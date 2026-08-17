"""
MODEL 3 — Anomaly & Volatility Detection (REAL DATASET VERSION)
MSME Cash-Flow Digital Twin | AI/ML Module

This version:
  1. Loads the real invoices.csv dataset (from data/raw/invoices.csv)
  2. Maps its columns onto the transaction schema Model 3 expects
  3. Runs the same rolling z-score + Isolation Forest pipeline
  4. FINE-TUNES both detectors against the real ground-truth column
     'is_big_ticket_spike' — so you can see actual precision/recall,
     not just "it printed some flags"

Run directly:  python model3_real_data.py --data path/to/invoices.csv
"""

import argparse
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------
# STEP 1: Load and map the real dataset onto our transaction schema
# ---------------------------------------------------------------------
def load_invoices_csv(path: str) -> pd.DataFrame:
    """
    Maps invoices.csv columns onto the schema the anomaly pipeline expects:
        transaction_id, business_id, date, amount, category, counterparty

    Mapping used:
        transaction_id  <- invoice_id
        business_id     <- constant "B001" (dataset is single-business)
        date             <- issue_date
        amount           <- invoice_amount
        category         <- "invoice" (constant — could also use sector)
        counterparty     <- cust_number  (customer, since these are receivables)

    Ground truth columns are kept alongside (not used as features):
        is_big_ticket_spike, customer_archetype_TRUE_LABEL
    """
    raw = pd.read_csv(path, parse_dates=["issue_date", "due_date", "actual_paid_date"])

    df = pd.DataFrame({
        "transaction_id": raw["invoice_id"],
        "business_id": "B001",
        "date": raw["issue_date"],
        "amount": raw["invoice_amount"],
        "category": "invoice",
        "counterparty": raw["cust_number"],
        # keep these for evaluation / richer labeling, not used as model features
        "sector": raw["sector"],
        "customer_name": raw["customer_name"],
        "delay_vs_due_date": raw["delay_vs_due_date"],
        "archetype_true_label": raw["customer_archetype_TRUE_LABEL"],
        "is_big_ticket_spike": raw["is_big_ticket_spike"],
    })

    df = df.sort_values(["business_id", "date"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------
# STEP 2+3: Feature engineering + rolling z-score baseline
# (same logic as the synthetic version, unchanged)
# ---------------------------------------------------------------------
def compute_zscore_features(
    df: pd.DataFrame,
    group_cols=("business_id", "category", "counterparty"),
    window=10,
    z_threshold=2.5,
) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    group_cols = list(group_cols)

    df["rolling_mean"] = df.groupby(group_cols)["amount"].transform(
        lambda x: x.rolling(window, min_periods=3).mean()
    )
    df["rolling_std"] = df.groupby(group_cols)["amount"].transform(
        lambda x: x.rolling(window, min_periods=3).std()
    )
    df["z_score"] = (df["amount"] - df["rolling_mean"]) / df["rolling_std"]
    df["z_score"] = df["z_score"].replace([np.inf, -np.inf], np.nan)
    df["z_flag"] = df["z_score"].abs() > z_threshold

    df["days_since_last"] = (
        df.groupby(["business_id", "counterparty"])["date"].diff().dt.days
    )
    df["day_of_month"] = df["date"].dt.day

    return df


# ---------------------------------------------------------------------
# STEP 4: Isolation Forest layer (trained per business)
# ---------------------------------------------------------------------
def run_isolation_forest(df: pd.DataFrame, contamination=0.05) -> pd.DataFrame:
    feature_cols = ["amount", "z_score", "days_since_last", "day_of_month"]
    output_frames = []

    for business_id, group in df.groupby("business_id"):
        g = group.copy()
        X = g[feature_cols].fillna(0)

        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=RANDOM_SEED,
        )
        model.fit(X)

        g["anomaly_score"] = model.decision_function(X)
        g["is_anomaly"] = model.predict(X) == -1
        output_frames.append(g)

    return pd.concat(output_frames).sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------
# STEP 5: Combine z-score + Isolation Forest into one flag/label
# ---------------------------------------------------------------------
def combine_flags(row) -> str:
    if row["is_anomaly"] and row["z_flag"]:
        return "high_confidence_anomaly"
    elif row["is_anomaly"] or row["z_flag"]:
        return "possible_anomaly"
    return "normal"


def label_anomaly(row) -> str:
    if row["flag_level"] == "normal":
        return ""
    if pd.notna(row["days_since_last"]) and row["days_since_last"] < 15:
        return "volatile payment pattern"
    return "unusual invoice amount"


# ---------------------------------------------------------------------
# STEP 6: JSON output builder
# ---------------------------------------------------------------------
def build_output_json(df: pd.DataFrame) -> list:
    cols = [
        "transaction_id", "business_id", "date", "amount", "category",
        "counterparty", "z_score", "anomaly_score", "flag_level", "label",
    ]
    out_df = df[cols].copy()
    out_df["date"] = out_df["date"].dt.strftime("%Y-%m-%d")
    out_df["z_score"] = out_df["z_score"].round(2)
    out_df["anomaly_score"] = out_df["anomaly_score"].round(4)
    return out_df.to_dict(orient="records")


# ---------------------------------------------------------------------
# FINE-TUNING: sweep parameters against the real ground-truth label
# ---------------------------------------------------------------------
def evaluate_against_ground_truth(df: pd.DataFrame) -> dict:
    """
    Compares model flags (flag_level != 'normal') against the real
    'is_big_ticket_spike' ground-truth column.
    Note: this ground truth only covers AMOUNT spikes, not timing/frequency
    anomalies, so recall will naturally be partial — that's expected and
    worth explaining in the demo.
    """
    y_true = df["is_big_ticket_spike"].astype(int)
    y_pred = (df["flag_level"] != "normal").astype(int)

    return {
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 3),
        "flagged_count": int(y_pred.sum()),
        "ground_truth_count": int(y_true.sum()),
    }


def tune_parameters(df_base: pd.DataFrame, z_thresholds, contaminations) -> pd.DataFrame:
    """
    Grid search over z_threshold and contamination.
    Returns a DataFrame of results sorted by F1 score (best first).
    """
    results = []
    for z_thresh in z_thresholds:
        df_z = compute_zscore_features(df_base, z_threshold=z_thresh)
        for cont in contaminations:
            df_iso = run_isolation_forest(df_z, contamination=cont)
            df_iso["flag_level"] = df_iso.apply(combine_flags, axis=1)
            metrics = evaluate_against_ground_truth(df_iso)
            results.append({
                "z_threshold": z_thresh,
                "contamination": cont,
                **metrics,
            })

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False).reset_index(drop=True)
    return results_df


# ---------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------
def run_pipeline(data_path: str, save_path="model3_real_output.json", do_tune=True):
    print(f"Step 1: Loading real dataset from {data_path} ...")
    df = load_invoices_csv(data_path)
    print(f"  -> {len(df)} invoices loaded, {df['counterparty'].nunique()} customers\n")

    if do_tune:
        print("Step 2: Fine-tuning parameters against ground truth (is_big_ticket_spike)...")
        z_thresholds = [1.5, 2.0, 2.5, 3.0]
        contaminations = [0.02, 0.03, 0.05, 0.08]
        tuning_results = tune_parameters(df, z_thresholds, contaminations)
        print(tuning_results.to_string(index=False))
        print()

        best = tuning_results.iloc[0]
        best_z = best["z_threshold"]
        best_cont = best["contamination"]
        print(f"  -> Best combo: z_threshold={best_z}, contamination={best_cont} "
              f"(F1={best['f1']}, precision={best['precision']}, recall={best['recall']})\n")
    else:
        best_z, best_cont = 2.5, 0.05

    print(f"Step 3: Running final pipeline with tuned parameters "
          f"(z_threshold={best_z}, contamination={best_cont})...")
    df = compute_zscore_features(df, z_threshold=best_z)
    df = run_isolation_forest(df, contamination=best_cont)
    df["flag_level"] = df.apply(combine_flags, axis=1)
    df["label"] = df.apply(label_anomaly, axis=1)

    final_metrics = evaluate_against_ground_truth(df)
    print(f"  -> Final metrics on full dataset: {final_metrics}\n")

    print("Step 4: Saving JSON output...")
    output = build_output_json(df)
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  -> saved to {save_path}\n")

    flagged = df[df["flag_level"] != "normal"][
        ["transaction_id", "counterparty", "date", "amount",
         "z_score", "anomaly_score", "flag_level", "label", "is_big_ticket_spike"]
    ].head(20)
    print("=== Flagged transactions preview (first 20) ===")
    print(flagged.to_string(index=False))

    return df, final_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/raw/invoices.csv",
                         help="Path to invoices.csv")
    parser.add_argument("--no-tune", action="store_true",
                         help="Skip the tuning grid search and use default params")
    args = parser.parse_args()

    run_pipeline(args.data, do_tune=not args.no_tune)
