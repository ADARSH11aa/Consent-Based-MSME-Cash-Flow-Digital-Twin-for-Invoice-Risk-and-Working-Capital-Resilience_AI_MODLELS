"""
Model 1 inference logic - shared between the training notebook and the FastAPI serving layer.

This module has NO training code. It only knows how to:

  1. Build customer-history features for a NEW invoice given past closed invoices.
  2. Build the exact model matrix used by the trained model.
  3. Run the saved RF quantile-forest model to get P10/P50/P90 days-to-payment.
  4. Apply the cold-start / sector-prior fallback consistently.

Keeping this in one file means Model 1 prediction and Model 5 SHAP explanation
use exactly the same feature-building logic.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


MIN_HISTORY_FOR_CONFIDENCE = 3

FEATURE_COLUMNS = [
    "customer_avg_payment_days",
    "customer_recent_avg_payment_days",
    "customer_invoice_count",
    "customer_payment_std",
    "payment_behavior_trend",
    "previous_payment_days",
    "invoice_amount",
    "payment_term_days",
    "sector",
]

QUANTILES = {
    "p10": 0.10,
    "p50": 0.50,
    "p90": 0.90,
}


class Model1Artifacts:
    """
    Loaded once at API startup and reused across requests.
    """

    def __init__(self, model_dir: Path):
        self.preprocessor = joblib.load(
            model_dir / "model1_preprocessor.joblib"
        )

        self.rf_model = joblib.load(
            model_dir / "model1_rf_quantile.joblib"
        )

        self.sector_priors = joblib.load(
            model_dir / "model1_sector_priors.joblib"
        )

        self.global_prior = joblib.load(
            model_dir / "model1_global_prior.joblib"
        )

        # Filled by refresh_customer_stats()
        self.customer_stats = None

    def refresh_customer_stats(self, closed_history_df: pd.DataFrame):
        """
        Rebuild the per-customer history lookup table from all currently
        closed invoices.

        This should be called:
          - at API startup
          - after new invoices are marked as closed/paid
        """

        closed_history_df = closed_history_df.sort_values(
            ["cust_number", "issue_date"]
        )

        stats = closed_history_df.groupby("cust_number").agg(
            customer_avg_payment_days=(
                "days_to_payment",
                "mean",
            ),
            customer_recent_avg_payment_days=(
                "days_to_payment",
                lambda x: x.tail(3).mean(),
            ),
            customer_payment_std=(
                "days_to_payment",
                "std",
            ),
            previous_payment_days=(
                "days_to_payment",
                "last",
            ),
            customer_invoice_count=(
                "days_to_payment",
                "count",
            ),
        )

        stats["payment_behavior_trend"] = (
            stats["customer_recent_avg_payment_days"]
            - stats["customer_avg_payment_days"]
        )

        self.customer_stats = stats


def build_model_matrix(
    invoice_rows: pd.DataFrame,
    artifacts: Model1Artifacts,
):
    """
    Build the exact feature frame and transformed matrix used by Model 1.

    Parameters
    ----------
    invoice_rows:
        DataFrame containing NEW invoices. At minimum:

            invoice_id
            cust_number
            sector
            invoice_amount
            payment_term_days
            issue_date

    artifacts:
        Loaded Model1Artifacts instance.

    Returns
    -------
    frame:
        DataFrame containing the original invoice information plus all
        engineered customer-history features.

    X:
        Preprocessed matrix actually passed into the RandomForest model.

    IMPORTANT:
    This function is shared by Model 1 prediction and Model 5 SHAP.
    """

    if artifacts.customer_stats is None:
        raise RuntimeError(
            "call artifacts.refresh_customer_stats(...) before building "
            "the model matrix"
        )

    # Make a copy so the caller's DataFrame is never modified.
    invoice_rows = invoice_rows.copy()

    # Ensure dates are in datetime format.
    if "issue_date" in invoice_rows.columns:
        invoice_rows["issue_date"] = pd.to_datetime(
            invoice_rows["issue_date"]
        )

    # ---------------------------------------------------------
    # 1. Attach historical customer statistics
    # ---------------------------------------------------------

    frame = invoice_rows.merge(
        artifacts.customer_stats,
        on="cust_number",
        how="left",
        suffixes=("", "_hist"),
    )

    # ---------------------------------------------------------
    # 2. Determine cold-start status
    # ---------------------------------------------------------

    frame["is_cold_start"] = (
        frame["customer_invoice_count"].isna()
        | (
            frame["customer_invoice_count"]
            < MIN_HISTORY_FOR_CONFIDENCE
        )
    )

    # ---------------------------------------------------------
    # 3. Sector/global fallback
    # ---------------------------------------------------------

    sector_fill = (
        frame["sector"]
        .map(artifacts.sector_priors)
        .fillna(artifacts.global_prior)
    )

    # These features use sector/global priors when customer
    # history is unavailable.
    for col in [
        "customer_avg_payment_days",
        "customer_recent_avg_payment_days",
        "previous_payment_days",
    ]:
        frame[col] = frame[col].fillna(sector_fill)

    # Standard deviation has no meaningful value for a customer
    # with insufficient history, so use zero.
    frame["customer_payment_std"] = (
        frame["customer_payment_std"].fillna(0)
    )

    # No recent-vs-average movement means no observed trend.
    frame["payment_behavior_trend"] = (
        frame["payment_behavior_trend"].fillna(0)
    )

    # No history means zero observed invoices.
    frame["customer_invoice_count"] = (
        frame["customer_invoice_count"].fillna(0)
    )

    # ---------------------------------------------------------
    # 4. Build exact model input
    # ---------------------------------------------------------

    X = artifacts.preprocessor.transform(
        frame[FEATURE_COLUMNS]
    )

    return frame, X


def rf_quantile_predict(
    rf_model,
    X,
    quantiles=QUANTILES,
) -> pd.DataFrame:
    """
    Generate P10/P50/P90 from the distribution of predictions
    across individual Random Forest trees.

    X must already be preprocessed.
    """

    all_tree_preds = np.array(
        [
            tree.predict(X)
            for tree in rf_model.estimators_
        ]
    )

    percentiles = np.percentile(
        all_tree_preds,
        [q * 100 for q in quantiles.values()],
        axis=0,
    )

    return pd.DataFrame(
        percentiles.T,
        columns=list(quantiles.keys()),
    )


def predict_payment_window(
    invoice_rows: pd.DataFrame,
    artifacts: Model1Artifacts,
) -> pd.DataFrame:
    """
    Predict P10/P50/P90 days-to-payment for one or more invoices.

    Historical features are generated automatically from
    artifacts.customer_stats.
    """

    frame, X = build_model_matrix(
        invoice_rows,
        artifacts,
    )

    preds = rf_quantile_predict(
        artifacts.rf_model,
        X,
    )

    out = frame[
        [
            "invoice_id",
            "cust_number",
            "issue_date",
        ]
    ].copy()

    out["predicted_days_p10"] = (
        np.round(preds["p10"].values)
        .astype(int)
    )

    out["predicted_days_p50"] = (
        np.round(preds["p50"].values)
        .astype(int)
    )

    out["predicted_days_p90"] = (
        np.round(preds["p90"].values)
        .astype(int)
    )

    out["confidence"] = np.where(
        frame["is_cold_start"],
        "low",
        "normal",
    )

    return out