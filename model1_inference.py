"""
Model 1 inference logic - shared between the training notebook and the FastAPI serving layer.

This module has NO training code in it. It only knows how to:
  1. build customer-history features for a NEW invoice given past closed invoices
  2. run the saved RF quantile-forest model to get P10/P50/P90 days-to-payment
  3. apply the cold-start / sector-prior fallback consistently

Keeping this in one file (instead of copy-pasting the logic into the API) means notebook 03
and the API can never silently drift apart.
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

QUANTILES = {"p10": 0.10, "p50": 0.50, "p90": 0.90}


class Model1Artifacts:
    """Loaded once at API startup, reused across requests."""

    def __init__(self, model_dir: Path):
        self.preprocessor = joblib.load(model_dir / "model1_preprocessor.joblib")
        self.rf_model = joblib.load(model_dir / "model1_rf_quantile.joblib")
        self.sector_priors = joblib.load(model_dir / "model1_sector_priors.joblib")
        self.global_prior = joblib.load(model_dir / "model1_global_prior.joblib")
        self.customer_stats = None  # set via refresh_customer_stats()

    def refresh_customer_stats(self, closed_history_df: pd.DataFrame):
        """
        Rebuild the per-customer history lookup table from all currently-closed invoices.
        Call this at startup, and again any time new invoices get closed (e.g. a nightly job,
        or right after the 'live correction' demo moment marks something as paid).
        """
        closed_history_df = closed_history_df.sort_values(["cust_number", "issue_date"])

        stats = closed_history_df.groupby("cust_number").agg(
            customer_avg_payment_days=("days_to_payment", "mean"),
            customer_recent_avg_payment_days=("days_to_payment", lambda x: x.tail(3).mean()),
            customer_payment_std=("days_to_payment", "std"),
            previous_payment_days=("days_to_payment", "last"),
            customer_invoice_count=("days_to_payment", "count"),
        )
        stats["payment_behavior_trend"] = (
            stats["customer_recent_avg_payment_days"] - stats["customer_avg_payment_days"]
        )
        self.customer_stats = stats


def rf_quantile_predict(rf_model, X, quantiles=QUANTILES) -> pd.DataFrame:
    """Per-tree prediction spread -> percentiles across trees. X must already be preprocessed."""
    all_tree_preds = np.array([tree.predict(X) for tree in rf_model.estimators_])
    percentiles = np.percentile(all_tree_preds, [q * 100 for q in quantiles.values()], axis=0)
    return pd.DataFrame(percentiles.T, columns=list(quantiles.keys()))


def predict_payment_window(invoice_rows: pd.DataFrame, artifacts: Model1Artifacts) -> pd.DataFrame:
    """
    invoice_rows: DataFrame with, at minimum, one row per invoice containing
        invoice_id, cust_number, sector, invoice_amount, payment_term_days, issue_date
    Historical features are looked up from artifacts.customer_stats (built by
    refresh_customer_stats) rather than expected to already be present on invoice_rows -
    this is the realistic production shape: the caller only knows about the NEW invoice,
    not that customer's whole history.
    """
    if artifacts.customer_stats is None:
        raise RuntimeError("call artifacts.refresh_customer_stats(...) before predicting")

    frame = invoice_rows.merge(
        artifacts.customer_stats, on="cust_number", how="left", suffixes=("", "_hist")
    )

    frame["is_cold_start"] = (
        frame["customer_invoice_count"].isna()
        | (frame["customer_invoice_count"] < MIN_HISTORY_FOR_CONFIDENCE)
    )

    sector_fill = frame["sector"].map(artifacts.sector_priors).fillna(artifacts.global_prior)
    for col in ["customer_avg_payment_days", "customer_recent_avg_payment_days", "previous_payment_days"]:
        frame[col] = frame[col].fillna(sector_fill)
    frame["customer_payment_std"] = frame["customer_payment_std"].fillna(0)
    frame["payment_behavior_trend"] = frame["payment_behavior_trend"].fillna(0)
    frame["customer_invoice_count"] = frame["customer_invoice_count"].fillna(0)

    X = artifacts.preprocessor.transform(frame[FEATURE_COLUMNS])
    preds = rf_quantile_predict(artifacts.rf_model, X)

    out = frame[["invoice_id", "cust_number", "issue_date"]].copy()
    out["predicted_days_p10"] = np.round(preds["p10"].values).astype(int)
    out["predicted_days_p50"] = np.round(preds["p50"].values).astype(int)
    out["predicted_days_p90"] = np.round(preds["p90"].values).astype(int)
    out["confidence"] = np.where(frame["is_cold_start"], "low", "normal")
    return out