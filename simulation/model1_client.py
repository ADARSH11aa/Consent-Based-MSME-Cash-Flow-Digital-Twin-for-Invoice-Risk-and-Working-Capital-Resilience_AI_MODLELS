"""
Calls Model 1's REAL, running FastAPI server (main.py, started with
`python -m uvicorn main:app --port 8000`) and joins the response with
invoice_amount / issue_date from data/raw/invoices.csv, since the API
response itself doesn't include those (confirmed from PredictionOutput's
schema in main.py).

Model 2 needs, per invoice: invoice_amount, days_since_issue, and the three
quantiles. This file's only job is bridging that gap - simulate_cashflow()
itself never needs to change, regardless of where the predictions come from.
"""
import pandas as pd
import requests


def load_model1_predictions(
    api_url="http://127.0.0.1:8000/predict/open-invoices",
    raw_invoices_path="data/raw/invoices.csv",
    today=None,
    timeout=30,
):
    """
    Returns a DataFrame with exactly the columns simulate_cashflow() expects:
        invoice_id, invoice_amount, days_since_issue,
        p10_payment_days, p50_payment_days, p90_payment_days

    today: reference date for "days_since_issue". Defaults to the real
    current date if not given - pass an explicit date (e.g. "2026-08-15")
    to pin the demo to a fixed day instead of whatever day you happen to run this.
    """
    if today is None:
        today = pd.Timestamp.now().normalize()
    else:
        today = pd.Timestamp(today)

    response = requests.get(api_url, timeout=timeout)
    response.raise_for_status()  # fail loudly if the server errored, not silently
    predictions = pd.DataFrame(response.json())

    if predictions.empty:
        raise ValueError(
            f"Model 1's API at {api_url} returned zero predictions - "
            f"check the server is running and its startup logs for errors."
        )

    raw = pd.read_csv(raw_invoices_path)
    raw["issue_date"] = pd.to_datetime(raw["issue_date"])

    merged = predictions.merge(
        raw[["invoice_id", "invoice_amount", "issue_date"]],
        on="invoice_id",
        how="left",
    )

    missing_amount = merged["invoice_amount"].isna().sum()
    if missing_amount > 0:
        raise ValueError(
            f"{missing_amount} invoice(s) from Model 1's API response had no "
            f"match in {raw_invoices_path} - check invoice_id formats line up."
        )

    merged["days_since_issue"] = (today - merged["issue_date"]).dt.days

    negative_days = (merged["days_since_issue"] < 0).sum()
    if negative_days > 0:
        raise ValueError(
            f"{negative_days} invoice(s) have an issue_date AFTER 'today' "
            f"({today.date()}) - check the `today` value you passed in."
        )

    return merged[[
        "invoice_id", "invoice_amount", "days_since_issue",
        "p10_payment_days", "p50_payment_days", "p90_payment_days",
    ]]