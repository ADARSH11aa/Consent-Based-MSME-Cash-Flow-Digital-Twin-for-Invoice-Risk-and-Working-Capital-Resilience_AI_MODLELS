from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model1_inference import Model1Artifacts, predict_payment_window

MODEL_DIR = Path("models")
RAW_INVOICES_PATH = Path("data/raw/invoices.csv")

app = FastAPI(title="Model 1 - Payment Prediction API")

# Loaded once at startup, reused across every request - do NOT reload/refit per-request.
artifacts: Model1Artifacts | None = None


@app.on_event("startup")
def load_model():
    global artifacts
    artifacts = Model1Artifacts(MODEL_DIR)
    raw = pd.read_csv(RAW_INVOICES_PATH)
    raw["issue_date"] = pd.to_datetime(raw["issue_date"])
    closed_history = raw[raw["status"] == "closed"].copy()
    artifacts.refresh_customer_stats(closed_history)


# ---------- request/response schemas ----------

class InvoiceInput(BaseModel):
    invoice_id: str
    cust_number: str
    sector: str
    invoice_amount: float
    payment_term_days: int
    issue_date: date


class PredictionOutput(BaseModel):
    invoice_id: str
    customer_id: str
    p10_payment_days: int
    p50_payment_days: int
    p90_payment_days: int
    confidence: Literal["normal", "low"]


# ---------- endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": artifacts is not None}


@app.post("/predict/invoices", response_model=list[PredictionOutput])
def predict_invoices(invoices: list[InvoiceInput]):
    """
    Predict P10/P50/P90 days-to-payment for arbitrary invoices - e.g. a single newly-created
    invoice, or a batch. This is also the endpoint the 'live correction' demo hits: when the
    user edits an OCR-extracted amount or term on stage, re-POST that one corrected invoice here
    and the forecast updates.
    """
    if artifacts is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")

    df = pd.DataFrame([inv.model_dump() for inv in invoices])
    df["issue_date"] = pd.to_datetime(df["issue_date"])

    result = predict_payment_window(df, artifacts)

    return [
        PredictionOutput(
            invoice_id=row.invoice_id,
            customer_id=row.cust_number,
            p10_payment_days=row.predicted_days_p10,
            p50_payment_days=row.predicted_days_p50,
            p90_payment_days=row.predicted_days_p90,
            confidence=row.confidence,
        )
        for row in result.itertuples()
    ]


@app.get("/predict/open-invoices", response_model=list[PredictionOutput])
def predict_open_invoices():
    """
    Convenience endpoint for the demo: predicts every currently open/disputed_open invoice
    straight from data/raw/invoices.csv. This is what Model 2 calls to get the full outstanding
    receivables forecast without needing to know anything about how Model 1 works internally.
    """
    if artifacts is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")

    raw = pd.read_csv(RAW_INVOICES_PATH)
    raw["issue_date"] = pd.to_datetime(raw["issue_date"])
    open_invoices = raw[raw["status"].isin(["open", "disputed_open"])].copy()

    result = predict_payment_window(open_invoices, artifacts)

    return [
        PredictionOutput(
            invoice_id=row.invoice_id,
            customer_id=row.cust_number,
            p10_payment_days=row.predicted_days_p10,
            p50_payment_days=row.predicted_days_p50,
            p90_payment_days=row.predicted_days_p90,
            confidence=row.confidence,
        )
        for row in result.itertuples()
    ]


@app.post("/admin/refresh-customer-stats")
def refresh_customer_stats():
    """
    Call this after invoices get marked as paid/closed (e.g. after the demo's 'mark as paid'
    action), so future predictions use the latest payment history instead of stale stats
    computed at startup.
    """
    if artifacts is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")

    raw = pd.read_csv(RAW_INVOICES_PATH)
    raw["issue_date"] = pd.to_datetime(raw["issue_date"])
    closed_history = raw[raw["status"] == "closed"].copy()
    artifacts.refresh_customer_stats(closed_history)
    return {"status": "refreshed", "customers": len(artifacts.customer_stats)}