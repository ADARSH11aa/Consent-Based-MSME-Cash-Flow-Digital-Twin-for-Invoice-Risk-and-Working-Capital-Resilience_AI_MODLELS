"""
Model 2 - Cash-Flow Forecast API.

Wraps simulate_cashflow() (which internally calls Model 1's live API through
model1_client.py) behind a FastAPI endpoint - same integration pattern as
your teammate's main.py for Model 1.

Run this as its OWN server, on a DIFFERENT port than Model 1's:
    python -m uvicorn api.cashflow_api:app --port 8001

Keep Model 1's server running on port 8000 at the same time in another
terminal - this API calls that one internally, same as model1_client.py
already did when we tested it directly.
"""
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from simulation.model1_client import load_model1_predictions
from simulation.monte_carlo import simulate_cashflow

app = FastAPI(title="Model 2 - Cash-Flow Forecast API")


class ForecastRequest(BaseModel):
    opening_cash: float = Field(..., description="Cash on hand today")
    daily_expense: float = Field(..., description="Flat expected daily outflow")
    horizon_days: int = Field(90, description="How many days ahead to forecast")
    n_sims: int = Field(5000, description="Number of Monte Carlo simulations")
    min_buffer: float = Field(0, description="Cash level considered a liquidity breach")
    overdue_cap: float = Field(
        0.4, description="Confidence cap for invoices already past their own P90"
    )
    model1_api_url: str = Field(
        "http://127.0.0.1:8000/predict/open-invoices",
        description="Where to fetch Model 1's predictions from",
    )


class DailyForecast(BaseModel):
    day: int
    cash_p10: float
    cash_p50: float
    cash_p90: float
    prob_breach: float


class ForecastSummary(BaseModel):
    expected_min_cash: float
    days_to_likely_breach: Optional[int]
    overdue_invoice_count: int
    overdue_invoice_value: float


class ForecastResponse(BaseModel):
    forecast: list[DailyForecast]
    summary: ForecastSummary


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/forecast-cashflow", response_model=ForecastResponse)
def forecast_cashflow(request: ForecastRequest):
    """
    Fetches live predictions from Model 1, runs the Monte Carlo cash-flow
    simulation, and returns a daily P10/P50/P90 forecast.
    """
    try:
        predictions = load_model1_predictions(api_url=request.model1_api_url)
    except Exception as e:
        # 502 = "we tried to reach another service and it failed" - correct
        # status code here, since the actual fault is Model 1's API, not this request
        raise HTTPException(
            status_code=502,
            detail=f"Could not get predictions from Model 1's API: {e}",
        )

    if predictions.empty:
        raise HTTPException(status_code=502, detail="Model 1 returned zero predictions")

    forecast_df, summary = simulate_cashflow(
        predictions,
        opening_cash=request.opening_cash,
        daily_expense=request.daily_expense,
        horizon_days=request.horizon_days,
        n_sims=request.n_sims,
        min_buffer=request.min_buffer,
        overdue_cap=request.overdue_cap,
    )

    return ForecastResponse(
        forecast=forecast_df.to_dict(orient="records"),
        summary=summary,
    )