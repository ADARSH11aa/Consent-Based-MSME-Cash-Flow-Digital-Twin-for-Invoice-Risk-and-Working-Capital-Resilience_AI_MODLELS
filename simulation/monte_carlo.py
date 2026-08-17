"""
Monte Carlo cash-flow simulation engine (v3).

Consumes Model 1's real output contract (invoice_id, invoice_amount,
days_since_issue, p10/p50/p90_payment_days) and produces a daily P10/P50/P90
cash forecast.

v3 change: invoices that are already past their OWN p90_payment_days (i.e.
Model 1's worst-case prediction has already been blown through, and it's
still unpaid) are no longer sampled from Model 1's distribution directly -
doing so would imply near-100% "paid any moment now", which is an
unjustified extrapolation of a model that was never confident about
invoices this overdue. Instead, a fraction of simulations (`overdue_cap`)
assume it gets collected soon; the rest assume it stays uncollected for
the whole forecast window. This keeps severely overdue invoices from
silently inflating the forecast.
"""
import numpy as np
import pandas as pd

from simulation.quantile_sampler import sample_payment_days


def sample_invoice_payment_day(row, n_sims, rng, horizon_days, overdue_cap=0.4):
    """
    Returns an array of shape (n_sims,): "days from today" this ONE invoice
    is paid, in each simulation.
    """
    if row.days_since_issue > row.p90_payment_days:
        # Severely overdue: don't trust Model 1's distribution here - cap
        # confidence instead of extrapolating it.
        is_collected_soon = rng.random(n_sims) < overdue_cap
        # horizon_days + 1 = "does not land inside this forecast window at all"
        return np.where(is_collected_soon, 0, horizon_days + 1)
    else:
        payment_days_from_issue = sample_payment_days(
            row.p10_payment_days, row.p50_payment_days, row.p90_payment_days,
            n_sims, rng
        )
        payment_day_from_today = payment_days_from_issue - row.days_since_issue
        return np.clip(payment_day_from_today, 0, None)


def simulate_cashflow(
    predictions,        # DataFrame: invoice_id, invoice_amount, days_since_issue,
                         #            p10_payment_days, p50_payment_days, p90_payment_days
    opening_cash,        # float: cash on hand today
    daily_expense,        # float: flat expected daily outflow
    horizon_days=90,       # how many days ahead to forecast
    n_sims=3000,
    min_buffer=0,          # cash level considered a "liquidity breach"
    overdue_cap=0.4,       # see module docstring - confidence cap for severely overdue invoices
    seed=42,
):
    rng = np.random.default_rng(seed)

    days = np.arange(0, horizon_days + 1)
    cumulative_inflow = np.zeros((n_sims, len(days)))

    overdue_count = 0
    overdue_value = 0.0

    for row in predictions.itertuples(index=False):
        if row.days_since_issue > row.p90_payment_days:
            overdue_count += 1
            overdue_value += row.invoice_amount

        payment_day_from_today = sample_invoice_payment_day(
            row, n_sims, rng, horizon_days, overdue_cap
        )
        paid_on_or_before = (payment_day_from_today[:, None] <= days[None, :])
        cumulative_inflow += paid_on_or_before * row.invoice_amount

    outflow = daily_expense * days[None, :]
    cash_balance = opening_cash - outflow + cumulative_inflow

    p10 = np.percentile(cash_balance, 10, axis=0)
    p50 = np.percentile(cash_balance, 50, axis=0)
    p90 = np.percentile(cash_balance, 90, axis=0)
    breach_prob = (cash_balance < min_buffer).mean(axis=0)

    forecast = pd.DataFrame({
        "day": days,
        "cash_p10": p10,
        "cash_p50": p50,
        "cash_p90": p90,
        "prob_breach": breach_prob,
    })

    expected_min_cash = cash_balance.min(axis=1).mean()
    breach_days = forecast.index[forecast["prob_breach"] > 0.5]
    days_to_likely_breach = int(forecast.loc[breach_days[0], "day"]) if len(breach_days) > 0 else None

    summary = {
        "expected_min_cash": float(expected_min_cash),
        "days_to_likely_breach": days_to_likely_breach,
        "overdue_invoice_count": overdue_count,
        "overdue_invoice_value": overdue_value,
    }

    return forecast, summary