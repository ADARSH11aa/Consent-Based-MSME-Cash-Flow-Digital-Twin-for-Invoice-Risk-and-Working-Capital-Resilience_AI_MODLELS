from simulation.model1_client import load_model1_predictions
from simulation.monte_carlo import simulate_cashflow

# Step 1: get real Model 1 predictions for all open invoices (via the live API)
predictions = load_model1_predictions()
print(f"Loaded {len(predictions)} real predictions from Model 1")
print(f"Total open invoice value: {predictions['invoice_amount'].sum():,.2f}\n")

# Step 2: run the Monte Carlo simulation using those real predictions.
# These 5 numbers were the business decisions we settled earlier - adjust
# opening_cash / daily_expense to whatever your team wants to demo.
forecast, summary = simulate_cashflow(
    predictions,
    opening_cash=1_200_000,   # mock opening cash - smaller than total receivables
                               # on purpose, so the forecast has something to show
    daily_expense=18_000,     # mock flat daily burn
    horizon_days=90,
    n_sims=5000,
    min_buffer=0,             # "liquidity breach" = cash below this
)

print("=== Forecast at key days ===")
print(forecast[forecast["day"].isin([0, 15, 30, 45, 60, 90])].to_string(index=False))

print("\n=== Summary ===")
print(summary)