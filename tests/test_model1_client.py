from simulation.model1_client import load_model1_predictions

df = load_model1_predictions()
print("Shape:", df.shape)
print(df.head(6).to_string(index=False))
print("\nAny nulls?", df.isna().sum().sum())
