"""
Model 5 - SHAP explainability for Model 1's payment-window predictions.

This module is deliberately numbers-only (USP 7: the arithmetic/ML layers compute, the LLM
only narrates). It never constructs English sentences - it returns a fully traceable, ranked
list of {feature, value, shap_value, direction} per invoice, built directly from
shap.TreeExplainer against the real trained RandomForestRegressor. No top-N truncation, no
canned template (USP 3) - every number here is a real SHAP value a judge could click into.

Important scope note: Model 1's P10/P50/P90 come from percentiles across individual trees'
predictions (see rf_quantile_predict in model1_inference.py), not from three separately
trained models. shap.TreeExplainer explains the forest's mean/central prediction - the
number closest to P50 for most invoices. P10/P90 remain genuine tree-spread bands (USP 1)
but are not independently SHAP-decomposed here. "predicted_value" below reflects that
central prediction, not literally predicted_days_p50 from Model 1 (the two will usually be
close, but are not guaranteed identical).

Does NOT reload the model or preprocessor - reuses the exact same fitted objects Model 1
already loaded at startup, wrapped inside the shared Model1Artifacts instance.
"""

import pandas as pd
import shap

from model1_inference import FEATURE_COLUMNS, Model1Artifacts, build_model_matrix


class Model5Artifacts:
    """
    Wraps an existing Model1Artifacts instance and builds one shap.TreeExplainer once,
    at API startup - reused across every /explain request. No background dataset is
    needed since feature_perturbation="tree_path_dependent" uses the tree structure itself.
    """

    def __init__(self, model1_artifacts: Model1Artifacts):
        self.model1_artifacts = model1_artifacts
        self.explainer = shap.TreeExplainer(
            model1_artifacts.rf_model,
            feature_perturbation="tree_path_dependent",
        )
        # Names of the columns actually fed to the model, post-preprocessing
        # (e.g. sector may expand into sector_manufacturing, sector_retail, ...)
        self.transformed_feature_names = list(
            model1_artifacts.preprocessor.get_feature_names_out(FEATURE_COLUMNS)
        )


def _to_native(val):
    """Cast numpy scalars (float64, int64, ...) to plain Python types for JSON safety."""
    return val.item() if hasattr(val, "item") else val


def _group_feature_name(transformed_name: str) -> str:
    """
    Collapse an encoded column (e.g. 'sector_manufacturing', or 'cat__sector_manufacturing'
    if the preprocessor prefixes transformer names) back to its original raw feature name
    ('sector'), so contributions are reported against columns a human - and Model 6 - can
    actually recognize. Numeric/passthrough columns are returned unchanged.
    """
    bare_name = transformed_name.split("__")[-1]  # strip any ColumnTransformer prefix
    for raw_col in FEATURE_COLUMNS:
        if bare_name == raw_col or bare_name.startswith(f"{raw_col}_"):
            return raw_col
    return bare_name


def explain_invoice(invoice_rows: pd.DataFrame, artifacts5: Model5Artifacts) -> list[dict]:
    """
    invoice_rows: same shape as predict_payment_window's input (one row per invoice).

    Returns one dict per invoice, in the same row order as invoice_rows:
        {
          "invoice_id": str,
          "base_value": float,        # explainer's expected value (avg model output)
          "predicted_value": float,   # this invoice's model output (base_value + sum(shap))
          "contributions": [
              {"feature": "customer_avg_payment_days", "value": 12.0,
               "shap_value": 4.1, "direction": "increases"},
              ...
          ]  # FULL ranked list, most-influential first - no truncation (USP 3)
        }
    """
    frame, X = build_model_matrix(invoice_rows, artifacts5.model1_artifacts)
    frame = frame.reset_index(drop=True)

    shap_values = artifacts5.explainer.shap_values(X)  # shape: (n_rows, n_transformed_features)
    base_value = float(artifacts5.explainer.expected_value)

    results = []
    for i, invoice_id in enumerate(frame["invoice_id"]):
        row_shap = shap_values[i]

        # Group one-hot columns back to their raw feature, summing their SHAP contribution
        grouped: dict[str, float] = {}
        for name, val in zip(artifacts5.transformed_feature_names, row_shap):
            raw_name = _group_feature_name(name)
            grouped[raw_name] = grouped.get(raw_name, 0.0) + float(val)

        contributions = [
            {
                "feature": raw_name,
                "value": _to_native(frame.iloc[i][raw_name]) if raw_name in frame.columns else None,
                "shap_value": round(shap_val, 4),
                "direction": "increases" if shap_val > 0 else "decreases",
            }
            for raw_name, shap_val in sorted(
                grouped.items(), key=lambda kv: abs(kv[1]), reverse=True
            )
        ]

        results.append(
            {
                "invoice_id": invoice_id,
                "base_value": round(base_value, 4),
                "predicted_value": round(base_value + float(row_shap.sum()), 4),
                "contributions": contributions,
            }
        )

    return results