"""
Model 5 - SHAP explainability for Model 1's payment-window predictions.

This module is deliberately numbers-only.

The ML layer computes:
    - SHAP values
    - feature values
    - feature contribution direction
    - base value
    - central model prediction

The LLM layer, if used later, is responsible only for narrating these
already-computed numbers.

Important scope note:

Model 1's P10/P50/P90 come from percentiles across individual
RandomForest tree predictions.

TreeExplainer explains the RandomForest's central/mean prediction,
not the individual P10/P50/P90 quantiles.

Therefore:

    predicted_value
        = SHAP base value + sum(SHAP values)

is the RandomForest central prediction.

It is NOT guaranteed to be numerically identical to
predicted_days_p50 from Model 1, although it will usually be close.

P10/P90 remain genuine tree-spread uncertainty bands but are not
independently SHAP-decomposed here.
"""

import pandas as pd
import shap

from model1_inference import (
    FEATURE_COLUMNS,
    Model1Artifacts,
    build_model_matrix,
)


class Model5Artifacts:
    """
    Holds the SHAP TreeExplainer built from the exact same RandomForest
    instance used by Model 1.

    The explainer is created once at API startup and reused across
    requests.
    """

    def __init__(
        self,
        model1_artifacts: Model1Artifacts,
    ):
        self.model1_artifacts = model1_artifacts

        self.explainer = shap.TreeExplainer(
            model1_artifacts.rf_model,
            feature_perturbation="tree_path_dependent",
        )

        # These are the feature names after preprocessing.
        #
        # Example:
        #   sector_manufacturing
        #   sector_retail
        #
        # depending on how the saved preprocessor was trained.
        self.transformed_feature_names = list(
            model1_artifacts.preprocessor.get_feature_names_out(
                FEATURE_COLUMNS
            )
        )


def _to_native(val):
    """
    Convert NumPy scalar values into normal Python values
    for JSON serialization.
    """

    return (
        val.item()
        if hasattr(val, "item")
        else val
    )


def _group_feature_name(
    transformed_name: str,
) -> str:
    """
    Convert transformed/preprocessed feature names back to the
    original raw feature name.

    Examples:

        sector_manufacturing
            -> sector

        cat__sector_retail
            -> sector

        invoice_amount
            -> invoice_amount
    """

    # Remove ColumnTransformer prefixes such as:
    #
    # cat__
    # num__
    #
    bare_name = transformed_name.split("__")[-1]

    for raw_col in FEATURE_COLUMNS:

        if bare_name == raw_col:
            return raw_col

        if bare_name.startswith(
            f"{raw_col}_"
        ):
            return raw_col

    return bare_name


def _get_base_value(explainer) -> float:
    """
    Safely extract the scalar expected/base value from SHAP.

    Different SHAP versions may represent expected_value as:
        float
        np.float64
        np.ndarray([value])
    """

    base_value = explainer.expected_value

    if hasattr(base_value, "item"):
        try:
            return float(base_value.item())
        except ValueError:
            pass

    if hasattr(base_value, "__len__"):
        return float(base_value[0])

    return float(base_value)


def explain_invoice(
    invoice_rows: pd.DataFrame,
    artifacts5: Model5Artifacts,
) -> list[dict]:
    """
    Generate a complete SHAP explanation for every invoice.

    Parameters
    ----------
    invoice_rows:
        Same invoice input structure used by Model 1.

    artifacts5:
        Model5Artifacts containing the SHAP explainer.

    Returns
    -------
    list[dict]

    Example structure:

    {
        "invoice_id": "INV001",
        "base_value": 35.2,
        "predicted_value": 41.8,
        "contributions": [
            {
                "feature": "customer_avg_payment_days",
                "value": 42.0,
                "shap_value": 5.13,
                "direction": "increases"
            },
            ...
        ]
    }

    The complete contribution list is returned.
    There is no top-N truncation.
    """

    # ---------------------------------------------------------
    # Build EXACTLY the same matrix used by Model 1.
    # ---------------------------------------------------------

    frame, X = build_model_matrix(
        invoice_rows,
        artifacts5.model1_artifacts,
    )

    frame = frame.reset_index(drop=True)

    # ---------------------------------------------------------
    # Calculate SHAP values
    # ---------------------------------------------------------

    shap_values = artifacts5.explainer.shap_values(X)

    # RandomForestRegressor normally returns:
    #
    #     (n_rows, n_features)
    #
    # But converting through numpy ensures consistent behavior.
    shap_values = pd.DataFrame(
        shap_values
    ).to_numpy()

    base_value = _get_base_value(
        artifacts5.explainer
    )

    results = []

    # ---------------------------------------------------------
    # Explain each invoice
    # ---------------------------------------------------------

    for i, invoice_id in enumerate(
        frame["invoice_id"]
    ):

        row_shap = shap_values[i]

        # -----------------------------------------------------
        # Group one-hot encoded columns back into raw features
        # -----------------------------------------------------

        grouped = {}

        for name, val in zip(
            artifacts5.transformed_feature_names,
            row_shap,
        ):

            raw_name = _group_feature_name(
                name
            )

            grouped[raw_name] = (
                grouped.get(raw_name, 0.0)
                + float(val)
            )

        # -----------------------------------------------------
        # Rank by absolute SHAP impact
        # -----------------------------------------------------

        contributions = []

        for raw_name, shap_val in sorted(
            grouped.items(),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        ):

            if raw_name in frame.columns:
                feature_value = _to_native(
                    frame.iloc[i][raw_name]
                )
            else:
                feature_value = None

            contributions.append(
                {
                    "feature": raw_name,
                    "value": feature_value,
                    "shap_value": round(
                        float(shap_val),
                        4,
                    ),
                    "direction": (
                        "increases"
                        if shap_val > 0
                        else "decreases"
                    ),
                }
            )

        # -----------------------------------------------------
        # SHAP reconstruction
        #
        # prediction = base_value + sum(SHAP values)
        # -----------------------------------------------------

        predicted_value = (
            base_value
            + float(row_shap.sum())
        )

        results.append(
            {
                "invoice_id": _to_native(
                    invoice_id
                ),
                "base_value": round(
                    base_value,
                    4,
                ),
                "predicted_value": round(
                    predicted_value,
                    4,
                ),
                "contributions": contributions,
            }
        )

    return results