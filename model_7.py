import json
import os
import pandas as pd

# MODEL 7 - NON-DEBT-FIRST RECOMMENDATION RANKER

CSV_FILE = "invoices.csv"
MODEL_2_FILE = "model_2_input.json"
OUTPUT_FILE = "model_7_output.json"

# MODEL 7 WEIGHTS

WEIGHTS = {
    "cost": 0.25,
    "speed": 0.25,
    "liquidity": 0.40,
    "risk": 0.10
}

# RECOVERY STRATEGIES

RECOVERY_OPTIONS = [
    {
        "strategy": "Supplier Payment Extension",
        "cost_rate": 0.005,
        "recovery_days": 15,
        "gap_reduction": 30,
        "risk": "low"
    },
    {
        "strategy": "Early Customer Payment",
        "cost_rate": 0.010,
        "recovery_days": 7,
        "gap_reduction": 45,
        "risk": "low"
    },
    {
        "strategy": "Invoice Financing",
        "cost_rate": 0.025,
        "recovery_days": 3,
        "gap_reduction": 80,
        "risk": "medium"
    },
    {
        "strategy": "Working Capital Facility",
        "cost_rate": 0.040,
        "recovery_days": 5,
        "gap_reduction": 90,
        "risk": "high"
    },
    {
        "strategy": "Combined Strategy",
        "cost_rate": 0.015,
        "recovery_days": 6,
        "gap_reduction": 75,
        "risk": "medium"
    }
]


# 1. LOAD DATASET

def load_dataset():
    """Load and clean the invoice dataset."""

    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(
            f"Could not find {CSV_FILE}. "
            f"Make sure it is in the same folder as model_7.py."
        )

    df = pd.read_csv(CSV_FILE)

    # Date columns
    date_columns = [
        "issue_date",
        "due_date",
        "actual_paid_date"
    ]

    for column in date_columns:
        df[column] = pd.to_datetime(
            df[column],
            errors="coerce"
        )

    # Numeric columns
    numeric_columns = [
        "payment_term_days",
        "invoice_amount",
        "days_to_payment",
        "delay_vs_due_date"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Boolean columns
    boolean_columns = [
        "had_partial_payment_flag",
        "is_big_ticket_spike"
    ]

    for column in boolean_columns:
        df[column] = (
            df[column]
            .astype(str)
            .str.lower()
            .map({
                "true": True,
                "false": False
            })
            .fillna(False)
        )

    return df


# 2. VALIDATE DATASET

def validate_dataset(df):
    """Validate required dataset columns."""

    required_columns = [
        "invoice_id",
        "cust_number",
        "customer_name",
        "sector",
        "payment_term_days",
        "invoice_amount",
        "issue_date",
        "due_date",
        "status",
        "actual_paid_date",
        "days_to_payment",
        "delay_vs_due_date",
        "had_partial_payment_flag",
        "is_big_ticket_spike"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("Dataset validation: PASSED")


# 3. CUSTOMER PAYMENT BEHAVIOUR

def calculate_customer_metrics(df):
    """
    Calculate historical payment behaviour
    for every customer.
    """

    closed = df[df["status"] == "closed"].copy()

    if closed.empty:
        return pd.DataFrame()

    metrics = (
        closed
        .groupby("cust_number")
        .agg(
            invoice_count=("invoice_id", "count"),
            total_invoice_value=("invoice_amount", "sum"),
            average_invoice_value=("invoice_amount", "mean"),
            average_payment_delay=("delay_vs_due_date", "mean"),
            average_days_to_payment=("days_to_payment", "mean"),
            partial_payment_rate=(
                "had_partial_payment_flag",
                "mean"
            ),
            big_ticket_rate=(
                "is_big_ticket_spike",
                "mean"
            )
        )
        .reset_index()
    )

    metrics["partial_payment_rate"] *= 100
    metrics["big_ticket_rate"] *= 100

    return metrics


# 4. OVERALL DATASET BEHAVIOUR

def calculate_overall_metrics(df):
    """Calculate overall payment behaviour."""

    closed = df[df["status"] == "closed"].copy()

    if closed.empty:
        return {
            "average_payment_delay": 0.0,
            "partial_payment_rate": 0.0,
            "big_ticket_rate": 0.0
        }

    return {
        "average_payment_delay": round(
            float(closed["delay_vs_due_date"].mean()),
            2
        ),
        "partial_payment_rate": round(
            float(
                closed["had_partial_payment_flag"].mean()
                * 100
            ),
            2
        ),
        "big_ticket_rate": round(
            float(
                closed["is_big_ticket_spike"].mean()
                * 100
            ),
            2
        )
    }

# 5. LOAD MODEL 2 OUTPUT

def load_model_2_output():
    """
    Load Model 2 JSON if available.

    If it is not available, Model 7 can still run using
    the invoice dataset.
    """

    if not os.path.exists(MODEL_2_FILE):

        print(
            f"\n{MODEL_2_FILE} not found."
        )

        print(
            "Running Model 7 using dataset information only."
        )

        return {
            "forecast": [],
            "summary": {}
        }

    try:
        with open(
            MODEL_2_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        print(
            f"\nModel 2 input loaded from "
            f"{MODEL_2_FILE}"
        )

        return data

    except Exception as error:

        print(
            f"\nWarning: Could not read Model 2 output: "
            f"{error}"
        )

        return {
            "forecast": [],
            "summary": {}
        }


# 6. CALCULATE LIQUIDITY GAP

def calculate_liquidity_gap(model_2_data, df):
    """
    Determine the liquidity pressure that Model 7
    needs to address.
    """

    summary = model_2_data.get(
        "summary",
        {}
    )

    overdue_value = float(
        summary.get(
            "overdue_invoice_value",
            0
        ) or 0
    )

    overdue_count = int(
        summary.get(
            "overdue_invoice_count",
            0
        ) or 0
    )

    expected_min_cash = summary.get(
        "expected_min_cash"
    )

    days_to_breach = summary.get(
        "days_to_likely_breach"
    )

    # If Model 2 doesn't provide overdue information,
    # calculate it from the CSV.

    if overdue_value <= 0:

        open_invoices = df[
            df["status"].isin(
                ["open", "disputed_open"]
            )
        ]

        overdue_value = float(
            open_invoices["invoice_amount"].sum()
        )

        overdue_count = int(
            len(open_invoices)
        )

    return {
        "overdue_invoice_count": overdue_count,
        "overdue_invoice_value": round(
            overdue_value,
            2
        ),
        "expected_min_cash": expected_min_cash,
        "days_to_likely_breach": days_to_breach
    }


# 7. NORMALIZATION FUNCTIONS

def calculate_cost_score(cost, maximum_cost):
    """Lower cost receives a higher score."""

    if maximum_cost <= 0:
        return 100.0

    score = (
        1 - (cost / maximum_cost)
    ) * 100

    return max(
        0,
        min(score, 100)
    )


def calculate_speed_score(
    recovery_days,
    maximum_days
):
    """Faster recovery receives a higher score."""

    if maximum_days <= 0:
        return 100.0

    score = (
        1 - (
            recovery_days
            / maximum_days
        )
    ) * 100

    return max(
        0,
        min(score, 100)
    )


def calculate_liquidity_score(
    gap_reduction
):
    """Higher liquidity recovery receives a higher score."""

    return max(
        0,
        min(
            float(gap_reduction),
            100
        )
    )


def calculate_risk_score(risk):
    """Convert risk level into a transparent score."""

    risk_scores = {
        "low": 100.0,
        "medium": 60.0,
        "high": 20.0
    }

    return risk_scores.get(
        risk.lower(),
        50.0
    )

# 8. GENERATE EXPLANATION

def generate_explanation(
    strategy,
    cost_score,
    speed_score,
    liquidity_score,
    risk_score,
    risk
):
    """Create a human-readable explanation."""

    strongest_factor = max(
        [
            ("cost", cost_score),
            ("speed", speed_score),
            ("liquidity", liquidity_score),
            ("risk", risk_score)
        ],
        key=lambda item: item[1]
    )[0]

    explanations = {
        "cost": "It has a relatively favorable cost profile.",
        "speed": "It provides relatively fast liquidity recovery.",
        "liquidity": "It provides a strong reduction in the liquidity gap.",
        "risk": "It has a relatively low risk profile."
    }

    explanation = explanations[
        strongest_factor
    ]

    return (
        f"{strategy} received a strong overall ranking because "
        f"{explanation} Risk level is {risk}."
    )

# 9. RANK RECOVERY STRATEGIES

def rank_recovery_options(
    liquidity_data
):
    """
    Rank all recovery strategies using
    transparent weighted scoring.
    """

    overdue_value = liquidity_data[
        "overdue_invoice_value"
    ]

    maximum_days = max(
        option["recovery_days"]
        for option in RECOVERY_OPTIONS
    )

    # Calculate estimated cost for each option
    calculated_options = []

    for option in RECOVERY_OPTIONS:

        estimated_cost = (
            overdue_value
            * option["cost_rate"]
        )

        calculated_options.append({
            **option,
            "estimated_cost": estimated_cost
        })

    maximum_cost = max(
        option["estimated_cost"]
        for option in calculated_options
    )

    results = []

    for option in calculated_options:

        cost_score = calculate_cost_score(
            option["estimated_cost"],
            maximum_cost
        )

        speed_score = calculate_speed_score(
            option["recovery_days"],
            maximum_days
        )

        liquidity_score = calculate_liquidity_score(
            option["gap_reduction"]
        )

        risk_score = calculate_risk_score(
            option["risk"]
        )

        final_score = (
            cost_score * WEIGHTS["cost"]
            + speed_score * WEIGHTS["speed"]
            + liquidity_score * WEIGHTS["liquidity"]
            + risk_score * WEIGHTS["risk"]
        )

        explanation = generate_explanation(
            option["strategy"],
            cost_score,
            speed_score,
            liquidity_score,
            risk_score,
            option["risk"]
        )

        results.append({
            "rank": 0,
            "strategy": option["strategy"],
            "score": round(
                final_score,
                2
            ),
            "cost": {
                "amount": round(
                    option["estimated_cost"],
                    2
                ),
                "score": round(
                    cost_score,
                    2
                )
            },
            "recovery": {
                "days": option["recovery_days"],
                "speed_score": round(
                    speed_score,
                    2
                )
            },
            "liquidity": {
                "gap_reduction_pct": option[
                    "gap_reduction"
                ],
                "liquidity_score": round(
                    liquidity_score,
                    2
                )
            },
            "risk": {
                "level": option["risk"],
                "score": round(
                    risk_score,
                    2
                )
            },
            "explanation": explanation
        })

    # Highest score first
    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    # Assign rank
    for index, result in enumerate(
        results,
        start=1
    ):
        result["rank"] = index

    return results

# 10. CREATE FINAL OUTPUT

def create_output(
    recommendations,
    liquidity_data,
    customer_metrics,
    overall_metrics
):
    """Create the final Model 7 JSON output."""

    best = (
        recommendations[0]
        if recommendations
        else None
    )

    output = {
        "recommendations": recommendations,

        "summary": {
            "recommended_strategy": (
                best["strategy"]
                if best
                else None
            ),

            "recommended_score": (
                best["score"]
                if best
                else None
            ),

            "liquidity_gap": (
                liquidity_data[
                    "overdue_invoice_value"
                ]
            ),

            "overdue_invoice_count": (
                liquidity_data[
                    "overdue_invoice_count"
                ]
            ),

            "number_of_options_evaluated": len(
                recommendations
            ),

            "customers_analyzed": len(
                customer_metrics
            ),

            "average_payment_delay": (
                overall_metrics[
                    "average_payment_delay"
                ]
            ),

            "partial_payment_rate": (
                overall_metrics[
                    "partial_payment_rate"
                ]
            ),

            "big_ticket_rate": (
                overall_metrics[
                    "big_ticket_rate"
                ]
            )
        },

        "weights": WEIGHTS
    }

    return output

# 11. SAVE JSON

def save_output(output):
    """Save Model 7 results to JSON."""

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2
        )

    print(
        f"\nModel 7 output saved to "
        f"{OUTPUT_FILE}"
    )

# 12. MAIN PROGRAM

def main():

    print("\n")
    print("=" * 60)
    print("MODEL 7 - NON-DEBT-FIRST RECOMMENDATION RANKER")
    print("=" * 60)

    # Load dataset
    df = load_dataset()

    print(
        f"\nDataset loaded successfully!"
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    # Validate
    validate_dataset(df)

    # Invoice status
    print("\nInvoice status:")
    print(
        df["status"].value_counts()
    )

    # Customer metrics
    customer_metrics = (
        calculate_customer_metrics(df)
    )

    print(
        "\nCustomer behaviour analysis:"
    )

    print(
        f"Customers analyzed: "
        f"{len(customer_metrics)}"
    )

    # Overall behaviour
    overall_metrics = (
        calculate_overall_metrics(df)
    )

    print(
        "\nOverall payment behaviour:"
    )

    print(
        f"Average payment delay: "
        f"{overall_metrics['average_payment_delay']} days"
    )

    print(
        f"Partial payment rate: "
        f"{overall_metrics['partial_payment_rate']}%"
    )

    print(
        f"Big ticket rate: "
        f"{overall_metrics['big_ticket_rate']}%"
    )

    # Model 2
    model_2_data = (
        load_model_2_output()
    )

    # Liquidity
    liquidity_data = (
        calculate_liquidity_gap(
            model_2_data,
            df
        )
    )

    print(
        "\nLiquidity situation:"
    )

    print(
        f"Overdue invoice count: "
        f"{liquidity_data['overdue_invoice_count']}"
    )

    print(
        f"Overdue invoice value: "
        f"₹{liquidity_data['overdue_invoice_value']:,.2f}"
    )

    if liquidity_data[
        "days_to_likely_breach"
    ] is not None:

        print(
            f"Days to likely breach: "
            f"{liquidity_data['days_to_likely_breach']}"
        )

    # Rank strategies
    recommendations = (
        rank_recovery_options(
            liquidity_data
        )
    )

    # Create output
    output = create_output(
        recommendations,
        liquidity_data,
        customer_metrics,
        overall_metrics
    )

    # Save output
    save_output(output)

    # Display ranking
    print("\n")
    print("=" * 60)
    print("MODEL 7 RECOMMENDATIONS")
    print("=" * 60)

    for recommendation in recommendations:

        print(
            f"\n#{recommendation['rank']} "
            f"{recommendation['strategy']}"
        )

        print(
            f"Score: "
            f"{recommendation['score']}"
        )

        print(
            f"Estimated cost: "
            f"₹{recommendation['cost']['amount']:,.2f}"
        )

        print(
            f"Recovery time: "
            f"{recommendation['recovery']['days']} days"
        )

        print(
            f"Liquidity reduction: "
            f"{recommendation['liquidity']['gap_reduction_pct']}%"
        )

        print(
            f"Risk: "
            f"{recommendation['risk']['level']}"
        )

        print(
            f"Why: "
            f"{recommendation['explanation']}"
        )

    print("\n")
    print("=" * 60)
    print("MODEL 7 COMPLETED SUCCESSFULLY")
    print("=" * 60)

# PROGRAM START

if __name__ == "__main__":
    main()