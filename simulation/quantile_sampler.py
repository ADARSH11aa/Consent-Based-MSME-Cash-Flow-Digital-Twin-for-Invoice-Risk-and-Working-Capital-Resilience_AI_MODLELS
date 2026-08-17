"""
Converts a per-invoice P10/P50/P90 payment-day forecast (Model 1's output shape)
into full samples, via linear interpolation of the inverse CDF.

Model 1 output contract (per invoice):
    {
        "invoice_id": "INV-1052",
        "customer_id": "C001",
        "p10_payment_days": 44,
        "p50_payment_days": 52,
        "p90_payment_days": 68
    }
"""
import numpy as np


def sample_payment_days(p10, p50, p90, n_sims, rng):
    """
    Draws n_sims samples of "days to payment" for ONE invoice, consistent with
    its P10/P50/P90 forecast.

    Method: piecewise-linear inverse-CDF.
      - known points: (0.10, p10), (0.50, p50), (0.90, p90)
      - tails (below 0.10 and above 0.90) extrapolate linearly using the slope
        of the nearest segment, then are floored at 0 (can't be paid before issue).

    This makes no assumption about the distribution's shape (normal, skewed,
    etc.) beyond what the three quantiles themselves imply.
    """
    # guard against a model output that isn't monotonic (data/model bug) -
    # clip so p10 <= p50 <= p90 rather than silently producing nonsense samples
    p10, p50, p90 = sorted([p10, p50, p90])

    quantile_points = np.array([0.10, 0.50, 0.90])
    value_points = np.array([p10, p50, p90], dtype=float)

    u = rng.uniform(0, 1, size=n_sims)
    samples = np.interp(u, quantile_points, value_points)

    # extrapolate the lower tail (u < 0.10) using the 0.10-0.50 slope
    lower_slope = (p50 - p10) / (0.50 - 0.10)
    lower_mask = u < 0.10
    samples[lower_mask] = p10 + (u[lower_mask] - 0.10) * lower_slope

    # extrapolate the upper tail (u > 0.90) using the 0.50-0.90 slope
    upper_slope = (p90 - p50) / (0.90 - 0.50)
    upper_mask = u > 0.90
    samples[upper_mask] = p90 + (u[upper_mask] - 0.90) * upper_slope

    return np.clip(samples, 0, None)


if __name__ == "__main__":
    # Quick sanity check using the exact example from Model 1's contract
    rng = np.random.default_rng(42)
    samples = sample_payment_days(p10=44, p50=52, p90=68, n_sims=20000, rng=rng)

    print("Input:  p10=44, p50=52, p90=68")
    print(f"Output: p10={np.percentile(samples,10):.1f}, "
          f"p50={np.percentile(samples,50):.1f}, "
          f"p90={np.percentile(samples,90):.1f}")
    print(f"Min: {samples.min():.1f}  Max: {samples.max():.1f}")