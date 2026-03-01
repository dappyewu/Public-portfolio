from __future__ import annotations

from dataclasses import asdict
import pandas as pd
import numpy as np

from .storage_dp import StorageParams, optimal_policy_perfect_foresight


def intrinsic_value_from_forward(
    forward_monthly: pd.DataFrame,
    params: StorageParams,
    start_date: str | pd.Timestamp,
    n_days: int,
) -> tuple[float, pd.DataFrame]:
    """Compute 'intrinsic' value using a deterministic daily curve built from monthly forwards.

    We expand monthly forward prices to daily by forward-filling within each month.
    This gives a clean, explainable baseline valuation.
    """

    fwd = forward_monthly.copy()
    if "delivery_month" in fwd.columns:
        col = "delivery_month"
        # accept either YYYY-MM or DD/MM/YYYY strings
        fwd[col] = fwd[col].astype(str).str.strip()
        dt_ym = pd.to_datetime(fwd[col], format="%Y-%m", errors="coerce")
        dt_dmy = pd.to_datetime(fwd[col], format="%d/%m/%Y", errors="coerce")
        fwd["contract_month"] = np.where(dt_ym.notna(), dt_ym, dt_dmy)
        fwd["contract_month"] = pd.to_datetime(fwd["contract_month"], errors="raise")
    elif "contract_month" in fwd.columns:
        col = "contract_month"
        # accept YYYY-MM and coerce to month start
        fwd["contract_month"] = pd.to_datetime(fwd[col].astype(str).str.strip(), errors="raise")
    else:
        raise ValueError("forward_monthly must have a 'delivery_month' or 'contract_month' column")

    # normalise to month-start timestamp (YYYY-MM-01)
    fwd["contract_month"] = fwd["contract_month"].dt.to_period("M").dt.to_timestamp()

    fwd = (
        fwd.sort_values("contract_month")
           .groupby("contract_month", as_index=False)["fwd_eur_mwh"]
           .last()
    )
    fwd = fwd.set_index("contract_month")

    dates = pd.date_range(pd.to_datetime(start_date), periods=n_days, freq="D")
    month_start = dates.to_period("M").to_timestamp()
    daily_prices = pd.Series(month_start, index=dates).map(fwd["fwd_eur_mwh"]).astype(float)
    # if curve shorter than horizon, carry last observed forward price forward
    daily_prices = daily_prices.ffill()
    if daily_prices.isna().any():
        missing = pd.Series(month_start[daily_prices.isna()]).unique()
        raise ValueError(f"Still missing forward prices for months: {missing[:10]}")

    # grid_size=200
    value, policy = optimal_policy_perfect_foresight(daily_prices, params=params, grid_size=200)
    return value, policy


def monte_carlo_upper_bound(
    paths: pd.DataFrame,
    dates: pd.DatetimeIndex,
    params: StorageParams,
    max_paths: int = 200,
) -> pd.DataFrame:
    """Compute a Monte Carlo estimate of an UPPER BOUND on extrinsic value.

    For each simulated path, run perfect-foresight DP. Average the path NPVs.


    """
    n_paths = min(paths.shape[1], max_paths)
    vals = []
    for i in range(n_paths):
        s = pd.Series(paths.iloc[:, i].values, index=dates, name=f"path_{i}")
        v, _ = optimal_policy_perfect_foresight(s, params=params, grid_size=200)
        vals.append(v)

    out = pd.DataFrame({"path": range(n_paths), "npv_eur": vals})
    out["npv_eur_mean"] = out["npv_eur"].mean()
    out["npv_eur_p50"] = out["npv_eur"].median()
    out["npv_eur_p10"] = out["npv_eur"].quantile(0.10)
    out["npv_eur_p90"] = out["npv_eur"].quantile(0.90)
    return out
