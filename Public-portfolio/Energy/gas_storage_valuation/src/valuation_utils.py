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
    fwd["contract_month"] = pd.to_datetime(fwd["contract_month"] + "-01")
    fwd = fwd.sort_values("contract_month").set_index("contract_month")

    dates = pd.date_range(pd.to_datetime(start_date), periods=n_days, freq="D")
    month_starts = dates.to_period("M").to_timestamp()

    # map each day to its contract month
    daily_prices = pd.Series(index=dates, dtype=float)
    for d in dates:
        m = d.to_period("M").to_timestamp()
        if m in fwd.index:
            daily_prices.loc[d] = float(fwd.loc[m, "fwd_eur_mwh"])
        else:
            # if beyond curve, use last known
            daily_prices.loc[d] = float(fwd["fwd_eur_mwh"].iloc[-1])

    value, policy = optimal_policy_perfect_foresight(daily_prices, params=params, grid_size=250)
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
