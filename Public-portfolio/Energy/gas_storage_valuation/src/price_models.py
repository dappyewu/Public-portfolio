from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd


@dataclass
class OUParams:
    """Ornstein-Uhlenbeck parameters for log-price X_t."""
    kappa: float          # mean reversion speed (per day)
    theta: float          # long-run mean of X
    sigma: float          # volatility of X (per sqrt(day))


def fit_ou_params(log_prices: pd.Series) -> OUParams:
    """Fit OU parameters via discrete-time regression:

        X_{t+1} = a + b X_t + eps
        => kappa = -ln(b)
        => theta = a/(1-b)
        => sigma from residual std

    Assumes daily spacing.
    """
    x = log_prices.dropna().astype(float).values
    if len(x) < 30:
        raise ValueError("Need at least 30 observations to fit OU params.")

    x_t = x[:-1]
    x_tp1 = x[1:]

    # OLS: x_tp1 = a + b*x_t
    b = np.cov(x_t, x_tp1, bias=True)[0, 1] / np.var(x_t)
    a = x_tp1.mean() - b * x_t.mean()

    # numeric safety
    b = float(np.clip(b, 1e-6, 0.999999))

    kappa = -np.log(b)  # per day
    theta = a / (1.0 - b)

    resid = x_tp1 - (a + b * x_t)
    sigma = float(np.std(resid, ddof=1))

    return OUParams(kappa=kappa, theta=float(theta), sigma=sigma)


def simulate_ou_paths(
    *,
    s0: float,
    params: OUParams,
    n_days: int,
    n_paths: int,
    seed: int = 7,
    seasonal_theta: pd.Series | None = None,
) -> pd.DataFrame:
    """Simulate OU log-price paths and return price paths.

    If seasonal_theta is provided, it should be a Series of length n_days with
    long-run mean (theta_t) for each day (log-price units). This lets you
    impose a seasonal forward-curve-like mean level.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0  # 1 day

    x = np.empty((n_days, n_paths), dtype=float)
    x[0, :] = np.log(float(s0))

    kappa, theta, sigma = params.kappa, params.theta, params.sigma

    if seasonal_theta is None:
        theta_t = np.full(n_days, theta, dtype=float)
    else:
        if len(seasonal_theta) != n_days:
            raise ValueError("seasonal_theta must have length n_days")
        theta_t = seasonal_theta.values.astype(float)

    for t in range(1, n_days):
        eps = rng.normal(0.0, 1.0, size=n_paths)
        x[t, :] = x[t-1, :] + kappa * (theta_t[t] - x[t-1, :]) * dt + sigma * np.sqrt(dt) * eps

    prices = np.exp(x)
    cols = [f"path_{i:04d}" for i in range(n_paths)]
    return pd.DataFrame(prices, columns=cols)


def build_seasonal_theta(
    dates: pd.DatetimeIndex,
    spot_history: pd.Series | pd.DataFrame,
    price_col: str = "spot_eur_mwh",
    use_log: bool = True,
    fill_method: str = "overall_mean",
) -> pd.Series:
    """
    Build seasonal theta(t) using monthly averages from historical spot data.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Dates you want theta for (e.g. simulation/valuation daily horizon).
    spot_history : pd.Series or pd.DataFrame
        Historical spot prices used to estimate seasonality.
        - If Series: index should be datetime-like and values are prices (or log-prices if use_log=False and already logged).
        - If DataFrame: must contain `price_col` and have a DatetimeIndex or a 'date' column.
    price_col : str
        Column name to use if spot_history is a DataFrame.
    use_log : bool
        If True, theta is computed in log space: log(price).
    fill_method : str
        What to do if some months are missing in history:
        - "overall_mean": fill missing months with the overall mean of the history (in same space: log or price)
        - "ffill_bfill": fill missing months by forward/backward filling the monthly means

    Returns
    -------
    pd.Series
        theta_log indexed by `dates` with name="theta_log".
    """
    # --- Coerce spot_history into a Series with DatetimeIndex ---
    if isinstance(spot_history, pd.DataFrame):
        if isinstance(spot_history.index, pd.DatetimeIndex):
            s = spot_history[price_col].copy()
        elif "date" in spot_history.columns:
            tmp = spot_history.copy()
            tmp["date"] = pd.to_datetime(tmp["date"])
            s = tmp.set_index("date")[price_col].copy()
        else:
            raise ValueError("spot_history DataFrame must have a DatetimeIndex or a 'date' column.")
    else:
        s = spot_history.copy()
        if not isinstance(s.index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index)

    s = s.sort_index().dropna()

    if use_log:
        # guard against non-positive prices
        if (s <= 0).any():
            raise ValueError("Spot history contains non-positive prices; cannot take log.")
        x = np.log(s)
    else:
        x = s.astype(float)

    # --- Compute monthly means from history (month-of-year seasonality) ---
    monthly_means = x.groupby(x.index.month).mean()  # index = 1..12

    # --- Handle missing months robustly ---
    all_months = pd.Index(range(1, 13), name="month")
    monthly_means = monthly_means.reindex(all_months)

    if monthly_means.isna().any():
        if fill_method == "overall_mean":
            monthly_means = monthly_means.fillna(x.mean())
        elif fill_method == "ffill_bfill":
            monthly_means = monthly_means.ffill().bfill()
        else:
            raise ValueError("fill_method must be 'overall_mean' or 'ffill_bfill'.")

    # --- Map requested dates to month-of-year mean ---
    dates = pd.DatetimeIndex(dates)
    theta_vals = monthly_means.loc[dates.month].to_numpy()

    theta = pd.Series(theta_vals, index=dates, name="theta_log")
    return theta
