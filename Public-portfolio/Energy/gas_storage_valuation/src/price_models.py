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
    base_price: float = 30.0,
    winter_amplitude: float = 0.5,
    phase_shift_days: int = 15,
) -> pd.Series:
    """Create a simple seasonal long-run mean for log-price.

    winter_amplitude controls how much higher winter is than summer (in log terms).
    """
    doy = dates.dayofyear.to_numpy()
    season = winter_amplitude * np.cos(2 * np.pi * (doy - phase_shift_days) / 365.25)
    theta = np.log(base_price) + season
    return pd.Series(theta, index=dates, name="theta_log")
