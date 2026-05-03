"""Feature engineering for day-ahead price forecasting.

Three families of features, each motivated by something a power analyst
would already believe about how this market works:

  1. Calendar features - prices follow strong daily, weekly, and seasonal
     cycles driven by working patterns and heating/cooling demand.

  2. Fundamental features - the merit-order curve says price is a function
     of demand minus zero-marginal-cost generation (residual load). The
     pipeline derives that, plus the renewable-share normalised version.

  3. Lagged price features - prices are autocorrelated. Yesterday's price
     and last-week-same-hour are strong baselines that any honest model
     must beat.

The fuel layer (``ttf_gas`` and ``eua_carbon``) already comes in lagged-by-
one-day from data_fetch.fetch_fuel_prices, so it can be added as a plain
feature here without any further shifting.

All target lags use SHIFTED values so future information never leaks into
the training set.
"""

from __future__ import annotations

from typing import Iterable

import holidays
import numpy as np
import pandas as pd


def add_calendar_features(
    df: pd.DataFrame,
    tz: str,
    use_holidays: bool = True,
    holiday_country: str = "GB",
) -> pd.DataFrame:
    """Add hour-of-day, day-of-week, month, and holiday flags."""
    out = df.copy()
    local = out.index.tz_convert(tz)

    out["hour"] = local.hour
    out["dayofweek"] = local.dayofweek
    out["month"] = local.month
    out["is_weekend"] = (local.dayofweek >= 5).astype(int)

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    if use_holidays:
        country_holidays = holidays.country_holidays(
            holiday_country,
            years=range(local.year.min(), local.year.max() + 1),
        )
        out["is_holiday"] = pd.Series(local.date, index=out.index).isin(country_holidays).astype(int)
    else:
        out["is_holiday"] = 0

    return out


def add_fundamental_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive merit-order-driven features from load, wind, solar.

    residual_load = load - wind - solar
        The slice of demand that thermal/imports must serve. The single
        most predictive engineered feature for short-term prices in any
        wind-heavy market.

    renewable_share = (wind + solar) / load
        Useful when the model needs a normalised view rather than absolute MW.
    """
    out = df.copy()
    out["residual_load"] = out["load_forecast"] - out["wind_forecast"] - out["solar_forecast"]
    out["renewable_share"] = (out["wind_forecast"] + out["solar_forecast"]) / out["load_forecast"]
    return out


def add_lag_features(
    df: pd.DataFrame,
    target: str,
    lags: Iterable[int],
    rolling_windows: Iterable[int],
) -> pd.DataFrame:
    """Add lagged target values and rolling statistics.

    Lags below 24h are intentionally excluded: at auction time (~11:00 the
    day before delivery), the most recent observed price is from earlier
    the same day, so single-hour lags would peek at unavailable info.
    """
    out = df.copy()
    for lag in lags:
        out[f"{target}_lag_{lag}h"] = out[target].shift(lag)
    for window in rolling_windows:
        shifted = out[target].shift(24)
        out[f"{target}_rollmean_{window}h"] = shifted.rolling(window).mean()
        out[f"{target}_rollstd_{window}h"] = shifted.rolling(window).std()
    return out


def build_feature_matrix(
    df: pd.DataFrame,
    tz: str,
    price_lags: Iterable[int],
    rolling_windows: Iterable[int],
    use_holidays: bool = True,
    holiday_country: str = "GB",
) -> pd.DataFrame:
    """Run all three feature families and drop rows with NaNs from the lags."""
    out = add_calendar_features(df, tz=tz, use_holidays=use_holidays, holiday_country=holiday_country)
    out = add_fundamental_features(out)
    out = add_lag_features(out, "day_ahead_price", price_lags, rolling_windows)
    out = out.dropna()
    return out


FEATURE_COLUMNS = [
    "load_forecast",
    "wind_forecast",
    "solar_forecast",
    "residual_load",
    "renewable_share",
    # v2 additions: fuel-cost drivers. Proxy CCGT marginal cost,
    # which sets the GB clearing price in most hours. Lagged 1 day
    # in data_fetch.fetch_fuel_prices so no auction-time leakage.
    "ttf_gas",
    "eua_carbon",
    "hour",
    "dayofweek",
    "month",
    "is_weekend",
    "is_holiday",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of model input columns present in the frame."""
    cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    cols += [c for c in df.columns if c.startswith("day_ahead_price_lag_")]
    cols += [c for c in df.columns if c.startswith("day_ahead_price_rollmean_")]
    cols += [c for c in df.columns if c.startswith("day_ahead_price_rollstd_")]
    return cols
