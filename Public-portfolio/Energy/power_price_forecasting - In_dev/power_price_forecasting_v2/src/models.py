"""Models, in increasing order of sophistication.

The point of stacking three models is pedagogical: each step has to *earn*
the next. If LightGBM only marginally beats a linear regression, the extra
complexity is not justified. If the linear regression only marginally beats
the seasonal naive forecast, the features are not adding signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb


class PriceModel(Protocol):
    name: str

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PriceModel": ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


@dataclass
class SeasonalNaive:
    """Predict price(t) = price(t - 168h). Same hour, same weekday, last week.

    This is the canonical baseline for hourly electricity prices because it
    captures the dominant intraday and intraweek seasonality for free.
    """

    name: str = "seasonal_naive_168h"
    season_hours: int = 168
    _history: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SeasonalNaive":
        # Only the target's history is needed to project forward.
        self._history = y.copy()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # For each row in X, look up the y value from `season_hours` ago.
        # If the history doesn't reach that far back, fall back to the mean.
        if self._history is None:
            raise RuntimeError("Call fit() first.")
        idx = X.index - pd.Timedelta(hours=self.season_hours)
        out = self._history.reindex(idx).to_numpy()
        out = np.where(np.isnan(out), self._history.mean(), out)
        return out


@dataclass
class LinearModel:
    """Ridge regression with standardised features.

    Ridge is preferred over OLS because the lag features and rolling means
    are highly collinear; a small amount of L2 keeps coefficients stable.
    """

    name: str = "ridge"
    alpha: float = 1.0
    _pipe: Pipeline | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearModel":
        self._pipe = Pipeline(
            [("scaler", StandardScaler()), ("ridge", Ridge(alpha=self.alpha))]
        )
        self._pipe.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._pipe is None:
            raise RuntimeError("Call fit() first.")
        return self._pipe.predict(X)

    def coefficients(self, feature_names: list[str]) -> pd.Series:
        if self._pipe is None:
            raise RuntimeError("Call fit() first.")
        return pd.Series(self._pipe.named_steps["ridge"].coef_, index=feature_names).sort_values()


@dataclass
class LightGBMModel:
    """Gradient-boosted trees. Captures non-linearities and feature interactions.

    Notable interactions in this market:
      - residual_load * hour-of-day (peak hours have steeper merit-order curves)
      - wind_forecast * is_weekend (weekend prices are more wind-sensitive)
    A linear model can't represent these without hand-crafted interaction terms.
    """

    name: str = "lightgbm"
    params: dict | None = None
    _model: lgb.LGBMRegressor | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMModel":
        params = self.params or {}
        self._model = lgb.LGBMRegressor(**params, verbose=-1)
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        return self._model.predict(X)

    def feature_importance(self, feature_names: list[str]) -> pd.Series:
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        return pd.Series(self._model.feature_importances_, index=feature_names).sort_values(ascending=False)


def build_models(lgb_params: dict) -> list[PriceModel]:
    """Return the standard model line-up used by the train/evaluate script."""
    return [
        SeasonalNaive(),
        LinearModel(alpha=1.0),
        LightGBMModel(params=lgb_params),
    ]
