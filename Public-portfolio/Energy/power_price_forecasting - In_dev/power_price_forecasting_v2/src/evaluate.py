"""Walk-forward evaluation and metrics.

Why walk-forward and not k-fold:
    Random k-fold cross validation leaks the future into the past. For time
    series the only honest validation is to train on a window ending at time
    T and predict times > T. The training window expands by one fold each
    iteration so the final fold uses the most data, mirroring how a model
    would be re-fit in production.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error, ignoring rows where |y_true| < 1 EUR/MWh.

    GB day-ahead prices occasionally settle near zero (high wind, low load).
    Dividing by tiny values blows up MAPE and gives a misleading score, so
    rows with |actual| < 1 are dropped from the calculation.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.abs(y_true) >= 1.0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def all_metrics(y_true, y_pred) -> dict[str, float]:
    return {"MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred), "MAPE_%": mape(y_true, y_pred)}


@dataclass
class Fold:
    train_idx: pd.DatetimeIndex
    test_idx: pd.DatetimeIndex


def expanding_window_folds(
    index: pd.DatetimeIndex, n_folds: int, test_horizon_days: int
) -> list[Fold]:
    """Build n_folds expanding-window folds, each with `test_horizon_days` of test data.

    Example with n_folds=3 and test_horizon_days=30 over a 1-year sample:

        Fold 1: train=[start,                t0],     test=[t0,   t0+30d]
        Fold 2: train=[start,                t0+30d], test=[t0+30d, t0+60d]
        Fold 3: train=[start,                t0+60d], test=[t0+60d, t0+90d]

    where t0 is chosen so the final test window ends at the index's last timestamp.
    """
    last = index.max()
    horizon = pd.Timedelta(days=test_horizon_days)
    folds: list[Fold] = []
    for i in range(n_folds):
        test_end = last - i * horizon
        test_start = test_end - horizon
        train_end = test_start
        train_idx = index[(index < train_end)]
        test_idx = index[(index >= test_start) & (index < test_end)]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        folds.append(Fold(train_idx=train_idx, test_idx=test_idx))
    folds.reverse()  # chronological order
    return folds


def walk_forward_evaluate(
    model_factory,
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[Fold],
) -> pd.DataFrame:
    """Train and score the model on every fold; return per-fold metrics.

    `model_factory` must be a zero-argument callable that returns a fresh
    model instance. Re-instantiating per fold prevents state from earlier
    folds bleeding into later ones.
    """
    rows = []
    for k, fold in enumerate(folds, start=1):
        X_tr, y_tr = X.loc[fold.train_idx], y.loc[fold.train_idx]
        X_te, y_te = X.loc[fold.test_idx], y.loc[fold.test_idx]
        model = model_factory()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        metrics = all_metrics(y_te, y_pred)
        metrics.update({"fold": k, "n_train": len(X_tr), "n_test": len(X_te)})
        rows.append(metrics)
    return pd.DataFrame(rows).set_index("fold")


def summarise(per_fold: pd.DataFrame) -> pd.Series:
    """Mean of the metric columns across folds."""
    metric_cols = [c for c in per_fold.columns if c not in {"n_train", "n_test"}]
    return per_fold[metric_cols].mean()
