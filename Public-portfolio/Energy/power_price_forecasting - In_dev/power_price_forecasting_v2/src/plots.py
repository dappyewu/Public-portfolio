"""Diagnostic plots used by the training script and the walkthrough notebook.

Each plot answers a specific diagnostic question:

    plot_actual_vs_predicted   "Does it track the level day-to-day?"
    plot_residuals_by_hour     "Is the error random or biased by time of day?"
    plot_feature_importance    "What's actually driving the forecast?"
    plot_price_vs_residual_load "Does the merit-order story hold in the data?"
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="talk")


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_actual_vs_predicted(
    y_true: pd.Series,
    predictions: dict[str, np.ndarray],
    out_path: str | Path,
    days: int = 14,
) -> Path:
    """Overlay actual prices with each model's prediction over the last `days`."""
    cutoff = y_true.index.max() - pd.Timedelta(days=days)
    mask = y_true.index >= cutoff

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y_true.index[mask], y_true.values[mask], label="Actual", color="black", linewidth=2)
    palette = sns.color_palette("viridis", n_colors=len(predictions))
    for (name, pred), colour in zip(predictions.items(), palette):
        pred = pd.Series(pred, index=y_true.index)
        ax.plot(pred.index[mask], pred.values[mask], label=name, color=colour, alpha=0.8)
    ax.set_title(f"Day-ahead price: actual vs predicted (last {days} days of test set)")
    ax.set_ylabel("Price (EUR/MWh)")
    ax.set_xlabel("")
    ax.legend(loc="upper left")
    return _save(fig, out_path)


def plot_residuals_by_hour(
    y_true: pd.Series, y_pred: np.ndarray, out_path: str | Path, model_name: str
) -> Path:
    residuals = pd.Series(y_true.values - y_pred, index=y_true.index)
    by_hour = residuals.groupby(residuals.index.hour)
    summary = pd.DataFrame({"mean": by_hour.mean(), "std": by_hour.std()})

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(summary.index, summary["mean"], yerr=summary["std"], capsize=3, color="#4c72b0")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(f"{model_name}: mean residual by hour of day (error bars = std)")
    ax.set_xlabel("Hour of day (local time)")
    ax.set_ylabel("Actual - Predicted (EUR/MWh)")
    return _save(fig, out_path)


def plot_feature_importance(
    importances: pd.Series, out_path: str | Path, top_n: int = 15
) -> Path:
    top = importances.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top.index, top.values, color="#55a868")
    ax.set_title(f"LightGBM feature importance (top {top_n})")
    ax.set_xlabel("Importance (gain)")
    return _save(fig, out_path)


def plot_price_vs_residual_load(
    df: pd.DataFrame, out_path: str | Path
) -> Path:
    """Scatter price vs residual load, coloured by hour. The 'merit order' chart."""
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(
        df["residual_load"] / 1000,
        df["day_ahead_price"],
        c=df.index.hour,
        cmap="twilight",
        alpha=0.4,
        s=8,
    )
    plt.colorbar(sc, ax=ax, label="Hour of day")
    ax.set_xlabel("Residual load = load - wind - solar (GW)")
    ax.set_ylabel("Day-ahead price (EUR/MWh)")
    ax.set_title("Empirical merit-order curve")
    return _save(fig, out_path)


def plot_metric_table(
    summary: pd.DataFrame, out_path: str | Path
) -> Path:
    """Render the per-model metric summary as a figure (handy for the README)."""
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(summary) + 1))
    ax.axis("off")
    rounded = summary.round(2)
    table = ax.table(
        cellText=rounded.values,
        colLabels=rounded.columns,
        rowLabels=rounded.index,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    ax.set_title("Walk-forward CV: mean metrics by model")
    return _save(fig, out_path)
