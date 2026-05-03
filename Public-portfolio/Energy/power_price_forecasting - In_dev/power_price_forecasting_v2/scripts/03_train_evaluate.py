"""Train all three models with walk-forward CV, score them, save figures.

End-to-end runner that reads the processed feature matrix, evaluates the
seasonal-naive baseline, ridge, and LightGBM with expanding-window CV,
then writes diagnostic plots and per-fold metrics. Output:

    reports/figures/actual_vs_predicted.png
    reports/figures/residuals_by_hour_lightgbm.png
    reports/figures/feature_importance_lightgbm.png
    reports/figures/merit_order.png
    reports/figures/metric_table.png
    stdout: per-fold and mean metrics for each model
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import expanding_window_folds, walk_forward_evaluate, summarise
from src.features import feature_columns
from src.models import LightGBMModel, LinearModel, SeasonalNaive
from src.plots import (
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_metric_table,
    plot_price_vs_residual_load,
    plot_residuals_by_hour,
)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    feats = pd.read_parquet(ROOT / cfg["data"]["processed_dir"] / "features.parquet")
    cols = feature_columns(feats)
    X, y = feats[cols], feats["day_ahead_price"]

    folds = expanding_window_folds(
        feats.index,
        n_folds=cfg["model"]["cv_folds"],
        test_horizon_days=cfg["model"]["test_horizon_days"] // cfg["model"]["cv_folds"],
    )
    print(f"Built {len(folds)} expanding-window folds.")
    for k, f in enumerate(folds, 1):
        print(f"  Fold {k}: train={len(f.train_idx):,}h, test={len(f.test_idx):,}h "
              f"({f.test_idx.min()} -> {f.test_idx.max()})")

    factories = {
        "seasonal_naive_168h": lambda: SeasonalNaive(),
        "ridge": lambda: LinearModel(alpha=1.0),
        "lightgbm": lambda: LightGBMModel(params=cfg["model"]["lightgbm"]),
    }

    summary_rows = {}
    print("\n=== Walk-forward CV ===")
    for name, factory in factories.items():
        per_fold = walk_forward_evaluate(factory, X, y, folds)
        mean = summarise(per_fold)
        summary_rows[name] = mean
        print(f"\n{name}")
        print(per_fold.round(2))
        print("  mean:", mean.round(2).to_dict())

    summary = pd.DataFrame(summary_rows).T
    print("\n=== Summary ===")
    print(summary.round(2))

    # Final fit on all data except the last test_horizon_days, score on the held-out tail.
    final_test = folds[-1]
    print(f"\nFinal hold-out test window: "
          f"{final_test.test_idx.min()} -> {final_test.test_idx.max()}")

    X_tr, y_tr = X.loc[final_test.train_idx], y.loc[final_test.train_idx]
    X_te, y_te = X.loc[final_test.test_idx], y.loc[final_test.test_idx]

    final_predictions = {}
    for name, factory in factories.items():
        m = factory().fit(X_tr, y_tr)
        final_predictions[name] = m.predict(X_te)

    # Plots ------------------------------------------------------------
    fig_dir = ROOT / cfg["output"]["figures_dir"]
    plot_actual_vs_predicted(y_te, final_predictions, fig_dir / "actual_vs_predicted.png", days=14)

    # LightGBM diagnostics
    lgbm = LightGBMModel(params=cfg["model"]["lightgbm"]).fit(X_tr, y_tr)
    plot_residuals_by_hour(y_te, lgbm.predict(X_te), fig_dir / "residuals_by_hour_lightgbm.png", "LightGBM")
    plot_feature_importance(lgbm.feature_importance(cols), fig_dir / "feature_importance_lightgbm.png")

    plot_price_vs_residual_load(feats, fig_dir / "merit_order.png")
    plot_metric_table(summary, fig_dir / "metric_table.png")

    print(f"\nFigures written to {fig_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
