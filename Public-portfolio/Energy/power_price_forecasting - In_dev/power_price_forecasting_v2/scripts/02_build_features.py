"""Build the model-ready feature matrix from the cached raw data.

Reads ``data/raw/GB_hourly.parquet`` (produced by ``01_fetch_data.py``),
applies calendar / fundamentals / lagged-price feature engineering, then
writes ``data/processed/features.parquet`` for the training step.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_fetch import load_raw
from src.features import build_feature_matrix, feature_columns


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    raw = load_raw(ROOT / cfg["data"]["raw_dir"], cfg["market"]["country_code"])

    feats = build_feature_matrix(
        raw,
        tz=cfg["market"]["timezone"],
        price_lags=cfg["features"]["price_lags"],
        rolling_windows=cfg["features"]["rolling_windows"],
        use_holidays=cfg["features"]["use_holidays"],
        holiday_country=cfg["features"]["holiday_country"],
    )

    cols = feature_columns(feats)
    print(f"Feature matrix: {len(feats):,} rows x {len(cols)} features")
    for c in cols:
        print(f"  - {c}")

    out_dir = ROOT / cfg["data"]["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "features.parquet"
    feats.to_parquet(out_path)
    print(f"Saved -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
