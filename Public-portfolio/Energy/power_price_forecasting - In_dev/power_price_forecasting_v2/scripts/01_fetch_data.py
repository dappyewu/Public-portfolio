"""v2 fetch: Elexon power data + Yahoo Finance fuel prices."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_fetch import fetch_all, save_raw


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    market = cfg["market"]
    data_cfg = cfg["data"]
    fuel_cfg = cfg.get("fuel", {})

    print(f"Fetching v2 GB data {data_cfg['start']} -> {data_cfg['end']}")
    df = fetch_all(
        api_base=data_cfg["api_base"],
        start=data_cfg["start"],
        end=data_cfg["end"],
        delay_s=data_cfg.get("request_delay_s", 0.4),
        fuel_cfg=fuel_cfg,
        raw_dir=ROOT / data_cfg["raw_dir"],
    )
    print(f"\nGot {len(df):,} hourly rows. Columns: {list(df.columns)}")
    print(df.describe().round(1))

    out = save_raw(df, ROOT / data_cfg["raw_dir"], market["country_code"])
    print(f"Saved -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
