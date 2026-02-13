import argparse
from pathlib import Path
import pandas as pd
import pyfolio as pf

def load_returns(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").set_index("date")
    rets = df["return"].astype(float).dropna()
    return rets

def load_benchmark_prices(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").set_index("date")
    prices = df["price"].astype(float)
    return prices.pct_change().dropna()


def load_risk_free_daily(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").set_index("date")
    rf = df["rf"].astype(float).dropna()
    return rf

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--returns", default="data/manager_returns.csv")
    p.add_argument("--benchmark", default="data/benchmark_prices.csv")
    p.add_argument("--risk_free", default="data/risk_free_daily.csv")
    p.add_argument("--outdir", default="reports")
    args = p.parse_args()

    returns_raw = load_returns(Path(args.returns))

    benchmark_rets_raw = None
    bench_path = Path(args.benchmark)
    if bench_path.exists():
        benchmark_rets_raw = load_benchmark_prices(bench_path)

    rf = None
    rf_path = Path(args.risk_free)
    if rf_path.exists():
        rf = load_risk_free_daily(rf_path)

    # align
    common_idx = returns_raw.index
    if benchmark_rets_raw is not None:
        common_idx = common_idx.intersection(benchmark_rets_raw.index)
    if rf is not None:
        common_idx = common_idx.intersection(rf.index)

    returns_raw = returns_raw.reindex(common_idx)
    if benchmark_rets_raw is not None:
        benchmark_rets_raw = benchmark_rets_raw.reindex(common_idx)
    if rf is not None:
        rf = rf.reindex(common_idx)

    # excess returns (for risk-adjusted metrics)
    returns = returns_raw - rf if rf is not None else returns_raw
    benchmark_rets = (benchmark_rets_raw - rf) if (benchmark_rets_raw is not None and rf is not None) else benchmark_rets_raw

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Save stats as CSV
    if benchmark_rets is not None:
        stats = pf.timeseries.perf_stats(returns, factor_returns=benchmark_rets)
    else:
        stats = pf.timeseries.perf_stats(returns)

    stats.to_csv(outdir / "perf_stats.csv", header=False)

    # Minimal plot export example (you can expand later)
    # For now: just confirm the script ran
    print("Saved:", outdir / "perf_stats.csv")
    print(stats)

if __name__ == "__main__":
    main()
