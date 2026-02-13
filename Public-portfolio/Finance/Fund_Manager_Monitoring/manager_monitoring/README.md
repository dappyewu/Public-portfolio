# Manager Monitoring

This project generates an **IC-style performance & risk report**  from:
- a manager/strategy **daily returns** CSV, and
- an optional **benchmark price** CSV.

For proper Sharpe/alpha with non-zero risk-free add a **daily risk-free return series** (`risk_free_daily.csv`).

It is Step 1 of a larger project:
1) reporting + rolling risk + monitoring flags
2) factor decomposition + style drift + stress tests + portfolio overlaps (next)

## Folder layout

- `data/`
  - `manager_returns.csv` (required)
  - `benchmark_prices.csv` (optional)
  - `risk_free_daily.csv` (recommended)
- `src/`
  - `report2.py` (main generator)
- `reports/` (generated)
  - `manager_report.docx`
  - `perf_stats.csv`, `perf_stats_formatted.csv`
  - `charts/` (PNG charts)

## Input formats

### `data/manager_returns.csv`
Must contain:
- `date` (YYYY-MM-DD)
- `return` (daily return as a decimal; e.g. `0.001` = +0.10%)

### `data/benchmark_prices.csv`
Must contain:
- `date` (YYYY-MM-DD)
- `price` (benchmark price level)

### `data/risk_free_daily.csv` 
Must contain:
- `date` (YYYY-MM-DD)
- `rf` (daily risk-free return as a decimal; e.g. `0.00015`)

Notes:
- Risk-free should be a **daily return** series (not an annualized yield). If you only have an annualized yield,
  convert it to a daily return first (roughly `annual_yield/252`).


## Quickstart

Create and activate a virtual environment, then:

```bash
pip install -r requirements.txt
python src/report2.py --returns data/manager_returns.csv --benchmark data/benchmark_prices.csv --risk_free data/risk_free_daily.csv --outdir reports --report_name manager_report.docx

```

The report includes:
- performance/risk summary metrics (via pyfolio)
- equity curve, drawdown, rolling vol, rolling Sharpe, rolling beta, return histogram
- **Monitoring Flags**: simple threshold-based alerts (drawdown, Sharpe deterioration, vol spike, beta regime shift)

## Monitoring flags (defaults)

You can tune thresholds from the CLI:

```bash
python src/report2.py --returns data/manager_returns.csv --benchmark data/benchmark_prices.csv --risk_free data/risk_free_daily.csv --outdir reports --report_name manager_report.docx --dd_warn -0.10 --dd_breach -0.20 --vol_breach 0.20 --sharpe_breach 0.0 --sharpe_lookback 126 --sharpe_breach_pct 0.80 --beta_shift_breach 0.30

```

Notes:
- Rolling Sharpe/alpha are computed on **excess returns** when `risk_free_daily.csv` is provided. If not provided, the scripts fall back to rf=0.
