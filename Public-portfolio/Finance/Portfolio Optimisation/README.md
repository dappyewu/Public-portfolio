# Portfolio Optimisation & Backtesting

A lightweight portfolio allocation + backtesting pipeline. It ingests a simple “tickers + amounts” CSV, pulls price history, computes portfolio stats, and generates backtest outputs (and optional reports).

Produces data-driven investment allocation recommendations to optimize risk-adjusted performance, primarily using quantitative methods with limited fundamental inputs. Historical performance is not indicative of future returns

---



## Quick start

1) **Add your portfolio**
- Edit `data/raw/Template_input.csv`
- Save as `data/raw/input.csv` (same columns, your tickers + amounts)

2) **Set your FRED API key (required)**
Create a `.env` file in the repo root (same level as `requirements.txt`) and add:

```
FRED_API_KEY=YOUR_KEY_HERE
```

3) **Install + run**
```bash
# Create & activate a virtual environment (recommended)
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the main pipeline - You can set the end_Date on the  TransformAndLoad.py file if required
python src/TransformAndLoad.py
```

Outputs are written to `data/processed/` (gitignored).

---

## What the pipeline does

- Loads a **portfolio input CSV** (tickers + amounts)
- Downloads historical prices via **yfinance**
- Cleans/filters tickers with limited history
- Computes key stats (return, volatility, Sharpe)
- Runs a backtest and writes summary outputs

---

## Project structure

```
Optimisation/
├─ src/
│  ├─ IngestRawData.py         # (Optional) load input CSV and build weights
│  ├─ TransformAndLoad.py      # main pipeline (download, clean, compute, outputs)
│  ├─ risk_kit.py              # portfolio math helper functions
│  ├─ wealth.py / wealthcompare.py  # (Optional) wealth comparisons
│  └─ ReportGenerator.py       # (Optional) Word summary report from outputs
├─ data/
│  ├─ raw/
│  │  ├─ Template_input.csv
│  │  └─ input.csv             # your inputs (edit for your portfolio)
│  └─ processed/               # generated outputs (gitignored)
├─ requirements.txt
└─ .env                        # contains FRED_API_KEY (do not commit)
```

---

## Optional scripts

```bash
# Wealth comparison outputs (after running TransformAndLoad)
python src/wealthcompare.py

# Generate a Word summary report (after running TransformAndLoad)
python src/ReportGenerator.py
```

---

## Typical outputs (in `data/processed/`)

- Allocation weights CSV
- Compunded Annual Growth
- Rolling Sharpe Ratio (1 year and 36months)
- Efficient frontier plot
- Drawdown summaries
- Backtest summary tables
- Optional wealth comparison outputs
- Optional Word report (`Backtest_Summary_Report_*.docx`)
