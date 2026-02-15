# gas_storage_valuation 

## Important

This project is a portfolio/learning implementation of gas storage valuation and optimisation. It is designed to showcase data handling, modelling structure, and decision logic  and not to represent a productiongrade valuatin system.

Some of the Limitation are
-  results are based on a simplified forward-curve-to-daily price series.
- The Monte Carlo experiment uses perfect-foresight optimisation on each simulated path, which provides an optimistic upper bound on optionality/extrinsic value rather than a tradable strategy.
- Several real-world features are intentionally omitted for scope (transport/basis constraints, bid–offer/market impact, risk-neutral calibration, robust testing).


## Flow of project
Value a natural gas storage facility using:
- a very simple **mean-reverting price model** ,
- **Monte Carlo simulation** of future price paths,
- **dynamic programming (DP)** to choose inject / withdraw / hold subject to constraints.


---



## Data

Sample data is included  
- `data/raw/spot_history_ttf_like.csv` – daily “TTF-like” spot history
- `data/raw/forward_curve_monthly.csv` – monthly forward curve (18 months)


---

## Method notes 

**Intrinsic value**  
Uses a deterministic daily curve built from monthly forwards. DP picks the best inject/withdraw schedule given that curve.

**Extrinsic value **  
We simulate many price paths, and for each path we solve the DP with *perfect foresight* over that path.
That produces an **upper bound** on true option value ( future prices in real life are unknown).
This still demonstrates:
- constraint handling
- DP mechanics
- how volatility increases optionality

---

## Project structure

- `src/price_models.py` – OU fit + simulation + seasonal mean
- `src/storage_dp.py` – DP for optimal policy (inject/withdraw/hold)
- `src/valuation_utils.py` – intrinsic valuation + MC upper bound wrapper
- `src/run_valuation.py` – CLI entry point


## additional repo info 
- Output CSVs in `results/` are generated when you run the pipeline
- Notebooks in `notebooks/` are runnable walkthroughs for the included sample data. 
- For the notebooks, you will need to update the column name on the forward curve file to 'delivery_month'. When running the CLI(python -m src.run_valuation --n_days 365 --n_paths 200) the column name should be 'contract_month'



## Quick start
- Use notebooks first, as CLI is still in development and has known issues -  Make sure to update column name on forward curve file to delivery_month

```bash
pip install -r requirements.txt
python -m src.run_valuation --n_days 365 --n_paths 200
```

Outputs are written to `results/`:
- `optimal_policy_intrinsic.csv` – the inject/withdraw schedule on the deterministic curve
- `storage_value_mc_upper_bound.csv` – distribution of NPVs across simulated paths
- `storage_value_summary.csv` – headline numbers + notes

---


