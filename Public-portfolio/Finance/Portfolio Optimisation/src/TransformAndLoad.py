from datetime import datetime, timedelta
#from tickers import tickers   # ✅ import the list, not the module
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from scipy.optimize import minimize
import risk_kit as rk

import pandas as pd
import numpy as np

# Load portfolio file
from IngestRawData import load_portfolio

dft, tickers, weights, portfolio_data = load_portfolio("data/raw/input.csv")


# Create end date variable
end_date = datetime.today()
#end_date = datetime(2023, 12, 31)

# Create start date variable in years
years = int(input("Enter your look back in years: "))
start_date = end_date - timedelta(days=years * 365)

print(f"Start date is {start_date}")
print(f"End date is {end_date}")

adj_close_df  = pd.DataFrame() #Daily returns


#Daily Returns

for ticker in tickers:
    
    data = yf.download(ticker, start= start_date, end=end_date, auto_adjust=False)
    adj_close_df[ticker] = data['Adj Close']
adj_close_df = adj_close_df[tickers] #This reorders the columns of adj_close_df to match the original tickers list.


#identify tickers with limted data
days_to_look_back_min = 0.65*years*252
filtered_tickers = adj_close_df.columns[adj_close_df.notnull().sum() < days_to_look_back_min].tolist()
print(filtered_tickers , 'has been deleted')


# drop all tickers with less than specified  non-null values
adj_close_df.drop(columns=filtered_tickers, inplace=True)

#to create a lit of the tickers as originally inserted
updatedtickers = adj_close_df.columns.tolist()


# Create a lookup dict: ticker -> weight
weight_map = dict(zip(portfolio_data["tickers"], portfolio_data["weights"]))

# Extract weights in the same order as updatedtickers
updated_weights = np.array([weight_map[t] for t in updatedtickers], dtype=float)
updated_weights = updated_weights / updated_weights.sum()



#Catch if weights do not sum to 1.0
total = updated_weights.sum()
tolerance = 1e-2  # Acceptable numerical error

updated_weights = updated_weights / updated_weights.sum()

if abs(total - 1.0) > tolerance:
    print(f" Warning: weights sum to {total:.10f}, not 1.0")
else:
    print(" Weights sum to 1.0")

# Ensure weights match the number of tickers
if len(updated_weights) != len(updatedtickers):
    print(f"❌ Error: You have {len(updated_weights)} weights but {len(updatedtickers)} tickers.")
    raise ValueError("Number of weights must match number of tickers.")
else:
    print(f"✅ OK: You have {len(updated_weights)} weights and {len(updatedtickers)} tickers. Continue to the next step")


#Daily Simple Return
simple_return = adj_close_df.pct_change().dropna()

#Portfolio holdings CAGR - Daily Data
er = rk.annualize_rets(simple_return,252)
#print(er.sort_values(ascending=False))

from fredapi import Fred
from dotenv import load_dotenv
import os

load_dotenv()
FRED_API_KEY = os.getenv('FRED_API_KEY')
if not FRED_API_KEY:
    raise ValueError('FRED_API_KEY not set. Create a .env file or export it as an env var.')
fred = Fred(api_key=FRED_API_KEY)
ten_year_treasury_rate = fred.get_series_latest_release('GS10')/100

#SET RISK FREE RATE
risk_free_rate =  ten_year_treasury_rate.iloc[-1]
#print(risk_free_rate)

#Average risk free rate for duration 
arisk_free_rate = ten_year_treasury_rate.loc[start_date:end_date].mean()
arisk_free_rate= arisk_free_rate #+ 0.005 #Adjusting for current higher interests!! 
#print(arisk_free_rate)

#Share ratio - Daily
sr = rk.sharpe_ratio(simple_return, arisk_free_rate, 252)

#Covariance matrix - Daily
cov = simple_return.cov()*252

PP = 100*rk.portfolio_return(updated_weights,er)
PV = 100*rk.portfolio_vol(updated_weights,cov)
PSR = ((PP/100) - arisk_free_rate)/(PV/100)

Portfolio = pd.DataFrame({
    "Portfolio Returns%": [PP],
    "Portfolio Volatility%": [PV],
    "Portfolio Sharpe Ratio": [PSR]

})

from pathlib import Path
from datetime import datetime

# Ensure processed folder exists
processed_dir = Path("data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

# Save with timestamp (so it doesn't overwrite)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = processed_dir / f"portfolio_results_{timestamp}.csv"

Portfolio.to_csv(output_path, index=False)

print(f"✅ Saved portfolio results to: {output_path}")

processed_dir = Path("data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = processed_dir / f"CAGR_{timestamp}.csv"

er_out = er.copy()

# If tickers are the index, move them into a column
if er_out.index.name is None or er_out.index.dtype == "object":
    er_out = er_out.reset_index().rename(columns={"index": "Ticker"})

er_out.sort_values(by="Ticker", inplace=True)
er_out.to_csv(output_path, index=False)

print(f"✅ Saved CAGR results to: {output_path}")

#Calculate returns if equally weighted
n = len(er)
w_ew  = np.repeat(1/n, n)

# Calcaulte weights of Global Min Volatility  and Max Sharpe Ratio using helper functions
w_gmv = rk.gmv(cov.values)
w_msr = rk.msr(arisk_free_rate, er.values, cov.values)

# Normalize (defensive)
w_gmv = w_gmv / w_gmv.sum()
w_msr = w_msr / w_msr.sum()

weights_df = pd.DataFrame(
    np.vstack([updated_weights,w_ew, w_gmv, w_msr]),
    columns=er.index,
    index=["Original", "EW", "GMV", "MSR"]
)
weights_df = 100*weights_df.round(4)

processed_dir = Path("data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

# Save with timestamp (so it doesn't overwrite)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = processed_dir / f"Allocationweights{timestamp}.csv"

if weights_df.index.name is None or weights_df.index.dtype == "object":
    weights_df = weights_df.reset_index().rename(columns={"index": "Method"})

weights_df.to_csv(output_path, index=False)

print(f"✅ Saved Allocation weights results to: {output_path}")


#function to display results easily

def portfolio_summary(weights, er_ann, cov_ann, rf=0.0):
    weights = np.asarray(weights)
    p_ret = rk.portfolio_return(weights, er_ann.values)
    p_vol = rk.portfolio_vol(weights, cov_ann.values)
    p_sr  = (p_ret - rf) / p_vol if p_vol != 0 else np.nan
    return pd.Series({"Return (ann.%)": 100*p_ret, "Vol (ann.%)": 100*p_vol, "Sharpe": p_sr})


summary = pd.concat([
    portfolio_summary(updated_weights,  er, cov, rf=arisk_free_rate).rename("Original"),
    portfolio_summary(w_ew,  er, cov, rf=arisk_free_rate).rename("EW"),
    portfolio_summary(w_gmv, er, cov, rf=arisk_free_rate).rename("GMV"),
    portfolio_summary(w_msr, er, cov, rf=arisk_free_rate).rename("MSR"),
], axis=1).T


processed_dir = Path("data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

# Save with timestamp (so it doesn't overwrite)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = processed_dir / f"Allocation_Style_summary_{timestamp}.csv"

if summary.index.name is None or summary.index.dtype == "object":
    summary = summary.reset_index().rename(columns={"index": "Method"})

summary.to_csv(output_path, index=False)

print(f"✅ Saved summary results to: {output_path}")


ax = rk.plot_ef(
    n_points=50,
    er=er.values, 
    cov=cov.values,
    show_cml=True,
    riskfree_rate=arisk_free_rate,
    show_ew=True,
    show_gmv=True,
    legend=False
)
ax.set_title("Efficient Frontier (annualized)")
ax.set_xlabel("Volatility (ann.)")
ax.set_ylabel("Return (ann.)")

processed_dir = Path("data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plot_path = processed_dir / f"efficient_frontier_{timestamp}.png"

plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"✅ Saved plot to: {plot_path}")

#max drawdown
ew_daily = (simple_return @ updated_weights)
dd_table = rk.drawdown(ew_daily)
dd_info  = rk.drawdowninfo(ew_daily)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save drawdown table (time series)
dd_table_path = processed_dir / f"drawdown_table_{timestamp}.csv"
dd_table.sort_values(by='Drawdown').head().to_csv(dd_table_path, index=True)


print(f"✅ Saved drawdown table to: {dd_table_path}")

# Save drawdown info (summary)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
txt_path = processed_dir / f"drawdown_info_{timestamp}.txt"

with open(txt_path, "w", encoding="utf-8") as f:
    f.write(str(dd_info))

print(f"✅ Saved dd_info text to: {txt_path}")

# Rolling Sharpe Ratio (portfolio)
# -----------------------------
# Common windows: ~3m=63d, ~6m=126d, ~1y=252d
rolling_sr_36m = rk.rolling_sharpe_ratio(
    ew_daily, window=756, riskfree_rate=arisk_free_rate, periods_per_year=252
)
#rolling_sr_6m = rk.rolling_sharpe_ratio(
   # ew_daily, window=126, riskfree_rate=arisk_free_rate, periods_per_year=252
#)
rolling_sr_1y = rk.rolling_sharpe_ratio(
    ew_daily, window=252, riskfree_rate=arisk_free_rate, periods_per_year=252
)

rolling_sr_df = pd.DataFrame(
    {
        "RollingSharpe_36M": rolling_sr_36m,
        # "RollingSharpe_6M": rolling_sr_6m,
        "RollingSharpe_1Y": rolling_sr_1y,
    }
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
rolling_sr_path = processed_dir / f"rolling_sharpe_{timestamp}.csv"
rolling_sr_df.to_csv(rolling_sr_path, index=True)
print(f"✅ Saved rolling Sharpe to: {rolling_sr_path}")

# Optional plot
plt.figure()
rolling_sr_df.plot()
plt.title("Rolling Sharpe Ratio (Portfolio)")
plt.xlabel("Date")
plt.ylabel("Sharpe")

rolling_sr_plot_path = processed_dir / f"rolling_sharpe_{timestamp}.png"
plt.savefig(rolling_sr_plot_path, dpi=300, bbox_inches="tight")
print(f"✅ Saved rolling Sharpe plot to: {rolling_sr_plot_path}")