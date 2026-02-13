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
    raise ValueError('FRED_API_KEY not set. Create a .env file (see .env.example) or export it as an env var.')
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






import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Assume you already have:
# simple_return : DataFrame of daily returns (rows = dates, cols = assets)
# weights_df : DataFrame of portfolio weights (rows = methods, cols = assets)

weights_df = weights_df.copy()
weights_df["Method"] = weights_df["Method"].astype(str).str.strip()

weights_numeric = (
    weights_df
    .set_index("Method")                 # ✅ keeps method names as row index
    .select_dtypes(include=[np.number])  # numeric weights only
)

# Align weights with simple_return columns
common_assets = simple_return.columns.intersection(weights_numeric.columns)
simple_return_aligned = simple_return[common_assets]
weights_aligned = weights_numeric[common_assets]

# Normalize weights to sum to 1
weights_aligned = weights_aligned.div(weights_aligned.sum(axis=1), axis=0)

# Dictionary to store wealth indices for each method
wealth_results = {}

initial_investment = float(input("Enter your initial investment: "))

for method in weights_aligned.index:   # ✅ now method = 'Original', 'EW', 'GMV', 'MSR'
    method_weights = weights_aligned.loc[method].values

    # Daily portfolio returns
    pret = simple_return_aligned.dot(method_weights)

    # Wealth index
    wealth_index = (1 + pret).cumprod() * initial_investment

    wealth_results[method] = wealth_index

wealth_df = pd.DataFrame(wealth_results)

# --- Plot cumulative wealth for all methods ---
plt.figure(figsize=(12, 6))
for method in wealth_df.columns:
    plt.plot(wealth_df.index, wealth_df[method], label=method)

plt.title(f'Cumulative Wealth Index by Allocation Style (Initial Investment = {initial_investment})')
plt.ylabel('Wealth Index Value')
plt.xlabel('Date')
plt.axhline(initial_investment, color='black', linewidth=0.8, linestyle='--', label='Initial Investment')
plt.legend()

# Save outputs
processed_dir = Path("data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plot_path = processed_dir / f"WealthIndexComparison_{timestamp}.png"

plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"✅ Saved comparison chart to: {plot_path}")

processed_dir = Path("data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plot_path = processed_dir / f"WealthIndexComparisonData_{timestamp}.csv"

wealth_df.to_csv(plot_path, index=True)
print(f"✅ Saved comparison data to: {plot_path}")
