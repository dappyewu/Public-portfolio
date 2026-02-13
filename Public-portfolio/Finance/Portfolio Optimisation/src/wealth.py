#optional

from datetime import datetime, timedelta
#from tickers import tickers   # ✅ import the list, not the module
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from scipy.optimize import minimize
import risk_kit as rk
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Load portfolio file
from IngestRawData import load_portfolio

dft, tickers, weights, portfolio_data = load_portfolio("data/raw/input.csv")


# Create end date variable
end_date = datetime.today()

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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# simple_return : DataFrame of daily returns (rows = dates, cols = assets)
# updated_weights : numpy array of portfolio weights (aligned with columns)

# 1. Calculate portfolio daily returns
pret = simple_return.dot(updated_weights)

# 2. Build wealth index (start with 1 = $1 invested)
initial_investment = float(input("Enter your initial investment: "))
wealth_index = (1 + pret).cumprod() * initial_investment

# 3. Combine into a DataFrame for clarity
wealth_df = pd.DataFrame({
    'pret': pret,
    'Wealth_Index': wealth_index
})

# --- NEW PART: Year-end wealth index ---
wealth_df.index = pd.to_datetime(wealth_df.index)

# Take the last wealth index value at the end of each year
year_end_wealth = wealth_df['Wealth_Index'].resample('Y').last()

# --- Plot cumulative wealth as line with year-end highlights ---
plt.figure(figsize=(12,6))
plt.plot(wealth_df.index, wealth_df['Wealth_Index'], label='Wealth Index', color='seagreen')

# Highlight year-end values with markers
plt.scatter(year_end_wealth.index, year_end_wealth.values, color='red', zorder=5, label='Year-End Value')

# Annotate year-end values
for date, value in year_end_wealth.items():
    plt.text(date, value, f"{value:.2f}", ha='center', va='bottom', fontsize=8, color='red')

plt.title(f'Cumulative Wealth Index (Initial Investment = {initial_investment})')
plt.ylabel('Wealth Index Value')
plt.xlabel('Date')
plt.axhline(initial_investment, color='black', linewidth=0.8, linestyle='--', label='Initial Investment')
plt.legend()

# Save to processed folder
processed_dir = Path("data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plot_path = processed_dir / f"WealthIndexLine_{timestamp}.png"

plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"✅ Saved cumulative line chart to: {plot_path}")

# Export wealth data
wealth_df.to_csv(processed_dir / f"WealthIndexData_{timestamp}.csv", index=True)
