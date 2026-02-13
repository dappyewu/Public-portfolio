##Redundant code, but I want to keep it here for now.
import pandas as pd
import numpy as np

def load_portfolio(path="data/raw/input.csv"):
    """
    Loads portfolio CSV and returns:
    - dft (DataFrame)
    - tickers (list)
    - weights (np array, sums to 1)
    - portfolio_data (dict)
    """
    dft = pd.read_csv(path)

    tickers = dft["Ticker"].tolist()

    weights = dft["Amount_USD"].to_numpy(dtype=float)
    weights = weights / weights.sum()

    portfolio_data = {
        "tickers": tickers,
        "weights": weights
    }

    return dft, tickers, weights, portfolio_data
