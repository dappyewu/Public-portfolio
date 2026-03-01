from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class StorageParams:
    capacity: float                 # max inventory (MWh)
    init_inventory: float           # starting inventory (MWh)
    inj_rate: float                 # max injection per day (MWh/day)
    wdr_rate: float                 # max withdrawal per day (MWh/day)
    inj_fee: float = 0.0            # €/MWh injection fee (variable)
    wdr_fee: float = 0.0            # €/MWh withdrawal fee (variable)
    loss_frac: float = 0.0          # fraction lost on injection/holding (e.g. 0.002)
    discount_rate_annual: float = 0.03  # risk-free discount for NPV


def _disc_factor(params: StorageParams) -> float:
    # continuous-ish daily discount factor
    return float(np.exp(-params.discount_rate_annual / 365.0))


def _idx_floor(grid: np.ndarray, x: float) -> int:
    """Largest index such that grid[idx] <= x (clipped to [0, len-1])."""
    j = int(np.searchsorted(grid, x, side='right') - 1)
    return int(np.clip(j, 0, len(grid) - 1))


def _idx_ceil(grid: np.ndarray, x: float) -> int:
    """Smallest index such that grid[idx] >= x (clipped to [0, len-1])."""
    j = int(np.searchsorted(grid, x, side='left'))
    return int(np.clip(j, 0, len(grid) - 1))


def optimal_policy_perfect_foresight(
    prices: pd.Series,
    params: StorageParams,
    grid_size: int = 200,
) -> tuple[float, pd.DataFrame]:
    """Value storage with  foresight over a single price path.



    Returns:
      (npv, policy_df)
    """
    if prices.isna().any():
        prices = prices.dropna()

    T = len(prices)
    if T < 2:
        raise ValueError("Need at least 2 price points")

    disc = _disc_factor(params)

    # inventory grid
    inv_grid = np.linspace(0.0, params.capacity, grid_size)
    # nearest index for init inventory
    init_idx = int(np.argmin(np.abs(inv_grid - params.init_inventory)))

    # DP arrays
    V = np.zeros((T, grid_size), dtype=float)        # value-to-go
    action = np.zeros((T, grid_size), dtype=int)     # -1 withdraw, 0 hold, +1 inject
    delta = np.zeros((T, grid_size), dtype=float)    # amount injected/withdrawn (MWh)

    # terminal value: assume leftover inventory is sold at last price (net of withdrawal fee)
    p_T = float(prices.iloc[-1])
    V[-1, :] = inv_grid * (p_T - params.wdr_fee)

    # backward induction
    for t in range(T-2, -1, -1):
        p = float(prices.iloc[t])

        for i, inv in enumerate(inv_grid):
            # HOLD
            hold_val = disc * V[t+1, i]

            # INJECT
            inj_amt = min(params.inj_rate, params.capacity - inv)
            inv_after_inj = inv + inj_amt * (1.0 - params.loss_frac)
            j_inj = _idx_floor(inv_grid, float(inv_after_inj))  # floor mapping avoids 'free inventory'
            inj_cash = -inj_amt * (p + params.inj_fee)  # pay to buy gas + fee
            inj_val = inj_cash + disc * V[t+1, j_inj]

            # WITHDRAW
            wdr_amt = min(params.wdr_rate, inv)
            inv_after_wdr = inv - wdr_amt
            j_wdr = _idx_ceil(inv_grid, float(inv_after_wdr))   # ceil mapping avoids 'free inventory'
            wdr_cash = wdr_amt * (p - params.wdr_fee)   # receive gas sale - fee
            wdr_val = wdr_cash + disc * V[t+1, j_wdr]

            best = max(hold_val, inj_val, wdr_val)
            V[t, i] = best

            if best == inj_val:
                action[t, i] = 1
                delta[t, i] = inj_amt
            elif best == wdr_val:
                action[t, i] = -1
                delta[t, i] = wdr_amt
            else:
                action[t, i] = 0
                delta[t, i] = 0.0

    # forward simulate the optimal policy starting at init inventory
    inv_idx = init_idx
    inv = float(inv_grid[inv_idx])
    rows = []
    npv = 0.0
    df_disc = 1.0

    for t, (dt, p) in enumerate(prices.items()):
        p = float(p)
        a = int(action[t, inv_idx])
        amt_dp = float(delta[t, inv_idx])

        # IMPORTANT:
        # The DP is solved on an inventory grid, but the forward rollout
        # can drift slightly off-grid due to mapping/rounding. Clamp both
        # (a) action volumes to feasible limits given *actual* inventory
        # and (b) inventory to physical bounds to avoid negative inventory.
        cash = 0.0
        if a == 1 and amt_dp > 0:
            # Can't inject beyond remaining capacity or daily inj_rate
            max_inj = min(params.inj_rate, max(0.0, params.capacity - inv))
            amt = min(amt_dp, max_inj)
            inv = inv + amt * (1.0 - params.loss_frac)
            inv = float(np.clip(inv, 0.0, params.capacity))
            cash = -amt * (p + params.inj_fee)
        elif a == -1 and amt_dp > 0:
            # Can't withdraw beyond inventory or daily wdr_rate
            max_wdr = min(params.wdr_rate, max(0.0, inv))
            amt = min(amt_dp, max_wdr)
            inv = inv - amt
            inv = float(np.clip(inv, 0.0, params.capacity))
            cash = amt * (p - params.wdr_fee)
        else:
            amt = 0.0
            inv = float(np.clip(inv, 0.0, params.capacity))

        npv += df_disc * cash
        rows.append({"date": dt, "price": p, "action": a, "quantity_mwh": amt, "inventory_mwh": inv, "cashflow_eur": cash})
        df_disc *= disc
        # Use floor mapping to avoid selecting a grid point above actual inv
        # (which can cause the DP to "withdraw" more than you physically have).
        inv_idx = _idx_floor(inv_grid, inv)

    # add terminal liquidation (already included in value function, but add explicit cash for policy table)
    terminal_cash = inv * (float(prices.iloc[-1]) - params.wdr_fee)
    rows[-1]["terminal_liquidation_eur"] = terminal_cash

    return float(npv + df_disc/ disc * terminal_cash), pd.DataFrame(rows)
