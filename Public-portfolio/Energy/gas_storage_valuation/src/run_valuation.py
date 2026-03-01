from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from .storage_dp import StorageParams
from .price_models import fit_ou_params, build_seasonal_theta, simulate_ou_paths
from .valuation_utils import intrinsic_value_from_forward, monte_carlo_upper_bound


def main():
    p = argparse.ArgumentParser(description="Gas storage valuation demo (DP + Monte Carlo upper bound)")
    p.add_argument("--spot_history", default="data/raw/spot_history_ttf_like.csv")
    p.add_argument("--forward_curve", default="data/raw/forward_curve_monthly.csv")
    p.add_argument("--outdir", default="results")
    p.add_argument("--n_days", type=int, default=365, help="Horizon in days (default matches notebooks)")
    p.add_argument("--n_paths", type=int, default=200, help="Number of Monte Carlo paths (default matches notebooks)")
    p.add_argument("--seed", type=int, default=7, help="RNG seed for Monte Carlo simulation (default matches notebooks)")
    args = p.parse_args()

    base = Path(__file__).resolve().parents[1]
    spot_path = base / args.spot_history
    fwd_path = base / args.forward_curve
    outdir = base / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    spot = pd.read_csv(spot_path, parse_dates=["date"]).sort_values("date")
    forward = pd.read_csv(fwd_path)


    logp = np.log(spot["spot_eur_mwh"]).dropna()
    params_ou = fit_ou_params(logp)

   
    sp = StorageParams(
        capacity=100.0,
        init_inventory=50.0,
        inj_rate=2.0,
        wdr_rate=2.0,
        inj_fee=0.02,
        wdr_fee=0.02,
        loss_frac=0.0,
        discount_rate_annual=0.0,
    )

    # -----------------
    # Intrinsic value
    # -----------------
    
    last_spot_plus_1 = pd.to_datetime(spot["date"].iloc[-1]) + pd.Timedelta(days=1)
    
    fwd_tmp = forward.copy()
    if "delivery_month" in fwd_tmp.columns:
        col = "delivery_month"
        fwd_tmp[col] = fwd_tmp[col].astype(str).str.strip()
        dt_ym = pd.to_datetime(fwd_tmp[col], format="%Y-%m", errors="coerce")
        dt_dmy = pd.to_datetime(fwd_tmp[col], format="%d/%m/%Y", errors="coerce")
        fwd_tmp["delivery_month"] = np.where(dt_ym.notna(), dt_ym, dt_dmy)
        fwd_tmp["delivery_month"] = pd.to_datetime(fwd_tmp["delivery_month"], errors="raise")
        first_fwd_month = fwd_tmp["delivery_month"].dt.to_period("M").dt.to_timestamp().min()
    elif "contract_month" in fwd_tmp.columns:
        first_fwd_month = pd.to_datetime(fwd_tmp["contract_month"], errors="raise").dt.to_period("M").dt.to_timestamp().min()
    else:
        raise ValueError("Forward curve must include 'delivery_month' or 'contract_month'.")

    start_intrinsic = max(last_spot_plus_1, first_fwd_month)
    intrinsic, pol_intr = intrinsic_value_from_forward(forward, sp, start_date=start_intrinsic, n_days=args.n_days)


    pol_intr.to_csv(outdir / "optimal_policy_intrinsic_CLI.csv", index=False)
    (outdir / "intrinsic_value_eur_CLI.txt").write_text(f"{intrinsic:.6f}\n")

    # -----------------
    # Monte Carlo upper bound (perfect-foresight DP per path)
    # -----------------
    start_mc = last_spot_plus_1
    future_dates = pd.date_range(start_mc, periods=args.n_days, freq="D")
    theta = build_seasonal_theta(future_dates)  

    paths = simulate_ou_paths(
        s0=float(spot["spot_eur_mwh"].iloc[-1]),
        params=params_ou,
        n_days=args.n_days,
        n_paths=args.n_paths,
        seed=int(args.seed),
        seasonal_theta=theta,
    )

    mc = monte_carlo_upper_bound(paths, future_dates, sp, max_paths=args.n_paths)
    mc.to_csv(outdir / "storage_value_mc_upper_bound_CLI.csv", index=False)

    summary = pd.DataFrame([
        {
            "intrinsic_npv_eur": float(intrinsic),
            "mc_upper_bound_mean_eur": float(mc["npv_eur"].mean()),
            "mc_upper_bound_p10_eur": float(mc["npv_eur"].quantile(0.10)),
            "mc_upper_bound_p90_eur": float(mc["npv_eur"].quantile(0.90)),
            "n_days": int(args.n_days),
            "n_paths": int(args.n_paths),
            "seed": int(args.seed),
            "notes": "MC uses perfect-foresight DP per path -> upper bound (not risk-neutral pricing).",
        }
    ])
    summary.to_csv(outdir / "storage_value_summary_CLI.csv", index=False)

    print("Wrote:")
    print(" -", outdir / "optimal_policy_intrinsic_CLI.csv")
    print(" -", outdir / "intrinsic_value_eur_CLI.txt")
    print(" -", outdir / "storage_value_mc_upper_bound_CLI.csv")
    print(" -", outdir / "storage_value_summary_CLI.csv")


if __name__ == "__main__":
    main()
