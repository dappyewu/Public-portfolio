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
    p.add_argument("--n_days", type=int, default=365)
    p.add_argument("--n_paths", type=int, default=200)
    args = p.parse_args()

    base = Path(__file__).resolve().parents[1]
    spot_path = base / args.spot_history
    fwd_path = base / args.forward_curve
    outdir = base / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    spot = pd.read_csv(spot_path, parse_dates=["date"]).sort_values("date")
    forward = pd.read_csv(fwd_path)

    # fit OU on log spot
    params_ou = fit_ou_params(np.log(spot["spot_eur_mwh"]))

    # storage assumptions (tweakable)
    sp = StorageParams(
        capacity=100_000.0,
        init_inventory=50_000.0,
        inj_rate=2_500.0,
        wdr_rate=2_500.0,
        inj_fee=0.10,
        wdr_fee=0.10,
        loss_frac=0.001,
        discount_rate_annual=0.03,
    )

    # Intrinsic value using deterministic forward curve
    start_date = pd.Timestamp.today().normalize()
    intrinsic, pol_intr = intrinsic_value_from_forward(forward, sp, start_date=start_date, n_days=args.n_days)
    pol_intr.to_csv(outdir / "optimal_policy_intrinsic.csv", index=False)

    # Simulate paths around a seasonal theta
    dates = pd.date_range(start_date, periods=args.n_days, freq="D")
    seasonal_theta = build_seasonal_theta(dates, base_price=float(forward["fwd_eur_mwh"].iloc[0]))
    paths = simulate_ou_paths(
        s0=float(spot["spot_eur_mwh"].iloc[-1]),
        params=params_ou,
        n_days=args.n_days,
        n_paths=args.n_paths,
        seed=7,
        seasonal_theta=seasonal_theta,
    )

    mc = monte_carlo_upper_bound(paths, dates, sp, max_paths=args.n_paths)
    mc.to_csv(outdir / "storage_value_mc_upper_bound.csv", index=False)

    summary = pd.DataFrame([{
        "intrinsic_npv_eur": intrinsic,
        "mc_upper_bound_mean_eur": float(mc["npv_eur"].mean()),
        "mc_upper_bound_p10_eur": float(mc["npv_eur"].quantile(0.10)),
        "mc_upper_bound_p90_eur": float(mc["npv_eur"].quantile(0.90)),
        "notes": "MC uses perfect-foresight DP per path -> upper bound (not risk-neutral pricing)."
    }])
    summary.to_csv(outdir / "storage_value_summary.csv", index=False)

    print("Wrote:")
    print(" -", outdir / "optimal_policy_intrinsic.csv")
    print(" -", outdir / "storage_value_mc_upper_bound.csv")
    print(" -", outdir / "storage_value_summary.csv")


if __name__ == "__main__":
    main()
