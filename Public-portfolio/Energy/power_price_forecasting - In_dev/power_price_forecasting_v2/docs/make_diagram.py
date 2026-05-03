"""Generate the project's architecture diagram.

A single PNG showing two panels:

  1. Market flow — how the GB day-ahead price actually gets formed.
     Generators bid, the operator publishes load/wind/solar forecasts before
     11:00 CET, the auction clears, and hourly prices fall out.

  2. Model pipeline — how this project consumes the same public information
     and tries to predict the auction's clearing price.

The two panels are linked by dashed arrows: the public forecasts feed the
model as inputs, and the auction's clearing price is what the model predicts.

Run this whenever you change the architecture; the PNG it writes is what the
READMEs link to.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "architecture.png"

# Palette
C_GENERATOR = "#F5D7B5"   # warm tan — physical assets
C_AUCTION = "#FFE699"     # yellow — the market mechanism
C_FORECAST = "#C8E0F4"    # light blue — public information
C_PRICE = "#F7B7B7"       # red — the target / clearing outcome
C_DATA = "#C8E0F4"        # blue — data layer
C_FEATURES = "#C8E6C9"    # green — engineered features
C_MODEL = "#E1BEE7"       # purple — models / ML
C_OUTPUT = "#FFE699"      # yellow — final forecast
ARROW_COLOR = "#444"
DASHED_ARROW_COLOR = "#888"


def box(ax, x, y, w, h, text, color, fontsize=9, weight="normal"):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.4,rounding_size=0.6",
        facecolor=color, edgecolor="#333", linewidth=1.2,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, weight=weight)


def arrow(ax, x1, y1, x2, y2, dashed=False, label=None, lw=1.5):
    style = "->,head_width=4,head_length=6"
    color = DASHED_ARROW_COLOR if dashed else ARROW_COLOR
    linestyle = (0, (4, 3)) if dashed else "-"
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, color=color, linewidth=lw,
        linestyle=linestyle, mutation_scale=1,
        shrinkA=0, shrinkB=0,
    )
    ax.add_patch(a)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.6, label,
                ha="center", va="bottom", fontsize=8,
                style="italic", color="#555")


def section_header(ax, y, text):
    ax.text(50, y, text, ha="center", va="center",
            fontsize=13, weight="bold", color="#222",
            bbox=dict(facecolor="#EEE", edgecolor="none", boxstyle="round,pad=0.4"))


def main() -> None:
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 108)
    ax.axis("off")

    # ---------- TITLE ----------
    ax.text(50, 105.5,
            "GB Day-Ahead Power Price Forecasting  —  v2",
            ha="center", va="center", fontsize=16, weight="bold")
    ax.text(50, 103,
            "",
            ha="center", va="center", fontsize=10, style="italic", color="#555")

    # =================================================================
    # PANEL 1 — MARKET FLOW
    # =================================================================
    section_header(ax, 100, "1.  Simplifid view of how the day-ahead price is set")

    # ---- Out-of-scope drivers callout (also feed the auction, not modelled) ----
    ghost_box = FancyBboxPatch(
        (12, 90.5), 76, 6,
        boxstyle="round,pad=0.5,rounding_size=0.6",
        facecolor="#F2F2F2", edgecolor="#888", linewidth=1.2,
        linestyle=(0, (5, 3)),
    )
    ax.add_patch(ghost_box)
    ax.text(50, 95, "Other price drivers — NOT modelled (out of scope)",
            ha="center", va="center", fontsize=9, weight="bold", color="#555")
    ax.text(50, 92.4,
            "Plant outages   ·   Interconnector flows   ·   Forecast errors   ·   Battery / DSR",
            ha="center", va="center", fontsize=8.5, color="#555", style="italic")

    # Dashed arrow from the ghost row down into the auction.
    arrow(ax, 50, 90.5, 50, 84, dashed=True, lw=1.2)

    # Generators (left column)
    ax.text(13.5, 88.5, "Generators bid into the auction",
            ha="center", fontsize=9, style="italic", color="#555")
    gens = [
        ("Wind",            83.5),
        ("Solar",           80.0),
        ("Nuclear",         76.5),
        ("CCGT (gas)",      73.0),
        ("Peakers",  69.5),
    ]
    for label, y in gens:
        box(ax, 4, y, 19, 3, label, C_GENERATOR, fontsize=8)

    # Day-ahead auction (centre)
    box(ax, 38, 75, 24, 9,
        "Day-Ahead Auction\nclears at 11:00 CET\nhourly prices for next 24h",
        C_AUCTION, fontsize=10, weight="bold")

    # Public forecasts (right column)
    ax.text(86.5, 88.5, "TSO publishes (before auction)",
            ha="center", fontsize=9, style="italic", color="#555")
    forecasts = [
        ("Load forecast",   83.5),
        ("Wind forecast",   80.0),
        ("Solar forecast",  76.5),
    ]
    for label, y in forecasts:
        box(ax, 77, y, 19, 3, label, C_FORECAST, fontsize=8)

    # Clearing price output
    box(ax, 38, 65.5, 24, 5,
        "Hourly clearing prices",
        C_PRICE, fontsize=10, weight="bold")

    # Arrows: generators → auction (bids)
    for _, y in gens:
        arrow(ax, 23, y + 1.5, 38, 79.5, lw=1.0)

    # Arrows: forecasts → auction (informing traders / price discovery)
    for _, y in forecasts:
        arrow(ax, 77, y + 1.5, 62, 79.5, lw=1.0)

    # Arrow: auction → clearing price
    arrow(ax, 50, 75, 50, 70.5, lw=1.8)

    # ---- v2: Commodity markets feeding into bid level (right of generators) ----
    ax.text(13.5, 67, "Commodity markets",
            ha="center", fontsize=9, style="italic", color="#555")
    fuel_inputs = [
        ("TTF gas (NBP proxy)",      63.5),
        ("EUA carbon (UKA proxy)",   60.0),
    ]
    for label, y in fuel_inputs:
        box(ax, 4, y, 19, 3, label, C_FORECAST, fontsize=7.5)
    # Arrows: commodity prices feed CCGT/peaker bids (visualised as into auction)
    for _, y in fuel_inputs:
        arrow(ax, 23, y + 1.5, 38, 76, lw=1.0)
    # Small annotation tying the visual story together
    ax.text(13.5, 57.5,
            "fuel + carbon set CCGT marginal cost\n→ shape gas-plant bids",
            ha="center", fontsize=7.5, style="italic", color="#666")

    # =================================================================
    # BRIDGE — top panel hands two things to the bottom panel
    # =================================================================
    # Dashed arrow: forecasts down → model inputs
    arrow(ax, 86.5, 76, 86.5, 56.5, dashed=True, lw=1.3,
          label="model inputs")
    # Dashed arrow: clearing price down → model target
    arrow(ax, 50, 65.5, 50, 56.5, dashed=True, lw=1.3,
          label="model target")

    # Divider line
    ax.plot([2, 98], [54, 54], color="#CCC", linewidth=1, linestyle="-")

    # =================================================================
    # PANEL 2 — MODEL PIPELINE
    # =================================================================
    section_header(ax, 50, "2.  How this project predicts that price")

    # Row 1: linear pipeline
    box(ax, 3, 36, 16, 9,
        "Data sources\n• Elexon / ENTSO-E\n  (power)\n• Yahoo Finance\n  (gas, carbon)",
        C_DATA, fontsize=8)
    box(ax, 23, 36, 16, 9,
        "Hourly aligned\ntable\nprice, load,\nwind, solar",
        C_DATA, fontsize=8.5)
    box(ax, 43, 36, 16, 9,
        "Feature engineering\n• Calendar\n• Fundamentals\n  (residual_load)\n• Lags 24/48/168 h",
        C_FEATURES, fontsize=8.5)
    box(ax, 63, 36, 16, 9,
        "Model ladder\n• Seasonal naive\n• Ridge\n• LightGBM",
        C_MODEL, fontsize=8.5)
    box(ax, 83, 36, 14, 9,
        "Hourly price\nforecast\nfor next day",
        C_OUTPUT, fontsize=9, weight="bold")

    # Forward arrows
    arrow(ax, 19, 40.5, 23, 40.5)
    arrow(ax, 39, 40.5, 43, 40.5)
    arrow(ax, 59, 40.5, 63, 40.5)
    arrow(ax, 79, 40.5, 83, 40.5)

    # Row 2: validation / diagnostics (sit beneath models)
    box(ax, 33, 18, 26, 9,
        "Walk-forward validation\n• Expanding window, 5 folds\n• Train: past, Test: strictly future\n• MAE / RMSE / MAPE",
        C_MODEL, fontsize=8.5)
    box(ax, 63, 18, 16, 9,
        "Diagnostics\n• Actual vs predicted\n• Residuals by hour\n• Feature importance",
        C_OUTPUT, fontsize=8.5)

    # Arrow: models down to validation, validation across to diagnostics
    arrow(ax, 71, 36, 53, 27)         # models -> validation
    arrow(ax, 59, 22.5, 63, 22.5)      # validation -> diagnostics

    # Footer note
    ax.text(50, 8,
            "All  model inputs are published BEFORE the auction closes so no look ahead.",
            ha="center", fontsize=9.5, style="italic", color="#444",
            bbox=dict(facecolor="#FFFDE7", edgecolor="#E0DCC8", boxstyle="round,pad=0.5"))
    ax.text(50, 4,
            "",
            ha="center", fontsize=9.5, style="italic", color="#444",
            bbox=dict(facecolor="#FFFDE7", edgecolor="#E0DCC8", boxstyle="round,pad=0.5"))

    plt.tight_layout()
    fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
