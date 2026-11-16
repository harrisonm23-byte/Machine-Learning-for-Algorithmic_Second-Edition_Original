"""Run the study: ``python -m volstrat --output-dir output``.

Optional first step, ``--calibrate``: rebuild the volatility-surface calibration
from Alpaca SPY option prices (needs APCA_API_KEY_ID / APCA_API_SECRET_KEY) and
write ``calibration/``.

Outputs (output directory):
  grid_calibrated.csv / grid_page.csv   strategy x exit rule x hedge x VIX filter, per surface
  sensitivity.csv       primary setup with the calibrated surface shifted -1/+1 vol point,
                        and the page surface at ATM = 0.8 / 0.85 / 0.9 / 1.0 x VIX
  episodes.csv, yearly.csv              primary setup, calibrated surface
  vrp_buckets.csv       VIX versus realised volatility over the next 21 trading days
  trades_<strategy>.csv trade lists, primary setup, calibrated surface
  status.json           latest data point, open positions and whether a new entry is allowed
  equity.png, sharpe_grid.png, sharpe_grid_page.png, vrp.png, episodes.png
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import (PRIMARY, episode_pnl, primary_config, run_grid, variance_risk_premium,
                       vrp_by_bucket, yearly_returns)
from .backtest import EXIT_RULES, run_backtest
from .data import load_index_data
from .strategies import STRATEGIES
from .surfaces import CALIBRATION_DIR, PAGE, PARAMS_FILE, calibrated

COLORS = {"book": "#d95926", "strangle": "#2a78d6", "condor": "#199e70",
          "longstr": "#9085e9", "longput": "#c98500", "longcall": "#888888"}
NAMES = {k: v.name for k, v in STRATEGIES.items()}
FILTER_ON, FILTER_OFF = str(PRIMARY["vix_filter"]), str(None)
COLS = ["strategy", "surface_name", "exit_rule", "hedge", "vix_filter", "n_trades", "time_in_market",
        "ann_return_on_notional", "ann_vol_on_notional", "sharpe", "max_drawdown_on_notional",
        "worst_day_on_notional", "beta_to_spx", "corr_to_spx", "win_rate", "avg_pnl_per_premium",
        "worst_trade_per_premium", "avg_premium_on_notional", "total_costs_on_notional", "open_at_end"]


def calibrate(index_data: pd.DataFrame, cache_dir: str) -> dict:
    from .calibrate import fit_surface_global, run_calibration, surface_errors

    CALIBRATION_DIR.mkdir(exist_ok=True)
    cals = {}
    for tenor, step in [(30, 1), (10, 2), (20, 2)]:
        cal = run_calibration(index_data, start="2024-02-01", cache_dir=f"{cache_dir}/alpaca",
                              step=step, target_days=tenor)
        cal.to_csv(CALIBRATION_DIR / f"spy_smile_{tenor}d.csv")
        cals[tenor] = cal
    params = fit_surface_global(cals)
    PARAMS_FILE.write_text(json.dumps(params, indent=2))
    surface_errors(cals, params).round(2).to_csv(CALIBRATION_DIR / "fit_errors.csv")
    return params


def main(argv=None):
    p = argparse.ArgumentParser(description="Backtest the SPX short/long volatility strategies.")
    p.add_argument("--start", default="2009-06-01")
    p.add_argument("--end", default=None)
    p.add_argument("--cache-dir", default="data_cache")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--calibrate", action="store_true", help="rebuild calibration/ from Alpaca first")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = load_index_data(a.start, a.end, cache_dir=a.cache_dir, refresh=a.refresh)
    logging.info("Data %s to %s, %d days", data.index[0].date(), data.index[-1].date(), len(data))
    if a.calibrate:
        calibrate(data, a.cache_dir)
    cal = calibrated()

    grid, results = run_grid(data, cal)
    grid_page, results_page = run_grid(data, PAGE)
    grid[COLS].to_csv(out / "grid_calibrated.csv")
    grid_page[COLS].to_csv(out / "grid_page.csv")

    sens_rows = []
    for surf in [calibrated(atm_shift=-1.0), calibrated(atm_shift=+1.0),
                 *[dict(PAGE, atm_ratio=r) for r in (0.8, 0.85, 1.0)]]:
        for s in STRATEGIES:
            sens_rows.append(run_backtest(data, primary_config(s, surf)).summary)
    sens = pd.concat([pd.DataFrame(sens_rows).set_index("label"),
                      grid.loc[[primary_config(s, cal).label for s in STRATEGIES]],
                      grid_page.loc[[primary_config(s, PAGE).label for s in STRATEGIES]]])
    sens["surface"] = [f"{n}" if n != "page" else f"page {r:g}xVIX"
                       for n, r in zip(sens["surface_name"], sens["atm_ratio"])]
    sens[["surface"] + COLS].to_csv(out / "sensitivity.csv")

    labels = [primary_config(s, cal).label for s in STRATEGIES]
    episode_pnl(results, labels).rename(index=lambda l: l.split("|")[0]).to_csv(out / "episodes.csv")
    yearly_returns(results, labels).rename(columns=lambda l: l.split("|")[0]).to_csv(out / "yearly.csv")
    for s, lab in zip(STRATEGIES, labels):
        results[lab].trades.to_csv(out / f"trades_{s}.csv", index=False)

    vrp = variance_risk_premium(data)
    vrp_by_bucket(vrp).to_csv(out / "vrp_buckets.csv")

    latest = data.iloc[-1]
    open_positions = {}
    for s, lab in zip(STRATEGIES, labels):
        t = results[lab].trades
        if len(t) and t["reason"].iloc[-1] == "end_of_data":
            r = t.iloc[-1]
            open_positions[s] = dict(
                entry_date=str(r["entry_date"].date()), expiry=str(r["expiry"].date()),
                legs=r["strikes"], premium=round(float(r["premium"]), 2),
                pnl_to_date_mid=round(float(r["pnl"] + r["exit_cost"]), 2),
                hedge_dollars_per_point=round(float(r["hedge_at_exit"]), 2),
                hedge_es_contracts=round(float(r["hedge_at_exit"]) / 50, 3),
                days_left=int((r["expiry"] - r["exit_date"]).days))
    in_band = PRIMARY["vix_filter"][0] <= latest["vix"] <= PRIMARY["vix_filter"][1]
    status = {
        "data_end": str(data.index[-1].date()), "spx": float(latest["spx"]), "vix": float(latest["vix"]),
        "vix_in_entry_band": bool(in_band),
        "note": ("Under the primary rules a new position is opened only when flat and the previous "
                 "close's VIX is in the band. 'open_positions' are positions the backtest holds at the "
                 "last close (P&L at mid, before exit costs)."),
        "open_positions": open_positions,
        "new_entry_allowed_next_session": {s: bool(in_band and s not in open_positions) for s in STRATEGIES},
        "vrp_mean_vix_minus_realized": float(vrp["premium"].mean()),
        "vrp_share_positive": float((vrp["premium"] > 0).mean()),
        "calibration": json.loads(PARAMS_FILE.read_text()).get("_fit"),
    }
    (out / "status.json").write_text(json.dumps(status, indent=2))

    _plots(out, data, grid, grid_page, results, labels, vrp)
    show = ["sharpe", "ann_return_on_notional", "max_drawdown_on_notional", "beta_to_spx", "win_rate",
            "worst_trade_per_premium"]
    print("Primary setup, calibrated surface:")
    print(grid.loc[labels, show].round(3).to_string())
    print("Primary setup, page surface:")
    print(grid_page.loc[[primary_config(s, PAGE).label for s in STRATEGIES], show].round(3).to_string())
    print(json.dumps(status, indent=2))


def _plots(out: Path, data, grid, grid_page, results, labels, vrp):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    # 1. equity curves, primary setup, calibrated surface, with VIX underneath
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for s, lab in zip(STRATEGIES, labels):
        ax.plot(results[lab].daily["ret"].cumsum(), lw=1.4, color=COLORS[s], label=NAMES[s])
    ax.axhline(0, color="black", lw=0.7)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Cumulative P&L as a share of one SPX contract's notional (calibrated surface)\n"
                 "hedged daily with ES at Black-Scholes delta, entries when VIX 12-18, held to expiry")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax2.plot(data["vix"], lw=0.8, color="#555555")
    ax2.axhspan(12, 18, color="#2a78d6", alpha=0.12, label="entry band 12-18")
    ax2.set_ylabel("VIX")
    ax2.legend(frameon=False, loc="upper left")
    for a_ in (ax, ax2):
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "equity.png", dpi=120)
    plt.close(fig)

    # 2. Sharpe grids
    for g, fname, title in [(grid, "sharpe_grid.png", "calibrated surface"),
                            (grid_page, "sharpe_grid_page.png", "strategy page's surface, ATM = 0.9 x VIX")]:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True, sharey=True)
        combos = [(True, FILTER_ON), (True, FILTER_OFF), (False, FILTER_ON), (False, FILTER_OFF)]
        rules = list(EXIT_RULES)
        for ax, (h, f) in zip(axes.flat, combos):
            sub = g[(g["hedge"] == h) & (g["vix_filter"] == f)]
            mat = sub.pivot_table(index="strategy", columns="exit_rule", values="sharpe").loc[list(STRATEGIES), rules]
            im = ax.imshow(mat.values, cmap="RdBu", vmin=-1.5, vmax=1.5, aspect="auto")
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    v = mat.values[i, j]
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                            color="white" if abs(v) > 0.9 else "black")
            ax.set_title(f"{'Hedged' if h else 'Unhedged'}, {'VIX 12-18 filter' if f == FILTER_ON else 'no VIX filter'}")
            ax.set_xticks(range(len(rules)), rules, rotation=30)
            ax.set_yticks(range(len(STRATEGIES)), [NAMES[s] for s in STRATEGIES])
        fig.colorbar(im, ax=axes, shrink=0.6, label="Sharpe ratio")
        fig.suptitle(f"Sharpe ratio by strategy and exit rule, after costs ({title})\n"
                     "Early-exit columns depend on how the model marks options mid-life; 'expiry' is the most robust")
        fig.savefig(out / fname, dpi=120, bbox_inches="tight")
        plt.close(fig)

    # 3. variance risk premium
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [2, 1]})
    ax.plot(vrp["vix"], lw=0.8, color="#d95926", label="VIX (implied, priced today)")
    ax.plot(vrp["realized_fwd"], lw=0.8, color="#2a78d6", label="Realised vol over the next 21 trading days")
    ax.set_ylim(0, 1.05 * vrp[["vix", "realized_fwd"]].max().max())
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Implied versus subsequently realised S&P 500 volatility")
    lo = -40
    n_lo = int((vrp["premium"] < lo).sum())
    ax2.hist(vrp["premium"].clip(lower=lo), bins=80, color="#199e70")
    ax2.axvline(0, color="black", lw=0.8)
    ax2.annotate(f"{n_lo} days at or below {lo}\n(min {vrp['premium'].min():.0f})", xy=(lo, 0),
                 xytext=(lo + 3, ax2.get_ylim()[1] * 0.6), fontsize=8)
    ax2.set_xlabel(f"VIX minus realised vol, points (mean {vrp['premium'].mean():.1f}, "
                   f"positive {100 * (vrp['premium'] > 0).mean():.0f}% of days; left bin collects everything below {lo})")
    for a_ in (ax, ax2):
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "vrp.png", dpi=120)
    plt.close(fig)

    # 4. stress episodes
    ep = episode_pnl(results, labels).rename(index=lambda l: l.split("|")[0])
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(ep.shape[1])
    w = 0.8 / ep.shape[0]
    for k, (key, row) in enumerate(ep.iterrows()):
        ax.bar(x + k * w - 0.4 + w / 2, row.values, width=w, color=COLORS[key], label=NAMES[key])
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x, ep.columns, rotation=25, ha="right")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("P&L through stress episodes (primary setup, calibrated surface, share of one contract's notional)")
    ax.legend(frameon=False, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "episodes.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
