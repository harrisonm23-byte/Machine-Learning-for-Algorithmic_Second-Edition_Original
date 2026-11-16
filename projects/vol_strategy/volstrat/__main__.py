"""Run the full study: ``python -m volstrat --output-dir output``.

Writes to the output directory:
  grid.csv              every strategy x exit rule x hedge x VIX filter (ATM = 0.9 x VIX)
  atm_sensitivity.csv   primary setup with ATM = 0.8, 0.9 and 1.0 x VIX
  episodes.csv          P&L through eight stress windows (primary setup)
  yearly.csv            calendar-year returns (primary setup)
  vrp_buckets.csv       VIX versus realised volatility over the next 21 trading days
  trades_<strategy>.csv trade lists for the primary setup
  equity.png, sharpe_grid.png, vrp.png, episodes.png
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import PRIMARY, episode_pnl, run_grid, variance_risk_premium, vrp_by_bucket, yearly_returns
from .backtest import EXIT_RULES, BacktestConfig
from .data import load_index_data
from .strategies import STRATEGIES

COLORS = {"book": "#d95926", "strangle": "#2a78d6", "condor": "#199e70",
          "longstr": "#9085e9", "longput": "#c98500", "longcall": "#888888"}


def primary_label(strategy: str, atm: float = 0.9) -> str:
    cfg = BacktestConfig(strategy=strategy, **{**PRIMARY, "atm_ratio": atm})
    return cfg.label


def main(argv=None):
    p = argparse.ArgumentParser(description="Backtest the SPX short/long volatility strategies.")
    p.add_argument("--start", default="2009-06-01")
    p.add_argument("--end", default=None)
    p.add_argument("--cache-dir", default="data_cache")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--refresh", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = load_index_data(a.start, a.end, cache_dir=a.cache_dir, refresh=a.refresh)
    logging.info("Data %s to %s, %d days", data.index[0].date(), data.index[-1].date(), len(data))

    grid, results = run_grid(data)
    sens, sens_results = run_grid(data, exit_rules=[PRIMARY["exit_rule"]], hedges=(True,),
                                  filters=(PRIMARY["vix_filter"],), atm_ratios=(0.8, 0.9, 1.0))
    results.update(sens_results)
    cols = ["strategy", "exit_rule", "hedge", "vix_filter", "atm_ratio", "n_trades", "time_in_market",
            "ann_return_on_notional", "ann_vol_on_notional", "sharpe", "max_drawdown_on_notional",
            "worst_day_on_notional", "win_rate", "avg_pnl_per_premium", "worst_trade_per_premium",
            "avg_premium_on_notional", "total_costs_on_notional"]
    grid[cols].to_csv(out / "grid.csv")
    sens[cols].to_csv(out / "atm_sensitivity.csv")

    labels = [primary_label(s) for s in STRATEGIES]
    episode_pnl(results, labels).rename(index=lambda l: l.split("|")[0]).to_csv(out / "episodes.csv")
    yearly_returns(results, labels).rename(columns=lambda l: l.split("|")[0]).to_csv(out / "yearly.csv")
    for s, lab in zip(STRATEGIES, labels):
        results[lab].trades.to_csv(out / f"trades_{s}.csv", index=False)

    vrp = variance_risk_premium(data)
    buckets = vrp_by_bucket(vrp)
    buckets.to_csv(out / "vrp_buckets.csv")

    latest = data.iloc[-1]
    status = {
        "data_end": str(data.index[-1].date()), "spx": float(latest["spx"]), "vix": float(latest["vix"]),
        "entry_allowed_next_session": bool(PRIMARY["vix_filter"][0] <= latest["vix"] <= PRIMARY["vix_filter"][1]),
        "vrp_mean": float(vrp["premium"].mean()), "vrp_share_positive": float((vrp["premium"] > 0).mean()),
    }
    (out / "status.json").write_text(json.dumps(status, indent=2))

    _plots(out, data, grid, results, labels, vrp)
    print(grid.loc[labels, ["sharpe", "ann_return_on_notional", "max_drawdown_on_notional",
                            "win_rate", "worst_trade_per_premium"]].round(3).to_string())
    print(json.dumps(status, indent=2))


def _plots(out: Path, data, grid, results, labels, vrp):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    names = {k: v.name for k, v in STRATEGIES.items()}

    # 1. equity curves, primary setup, with VIX underneath
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for s, lab in zip(STRATEGIES, labels):
        ax.plot(results[lab].daily["ret"].cumsum(), lw=1.4, color=COLORS[s], label=names[s])
    ax.axhline(0, color="black", lw=0.7)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Cumulative P&L as a share of one SPX contract's notional\n"
                 "hedged daily with ES, entries when VIX 12-18, held to expiry")
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

    # 2. Sharpe grid: strategy x exit rule, one panel per hedge/filter combination
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True, sharey=True)
    combos = [(True, "(12.0, 18.0)"), (True, "None"), (False, "(12.0, 18.0)"), (False, "None")]
    rules = list(EXIT_RULES)
    for ax, (h, f) in zip(axes.flat, combos):
        sub = grid[(grid["hedge"] == h) & (grid["vix_filter"] == f) & (grid["atm_ratio"] == 0.9)]
        mat = sub.pivot_table(index="strategy", columns="exit_rule", values="sharpe").loc[list(STRATEGIES), rules]
        im = ax.imshow(mat.values, cmap="RdBu", vmin=-1.5, vmax=1.5, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(v) > 0.9 else "black")
        ax.set_title(f"{'Hedged' if h else 'Unhedged'}, {'VIX 12-18 filter' if f != 'None' else 'no VIX filter'}")
        ax.set_xticks(range(len(rules)), rules, rotation=30)
        ax.set_yticks(range(len(STRATEGIES)), [names[s] for s in STRATEGIES])
    fig.colorbar(im, ax=axes, shrink=0.6, label="Sharpe ratio")
    fig.suptitle("Sharpe ratio by strategy and exit rule (ATM vol = 0.9 x VIX, after costs)")
    fig.savefig(out / "sharpe_grid.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 3. variance risk premium
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [2, 1]})
    ax.plot(vrp["vix"], lw=0.8, color="#d95926", label="VIX (implied, priced today)")
    ax.plot(vrp["realized_fwd"], lw=0.8, color="#2a78d6", label="Realised vol over the next 21 trading days")
    ax.set_ylim(0, 90)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Implied versus subsequently realised S&P 500 volatility")
    prem = vrp["premium"].clip(-40, 25)
    ax2.hist(prem, bins=80, color="#199e70")
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_xlabel(f"VIX minus realised vol, points (mean {vrp['premium'].mean():.1f}, "
                   f"positive {100 * (vrp['premium'] > 0).mean():.0f}% of days)")
    for a_ in (ax, ax2):
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "vrp.png", dpi=120)
    plt.close(fig)

    # 4. stress episodes
    ep = episode_pnl(results, labels).rename(index=lambda l: names[l.split("|")[0]])
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(ep.shape[1])
    w = 0.8 / ep.shape[0]
    for k, (name, row) in enumerate(ep.iterrows()):
        key = [s for s in STRATEGIES if names[s] == name][0]
        ax.bar(x + k * w - 0.4 + w / 2, row.values, width=w, color=COLORS[key], label=name)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x, ep.columns, rotation=25, ha="right")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("P&L through stress episodes (primary setup, share of one contract's notional)")
    ax.legend(frameon=False, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "episodes.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
