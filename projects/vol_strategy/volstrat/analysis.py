"""Grid runs, the variance risk premium diagnostic and stress-episode P&L."""
from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd

from .backtest import EXIT_RULES, BacktestConfig, run_backtest
from .strategies import STRATEGIES

log = logging.getLogger(__name__)

# Primary rules, fixed before looking at results: the strategy page's own setup.
PRIMARY = dict(hedge=True, vix_filter=(12.0, 18.0), exit_rule="expiry")

EPISODES = {
    "Aug 2011 (US downgrade)": ("2011-07-22", "2011-08-31"),
    "Aug 2015 (China deval.)": ("2015-08-14", "2015-09-15"),
    "Feb 2018 (Volmageddon)": ("2018-01-26", "2018-02-28"),
    "Q4 2018 selloff": ("2018-10-01", "2018-12-31"),
    "Mar 2020 (COVID)": ("2020-02-19", "2020-04-15"),
    "2022 bear market": ("2022-01-03", "2022-10-14"),
    "Aug 2024 (yen carry)": ("2024-07-24", "2024-08-15"),
    "Apr 2025 (tariffs)": ("2025-03-28", "2025-04-25"),
}


def primary_config(strategy: str, surface: dict) -> BacktestConfig:
    return BacktestConfig(strategy=strategy, **PRIMARY, **surface)


def run_grid(data: pd.DataFrame, surface: dict, strategies=None, exit_rules=None,
             hedges=(True, False), filters=((12.0, 18.0), None), **overrides) -> tuple[pd.DataFrame, dict]:
    """Run every combination on one surface; return a summary table and results by label."""
    strategies = strategies or list(STRATEGIES)
    exit_rules = exit_rules or list(EXIT_RULES)
    rows, results = [], {}
    for s, e, h, f in itertools.product(strategies, exit_rules, hedges, filters):
        cfg = BacktestConfig(strategy=s, exit_rule=e, hedge=h, vix_filter=f, **surface, **overrides)
        res = run_backtest(data, cfg)
        results[cfg.label] = res
        rows.append(res.summary)
        log.info("%s sharpe=%.2f", cfg.label, res.summary["sharpe"])
    return pd.DataFrame(rows).set_index("label"), results


def variance_risk_premium(data: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """VIX (implied, 30 calendar days) versus realised S&P 500 volatility over the
    next ``horizon`` trading days (t+1 .. t+horizon), both in annualised vol points."""
    logret = np.log(data["spx"]).diff()
    fwd_var = logret.pow(2).rolling(horizon).sum().shift(-horizon) * 252 / horizon
    out = pd.DataFrame({"vix": data["vix"], "realized_fwd": np.sqrt(fwd_var) * 100}).dropna()
    out["premium"] = out["vix"] - out["realized_fwd"]
    return out


def vrp_by_bucket(vrp: pd.DataFrame, bins=(0, 12, 15, 18, 22, 30, 200)) -> pd.DataFrame:
    b = pd.cut(vrp["vix"], bins=list(bins), right=False)
    g = vrp.groupby(b, observed=True)
    return pd.DataFrame({
        "days": g.size(),
        "avg_vix": g["vix"].mean(),
        "avg_realized_next_21d": g["realized_fwd"].mean(),
        "avg_premium": g["premium"].mean(),
        "share_implied_above_realized": g["premium"].apply(lambda x: (x > 0).mean()),
        "worst_premium": g["premium"].min(),
    })


def episode_pnl(results: dict, labels: list[str]) -> pd.DataFrame:
    """P&L over each stress window as a share of one contract's notional."""
    rows = {}
    for lab in labels:
        r = results[lab].daily["ret"]
        rows[lab] = {name: float(r.loc[a:b].sum()) for name, (a, b) in EPISODES.items()}
    return pd.DataFrame(rows).T


def yearly_returns(results: dict, labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame({lab: results[lab].daily["ret"].groupby(results[lab].daily.index.year).sum()
                         for lab in labels})
