"""Daily backtest of a delta-hedged SPX option position.

How one day is processed, at the close of date t:

1. Mark the open position: every leg is repriced with Black-Scholes on the
   skewed surface whose ATM level is ``atm_ratio * VIX_t``. The option P&L
   is the change in value, and the hedge P&L is
   ``hedge_{t-1} * (S_t - S_{t-1})``.
2. Check exits in this order: expiry, take profit, stop loss, days to expiry.
   An exit pays the option half-spread and commission (none at cash-settled
   expiry) and closes the hedge.
3. Otherwise re-hedge to delta neutral (if hedging is on) and pay the futures
   cost on the traded amount.
4. If flat, open a new position when the entry filter allows it. The filter
   reads the VIX ``signal_lag`` days earlier (1 by default), so the decision
   only uses information available before the trade.

Units: option quantities are contracts with a $100 multiplier. The hedge is
held in dollars per index point (one ES contract = $50/pt, one MES = $5/pt)
and may be fractional. Daily returns are P&L divided by the notional of one
SPX contract (index level x 100) at the previous close.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import bs
from .strategies import STRATEGIES, StrategySpec

EXIT_RULES = {
    "expiry": dict(tp=False, sl=False, dte=False),
    "tp": dict(tp=True, sl=False, dte=False),
    "tp_sl": dict(tp=True, sl=True, dte=False),
    "dte": dict(tp=False, sl=False, dte=True),
    "tp_dte": dict(tp=True, sl=False, dte=True),
}


@dataclass
class BacktestConfig:
    strategy: str = "book"
    hedge: bool = True
    vix_filter: tuple[float, float] | None = (12.0, 18.0)
    signal_lag: int = 1
    dte: int = 30                    # calendar days to expiry at entry
    exit_rule: str = "expiry"        # see EXIT_RULES
    tp_short: float = 0.5            # short vol: take profit at 50% of the credit
    sl_short: float = 2.0            # short vol: stop at a loss of 2x the credit
    tp_long: float = 1.0             # long vol: take profit at +100% of the debit
    sl_long: float = 0.5             # long vol: stop at -50% of the debit
    exit_dte: int = 10               # close with this many calendar days left
    atm_ratio: float = 0.9           # ATM implied vol = atm_ratio * VIX
    skew_a: float = 0.6
    skew_b: float = 1.0
    opt_half_spread_pct: float = 0.01
    opt_half_spread_min: float = 0.05   # index points
    opt_commission: float = 1.50        # $ per contract per side, incl. exchange fees
    fut_cost_per_unit: float = 0.175    # $ per ($1/pt) of hedge traded: ES half-spread 0.125 pt + $2.50 commission per $50/pt
    multiplier: float = 100.0
    strike_tick: float = 5.0

    def __post_init__(self):
        if self.strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {self.strategy!r}")
        if self.exit_rule not in EXIT_RULES:
            raise ValueError(f"unknown exit rule {self.exit_rule!r}")
        if self.signal_lag < 0:
            raise ValueError("signal_lag must be >= 0")

    @property
    def label(self) -> str:
        filt = "vix{:g}-{:g}".format(*self.vix_filter) if self.vix_filter else "nofilter"
        return f"{self.strategy}|{self.exit_rule}|{'hedged' if self.hedge else 'unhedged'}|{filt}|atm{self.atm_ratio:g}"


@dataclass
class _Position:
    legs: list            # (call, strike, qty)
    entry_date: pd.Timestamp
    expiry: pd.Timestamp
    spx_entry: float
    vix_entry: float
    premium: float        # + credit received, - debit paid (mid, before costs)
    value: float          # current mark of the options ($, signed)
    hedge: float = 0.0    # $ per index point
    pnl: float = 0.0      # running P&L incl. costs
    pnl_options: float = 0.0
    pnl_hedge: float = 0.0
    costs: float = 0.0
    hedge_traded: float = 0.0
    worst_pnl: float = 0.0


@dataclass
class BacktestResult:
    config: BacktestConfig
    daily: pd.DataFrame
    trades: pd.DataFrame
    summary: dict = field(default_factory=dict)


def _mark(legs, S: float, T: float, surf: bs.Surface, mult: float) -> tuple[float, float]:
    value = dlt = 0.0
    for call, K, qty in legs:
        vol = surf.vol(S, K)
        value += qty * bs.price(S, K, T, vol, call) * mult
        dlt += qty * bs.delta(S, K, T, vol, call) * mult
    return value, dlt


def _option_trade_cost(legs, S: float, T: float, surf: bs.Surface, cfg: BacktestConfig) -> float:
    cost = 0.0
    for call, K, qty in legs:
        px = bs.price(S, K, T, surf.vol(S, K), call)
        half = max(cfg.opt_half_spread_min, cfg.opt_half_spread_pct * px)
        cost += abs(qty) * (half * cfg.multiplier + cfg.opt_commission)
    return cost


def _expiry_for(dates: pd.DatetimeIndex, d: pd.Timestamp, dte: int) -> pd.Timestamp:
    """Last trading day on or before d + dte calendar days (the listed calendar is known in advance)."""
    target = d + pd.Timedelta(days=dte)
    if target > dates[-1]:
        return target
    return dates[dates.searchsorted(target, side="right") - 1]


def run_backtest(data: pd.DataFrame, cfg: BacktestConfig) -> BacktestResult:
    spec: StrategySpec = STRATEGIES[cfg.strategy]
    rules = EXIT_RULES[cfg.exit_rule]
    dates = data.index
    S = data["spx"].to_numpy(float)
    V = data["vix"].to_numpy(float)
    sig = data["vix"].shift(cfg.signal_lag).to_numpy(float)
    mult = cfg.multiplier

    daily, trades = [], []
    pos: _Position | None = None

    def close(p: _Position, d, reason, S_now, T_now, surf, expired):
        cost = 0.0 if expired else _option_trade_cost(p.legs, S_now, T_now, surf, cfg)
        cost += abs(p.hedge) * cfg.fut_cost_per_unit
        p.hedge_traded += abs(p.hedge)
        p.pnl -= cost
        p.costs += cost
        trades.append(dict(
            entry_date=p.entry_date, exit_date=d, expiry=p.expiry, reason=reason,
            spx_entry=p.spx_entry, spx_exit=S_now, vix_entry=p.vix_entry,
            strikes=";".join(f"{'C' if c else 'P'}{k:g}x{q}" for c, k, q in p.legs),
            premium=p.premium, pnl=p.pnl, pnl_options=p.pnl_options, pnl_hedge=p.pnl_hedge,
            costs=p.costs, hedge_traded=p.hedge_traded, worst_pnl=min(p.worst_pnl, p.pnl),
            days_held=(d - p.entry_date).days,
            notional=p.spx_entry * mult,
        ))
        return cost

    for i, d in enumerate(dates):
        s, vix = S[i], V[i]
        surf = bs.Surface(cfg.atm_ratio * vix / 100.0, cfg.skew_a, cfg.skew_b)
        pnl_opt = pnl_hedge = cost = 0.0
        exited = False

        # 1-3: mark, exit or re-hedge an open position
        if pos is not None:
            days_left = (pos.expiry - d).days
            T = max(days_left, 0) / 365.0
            value, dlt = _mark(pos.legs, s, T, surf, mult)
            pnl_opt = value - pos.value
            pnl_hedge = pos.hedge * (s - S[i - 1])
            pos.value = value
            pos.pnl += pnl_opt + pnl_hedge
            pos.pnl_options += pnl_opt
            pos.pnl_hedge += pnl_hedge
            pos.worst_pnl = min(pos.worst_pnl, pos.pnl)

            size = abs(pos.premium)
            if spec.is_short_vol:
                tp_level, sl_level = cfg.tp_short * size, cfg.sl_short * size
            else:
                tp_level, sl_level = cfg.tp_long * size, cfg.sl_long * size

            reason = None
            if days_left <= 0:
                reason = "expiry"
            elif rules["tp"] and pos.pnl >= tp_level:
                reason = "take_profit"
            elif rules["sl"] and pos.pnl <= -sl_level:
                reason = "stop_loss"
            elif rules["dte"] and days_left <= cfg.exit_dte:
                reason = "dte_exit"
            elif i == len(dates) - 1:
                reason = "end_of_data"

            if reason is not None:
                cost += close(pos, d, reason, s, T, surf, expired=(reason == "expiry"))
                pos = None
                exited = True
            elif cfg.hedge:
                new_hedge = -dlt
                traded = abs(new_hedge - pos.hedge)
                c = traded * cfg.fut_cost_per_unit
                pos.hedge = new_hedge
                pos.hedge_traded += traded
                pos.pnl -= c
                pos.costs += c
                cost += c

        # 4: open a new position (same-day roll allowed after an exit)
        if pos is None and i < len(dates) - 1:
            sv = sig[i]
            allowed = not np.isnan(sv) and (
                cfg.vix_filter is None or cfg.vix_filter[0] <= sv <= cfg.vix_filter[1])
            if allowed:
                expiry = _expiry_for(dates, d, cfg.dte)
                T0 = (expiry - d).days / 365.0
                if T0 > 0:
                    legs = [(leg.call,
                             bs.strike_for_delta(s, T0, surf, leg.strike_delta, leg.call, tick=cfg.strike_tick),
                             leg.qty) for leg in spec.legs]
                    value, dlt = _mark(legs, s, T0, surf, mult)
                    hedge = -dlt if cfg.hedge else 0.0
                    c = _option_trade_cost(legs, s, T0, surf, cfg) + abs(hedge) * cfg.fut_cost_per_unit
                    pos = _Position(legs=legs, entry_date=d, expiry=expiry, spx_entry=s, vix_entry=vix,
                                    premium=-value, value=value, hedge=hedge, pnl=-c, costs=c,
                                    hedge_traded=abs(hedge), worst_pnl=-c)
                    cost += c

        prev_notional = (S[i - 1] if i > 0 else s) * mult
        pnl = pnl_opt + pnl_hedge - cost
        daily.append(dict(date=d, spx=s, vix=vix, pnl=pnl, pnl_options=pnl_opt, pnl_hedge=pnl_hedge,
                          cost=cost, ret=pnl / prev_notional, in_position=pos is not None or exited))

    daily_df = pd.DataFrame(daily).set_index("date")
    trades_df = pd.DataFrame(trades)
    result = BacktestResult(cfg, daily_df, trades_df)
    result.summary = summarize(result)
    return result


def _max_drawdown(cum: pd.Series) -> float:
    return float((cum - cum.cummax()).min()) if len(cum) else 0.0


def summarize(res: BacktestResult) -> dict:
    d, t = res.daily, res.trades
    r = d["ret"]
    n = len(r)
    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    cum = r.cumsum()
    out = dict(
        label=res.config.label, **{k: v for k, v in asdict(res.config).items()
                                   if k in ("strategy", "exit_rule", "hedge", "atm_ratio")},
        vix_filter=str(res.config.vix_filter),
        start=str(d.index[0].date()), end=str(d.index[-1].date()), n_days=n,
        time_in_market=float(d["in_position"].mean()),
        ann_return_on_notional=float(ann_ret),
        ann_vol_on_notional=float(ann_vol),
        sharpe=float(ann_ret / ann_vol) if ann_vol > 0 else np.nan,
        max_drawdown_on_notional=_max_drawdown(cum),
        worst_day_on_notional=float(r.min()),
        cvar5_daily=float(r[r <= r.quantile(0.05)].mean()),
        total_costs_on_notional=float((d["cost"] / (d["spx"].shift().fillna(d["spx"]) * res.config.multiplier)).sum()),
        n_trades=int(len(t)),
    )
    if len(t):
        prem = t["premium"].abs()
        out.update(
            win_rate=float((t["pnl"] > 0).mean()),
            avg_pnl_per_premium=float((t["pnl"] / prem).mean()),
            median_pnl_per_premium=float((t["pnl"] / prem).median()),
            worst_trade_per_premium=float((t["pnl"] / prem).min()),
            best_trade_per_premium=float((t["pnl"] / prem).max()),
            worst_trade_on_notional=float((t["pnl"] / t["notional"]).min()),
            avg_days_held=float(t["days_held"].mean()),
            avg_premium_on_notional=float((t["premium"] / t["notional"]).mean()),
            exit_reasons=t["reason"].value_counts().to_dict(),
        )
    return out
