"""Regression tests for the issues confirmed in the multi-agent code review."""
import numpy as np
import pandas as pd
import pytest

from volstrat import bs
from volstrat.backtest import BacktestConfig, _expiry_for, _option_trade_cost, run_backtest
from volstrat.strategies import STRATEGIES

FREE = dict(opt_half_spread_pct=0.0, opt_half_spread_min=0.0, opt_commission=0.0, fut_cost_per_unit=0.0)


def market(spx, vix, start="2020-01-01"):
    dates = pd.bdate_range(start, periods=len(spx))
    return pd.DataFrame({"spx": np.asarray(spx, float), "vix": np.asarray(vix, float)}, index=dates)


def gbm(n, sigma, seed, s0=5000.0):
    rng = np.random.default_rng(seed)
    r = rng.normal(-0.5 * sigma**2 / 252, sigma / np.sqrt(252), n)
    return s0 * np.exp(np.concatenate([[0.0], np.cumsum(r[1:])]))


# --- iron condor was classified as long vol --------------------------------------------
@pytest.mark.parametrize("key,short", [("book", True), ("strangle", True), ("condor", True),
                                       ("longstr", False), ("longput", False), ("longcall", False)])
def test_short_vol_classification(key, short):
    assert STRATEGIES[key].is_short_vol is short


@pytest.mark.parametrize("key", ["book", "strangle", "condor"])
def test_credit_positions_use_short_vol_take_profit(key):
    vix = np.r_[np.full(3, 17.0), np.full(57, 9.0)]          # vol crush: shorts profit
    res = run_backtest(market(np.full(60, 5000.0), vix),
                       BacktestConfig(strategy=key, exit_rule="tp", hedge=False, **FREE))
    t = res.trades.iloc[0]
    assert t["premium"] > 0
    assert t["reason"] == "take_profit"
    assert 0.5 * t["premium"] <= t["pnl"] < 1.0 * t["premium"]


def test_condor_stop_uses_short_vol_level():
    spx = np.r_[np.full(3, 5000.0), np.full(57, 4500.0)]
    res = run_backtest(market(spx, np.full(60, 15.0)),
                       BacktestConfig(strategy="condor", exit_rule="tp_sl", hedge=False, **FREE))
    t = res.trades.iloc[0]
    # with the old long-vol level it would have stopped at -0.5x the credit
    assert t["reason"] in ("stop_loss", "expiry")
    if t["reason"] == "stop_loss":
        assert t["pnl"] <= -2.0 * t["premium"]


# --- strike solver with realistic smile curvature ---------------------------------------
@pytest.mark.parametrize("b,wings", [(1.0, None), (2.0, None), (3.0, None), (3.0, (-5, 2)), (6.0, (-5, 2))])
@pytest.mark.parametrize("target,call", [(-0.15, False), (0.15, True), (-0.05, False), (0.05, True)])
def test_strike_for_delta_stays_near_the_money(b, wings, target, call):
    S, T = 5000.0, 30 / 365
    surf = bs.Surface(0.13, a=0.7, b=b, wing_z=wings)
    k = bs.strike_for_delta(S, T, surf, target, call, tick=None)
    assert 0.75 < k / S < 1.20
    assert bs.delta(S, k, T, surf.vol(S, k, T), call) == pytest.approx(target, abs=1e-5)


# --- tenor-dependent smile --------------------------------------------------------------
def test_surface_tenor_scaling():
    flat = bs.Surface(0.15, a=0.6, b=1.0)
    term = bs.Surface(0.15, a=0.6, b=1.0, term_atm=0.1, term_a=0.5, term_b=0.5)
    S, K = 5000.0, 4700.0
    # identical at the 30-day reference tenor and when no tenor is given
    assert term.vol(S, K, bs.T_REF) == pytest.approx(flat.vol(S, K))
    assert term.vol(S, K) == pytest.approx(flat.vol(S, K))
    # shorter tenor: steeper put skew (and the ATM level moves by exp(term_atm * x))
    assert term.vol(S, K, 7 / 365) > term.vol(S, K, 30 / 365)
    x = np.log(bs.T_REF / (7 / 365))
    assert term.vol(S, S, 7 / 365) == pytest.approx(0.15 * np.exp(0.1 * x))
    # finite at expiry
    assert np.isfinite(term.vol(S, K, 0.0))


def test_from_vix_defaults_reproduce_page_model():
    s = bs.Surface.from_vix(15.0)
    assert (s.atm, s.a, s.b) == pytest.approx((0.135, 0.6, 1.0))
    cfg = BacktestConfig()
    assert cfg.surface(15.0).vol(5000, 4800, 0.05) == pytest.approx(s.vol(5000, 4800))


# --- costs are charged exactly once -----------------------------------------------------
def test_costs_entry_only_at_expiry_and_entry_plus_exit_otherwise():
    data = market(np.full(60, 5000.0), np.full(60, 15.0))
    cfg = BacktestConfig(strategy="strangle", hedge=False)
    t = run_backtest(data, cfg).trades.iloc[0]
    legs = [(s[0] == "C", float(s[1:].split("x")[0]), int(s.split("x")[1])) for s in t["strikes"].split(";")]
    T0 = (t["expiry"] - t["entry_date"]).days / 365
    entry_cost = _option_trade_cost(legs, 5000.0, T0, cfg.surface(15.0), cfg)
    assert t["reason"] == "expiry"
    assert t["costs"] == pytest.approx(entry_cost)

    cfg = BacktestConfig(strategy="strangle", hedge=False, exit_rule="dte")
    t = run_backtest(data, cfg).trades.iloc[0]
    T_exit = (t["expiry"] - t["exit_date"]).days / 365
    exit_cost = _option_trade_cost(legs, 5000.0, T_exit, cfg.surface(15.0), cfg)
    assert t["costs"] == pytest.approx(entry_cost + exit_cost)


def test_hedge_costs_equal_traded_amount():
    data = market(gbm(300, 0.2, seed=3), np.full(300, 15.0))
    cfg = BacktestConfig(strategy="book", hedge=True, opt_half_spread_pct=0.0, opt_half_spread_min=0.0,
                         opt_commission=0.0)
    t = run_backtest(data, cfg).trades
    assert t["costs"].to_numpy() == pytest.approx((t["hedge_traded"] * cfg.fut_cost_per_unit).to_numpy())


# --- reporting fixes ----------------------------------------------------------------------
def test_vix_signal_recorded_and_inside_filter():
    rng = np.random.default_rng(0)
    vix = np.clip(15 + rng.normal(0, 4, 400).cumsum() * 0.2, 9, 40)
    res = run_backtest(market(gbm(400, 0.15, seed=4), vix), BacktestConfig(strategy="book"))
    t = res.trades
    assert len(t) and t["vix_signal"].between(12, 18).all()


def test_open_position_at_end_is_not_a_trade_statistic():
    data = market(np.full(45, 5000.0), np.full(45, 15.0))
    res = run_backtest(data, BacktestConfig(strategy="strangle", hedge=False, **FREE))
    assert res.trades["reason"].iloc[-1] == "end_of_data"
    assert res.summary["open_at_end"] == 1
    assert res.summary["n_trades"] == len(res.trades) - 1


def test_expiry_past_data_end_is_a_weekday():
    dates = pd.bdate_range("2026-11-02", "2026-11-13")
    e = _expiry_for(dates, pd.Timestamp("2026-11-12"), 30)    # target Sat 12 Dec
    assert e == pd.Timestamp("2026-12-11") and e.weekday() == 4


def test_hedged_short_vol_beta_is_reported():
    data = market(gbm(500, 0.18, seed=5), np.full(500, 15.0))
    s = run_backtest(data, BacktestConfig(strategy="book", vix_filter=None)).summary
    assert np.isfinite(s["beta_to_spx"]) and np.isfinite(s["corr_to_spx"])


def test_wings_flatten_outside_fit_range():
    S, T = 5000.0, 30 / 365
    surf = bs.Surface(0.13, a=0.7, b=3.0, wing_z=(-5, 2))
    width = 0.13 * np.sqrt(T)
    edge = S * np.exp(-5 * width)
    assert surf.vol(S, edge * 0.9, T) == pytest.approx(surf.vol(S, edge, T))
    assert surf.vol(S, S * 0.97, T) == pytest.approx(bs.Surface(0.13, a=0.7, b=3.0).vol(S, S * 0.97, T))
