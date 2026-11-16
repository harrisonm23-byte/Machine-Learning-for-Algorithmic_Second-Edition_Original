import numpy as np
import pandas as pd
import pytest

from volstrat import bs
from volstrat.backtest import BacktestConfig, _expiry_for, run_backtest

FREE = dict(opt_half_spread_pct=0.0, opt_half_spread_min=0.0, opt_commission=0.0, fut_cost_per_unit=0.0)


def market(spx, vix, start="2020-01-01"):
    dates = pd.bdate_range(start, periods=len(spx))
    return pd.DataFrame({"spx": np.asarray(spx, float), "vix": np.asarray(vix, float)}, index=dates)


def gbm(n, sigma, seed, s0=5000.0):
    rng = np.random.default_rng(seed)
    r = rng.normal(-0.5 * sigma**2 / 252, sigma / np.sqrt(252), n)
    return s0 * np.exp(np.concatenate([[0.0], np.cumsum(r[1:])]))


@pytest.mark.parametrize("strategy", ["book", "strangle", "condor", "longstr", "longput", "longcall"])
@pytest.mark.parametrize("hedge", [True, False])
def test_daily_and_trade_pnl_reconcile(strategy, hedge):
    data = market(gbm(400, 0.18, seed=1), np.full(400, 15.0))
    res = run_backtest(data, BacktestConfig(strategy=strategy, hedge=hedge, exit_rule="tp_sl"))
    d, t = res.daily, res.trades
    assert np.allclose(d["pnl"], d["pnl_options"] + d["pnl_hedge"] - d["cost"])
    assert d["pnl"].sum() == pytest.approx(t["pnl"].sum(), rel=1e-9, abs=1e-6)
    assert t["pnl"].sum() == pytest.approx((t["pnl_options"] + t["pnl_hedge"] - t["costs"]).sum(), abs=1e-6)
    if not hedge:
        assert (t["pnl_hedge"] == 0).all()


def test_flat_market_short_strangle_keeps_premium():
    data = market(np.full(60, 5000.0), np.full(60, 15.0))
    res = run_backtest(data, BacktestConfig(strategy="strangle", hedge=False, **FREE))
    first = res.trades.iloc[0]
    assert first["reason"] == "expiry"
    assert first["premium"] > 0
    assert first["pnl"] == pytest.approx(first["premium"], rel=1e-9)


def test_expiry_settles_at_intrinsic():
    spx = np.r_[np.full(10, 5000.0), np.full(50, 4000.0)]    # crash through the put strike
    data = market(spx, np.full(60, 15.0))
    res = run_backtest(data, BacktestConfig(strategy="longput", hedge=False, **FREE))
    t = res.trades.iloc[0]
    strike = float(t["strikes"].split("x")[0][1:])
    assert t["reason"] == "expiry"
    assert t["pnl"] == pytest.approx((strike - 4000.0) * 100 - abs(t["premium"]), rel=1e-9)


def test_hedge_pnl_uses_previous_hedge():
    data = market([5000, 5010, 4990, 5000, 5000], np.full(5, 15.0))
    cfg = BacktestConfig(strategy="book", hedge=True, vix_filter=None, signal_lag=0, **FREE)
    res = run_backtest(data, cfg)
    trade = res.trades.iloc[0]
    assert trade["entry_date"] == data.index[0]
    surf = bs.Surface(0.9 * 0.15)
    legs = [(s[0] == "C", float(s[1:].split("x")[0]), int(s.split("x")[1])) for s in trade["strikes"].split(";")]
    T = (trade["expiry"] - data.index[0]).days / 365
    delta0 = sum(q * bs.delta(5000, k, T, surf.vol(5000, k), c) * 100 for c, k, q in legs)
    # the hedge set at day 0's close (-delta0 $/pt) earns day 1's +10 point move
    assert res.daily["pnl_hedge"].iloc[1] == pytest.approx(-delta0 * 10, rel=1e-9)


def test_entry_filter_uses_lagged_vix():
    # day 0 VIX 15 (allowed), day 1 VIX 50: with lag 1 the day-1 entry is still allowed
    spx = np.full(40, 5000.0)
    vix = np.r_[30.0, 15.0, 50.0, np.full(37, 30.0)]
    res = run_backtest(market(spx, vix), BacktestConfig(strategy="strangle", signal_lag=1))
    assert res.trades.iloc[0]["entry_date"] == market(spx, vix).index[2]
    # with lag 0 the entry happens on the day the VIX itself is 15
    res0 = run_backtest(market(spx, vix), BacktestConfig(strategy="strangle", signal_lag=0))
    assert res0.trades.iloc[0]["entry_date"] == market(spx, vix).index[1]


def test_no_entries_when_filter_never_met():
    res = run_backtest(market(np.full(50, 5000.0), np.full(50, 30.0)), BacktestConfig(strategy="book"))
    assert res.trades.empty and (res.daily["pnl"] == 0).all()


def test_take_profit_triggers_on_vol_crush():
    vix = np.r_[np.full(3, 17.0), np.full(57, 9.0)]
    res = run_backtest(market(np.full(60, 5000.0), vix),
                       BacktestConfig(strategy="strangle", exit_rule="tp", **FREE))
    t = res.trades.iloc[0]
    assert t["reason"] == "take_profit"
    assert t["pnl"] >= 0.5 * t["premium"]


def test_stop_loss_and_dte_exit():
    spx = np.r_[np.full(3, 5000.0), np.full(57, 4400.0)]
    res = run_backtest(market(spx, np.full(60, 15.0)),
                       BacktestConfig(strategy="book", hedge=False, exit_rule="tp_sl", **FREE))
    t = res.trades.iloc[0]
    assert t["reason"] == "stop_loss" and t["pnl"] <= -2 * t["premium"]
    res = run_backtest(market(np.full(60, 5000.0), np.full(60, 15.0)),
                       BacktestConfig(strategy="book", exit_rule="dte"))
    t = res.trades.iloc[0]
    days_left = (t["expiry"] - t["exit_date"]).days
    assert t["reason"] == "dte_exit" and 7 < days_left <= 10   # first close with <= 10 days left


def test_expiry_is_last_trading_day_on_or_before_target():
    dates = pd.bdate_range("2024-01-01", periods=60)
    d = dates[0]                      # Mon 1 Jan; +30 days = Wed 31 Jan
    assert _expiry_for(dates, d, 30) == pd.Timestamp("2024-01-31")
    d = pd.Timestamp("2024-01-04")    # Thu; +30 = Sat 3 Feb -> Fri 2 Feb
    assert _expiry_for(dates, d, 30) == pd.Timestamp("2024-02-02")


@pytest.mark.parametrize("implied,expect_edge", [(0.15, False), (0.21, True)])
def test_hedged_short_vol_edge_is_implied_minus_realized(implied, expect_edge):
    """Theory check: daily delta-hedged short options earn roughly nothing when
    implied equals realized vol, and earn the premium spread when implied is higher."""
    realized = 0.15
    n = 252 * 30
    spx = gbm(n, realized, seed=7)
    data = market(spx, np.full(n, implied * 100), start="1990-01-01")
    cfg = BacktestConfig(strategy="strangle", hedge=True, vix_filter=None, atm_ratio=1.0,
                         skew_a=0.0, skew_b=0.0, **FREE)
    t = run_backtest(data, cfg).trades
    ratio = (t["pnl"] / t["premium"]).mean()
    stderr = (t["pnl"] / t["premium"]).std() / np.sqrt(len(t))
    if expect_edge:
        assert ratio > 0.25 and ratio > 4 * stderr
    else:
        assert abs(ratio) < max(0.08, 3 * stderr)
