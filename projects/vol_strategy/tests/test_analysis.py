import json

import numpy as np
import pandas as pd
import pytest

from volstrat import __main__ as runner
from volstrat.analysis import (EPISODES, PRIMARY, episode_pnl, primary_config, run_grid,
                               variance_risk_premium, yearly_returns)
from volstrat.surfaces import PAGE, calibrated


def market(n=320, seed=0, start="2018-01-01"):
    rng = np.random.default_rng(seed)
    spx = 2500 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    vix = np.clip(15 + np.cumsum(rng.normal(0, 0.5, n)) * 0.3, 10, 40)
    return pd.DataFrame({"spx": spx, "vix": vix}, index=pd.bdate_range(start, periods=n))


def test_vrp_uses_the_next_21_days():
    d = market()
    vrp = variance_risk_premium(d)
    t = d.index[50]
    logret = np.log(d["spx"]).diff()
    manual = np.sqrt((logret.iloc[51:72] ** 2).sum() * 252 / 21) * 100
    assert vrp.loc[t, "realized_fwd"] == pytest.approx(manual)
    assert vrp.loc[t, "premium"] == pytest.approx(d.loc[t, "vix"] - manual)
    assert vrp.index[-1] == d.index[-22]          # no forward window past the data


def test_grid_labels_filters_and_reconciliation():
    d = market()
    grid, results = run_grid(d, PAGE, strategies=["strangle"], exit_rules=["expiry"])
    primary = primary_config("strangle", PAGE).label
    assert primary in results
    assert set(grid["vix_filter"]) == {runner.FILTER_ON, runner.FILTER_OFF}
    yr = yearly_returns(results, [primary])
    assert yr[primary].sum() == pytest.approx(results[primary].daily["ret"].sum())
    ep = episode_pnl(results, [primary])
    a, b = EPISODES["Feb 2018 (Volmageddon)"]
    assert ep.loc[primary, "Feb 2018 (Volmageddon)"] == pytest.approx(results[primary].daily["ret"].loc[a:b].sum())


def test_primary_rules_are_the_strategy_page_setup():
    assert PRIMARY == dict(hedge=True, vix_filter=(12.0, 18.0), exit_rule="expiry")


def test_calibrated_preset_loads_and_shifts(tmp_path):
    params = {"atm_ratio": 0.8, "atm_intercept": 1.0, "skew_a": 0.4, "skew_a_vix": 0.02, "skew_b": 5.0,
              "skew_b_vix": -0.18, "term_atm": 0.0, "term_atm_vix": 0.0, "term_a": 0.0, "term_b": 0.0,
              "wing_z": [-5.0, 2.0], "_fit": {"n_obs": 1}}
    f = tmp_path / "p.json"
    f.write_text(json.dumps(params))
    c = calibrated(f)
    assert c["surface_name"] == "calibrated" and c["wing_z"] == (-5.0, 2.0) and "_fit" not in c
    up = calibrated(f, atm_shift=1.0)
    assert up["atm_intercept"] == 2.0 and up["surface_name"] == "calibrated+1"
    cfg = primary_config("book", c)
    assert cfg.surface(15).atm == pytest.approx((0.8 * 15 + 1.0) / 100)
