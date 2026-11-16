import math

import numpy as np
import pandas as pd
import pytest

from volstrat.calibrate import _parity_forward, black76, implied_vol, occ_symbol


def test_occ_symbol_format():
    assert occ_symbol("SPY", pd.Timestamp("2024-12-20"), False, 560) == "SPY241220P00560000"
    assert occ_symbol("SPY", pd.Timestamp("2024-12-20"), True, 612.5) == "SPY241220C00612500"


@pytest.mark.parametrize("call", [True, False])
@pytest.mark.parametrize("K", [540.0, 590.0, 600.0, 640.0])
@pytest.mark.parametrize("vol", [0.08, 0.15, 0.45])
def test_implied_vol_round_trip(call, K, vol):
    F, T = 600.0, 30 / 365
    px = black76(F, K, T, vol, call)
    if px < 1e-4:
        pytest.skip("price too small to invert reliably")
    assert implied_vol(px, F, K, T, call) == pytest.approx(vol, abs=1e-5)


def test_implied_vol_rejects_prices_below_intrinsic():
    assert np.isnan(implied_vol(5.0, 600.0, 590.0, 30 / 365, True))   # intrinsic is 10


def test_black76_put_call_parity():
    F, K, T, df = 600.0, 580.0, 0.1, 0.995
    c, p = black76(F, K, T, 0.2, True, df), black76(F, K, T, 0.2, False, df)
    assert c - p == pytest.approx(df * (F - K))


def test_parity_forward_recovers_forward():
    F, T, df = 603.2, 30 / 365, math.exp(-0.045 * 30 / 365)
    ks = np.arange(592, 610, 1.0)
    calls = {k: black76(F, k, T, 0.13, True, df) for k in ks}
    puts = {k: black76(F, k, T, 0.13, False, df) for k in ks}
    assert _parity_forward(calls, puts, S=600.0) == pytest.approx(F, rel=1e-9)
    assert _parity_forward({}, {}, S=600.0) is None
