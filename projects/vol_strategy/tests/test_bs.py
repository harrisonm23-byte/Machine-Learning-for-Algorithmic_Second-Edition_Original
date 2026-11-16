import math

import pytest

from volstrat import bs

S, T = 5500.0, 30 / 365


@pytest.mark.parametrize("K", [4500.0, 5300.0, 5500.0, 5800.0, 6500.0])
@pytest.mark.parametrize("r,q", [(0.0, 0.0), (0.05, 0.015)])
def test_put_call_parity(K, r, q):
    c = bs.price(S, K, T, 0.2, True, r, q)
    p = bs.price(S, K, T, 0.2, False, r, q)
    assert c - p == pytest.approx(S * math.exp(-q * T) - K * math.exp(-r * T), abs=1e-8)


@pytest.mark.parametrize("call", [True, False])
@pytest.mark.parametrize("K", [5000.0, 5500.0, 6000.0])
def test_greeks_match_finite_differences(call, K):
    h, sig = 0.5, 0.18
    up, dn = bs.price(S + h, K, T, sig, call), bs.price(S - h, K, T, sig, call)
    assert bs.delta(S, K, T, sig, call) == pytest.approx((up - dn) / (2 * h), rel=1e-5, abs=1e-8)
    d_up, d_dn = bs.delta(S + h, K, T, sig, call), bs.delta(S - h, K, T, sig, call)
    assert bs.gamma(S, K, T, sig) == pytest.approx((d_up - d_dn) / (2 * h), rel=1e-4, abs=1e-10)
    v_up, v_dn = bs.price(S, K, T, sig + 1e-4, call), bs.price(S, K, T, sig - 1e-4, call)
    assert bs.vega(S, K, T, sig) == pytest.approx((v_up - v_dn) / 2e-4, rel=1e-5)


def test_expiry_values_are_intrinsic():
    assert bs.price(5500, 5400, 0, 0.2, True) == 100
    assert bs.price(5500, 5400, 0, 0.2, False) == 0
    assert bs.delta(5300, 5400, 0, 0.2, False) == -1.0
    assert bs.delta(5300, 5400, 0, 0.2, True) == 0.0


def test_surface_shape():
    surf = bs.Surface(0.15, a=0.6, b=1.0)
    assert surf.vol(S, S) == pytest.approx(0.15)
    down, up = surf.vol(S, S * math.exp(-0.05)), surf.vol(S, S * math.exp(0.05))
    assert down > 0.15 and down > up            # puts richer than calls
    assert bs.Surface(0.01).vol(S, S) == bs.MIN_VOL


@pytest.mark.parametrize("target,call", [(-0.15, False), (0.15, True), (-0.05, False), (0.05, True)])
def test_strike_for_delta_hits_target(target, call):
    surf = bs.Surface(0.15)
    k = bs.strike_for_delta(S, T, surf, target, call, tick=None)
    assert bs.delta(S, k, T, surf.vol(S, k), call) == pytest.approx(target, abs=1e-5)
    assert (k > S) if call else (k < S)
    k5 = bs.strike_for_delta(S, T, surf, target, call)
    assert k5 % 5 == 0 and abs(k5 - k) <= 2.5


def test_strike_for_delta_rejects_wrong_sign():
    with pytest.raises(ValueError):
        bs.strike_for_delta(S, T, bs.Surface(0.15), 0.15, call=False)
