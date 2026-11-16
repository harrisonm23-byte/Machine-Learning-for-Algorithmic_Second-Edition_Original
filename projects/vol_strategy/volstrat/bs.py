"""Black-Scholes pricing, Greeks and the skewed volatility surface.

Conventions
-----------
* ``T`` is time to expiry in years (calendar days / 365).
* ``sigma`` is annualised volatility as a decimal (0.15 = 15 vol points).
* ``vega`` is per 1.00 of vol (divide by 100 for "per vol point").
* Rates and dividends default to zero, as in the strategy page's model.
"""
from __future__ import annotations

import math

from scipy.optimize import brentq
from scipy.stats import norm

MIN_VOL = 0.03   # same floor as the strategy page
MAX_VOL = 3.00   # guard against the quadratic smile exploding far from the money


def _d1d2(S: float, K: float, T: float, sigma: float, r: float, q: float):
    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    return d1, d1 - sigma * sqrt_t


def intrinsic(S: float, K: float, call: bool) -> float:
    return max(S - K, 0.0) if call else max(K - S, 0.0)


def price(S: float, K: float, T: float, sigma: float, call: bool, r: float = 0.0, q: float = 0.0) -> float:
    if T <= 0 or sigma <= 0:
        return intrinsic(S, K, call)
    d1, d2 = _d1d2(S, K, T, sigma, r, q)
    df_r, df_q = math.exp(-r * T), math.exp(-q * T)
    if call:
        return S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)
    return K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1)


def delta(S: float, K: float, T: float, sigma: float, call: bool, r: float = 0.0, q: float = 0.0) -> float:
    if T <= 0 or sigma <= 0:
        if call:
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1, _ = _d1d2(S, K, T, sigma, r, q)
    df_q = math.exp(-q * T)
    return df_q * norm.cdf(d1) if call else -df_q * norm.cdf(-d1)


def gamma(S: float, K: float, T: float, sigma: float, r: float = 0.0, q: float = 0.0) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1d2(S, K, T, sigma, r, q)
    return math.exp(-q * T) * norm.pdf(d1) / (S * sigma * math.sqrt(T))


def vega(S: float, K: float, T: float, sigma: float, r: float = 0.0, q: float = 0.0) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1d2(S, K, T, sigma, r, q)
    return S * math.exp(-q * T) * norm.pdf(d1) * math.sqrt(T)


# --------------------------------------------------------------------------- #
# volatility surface
# --------------------------------------------------------------------------- #
class Surface:
    """Quadratic smile in log-moneyness around the current spot.

    ``sigma(K) = atm - a*m + b*m**2`` with ``m = ln(K / S)``, the strategy
    page's skew model (a=0.6, b=1.0). ``a > 0`` makes downside strikes richer.
    The smile is anchored to today's spot ("sticky moneyness"); the ATM level
    comes from the VIX each day.
    """

    def __init__(self, atm: float, a: float = 0.6, b: float = 1.0):
        self.atm, self.a, self.b = atm, a, b

    @classmethod
    def from_vix(cls, vix: float, atm_ratio: float = 0.9, atm_intercept: float = 0.0,
                 skew_a: float = 0.6, skew_a_vix: float = 0.0,
                 skew_b: float = 1.0, skew_b_vix: float = 0.0) -> "Surface":
        """Surface for a VIX level (in points). With the ``*_vix`` slopes at zero this is
        the strategy page's model; the calibrated preset makes level and skew depend on the VIX.
        Skew slope is clipped to [0, 2.5] and curvature to [0, 6] to stay inside the fitted range."""
        atm = max(MIN_VOL, (atm_ratio * vix + atm_intercept) / 100.0)
        a = min(2.5, max(0.0, skew_a + skew_a_vix * vix))
        b = min(6.0, max(0.0, skew_b + skew_b_vix * vix))
        return cls(atm, a, b)

    def vol(self, S: float, K: float) -> float:
        m = math.log(K / S)
        return min(MAX_VOL, max(MIN_VOL, self.atm - self.a * m + self.b * m * m))


def strike_for_delta(S: float, T: float, surface: Surface, target: float, call: bool,
                     r: float = 0.0, q: float = 0.0, tick: float | None = 5.0) -> float:
    """Strike whose delta, priced at its own smile vol, equals ``target``.

    ``target`` is signed: +0.15 for a 15-delta call, -0.15 for a 15-delta put.
    The result is rounded to the nearest listed strike (``tick`` points).
    """
    if call and not 0 < target < 1:
        raise ValueError("call delta must be in (0, 1)")
    if not call and not -1 < target < 0:
        raise ValueError("put delta must be in (-1, 0)")

    def err(k):
        return delta(S, k, T, surface.vol(S, k), call, r, q) - target

    lo, hi = S * 0.2, S * 5.0
    k = brentq(err, lo, hi, xtol=1e-6 * S)
    return round(k / tick) * tick if tick else k
