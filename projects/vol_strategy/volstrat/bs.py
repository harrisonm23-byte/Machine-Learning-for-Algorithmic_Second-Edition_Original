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
T_REF = 30 / 365   # tenor the VIX and the 30-day calibration refer to
T_MIN = 2 / 365    # shortest tenor used when scaling the smile, so it stays finite near expiry


class Surface:
    """Quadratic smile in log-moneyness around the current spot.

    ``sigma(K, T) = atm(T) - a(T)*m + b(T)*m**2`` with ``m = ln(K / S)``.
    At the 30-day reference tenor this is the strategy page's skew model
    (a=0.6, b=1.0 by default). ``a > 0`` makes downside strikes richer.
    The smile is anchored to today's spot ("sticky moneyness").

    Tenor dependence (all zero by default, i.e. the same smile at every tenor):
    with ``x = ln(T_REF / max(T, T_MIN))``, which is positive for shorter options,
    ``atm(T) = atm * exp(term_atm * x)``, ``a(T) = a * exp(term_a * x)`` and
    ``b(T) = b * exp(term_b * x)``. Real SPX smiles steepen as expiry nears,
    which matters when an option is marked part-way through its life.

    ``wing_z = (z_lo, z_hi)`` holds the smile flat beyond that many standard
    deviations from the money (``z = m / (atm(T) * sqrt(T))``), i.e. outside the
    range the quadratic was fitted on. Without it (the page's model) the
    quadratic is used everywhere, which for realistic curvature makes far-wing
    vols explode.
    """

    def __init__(self, atm: float, a: float = 0.6, b: float = 1.0,
                 term_atm: float = 0.0, term_a: float = 0.0, term_b: float = 0.0,
                 wing_z: tuple[float, float] | None = None):
        self.atm, self.a, self.b = atm, a, b
        self.term_atm, self.term_a, self.term_b = term_atm, term_a, term_b
        self.wing_z = wing_z

    @classmethod
    def from_vix(cls, vix: float, atm_ratio: float = 0.9, atm_intercept: float = 0.0,
                 skew_a: float = 0.6, skew_a_vix: float = 0.0,
                 skew_b: float = 1.0, skew_b_vix: float = 0.0,
                 term_atm: float = 0.0, term_atm_vix: float = 0.0,
                 term_a: float = 0.0, term_b: float = 0.0,
                 wing_z: tuple[float, float] | None = None) -> "Surface":
        """Surface for a VIX level (in points).

        ATM (vol points) = atm_ratio * VIX + atm_intercept; a = skew_a + skew_a_vix * VIX;
        b = skew_b + skew_b_vix * VIX; ATM term exponent = term_atm + term_atm_vix * VIX.
        With every ``*_vix``/``term_*`` value at zero this is the strategy page's model.
        Skew slope is clipped to [0, 2.5] and curvature to [0, 6] to stay inside the fitted range.
        """
        atm = max(MIN_VOL, (atm_ratio * vix + atm_intercept) / 100.0)
        a = min(2.5, max(0.0, skew_a + skew_a_vix * vix))
        b = min(6.0, max(0.0, skew_b + skew_b_vix * vix))
        return cls(atm, a, b, term_atm=term_atm + term_atm_vix * vix, term_a=term_a, term_b=term_b,
                   wing_z=wing_z)

    def vol(self, S: float, K: float, T: float | None = None) -> float:
        m = math.log(K / S)
        atm, a, b = self.atm, self.a, self.b
        if T is not None and (self.term_atm or self.term_a or self.term_b):
            x = math.log(T_REF / max(T, T_MIN))
            atm *= math.exp(self.term_atm * x)
            a *= math.exp(self.term_a * x)
            b *= math.exp(self.term_b * x)
        if self.wing_z is not None:
            width = max(atm, MIN_VOL) * math.sqrt(max(T if T is not None else T_REF, T_MIN))
            m = min(max(m, self.wing_z[0] * width), self.wing_z[1] * width)
        return min(MAX_VOL, max(MIN_VOL, atm - a * m + b * m * m))


def strike_for_delta(S: float, T: float, surface: Surface, target: float, call: bool,
                     r: float = 0.0, q: float = 0.0, tick: float | None = 5.0) -> float:
    """Strike whose delta, priced at its own smile vol, equals ``target``.

    ``target`` is signed: +0.15 for a 15-delta call, -0.15 for a 15-delta put.
    The result is rounded to the nearest listed strike (``tick`` points).

    The search walks outward from the money in 0.25% steps and returns the
    first crossing. With a strongly curved smile, delta is not monotone far
    out in the wings, so a wide fixed bracket could land on a nonsense root
    (e.g. a "15-delta call" at 3.5x spot) or fail outright.
    """
    if call and not 0 < target < 1:
        raise ValueError("call delta must be in (0, 1)")
    if not call and not -1 < target < 0:
        raise ValueError("put delta must be in (-1, 0)")

    def err(k):
        return delta(S, k, T, surface.vol(S, k, T), call, r, q) - target

    step = 0.0025 * S * (1 if call else -1)
    k0, e0 = S, err(S)
    while True:
        k1 = k0 + step
        if not 0.05 * S < k1 < 10 * S:
            raise ValueError(f"no strike with delta {target} found")
        e1 = err(k1)
        if e0 == 0:
            k = k0
            break
        if e0 * e1 <= 0:
            k = brentq(err, min(k0, k1), max(k0, k1), xtol=1e-6 * S)
            break
        k0, e0 = k1, e1
    return round(k / tick) * tick if tick else k
