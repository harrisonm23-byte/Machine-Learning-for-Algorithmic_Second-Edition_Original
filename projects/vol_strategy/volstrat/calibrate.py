"""Calibrate the model's volatility surface against real option prices.

The backtest prices options from the VIX with two assumptions: the 30-day ATM
implied vol is ``atm_ratio x VIX`` and the smile is
``sigma(K) = atm - a*m + b*m**2`` with ``m = ln(K/F)``. This module measures
both from real SPY option trades (Alpaca historical options data, available
from February 2024):

1. For each trading day, pick the SPY expiry closest to 30 calendar days out.
2. Download that day's bar for every out-of-the-money strike from 80% to 110%
   of spot (puts below the forward, calls above).
3. Estimate the forward from put-call parity on near-the-money pairs, then
   invert Black-76 on each option's volume-weighted price to get its implied vol.
4. Fit the quadratic smile and compare its ATM level with the day's average VIX.

SPY tracks the S&P 500 and its options trade far more than single SPX strikes,
so its implied vols are a good proxy for SPX's. Prices are daily VWAPs matched
with SPY's daily VWAP and the VIX's average of open/high/low/close, so each
day's numbers are noisy; the calibration uses medians over hundreds of days.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

log = logging.getLogger(__name__)

ALPACA_DATA = "https://data.alpaca.markets"
MIN_VOLUME, MIN_TRADES = 10, 3


def occ_symbol(root: str, expiry: pd.Timestamp, call: bool, strike: float) -> str:
    return f"{root}{expiry:%y%m%d}{'C' if call else 'P'}{int(round(strike * 1000)):08d}"


def black76(F: float, K: float, T: float, sigma: float, call: bool, df: float = 1.0) -> float:
    if T <= 0 or sigma <= 0:
        return df * (max(F - K, 0.0) if call else max(K - F, 0.0))
    s = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * s * s) / s
    d2 = d1 - s
    if call:
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def implied_vol(price: float, F: float, K: float, T: float, call: bool, df: float = 1.0) -> float:
    intrinsic = df * (max(F - K, 0.0) if call else max(K - F, 0.0))
    if not np.isfinite(price) or price <= intrinsic + 1e-6 or T <= 0:
        return np.nan
    try:
        return brentq(lambda v: black76(F, K, T, v, call, df) - price, 1e-4, 5.0, xtol=1e-7)
    except ValueError:
        return np.nan


class AlpacaClient:
    def __init__(self, key_id: str | None = None, secret: str | None = None, cache_dir: str | Path | None = None):
        import requests

        self.s = requests.Session()
        self.s.headers.update({
            "APCA-API-KEY-ID": key_id or os.environ.get("APCA_API_KEY_ID", "proxy-injected"),
            "APCA-API-SECRET-KEY": secret or os.environ.get("APCA_API_SECRET_KEY", "proxy-injected"),
        })
        self.cache = Path(cache_dir) if cache_dir else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)

    def _get(self, path: str, params: dict) -> dict:
        for attempt in range(6):
            r = self.s.get(ALPACA_DATA + path, params=params, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        r.raise_for_status()
        return r.json()

    def bars(self, path: str, symbols: list[str], start: str, end: str, **extra) -> dict[str, list]:
        out: dict[str, list] = {}
        for i in range(0, len(symbols), 100):
            params = {"symbols": ",".join(symbols[i:i + 100]), "timeframe": "1Day",
                      "start": start, "end": end, "limit": 10000, **extra}
            while True:
                payload = self._get(path, params)
                for sym, rows in (payload.get("bars") or {}).items():
                    out.setdefault(sym, []).extend(rows)
                token = payload.get("next_page_token")
                if not token:
                    break
                params["page_token"] = token
        return out

    def option_day(self, symbols: list[str], day: pd.Timestamp) -> dict[str, dict]:
        """One day's bar for each option symbol that traded (cached per day and expiry)."""
        key = None
        if self.cache:
            key = self.cache / f"opt_{day:%Y%m%d}_{symbols[0][:9]}.json"
            if key.exists():
                return json.loads(key.read_text())
        d = day.strftime("%Y-%m-%d")
        raw = self.bars("/v1beta1/options/bars", symbols, d, d)
        out = {k: v[0] for k, v in raw.items() if v}
        if key:
            key.write_text(json.dumps(out))
        return out

    def stock_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        raw = self.bars("/v2/stocks/bars", [symbol], start, end, adjustment="raw", feed="sip")
        df = pd.DataFrame(raw.get(symbol, []))
        df["date"] = pd.to_datetime(df["t"]).dt.tz_localize(None).dt.normalize()
        return df.set_index("date").rename(columns={"o": "open", "h": "high", "l": "low", "c": "close",
                                                   "v": "volume", "vw": "vwap"})


def _parity_forward(calls: dict, puts: dict, S: float) -> float | None:
    """Forward from C - P = DF*(F - K) on near-the-money strikes traded on both sides."""
    ks = sorted(k for k in set(calls) & set(puts) if abs(k / S - 1) < 0.02)
    if len(ks) < 3:
        return None
    k = np.array(ks)
    y = np.array([calls[x] - puts[x] for x in ks])
    slope, intercept = np.polyfit(k, y, 1)       # y = DF*F - DF*K
    df = -slope
    if not 0.9 < df < 1.01:
        return None
    return intercept / df


def calibrate_day(client: AlpacaClient, day: pd.Timestamp, S: float, root: str = "SPY",
                  target_days: int = 30, calendar: pd.DatetimeIndex | None = None) -> dict | None:
    """Measure the smile for one day. ``S`` is SPY's VWAP that day."""
    # nearest listed weekday expiry to day + target_days (SPY lists daily expiries)
    target = day + pd.Timedelta(days=target_days)
    candidates = sorted({target + pd.Timedelta(days=k) for k in range(-3, 4)
                         if (target + pd.Timedelta(days=k)).weekday() < 5},
                        key=lambda e: abs((e - target).days))
    strikes = np.arange(math.floor(S * 0.80), math.ceil(S * 1.10) + 1, 1.0)
    for expiry in candidates:
        if calendar is not None and expiry <= calendar[-1] and expiry not in calendar:
            continue                      # market holiday, no contract
        syms = [occ_symbol(root, expiry, False, k) for k in strikes if k <= S * 1.02] + \
               [occ_symbol(root, expiry, True, k) for k in strikes if k >= S * 0.98]
        bars = client.option_day(syms, day)
        if len(bars) < 20:
            continue
        calls, puts = {}, {}
        for sym, b in bars.items():
            if b.get("v", 0) < MIN_VOLUME or b.get("n", 0) < MIN_TRADES:
                continue
            k = int(sym[-8:]) / 1000.0
            (calls if sym[-9] == "C" else puts)[k] = float(b["vw"])
        T = ((expiry - day).days + 0.0) / 365.0   # to the close of expiry day, calendar convention
        F = _parity_forward(calls, puts, S) or S * math.exp((0.045 - 0.013) * T)
        pts = []
        for k, px in puts.items():
            if k < F:
                pts.append((k, False, implied_vol(px, F, k, T, False)))
        for k, px in calls.items():
            if k > F:
                pts.append((k, True, implied_vol(px, F, k, T, True)))
        pts = [(k, c, v) for k, c, v in pts if np.isfinite(v) and 0.03 < v < 2.0]
        if len(pts) < 10:
            continue
        m = np.array([math.log(k / F) for k, _, _ in pts])
        iv = np.array([v for _, _, v in pts])
        keep = (m > -0.20) & (m < 0.08)
        if keep.sum() < 8:
            continue
        c2, c1, c0 = np.polyfit(m[keep], iv[keep], 2)

        def iv_at_delta(target_delta, call):
            best = None
            for k, c, v in pts:
                if c != call:
                    continue
                d1 = (math.log(F / k) + 0.5 * v * v * T) / (v * math.sqrt(T))
                dl = norm.cdf(d1) if call else -norm.cdf(-d1)
                if best is None or abs(dl - target_delta) < abs(best[2] - target_delta):
                    best = (k, v, dl)
            return best

        out = dict(date=day, expiry=expiry, days=(expiry - day).days, spot=S, forward=F,
                   n_points=int(keep.sum()), atm_iv=float(c0), skew_a=float(-c1), skew_b=float(c2))
        for name, tgt, call in [("p15", -0.15, False), ("c15", 0.15, True), ("p05", -0.05, False), ("c05", 0.05, True)]:
            b = iv_at_delta(tgt, call)
            if b and abs(b[2] - tgt) < 0.03:
                out.update({f"k_{name}": b[0], f"iv_{name}": b[1], f"delta_{name}": b[2]})
        return out
    return None


def run_calibration(index_data: pd.DataFrame, start: str = "2024-02-01", end: str | None = None,
                    cache_dir: str | Path | None = "data_cache/alpaca", step: int = 1,
                    target_days: int = 30) -> pd.DataFrame:
    """Calibrate every ``step``-th trading day between ``start`` and ``end``, using the
    expiry closest to ``target_days`` calendar days out."""
    client = AlpacaClient(cache_dir=cache_dir)
    end = end or index_data.index[-1].strftime("%Y-%m-%d")
    spy = client.stock_bars("SPY", start, end)
    days = index_data.loc[start:end].index
    rows = []
    for i, day in enumerate(days[::step]):
        if day not in spy.index:
            continue
        r = calibrate_day(client, day, float(spy.loc[day, "vwap"]), target_days=target_days,
                          calendar=index_data.index)
        if r is None:
            log.warning("no calibration for %s", day.date())
            continue
        v = index_data.loc[day]
        r["vix_close"] = float(v["vix"])
        r["vix_avg"] = float((v["vix_open"] + v["vix_high"] + v["vix_low"] + v["vix"]) / 4)
        rows.append(r)
        if i % 50 == 0:
            log.info("calibrated %s (%d days)", day.date(), len(rows))
    cal = pd.DataFrame(rows).set_index("date")
    cal["atm_ratio"] = cal["atm_iv"] * 100 / cal["vix_avg"]
    return cal


def fit_surface_params(cal: pd.DataFrame) -> dict:
    """Linear-in-VIX surface parameters fitted across calibration days:

    atm (vol points) = atm_ratio * VIX + atm_intercept
    a = skew_a + skew_a_vix * VIX,   b = skew_b + skew_b_vix * VIX

    The keys match ``BacktestConfig`` fields, so the result can be passed straight in.
    """
    vix = cal["vix_avg"].to_numpy()

    def ols(y):
        slope, intercept = np.polyfit(vix, y, 1)
        resid = y - (slope * vix + intercept)
        r2 = 1 - resid.var() / y.var()
        return float(slope), float(intercept), float(r2)

    ar, ai, ar2 = ols(cal["atm_iv"].to_numpy() * 100)
    sa, ia, ra2 = ols(cal["skew_a"].to_numpy())
    sb, ib, rb2 = ols(cal["skew_b"].to_numpy())
    return {"atm_ratio": ar, "atm_intercept": ai, "skew_a": ia, "skew_a_vix": sa,
            "skew_b": ib, "skew_b_vix": sb, "_r2": {"atm": ar2, "skew_a": ra2, "skew_b": rb2}}


POINTS = ["p15", "c15", "p05", "c05"]
WING_Z = (-5.0, 2.0)     # flat smile beyond these z-scores, roughly the strike range fitted on
SURFACE_KEYS = ["atm_ratio", "atm_intercept", "skew_a", "skew_a_vix", "skew_b", "skew_b_vix",
                "term_atm", "term_atm_vix", "term_a", "term_b"]


def _observations(cals: dict[int, pd.DataFrame], vix_col: str = "vix_close") -> pd.DataFrame:
    """One row per (day, tenor, strike) with the real IV at the ATM, 15- and 5-delta points."""
    rows = []
    for tenor, cal in cals.items():
        for day, r in cal.iterrows():
            T = r["days"] / 365.0
            rows.append((day, tenor, r[vix_col], r["spot"], r["forward"], T, r["atm_iv"], "atm"))
            for p in POINTS:
                if pd.notna(r.get(f"k_{p}")) and pd.notna(r.get(f"iv_{p}")):
                    rows.append((day, tenor, r[vix_col], r["spot"], r[f"k_{p}"], T, r[f"iv_{p}"], p))
    return pd.DataFrame(rows, columns=["date", "tenor", "vix", "spot", "strike", "T", "iv", "point"])


def fit_surface_global(cals: dict[int, pd.DataFrame], vix_col: str = "vix_close",
                       fit_term: bool | None = None) -> dict:
    """Least-squares fit of the backtest's own surface formula (bs.Surface.from_vix,
    with flat wings) to real implied vols at the strikes the strategies trade.

    ``cals`` maps tenor (days) to a calibration table from ``run_calibration``.
    Term-structure parameters are fitted only when more than one tenor is given.
    The model is evaluated with m = ln(K / spot), the convention the backtest uses.
    """
    from scipy.optimize import least_squares

    from .bs import Surface

    obs = _observations(cals, vix_col).dropna()
    fit_term = len(cals) > 1 if fit_term is None else fit_term
    keys = SURFACE_KEYS if fit_term else SURFACE_KEYS[:6]
    x0 = np.array([0.85, 0.0, 0.6, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0][:len(keys)])
    V, Sp, K, T, iv = (obs[c].to_numpy(float) for c in ["vix", "spot", "strike", "T", "iv"])

    def resid(x):
        p = dict(zip(keys, x))
        out = np.empty(len(obs))
        for i in range(len(obs)):
            out[i] = Surface.from_vix(V[i], wing_z=WING_Z, **p).vol(Sp[i], K[i], T[i]) - iv[i]
        return out * 100     # vol points

    sol = least_squares(resid, x0, loss="soft_l1", f_scale=1.0)
    params = dict(zip(keys, map(float, sol.x)))
    for k in SURFACE_KEYS:
        params.setdefault(k, 0.0)
    r = resid(sol.x)
    params["wing_z"] = WING_Z
    params["_fit"] = {"n_obs": int(len(obs)), "rmse_vol_pts": float(np.sqrt(np.mean(r ** 2))),
                      "median_abs_err_vol_pts": float(np.median(np.abs(r))), "tenors": sorted(cals)}
    return params


def surface_errors(cals: dict[int, pd.DataFrame], params: dict, vix_col: str = "vix_close") -> pd.DataFrame:
    """Median model-minus-real IV (vol points) by tenor, VIX bucket and strike point."""
    from .bs import Surface

    obs = _observations(cals, vix_col).dropna()
    p = {k: v for k, v in params.items() if not k.startswith("_")}
    obs["error"] = [(Surface.from_vix(v, **p).vol(s, k, t) - y) * 100
                    for v, s, k, t, y in obs[["vix", "spot", "strike", "T", "iv"]].itertuples(index=False)]
    obs["bucket"] = pd.cut(obs["vix"], [0, 12, 15, 18, 22, 30, 200], right=False).astype(str)
    return obs.pivot_table(index=["tenor", "bucket"], columns="point", values="error",
                           aggfunc="median", observed=True)


def validate_surface(cal: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Model IV minus real IV (vol points) at the real 15- and 5-delta strikes, by VIX bucket."""
    from .bs import Surface

    rows = []
    for day, r in cal.iterrows():
        surf = Surface.from_vix(r["vix_avg"], **{k: v for k, v in params.items() if not k.startswith("_")})
        for name in ["p15", "c15", "p05", "c05"]:
            k, iv = r.get(f"k_{name}"), r.get(f"iv_{name}")
            if pd.notna(k) and pd.notna(iv):
                rows.append(dict(date=day, vix=r["vix_avg"], point=name,
                                 error=(surf.vol(r["forward"], k) - iv) * 100))
    err = pd.DataFrame(rows)
    err["bucket"] = pd.cut(err["vix"], [0, 12, 15, 18, 22, 30, 200], right=False).astype(str)
    return err.pivot_table(index="bucket", columns="point", values="error", aggfunc="median", observed=True)


def check_against_spxw(trades: pd.DataFrame, index_data: pd.DataFrame, surfaces: dict,
                       cache_dir: str | Path | None = "data_cache/alpaca_spx", min_volume: int = 5) -> pd.DataFrame:
    """Compare model prices with real SPX weekly-option (SPXW) trades at the exact strikes
    and expiries the backtest traded on its entry days (Feb 2024 onwards).

    The real price is the day's VWAP, so the model is evaluated at day-average conditions:
    SPX scaled by SPY's VWAP/close ratio and the VIX's average of open/high/low/close.
    ``surfaces`` maps a name to a ``BacktestConfig`` whose ``surface(vix)`` is used.
    Returns one row per traded leg with real and model prices and implied-vol errors.
    """
    from .bs import price as bs_price

    client = AlpacaClient(cache_dir=cache_dir)
    t = trades[trades["entry_date"] >= "2024-02-01"].drop_duplicates(["entry_date", "strikes"])
    if t.empty:
        return pd.DataFrame()
    spy = client.stock_bars("SPY", t["entry_date"].min().strftime("%Y-%m-%d"),
                            t["entry_date"].max().strftime("%Y-%m-%d"))
    rows = []
    for _, r in t.iterrows():
        day = r["entry_date"]
        if day not in spy.index:
            continue
        legs = [(s[0] == "C", float(s[1:].split("x")[0]), int(s.split("x")[1])) for s in r["strikes"].split(";")]
        syms = [occ_symbol("SPXW", r["expiry"], c, k) for c, k, _ in legs]
        bars = client.option_day(syms, day)
        v = index_data.loc[day]
        s_avg = v["spx"] * spy.loc[day, "vwap"] / spy.loc[day, "close"]
        vix_avg = (v["vix_open"] + v["vix_high"] + v["vix_low"] + v["vix"]) / 4
        T = (r["expiry"] - day).days / 365.0
        F = s_avg * math.exp((0.045 - 0.013) * T)
        for (call, k, q), sym in zip(legs, syms):
            b = bars.get(sym)
            if not b or b.get("v", 0) < min_volume:
                continue
            row = dict(date=day, leg=("C" if call else "P") + ("_wing" if q > 0 and len(legs) == 4 else ""),
                       strike=k, spx_avg=s_avg, vix_avg=vix_avg, days=(r["expiry"] - day).days,
                       real_vwap=float(b["vw"]), volume=int(b["v"]),
                       iv_real=implied_vol(float(b["vw"]), F, k, T, call))
            for name, cfg in surfaces.items():
                vol = cfg.surface(vix_avg).vol(s_avg, k, T)
                row[f"model_{name}"] = bs_price(s_avg, k, T, vol, call)
                row[f"iv_err_{name}"] = (vol - row["iv_real"]) * 100
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_calibration(cal: pd.DataFrame) -> dict:
    """Robust (median) parameters plus how they vary with the VIX level."""
    out = {
        "days": int(len(cal)), "start": str(cal.index.min().date()), "end": str(cal.index.max().date()),
        "atm_ratio_median": float(cal["atm_ratio"].median()),
        "atm_ratio_iqr": [float(cal["atm_ratio"].quantile(0.25)), float(cal["atm_ratio"].quantile(0.75))],
        "skew_a_median": float(cal["skew_a"].median()),
        "skew_b_median": float(cal["skew_b"].median()),
    }
    b = pd.cut(cal["vix_avg"], [0, 12, 15, 18, 22, 30, 200], right=False)
    cols = [c for c in ["atm_ratio", "skew_a", "skew_b", "iv_p15", "iv_c15", "iv_p05", "iv_c05", "atm_iv"] if c in cal]
    out["by_vix_bucket"] = cal.groupby(b, observed=True)[cols].median().round(4).assign(
        days=cal.groupby(b, observed=True).size()).reset_index().astype({"vix_avg": str}).to_dict("records")
    return out
