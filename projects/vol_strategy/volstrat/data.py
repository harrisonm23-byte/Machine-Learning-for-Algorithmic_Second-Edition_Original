"""Daily S&P 500 and VIX closes from Financial Modeling Prep.

The S&P 500 trading calendar is used: the VIX also prints on a few days the
equity market is closed (e.g. Juneteenth), and those rows are dropped.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

FMP_EOD_URL = "https://financialmodelingprep.com/stable/historical-price-eod/full"


def _fetch(symbol: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    import requests

    resp = requests.get(FMP_EOD_URL, params={"symbol": symbol, "from": start, "to": end},
                        headers={"apikey": api_key}, timeout=60)
    if resp.status_code in (401, 402, 403):
        raise PermissionError(f"FMP {resp.status_code} for {symbol}: {resp.text[:200]}")
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        raise ValueError(f"FMP returned no rows for {symbol}")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()[["open", "high", "low", "close"]].astype(float)


def load_index_data(start: str = "2009-06-01", end: str | None = None,
                    cache_dir: str | Path | None = "data_cache", refresh: bool = False,
                    api_key: str | None = None) -> pd.DataFrame:
    """Return a DataFrame indexed by date with columns ``spx`` and ``vix`` (closes),
    plus ``spx_open/high/low`` and ``vix_open/high/low``."""
    end = end or (pd.Timestamp.today().normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    cache = None
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cache = Path(cache_dir) / f"spx_vix_{start}_{end}.parquet"
        if cache.exists() and not refresh:
            return pd.read_parquet(cache)

    api_key = api_key or os.environ.get("FMP_API_KEY")
    if not api_key:
        raise RuntimeError("Set FMP_API_KEY to download index data.")
    spx = _fetch("^GSPC", start, end, api_key).add_prefix("spx_")
    vix = _fetch("^VIX", start, end, api_key).add_prefix("vix_")
    df = spx.join(vix, how="left").rename(columns={"spx_close": "spx", "vix_close": "vix"})
    missing = df["vix"].isna().sum()
    if missing:
        log.warning("Forward-filling %d missing VIX closes", missing)
        df = df.ffill()
    df = df.dropna()
    if cache is not None:
        df.to_parquet(cache)
    return df
