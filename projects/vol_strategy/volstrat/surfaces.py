"""Volatility-surface presets for the backtest.

PAGE        the strategy page's model: ATM = 0.9 x VIX, skew a=0.6, b=1.0, same at every tenor.
CALIBRATED  fitted to real SPY option prices (Alpaca, Feb 2024 onwards) by
            ``python -m volstrat --calibrate``; parameters live in
            ``calibration/surface_params.json`` next to the data they came from.
"""
from __future__ import annotations

import json
from pathlib import Path

CALIBRATION_DIR = Path(__file__).resolve().parents[1] / "calibration"
PARAMS_FILE = CALIBRATION_DIR / "surface_params.json"

PAGE = dict(surface_name="page", atm_ratio=0.9, skew_a=0.6, skew_b=1.0)


def calibrated(path: str | Path = PARAMS_FILE, atm_shift: float = 0.0) -> dict:
    """BacktestConfig keyword arguments for the calibrated surface.

    ``atm_shift`` moves the whole surface up or down by that many vol points,
    for testing how sensitive results are to calibration error.
    """
    params = json.loads(Path(path).read_text())
    out = {k: v for k, v in params.items() if not k.startswith("_")}
    out["wing_z"] = tuple(out["wing_z"])
    out["atm_intercept"] = out.get("atm_intercept", 0.0) + atm_shift
    name = "calibrated" if atm_shift == 0 else f"calibrated{atm_shift:+g}"
    return dict(surface_name=name, **out)
