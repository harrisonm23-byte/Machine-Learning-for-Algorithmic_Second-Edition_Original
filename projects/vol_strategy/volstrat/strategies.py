"""The six positions from the strategy page.

Each leg is (option type, strike delta, quantity). Quantity is in contracts
per "unit" of the strategy (negative = sold). Strike deltas are the signed
Black-Scholes deltas used to pick strikes at entry: -0.15 is a 15-delta put.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LegSpec:
    call: bool
    strike_delta: float
    qty: int


@dataclass(frozen=True)
class StrategySpec:
    key: str
    name: str
    legs: tuple[LegSpec, ...]

    @property
    def is_short_vol(self) -> bool:
        """Sells the 15-delta (core) strikes. Bought 5-delta wings, as in the iron condor,
        cap the tail but leave the position a net premium seller. The backtest decides the
        side of each trade from the sign of the premium actually collected."""
        return sum(leg.qty for leg in self.legs if abs(leg.strike_delta) >= 0.15) < 0


P15, C15 = -0.15, 0.15
P05, C05 = -0.05, 0.05

STRATEGIES: dict[str, StrategySpec] = {s.key: s for s in [
    StrategySpec("book", "Short (put + strangle)",
                 (LegSpec(False, P15, -2), LegSpec(True, C15, -1))),
    StrategySpec("strangle", "Short strangle",
                 (LegSpec(False, P15, -1), LegSpec(True, C15, -1))),
    StrategySpec("condor", "Iron condor",
                 (LegSpec(False, P15, -1), LegSpec(True, C15, -1),
                  LegSpec(False, P05, +1), LegSpec(True, C05, +1))),
    StrategySpec("longstr", "Long strangle",
                 (LegSpec(False, P15, +1), LegSpec(True, C15, +1))),
    StrategySpec("longput", "Long put", (LegSpec(False, P15, +1),)),
    StrategySpec("longcall", "Long call", (LegSpec(True, C15, +1),)),
]}
