"""Trading cost model: real bid/ask spread from the candle itself, plus configurable
slippage and commission. Entries/exits always fill on the correct side of the spread
(buy at ask, sell at bid) -- never at the more favourable mid price.
"""

from __future__ import annotations

from dataclasses import dataclass


def pip_size(instrument: str) -> float:
    """0.01 for JPY-quoted pairs, 0.0001 otherwise."""
    return 0.01 if instrument.endswith("_JPY") else 0.0001


@dataclass(frozen=True)
class CostModel:
    slippage_pips: float = 0.2
    commission_per_trade: float = 0.0

    def buy_fill_price(self, ask_price: float, instrument: str) -> float:
        return ask_price + self.slippage_pips * pip_size(instrument)

    def sell_fill_price(self, bid_price: float, instrument: str) -> float:
        return bid_price - self.slippage_pips * pip_size(instrument)
