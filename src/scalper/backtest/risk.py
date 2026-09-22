"""ATR-relative stop/take-profit distances.

A fixed pip stop is the same width whether the market is dead quiet or
violently volatile. ATR (Average True Range) tracks recent bar-to-bar range,
so multiplying it by a factor gives a stop that's tight in quiet conditions
and wide in volatile ones -- sized to what the market is actually doing at
entry, rather than one static number searched over a fixed grid.
"""

from __future__ import annotations

import pandas as pd

from scalper.backtest.costs import pip_size


def atr_pips(df: pd.DataFrame, period: int, instrument: str) -> pd.Series:
    """Wilder-smoothed Average True Range of `df`'s mid price, in pips.

    Uses mid bid/ask (consistent with how strategies compute their own
    signals) rather than a single side, since the bid-ask spread itself isn't
    part of what a volatility-based stop should be sized to.
    """
    mid_h = (df["bid_h"] + df["ask_h"]) / 2.0
    mid_l = (df["bid_l"] + df["ask_l"]) / 2.0
    mid_c = (df["bid_c"] + df["ask_c"]) / 2.0
    prev_close = mid_c.shift(1)

    true_range = pd.concat(
        [mid_h - mid_l, (mid_h - prev_close).abs(), (mid_l - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    # Warm-up (fewer than `period` bars of history) has no real ATR yet --
    # back-fill from the first bar that has one rather than using 0, which
    # would otherwise size a same-bar stop/take-profit at zero distance.
    return (atr / pip_size(instrument)).bfill().fillna(0.0)
