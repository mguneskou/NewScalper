from __future__ import annotations

import pandas as pd

from scalper.strategies.base import Strategy


class EmaCrossStrategy(Strategy):
    """Always-in-market trend following: long while the fast EMA is above the
    slow EMA, short while it's below."""

    name = "ema_cross"
    # Non-overlapping ranges so every combo has fast_period < slow_period: an equal
    # pair would produce an always-flat (zero-trade, zero-P&L) strategy that a
    # pnl-maximizing optimizer could wrongly pick as "best" over any real, lossy combo.
    param_grid = {
        "fast_period": [5, 8, 12, 16],
        "slow_period": [24, 32, 50, 100],
    }

    def signals(self, df: pd.DataFrame, params: dict) -> pd.Series:
        fast_period = params["fast_period"]
        slow_period = params["slow_period"]
        if fast_period >= slow_period:
            return pd.Series(0, index=df.index)

        price = self.mid_close(df)
        fast_ema = price.ewm(span=fast_period, adjust=False).mean()
        slow_ema = price.ewm(span=slow_period, adjust=False).mean()

        signal = pd.Series(0, index=df.index)
        signal[fast_ema > slow_ema] = 1
        signal[fast_ema < slow_ema] = -1
        # warm-up period before the slow EMA has enough data is unreliable -> stay flat
        signal.iloc[: slow_period - 1] = 0
        return signal
