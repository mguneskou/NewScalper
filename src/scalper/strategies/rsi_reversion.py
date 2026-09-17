from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from scalper.strategies.base import Strategy


@njit(cache=True)
def _rsi_positions(rsi_vals: np.ndarray, oversold: float, overbought: float) -> np.ndarray:
    n = rsi_vals.shape[0]
    out = np.empty(n, dtype=np.int64)
    position = 0
    for i in range(n):
        r = rsi_vals[i]
        if position == 0:
            if r <= oversold:
                position = 1
            elif r >= overbought:
                position = -1
        elif position == 1 and r >= 50:
            position = 0
        elif position == -1 and r <= 50:
            position = 0
        out[i] = position
    return out


def _rsi(price: pd.Series, period: int) -> pd.Series:
    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    # avg_loss == 0: pure uptrend over the window -> maximally overbought (100),
    # unless avg_gain is also 0 (flat price), which is truly neutral (50).
    rsi = rsi.where(avg_loss != 0, other=np.where(avg_gain > 0, 100.0, 50.0))
    return rsi.fillna(50)  # warm-up period (fewer than `period` bars of history) -> neutral


class RsiReversionStrategy(Strategy):
    """Mean reversion: go long when RSI dips below the oversold threshold (expecting
    a bounce), short when it rises above overbought, flat once RSI returns to the
    neutral midpoint."""

    name = "rsi_reversion"
    param_grid = {
        "period": [7, 14, 21],
        "oversold": [20, 25, 30],
        "overbought": [70, 75, 80],
    }

    def signals(self, df: pd.DataFrame, params: dict) -> pd.Series:
        period = params["period"]
        oversold = params["oversold"]
        overbought = params["overbought"]

        price = self.mid_close(df)
        rsi = _rsi(price, period)
        sig_vals = _rsi_positions(rsi.to_numpy(dtype=np.float64), float(oversold), float(overbought))
        return pd.Series(sig_vals, index=df.index)
