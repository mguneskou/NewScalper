from __future__ import annotations

import numpy as np
from numba import njit
import pandas as pd

from scalper.strategies.base import Strategy


@njit(cache=True)
def _breakout_positions(price: np.ndarray, upper: np.ndarray, lower: np.ndarray, middle: np.ndarray) -> np.ndarray:
    n = price.shape[0]
    out = np.empty(n, dtype=np.int64)
    position = 0
    for i in range(n):
        if np.isnan(upper[i]) or np.isnan(lower[i]):
            out[i] = 0
            continue
        if position == 0:
            if price[i] > upper[i]:
                position = 1
            elif price[i] < lower[i]:
                position = -1
        elif position == 1 and price[i] <= middle[i]:
            position = 0
        elif position == -1 and price[i] >= middle[i]:
            position = 0
        out[i] = position
    return out


class BollingerBreakoutStrategy(Strategy):
    """Breakout continuation: go long when price closes above the upper Bollinger
    Band (expecting momentum to continue), short below the lower band, flat once
    price reverts back through the middle band (the moving average)."""

    name = "bollinger_breakout"
    param_grid = {
        "period": [10, 20, 30],
        "num_std": [1.5, 2.0, 2.5],
    }

    def signals(self, df: pd.DataFrame, params: dict) -> pd.Series:
        period = params["period"]
        num_std = params["num_std"]

        price = self.mid_close(df)
        middle = price.rolling(period).mean()
        std = price.rolling(period).std()
        upper = middle + num_std * std
        lower = middle - num_std * std

        sig_vals = _breakout_positions(
            price.to_numpy(dtype=np.float64),
            upper.to_numpy(dtype=np.float64),
            lower.to_numpy(dtype=np.float64),
            middle.to_numpy(dtype=np.float64),
        )
        return pd.Series(sig_vals, index=df.index)
