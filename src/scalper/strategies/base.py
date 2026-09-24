"""Strategy interface: a strategy turns a candle DataFrame + parameters into a
desired-position signal per bar. The backtest engine handles all execution
timing (shift-by-one, fills, costs) -- strategies only decide direction."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Strategy(ABC):
    name: str
    # param name -> list of candidate values searched by the optimizer.
    param_grid: dict[str, list]

    @abstractmethod
    def signals(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Returns a Series aligned with df.index: 1 (want long), -1 (want short),
        0 (want flat), computed only from each row's own close and earlier data
        (no look-ahead within this function itself)."""
        raise NotImplementedError

    def mid_close(self, df: pd.DataFrame) -> pd.Series:
        return (df["bid_c"] + df["ask_c"]) / 2.0

    def risk_distances(
        self, df: pd.DataFrame, params: dict, sl: float, tp: float
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        """Converts a risk-grid (sl, tp) pair into the actual pip distances
        passed to `run_backtest`. Default: fixed pips, passed through
        unchanged. A strategy can override this to interpret sl/tp
        differently based on its own params -- e.g. as a fraction of a
        self-computed range (a stop/target sized to what the market is
        actually doing at entry, rather than one static pip count) -- in
        which case the optimizer's risk_search grid values mean whatever
        that strategy defines them to mean instead of pips."""
        return sl, tp
