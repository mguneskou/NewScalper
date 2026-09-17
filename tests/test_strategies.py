import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scalper.strategies.bollinger_breakout import BollingerBreakoutStrategy
from scalper.strategies.ema_cross import EmaCrossStrategy
from scalper.strategies.rsi_reversion import RsiReversionStrategy


def make_price_df(closes: list[float]) -> pd.DataFrame:
    n = len(closes)
    return pd.DataFrame(
        {
            "time": [f"2026-01-01T00:{i:02d}:00.000000000Z" for i in range(n)],
            "bid_o": closes,
            "bid_h": closes,
            "bid_l": closes,
            "bid_c": closes,
            "ask_o": closes,
            "ask_h": closes,
            "ask_l": closes,
            "ask_c": closes,
            "volume": [1] * n,
            "complete": [True] * n,
        }
    )


def test_ema_cross_goes_long_in_uptrend():
    closes = list(np.linspace(1.0, 1.1, 60))
    df = make_price_df(closes)
    strat = EmaCrossStrategy()
    sig = strat.signals(df, {"fast_period": 5, "slow_period": 20})
    assert sig.iloc[-1] == 1


def test_ema_cross_goes_short_in_downtrend():
    closes = list(np.linspace(1.1, 1.0, 60))
    df = make_price_df(closes)
    strat = EmaCrossStrategy()
    sig = strat.signals(df, {"fast_period": 5, "slow_period": 20})
    assert sig.iloc[-1] == -1


def test_ema_cross_invalid_params_stays_flat():
    closes = list(np.linspace(1.0, 1.1, 30))
    df = make_price_df(closes)
    strat = EmaCrossStrategy()
    sig = strat.signals(df, {"fast_period": 20, "slow_period": 5})
    assert (sig == 0).all()


def test_rsi_reversion_goes_long_after_sharp_drop():
    closes = [1.10] * 20 + list(np.linspace(1.10, 1.05, 15))  # sharp decline -> oversold
    df = make_price_df(closes)
    strat = RsiReversionStrategy()
    sig = strat.signals(df, {"period": 14, "oversold": 30, "overbought": 70})
    assert sig.iloc[-1] == 1


def test_rsi_reversion_goes_short_after_sharp_rise():
    closes = [1.05] * 20 + list(np.linspace(1.05, 1.10, 15))  # sharp rise -> overbought
    df = make_price_df(closes)
    strat = RsiReversionStrategy()
    sig = strat.signals(df, {"period": 14, "oversold": 30, "overbought": 70})
    assert sig.iloc[-1] == -1


def test_bollinger_breakout_goes_long_on_upside_break():
    flat = [1.10] * 25
    spike = [1.10 + 0.001 * i for i in range(1, 6)]
    closes = flat + spike
    df = make_price_df(closes)
    strat = BollingerBreakoutStrategy()
    sig = strat.signals(df, {"period": 20, "num_std": 2.0})
    assert sig.iloc[-1] == 1


def test_bollinger_breakout_flat_before_warmup():
    closes = [1.10 + 0.0001 * i for i in range(10)]
    df = make_price_df(closes)
    strat = BollingerBreakoutStrategy()
    sig = strat.signals(df, {"period": 20, "num_std": 2.0})
    assert (sig == 0).all()
