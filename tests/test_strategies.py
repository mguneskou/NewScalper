import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scalper.strategies.bollinger_breakout import BollingerBreakoutStrategy
from scalper.strategies.ema_cross import EmaCrossStrategy
from scalper.strategies.opening_range_breakout import OpeningRangeBreakoutStrategy
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


def make_orb_df(prices: list[float], start: str = "2026-01-05T00:00:00Z") -> pd.DataFrame:
    """1-minute bars starting at UTC midnight. Europe/London is plain GMT
    (UTC+0) in January, so UTC times double as the strategy's local session time."""
    n = len(prices)
    times = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "time": times.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
            "bid_o": prices, "bid_h": prices, "bid_l": prices, "bid_c": prices,
            "ask_o": prices, "ask_h": prices, "ask_l": prices, "ask_c": prices,
            "volume": [1] * n, "complete": [True] * n,
        }
    )


def test_orb_flat_during_box_formation():
    # session_open_hour=0, box_minutes=10 -> box window is minutes 0-9.
    box_prices = [1.1000, 1.1010, 1.1002, 1.1008, 1.1000, 1.1005, 1.1001, 1.1009, 1.1003, 1.1006]
    df = make_orb_df(box_prices)
    strat = OpeningRangeBreakoutStrategy()
    sig = strat.signals(df, {"session_open_hour": 0, "box_minutes": 10})
    assert (sig == 0).all()


def test_orb_goes_long_on_upside_breakout():
    box_prices = [1.1000, 1.1010, 1.1002, 1.1008, 1.1000, 1.1005, 1.1001, 1.1009, 1.1003, 1.1006]
    # box high = 1.1010, box low = 1.1000
    breakout_prices = [1.1015, 1.1025, 1.1030]
    df = make_orb_df(box_prices + breakout_prices)
    strat = OpeningRangeBreakoutStrategy()
    sig = strat.signals(df, {"session_open_hour": 0, "box_minutes": 10})
    assert (sig.iloc[:10] == 0).all()
    assert (sig.iloc[10:] == 1).all()


def test_orb_exits_to_flat_on_reversion_to_box_mid():
    box_prices = [1.1000, 1.1010, 1.1002, 1.1008, 1.1000, 1.1005, 1.1001, 1.1009, 1.1003, 1.1006]
    # box high=1.1010, low=1.1000 -> mid=1.1005
    prices_after = [1.1015, 1.1020, 1.1005, 1.1004]
    df = make_orb_df(box_prices + prices_after)
    strat = OpeningRangeBreakoutStrategy()
    sig = strat.signals(df, {"session_open_hour": 0, "box_minutes": 10})
    assert sig.iloc[10] == 1  # 1.1015 > 1.1010 -> breaks out long
    assert sig.iloc[11] == 1  # 1.1020 still above mid -> stays long
    assert sig.iloc[12] == 0  # 1.1005 <= mid(1.1005) -> reverts to flat
    assert sig.iloc[13] == 0


def test_orb_resets_for_a_new_day():
    day1_box = [1.1000, 1.1010, 1.1002, 1.1008, 1.1000]  # box_minutes=5 -> high=1.1010 low=1.1000
    day1_rest = [1.1020] * (1440 - len(day1_box))  # breaks out and never reverts for the rest of day 1
    day2_box = [1.0500, 1.0510, 1.0502, 1.0508, 1.0500]  # an unrelated day-2 box
    day2_after = [1.0520]  # breaks day 2's own box high (1.0510)

    df = make_orb_df(day1_box + day1_rest + day2_box + day2_after)
    strat = OpeningRangeBreakoutStrategy()
    sig = strat.signals(df, {"session_open_hour": 0, "box_minutes": 5})

    day2_start = 1440
    # day 2's box-forming window forces flat regardless of day 1's still-open long.
    assert (sig.iloc[day2_start:day2_start + 5] == 0).all()
    # day 2's own breakout is detected fresh, independent of day 1's box levels.
    assert sig.iloc[-1] == 1
