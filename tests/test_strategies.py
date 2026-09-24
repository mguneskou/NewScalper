import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scalper.strategies.opening_range_breakout import OpeningRangeBreakoutStrategy


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


def test_risk_distances_defaults_to_fixed_pips_passthrough():
    df = make_orb_df([1.1000] * 5)
    strat = OpeningRangeBreakoutStrategy()
    sl, tp = strat.risk_distances(df, {"session_open_hour": 0, "box_minutes": 10}, 60, 15)
    assert (sl, tp) == (60, 15)


def test_risk_distances_box_mode_scales_by_box_height():
    # box high=1.1010, low=1.1000 -> height = 0.0010 = 10 pips (EUR_USD pip = 0.0001)
    box_prices = [1.1000, 1.1010, 1.1002, 1.1008, 1.1000, 1.1005, 1.1001, 1.1009, 1.1003, 1.1006]
    df = make_orb_df(box_prices + [1.1015])  # one bar past the box so box_high/low are known
    strat = OpeningRangeBreakoutStrategy()
    params = {
        "session_open_hour": 0, "box_minutes": 10, "risk_mode": "box", "instrument": "EUR_USD",
    }
    sl, tp = strat.risk_distances(df, params, 1.5, 0.5)
    assert sl[-1] == pytest.approx(15.0)  # 1.5x 10-pip box height
    assert tp[-1] == pytest.approx(5.0)  # 0.5x 10-pip box height


def test_direction_filter_long_blocks_short_entries():
    box_prices = [1.1000, 1.1010, 1.1002, 1.1008, 1.1000, 1.1005, 1.1001, 1.1009, 1.1003, 1.1006]
    breakdown_prices = [1.0990, 1.0980]  # breaks box low (1.1000) -> would normally go short
    df = make_orb_df(box_prices + breakdown_prices)
    strat = OpeningRangeBreakoutStrategy()
    sig = strat.signals(df, {"session_open_hour": 0, "box_minutes": 10, "direction_filter": "long"})
    assert (sig.iloc[10:] == 0).all()  # short entry blocked, stays flat


def test_direction_filter_short_blocks_long_entries():
    box_prices = [1.1000, 1.1010, 1.1002, 1.1008, 1.1000, 1.1005, 1.1001, 1.1009, 1.1003, 1.1006]
    breakout_prices = [1.1015, 1.1025]  # breaks box high (1.1010) -> would normally go long
    df = make_orb_df(box_prices + breakout_prices)
    strat = OpeningRangeBreakoutStrategy()
    sig = strat.signals(df, {"session_open_hour": 0, "box_minutes": 10, "direction_filter": "short"})
    assert (sig.iloc[10:] == 0).all()  # long entry blocked, stays flat


def test_direction_filter_short_still_allows_short_entries_and_exits():
    box_prices = [1.1000, 1.1010, 1.1002, 1.1008, 1.1000, 1.1005, 1.1001, 1.1009, 1.1003, 1.1006]
    # box low=1.1000, mid=1.1005
    prices_after = [1.0990, 1.0980, 1.1005, 1.1006]
    df = make_orb_df(box_prices + prices_after)
    strat = OpeningRangeBreakoutStrategy()
    sig = strat.signals(df, {"session_open_hour": 0, "box_minutes": 10, "direction_filter": "short"})
    assert sig.iloc[10] == -1  # 1.0990 < 1.1000 -> breaks out short
    assert sig.iloc[11] == -1
    assert sig.iloc[12] == 0  # 1.1005 >= mid(1.1005) -> reverts to flat
