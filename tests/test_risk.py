import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scalper.backtest.risk import atr_pips

INSTRUMENT = "EUR_USD"


def test_atr_pips_matches_hand_computed_true_range():
    # Equal bid/ask so mid H/L/C are just the given values, easy to hand-check.
    # Bar 0 TR = 10 pips (high-low, no prior close).
    # Bar 1 TR = max(15, 10, 5) = 15 pips (prior close 1.1010).
    # Bar 2 TR = max(20, 20, 0) = 20 pips (prior close 1.1010).
    df = pd.DataFrame(
        {
            "time": ["2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z", "2026-01-01T00:02:00Z"],
            "bid_o": [1.1000, 1.1010, 1.1015], "ask_o": [1.1000, 1.1010, 1.1015],
            "bid_h": [1.1010, 1.1020, 1.1030], "ask_h": [1.1010, 1.1020, 1.1030],
            "bid_l": [1.1000, 1.1005, 1.1010], "ask_l": [1.1000, 1.1005, 1.1010],
            "bid_c": [1.1010, 1.1010, 1.1020], "ask_c": [1.1010, 1.1010, 1.1020],
            "volume": [10, 10, 10], "complete": [True, True, True],
        }
    )

    result = atr_pips(df, period=2, instrument=INSTRUMENT)

    # ewm(alpha=1/2, adjust=False) internally: atr0=10, atr1=0.5*15+0.5*10=12.5, atr2=0.5*20+0.5*12.5=16.25.
    # min_periods=2 masks bar 0 (only 1 observation) as NaN, then bfill() pulls bar 1's
    # value back into it -- both read as the first fully-supported ATR estimate.
    assert abs(result.iloc[0] - 12.5) < 1e-6
    assert abs(result.iloc[1] - 12.5) < 1e-6
    assert abs(result.iloc[2] - 16.25) < 1e-6


def test_atr_pips_never_nan():
    df = pd.DataFrame(
        {
            "time": [f"2026-01-01T00:{i:02d}:00Z" for i in range(5)],
            "bid_o": [1.10] * 5, "ask_o": [1.1002] * 5,
            "bid_h": [1.1005] * 5, "ask_h": [1.1007] * 5,
            "bid_l": [1.0995] * 5, "ask_l": [1.0997] * 5,
            "bid_c": [1.10] * 5, "ask_c": [1.1002] * 5,
            "volume": [10] * 5, "complete": [True] * 5,
        }
    )
    result = atr_pips(df, period=14, instrument=INSTRUMENT)
    assert not result.isna().any()
