import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scalper.backtest.costs import CostModel
from scalper.backtest.engine import run_backtest

INSTRUMENT = "EUR_USD"
PIP = 0.0001


def make_df(rows: list[dict]) -> pd.DataFrame:
    """rows: list of dicts with bid_o/h/l, ask_o/h/l. Close columns are unused by
    the engine but included for realism / parquet-shape parity."""
    for i, r in enumerate(rows):
        r.setdefault("time", f"2026-01-01T00:{i:02d}:00.000000000Z")
        r.setdefault("bid_c", r["bid_o"])
        r.setdefault("ask_c", r["ask_o"])
        r.setdefault("volume", 10)
        r.setdefault("complete", True)
    return pd.DataFrame(rows)


def test_signal_exit_pnl_and_no_lookahead():
    """Signal appears on bar 1's close; must NOT enter until bar 2's open (no lookahead),
    and must exit at bar 3's open when the signal returns to flat."""
    df = make_df(
        [
            {"bid_o": 1.1000, "bid_h": 1.1000, "bid_l": 1.1000, "ask_o": 1.1002, "ask_h": 1.1002, "ask_l": 1.1002},
            {"bid_o": 1.1000, "bid_h": 1.1000, "bid_l": 1.1000, "ask_o": 1.1002, "ask_h": 1.1002, "ask_l": 1.1002},
            {"bid_o": 1.1010, "bid_h": 1.1010, "bid_l": 1.1010, "ask_o": 1.1012, "ask_h": 1.1012, "ask_l": 1.1012},
            {"bid_o": 1.1020, "bid_h": 1.1020, "bid_l": 1.1020, "ask_o": 1.1022, "ask_h": 1.1022, "ask_l": 1.1022},
            {"bid_o": 1.1020, "bid_h": 1.1020, "bid_l": 1.1020, "ask_o": 1.1022, "ask_h": 1.1022, "ask_l": 1.1022},
        ]
    )
    # signal goes long at bar index 1 (known at bar 1's close), flat again at bar 3.
    signals = pd.Series([0, 1, 1, 0, 0])
    cost_model = CostModel(slippage_pips=0.0, commission_per_trade=0.0)

    result = run_backtest(
        df, signals, INSTRUMENT,
        stop_loss_pips=1000, take_profit_pips=1000,  # effectively disabled
        position_size_units=1000,
        cost_model=cost_model,
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    # entered at bar 2's open ask (signal known at bar 1 close acts on bar 2), not bar 1's.
    assert trade.entry_price == 1.1012
    # exited at bar 3's open bid (signal flat known at bar 2 close acts on bar 3).
    assert trade.exit_price == 1.1020
    assert trade.exit_reason == "signal"
    expected_pnl = (1.1020 - 1.1012) * 1000
    assert abs(trade.pnl - expected_pnl) < 1e-9


def test_stop_loss_hit_intrabar():
    df = make_df(
        [
            {"bid_o": 1.1000, "bid_h": 1.1000, "bid_l": 1.1000, "ask_o": 1.1002, "ask_h": 1.1002, "ask_l": 1.1002},
            {"bid_o": 1.1000, "bid_h": 1.1000, "bid_l": 1.1000, "ask_o": 1.1002, "ask_h": 1.1002, "ask_l": 1.1002},
            # entry bar (signal known at bar 1's close acts here): buy at ask_o=1.1002, stop = 1.0997
            {"bid_o": 1.1000, "bid_h": 1.1000, "bid_l": 1.1000, "ask_o": 1.1002, "ask_h": 1.1002, "ask_l": 1.1002},
            # first bar the stop can be checked on (position wasn't open yet during the entry bar's check)
            {"bid_o": 1.1000, "bid_h": 1.1000, "bid_l": 1.0940, "ask_o": 1.1002, "ask_h": 1.1002, "ask_l": 1.0942},
        ]
    )
    # signals[2]=0 -> effective signal is flat on the stop-check bar, so once the
    # stop closes the position the engine won't immediately re-enter on the same bar.
    signals = pd.Series([0, 1, 0, 0])
    cost_model = CostModel(slippage_pips=0.0, commission_per_trade=0.0)

    result = run_backtest(
        df, signals, INSTRUMENT,
        stop_loss_pips=5, take_profit_pips=1000,
        position_size_units=1000,
        cost_model=cost_model,
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "stop_loss"
    expected_stop = 1.1002 - 5 * PIP
    assert abs(trade.exit_price - expected_stop) < 1e-9
    assert trade.pnl < 0


def test_slippage_and_commission_applied():
    df = make_df(
        [
            {"bid_o": 1.1000, "bid_h": 1.1000, "bid_l": 1.1000, "ask_o": 1.1002, "ask_h": 1.1002, "ask_l": 1.1002},
            {"bid_o": 1.1000, "bid_h": 1.1000, "bid_l": 1.1000, "ask_o": 1.1002, "ask_h": 1.1002, "ask_l": 1.1002},
            {"bid_o": 1.1050, "bid_h": 1.1050, "bid_l": 1.1050, "ask_o": 1.1052, "ask_h": 1.1052, "ask_l": 1.1052},
            {"bid_o": 1.1050, "bid_h": 1.1050, "bid_l": 1.1050, "ask_o": 1.1052, "ask_h": 1.1052, "ask_l": 1.1052},
        ]
    )
    signals = pd.Series([0, 1, 0, 0])
    cost_model = CostModel(slippage_pips=0.5, commission_per_trade=2.0)

    result = run_backtest(
        df, signals, INSTRUMENT,
        stop_loss_pips=1000, take_profit_pips=1000,
        position_size_units=1000,
        cost_model=cost_model,
    )

    # signals[1]=1 is known at bar 1's close -> entry executes at bar 2's open;
    # signals[2]=0 is known at bar 2's close -> exit executes at bar 3's open.
    trade = result.trades[0]
    # buy fills 0.5 pip worse than bar 2's ask_o (1.1052): 1.10525
    assert abs(trade.entry_price - (1.1052 + 0.5 * PIP)) < 1e-9
    # sell fills 0.5 pip worse than bar 3's bid_o (1.1050): 1.10495
    assert abs(trade.exit_price - (1.1050 - 0.5 * PIP)) < 1e-9
    expected_pnl = (trade.exit_price - trade.entry_price) * 1000 - 2.0
    assert abs(trade.pnl - expected_pnl) < 1e-9


def test_short_trade_take_profit():
    df = make_df(
        [
            {"bid_o": 1.1000, "bid_h": 1.1000, "bid_l": 1.1000, "ask_o": 1.1002, "ask_h": 1.1002, "ask_l": 1.1002},
            {"bid_o": 1.1000, "bid_h": 1.1000, "bid_l": 1.1000, "ask_o": 1.1002, "ask_h": 1.1002, "ask_l": 1.1002},
            # entry bar (signal known at bar 1's close acts here): sell at bid_o=1.1000, tp = 1.0995
            {"bid_o": 1.1000, "bid_h": 1.1000, "bid_l": 1.1000, "ask_o": 1.1002, "ask_h": 1.1002, "ask_l": 1.1002},
            # first bar the tp can be checked on
            {"bid_o": 1.1000, "bid_h": 1.1000, "bid_l": 1.0994, "ask_o": 1.1002, "ask_h": 1.1002, "ask_l": 1.0990},
        ]
    )
    signals = pd.Series([0, -1, 0, 0])
    cost_model = CostModel(slippage_pips=0.0, commission_per_trade=0.0)

    result = run_backtest(
        df, signals, INSTRUMENT,
        stop_loss_pips=1000, take_profit_pips=5,
        position_size_units=1000,
        cost_model=cost_model,
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.direction == -1
    assert trade.exit_reason == "take_profit"
    expected_tp = 1.1000 - 5 * PIP
    assert abs(trade.exit_price - expected_tp) < 1e-9
    assert trade.pnl > 0


def test_open_position_force_closed_at_end_of_data():
    df = make_df(
        [
            {"bid_o": 1.1000, "bid_h": 1.1000, "bid_l": 1.1000, "ask_o": 1.1002, "ask_h": 1.1002, "ask_l": 1.1002},
            {"bid_o": 1.1000, "bid_h": 1.1000, "bid_l": 1.1000, "ask_o": 1.1002, "ask_h": 1.1002, "ask_l": 1.1002},
            {"bid_o": 1.1010, "bid_h": 1.1010, "bid_l": 1.1010, "ask_o": 1.1012, "ask_h": 1.1012, "ask_l": 1.1012},
        ]
    )
    signals = pd.Series([0, 1, 1])
    cost_model = CostModel(slippage_pips=0.0, commission_per_trade=0.0)

    result = run_backtest(
        df, signals, INSTRUMENT,
        stop_loss_pips=1000, take_profit_pips=1000,
        position_size_units=1000,
        cost_model=cost_model,
    )

    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "end_of_data"


def test_per_bar_stop_array_uses_entry_bar_value():
    """A per-bar stop/take-profit array (e.g. ATR-relative) locks in whichever
    value sits at the trade's *entry* bar, not a single fixed distance."""
    df = make_df(
        [
            {"bid_o": 1.1000, "bid_h": 1.1000, "bid_l": 1.1000, "ask_o": 1.1002, "ask_h": 1.1002, "ask_l": 1.1002},
            {"bid_o": 1.1000, "bid_h": 1.1000, "bid_l": 1.1000, "ask_o": 1.1002, "ask_h": 1.1002, "ask_l": 1.1002},
            # entry bar (signal known at bar 1's close acts here): buy at ask_o=1.1002
            {"bid_o": 1.1000, "bid_h": 1.1000, "bid_l": 1.1000, "ask_o": 1.1002, "ask_h": 1.1002, "ask_l": 1.1002},
            # stop-check bar: low dips 3 pips below entry -- the entry bar's 2-pip
            # stop should catch this, even though every other bar's value is 50.
            {"bid_o": 1.1000, "bid_h": 1.1000, "bid_l": 1.0969, "ask_o": 1.1002, "ask_h": 1.1002, "ask_l": 1.0971},
        ]
    )
    signals = pd.Series([0, 1, 0, 0])
    cost_model = CostModel(slippage_pips=0.0, commission_per_trade=0.0)
    stop_array = pd.Series([50.0, 50.0, 2.0, 50.0])

    result = run_backtest(
        df, signals, INSTRUMENT,
        stop_loss_pips=stop_array, take_profit_pips=1000,
        position_size_units=1000,
        cost_model=cost_model,
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "stop_loss"
    expected_stop = 1.1002 - 2 * PIP
    assert abs(trade.exit_price - expected_stop) < 1e-9
