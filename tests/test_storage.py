import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scalper.backtest.engine import BacktestResult, Trade
from scalper.backtest.metrics import compute_metrics
from scalper.data.storage import connect, get_active_params, save_active_params, save_backtest_run


def make_result() -> BacktestResult:
    trades = [
        Trade(
            instrument="EUR_USD", direction=1,
            entry_time=pd.Timestamp("2026-01-01T00:00:00Z"), entry_price=1.1000,
            exit_time=pd.Timestamp("2026-01-01T01:00:00Z"), exit_price=1.1010,
            units=1000, pnl=10.0, pnl_quote_ccy=10.0, risk=5.0, exit_reason="signal",
        ),
        Trade(
            instrument="EUR_USD", direction=-1,
            entry_time=pd.Timestamp("2026-01-02T00:00:00Z"), entry_price=1.1010,
            exit_time=pd.Timestamp("2026-01-02T01:00:00Z"), exit_price=1.1000,
            units=1000, pnl=10.0, pnl_quote_ccy=10.0, risk=5.0, exit_reason="take_profit",
        ),
    ]
    equity = pd.Series([1010.0, 1020.0], index=pd.DatetimeIndex([t.exit_time for t in trades]))
    return BacktestResult(trades=trades, equity_curve=equity, starting_balance=1000.0)


def test_save_and_read_back_backtest_run():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        result = make_result()
        metrics = compute_metrics(result)

        with connect(db_path) as conn:
            run_id = save_backtest_run(
                conn,
                strategy_name="ema_cross",
                instrument="EUR_USD",
                granularity="M5",
                params={"fast_period": 5, "slow_period": 20, "stop_loss_pips": 5, "take_profit_pips": 8},
                train_start="2025-01-01",
                train_end="2025-07-01",
                validate_start="2025-07-01",
                validate_end="2025-09-01",
                is_out_of_sample=True,
                account_currency="USD",
                metrics=metrics,
                trades=result.trades,
            )

        # reopen to confirm the commit actually persisted to disk, not just in-memory
        with connect(db_path) as conn:
            run_row = conn.execute(
                "SELECT strategy_name, instrument, granularity, trade_count, total_pnl, is_out_of_sample "
                "FROM backtest_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            trade_rows = conn.execute(
                "SELECT pnl, exit_reason FROM backtest_trades WHERE run_id = ? ORDER BY id", (run_id,)
            ).fetchall()

        assert run_row == ("ema_cross", "EUR_USD", "M5", 2, 20.0, 1)
        assert trade_rows == [(10.0, "signal"), (10.0, "take_profit")]


def test_multiple_runs_accumulate_independently():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        result = make_result()
        metrics = compute_metrics(result)

        with connect(db_path) as conn:
            for strategy_name in ["ema_cross", "rsi_reversion"]:
                save_backtest_run(
                    conn,
                    strategy_name=strategy_name,
                    instrument="EUR_USD",
                    granularity="M5",
                    params={},
                    train_start=None, train_end=None, validate_start=None, validate_end=None,
                    is_out_of_sample=True,
                    account_currency="USD",
                    metrics=metrics,
                    trades=result.trades,
                )

        with connect(db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0]
            trade_count = conn.execute("SELECT COUNT(*) FROM backtest_trades").fetchone()[0]

        assert count == 2
        assert trade_count == 4


def test_active_params_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        metrics = compute_metrics(make_result())

        with connect(db_path) as conn:
            assert get_active_params(
                conn, strategy_name="opening_range_breakout", instrument="USD_JPY", granularity="M5"
            ) is None

            save_active_params(
                conn,
                strategy_name="opening_range_breakout",
                instrument="USD_JPY",
                granularity="M5",
                params={"stop_loss_pips": 80, "take_profit_pips": 20},
                train_start="2025-01-01", train_end="2025-05-01",
                validate_start="2025-05-01", validate_end="2025-07-01",
                metrics=metrics,
            )

        with connect(db_path) as conn:
            params = get_active_params(
                conn, strategy_name="opening_range_breakout", instrument="USD_JPY", granularity="M5"
            )
        assert params == {"stop_loss_pips": 80, "take_profit_pips": 20}


def test_active_params_overwrite_on_re_tune():
    """A second save for the same (strategy, instrument, granularity) replaces
    the first rather than accumulating -- this is what makes it "evolving"
    instead of a permanent, named, set-and-forget config."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        metrics = compute_metrics(make_result())
        key = dict(strategy_name="opening_range_breakout", instrument="USD_JPY", granularity="M5")

        with connect(db_path) as conn:
            save_active_params(
                conn, **key, params={"stop_loss_pips": 80, "take_profit_pips": 20},
                train_start="2025-01-01", train_end="2025-05-01",
                validate_start="2025-05-01", validate_end="2025-07-01", metrics=metrics,
            )
            save_active_params(
                conn, **key, params={"stop_loss_pips": 60, "take_profit_pips": 15},
                train_start="2025-05-01", train_end="2025-09-01",
                validate_start="2025-09-01", validate_end="2025-11-01", metrics=metrics,
            )

            row_count = conn.execute("SELECT COUNT(*) FROM active_params").fetchone()[0]
            params = get_active_params(conn, **key)

        assert row_count == 1
        assert params == {"stop_loss_pips": 60, "take_profit_pips": 15}
