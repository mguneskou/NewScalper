import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scalper.backtest.engine import BacktestResult, Trade
from scalper.backtest.metrics import compute_metrics

STARTING_BALANCE = 1000.0


def make_trade(pnl: float, exit_time: str) -> Trade:
    return Trade(
        instrument="EUR_USD",
        direction=1,
        entry_time=pd.Timestamp(exit_time) - pd.Timedelta(hours=1),
        entry_price=1.1000,
        exit_time=pd.Timestamp(exit_time),
        exit_price=1.1010,
        units=1000,
        pnl=pnl,
        exit_reason="signal",
    )


def make_result(pnls: list[float]) -> BacktestResult:
    trades = [make_trade(pnl, f"2026-01-0{i + 1}T12:00:00Z") for i, pnl in enumerate(pnls)]
    balance = STARTING_BALANCE
    values, times = [], []
    for t in trades:
        balance += t.pnl
        values.append(balance)
        times.append(t.exit_time)
    equity_curve = pd.Series(values, index=pd.DatetimeIndex(times))
    return BacktestResult(trades=trades, equity_curve=equity_curve, starting_balance=STARTING_BALANCE)


def test_empty_trades_gives_zeroed_metrics():
    result = BacktestResult(trades=[], equity_curve=pd.Series(dtype=float), starting_balance=STARTING_BALANCE)
    m = compute_metrics(result)
    assert m.trade_count == 0
    assert m.win_rate == 0.0
    assert m.profit_factor == 0.0
    assert m.total_pnl == 0.0
    assert m.max_drawdown == 0.0
    assert m.sharpe_ratio == 0.0
    assert m.ending_balance == STARTING_BALANCE


def test_known_trade_sequence_basic_stats():
    pnls = [100, -50, 80, -30, 20]
    result = make_result(pnls)
    m = compute_metrics(result)

    assert m.trade_count == 5
    assert m.win_rate == 3 / 5
    assert abs(m.total_pnl - 120) < 1e-9
    assert abs(m.expectancy - 24) < 1e-9
    # gross_profit=200, gross_loss=80
    assert abs(m.profit_factor - 2.5) < 1e-9
    assert abs(m.ending_balance - 1120) < 1e-9


def test_known_trade_sequence_drawdown():
    # equity path: 1000 -> 1100 -> 1050 -> 1130 -> 1100 -> 1120
    pnls = [100, -50, 80, -30, 20]
    result = make_result(pnls)
    m = compute_metrics(result)

    # worst drawdown: peak 1130 -> trough 1100 (after the 4th trade) = -30,
    # but also peak 1100 -> trough 1050 (after 2nd trade) = -50 -- the deeper one.
    assert abs(m.max_drawdown - (-50)) < 1e-9
    assert abs(m.max_drawdown_pct - (-50 / 1100)) < 1e-6


def test_sharpe_matches_independent_daily_return_calc():
    pnls = [100, -50, 80, -30, 20]
    result = make_result(pnls)
    m = compute_metrics(result)

    equity = [1100, 1050, 1130, 1100, 1120]
    daily_returns = np.diff(equity) / np.array(equity[:-1])
    expected_sharpe = daily_returns.mean() / daily_returns.std(ddof=1) * np.sqrt(252)
    assert abs(m.sharpe_ratio - expected_sharpe) < 1e-6


def test_all_wins_gives_infinite_profit_factor():
    result = make_result([10, 20, 30])
    m = compute_metrics(result)
    assert m.win_rate == 1.0
    assert m.profit_factor == float("inf")


def test_all_losses_gives_zero_win_rate_and_profit_factor():
    result = make_result([-10, -20, -30])
    m = compute_metrics(result)
    assert m.win_rate == 0.0
    assert m.profit_factor == 0.0
    assert abs(m.total_pnl - (-60)) < 1e-9
