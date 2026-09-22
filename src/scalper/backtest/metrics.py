"""Performance metrics computed from a BacktestResult."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scalper.backtest.engine import BacktestResult


@dataclass
class Metrics:
    trade_count: int
    win_rate: float
    profit_factor: float
    total_pnl: float
    expectancy: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    avg_r_multiple: float
    ending_balance: float


def compute_metrics(result: BacktestResult) -> Metrics:
    trades = result.trades
    trade_count = len(trades)

    if trade_count == 0:
        return Metrics(
            trade_count=0,
            win_rate=0.0,
            profit_factor=0.0,
            total_pnl=0.0,
            expectancy=0.0,
            max_drawdown=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            avg_r_multiple=0.0,
            ending_balance=result.starting_balance,
        )

    pnls = np.array([t.pnl for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    win_rate = len(wins) / trade_count
    gross_profit = wins.sum() if len(wins) else 0.0
    gross_loss = -losses.sum() if len(losses) else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0
    total_pnl = pnls.sum()
    expectancy = pnls.mean()
    # Mean P&L expressed as a multiple of what each trade risked at entry -- unlike
    # total_pnl/expectancy, this is comparable across combos with different stop
    # distances (fixed-pip or ATR-relative) regardless of position size, since a
    # tight-stop combo's trades are scored against their own (smaller) risk rather
    # than looking artificially cheap-and-plentiful next to a wide-stop combo's.
    avg_r_multiple = float(np.mean([t.r_multiple for t in trades]))

    equity = pd.concat(
        [pd.Series([result.starting_balance]), result.equity_curve.reset_index(drop=True)]
    ).reset_index(drop=True)
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_drawdown = float(drawdown.min())
    max_drawdown_pct = float((drawdown / running_max).min()) if running_max.max() > 0 else 0.0

    # Daily-resampled, annualized Sharpe -- comparable across strategies regardless
    # of trade frequency (a trade-count-scaled Sharpe would favor high-frequency
    # strategies purely for taking more trades, not for a better risk-adjusted edge).
    daily_equity = result.equity_curve.resample("1D").last().ffill()
    daily_returns = daily_equity.pct_change().dropna()
    if len(daily_returns) > 1 and daily_returns.std(ddof=1) > 0:
        sharpe = float(daily_returns.mean() / daily_returns.std(ddof=1) * np.sqrt(252))
    else:
        sharpe = 0.0

    return Metrics(
        trade_count=trade_count,
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_pnl=float(total_pnl),
        expectancy=float(expectancy),
        max_drawdown=max_drawdown,
        max_drawdown_pct=max_drawdown_pct,
        sharpe_ratio=sharpe,
        avg_r_multiple=avg_r_multiple,
        ending_balance=result.starting_balance + float(total_pnl),
    )
