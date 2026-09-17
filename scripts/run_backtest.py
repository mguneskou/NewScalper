"""CLI: walk-forward optimizes every strategy against every cached instrument/
granularity, and prints + stores the out-of-sample (truthful) results.

Out-of-sample here means: for each walk-forward window, parameters were chosen
using only the training slice, then evaluated on the following, unseen
validation slice. The summary below stitches all validation-window trades
together into one continuous track record -- this is the number to trust, not
any single window's training-set performance.

Progress (window count, elapsed time, ETA) is printed after every single
walk-forward window -- not just after each strategy/instrument -- and each
window's result is committed to the database immediately, so progress can be
checked mid-run instead of only after the whole (potentially long) job ends.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scalper.backtest.costs import CostModel
from scalper.backtest.engine import BacktestResult, equity_curve_from_trades
from scalper.backtest.metrics import compute_metrics
from scalper.backtest.optimizer import run_walk_forward
from scalper.config import load_settings
from scalper.data.history import load_cached
from scalper.data.storage import connect, save_backtest_run
from scalper.strategies.bollinger_breakout import BollingerBreakoutStrategy
from scalper.strategies.ema_cross import EmaCrossStrategy
from scalper.strategies.rsi_reversion import RsiReversionStrategy

STRATEGIES = [EmaCrossStrategy, RsiReversionStrategy, BollingerBreakoutStrategy]
STARTING_BALANCE = 10_000.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--granularities", nargs="+", default=None,
        help="Override config/settings.yaml granularities, e.g. --granularities M5",
    )
    parser.add_argument("--max-workers", type=int, default=None)
    args = parser.parse_args()

    settings = load_settings()
    instruments = settings["instruments"]
    granularities = args.granularities or settings["granularities"]
    bt_cfg = settings["backtest"]

    # max_concurrent_positions is a live-trading portfolio control, not meaningful
    # for this single-instrument, single-position-at-a-time backtest engine.
    risk_grid = {k: v for k, v in settings["risk_search"].items() if k != "max_concurrent_positions"}

    cost_model = CostModel(
        slippage_pips=bt_cfg["slippage_pips"],
        commission_per_trade=bt_cfg["commission_per_trade"],
    )

    # Total (strategy, instrument, granularity) jobs, used for the overall progress line.
    jobs = [
        (strategy_cls, instrument, granularity)
        for instrument in instruments
        for granularity in granularities
        for strategy_cls in STRATEGIES
        if not load_cached(instrument, granularity).empty
    ]
    total_jobs = len(jobs)
    run_start = time.time()
    summary_rows = []

    with connect() as conn:
        for job_index, (strategy_cls, instrument, granularity) in enumerate(jobs, start=1):
            df = load_cached(instrument, granularity)
            strategy = strategy_cls()
            print(
                f"[job {job_index}/{total_jobs}] {strategy.name} on {instrument} {granularity} "
                f"({len(df)} bars)...",
                flush=True,
            )

            job_start = time.time()

            def on_window_done(window_index, total_windows, window_result, _strategy=strategy,
                                _instrument=instrument, _granularity=granularity, _job_start=job_start):
                save_backtest_run(
                    conn,
                    strategy_name=_strategy.name,
                    instrument=_instrument,
                    granularity=_granularity,
                    params=window_result.best_params,
                    train_start=str(window_result.train_start),
                    train_end=str(window_result.train_end),
                    validate_start=str(window_result.validate_start),
                    validate_end=str(window_result.validate_end),
                    is_out_of_sample=True,
                    metrics=window_result.validate_metrics,
                    trades=window_result.validate_trades,
                )
                conn.commit()

                elapsed = time.time() - _job_start
                per_window = elapsed / window_index
                eta = per_window * (total_windows - window_index)
                m = window_result.validate_metrics
                print(
                    f"    window {window_index}/{total_windows} "
                    f"[{window_result.validate_start.date()} -> {window_result.validate_end.date()}] "
                    f"trades={m.trade_count} pnl={m.total_pnl:.2f}  "
                    f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
                    flush=True,
                )

            window_results = run_walk_forward(
                df, strategy, instrument, risk_grid, cost_model,
                train_months=bt_cfg["train_months"],
                validate_months=bt_cfg["validate_months"],
                max_workers=args.max_workers,
                on_window_done=on_window_done,
            )
            if not window_results:
                print("  not enough data for even one walk-forward window, skipping", flush=True)
                continue

            oos_trades = sorted(
                (t for w in window_results for t in w.validate_trades),
                key=lambda t: t.entry_time,
            )
            combined = BacktestResult(
                trades=oos_trades,
                equity_curve=equity_curve_from_trades(oos_trades, STARTING_BALANCE),
                starting_balance=STARTING_BALANCE,
            )
            oos_metrics = compute_metrics(combined)

            total_elapsed = time.time() - run_start
            print(
                f"  done in {time.time() - job_start:.0f}s (total elapsed {total_elapsed:.0f}s): "
                f"windows={len(window_results)} oos_trades={oos_metrics.trade_count} "
                f"win_rate={oos_metrics.win_rate:.1%} profit_factor={oos_metrics.profit_factor:.2f} "
                f"total_pnl={oos_metrics.total_pnl:.2f} sharpe={oos_metrics.sharpe_ratio:.2f} "
                f"max_dd={oos_metrics.max_drawdown:.2f}",
                flush=True,
            )
            summary_rows.append((strategy.name, instrument, granularity, oos_metrics))

    print("\n=== Out-of-sample summary (best to worst by total P&L) ===", flush=True)
    summary_rows.sort(key=lambda r: r[3].total_pnl, reverse=True)
    header = f"{'strategy':20s} {'instrument':10s} {'gran':4s} {'trades':>7s} {'win%':>6s} {'pf':>6s} {'pnl':>10s} {'sharpe':>7s} {'max_dd':>10s}"
    print(header, flush=True)
    for name, instrument, granularity, m in summary_rows:
        print(
            f"{name:20s} {instrument:10s} {granularity:4s} {m.trade_count:7d} "
            f"{m.win_rate:6.1%} {m.profit_factor:6.2f} {m.total_pnl:10.2f} "
            f"{m.sharpe_ratio:7.2f} {m.max_drawdown:10.2f}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
