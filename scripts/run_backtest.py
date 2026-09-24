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

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scalper.backtest.costs import CostModel
from scalper.backtest.engine import BacktestResult, equity_curve_from_trades
from scalper.backtest.metrics import compute_metrics
from scalper.backtest.optimizer import run_walk_forward
from scalper.config import load_settings
from scalper.data.history import load_cached
from scalper.data.storage import connect, save_active_params, save_backtest_run
from scalper.strategies.opening_range_breakout import OpeningRangeBreakoutStrategy

ALL_STRATEGIES = [OpeningRangeBreakoutStrategy]
STARTING_BALANCE = 10_000.0


def _clip_to_recent_years(df: pd.DataFrame, years: float) -> pd.DataFrame:
    """Keeps only the most recent `years` of cached candles -- lets a run be
    scoped shorter than the full cache without re-fetching or re-caching."""
    if df.empty:
        return df
    times = pd.to_datetime(df["time"])
    cutoff = times.max() - pd.DateOffset(days=round(years * 365))
    return df.loc[times >= cutoff].reset_index(drop=True)


def _clip_to_range(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Keeps only candles in [start, end) -- lets a run be scoped to a
    specific historical slice (e.g. re-validating a previously tuned window
    against an earlier, non-overlapping period) instead of only "the most
    recent N years"."""
    if df.empty:
        return df
    times = pd.to_datetime(df["time"])
    mask = (times >= pd.Timestamp(start, tz="UTC")) & (times < pd.Timestamp(end, tz="UTC"))
    return df.loc[mask].reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--granularities", nargs="+", default=None,
        help="Override config/settings.yaml granularities, e.g. --granularities M5",
    )
    parser.add_argument(
        "--strategies", nargs="+", default=None,
        help="Override which strategies to run, by name, e.g. --strategies opening_range_breakout",
    )
    parser.add_argument(
        "--instruments", nargs="+", default=None,
        help="Override config/settings.yaml instruments, e.g. --instruments GBP_USD",
    )
    parser.add_argument(
        "--years", type=float, default=None,
        help="Only use the most recent N years of cached data (default: settings.yaml backtest.years)",
    )
    parser.add_argument(
        "--start", type=str, default=None,
        help="Explicit ISO start date (e.g. 2025-03-01) -- overrides --years. Use with --end.",
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="Explicit ISO end date (exclusive) -- overrides --years. Use with --start.",
    )
    parser.add_argument(
        "--skip-active-params", action="store_true",
        help="Don't update active_params from this run -- use for validation-only runs on "
             "historical windows, so a check against older data doesn't clobber the current "
             "best (most recent) live params with stale ones.",
    )
    parser.add_argument(
        "--train-months", type=int, default=None,
        help="Override settings.yaml backtest.train_months (walk-forward training window length)",
    )
    parser.add_argument(
        "--validate-months", type=int, default=None,
        help="Override settings.yaml backtest.validate_months (walk-forward validation window length)",
    )
    parser.add_argument("--max-workers", type=int, default=None)
    args = parser.parse_args()

    settings = load_settings()
    instruments = args.instruments or settings["instruments"]
    granularities = args.granularities or settings["granularities"]
    bt_cfg = settings["backtest"]
    account_currency = settings["account_currency"]
    years = args.years if args.years is not None else bt_cfg["years"]
    train_months = args.train_months if args.train_months is not None else bt_cfg["train_months"]
    validate_months = args.validate_months if args.validate_months is not None else bt_cfg["validate_months"]

    strategies = ALL_STRATEGIES
    if args.strategies:
        wanted = set(args.strategies)
        strategies = [s for s in ALL_STRATEGIES if s.name in wanted]
        missing = wanted - {s.name for s in strategies}
        if missing:
            raise SystemExit(f"Unknown --strategies: {sorted(missing)}")

    # max_concurrent_positions is a live-trading portfolio control, not meaningful
    # for this single-instrument, single-position-at-a-time backtest engine.
    base_risk_grid = {k: v for k, v in settings["risk_search"].items() if k != "max_concurrent_positions"}
    instrument_overrides = settings.get("instrument_overrides", {})

    cost_model = CostModel(
        slippage_pips=bt_cfg["slippage_pips"],
        commission_per_trade=bt_cfg["commission_per_trade"],
    )

    # Total (strategy, instrument, granularity) jobs, used for the overall progress line.
    jobs = [
        (strategy_cls, instrument, granularity)
        for instrument in instruments
        for granularity in granularities
        for strategy_cls in strategies
        if not load_cached(instrument, granularity).empty
    ]
    total_jobs = len(jobs)
    run_start = time.time()
    summary_rows = []
    window_desc = f"{args.start}->{args.end}" if args.start else f"years={years}"
    print(f"account_currency={account_currency} {window_desc}", flush=True)

    with connect() as conn:
        for job_index, (strategy_cls, instrument, granularity) in enumerate(jobs, start=1):
            cached = load_cached(instrument, granularity)
            df = _clip_to_range(cached, args.start, args.end) if args.start else _clip_to_recent_years(cached, years)
            strategy = strategy_cls()
            print(
                f"[job {job_index}/{total_jobs}] {strategy.name} on {instrument} {granularity} "
                f"({len(df)} bars)...",
                flush=True,
            )

            overrides = instrument_overrides.get(instrument, {})
            risk_grid = {**base_risk_grid, **overrides.get("risk_search", {})}
            param_grid = {**strategy.param_grid, **overrides.get("strategy_params", {})}
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
                    account_currency=account_currency,
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
                train_months=train_months,
                validate_months=validate_months,
                max_workers=args.max_workers,
                on_window_done=on_window_done,
                account_currency=account_currency,
                granularity=granularity,
                param_grid=param_grid,
            )
            if not window_results:
                print("  not enough data for even one walk-forward window, skipping", flush=True)
                continue

            # "Active" params always reflect the most recently trained window, not a
            # frozen/named config -- re-running this script with fresh data and
            # calling this again is the entire update mechanism (see storage.py).
            # Skipped for validation-only runs on historical windows (--skip-active-params),
            # so checking an older period doesn't clobber the current best live params.
            if not args.skip_active_params:
                latest_window = window_results[-1]
                save_active_params(
                    conn,
                    strategy_name=strategy.name,
                    instrument=instrument,
                    granularity=granularity,
                    params=latest_window.best_params,
                    train_start=str(latest_window.train_start),
                    train_end=str(latest_window.train_end),
                    validate_start=str(latest_window.validate_start),
                    validate_end=str(latest_window.validate_end),
                    metrics=latest_window.validate_metrics,
                )
                conn.commit()

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

    print(f"\n=== Out-of-sample summary (best to worst by total P&L, in {account_currency}) ===", flush=True)
    summary_rows.sort(key=lambda r: r[3].total_pnl, reverse=True)
    header = f"{'strategy':20s} {'instrument':10s} {'gran':4s} {'trades':>7s} {'win%':>6s} {'pf':>6s} {'pnl_' + account_currency:>10s} {'sharpe':>7s} {'max_dd':>10s}"
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
