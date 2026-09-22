"""Prototype: walk-forward backtests rsi_reversion with an ATR-relative
stop-loss/take-profit instead of the fixed pip grid used by run_backtest.py.

Every fixed-pip combo tested so far (M5 and M1, 44 strategy/instrument/
granularity combos) landed below profit factor 1.0 out-of-sample. A static
pip stop is the same width in a dead-quiet market as a violent one; this
script tests whether sizing the stop/take-profit to *current* volatility
(ATR) instead closes any of that gap for rsi_reversion specifically -- the
strategy whose fixed-pip results were the closest to breakeven.

Deliberately kept separate from optimizer.py/run_backtest.py rather than
generalizing their fixed-pip risk grid to also support ATR: this is a
one-strategy experiment, not a permanent second risk model for every
strategy, so a small standalone script is less invasive than reshaping the
shared walk-forward machinery around a risk-model abstraction it doesn't
otherwise need.
"""

import argparse
import itertools
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scalper.backtest.costs import CostModel
from scalper.backtest.engine import BacktestResult, equity_curve_from_trades, run_backtest
from scalper.backtest.fx import FxConverter
from scalper.backtest.metrics import Metrics, compute_metrics
from scalper.backtest.optimizer import walk_forward_windows
from scalper.backtest.risk import atr_pips
from scalper.config import load_settings
from scalper.data.history import load_cached
from scalper.data.storage import connect, save_backtest_run
from scalper.strategies.rsi_reversion import RsiReversionStrategy

STARTING_BALANCE = 10_000.0

ATR_RISK_GRID = {
    # atr_period fixed at 7 (the value picked in 8/12 windows of the previous run)
    # and atr_mult_tp narrowed to values near its previous pick of 6 (the grid's
    # *minimum* last time, never the wide end) -- both axes already have an answer,
    # so this pass spends its whole budget on atr_mult_sl, which pinned at 14 (the
    # old max) in every window and needs to be pushed further to find where it stops
    # helping.
    "atr_period": [7],
    "atr_mult_sl": [14, 20, 28],
    "atr_mult_tp": [6, 10],
    "position_size_units": [1000],
}

_worker_df: pd.DataFrame | None = None
_worker_cost_model: CostModel | None = None
_worker_instrument: str | None = None
_worker_account_currency: str | None = None
_worker_fx: FxConverter | None = None


def _init_worker(df, cost_model, instrument, account_currency, granularity):
    global _worker_df, _worker_cost_model, _worker_instrument, _worker_account_currency, _worker_fx
    _worker_df = df
    _worker_cost_model = cost_model
    _worker_instrument = instrument
    _worker_account_currency = account_currency
    _worker_fx = FxConverter(account_currency, granularity)


def _param_combinations(grid: dict[str, list]) -> list[dict]:
    keys = list(grid.keys())
    return [dict(zip(keys, values)) for values in itertools.product(*[grid[k] for k in keys])]


def _evaluate_combo(args: tuple) -> tuple[dict, Metrics]:
    strategy_params, atr_period, atr_mult_sl, atr_mult_tp, size = args
    strategy = RsiReversionStrategy()
    signals = strategy.signals(_worker_df, strategy_params)
    atr = atr_pips(_worker_df, atr_period, _worker_instrument).to_numpy()
    result = run_backtest(
        _worker_df, signals, _worker_instrument,
        stop_loss_pips=atr * atr_mult_sl, take_profit_pips=atr * atr_mult_tp,
        position_size_units=size, cost_model=_worker_cost_model,
        account_currency=_worker_account_currency, fx=_worker_fx,
    )
    metrics = compute_metrics(result)
    params = {
        **strategy_params,
        "atr_period": atr_period, "atr_mult_sl": atr_mult_sl, "atr_mult_tp": atr_mult_tp,
        "position_size_units": size,
    }
    return params, metrics


def _slice_window(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    times = pd.to_datetime(df["time"])
    return df.loc[(times >= start) & (times < end)].reset_index(drop=True)


def grid_search_train(
    df_train: pd.DataFrame, risk_grid: dict, instrument: str, cost_model: CostModel,
    account_currency: str, granularity: str, max_workers: int | None,
    objective: str = "avg_r_multiple", min_trades: int = 30,
) -> list[tuple[dict, Metrics]]:
    # See optimizer.grid_search_train for why this ranks by a risk-normalized
    # objective with a minimum-sample floor rather than raw total_pnl: with a
    # fixed position size, total_pnl rewards trade volume as much as edge.
    strategy_combos = _param_combinations(RsiReversionStrategy.param_grid)
    risk_combos = _param_combinations(risk_grid)
    tasks = [
        (sp, rp["atr_period"], rp["atr_mult_sl"], rp["atr_mult_tp"], rp["position_size_units"])
        for sp in strategy_combos
        for rp in risk_combos
    ]
    with ProcessPoolExecutor(
        max_workers=max_workers, initializer=_init_worker,
        initargs=(df_train, cost_model, instrument, account_currency, granularity),
    ) as pool:
        results = list(pool.map(_evaluate_combo, tasks, chunksize=max(1, len(tasks) // 32)))
    results = [r for r in results if r[1].trade_count >= min_trades] or results
    results.sort(key=lambda r: getattr(r[1], objective), reverse=True)
    return results


def run_walk_forward_atr(
    df: pd.DataFrame, instrument: str, risk_grid: dict, cost_model: CostModel,
    train_months: int, validate_months: int, account_currency: str, granularity: str,
    max_workers: int | None = None, on_window_done=None,
) -> list[dict]:
    windows = walk_forward_windows(df, train_months, validate_months)
    fx = FxConverter(account_currency, granularity)
    results = []

    for window_index, (train_start, train_end, validate_start, validate_end) in enumerate(windows, start=1):
        df_train = _slice_window(df, train_start, train_end)
        df_validate = _slice_window(df, validate_start, validate_end)
        if df_train.empty or df_validate.empty:
            continue

        ranked = grid_search_train(
            df_train, risk_grid, instrument, cost_model, account_currency, granularity, max_workers
        )
        if not ranked:
            continue
        best_params, _train_metrics = ranked[0]

        strategy = RsiReversionStrategy()
        signals = strategy.signals(df_validate, best_params)
        atr = atr_pips(df_validate, best_params["atr_period"], instrument).to_numpy()
        validate_result = run_backtest(
            df_validate, signals, instrument,
            stop_loss_pips=atr * best_params["atr_mult_sl"],
            take_profit_pips=atr * best_params["atr_mult_tp"],
            position_size_units=best_params["position_size_units"],
            cost_model=cost_model, account_currency=account_currency, fx=fx,
        )
        validate_metrics = compute_metrics(validate_result)

        window_result = dict(
            train_start=train_start, train_end=train_end,
            validate_start=validate_start, validate_end=validate_end,
            best_params=best_params, validate_metrics=validate_metrics,
            validate_trades=validate_result.trades,
        )
        results.append(window_result)
        if on_window_done is not None:
            on_window_done(window_index, len(windows), window_result)

    return results


def _clip_to_recent_years(df: pd.DataFrame, years: float) -> pd.DataFrame:
    if df.empty:
        return df
    times = pd.to_datetime(df["time"])
    cutoff = times.max() - pd.DateOffset(days=round(years * 365))
    return df.loc[times >= cutoff].reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--granularities", nargs="+", default=None)
    parser.add_argument("--years", type=float, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    args = parser.parse_args()

    settings = load_settings()
    instruments = settings["instruments"]
    granularities = args.granularities or settings["granularities"]
    bt_cfg = settings["backtest"]
    account_currency = settings["account_currency"]
    years = args.years if args.years is not None else bt_cfg["years"]

    cost_model = CostModel(
        slippage_pips=bt_cfg["slippage_pips"],
        commission_per_trade=bt_cfg["commission_per_trade"],
    )

    jobs = [
        (instrument, granularity)
        for instrument in instruments
        for granularity in granularities
        if not load_cached(instrument, granularity).empty
    ]
    total_jobs = len(jobs)
    run_start = time.time()
    summary_rows = []
    print(f"strategy=rsi_reversion (ATR stops) account_currency={account_currency} years={years}", flush=True)

    with connect() as conn:
        for job_index, (instrument, granularity) in enumerate(jobs, start=1):
            df = _clip_to_recent_years(load_cached(instrument, granularity), years)
            print(f"[job {job_index}/{total_jobs}] rsi_reversion_atr on {instrument} {granularity} ({len(df)} bars)...", flush=True)
            job_start = time.time()

            def on_window_done(window_index, total_windows, window_result, _instrument=instrument,
                                _granularity=granularity, _job_start=job_start):
                save_backtest_run(
                    conn,
                    strategy_name="rsi_reversion_atr",
                    instrument=_instrument,
                    granularity=_granularity,
                    params=window_result["best_params"],
                    train_start=str(window_result["train_start"]),
                    train_end=str(window_result["train_end"]),
                    validate_start=str(window_result["validate_start"]),
                    validate_end=str(window_result["validate_end"]),
                    is_out_of_sample=True,
                    account_currency=account_currency,
                    metrics=window_result["validate_metrics"],
                    trades=window_result["validate_trades"],
                )
                conn.commit()
                elapsed = time.time() - _job_start
                eta = (elapsed / window_index) * (total_windows - window_index)
                m = window_result["validate_metrics"]
                print(
                    f"    window {window_index}/{total_windows} "
                    f"[{window_result['validate_start'].date()} -> {window_result['validate_end'].date()}] "
                    f"trades={m.trade_count} pnl={m.total_pnl:.2f}  elapsed={elapsed:.0f}s eta={eta:.0f}s",
                    flush=True,
                )

            window_results = run_walk_forward_atr(
                df, instrument, ATR_RISK_GRID, cost_model,
                train_months=bt_cfg["train_months"], validate_months=bt_cfg["validate_months"],
                account_currency=account_currency, granularity=granularity,
                max_workers=args.max_workers, on_window_done=on_window_done,
            )
            if not window_results:
                print("  not enough data for even one walk-forward window, skipping", flush=True)
                continue

            oos_trades = sorted(
                (t for w in window_results for t in w["validate_trades"]),
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
            summary_rows.append((instrument, granularity, oos_metrics))

    print(f"\n=== rsi_reversion_atr out-of-sample summary (best to worst by total P&L, in {account_currency}) ===", flush=True)
    summary_rows.sort(key=lambda r: r[2].total_pnl, reverse=True)
    header = f"{'instrument':10s} {'gran':4s} {'trades':>7s} {'win%':>6s} {'pf':>6s} {'pnl_' + account_currency:>10s} {'sharpe':>7s} {'max_dd':>10s}"
    print(header, flush=True)
    for instrument, granularity, m in summary_rows:
        print(
            f"{instrument:10s} {granularity:4s} {m.trade_count:7d} "
            f"{m.win_rate:6.1%} {m.profit_factor:6.2f} {m.total_pnl:10.2f} "
            f"{m.sharpe_ratio:7.2f} {m.max_drawdown:10.2f}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
