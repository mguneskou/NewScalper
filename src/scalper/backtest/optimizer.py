"""Walk-forward parameter optimizer.

Strategy + risk parameters are grid-searched on a training window, then the best
combo (by the chosen objective) is scored on the following, unseen validation
window. Only the validation-window ("out-of-sample") results are trustworthy as
an estimate of live performance -- training-window results are always
optimistic because the parameters were chosen to fit exactly that data. This is
what makes the backtest "truthful" rather than curve-fit.

Grid search within a window is parallelized across processes: each worker loads
the training slice once (via the pool initializer) and then evaluates many
parameter combos against it, since serializing the whole DataFrame per task
would dominate runtime otherwise.
"""

from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import pandas as pd

from scalper.backtest.costs import CostModel
from scalper.backtest.engine import Trade, run_backtest
from scalper.backtest.metrics import Metrics, compute_metrics
from scalper.strategies.base import Strategy

_worker_df: pd.DataFrame | None = None
_worker_cost_model: CostModel | None = None
_worker_instrument: str | None = None


def _init_worker(df: pd.DataFrame, cost_model: CostModel, instrument: str) -> None:
    global _worker_df, _worker_cost_model, _worker_instrument
    _worker_df = df
    _worker_cost_model = cost_model
    _worker_instrument = instrument


def _evaluate_combo(args: tuple) -> tuple[dict, Metrics]:
    strategy_cls, strategy_params, sl, tp, size = args
    strategy: Strategy = strategy_cls()
    signals = strategy.signals(_worker_df, strategy_params)
    result = run_backtest(
        _worker_df, signals, _worker_instrument,
        stop_loss_pips=sl, take_profit_pips=tp, position_size_units=size,
        cost_model=_worker_cost_model,
    )
    metrics = compute_metrics(result)
    params = {
        **strategy_params,
        "stop_loss_pips": sl,
        "take_profit_pips": tp,
        "position_size_units": size,
    }
    return params, metrics


def _param_combinations(grid: dict[str, list]) -> list[dict]:
    keys = list(grid.keys())
    return [dict(zip(keys, values)) for values in itertools.product(*[grid[k] for k in keys])]


def walk_forward_windows(
    df: pd.DataFrame, train_months: int, validate_months: int
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    times = pd.to_datetime(df["time"])
    start, end = times.iloc[0], times.iloc[-1]

    windows = []
    train_start = start
    while True:
        train_end = train_start + pd.DateOffset(months=train_months)
        validate_end = train_end + pd.DateOffset(months=validate_months)
        if validate_end > end:
            break
        windows.append((train_start, train_end, train_end, validate_end))
        train_start = train_start + pd.DateOffset(months=validate_months)
    return windows


def _slice_window(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    times = pd.to_datetime(df["time"])
    mask = (times >= start) & (times < end)
    return df.loc[mask].reset_index(drop=True)


def grid_search_train(
    df_train: pd.DataFrame,
    strategy: Strategy,
    risk_grid: dict[str, list],
    instrument: str,
    cost_model: CostModel,
    objective: str = "total_pnl",
    max_workers: int | None = None,
) -> list[tuple[dict, Metrics]]:
    """Evaluates every (strategy params x risk params) combo on df_train,
    returns all (params, metrics) pairs sorted best-first by `objective`."""
    strategy_combos = _param_combinations(strategy.param_grid)
    risk_combos = _param_combinations(risk_grid)

    tasks = [
        (type(strategy), sp, rp["stop_loss_pips"], rp["take_profit_pips"], rp["position_size_units"])
        for sp in strategy_combos
        for rp in risk_combos
    ]

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(df_train, cost_model, instrument),
    ) as pool:
        results = list(pool.map(_evaluate_combo, tasks, chunksize=max(1, len(tasks) // 32)))

    results.sort(key=lambda r: getattr(r[1], objective), reverse=True)
    return results


@dataclass
class WindowResult:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validate_start: pd.Timestamp
    validate_end: pd.Timestamp
    best_params: dict
    train_metrics: Metrics
    validate_metrics: Metrics  # out-of-sample -- the number that matters
    validate_trades: list[Trade]


def run_walk_forward(
    df: pd.DataFrame,
    strategy: Strategy,
    instrument: str,
    risk_grid: dict[str, list],
    cost_model: CostModel,
    train_months: int,
    validate_months: int,
    objective: str = "total_pnl",
    max_workers: int | None = None,
    on_window_done=None,
) -> list[WindowResult]:
    """`on_window_done`, if given, is called as on_window_done(window_index,
    total_windows, WindowResult) right after each window finishes -- lets a
    caller show live progress / persist results incrementally instead of
    waiting for the entire (potentially long) walk-forward run to complete."""
    windows = walk_forward_windows(df, train_months, validate_months)
    results: list[WindowResult] = []

    for window_index, (train_start, train_end, validate_start, validate_end) in enumerate(windows, start=1):
        df_train = _slice_window(df, train_start, train_end)
        df_validate = _slice_window(df, validate_start, validate_end)
        if df_train.empty or df_validate.empty:
            continue

        ranked = grid_search_train(
            df_train, strategy, risk_grid, instrument, cost_model,
            objective=objective, max_workers=max_workers,
        )
        if not ranked:
            continue
        best_params, train_metrics = ranked[0]

        signals = strategy.signals(df_validate, best_params)
        validate_result = run_backtest(
            df_validate, signals, instrument,
            stop_loss_pips=best_params["stop_loss_pips"],
            take_profit_pips=best_params["take_profit_pips"],
            position_size_units=best_params["position_size_units"],
            cost_model=cost_model,
        )
        validate_metrics = compute_metrics(validate_result)

        window_result = WindowResult(
            train_start=train_start,
            train_end=train_end,
            validate_start=validate_start,
            validate_end=validate_end,
            best_params=best_params,
            train_metrics=train_metrics,
            validate_metrics=validate_metrics,
            validate_trades=validate_result.trades,
        )
        results.append(window_result)
        if on_window_done is not None:
            on_window_done(window_index, len(windows), window_result)

    return results
