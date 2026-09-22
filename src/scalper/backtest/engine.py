"""Event-driven, single-instrument, single-position backtest engine.

The per-bar simulation loop is JIT-compiled with numba: a plain Python loop over
5 years of M1 data (~1.8M bars) repeated across a walk-forward x grid-search
optimization would otherwise take hours. Numba gets a single run down to
milliseconds, which is what makes exhaustive walk-forward validation practical
rather than something we'd be tempted to skip for speed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit

from scalper.backtest.costs import CostModel, pip_size
from scalper.backtest.fx import FxConverter, quote_currency

EXIT_REASONS = {0: "signal", 1: "stop_loss", 2: "take_profit", 3: "end_of_data"}


@dataclass
class Trade:
    instrument: str
    direction: int  # 1 long, -1 short
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    units: int
    pnl: float  # in account_currency -- the figure to use for any cross-instrument comparison
    pnl_quote_ccy: float  # raw P&L in the instrument's own quote currency, kept for auditability
    risk: float  # amount risked at entry (stop distance x units), in account_currency
    exit_reason: str

    @property
    def r_multiple(self) -> float:
        """P&L expressed as a multiple of what the trade risked at entry --
        lets trades with different stop distances (fixed-pip or ATR-relative)
        be compared and averaged on equal footing, unlike raw P&L, which a
        fixed position size lets a tight-stop/high-frequency combo dominate
        purely through volume rather than genuine edge."""
        return self.pnl / self.risk if self.risk > 0 else 0.0


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: pd.Series  # balance after each closed trade, indexed by exit time
    starting_balance: float


@njit(cache=True)
def _simulate(
    bid_o, bid_h, bid_l, ask_o, ask_h, ask_l,
    sig,
    sl_dist_arr, tp_dist_arr,
    position_size_units,
    slippage_price,
    commission_per_trade,
):
    n = bid_o.shape[0]
    out_entry_idx = np.empty(n, dtype=np.int64)
    out_exit_idx = np.empty(n, dtype=np.int64)
    out_direction = np.empty(n, dtype=np.int64)
    out_entry_price = np.empty(n, dtype=np.float64)
    out_exit_price = np.empty(n, dtype=np.float64)
    out_pnl = np.empty(n, dtype=np.float64)
    out_risk = np.empty(n, dtype=np.float64)
    out_reason = np.empty(n, dtype=np.int64)
    trade_count = 0

    position_dir = 0
    entry_idx = -1
    entry_price = 0.0
    stop_price = 0.0
    tp_price = 0.0
    entry_risk = 0.0

    for i in range(n):
        if position_dir != 0:
            exit_price = np.nan
            exit_reason = -1

            if position_dir == 1:
                if bid_l[i] <= stop_price:
                    exit_price = stop_price
                    exit_reason = 1
                elif bid_h[i] >= tp_price:
                    exit_price = tp_price
                    exit_reason = 2
            else:
                if ask_h[i] >= stop_price:
                    exit_price = stop_price
                    exit_reason = 1
                elif ask_l[i] <= tp_price:
                    exit_price = tp_price
                    exit_reason = 2

            if exit_reason == -1 and sig[i] != position_dir:
                if position_dir == 1:
                    exit_price = bid_o[i] - slippage_price
                else:
                    exit_price = ask_o[i] + slippage_price
                exit_reason = 0

            if exit_reason != -1:
                if position_dir == 1:
                    pnl = (exit_price - entry_price) * position_size_units - commission_per_trade
                else:
                    pnl = (entry_price - exit_price) * position_size_units - commission_per_trade

                out_entry_idx[trade_count] = entry_idx
                out_exit_idx[trade_count] = i
                out_direction[trade_count] = position_dir
                out_entry_price[trade_count] = entry_price
                out_exit_price[trade_count] = exit_price
                out_pnl[trade_count] = pnl
                out_risk[trade_count] = entry_risk
                out_reason[trade_count] = exit_reason
                trade_count += 1
                position_dir = 0

        if position_dir == 0 and sig[i] != 0:
            position_dir = sig[i]
            entry_idx = i
            # SL/TP distance is locked in at entry from that bar's value -- for an
            # ATR-relative stop this means "sized to volatility when the trade was
            # opened", not continuously re-sized while the position is held.
            entry_risk = sl_dist_arr[i] * position_size_units
            if position_dir == 1:
                entry_price = ask_o[i] + slippage_price
                stop_price = entry_price - sl_dist_arr[i]
                tp_price = entry_price + tp_dist_arr[i]
            else:
                entry_price = bid_o[i] - slippage_price
                stop_price = entry_price + sl_dist_arr[i]
                tp_price = entry_price - tp_dist_arr[i]

    # force-close any still-open position at the final bar's price
    if position_dir != 0:
        i = n - 1
        if position_dir == 1:
            exit_price = bid_o[i] - slippage_price
            pnl = (exit_price - entry_price) * position_size_units - commission_per_trade
        else:
            exit_price = ask_o[i] + slippage_price
            pnl = (entry_price - exit_price) * position_size_units - commission_per_trade
        out_entry_idx[trade_count] = entry_idx
        out_exit_idx[trade_count] = i
        out_direction[trade_count] = position_dir
        out_entry_price[trade_count] = entry_price
        out_exit_price[trade_count] = exit_price
        out_pnl[trade_count] = pnl
        out_risk[trade_count] = entry_risk
        out_reason[trade_count] = 3
        trade_count += 1

    return (
        out_entry_idx[:trade_count],
        out_exit_idx[:trade_count],
        out_direction[:trade_count],
        out_entry_price[:trade_count],
        out_exit_price[:trade_count],
        out_pnl[:trade_count],
        out_risk[:trade_count],
        out_reason[:trade_count],
    )


def _as_pip_array(value: float | np.ndarray | pd.Series, n: int) -> np.ndarray:
    """Broadcasts a fixed pip distance to every bar, or passes through a
    per-bar distance (e.g. ATR-relative) already computed by the caller."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return np.full(n, float(value))
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape[0] != n:
        raise ValueError(f"per-bar stop/take-profit array must have length {n}, got {arr.shape[0]}")
    return arr


def run_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    instrument: str,
    stop_loss_pips: float | np.ndarray | pd.Series,
    take_profit_pips: float | np.ndarray | pd.Series,
    position_size_units: int,
    cost_model: CostModel,
    starting_balance: float = 10_000.0,
    account_currency: str = "USD",
    fx: FxConverter | None = None,
) -> BacktestResult:
    """Runs the simulation and returns realized trades + equity curve.

    IMPORTANT (look-ahead bias): `signals` must represent the desired position
    computed from each bar's *close*. This function shifts it forward by one bar
    internally, so a signal known at bar i's close is only acted on starting at
    bar i+1's open -- exactly what a real strategy could have done live. Do not
    pre-shift signals yourself.

    `stop_loss_pips`/`take_profit_pips` are each either a single fixed distance
    (the whole backtest uses the same SL/TP) or a per-bar array/Series aligned
    with `df` (e.g. an ATR-relative distance that varies with volatility) --
    whichever a trade's entry bar holds is what gets locked in for that trade.

    P&L is computed in `instrument`'s own quote currency, then converted to
    `account_currency` (via `fx`, required whenever the two differ) at the rate
    prevailing at each trade's exit time -- the same way a broker marks a closed
    trade's P&L to the account currency. This is what makes `total_pnl` safe to
    sum or compare across instruments with different quote currencies.
    """
    pip = pip_size(instrument)
    sl_dist_arr = _as_pip_array(stop_loss_pips, len(df)) * pip
    tp_dist_arr = _as_pip_array(take_profit_pips, len(df)) * pip
    slippage_price = cost_model.slippage_pips * pip

    effective_signal = signals.shift(1, fill_value=0).to_numpy().astype(np.int64)

    times = pd.to_datetime(df["time"]).to_numpy()
    bid_o = df["bid_o"].to_numpy(dtype=np.float64)
    bid_h = df["bid_h"].to_numpy(dtype=np.float64)
    bid_l = df["bid_l"].to_numpy(dtype=np.float64)
    ask_o = df["ask_o"].to_numpy(dtype=np.float64)
    ask_h = df["ask_h"].to_numpy(dtype=np.float64)
    ask_l = df["ask_l"].to_numpy(dtype=np.float64)

    (entry_idx, exit_idx, direction, entry_price, exit_price, pnl, risk, reason) = _simulate(
        bid_o, bid_h, bid_l, ask_o, ask_h, ask_l,
        effective_signal,
        sl_dist_arr, tp_dist_arr,
        float(position_size_units),
        slippage_price,
        cost_model.commission_per_trade,
    )

    quote_ccy = quote_currency(instrument)
    exit_times = times[exit_idx]
    if quote_ccy == account_currency:
        conversion_rate = np.ones(len(exit_idx))
    elif fx is not None:
        conversion_rate = fx.rate_to_account_currency(quote_ccy, exit_times)
    else:
        raise ValueError(
            f"{instrument} is quoted in {quote_ccy}, which differs from "
            f"account_currency={account_currency!r} -- pass an FxConverter via `fx`."
        )
    pnl_account_ccy = pnl * conversion_rate
    # Risk was locked in at entry, but it's converted at the same exit-time rate as
    # P&L (rather than the entry-time rate) purely for simplicity -- FX drift over a
    # single trade's short holding period is negligible next to what it's measuring.
    risk_account_ccy = risk * conversion_rate

    trades: list[Trade] = []
    balance = starting_balance
    equity_times = []
    equity_values = []
    for k in range(len(entry_idx)):
        balance += pnl_account_ccy[k]
        trades.append(
            Trade(
                instrument=instrument,
                direction=int(direction[k]),
                entry_time=pd.Timestamp(times[entry_idx[k]]),
                entry_price=float(entry_price[k]),
                exit_time=pd.Timestamp(times[exit_idx[k]]),
                exit_price=float(exit_price[k]),
                units=position_size_units,
                pnl=float(pnl_account_ccy[k]),
                pnl_quote_ccy=float(pnl[k]),
                risk=float(risk_account_ccy[k]),
                exit_reason=EXIT_REASONS[int(reason[k])],
            )
        )
        equity_times.append(times[exit_idx[k]])
        equity_values.append(balance)

    equity_curve = (
        pd.Series(equity_values, index=pd.to_datetime(equity_times))
        if equity_values
        else pd.Series(dtype=float)
    )
    return BacktestResult(trades=trades, equity_curve=equity_curve, starting_balance=starting_balance)


def equity_curve_from_trades(trades: list[Trade], starting_balance: float) -> pd.Series:
    """Rebuilds a running-balance equity curve from an already-assembled trade list
    (e.g. out-of-sample trades stitched together across walk-forward windows)."""
    if not trades:
        return pd.Series(dtype=float)
    balance = starting_balance
    values = []
    times = []
    for t in trades:
        balance += t.pnl
        values.append(balance)
        times.append(t.exit_time)
    return pd.Series(values, index=pd.to_datetime(times))
