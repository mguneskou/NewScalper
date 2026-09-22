"""Fast single-trial tool for manually fine-tuning OpeningRangeBreakoutStrategy.

Unlike run_backtest.py, this does NOT grid-search or walk-forward split --
it runs exactly one named configuration directly against a fixed window
(default: a locked 6-month tuning slice) and prints a rich diagnostic in
seconds, so a person can read the result, change one setting, and re-run.

This trades statistical rigor for iteration speed on purpose: the 6-month
window is small enough to be regime-specific, and repeatedly trying settings
against the same slice risks curve-fitting to it. Treat every number this
script prints as a *direction to try next*, not a result to report -- once
settings look good here, validate them with the real walk-forward optimizer
(run_backtest.py) over the full history before trusting them.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scalper.backtest.costs import CostModel, pip_size
from scalper.backtest.engine import run_backtest
from scalper.backtest.fx import FxConverter
from scalper.backtest.metrics import compute_metrics
from scalper.config import load_settings
from scalper.data.history import load_cached
from scalper.strategies.opening_range_breakout import OpeningRangeBreakoutStrategy, compute_box

# Locked tuning window: a middle slice of the cache, deliberately not the most
# recent data, so both earlier and later history stay untouched as a genuine
# held-out check once settings are chosen here.
DEFAULT_START = "2023-07-01"
DEFAULT_END = "2024-01-01"
STARTING_BALANCE = 10_000.0


def box_relative_pips(df: pd.DataFrame, instrument: str, session_open_hour: int, box_minutes: int,
                       sl_frac: float, tp_frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Stop/take-profit sized as a fraction of *that day's own* box height,
    instead of a fixed pip count or ATR -- the box already measures the local
    range at the moment of the trade, so this may fit ORB more naturally than
    either alternative tried on other strategies."""
    box_high, box_low = compute_box(df, session_open_hour, box_minutes)
    box_height_pips = (box_high - box_low) / pip_size(instrument)
    # Never actually used on a NaN bar (the strategy is flat whenever the box
    # itself is NaN, so no trade can enter there) -- bfill purely for hygiene.
    box_height_pips = box_height_pips.bfill().fillna(0.0)
    return (box_height_pips * sl_frac).to_numpy(), (box_height_pips * tp_frac).to_numpy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", default="EUR_USD")
    parser.add_argument("--granularity", default="M5")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--session-open-hour", type=int, default=8)
    parser.add_argument("--box-minutes", type=int, default=30)
    parser.add_argument("--exit-buffer-frac", type=float, default=0.0,
                         help="push the box_mid exit trigger this fraction of the box height further in")
    parser.add_argument("--min-box-height-pips", type=float, default=0.0,
                         help="skip days whose box height is below this many pips")
    parser.add_argument("--risk-mode", choices=["fixed", "box"], default="fixed")
    parser.add_argument("--sl", type=float, required=True,
                         help="fixed mode: stop-loss pips. box mode: fraction of box height.")
    parser.add_argument("--tp", type=float, required=True,
                         help="fixed mode: take-profit pips. box mode: fraction of box height.")
    parser.add_argument("--position-size", type=int, default=1000)
    args = parser.parse_args()

    settings = load_settings()
    account_currency = settings["account_currency"]
    bt_cfg = settings["backtest"]
    cost_model = CostModel(slippage_pips=bt_cfg["slippage_pips"], commission_per_trade=bt_cfg["commission_per_trade"])

    df = load_cached(args.instrument, args.granularity)
    times = pd.to_datetime(df["time"])
    df = df.loc[(times >= args.start) & (times < args.end)].reset_index(drop=True)
    print(f"{args.instrument} {args.granularity} [{args.start} -> {args.end}] ({len(df)} bars)")

    strategy_params = {
        "session_open_hour": args.session_open_hour,
        "box_minutes": args.box_minutes,
        "exit_buffer_frac": args.exit_buffer_frac,
        "min_box_height_pips": args.min_box_height_pips,
        "instrument": args.instrument,
    }
    signals = OpeningRangeBreakoutStrategy().signals(df, strategy_params)

    if args.risk_mode == "fixed":
        sl_pips, tp_pips = args.sl, args.tp
        risk_desc = f"fixed sl={args.sl}pips tp={args.tp}pips"
    else:
        sl_pips, tp_pips = box_relative_pips(
            df, args.instrument, args.session_open_hour, args.box_minutes, args.sl, args.tp
        )
        risk_desc = f"box-relative sl={args.sl}x tp={args.tp}x box-height"

    fx = FxConverter(account_currency, args.granularity)
    result = run_backtest(
        df, signals, args.instrument,
        stop_loss_pips=sl_pips, take_profit_pips=tp_pips,
        position_size_units=args.position_size, cost_model=cost_model,
        starting_balance=STARTING_BALANCE, account_currency=account_currency, fx=fx,
    )
    m = compute_metrics(result)

    print(
        f"session_open_hour={args.session_open_hour} box_minutes={args.box_minutes} "
        f"exit_buffer_frac={args.exit_buffer_frac} min_box_height_pips={args.min_box_height_pips} {risk_desc}"
    )
    print(
        f"trades={m.trade_count} win_rate={m.win_rate:.1%} profit_factor={m.profit_factor:.2f} "
        f"avg_r_multiple={m.avg_r_multiple:.3f} sharpe={m.sharpe_ratio:.2f} "
        f"total_pnl={m.total_pnl:.2f}{account_currency} max_dd={m.max_drawdown:.2f}"
    )

    if result.trades:
        by_reason: dict[str, list[float]] = {}
        for t in result.trades:
            by_reason.setdefault(t.exit_reason, []).append(t.pnl)
        print("exit reasons:")
        for reason, pnls in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
            arr = np.array(pnls)
            print(f"  {reason:12s} count={len(arr):5d} avg_pnl={arr.mean():8.3f} total_pnl={arr.sum():9.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
