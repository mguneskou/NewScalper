from __future__ import annotations

import numpy as np
from numba import njit
import pandas as pd

from scalper.backtest.costs import pip_size
from scalper.strategies.base import Strategy

# The session-open hour is interpreted in this timezone (matches
# config/settings.yaml's live.day_boundary_timezone convention), so the box
# lines up with a trader's local wall-clock session open year-round, DST included.
SESSION_TIMEZONE = "Europe/London"


@njit(cache=True)
def _orb_positions(
    price: np.ndarray, box_high: np.ndarray, box_low: np.ndarray,
    exit_long_level: np.ndarray, exit_short_level: np.ndarray,
) -> np.ndarray:
    n = price.shape[0]
    out = np.empty(n, dtype=np.int64)
    position = 0
    for i in range(n):
        if np.isnan(box_high[i]) or np.isnan(box_low[i]):
            # No box yet for today (still forming, or before the session opens).
            # Resetting position (not just the output) here is what guarantees a
            # position never carries from one day into the next -- without it, a
            # still-open trade would resume using yesterday's exit level as its
            # exit once today's box became NaN-free again, instead of taking a
            # fresh breakout on today's own range.
            position = 0
            out[i] = 0
            continue
        if position == 0:
            if price[i] > box_high[i]:
                position = 1
            elif price[i] < box_low[i]:
                position = -1
        elif position == 1 and price[i] <= exit_long_level[i]:
            position = 0
        elif position == -1 and price[i] >= exit_short_level[i]:
            position = 0
        out[i] = position
    return out


def compute_box(df: pd.DataFrame, session_open_hour: int, box_minutes: int) -> tuple[pd.Series, pd.Series]:
    """Returns (box_high, box_low), aligned with df: NaN before that day's box
    is known (still forming, or before the session opens), the day's fixed
    bounds for every bar afterward.

    Exposed as a module-level function (not inlined in `signals`) so a caller
    tuning the strategy can also derive a risk distance from the same box --
    e.g. a stop-loss sized as a fraction of that day's own box height --
    without recomputing the box independently and risking it drift out of
    sync with what `signals` actually used.
    """
    local_time = pd.to_datetime(df["time"]).dt.tz_convert(SESSION_TIMEZONE)
    local_date = local_time.dt.date
    minutes_since_midnight = local_time.dt.hour * 60 + local_time.dt.minute

    box_start_min = session_open_hour * 60
    box_end_min = box_start_min + box_minutes
    in_box_window = (minutes_since_midnight >= box_start_min) & (minutes_since_midnight < box_end_min)
    past_box_window = minutes_since_midnight >= box_end_min

    mid_h = (df["bid_h"] + df["ask_h"]) / 2.0
    mid_l = (df["bid_l"] + df["ask_l"]) / 2.0

    # Each day's box is computed only from that day's own box-window bars
    # (aggregation skips NaN), then broadcast to every bar of that day;
    # masking to `past_box_window` afterwards is what keeps it from being used
    # before it's actually known (no look-ahead) and forces every pre-box/
    # forming bar to NaN.
    day_box_high = mid_h.where(in_box_window).groupby(local_date).transform("max")
    day_box_low = mid_l.where(in_box_window).groupby(local_date).transform("min")
    box_high = day_box_high.where(past_box_window)
    box_low = day_box_low.where(past_box_window)
    return box_high, box_low


class OpeningRangeBreakoutStrategy(Strategy):
    """Marks a box from the high/low of the first `box_minutes` after
    `session_open_hour` each day, then goes long above the box / short below
    it, flat once price reverts to the box's midpoint -- or once the next
    day's own box starts forming, whichever comes first (no overnight carry).
    """

    name = "opening_range_breakout"
    param_grid = {
        # 8 ~= London session open, 13 ~= New York session open, both in the
        # Europe/London local time used to build the box.
        "session_open_hour": [8, 13],
        # Locked at 30 (not searched): consistently the best of [15, 30, 60] across
        # every manual tuning pass so far, by a clearer margin than session_open_hour.
        "box_minutes": [30],
        # 0.0 = exit exactly at the box midpoint (original behavior). A positive
        # value pushes the exit trigger further past the midpoint, deeper into
        # the box (as a fraction of box height) -- requires more than a bare
        # touch before giving up, to cut down on whipsaw exits right at the line.
        # 0.4 was the best single value found manually; the neighbors are here so
        # the real walk-forward can pick per-instrument/per-window rather than
        # trusting one manually-tuned window's exact peak.
        "exit_buffer_frac": [0.0, 0.25, 0.4],
        # 0.0 = trade every day's box regardless of size. A positive value skips
        # days whose box is narrower than this many pips -- a very small opening
        # range is more likely noise than a meaningful level, and a "breakout" of
        # it correspondingly less likely to mean anything. 8 pips was the best
        # found manually (roughly EUR_USD's 25th-percentile box height); results
        # were bumpy enough between 7-10 pips that this is a rough answer, not
        # a precisely-located optimum.
        "min_box_height_pips": [0.0, 8.0],
    }

    def signals(self, df: pd.DataFrame, params: dict) -> pd.Series:
        box_high, box_low = compute_box(df, params["session_open_hour"], params["box_minutes"])
        exit_buffer_frac = params.get("exit_buffer_frac", 0.0)
        min_box_height_pips = params.get("min_box_height_pips", 0.0)

        if min_box_height_pips > 0:
            instrument = params.get("instrument")
            if instrument is None:
                raise ValueError("min_box_height_pips > 0 requires 'instrument' in params")
            box_height_pips = (box_high - box_low) / pip_size(instrument)
            too_small = box_height_pips < min_box_height_pips
            box_high = box_high.where(~too_small)
            box_low = box_low.where(~too_small)

        box_mid = (box_high + box_low) / 2.0
        box_height = box_high - box_low
        exit_long_level = box_mid - exit_buffer_frac * box_height
        exit_short_level = box_mid + exit_buffer_frac * box_height
        price = self.mid_close(df)

        sig_vals = _orb_positions(
            price.to_numpy(dtype=np.float64),
            box_high.to_numpy(dtype=np.float64),
            box_low.to_numpy(dtype=np.float64),
            exit_long_level.to_numpy(dtype=np.float64),
            exit_short_level.to_numpy(dtype=np.float64),
        )
        return pd.Series(sig_vals, index=df.index)
