import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scalper.backtest.costs import CostModel
from scalper.backtest.optimizer import run_walk_forward, walk_forward_windows
from scalper.strategies.base import Strategy


class _TinyTestStrategy(Strategy):
    """A minimal EMA-cross-style strategy with a 1-combo grid, used only to
    exercise the generic walk-forward machinery cheaply -- not tied to any
    real trading strategy."""

    name = "tiny_test_strategy"
    param_grid = {"fast_period": [3], "slow_period": [10]}

    def signals(self, df, params):
        price = self.mid_close(df)
        fast = price.ewm(span=params["fast_period"], adjust=False).mean()
        slow = price.ewm(span=params["slow_period"], adjust=False).mean()
        sig = pd.Series(0, index=df.index)
        sig[fast > slow] = 1
        sig[fast < slow] = -1
        return sig


def make_synthetic_df(n_days: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = n_days * 24 * 60  # 1-minute bars
    times = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    walk = np.cumsum(rng.normal(0, 0.00005, n)) + 1.10
    spread = 0.00015
    return pd.DataFrame(
        {
            "time": times.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
            "bid_o": walk,
            "bid_h": walk + 0.00002,
            "bid_l": walk - 0.00002,
            "bid_c": walk,
            "ask_o": walk + spread,
            "ask_h": walk + spread + 0.00002,
            "ask_l": walk + spread - 0.00002,
            "ask_c": walk + spread,
            "volume": 10,
            "complete": True,
        }
    )


def test_walk_forward_windows_cover_expected_span():
    df = make_synthetic_df(n_days=300)  # ~10 months
    windows = walk_forward_windows(df, train_months=3, validate_months=2)
    assert len(windows) >= 2
    for train_start, train_end, validate_start, validate_end in windows:
        assert train_start < train_end == validate_start < validate_end


def test_walk_forward_windows_clamps_validate_end_instead_of_dropping_window():
    """4mo train + 2mo validate = 6 calendar months, which is NOT a fixed
    number of days (28-31 days/month) -- a day-based span of "exactly" 6
    months worth of data can fall a few hours to a few days short of what
    DateOffset(months=6) expects. That used to make the entire window
    vanish; it should instead just produce one window with a slightly
    shorter validation slice."""
    df = make_synthetic_df(n_days=182)  # exactly 6 months' worth of days, no padding
    windows = walk_forward_windows(df, train_months=4, validate_months=2)
    assert len(windows) == 1
    train_start, train_end, validate_start, validate_end = windows[0]
    assert validate_start == train_end
    assert validate_end == pd.to_datetime(df["time"]).iloc[-1]


def test_walk_forward_windows_still_rejects_incomplete_training_window():
    df = make_synthetic_df(n_days=90)  # 3 months, less than the 4-month training window
    windows = walk_forward_windows(df, train_months=4, validate_months=2)
    assert windows == []


def test_run_walk_forward_produces_out_of_sample_results():
    df = make_synthetic_df(n_days=300)
    strategy = _TinyTestStrategy()
    risk_grid = {
        "stop_loss_pips": [5],
        "take_profit_pips": [8],
        "position_size_units": [1000],
    }
    cost_model = CostModel(slippage_pips=0.2, commission_per_trade=0.0)

    results = run_walk_forward(
        df, strategy, "EUR_USD", risk_grid, cost_model,
        train_months=3, validate_months=2, max_workers=2,
    )

    assert len(results) >= 2
    for r in results:
        assert r.best_params["fast_period"] == 3
        assert r.validate_start == r.train_end
        # metrics objects should always be populated, even if trade_count is 0
        assert r.validate_metrics.trade_count >= 0


def test_run_walk_forward_param_grid_override_replaces_strategy_default():
    """An instrument-specific param_grid (e.g. a narrower min_box_height_pips
    for EUR_GBP) must actually be searched instead of the strategy's own
    param_grid -- this is what lets different instruments get different
    search spaces, not just different winners from one shared grid."""
    df = make_synthetic_df(n_days=300)
    strategy = _TinyTestStrategy()
    risk_grid = {"stop_loss_pips": [5], "take_profit_pips": [8], "position_size_units": [1000]}
    cost_model = CostModel(slippage_pips=0.2, commission_per_trade=0.0)

    results = run_walk_forward(
        df, strategy, "EUR_USD", risk_grid, cost_model,
        train_months=3, validate_months=2, max_workers=2,
        param_grid={"fast_period": [7], "slow_period": [10]},
    )

    assert len(results) >= 2
    for r in results:
        assert r.best_params["fast_period"] == 7  # not the strategy's default (3)
