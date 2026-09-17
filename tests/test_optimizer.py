import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scalper.backtest.costs import CostModel
from scalper.backtest.optimizer import run_walk_forward, walk_forward_windows
from scalper.strategies.base import Strategy


class _TinyEmaCross(Strategy):
    """Same logic as EmaCrossStrategy but a 1-combo grid, to keep the test fast."""

    name = "tiny_ema_cross"
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


def test_run_walk_forward_produces_out_of_sample_results():
    df = make_synthetic_df(n_days=300)
    strategy = _TinyEmaCross()
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
