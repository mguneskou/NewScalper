"""Converts backtest P&L from an instrument's quote currency into the trading
account's currency, using contemporaneous FX rates read from the same cached
candle data the backtest already trusts as its source of truth.

Without this, P&L from different instruments cannot be summed, ranked, or
compared: a EUR_USD trade's raw P&L is denominated in USD, a USD_JPY trade's
in JPY, an EUR_GBP trade's in GBP -- mixing those (as the backtest used to)
makes a JPY-quoted result look ~100x larger than an equivalent USD one purely
from unit mismatch, not real performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scalper.data.history import load_cached


def quote_currency(instrument: str) -> str:
    return instrument.split("_")[1]


class FxConverter:
    """Looks up conversion rates on demand from cached candle data at a given
    granularity, via a direct pair, its inverse, or a two-hop path through USD
    (the currency virtually every FX pair is quoted against or from)."""

    def __init__(self, account_currency: str, granularity: str):
        self.account_currency = account_currency
        self.granularity = granularity
        self._mid_cache: dict[str, pd.Series | None] = {}

    def _mid_series(self, instrument: str) -> pd.Series | None:
        if instrument not in self._mid_cache:
            df = load_cached(instrument, self.granularity)
            if df.empty:
                self._mid_cache[instrument] = None
            else:
                times = pd.to_datetime(df["time"]).to_numpy()
                mid = ((df["bid_c"] + df["ask_c"]) / 2.0).to_numpy()
                series = pd.Series(mid, index=pd.DatetimeIndex(times)).sort_index()
                # Defensive: history.backfill() used to leave occasional exact-
                # duplicate timestamps in the cache from a pagination artifact
                # (fixed in OandaClient.candles_range via includeFirst=false).
                # Kept here since reindex() just needs a unique index anyway.
                self._mid_cache[instrument] = series[~series.index.duplicated(keep="last")]
        return self._mid_cache[instrument]

    @staticmethod
    def _nearest(series: pd.Series, times: np.ndarray) -> np.ndarray:
        target = pd.DatetimeIndex(times)
        # Cached candle timestamps are tz-aware (UTC); normalize so a caller
        # passing naive timestamps (or a different tz) still matches correctly.
        if series.index.tz is not None and target.tz is None:
            target = target.tz_localize("UTC")
        elif series.index.tz is None and target.tz is not None:
            target = target.tz_convert("UTC").tz_localize(None)
        return series.reindex(target, method="nearest").to_numpy()

    def _to_usd_rate(self, currency: str, times: np.ndarray) -> np.ndarray:
        """How many USD 1 unit of `currency` is worth, at each of `times`."""
        if currency == "USD":
            return np.ones(len(times))
        direct = self._mid_series(f"{currency}_USD")
        if direct is not None:
            return self._nearest(direct, times)
        inverse = self._mid_series(f"USD_{currency}")
        if inverse is not None:
            return 1.0 / self._nearest(inverse, times)
        raise ValueError(
            f"No cached {self.granularity} candles found to convert {currency} to USD "
            f"(looked for {currency}_USD and USD_{currency})"
        )

    def rate_to_account_currency(self, currency: str, times: np.ndarray) -> np.ndarray:
        """How many units of the account currency 1 unit of `currency` is worth,
        at each of `times`. `times` is a datetime64 array, typically trade exit
        times -- P&L is converted at the rate when it was realized, same as a
        broker would mark it to the account currency."""
        if currency == self.account_currency:
            return np.ones(len(times))
        usd_rate = self._to_usd_rate(currency, times)
        if self.account_currency == "USD":
            return usd_rate
        account_to_usd = self._to_usd_rate(self.account_currency, times)
        return usd_rate / account_to_usd
