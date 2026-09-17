"""Backfills and incrementally updates a local Parquet cache of Oanda candles.

Oanda's candle API is the single source of truth for both the backtest and the
live engine, so caching its own history (rather than a third-party data vendor)
guarantees the backtest sees exactly the prices the live engine would have seen.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from scalper.oanda.client import OandaClient

CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache"

CANDLE_COLUMNS = [
    "time",
    "volume",
    "complete",
    "bid_o",
    "bid_h",
    "bid_l",
    "bid_c",
    "ask_o",
    "ask_h",
    "ask_l",
    "ask_c",
]


def cache_path(instrument: str, granularity: str) -> Path:
    return CACHE_DIR / f"{instrument}_{granularity}.parquet"


def _candle_to_row(candle: dict) -> dict:
    bid = candle["bid"]
    ask = candle["ask"]
    return {
        "time": candle["time"],
        "volume": candle["volume"],
        "complete": candle["complete"],
        "bid_o": float(bid["o"]),
        "bid_h": float(bid["h"]),
        "bid_l": float(bid["l"]),
        "bid_c": float(bid["c"]),
        "ask_o": float(ask["o"]),
        "ask_h": float(ask["h"]),
        "ask_l": float(ask["l"]),
        "ask_c": float(ask["c"]),
    }


def load_cached(instrument: str, granularity: str) -> pd.DataFrame:
    path = cache_path(instrument, granularity)
    if not path.exists():
        return pd.DataFrame(columns=CANDLE_COLUMNS)
    return pd.read_parquet(path)


def backfill(
    client: OandaClient,
    instrument: str,
    granularity: str,
    years: int = 5,
    request_pause_seconds: float = 0.1,
    progress_every: int = 20,
    on_progress=None,
) -> pd.DataFrame:
    """Fetches (or resumes fetching) candle history and writes it to the Parquet cache.

    If a cache already exists, only fetches candles after the last cached one,
    so re-running this is cheap and safe (used both for the initial 5y backfill
    and later top-ups).
    """
    existing = load_cached(instrument, granularity)

    now = datetime.now(timezone.utc)
    default_from = now - timedelta(days=365 * years)

    if not existing.empty:
        last_time = pd.to_datetime(existing["time"].iloc[-1])
        from_time = max(last_time.to_pydatetime(), default_from.replace(tzinfo=last_time.tzinfo))
    else:
        from_time = default_from

    from_iso = from_time.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
    to_iso = now.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")

    new_rows: list[dict] = []
    request_count = 0
    for candle in client.candles_range(instrument, granularity, from_iso, to_iso):
        if not candle["complete"]:
            continue
        new_rows.append(_candle_to_row(candle))
        request_count += 1
        if request_count % 5000 == 0:
            time.sleep(request_pause_seconds)
            if on_progress and (request_count // 5000) % progress_every == 0:
                on_progress(instrument, granularity, len(new_rows))

    if not new_rows:
        return existing

    new_df = pd.DataFrame(new_rows, columns=CANDLE_COLUMNS)
    if not existing.empty:
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset="time", keep="last").sort_values("time")
    else:
        combined = new_df.sort_values("time")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(cache_path(instrument, granularity), index=False)
    return combined
