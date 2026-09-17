"""CLI: backfills (or tops up) the 5-year M1+M5 candle cache for all configured instruments.

This makes many thousands of paged Oanda API requests and can take a while --
run it standalone, not from inside the dashboard.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scalper.config import load_credentials, load_settings
from scalper.data.history import backfill, load_cached
from scalper.oanda.client import OandaClient


def main() -> int:
    creds = load_credentials()
    settings = load_settings()
    instruments = settings["instruments"]
    granularities = settings["granularities"]
    years = settings["backtest"]["years"]

    def on_progress(instrument, granularity, count):
        print(f"  ... {instrument} {granularity}: {count} new candles fetched so far")

    with OandaClient(creds) as client:
        for instrument in instruments:
            for granularity in granularities:
                start = time.time()
                print(f"Backfilling {instrument} {granularity} ({years}y)...")
                df = backfill(
                    client,
                    instrument,
                    granularity,
                    years=years,
                    on_progress=on_progress,
                )
                elapsed = time.time() - start
                print(
                    f"  done: {len(df)} total candles cached "
                    f"({df['time'].iloc[0] if len(df) else 'n/a'} -> "
                    f"{df['time'].iloc[-1] if len(df) else 'n/a'}) in {elapsed:.1f}s"
                )

    print("\nCache summary:")
    for instrument in instruments:
        for granularity in granularities:
            df = load_cached(instrument, granularity)
            print(f"  {instrument} {granularity}: {len(df)} candles")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
