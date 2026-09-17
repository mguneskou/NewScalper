"""Verifies Oanda practice-account credentials work. Prints only non-secret info."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scalper.config import load_credentials
from scalper.oanda.client import OandaClient, OandaError


def main() -> int:
    try:
        creds = load_credentials()
    except RuntimeError as e:
        print(f"Config error: {e}")
        return 1

    try:
        with OandaClient(creds) as client:
            account = client.account_summary()
            prices = client.current_prices(["EUR_USD"])
    except OandaError as e:
        print(f"Oanda API error: {e}")
        return 1

    print("Connected to Oanda practice account successfully.")
    print(f"  Account ID:       {account['id']}")
    print(f"  Currency:         {account['currency']}")
    print(f"  Balance:          {account['balance']}")
    print(f"  Open trade count: {account['openTradeCount']}")
    if prices:
        p = prices[0]
        print(f"  EUR_USD bid/ask:  {p['bids'][0]['price']} / {p['asks'][0]['price']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
