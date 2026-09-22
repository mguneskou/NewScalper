"""Thin wrapper around the Oanda v20 REST API (practice environment only).

Deliberately hand-rolled rather than using the unmaintained oandapyV20 package --
this keeps us in full control of pagination, error handling, and rate limiting.
"""

from __future__ import annotations

from typing import Any, Iterator

import httpx

from scalper.config import OandaCredentials

MAX_CANDLES_PER_REQUEST = 5000


class OandaError(RuntimeError):
    def __init__(self, status_code: int, body: Any):
        super().__init__(f"Oanda API error {status_code}: {body}")
        self.status_code = status_code
        self.body = body


class OandaClient:
    def __init__(self, credentials: OandaCredentials, timeout: float = 10.0):
        self._creds = credentials
        self._client = httpx.Client(
            base_url=credentials.rest_host,
            headers={
                "Authorization": f"Bearer {credentials.api_token}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OandaClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _request(self, method: str, path: str, **kwargs) -> dict:
        response = self._client.request(method, path, **kwargs)
        if response.status_code >= 400:
            try:
                body = response.json()
            except ValueError:
                body = response.text
            raise OandaError(response.status_code, body)
        return response.json()

    # -- Account -----------------------------------------------------------

    def account_summary(self) -> dict:
        data = self._request("GET", f"/v3/accounts/{self._creds.account_id}/summary")
        return data["account"]

    def open_positions(self) -> list[dict]:
        data = self._request("GET", f"/v3/accounts/{self._creds.account_id}/openPositions")
        return data["positions"]

    def open_trades(self) -> list[dict]:
        data = self._request("GET", f"/v3/accounts/{self._creds.account_id}/openTrades")
        return data["trades"]

    def transactions_since(self, from_time_iso: str) -> list[dict]:
        """Transactions (includes ORDER_FILL / trade closes) since an RFC3339 timestamp."""
        data = self._request(
            "GET",
            f"/v3/accounts/{self._creds.account_id}/transactions",
            params={"from": from_time_iso, "type": "ORDER_FILL"},
        )
        pages: list[dict] = []
        for page_url in data.get("pages", []):
            path = page_url.split(self._creds.rest_host, 1)[-1]
            page = self._request("GET", path)
            pages.extend(page.get("transactions", []))
        if not data.get("pages"):
            pages.extend(data.get("transactions", []))
        return pages

    # -- Pricing -------------------------------------------------------------

    def current_prices(self, instruments: list[str]) -> list[dict]:
        data = self._request(
            "GET",
            f"/v3/accounts/{self._creds.account_id}/pricing",
            params={"instruments": ",".join(instruments)},
        )
        return data["prices"]

    # -- Candles ---------------------------------------------------------------

    def candles_page(
        self,
        instrument: str,
        granularity: str,
        from_time_iso: str,
        count: int = MAX_CANDLES_PER_REQUEST,
        price: str = "BA",  # Bid + Ask, needed for realistic backtest fills
        include_first: bool = True,
    ) -> list[dict]:
        data = self._request(
            "GET",
            f"/v3/instruments/{instrument}/candles",
            params={
                "granularity": granularity,
                "from": from_time_iso,
                "count": count,
                "price": price,
                "includeFirst": "true" if include_first else "false",
            },
        )
        return data["candles"]

    def candles_range(
        self,
        instrument: str,
        granularity: str,
        from_time_iso: str,
        to_time_iso: str,
        price: str = "BA",
    ) -> Iterator[dict]:
        """Pages through candles between two RFC3339 timestamps, yielding each candle.

        Oanda's `from` param is inclusive, so re-querying with `cursor = last_time`
        of the previous page would re-yield that last candle as the first candle
        of the next page. `includeFirst=false` on every page after the first
        suppresses that re-fetch instead of relying on dedup downstream.
        """
        cursor = from_time_iso
        include_first = True
        seen_last_time: str | None = None
        while True:
            batch = self.candles_page(
                instrument, granularity, cursor, price=price, include_first=include_first
            )
            if not batch:
                return
            for candle in batch:
                if candle["time"] > to_time_iso:
                    return
                yield candle
            last_time = batch[-1]["time"]
            if last_time == seen_last_time:
                return  # no progress -- avoid infinite loop
            seen_last_time = last_time
            cursor = last_time
            include_first = False
            if len(batch) < MAX_CANDLES_PER_REQUEST:
                return

    # -- Orders --------------------------------------------------------------

    def place_market_order(
        self,
        instrument: str,
        units: int,
        stop_loss_price: str | None = None,
        take_profit_price: str | None = None,
    ) -> dict:
        """units: positive = buy, negative = sell."""
        order: dict[str, Any] = {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(units),
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
        }
        if stop_loss_price is not None:
            order["stopLossOnFill"] = {"price": stop_loss_price}
        if take_profit_price is not None:
            order["takeProfitOnFill"] = {"price": take_profit_price}
        return self._request(
            "POST",
            f"/v3/accounts/{self._creds.account_id}/orders",
            json={"order": order},
        )

    def close_trade(self, trade_id: str) -> dict:
        return self._request(
            "PUT",
            f"/v3/accounts/{self._creds.account_id}/trades/{trade_id}/close",
        )
