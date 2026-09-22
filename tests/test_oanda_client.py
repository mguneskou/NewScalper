import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scalper.config import OandaCredentials
from scalper.oanda.client import OandaClient


def _candle(time_iso: str, complete: bool = True) -> dict:
    return {
        "time": time_iso,
        "volume": 10,
        "complete": complete,
        "bid": {"o": "1.1", "h": "1.1", "l": "1.1", "c": "1.1"},
        "ask": {"o": "1.1", "h": "1.1", "l": "1.1", "c": "1.1"},
    }


def _make_client(handler) -> OandaClient:
    creds = OandaCredentials(api_token="t", account_id="a", environment="practice")
    client = OandaClient(creds)
    client._client = httpx.Client(
        base_url=creds.rest_host, transport=httpx.MockTransport(handler)
    )
    return client


def test_candles_range_does_not_repeat_page_boundary_candle(monkeypatch):
    # Oanda's `from` is inclusive, so page 2 starting at page 1's last candle
    # would re-yield it unless includeFirst=false is sent -- this is the
    # pagination bug that produced duplicate-timestamp rows in the cache.
    # A full page triggers another fetch, so shrink the page-size threshold
    # to 2 to exercise the multi-page path without 5000 fake candles.
    import scalper.oanda.client as client_module

    monkeypatch.setattr(client_module, "MAX_CANDLES_PER_REQUEST", 2)

    all_times = [
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:01:00Z",
        "2026-01-01T00:02:00Z",
    ]
    seen_include_first = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        from_time = params["from"]
        include_first = params.get("includeFirst") == "true"
        seen_include_first.append(params.get("includeFirst"))
        start = all_times.index(from_time) + (0 if include_first else 1)
        page_times = all_times[start : start + 2]
        return httpx.Response(200, json={"candles": [_candle(t) for t in page_times]})

    client = _make_client(handler)
    candles = list(
        client.candles_range(
            "EUR_USD", "M1", "2026-01-01T00:00:00Z", "2026-01-01T00:02:00Z"
        )
    )

    times = [c["time"] for c in candles]
    assert times == all_times
    assert seen_include_first == ["true", "false"]


def test_candles_page_defaults_to_include_first_true():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params.get("includeFirst") == "true"
        return httpx.Response(200, json={"candles": [_candle("2026-01-01T00:00:00Z")]})

    client = _make_client(handler)
    candles = client.candles_page("EUR_USD", "M1", "2026-01-01T00:00:00Z")
    assert len(candles) == 1
