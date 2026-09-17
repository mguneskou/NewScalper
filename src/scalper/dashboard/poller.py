"""Background poller: periodically pulls balance/prices/positions/today's closed
trades from Oanda's REST API and publishes a fresh LiveState via a Qt signal.

Polls rather than uses Oanda's streaming endpoint for this first pass -- simpler,
and a REST poll every couple of seconds is plenty responsive for a dashboard.
Swapping in a streaming client later won't change what widgets consume (LiveState).
"""

from __future__ import annotations

from datetime import datetime, timezone

from PySide6.QtCore import QObject, QTimer, Signal

from scalper.live.risk import day_start_utc
from scalper.live.state import LiveState
from scalper.live.sync import parse_closed_trades, parse_open_positions
from scalper.oanda.client import OandaClient, OandaError


class OandaPoller(QObject):
    state_updated = Signal(object)  # LiveState

    def __init__(
        self,
        client: OandaClient,
        instruments: list[str],
        poll_interval_ms: int = 2000,
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self._client = client
        self._instruments = instruments
        self._timer = QTimer(self)
        self._timer.setInterval(poll_interval_ms)
        self._timer.timeout.connect(self._poll_once)

    def start(self) -> None:
        self._poll_once()
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def _poll_once(self) -> None:
        state = LiveState()
        now = datetime.now(timezone.utc)
        try:
            account = self._client.account_summary()
            state.balance = float(account["balance"])
            state.currency = account["currency"]

            raw_prices = self._client.current_prices(self._instruments)
            state.current_prices = {
                p["instrument"]: (float(p["bids"][0]["price"]), float(p["asks"][0]["price"]))
                for p in raw_prices
                if p.get("bids") and p.get("asks")
            }

            open_trades = self._client.open_trades()
            state.open_positions = parse_open_positions(open_trades, state.current_prices)

            day_start_iso = day_start_utc(now).strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
            transactions = self._client.transactions_since(day_start_iso)
            state.closed_trades_today = parse_closed_trades(transactions)
            state.daily_realized_pnl = state.closed_pnl_today_total

            state.last_updated = now
        except OandaError as e:
            state.last_error = str(e)
        except Exception as e:  # network errors etc. -- keep the dashboard alive
            state.last_error = f"{type(e).__name__}: {e}"

        self.state_updated.emit(state)
