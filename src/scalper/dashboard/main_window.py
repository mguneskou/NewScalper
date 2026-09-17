from __future__ import annotations

from PySide6.QtWidgets import QLabel, QMainWindow, QTabWidget, QVBoxLayout, QWidget

from scalper.dashboard.poller import OandaPoller
from scalper.dashboard.widgets.balance_panel import BalancePanel
from scalper.dashboard.widgets.closed_positions_table import ClosedPositionsTable
from scalper.dashboard.widgets.open_positions_table import OpenPositionsTable
from scalper.dashboard.widgets.price_ticker import PriceTicker
from scalper.live.state import LiveState
from scalper.oanda.client import OandaClient


class MainWindow(QMainWindow):
    def __init__(self, client: OandaClient, instruments: list[str]):
        super().__init__()
        self.setWindowTitle("Oanda Auto-Scalper (practice account)")
        self.resize(900, 650)

        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self._balance_panel = BalancePanel(self)
        layout.addWidget(self._balance_panel)

        tabs = QTabWidget(self)
        layout.addWidget(tabs)

        self._price_ticker = PriceTicker(instruments, self)
        tabs.addTab(self._price_ticker, "Live Prices")

        self._open_positions = OpenPositionsTable(self)
        tabs.addTab(self._open_positions, "Open Positions")

        self._closed_positions = ClosedPositionsTable(self)
        tabs.addTab(self._closed_positions, "Closed Today")

        # Strategy controls / auto-vs-manual execution toggle land here once the
        # winning strategy from the Phase 5 backtest checkpoint is known.
        placeholder = QLabel(
            "Strategy execution controls will appear here once a strategy has "
            "been selected from the backtest results.",
            self,
        )
        placeholder.setWordWrap(True)
        tabs.addTab(placeholder, "Strategy")

        self._poller = OandaPoller(client, instruments)
        self._poller.state_updated.connect(self._on_state_updated)
        self._poller.start()

    def _on_state_updated(self, state: LiveState) -> None:
        self._balance_panel.update_state(state)
        self._price_ticker.update_state(state)
        self._open_positions.update_state(state)
        self._closed_positions.update_state(state)

    def closeEvent(self, event) -> None:
        self._poller.stop()
        super().closeEvent(event)
