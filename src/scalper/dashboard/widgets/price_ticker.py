from __future__ import annotations

from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QWidget

from scalper.live.state import LiveState

COLUMNS = ["Instrument", "Bid", "Ask", "Spread (pips)"]


def _pip_size(instrument: str) -> float:
    return 0.01 if instrument.endswith("_JPY") else 0.0001


class PriceTicker(QTableWidget):
    def __init__(self, instruments: list[str], parent: QWidget | None = None):
        super().__init__(len(instruments), len(COLUMNS), parent)
        self._instruments = instruments
        self.setHorizontalHeaderLabels(COLUMNS)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for row, instrument in enumerate(instruments):
            self.setItem(row, 0, QTableWidgetItem(instrument))

    def update_state(self, state: LiveState) -> None:
        for row, instrument in enumerate(self._instruments):
            prices = state.current_prices.get(instrument)
            if prices is None:
                continue
            bid, ask = prices
            spread_pips = (ask - bid) / _pip_size(instrument)
            self.setItem(row, 1, QTableWidgetItem(f"{bid:.5f}"))
            self.setItem(row, 2, QTableWidgetItem(f"{ask:.5f}"))
            self.setItem(row, 3, QTableWidgetItem(f"{spread_pips:.1f}"))
