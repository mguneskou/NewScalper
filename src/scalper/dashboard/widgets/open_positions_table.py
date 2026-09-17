from __future__ import annotations

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QWidget

from scalper.live.state import LiveState

COLUMNS = ["Instrument", "Direction", "Units", "Entry", "Current", "Unrealized P/L"]


class OpenPositionsTable(QTableWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(0, len(COLUMNS), parent)
        self.setHorizontalHeaderLabels(COLUMNS)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

    def update_state(self, state: LiveState) -> None:
        positions = state.open_positions
        self.setRowCount(len(positions))
        for row, pos in enumerate(positions):
            self.setItem(row, 0, QTableWidgetItem(pos.instrument))
            self.setItem(row, 1, QTableWidgetItem("Long" if pos.direction == 1 else "Short"))
            self.setItem(row, 2, QTableWidgetItem(str(pos.units)))
            self.setItem(row, 3, QTableWidgetItem(f"{pos.entry_price:.5f}"))
            self.setItem(row, 4, QTableWidgetItem(f"{pos.current_price:.5f}"))
            pnl_item = QTableWidgetItem(f"{pos.unrealized_pnl:+,.2f}")
            if pos.unrealized_pnl > 0:
                pnl_item.setForeground(QColor("#1e8449"))
            elif pos.unrealized_pnl < 0:
                pnl_item.setForeground(QColor("#c0392b"))
            self.setItem(row, 5, pnl_item)
