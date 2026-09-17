from __future__ import annotations

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QWidget

from scalper.live.state import LiveState

COLUMNS = ["Instrument", "Direction", "Units", "Exit Price", "P/L", "Closed At"]


class ClosedPositionsTable(QTableWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(0, len(COLUMNS), parent)
        self.setHorizontalHeaderLabels(COLUMNS)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

    def update_state(self, state: LiveState) -> None:
        # most recently closed first
        trades = sorted(state.closed_trades_today, key=lambda t: t.close_time, reverse=True)
        self.setRowCount(len(trades))
        for row, t in enumerate(trades):
            self.setItem(row, 0, QTableWidgetItem(t.instrument))
            self.setItem(row, 1, QTableWidgetItem("Long" if t.direction == 1 else "Short"))
            self.setItem(row, 2, QTableWidgetItem(str(t.units)))
            self.setItem(row, 3, QTableWidgetItem(f"{t.exit_price:.5f}"))
            pnl_item = QTableWidgetItem(f"{t.pnl:+,.2f}")
            if t.pnl > 0:
                pnl_item.setForeground(QColor("#1e8449"))
            elif t.pnl < 0:
                pnl_item.setForeground(QColor("#c0392b"))
            self.setItem(row, 4, pnl_item)
            self.setItem(row, 5, QTableWidgetItem(t.close_time.astimezone().strftime("%H:%M:%S")))
