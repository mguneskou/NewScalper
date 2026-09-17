from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QLabel, QWidget

from scalper.live.state import LiveState


class BalancePanel(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QGridLayout(self)

        self._balance_value = self._make_value_label()
        self._unrealized_value = self._make_value_label()
        self._realized_today_value = self._make_value_label()
        self._updated_value = self._make_value_label(small=True)
        self._error_label = QLabel("")
        self._error_label.setStyleSheet("color: #c0392b;")

        layout.addWidget(QLabel("Balance"), 0, 0)
        layout.addWidget(self._balance_value, 0, 1)
        layout.addWidget(QLabel("Unrealized P/L"), 1, 0)
        layout.addWidget(self._unrealized_value, 1, 1)
        layout.addWidget(QLabel("Realized P/L today"), 2, 0)
        layout.addWidget(self._realized_today_value, 2, 1)
        layout.addWidget(QLabel("Last updated"), 3, 0)
        layout.addWidget(self._updated_value, 3, 1)
        layout.addWidget(self._error_label, 4, 0, 1, 2)

    @staticmethod
    def _make_value_label(small: bool = False) -> QLabel:
        label = QLabel("--")
        label.setAlignment(Qt.AlignmentFlag.AlignRight)
        font = label.font()
        font.setPointSize(9 if small else 14)
        font.setBold(not small)
        label.setFont(font)
        return label

    def update_state(self, state: LiveState) -> None:
        self._balance_value.setText(f"{state.balance:,.2f} {state.currency}")

        unrealized = state.unrealized_pnl_total
        self._unrealized_value.setText(f"{unrealized:+,.2f}")
        self._unrealized_value.setStyleSheet(self._pnl_color(unrealized))

        realized = state.daily_realized_pnl
        self._realized_today_value.setText(f"{realized:+,.2f}")
        self._realized_today_value.setStyleSheet(self._pnl_color(realized))

        if state.last_updated is not None:
            self._updated_value.setText(state.last_updated.astimezone().strftime("%H:%M:%S"))

        self._error_label.setText(f"Connection issue: {state.last_error}" if state.last_error else "")

    @staticmethod
    def _pnl_color(value: float) -> str:
        if value > 0:
            return "color: #1e8449;"
        if value < 0:
            return "color: #c0392b;"
        return ""
