"""Shared in-memory state for the live trading engine and dashboard.

Plain data containers only -- the live engine (or the poller, for now) writes to
these, the dashboard reads them. No networking or persistence here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class OpenPosition:
    trade_id: str
    instrument: str
    direction: int  # 1 long, -1 short
    units: int
    entry_price: float
    current_price: float
    unrealized_pnl: float


@dataclass
class ClosedTrade:
    trade_id: str
    instrument: str
    direction: int  # 1 long, -1 short
    units: int
    exit_price: float
    pnl: float
    close_time: datetime
    entry_price: float | None = None  # not always available from the fill transaction alone


@dataclass
class LiveState:
    balance: float = 0.0
    currency: str = "GBP"
    open_positions: list[OpenPosition] = field(default_factory=list)
    closed_trades_today: list[ClosedTrade] = field(default_factory=list)
    current_prices: dict[str, tuple[float, float]] = field(default_factory=dict)  # instrument -> (bid, ask)
    execution_mode: str = "manual"  # "auto" or "manual"
    circuit_breaker_active: bool = False
    daily_realized_pnl: float = 0.0
    last_updated: datetime | None = None
    last_error: str | None = None

    @property
    def unrealized_pnl_total(self) -> float:
        return sum(p.unrealized_pnl for p in self.open_positions)

    @property
    def closed_pnl_today_total(self) -> float:
        return sum(t.pnl for t in self.closed_trades_today)
