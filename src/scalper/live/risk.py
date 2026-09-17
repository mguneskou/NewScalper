"""Live-trading risk controls: the daily realized-loss circuit breaker, a max
concurrent open positions cap, and the Europe/London day boundary both are
measured against (matching the UK user's own sense of "today", not Oanda's
NY-rollover trading day)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

DAY_BOUNDARY_TZ = ZoneInfo("Europe/London")


def trading_day(instant: datetime) -> date:
    """The Europe/London calendar date an instant falls on."""
    if instant.tzinfo is None:
        raise ValueError("instant must be timezone-aware")
    return instant.astimezone(DAY_BOUNDARY_TZ).date()


def day_start_utc(instant: datetime) -> datetime:
    """The UTC instant corresponding to Europe/London midnight for the trading
    day containing `instant`. Used as the `from` bound when fetching "today's"
    closed trades from Oanda."""
    day = trading_day(instant)
    local_midnight = datetime.combine(day, time.min, tzinfo=DAY_BOUNDARY_TZ)
    return local_midnight.astimezone(timezone.utc)


@dataclass
class DailyLossLimit:
    limit_gbp: float
    _day: date | None = field(default=None, repr=False)
    _realized_pnl_today: float = field(default=0.0, repr=False)

    def record_closed_trade_pnl(self, pnl: float, close_time: datetime) -> None:
        self.reset_if_new_day(close_time)
        self._realized_pnl_today += pnl

    def reset_if_new_day(self, now: datetime) -> None:
        day = trading_day(now)
        if day != self._day:
            self._day = day
            self._realized_pnl_today = 0.0

    @property
    def realized_pnl_today(self) -> float:
        return self._realized_pnl_today

    @property
    def breached(self) -> bool:
        return self._realized_pnl_today <= -abs(self.limit_gbp)


@dataclass
class RiskManager:
    daily_loss_limit: DailyLossLimit
    max_concurrent_positions: int

    def can_open_new_trade(self, open_position_count: int, now: datetime) -> bool:
        self.daily_loss_limit.reset_if_new_day(now)
        if self.daily_loss_limit.breached:
            return False
        if open_position_count >= self.max_concurrent_positions:
            return False
        return True
