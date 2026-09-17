import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scalper.live.risk import DailyLossLimit, RiskManager, day_start_utc, trading_day

LONDON = ZoneInfo("Europe/London")


def test_trading_day_requires_timezone_aware_input():
    with pytest.raises(ValueError):
        trading_day(datetime(2026, 1, 1))


def test_trading_day_uses_london_calendar_date():
    # 23:30 UTC on Jan 1 is still Jan 1 in London (winter, no DST offset)
    instant = datetime(2026, 1, 1, 23, 30, tzinfo=timezone.utc)
    assert trading_day(instant) == datetime(2026, 1, 1).date()

    # 23:30 UTC in summer (BST, UTC+1) is already the next day locally
    instant_summer = datetime(2026, 6, 1, 23, 30, tzinfo=timezone.utc)
    assert trading_day(instant_summer) == datetime(2026, 6, 2).date()


def test_day_start_utc_matches_london_midnight():
    instant = datetime(2026, 6, 15, 10, 0, tzinfo=timezone.utc)  # BST in effect (UTC+1)
    start = day_start_utc(instant)
    # London midnight in BST is 23:00 UTC the previous day
    assert start == datetime(2026, 6, 14, 23, 0, tzinfo=timezone.utc)


def test_daily_loss_limit_breaches_at_threshold():
    limit = DailyLossLimit(limit_gbp=50.0)
    now = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)

    limit.record_closed_trade_pnl(-20.0, now)
    assert not limit.breached
    limit.record_closed_trade_pnl(-30.0, now)
    assert limit.breached  # exactly at -50
    assert limit.realized_pnl_today == -50.0


def test_daily_loss_limit_resets_on_new_london_day():
    limit = DailyLossLimit(limit_gbp=50.0)
    day1 = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    day2 = datetime(2026, 1, 2, 10, tzinfo=timezone.utc)

    limit.record_closed_trade_pnl(-60.0, day1)
    assert limit.breached

    limit.record_closed_trade_pnl(-5.0, day2)
    assert not limit.breached
    assert limit.realized_pnl_today == -5.0


def test_daily_loss_limit_wins_do_not_trigger_breach():
    limit = DailyLossLimit(limit_gbp=50.0)
    now = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    limit.record_closed_trade_pnl(100.0, now)
    assert not limit.breached


def test_risk_manager_blocks_on_loss_limit_breach():
    rm = RiskManager(daily_loss_limit=DailyLossLimit(limit_gbp=50.0), max_concurrent_positions=5)
    now = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    rm.daily_loss_limit.record_closed_trade_pnl(-60.0, now)
    assert rm.can_open_new_trade(open_position_count=0, now=now) is False


def test_risk_manager_blocks_on_max_concurrent_positions():
    rm = RiskManager(daily_loss_limit=DailyLossLimit(limit_gbp=50.0), max_concurrent_positions=2)
    now = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    assert rm.can_open_new_trade(open_position_count=2, now=now) is False
    assert rm.can_open_new_trade(open_position_count=1, now=now) is True


def test_risk_manager_allows_new_day_after_breach():
    rm = RiskManager(daily_loss_limit=DailyLossLimit(limit_gbp=50.0), max_concurrent_positions=5)
    day1 = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    day2 = datetime(2026, 1, 2, 10, tzinfo=timezone.utc)
    rm.daily_loss_limit.record_closed_trade_pnl(-60.0, day1)
    assert rm.can_open_new_trade(open_position_count=0, now=day1) is False
    assert rm.can_open_new_trade(open_position_count=0, now=day2) is True
