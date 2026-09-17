import sys
from datetime import timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scalper.live.sync import parse_closed_trades, parse_open_positions


def test_parse_open_positions_long_and_short():
    open_trades = [
        {"id": "101", "instrument": "EUR_USD", "currentUnits": "1000", "price": "1.1000", "unrealizedPL": "5.20"},
        {"id": "102", "instrument": "GBP_USD", "currentUnits": "-2000", "price": "1.2700", "unrealizedPL": "-3.10"},
    ]
    current_prices = {
        "EUR_USD": (1.1005, 1.1007),
        "GBP_USD": (1.2690, 1.2692),
    }

    positions = parse_open_positions(open_trades, current_prices)
    assert len(positions) == 2

    long_pos = positions[0]
    assert long_pos.trade_id == "101"
    assert long_pos.direction == 1
    assert long_pos.units == 1000
    assert long_pos.entry_price == 1.1000
    assert long_pos.current_price == 1.1005  # long marks at bid
    assert long_pos.unrealized_pnl == 5.20

    short_pos = positions[1]
    assert short_pos.direction == -1
    assert short_pos.units == 2000
    assert short_pos.current_price == 1.2692  # short marks at ask
    assert short_pos.unrealized_pnl == -3.10


def test_parse_open_positions_missing_price_falls_back_to_entry():
    open_trades = [{"id": "1", "instrument": "USD_JPY", "currentUnits": "500", "price": "150.00"}]
    positions = parse_open_positions(open_trades, current_prices={})
    assert positions[0].current_price == 150.00
    assert positions[0].unrealized_pnl == 0.0


def test_parse_closed_trades_from_order_fill_transactions():
    transactions = [
        {"type": "MARKET_ORDER", "id": "1"},  # non-fill transactions should be ignored
        {
            "type": "ORDER_FILL",
            "id": "2",
            "instrument": "EUR_USD",
            "time": "2026-01-01T10:15:00.000000000Z",
            "price": "1.1050",
            "tradesClosed": [
                {"tradeID": "101", "units": "-1000", "price": "1.1050", "realizedPL": "5.00"},
            ],
        },
        {
            "type": "ORDER_FILL",
            "id": "3",
            "instrument": "GBP_USD",
            "time": "2026-01-01T11:00:00.000000000Z",
            "price": "1.2650",
            "tradesClosed": [
                {"tradeID": "102", "units": "2000", "price": "1.2650", "realizedPL": "-8.50"},
            ],
        },
        {
            "type": "ORDER_FILL",
            "id": "4",
            "instrument": "USD_JPY",
            "time": "2026-01-01T12:00:00.000000000Z",
            "price": "150.00",
            "tradesClosed": [],  # a fill that opened a trade, closed nothing
        },
    ]

    closed = parse_closed_trades(transactions)
    assert len(closed) == 2

    first = closed[0]
    assert first.trade_id == "101"
    assert first.instrument == "EUR_USD"
    assert first.direction == 1  # negative closing units -> original was long
    assert first.units == 1000
    assert first.exit_price == 1.1050
    assert first.pnl == 5.00
    assert first.close_time.tzinfo is not None
    assert first.close_time.astimezone(timezone.utc).hour == 10

    second = closed[1]
    assert second.direction == -1  # positive closing units -> original was short
    assert second.pnl == -8.50
