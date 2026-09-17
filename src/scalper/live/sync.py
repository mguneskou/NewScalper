"""Pure functions that translate raw Oanda v20 API responses into our domain
objects. Kept separate from the Qt poller so they can be unit-tested without a
Qt event loop or network access.
"""

from __future__ import annotations

from datetime import datetime

from scalper.live.state import ClosedTrade, OpenPosition


def parse_open_positions(
    open_trades: list[dict], current_prices: dict[str, tuple[float, float]]
) -> list[OpenPosition]:
    positions = []
    for t in open_trades:
        instrument = t["instrument"]
        units = float(t["currentUnits"])
        direction = 1 if units > 0 else -1
        entry_price = float(t["price"])
        bid, ask = current_prices.get(instrument, (entry_price, entry_price))
        # mark-to-market a long at the bid (what you'd sell to close at), a short at the ask
        current_price = bid if direction == 1 else ask
        positions.append(
            OpenPosition(
                trade_id=t["id"],
                instrument=instrument,
                direction=direction,
                units=int(abs(units)),
                entry_price=entry_price,
                current_price=current_price,
                unrealized_pnl=float(t.get("unrealizedPL", 0.0)),
            )
        )
    return positions


def parse_closed_trades(transactions: list[dict]) -> list[ClosedTrade]:
    """Extracts closed-trade records from ORDER_FILL transactions' `tradesClosed`
    entries. Entry price isn't available from the closing fill alone (that lives
    on the original opening transaction), so it's left as None here."""
    closed = []
    for txn in transactions:
        if txn.get("type") != "ORDER_FILL":
            continue
        for tc in txn.get("tradesClosed", []):
            closed_units = float(tc["units"])
            # a negative closing-fill unit count means the original position was
            # long (you sold to close it); positive means it was short.
            direction = 1 if closed_units < 0 else -1
            closed.append(
                ClosedTrade(
                    trade_id=tc["tradeID"],
                    instrument=txn.get("instrument", ""),
                    direction=direction,
                    units=int(abs(closed_units)),
                    exit_price=float(tc.get("price", txn.get("price", 0.0))),
                    pnl=float(tc.get("realizedPL", 0.0)),
                    close_time=datetime.fromisoformat(txn["time"].replace("Z", "+00:00")),
                )
            )
    return closed
