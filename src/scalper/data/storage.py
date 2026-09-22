"""SQLite persistence for live trades and backtest results."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[3] / "data" / "scalper.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS live_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    oanda_trade_id TEXT,
    instrument TEXT NOT NULL,
    direction INTEGER NOT NULL,
    units INTEGER NOT NULL,
    entry_time TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_time TEXT,
    exit_price REAL,
    pnl REAL,
    strategy_name TEXT NOT NULL,
    execution_mode TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS backtest_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    instrument TEXT NOT NULL,
    granularity TEXT NOT NULL,
    params_json TEXT NOT NULL,
    train_start TEXT,
    train_end TEXT,
    validate_start TEXT,
    validate_end TEXT,
    is_out_of_sample INTEGER NOT NULL,
    account_currency TEXT NOT NULL,
    trade_count INTEGER NOT NULL,
    win_rate REAL NOT NULL,
    profit_factor REAL NOT NULL,
    total_pnl REAL NOT NULL,
    expectancy REAL NOT NULL,
    max_drawdown REAL NOT NULL,
    max_drawdown_pct REAL NOT NULL,
    sharpe_ratio REAL NOT NULL,
    avg_r_multiple REAL NOT NULL,
    ending_balance REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS backtest_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES backtest_runs(id),
    direction INTEGER NOT NULL,
    entry_time TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_time TEXT NOT NULL,
    exit_price REAL NOT NULL,
    units INTEGER NOT NULL,
    pnl REAL NOT NULL,
    pnl_quote_ccy REAL NOT NULL,
    risk REAL NOT NULL,
    exit_reason TEXT NOT NULL
);
"""

# Columns added to a table after its original CREATE TABLE above, so `connect()`
# can retrofit them onto a database file created before they existed --
# `CREATE TABLE IF NOT EXISTS` only ever helps for a brand-new file, not one that
# already has the table in an older shape. Every column here must also appear in
# the CREATE TABLE statement above (so a fresh database gets it immediately);
# this dict is purely what makes an *existing* database catch up.
_COLUMN_MIGRATIONS: dict[str, list[tuple[str, str]]] = {
    "backtest_runs": [
        ("account_currency", "TEXT NOT NULL DEFAULT ''"),
        ("avg_r_multiple", "REAL NOT NULL DEFAULT 0"),
    ],
    "backtest_trades": [
        ("pnl_quote_ccy", "REAL NOT NULL DEFAULT 0"),
        ("risk", "REAL NOT NULL DEFAULT 0"),
    ],
}


def _apply_column_migrations(conn: sqlite3.Connection) -> None:
    for table, columns in _COLUMN_MIGRATIONS.items():
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name, column_def in columns:
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {column_def}")


@contextmanager
def connect(db_path: Path | None = None):
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    # WAL mode lets other connections read committed data while this one still
    # has a transaction open -- needed so progress can be checked mid-run.
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        conn.executescript(SCHEMA)
        _apply_column_migrations(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


def save_backtest_run(
    conn: sqlite3.Connection,
    *,
    strategy_name: str,
    instrument: str,
    granularity: str,
    params: dict,
    train_start: str | None,
    train_end: str | None,
    validate_start: str | None,
    validate_end: str | None,
    is_out_of_sample: bool,
    account_currency: str,
    metrics,
    trades,
) -> int:
    from datetime import datetime, timezone

    cur = conn.execute(
        """
        INSERT INTO backtest_runs (
            created_at, strategy_name, instrument, granularity, params_json,
            train_start, train_end, validate_start, validate_end, is_out_of_sample,
            account_currency, trade_count, win_rate, profit_factor, total_pnl, expectancy,
            max_drawdown, max_drawdown_pct, sharpe_ratio, avg_r_multiple, ending_balance
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            strategy_name,
            instrument,
            granularity,
            json.dumps(params),
            train_start,
            train_end,
            validate_start,
            validate_end,
            int(is_out_of_sample),
            account_currency,
            metrics.trade_count,
            metrics.win_rate,
            metrics.profit_factor,
            metrics.total_pnl,
            metrics.expectancy,
            metrics.max_drawdown,
            metrics.max_drawdown_pct,
            metrics.sharpe_ratio,
            metrics.avg_r_multiple,
            metrics.ending_balance,
        ),
    )
    run_id = cur.lastrowid

    conn.executemany(
        """
        INSERT INTO backtest_trades (
            run_id, direction, entry_time, entry_price, exit_time, exit_price,
            units, pnl, pnl_quote_ccy, risk, exit_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                run_id,
                t.direction,
                str(t.entry_time),
                t.entry_price,
                str(t.exit_time),
                t.exit_price,
                t.units,
                t.pnl,
                t.pnl_quote_ccy,
                t.risk,
                t.exit_reason,
            )
            for t in trades
        ],
    )
    return run_id
