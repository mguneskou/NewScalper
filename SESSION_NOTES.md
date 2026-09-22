# Session notes — backtest investigation (2026-09-21 / 2026-09-22)

Handoff note for picking this up in a new session. Written for future-Claude,
not the user directly — read this first, then check `git status` and
`git diff` to see the actual uncommitted state before doing anything else.

## TL;DR of where things stand

Started from "what's the backtest status" and ended up: fixing a real
currency-conversion bug, fixing a real optimizer-objective bug, adding a new
strategy (Opening Range Breakout), and manually tuning it to the **first
genuinely profitable out-of-sample result in the whole investigation**:
**USD_JPY, opening_range_breakout, M5, full 5yr walk-forward: +3.19% return,
profit factor 1.09, Sharpe +1.03.** EUR_USD is ~breakeven (+0.19%), GBP_USD
flat (-0.34%), EUR_GBP still clearly negative (-2.54%, unresolved).

**Nothing is committed yet.** Everything below is sitting in the working tree.
Run `git status` first thing.

## Chronological summary

1. **Initial state**: 3 strategies existed (`ema_cross`, `rsi_reversion`,
   `bollinger_breakout`), partially backtested. All eventually shown to be
   net-losing out-of-sample everywhere tested.
2. **Found and fixed a currency-mixing bug**: the backtest compared raw P&L
   across instruments with different quote currencies (USD/JPY/GBP) with no
   conversion — made JPY-quoted results look ~100x larger than they were.
   Fixed with `src/scalper/backtest/fx.py` (new `FxConverter`, converts every
   trade's P&L to `account_currency` = GBP, set in `config/settings.yaml`).
   Also found/worked around a pre-existing data bug: ~74 duplicate-timestamp
   rows per cached instrument (`data/cache/*.parquet`), likely an OANDA fetch
   pagination artifact in `history.py`'s `backfill()` — **not fixed at the
   root**, just deduped defensively inside `fx.py`.
3. **Diagnosed why the 3 original strategies lose**: no look-ahead bugs in the
   engine; the real issue is strategy/timeframe fit (slow TA indicators forced
   into tight scalping brackets) plus a fixed-pip risk grid that always
   pinned to its own boundary (evidence the grid was too narrow, not that
   wide stops don't help).
4. **M1 sweep** (`rsi_reversion` + `bollinger_breakout`, 2yr, all 4
   instruments) — bollinger got much worse on M1 (overtrading noise),
   rsi_reversion held up better relatively.
5. **ATR-relative stop prototype** for `rsi_reversion`
   (`scripts/run_backtest_atr.py`, `src/scalper/backtest/risk.py` for
   `atr_pips()`). Two failed calibration passes (multiplier range too narrow,
   found via checking what params got selected — always pinned to grid max)
   before a working one: wide SL (sl_mult up to 28x ATR) + tight TP (6x ATR)
   cut losses 35-81% vs fixed pips across all 4 instruments. Still net
   negative everywhere, but directionally the best family found up to that
   point.
6. **Built `OpeningRangeBreakoutStrategy`**
   (`src/scalper/strategies/opening_range_breakout.py`): box from first
   `box_minutes` after `session_open_hour` (Europe/London time), long/short
   the breakout, flat on reversion to box midpoint or next day's box forming
   (no overnight carry — verified with a dedicated test for the reset).
7. **Found and fixed a real optimizer bug**: `grid_search_train`'s objective
   was raw `total_pnl` with a fixed position size, which let the grid search
   pick a degenerate ultra-tight stop (e.g. 3 pips) purely because trading it
   thousands of times racked up a bigger in-sample total than genuine edge —
   then it collapsed out-of-sample. **First fix attempt (switch objective to
   Sharpe) was verified NOT to work** (Sharpe penalizes return variance, not
   trade frequency — same combo still won). Real fix: added a `risk` field to
   `Trade` (stop distance x position size, in account currency) and an
   `avg_r_multiple` metric (P&L normalized by each trade's own risk) to
   `Metrics`; objective default changed to `avg_r_multiple` with a
   `min_trades=30` floor. Verified against the exact window that exposed the
   bug before trusting it this time.
8. **Schema migration got missed twice** — `CREATE TABLE IF NOT EXISTS`
   doesn't retrofit columns onto an existing `data/scalper.db`. Fixed
   permanently: `storage.py` now has `_apply_column_migrations()` /
   `_COLUMN_MIGRATIONS`, run automatically inside `connect()`. Any new column
   added to a table in the future needs an entry there too, or this bites
   again.
9. **Manual iterative tuning of ORB** (`scripts/tune_orb.py` — fast
   single-trial tool, no grid search, prints exit-reason breakdown). Two
   rounds: a fixed 6-month window, then a randomly-picked, non-overlapping
   1-year window (2025-08-01 to 2026-08-01) to check the findings weren't
   specific to one slice. Findings, in order of leverage:
   - Wide stop-loss (plateaus ~40-80 pips depending on window) + moderate
     take-profit (~10 pips) beats the original tight fixed-pip grid, same
     "wide SL, tight TP" shape as the ATR experiment on rsi_reversion.
   - `session_open_hour=8` (London open) and `box_minutes=30` consistently
     beat NY open / 15min / 60min boxes.
   - New param `exit_buffer_frac` (push the box_mid exit trigger further into
     the box, as a fraction of box height) — peak around 0.4, modest but real
     improvement.
   - New param `min_box_height_pips` (skip days with a too-small opening
     range) — peak around 8 pips (roughly EUR_USD's 25th percentile box
     height), non-monotonic/bumpy 7-10, so treated as a rough answer.
   - Widened the shared `risk_search.stop_loss_pips` grid in
     `config/settings.yaml` to `[15, 25, 40, 60, 80]` (dropped 3/5/10 — never
     won a single window across 4 separate experiments now).
10. **Final full 5yr walk-forward validation** with the tuned param grid —
    see TL;DR above for the result.

## Current DB contents (`data/scalper.db`)

Query `SELECT strategy_name, instrument, granularity, COUNT(*) FROM
backtest_runs GROUP BY 1,2,3` to check freshness before trusting anything
below — this reflects state as of session end:

- `opening_range_breakout`: all 4 instruments, **M5, full 5yr** (11 windows
  each) — the tuned, validated version (final one, superseding 2 earlier
  cleared attempts).
- `rsi_reversion`, `bollinger_breakout`: all 4 instruments, **M1, 2yr only**
  (3 windows each) — from before the optimizer fix, still using the old
  fixed-pip fixed-position-size approach. Worth re-running with the fixed
  objective for a fair comparison to ORB.
- `rsi_reversion_atr`: all 4 instruments, **M1, 2yr** (3 windows each) — the
  ATR-relative-stop version, also from before the optimizer objective fix.
- `ema_cross`: **nothing currently stored** (was tested earlier in the
  session on M5 full history, but those rows got cleared during the currency
  bug fix cleanup and never re-run since — it was the worst performer by far,
  low priority to redo unless doing a full clean sweep).
- Every table row from before the currency-conversion fix and before the
  optimizer-objective fix has been deleted at various points this session —
  nothing stale should remain, but double check `account_currency` and
  `avg_r_multiple` columns are non-empty/non-zero-default on any row before
  trusting it (a `0`/`''` in those columns means it predates one of the
  fixes and slipped through).

## Uncommitted files (as of session end)

New:
- `src/scalper/backtest/fx.py`, `src/scalper/backtest/risk.py`
- `src/scalper/strategies/opening_range_breakout.py`
- `scripts/run_backtest_atr.py`, `scripts/tune_orb.py`
- `tests/test_risk.py` (shows as modified in git status, not untracked —
  double check nothing was accidentally clobbered from an earlier scaffold)
- `logs_*.txt` at repo root (8 files) — raw stdout from various background
  sweep runs, kept for reference/debugging, not meant to be permanent
  artifacts. Safe to delete or gitignore if cluttering.

Modified: `config/settings.yaml`, `scripts/run_backtest.py`,
`src/scalper/backtest/engine.py`, `src/scalper/backtest/metrics.py`,
`src/scalper/backtest/optimizer.py`, `src/scalper/data/storage.py`,
`tests/test_backtest_engine.py`, `tests/test_metrics.py`,
`tests/test_storage.py`, `tests/test_strategies.py`.

All 33 tests pass as of last run (`python -m pytest tests/ -q`).

## Known gaps / natural next steps

- **EUR_GBP is unexplained.** It's the weakest ORB result (-2.54%) in every
  version of this experiment, and nobody's dug into *why* (likely: it's the
  lowest-volatility pair traded, so its opening ranges are small and the
  EUR_USD-tuned settings — box height filter especially — may not transfer).
- **ORB has never been run on M1**, only M5. Given rsi_reversion actually got
  *relatively* better on M1 earlier in the session, this is an open question,
  not a dead end.
- **rsi_reversion / bollinger_breakout / rsi_reversion_atr predate the
  optimizer fix.** Their M1 numbers in the DB were selected using the old,
  now-known-flawed `total_pnl` objective. Re-running them with
  `avg_r_multiple` might change their picture the way it did for ORB.
- **Bollinger mean-reversion (fade the bands instead of breaking them)** was
  proposed as a promising next strategy and never built — bollinger_breakout
  was the worst-performing continuation strategy tested, and everything
  learned since points toward mean-reversion being the better-fitting family
  for these instruments/timeframes.
- Other strategies discussed but not built: Stochastic, MACD, Keltner
  Channel, VWAP bounce, Ichimoku, pivot-point bounce/breakout,
  trend-filtered breakout.
- No live/paper-trading validation of the USD_JPY ORB config — everything so
  far is backtest-only. `src/scalper/live/` (risk.py, state.py, sync.py) and
  the dashboard weren't touched this session.
- ~~`history.py`'s duplicate-candle bug (pagination artifact) was worked
  around in `fx.py` but never fixed at the source.~~ **Fixed 2026-09-22**:
  root cause was `OandaClient.candles_page` never passing `includeFirst`, so
  Oanda's inclusive `from` param re-returned each page's last candle as the
  next page's first candle. Fixed by sending `includeFirst=false` on every
  page after the first (`src/scalper/oanda/client.py`), added
  `tests/test_oanda_client.py` to cover it, and one-off deduped the 8
  existing `data/cache/*.parquet` files in place (366-369 dup rows on M1,
  74 on M5, matching the notes below exactly). `history.py`'s `backfill()`
  dedup also now runs unconditionally (previously skipped on a fresh/empty
  cache). The defensive dedup in `fx.py` is left in place but is now
  belt-and-suspenders, not load-bearing.

## Caveats already given to the user, worth remembering

The +3.19% USD_JPY result was explicitly caveated to the user as: not
literally 5 years (≈3.7yr OOS span, first 12mo consumed by initial
training), not compounding (fixed 1000-unit position size regardless of
balance), a thin edge (PF 1.09) that real-world slippage/spread could
plausibly erase, and the product of a walk-forward process that
*re-optimizes periodically* rather than one fixed rule set — replicating it
live would mean re-running the optimization on a schedule, not "set and
forget."
