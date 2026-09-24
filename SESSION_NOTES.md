# Session notes — backtest investigation (2026-09-21 / 2026-09-22)

**Self-note (user instruction, 2026-09-22): keep chat responses short from now
on to conserve credits.** Don't repeat plans/diffs/explanations at length in
the reply text — do the work, report results tersely.

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

**Committed and pushed as of 2026-09-22**: `acfc3bb` (pagination fix) and
`6020350` (currency/optimizer fixes + ORB) are on `origin/main`. Run
`git status` first thing to check for anything newer left uncommitted.

**2026-09-22, later same day: pivoted to ORB-only.** `ema_cross`,
`rsi_reversion`, and `bollinger_breakout` (plus the `rsi_reversion_atr`
prototype) are deleted — all net-losing everywhere tested, no rescue from the
optimizer fix, ORB is the only strategy with a genuine edge. User decision:
"we will only carry on with ORB from now on." Removed: the 3 strategy files,
`src/scalper/backtest/risk.py` + `tests/test_risk.py` (ATR prototype, ORB
doesn't use ATR), `scripts/run_backtest_atr.py`, their tests/fixtures, and
`risk_search` in `config/settings.yaml` now directly holds ORB's own wider
grid (the short-lived `risk_search_overrides` per-strategy mechanism was
added and then removed again the same day — see phase 6 below). **DB
cleanup still pending user permission** — see "Known gaps" below, first
item.

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

## Per-instrument fine-tuning + evolving active_params (2026-09-22)

New capability, built per user request: tune ORB per (instrument,
granularity) on a short recent window instead of only the full 5yr
walk-forward, and store the winning params so live code can look them up
without a hardcoded/frozen config.

- `scripts/run_backtest.py` now takes `--train-months`/`--validate-months`
  overrides (previously only `--years`). A short tuning run uses e.g.
  `--years 0.55 --train-months 4 --validate-months 2` -- **use 0.55, not
  0.5**, for "6 months": calendar-month arithmetic (train_months=4 +
  validate_months=2) needs ~181-186 days depending on which months are
  covered, and clipping to exactly 0.5yr (182.5 days) has zero slack, so
  every job failed with "not enough data" on the first attempt. 0.55yr
  (~201 days) leaves enough buffer for exactly one window without spilling
  into a second one (that would need ~8 months).
- New `active_params` table (`src/scalper/data/storage.py`) + `save_active_params()`/
  `get_active_params()`, keyed by (strategy_name, instrument, granularity).
  `run_backtest.py` upserts it from the *last* walk-forward window of every
  run (any run, not just short tuning ones) -- this is what makes it
  "evolving": there's no separate promote-to-live step, no frozen named
  config. Re-run the backtest with fresh data, and whatever live/dashboard
  code eventually calls `get_active_params()` picks up the update
  automatically. Nothing reads this table yet (`src/scalper/live/` wasn't
  touched) -- wiring that up is a natural next step whenever live trading
  is revisited.
- Tests: `test_active_params_round_trip`, `test_active_params_overwrite_on_re_tune`.

**First run's results** (4mo train 2026-03-01→07-01, 2mo OOS validate
2026-07-01→09-01, all 4 instruments x M1/M5, `logs_per_instrument_6mo_tune.txt`):

| Instrument | Gran | Trades | Win% | PF | P&L (GBP) | Return | Sharpe | Chosen SL/TP |
|---|---|---|---|---|---|---|---|---|
| USD_JPY | M1 | 163 | 54.6% | 1.33 | +27.81 | +0.28% | 2.10 | 60/**30** |
| USD_JPY | M5 | 135 | 54.8% | 1.20 | +15.20 | +0.15% | 1.67 | 60/**30** |
| GBP_USD | M5 | 215 | 56.7% | 1.11 | +12.68 | +0.13% | 1.27 | 80/15 |
| EUR_USD | M5 | 74 | 43.2% | 0.87 | -4.67 | -0.05% | -0.51 | 60/15 |
| EUR_USD | M1 | 85 | 54.1% | 0.83 | -6.74 | -0.07% | -1.73 | 120/10 |
| GBP_USD | M1 | 222 | 50.0% | 0.87 | -17.04 | -0.17% | -1.20 | 120/15 |
| EUR_GBP | M1 | 2 | 0.0% | 0.00 | -1.65 | -0.02% | 0.00 | 40/10 |
| EUR_GBP | M5 | 2 | 0.0% | 0.00 | -2.14 | -0.02% | 0.00 | 120/10 |

Return% is on the 2-month OOS slice only, not annualized. Findings:
- **USD_JPY wins on both granularities** and picked `take_profit_pips=30` —
  the grid's current max on both. Same "pinned to the boundary" signature
  as before; worth widening the TP grid further specifically for USD_JPY if
  we keep tuning it.
- **GBP_USD splits by granularity**: profitable on M5, losing on M1.
- **EUR_USD is marginally negative on both.**
- **EUR_GBP is broken, not just weak**: only 2 trades in 2 months on both
  granularities. `min_box_height_pips=8.0` (one of the two grid choices)
  combined with EUR_GBP's already-small opening ranges is filtering out
  almost every day. This is the same EUR_GBP problem flagged earlier
  (phase 1 below) surfacing immediately at 6-month scale, more starkly than
  in the 5yr run.
- **Caveat**: this is a *single* 2-month out-of-sample window per instrument,
  not a multi-window walk-forward like the 5yr validation. It's a fast
  per-instrument signal, useful for phase 6/1 iteration, but weaker evidence
  than the full validation -- don't treat these `active_params` as
  live-ready without a broader check first.

### Round 2: per-instrument search grids, not just per-instrument winners

User correction: a shared grid that happens to pick different winners per
instrument isn't "per-instrument fine-tuning" -- each instrument needed its
own search *space*. Added `instrument_overrides` to `config/settings.yaml`
(risk_search and/or strategy_params, merged on top of the shared grid) and
threaded an optional `param_grid` argument through
`grid_search_train`/`run_walk_forward` in `optimizer.py` so an instrument can
override the strategy's own param_grid too (needed for EUR_GBP's
min_box_height_pips). Test: `test_run_walk_forward_param_grid_override_replaces_strategy_default`.

Same 6-month window, re-run with: USD_JPY `take_profit_pips: [20,25,30,40,50]`,
EUR_USD/GBP_USD `stop_loss_pips: [60,80,100,120,150]`, EUR_GBP
`min_box_height_pips: [0.0,2.0,4.0]` (`logs_per_instrument_6mo_tune2.txt`):

| Instrument | Gran | Trades | PF | P&L (GBP) | Return | Sharpe | Chosen SL/TP |
|---|---|---|---|---|---|---|---|
| USD_JPY | M1 | 140 | 1.36 | +30.69 | +0.31% | 2.03 | 40/**50** |
| USD_JPY | M5 | 110 | 1.25 | +18.36 | +0.18% | 1.74 | 60/**50** |
| GBP_USD | M5 | 215 | 1.11 | +12.68 | +0.13% | 1.27 | 80/15 |
| EUR_USD | M5 | 74 | 0.87 | -4.67 | -0.05% | -0.51 | 60/15 |
| EUR_USD | M1 | 85 | 0.83 | -6.74 | -0.07% | -1.73 | **150**/10 |
| GBP_USD | M1 | 287 | 0.88 | -16.41 | -0.16% | -1.16 | **150**/10 |
| EUR_GBP | M5 | 101 | 0.51 | -26.64 | -0.27% | -4.66 | 120/10 |
| EUR_GBP | M1 | 117 | 0.48 | -30.04 | -0.30% | -5.49 | 120/10 |

**Important finding**: fixing EUR_GBP's trade-starvation didn't rescue it --
with a real sample size (101-117 trades vs. the previous 2), it's PF ~0.5,
Sharpe -4.7 to -5.5, ~30% win rate. Not a filter artifact; a genuine loser.
USD_JPY still pinned to the new TP max (50) on both granularities; EUR_USD/
GBP_USD still pinned to the new SL max (150) on M1 only.

### Round 3: drop EUR_GBP, widen the still-pinned grids again

User decision: drop EUR_GBP entirely (removed from `instruments` in
`config/settings.yaml`, its `instrument_overrides` entry deleted, its rows
purged from `backtest_runs`/`active_params`). Widened again: USD_JPY
`take_profit_pips: [40,50,70,90,120]`, EUR_USD/GBP_USD
`stop_loss_pips: [100,150,200,250,300]` (`logs_per_instrument_6mo_tune3.txt`):

| Instrument | Gran | Trades | PF | P&L (GBP) | Return | Sharpe | Chosen SL/TP |
|---|---|---|---|---|---|---|---|
| USD_JPY | M5 | 94 | **1.58** | +34.66 | +0.35% | 2.27 | 60/70 |
| USD_JPY | M1 | 140 | 1.36 | +30.69 | +0.31% | 2.03 | 40/50 |
| GBP_USD | M5 | 215 | 1.11 | +12.68 | +0.13% | 1.27 | 100/15 |
| EUR_USD | M5 | 74 | 0.87 | -4.67 | -0.05% | -0.51 | 150/15 |
| EUR_USD | M1 | 85 | 0.83 | -6.74 | -0.07% | -1.73 | **300**/10 |
| GBP_USD | M1 | 287 | 0.88 | -16.41 | -0.16% | -1.16 | **300**/10 |

**USD_JPY M5 converged and improved substantially** (PF 1.20 -> 1.25 -> 1.58
across the three rounds; TP settled at 70, no longer at the grid's edge).
USD_JPY M1 was already at its true optimum in round 2 (TP=50 unchanged here).

**EUR_USD/GBP_USD M1's SL pinning is a dead end, not a real signal** --
verified via `backtest_trades.exit_reason` for both instruments' latest M1
run: **zero trades exited via `stop_loss`** (EUR_USD: 42 signal + 42
take_profit + 1 end_of_data; GBP_USD: 120 signal + 166 take_profit + 1
end_of_data). Every SL widening from 120 to 300 produced byte-identical
validate-window results because the stop was never once hit -- grid search
was picking the largest value as a meaningless tie-break, not converging on
a real edge. **Stop widening SL for EUR_USD/GBP_USD M1.** Their negative
performance isn't a stop-sizing problem; if worth pursuing, the lever is
elsewhere (take_profit, exit_buffer_frac, min_box_height_pips) or the
conclusion is simply that M1 doesn't suit these two (both are already
solidly better on M5: GBP_USD PF 1.11, EUR_USD still marginally negative
but far less so).

**Current best known active_params** (6 combos, EUR_GBP excluded): USD_JPY
M5 is the strongest result in the whole project (PF 1.58); USD_JPY M1,
GBP_USD M5 are solid; EUR_USD (both) and GBP_USD M1 remain net-negative on
this window and are not stop-loss-fixable.

## 11-item improvement backlog (2026-09-22) -- ranked, working through in phases

User asked "have we tried everything" -- answer was no. Full ranked list
(high to low impact): (1) multi-window validation of current active_params,
(2) per-instrument session_open_hour incl. Asian session, (3) per-instrument
min_box_height_pips/exit_buffer_frac, (4) per-instrument box_minutes,
(5) box-relative SL/TP sizing in the real optimizer (not just tune_orb.py's
manual mode), (6) long-only/short-only filtering per instrument,
(7) volatility regime filter, (8) time-based/trailing exits, (9) day-of-week
/news-calendar filtering, (10) alternative ORB variants (retest entries,
multiple boxes/day), (11) cross-instrument portfolio/ensemble logic.
Process: implement one phase at a time, only backtest when checking a
phase's impact, report in detail, user decides keep/drop before the next
phase. Added `--start`/`--end`/`--skip-active-params` to `run_backtest.py`
for this (see below).

### Phase 1: multi-window validation -- DONE, encouraging result

Re-ran the current per-instrument grids (EUR_USD/GBP_USD/USD_JPY x M1/M5)
on two more non-overlapping 6-month windows, in addition to the original
(2026-07->09). `--skip-active-params` used so historical-window checks
don't clobber the live config with stale params.

**Gotcha, hit twice**: an exact N-month `--start`/`--end` span has zero
slack for `walk_forward_windows`' calendar-month arithmetic (same root
cause as the `--years 0.5` bug from earlier) -- both new windows failed
"not enough data" until start was pulled back ~15 days. **Fixed
permanently 2026-09-22** (see below) -- no padding needed anymore.

| Instrument | Gran | Win A (Jul-Sep '26) | Win B (Dec-Feb '26) | Win C (Jun-Aug '25) | Verdict |
|---|---|---|---|---|---|
| USD_JPY | M5 | PF 1.58 | PF 1.04 | PF 1.26 | **Robust: 3/3 profitable** |
| GBP_USD | M5 | PF 1.11 | PF 1.13 | PF 1.07 | **Robust: 3/3 profitable, tightest band** |
| USD_JPY | M1 | PF 1.36 | PF 0.85 | PF 1.21 | Mixed: 2/3 profitable |
| EUR_USD | M5 | PF 0.87 | PF 1.01 | PF 0.87 | Inconclusive: ~breakeven |
| EUR_USD | M1 | PF 0.83 | PF 0.90 | PF 0.94 | **Robust: 3/3 losing** |
| GBP_USD | M1 | PF 0.88 | PF 0.89 | PF 0.98 | **Robust: 3/3 losing** |

This is the most important finding of the whole tuning effort so far:
**USD_JPY M5 and GBP_USD M5 have a real, window-independent edge** (not
curve-fit to one slice) -- exactly what walk-forward validation is for.
Equally useful negative result: **EUR_USD M1 and GBP_USD M1 are
consistently losing across all 3 independent windows**, confirming the
earlier single-window finding was real, not noise -- their SL-pinning issue
was correctly identified as a dead end (see round 3 above); the M1
underperformance itself looks structural, not a sizing problem.
`logs_phase1_windowB.txt`, `logs_phase1_windowC.txt`.

**Boundary bug fixed permanently**: `walk_forward_windows` (`optimizer.py`)
now clamps `validate_end` to the last available timestamp instead of
rejecting the whole window when it overshoots by a few hours/days --
calendar months aren't a fixed day count, so an "exact" N-month day-based
span (via `--years` or `--start`/`--end`) was frequently just short of what
`DateOffset(months=...)` needed. `train_end` is still NOT clamped (a full
training window is still required). Verified `--years 0.5` now works with
no padding. Tests: `test_walk_forward_windows_clamps_validate_end_instead_of_dropping_window`,
`test_walk_forward_windows_still_rejects_incomplete_training_window`.

### Phase 4: per-instrument box_minutes -- DONE, kept

Re-opened `box_minutes` from locked-at-30 to `[15, 30, 60]` (real prior
evidence behind the values, unlike phase 3's blind widening).
`logs_phase4_box_minutes.txt` surfaced a broken EUR_USD M5 result (0 trades,
`session_open_hour=0` + `box_minutes=15` + `min_box_height_pips=8.0`) --
same trade-starvation pattern as GBP_USD's Asian-session issue in phase 2.
Extended the Asian-session exclusion to EUR_USD too
(`instrument_overrides.EUR_USD.strategy_params.session_open_hour: [8,13]`,
mirrors GBP_USD's), re-ran EUR_USD/GBP_USD (`logs_phase4_refix.txt`) for a
clean read: EUR_USD M5 fell back to the exact phase-2 baseline (confirms
the fix worked, no hidden benefit lost), EUR_USD M1 and GBP_USD M5 improved
genuinely, GBP_USD M1 unchanged, **USD_JPY M1 regressed** (1.24->1.04,
`box_minutes=15`, still profitable, trade count went *up* so not a
thin-sample artifact). User decided to keep everything as-is (net positive
across the 6 combos) rather than surgically revert USD_JPY M1.
Current `active_params` (post phase 4, final):

| Instrument | Gran | PF | box_minutes |
|---|---|---|---|
| USD_JPY | M5 | 1.29 | 15 |
| USD_JPY | M1 | 1.04 | 15 |
| EUR_USD | M1 | 0.92 | 60 |
| EUR_USD | M5 | 0.83 | 30 |
| GBP_USD | M5 | 1.01 | 60 |
| GBP_USD | M1 | 0.79 | 60 |

Added `--instruments` CLI flag to `run_backtest.py` (mirrors
`--strategies`/`--granularities`) after accidentally re-running all 3
instruments instead of scoping to just GBP_USD during the phase-2 re-fix.

### Phase 5: box-relative SL/TP sizing -- IN PROGRESS

Added a generic `Strategy.risk_distances(df, params, sl, tp)` hook (default:
passthrough, i.e. sl/tp stay fixed pips) to `strategies/base.py`, wired
through `optimizer._evaluate_combo` and `run_walk_forward`'s validation-window
backtest call. `OpeningRangeBreakoutStrategy` overrides it: when
`params["risk_mode"] == "box"`, sl/tp are treated as fractions of that day's
own box height (computed via the already-existing `compute_box`) instead of
pip counts. New `risk_mode` param_grid entry, default `["fixed"]` (a single
value, so it doesn't change existing combo counts) -- box mode is only ever
enabled per-instrument via `instrument_overrides`, since it needs a
completely differently-scaled `risk_search` grid (fractions like 1.5, not
pip counts like 60) that can't coexist with fixed-pip values in the same
grid. Tests: `test_risk_distances_defaults_to_fixed_pips_passthrough`,
`test_risk_distances_box_mode_scales_by_box_height`.

**Process improvement adopted here**: piloting untested dimensions with
`--skip-active-params` from now on, so a rejected experiment (like phase 3)
never touches the live config in the first place -- phase 3's mistake was
letting a bad result into `active_params` before evaluating it, which then
needed a manual restore. Piloting box-relative sizing on USD_JPY only first
(strongest, most walk-forward-validated instrument: PF>1 in 3/3 windows on
M5 in phase 1) rather than all 3 instruments at once, given phase 3's
lesson about new-dimension overfitting risk. Temporary override in
`config/settings.yaml` (`instrument_overrides.USD_JPY`): `risk_mode: ["box"]`,
`stop_loss_pips: [1.0,1.5,2.0,2.5,3.0]` and `take_profit_pips: [0.3,0.5,0.7,0.9,1.2]`
(now box-height fractions) -- **replaces**, not adds to, USD_JPY's normal
fixed-pip override (`take_profit_pips: [40,50,70,90,120]`); revert to that
if box mode doesn't win.

**Result: rejected.** `logs_phase5_box_relative.txt` -- PF dropped on both
granularities (M5 1.29->0.86, M1 1.04->0.85), P&L went negative on both
(-£18.27, -£28.92), trade count roughly tripled (132->394, 130->273) with
a *higher* win rate (~62% vs ~44%) -- the classic "many small wins wiped
out by fewer larger losses" signature of a stop too tight for the
instrument's real volatility. Box-height fractions of 1.0-3.0x didn't
capture what USD_JPY actually needs the way the wide fixed-pip stops
(40-120 pips) do. Since this ran with `--skip-active-params`, nothing live
needed restoring -- just reverted `config/settings.yaml`'s USD_JPY override
back to its fixed-pip form. `risk_distances`/`risk_mode` infrastructure is
kept (harmless at default "fixed", and available if a future strategy or
instrument wants it), but ORB's box-relative sizing itself is a dead end
for now.

### Phase 6: long-only/short-only filtering per instrument -- DONE, kept for USD_JPY

Diagnostic first (no new backtest needed): split each instrument's existing
`backtest_trades` by direction across all 3 windows already computed in
phase 1. Result:
- **USD_JPY: short beats long in all 6 window x granularity combos** --
  long PF 0.51-1.02, short PF 0.99-1.66, every single time, on both M1 and
  M5. A real, consistent structural bias, not noise.
- EUR_USD: leans long in 2/3 windows, but weakly and inconsistently (near
  ties in places).
- GBP_USD: **flips direction between windows** (long favored in the two
  phase-1 windows, short favored in the current window) -- not a real
  signal, just noise.

Implemented `direction_filter` param ("both"/"long"/"short") in
`opening_range_breakout.py`: blocks the disallowed side's *entry* condition
(pushes that side's box level to +-inf, preserving NaN so the day-reset
logic is untouched) while leaving `box_mid`/`box_height` -- and therefore
the *allowed* side's exit level -- computed from the real, unfiltered box.
Getting this right required care: naively filtering the returned signal
array post-hoc would desync the internal position state machine from what
it's actually reporting. Tests:
`test_direction_filter_long_blocks_short_entries`,
`test_direction_filter_short_blocks_long_entries`,
`test_direction_filter_short_still_allows_short_entries_and_exits`.

Piloted short-only on USD_JPY only (`--skip-active-params`,
`logs_phase6_short_only.txt`) -- **clear win, kept**: PF 1.29->1.68 (M5),
1.04->1.34 (M1), P&L +£25.01->+£39.79 (M5) and +£5.35->+£27.88 (M1), trade
counts still reasonable (70/112, not thin-sample). Matches almost exactly
what the diagnostic predicted (PF 1.66/1.35), confirming it's a real effect
of freeing the optimizer to fully re-tune under the short-only constraint,
not a fluke. Promoted directly from `backtest_runs` into `active_params`
(same zero-recomputation technique as the phase-3 revert). Not applied to
EUR_USD/GBP_USD given their weaker/inconsistent direction signal.

**Current active_params, all 6 combos, final state**:

| Instrument | Gran | PF | Notes |
|---|---|---|---|
| USD_JPY | M5 | **1.68** | short-only, box_minutes=15 |
| USD_JPY | M1 | 1.34 | short-only, box_minutes=15 |
| GBP_USD | M5 | 1.01 | box_minutes=60 |
| EUR_USD | M1 | 0.92 | box_minutes=60 |
| EUR_USD | M5 | 0.83 | box_minutes=30 |
| GBP_USD | M1 | 0.79 | box_minutes=60 |

Remaining backlog (ranked, from the original 11): 7 (volatility regime
filter), 8 (time-based/trailing exits), 9 (day-of-week/news filtering),
10 (alternative ORB variants), 11 (cross-instrument portfolio logic).

### Phase 2: per-instrument session_open_hour incl. Asian session -- DONE, mixed with an important catch

Added `session_open_hour: [0, 8, 13]` to `OpeningRangeBreakoutStrategy.param_grid`
(0 = Tokyo/Asian session, approximated as midnight Europe/London local time,
same DST caveat as the existing 8/13 options -- see code comment).
`logs_phase2_asian_session.txt`:

| Instrument | Gran | Trades | PF | Sharpe | Chosen hour |
|---|---|---|---|---|---|
| USD_JPY | M1 | 132 | 1.24 | 0.72 | **0 (Asian)** -- kept |
| USD_JPY | M5 | 130 | 1.19 | 1.88 | 8 (unchanged) |
| GBP_USD | M1 | 17 | 2.55 | 7.94 | 0 (Asian) -- **rejected** |
| GBP_USD | M5 | 11 | 2.77 | 7.94 | 0 (Asian) -- **rejected** |
| EUR_USD | M1/M5 | 59/66 | 0.76/0.83 | -1.63/-1.05 | 8 (unchanged) |

**Caught before it stuck**: GBP_USD picked the Asian session and produced
only 11-17 validation trades -- the same trade-starvation overfitting trap
as EUR_GBP's `min_box_height_pips` issue earlier, not a real edge (PF 2.77
/ Sharpe 7.94 on 11 trades is noise). This had already overwritten
`active_params` with the unreliable config. Fixed by adding a GBP_USD-only
`instrument_overrides.GBP_USD.strategy_params.session_open_hour: [8, 13]`
(excludes Asian session for GBP_USD specifically; USD_JPY keeps it),
then re-ran to restore a trustworthy `active_params` entry
(`logs_phase2_gbpusd_refix.txt`): GBP_USD M5 201 trades PF 0.93 (pnl -8.19),
M1 268 trades PF 0.79 (pnl -28.73).

**Don't over-read the corrected GBP_USD M5 number as a regression** -- this
run's window (`--years 0.5`, now exact thanks to the boundary fix) is
2026-07-19->09-17, not round 3's 2026-07-01->09-01, so it's a different
2-month slice, not a same-window comparison. Phase 1 already showed GBP_USD
M5 varies PF 1.07-1.13 across 3 independent windows; a 4th window landing at
0.93 is within that kind of window-to-window variance, not evidence Asian
session's exclusion hurt anything. GBP_USD M1's -28.73/PF 0.79 is,
if anything, further confirmation of phase 1's finding that GBP_USD M1 is a
consistent loser.

**Process note**: accidentally re-ran all 3 instruments instead of scoping
to just GBP_USD for the re-fix, because `run_backtest.py` had no
`--instruments` flag -- added one (mirrors `--strategies`/`--granularities`)
so this doesn't happen again.

### Phase 3: per-instrument exit_buffer_frac/min_box_height_pips -- DONE, reverted

Widened `exit_buffer_frac` from `[0.0,0.25,0.4]` to `[0.0,0.15,0.25,0.4,0.5]`
and `min_box_height_pips` from `[0.0,8.0]` to `[0.0,4.0,8.0,12.0]` (~3.3x
more combos per job) and re-ran the same window (`logs_phase3_box_exit_grid.txt`):

| Instrument | Gran | Phase 2 PF | Phase 3 PF | Chosen exit_buf / min_box |
|---|---|---|---|---|
| USD_JPY | M5 | 1.19 | 0.96 | 0.4 / **12.0 (new max)** |
| USD_JPY | M1 | 1.24 | 0.89 | **0.5 (new max)** / 12.0 (new max) |
| EUR_USD | M1 | 0.76 | 0.79 | 0.15 / 8.0 |
| EUR_USD | M5 | 0.83 | 0.77 | **0.5 (new max)** / 4.0 |
| GBP_USD | M1 | 0.79 | 0.64 | **0.5 (new max)** / **12.0 (new max)** |
| GBP_USD | M5 | 0.93 | 0.77 | **0.5 (new max)** / **12.0 (new max)** |

**Uniform regression, not a mixed result**: every instrument/granularity got
worse, and nearly every one pinned to the new grid's max on one or both
params, with trade counts dropping hard (USD_JPY M5 130->83, GBP_USD M1
268->67). Reading: tripling the combo count let the optimizer fit
training-window noise rather than finding real edge -- the same
overfitting failure mode as the original `total_pnl` objective bug from
earlier in the session, just triggered by grid size instead of the
objective function. **Rejected and reverted same day**: both param_grid
entries back to their phase-2 values in `opening_range_breakout.py`.

**How the revert worked without re-running the backtest**: `active_params`
is upsert-only (see its own doc-comment) and had already been overwritten
by phase 3's bad results, but `backtest_runs` is append-only and still had
phase 2's exact rows (ids 428-433, right before phase 3's 434-439) sitting
in the database untouched. Restored `active_params` by copying those 6 rows
back in directly via SQL -- zero recomputation needed. Lesson: `active_params`
being "just a pointer" cuts both ways -- convenient for evolving live
config, but it means there's no automatic undo; the fix each time is
knowing `backtest_runs` is the real history and active_params can always be
rebuilt from it by id/timestamp.

## ORB-only improvement roadmap (started 2026-09-22)

User asked for phases 6, then 1-5, in order, reporting back after each.
Status:

6. **Dedicated risk grid for ORB** — IN PROGRESS. DB check showed the shared
   grid was still pinning to its boundary for ORB specifically:
   `stop_loss_pips=80` won 35/44 windows, and USD_JPY's `take_profit_pips=20`
   won 8/11 — both the grid's old max. Widened `risk_search` in
   `config/settings.yaml` to `stop_loss_pips: [40,60,80,100,120]`,
   `take_profit_pips: [10,15,20,25,30]` (dropped 15/25 SL — never won).
   Full 5yr M5 re-run was started, then **stopped by user request mid-run**
   (completed EUR_USD's 11 windows, partial GBP_USD). Those partial/old-grid
   rows need cleanup (see "Known gaps") before re-running clean.
1. Fix EUR_GBP (per-instrument `min_box_height_pips`) — not started.
2. Test ORB on M1 — not started.
3. Session/timezone refinement (Asian/NY boxes) — not started.
4. Volatility regime filter (skip abnormal-ATR days) — not started.
5. Restrict to winning instrument/direction combo — not started.

## Known gaps / natural next steps

- ~~DB cleanup blocked by a permission classifier~~ **Done 2026-09-22** (user
  approved retry). `data/scalper.db`'s `backtest_runs` now has only
  `opening_range_breakout`/M5, 11 windows x 4 instruments — the original
  tuned baseline (`id < 369`); old strategies and the interrupted phase-6
  duplicate rows are gone.
- **EUR_GBP is unexplained.** It's the weakest ORB result (-2.54%) in every
  version of this experiment, and nobody's dug into *why* (likely: it's the
  lowest-volatility pair traded, so its opening ranges are small and the
  EUR_USD-tuned settings — box height filter especially — may not transfer).
  Phase 1 above.
- **ORB has never been run on M1**, only M5. Phase 2 above.
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

## Backlog items 7-11 (2026-09-24)

Prior session (2026-09-22) committed and pushed as `f2bdd92` (phases 1-6:
EUR_GBP drop, per-instrument grids, box_minutes, USD_JPY short-only). User
then said "carry on with items 7-11" — the remaining ranked backlog:
volatility regime filter, time-based/trailing exits, day-of-week filtering,
alternative ORB variants, cross-instrument portfolio logic. Each piloted the
same way as phases 2/4/6: add the new dimension to `OpeningRangeBreakoutStrategy`
(default single value, so it doesn't change existing combo counts unless
deliberately widened), pilot via a temporary `instrument_overrides.USD_JPY`
grid widening (USD_JPY chosen as the most walk-forward-validated instrument,
same reasoning as phase 5), decide keep/drop from real OOS results, revert
the temp override if rejected. Pilot window: the same `--years 0.55
--train-months 4 --validate-months 2` 6-month slice (2026-07-01 -> 2026-09-17,
2 windows) used throughout the per-instrument tuning work, baseline =
trades=68 pf=1.53 pnl=28.28 sharpe=1.72 max_dd=-9.81 (USD_JPY M5, current
settings, no new dimension enabled).

New `OpeningRangeBreakoutStrategy` params added for this (all default to a
no-op single value in the base `param_grid`, see the code comments there):
`excluded_weekdays`, `vol_filter_mode`/`vol_lookback_days`/`vol_low_ratio`/
`vol_high_ratio`, `entry_mode`, `max_hold_bars`, `trailing_exit_pips`.
`max_hold_bars`/`trailing_exit_pips` are implemented entirely inside
`_orb_positions` (the numba signal loop), not the backtest engine -- keeps
"strategy decides direction, engine decides execution timing" intact, and
avoids threading two more dimensions through `optimizer.py`'s risk-grid
machinery. Tests: `test_excluded_weekdays_skips_that_days_box`,
`test_vol_filter_skips_abnormally_wide_box_day`,
`test_vol_filter_baseline_survives_a_nan_height_day`,
`test_entry_mode_retest_waits_for_pullback_before_entering`,
`test_max_hold_bars_force_exits_after_n_bars`,
`test_trailing_exit_pips_exits_on_giveback`.

**Bug caught and fixed during phase 7**: `_volatility_regime_mask`'s rolling
baseline (median box height over the trailing `vol_lookback_days` days) was
*permanently* NaN on real data -- some calendar days (weekend slivers with too
few bars to ever fill the box window) have a NaN box height, recurring
roughly weekly. `rolling(lookback_days, min_periods=lookback_days)` requires
zero NaNs in a `lookback_days`-sized window to produce a value; since these
NaN days recur more often than `lookback_days` apart, *every* window failed
the count, silently disabling the whole filter (confirmed: first pilot run's
result was byte-identical to baseline). Fixed by dropping NaN-height days
before rolling (`_volatility_regime_mask`, `valid_heights = daily_heights.dropna()`)
-- those days are already excluded from trading anyway via NaN box
high/low, so this only changes what the *other* days' baseline is built
from. `test_vol_filter_baseline_survives_a_nan_height_day` regression-tests
this specifically (the earlier vol-filter test used only fully-formed days
and would not have caught it).

### Phase 7: volatility regime filter -- REJECTED

After the NaN-baseline fix, forced `vol_filter_mode=atr_ratio` (default
`vol_lookback_days=10`, `vol_low_ratio=0.4`, `vol_high_ratio=2.5`) on USD_JPY
M5 and re-ran the pilot window: trades 68->51 (-25%), pnl 28.28->20.47
(-28%), sharpe 1.72->1.29, max_dd -9.81->-12.04 (worse). PF barely moved
(1.53->1.54). Filtering out "abnormal" days cut volume and hurt risk-adjusted
return with no offsetting benefit -- rejected, override reverted.

### Phase 8: time-based / trailing exits -- REJECTED (both)

Piloted as searched options (grid `[0, 24, 48, 96]` bars and `[0.0, 10.0,
20.0, 40.0]` pips respectively, 0/0.0 = disabled, included specifically so
the optimizer could reject them) rather than forced values, letting training
choose per window like `session_open_hour`. Both pilots came back **byte-
identical to baseline** (same 68 trades, same PF 1.53, same £28.28) --
the optimizer never once chose a non-zero value in either walk-forward
window. Strongest possible rejection signal available from this method:
given the choice, training itself never wanted either feature. Box-mid
reversion + the existing fixed SL/TP already dominates every time/trailing
variant tried for USD_JPY M5. Both overrides reverted (never landed
permanently since they were tested as temporary grid widenings only).

### Phase 9: day-of-week filtering -- KEPT (as a searched option, USD_JPY only)

Piloted `excluded_weekdays: [[], [4]]` (Friday) as a searched option, same
method as phase 8. Unlike phase 8, this showed a real effect: both windows
in the pilot slice chose `[4]` in training and improved substantially OOS
(window 1 PF 1.64->2.09 pnl 27.88->30.71, window 2 PF 1.04->1.90 pnl
0.40->10.65; combined stitched PF 1.53->2.04, pnl 28.28->41.35 (+46%),
sharpe 1.72->1.83, though max_dd worsened -9.81->-14.02).

**Second-window check (independent, non-overlapping) before trusting it**:
`--start 2025-06-01 --end 2025-12-01` (validate window 2025-10-01 ->
2025-11-30) -- training chose `excluded_weekdays=[]` (opted out) and got PF
0.84 OOS. At first glance a regression, but the optimizer had the *same*
choice available as the other two windows and didn't use `[4]` here --
meaning this window's weak result reflects that window's OOS conditions
generally (consistent with phase 1's original finding that USD_JPY M5 varies
PF 1.04-1.58 across windows), not a cost imposed by having the option. With
the dimension available, training only ever uses it when it actually helps
in-sample; it never made a window *worse* by forcing an unwanted filter.

**Decision: kept as a permanent, searched (not forced) `instrument_overrides.USD_JPY`
entry** -- same treatment as `session_open_hour`/`box_minutes`, not a
blanket "always skip Friday" rule. Not tried for EUR_USD/GBP_USD this pass.

### Phase 10: alternative ORB variants -- retest entries REJECTED, multiple boxes/day not attempted

Implemented `entry_mode="retest"` in `_orb_positions`: a breakout only arms
entry, which fires once price pulls back to (re)touch the broken level
(state machine: `pending_dir`, see the function's docstring). Scoped to this
one variant for phase 10 -- "multiple boxes/day" (e.g. a second London+NY
box) was not attempted this pass, left for a future session.

Piloted as a searched option (`entry_mode: ["breakout", "retest"]`) same
method as phases 8/9. Window 1 chose `retest` in training and it collapsed
OOS: PF 2.09->1.37, pnl 30.71->3.18, trades 39->19. Window 2 didn't use it
(picked `breakout`, unchanged). Combined stitched result worse than without
it (PF 2.04->1.68, pnl 41.35->13.83). Same overfit-to-training-noise
signature as phase 3's rejected grid widening -- the dimension gave the
optimizer a way to fit window 1's training noise that didn't generalize.
**Rejected, override reverted.** `entry_mode`/`_orb_positions`' retest state
machine kept as infrastructure (harmless at default "breakout"), same
treatment as phase 5's `risk_mode`/`risk_distances`.

### Phase 11: cross-instrument portfolio logic -- scoped down, naive combination REJECTED

The real single-instrument backtest engine (`run_backtest`/`_simulate`) has
no concept of shared capital or simultaneous multi-instrument positions --
`max_concurrent_positions` in `risk_search` is explicitly documented as a
live-only knob the backtest doesn't model. Building an actual shared-capital,
concurrent-position portfolio engine was out of scope for this pass; instead
ran a scoped diagnostic: combined the three instruments' independently-
computed OOS trade streams (same pilot window, same run, fresh
`--skip-active-params` run of all three: USD_JPY pf=2.04 pnl=41.35,
GBP_USD pf=1.09 pnl=9.11, EUR_USD pf=0.80 pnl=-8.96) into one shared,
sequential equity curve (ad-hoc script, not committed -- see chat), as if
one account traded all three with the current per-instrument params and no
allocation weighting.

Result: **naive equal-weight combination does not help**.
USD_JPY+GBP_USD: pnl 50.46 (higher, since GBP_USD is mildly positive) but
sharpe roughly flat (1.83->1.80) and max_dd roughly doubles (-14.02->-32.05).
USD_JPY+EUR_USD: pnl drops to 32.39, sharpe drops to 1.23, max_dd worsens to
-19.91 (EUR_USD's negative edge just drags on the good instrument). All
three combined: pnl 41.50 (GBP_USD's gain and EUR_USD's loss roughly cancel,
netting out near USD_JPY alone), sharpe drops to 1.36, max_dd worsens to
-44.90 (>3x USD_JPY alone). Blending a strong, validated edge (USD_JPY) with
weaker/negative ones at equal position size dilutes risk-adjusted return
without a compensating diversification benefit in this window -- there's no
evidence here that trading all three together beats trading USD_JPY M5 alone.
**Rejected as tested.** If portfolio logic is revisited, the lever most
likely to matter is risk-weighted sizing (scale each instrument's position
by its own edge strength, not a flat 1000 units each) rather than equal-
weight blending, and it would need a real shared-capital engine change (not
just this trade-stream-concatenation diagnostic) to be trustworthy.

## Status after phases 7-11

Net effect of this pass: one durable addition (`excluded_weekdays` as a
searched USD_JPY option, phase 9), everything else rejected after real
testing (not skipped). `active_params` was **not** touched this pass (every
run used `--skip-active-params`) -- it still reflects the 2026-09-22
phase-6 end state. A full walk-forward re-run (without `--skip-active-params`)
to let `excluded_weekdays` actually reach `active_params`, and to re-validate
phase 9 across the full 5yr multi-window history rather than just the 2
pilot windows checked here, is a natural next step.
