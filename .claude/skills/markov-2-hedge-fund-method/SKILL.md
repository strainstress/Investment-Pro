---
name: markov-2-hedge-fund-method
description: >-
  Markov 2.0 — Hedge Fund Method (corrected). Regime-based market analysis and
  backtesting: label an asset's history into BULL/BEAR/SIDEWAYS states, build a
  state-transition matrix, read stickiness off the diagonal, and turn the next-step
  distribution into a directional signal P(bull) - P(bear). Use this whenever the
  user wants regime analysis, a transition matrix, a regime/Markov signal, regime
  filtering of a strategy, or a walk-forward backtest of a ticker. Ships three
  non-negotiable fixes over the original method: stride sampling (kills the
  overlapping-window autocorrelation flaw), programmatic label self-verification
  (no more bull/bear swapped in a display), and two explicit trade modes
  (FILTER vs STANDALONE). Optional enhanced states (return+ATR+volume) and an
  unsupervised HMM cross-check.
---

# Markov 2.0 — Hedge Fund Method (corrected)

Same core as the original Markov hedge-fund method (states → transition matrix →
stickiness → signal), with three documented flaws fixed. **The fixes are not
optional — they are what makes this 2.0.**

## The method

1. **States.** Default: 20-day cumulative return ≥ +5% = BULL, ≤ −5% = BEAR, else
   SIDEWAYS. Label the asset's full history.
2. **Transition matrix.** Count state→state transitions, normalize rows to
   probabilities (rows sum to 1). The diagonal is **stickiness**.
3. **Signal.** `P(bull tomorrow) − P(bear tomorrow)`. Sign = direction, magnitude
   = conviction.
4. **Multi-day forecasts** via matrix powers `P^h`; note convergence to the
   stationary distribution — long-horizon forecasts carry no signal.
5. **HMM mode (optional).** Fit an unsupervised Gaussian HMM (no hand-made labels)
   and report directional agreement with the threshold labels. Agreement is the
   green light.

## The three fixes (non-negotiable)

- **FIX 1 — Stride sampling.** NEVER build the matrix from overlapping rolling
  windows: consecutive 20-day windows share 19 days, which fakes persistence on
  the diagonal. Count transitions between NON-overlapping windows (stride = window
  length, default 20 bars). The tool ALWAYS computes **both** matrices —
  overlapping (legacy) and stride-sampled (true) — and shows them side by side.
  **Only the stride-sampled matrix is statistically honest.**
- **FIX 2 — Label verification.** After labelling, programmatically verify the
  state→name mapping against three known historical periods (a famous crash, a
  famous bull run, a flat stretch). If the rendered labels disagree with the data,
  fail loudly and fix before showing the user. (The original shipped with
  bull/bear swapped in a display — never repeat that.)
- **FIX 3 — Two explicit modes.** Always ask / state which:
  - **FILTER** (default): the regime gates an *existing* strategy — longs only
    when signal > +threshold, shorts only when < −threshold, flat in chop. The
    user's strategy stays theirs; Markov 2.0 decides WHEN it may act.
  - **STANDALONE**: trade the differential directly; position size scaled to
    |signal|, capped at a user-set leverage cap.

## Optional richer states (offer, don't force)

Price-only states are the default. **Enhanced states** cluster on 20-day return +
ATR (volatility) + relative volume, so "bear and violent" ≠ "bear and asleep".
When chosen, report how the matrix and signal change vs price-only.

## How to run

The engine is `scripts/markov2.py` (numpy, pandas, matplotlib; scikit-learn for
enhanced states; hmmlearn for HMM mode).

```bash
# Filter mode, price-only states, full report:
python scripts/markov2.py --csv PRICES.csv

# Standalone, enhanced states, HMM cross-check, walk-forward backtest + plot:
python scripts/markov2.py --csv PRICES.csv --states enhanced --mode standalone \
    --hmm --backtest --plot equity.png --json summary.json
```

Flags: `--states price|enhanced` · `--mode filter|standalone` · `--window 20` ·
`--bull 0.05 --bear -0.05` · `--cap 1.0` (standalone) · `--filter-thr 0.10`
(filter) · `--k 6` (enhanced clusters) · `--min-train 504` (walk-forward warmup) ·
`--hmm` · `--backtest` · `--plot FILE` · `--json FILE`.

**CSV format:** columns `date,open,high,low,close,volume` (volume/high/low optional
for price-only; required for enhanced states' ATR and relative volume). Use
**split-adjusted** prices for backtests.

## Getting price data (no API keys)

Public CSV endpoints (Stooq, Yahoo) may be blocked by an egress policy. If a
Robinhood MCP server is connected, pull split-adjusted daily OHLCV with
`get_equity_historicals` (symbol, start_time, interval=day, adjustment_type=split).
Large responses are saved to a file; convert it with the helper:

```bash
python scripts/rh_to_csv.py ROBINHOOD_RESULT.json OUT.csv
```

## Backtest discipline

The backtest is **strict walk-forward**: the matrix used to decide day *t+1* is
built only from data up to and including day *t*. For enhanced states the
clustering is refit on a schedule on past data only. The tool also prints an
**in-sample (lookahead) row** purely to show how much in-sample fitting flatters.

## The caveat (say it every time)

> Backtests flatter. The fixed matrix shows uglier, truer numbers — those are the
> only ones worth trading.

## Not financial advice

This is a research/analysis tool. Regime signals are probabilistic, fit to past
data, and can fail. Nothing here is a recommendation to buy or sell any security.
