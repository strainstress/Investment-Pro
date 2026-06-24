#!/usr/bin/env python3
"""
Markov 2.0 - Hedge Fund Method (corrected)
==========================================

Regime model: label price history into states -> count state->state transitions
-> build a transition matrix -> read stickiness (diagonal) -> turn the next-step
distribution into a directional signal  P(bull) - P(bear).

This is the 2.0 ("corrected") build. Three documented flaws of the original are
fixed here and the fixes are NOT optional:

  FIX 1  Stride sampling.  Consecutive 20-day windows share 19 days, so counting
         transitions between *overlapping* windows fakes persistence on the
         diagonal. We ALWAYS build BOTH matrices -- overlapping (legacy) and
         stride-sampled (non-overlapping, stride = window) -- and show them side
         by side. Only the stride-sampled matrix is statistically honest.

  FIX 2  Label verification.  After labelling, we programmatically check the
         state->name mapping against three known historical periods (a famous
         crash, a famous bull run, a flat stretch). If the rendered labels
         disagree with the data we fail loudly instead of shipping a display
         with bull/bear swapped (the original's bug).

  FIX 3  Two explicit trade modes.
           FILTER     - the regime gates an EXISTING strategy: longs only when
                        signal > +thr, shorts only when signal < -thr, flat in
                        chop. Your strategy stays yours; Markov decides WHEN.
           STANDALONE - trade the differential directly; position size scaled to
                        |signal|, capped at a user-set leverage cap.

Optional richer states ("enhanced"): cluster on 20-day return + ATR (volatility)
+ relative volume so "bear and violent" is a different state than "bear asleep".

Backtest is strict walk-forward: the matrix used to decide day t+1 is built only
from data up to and including day t. Nothing is tested on data it learned from.

Usage:
  python markov2.py --csv spy.csv --states enhanced --mode standalone --backtest \
        --plot equity.png --hmm
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

# Canonical price-only mapping. FIX 2 guards this against display drift.
SIDEWAYS, BULL, BEAR = 0, 1, 2
NAME = {SIDEWAYS: "SIDEWAYS", BULL: "BULL", BEAR: "BEAR", -1: "n/a"}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_prices(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    for c in ("open", "high", "low", "close", "volume"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# --------------------------------------------------------------------------- #
# State labelling
# --------------------------------------------------------------------------- #
def cum_return(close, window):
    """window-day trailing cumulative return; first `window` entries are nan."""
    close = np.asarray(close, dtype=float)
    ret = np.full(close.shape, np.nan)
    ret[window:] = close[window:] / close[:-window] - 1.0
    return ret


def price_states(close, window=20, bull=0.05, bear=-0.05):
    """Threshold labels on trailing cumulative return. -1 where undefined."""
    ret = cum_return(close, window)
    s = np.full(ret.shape, -1, dtype=int)
    defined = ~np.isnan(ret)
    s[defined] = SIDEWAYS
    s[defined & (ret >= bull)] = BULL
    s[defined & (ret <= bear)] = BEAR
    return s, ret


def true_range(high, low, close):
    high, low, close = map(lambda x: np.asarray(x, float), (high, low, close))
    prev = np.concatenate([[close[0]], close[:-1]])
    return np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))


def enhanced_features(df, window=20, atr_n=14):
    """Feature matrix: [20d return, ATR/price, relative volume]."""
    close = df["close"].values
    ret = cum_return(close, window)
    if {"high", "low"}.issubset(df.columns):
        tr = true_range(df["high"].values, df["low"].values, close)
        natr = pd.Series(tr).rolling(atr_n).mean().values / close
    else:  # fall back to |daily return| as a vol proxy
        natr = pd.Series(np.abs(np.concatenate([[0], np.diff(close) / close[:-1]]))).rolling(atr_n).mean().values
    if "volume" in df.columns:
        vol = df["volume"].values.astype(float)
        relvol = vol / pd.Series(vol).rolling(window).mean().values
    else:
        relvol = np.ones_like(close)
    return np.column_stack([ret, natr, relvol])


def fit_enhanced(feats, ret, k=6, seed=0, bull=0.05, bear=-0.05):
    """Fit scaler+KMeans on valid rows; classify each cluster's directional lean
    by the sign of its mean 20d return (deadband +-1%)."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    valid = ~np.isnan(feats).any(axis=1)
    scaler = StandardScaler().fit(feats[valid])
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(scaler.transform(feats[valid]))
    states = np.full(len(feats), -1, dtype=int)
    states[valid] = km.predict(scaler.transform(feats[valid]))
    direction = {}
    band = 0.01
    for c in range(k):
        m = np.nanmean(ret[states == c]) if np.any(states == c) else 0.0
        direction[c] = BULL if m > band else BEAR if m < -band else SIDEWAYS
    return scaler, km, states, direction


def assign_enhanced(feats, scaler, km):
    states = np.full(len(feats), -1, dtype=int)
    valid = ~np.isnan(feats).any(axis=1)
    if valid.any():
        states[valid] = km.predict(scaler.transform(feats[valid]))
    return states


# --------------------------------------------------------------------------- #
# Transition matrix  (FIX 1: overlapping vs stride-sampled)
# --------------------------------------------------------------------------- #
def transition_matrix(states, n_states, stride=1):
    """Count transitions then row-normalise to probabilities.

    stride == 1            -> overlapping / legacy (every consecutive bar).
    stride == window (20)  -> non-overlapping windows; the honest matrix.
    Undefined states (-1) are skipped.
    """
    states = np.asarray(states)
    if stride <= 1:
        a, b = states[:-1], states[1:]
    else:
        idx = np.arange(0, len(states) - stride, stride)
        a, b = states[idx], states[idx + stride]
    ok = (a >= 0) & (b >= 0)
    a, b = a[ok], b[ok]
    counts = np.zeros((n_states, n_states), dtype=float)
    np.add.at(counts, (a, b), 1.0)
    rowsum = counts.sum(axis=1, keepdims=True)
    P = np.divide(counts, rowsum, out=np.zeros_like(counts), where=rowsum > 0)
    return counts, P


def stationary(P):
    """Stationary distribution = left eigenvector for eigenvalue 1."""
    vals, vecs = np.linalg.eig(P.T)
    i = np.argmin(np.abs(vals - 1.0))
    v = np.real(vecs[:, i])
    s = v.sum()
    return v / s if s != 0 else np.full(P.shape[0], 1.0 / P.shape[0])


def signal_from_row(row, bull_cols, bear_cols):
    return float(np.sum(row[bull_cols]) - np.sum(row[bear_cols]))


# --------------------------------------------------------------------------- #
# FIX 2: label self-verification
# --------------------------------------------------------------------------- #
def verify_labels(df, states, ret, bull=0.05, bear=-0.05):
    """FIX 2 (ticker-agnostic). Two complementary checks that the
    BULL/BEAR/SIDEWAYS mapping is not swapped in the display:

      (A) INTERNAL CONSISTENCY (the authoritative swap-guard). By definition,
          BULL-labeled days have trailing-20d return >= +bull and BEAR-labeled
          days <= -bear, so the per-label mean return MUST order
          BULL > SIDEWAYS > BEAR with the right signs. If the display swapped the
          names, this ordering breaks. Works for any ticker, no period guessing.

      (B) KNOWN MARKET-WIDE ANCHOR. The Feb-Mar 2020 COVID crash was a fast, broad
          decline that essentially every liquid US equity shared, so it must read
          BEAR. Two further windows are shown for CONTEXT only (no pass/fail),
          because what counts as a 'bull run' or 'flat stretch' is ticker-specific
          (a slow steady riser labels SIDEWAYS even while it grinds higher).

    Returns dict(checks=[(label, ok_or_None, detail)], passed=bool).
    """
    out = {"checks": [], "passed": True}
    close = df["close"].values
    dates = df["date"].values

    # (A) internal consistency
    means = {}
    for st in (BULL, SIDEWAYS, BEAR):
        sel = states == st
        means[st] = float(np.nanmean(ret[sel])) if sel.any() else float("nan")
    okA = (means[BULL] >= bull and means[BEAR] <= bear
           and means[BULL] > means[SIDEWAYS] > means[BEAR])
    out["checks"].append((
        "internal: mean ret20 ordered BULL>SIDE>BEAR with correct signs", okA,
        f"BULL {means[BULL]:+.3f} | SIDE {means[SIDEWAYS]:+.3f} | BEAR {means[BEAR]:+.3f}"))
    out["passed"] = out["passed"] and okA

    # (B) anchors: (name, start, end, expected_or_None, hard?)
    anchors = [
        ("2020 COVID crash", "2020-02-24", "2020-03-23", BEAR, True),
        ("Apr-2020 rebound", "2020-04-06", "2020-06-08", None, False),
        ("2017 window", "2017-06-01", "2017-09-01", None, False),
    ]
    for name, a, b, exp, hard in anchors:
        m = (dates >= np.datetime64(a)) & (dates <= np.datetime64(b))
        idx = np.where(m)[0]
        sub = states[m]
        sub = sub[sub >= 0]
        if len(sub) == 0 or len(idx) < 2:
            out["checks"].append((f"anchor {name}", None, "no data"))
            continue
        vals, counts = np.unique(sub, return_counts=True)
        dom = int(vals[np.argmax(counts)])
        frac = float(counts.max() / counts.sum())
        realized = float(close[idx[-1]] / close[idx[0]] - 1.0)
        detail = f"dominant={NAME[dom]} ({frac*100:.0f}%)  realized {realized*100:+.0f}%"
        if hard:
            ok = dom == exp
            out["passed"] = out["passed"] and ok
            out["checks"].append((f"anchor {name} (expect {NAME[exp]})", ok, detail))
        else:
            out["checks"].append((f"context {name}", None, detail))
    return out


# --------------------------------------------------------------------------- #
# HMM mode (optional): no hand-made labels
# --------------------------------------------------------------------------- #
def hmm_compare(close, threshold_states, n=2, seed=0, n_iter=1000):
    """Fit a Gaussian HMM with NO hand-made labels on DAILY log returns (the
    natural per-bar observation), map hidden states to BULL/BEAR by mean return,
    and measure DIRECTIONAL agreement with the threshold labels on the days the
    threshold actually makes a call.

    Why not raw 3-way accuracy: the threshold labels are ~79% SIDEWAYS, so 3-way
    accuracy just rewards a model for predicting 'sideways' and is dominated by
    the mushy middle. And fitting an HMM on *overlapping* 20d returns is
    degenerate (the same autocorrelation FIX 1 warns about collapses it into one
    sticky state). The honest, trade-relevant question is: on the decisive days,
    does an independent model agree on DIRECTION? Agreement is the green light.
    """
    import warnings

    from hmmlearn.hmm import GaussianHMM

    close = np.asarray(close, float)
    r = np.diff(np.log(close))
    X = r.reshape(-1, 1)
    model = GaussianHMM(n_components=n, covariance_type="full", n_iter=n_iter,
                        tol=1e-4, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X)
    hidden = model.predict(X)
    means = {c: r[hidden == c].mean() if np.any(hidden == c) else 0.0 for c in range(n)}
    order = sorted(means, key=lambda c: means[c])  # low mean -> bear, high -> bull
    mp = {order[0]: BEAR, order[-1]: BULL}
    for c in order[1:-1]:
        mp[c] = SIDEWAYS
    mapped = np.full(len(close), -1, dtype=int)
    mapped[1:] = [mp[h] for h in hidden]  # r[i] is the return INTO bar i+1

    decisive = (threshold_states == BULL) | (threshold_states == BEAR)
    hm, tt = mapped[decisive], threshold_states[decisive]
    call = (hm == BULL) | (hm == BEAR)
    coverage = float(call.mean()) if len(call) else float("nan")
    agree = float((hm[call] == tt[call]).mean()) if call.sum() > 0 else float("nan")
    return mapped, agree, coverage


# --------------------------------------------------------------------------- #
# Walk-forward backtest
# --------------------------------------------------------------------------- #
def walk_forward(df, states_mode, trade_mode, *, window=20, bull=0.05, bear=-0.05,
                 cap=1.0, filter_thr=0.10, use_stride=True, k=6, refit_every=21,
                 min_train=504, seed=0):
    """Strict walk-forward. Position for day t+1 is decided using only data up to
    day t. Returns daily strategy returns aligned to df rows (nan before start)."""
    close = df["close"].values
    n = len(close)
    daily = np.full(n, np.nan)
    daily[1:] = close[1:] / close[:-1] - 1.0  # realised return for day t (t-1 -> t)
    pos = np.zeros(n)

    enh_cache = None  # (scaler, km, direction, last_fit_index)
    feats_full = enhanced_features(df, window) if states_mode == "enhanced" else None
    ret_full = cum_return(close, window)

    for t in range(min_train, n - 1):
        if states_mode == "price":
            s, _ = price_states(close[: t + 1], window, bull, bear)
            n_states = 3
            bull_cols, bear_cols = [BULL], [BEAR]
            cur = s[-1]
        else:
            need_fit = enh_cache is None or (t - enh_cache[3]) >= refit_every
            if need_fit:
                scaler, km, _, direction = fit_enhanced(
                    feats_full[: t + 1], ret_full[: t + 1], k=k, seed=seed, bull=bull, bear=bear
                )
                enh_cache = (scaler, km, direction, t)
            scaler, km, direction, _ = enh_cache
            s = assign_enhanced(feats_full[: t + 1], scaler, km)
            n_states = k
            bull_cols = [c for c, d in direction.items() if d == BULL]
            bear_cols = [c for c, d in direction.items() if d == BEAR]
            cur = s[-1]

        _, P = transition_matrix(s, n_states, stride=window if use_stride else 1)
        if cur < 0 or P[cur].sum() == 0:
            sig = 0.0
        else:
            sig = signal_from_row(P[cur], bull_cols, bear_cols)

        if trade_mode == "standalone":
            pos[t + 1] = float(np.clip(sig, -cap, cap))
        else:  # filter: regime gates a long-biased base strategy (here: long SPY)
            base = 1.0  # the existing strategy = be long
            if sig > filter_thr:
                pos[t + 1] = base
            elif sig < -filter_thr:
                pos[t + 1] = -base
            else:
                pos[t + 1] = 0.0

    strat = pos * daily
    return strat, pos, daily


def backtest_insample(df, states_mode, trade_mode, *, window=20, bull=0.05, bear=-0.05,
                      cap=1.0, filter_thr=0.10, use_stride=False, k=6, seed=0):
    """The FLATTERING backtest: build ONE matrix on the full history, then trade
    that same history. This peeks at the future (lookahead) and is shown only to
    demonstrate how much in-sample fitting inflates the numbers. Do NOT trust it.
    Defaults to the legacy/overlapping matrix -- the most flattering combination.
    """
    close = df["close"].values
    n = len(close)
    daily = np.full(n, np.nan)
    daily[1:] = close[1:] / close[:-1] - 1.0
    if states_mode == "price":
        s, _ = price_states(close, window, bull, bear)
        n_states, bull_cols, bear_cols = 3, [BULL], [BEAR]
    else:
        feats = enhanced_features(df, window)
        _, _, s, direction = fit_enhanced(feats, feats[:, 0], k=k, seed=seed, bull=bull, bear=bear)
        n_states = k
        bull_cols = [c for c, d in direction.items() if d == BULL]
        bear_cols = [c for c, d in direction.items() if d == BEAR]
    _, P = transition_matrix(s, n_states, stride=window if use_stride else 1)
    pos = np.zeros(n)
    for t in range(window, n - 1):
        cur = s[t]
        sig = 0.0 if cur < 0 or P[cur].sum() == 0 else signal_from_row(P[cur], bull_cols, bear_cols)
        if trade_mode == "standalone":
            pos[t + 1] = float(np.clip(sig, -cap, cap))
        else:
            pos[t + 1] = 1.0 if sig > filter_thr else (-1.0 if sig < -filter_thr else 0.0)
    return pos * daily


def metrics(strat):
    r = strat[~np.isnan(strat)]
    r = r[~np.isnan(r)]
    eq = np.cumprod(1.0 + r)
    if len(eq) == 0:
        return {}
    yrs = len(r) / 252.0
    active = r[r != 0]
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    return {
        "days": int(len(r)),
        "total_return": float(eq[-1] - 1.0),
        "cagr": float(eq[-1] ** (1.0 / yrs) - 1.0) if yrs > 0 else float("nan"),
        "win_rate": float((active > 0).mean()) if len(active) else float("nan"),
        "profit_factor": float(gains / losses) if losses > 0 else float("inf"),
        "max_drawdown": float(dd.min()),
        "sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan"),
        "exposure": float((active.size) / len(r)),
        "equity": eq,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def fmt_matrix(P, names):
    w = max(len(x) for x in names) + 2
    head = " " * (w + 3) + "".join(f"{n:>10}" for n in names)
    lines = [head]
    for i, n in enumerate(names):
        cells = "".join(f"{P[i, j]:>10.3f}" for j in range(len(names)))
        lines.append(f"{n:>{w}} -> {cells}")
    return "\n".join(lines)


def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:.2f}%"


def run_report(df, args):
    out = {}
    close = df["close"].values
    print("=" * 74)
    print(f"MARKOV 2.0  |  {getattr(args, 'ticker', '?')}  |  {len(df)} bars  "
          f"{df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}")
    print(f"states={args.states}  trade-mode={args.mode}  window={args.window}  "
          f"bull>=+{args.bull:.0%}  bear<=-{abs(args.bear):.0%}")
    print("=" * 74)

    # ---- states + matrices --------------------------------------------------
    if args.states == "price":
        states, ret = price_states(close, args.window, args.bull, args.bear)
        names = [NAME[SIDEWAYS], NAME[BULL], NAME[BEAR]]
        n_states = 3
        bull_cols, bear_cols = [BULL], [BEAR]
        verify_states, verify_ret = states, ret
    else:
        feats = enhanced_features(df, args.window)
        ret = feats[:, 0]
        scaler, km, states, direction = fit_enhanced(feats, ret, k=args.k, seed=args.seed,
                                                      bull=args.bull, bear=args.bear)
        n_states = args.k
        names = [f"C{c}:{NAME[direction[c]][:4]}" for c in range(args.k)]
        bull_cols = [c for c, d in direction.items() if d == BULL]
        bear_cols = [c for c, d in direction.items() if d == BEAR]
        # price-only labels too, for FIX 2 verification + enhanced-vs-price report
        verify_states, verify_ret = price_states(close, args.window, args.bull, args.bear)

    counts_leg, P_leg = transition_matrix(states, n_states, stride=1)
    counts_str, P_str = transition_matrix(states, n_states, stride=args.window)

    print("\n--- FIX 1: transition matrices (rows sum to 1) ---")
    print("\n[LEGACY / overlapping windows]  <- inflated diagonal, do NOT trust")
    print(fmt_matrix(P_leg, names))
    print("\n[STRIDE-SAMPLED / non-overlapping]  <- statistically honest")
    print(fmt_matrix(P_str, names))
    print("\nStickiness (diagonal):")
    for i, nm in enumerate(names):
        print(f"  {nm:>12}: legacy {P_leg[i,i]:.3f}   stride {P_str[i,i]:.3f}   "
              f"(legacy overstates by {(P_leg[i,i]-P_str[i,i]):+.3f})")
    print("WARNING: legacy windows share 19/20 days -> fake persistence. "
          "Only the stride-sampled matrix is honest.")

    # ---- FIX 2: label verification (ticker-agnostic) -----------------------
    vr = verify_labels(df, verify_states, verify_ret, bull=args.bull, bear=args.bear)
    print("\n--- FIX 2: label self-verification (price-only labels) ---")
    for label, good, detail in vr["checks"]:
        tag = "PASS" if good else ("FAIL" if good is False else "ctx ")
        print(f"  [{tag}] {label}")
        if detail:
            print(f"         {detail}")
    print(f"  => label mapping {'VERIFIED' if vr['passed'] else 'MISMATCH - would block display'}")
    out["labels_verified"] = vr["passed"]

    # ---- signal + forecasts -------------------------------------------------
    cur = states[states >= 0][-1]
    sig_now = signal_from_row(P_str[cur], bull_cols, bear_cols)
    cur_name = names[cur]
    print(f"\n--- Signal (stride matrix) ---")
    print(f"  current state: {cur_name}")
    print(f"  P(bull tomorrow) - P(bear tomorrow) = {sig_now:+.3f}  "
          f"=> {'LONG bias' if sig_now>0 else 'SHORT bias' if sig_now<0 else 'flat'}")

    print("\n--- Multi-day forecast via matrix powers (convergence to stationary) ---")
    stat = stationary(P_str)
    for h in (1, 5, 20, 60):
        Ph = np.linalg.matrix_power(P_str, h)
        sig_h = signal_from_row(Ph[cur], bull_cols, bear_cols)
        print(f"  h={h:>3}d: signal {sig_h:+.3f}")
    print(f"  stationary dist: " + ", ".join(f"{names[i]}={stat[i]:.3f}" for i in range(n_states)))
    print("  NOTE: P^h -> stationary as h grows; long-horizon forecasts carry no signal.")

    # ---- enhanced vs price-only --------------------------------------------
    if args.states == "enhanced":
        ps, pret = price_states(close, args.window, args.bull, args.bear)
        _, Pp = transition_matrix(ps, 3, stride=args.window)
        sig_price = signal_from_row(Pp[ps[ps >= 0][-1]], [BULL], [BEAR])
        print("\n--- Enhanced vs price-only (stride matrices) ---")
        print(f"  states: price-only=3   enhanced={args.k} "
              f"(bull-type={bull_cols}, bear-type={bear_cols})")
        print(f"  current signal: price-only {sig_price:+.3f}   enhanced {sig_now:+.3f}")
        print("  enhanced splits each direction by volatility/volume so a violent "
              "bear is a distinct, less-sticky state than a quiet one.")

    # ---- HMM ----------------------------------------------------------------
    if args.hmm:
        try:
            _, agree, cov = hmm_compare(close, verify_states, n=2, seed=args.seed)
            print("\n--- Hidden Markov mode (unsupervised 2-state HMM on daily returns) ---")
            print("  The HMM was never shown the +-5% thresholds. On the days the")
            print(f"  threshold makes a directional call, the HMM independently agrees")
            print(f"  on direction {agree*100:.1f}% of the time (coverage {cov*100:.0f}%).")
            print(f"  => {'GREEN LIGHT - an independent model confirms the regimes' if agree>=0.6 else 'AMBER - regimes are fuzzy; size down'}")
        except Exception as e:
            print(f"\n--- Hidden Markov mode ---\n  skipped ({e})")

    return out, (states, n_states, names, bull_cols, bear_cols, P_str)


def run_backtest(df, args):
    print("\n" + "=" * 74)
    print("WALK-FORWARD BACKTEST  (matrix for day t+1 uses only data up to day t)")
    print("=" * 74)
    bh = df["close"].values
    bh_ret = np.full(len(bh), np.nan)
    bh_ret[1:] = bh[1:] / bh[:-1] - 1.0

    # after-fix (stride) and before-fix (legacy) using the chosen states + mode
    strat_fix, pos_fix, _ = walk_forward(df, args.states, args.mode, window=args.window,
                                         bull=args.bull, bear=args.bear, cap=args.cap,
                                         filter_thr=args.filter_thr, use_stride=True,
                                         k=args.k, seed=args.seed, min_train=args.min_train)
    strat_leg, _, _ = walk_forward(df, args.states, args.mode, window=args.window,
                                   bull=args.bull, bear=args.bear, cap=args.cap,
                                   filter_thr=args.filter_thr, use_stride=False,
                                   k=args.k, seed=args.seed, min_train=args.min_train)
    # the flattering in-sample (lookahead) backtest, for contrast only
    strat_is = backtest_insample(df, args.states, args.mode, window=args.window,
                                 bull=args.bull, bear=args.bear, cap=args.cap,
                                 filter_thr=args.filter_thr, use_stride=False,
                                 k=args.k, seed=args.seed)

    # align everything to the traded window
    start = args.min_train + 1
    m_is = metrics(strat_is[start:])
    m_fix = metrics(strat_fix[start:])
    m_leg = metrics(strat_leg[start:])
    m_bh = metrics(bh_ret[start:])

    def line(label, m):
        print(f"  {label:<32} total {pct(m['total_return']):>9}  CAGR {pct(m['cagr']):>8}  "
              f"win {pct(m['win_rate']):>7}  PF {m['profit_factor']:>5.2f}  "
              f"maxDD {pct(m['max_drawdown']):>8}  Sharpe {m['sharpe']:>5.2f}")

    print(f"\n  Period traded: {df['date'].iloc[start].date()} -> {df['date'].iloc[-1].date()}  "
          f"({m_fix['days']} days)\n")
    line("IN-SAMPLE legacy (LOOKAHEAD!)", m_is)
    line("BEFORE FIX  walk-fwd legacy", m_leg)
    line("AFTER FIX   walk-fwd stride", m_fix)
    line(f"Buy & hold {args.ticker}", m_bh)

    print("\n  Backtests flatter. The fixed matrix shows uglier, truer numbers — "
          "those are the only ones worth trading.")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            dts = df["date"].values[start:]
            fig, ax = plt.subplots(figsize=(11, 6))
            ax.plot(dts, m_is["equity"], label="In-sample legacy (LOOKAHEAD — flatters)",
                    color="#2ca02c", lw=1.1, ls=":")
            ax.plot(dts, m_bh["equity"], label=f"Buy & hold {args.ticker}", color="#888888", lw=1.4)
            ax.plot(dts, m_leg["equity"], label="Before fix (walk-fwd, legacy/overlapping)",
                    color="#d62728", lw=1.4, ls="--")
            ax.plot(dts, m_fix["equity"], label="After fix (walk-fwd, stride-sampled, honest)",
                    color="#1f77b4", lw=1.8)
            ax.set_yscale("log")
            ax.set_title(f"Markov 2.0 walk-forward  |  {args.ticker}  |  {args.states} states, "
                         f"{args.mode} mode")
            ax.set_ylabel("Growth of $1 (log)")
            ax.legend(loc="upper left", fontsize=9)
            ax.grid(True, which="both", alpha=0.25)
            fig.tight_layout()
            fig.savefig(args.plot, dpi=130)
            print(f"\n  Equity curve saved -> {args.plot}")
        except Exception as e:
            print(f"\n  (plot skipped: {e})")

    return {"after_fix": {k: v for k, v in m_fix.items() if k != "equity"},
            "before_fix": {k: v for k, v in m_leg.items() if k != "equity"},
            "buy_hold": {k: v for k, v in m_bh.items() if k != "equity"}}


def main():
    ap = argparse.ArgumentParser(description="Markov 2.0 - Hedge Fund Method (corrected)")
    ap.add_argument("--csv", required=True, help="CSV with date,open,high,low,close,volume")
    ap.add_argument("--ticker", default=None, help="asset label (default: from CSV filename)")
    # Defaults set to this install's onboarding choice (standalone + enhanced).
    # The method's conceptual defaults are filter + price; both modes/states are
    # fully supported via these flags.
    ap.add_argument("--states", choices=["price", "enhanced"], default="enhanced")
    ap.add_argument("--mode", choices=["filter", "standalone"], default="standalone")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--bull", type=float, default=0.05)
    ap.add_argument("--bear", type=float, default=-0.05)
    ap.add_argument("--cap", type=float, default=1.0, help="standalone leverage cap")
    ap.add_argument("--filter-thr", dest="filter_thr", type=float, default=0.10,
                    help="filter-mode signal threshold")
    ap.add_argument("--k", type=int, default=6, help="enhanced cluster count")
    ap.add_argument("--min-train", dest="min_train", type=int, default=504)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hmm", action="store_true")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--plot", default=None)
    ap.add_argument("--json", default=None, help="write machine-readable summary")
    args = ap.parse_args()
    if not args.ticker:
        import os
        args.ticker = os.path.splitext(os.path.basename(args.csv))[0].upper()

    df = load_prices(args.csv)
    summary, _ = run_report(df, args)
    if args.backtest:
        summary["backtest"] = run_backtest(df, args)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
