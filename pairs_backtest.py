"""
pairs_backtest.py  v6.1
========================
Architecture: Fixed OLS hedge ratio.

Why previous versions failed:
  v1-v5 used Kalman filter to build the spread.
  The Kalman filter updates beta every bar to
  MINIMISE the spread residual. This means the
  spread reverts to zero by mathematical construction,
  not by market behaviour. Win rate approaches 100%
  regardless of market conditions. This is circular
  and not a real edge.

This version:
  1. Fits OLS on training bars only (beta is FIXED)
  2. Applies fixed beta to test bars (true OOS)
  3. Z-score uses training mean/std only (no leakage)
  4. Windows do not overlap (step = test_bars)
  5. OLS via numpy lstsq (no .iloc bug)

Expected realistic results:
  Trades:    30-200 per pair
  Win rate:  45-65%
  PF:        0.9-2.0
  Sharpe:    0.1-1.2
  Avg hold:  40-150 bars
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from typing      import List, Optional
import json, os, traceback

from pairs_research import (
    load_price_series,
    align_series,
    compute_half_life,
)

OUTPUT_DIR = "pairs_artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)
BT_VERSION = "backtest-v6.1"

# ── Parameters ────────────────────────────────────────────────
ENTRY_Z    = 2.0    # enter when |z| exceeds this
EXIT_Z     = 0.5    # exit when z returns to this
STOP_Z     = 3.5    # stop loss when |z| exceeds this
TRAIN_BARS = 2016   # ~3 months H1: OLS estimation
TEST_BARS  = 1008   # ~6 weeks H1: OOS trading
MAX_TRADES = 300    # sanity gate per pair
MIN_BT_PF  = 1.0   # minimum OOS profit factor


# ─────────────────────────────────────────────────────────────
#  TRADE RECORD
# ─────────────────────────────────────────────────────────────
@dataclass
class Trade:
    entry_bar    : int
    entry_time   : pd.Timestamp
    direction    : int           # +1 long, -1 short
    entry_spread : float
    entry_z      : float
    spread_std   : float         # training spread std
    exit_bar     : int           = 0
    exit_time    : pd.Timestamp  = None
    exit_spread  : float         = 0.0
    exit_z       : float         = 0.0
    exit_reason  : str           = ""
    pnl_raw      : float         = 0.0
    pnl_norm     : float         = 0.0
    bars_held    : int           = 0


# ─────────────────────────────────────────────────────────────
#  OLS via numpy — no statsmodels, no .iloc bug
# ─────────────────────────────────────────────────────────────
def estimate_ols(s1_train: np.ndarray,
                 s2_train: np.ndarray
                 ) -> tuple:
    """
    Fit log(s1) = beta*log(s2) + alpha + e
    using numpy least squares.

    Returns (beta, alpha, spread_std).

    Why numpy and not statsmodels:
      When numpy arrays are passed to statsmodels OLS,
      m.params is returned as a numpy array.
      Calling .iloc[0] on a numpy array raises:
        AttributeError: numpy.ndarray has no attribute iloc
      numpy lstsq avoids this entirely.
    """
    log1 = np.log(s1_train.astype(float))
    log2 = np.log(s2_train.astype(float))
    n    = len(log1)

    if n < 30:
        raise ValueError(
            f"Too few training bars: {n}")

    # Design matrix: [log2 | 1]
    X = np.column_stack([log2, np.ones(n)])

    # Solve normal equations
    coeffs, _, _, _ = np.linalg.lstsq(
        X, log1, rcond=None)

    beta  = float(coeffs[0])
    alpha = float(coeffs[1])

    # In-sample residuals
    residuals = log1 - beta * log2 - alpha
    std = float(np.std(residuals, ddof=1))
    if std < 1e-10:
        std = 1.0

    return beta, alpha, std


# ─────────────────────────────────────────────────────────────
#  APPLY FIXED RATIO TO TEST BARS
# ─────────────────────────────────────────────────────────────
def apply_fixed_spread(s1_test: np.ndarray,
                       s2_test: np.ndarray,
                       beta:    float,
                       alpha:   float
                       ) -> np.ndarray:
    """
    Apply training beta/alpha to test prices.
    This spread is genuinely OOS:
      - beta was estimated before seeing any test bar
      - the spread is NOT the fitting residual
      - mean reversion must come from the market
    """
    log1 = np.log(s1_test.astype(float))
    log2 = np.log(s2_test.astype(float))
    return log1 - beta * log2 - alpha


# ─────────────────────────────────────────────────────────────
#  Z-SCORE — training stats only, no future leak
# ─────────────────────────────────────────────────────────────
def compute_test_zscore(test_spread:  np.ndarray,
                        train_spread: np.ndarray
                        ) -> np.ndarray:
    """
    Standardise test spread using training mean/std.

    Why training stats:
      Using rolling test stats would still leak
      because future test bars shift the window mean.
      Training mean/std are fixed before test starts.
      This is the only correct OOS z-score method.
    """
    mu  = float(np.mean(train_spread))
    sig = float(np.std(train_spread, ddof=1))
    if sig < 1e-10:
        sig = 1.0
    return (test_spread - mu) / sig


# ─────────────────────────────────────────────────────────────
#  CORE BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────
def backtest_pair(sym1:       str,
                  sym2:       str,
                  s1:         pd.Series,
                  s2:         pd.Series,
                  entry_z:    float = ENTRY_Z,
                  exit_z:     float = EXIT_Z,
                  stop_z:     float = STOP_Z,
                  train_bars: int   = TRAIN_BARS,
                  test_bars:  int   = TEST_BARS,
                  ) -> tuple:
    """
    Walk-forward backtest with fixed OLS hedge ratio.

    Window structure (no overlap):
      |<── train_bars ──>|<── test_bars ──>|
      Fit OLS here        Trade here (OOS)

    After each window: train_start += test_bars
    This ensures zero overlap between test windows.
    """
    s1v   = s1.values.astype(float)
    s2v   = s2.values.astype(float)
    times = s1.index
    n     = len(s1v)

    print(f"\n  ── {sym1}/{sym2}  ({n:,} bars) ──")
    print(f"    Train={train_bars} bars  "
          f"Test={test_bars} bars  "
          f"(step=test, zero overlap)")

    # Validate prices
    if np.any(s1v <= 0) or np.any(s2v <= 0):
        raise ValueError(
            "Non-positive prices detected")

    trades:    List[Trade] = []
    eq_times:  list        = []
    eq_vals:   list        = []
    cum_pnl_n  = 0.0

    window_n    = 0
    train_start = 0

    while (train_start + train_bars +
           test_bars <= n):
        window_n  += 1
        train_end  = train_start + train_bars
        test_end   = min(
            train_end + test_bars, n)
        test_len   = test_end - train_end

        if test_len < 10:
            train_start += test_bars
            continue

        # ── Step 1: OLS on training bars ─────────
        try:
            beta, alpha, train_std = estimate_ols(
                s1v[train_start: train_end],
                s2v[train_start: train_end])
        except Exception as e:
            print(f"    Win {window_n}: "
                  f"OLS failed: {e}")
            train_start += test_bars
            continue

        # ── Step 2: Training spread ───────────────
        log1_tr  = np.log(
            s1v[train_start: train_end])
        log2_tr  = np.log(
            s2v[train_start: train_end])
        train_sp = (log1_tr -
                    beta * log2_tr - alpha)

        # Half-life from training spread
        hl = compute_half_life(
            pd.Series(train_sp))
        if not np.isfinite(hl) or hl <= 0:
            hl = 30.0
        hl = float(np.clip(hl, 2.0, 198.0))

        # Skip window if spread not mean-reverting
        if hl >= 198.0:
            print(f"    Win {window_n}: "
                  f"HL>=198 (not mean-reverting) "
                  f"-> skip")
            train_start += test_bars
            continue

        # Minimum hold based on half-life
        min_hold = int(
            np.clip(hl / 2, 6, 60))

        # ── Step 3: OOS test spread ───────────────
        test_sp = apply_fixed_spread(
            s1v[train_end: test_end],
            s2v[train_end: test_end],
            beta, alpha)

        # ── Step 4: Z-score (training stats) ─────
        test_z = compute_test_zscore(
            test_sp, train_sp)

        t0 = times[train_end]
        t1 = times[test_end - 1]
        print(f"    Win {window_n}: "
              f"{t0.date()} -> {t1.date()}  "
              f"B={beta:.4f}  "
              f"HL={hl:.1f}  "
              f"Std={train_std:.6f}",
              end="")

        # ── Step 5: Trade the test window ─────────
        position:      int             = 0
        current_trade: Optional[Trade] = None
        window_trades  = 0

        for li in range(test_len):
            gi = train_end + li
            sp = float(test_sp[li])
            z  = float(test_z[li])
            t  = times[gi]

            # ── Entry ─────────────────────────────
            if position == 0:
                if z < -entry_z:
                    # Long spread:
                    # spread too low, expect rise
                    position      = 1
                    current_trade = Trade(
                        entry_bar    = gi,
                        entry_time   = t,
                        direction    = 1,
                        entry_spread = sp,
                        entry_z      = z,
                        spread_std   = train_std)
                    window_trades += 1

                elif z > entry_z:
                    # Short spread:
                    # spread too high, expect fall
                    position      = -1
                    current_trade = Trade(
                        entry_bar    = gi,
                        entry_time   = t,
                        direction    = -1,
                        entry_spread = sp,
                        entry_z      = z,
                        spread_std   = train_std)
                    window_trades += 1

            # ── Exit: Long ────────────────────────
            elif position == 1:
                bars_in = (gi -
                           current_trade.entry_bar)
                pnl_raw = (sp -
                           current_trade.entry_spread)
                pnl_n   = pnl_raw / train_std

                if bars_in < min_hold:
                    # Too soon — record unrealised
                    eq_times.append(t)
                    eq_vals.append(
                        cum_pnl_n + pnl_n)
                    continue

                reason = None
                if z >= exit_z:
                    # Spread returned toward mean
                    reason = "MEAN_CROSS"
                elif z < -stop_z:
                    # Spread moved further against us
                    reason = "STOP_LOSS"
                elif li == test_len - 1:
                    # End of test window
                    reason = "END_OF_WINDOW"

                if reason:
                    current_trade.exit_bar    = gi
                    current_trade.exit_time   = t
                    current_trade.exit_spread = sp
                    current_trade.exit_z      = z
                    current_trade.exit_reason = reason
                    current_trade.pnl_raw     = pnl_raw
                    current_trade.pnl_norm    = pnl_n
                    current_trade.bars_held   = bars_in
                    trades.append(current_trade)
                    cum_pnl_n    += pnl_n
                    position      = 0
                    current_trade = None
                else:
                    # Still in trade — unrealised
                    eq_times.append(t)
                    eq_vals.append(
                        cum_pnl_n + pnl_n)
                    continue

            # ── Exit: Short ───────────────────────
            elif position == -1:
                bars_in = (gi -
                           current_trade.entry_bar)
                pnl_raw = (current_trade.entry_spread
                           - sp)
                pnl_n   = pnl_raw / train_std

                if bars_in < min_hold:
                    eq_times.append(t)
                    eq_vals.append(
                        cum_pnl_n + pnl_n)
                    continue

                reason = None
                if z <= exit_z:
                    reason = "MEAN_CROSS"
                elif z > stop_z:
                    reason = "STOP_LOSS"
                elif li == test_len - 1:
                    reason = "END_OF_WINDOW"

                if reason:
                    current_trade.exit_bar    = gi
                    current_trade.exit_time   = t
                    current_trade.exit_spread = sp
                    current_trade.exit_z      = z
                    current_trade.exit_reason = reason
                    current_trade.pnl_raw     = pnl_raw
                    current_trade.pnl_norm    = pnl_n
                    current_trade.bars_held   = bars_in
                    trades.append(current_trade)
                    cum_pnl_n    += pnl_n
                    position      = 0
                    current_trade = None
                else:
                    eq_times.append(t)
                    eq_vals.append(
                        cum_pnl_n + pnl_n)
                    continue

            # Realised equity point
            eq_times.append(t)
            eq_vals.append(cum_pnl_n)

        print(f"  -> {window_trades} trades")

        # Advance — zero overlap guaranteed
        train_start += test_bars

    # ── Build equity curve ────────────────────────
    if eq_times:
        equity = pd.Series(
            eq_vals,
            index=pd.DatetimeIndex(eq_times))
        equity = equity[
            ~equity.index.duplicated(
                keep='last')]
        equity = equity.sort_index()
        # Convert: 1 sigma-unit = 3% equity move
        equity = 1.0 + equity * 0.03
        equity = equity.clip(lower=0.01)
    else:
        equity = pd.Series(
            [1.0], index=[s1.index[0]])

    return trades, equity, {
        'n_windows': window_n}


# ─────────────────────────────────────────────────────────────
#  STATISTICS
# ─────────────────────────────────────────────────────────────
def compute_stats(trades: List[Trade],
                  equity: pd.Series) -> dict:
    if not trades:
        return {}

    pnl  = np.array(
        [t.pnl_norm for t in trades])
    wins = pnl[pnl > 0]
    loss = pnl[pnl < 0]
    n    = len(pnl)
    gp   = wins.sum() if len(wins) else 0.0
    gl   = (abs(loss.sum())
            if len(loss) else 1e-10)

    ret  = equity.pct_change().dropna()
    sh   = float(
        ret.mean() / (ret.std() + 1e-10) *
        np.sqrt(252 * 24))

    rm   = equity.cummax()
    dd   = float(((equity - rm) / rm).min())

    holds = [t.bars_held for t in trades]
    exits = {}
    for t in trades:
        exits[t.exit_reason] = (
            exits.get(t.exit_reason, 0) + 1)

    return {
        'n_trades'     : n,
        'win_rate'     : round(len(wins)/n, 4),
        'profit_factor': round(gp/gl, 4),
        'sharpe'       : round(sh, 4),
        'max_dd'       : round(dd, 4),
        'avg_hold'     : round(
            float(np.mean(holds)), 1),
        'min_hold'     : int(np.min(holds)),
        'max_hold'     : int(np.max(holds)),
        'total_pnl_n'  : round(
            float(pnl.sum()), 4),
        'exit_reasons' : exits,
    }


# ─────────────────────────────────────────────────────────────
#  SANITY CHECKER
# ─────────────────────────────────────────────────────────────
def sanity_check(trades: List[Trade],
                 sym1:   str,
                 sym2:   str,
                 stats:  dict) -> bool:
    n  = len(trades)
    wr = stats.get('win_rate', 0)
    pf = stats.get('profit_factor', 0)
    ah = stats.get('avg_hold', 0)

    print(f"\n    ── SANITY: {sym1}/{sym2} ──")
    print(f"      Trades     : {n}")
    print(f"      Win rate   : {wr:.1%}")
    print(f"      PF         : {pf:.2f}")
    print(f"      Avg hold   : {ah:.1f} bars")

    failures = []

    if n > MAX_TRADES:
        failures.append(
            f"Trade count {n} > {MAX_TRADES} "
            f"-- exit logic too aggressive")

    if wr > 0.75:
        failures.append(
            f"Win rate {wr:.1%} > 75% "
            f"-- look-ahead bias likely")

    if pf > 5.0:
        failures.append(
            f"PF {pf:.2f} > 5.0 "
            f"-- unrealistic")

    if ah < 20 and n > 10:
        failures.append(
            f"Avg hold {ah:.1f} < 20 bars "
            f"-- exits too fast")

    if failures:
        print(f"      SANITY FAILURES:")
        for f in failures:
            print(f"        - {f}")
        return False

    print(f"      All checks passed")

    # Sample trades for manual inspection
    print(f"\n      Sample (first 6 trades):")
    print(f"      {'Date':>10} D "
          f"{'EnZ':>6} {'ExZ':>6} "
          f"{'Hold':>5} {'PnL':>7} Reason")
    for t in trades[:6]:
        d = '+L' if t.direction == 1 else '-S'
        print(
            f"      "
            f"{str(t.entry_time)[:10]} {d} "
            f"{t.entry_z:>6.2f} "
            f"{t.exit_z:>6.2f} "
            f"{t.bars_held:>5} "
            f"{t.pnl_norm:>7.4f} "
            f"{t.exit_reason}")

    return True


# ─────────────────────────────────────────────────────────────
#  PLOT — individual pair
# ─────────────────────────────────────────────────────────────
def plot_results(sym1:   str,
                 sym2:   str,
                 trades: List[Trade],
                 equity: pd.Series,
                 stats:  dict):
    if not trades or equity.empty:
        return
    try:
        fig = plt.figure(figsize=(16, 10))
        gs  = gridspec.GridSpec(
            2, 2, hspace=0.4, wspace=0.3)

        fig.suptitle(
            f"{sym1}/{sym2}  "
            f"n={stats['n_trades']}  "
            f"WR={stats['win_rate']:.1%}  "
            f"PF={stats['profit_factor']:.2f}  "
            f"Sharpe={stats['sharpe']:.2f}  "
            f"MaxDD={stats['max_dd']:.1%}  "
            f"AvgHold={stats['avg_hold']:.0f}b",
            fontsize=10)

        # Equity curve
        ax1 = fig.add_subplot(gs[0, :])
        equity.plot(
            ax=ax1, color='steelblue', lw=1.5)
        ax1.axhline(
            1.0, color='gray',
            ls='--', lw=0.8)
        ax1.set_title(
            "OOS Equity — Fixed OLS ratio, "
            "zero window overlap")
        ax1.set_ylabel("Equity (1 = start)")
        ax1.grid(True, alpha=0.3)

        # Mark exits on equity curve
        for t in trades[:200]:
            if t.exit_time is None:
                continue
            c = ('green'
                 if t.pnl_norm > 0 else 'red')
            try:
                ax1.axvline(
                    t.exit_time, color=c,
                    alpha=0.08, lw=0.5)
            except Exception:
                pass

        # PnL distribution
        ax2 = fig.add_subplot(gs[1, 0])
        pd.Series(
            [t.pnl_norm for t in trades]
        ).hist(ax=ax2, bins=40,
               color='steelblue',
               edgecolor='white',
               alpha=0.8)
        ax2.axvline(0, color='red', lw=1.5)
        ax2.set_title(
            "PnL Distribution (sigma-normalised)")
        ax2.set_xlabel("PnL (sigma units)")
        ax2.grid(True, alpha=0.3)

        # Hold time distribution
        ax3 = fig.add_subplot(gs[1, 1])
        pd.Series(
            [t.bars_held for t in trades]
        ).hist(ax=ax3, bins=30,
               color='darkorange',
               edgecolor='white',
               alpha=0.8)
        ax3.set_title(
            f"Hold Periods "
            f"(avg={stats['avg_hold']:.0f}b)")
        ax3.set_xlabel("Bars held (H1)")
        ax3.grid(True, alpha=0.3)

        fname = os.path.join(
            OUTPUT_DIR,
            f"bt_{sym1}_{sym2}.png")
        plt.savefig(
            fname, dpi=120,
            bbox_inches='tight')
        plt.close()
        print(f"    [PLOT] {fname}")

    except Exception as e:
        print(f"    [PLOT ERR] {e}")
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────
#  PLOT — portfolio
# ─────────────────────────────────────────────────────────────
def plot_portfolio(results: list):
    curves = [
        (r['pair'], r['equity'])
        for r in results
        if len(r.get('equity', [])) > 10
    ]
    if not curves:
        print("  [PORTFOLIO] No curves to plot")
        return

    frames = []
    for pair, eq in curves:
        eq = eq[
            ~eq.index.duplicated(keep='last')]
        eq = eq.sort_index()
        frames.append(eq.rename(pair))

    combined  = pd.concat(frames, axis=1)
    combined  = combined.ffill().fillna(1.0)
    portfolio = combined.mean(axis=1)
    portfolio = portfolio / portfolio.iloc[0]

    ret    = portfolio.pct_change().dropna()
    sharpe = float(
        ret.mean() / (ret.std() + 1e-10) *
        np.sqrt(252 * 24))
    rm     = portfolio.cummax()
    max_dd = float(
        ((portfolio - rm) / rm).min())
    tot_r  = float(portfolio.iloc[-1] - 1.0)

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 10))
    fig.suptitle(
        f"Portfolio: {len(curves)} pairs  "
        f"Sharpe={sharpe:.2f}  "
        f"MaxDD={max_dd:.1%}  "
        f"Return={tot_r:.1%}",
        fontsize=12)

    combined.plot(
        ax=axes[0], lw=0.8, alpha=0.7)
    axes[0].axhline(
        1.0, color='k', ls='--', lw=0.8)
    axes[0].set_title("Individual Pair Curves")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    portfolio.plot(
        ax=axes[1], color='darkblue', lw=2)
    axes[1].axhline(
        1.0, color='k', ls='--', lw=0.8)
    axes[1].fill_between(
        portfolio.index,
        portfolio.values, 1.0,
        where=(portfolio.values >= 1.0),
        alpha=0.3, color='green')
    axes[1].fill_between(
        portfolio.index,
        portfolio.values, 1.0,
        where=(portfolio.values < 1.0),
        alpha=0.3, color='red')
    axes[1].set_title(
        "Portfolio (equal weight)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR,
                     "portfolio.png"),
        dpi=120, bbox_inches='tight')
    plt.close()

    print(f"\n  PORTFOLIO:")
    print(f"    Pairs  : {len(curves)}")
    print(f"    Sharpe : {sharpe:.2f}")
    print(f"    Max DD : {max_dd:.1%}")
    print(f"    Return : {tot_r:.1%}")

    with open(os.path.join(
            OUTPUT_DIR,
            "portfolio_stats.json"), 'w') as f:
        json.dump({
            'version': BT_VERSION,
            'n_pairs': len(curves),
            'sharpe' : round(sharpe, 4),
            'max_dd' : round(max_dd, 4),
            'return' : round(tot_r, 4),
            'pairs'  : [p for p, _ in curves],
        }, f, indent=2)


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def run_all_backtests(timeframe: str = "H1"):
    print(f"\n{'='*60}")
    print(f"  {BT_VERSION}")
    print(f"  Architecture : Fixed OLS hedge ratio")
    print(f"  Entry        : +/-{ENTRY_Z} sigma")
    print(f"  Exit         : {EXIT_Z} sigma")
    print(f"  Stop         : +/-{STOP_Z} sigma")
    print(f"  Train        : {TRAIN_BARS} bars")
    print(f"  Test         : {TEST_BARS} bars "
          f"(step=test, zero overlap)")
    print(f"  Min OOS PF   : {MIN_BT_PF}")
    print(f"{'='*60}")

    vp_file = os.path.join(
        OUTPUT_DIR, "valid_pairs.json")
    if not os.path.exists(vp_file):
        print("[ERR] valid_pairs.json not found")
        print("      Run pairs_research.py first")
        return []

    with open(vp_file) as f:
        valid_pairs = json.load(f)

    print(f"\n  {len(valid_pairs)} valid pairs loaded")

    all_results  = []
    summary_rows = []
    min_bars     = TRAIN_BARS + TEST_BARS + 100

    for p in valid_pairs:
        sym1 = p['symbol1']
        sym2 = p['symbol2']

        s1 = load_price_series(sym1, timeframe)
        s2 = load_price_series(sym2, timeframe)

        if min(len(s1), len(s2)) < min_bars:
            print(f"  [SKIP] {sym1}/{sym2}: "
                  f"need {min_bars} bars")
            continue

        s1a, s2a = align_series(s1, s2)
        if len(s1a) < min_bars:
            print(f"  [SKIP] {sym1}/{sym2}: "
                  f"insufficient aligned bars")
            continue

        try:
            trades, equity, diag = backtest_pair(
                sym1       = sym1,
                sym2       = sym2,
                s1         = s1a,
                s2         = s2a,
                entry_z    = ENTRY_Z,
                exit_z     = EXIT_Z,
                stop_z     = STOP_Z,
                train_bars = TRAIN_BARS,
                test_bars  = TEST_BARS,
            )
        except Exception as e:
            print(f"  [ERR] {sym1}/{sym2}: {e}")
            traceback.print_exc()
            continue

        if not trades:
            print(f"  [SKIP] No trades: "
                  f"{sym1}/{sym2}")
            continue

        stats  = compute_stats(trades, equity)

        # Post-backtest profitability filter
        oos_pf = stats.get('profit_factor', 0)
        if oos_pf < MIN_BT_PF:
            print(
                f"\n  [REJECT] {sym1}/{sym2}: "
                f"OOS PF={oos_pf:.2f} < "
                f"{MIN_BT_PF} -- not profitable")
            # Save for analysis but skip portfolio
            summary_rows.append({
                'pair'         : f"{sym1}/{sym2}",
                'n_trades'     : stats['n_trades'],
                'win_rate'     : stats['win_rate'],
                'profit_factor': oos_pf,
                'sharpe'       : stats['sharpe'],
                'max_dd'       : stats['max_dd'],
                'avg_hold_bars': stats['avg_hold'],
                'sanity_ok'    : False,
                'in_portfolio' : False,
            })
            continue

        passed = sanity_check(
            trades, sym1, sym2, stats)

        plot_results(
            sym1, sym2, trades, equity, stats)

        # Save trade log CSV
        rows = [{
            'entry_time' : str(t.entry_time),
            'exit_time'  : str(t.exit_time),
            'direction'  : t.direction,
            'entry_z'    : round(t.entry_z, 4),
            'exit_z'     : round(t.exit_z, 4),
            'bars_held'  : t.bars_held,
            'pnl_raw'    : t.pnl_raw,
            'pnl_norm'   : round(t.pnl_norm, 6),
            'exit_reason': t.exit_reason,
            'spread_std' : t.spread_std,
        } for t in trades]

        pd.DataFrame(rows).to_csv(
            os.path.join(
                OUTPUT_DIR,
                f"trades_{sym1}_{sym2}.csv"),
            index=False)

        print(f"\n  == {sym1}/{sym2} ==")
        for k, v in stats.items():
            print(f"    {k:<16}: {v}")

        all_results.append({
            'pair'  : f"{sym1}/{sym2}",
            'equity': equity,
            **stats,
        })
        summary_rows.append({
            'pair'         : f"{sym1}/{sym2}",
            'n_trades'     : stats['n_trades'],
            'win_rate'     : stats['win_rate'],
            'profit_factor': stats['profit_factor'],
            'sharpe'       : stats['sharpe'],
            'max_dd'       : stats['max_dd'],
            'avg_hold_bars': stats['avg_hold'],
            'sanity_ok'    : passed,
            'in_portfolio' : True,
        })

    # ── Summary table ─────────────────────────────
    if summary_rows:
        df_s = pd.DataFrame(
            summary_rows
        ).sort_values(
            'profit_factor', ascending=False)

        print(f"\n{'='*72}")
        print(f"  BACKTEST SUMMARY  ({BT_VERSION})")
        print(f"{'='*72}")
        print(df_s.to_string(index=False))

        df_s.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "backtest_summary.csv"),
            index=False)

    # ── Portfolio ─────────────────────────────────
    if all_results:
        plot_portfolio(all_results)
    else:
        print(f"\n  [WARN] No pairs passed "
              f"PF>={MIN_BT_PF} filter")
        print(f"         Consider lowering "
              f"MIN_BT_PF or reviewing pairs")

    print(f"\n  [DONE] {BT_VERSION}")
    return all_results


if __name__ == "__main__":
    run_all_backtests(timeframe="H1")
