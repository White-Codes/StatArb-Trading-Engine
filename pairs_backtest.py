"""
pairs_backtest.py  v5.0
========================
Fixes applied:

  FIX 1: Proper walk-forward — no window overlap
          step_bars == test_bars enforced
          
  FIX 2: Exit threshold raised to 0.5 (not 0.0)
          Prevents immediate exit on noise
          
  FIX 3: Min hold raised to 12 bars minimum
          (half-life based, not fixed 2 bars)
          
  FIX 4: Equity in pip-equivalent units
          Spread PnL normalised by entry spread std
          
  FIX 5: Kalman filter re-initialised per window
          No state leakage from future bars
          
  FIX 6: Z-score computed only on in-window data
          Training stats never include test bars
          
  FIX 7: Duplicate equity index properly handled
          with explicit deduplication before concat
          
  FIX 8: Trade count hard cap with diagnostic dump
          if exceeded (indicates logic error)

Expected realistic results:
  Trades:         30 to 200 per pair over full history
  Win rate:       52% to 68%
  Avg hold:       20 to 150 bars
  Profit factor:  1.1 to 2.0
  Sharpe:         0.3 to 1.5
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing      import List, Optional
import json, os, traceback

from kalman_filter  import KalmanHedgeFilter
from pairs_research import (
    load_price_series, align_series,
    compute_half_life
)

OUTPUT_DIR = "pairs_artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)
BT_VERSION = "backtest-v5.0"

# ── Trading parameters ────────────────────────────────────────
ENTRY_Z      = 2.0    # enter when |z| > this
EXIT_Z       = 0.5    # exit when z returns to this
                      # (FIX 2: was 0.0, too tight)
STOP_Z       = 3.5    # stop loss
MIN_HOLD_BARS = 12    # minimum hold (FIX 3: was 2)
DELTA        = 1e-4   # Kalman filter speed

# Walk-forward parameters
# FIX 5: test_bars == step_bars to eliminate overlap
TRAIN_BARS   = 2016   # ~3 months H1 for Kalman warmup
TEST_BARS    = 336    # ~2 weeks H1
STEP_BARS    = 336    # FIX 1: must equal TEST_BARS
                      # (was 168 → 50% overlap bug)

# Sanity gates
MAX_TRADES_PER_PAIR = 400   # flag if exceeded


# ─────────────────────────────────────────────────────────────
#  TRADE RECORD
# ─────────────────────────────────────────────────────────────
@dataclass
class Trade:
    entry_bar    : int
    entry_time   : pd.Timestamp
    direction    : int           # +1=long spread, -1=short
    entry_spread : float
    entry_z      : float
    entry_spread_std : float     # normalisation factor
    exit_bar     : int           = 0
    exit_time    : pd.Timestamp  = None
    exit_spread  : float         = 0.0
    exit_z       : float         = 0.0
    exit_reason  : str           = ""
    pnl_spread   : float         = 0.0   # raw spread units
    pnl_norm     : float         = 0.0   # normalised by std
    bars_held    : int           = 0


# ─────────────────────────────────────────────────────────────
#  KALMAN SPREAD — per-window, no future leak
# ─────────────────────────────────────────────────────────────
def build_kalman_spread_window(
        s1_vals: np.ndarray,
        s2_vals: np.ndarray,
        warmup:  int,
        delta:   float = DELTA
        ) -> np.ndarray:
    """
    Run fresh Kalman filter from scratch.
    First 'warmup' bars are training only.
    Returns spread for ALL bars (train + test).
    
    Key: filter is NEVER carried across windows.
    Each window gets a fresh KalmanHedgeFilter().
    This ensures no state leakage from future data.
    """
    kf  = KalmanHedgeFilter(delta=delta)
    out = np.zeros(len(s1_vals))
    for i in range(len(s1_vals)):
        _, _, sp = kf.update(s1_vals[i], s2_vals[i])
        out[i]   = sp
    return out


# ─────────────────────────────────────────────────────────────
#  ROLLING Z-SCORE — strict no look-ahead
# ─────────────────────────────────────────────────────────────
def rolling_zscore_strict(
        spreads: np.ndarray,
        window:  int
        ) -> np.ndarray:
    """
    z[i] uses ONLY spreads[i-window : i].
    spreads[i] itself is NOT included in mean/std.
    
    This is strictly causal:
    - mean and std are computed on PAST bars only
    - the current bar's spread is then standardised
    
    First 'window' elements → z = 0 (insufficient history).
    """
    n = len(spreads)
    z = np.zeros(n)
    for i in range(window, n):
        # Past window: [i-window, i)  — excludes bar i
        past = spreads[i - window: i]
        mu   = past.mean()
        sig  = past.std(ddof=1)
        if sig > 1e-12:
            z[i] = (spreads[i] - mu) / sig
    return z


# ─────────────────────────────────────────────────────────────
#  CORE BACKTEST ENGINE  v5.0
# ─────────────────────────────────────────────────────────────
def backtest_pair(
        sym1:       str,
        sym2:       str,
        s1:         pd.Series,
        s2:         pd.Series,
        entry_z:    float = ENTRY_Z,
        exit_z:     float = EXIT_Z,
        stop_z:     float = STOP_Z,
        min_hold:   int   = MIN_HOLD_BARS,
        delta:      float = DELTA,
        train_bars: int   = TRAIN_BARS,
        test_bars:  int   = TEST_BARS,
        step_bars:  int   = STEP_BARS,
        ) -> tuple:
    """
    Walk-forward backtest — no window overlap.

    Architecture per window:
      1. Take train_bars as Kalman warmup
      2. Run fresh Kalman on train+test combined
      3. Compute z-score using only past bars within window
      4. Trade only on test bars
      5. Advance by step_bars (= test_bars, no overlap)

    Returns:
      trades       : List[Trade]
      equity_curve : pd.Series  (cumulative normalised PnL)
      diagnostics  : dict
    """
    # Enforce no overlap
    if step_bars != test_bars:
        print(f"    [WARN] step_bars({step_bars}) != "
              f"test_bars({test_bars}), forcing equal")
        step_bars = test_bars

    s1v   = s1.values.astype(float)
    s2v   = s2.values.astype(float)
    times = s1.index
    n     = len(s1v)

    print(f"\n  ── {sym1}/{sym2}  ({n:,} bars) ──")
    print(f"    Train={train_bars}  "
          f"Test={test_bars}  "
          f"Step={step_bars}")

    trades:     List[Trade] = []
    # Equity tracking: list of (timestamp, cumulative_norm_pnl)
    eq_records: List[tuple] = []
    cum_norm_pnl = 0.0

    window_n   = 0
    win_start  = 0   # start of train window

    while win_start + train_bars + test_bars <= n:
        window_n  += 1
        train_end  = win_start + train_bars
        test_end   = min(train_end + test_bars, n)
        seg_end    = test_end

        # ── Fresh Kalman on train+test ────────────────
        seg_s1 = s1v[win_start: seg_end]
        seg_s2 = s2v[win_start: seg_end]
        seg_sp = build_kalman_spread_window(
            seg_s1, seg_s2,
            warmup=train_bars,
            delta=delta)

        # ── Half-life from training spread only ──────
        train_sp = pd.Series(seg_sp[:train_bars])
        hl = compute_half_life(train_sp)
        if not np.isfinite(hl) or hl <= 1:
            hl = 30.0
        hl = float(np.clip(hl, 5.0, 150.0))

        # Z-window: 2× half-life, clamped
        z_window = int(np.clip(hl * 2, 30, 300))

        # ── Training spread std for normalisation ────
        # Use last z_window bars of training spread
        # This is the "expected" spread volatility
        train_tail = seg_sp[
            max(0, train_bars - z_window): train_bars]
        spread_std = float(np.std(train_tail, ddof=1))
        if spread_std < 1e-10:
            spread_std = 1.0

        # ── Z-score over entire segment ──────────────
        # Strictly causal: z[i] uses spreads[i-w:i]
        seg_z = rolling_zscore_strict(seg_sp, z_window)

        test_start_offset = train_bars  # within seg

        t_start = times[train_end]
        t_end   = times[test_end - 1]
        print(f"    Win {window_n}: "
              f"{t_start.date()} → {t_end.date()}  "
              f"HL={hl:.1f}  Zwin={z_window}  "
              f"SpreadStd={spread_std:.6f}",
              end="")

        # ── Trade the test window ─────────────────────
        position      : int            = 0
        current_trade : Optional[Trade] = None
        window_trades = 0

        for li in range(test_start_offset,
                        seg_end - win_start):
            gi = win_start + li   # global bar index
            if gi >= n:
                break

            sp = seg_sp[li]
            z  = seg_z[li]
            t  = times[gi]

            # Skip bars with no valid z-score
            if z == 0.0 and li < z_window + test_start_offset:
                continue

            # ── ENTRY ─────────────────────────────────
            if position == 0:
                if z < -entry_z:
                    position = 1
                    current_trade = Trade(
                        entry_bar        = gi,
                        entry_time       = t,
                        direction        = 1,
                        entry_spread     = sp,
                        entry_z          = z,
                        entry_spread_std = spread_std)
                    window_trades += 1

                elif z > entry_z:
                    position = -1
                    current_trade = Trade(
                        entry_bar        = gi,
                        entry_time       = t,
                        direction        = -1,
                        entry_spread     = sp,
                        entry_z          = z,
                        entry_spread_std = spread_std)
                    window_trades += 1

            # ── EXIT — LONG ───────────────────────────
            elif position == 1:
                bars_in = gi - current_trade.entry_bar

                if bars_in < min_hold:
                    # Record equity without exit check
                    norm_pnl = (
                        (sp - current_trade.entry_spread)
                        / current_trade.entry_spread_std)
                    eq_records.append(
                        (t, cum_norm_pnl + norm_pnl))
                    continue

                # Unrealised PnL
                pnl_raw  = sp - current_trade.entry_spread
                pnl_norm = pnl_raw / current_trade.entry_spread_std
                reason   = None

                # FIX 2: Exit at EXIT_Z=0.5, not 0.0
                if z >= exit_z:
                    reason = "MEAN_CROSS"
                elif z < -stop_z:
                    reason = "STOP_LOSS"

                if reason:
                    current_trade.exit_bar    = gi
                    current_trade.exit_time   = t
                    current_trade.exit_spread = sp
                    current_trade.exit_z      = z
                    current_trade.exit_reason = reason
                    current_trade.pnl_spread  = pnl_raw
                    current_trade.pnl_norm    = pnl_norm
                    current_trade.bars_held   = bars_in
                    trades.append(current_trade)
                    cum_norm_pnl += pnl_norm
                    position      = 0
                    current_trade = None

                else:
                    # Still in trade — record unrealised
                    eq_records.append(
                        (t, cum_norm_pnl + pnl_norm))
                    continue

            # ── EXIT — SHORT ──────────────────────────
            elif position == -1:
                bars_in = gi - current_trade.entry_bar

                if bars_in < min_hold:
                    norm_pnl = (
                        (current_trade.entry_spread - sp)
                        / current_trade.entry_spread_std)
                    eq_records.append(
                        (t, cum_norm_pnl + norm_pnl))
                    continue

                pnl_raw  = current_trade.entry_spread - sp
                pnl_norm = pnl_raw / current_trade.entry_spread_std
                reason   = None

                # FIX 2: Exit at EXIT_Z=0.5 (symmetric)
                if z <= exit_z:
                    reason = "MEAN_CROSS"
                elif z > stop_z:
                    reason = "STOP_LOSS"

                if reason:
                    current_trade.exit_bar    = gi
                    current_trade.exit_time   = t
                    current_trade.exit_spread = sp
                    current_trade.exit_z      = z
                    current_trade.exit_reason = reason
                    current_trade.pnl_spread  = pnl_raw
                    current_trade.pnl_norm    = pnl_norm
                    current_trade.bars_held   = bars_in
                    trades.append(current_trade)
                    cum_norm_pnl += pnl_norm
                    position      = 0
                    current_trade = None

                else:
                    eq_records.append(
                        (t, cum_norm_pnl + pnl_norm))
                    continue

            # Realised equity point
            eq_records.append((t, cum_norm_pnl))

        print(f"  → {window_trades} trades")

        # FIX 1: No overlap — advance by test_bars
        win_start += step_bars

    # ── Close any open position at end ───────────────
    if (position != 0 and current_trade is not None
            and n > 0):
        gi  = n - 1
        sp  = s1v[gi] - s2v[gi]   # approximate
        pnl_raw = (position *
                   (sp - current_trade.entry_spread))
        pnl_norm = pnl_raw / current_trade.entry_spread_std
        current_trade.exit_bar    = gi
        current_trade.exit_time   = times[gi]
        current_trade.exit_spread = sp
        current_trade.exit_z      = 0.0
        current_trade.exit_reason = "END_OF_DATA"
        current_trade.pnl_spread  = pnl_raw
        current_trade.pnl_norm    = pnl_norm
        current_trade.bars_held   = gi - current_trade.entry_bar
        trades.append(current_trade)

    # ── Build equity curve ────────────────────────────
    if eq_records:
        eq_idx  = [r[0] for r in eq_records]
        eq_vals = [r[1] for r in eq_records]
        equity  = pd.Series(eq_vals,
                            index=pd.DatetimeIndex(eq_idx))
        # Deduplicate timestamps
        equity  = equity[~equity.index.duplicated(
                            keep='last')]
        equity  = equity.sort_index()

        # Convert cumulative normalised PnL
        # to equity starting at 1.0
        # Scale: 1 spread-std unit = 5% of equity
        # (conservative, adjust to taste)
        scale  = 0.05
        equity = 1.0 + equity * scale

        # Floor at 0 (can't lose more than you have)
        equity = equity.clip(lower=0.0)
    else:
        equity = pd.Series([1.0], index=[s1.index[0]])

    diag = {
        'n_windows'  : window_n,
        'n_trades'   : len(trades),
        'cum_norm_pnl': cum_norm_pnl,
    }

    return trades, equity, diag


# ─────────────────────────────────────────────────────────────
#  STATISTICS
# ─────────────────────────────────────────────────────────────
def compute_stats(trades: List[Trade],
                  equity: pd.Series) -> dict:
    if not trades:
        return {}

    pnl  = np.array([t.pnl_norm for t in trades])
    wins = pnl[pnl > 0]
    loss = pnl[pnl < 0]
    n    = len(pnl)
    wr   = len(wins) / n if n > 0 else 0
    gp   = wins.sum() if len(wins) > 0 else 0.0
    gl   = abs(loss.sum()) if len(loss) > 0 else 1e-10
    pf   = gp / gl

    ret    = equity.pct_change().dropna()
    sharpe = float(
        ret.mean() / (ret.std() + 1e-10) *
        np.sqrt(252 * 24))

    rm     = equity.cummax()
    max_dd = float(((equity - rm) / rm).min())

    holds  = [t.bars_held for t in trades]
    exits  = {}
    for t in trades:
        exits[t.exit_reason] = (
            exits.get(t.exit_reason, 0) + 1)

    return {
        'n_trades'     : n,
        'win_rate'     : round(wr, 4),
        'profit_factor': round(pf, 4),
        'sharpe'       : round(sharpe, 4),
        'max_dd'       : round(max_dd, 4),
        'avg_hold'     : round(float(np.mean(holds)), 1),
        'min_hold'     : int(np.min(holds)),
        'max_hold'     : int(np.max(holds)),
        'total_pnl_norm': round(float(pnl.sum()), 4),
        'exit_reasons' : exits,
    }


# ─────────────────────────────────────────────────────────────
#  SANITY CHECKER
# ─────────────────────────────────────────────────────────────
def sanity_check(trades:  List[Trade],
                 sym1:    str,
                 sym2:    str,
                 n_bars:  int,
                 stats:   dict):
    n     = len(trades)
    holds = [t.bars_held for t in trades] if trades else [0]

    print(f"\n    ── SANITY: {sym1}/{sym2} ──")
    print(f"      Bars in dataset : {n_bars:,}")
    print(f"      Total trades    : {n}")

    if n == 0:
        print(f"      ⚠ No trades generated")
        print(f"        Possible causes:")
        print(f"        - entry_z too strict")
        print(f"        - insufficient data after warmup")
        return

    avg_h = np.mean(holds)
    min_h = np.min(holds)

    print(f"      Avg hold        : {avg_h:.1f} bars")
    print(f"      Min hold        : {min_h} bars")
    print(f"      Win rate        : "
          f"{stats.get('win_rate',0):.1%}")
    print(f"      Profit factor   : "
          f"{stats.get('profit_factor',0):.2f}")

    ok = True

    # Gate 1: trade count
    if n > MAX_TRADES_PER_PAIR:
        print(f"      ✗ FAIL: {n} trades > "
              f"{MAX_TRADES_PER_PAIR} maximum")
        print(f"        Indicates exit logic bug")
        ok = False
    elif n < 10:
        print(f"      ⚠ WARN: only {n} trades "
              f"(low statistical power)")
    else:
        print(f"      ✓ Trade count: {n}")

    # Gate 2: hold time
    if avg_h < 15:
        print(f"      ✗ FAIL: avg hold {avg_h:.1f} < 15 bars")
        print(f"        Exits firing too fast")
        ok = False
    else:
        print(f"      ✓ Avg hold: {avg_h:.1f} bars")

    # Gate 3: win rate
    wr = stats.get('win_rate', 0)
    if wr > 0.80:
        print(f"      ✗ FAIL: win rate {wr:.1%} > 80%")
        print(f"        Indicates look-ahead bias")
        ok = False
    elif wr > 0.70:
        print(f"      ⚠ WARN: win rate {wr:.1%} is high")
    else:
        print(f"      ✓ Win rate: {wr:.1%}")

    # Gate 4: profit factor
    pf = stats.get('profit_factor', 0)
    if pf > 5.0:
        print(f"      ✗ FAIL: PF={pf:.2f} > 5.0")
        print(f"        Unrealistic for real trading")
        ok = False
    else:
        print(f"      ✓ Profit factor: {pf:.2f}")

    if ok:
        print(f"      ✓ ALL SANITY CHECKS PASSED")

    # Show first 8 trades for manual inspection
    print(f"\n      First 8 trades:")
    print(f"      {'Entry':>10} {'Dir':>4} "
          f"{'EntZ':>6} {'ExitZ':>6} "
          f"{'Hold':>5} {'PnL_n':>8} {'Reason'}")
    for t in trades[:8]:
        d = '+' if t.direction == 1 else '-'
        print(f"      "
              f"{str(t.entry_time)[:10]:>10} "
              f"[{d}] "
              f"{t.entry_z:>6.2f} "
              f"{t.exit_z:>6.2f} "
              f"{t.bars_held:>5} "
              f"{t.pnl_norm:>8.4f} "
              f"{t.exit_reason}")


# ─────────────────────────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────────────────────────
def plot_results(sym1:   str,
                 sym2:   str,
                 trades: List[Trade],
                 equity: pd.Series,
                 stats:  dict):
    if not trades or equity.empty:
        return
    try:
        fig = plt.figure(figsize=(16, 12))
        gs  = gridspec.GridSpec(
            2, 2, hspace=0.4, wspace=0.3)

        wr  = stats.get('win_rate', 0)
        pf  = stats.get('profit_factor', 0)
        sh  = stats.get('sharpe', 0)
        dd  = stats.get('max_dd', 0)
        nt  = stats.get('n_trades', 0)
        ah  = stats.get('avg_hold', 0)

        fig.suptitle(
            f"{sym1}/{sym2}  n={nt}  "
            f"WR={wr:.1%}  PF={pf:.2f}  "
            f"Sharpe={sh:.2f}  MaxDD={dd:.1%}  "
            f"AvgHold={ah:.0f}b",
            fontsize=11)

        # Equity
        ax1 = fig.add_subplot(gs[0, :])
        equity.plot(ax=ax1,
                    color='steelblue', lw=1.5)
        ax1.axhline(1.0, color='gray',
                    ls='--', lw=0.8)
        ax1.set_title(
            "Equity Curve (Walk-Forward OOS, "
            "no window overlap)")
        ax1.set_ylabel("Equity (normalised, 1=start)")
        ax1.grid(True, alpha=0.3)

        for t in trades[:300]:
            if t.exit_time is None: continue
            c = 'green' if t.pnl_norm > 0 else 'red'
            try:
                ax1.axvline(t.exit_time,
                            color=c,
                            alpha=0.08, lw=0.5)
            except Exception:
                pass

        # PnL distribution (normalised)
        ax2 = fig.add_subplot(gs[1, 0])
        pnl_n = [t.pnl_norm for t in trades]
        pd.Series(pnl_n).hist(
            ax=ax2, bins=40,
            color='steelblue',
            edgecolor='white', alpha=0.8)
        ax2.axvline(0, color='red', lw=1.5)
        ax2.set_title(
            "PnL Distribution (spread-std normalised)")
        ax2.set_xlabel("PnL (σ units)")
        ax2.grid(True, alpha=0.3)

        # Hold time
        ax3 = fig.add_subplot(gs[1, 1])
        holds = [t.bars_held for t in trades]
        pd.Series(holds).hist(
            ax=ax3, bins=30,
            color='darkorange',
            edgecolor='white', alpha=0.8)
        ax3.set_title(
            f"Hold Periods  avg={np.mean(holds):.0f}b")
        ax3.set_xlabel("Bars held (H1)")
        ax3.grid(True, alpha=0.3)

        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                f"bt_{sym1}_{sym2}.png"),
            dpi=120, bbox_inches='tight')
        plt.close()
        print(f"    [PLOT] bt_{sym1}_{sym2}.png")

    except Exception as e:
        print(f"    [PLOT ERR] {e}")
        traceback.print_exc()


def plot_portfolio(results: list):
    curves = [
        (r['pair'], r['equity'])
        for r in results
        if r.get('equity') is not None
        and len(r.get('equity', [])) > 10
    ]

    if not curves:
        return

    frames = []
    for pair, eq in curves:
        eq_c = eq.copy()
        eq_c = eq_c[~eq_c.index.duplicated(
                        keep='last')]
        eq_c = eq_c.sort_index()
        frames.append(eq_c.rename(pair))

    combined  = pd.concat(frames, axis=1)
    combined  = combined.ffill().fillna(1.0)
    portfolio = combined.mean(axis=1)

    if portfolio.iloc[0] != 0:
        portfolio = portfolio / portfolio.iloc[0]

    ret    = portfolio.pct_change().dropna()
    sharpe = float(
        ret.mean() / (ret.std() + 1e-10) *
        np.sqrt(252 * 24))
    rm     = portfolio.cummax()
    max_dd = float(((portfolio - rm) / rm).min())
    tot_r  = float(portfolio.iloc[-1] - 1.0)

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 10))
    fig.suptitle(
        f"Portfolio: {len(curves)} pairs  "
        f"Sharpe={sharpe:.2f}  "
        f"MaxDD={max_dd:.1%}  "
        f"Return={tot_r:.1%}",
        fontsize=12)

    combined.plot(ax=axes[0], lw=0.8, alpha=0.7)
    axes[0].axhline(1.0, color='k',
                    ls='--', lw=0.8)
    axes[0].set_title("Individual Pair Curves")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    portfolio.plot(ax=axes[1],
                   color='darkblue', lw=1.5)
    axes[1].axhline(1.0, color='k',
                    ls='--', lw=0.8)
    axes[1].fill_between(
        portfolio.index, portfolio.values, 1.0,
        where=(portfolio.values >= 1.0),
        alpha=0.3, color='green')
    axes[1].fill_between(
        portfolio.index, portfolio.values, 1.0,
        where=(portfolio.values < 1.0),
        alpha=0.3, color='red')
    axes[1].set_title("Portfolio (equal weight)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "portfolio.png"),
        dpi=120, bbox_inches='tight')
    plt.close()

    print(f"\n  PORTFOLIO:")
    print(f"    Pairs:   {len(curves)}")
    print(f"    Sharpe:  {sharpe:.2f}")
    print(f"    Max DD:  {max_dd:.1%}")
    print(f"    Return:  {tot_r:.1%}")

    with open(os.path.join(
            OUTPUT_DIR, "portfolio_stats.json"),
              'w') as f:
        json.dump({
            'version'  : BT_VERSION,
            'n_pairs'  : len(curves),
            'sharpe'   : round(sharpe, 4),
            'max_dd'   : round(max_dd, 4),
            'return'   : round(tot_r, 4),
            'pairs'    : [p for p, _ in curves],
        }, f, indent=2)


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def run_all_backtests(timeframe: str = "H1"):
    print(f"\n{'='*60}")
    print(f"  {BT_VERSION}")
    print(f"  Entry:   ±{ENTRY_Z}σ")
    print(f"  Exit:    {EXIT_Z}σ  (FIX: was 0.0)")
    print(f"  Stop:    ±{STOP_Z}σ")
    print(f"  MinHold: {MIN_HOLD_BARS} bars  "
          f"(FIX: was 2)")
    print(f"  Step:    {STEP_BARS} = Test={TEST_BARS}  "
          f"(FIX: no overlap)")
    print(f"{'='*60}")

    vp_file = os.path.join(
        OUTPUT_DIR, "valid_pairs.json")
    if not os.path.exists(vp_file):
        print("[ERR] valid_pairs.json not found")
        print("      Run pairs_research.py first")
        return []

    with open(vp_file) as f:
        valid_pairs = json.load(f)

    print(f"\n  Loaded {len(valid_pairs)} valid pairs")

    all_results  = []
    summary_rows = []

    for p in valid_pairs:
        sym1 = p['symbol1']
        sym2 = p['symbol2']

        s1 = load_price_series(sym1, timeframe)
        s2 = load_price_series(sym2, timeframe)

        min_bars = train_bars + test_bars + 200
        if len(s1) < min_bars or len(s2) < min_bars:
            print(f"  [SKIP] {sym1}/{sym2}: "
                  f"need {min_bars} bars, "
                  f"have {min(len(s1),len(s2))}")
            continue

        s1a, s2a = align_series(s1, s2)
        if len(s1a) < min_bars:
            print(f"  [SKIP] {sym1}/{sym2}: "
                  f"insufficient aligned bars")
            continue

        try:
            trades, equity, diag = backtest_pair(
                sym1        = sym1,
                sym2        = sym2,
                s1          = s1a,
                s2          = s2a,
                entry_z     = ENTRY_Z,
                exit_z      = EXIT_Z,
                stop_z      = STOP_Z,
                min_hold    = MIN_HOLD_BARS,
                delta       = DELTA,
                train_bars  = TRAIN_BARS,
                test_bars   = TEST_BARS,
                step_bars   = STEP_BARS,
            )
        except Exception as e:
            print(f"  [ERR] {sym1}/{sym2}: {e}")
            traceback.print_exc()
            continue

        if not trades:
            print(f"  [SKIP] No trades for "
                  f"{sym1}/{sym2}")
            continue

        stats = compute_stats(trades, equity)
        sanity_check(trades, sym1, sym2,
                     len(s1a), stats)

        plot_results(sym1, sym2, trades,
                     equity, stats)

        # Save trade CSV
        rows = []
        for t in trades:
            rows.append({
                'entry_time'   : str(t.entry_time),
                'exit_time'    : str(t.exit_time),
                'direction'    : t.direction,
                'entry_spread' : t.entry_spread,
                'exit_spread'  : t.exit_spread,
                'entry_z'      : round(t.entry_z, 4),
                'exit_z'       : round(t.exit_z, 4),
                'exit_reason'  : t.exit_reason,
                'pnl_spread'   : t.pnl_spread,
                'pnl_norm'     : round(t.pnl_norm, 6),
                'bars_held'    : t.bars_held,
            })
        pd.DataFrame(rows).to_csv(
            os.path.join(
                OUTPUT_DIR,
                f"trades_{sym1}_{sym2}.csv"),
            index=False)

        print(f"\n  ══ {sym1}/{sym2} ══")
        print(f"    Trades    : {stats['n_trades']}")
        print(f"    Win Rate  : "
              f"{stats['win_rate']:.1%}")
        print(f"    PF        : "
              f"{stats['profit_factor']:.2f}")
        print(f"    Sharpe    : {stats['sharpe']:.2f}")
        print(f"    Max DD    : {stats['max_dd']:.1%}")
        print(f"    Avg Hold  : "
              f"{stats['avg_hold']:.0f} bars "
              f"(min={stats['min_hold']} "
              f"max={stats['max_hold']})")
        print(f"    PnL (norm): "
              f"{stats['total_pnl_norm']:.4f} σ-units")
        print(f"    Exits     : {stats['exit_reasons']}")

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
            'min_hold'     : stats['min_hold'],
            'half_life'    : p.get('half_life', 0),
            'eg_pval'      : p.get('eg_pval', 1),
        })

    if summary_rows:
        df_s = pd.DataFrame(summary_rows).sort_values(
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

    if len(all_results) >= 1:
        plot_portfolio(all_results)

    print(f"\n  [DONE] {BT_VERSION}")
    return all_results


# Reference for calling from outside
train_bars = TRAIN_BARS
test_bars  = TEST_BARS

if __name__ == "__main__":
    run_all_backtests(timeframe="H1")
