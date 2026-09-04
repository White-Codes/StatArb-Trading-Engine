"""
pairs_backtest.py  v4.0
========================
Root cause fixes vs v3.0:

  FIX 1: Exit threshold set to 0.0 (exact mean cross)
          not 0.3 which caused asymmetric exit logic

  FIX 2: SHORT spread exit corrected
          Short entered at z=+2.0, exit when z<=0.0
          not when z<=-0.3 which was confusing direction

  FIX 3: Equity model replaced
          Instead of compounding every trade:
          - Track cumulative PnL in spread units
          - Convert to equity only at the end
          - No compounding overflow

  FIX 4: Minimum hold time enforced
          Cannot exit on the same bar as entry
          Minimum 2 bars held before exit checked
          This prevents 1-bar micro-trades

  FIX 5: Trade count sanity gate
          If > 500 trades generated: print clear warning
          and show sample of first 10 trades to diagnose

Expected realistic results:
  Trades:         50 to 300 per pair over full history
  Win rate:       55% to 72%
  Avg hold:       15 to 100 bars
  Profit factor:  1.2 to 2.5
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

from kalman_filter import KalmanHedgeFilter
from pairs_research import (
    load_price_series, align_series,
    compute_half_life
)

OUTPUT_DIR = "pairs_artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)
BT_VERSION = "backtest-v4.0"

# ── Trading parameters ────────────────────────────────────────
ENTRY_Z     = 2.0    # enter when |z| > this
EXIT_Z      = 0.0    # exit when z returns past this toward 0
STOP_Z      = 3.5    # stop loss when |z| exceeds this
MIN_HOLD    = 2      # minimum bars before exit is checked
DELTA       = 1e-4   # Kalman filter speed
TRAIN_BARS  = 2016   # ~3 months H1
TEST_BARS   = 336    # ~2 weeks H1
STEP_BARS   = 168    # step 1 week


# ─────────────────────────────────────────────────────────────
#  TRADE RECORD
# ─────────────────────────────────────────────────────────────
@dataclass
class Trade:
    entry_bar    : int
    entry_time   : pd.Timestamp
    direction    : int           # +1=long spread -1=short
    entry_spread : float
    entry_z      : float
    exit_bar     : int           = 0
    exit_time    : pd.Timestamp  = None
    exit_spread  : float         = 0.0
    exit_z       : float         = 0.0
    exit_reason  : str           = ""
    pnl_spread   : float         = 0.0
    bars_held    : int           = 0


# ─────────────────────────────────────────────────────────────
#  KALMAN SPREAD — runs filter once over full history
# ─────────────────────────────────────────────────────────────
def build_kalman_spread(s1_vals: np.ndarray,
                        s2_vals: np.ndarray,
                        delta:   float = DELTA
                        ) -> np.ndarray:
    """
    Run Kalman filter over entire price history.
    Returns spread array of same length.

    The filter is CAUSAL: spread[i] uses only
    data from bars 0..i, never future bars.
    Safe to use in walk-forward testing.
    """
    kf   = KalmanHedgeFilter(delta=delta)
    out  = np.zeros(len(s1_vals))
    for i in range(len(s1_vals)):
        _, _, sp = kf.update(s1_vals[i], s2_vals[i])
        out[i]   = sp
    return out


# ─────────────────────────────────────────────────────────────
#  ROLLING Z-SCORE — correct implementation
# ─────────────────────────────────────────────────────────────
def build_rolling_zscore(spreads: np.ndarray,
                          window:  int
                          ) -> np.ndarray:
    """
    z[i] = (spreads[i] - mean(spreads[i-w:i]))
           / std(spreads[i-w:i])

    Uses ONLY past data (no look-ahead).
    First 'window' elements are set to 0.
    """
    n = len(spreads)
    z = np.zeros(n)
    for i in range(window, n):
        sub  = spreads[i - window: i]  # past window bars
        mu   = sub.mean()
        sig  = sub.std(ddof=1)
        if sig > 1e-12:
            z[i] = (spreads[i] - mu) / sig
    return z


# ─────────────────────────────────────────────────────────────
#  CORE BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────
def backtest_pair(sym1:        str,
                  sym2:        str,
                  s1:          pd.Series,
                  s2:          pd.Series,
                  entry_z:     float = ENTRY_Z,
                  exit_z:      float = EXIT_Z,
                  stop_z:      float = STOP_Z,
                  min_hold:    int   = MIN_HOLD,
                  delta:       float = DELTA,
                  train_bars:  int   = TRAIN_BARS,
                  test_bars:   int   = TEST_BARS,
                  step_bars:   int   = STEP_BARS,
                  max_dd_pct:  float = 0.20,
                  ) -> tuple:
    """
    Walk-forward backtest for one pair.

    Returns:
      trades       : List[Trade]
      equity_curve : pd.Series  (starts at 1.0)
      z_series     : pd.Series  (full z-score history)
    """
    s1v   = s1.values.astype(float)
    s2v   = s2.values.astype(float)
    times = s1.index
    n     = len(s1v)

    print(f"\n  ── {sym1}/{sym2}  ({n:,} bars) ──")

    # Step 1: Build full Kalman spread (causal)
    print(f"    Building Kalman spread...")
    spreads = build_kalman_spread(s1v, s2v, delta)

    # Step 2: Walk-forward loop
    trades:   List[Trade] = []
    pnl_log:  List[float] = []   # raw PnL each trade
    eq_index: List        = []
    eq_vals:  List[float] = []

    halted     = False
    peak_eq    = 1.0
    cum_pnl    = 0.0   # cumulative spread PnL
    test_start = train_bars
    window_n   = 0

    while test_start + test_bars <= n:
        window_n += 1
        test_end  = min(test_start + test_bars, n)

        # ── Calibrate z-window from training spread ───
        tr_spread = pd.Series(
            spreads[max(0, test_start - train_bars):
                    test_start])
        hl = compute_half_life(tr_spread)
        if not np.isfinite(hl) or hl <= 0:
            hl = 30
        z_window = int(np.clip(hl * 2, 20, 200))

        print(f"    Win {window_n}: "
              f"{times[test_start].date()} → "
              f"{times[test_end-1].date()}  "
              f"HL={hl:.1f}  Zwin={z_window}",
              end="")

        # ── Compute z-score for this segment ─────────
        # Include lookback so test bars have valid z
        seg_start  = max(0, test_start - z_window)
        seg_spread = spreads[seg_start: test_end]
        seg_z      = build_rolling_zscore(
            seg_spread, z_window)

        # Offset: index within seg that is test_start
        offset = test_start - seg_start

        # ── Trade this window ─────────────────────────
        position:       int            = 0
        current_trade:  Optional[Trade] = None
        window_trades   = 0

        for li in range(offset, len(seg_spread)):
            gi = seg_start + li   # global bar index
            if gi >= n:
                break

            sp = seg_spread[li]
            z  = seg_z[li]
            t  = times[gi]

            # Drawdown check
            # Equity approximated as 1 + cum_pnl * scale
            approx_eq = 1.0 + cum_pnl * 10
            if (approx_eq < peak_eq * (1 - max_dd_pct)
                    and not halted):
                halted = True
                print(f"\n      [HALT] DD breached "
                      f"at {t.date()}")

            if halted:
                eq_index.append(t)
                eq_vals.append(
                    max(0.5, 1.0 + cum_pnl * 10))
                continue

            peak_eq = max(peak_eq,
                          1.0 + cum_pnl * 10)

            # ── ENTRY ─────────────────────────────────
            if position == 0:

                if z < -entry_z:
                    # LONG spread
                    # The spread is too LOW
                    # We expect it to rise back to 0
                    # Action: BUY instrument A
                    #         SELL instrument B
                    position = 1
                    current_trade = Trade(
                        entry_bar    = gi,
                        entry_time   = t,
                        direction    = 1,
                        entry_spread = sp,
                        entry_z      = z)
                    window_trades += 1

                elif z > entry_z:
                    # SHORT spread
                    # The spread is too HIGH
                    # We expect it to fall back to 0
                    # Action: SELL instrument A
                    #         BUY instrument B
                    position = -1
                    current_trade = Trade(
                        entry_bar    = gi,
                        entry_time   = t,
                        direction    = -1,
                        entry_spread = sp,
                        entry_z      = z)
                    window_trades += 1

            # ── EXIT — LONG spread ────────────────────
            elif position == 1:
                # We are long spread
                # Entered at z ~ -2.0 (spread too low)
                # Spread rising → z rising toward 0
                # Exit when z crosses back above 0

                bars_in = gi - current_trade.entry_bar
                if bars_in < min_hold:
                    # Too soon — must hold minimum bars
                    eq_index.append(t)
                    eq_vals.append(
                        1.0 + cum_pnl * 10)
                    continue

                pnl    = sp - current_trade.entry_spread
                reason = None

                if z >= exit_z:
                    # Spread returned to mean → profit
                    reason = "MEAN_CROSS"

                elif z < -stop_z:
                    # Spread went further against us
                    # Cointegration may be breaking
                    reason = "STOP_LOSS"

                if reason:
                    current_trade.exit_bar    = gi
                    current_trade.exit_time   = t
                    current_trade.exit_spread = sp
                    current_trade.exit_z      = z
                    current_trade.exit_reason = reason
                    current_trade.pnl_spread  = pnl
                    current_trade.bars_held   = bars_in
                    trades.append(current_trade)
                    pnl_log.append(pnl)
                    cum_pnl += pnl
                    position = 0
                    current_trade = None

            # ── EXIT — SHORT spread ───────────────────
            elif position == -1:
                # We are short spread
                # Entered at z ~ +2.0 (spread too high)
                # Spread falling → z falling toward 0
                # Exit when z crosses back below 0

                bars_in = gi - current_trade.entry_bar
                if bars_in < min_hold:
                    eq_index.append(t)
                    eq_vals.append(
                        1.0 + cum_pnl * 10)
                    continue

                # PnL for short: profit when spread falls
                pnl    = (current_trade.entry_spread - sp)
                reason = None

                if z <= exit_z:
                    # Spread returned to mean → profit
                    # z has fallen back to 0 or below
                    reason = "MEAN_CROSS"

                elif z > stop_z:
                    # Spread went further against us
                    reason = "STOP_LOSS"

                if reason:
                    current_trade.exit_bar    = gi
                    current_trade.exit_time   = t
                    current_trade.exit_spread = sp
                    current_trade.exit_z      = z
                    current_trade.exit_reason = reason
                    current_trade.pnl_spread  = pnl
                    current_trade.bars_held   = bars_in
                    trades.append(current_trade)
                    pnl_log.append(pnl)
                    cum_pnl += pnl
                    position = 0
                    current_trade = None

            eq_index.append(t)
            eq_vals.append(1.0 + cum_pnl * 10)

        print(f"  → {window_trades} trades")
        test_start += step_bars

    # Close any open position at end of data
    if position != 0 and current_trade and n > 0:
        gi = n - 1
        sp = spreads[gi]
        pnl = (position *
               (sp - current_trade.entry_spread))
        current_trade.exit_bar    = gi
        current_trade.exit_time   = times[gi]
        current_trade.exit_spread = sp
        current_trade.exit_z      = 0.0
        current_trade.exit_reason = "END_OF_DATA"
        current_trade.pnl_spread  = pnl
        current_trade.bars_held   = (
            gi - current_trade.entry_bar)
        trades.append(current_trade)

    # Build equity curve
    # Remove duplicate timestamps
    eq_df = pd.Series(eq_vals,
                      index=pd.DatetimeIndex(eq_index))
    eq_df = eq_df[~eq_df.index.duplicated(keep='last')]
    eq_df = eq_df.sort_index()

    # Normalise so it starts at exactly 1.0
    if len(eq_df) > 0 and eq_df.iloc[0] != 0:
        eq_df = eq_df / eq_df.iloc[0]

    return trades, eq_df, pd.Series(
        build_rolling_zscore(
            spreads,
            int(np.clip(30 * 2, 20, 200))),
        index=s1.index)


# ─────────────────────────────────────────────────────────────
#  STATISTICS
# ─────────────────────────────────────────────────────────────
def compute_stats(trades: List[Trade],
                  equity: pd.Series) -> dict:
    if not trades:
        return {}

    pnl  = np.array([t.pnl_spread for t in trades])
    wins = pnl[pnl > 0]
    loss = pnl[pnl < 0]
    n    = len(pnl)
    wr   = len(wins) / n if n > 0 else 0
    gp   = wins.sum() if len(wins) > 0 else 0
    gl   = abs(loss.sum()) if len(loss) > 0 else 1e-10
    pf   = gp / gl

    ret    = equity.pct_change().dropna()
    sharpe = float(
        ret.mean() / (ret.std() + 1e-10) *
        np.sqrt(252 * 24))  # H1 annualised

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
        'avg_hold'     : round(np.mean(holds), 1),
        'min_hold'     : int(np.min(holds)),
        'max_hold'     : int(np.max(holds)),
        'total_pnl'    : round(float(pnl.sum()), 8),
        'exit_reasons' : exits,
    }


# ─────────────────────────────────────────────────────────────
#  SANITY CHECKER
# ─────────────────────────────────────────────────────────────
def sanity_check(trades: List[Trade],
                 sym1:   str,
                 sym2:   str,
                 n_bars: int):
    """
    Check trade count and hold times are realistic.
    Print first 5 trades for manual inspection.
    """
    n = len(trades)
    print(f"\n    SANITY ({sym1}/{sym2}):")
    print(f"      Total trades:  {n}")

    if n == 0:
        print(f"      ⚠ No trades — "
              f"entry_z may be too strict")
        return

    holds = [t.bars_held for t in trades]
    print(f"      Avg hold:      {np.mean(holds):.1f} bars")
    print(f"      Min hold:      {np.min(holds)} bars")
    print(f"      Max hold:      {np.max(holds)} bars")

    # Flags
    if n > 500:
        print(f"      ⚠ WARNING: {n} trades is too many")
        print(f"        Expected 50-300 for H1 data")
    else:
        print(f"      ✓ Trade count OK")

    if np.mean(holds) < 10:
        print(f"      ⚠ WARNING: avg hold < 10 bars")
        print(f"        Exits firing too fast")
    else:
        print(f"      ✓ Hold time OK")

    # Show first 5 trades
    print(f"\n      First 5 trades:")
    print(f"      {'Entry':>10} {'ExitZ':>7} "
          f"{'Hold':>5} {'PnL':>10} {'Reason'}")
    for t in trades[:5]:
        print(f"      {str(t.entry_time)[:10]:>10} "
              f"{t.exit_z:>7.3f} "
              f"{t.bars_held:>5} "
              f"{t.pnl_spread:>10.6f} "
              f"{t.exit_reason}")


# ─────────────────────────────────────────────────────────────
#  PLOT
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
        gs  = gridspec.GridSpec(2, 2,
                                hspace=0.4, wspace=0.3)

        wr = stats.get('win_rate', 0)
        pf = stats.get('profit_factor', 0)
        sh = stats.get('sharpe', 0)
        dd = stats.get('max_dd', 0)
        nt = stats.get('n_trades', 0)
        ah = stats.get('avg_hold', 0)

        fig.suptitle(
            f"{sym1}/{sym2}  "
            f"n={nt}  WR={wr:.1%}  "
            f"PF={pf:.2f}  Sharpe={sh:.2f}  "
            f"MaxDD={dd:.1%}  AvgHold={ah:.0f}bars",
            fontsize=11)

        # 1. Equity curve
        ax1 = fig.add_subplot(gs[0, :])
        equity.plot(ax=ax1, color='steelblue', lw=1.5)
        ax1.axhline(1.0, color='gray',
                    ls='--', lw=0.8)
        ax1.set_title(
            "Equity Curve (Walk-Forward OOS)")
        ax1.set_ylabel("Equity (normalised)")
        ax1.grid(True, alpha=0.3)

        # Mark exits on equity curve
        for t in trades[:200]:  # cap for speed
            if t.exit_time is None:
                continue
            c = ('green' if t.pnl_spread > 0
                 else 'red')
            try:
                ax1.axvline(t.exit_time,
                            color=c, alpha=0.1,
                            lw=0.4)
            except Exception:
                pass

        # 2. PnL distribution
        ax2 = fig.add_subplot(gs[1, 0])
        pnl_vals = [t.pnl_spread for t in trades]
        pd.Series(pnl_vals).hist(
            ax=ax2, bins=40,
            color='steelblue',
            edgecolor='white', alpha=0.8)
        ax2.axvline(0, color='red', lw=1.5)
        ax2.set_title("PnL Distribution")
        ax2.set_xlabel("PnL (spread units)")
        ax2.grid(True, alpha=0.3)

        # 3. Hold time distribution
        ax3 = fig.add_subplot(gs[1, 1])
        holds = [t.bars_held for t in trades]
        pd.Series(holds).hist(
            ax=ax3, bins=30,
            color='darkorange',
            edgecolor='white', alpha=0.8)
        ax3.set_title(
            f"Hold Periods  avg={np.mean(holds):.0f}b")
        ax3.set_xlabel("Bars held")
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


# ─────────────────────────────────────────────────────────────
#  PORTFOLIO PLOT — no compounding, simple average
# ─────────────────────────────────────────────────────────────
def plot_portfolio(results: list):
    curves = [(r['pair'], r['equity'])
              for r in results
              if r.get('equity') is not None
              and len(r.get('equity', [])) > 10]

    if len(curves) < 1:
        return

    frames = []
    for pair, eq in curves:
        # Remove duplicate index entries
        eq_clean = eq.copy()
        eq_clean = eq_clean[
            ~eq_clean.index.duplicated(keep='last')]
        eq_clean = eq_clean.sort_index()
        frames.append(eq_clean.rename(pair))

    combined  = pd.concat(frames, axis=1)
    combined  = combined.ffill().fillna(1.0)
    portfolio = combined.mean(axis=1)

    # Normalise portfolio to start at 1.0
    if portfolio.iloc[0] != 0:
        portfolio = portfolio / portfolio.iloc[0]

    ret    = portfolio.pct_change().dropna()
    sharpe = float(
        ret.mean() / (ret.std() + 1e-10) *
        np.sqrt(252 * 24))
    rm     = portfolio.cummax()
    max_dd = float(((portfolio - rm) / rm).min())
    tot_r  = float(portfolio.iloc[-1] - 1.0)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
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
        portfolio.index,
        portfolio.values, 1.0,
        where=(portfolio.values >= 1.0),
        alpha=0.3, color='green')
    axes[1].fill_between(
        portfolio.index,
        portfolio.values, 1.0,
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

    with open(os.path.join(OUTPUT_DIR,
                            "portfolio_stats.json"),
              'w') as f:
        json.dump({
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
    print(f"\n{'='*55}")
    print(f"  {BT_VERSION}")
    print(f"  Entry: ±{ENTRY_Z}σ  "
          f"Exit: {EXIT_Z}σ  "
          f"Stop: ±{STOP_Z}σ  "
          f"MinHold: {MIN_HOLD} bars")
    print(f"{'='*55}")

    vp_file = os.path.join(OUTPUT_DIR,
                            "valid_pairs.json")
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

        if len(s1) < 3000 or len(s2) < 3000:
            print(f"  [SKIP] {sym1}/{sym2}: "
                  f"insufficient data")
            continue

        s1a, s2a = align_series(s1, s2)
        if len(s1a) < 3000:
            print(f"  [SKIP] {sym1}/{sym2}: "
                  f"insufficient aligned bars")
            continue

        try:
            trades, equity, z_ser = backtest_pair(
                sym1       = sym1,
                sym2       = sym2,
                s1         = s1a,
                s2         = s2a,
                entry_z    = ENTRY_Z,
                exit_z     = EXIT_Z,
                stop_z     = STOP_Z,
                min_hold   = MIN_HOLD,
                delta      = DELTA,
                train_bars = TRAIN_BARS,
                test_bars  = TEST_BARS,
                step_bars  = STEP_BARS,
                max_dd_pct = 0.20,
            )
        except Exception as e:
            print(f"  [ERR] {sym1}/{sym2}: {e}")
            traceback.print_exc()
            continue

        sanity_check(trades, sym1, sym2, len(s1a))

        if not trades:
            print(f"  [SKIP] No trades generated")
            continue

        stats = compute_stats(trades, equity)
        plot_results(sym1, sym2, trades,
                     equity, stats)

        # Save trade log
        rows = []
        for t in trades:
            rows.append({
                'entry_time' : str(t.entry_time),
                'exit_time'  : str(t.exit_time),
                'direction'  : t.direction,
                'entry_spread': t.entry_spread,
                'exit_spread': t.exit_spread,
                'entry_z'    : t.entry_z,
                'exit_z'     : t.exit_z,
                'exit_reason': t.exit_reason,
                'pnl_spread' : t.pnl_spread,
                'bars_held'  : t.bars_held,
            })
        pd.DataFrame(rows).to_csv(
            os.path.join(
                OUTPUT_DIR,
                f"trades_{sym1}_{sym2}.csv"),
            index=False)

        print(f"\n  ══ {sym1}/{sym2} ══")
        print(f"    Trades:   {stats['n_trades']}")
        print(f"    Win Rate: {stats['win_rate']:.1%}")
        print(f"    PF:       "
              f"{stats['profit_factor']:.2f}")
        print(f"    Sharpe:   {stats['sharpe']:.2f}")
        print(f"    Max DD:   {stats['max_dd']:.1%}")
        print(f"    Avg Hold: "
              f"{stats['avg_hold']:.0f} bars  "
              f"(min={stats['min_hold']} "
              f"max={stats['max_hold']})")
        print(f"    Exits:    {stats['exit_reasons']}")

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

    # Summary table
    if summary_rows:
        df_s = pd.DataFrame(summary_rows).sort_values(
            'profit_factor', ascending=False)
        print(f"\n{'='*72}")
        print(f"  BACKTEST SUMMARY  ({BT_VERSION})")
        print(f"{'='*72}")
        print(df_s.to_string(index=False))
        df_s.to_csv(
            os.path.join(OUTPUT_DIR,
                         "backtest_summary.csv"),
            index=False)

    # Portfolio
    if len(all_results) >= 1:
        plot_portfolio(all_results)

    print(f"\n  [DONE] {BT_VERSION}")
    return all_results


if __name__ == "__main__":
    run_all_backtests(timeframe="H1")
