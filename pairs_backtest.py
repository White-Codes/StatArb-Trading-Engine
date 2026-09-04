"""
pairs_backtest.py  v3.0
========================
CORRECTED walk-forward backtest.

Bug fixed vs v2.0:
  - Exit condition was triggering too early
    because the z-score buffer was not being
    maintained correctly across bars
  - Solution: compute z-score from a fixed
    rolling window on the spread series
    using only bars WITHIN the current test window
    not from the training buffer

Rules implemented:
  ENTRY:  z < -2.0  →  long spread  (buy A, sell B)
          z > +2.0  →  short spread (sell A, buy B)
  EXIT:   z > +0.3  →  close long  (mean restored)
          z < -0.3  →  close short (mean restored)
  STOP:   z < -3.5  →  stop long   (loss too large)
          z > +3.5  →  stop short  (loss too large)

Expected results (realistic):
  Trades per pair:  30-150 over full history
  Win rate:         55-70%
  Profit factor:    1.2-2.5
  Avg hold:         20-80 bars
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

from kalman_filter  import KalmanHedgeFilter, kalman_filter_batch
from pairs_research  import (
    load_price_series, align_series,
    estimate_hedge_ratio, compute_half_life
)

OUTPUT_DIR = "pairs_artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)
BT_VERSION = "backtest-v3.0-fixed"


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
    entry_beta   : float
    exit_bar     : int           = 0
    exit_time    : pd.Timestamp  = None
    exit_spread  : float         = 0.0
    exit_z       : float         = 0.0
    exit_reason  : str           = ""
    pnl_raw      : float         = 0.0   # in spread units
    bars_held    : int           = 0


# ─────────────────────────────────────────────────────────────
#  KALMAN SPREAD COMPUTATION
#  Runs Kalman filter over a price window and returns
#  the full spread series as a numpy array
# ─────────────────────────────────────────────────────────────
def compute_kalman_spread(s1_vals: np.ndarray,
                          s2_vals: np.ndarray,
                          delta:   float = 1e-4
                          ) -> np.ndarray:
    """
    Run Kalman filter over s1, s2 price arrays.
    Returns spread array of same length.

    The Kalman filter updates the hedge ratio β
    every single bar, so the spread always reflects
    the CURRENT relationship between the two prices.

    This is better than OLS (fixed hedge ratio) because:
    - OLS assumes β is constant forever
    - Kalman allows β to slowly drift over time
    - This matches reality — relationships shift slowly
    """
    kf      = KalmanHedgeFilter(delta=delta)
    spreads = np.zeros(len(s1_vals))

    for i in range(len(s1_vals)):
        beta, alpha, spread = kf.update(
            s1_vals[i], s2_vals[i])
        spreads[i] = spread

    return spreads


# ─────────────────────────────────────────────────────────────
#  Z-SCORE COMPUTATION — ROLLING WINDOW
#  This is the CORRECTED version.
#  Key rule: z-score at bar i uses only bars
#  from max(0, i-window) to i (rolling window).
#  It does NOT mix training and test data.
# ─────────────────────────────────────────────────────────────
def compute_rolling_zscore(spreads: np.ndarray,
                            window: int
                            ) -> np.ndarray:
    """
    Compute rolling z-score for every bar.

    z[i] = (spread[i] - mean(spread[i-w:i])) 
           / std(spread[i-w:i])

    where w = window size

    The window size is set to 2 × half_life
    which means we are normalising against
    roughly 2 mean-reversion cycles.

    Returns array of same length as spreads.
    First 'window' bars will be 0 (insufficient data).
    """
    n      = len(spreads)
    z      = np.zeros(n)

    for i in range(window, n):
        sub  = spreads[i - window : i]
        mu   = sub.mean()
        sig  = sub.std()
        if sig > 1e-12:
            z[i] = (spreads[i] - mu) / sig

    return z


# ─────────────────────────────────────────────────────────────
#  SINGLE PAIR WALK-FORWARD BACKTEST
# ─────────────────────────────────────────────────────────────
def backtest_pair(sym1: str, sym2: str,
                  s1:   pd.Series,
                  s2:   pd.Series,
                  # Signal thresholds
                  entry_z:    float = 2.0,
                  exit_z:     float = 0.3,
                  stop_z:     float = 3.5,
                  # Kalman delta
                  delta:      float = 1e-4,
                  # Walk-forward windows (bars)
                  train_bars: int   = 2016,
                  test_bars:  int   = 336,
                  step_bars:  int   = 168,
                  # Risk
                  max_dd_pct: float = 0.20
                  ) -> tuple:
    """
    Walk-forward backtest for one pair.

    Returns:
      trades      : list of Trade objects
      equity_curve: pd.Series of equity values
    """
    s1_vals = s1.values.astype(float)
    s2_vals = s2.values.astype(float)
    times   = s1.index
    n       = len(s1_vals)

    print(f"\n  ── {sym1} / {sym2}  ({n} bars) ──")

    # ── PRE-COMPUTE full Kalman spread ────────────────────
    # We run the Kalman filter ONCE over the full history.
    # This is equivalent to running it bar-by-bar in live
    # because Kalman is causal (uses only past data).
    print(f"    Computing Kalman spread...")
    all_spreads = compute_kalman_spread(
        s1_vals, s2_vals, delta=delta)

    trades:     List[Trade] = []
    eq_vals:    List[float] = [1.0]
    eq_times:   List        = [times[0]]
    equity      = 1.0
    peak_eq     = 1.0
    halted      = False

    test_start  = train_bars
    window_n    = 0

    while test_start + test_bars <= n:
        window_n += 1
        test_end  = min(test_start + test_bars, n)

        # ── TRAIN: determine z-score window ──────────────
        # Use the spread from the training window
        # to estimate the half-life and set z-window size
        tr_spreads = all_spreads[
            max(0, test_start - train_bars):test_start]
        tr_series  = pd.Series(tr_spreads)
        hl         = compute_half_life(tr_series)

        if not np.isfinite(hl) or hl <= 0:
            hl = 30   # default fallback

        # Z-score window = 2 × half_life
        # Minimum 20 bars, maximum 200 bars
        z_window = int(np.clip(hl * 2, 20, 200))

        print(f"    Win {window_n}: "
              f"{times[test_start].date()} → "
              f"{times[test_end-1].date()}  "
              f"HL={hl:.1f}  Zwin={z_window}")

        # ── TEST: compute z-scores for test window ───────
        # IMPORTANT: we compute z-score using a window
        # that STARTS from training data.
        # This means bar test_start's z-score uses
        # bars [test_start - z_window : test_start]
        # which are all training bars → no look-ahead.

        # Get spread for test window + lookback
        lookback_start = max(0,
                             test_start - z_window)
        segment_end    = test_end

        seg_spreads = all_spreads[
            lookback_start:segment_end]
        seg_z       = compute_rolling_zscore(
            seg_spreads, z_window)

        # Offset: position within seg that corresponds
        # to test_start
        offset = test_start - lookback_start

        # ── BAR-BY-BAR TRADING ───────────────────────────
        position:      int            = 0
        current_trade: Optional[Trade] = None

        for local_i in range(offset,
                             len(seg_spreads)):
            global_i = lookback_start + local_i
            if global_i >= n:
                break

            spread = seg_spreads[local_i]
            z      = seg_z[local_i]
            t      = times[global_i]

            # ── Drawdown check ────────────────────────
            if equity < peak_eq * (1 - max_dd_pct):
                if not halted:
                    halted = True
                    print(f"      [HALT] DD at "
                          f"{t.date()}")
                # Close any open trade
                if position != 0 and current_trade:
                    pnl = (position *
                           (spread -
                            current_trade.entry_spread))
                    current_trade.exit_bar    = global_i
                    current_trade.exit_time   = t
                    current_trade.exit_spread = spread
                    current_trade.exit_z      = z
                    current_trade.exit_reason = "DD_HALT"
                    current_trade.pnl_raw     = pnl
                    current_trade.bars_held   = (
                        global_i - current_trade.entry_bar)
                    trades.append(current_trade)
                    equity = _apply_pnl(
                        equity,
                        current_trade.pnl_raw,
                        current_trade.entry_spread)
                    peak_eq = max(peak_eq, equity)
                    position = 0
                    current_trade = None

            if halted:
                eq_vals.append(equity)
                eq_times.append(t)
                continue

            peak_eq = max(peak_eq, equity)

            # ── Entry logic ───────────────────────────
            if position == 0:
                # Only enter on new bars with valid z
                if z < -entry_z:
                    # LONG spread: buy A, sell B
                    # Betting spread will rise to zero
                    position = 1
                    current_trade = Trade(
                        entry_bar    = global_i,
                        entry_time   = t,
                        direction    = 1,
                        entry_spread = spread,
                        entry_z      = z,
                        entry_beta   = 0.0)

                elif z > entry_z:
                    # SHORT spread: sell A, buy B
                    # Betting spread will fall to zero
                    position = -1
                    current_trade = Trade(
                        entry_bar    = global_i,
                        entry_time   = t,
                        direction    = -1,
                        entry_spread = spread,
                        entry_z      = z,
                        entry_beta   = 0.0)

            # ── Exit logic — LONG spread ──────────────
            elif position == 1:
                # We are long spread (bought A, sold B)
                # Profit when spread rises (z increases)
                # Exit when z returns to near zero
                pnl = spread - current_trade.entry_spread

                reason = None
                if z >= exit_z:
                    # Spread returned to mean → take profit
                    reason = "MEAN_CROSS"
                elif z < -stop_z:
                    # Spread moved further against us
                    # Cut loss — cointegration may be broken
                    reason = "STOP_LOSS"

                if reason:
                    current_trade.exit_bar    = global_i
                    current_trade.exit_time   = t
                    current_trade.exit_spread = spread
                    current_trade.exit_z      = z
                    current_trade.exit_reason = reason
                    current_trade.pnl_raw     = pnl
                    current_trade.bars_held   = (
                        global_i -
                        current_trade.entry_bar)
                    trades.append(current_trade)
                    equity = _apply_pnl(
                        equity, pnl,
                        current_trade.entry_spread)
                    peak_eq = max(peak_eq, equity)
                    position = 0
                    current_trade = None

            # ── Exit logic — SHORT spread ─────────────
            elif position == -1:
                # We are short spread (sold A, bought B)
                # Profit when spread falls (z decreases)
                # Exit when z returns to near zero
                pnl = (current_trade.entry_spread -
                       spread)

                reason = None
                if z <= -exit_z:
                    # Spread returned to mean → take profit
                    reason = "MEAN_CROSS"
                elif z > stop_z:
                    # Spread moved further against us
                    reason = "STOP_LOSS"

                if reason:
                    current_trade.exit_bar    = global_i
                    current_trade.exit_time   = t
                    current_trade.exit_spread = spread
                    current_trade.exit_z      = z
                    current_trade.exit_reason = reason
                    current_trade.pnl_raw     = pnl
                    current_trade.bars_held   = (
                        global_i -
                        current_trade.entry_bar)
                    trades.append(current_trade)
                    equity = _apply_pnl(
                        equity, pnl,
                        current_trade.entry_spread)
                    peak_eq = max(peak_eq, equity)
                    position = 0
                    current_trade = None

            eq_vals.append(equity)
            eq_times.append(t)

        # Move to next window
        test_start += step_bars

    # Close any trade still open at the end
    if position != 0 and current_trade and n > 0:
        final_i      = n - 1
        final_spread = all_spreads[final_i]
        pnl = (position *
               (final_spread -
                current_trade.entry_spread))
        current_trade.exit_bar    = final_i
        current_trade.exit_time   = times[final_i]
        current_trade.exit_spread = final_spread
        current_trade.exit_z      = 0.0
        current_trade.exit_reason = "END_OF_TEST"
        current_trade.pnl_raw     = pnl
        current_trade.bars_held   = (final_i -
                                     current_trade.entry_bar)
        trades.append(current_trade)

    equity_curve = pd.Series(
        eq_vals,
        index=pd.DatetimeIndex(eq_times))

    return trades, equity_curve


def _apply_pnl(equity: float,
               pnl: float,
               entry_spread: float,
               risk: float = 0.01) -> float:
    """
    Convert raw spread PnL to equity change.

    We normalise by the entry spread magnitude
    so that a full mean-reversion (spread moves
    from ±2σ back to 0) generates ~risk return.

    This is a simplified model — in live trading
    we would use actual lot sizes and pip values.
    """
    base = abs(entry_spread) + 1e-10
    # Fraction of spread recovered
    frac = pnl / base
    # Scale to risk per trade
    # frac=1 means full recovery → +risk return
    eq_change = frac * risk
    # Cap individual trade impact at ±5%
    eq_change = float(np.clip(eq_change, -0.05, 0.05))
    return equity * (1.0 + eq_change)


# ─────────────────────────────────────────────────────────────
#  STATISTICS
# ─────────────────────────────────────────────────────────────
def compute_stats(trades: List[Trade],
                  equity_curve: pd.Series) -> dict:
    if not trades:
        return {}

    pnl  = np.array([t.pnl_raw for t in trades])
    wins = pnl[pnl > 0]
    loss = pnl[pnl < 0]
    n    = len(pnl)

    wr   = len(wins) / n if n > 0 else 0
    gp   = wins.sum() if len(wins) > 0 else 0
    gl   = abs(loss.sum()) if len(loss) > 0 else 1e-10
    pf   = gp / gl

    ret    = equity_curve.pct_change().dropna()
    sharpe = float(ret.mean() /
                   (ret.std() + 1e-10) *
                   np.sqrt(252 * 24))

    roll_max = equity_curve.cummax()
    max_dd   = float(((equity_curve - roll_max) /
                      roll_max).min())

    holds = [t.bars_held for t in trades]
    exits = {}
    for t in trades:
        exits[t.exit_reason] = (
            exits.get(t.exit_reason, 0) + 1)

    return {
        'n_trades'     : n,
        'win_rate'     : round(wr, 4),
        'profit_factor': round(pf, 4),
        'sharpe'       : round(sharpe, 4),
        'max_dd'       : round(max_dd, 4),
        'avg_bars'     : round(np.mean(holds), 1),
        'min_bars'     : int(np.min(holds)),
        'max_bars'     : int(np.max(holds)),
        'total_pnl'    : round(float(pnl.sum()), 8),
        'avg_win'      : round(float(wins.mean()), 8)
                         if len(wins) > 0 else 0,
        'avg_loss'     : round(float(loss.mean()), 8)
                         if len(loss) > 0 else 0,
        'exit_reasons' : exits,
    }


# ─────────────────────────────────────────────────────────────
#  SANITY CHECK ON TRADE COUNT
# ─────────────────────────────────────────────────────────────
def sanity_check_trades(trades: List[Trade],
                        sym1: str, sym2: str,
                        n_bars: int):
    """
    Check that trade count and hold times
    are in realistic range.

    Expected for H1 pairs trading:
      - 20 to 500 trades over full history
      - Average hold: 15 to 200 bars
      - Win rate: 50% to 75%

    If outside these ranges → print WARNING
    """
    n = len(trades)
    trades_per_bar = n / n_bars if n_bars > 0 else 0

    print(f"\n    SANITY CHECK ({sym1}/{sym2}):")
    print(f"      Total trades:    {n}")
    print(f"      Trades per bar:  {trades_per_bar:.4f}")

    if n == 0:
        print(f"      ⚠ WARNING: No trades generated")
        print(f"        Check entry_z threshold")
    elif n > 1000:
        print(f"      ⚠ WARNING: Too many trades ({n})")
        print(f"        Expected 20-500 for H1 pairs trading")
        print(f"        This suggests exit is triggering "
              f"too early")
        print(f"        Check exit_z and z-score computation")
    elif n < 10:
        print(f"      ⚠ WARNING: Very few trades ({n})")
        print(f"        entry_z={2.0} may be too strict")
    else:
        print(f"      ✓ Trade count looks reasonable")

    if trades:
        holds = [t.bars_held for t in trades]
        avg_h = np.mean(holds)
        if avg_h < 5:
            print(f"      ⚠ WARNING: Avg hold = {avg_h:.1f} bars")
            print(f"        This is too short for H1 pairs")
            print(f"        Exit condition triggering too fast")
        elif avg_h > 300:
            print(f"      ⚠ WARNING: Avg hold = {avg_h:.1f} bars")
            print(f"        This may indicate exits are too tight")
        else:
            print(f"      ✓ Avg hold {avg_h:.1f} bars — OK")


# ─────────────────────────────────────────────────────────────
#  PLOT
# ─────────────────────────────────────────────────────────────
def plot_pair_backtest(sym1: str, sym2: str,
                       trades: List[Trade],
                       equity_curve: pd.Series,
                       stats: dict,
                       s1: pd.Series,
                       s2: pd.Series,
                       all_spreads: np.ndarray,
                       z_window: int):
    try:
        fig = plt.figure(figsize=(16, 14))
        gs  = gridspec.GridSpec(3, 2,
                                hspace=0.4, wspace=0.3)

        wr = stats.get('win_rate', 0)
        pf = stats.get('profit_factor', 0)
        sh = stats.get('sharpe', 0)
        dd = stats.get('max_dd', 0)
        n  = stats.get('n_trades', 0)

        fig.suptitle(
            f"{sym1}/{sym2}  "
            f"Trades={n}  "
            f"WR={wr:.1%}  "
            f"PF={pf:.2f}  "
            f"Sharpe={sh:.2f}  "
            f"MaxDD={dd:.1%}",
            fontsize=12)

        # 1. Equity curve
        ax1 = fig.add_subplot(gs[0, :])
        equity_curve.plot(ax=ax1, color='steelblue',
                          lw=1.5)
        ax1.axhline(1.0, color='gray',
                    ls='--', lw=0.8)
        ax1.set_title("Equity Curve (OOS walk-forward)")
        ax1.set_ylabel("Equity")
        ax1.grid(True, alpha=0.3)

        # Mark wins and losses on equity curve
        for t in trades:
            if t.exit_time is None:
                continue
            c = ('green' if t.pnl_raw > 0
                 else 'red')
            ax1.axvline(t.exit_time,
                        color=c, alpha=0.15, lw=0.5)

        # 2. Z-score with trade markers
        ax2 = fig.add_subplot(gs[1, :])
        z_all = compute_rolling_zscore(
            all_spreads, z_window)
        z_series = pd.Series(z_all, index=s1.index)
        z_series.iloc[-2016:].plot(
            ax=ax2, color='darkgreen', lw=0.6,
            alpha=0.8)
        for level, c, ls in [
                (entry_z,  'red',    '--'),
                (-entry_z, 'red',    '--'),
                (stop_z,   'darkred',':'),
                (-stop_z,  'darkred',':'),
                (0,        'black',  '-')]:
            ax2.axhline(level, color=c,
                        ls=ls, lw=0.8)
        ax2.set_title("Z-Score (recent 3 months shown)")
        ax2.grid(True, alpha=0.3)

        # Mark entries on z-score chart
        for t in trades:
            if t.entry_time not in z_series.index:
                continue
            c = 'blue' if t.direction == 1 else 'orange'
            ax2.scatter(t.entry_time, t.entry_z,
                        color=c, s=15, zorder=5,
                        alpha=0.5)

        # 3. PnL distribution
        ax3 = fig.add_subplot(gs[2, 0])
        pnl_vals = [t.pnl_raw for t in trades]
        if pnl_vals:
            pd.Series(pnl_vals).hist(
                ax=ax3, bins=30,
                color='steelblue',
                edgecolor='white', alpha=0.8)
        ax3.axvline(0, color='red', lw=1.5)
        ax3.set_title("PnL Distribution (spread units)")
        ax3.grid(True, alpha=0.3)

        # 4. Hold time distribution
        ax4 = fig.add_subplot(gs[2, 1])
        holds = [t.bars_held for t in trades]
        if holds:
            pd.Series(holds).hist(
                ax=ax4, bins=25,
                color='darkorange',
                edgecolor='white', alpha=0.8)
        avg_h = np.mean(holds) if holds else 0
        ax4.set_title(
            f"Hold Period Distribution  "
            f"avg={avg_h:.0f} bars")
        ax4.set_xlabel("Bars held")
        ax4.grid(True, alpha=0.3)

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


# entry_z for plot — needs to be visible
entry_z = 2.0
stop_z  = 3.5


# ─────────────────────────────────────────────────────────────
#  PORTFOLIO COMBINATION
# ─────────────────────────────────────────────────────────────
def plot_portfolio(results: list):
    """
    Combine all pair equity curves into one portfolio.
    Equal weight.
    """
    curves = [(r['pair'], r['equity'])
              for r in results
              if r.get('equity') is not None
              and len(r['equity']) > 10]

    if len(curves) < 2:
        return

    # Build combined DataFrame
    frames = []
    for pair, eq in curves:
        # Remove duplicate timestamps
        eq = eq[~eq.index.duplicated(keep='last')]
        frames.append(eq.rename(pair))

    combined = pd.concat(frames, axis=1)
    combined = combined.ffill().fillna(1.0)
    portfolio = combined.mean(axis=1)

    ret    = portfolio.pct_change().dropna()
    sharpe = float(ret.mean() /
                   (ret.std() + 1e-10) *
                   np.sqrt(252 * 24))
    rm     = portfolio.cummax()
    max_dd = float(((portfolio - rm) / rm).min())
    ret_t  = float(portfolio.iloc[-1] /
                   portfolio.iloc[0] - 1)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(
        f"Portfolio: {len(curves)} pairs  "
        f"Sharpe={sharpe:.2f}  "
        f"MaxDD={max_dd:.1%}  "
        f"Return={ret_t:.1%}",
        fontsize=12)

    combined.plot(ax=axes[0], lw=0.8, alpha=0.7)
    axes[0].axhline(1.0, color='k', ls='--', lw=0.8)
    axes[0].set_title("Individual Pair Curves")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    portfolio.plot(ax=axes[1], color='darkblue', lw=1.5)
    axes[1].axhline(1.0, color='k', ls='--', lw=0.8)
    axes[1].fill_between(
        portfolio.index, portfolio.values, 1.0,
        where=(portfolio.values >= 1.0),
        alpha=0.3, color='green', label='Profit')
    axes[1].fill_between(
        portfolio.index, portfolio.values, 1.0,
        where=(portfolio.values < 1.0),
        alpha=0.3, color='red', label='Loss')
    axes[1].set_title("Portfolio Equity")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "portfolio.png"),
        dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n  [PLOT] portfolio.png")

    print(f"\n  PORTFOLIO STATS:")
    print(f"    Pairs:      {len(curves)}")
    print(f"    Sharpe:     {sharpe:.2f}")
    print(f"    Max DD:     {max_dd:.1%}")
    print(f"    Return:     {ret_t:.1%}")


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def run_all_backtests(timeframe: str = "H1"):
    print(f"\n{'='*55}")
    print(f"  {BT_VERSION}")
    print(f"{'='*55}")

    vp_file = os.path.join(OUTPUT_DIR,
                            "valid_pairs.json")
    if not os.path.exists(vp_file):
        print("[ERR] valid_pairs.json not found.")
        print("      Run pairs_research.py first.")
        return []

    with open(vp_file) as f:
        valid_pairs = json.load(f)

    print(f"\nFound {len(valid_pairs)} valid pairs")

    all_results  = []
    summary_rows = []

    for p in valid_pairs:
        sym1 = p['symbol1']
        sym2 = p['symbol2']

        s1 = load_price_series(sym1, timeframe)
        s2 = load_price_series(sym2, timeframe)

        if len(s1) < 3000 or len(s2) < 3000:
            print(f"  [SKIP] {sym1}/{sym2}: "
                  f"not enough data")
            continue

        s1a, s2a = align_series(s1, s2)
        if len(s1a) < 3000:
            print(f"  [SKIP] {sym1}/{sym2}: "
                  f"not enough aligned bars")
            continue

        try:
            trades, equity = backtest_pair(
                sym1        = sym1,
                sym2        = sym2,
                s1          = s1a,
                s2          = s2a,
                entry_z     = 2.0,
                exit_z      = 0.3,
                stop_z      = 3.5,
                delta       = 1e-4,
                train_bars  = 2016,
                test_bars   = 336,
                step_bars   = 168,
                max_dd_pct  = 0.20,
            )
        except Exception as e:
            print(f"  [ERR] {sym1}/{sym2}: {e}")
            traceback.print_exc()
            continue

        # Sanity check trade count
        sanity_check_trades(trades, sym1, sym2,
                            len(s1a))

        if not trades:
            print(f"  [SKIP] No trades for "
                  f"{sym1}/{sym2}")
            continue

        stats = compute_stats(trades, equity)

        # Pre-compute spread for plot
        spreads = compute_kalman_spread(
            s1a.values.astype(float),
            s2a.values.astype(float))
        hl = p.get('half_life', 30)
        if not np.isfinite(hl) or hl <= 0:
            hl = 30
        z_win = int(np.clip(hl * 2, 20, 200))

        plot_pair_backtest(
            sym1, sym2, trades, equity, stats,
            s1a, s2a, spreads, z_win)

        # Save trade log
        trade_rows = []
        for t in trades:
            trade_rows.append({
                'entry_time'  : str(t.entry_time),
                'exit_time'   : str(t.exit_time),
                'direction'   : t.direction,
                'entry_spread': t.entry_spread,
                'exit_spread' : t.exit_spread,
                'entry_z'     : t.entry_z,
                'exit_z'      : t.exit_z,
                'exit_reason' : t.exit_reason,
                'pnl_raw'     : t.pnl_raw,
                'bars_held'   : t.bars_held,
            })
        pd.DataFrame(trade_rows).to_csv(
            os.path.join(OUTPUT_DIR,
                         f"trades_{sym1}_{sym2}.csv"),
            index=False)

        print(f"\n  ══ {sym1}/{sym2} RESULTS ══")
        print(f"    Trades:        {stats['n_trades']}")
        print(f"    Win Rate:      "
              f"{stats['win_rate']:.1%}")
        print(f"    Profit Factor: "
              f"{stats['profit_factor']:.2f}")
        print(f"    Sharpe:        {stats['sharpe']:.2f}")
        print(f"    Max Drawdown:  {stats['max_dd']:.1%}")
        print(f"    Avg Hold:      "
              f"{stats['avg_bars']:.0f} bars  "
              f"(min={stats['min_bars']} "
              f"max={stats['max_bars']})")
        print(f"    Exit reasons:  "
              f"{stats['exit_reasons']}")

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
            'avg_bars_held': stats['avg_bars'],
            'min_hold'     : stats['min_bars'],
            'half_life'    : p.get('half_life', 0),
            'eg_pval'      : p.get('eg_pval', 1),
        })

    # Summary
    if summary_rows:
        df_s = pd.DataFrame(summary_rows).sort_values(
            'profit_factor', ascending=False)
        print(f"\n{'='*70}")
        print(f"  BACKTEST SUMMARY")
        print(f"{'='*70}")
        print(df_s.to_string(index=False))
        df_s.to_csv(
            os.path.join(OUTPUT_DIR,
                         "backtest_summary.csv"),
            index=False)

    # Portfolio
    if len(all_results) >= 2:
        plot_portfolio(all_results)

    print(f"\n  [DONE] {BT_VERSION}")
    return all_results


if __name__ == "__main__":
    run_all_backtests(timeframe="H1")
