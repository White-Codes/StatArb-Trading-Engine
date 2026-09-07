"""
pairs_backtest.py  v6.0
========================
Root cause identified and fixed:

  THE FUNDAMENTAL BUG (all previous versions):
  ─────────────────────────────────────────────
  Kalman filter updates beta EVERY bar using the
  current prices. This means spread[i] is always
  the residual of a model fitted ON bar i.
  
  By construction this residual mean-reverts to zero
  because that is what the Kalman filter minimises.
  
  Trading this spread is not a market strategy —
  it is trading the fitting error of an adaptive
  model. Win rate approaches 100% mathematically.

  THE FIX:
  ─────────────────────────────────────────────
  1. Estimate OLS hedge ratio on TRAINING bars only
  2. Apply that FIXED ratio to TEST bars
  3. The test spread is now genuinely out-of-sample
  4. Mean reversion must come from the MARKET,
     not from the fitting procedure
  
  Walk-forward: re-estimate ratio each new window
  using training bars, test on next window only.

Expected realistic results after fix:
  Trades:       30-150 per pair
  Win rate:     52-65%
  PF:           1.1-1.8  
  Sharpe:       0.3-1.2
  Avg hold:     20-100 bars
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from typing      import List, Optional
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import json, os, traceback

from pairs_research import (
    load_price_series, align_series,
    compute_half_life
)

OUTPUT_DIR = "pairs_artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)
BT_VERSION = "backtest-v6.0"

# ── Parameters ────────────────────────────────────────────────
ENTRY_Z       = 2.0
EXIT_Z        = 0.5
STOP_Z        = 3.5
TRAIN_BARS    = 2016    # ~3 months H1: OLS estimation
TEST_BARS     = 336     # ~2 weeks H1: OOS trading
# step = test: zero overlap guaranteed
MAX_TRADES    = 300     # sanity gate


# ─────────────────────────────────────────────────────────────
#  TRADE RECORD
# ─────────────────────────────────────────────────────────────
@dataclass
class Trade:
    entry_bar     : int
    entry_time    : pd.Timestamp
    direction     : int        # +1 long spread, -1 short
    entry_spread  : float
    entry_z       : float
    spread_std    : float      # training spread std
    exit_bar      : int           = 0
    exit_time     : pd.Timestamp  = None
    exit_spread   : float         = 0.0
    exit_z        : float         = 0.0
    exit_reason   : str           = ""
    pnl_raw       : float         = 0.0
    pnl_norm      : float         = 0.0  # pnl / spread_std
    bars_held     : int           = 0


# ─────────────────────────────────────────────────────────────
#  OLS HEDGE RATIO — training only
# ─────────────────────────────────────────────────────────────
def estimate_ols(s1_train: np.ndarray,
                 s2_train: np.ndarray
                 ) -> tuple[float, float, float]:
    """
    Fit OLS on log prices using training bars.
    
    Returns: (beta, alpha, spread_std)
    
    beta and alpha are FIXED for the test window.
    spread_std is used to normalise PnL.
    """
    log1 = np.log(s1_train)
    log2 = np.log(s2_train)

    x = add_constant(log2)
    m = OLS(log1, x).fit()

    alpha = float(m.params.iloc[0])
    beta  = float(m.params.iloc[1])

    # In-sample spread std for normalisation
    spread_train = log1 - beta * log2 - alpha
    std          = float(np.std(spread_train, ddof=1))
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
    Apply training-estimated beta/alpha to test prices.
    
    spread[i] = log(s1[i]) - beta*log(s2[i]) - alpha
    
    This spread is genuinely OOS — the model parameters
    were fixed before seeing any test bar.
    """
    log1 = np.log(s1_test)
    log2 = np.log(s2_test)
    return log1 - beta * log2 - alpha


# ─────────────────────────────────────────────────────────────
#  Z-SCORE — uses training stats (fixed mean/std)
# ─────────────────────────────────────────────────────────────
def compute_test_zscore(test_spread:   np.ndarray,
                        train_spread:  np.ndarray,
                        ) -> np.ndarray:
    """
    Standardise test spread using TRAINING mean and std.
    
    z[i] = (test_spread[i] - train_mean) / train_std
    
    Why training stats:
    - Using rolling test stats still leaks information
      because future test bars shift the rolling window
    - Training mean/std are fixed before test starts
    - This is the correct OOS z-score
    
    Note: if the cointegration relationship is stable,
    the test spread should have mean ~0 and std ~1
    relative to training stats.
    """
    mu  = float(np.mean(train_spread))
    sig = float(np.std(train_spread, ddof=1))
    if sig < 1e-10:
        sig = 1.0
    return (test_spread - mu) / sig


# ─────────────────────────────────────────────────────────────
#  CORE ENGINE  v6.0
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
    Walk-forward backtest with FIXED OLS hedge ratio.
    
    Per window:
      1. Fit OLS on s1[train_start:train_end]
         → fixed beta, alpha
      2. Compute training spread → get mean, std
      3. Apply fixed ratio to s1[test_start:test_end]
         → OOS spread (no future info)
      4. Z-score using training mean/std (no future info)
      5. Trade on z-score signals
      6. Advance: train_start += test_bars (no overlap)
    
    Returns: trades, equity_curve, diagnostics
    """
    s1v   = s1.values.astype(float)
    s2v   = s2.values.astype(float)
    times = s1.index
    n     = len(s1v)

    print(f"\n  ── {sym1}/{sym2}  ({n:,} bars) ──")
    print(f"    Train={train_bars}  Test={test_bars}  "
          f"(step=test, zero overlap)")

    trades:     List[Trade] = []
    eq_times:   List        = []
    eq_vals:    List[float] = []
    cum_pnl_n   = 0.0   # cumulative normalised PnL

    window_n   = 0
    train_start = 0

    while train_start + train_bars + test_bars <= n:
        window_n   += 1
        train_end   = train_start + train_bars
        test_end    = min(train_end + test_bars, n)
        test_len    = test_end - train_end

        if test_len < 20:
            train_start += test_bars
            continue

        # ── Step 1: OLS on training bars ─────────────
        try:
            beta, alpha, train_std = estimate_ols(
                s1v[train_start: train_end],
                s2v[train_start: train_end])
        except Exception as e:
            print(f"    Win {window_n}: OLS failed: {e}")
            train_start += test_bars
            continue

        # ── Step 2: Training spread stats ────────────
        log1_tr = np.log(s1v[train_start: train_end])
        log2_tr = np.log(s2v[train_start: train_end])
        train_sp = log1_tr - beta * log2_tr - alpha
        # Half-life from training (for info only)
        hl = compute_half_life(
            pd.Series(train_sp))
        if not np.isfinite(hl) or hl <= 0:
            hl = 30.0
        hl = float(np.clip(hl, 2.0, 200.0))

        # ── Step 3: OOS test spread ───────────────────
        test_sp = apply_fixed_spread(
            s1v[train_end: test_end],
            s2v[train_end: test_end],
            beta, alpha)

        # ── Step 4: Z-score using training stats ─────
        test_z = compute_test_zscore(
            test_sp, train_sp)

        t_start = times[train_end]
        t_end   = times[test_end - 1]
        print(f"    Win {window_n}: "
              f"{t_start.date()} → {t_end.date()}  "
              f"β={beta:.4f}  "
              f"HL={hl:.1f}  "
              f"TrainStd={train_std:.6f}",
              end="")

        # ── Step 5: Trade ─────────────────────────────
        position:      int             = 0
        current_trade: Optional[Trade] = None
        window_trades  = 0

        # Min hold: half-life based, at least 6 bars
        min_hold = int(np.clip(hl / 2, 6, 60))

        for li in range(test_len):
            gi = train_end + li
            sp = test_sp[li]
            z  = test_z[li]
            t  = times[gi]

            # ── Entry ─────────────────────────────────
            if position == 0:
                if z < -entry_z:
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
                    position      = -1
                    current_trade = Trade(
                        entry_bar    = gi,
                        entry_time   = t,
                        direction    = -1,
                        entry_spread = sp,
                        entry_z      = z,
                        spread_std   = train_std)
                    window_trades += 1

            # ── Exit: Long ────────────────────────────
            elif position == 1:
                bars_in = gi - current_trade.entry_bar
                pnl_raw = sp - current_trade.entry_spread
                pnl_n   = pnl_raw / train_std

                if bars_in < min_hold:
                    # Enforce hold, track unrealised
                    eq_times.append(t)
                    eq_vals.append(
                        cum_pnl_n + pnl_n)
                    continue

                reason = None
                if z >= exit_z:
                    reason = "MEAN_CROSS"
                elif z < -stop_z:
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
                    cum_pnl_n  += pnl_n
                    position    = 0
                    current_trade = None
                else:
                    # Still in trade
                    eq_times.append(t)
                    eq_vals.append(cum_pnl_n + pnl_n)
                    continue

            # ── Exit: Short ───────────────────────────
            elif position == -1:
                bars_in = gi - current_trade.entry_bar
                pnl_raw = current_trade.entry_spread - sp
                pnl_n   = pnl_raw / train_std

                if bars_in < min_hold:
                    eq_times.append(t)
                    eq_vals.append(cum_pnl_n + pnl_n)
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
                    cum_pnl_n  += pnl_n
                    position    = 0
                    current_trade = None
                else:
                    eq_times.append(t)
                    eq_vals.append(cum_pnl_n + pnl_n)
                    continue

            # Realised point
            eq_times.append(t)
            eq_vals.append(cum_pnl_n)

        print(f"  → {window_trades} trades")
        # FIX: step by test_bars only (zero overlap)
        train_start += test_bars

    # ── Equity curve ──────────────────────────────────
    if eq_times:
        equity = pd.Series(
            eq_vals,
            index=pd.DatetimeIndex(eq_times))
        equity = equity[
            ~equity.index.duplicated(keep='last')]
        equity = equity.sort_index()
        # Convert normalised PnL to equity
        # 1 train_std unit = 3% equity move
        scale  = 0.03
        equity = 1.0 + equity * scale
        equity = equity.clip(lower=0.01)
    else:
        equity = pd.Series(
            [1.0], index=[s1.index[0]])

    return trades, equity, {'n_windows': window_n}


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

    gp  = wins.sum() if len(wins) > 0 else 0.0
    gl  = abs(loss.sum()) if len(loss) > 0 else 1e-10

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
        'win_rate'     : round(len(wins)/n, 4),
        'profit_factor': round(gp/gl, 4),
        'sharpe'       : round(sharpe, 4),
        'max_dd'       : round(max_dd, 4),
        'avg_hold'     : round(float(np.mean(holds)), 1),
        'min_hold'     : int(np.min(holds)),
        'max_hold'     : int(np.max(holds)),
        'total_pnl_n'  : round(float(pnl.sum()), 4),
        'exit_reasons' : exits,
    }


# ─────────────────────────────────────────────────────────────
#  SANITY CHECKER — hard gates with clear diagnostics
# ─────────────────────────────────────────────────────────────
def sanity_check(trades: List[Trade],
                 sym1:   str,
                 sym2:   str,
                 stats:  dict) -> bool:
    n   = len(trades)
    wr  = stats.get('win_rate', 0)
    pf  = stats.get('profit_factor', 0)
    ah  = stats.get('avg_hold', 0)
    mh  = stats.get('min_hold', 0)

    print(f"\n    ── SANITY: {sym1}/{sym2} ──")
    print(f"      Trades      : {n}")
    print(f"      Win rate    : {wr:.1%}")
    print(f"      PF          : {pf:.2f}")
    print(f"      Avg hold    : {ah:.1f} bars")
    print(f"      Min hold    : {mh} bars")

    failures = []

    if n > MAX_TRADES:
        failures.append(
            f"Trade count {n} > {MAX_TRADES}: "
            f"exit logic still too aggressive")

    if wr > 0.75:
        failures.append(
            f"Win rate {wr:.1%} > 75%: "
            f"look-ahead bias likely present")

    if pf > 4.0:
        failures.append(
            f"Profit factor {pf:.2f} > 4.0: "
            f"unrealistic")

    if ah < 20 and n > 10:
        failures.append(
            f"Avg hold {ah:.1f} < 20 bars: "
            f"exits still too fast")

    if failures:
        print(f"      ✗ SANITY FAILURES:")
        for f in failures:
            print(f"        - {f}")
        return False
    else:
        print(f"      ✓ All checks passed")

    # Sample trades
    print(f"\n      Sample trades (first 6):")
    print(f"      {'Date':>10} {'Dir':>4} "
          f"{'EnZ':>6} {'ExZ':>6} "
          f"{'Hold':>5} {'PnL_n':>7} Reason")
    for t in trades[:6]:
        d = '+L' if t.direction == 1 else '-S'
        print(f"      "
              f"{str(t.entry_time)[:10]} "
              f"{d} "
              f"{t.entry_z:>6.2f} "
              f"{t.exit_z:>6.2f} "
              f"{t.bars_held:>5} "
              f"{t.pnl_norm:>7.4f} "
              f"{t.exit_reason}")
    return True


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

        ax1 = fig.add_subplot(gs[0, :])
        equity.plot(ax=ax1, color='steelblue', lw=1.5)
        ax1.axhline(1.0, color='gray',
                    ls='--', lw=0.8)
        ax1.set_title(
            "Equity — OOS only, fixed OLS hedge ratio, "
            "zero window overlap")
        ax1.set_ylabel("Equity (1 = start)")
        ax1.grid(True, alpha=0.3)

        for t in trades[:200]:
            if t.exit_time is None:
                continue
            c = ('green' if t.pnl_norm > 0
                 else 'red')
            try:
                ax1.axvline(
                    t.exit_time, color=c,
                    alpha=0.08, lw=0.5)
            except Exception:
                pass

        ax2 = fig.add_subplot(gs[1, 0])
        pd.Series([t.pnl_norm for t in trades]
                  ).hist(ax=ax2, bins=40,
                         color='steelblue',
                         edgecolor='white',
                         alpha=0.8)
        ax2.axvline(0, color='red', lw=1.5)
        ax2.set_title("PnL (σ-normalised)")
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[1, 1])
        pd.Series([t.bars_held for t in trades]
                  ).hist(ax=ax3, bins=30,
                         color='darkorange',
                         edgecolor='white',
                         alpha=0.8)
        ah = stats['avg_hold']
        ax3.set_title(f"Hold Periods (avg={ah:.0f}b)")
        ax3.set_xlabel("Bars (H1)")
        ax3.grid(True, alpha=0.3)

        fname = os.path.join(
            OUTPUT_DIR,
            f"bt_{sym1}_{sym2}.png")
        plt.savefig(fname, dpi=120,
                    bbox_inches='tight')
        plt.close()
        print(f"    [PLOT] {fname}")

    except Exception as e:
        print(f"    [PLOT ERR] {e}")
        traceback.print_exc()


def plot_portfolio(results: list):
    curves = [
        (r['pair'], r['equity'])
        for r in results
        if (r.get('equity') is not None
            and len(r.get('equity', [])) > 10)
    ]
    if not curves:
        return

    frames = []
    for pair, eq in curves:
        eq = eq[~eq.index.duplicated(keep='last')]
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
    max_dd = float(((portfolio - rm) / rm).min())
    tot_r  = float(portfolio.iloc[-1] - 1.0)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(
        f"Portfolio {len(curves)} pairs  "
        f"Sharpe={sharpe:.2f}  "
        f"MaxDD={max_dd:.1%}  "
        f"Return={tot_r:.1%}",
        fontsize=12)

    combined.plot(ax=axes[0], lw=0.8, alpha=0.7)
    axes[0].axhline(1.0, color='k',
                    ls='--', lw=0.8)
    axes[0].set_title("Individual Pairs")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    portfolio.plot(ax=axes[1],
                   color='darkblue', lw=2)
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
    print(f"    Pairs  : {len(curves)}")
    print(f"    Sharpe : {sharpe:.2f}")
    print(f"    Max DD : {max_dd:.1%}")
    print(f"    Return : {tot_r:.1%}")

    with open(os.path.join(
            OUTPUT_DIR, "portfolio_stats.json"),
              'w') as f:
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
    print(f"  Architecture: Fixed OLS hedge ratio")
    print(f"  OLS fit on training bars only")
    print(f"  Fixed ratio applied to test bars (OOS)")
    print(f"  Z-score uses training mean/std only")
    print(f"  Zero window overlap (step = test_bars)")
    print(f"  Entry: ±{ENTRY_Z}σ  Exit: {EXIT_Z}σ  "
          f"Stop: ±{STOP_Z}σ")
    print(f"{'='*60}")

    vp_file = os.path.join(
        OUTPUT_DIR, "valid_pairs.json")
    if not os.path.exists(vp_file):
        print("[ERR] valid_pairs.json not found")
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
                  f"insufficient data")
            continue

        s1a, s2a = align_series(s1, s2)
        if len(s1a) < min_bars:
            print(f"  [SKIP] {sym1}/{sym2}: "
                  f"insufficient aligned data")
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
                train_bars  = TRAIN_BARS,
                test_bars   = TEST_BARS,
            )
        except Exception as e:
            print(f"  [ERR] {sym1}/{sym2}: {e}")
            traceback.print_exc()
            continue

        if not trades:
            print(f"  [SKIP] No trades: {sym1}/{sym2}")
            continue

        stats = compute_stats(trades, equity)
        passed = sanity_check(
            trades, sym1, sym2, stats)

        plot_results(
            sym1, sym2, trades, equity, stats)

        # Save CSV
        rows = [{
            'entry_time'  : str(t.entry_time),
            'exit_time'   : str(t.exit_time),
            'direction'   : t.direction,
            'entry_z'     : round(t.entry_z, 4),
            'exit_z'      : round(t.exit_z, 4),
            'bars_held'   : t.bars_held,
            'pnl_raw'     : t.pnl_raw,
            'pnl_norm'    : round(t.pnl_norm, 6),
            'exit_reason' : t.exit_reason,
            'spread_std'  : t.spread_std,
        } for t in trades]

        pd.DataFrame(rows).to_csv(
            os.path.join(
                OUTPUT_DIR,
                f"trades_{sym1}_{sym2}.csv"),
            index=False)

        print(f"\n  ══ {sym1}/{sym2} ══")
        for k, v in stats.items():
            print(f"    {k:<16}: {v}")

        all_results.append({
            'pair'         : f"{sym1}/{sym2}",
            'equity'       : equity,
            'sanity_passed': passed,
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
            'half_life'    : p.get('half_life', 0),
        })

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

    if all_results:
        plot_portfolio(all_results)

    print(f"\n  [DONE] {BT_VERSION}")
    return all_results


if __name__ == "__main__":
    run_all_backtests(timeframe="H1")
