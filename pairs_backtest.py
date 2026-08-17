"""
pairs_backtest.py
=================
Full walk-forward backtest for the pairs trading system.
Uses Kalman filter for adaptive hedge ratio.

Walk-forward design:
  - Train window: 6 months  (calibrate parameters)
  - Test window:  1 month   (out-of-sample)
  - Step:         1 month   (re-calibrate monthly)
  - Never look ahead

This is the ONLY honest way to measure performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import List, Optional
import json
import os

from kalman_filter import KalmanHedgeFilter, kalman_filter_batch
from pairs_research import (
    compute_half_life, compute_hurst,
    load_price_series, align_series,
    estimate_hedge_ratio
)

OUTPUT_DIR = "pairs_artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
#  TRADE RECORD
# ─────────────────────────────────────────────
@dataclass
class Trade:
    entry_bar    : int
    entry_time   : pd.Timestamp
    direction    : int        # +1=long spread, -1=short
    entry_spread : float
    entry_z      : float
    entry_beta   : float
    exit_bar     : int        = 0
    exit_time    : pd.Timestamp = None
    exit_spread  : float      = 0.0
    exit_z       : float      = 0.0
    exit_reason  : str        = ""
    pnl_spread   : float      = 0.0
    pnl_pct      : float      = 0.0
    bars_held    : int        = 0


# ─────────────────────────────────────────────
#  BACKTEST ENGINE
# ─────────────────────────────────────════════
class PairsBacktester:

    def __init__(self,
                 sym1: str, sym2: str,
                 s1:   pd.Series, s2: pd.Series,
                 # Signal parameters
                 entry_z:    float = 2.0,
                 exit_z:     float = 0.3,
                 stop_z:     float = 3.5,
                 # Kalman parameters
                 delta:      float = 1e-4,
                 warmup:     int   = 200,
                 # Walk-forward
                 train_bars: int   = 1000,
                 test_bars:  int   = 168,   # 1 week H1
                 step_bars:  int   = 168,
                 # Risk
                 max_dd:     float = 0.15):

        self.sym1       = sym1
        self.sym2       = sym2
        self.s1         = s1
        self.s2         = s2
        self.entry_z    = entry_z
        self.exit_z     = exit_z
        self.stop_z     = stop_z
        self.delta      = delta
        self.warmup     = warmup
        self.train_bars = train_bars
        self.test_bars  = test_bars
        self.step_bars  = step_bars
        self.max_dd     = max_dd

        self.trades: List[Trade] = []
        self.equity_curve = []

    def _compute_zscore_window(self,
                                spread_arr: np.ndarray,
                                window: int,
                                idx: int) -> float:
        """Rolling z-score at index idx."""
        start = max(0, idx - window + 1)
        sub   = spread_arr[start:idx+1]
        if len(sub) < 5:
            return 0.0
        mu  = sub.mean()
        sig = sub.std()
        if sig < 1e-10:
            return 0.0
        return (sub[-1] - mu) / sig

    def run_walkforward(self) -> pd.DataFrame:
        """
        Walk-forward backtest.
        Returns trade DataFrame.
        """
        n       = len(self.s1)
        s1_vals = self.s1.values
        s2_vals = self.s2.values
        times   = self.s1.index

        equity      = 1.0
        peak_equity = 1.0
        halted      = False
        all_equity  = [1.0]
        all_times   = [times[0]]

        print(f"\n  Walk-Forward: {self.sym1}/{self.sym2}")
        print(f"  Bars: {n}  Train:{self.train_bars}  "
              f"Test:{self.test_bars}  Step:{self.step_bars}")

        # First test window starts after training data
        test_start = self.train_bars

        window_num = 0
        while test_start + self.test_bars <= n:
            window_num += 1
            test_end = test_start + self.test_bars

            # ── TRAIN: calibrate Kalman on train window ──
            train_s1 = self.s1.iloc[
                max(0, test_start - self.train_bars):
                test_start]
            train_s2 = self.s2.iloc[
                max(0, test_start - self.train_bars):
                test_start]

            # Compute half-life on training data
            kf_train = kalman_filter_batch(
                train_s1, train_s2, delta=self.delta)
            spread_train = pd.Series(
                kf_train['spread'].values)
            half_life = compute_half_life(spread_train)
            if not np.isfinite(half_life):
                half_life = 20
            zscore_window = max(20,
                                min(100,
                                    int(half_life * 2)))

            # ── TEST: run on out-of-sample window ───────
            # Initialise Kalman with full training history
            # then continue updating in test period
            kf = KalmanHedgeFilter(delta=self.delta)

            # Warm up on training data
            for pa, pb in zip(train_s1.values,
                               train_s2.values):
                kf.update(pa, pb)

            # Now trade the test window
            position   = 0   # 0=flat, 1=long, -1=short
            current_trade: Optional[Trade] = None

            test_s1 = s1_vals[test_start:test_end]
            test_s2 = s2_vals[test_start:test_end]
            test_tm = times[test_start:test_end]

            # Spread buffer for rolling z-score
            spread_buf = np.array(
                kf.spread_history[-zscore_window*2:])

            for i in range(len(test_s1)):
                pa = test_s1[i]
                pb = test_s2[i]
                t  = test_tm[i]

                # Update Kalman
                beta, alpha, spread = kf.update(pa, pb)
                spread_buf = np.append(spread_buf, spread)

                # Compute z-score
                z = self._compute_zscore_window(
                    spread_buf, zscore_window,
                    len(spread_buf)-1)

                abs_idx = test_start + i

                # ── Check drawdown halt ──────────────────
                if equity < peak_equity * (1 - self.max_dd):
                    if not halted:
                        print(f"    [HALT] DD>={self.max_dd:.0%}"
                              f" at bar {abs_idx}")
                        halted = True
                    # Close position if open
                    if position != 0 and current_trade:
                        pnl = (position *
                               (spread - current_trade
                                .entry_spread))
                        current_trade.exit_bar    = abs_idx
                        current_trade.exit_time   = t
                        current_trade.exit_spread = spread
                        current_trade.exit_z      = z
                        current_trade.exit_reason = "DD_HALT"
                        current_trade.pnl_spread  = pnl
                        current_trade.bars_held   = i
                        self.trades.append(current_trade)
                        equity *= (1 + pnl /
                                   abs(current_trade
                                       .entry_spread +
                                       1e-10))
                        position = 0
                        current_trade = None

                if halted:
                    all_equity.append(equity)
                    all_times.append(t)
                    continue

                peak_equity = max(peak_equity, equity)

                # ── POSITION MANAGEMENT ──────────────────

                if position == 0:
                    # Entry signals
                    if z < -self.entry_z:
                        position = 1
                        current_trade = Trade(
                            entry_bar   = abs_idx,
                            entry_time  = t,
                            direction   = 1,
                            entry_spread= spread,
                            entry_z     = z,
                            entry_beta  = beta,
                        )
                    elif z > self.entry_z:
                        position = -1
                        current_trade = Trade(
                            entry_bar   = abs_idx,
                            entry_time  = t,
                            direction   = -1,
                            entry_spread= spread,
                            entry_z     = z,
                            entry_beta  = beta,
                        )

                elif position == 1:
                    # Long spread — waiting for z to rise
                    pnl_raw = spread - current_trade.entry_spread

                    exit_reason = None
                    if z >= self.exit_z:
                        exit_reason = "MEAN_CROSS"
                    elif z < -self.stop_z:
                        exit_reason = "STOP_LOSS"

                    if exit_reason:
                        current_trade.exit_bar    = abs_idx
                        current_trade.exit_time   = t
                        current_trade.exit_spread = spread
                        current_trade.exit_z      = z
                        current_trade.exit_reason = exit_reason
                        current_trade.pnl_spread  = pnl_raw
                        current_trade.bars_held   = (
                            abs_idx - current_trade.entry_bar)
                        self.trades.append(current_trade)

                        eq_chg = pnl_raw / (
                            abs(current_trade.entry_spread)
                            + 1e-10)
                        equity *= (1 + eq_chg * 0.1)
                        peak_equity = max(peak_equity, equity)
                        position = 0
                        current_trade = None

                elif position == -1:
                    # Short spread — waiting for z to fall
                    pnl_raw = (current_trade.entry_spread
                               - spread)

                    exit_reason = None
                    if z <= self.exit_z:
                        exit_reason = "MEAN_CROSS"
                    elif z > self.stop_z:
                        exit_reason = "STOP_LOSS"

                    if exit_reason:
                        current_trade.exit_bar    = abs_idx
                        current_trade.exit_time   = t
                        current_trade.exit_spread = spread
                        current_trade.exit_z      = z
                        current_trade.exit_reason = exit_reason
                        current_trade.pnl_spread  = pnl_raw
                        current_trade.bars_held   = (
                            abs_idx - current_trade.entry_bar)
                        self.trades.append(current_trade)

                        eq_chg = pnl_raw / (
                            abs(current_trade.entry_spread)
                            + 1e-10)
                        equity *= (1 + eq_chg * 0.1)
                        peak_equity = max(peak_equity, equity)
                        position = 0
                        current_trade = None

                all_equity.append(equity)
                all_times.append(t)

            test_start += self.step_bars

        self.equity_curve = pd.Series(
            all_equity, index=all_times)
        return self._build_trade_df()

    def _build_trade_df(self) -> pd.DataFrame:
        """Convert trade list to DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        rows = []
        for t in self.trades:
            rows.append({
                'entry_time'  : t.entry_time,
                'exit_time'   : t.exit_time,
                'direction'   : t.direction,
                'entry_spread': t.entry_spread,
                'exit_spread' : t.exit_spread,
                'entry_z'     : t.entry_z,
                'exit_z'      : t.exit_z,
                'exit_reason' : t.exit_reason,
                'pnl_spread'  : t.pnl_spread,
                'bars_held'   : t.bars_held,
            })
        return pd.DataFrame(rows)

    def get_statistics(self,
                       trade_df: pd.DataFrame) -> dict:
        """Compute full performance statistics."""
        if trade_df.empty:
            return {}

        pnl = trade_df['pnl_spread'].values
        wins   = pnl[pnl > 0]
        losses = pnl[pnl < 0]

        n      = len(pnl)
        wr     = len(wins) / n if n > 0 else 0
        gp     = wins.sum()   if len(wins)   > 0 else 0
        gl     = abs(losses.sum()) if len(losses) > 0 else 1e-10
        pf     = gp / gl

        # Equity curve metrics
        eq     = self.equity_curve
        ret    = eq.pct_change().dropna()
        sharpe = (ret.mean() / (ret.std() + 1e-10)
                  * np.sqrt(252 * 24))  # H1 annualised

        roll_max = eq.cummax()
        dd       = (eq - roll_max) / roll_max
        max_dd   = dd.min()

        # Exit reason breakdown
        reasons = trade_df['exit_reason'].value_counts()

        avg_bars = trade_df['bars_held'].mean()

        stats = {
            'n_trades'       : n,
            'win_rate'       : round(wr, 4),
            'profit_factor'  : round(pf, 4),
            'sharpe'         : round(sharpe, 4),
            'max_drawdown'   : round(max_dd, 4),
            'avg_bars_held'  : round(avg_bars, 1),
            'total_pnl'      : round(pnl.sum(), 6),
            'avg_win'        : round(wins.mean(), 6)
                               if len(wins) > 0 else 0,
            'avg_loss'       : round(losses.mean(), 6)
                               if len(losses) > 0 else 0,
            'exit_mean_cross': int(reasons.get(
                                   'MEAN_CROSS', 0)),
            'exit_stop'      : int(reasons.get(
                                   'STOP_LOSS', 0)),
        }
        return stats

    def plot_results(self, trade_df: pd.DataFrame,
                     stats: dict):
        """Full performance visualisation."""
        fig = plt.figure(figsize=(16, 12))
        gs  = gridspec.GridSpec(3, 2, figure=fig)

        pair_label = f"{self.sym1} / {self.sym2}"
        fig.suptitle(
            f"Walk-Forward Backtest: {pair_label}\n"
            f"WR={stats['win_rate']:.1%}  "
            f"PF={stats['profit_factor']:.2f}  "
            f"Sharpe={stats['sharpe']:.2f}  "
            f"MaxDD={stats['max_drawdown']:.1%}",
            fontsize=13
        )

        # Panel 1: Equity curve
        ax = fig.add_subplot(gs[0, :])
        self.equity_curve.plot(ax=ax, color='steelblue',
                               linewidth=1.2)
        ax.set_title("Equity Curve (Walk-Forward OOS)")
        ax.set_ylabel("Equity (normalised)")
        ax.axhline(1.0, color='gray', linestyle='--',
                   linewidth=0.8)
        ax.grid(True, alpha=0.3)

        if not trade_df.empty:
            # Panel 2: PnL distribution
            ax2 = fig.add_subplot(gs[1, 0])
            trade_df['pnl_spread'].hist(
                ax=ax2, bins=30, color='steelblue',
                edgecolor='white', alpha=0.8)
            ax2.axvline(0, color='red', linewidth=1)
            ax2.set_title("PnL Distribution")
            ax2.set_xlabel("PnL (spread units)")
            ax2.grid(True, alpha=0.3)

            # Panel 3: Win rate by exit reason
            ax3 = fig.add_subplot(gs[1, 1])
            reasons = trade_df.groupby(
                'exit_reason')['pnl_spread'].agg(
                ['count', 'mean', 'sum'])
            reasons['count'].plot(kind='bar', ax=ax3,
                                  color=['green', 'red',
                                         'orange'])
            ax3.set_title("Exit Reason Counts")
            ax3.set_ylabel("Number of Trades")
            ax3.tick_params(axis='x', rotation=30)
            ax3.grid(True, alpha=0.3)

            # Panel 4: Bars held distribution
            ax4 = fig.add_subplot(gs[2, 0])
            trade_df['bars_held'].hist(
                ax=ax4, bins=25, color='darkorange',
                edgecolor='white', alpha=0.8)
            ax4.set_title("Holding Period (bars)")
            ax4.set_xlabel("Bars held")
            ax4.grid(True, alpha=0.3)

            # Panel 5: Cumulative wins vs losses
            ax5 = fig.add_subplot(gs[2, 1])
            pnl = trade_df['pnl_spread'].values
            cum_wins   = np.cumsum(pnl > 0)
            cum_losses = np.cumsum(pnl < 0)
            ax5.plot(cum_wins,   label='Wins',
                     color='green')
            ax5.plot(cum_losses, label='Losses',
                     color='red')
            ax5.set_title("Cumulative Wins vs Losses")
            ax5.legend()
            ax5.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = os.path.join(
            OUTPUT_DIR,
            f"backtest_{self.sym1}_{self.sym2}.png")
        plt.savefig(fname, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  [PLOT] {fname}")


# ─────────────────────────────────────────────
#  RUN BACKTEST ON ALL VALID PAIRS
# ─────────────────────────────────────────────
def run_all_backtests(timeframe: str = "H1"):
    """Load valid pairs and backtest each."""
    pairs_file = os.path.join(OUTPUT_DIR,
                               "valid_pairs.json")
    if not os.path.exists(pairs_file):
        print("[ERR] Run pairs_research.py first")
        return

    with open(pairs_file) as f:
        valid_pairs = json.load(f)

    print(f"\nFound {len(valid_pairs)} pairs to backtest")

    all_results = []

    for p in valid_pairs:
        sym1, sym2 = p['symbol1'], p['symbol2']

        s1 = load_price_series(sym1, timeframe)
        s2 = load_price_series(sym2, timeframe)

        if len(s1) < 2000 or len(s2) < 2000:
            print(f"  [SKIP] {sym1}/{sym2}: "
                  f"insufficient data")
            continue

        s1, s2 = align_series(s1, s2)

        bt = PairsBacktester(
            sym1=sym1, sym2=sym2,
            s1=s1, s2=s2,
            entry_z    = 2.0,
            exit_z     = 0.3,
            stop_z     = 3.5,
            delta      = 1e-4,
            warmup     = 200,
            train_bars = 2000,
            test_bars  = 336,    # 2 weeks H1
            step_bars  = 168,    # step 1 week
            max_dd     = 0.15,
        )

        trade_df = bt.run_walkforward()

        if trade_df.empty:
            print(f"  [SKIP] {sym1}/{sym2}: no trades")
            continue

        stats = bt.get_statistics(trade_df)
        bt.plot_results(trade_df, stats)

        print(f"\n  ══ {sym1}/{sym2} ══")
        print(f"  Trades:       {stats['n_trades']}")
        print(f"  Win Rate:     {stats['win_rate']:.1%}")
        print(f"  Profit Factor:{stats['profit_factor']:.2f}")
        print(f"  Sharpe:       {stats['sharpe']:.2f}")
        print(f"  Max DD:       {stats['max_drawdown']:.1%}")
        print(f"  Avg Hold:     {stats['avg_bars_held']:.0f} bars")

        result = {
            'symbol1': sym1,
            'symbol2': sym2,
            **stats,
            'half_life' : p['half_life'],
            'hurst'     : p['hurst'],
            'coint_pval': p['pvalue'],
        }
        all_results.append(result)

        # Save trade log
        trade_df.to_csv(
            os.path.join(OUTPUT_DIR,
                         f"trades_{sym1}_{sym2}.csv"),
            index=False)

    # Summary table
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values(
            'profit_factor', ascending=False)

        print("\n" + "="*70)
        print("  BACKTEST SUMMARY — RANKED BY PROFIT FACTOR")
        print("="*70)
        print(results_df[[
            'symbol1', 'symbol2',
            'n_trades', 'win_rate',
            'profit_factor', 'sharpe', 'max_drawdown'
        ]].to_string(index=False))

        results_df.to_csv(
            os.path.join(OUTPUT_DIR,
                         "backtest_summary.csv"),
            index=False)
        print(f"\n[SAVED] backtest_summary.csv")

    return all_results


if __name__ == "__main__":
    run_all_backtests(timeframe="H1")
