"""
pairs_backtest.py  v2.0
========================
Walk-forward backtest using Kalman Filter hedge ratio.

Uses valid_pairs.json produced by pairs_research.py v4.0

Walk-forward design:
  Train: 3 months (calibrate Kalman delta parameter)
  Test:  1 month  (trade out-of-sample)
  Step:  2 weeks  (re-calibrate every 2 weeks)

Key difference from static backtest:
  The hedge ratio updates EVERY BAR via Kalman
  This handles the drift visible in the top panels
  of the research charts
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

from kalman_filter  import KalmanHedgeFilter, kalman_filter_batch
from pairs_research  import (
    load_price_series, align_series,
    estimate_hedge_ratio, compute_half_life,
    _run_adf, VERSION
)

OUTPUT_DIR = "pairs_artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BT_VERSION = "backtest-v2.0"


# ─────────────────────────────────────────────────────────────
#  TRADE RECORD
# ─────────────────────────────────────────────────────────────
@dataclass
class Trade:
    entry_bar    : int
    entry_time   : pd.Timestamp
    direction    : int        # +1=long spread  -1=short spread
    entry_spread : float
    entry_z      : float
    entry_beta   : float
    exit_bar     : int              = 0
    exit_time    : pd.Timestamp     = None
    exit_spread  : float            = 0.0
    exit_z       : float            = 0.0
    exit_reason  : str              = ""
    pnl_spread   : float            = 0.0
    bars_held    : int              = 0


# ─────────────────────────────────────────────────────────────
#  ROLLING Z-SCORE HELPER
# ─────────────────────────────────────────────────────────────
def rolling_zscore(spread_buf: np.ndarray,
                   window: int) -> float:
    """
    Compute z-score of the last element
    using a rolling window.
    """
    n = len(spread_buf)
    if n < 5:
        return 0.0
    start = max(0, n - window)
    sub   = spread_buf[start:]
    mu    = sub.mean()
    sig   = sub.std()
    if sig < 1e-12:
        return 0.0
    return float((spread_buf[-1] - mu) / sig)


# ─────────────────────────────────────────────────────────────
#  BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────
class PairsBacktester:

    def __init__(self,
                 sym1: str,        sym2: str,
                 s1:   pd.Series,  s2:   pd.Series,
                 # Signal
                 entry_z:    float = 2.0,
                 exit_z:     float = 0.3,
                 stop_z:     float = 3.5,
                 # Kalman
                 delta:      float = 1e-4,
                 # Walk-forward windows (in bars)
                 train_bars: int   = 2016,  # ~3 months H1
                 test_bars:  int   = 336,   # ~2 weeks H1
                 step_bars:  int   = 168,   # step 1 week
                 # Risk
                 max_dd:     float = 0.20,
                 # Position sizing
                 risk_per_trade: float = 0.01):

        self.sym1           = sym1
        self.sym2           = sym2
        self.s1             = s1
        self.s2             = s2
        self.entry_z        = entry_z
        self.exit_z         = exit_z
        self.stop_z         = stop_z
        self.delta          = delta
        self.train_bars     = train_bars
        self.test_bars      = test_bars
        self.step_bars      = step_bars
        self.max_dd         = max_dd
        self.risk_per_trade = risk_per_trade

        self.trades: List[Trade] = []
        self.equity_curve: pd.Series = pd.Series(dtype=float)

    # ── CORE WALK-FORWARD LOOP ────────────────────────────
    def run(self) -> pd.DataFrame:
        s1v    = self.s1.values
        s2v    = self.s2.values
        times  = self.s1.index
        n      = len(s1v)

        print(f"\n  {'─'*50}")
        print(f"  {self.sym1} / {self.sym2}")
        print(f"  Bars: {n}  "
              f"Train: {self.train_bars}  "
              f"Test: {self.test_bars}  "
              f"Step: {self.step_bars}")

        equity      = 1.0
        peak_eq     = 1.0
        halted      = False
        eq_times    = [times[0]]
        eq_vals     = [1.0]

        test_start = self.train_bars
        window_n   = 0

        while test_start + self.test_bars <= n:
            window_n += 1
            test_end  = test_start + self.test_bars

            # ── TRAIN PHASE ──────────────────────────────
            tr_s  = max(0, test_start - self.train_bars)
            tr_e  = test_start
            tr_s1 = self.s1.iloc[tr_s:tr_e]
            tr_s2 = self.s2.iloc[tr_s:tr_e]

            # Compute z-score window from training half-life
            kf_tr = kalman_filter_batch(
                tr_s1, tr_s2, delta=self.delta)
            sp_tr = pd.Series(kf_tr['spread'].values)
            hl_tr = compute_half_life(sp_tr)
            if not np.isfinite(hl_tr) or hl_tr <= 0:
                hl_tr = 30
            z_window = int(np.clip(hl_tr * 2, 20, 200))

            print(f"  Window {window_n}: "
                  f"{times[test_start].date()} → "
                  f"{times[min(test_end-1,n-1)].date()}  "
                  f"HL={hl_tr:.1f}  Zwin={z_window}")

            # ── TEST PHASE ───────────────────────────────
            # Initialise Kalman on training data
            kf = KalmanHedgeFilter(delta=self.delta)
            for pa, pb in zip(tr_s1.values, tr_s2.values):
                kf.update(pa, pb)

            # Spread buffer pre-seeded with training spreads
            spread_buf = np.array(
                kf.spread_history[-z_window * 2:],
                dtype=float)

            position      = 0   # 0=flat 1=long -1=short
            current_trade: Optional[Trade] = None

            for i in range(test_start, test_end):
                if i >= n:
                    break

                pa = s1v[i]
                pb = s2v[i]
                t  = times[i]

                beta, alpha, spread = kf.update(pa, pb)
                spread_buf = np.append(spread_buf, spread)

                z = rolling_zscore(spread_buf, z_window)

                # Drawdown circuit breaker
                if (equity < peak_eq * (1 - self.max_dd)
                        and not halted):
                    halted = True
                    print(f"    [HALT] DD at bar {i}")

                if halted:
                    if position != 0 and current_trade:
                        pnl = (position *
                               (spread -
                                current_trade.entry_spread))
                        self._close_trade(
                            current_trade, i, t,
                            spread, z, "DD_HALT", pnl)
                        equity = self._update_equity(
                            equity,
                            current_trade.pnl_spread,
                            current_trade.entry_spread)
                        peak_eq = max(peak_eq, equity)
                        position = 0
                        current_trade = None
                    eq_times.append(t)
                    eq_vals.append(equity)
                    continue

                peak_eq = max(peak_eq, equity)

                # ── POSITION LOGIC ───────────────────────
                if position == 0:
                    if z < -self.entry_z:
                        position = 1
                        current_trade = Trade(
                            entry_bar   = i,
                            entry_time  = t,
                            direction   = 1,
                            entry_spread= spread,
                            entry_z     = z,
                            entry_beta  = beta)

                    elif z > self.entry_z:
                        position = -1
                        current_trade = Trade(
                            entry_bar   = i,
                            entry_time  = t,
                            direction   = -1,
                            entry_spread= spread,
                            entry_z     = z,
                            entry_beta  = beta)

                elif position == 1:
                    pnl = spread - current_trade.entry_spread
                    reason = None
                    if z >= self.exit_z:
                        reason = "MEAN_CROSS"
                    elif z < -self.stop_z:
                        reason = "STOP_LOSS"
                    if reason:
                        self._close_trade(
                            current_trade, i, t,
                            spread, z, reason, pnl)
                        equity = self._update_equity(
                            equity, pnl,
                            current_trade.entry_spread)
                        peak_eq = max(peak_eq, equity)
                        position = 0
                        current_trade = None

                elif position == -1:
                    pnl = (current_trade.entry_spread
                           - spread)
                    reason = None
                    if z <= self.exit_z:
                        reason = "MEAN_CROSS"
                    elif z > self.stop_z:
                        reason = "STOP_LOSS"
                    if reason:
                        self._close_trade(
                            current_trade, i, t,
                            spread, z, reason, pnl)
                        equity = self._update_equity(
                            equity, pnl,
                            current_trade.entry_spread)
                        peak_eq = max(peak_eq, equity)
                        position = 0
                        current_trade = None

                eq_times.append(t)
                eq_vals.append(equity)

            test_start += self.step_bars

        # Close any open trade at end
        if position != 0 and current_trade and len(s1v) > 0:
            i  = n - 1
            pa = s1v[i]; pb = s2v[i]; t = times[i]
            beta, alpha, spread = kf.update(pa, pb)
            pnl = (position *
                   (spread - current_trade.entry_spread))
            self._close_trade(
                current_trade, i, t,
                spread, 0.0, "END_OF_TEST", pnl)
            self.trades.append(current_trade)

        self.equity_curve = pd.Series(
            eq_vals,
            index=pd.DatetimeIndex(eq_times))

        trade_df = self._to_df()
        self._print_summary(trade_df)
        return trade_df

    def _close_trade(self, trade: Trade,
                     bar, t, spread, z,
                     reason, pnl):
        trade.exit_bar    = bar
        trade.exit_time   = t
        trade.exit_spread = spread
        trade.exit_z      = z
        trade.exit_reason = reason
        trade.pnl_spread  = pnl
        trade.bars_held   = bar - trade.entry_bar
        self.trades.append(trade)

    def _update_equity(self, equity: float,
                       pnl: float,
                       entry_spread: float) -> float:
        """
        Convert spread PnL to equity change.
        Normalise by entry spread magnitude so that
        a 1-sigma move corresponds to ~risk_per_trade.
        """
        base = abs(entry_spread) + 1e-10
        frac = pnl / base * self.risk_per_trade * 10
        frac = float(np.clip(frac, -0.20, 0.20))
        return equity * (1.0 + frac)

    def _to_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([{
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
        } for t in self.trades])

    def _print_summary(self, df: pd.DataFrame):
        if df.empty:
            print(f"    No trades generated")
            return
        stats = self.get_stats(df)
        print(f"  ══ {self.sym1}/{self.sym2} RESULTS ══")
        print(f"  Trades:        {stats['n_trades']}")
        print(f"  Win Rate:      {stats['win_rate']:.1%}")
        print(f"  Profit Factor: {stats['profit_factor']:.2f}")
        print(f"  Sharpe:        {stats['sharpe']:.2f}")
        print(f"  Max Drawdown:  {stats['max_dd']:.1%}")
        print(f"  Avg Hold:      {stats['avg_bars']:.0f} bars")
        exits = df['exit_reason'].value_counts().to_dict()
        print(f"  Exits:         {exits}")

    def get_stats(self, df: pd.DataFrame) -> dict:
        if df.empty:
            return {}
        pnl  = df['pnl_spread'].values
        wins = pnl[pnl > 0]
        loss = pnl[pnl < 0]
        n    = len(pnl)
        wr   = len(wins) / n
        gp   = wins.sum() if len(wins) > 0 else 0
        gl   = abs(loss.sum()) if len(loss) > 0 else 1e-10
        pf   = gp / gl

        eq      = self.equity_curve
        ret     = eq.pct_change().dropna()
        sharpe  = float(ret.mean() /
                        (ret.std() + 1e-10) *
                        np.sqrt(252 * 24))
        roll_mx = eq.cummax()
        max_dd  = float(((eq - roll_mx) / roll_mx).min())

        return {
            'n_trades'     : n,
            'win_rate'     : round(wr, 4),
            'profit_factor': round(pf, 4),
            'sharpe'       : round(sharpe, 4),
            'max_dd'       : round(max_dd, 4),
            'avg_bars'     : round(df['bars_held'].mean(),1),
            'total_pnl'    : round(pnl.sum(), 8),
            'avg_win'      : round(wins.mean(),8)
                             if len(wins)>0 else 0,
            'avg_loss'     : round(loss.mean(),8)
                             if len(loss)>0 else 0,
        }

    # ── PLOT ─────────────────────────────────────────────
    def plot(self, df: pd.DataFrame, stats: dict):
        if df.empty:
            return

        fig = plt.figure(figsize=(16, 14))
        gs  = gridspec.GridSpec(3, 2, figure=fig,
                                hspace=0.4, wspace=0.3)

        title = (f"{self.sym1}/{self.sym2}  "
                 f"WR={stats['win_rate']:.1%}  "
                 f"PF={stats['profit_factor']:.2f}  "
                 f"Sharpe={stats['sharpe']:.2f}  "
                 f"MaxDD={stats['max_dd']:.1%}")
        fig.suptitle(f"Walk-Forward: {title}", fontsize=12)

        # 1. Equity curve
        ax1 = fig.add_subplot(gs[0, :])
        self.equity_curve.plot(ax=ax1, color='steelblue',
                               lw=1.2)
        ax1.axhline(1.0, color='gray', ls='--', lw=0.8)
        ax1.set_title("Equity Curve (OOS only)")
        ax1.set_ylabel("Equity (normalised)")
        ax1.grid(True, alpha=0.3)

        # Mark trades on equity curve
        for _, row in df.iterrows():
            c = 'green' if row['pnl_spread'] > 0 else 'red'
            if row['exit_time'] is not None:
                ax1.axvline(row['exit_time'],
                            color=c, alpha=0.2,
                            lw=0.5)

        # 2. PnL distribution
        ax2 = fig.add_subplot(gs[1, 0])
        df['pnl_spread'].hist(ax=ax2, bins=25,
                              color='steelblue',
                              edgecolor='white',
                              alpha=0.8)
        ax2.axvline(0, color='red', lw=1.5)
        ax2.set_title("PnL Distribution")
        ax2.set_xlabel("PnL (spread units)")
        ax2.grid(True, alpha=0.3)

        # 3. Exit reasons
        ax3 = fig.add_subplot(gs[1, 1])
        reasons = df['exit_reason'].value_counts()
        colors  = ['green' if 'MEAN' in r else 'red'
                   for r in reasons.index]
        reasons.plot(kind='bar', ax=ax3, color=colors,
                     edgecolor='white')
        ax3.set_title("Exit Reason Counts")
        ax3.tick_params(axis='x', rotation=30)
        ax3.grid(True, alpha=0.3)

        # 4. Holding period
        ax4 = fig.add_subplot(gs[2, 0])
        df['bars_held'].hist(ax=ax4, bins=20,
                             color='darkorange',
                             edgecolor='white',
                             alpha=0.8)
        ax4.set_title(f"Hold Period (bars)  "
                      f"avg={stats['avg_bars']:.0f}")
        ax4.set_xlabel("Bars held")
        ax4.grid(True, alpha=0.3)

        # 5. Cumulative PnL
        ax5 = fig.add_subplot(gs[2, 1])
        pnl_cum = df['pnl_spread'].cumsum()
        pnl_cum.reset_index(drop=True).plot(
            ax=ax5, color='purple', lw=1.2)
        ax5.axhline(0, color='gray', ls='--', lw=0.8)
        ax5.set_title("Cumulative Spread PnL")
        ax5.grid(True, alpha=0.3)

        plt.savefig(
            os.path.join(OUTPUT_DIR,
                         f"bt_{self.sym1}_{self.sym2}.png"),
            dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  [PLOT] bt_{self.sym1}_{self.sym2}.png")


# ─────────────────────────────────────────────────────────────
#  PORTFOLIO BACKTEST — combine all valid pairs
# ─────────────────────────────────────────────────────────────
def run_portfolio_backtest(results: list) -> dict:
    """
    Combine equity curves from all valid pairs
    into a single portfolio equity curve.
    Equal weight per pair.
    """
    if not results:
        return {}

    curves = [r['equity_curve'] for r in results
              if r.get('equity_curve') is not None
              and len(r['equity_curve']) > 10]

    if not curves:
        return {}

    # Align all curves to common index
    combined = pd.concat(curves, axis=1).fillna(
        method='ffill').fillna(1.0)
    combined.columns = [r['pair'] for r in results
                        if r.get('equity_curve')
                        is not None
                        and len(r['equity_curve']) > 10]

    # Equal-weight portfolio return
    portfolio = combined.mean(axis=1)

    ret     = portfolio.pct_change().dropna()
    sharpe  = float(ret.mean() /
                    (ret.std() + 1e-10) *
                    np.sqrt(252 * 24))
    roll_mx = portfolio.cummax()
    max_dd  = float(((portfolio - roll_mx) /
                     roll_mx).min())
    total_r = float(portfolio.iloc[-1] /
                    portfolio.iloc[0] - 1)

    # Plot portfolio equity
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(
        f"Portfolio of {len(combined.columns)} pairs  "
        f"Sharpe={sharpe:.2f}  "
        f"MaxDD={max_dd:.1%}  "
        f"Return={total_r:.1%}",
        fontsize=12)

    combined.plot(ax=axes[0], lw=0.8, alpha=0.7)
    axes[0].set_title("Individual Pair Equity Curves")
    axes[0].axhline(1.0, color='k', ls='--', lw=0.8)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    portfolio.plot(ax=axes[1], color='darkblue', lw=1.5)
    axes[1].set_title("Portfolio Equity (equal weight)")
    axes[1].axhline(1.0, color='k', ls='--', lw=0.8)
    axes[1].fill_between(
        portfolio.index,
        portfolio.values,
        1.0,
        where=(portfolio.values >= 1.0),
        alpha=0.3, color='green')
    axes[1].fill_between(
        portfolio.index,
        portfolio.values,
        1.0,
        where=(portfolio.values < 1.0),
        alpha=0.3, color='red')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "portfolio_equity.png"),
        dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n  [PLOT] portfolio_equity.png")

    return {
        'n_pairs'      : len(combined.columns),
        'sharpe'       : round(sharpe, 4),
        'max_dd'       : round(max_dd, 4),
        'total_return' : round(total_r, 4),
        'pairs'        : list(combined.columns),
    }


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def run_all_backtests(timeframe: str = "H1"):
    print(f"\n{'='*55}")
    print(f"  {BT_VERSION}  (research: {VERSION})")
    print(f"{'='*55}")

    pairs_file = os.path.join(OUTPUT_DIR,
                               "valid_pairs.json")
    if not os.path.exists(pairs_file):
        print("[ERR] Run pairs_research.py first")
        return

    with open(pairs_file) as f:
        valid_pairs = json.load(f)

    print(f"\nLoaded {len(valid_pairs)} valid pairs")

    all_bt_results = []
    summary_rows   = []

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

        bt = PairsBacktester(
            sym1        = sym1,
            sym2        = sym2,
            s1          = s1a,
            s2          = s2a,
            entry_z     = 2.0,
            exit_z      = 0.3,
            stop_z      = 3.5,
            delta       = 1e-4,
            train_bars  = 2016,   # 3 months H1
            test_bars   = 336,    # 2 weeks H1
            step_bars   = 168,    # step 1 week
            max_dd      = 0.20,
            risk_per_trade = 0.01,
        )

        try:
            trade_df = bt.run()
        except Exception as e:
            print(f"  [ERR] {sym1}/{sym2}: {e}")
            traceback.print_exc()
            continue

        if trade_df.empty:
            print(f"  [SKIP] {sym1}/{sym2}: no trades")
            continue

        stats = bt.get_stats(trade_df)
        bt.plot(trade_df, stats)

        # Save trade log
        trade_df.to_csv(
            os.path.join(OUTPUT_DIR,
                         f"trades_{sym1}_{sym2}.csv"),
            index=False)

        all_bt_results.append({
            'pair'        : f"{sym1}/{sym2}",
            'equity_curve': bt.equity_curve,
            **stats,
        })

        summary_rows.append({
            'pair'         : f"{sym1}/{sym2}",
            'n_trades'     : stats['n_trades'],
            'win_rate'     : stats['win_rate'],
            'profit_factor': stats['profit_factor'],
            'sharpe'       : stats['sharpe'],
            'max_dd'       : stats['max_dd'],
            'avg_bars'     : stats['avg_bars'],
            'half_life'    : p.get('half_life', 0),
            'eg_pval'      : p.get('eg_pval', 1),
            'adf_pval'     : p.get('adf_pval', 1),
        })

    # Summary table
    if summary_rows:
        df_sum = pd.DataFrame(summary_rows).sort_values(
            'profit_factor', ascending=False)

        print(f"\n{'='*70}")
        print(f"  BACKTEST SUMMARY — RANKED BY PROFIT FACTOR")
        print(f"{'='*70}")
        print(df_sum.to_string(index=False))

        df_sum.to_csv(
            os.path.join(OUTPUT_DIR,
                         "backtest_summary.csv"),
            index=False)
        print(f"\n  [SAVED] backtest_summary.csv")

    # Portfolio combination
    if len(all_bt_results) > 1:
        print(f"\n  Computing portfolio metrics...")
        port = run_portfolio_backtest(all_bt_results)
        print(f"\n  PORTFOLIO SUMMARY:")
        print(f"  Pairs:         {port['n_pairs']}")
        print(f"  Sharpe:        {port['sharpe']:.2f}")
        print(f"  Max Drawdown:  {port['max_dd']:.1%}")
        print(f"  Total Return:  {port['total_return']:.1%}")

        with open(os.path.join(OUTPUT_DIR,
                               "portfolio_stats.json"),
                  'w') as f:
            json.dump(port, f, indent=2)

    print(f"\n  [DONE] {BT_VERSION}")
    return all_bt_results


if __name__ == "__main__":
    run_all_backtests(timeframe="H1")
