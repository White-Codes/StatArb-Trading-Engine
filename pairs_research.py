"""
pairs_research.py
=================
Full pipeline for:
  1. Loading OHLCV data
  2. Testing all pairs for cointegration
  3. Computing half-life of mean reversion
  4. Computing Hurst exponent
  5. Ranking pairs by tradability score
  6. Saving results for the trading engine

Run this OFFLINE to identify valid pairs.
Re-run monthly to refresh pair validity.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from itertools import combinations
import warnings
import json
import os

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
DATA_DIR    = "."                # Reading CSVs directly from root directory
OUTPUT_DIR  = "pairs_artifacts"  # Results saved here
MIN_BARS    = 2000               # Minimum history required
MAX_HALF_LIFE = 30                # Bars — reject slow reverters
MIN_HALF_LIFE = 2                 # Bars — reject too fast (noise)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  CANDIDATE PAIRS
#  These share common economic drivers
# ─────────────────────────────────────────────
UNIVERSE = [
    # FX Majors vs USD
    "EURUSD", "GBPUSD", "AUDUSD",
    "NZDUSD", "USDCAD", "USDCHF",

    # FX Crosses (naturally related)
    "EURGBP", "EURAUD", "GBPAUD",
    "AUDNZD", "EURCAD", "GBPCAD",

    # Add commodities if available
    # "XAUUSD", "XAGUSD",
]

# Natural economic pairs to prioritise
PRIORITY_PAIRS = [
    ("EURUSD", "GBPUSD"),   # Both vs USD, European
    ("AUDUSD", "NZDUSD"),   # Both commodity / Oceania
    ("EURUSD", "USDCHF"),   # EUR/CHF relationship
    ("AUDUSD", "XAUUSD"),   # AUD is gold-correlated
    ("GBPUSD", "EURUSD"),   # Brexit-era divergence
    ("USDCAD", "XAUUSD"),   # CAD-oil, Gold-USD
]


# ─────────────────────────────────────────────
#  DATA LOADING
#  Expects CSV with columns: time, open, high,
#  low, close, volume  (MT5 export format)
# ─────────────────────────────────────────────
def load_price_series(symbol: str,
                      timeframe: str = "H1") -> pd.Series:
    """Load close prices for a symbol."""
    fname = os.path.join(DATA_DIR,
                         f"{symbol}_{timeframe}.csv")
    if not os.path.exists(fname):
        print(f"  [MISS] {fname}")
        return pd.Series(dtype=float)

    # Read CSV and handle flexible date column names
    df = pd.read_csv(fname)
    
    date_col = next((c for c in ['time', 'Time', 'date', 'Date', '<DATE>'] if c in df.columns), None)
    close_col = next((c for c in ['close', 'Close', '<CLOSE>'] if c in df.columns), None)

    if not date_col or not close_col:
        print(f"  [ERR] Invalid column structure in {fname}")
        return pd.Series(dtype=float)

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    series = df[close_col].dropna()
    print(f"  [LOAD] {symbol}: {len(series)} bars")
    return series


def align_series(s1: pd.Series,
                 s2: pd.Series) -> tuple:
    """Align two series to common timestamps."""
    combined = pd.concat([s1, s2], axis=1).dropna()
    return combined.iloc[:, 0], combined.iloc[:, 1]


# ─────────────────────────────────────────────
#  HALF-LIFE OF MEAN REVERSION
#  Derived from Ornstein-Uhlenbeck process
#  ΔZ_t = λ * Z_{t-1} + ε_t
#  half_life = -log(2) / λ
# ─────────────────────────────────────────────
def compute_half_life(spread: pd.Series) -> float:
    """
    Compute half-life of mean reversion in bars.
    Returns np.inf if spread is not mean-reverting.
    """
    spread = spread.dropna()
    lag    = spread.shift(1).dropna()
    delta  = spread.diff().dropna()

    # Align
    lag   = lag.iloc[1:]
    delta = delta.iloc[1:]

    if len(lag) < 50:
        return np.inf

    lag_const = add_constant(lag)
    model     = OLS(delta, lag_const).fit()
    lam       = model.params.iloc[1]

    if lam >= 0:
        return np.inf   # Not mean reverting

    half_life = -np.log(2) / lam
    return half_life


# ─────────────────────────────────────────────
#  HURST EXPONENT
#  H < 0.5 → mean reverting (we want this)
#  H = 0.5 → random walk
#  H > 0.5 → trending
# ─────────────────────────────────────────────
def compute_hurst(series: pd.Series,
                  max_lag: int = 100) -> float:
    """
    Compute Hurst exponent using R/S analysis.
    """
    series = series.dropna().values
    n      = len(series)

    if n < max_lag * 2:
        return 0.5

    lags    = range(10, max_lag)
    rs_vals = []

    for lag in lags:
        sub_series = [series[i:i+lag]
                      for i in range(0, n-lag, lag)]
        rs_list = []
        for sub in sub_series:
            if len(sub) < 4:
                continue
            mean     = np.mean(sub)
            demeaned = sub - mean
            cumdev   = np.cumsum(demeaned)
            R        = np.max(cumdev) - np.min(cumdev)
            S        = np.std(sub, ddof=1)
            if S > 0:
                rs_list.append(R / S)
        if rs_list:
            rs_vals.append(np.mean(rs_list))

    if len(rs_vals) < 5:
        return 0.5

    log_lags = np.log(list(lags)[:len(rs_vals)])
    log_rs   = np.log(rs_vals)

    hurst = np.polyfit(log_lags, log_rs, 1)[0]
    return hurst


# ─────────────────────────────────────────────
#  HEDGE RATIO ESTIMATION
#  OLS regression: A = alpha + beta * B + epsilon
#  beta is the hedge ratio
# ─────────────────────────────────────────────
def estimate_hedge_ratio(s1: pd.Series,
                         s2: pd.Series) -> tuple:
    """
    Estimate OLS hedge ratio.
    Returns (hedge_ratio, alpha, residuals)
    """
    s2_const = add_constant(s2)
    model    = OLS(s1, s2_const).fit()
    alpha    = model.params.iloc[0]
    beta     = model.params.iloc[1]
    spread   = s1 - beta * s2 - alpha
    return beta, alpha, spread


# ─────────────────────────────────────────────
#  COINTEGRATION TEST
#  Engle-Granger two-step test
# ─────────────────────────────────────────────
def test_cointegration(s1: pd.Series,
                       s2: pd.Series,
                       significance: float = 0.05
                       ) -> dict:
    """
    Full cointegration analysis of a pair.
    Returns dict with all metrics.
    """
    result = {
        'cointegrated'  : False,
        'pvalue'        : 1.0,
        'hedge_ratio'   : 0.0,
        'alpha'         : 0.0,
        'half_life'     : np.inf,
        'hurst'         : 0.5,
        'spread_mean'   : 0.0,
        'spread_std'    : 0.0,
        'adf_pvalue'    : 1.0,
        'score'         : 0.0,
    }

    try:
        # Engle-Granger test
        eg_stat, eg_pval, _ = coint(s1, s2)

        # Hedge ratio + spread
        beta, alpha, spread = estimate_hedge_ratio(s1, s2)

        # ADF test on spread directly
        adf_result = adfuller(spread, maxlags=1,
                              autolag=None)
        adf_pval   = adf_result[1]

        # Mean reversion metrics
        half_life = compute_half_life(spread)
        hurst     = compute_hurst(spread)

        result.update({
            'pvalue'      : eg_pval,
            'adf_pvalue'  : adf_pval,
            'hedge_ratio' : beta,
            'alpha'       : alpha,
            'half_life'   : half_life,
            'hurst'       : hurst,
            'spread_mean' : float(spread.mean()),
            'spread_std'  : float(spread.std()),
        })

        # Pair is valid if:
        #   1. Engle-Granger p < significance
        #   2. ADF on spread p < significance
        #   3. Half-life is reasonable
        #   4. Hurst < 0.5 (mean reverting)
        is_coint = (eg_pval   < significance and
                    adf_pval  < significance and
                    MIN_HALF_LIFE < half_life < MAX_HALF_LIFE and
                    hurst     < 0.5)

        result['cointegrated'] = is_coint

        if is_coint:
            # Tradability score (higher is better)
            # Rewards: low p-value, short half-life, low Hurst
            score = (
                (1 - eg_pval)   * 0.30 +
                (1 - adf_pval)  * 0.30 +
                (1 - hurst)     * 0.20 +
                (1 - min(half_life, MAX_HALF_LIFE) /
                 MAX_HALF_LIFE) * 0.20
            )
            result['score'] = score

    except Exception as e:
        print(f"    [ERR] Coint test failed: {e}")

    return result


# ─────────────────────────────────────────────
#  SPREAD Z-SCORE ANALYSIS
# ─────────────────────────────────────────────
def analyse_spread(spread: pd.Series,
                   half_life: float) -> dict:
    """
    Compute rolling z-score and entry statistics.
    Window size derived from half-life.
    """
    window = max(20, int(half_life * 2))

    roll_mean = spread.rolling(window).mean()
    roll_std  = spread.rolling(window).std()
    zscore    = (spread - roll_mean) / (roll_std + 1e-10)

    # Entry/exit statistics
    entries_long  = (zscore < -2.0).sum()
    entries_short = (zscore >  2.0).sum()
    crossings     = ((zscore.shift(1) * zscore) < 0).sum()

    return {
        'zscore'        : zscore,
        'roll_mean'     : roll_mean,
        'roll_std'      : roll_std,
        'entries_long'  : int(entries_long),
        'entries_short' : int(entries_short),
        'mean_crossings': int(crossings),
        'window'        : window,
    }


# ─────────────────────────────────────────────
#  PAIR BACKTEST (Simple)
#  Quick check of theoretical P&L before
#  committing to detailed backtest
# ─────────────────────────────────────────────
def quick_backtest(s1: pd.Series,
                   s2: pd.Series,
                   spread: pd.Series,
                   half_life: float,
                   entry_z: float  = 2.0,
                   exit_z:  float  = 0.0,
                   stop_z:  float  = 3.5) -> dict:
    """
    Vectorised quick backtest.
    Returns basic P&L statistics.
    """
    window   = max(20, int(half_life * 2))
    roll_mean = spread.rolling(window).mean()
    roll_std  = spread.rolling(window).std()
    zscore    = (spread - roll_mean) / (roll_std + 1e-10)

    position  = 0   # 1=long spread, -1=short spread
    pnl_list  = []
    entry_spread = 0.0

    for i in range(window + 1, len(zscore)):
        z = zscore.iloc[i]

        if position == 0:
            # Look for entry
            if z < -entry_z:
                position    = 1
                entry_spread = spread.iloc[i]
            elif z > entry_z:
                position    = -1
                entry_spread = spread.iloc[i]

        elif position == 1:
            # Long spread — exit when z crosses 0
            # or stop at -stop_z (spread going further negative)
            current_spread = spread.iloc[i]
            pnl_raw        = current_spread - entry_spread

            if z >= exit_z or z < -stop_z:
                pnl_list.append(pnl_raw)
                position = 0

        elif position == -1:
            # Short spread — exit when z crosses 0
            current_spread = spread.iloc[i]
            pnl_raw        = entry_spread - current_spread

            if z <= exit_z or z > stop_z:
                pnl_list.append(pnl_raw)
                position = 0

    if not pnl_list:
        return {'n_trades': 0, 'win_rate': 0,
                'profit_factor': 0, 'sharpe': 0}

    pnl_arr   = np.array(pnl_list)
    wins      = pnl_arr[pnl_arr > 0]
    losses    = pnl_arr[pnl_arr < 0]
    win_rate  = len(wins) / len(pnl_arr)
    gross_profit = wins.sum()  if len(wins)   > 0 else 0
    gross_loss   = abs(losses.sum()) if len(losses) > 0 else 1e-10
    pf        = gross_profit / gross_loss
    sharpe    = (pnl_arr.mean() /
                 (pnl_arr.std() + 1e-10) *
                 np.sqrt(252))

    return {
        'n_trades'     : len(pnl_arr),
        'win_rate'     : round(win_rate, 4),
        'profit_factor': round(pf, 4),
        'sharpe'       : round(sharpe, 4),
        'avg_pnl'      : round(pnl_arr.mean(), 6),
        'total_pnl'    : round(pnl_arr.sum(), 6),
    }


# ─────────────────────────────────────────────
#  VISUALISE PAIR
# ─────────────────────────────────────────────
def plot_pair(sym1: str, sym2: str,
              s1: pd.Series, s2: pd.Series,
              spread: pd.Series,
              zscore: pd.Series,
              result: dict):
    """Save diagnostic chart for a valid pair."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(
        f"{sym1} / {sym2}  |  "
        f"p={result['pvalue']:.4f}  "
        f"HL={result['half_life']:.1f}  "
        f"Hurst={result['hurst']:.3f}  "
        f"Score={result['score']:.3f}",
        fontsize=12
    )

    # Panel 1: Normalised price series
    ax = axes[0]
    (s1 / s1.iloc[0]).plot(ax=ax, label=sym1,
                            color='steelblue')
    (s2 / s2.iloc[0]).plot(ax=ax, label=sym2,
                            color='darkorange')
    ax.set_title("Normalised Price Series")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Spread
    ax = axes[1]
    spread.plot(ax=ax, color='purple', linewidth=0.8)
    ax.axhline(result['spread_mean'], color='black',
               linestyle='--', linewidth=1,
               label=f"Mean={result['spread_mean']:.5f}")
    ax.set_title("Spread (A - β·B)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Z-score with entry/exit levels
    ax = axes[2]
    zscore.plot(ax=ax, color='darkgreen', linewidth=0.8)
    ax.axhline( 2.0, color='red',   linestyle='--',
                linewidth=1, label='Entry ±2σ')
    ax.axhline(-2.0, color='red',   linestyle='--')
    ax.axhline( 3.5, color='darkred', linestyle=':',
                linewidth=1, label='Stop ±3.5σ')
    ax.axhline(-3.5, color='darkred', linestyle=':')
    ax.axhline( 0.0, color='black', linestyle='-',
                linewidth=0.5, label='Exit 0')
    ax.set_title("Z-Score of Spread")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR,
                         f"{sym1}_{sym2}_analysis.png")
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"    [PLOT] Saved: {fname}")


# ─────────────────────────────────────────────
#  MAIN RESEARCH PIPELINE
# ─────────────────────────────────────────────
def run_research(timeframe: str = "H1"):
    print("\n" + "="*60)
    print("  PAIRS TRADING RESEARCH PIPELINE")
    print("="*60)

    # Load all available series
    print("\n[1] Loading price data...")
    price_data = {}
    for sym in UNIVERSE:
        s = load_price_series(sym, timeframe)
        if len(s) >= MIN_BARS:
            price_data[sym] = s

    available = list(price_data.keys())
    print(f"\n  Available symbols: {available}")

    if len(available) < 2:
        print("[ERR] Need at least 2 symbols")
        return

    # Test all combinations
    print("\n[2] Testing cointegration for all pairs...")
    all_pairs   = list(combinations(available, 2))
    valid_pairs = []

    for sym1, sym2 in all_pairs:
        s1, s2 = align_series(price_data[sym1],
                               price_data[sym2])

        if len(s1) < MIN_BARS:
            continue

        print(f"\n  Testing: {sym1} / {sym2} "
              f"({len(s1)} bars)")

        result = test_cointegration(s1, s2)

        status = "✓ VALID" if result['cointegrated'] \
                           else "✗ invalid"
        print(f"    {status}  "
              f"p={result['pvalue']:.4f}  "
              f"HL={result['half_life']:.1f}  "
              f"Hurst={result['hurst']:.3f}")

        if result['cointegrated']:
            # Quick backtest
            _, _, spread = estimate_hedge_ratio(s1, s2)
            bt = quick_backtest(s1, s2, spread,
                                result['half_life'])
            result.update({
                'symbol1'  : sym1,
                'symbol2'  : sym2,
                'backtest' : bt,
                'n_bars'   : len(s1),
            })

            print(f"    Backtest: n={bt['n_trades']} "
                  f"WR={bt['win_rate']:.1%} "
                  f"PF={bt['profit_factor']:.2f} "
                  f"Sharpe={bt['sharpe']:.2f}")

            # Plot
            spread_analysis = analyse_spread(
                spread, result['half_life'])
            plot_pair(sym1, sym2, s1, s2,
                      spread,
                      spread_analysis['zscore'],
                      result)

            valid_pairs.append(result)

    # Rank by score
    valid_pairs.sort(key=lambda x: x['score'],
                     reverse=True)

    print("\n" + "="*60)
    print(f"  FOUND {len(valid_pairs)} VALID PAIRS")
    print("="*60)

    for i, p in enumerate(valid_pairs):
        print(f"\n  #{i+1}: {p['symbol1']} / {p['symbol2']}")
        print(f"    Score:   {p['score']:.4f}")
        print(f"    p-value: {p['pvalue']:.4f}")
        print(f"    Half-life: {p['half_life']:.1f} bars")
        print(f"    Hurst:   {p['hurst']:.3f}")
        bt = p['backtest']
        print(f"    Trades:  {bt['n_trades']}  "
              f"WR={bt['win_rate']:.1%}  "
              f"PF={bt['profit_factor']:.2f}")

    # Save results
    save_data = []
    for p in valid_pairs:
        row = {k: v for k, v in p.items()
               if k != 'backtest'}
        row['hedge_ratio'] = float(row['hedge_ratio'])
        row['alpha']       = float(row['alpha'])
        row['half_life']   = float(row['half_life'])
        row['hurst']       = float(row['hurst'])
        row['spread_mean'] = float(row['spread_mean'])
        row['spread_std']  = float(row['spread_std'])
        row['pvalue']      = float(row['pvalue'])
        row['adf_pvalue']  = float(row['adf_pvalue'])
        row['score']       = float(row['score'])
        row.update(p['backtest'])
        save_data.append(row)

    out_path = os.path.join(OUTPUT_DIR, "valid_pairs.json")
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n[SAVED] {out_path}")

    return valid_pairs


if __name__ == "__main__":
    pairs = run_research(timeframe="H1")
