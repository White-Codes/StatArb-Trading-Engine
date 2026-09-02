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

Run this OFFLINE or via GitHub Actions to identify valid pairs.
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
DATA_DIR      = "."                # Reading CSVs directly from root directory
OUTPUT_DIR    = "pairs_artifacts"  # Results saved here
MIN_BARS      = 2000               # Minimum history required
LOOKBACK_BARS = 10000              # Focus on recent history for cointegration
MAX_HALF_LIFE = 60                 # Relaxed half-life ceiling (in bars)
MIN_HALF_LIFE = 2                  # Bars — reject noise
SIGNIFICANCE  = 0.10               # Slightly broader p-value threshold

os.makedirs(OUTPUT_DIR, exist_ok=True)

UNIVERSE = [
    "EURUSD", "GBPUSD", "AUDUSD",
    "NZDUSD", "USDCAD", "USDCHF",
    "EURGBP", "EURAUD", "GBPAUD",
    "AUDNZD", "EURCAD", "GBPCAD",
]


# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────
def load_price_series(symbol: str, timeframe: str = "H1") -> pd.Series:
    """Load close prices supporting Dukascopy and standard CSV formats."""
    fname = os.path.join(DATA_DIR, f"{symbol}_{timeframe}.csv")
    if not os.path.exists(fname):
        print(f"  [MISS] {fname}")
        return pd.Series(dtype=float)

    try:
        # Read CSV file
        df = pd.read_csv(fname)
        
        # Clean column names (strip spaces, lowercase)
        df.columns = [c.strip().lower() for c in df.columns]

        # Handle Dukascopy 'gmt time', 'timestamp', or standard 'time'/'date'
        date_col = next((c for c in ['gmt time', 'gmt_time', 'timestamp', 'time', 'date', 'datetime'] if c in df.columns), None)
        close_col = next((c for c in ['close', 'c', 'price'] if c in df.columns), None)

        if not date_col or not close_col:
            print(f"  [ERR] Unrecognized columns in {fname}: {list(df.columns)}")
            return pd.Series(dtype=float)

        # Parse Dukascopy timestamps
        df[date_col] = pd.to_datetime(df[date_col], format='mixed', errors='coerce')
        df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
        
        series = df[close_col].astype(float).dropna()
        print(f"  [LOAD] {symbol}: {len(series)} bars")
        return series

    except Exception as e:
        print(f"  [ERR] Failed to read {fname}: {e}")
        return pd.Series(dtype=float)


def align_series(s1: pd.Series, s2: pd.Series) -> tuple:
    """Align two series to common timestamps."""
    combined = pd.concat([s1, s2], axis=1).dropna()
    return combined.iloc[:, 0], combined.iloc[:, 1]


# ─────────────────────────────────────────────
#  STATISTICAL METRICS
# ─────────────────────────────────────────────
def compute_half_life(spread: pd.Series) -> float:
    spread = spread.dropna()
    lag    = spread.shift(1).dropna()
    delta  = spread.diff().dropna()

    lag   = lag.iloc[1:]
    delta = delta.iloc[1:]

    if len(lag) < 50:
        return np.inf

    lag_const = add_constant(lag)
    model     = OLS(delta, lag_const).fit()
    lam       = model.params.iloc[1]

    if lam >= 0:
        return np.inf

    return -np.log(2) / lam


def compute_hurst(series: pd.Series, max_lag: int = 100) -> float:
    series = series.dropna().values
    n      = len(series)

    if n < max_lag * 2:
        return 0.5

    lags    = range(10, max_lag)
    rs_vals = []

    for lag in lags:
        sub_series = [series[i:i+lag] for i in range(0, n-lag, lag)]
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
    return np.polyfit(log_lags, log_rs, 1)[0]


def estimate_hedge_ratio(s1: pd.Series, s2: pd.Series) -> tuple:
    s2_const = add_constant(s2)
    model    = OLS(s1, s2_const).fit()
    alpha    = model.params.iloc[0]
    beta     = model.params.iloc[1]
    spread   = s1 - beta * s2 - alpha
    return beta, alpha, spread


def test_cointegration(s1: pd.Series, s2: pd.Series, significance: float = 0.05) -> dict:
    result = {
        'cointegrated': False, 'pvalue': 1.0, 'hedge_ratio': 0.0,
        'alpha': 0.0, 'half_life': np.inf, 'hurst': 0.5,
        'spread_mean': 0.0, 'spread_std': 0.0, 'adf_pvalue': 1.0, 'score': 0.0
    }

    try:
        eg_stat, eg_pval, _ = coint(s1, s2)
        beta, alpha, spread = estimate_hedge_ratio(s1, s2)
        adf_pval = adfuller(spread, maxlag=1, autolag=None)[1]
        half_life = compute_half_life(spread)
        hurst     = compute_hurst(spread)

        result.update({
            'pvalue': eg_pval, 'adf_pvalue': adf_pval,
            'hedge_ratio': beta, 'alpha': alpha,
            'half_life': half_life, 'hurst': hurst,
            'spread_mean': float(spread.mean()),
            'spread_std': float(spread.std()),
        })

        is_coint = (eg_pval < significance and adf_pval < significance and
                    MIN_HALF_LIFE < half_life < MAX_HALF_LIFE and hurst < 0.5)

        result['cointegrated'] = is_coint

        if is_coint:
            score = ((1 - eg_pval) * 0.30 + (1 - adf_pval) * 0.30 +
                     (1 - hurst) * 0.20 + (1 - min(half_life, MAX_HALF_LIFE) / MAX_HALF_LIFE) * 0.20)
            result['score'] = score

    except Exception as e:
        print(f"    [ERR] Coint test failed: {e}")

    return result


def quick_backtest(s1: pd.Series, s2: pd.Series, spread: pd.Series, half_life: float) -> dict:
    window   = max(20, int(half_life * 2))
    roll_mean = spread.rolling(window).mean()
    roll_std  = spread.rolling(window).std()
    zscore    = (spread - roll_mean) / (roll_std + 1e-10)

    position, pnl_list, entry_spread = 0, [], 0.0

    for i in range(window + 1, len(zscore)):
        z = zscore.iloc[i]
        if position == 0:
            if z < -2.0: position, entry_spread = 1, spread.iloc[i]
            elif z > 2.0: position, entry_spread = -1, spread.iloc[i]
        elif position == 1:
            pnl_raw = spread.iloc[i] - entry_spread
            if z >= 0.0 or z < -3.5:
                pnl_list.append(pnl_raw)
                position = 0
        elif position == -1:
            pnl_raw = entry_spread - spread.iloc[i]
            if z <= 0.0 or z > 3.5:
                pnl_list.append(pnl_raw)
                position = 0

    if not pnl_list:
        return {'n_trades': 0, 'win_rate': 0, 'profit_factor': 0, 'sharpe': 0}

    pnl_arr   = np.array(pnl_list)
    wins      = pnl_arr[pnl_arr > 0]
    losses    = pnl_arr[pnl_arr < 0]
    win_rate  = len(wins) / len(pnl_arr)
    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss   = abs(losses.sum()) if len(losses) > 0 else 1e-10
    pf        = gross_profit / gross_loss
    sharpe    = (pnl_arr.mean() / (pnl_arr.std() + 1e-10) * np.sqrt(252))

    return {
        'n_trades': len(pnl_arr), 'win_rate': round(win_rate, 4),
        'profit_factor': round(pf, 4), 'sharpe': round(sharpe, 4)
    }


def plot_pair(sym1: str, sym2: str, s1: pd.Series, s2: pd.Series, spread: pd.Series, result: dict):
    window    = max(20, int(result['half_life'] * 2))
    roll_mean = spread.rolling(window).mean()
    roll_std  = spread.rolling(window).std()
    zscore    = (spread - roll_mean) / (roll_std + 1e-10)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f"{sym1} / {sym2} | p={result['pvalue']:.4f} | HL={result['half_life']:.1f} | Hurst={result['hurst']:.3f}")

    (s1 / s1.iloc[0]).plot(ax=axes[0], label=sym1, color='steelblue')
    (s2 / s2.iloc[0]).plot(ax=axes[0], label=sym2, color='darkorange')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    spread.plot(ax=axes[1], color='purple', linewidth=0.8)
    axes[1].axhline(result['spread_mean'], color='black', linestyle='--')
    axes[1].grid(True, alpha=0.3)

    zscore.plot(ax=axes[2], color='darkgreen', linewidth=0.8)
    axes[2].axhline(2.0, color='red', linestyle='--')
    axes[2].axhline(-2.0, color='red', linestyle='--')
    axes[2].axhline(0.0, color='black', linestyle='-')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{sym1}_{sym2}_analysis.png"), dpi=120)
    plt.close()


# ─────────────────────────────────────────────
#  MAIN EXECUTION
# ─────────────────────────────────────────────
def run_research(timeframe: str = "H1"):
    print("="*60 + "\n  PAIRS TRADING RESEARCH PIPELINE\n" + "="*60)

    price_data = {}
    for sym in UNIVERSE:
        s = load_price_series(sym, timeframe)
        if len(s) >= MIN_BARS:
            price_data[sym] = s

    available = list(price_data.keys())
    if len(available) < 2:
        print("[ERR] Need at least 2 valid price series to form pairs.")
        return

    all_pairs, valid_pairs = list(combinations(available, 2)), []

    for sym1, sym2 in all_pairs:
        s1, s2 = align_series(price_data[sym1], price_data[sym2])
        if len(s1) < MIN_BARS: continue

        result = test_cointegration(s1, s2)
        if result['cointegrated']:
            _, _, spread = estimate_hedge_ratio(s1, s2)
            bt = quick_backtest(s1, s2, spread, result['half_life'])
            result.update({'symbol1': sym1, 'symbol2': sym2, 'backtest': bt})
            plot_pair(sym1, sym2, s1, s2, spread, result)
            valid_pairs.append(result)

    valid_pairs.sort(key=lambda x: x['score'], reverse=True)

    save_data = []
    for p in valid_pairs:
        row = {k: v for k, v in p.items() if k != 'backtest'}
        row.update(p['backtest'])
        save_data.append(row)

    out_path = os.path.join(OUTPUT_DIR, "valid_pairs.json")
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n[SUCCESS] Found {len(valid_pairs)} valid pairs. Saved to {out_path}")


if __name__ == "__main__":
    run_research(timeframe="H1")
