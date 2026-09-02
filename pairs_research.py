"""
pairs_research.py  v2.0
========================
Key changes vs v1.0:
  - Rolling window cointegration (recent 3-6 months)
    instead of full history testing
  - Diagnostic mode: prints ACTUAL p-values and metrics
    so you can see exactly why pairs fail
  - Relaxed but sensible filters with clear logging
  - Johansen test added as second cointegration method
  - Sub-period stability check
  - Works correctly with 60,000+ bar Dukascopy data
  - Testing for shorter times over long periods are more advisable
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools  import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from itertools import combinations
import warnings
import json
import os

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────
DATA_DIR    = "."
OUTPUT_DIR  = "pairs_artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Rolling window settings ───────────────────────────────────
# Test cointegration on recent N bars only
# 2016 H1 bars = ~3 months of trading hours
# 4032 H1 bars = ~6 months
COINT_WINDOW   = 2016    # bars used for cointegration test
MIN_BARS       = 5000    # minimum total history required

# ── Filter thresholds ─────────────────────────────────────────
# These are applied to the ROLLING WINDOW test
EG_PVAL_MAX    = 0.15    # Engle-Granger p-value ceiling
ADF_PVAL_MAX   = 0.15    # ADF on spread p-value ceiling
MAX_HALF_LIFE  = 168     # bars — 1 week on H1
MIN_HALF_LIFE  = 2       # bars — avoid noise
HURST_MAX      = 0.58    # Hurst exponent ceiling

# ── Sub-period stability ──────────────────────────────────────
# Pair must pass in at least N of last M windows
STABILITY_WINDOWS = 4    # number of windows to check
STABILITY_MIN_PASS = 2   # minimum windows that must pass

UNIVERSE = [
    "EURUSD", "GBPUSD", "AUDUSD",
    "NZDUSD", "USDCAD", "USDCHF",
    "EURGBP", "EURAUD", "GBPAUD",
    "AUDNZD", "EURCAD", "GBPCAD",
]


# ─────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────
def load_price_series(symbol: str,
                      timeframe: str = "H1") -> pd.Series:
    fname = os.path.join(DATA_DIR,
                         f"{symbol}_{timeframe}.csv")
    if not os.path.exists(fname):
        print(f"  [MISS] {fname}")
        return pd.Series(dtype=float)

    try:
        df = pd.read_csv(fname)
        df.columns = [c.strip().lower() for c in df.columns]

        date_col  = next((c for c in [
            'gmt time', 'gmt_time', 'timestamp',
            'time', 'date', 'datetime']
            if c in df.columns), None)
        close_col = next((c for c in [
            'close', 'c', 'price']
            if c in df.columns), None)

        if not date_col or not close_col:
            print(f"  [ERR] Bad columns in {fname}: "
                  f"{list(df.columns)}")
            return pd.Series(dtype=float)

        df[date_col] = pd.to_datetime(
            df[date_col], format='mixed', errors='coerce')
        df = (df.dropna(subset=[date_col])
                .sort_values(date_col)
                .set_index(date_col))

        series = df[close_col].astype(float).dropna()
        print(f"  [LOAD] {symbol}: {len(series):,} bars  "
              f"({series.index[0].date()} → "
              f"{series.index[-1].date()})")
        return series

    except Exception as e:
        print(f"  [ERR] {fname}: {e}")
        return pd.Series(dtype=float)


def align_series(s1: pd.Series,
                 s2: pd.Series) -> tuple:
    combined = pd.concat([s1, s2], axis=1).dropna()
    return combined.iloc[:, 0], combined.iloc[:, 1]


# ─────────────────────────────────────────────────────────────
#  STATISTICAL METRICS
# ─────────────────────────────────────────────────────────────
def compute_half_life(spread: pd.Series) -> float:
    spread = spread.dropna()
    if len(spread) < 30:
        return np.inf

    lag   = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    # align
    common = lag.index.intersection(delta.index)
    lag, delta = lag[common], delta[common]

    if len(lag) < 20:
        return np.inf

    try:
        model = OLS(delta, add_constant(lag)).fit()
        lam   = model.params.iloc[1]
        if lam >= 0:
            return np.inf
        return -np.log(2) / lam
    except Exception:
        return np.inf


def compute_hurst(series: pd.Series,
                  max_lag: int = 50) -> float:
    """Hurst exponent via R/S analysis."""
    vals = series.dropna().values
    n    = len(vals)
    if n < max_lag * 2:
        return 0.5

    lags    = range(8, max_lag)
    rs_vals = []

    for lag in lags:
        chunks  = [vals[i:i+lag]
                   for i in range(0, n-lag, lag)]
        rs_list = []
        for chunk in chunks:
            if len(chunk) < 4:
                continue
            dm  = chunk - chunk.mean()
            R   = np.max(np.cumsum(dm)) - \
                  np.min(np.cumsum(dm))
            S   = np.std(chunk, ddof=1)
            if S > 0:
                rs_list.append(R / S)
        if rs_list:
            rs_vals.append(np.mean(rs_list))

    if len(rs_vals) < 3:
        return 0.5

    log_lags = np.log(list(lags)[:len(rs_vals)])
    log_rs   = np.log(rs_vals)
    return float(np.polyfit(log_lags, log_rs, 1)[0])


def estimate_hedge_ratio(s1: pd.Series,
                         s2: pd.Series,
                         use_log: bool = True) -> tuple:
    """OLS hedge ratio on log prices."""
    if use_log:
        y = np.log(s1)
        x = np.log(s2)
    else:
        y, x = s1, s2

    model  = OLS(y, add_constant(x)).fit()
    alpha  = model.params.iloc[0]
    beta   = model.params.iloc[1]
    spread = y - beta * x - alpha
    return beta, alpha, spread


def johansen_test(s1: pd.Series,
                  s2: pd.Series) -> float:
    """
    Johansen cointegration test.
    Returns p-value proxy (trace statistic vs 95% CV).
    Returns 1.0 if test fails or no cointegration.
    """
    try:
        data   = pd.concat([np.log(s1), np.log(s2)],
                           axis=1).dropna().values
        result = coint_johansen(data, det_order=0,
                                k_ar_diff=1)
        # Trace statistic vs 95% critical value
        trace_stat = result.lr1[0]
        crit_95    = result.cvt[0, 1]
        # Convert to pseudo p-value
        # >1 means significant at 95%
        ratio = trace_stat / (crit_95 + 1e-10)
        # Map ratio to [0,1]: ratio>1 → p<0.05
        return max(0.0, min(1.0, 1.0 / ratio))
    except Exception:
        return 1.0


# ─────────────────────────────────────────────────────────────
#  CORE COINTEGRATION TEST — SINGLE WINDOW
# ─────────────────────────────────────────────────────────────
def test_window(s1: pd.Series,
                s2: pd.Series,
                verbose: bool = False) -> dict:
    """
    Test one pair on one time window.
    Returns dict with all metrics.
    """
    result = {
        'cointegrated' : False,
        'eg_pval'      : 1.0,
        'adf_pval'     : 1.0,
        'johansen_pval': 1.0,
        'half_life'    : np.inf,
        'hurst'        : 0.5,
        'hedge_ratio'  : 0.0,
        'alpha'        : 0.0,
        'spread_std'   : 0.0,
        'score'        : 0.0,
    }

    if len(s1) < 100 or len(s2) < 100:
        return result

    try:
        # Use log prices
        log_s1 = np.log(s1)
        log_s2 = np.log(s2)

        # 1. Engle-Granger
        _, eg_pval, _ = coint(log_s1, log_s2)

        # 2. Hedge ratio + spread
        beta, alpha, spread = estimate_hedge_ratio(
            s1, s2, use_log=True)

        # 3. ADF directly on spread
        adf_pval = adfuller(spread.dropna(),
                            maxlags=1,
                            autolag=None)[1]

        # 4. Johansen
        j_pval = johansen_test(s1, s2)

        # 5. Half-life
        hl = compute_half_life(spread)

        # 6. Hurst
        hurst = compute_hurst(spread)

        result.update({
            'eg_pval'      : float(eg_pval),
            'adf_pval'     : float(adf_pval),
            'johansen_pval': float(j_pval),
            'half_life'    : float(hl),
            'hurst'        : float(hurst),
            'hedge_ratio'  : float(beta),
            'alpha'        : float(alpha),
            'spread_std'   : float(spread.std()),
        })

        if verbose:
            print(f"      EG={eg_pval:.4f}  "
                  f"ADF={adf_pval:.4f}  "
                  f"J={j_pval:.4f}  "
                  f"HL={hl:.1f}  "
                  f"Hurst={hurst:.3f}")

        # ── Acceptance criteria ───────────────────────
        passes_eg   = eg_pval   < EG_PVAL_MAX
        passes_adf  = adf_pval  < ADF_PVAL_MAX
        passes_j    = j_pval    < 0.10
        passes_hl   = (MIN_HALF_LIFE < hl < MAX_HALF_LIFE
                       and np.isfinite(hl))
        passes_h    = hurst     < HURST_MAX

        # Need EG + ADF, OR Johansen + ADF
        coint_ok = ((passes_eg or passes_j) and
                    passes_adf and
                    passes_hl and
                    passes_h)

        result['cointegrated'] = coint_ok

        if coint_ok:
            score = (
                (1 - min(eg_pval, 0.99))  * 0.25 +
                (1 - min(adf_pval, 0.99)) * 0.25 +
                (1 - min(j_pval, 0.99))   * 0.15 +
                (1 - min(hurst, 1.0))     * 0.20 +
                (1 - min(hl, MAX_HALF_LIFE) /
                 MAX_HALF_LIFE)           * 0.15
            )
            result['score'] = float(score)

    except Exception as e:
        if verbose:
            print(f"      [ERR] {e}")

    return result


# ─────────────────────────────────────────────────────────────
#  ROLLING WINDOW COINTEGRATION TEST
# ─────────────────────────────────────────────────────────────
def test_pair_rolling(sym1: str, sym2: str,
                      s1: pd.Series,
                      s2: pd.Series) -> dict:
    """
    Test cointegration on multiple rolling windows.

    Strategy:
      1. Test on the MOST RECENT COINT_WINDOW bars
         (primary — is it cointegrated RIGHT NOW?)
      2. Test on STABILITY_WINDOWS evenly-spaced
         sub-windows over the last year
         (is the relationship consistent?)
      3. Accept if recent window passes AND
         at least STABILITY_MIN_PASS sub-windows pass
    """
    print(f"\n  ── {sym1} / {sym2} ──")

    # Align
    s1a, s2a = align_series(s1, s2)
    n = len(s1a)

    if n < COINT_WINDOW + 200:
        print(f"    [SKIP] Only {n} aligned bars")
        return {'valid': False}

    # ── Test 1: Most recent window ───────────────────────
    print(f"    [RECENT] Last {COINT_WINDOW} bars:")
    recent_s1 = s1a.iloc[-COINT_WINDOW:]
    recent_s2 = s2a.iloc[-COINT_WINDOW:]
    recent    = test_window(recent_s1, recent_s2,
                            verbose=True)

    recent_pass = recent['cointegrated']
    status = "✓ PASS" if recent_pass else "✗ FAIL"
    print(f"    Recent window: {status}")

    # ── Test 2: Sub-period stability ─────────────────────
    # Use last 1 year = 8736 H1 bars
    lookback  = min(n, 8736)
    step      = lookback // STABILITY_WINDOWS
    stab_pass = 0
    stab_results = []

    print(f"    [STABILITY] {STABILITY_WINDOWS} windows "
          f"over last {lookback} bars:")

    for w in range(STABILITY_WINDOWS):
        end_idx   = n - w * step
        start_idx = max(0, end_idx - COINT_WINDOW)
        if end_idx - start_idx < 500:
            continue

        w_s1 = s1a.iloc[start_idx:end_idx]
        w_s2 = s2a.iloc[start_idx:end_idx]
        wr   = test_window(w_s1, w_s2, verbose=True)
        stab_results.append(wr)
        if wr['cointegrated']:
            stab_pass += 1

    print(f"    Stability: {stab_pass}/{STABILITY_WINDOWS} "
          f"windows passed "
          f"(need {STABILITY_MIN_PASS})")

    # ── Final decision ───────────────────────────────────
    # Recent window must pass
    # Enough stability windows must pass
    is_valid = (recent_pass and
                stab_pass >= STABILITY_MIN_PASS)

    status = "✓✓ VALID PAIR" if is_valid else "✗ REJECTED"
    print(f"    Decision: {status}")

    if not is_valid:
        return {'valid': False,
                'symbol1': sym1, 'symbol2': sym2,
                'recent': recent,
                'stability_pass': stab_pass}

    # Build final result from recent window
    result = {
        'valid'          : True,
        'symbol1'        : sym1,
        'symbol2'        : sym2,
        'eg_pval'        : recent['eg_pval'],
        'adf_pval'       : recent['adf_pval'],
        'johansen_pval'  : recent['johansen_pval'],
        'half_life'      : recent['half_life'],
        'hurst'          : recent['hurst'],
        'hedge_ratio'    : recent['hedge_ratio'],
        'alpha'          : recent['alpha'],
        'spread_std'     : recent['spread_std'],
        'score'          : recent['score'],
        'stability_pass' : stab_pass,
        'coint_window'   : COINT_WINDOW,
        'n_bars_total'   : n,
        'spread_mean'    : 0.0,
    }
    return result


# ─────────────────────────────────────────────────────────────
#  DIAGNOSTIC REPORT — ALWAYS PRINT FULL DETAILS
# ─────────────────────────────────────────────────────────────
def print_diagnostic_report(all_results: list):
    """
    Print full report of ALL pairs tested,
    showing exactly why each one passed or failed.
    Helps you tune the filters intelligently.
    """
    print("\n" + "="*70)
    print("  FULL DIAGNOSTIC REPORT")
    print("="*70)

    headers = (f"{'Pair':<18} {'EG':>7} {'ADF':>7} "
               f"{'J':>7} {'HL':>8} {'Hurst':>7} "
               f"{'Stab':>6} {'Result'}")
    print(headers)
    print("-"*70)

    for r in all_results:
        sym1 = r.get('symbol1', '?')
        sym2 = r.get('symbol2', '?')
        pair = f"{sym1}/{sym2}"

        recent = r.get('recent', r)
        eg   = recent.get('eg_pval', 1.0)
        adf  = recent.get('adf_pval', 1.0)
        j    = recent.get('johansen_pval', 1.0)
        hl   = recent.get('half_life', np.inf)
        h    = recent.get('hurst', 0.5)
        stab = r.get('stability_pass', 0)
        valid = r.get('valid', False)

        hl_str = f"{hl:.1f}" if np.isfinite(hl) else "∞"
        status = "✓ VALID" if valid else "✗"

        print(f"{pair:<18} {eg:>7.4f} {adf:>7.4f} "
              f"{j:>7.4f} {hl_str:>8} {h:>7.3f} "
              f"{stab:>6} {status}")

    print("-"*70)
    valid_n = sum(1 for r in all_results
                  if r.get('valid', False))
    print(f"  Valid pairs: {valid_n} / {len(all_results)}")

    # ── Filter failure analysis ───────────────────────────
    print("\n  FILTER FAILURE BREAKDOWN:")
    print(f"  (thresholds: EG<{EG_PVAL_MAX}  "
          f"ADF<{ADF_PVAL_MAX}  "
          f"HL {MIN_HALF_LIFE}-{MAX_HALF_LIFE}  "
          f"Hurst<{HURST_MAX}  "
          f"StabPass>={STABILITY_MIN_PASS})")

    fail_eg   = sum(1 for r in all_results
                    if not r.get('valid', False) and
                    r.get('recent', r).get(
                        'eg_pval', 1.0) >= EG_PVAL_MAX and
                    r.get('recent', r).get(
                        'johansen_pval', 1.0) >= 0.10)
    fail_adf  = sum(1 for r in all_results
                    if not r.get('valid', False) and
                    r.get('recent', r).get(
                        'adf_pval', 1.0) >= ADF_PVAL_MAX)
    fail_hl   = sum(1 for r in all_results
                    if not r.get('valid', False) and
                    not (MIN_HALF_LIFE <
                         r.get('recent', r).get(
                             'half_life', np.inf) <
                         MAX_HALF_LIFE))
    fail_h    = sum(1 for r in all_results
                    if not r.get('valid', False) and
                    r.get('recent', r).get(
                        'hurst', 1.0) >= HURST_MAX)
    fail_stab = sum(1 for r in all_results
                    if not r.get('valid', False) and
                    r.get('stability_pass', 0) <
                    STABILITY_MIN_PASS)

    print(f"  Failed EG+Johansen: {fail_eg}")
    print(f"  Failed ADF:         {fail_adf}")
    print(f"  Failed half-life:   {fail_hl}")
    print(f"  Failed Hurst:       {fail_h}")
    print(f"  Failed stability:   {fail_stab}")


# ─────────────────────────────────────────────────────────────
#  QUICK BACKTEST
# ─────────────────────────────────────────────────────────────
def quick_backtest(spread: pd.Series,
                   half_life: float,
                   entry_z: float = 2.0,
                   exit_z:  float = 0.0,
                   stop_z:  float = 3.5) -> dict:
    """Simple vectorised backtest on spread series."""
    window    = max(20, int(half_life * 2))
    rm        = spread.rolling(window).mean()
    rs        = spread.rolling(window).std()
    zscore    = (spread - rm) / (rs + 1e-10)

    position, pnl_list, entry_sp = 0, [], 0.0

    for i in range(window+1, len(zscore)):
        z = zscore.iloc[i]
        sp = spread.iloc[i]
        if position == 0:
            if   z < -entry_z:
                position, entry_sp = 1, sp
            elif z >  entry_z:
                position, entry_sp = -1, sp
        elif position == 1:
            if z >= exit_z or z < -stop_z:
                pnl_list.append(sp - entry_sp)
                position = 0
        elif position == -1:
            if z <= exit_z or z > stop_z:
                pnl_list.append(entry_sp - sp)
                position = 0

    if not pnl_list:
        return {'n_trades': 0, 'win_rate': 0,
                'profit_factor': 0, 'sharpe': 0}

    pnl  = np.array(pnl_list)
    wins = pnl[pnl > 0];  losses = pnl[pnl < 0]
    wr   = len(wins) / len(pnl)
    gp   = wins.sum()  if len(wins)   > 0 else 0
    gl   = abs(losses.sum()) if len(losses) > 0 else 1e-10

    return {
        'n_trades'     : len(pnl),
        'win_rate'     : round(wr, 4),
        'profit_factor': round(gp / gl, 4),
        'sharpe'       : round(
            pnl.mean() / (pnl.std()+1e-10) *
            np.sqrt(252), 4),
        'avg_pnl'      : round(pnl.mean(), 8),
    }


# ─────────────────────────────────────────────────────────────
#  PLOT PAIR
# ─────────────────────────────────────────────────────────────
def plot_pair(sym1: str, sym2: str,
              s1: pd.Series, s2: pd.Series,
              spread: pd.Series, result: dict):
    hl     = result.get('half_life', 20)
    window = max(20, int(hl * 2))
    rm     = spread.rolling(window).mean()
    rs     = spread.rolling(window).std()
    z      = (spread - rm) / (rs + 1e-10)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(
        f"{sym1}/{sym2}  "
        f"EG={result.get('eg_pval',1):.4f}  "
        f"ADF={result.get('adf_pval',1):.4f}  "
        f"HL={hl:.1f}  "
        f"Hurst={result.get('hurst',0.5):.3f}  "
        f"Stab={result.get('stability_pass',0)}/4",
        fontsize=11)

    (s1/s1.iloc[0]).iloc[-COINT_WINDOW:].plot(
        ax=axes[0], label=sym1, color='steelblue')
    (s2/s2.iloc[0]).iloc[-COINT_WINDOW:].plot(
        ax=axes[0], label=sym2, color='darkorange')
    axes[0].set_title("Normalised Prices (recent window)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    spread.iloc[-COINT_WINDOW:].plot(
        ax=axes[1], color='purple', linewidth=0.8)
    axes[1].axhline(spread.iloc[-COINT_WINDOW:].mean(),
                    color='black', linestyle='--',
                    linewidth=1)
    axes[1].set_title("Spread"); axes[1].grid(True,alpha=0.3)

    z.iloc[-COINT_WINDOW:].plot(
        ax=axes[2], color='darkgreen', linewidth=0.8)
    for level, col in [(2,'red'),(-2,'red'),
                       (3.5,'darkred'),(-3.5,'darkred'),
                       (0,'black')]:
        axes[2].axhline(level, color=col,
                        linestyle='--' if abs(level)>0.1
                        else '-', linewidth=0.8)
    axes[2].set_title("Z-Score")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR,
                     f"{sym1}_{sym2}_analysis.png"),
        dpi=120, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def run_research(timeframe: str = "H1"):
    print("="*60)
    print("  PAIRS TRADING RESEARCH PIPELINE  v2.0")
    print(f"  Coint window: {COINT_WINDOW} bars  "
          f"EG<{EG_PVAL_MAX}  ADF<{ADF_PVAL_MAX}  "
          f"HL<{MAX_HALF_LIFE}  Hurst<{HURST_MAX}")
    print("="*60)

    # Load data
    print("\n[1] Loading price data...")
    price_data = {}
    for sym in UNIVERSE:
        s = load_price_series(sym, timeframe)
        if len(s) >= MIN_BARS:
            price_data[sym] = s

    available = list(price_data.keys())
    print(f"\n  Loaded {len(available)} symbols: "
          f"{available}")

    if len(available) < 2:
        print("[ERR] Need at least 2 symbols")
        return

    # Test all pairs
    print(f"\n[2] Testing {len(list(combinations(available,2)))} "
          f"pairs with rolling window cointegration...")

    all_results = []
    valid_pairs = []

    for sym1, sym2 in combinations(available, 2):
        r = test_pair_rolling(
            sym1, sym2,
            price_data[sym1],
            price_data[sym2])

        all_results.append(r)

        if r.get('valid', False):
            # Quick backtest on recent window
            recent_s1 = price_data[sym1].iloc[
                -COINT_WINDOW:]
            recent_s2 = price_data[sym2].iloc[
                -COINT_WINDOW:]
            recent_s1, recent_s2 = align_series(
                recent_s1, recent_s2)

            _, _, spread = estimate_hedge_ratio(
                recent_s1, recent_s2, use_log=True)
            bt = quick_backtest(spread, r['half_life'])
            r['backtest'] = bt

            print(f"    Backtest: n={bt['n_trades']} "
                  f"WR={bt['win_rate']:.1%} "
                  f"PF={bt['profit_factor']:.2f}")

            plot_pair(sym1, sym2,
                      recent_s1, recent_s2,
                      spread, r)
            valid_pairs.append(r)

    # Diagnostic report — always printed
    print_diagnostic_report(all_results)

    # Save valid pairs
    valid_pairs.sort(key=lambda x: x['score'],
                     reverse=True)

    save_data = []
    for p in valid_pairs:
        row = {k: v for k, v in p.items()
               if k not in ('backtest',) and
               isinstance(v, (int, float, str, bool))}
        if 'backtest' in p:
            row.update(p['backtest'])
        save_data.append(row)

    out = os.path.join(OUTPUT_DIR, "valid_pairs.json")
    with open(out, 'w') as f:
        json.dump(save_data, f, indent=2,
                  default=str)

    # Also save full diagnostic for tuning
    diag_out = os.path.join(OUTPUT_DIR,
                             "diagnostic_report.json")
    diag_data = []
    for r in all_results:
        row = {}
        for k, v in r.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    row[f"recent_{k2}"] = (
                        float(v2)
                        if isinstance(v2, (np.floating,
                                           np.integer,
                                           float, int))
                        else str(v2))
            elif isinstance(v, (np.floating,
                                np.integer)):
                row[k] = float(v)
            else:
                row[k] = v
        diag_data.append(row)

    with open(diag_out, 'w') as f:
        json.dump(diag_data, f, indent=2, default=str)

    print(f"\n[SAVED] valid_pairs.json  → "
          f"{len(valid_pairs)} pairs")
    print(f"[SAVED] diagnostic_report.json  → "
          f"full metrics for all pairs")
    print(f"\n[DONE] "
          f"{'No valid pairs found.' if not valid_pairs else f'{len(valid_pairs)} valid pairs ready for backtest.'}")

    if not valid_pairs:
        print("\n── TUNING SUGGESTIONS ──────────────────")
        print("  If all pairs failed, check the")
        print("  diagnostic_report.json to see which")
        print("  filter is blocking the most pairs.")
        print("  Common fixes:")
        print("    1. EG all > 0.10 → "
              "try shorter COINT_WINDOW (1008)")
        print("    2. HL all > 168  → "
              "try daily data (D1) instead of H1")
        print("    3. Hurst > 0.55  → "
              "try COINT_WINDOW = 504 (1 month)")
        print("    4. Stability<2   → "
              "lower STABILITY_MIN_PASS to 1")

    return valid_pairs


if __name__ == "__main__":
    run_research(timeframe="H1")
