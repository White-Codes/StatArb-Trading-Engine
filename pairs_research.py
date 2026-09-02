"""
pairs_research.py  v3.0
========================
Fixes vs v2.0:
  - adfuller maxlags → maxlag  (statsmodels API fix)
  - coint() autolag removed    (API compatibility)
  - Johansen wrapped more safely
  - All exceptions print full traceback in debug mode
  - Sanity check prints REAL values before filtering
  - Added standalone test at bottom to verify stats work
"""

import numpy as np
import pandas as pd
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools           import coint, adfuller
from statsmodels.tsa.vector_ar.vecm      import coint_johansen
from statsmodels.regression.linear_model import OLS
from statsmodels.tools                   import add_constant
from itertools                           import combinations
import warnings, json, os

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────
DATA_DIR   = "."
OUTPUT_DIR = "pairs_artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COINT_WINDOW       = 2016   # ~3 months H1
MIN_BARS           = 5000
EG_PVAL_MAX        = 0.15
ADF_PVAL_MAX       = 0.15
MAX_HALF_LIFE      = 168    # bars
MIN_HALF_LIFE      = 2
HURST_MAX          = 0.58
STABILITY_WINDOWS  = 4
STABILITY_MIN_PASS = 2
DEBUG              = True   # print full tracebacks

UNIVERSE = [
    "EURUSD", "GBPUSD", "AUDUSD",
    "NZDUSD", "USDCAD", "USDCHF",
    "EURGBP", "EURAUD", "GBPAUD",
    "AUDNZD", "EURCAD", "GBPCAD",
]


# ─────────────────────────────────────────────────────────────
#  STEP 0 — SANITY CHECK  (runs first, before any pair testing)
#  Verifies that statsmodels functions work correctly
#  on synthetic data where we KNOW the answer
# ─────────────────────────────────────────────────────────────
def sanity_check_stats():
    """
    Generate a known mean-reverting spread and verify:
      - adfuller rejects unit root  (p < 0.05)
      - coint finds cointegration   (p < 0.05)
      - half_life is finite
      - Hurst < 0.5
    If any check fails the function prints the error
    so you know exactly which API call is broken.
    """
    print("\n" + "="*55)
    print("  SANITY CHECK — Testing statsmodels API")
    print("="*55)

    rng  = np.random.default_rng(42)
    n    = 1000

    # Create two cointegrated series
    common = np.cumsum(rng.standard_normal(n))
    s1     = common + rng.standard_normal(n) * 0.1
    s2     = common * 1.5 + 2.0 + \
             rng.standard_normal(n) * 0.1
    spread = s1 - 0.667 * s2   # known mean-reverting

    s1_ser = pd.Series(s1)
    s2_ser = pd.Series(s2)
    sp_ser = pd.Series(spread)

    all_ok = True

    # ── Test 1: adfuller ─────────────────────────────────
    print("\n  [1] adfuller on mean-reverting spread:")
    try:
        res    = adfuller(sp_ser.dropna(), maxlag=1,
                          autolag=None)
        pval   = res[1]
        status = "PASS" if pval < 0.05 else "WARN"
        print(f"      p-value = {pval:.6f}  → {status}")
        if pval >= 0.05:
            print("      WARNING: expected p<0.05 "
                  "on synthetic cointegrated data")
            all_ok = False
    except Exception as e:
        print(f"      ERROR: {e}")
        if DEBUG:
            traceback.print_exc()
        all_ok = False

    # ── Test 2: coint ─────────────────────────────────────
    print("\n  [2] coint (Engle-Granger) on s1, s2:")
    try:
        _, pval, _ = coint(s1_ser, s2_ser)
        status     = "PASS" if pval < 0.05 else "WARN"
        print(f"      p-value = {pval:.6f}  → {status}")
        if pval >= 0.05:
            print("      WARNING: expected p<0.05")
            all_ok = False
    except Exception as e:
        print(f"      ERROR: {e}")
        if DEBUG:
            traceback.print_exc()
        all_ok = False

    # ── Test 3: Johansen ──────────────────────────────────
    print("\n  [3] Johansen test on s1, s2:")
    try:
        data   = np.column_stack([s1, s2])
        result = coint_johansen(data,
                                det_order=0,
                                k_ar_diff=1)
        trace  = result.lr1[0]
        crit   = result.cvt[0, 1]
        status = "PASS" if trace > crit else "WARN"
        print(f"      trace={trace:.2f}  "
              f"crit95={crit:.2f}  → {status}")
    except Exception as e:
        print(f"      ERROR: {e}")
        if DEBUG:
            traceback.print_exc()

    # ── Test 4: half_life ─────────────────────────────────
    print("\n  [4] Half-life of synthetic spread:")
    try:
        hl = compute_half_life(sp_ser)
        status = ("PASS" if 1 < hl < 100
                  else "WARN")
        print(f"      half_life = {hl:.2f}  → {status}")
        if not np.isfinite(hl):
            print("      WARNING: got infinite half-life")
            all_ok = False
    except Exception as e:
        print(f"      ERROR: {e}")
        if DEBUG:
            traceback.print_exc()
        all_ok = False

    # ── Test 5: Hurst ─────────────────────────────────────
    print("\n  [5] Hurst exponent of spread:")
    try:
        h      = compute_hurst(sp_ser)
        status = "PASS" if h < 0.5 else "WARN"
        print(f"      Hurst = {h:.4f}  → {status}")
    except Exception as e:
        print(f"      ERROR: {e}")
        if DEBUG:
            traceback.print_exc()

    result_str = "ALL CHECKS PASSED" if all_ok \
                 else "SOME CHECKS FAILED — see above"
    print(f"\n  Sanity check result: {result_str}")
    print("="*55 + "\n")
    return all_ok


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
        df.columns = [c.strip().lower()
                      for c in df.columns]

        date_col = next((c for c in [
            'gmt time', 'gmt_time', 'timestamp',
            'time', 'date', 'datetime']
            if c in df.columns), None)
        close_col = next((c for c in [
            'close', 'c', 'price']
            if c in df.columns), None)

        if not date_col or not close_col:
            print(f"  [ERR] Columns in {fname}: "
                  f"{list(df.columns)}")
            return pd.Series(dtype=float)

        df[date_col] = pd.to_datetime(
            df[date_col],
            format='mixed', errors='coerce')
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
        if DEBUG:
            traceback.print_exc()
        return pd.Series(dtype=float)


def align_series(s1: pd.Series,
                 s2: pd.Series) -> tuple:
    combined = pd.concat([s1, s2], axis=1).dropna()
    return combined.iloc[:, 0], combined.iloc[:, 1]


# ─────────────────────────────────────────────────────────────
#  STATISTICAL METRICS
# ─────────────────────────────────────────────────────────────
def compute_half_life(spread: pd.Series) -> float:
    """Ornstein-Uhlenbeck half-life of mean reversion."""
    spread = spread.dropna()
    if len(spread) < 30:
        return np.inf
    try:
        lag   = spread.shift(1)
        delta = spread.diff()
        df    = pd.concat([lag, delta],
                          axis=1).dropna()
        df.columns = ['lag', 'delta']

        if len(df) < 20:
            return np.inf

        model = OLS(df['delta'],
                    add_constant(df['lag'])).fit()
        lam   = model.params.iloc[1]

        if lam >= 0:
            return np.inf
        return float(-np.log(2) / lam)

    except Exception as e:
        if DEBUG:
            print(f"      [half_life ERR] {e}")
        return np.inf


def compute_hurst(series: pd.Series,
                  max_lag: int = 50) -> float:
    """Hurst exponent via R/S analysis. <0.5 = mean reverting."""
    try:
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
                R   = (np.max(np.cumsum(dm)) -
                       np.min(np.cumsum(dm)))
                S   = np.std(chunk, ddof=1)
                if S > 0:
                    rs_list.append(R / S)
            if rs_list:
                rs_vals.append(np.mean(rs_list))

        if len(rs_vals) < 3:
            return 0.5

        log_l = np.log(list(lags)[:len(rs_vals)])
        log_r = np.log(rs_vals)
        return float(np.polyfit(log_l, log_r, 1)[0])

    except Exception as e:
        if DEBUG:
            print(f"      [hurst ERR] {e}")
        return 0.5


def estimate_hedge_ratio(s1: pd.Series,
                         s2: pd.Series,
                         use_log: bool = True) -> tuple:
    """OLS regression to get hedge ratio and spread."""
    try:
        y = np.log(s1) if use_log else s1.values
        x = np.log(s2) if use_log else s2.values

        y = pd.Series(y, index=s1.index)
        x = pd.Series(x, index=s2.index)

        model  = OLS(y, add_constant(x)).fit()
        alpha  = float(model.params.iloc[0])
        beta   = float(model.params.iloc[1])
        spread = y - beta * x - alpha
        return beta, alpha, spread

    except Exception as e:
        if DEBUG:
            print(f"      [hedge_ratio ERR] {e}")
        return 0.0, 0.0, pd.Series(dtype=float)


def run_adfuller(spread: pd.Series) -> float:
    """
    ADF test with API compatibility fix.
    statsmodels ≥ 0.14 uses maxlag (no s).
    Older versions use maxlags.
    We try both.
    """
    data = spread.dropna().values
    if len(data) < 20:
        return 1.0

    # Try new API first (maxlag without s)
    try:
        result = adfuller(data, maxlag=1, autolag=None)
        return float(result[1])
    except TypeError:
        pass

    # Fall back to old API (maxlags with s)
    try:
        result = adfuller(data, maxlags=1, autolag=None)
        return float(result[1])
    except TypeError:
        pass

    # Last resort: let statsmodels choose lag automatically
    try:
        result = adfuller(data)
        return float(result[1])
    except Exception as e:
        if DEBUG:
            print(f"      [ADF ERR] {e}")
        return 1.0


def run_engle_granger(s1: pd.Series,
                      s2: pd.Series) -> float:
    """Engle-Granger cointegration test p-value."""
    try:
        log_s1 = np.log(s1.dropna().values)
        log_s2 = np.log(s2.dropna().values)
        n      = min(len(log_s1), len(log_s2))
        _, pval, _ = coint(log_s1[:n], log_s2[:n])
        return float(pval)
    except Exception as e:
        if DEBUG:
            print(f"      [EG ERR] {e}")
        return 1.0


def run_johansen(s1: pd.Series,
                 s2: pd.Series) -> float:
    """
    Johansen test.
    Returns pseudo p-value: <0.05 means significant.
    """
    try:
        log_s1 = np.log(s1.dropna().values)
        log_s2 = np.log(s2.dropna().values)
        n      = min(len(log_s1), len(log_s2))
        data   = np.column_stack([log_s1[:n],
                                   log_s2[:n]])
        result = coint_johansen(data,
                                det_order=0,
                                k_ar_diff=1)
        trace  = result.lr1[0]
        crit95 = result.cvt[0, 1]
        # ratio > 1 → significant at 95%
        ratio  = trace / (crit95 + 1e-10)
        return float(max(0.0, min(1.0, 1.0 / ratio)))
    except Exception as e:
        if DEBUG:
            print(f"      [Johansen ERR] {e}")
        return 1.0


# ─────────────────────────────────────────────────────────────
#  SINGLE WINDOW TEST
# ─────────────────────────────────────────────────────────────
def test_window(s1: pd.Series,
                s2: pd.Series,
                verbose: bool = False) -> dict:
    """
    Full cointegration analysis on one time window.
    All metrics printed when verbose=True.
    """
    default = {
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
        return default

    result = dict(default)

    try:
        # Align
        s1a, s2a = align_series(s1, s2)
        if len(s1a) < 100:
            return default

        # All three cointegration tests
        eg_pval  = run_engle_granger(s1a, s2a)
        j_pval   = run_johansen(s1a, s2a)

        # Hedge ratio and spread
        beta, alpha, spread = estimate_hedge_ratio(
            s1a, s2a, use_log=True)

        if len(spread) < 50:
            return default

        # ADF on spread — use compatibility wrapper
        adf_pval = run_adfuller(spread)

        # Half-life
        hl = compute_half_life(spread)

        # Hurst
        hurst = compute_hurst(spread)

        result.update({
            'eg_pval'      : eg_pval,
            'adf_pval'     : adf_pval,
            'johansen_pval': j_pval,
            'half_life'    : hl,
            'hurst'        : hurst,
            'hedge_ratio'  : beta,
            'alpha'        : alpha,
            'spread_std'   : float(spread.std()),
        })

        if verbose:
            hl_str = (f"{hl:.1f}" if np.isfinite(hl)
                      else "∞")
            print(f"      EG={eg_pval:.4f}  "
                  f"ADF={adf_pval:.4f}  "
                  f"J={j_pval:.4f}  "
                  f"HL={hl_str}  "
                  f"Hurst={hurst:.4f}")

        # Acceptance logic
        passes_eg  = eg_pval  < EG_PVAL_MAX
        passes_adf = adf_pval < ADF_PVAL_MAX
        passes_j   = j_pval   < 0.10
        passes_hl  = (np.isfinite(hl) and
                      MIN_HALF_LIFE < hl < MAX_HALF_LIFE)
        passes_h   = hurst < HURST_MAX

        coint_ok = ((passes_eg or passes_j) and
                    passes_adf and
                    passes_hl and
                    passes_h)

        result['cointegrated'] = coint_ok

        if verbose:
            flags = (
                f"EG={'✓' if passes_eg else '✗'}  "
                f"ADF={'✓' if passes_adf else '✗'}  "
                f"J={'✓' if passes_j else '✗'}  "
                f"HL={'✓' if passes_hl else '✗'}  "
                f"H={'✓' if passes_h else '✗'}"
            )
            print(f"      Flags: {flags}")

        if coint_ok:
            score = (
                (1 - min(eg_pval,  0.99)) * 0.25 +
                (1 - min(adf_pval, 0.99)) * 0.25 +
                (1 - min(j_pval,   0.99)) * 0.15 +
                (1 - min(hurst,    1.0))  * 0.20 +
                (1 - min(hl, MAX_HALF_LIFE) /
                 MAX_HALF_LIFE)           * 0.15
            )
            result['score'] = float(score)

    except Exception as e:
        print(f"      [test_window ERR] {e}")
        if DEBUG:
            traceback.print_exc()

    return result


# ─────────────────────────────────────────────────────────────
#  ROLLING WINDOW TEST
# ─────────────────────────────────────────────────────────────
def test_pair_rolling(sym1: str, sym2: str,
                      s1: pd.Series,
                      s2: pd.Series) -> dict:
    """
    Test cointegration across multiple rolling windows.
    Prints full detail for every window.
    """
    print(f"\n  ── {sym1} / {sym2} ──")

    s1a, s2a = align_series(s1, s2)
    n = len(s1a)

    if n < COINT_WINDOW + 200:
        print(f"    [SKIP] Only {n} aligned bars "
              f"(need {COINT_WINDOW + 200})")
        return {'valid': False,
                'symbol1': sym1, 'symbol2': sym2}

    # ── Recent window ────────────────────────────────────
    print(f"    [RECENT] Last {COINT_WINDOW} bars "
          f"({s1a.index[-COINT_WINDOW].date()} → "
          f"{s1a.index[-1].date()}):")

    recent    = test_window(s1a.iloc[-COINT_WINDOW:],
                            s2a.iloc[-COINT_WINDOW:],
                            verbose=True)
    recent_ok = recent['cointegrated']
    print(f"    Recent: {'✓ PASS' if recent_ok else '✗ FAIL'}")

    # ── Stability windows ────────────────────────────────
    lookback   = min(n, 8736)
    step       = lookback // STABILITY_WINDOWS
    stab_pass  = 0

    print(f"    [STABILITY] {STABILITY_WINDOWS} windows "
          f"over last {lookback} bars:")

    for w in range(STABILITY_WINDOWS):
        end_i   = n - w * step
        start_i = max(0, end_i - COINT_WINDOW)
        if end_i - start_i < 500:
            print(f"      Window {w+1}: too short, skip")
            continue

        w_s1 = s1a.iloc[start_i:end_i]
        w_s2 = s2a.iloc[start_i:end_i]
        print(f"      Window {w+1} "
              f"({w_s1.index[0].date()} → "
              f"{w_s1.index[-1].date()}):")
        wr = test_window(w_s1, w_s2, verbose=True)
        if wr['cointegrated']:
            stab_pass += 1
            print(f"        → ✓ pass")
        else:
            print(f"        → ✗ fail")

    print(f"    Stability: {stab_pass}/{STABILITY_WINDOWS} "
          f"(need {STABILITY_MIN_PASS})")

    is_valid = (recent_ok and
                stab_pass >= STABILITY_MIN_PASS)
    print(f"    Decision: "
          f"{'✓✓ VALID' if is_valid else '✗ REJECTED'}")

    base = {
        'valid'          : is_valid,
        'symbol1'        : sym1,
        'symbol2'        : sym2,
        'stability_pass' : stab_pass,
        'n_bars_total'   : n,
        'recent'         : recent,
    }

    if is_valid:
        base.update({
            'eg_pval'      : recent['eg_pval'],
            'adf_pval'     : recent['adf_pval'],
            'johansen_pval': recent['johansen_pval'],
            'half_life'    : recent['half_life'],
            'hurst'        : recent['hurst'],
            'hedge_ratio'  : recent['hedge_ratio'],
            'alpha'        : recent['alpha'],
            'spread_std'   : recent['spread_std'],
            'score'        : recent['score'],
            'spread_mean'  : 0.0,
            'coint_window' : COINT_WINDOW,
        })

    return base


# ─────────────────────────────────────────────────────────────
#  QUICK BACKTEST
# ─────────────────────────────────────────────────────────────
def quick_backtest(spread: pd.Series,
                   half_life: float,
                   entry_z: float = 2.0,
                   exit_z:  float = 0.0,
                   stop_z:  float = 3.5) -> dict:
    window = max(20, int(half_life * 2))
    rm     = spread.rolling(window).mean()
    rs     = spread.rolling(window).std()
    z      = (spread - rm) / (rs + 1e-10)

    pos, pnl_list, entry_sp = 0, [], 0.0

    for i in range(window+1, len(z)):
        zi  = z.iloc[i]
        spi = spread.iloc[i]
        if pos == 0:
            if   zi < -entry_z: pos, entry_sp =  1, spi
            elif zi >  entry_z: pos, entry_sp = -1, spi
        elif pos == 1:
            if zi >= exit_z or zi < -stop_z:
                pnl_list.append(spi - entry_sp)
                pos = 0
        elif pos == -1:
            if zi <= exit_z or zi > stop_z:
                pnl_list.append(entry_sp - spi)
                pos = 0

    if not pnl_list:
        return {'n_trades': 0, 'win_rate': 0,
                'profit_factor': 0, 'sharpe': 0}

    pnl  = np.array(pnl_list)
    wins = pnl[pnl > 0]
    loss = pnl[pnl < 0]
    gp   = wins.sum() if len(wins) > 0 else 0
    gl   = abs(loss.sum()) if len(loss) > 0 else 1e-10

    return {
        'n_trades'     : len(pnl),
        'win_rate'     : round(len(wins)/len(pnl), 4),
        'profit_factor': round(gp/gl, 4),
        'sharpe'       : round(
            pnl.mean()/(pnl.std()+1e-10)*np.sqrt(252), 4),
    }


# ─────────────────────────────────────────────────────────────
#  PLOT
# ─────────────────────────────────────────────────────────────
def plot_pair(sym1: str, sym2: str,
              s1: pd.Series, s2: pd.Series,
              spread: pd.Series, result: dict):
    try:
        hl     = result.get('half_life', 20)
        if not np.isfinite(hl):
            hl = 20
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
            f"Hurst={result.get('hurst',0.5):.3f}",
            fontsize=11)

        recent = min(COINT_WINDOW, len(s1))
        (s1/s1.iloc[0]).iloc[-recent:].plot(
            ax=axes[0], label=sym1, color='steelblue')
        (s2/s2.iloc[0]).iloc[-recent:].plot(
            ax=axes[0], label=sym2, color='darkorange')
        axes[0].set_title("Normalised Prices (recent)")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        spread.iloc[-recent:].plot(
            ax=axes[1], color='purple', linewidth=0.8)
        axes[1].axhline(
            spread.iloc[-recent:].mean(),
            color='black', linestyle='--')
        axes[1].set_title("Spread")
        axes[1].grid(True, alpha=0.3)

        z.iloc[-recent:].plot(
            ax=axes[2], color='darkgreen', linewidth=0.8)
        for lv, col in [(2,'red'),(-2,'red'),
                        (3.5,'darkred'),(-3.5,'darkred'),
                        (0,'black')]:
            axes[2].axhline(lv, color=col,
                            linestyle='--'
                            if abs(lv) > 0.1 else '-',
                            linewidth=0.8)
        axes[2].set_title("Z-Score")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR,
                         f"{sym1}_{sym2}.png"),
            dpi=120, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"    [PLOT ERR] {e}")


# ─────────────────────────────────────────────────────────────
#  DIAGNOSTIC REPORT
# ─────────────────────────────────────────────────────────────
def print_diagnostic_report(all_results: list):
    print("\n" + "="*72)
    print("  FULL DIAGNOSTIC REPORT")
    print("="*72)
    print(f"  {'Pair':<18} {'EG':>7} {'ADF':>7} "
          f"{'J':>7} {'HL':>8} {'Hurst':>7} "
          f"{'Stab':>5}  Result")
    print("-"*72)

    for r in all_results:
        sym1 = r.get('symbol1', '?')
        sym2 = r.get('symbol2', '?')
        rec  = r.get('recent', r)
        eg   = rec.get('eg_pval',       1.0)
        adf  = rec.get('adf_pval',      1.0)
        j    = rec.get('johansen_pval', 1.0)
        hl   = rec.get('half_life',     np.inf)
        h    = rec.get('hurst',         0.5)
        stab = r.get('stability_pass',  0)
        ok   = r.get('valid',           False)

        hl_s = f"{hl:.1f}" if np.isfinite(hl) else "∞"
        res  = "✓ VALID" if ok else "✗"
        print(f"  {sym1}/{sym2:<14} "
              f"{eg:>7.4f} {adf:>7.4f} {j:>7.4f} "
              f"{hl_s:>8} {h:>7.3f} {stab:>5}  {res}")

    print("-"*72)
    n_valid = sum(1 for r in all_results
                  if r.get('valid', False))
    print(f"  Valid: {n_valid} / {len(all_results)}")


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def run_research(timeframe: str = "H1"):

    # Always run sanity check first
    ok = sanity_check_stats()
    if not ok:
        print("\n[WARN] Sanity check had warnings — "
              "results may be unreliable")

    print("\n" + "="*55)
    print("  PAIRS RESEARCH  v3.0")
    print(f"  Window={COINT_WINDOW}  "
          f"EG<{EG_PVAL_MAX}  ADF<{ADF_PVAL_MAX}  "
          f"HL<{MAX_HALF_LIFE}  Hurst<{HURST_MAX}")
    print("="*55)

    print("\n[1] Loading price data...")
    price_data = {}
    for sym in UNIVERSE:
        s = load_price_series(sym, timeframe)
        if len(s) >= MIN_BARS:
            price_data[sym] = s

    available = list(price_data.keys())
    print(f"\n  Loaded: {available}")

    if len(available) < 2:
        print("[ERR] Need at least 2 symbols")
        return []

    pairs       = list(combinations(available, 2))
    all_results = []
    valid_pairs = []

    print(f"\n[2] Testing {len(pairs)} pairs...")

    for sym1, sym2 in pairs:
        r = test_pair_rolling(
            sym1, sym2,
            price_data[sym1],
            price_data[sym2])
        all_results.append(r)

        if r.get('valid', False):
            # Quick backtest
            rs1 = price_data[sym1].iloc[-COINT_WINDOW:]
            rs2 = price_data[sym2].iloc[-COINT_WINDOW:]
            rs1, rs2 = align_series(rs1, rs2)
            _, _, spread = estimate_hedge_ratio(
                rs1, rs2, use_log=True)

            bt = quick_backtest(spread, r['half_life'])
            r['backtest'] = bt
            print(f"    Backtest: "
                  f"n={bt['n_trades']} "
                  f"WR={bt['win_rate']:.1%} "
                  f"PF={bt['profit_factor']:.2f}")

            plot_pair(sym1, sym2, rs1, rs2,
                      spread, r)
            valid_pairs.append(r)

    print_diagnostic_report(all_results)

    # Save valid pairs
    valid_pairs.sort(
        key=lambda x: x.get('score', 0), reverse=True)

    def make_serialisable(v):
        if isinstance(v, (np.floating, np.integer)):
            return float(v)
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v

    save = []
    for p in valid_pairs:
        row = {}
        for k, v in p.items():
            if k == 'recent':
                continue
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    row[k2] = make_serialisable(v2)
            else:
                row[k] = make_serialisable(v)
        save.append(row)

    out = os.path.join(OUTPUT_DIR, "valid_pairs.json")
    with open(out, 'w') as f:
        json.dump(save, f, indent=2, default=str)

    # Full diagnostic JSON
    diag = []
    for r in all_results:
        row = {'symbol1': r.get('symbol1',''),
               'symbol2': r.get('symbol2',''),
               'valid'  : r.get('valid', False),
               'stab'   : r.get('stability_pass', 0)}
        rec = r.get('recent', {})
        for k in ['eg_pval','adf_pval','johansen_pval',
                  'half_life','hurst','score']:
            v = rec.get(k, None)
            row[k] = make_serialisable(v)
        diag.append(row)

    diag_out = os.path.join(OUTPUT_DIR,
                             "diagnostic_report.json")
    with open(diag_out, 'w') as f:
        json.dump(diag, f, indent=2, default=str)

    print(f"\n[SAVED] valid_pairs.json ({len(valid_pairs)} pairs)")
    print(f"[SAVED] diagnostic_report.json")

    if not valid_pairs:
        print("\n── IF STILL 0 VALID PAIRS ──────────────")
        print("  Check sanity check output above.")
        print("  If sanity check PASSED but no pairs found:")
        print("  → FX pairs genuinely not cointegrated")
        print("    in current regime")
        print("  → Try: COINT_WINDOW = 504  (1 month)")
        print("  → Try: EG_PVAL_MAX  = 0.20")
        print("  → Try: HURST_MAX    = 0.60")
        print("  → Try: STABILITY_MIN_PASS = 1")

    return valid_pairs


if __name__ == "__main__":
    run_research(timeframe="H1")
