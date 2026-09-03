"""
pairs_research.py  v4.0
========================
Complete rewrite with all fixes integrated:

  FIX 1: Hurst computed on spread.diff() not raw spread
          → eliminates Hurst > 1 bug
  FIX 2: Johansen uses critical value table correctly
          → eliminates J=1.0000 for all pairs  
  FIX 3: Hurst filter removed entirely as primary gate
          → EG + ADF are sufficient for cointegration
          → Hurst used only for scoring, not rejection
  FIX 4: Stability filter lowered to 1 window
  FIX 5: Version tag printed at startup to confirm
          new code is actually running
"""

import numpy as np
import pandas as pd
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools            import coint, adfuller
from statsmodels.tsa.vector_ar.vecm       import coint_johansen
from statsmodels.regression.linear_model  import OLS
from statsmodels.tools                    import add_constant
from itertools                            import combinations
import warnings, json, os

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
#  VERSION — check this prints in your Actions log
#  If you still see v3.0 the old file is running
# ─────────────────────────────────────────────────────────────
VERSION = "v4.0-HURST-FIX"

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────
DATA_DIR   = "."
OUTPUT_DIR = "pairs_artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COINT_WINDOW       = 2016   # ~3 months H1
MIN_BARS           = 5000

# Cointegration filters
EG_PVAL_MAX        = 0.15
ADF_PVAL_MAX       = 0.15

# Half-life filter
MAX_HALF_LIFE      = 200
MIN_HALF_LIFE      = 2

# Hurst: used for SCORING only, NOT for rejection
# After the diff() fix values will be in [0, 1]
# Kept here for reference but not used as a hard gate
HURST_SCORE_WEIGHT = 0.20

# Stability
STABILITY_WINDOWS  = 4
STABILITY_MIN_PASS = 1      # lowered: FX is regime-dependent

DEBUG = True

UNIVERSE = [
    "EURUSD", "GBPUSD", "AUDUSD",
    "NZDUSD", "USDCAD", "USDCHF",
    "EURGBP", "EURAUD", "GBPAUD",
    "AUDNZD", "EURCAD", "GBPCAD",
]


# ─────────────────────────────────────────────────────────────
#  SANITY CHECK
# ─────────────────────────────────────────────────────────────
def sanity_check():
    print(f"\n{'='*55}")
    print(f"  PAIRS RESEARCH {VERSION}")
    print(f"  Sanity checking statsmodels API...")
    print(f"{'='*55}")

    rng    = np.random.default_rng(42)
    n      = 1000
    common = np.cumsum(rng.standard_normal(n))
    s1     = pd.Series(common +
                       rng.standard_normal(n) * 0.1)
    s2     = pd.Series(common * 1.5 + 2.0 +
                       rng.standard_normal(n) * 0.1)
    spread = pd.Series(s1 - 0.667 * s2)

    # Test adfuller — try both keyword spellings
    adf_ok = False
    try:
        r = adfuller(spread.values, maxlag=1,
                     autolag=None)
        print(f"  adfuller(maxlag):  p={r[1]:.4f}  OK")
        adf_ok = True
    except TypeError:
        try:
            r = adfuller(spread.values, maxlags=1,
                         autolag=None)
            print(f"  adfuller(maxlags): p={r[1]:.4f}  OK")
            adf_ok = True
        except Exception as e:
            print(f"  adfuller FAILED: {e}")

    # Test coint
    try:
        _, p, _ = coint(s1.values, s2.values)
        print(f"  coint:             p={p:.4f}  "
              f"{'OK' if p < 0.05 else 'WARN'}")
    except Exception as e:
        print(f"  coint FAILED: {e}")

    # Test Johansen
    try:
        data = np.column_stack([s1.values, s2.values])
        res  = coint_johansen(data, det_order=0,
                              k_ar_diff=1)
        tr   = res.lr1[0]
        cv   = res.cvt[0, 1]
        print(f"  johansen:  trace={tr:.2f} "
              f"cv95={cv:.2f}  "
              f"{'OK significant' if tr > cv else 'WARN not significant'}")
    except Exception as e:
        print(f"  johansen FAILED: {e}")

    # Test Hurst on known data
    ou = np.zeros(500)
    for t in range(1, 500):
        ou[t] = 0.7 * ou[t-1] + rng.standard_normal()
    h_ou = _hurst_on_diffs(pd.Series(ou))
    print(f"  Hurst(OU process):  {h_ou:.4f}  "
          f"{'OK <0.5' if h_ou < 0.5 else 'WARN >=0.5'}")

    rw   = pd.Series(np.cumsum(rng.standard_normal(500)))
    h_rw = _hurst_on_diffs(rw)
    print(f"  Hurst(random walk): {h_rw:.4f}  "
          f"(expect ~0.5)")

    h_sp = _hurst_on_diffs(spread)
    print(f"  Hurst(coint spread):{h_sp:.4f}  "
          f"{'OK <0.5' if h_sp < 0.5 else 'WARN >=0.5'}")

    print(f"{'='*55}\n")


# ─────────────────────────────────────────────────────────────
#  HURST — operates on DIFFERENCES to avoid level trend
# ─────────────────────────────────────────────────────────────
def _hurst_on_diffs(series: pd.Series,
                    max_lag: int = 50) -> float:
    """
    Compute Hurst exponent on first differences.

    WHY DIFFS:
      Raw spread levels retain low-frequency drift
      even after OLS hedge ratio estimation.
      R/S on raw levels reads the drift as persistence
      and returns Hurst > 1 (mathematically impossible
      for a proper R/S estimate, but happens due to
      non-stationarity in the input).

      Taking first differences removes the level trend
      and measures autocorrelation in the CHANGES,
      which is the correct quantity for assessing
      mean reversion speed.

    Returns value in [0.0, 1.0] always.
    """
    try:
        diffs = series.dropna().diff().dropna().values
        n     = len(diffs)

        if n < max_lag * 2:
            return 0.5

        lag_range = range(8, min(max_lag, n // 4))
        if len(lag_range) < 3:
            return 0.5

        rs_vals = []
        for lag in lag_range:
            chunks = [diffs[i:i+lag]
                      for i in range(0, n - lag, lag)]
            rs_chunk = []
            for chunk in chunks:
                if len(chunk) < 4:
                    continue
                dm = chunk - chunk.mean()
                R  = (np.max(np.cumsum(dm)) -
                      np.min(np.cumsum(dm)))
                S  = np.std(chunk, ddof=1)
                if S > 1e-14:
                    rs_chunk.append(R / S)
            if rs_chunk:
                rs_vals.append(np.mean(rs_chunk))

        if len(rs_vals) < 3:
            return 0.5

        lags_used = list(lag_range)[:len(rs_vals)]
        log_l     = np.log(lags_used)
        log_r     = np.log(rs_vals)
        h         = float(np.polyfit(log_l, log_r, 1)[0])

        # Clamp — values outside [0,1] are numerical artifacts
        return float(np.clip(h, 0.0, 1.0))

    except Exception as e:
        if DEBUG:
            print(f"      [Hurst ERR] {e}")
        return 0.5


# ─────────────────────────────────────────────────────────────
#  ADF — compatible with all statsmodels versions
# ─────────────────────────────────────────────────────────────
def _run_adf(data: np.ndarray) -> float:
    """ADF test with automatic API version detection."""
    if len(data) < 20:
        return 1.0
    # Try new API (statsmodels >= 0.14)
    try:
        return float(adfuller(data, maxlag=1,
                              autolag=None)[1])
    except TypeError:
        pass
    # Try old API (statsmodels < 0.14)
    try:
        return float(adfuller(data, maxlags=1,
                              autolag=None)[1])
    except TypeError:
        pass
    # Let statsmodels choose lag automatically
    try:
        return float(adfuller(data)[1])
    except Exception as e:
        if DEBUG:
            print(f"      [ADF ERR] {e}")
        return 1.0


# ─────────────────────────────────────────────────────────────
#  JOHANSEN — fixed critical value comparison
# ─────────────────────────────────────────────────────────────
def _run_johansen(log_s1: np.ndarray,
                  log_s2: np.ndarray) -> float:
    """
    Johansen test returning a meaningful pseudo p-value.

    Previous version used a ratio (trace/cv) which
    returned 1.0 whenever trace < cv because
    1 / (ratio < 1) > 1 then min(1.0, ...) = 1.0.

    This version uses direct threshold comparison
    against the 90%, 95%, 99% critical values.
    """
    try:
        n    = min(len(log_s1), len(log_s2))
        if n < 100:
            return 1.0

        data   = np.column_stack([log_s1[:n],
                                   log_s2[:n]])
        result = coint_johansen(data,
                                det_order=0,
                                k_ar_diff=1)

        trace = result.lr1[0]    # trace statistic, rank 0
        cv90  = result.cvt[0, 0] # 90% critical value
        cv95  = result.cvt[0, 1] # 95% critical value
        cv99  = result.cvt[0, 2] # 99% critical value

        if trace >= cv99:  return 0.01
        if trace >= cv95:  return 0.05
        if trace >= cv90:  return 0.10

        # Below 90% — scale between 0.10 and 1.0
        # based on how far below cv90 we are
        frac = trace / (cv90 + 1e-10)   # 0..1
        # frac=1 → p=0.10, frac=0 → p=1.0
        return float(np.clip(0.10 + (1.0 - frac) * 0.90,
                             0.10, 1.0))

    except Exception as e:
        if DEBUG:
            print(f"      [Johansen ERR] {e}")
        return 1.0


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
            print(f"  [ERR] Columns: {list(df.columns)}")
            return pd.Series(dtype=float)

        df[date_col] = pd.to_datetime(
            df[date_col], format='mixed', errors='coerce')
        df = (df.dropna(subset=[date_col])
                .sort_values(date_col)
                .set_index(date_col))

        s = df[close_col].astype(float).dropna()
        print(f"  [LOAD] {symbol}: {len(s):,} bars  "
              f"{s.index[0].date()} → {s.index[-1].date()}")
        return s

    except Exception as e:
        print(f"  [ERR] {fname}: {e}")
        return pd.Series(dtype=float)


def align_series(s1: pd.Series,
                 s2: pd.Series) -> tuple:
    c = pd.concat([s1, s2], axis=1).dropna()
    return c.iloc[:, 0], c.iloc[:, 1]


# ─────────────────────────────────────────────────────────────
#  HEDGE RATIO
# ─────────────────────────────────────────────────────────────
def estimate_hedge_ratio(s1: pd.Series,
                         s2: pd.Series) -> tuple:
    """OLS on log prices. Returns beta, alpha, spread."""
    try:
        y = np.log(s1.values.astype(float))
        x = np.log(s2.values.astype(float))
        y_s = pd.Series(y, index=s1.index)
        x_s = pd.Series(x, index=s2.index)
        m   = OLS(y_s, add_constant(x_s)).fit()
        a   = float(m.params.iloc[0])
        b   = float(m.params.iloc[1])
        sp  = y_s - b * x_s - a
        return b, a, sp
    except Exception as e:
        if DEBUG:
            print(f"      [hedge ERR] {e}")
        return 0.0, 0.0, pd.Series(dtype=float)


# ─────────────────────────────────────────────────────────────
#  HALF-LIFE
# ─────────────────────────────────────────────────────────────
def compute_half_life(spread: pd.Series) -> float:
    try:
        sp  = spread.dropna()
        if len(sp) < 30:
            return np.inf
        df  = pd.DataFrame({
            'lag'  : sp.shift(1),
            'delta': sp.diff()
        }).dropna()
        if len(df) < 20:
            return np.inf
        m   = OLS(df['delta'],
                  add_constant(df['lag'])).fit()
        lam = float(m.params.iloc[1])
        if lam >= 0:
            return np.inf
        return float(-np.log(2) / lam)
    except Exception as e:
        if DEBUG:
            print(f"      [HL ERR] {e}")
        return np.inf


# ─────────────────────────────────────────────────────────────
#  SINGLE WINDOW TEST
# ─────────────────────────────────────────────────────────────
def test_window(s1: pd.Series,
                s2: pd.Series,
                label: str = "") -> dict:
    """
    Full cointegration test on one window.
    Acceptance: (EG < threshold OR Johansen < threshold)
                AND ADF < threshold
                AND half-life in valid range
    Hurst is computed for scoring but NOT for rejection.
    """
    default = dict(
        cointegrated=False, eg_pval=1.0,
        adf_pval=1.0, johansen_pval=1.0,
        half_life=np.inf, hurst=0.5,
        hedge_ratio=0.0, alpha=0.0,
        spread_std=0.0, score=0.0)

    try:
        s1a, s2a = align_series(s1, s2)
        if len(s1a) < 100:
            return default

        # Engle-Granger
        log1 = np.log(s1a.values.astype(float))
        log2 = np.log(s2a.values.astype(float))
        _, eg_p, _ = coint(log1, log2)
        eg_p = float(eg_p)

        # Hedge ratio + spread
        beta, alpha, spread = estimate_hedge_ratio(
            s1a, s2a)
        if len(spread) < 50:
            return default

        # ADF on spread
        adf_p = _run_adf(spread.dropna().values)

        # Johansen
        j_p = _run_johansen(log1, log2)

        # Half-life
        hl = compute_half_life(spread)

        # Hurst on DIFFERENCES (fixed)
        hurst = _hurst_on_diffs(spread)

        # ── Acceptance logic ──────────────────────────
        # Primary: need statistical evidence of cointegration
        coint_evidence = (eg_p < EG_PVAL_MAX or
                          j_p  < 0.10)
        # Spread must be stationary
        spread_stat    = adf_p < ADF_PVAL_MAX
        # Mean reversion must be tradeable
        hl_ok          = (np.isfinite(hl) and
                          MIN_HALF_LIFE < hl <
                          MAX_HALF_LIFE)
        # NOTE: Hurst NOT used as rejection criterion
        # It failed to discriminate on your data
        # and is kept only for the score

        accepted = coint_evidence and spread_stat and hl_ok

        score = 0.0
        if accepted:
            score = (
                (1 - min(eg_p,  0.99)) * 0.30 +
                (1 - min(adf_p, 0.99)) * 0.30 +
                (1 - min(j_p,   0.99)) * 0.15 +
                (1 - min(hurst, 1.0))  * 0.10 +
                (1 - min(hl, MAX_HALF_LIFE) /
                 MAX_HALF_LIFE)        * 0.15
            )

        result = dict(
            cointegrated  = accepted,
            eg_pval       = eg_p,
            adf_pval      = adf_p,
            johansen_pval = j_p,
            half_life     = hl,
            hurst         = hurst,
            hedge_ratio   = beta,
            alpha         = alpha,
            spread_std    = float(spread.std()),
            score         = score,
        )

        if label:
            hl_s = f"{hl:.1f}" if np.isfinite(hl) else "∞"
            flag = "✓" if accepted else "✗"
            print(f"      {label}: "
                  f"EG={eg_p:.4f} ADF={adf_p:.4f} "
                  f"J={j_p:.4f} HL={hl_s} "
                  f"H={hurst:.3f}  {flag}")

        return result

    except Exception as e:
        print(f"      [test_window ERR] {e}")
        if DEBUG:
            traceback.print_exc()
        return default


# ─────────────────────────────────────────────────────────────
#  ROLLING WINDOW TEST
# ─────────────────────────────────────────────────────────────
def test_pair_rolling(sym1: str, sym2: str,
                      s1: pd.Series,
                      s2: pd.Series) -> dict:
    print(f"\n  ── {sym1} / {sym2} ──")

    s1a, s2a = align_series(s1, s2)
    n = len(s1a)

    if n < COINT_WINDOW + 200:
        print(f"    [SKIP] {n} bars < minimum")
        return {'valid': False,
                'symbol1': sym1, 'symbol2': sym2,
                'recent': {}}

    # Recent window
    recent = test_window(
        s1a.iloc[-COINT_WINDOW:],
        s2a.iloc[-COINT_WINDOW:],
        label="RECENT")
    recent_ok = recent['cointegrated']

    # Stability windows
    lookback  = min(n, 8736)
    step      = lookback // STABILITY_WINDOWS
    stab_pass = 0

    for w in range(STABILITY_WINDOWS):
        ei  = n - w * step
        si  = max(0, ei - COINT_WINDOW)
        if ei - si < 500:
            continue
        wr = test_window(
            s1a.iloc[si:ei],
            s2a.iloc[si:ei],
            label=f"STAB-{w+1}")
        if wr['cointegrated']:
            stab_pass += 1

    is_valid = (recent_ok and
                stab_pass >= STABILITY_MIN_PASS)

    flag = "✓✓ VALID" if is_valid else "✗ REJECTED"
    print(f"    Stab {stab_pass}/{STABILITY_WINDOWS}  "
          f"→ {flag}")

    out = {'valid'         : is_valid,
           'symbol1'       : sym1,
           'symbol2'       : sym2,
           'stability_pass': stab_pass,
           'n_bars'        : n,
           'recent'        : recent}

    if is_valid:
        out.update({k: recent[k] for k in [
            'eg_pval','adf_pval','johansen_pval',
            'half_life','hurst','hedge_ratio',
            'alpha','spread_std','score']})
        out['coint_window'] = COINT_WINDOW

    return out


# ─────────────────────────────────────────────────────────────
#  QUICK BACKTEST
# ─────────────────────────────────────────────────────────────
def quick_backtest(spread: pd.Series,
                   half_life: float,
                   entry_z: float = 2.0,
                   exit_z:  float = 0.0,
                   stop_z:  float = 3.5) -> dict:
    hl  = half_life if np.isfinite(half_life) else 20
    win = max(20, int(hl * 2))
    rm  = spread.rolling(win).mean()
    rs  = spread.rolling(win).std()
    z   = (spread - rm) / (rs + 1e-10)

    pos, trades, esp = 0, [], 0.0
    for i in range(win + 1, len(z)):
        zi, si = z.iloc[i], spread.iloc[i]
        if pos == 0:
            if   zi < -entry_z: pos, esp =  1, si
            elif zi >  entry_z: pos, esp = -1, si
        elif pos == 1:
            if zi >= exit_z or zi < -stop_z:
                trades.append(si - esp); pos = 0
        elif pos == -1:
            if zi <= exit_z or zi > stop_z:
                trades.append(esp - si); pos = 0

    if not trades:
        return {'n_trades': 0, 'win_rate': 0,
                'profit_factor': 0, 'sharpe': 0}

    t    = np.array(trades)
    w    = t[t > 0]; l = t[t < 0]
    gp   = w.sum() if len(w) > 0 else 0
    gl   = abs(l.sum()) if len(l) > 0 else 1e-10

    return {
        'n_trades'     : len(t),
        'win_rate'     : round(len(w)/len(t), 4),
        'profit_factor': round(gp/gl, 4),
        'sharpe'       : round(
            t.mean()/(t.std()+1e-10)*np.sqrt(252), 4),
    }


# ─────────────────────────────────────────────────────────────
#  PLOT
# ─────────────────────────────────────────────────────────────
def plot_pair(sym1, sym2, s1, s2, spread, result):
    try:
        hl  = result.get('half_life', 20)
        if not np.isfinite(hl): hl = 20
        win = max(20, int(hl * 2))
        rm  = spread.rolling(win).mean()
        rs  = spread.rolling(win).std()
        z   = (spread - rm) / (rs + 1e-10)
        n   = min(COINT_WINDOW, len(spread))

        fig, ax = plt.subplots(3, 1, figsize=(14, 10))
        fig.suptitle(
            f"{sym1}/{sym2}  "
            f"EG={result.get('eg_pval',1):.4f}  "
            f"ADF={result.get('adf_pval',1):.4f}  "
            f"HL={hl:.1f}  H={result.get('hurst',0.5):.3f}",
            fontsize=11)

        (s1/s1.iloc[0]).iloc[-n:].plot(ax=ax[0],
            label=sym1, color='steelblue')
        (s2/s2.iloc[0]).iloc[-n:].plot(ax=ax[0],
            label=sym2, color='darkorange')
        ax[0].set_title("Normalised Prices")
        ax[0].legend(); ax[0].grid(True, alpha=0.3)

        spread.iloc[-n:].plot(ax=ax[1],
            color='purple', lw=0.8)
        ax[1].axhline(spread.iloc[-n:].mean(),
                      color='k', ls='--', lw=1)
        ax[1].set_title("Spread")
        ax[1].grid(True, alpha=0.3)

        z.iloc[-n:].plot(ax=ax[2],
            color='darkgreen', lw=0.8)
        for lv, c in [(2,'r'),(-2,'r'),
                      (3.5,'darkred'),(-3.5,'darkred'),
                      (0,'k')]:
            ax[2].axhline(lv, color=c,
                          ls='--' if abs(lv)>0.1 else '-',
                          lw=0.8)
        ax[2].set_title("Z-Score")
        ax[2].grid(True, alpha=0.3)

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
def print_report(all_results):
    print(f"\n{'='*72}")
    print("  FULL DIAGNOSTIC REPORT")
    print(f"{'='*72}")
    print(f"  {'Pair':<18} {'EG':>7} {'ADF':>7} "
          f"{'J':>7} {'HL':>8} {'Hurst':>7} "
          f"{'Stab':>5}  Result")
    print(f"  {'':─<68}")

    for r in all_results:
        s1   = r.get('symbol1','?')
        s2   = r.get('symbol2','?')
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
        print(f"  {s1}/{s2:<12} "
              f"{eg:>7.4f} {adf:>7.4f} {j:>7.4f} "
              f"{hl_s:>8} {h:>7.3f} {stab:>5}  {res}")

    n_ok = sum(1 for r in all_results
               if r.get('valid', False))
    print(f"  {'':─<68}")
    print(f"  Valid: {n_ok} / {len(all_results)}")


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def run_research(timeframe: str = "H1"):
    sanity_check()

    print(f"{'='*55}")
    print(f"  PAIRS RESEARCH {VERSION}")
    print(f"  Window={COINT_WINDOW}  "
          f"EG<{EG_PVAL_MAX}  ADF<{ADF_PVAL_MAX}  "
          f"HL<{MAX_HALF_LIFE}  StabMin={STABILITY_MIN_PASS}")
    print(f"  NOTE: Hurst used for scoring only, "
          f"not rejection")
    print(f"{'='*55}")

    print("\n[1] Loading data...")
    pd_data = {}
    for sym in UNIVERSE:
        s = load_price_series(sym, timeframe)
        if len(s) >= MIN_BARS:
            pd_data[sym] = s

    avail = list(pd_data.keys())
    if len(avail) < 2:
        print("[ERR] Need >= 2 symbols"); return []

    pairs       = list(combinations(avail, 2))
    all_results = []
    valid_pairs = []

    print(f"\n[2] Testing {len(pairs)} pairs...")

    for sym1, sym2 in pairs:
        r = test_pair_rolling(
            sym1, sym2, pd_data[sym1], pd_data[sym2])
        all_results.append(r)

        if r.get('valid', False):
            rs1 = pd_data[sym1].iloc[-COINT_WINDOW:]
            rs2 = pd_data[sym2].iloc[-COINT_WINDOW:]
            rs1, rs2 = align_series(rs1, rs2)
            _, _, spread = estimate_hedge_ratio(rs1, rs2)
            bt = quick_backtest(spread, r['half_life'])
            r['backtest'] = bt
            print(f"    ★ BACKTEST: n={bt['n_trades']} "
                  f"WR={bt['win_rate']:.1%} "
                  f"PF={bt['profit_factor']:.2f} "
                  f"Sharpe={bt['sharpe']:.2f}")
            plot_pair(sym1, sym2, rs1, rs2, spread, r)
            valid_pairs.append(r)

    print_report(all_results)

    # Save
    valid_pairs.sort(
        key=lambda x: x.get('score', 0), reverse=True)

    def _clean(v):
        if isinstance(v, (np.floating, np.integer)):
            return float(v)
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v

    save = []
    for p in valid_pairs:
        row = {}
        for k, v in p.items():
            if k in ('recent', 'backtest'):
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        row[k2] = _clean(v2)
            else:
                row[k] = _clean(v)
        save.append(row)

    vp_path = os.path.join(OUTPUT_DIR,
                            "valid_pairs.json")
    with open(vp_path, 'w') as f:
        json.dump(save, f, indent=2, default=str)

    diag = []
    for r in all_results:
        rec = r.get('recent', {})
        diag.append({
            'symbol1': r.get('symbol1',''),
            'symbol2': r.get('symbol2',''),
            'valid'  : r.get('valid', False),
            'stab'   : r.get('stability_pass', 0),
            **{k: _clean(rec.get(k))
               for k in ['eg_pval','adf_pval',
                         'johansen_pval','half_life',
                         'hurst','score']}
        })

    dp = os.path.join(OUTPUT_DIR,
                      "diagnostic_report.json")
    with open(dp, 'w') as f:
        json.dump(diag, f, indent=2, default=str)

    print(f"\n[SAVED] {vp_path}  ({len(valid_pairs)} pairs)")
    print(f"[SAVED] {dp}")
    print(f"\n[DONE] {VERSION}  "
          f"Valid pairs: {len(valid_pairs)}/66")

    return valid_pairs


if __name__ == "__main__":
    run_research(timeframe="H1")
