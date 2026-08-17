"""
kalman_filter.py
================
Adaptive hedge ratio estimation using Kalman Filter.

Why Kalman instead of OLS:
  - OLS assumes hedge ratio is CONSTANT over time
  - In reality the relationship drifts slowly
  - Kalman updates the ratio every bar
  - Result: spread is always correctly constructed
  - Fewer false signals from stale hedge ratios

The model:
  Observation:  y_t = β_t * x_t + α_t + ε_t
  State update: β_t = β_{t-1} + η_t
                (hedge ratio performs a random walk)

Parameters:
  delta   — how fast the ratio is allowed to drift
            small delta = slow adaptation (more stable)
            large delta = fast adaptation (more responsive)
            typical range: 1e-5 to 1e-3
  vt      — observation noise variance
"""

import numpy as np
import pandas as pd


class KalmanHedgeFilter:
    """
    Online Kalman filter for adaptive hedge ratio.
    
    Usage:
        kf = KalmanHedgeFilter(delta=1e-4)
        for price_a, price_b in zip(series_a, series_b):
            beta, alpha, spread = kf.update(price_a, price_b)
    """

    def __init__(self, delta: float = 1e-4,
                 vt: float = 1e-3):
        """
        Parameters
        ----------
        delta : float
            State transition noise. Controls how fast
            the hedge ratio adapts.
            1e-5 = very slow, 1e-3 = fast
        vt : float
            Observation noise variance.
        """
        self.delta  = delta
        self.vt     = vt

        # State vector: [beta, alpha]
        # beta  = hedge ratio
        # alpha = intercept
        self.theta  = np.zeros(2)

        # State covariance matrix (2x2)
        self.P      = np.eye(2) * 1.0

        # Process noise covariance
        self.Q      = delta / (1 - delta) * np.eye(2)

        # History for analysis
        self.beta_history   = []
        self.alpha_history  = []
        self.spread_history = []
        self.e_history      = []   # innovations

        self._initialised   = False
        self._n_updates     = 0

    def update(self, price_a: float,
               price_b: float) -> tuple:
        """
        Process one new observation.
        
        Parameters
        ----------
        price_a : float  — price of instrument A
        price_b : float  — price of instrument B
        
        Returns
        -------
        beta   : float  — current hedge ratio
        alpha  : float  — current intercept
        spread : float  — current spread value
        """
        # Observation vector: [price_b, 1]
        # (we regress price_a on price_b + constant)
        F = np.array([price_b, 1.0])

        # Prediction step
        # State prediction: theta = theta (random walk)
        # Covariance prediction: P = P + Q
        P_pred = self.P + self.Q

        # Innovation (prediction error)
        y_pred  = F @ self.theta          # predicted price_a
        e       = price_a - y_pred         # innovation

        # Innovation variance
        S       = F @ P_pred @ F.T + self.vt

        # Kalman gain
        K       = P_pred @ F.T / S

        # Update step
        self.theta = self.theta + K * e
        self.P     = (np.eye(2) - np.outer(K, F)) @ P_pred

        beta   = self.theta[0]
        alpha  = self.theta[1]
        spread = price_a - beta * price_b - alpha

        # Store history
        self.beta_history.append(beta)
        self.alpha_history.append(alpha)
        self.spread_history.append(spread)
        self.e_history.append(e)

        self._n_updates += 1

        return beta, alpha, spread

    def get_zscore(self, window: int = None) -> float:
        """
        Compute current z-score of spread.
        Uses rolling window = 2 * half_life if not specified.
        Default window: last 50 observations.
        """
        if len(self.spread_history) < 20:
            return 0.0

        w = window if window else min(
            50, len(self.spread_history))
        recent = np.array(self.spread_history[-w:])
        mean   = recent.mean()
        std    = recent.std()

        if std < 1e-10:
            return 0.0

        return (self.spread_history[-1] - mean) / std

    def get_state(self) -> dict:
        """Return current filter state."""
        return {
            'beta'   : self.theta[0],
            'alpha'  : self.theta[1],
            'spread' : self.spread_history[-1]
                       if self.spread_history else 0.0,
            'n_obs'  : self._n_updates,
        }

    def to_series(self) -> pd.DataFrame:
        """Return full history as DataFrame."""
        n = len(self.beta_history)
        return pd.DataFrame({
            'beta'  : self.beta_history,
            'alpha' : self.alpha_history,
            'spread': self.spread_history,
            'innov' : self.e_history,
        })


# ─────────────────────────────────────────────
#  OFFLINE BATCH VERSION
#  For research / backtesting with full history
# ─────────────────────────────────────────────
def kalman_filter_batch(s1: pd.Series,
                        s2: pd.Series,
                        delta: float = 1e-4,
                        vt: float    = 1e-3
                        ) -> pd.DataFrame:
    """
    Run Kalman filter over entire history.
    Returns DataFrame with all state variables.
    """
    kf = KalmanHedgeFilter(delta=delta, vt=vt)

    betas   = []
    alphas  = []
    spreads = []

    for pa, pb in zip(s1.values, s2.values):
        beta, alpha, spread = kf.update(pa, pb)
        betas.append(beta)
        alphas.append(alpha)
        spreads.append(spread)

    result = pd.DataFrame({
        'beta'  : betas,
        'alpha' : alphas,
        'spread': spreads,
    }, index=s1.index)

    return result


# ─────────────────────────────────────────────
#  KALMAN PARAMETER OPTIMISATION
#  Grid search for best delta parameter
# ─────────────────────────────────────────────
def optimise_delta(s1: pd.Series,
                   s2: pd.Series,
                   deltas: list = None) -> dict:
    """
    Find the delta value that produces the most
    mean-reverting spread (lowest Hurst exponent).
    """
    from pairs_research import (compute_half_life,
                                  compute_hurst)

    if deltas is None:
        deltas = [1e-6, 1e-5, 5e-5,
                  1e-4, 5e-4, 1e-3, 1e-2]

    results = []
    for d in deltas:
        kf_df = kalman_filter_batch(s1, s2, delta=d)
        spread = pd.Series(kf_df['spread'].values,
                           index=s1.index)

        hl    = compute_half_life(spread)
        hurst = compute_hurst(spread)

        results.append({
            'delta'    : d,
            'half_life': hl,
            'hurst'    : hurst,
            'valid'    : (2 < hl < 30 and hurst < 0.5),
        })
        print(f"  delta={d:.0e}  "
              f"HL={hl:.1f}  Hurst={hurst:.3f}")

    # Best: lowest Hurst among valid configs
    valid = [r for r in results if r['valid']]
    if not valid:
        valid = results   # fallback

    best = min(valid, key=lambda x: x['hurst'])
    print(f"\n  Best delta: {best['delta']:.0e}  "
          f"HL={best['half_life']:.1f}  "
          f"Hurst={best['hurst']:.3f}")
    return best
