"""
dashboard/live_feed.py
======================
Fetches current and historical prices.
Fix: yfinance returns MultiIndex DataFrame.
     Must extract single column correctly.
"""

import numpy as np
import pandas as pd
import os

try:
    import yfinance as yf
    YFINANCE_OK = True
except ImportError:
    YFINANCE_OK = False

DATA_DIR = "."

YAHOO_MAP = {
    "EURUSD" : "EURUSD=X",
    "GBPUSD" : "GBPUSD=X",
    "AUDUSD" : "AUDUSD=X",
    "NZDUSD" : "NZDUSD=X",
    "USDCAD" : "USDCAD=X",
    "USDCHF" : "USDCHF=X",
    "EURGBP" : "EURGBP=X",
    "EURAUD" : "EURAUD=X",
    "GBPAUD" : "GBPAUD=X",
    "AUDNZD" : "AUDNZD=X",
    "EURCAD" : "EURCAD=X",
    "GBPCAD" : "GBPCAD=X",
}


def _to_clean_series(
        data,
        name: str = ""
        ) -> pd.Series:
    """
    Convert any DataFrame or Series from
    yfinance or CSV into a clean
    single-column pd.Series with
    DatetimeIndex.

    yfinance >= 0.2 returns a DataFrame
    with MultiIndex columns like:
      ('Close', 'EURUSD=X')
    or sometimes just:
      ('Close',)

    This function handles all cases.
    """
    # Already a clean Series
    if isinstance(data, pd.Series):
        s = data.copy()
        # Remove timezone if present
        if hasattr(s.index, 'tz') and (
                s.index.tz is not None):
            s.index = s.index.tz_localize(
                None)
        return s.astype(float).dropna()

    # DataFrame — need to extract one column
    if isinstance(data, pd.DataFrame):
        # Flatten MultiIndex columns
        if isinstance(data.columns,
                      pd.MultiIndex):
            data.columns = [
                '_'.join(
                    str(c) for c in col
                ).strip()
                for col in data.columns]

        # Find Close column
        close_col = None
        for col in data.columns:
            col_lower = str(col).lower()
            if 'close' in col_lower:
                close_col = col
                break

        if close_col is None:
            # Try first column
            close_col = data.columns[0]

        s = data[close_col].copy()
        if hasattr(s.index, 'tz') and (
                s.index.tz is not None):
            s.index = s.index.tz_localize(
                None)
        return s.astype(float).dropna()

    return pd.Series(dtype=float)


def load_csv_series(
        symbol:    str,
        timeframe: str = "H1"
        ) -> pd.Series:
    """Load price series from local CSV."""
    fname = os.path.join(
        DATA_DIR,
        f"{symbol}_{timeframe}.csv")

    if not os.path.exists(fname):
        return pd.Series(dtype=float)

    try:
        df = pd.read_csv(fname)
        df.columns = [
            c.strip().lower()
            for c in df.columns]

        date_col = next((c for c in [
            'gmt time', 'gmt_time',
            'timestamp', 'time',
            'date', 'datetime']
            if c in df.columns), None)

        close_col = next((c for c in [
            'close', 'c', 'price']
            if c in df.columns), None)

        if not date_col or not close_col:
            print(f"  [CSV ERR] {symbol}: "
                  f"columns not found. "
                  f"Have: {list(df.columns)}")
            return pd.Series(dtype=float)

        df[date_col] = pd.to_datetime(
            df[date_col],
            format='mixed',
            errors='coerce')
        df = (df
              .dropna(subset=[date_col])
              .sort_values(date_col)
              .set_index(date_col))

        s = df[close_col].astype(
            float).dropna()

        # Remove timezone
        if hasattr(s.index, 'tz') and (
                s.index.tz is not None):
            s.index = s.index.tz_localize(
                None)

        return s

    except Exception as e:
        print(f"  [CSV ERR] {symbol}: {e}")
        return pd.Series(dtype=float)


def fetch_yfinance_series(
        symbol: str,
        n_bars: int = 500
        ) -> pd.Series:
    """
    Fetch recent bars from yfinance.
    Handles MultiIndex columns correctly.
    """
    if not YFINANCE_OK:
        return pd.Series(dtype=float)

    ticker = YAHOO_MAP.get(symbol)
    if not ticker:
        return pd.Series(dtype=float)

    try:
        raw = yf.download(
            ticker,
            period      = "60d",
            interval    = "1h",
            progress    = False,
            auto_adjust = True)

        if raw is None or (
                hasattr(raw, 'empty') and
                raw.empty):
            return pd.Series(dtype=float)

        # Use helper to extract clean Series
        s = _to_clean_series(raw, symbol)

        if len(s) == 0:
            print(f"  [YF WARN] {symbol}: "
                  f"0 bars after cleaning. "
                  f"Raw type: {type(raw)}, "
                  f"Raw shape: "
                  f"{raw.shape if hasattr(raw, 'shape') else 'N/A'}, "
                  f"Raw cols: "
                  f"{list(raw.columns) if hasattr(raw, 'columns') else 'N/A'}")
            return pd.Series(dtype=float)

        return s.iloc[-n_bars:]

    except Exception as e:
        print(f"  [YF ERR] {symbol}: {e}")
        return pd.Series(dtype=float)


def fetch_recent_history(
        symbol:    str,
        n_bars:    int = 5000,
        timeframe: str = "H1"
        ) -> pd.Series:
    """
    Load price history.
    CSV (long history) + yfinance (recent).
    Returns a clean single pd.Series.
    """
    csv_s = load_csv_series(
        symbol, timeframe)
    yf_s  = fetch_yfinance_series(symbol)

    # Validate both are proper Series
    assert isinstance(csv_s, pd.Series), (
        f"csv_s is {type(csv_s)}")
    assert isinstance(yf_s, pd.Series), (
        f"yf_s is {type(yf_s)}")

    if len(csv_s) == 0 and len(yf_s) == 0:
        print(f"  [NO DATA] {symbol}")
        return pd.Series(dtype=float)

    if len(csv_s) == 0:
        return yf_s.iloc[-n_bars:]

    if len(yf_s) == 0:
        return csv_s.iloc[-n_bars:]

    # Combine
    try:
        combined = pd.concat([csv_s, yf_s])
        combined = combined[
            ~combined.index.duplicated(
                keep='last')]
        combined = combined.sort_index()
    except Exception as e:
        print(f"  [COMBINE ERR] {symbol}: "
              f"{e} — using CSV only")
        combined = csv_s

    print(f"  [FEED] {symbol}: "
          f"{len(combined):,} bars  "
          f"({combined.index[0].date()} "
          f"-> "
          f"{combined.index[-1].date()})")

    return combined.iloc[-n_bars:]


def load_all_price_data(
        symbols:   list,
        n_bars:    int = 5000,
        timeframe: str = "H1"
        ) -> dict:
    """
    Load price data for all symbols.
    Returns dict of {symbol: pd.Series}.
    All Series are guaranteed to be
    simple single-column with DatetimeIndex.
    """
    print("\n[FEED] Loading price data...")
    price_data = {}

    for sym in symbols:
        s = fetch_recent_history(
            sym, n_bars, timeframe)

        # Final validation
        if not isinstance(s, pd.Series):
            print(f"  [SKIP] {sym}: "
                  f"not a Series "
                  f"({type(s)})")
            continue

        if len(s) < 500:
            print(f"  [SKIP] {sym}: "
                  f"only {len(s)} bars")
            continue

        # Confirm it is truly 1-dimensional
        if hasattr(s, 'ndim') and s.ndim != 1:
            print(f"  [SKIP] {sym}: "
                  f"ndim={s.ndim} "
                  f"(expected 1)")
            continue

        price_data[sym] = s

    print(f"  Loaded {len(price_data)} "
          f"symbols")

    # Debug: show structure of first series
    if price_data:
        first_sym = list(price_data.keys())[0]
        first_s   = price_data[first_sym]
        print(f"  [DEBUG] {first_sym}: "
              f"type={type(first_s).__name__} "
              f"ndim={first_s.ndim} "
              f"len={len(first_s)} "
              f"dtype={first_s.dtype}")

    return price_data
