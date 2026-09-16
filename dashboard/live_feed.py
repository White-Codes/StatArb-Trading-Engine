"""
live_feed.py
============
Fetches current and historical prices.
Primary source: your existing CSV files.
Secondary source: yfinance for recent bars.
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
            return pd.Series(dtype=float)

        df[date_col] = pd.to_datetime(
            df[date_col],
            format='mixed',
            errors='coerce')
        df = (df
              .dropna(subset=[date_col])
              .sort_values(date_col)
              .set_index(date_col))

        return (df[close_col]
                .astype(float)
                .dropna())

    except Exception as e:
        print(f"  [CSV ERR] {symbol}: {e}")
        return pd.Series(dtype=float)


def fetch_yfinance_series(
        symbol: str,
        n_bars: int = 500
        ) -> pd.Series:
    """Fetch recent bars from yfinance."""
    if not YFINANCE_OK:
        return pd.Series(dtype=float)

    ticker = YAHOO_MAP.get(symbol)
    if not ticker:
        return pd.Series(dtype=float)

    try:
        data = yf.download(
            ticker,
            period      = "60d",
            interval    = "1h",
            progress    = False,
            auto_adjust = True)

        if data.empty:
            return pd.Series(dtype=float)

        s = data['Close'].astype(float).dropna()
        s.index = (
            s.index.tz_localize(None)
            if s.index.tz
            else s.index)
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
    Combines CSV (long history) +
    yfinance (recent bars).
    """
    csv_s = load_csv_series(symbol, timeframe)
    yf_s  = fetch_yfinance_series(symbol)

    if len(csv_s) == 0 and len(yf_s) == 0:
        print(f"  [NO DATA] {symbol}")
        return pd.Series(dtype=float)

    if len(csv_s) == 0:
        return yf_s.iloc[-n_bars:]

    if len(yf_s) == 0:
        return csv_s.iloc[-n_bars:]

    # Combine: CSV history + yfinance recent
    csv_s.index = (
        csv_s.index.tz_localize(None)
        if csv_s.index.tz
        else csv_s.index)

    combined = pd.concat([csv_s, yf_s])
    combined = combined[
        ~combined.index.duplicated(
            keep='last')]
    combined = combined.sort_index()

    print(f"  [FEED] {symbol}: "
          f"{len(combined):,} bars  "
          f"({combined.index[0].date()} "
          f"→ {combined.index[-1].date()})")

    return combined.iloc[-n_bars:]


def load_all_price_data(
        symbols:   list,
        n_bars:    int = 5000,
        timeframe: str = "H1"
        ) -> dict:
    """Load price data for all symbols."""
    print("\n[FEED] Loading price data...")
    price_data = {}
    for sym in symbols:
        s = fetch_recent_history(
            sym, n_bars, timeframe)
        if len(s) >= 500:
            price_data[sym] = s
        else:
            print(f"  [SKIP] {sym}: "
                  f"only {len(s)} bars")
    print(f"  Loaded {len(price_data)} "
          f"symbols")
    return price_data
