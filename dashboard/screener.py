"""
screener.py
===========
Full pipeline engine.
Runs research -> backtest -> signals -> sizing.
Returns structured DataFrames for Sheets.

Fix: KeyError 'Score' in run_screener()
     - Consistent column naming throughout
     - Empty rows guard before sort_values
     - Robust error handling per pair
"""

import numpy as np
import pandas as pd
import sys, os
from datetime import datetime, timezone
from itertools import combinations

sys.path.insert(0, os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__))))

from pairs_research import (
    align_series,
    compute_half_life,
    test_window,
)
from pairs_backtest import (
    estimate_ols,
    apply_fixed_spread,
    compute_test_zscore,
    backtest_pair,
    compute_stats,
)
from dashboard.live_feed import (
    load_all_price_data,
)

UNIVERSE = [
    "EURUSD", "GBPUSD", "AUDUSD",
    "NZDUSD", "USDCAD", "USDCHF",
    "EURGBP", "EURAUD", "GBPAUD",
    "AUDNZD", "EURCAD", "GBPCAD",
]

TRAIN_BARS = 2016
TEST_BARS  = 1008
ENTRY_Z    = 2.0
EXIT_Z     = 0.5
STOP_Z     = 3.5
MIN_BT_PF  = 1.0


# ─────────────────────────────────────────────
#  STEP 1: SCREENER
# ─────────────────────────────────────────────
def run_screener(
        price_data: dict
        ) -> pd.DataFrame:
    """
    Test all pair combinations for
    cointegration. Returns Sheet 1 data.
    """
    avail = list(price_data.keys())
    pairs = list(combinations(avail, 2))
    print(f"\n[1-SCREENER] Testing "
          f"{len(pairs)} pairs...")

    rows = []
    for sym1, sym2 in pairs:
        try:
            s1, s2 = align_series(
                price_data[sym1],
                price_data[sym2])

            if len(s1) < 500:
                continue

            n = len(s1)
            w = min(TRAIN_BARS, n)

            result = test_window(
                s1.iloc[-w:],
                s2.iloc[-w:])

            hl    = result.get(
                'half_life', np.inf)
            score = result.get('score', 0.0)

            rows.append({
                'Pair'         : f"{sym1}/{sym2}",
                'Symbol1'      : sym1,
                'Symbol2'      : sym2,
                'EG_pval'      : round(
                    float(result.get(
                        'eg_pval', 1.0)), 4),
                'ADF_pval'     : round(
                    float(result.get(
                        'adf_pval', 1.0)), 4),
                'Johansen_pval': round(
                    float(result.get(
                        'johansen_pval', 1.0)),
                    4),
                'Half_Life'    : (
                    round(float(hl), 1)
                    if np.isfinite(hl)
                    else 999.0),
                'Hurst'        : round(
                    float(result.get(
                        'hurst', 0.5)), 3),
                'Hedge_Ratio'  : round(
                    float(result.get(
                        'hedge_ratio', 0.0)),
                    4),
                'Valid'        : (
                    "YES"
                    if result.get(
                        'cointegrated', False)
                    else "NO"),
                'Score'        : round(
                    float(score), 4),
            })

        except Exception as e:
            print(f"  [ERR] {sym1}/{sym2}: "
                  f"{e}")
            # Add a failed row so we don't
            # lose the pair silently
            rows.append({
                'Pair'         : f"{sym1}/{sym2}",
                'Symbol1'      : sym1,
                'Symbol2'      : sym2,
                'EG_pval'      : 1.0,
                'ADF_pval'     : 1.0,
                'Johansen_pval': 1.0,
                'Half_Life'    : 999.0,
                'Hurst'        : 0.5,
                'Hedge_Ratio'  : 0.0,
                'Valid'        : "NO",
                'Score'        : 0.0,
            })

    if not rows:
        print("  [WARN] No pairs processed")
        return pd.DataFrame(columns=[
            'Pair', 'Symbol1', 'Symbol2',
            'EG_pval', 'ADF_pval',
            'Johansen_pval', 'Half_Life',
            'Hurst', 'Hedge_Ratio',
            'Valid', 'Score'])

    df = pd.DataFrame(rows)

    # Verify column exists before sorting
    if 'Score' not in df.columns:
        print(f"  [WARN] Score column missing. "
              f"Columns: {df.columns.tolist()}")
        df['Score'] = 0.0

    df = df.sort_values(
        'Score', ascending=False
    ).reset_index(drop=True)

    n_valid = (df['Valid'] == 'YES').sum()
    print(f"  Done: {n_valid} valid "
          f"/ {len(df)} total")
    return df


# ─────────────────────────────────────────────
#  STEP 2: BACKTEST
# ─────────────────────────────────────────────
def run_backtests(
        screener_df: pd.DataFrame,
        price_data:  dict
        ) -> pd.DataFrame:
    """
    Backtest all valid pairs.
    Returns Sheet 2 data.
    """
    if screener_df.empty:
        print("\n[2-BACKTEST] No valid pairs "
              "to backtest")
        return pd.DataFrame()

    valid = screener_df[
        screener_df['Valid'] == 'YES']

    if valid.empty:
        print("\n[2-BACKTEST] No valid pairs "
              "found in screener")
        return pd.DataFrame()

    print(f"\n[2-BACKTEST] Running "
          f"{len(valid)} pairs...")

    rows = []
    for _, row in valid.iterrows():
        sym1 = row['Symbol1']
        sym2 = row['Symbol2']
        pair = row['Pair']

        if (sym1 not in price_data or
                sym2 not in price_data):
            print(f"  [SKIP] {pair}: "
                  f"price data missing")
            continue

        try:
            s1, s2 = align_series(
                price_data[sym1],
                price_data[sym2])

            min_needed = TRAIN_BARS + TEST_BARS
            if len(s1) < min_needed:
                print(f"  [SKIP] {pair}: "
                      f"need {min_needed} bars, "
                      f"have {len(s1)}")
                continue

            trades, equity, _ = backtest_pair(
                sym1       = sym1,
                sym2       = sym2,
                s1         = s1,
                s2         = s2,
                entry_z    = ENTRY_Z,
                exit_z     = EXIT_Z,
                stop_z     = STOP_Z,
                train_bars = TRAIN_BARS,
                test_bars  = TEST_BARS,
            )

            if not trades:
                print(f"  {pair}: "
                      f"no trades generated")
                continue

            st = compute_stats(trades, equity)
            pf = st.get('profit_factor', 0.0)

            rows.append({
                'Pair'         : pair,
                'Symbol1'      : sym1,
                'Symbol2'      : sym2,
                'Trades'       : st.get(
                    'n_trades', 0),
                'Win_Rate'     : round(
                    float(st.get(
                        'win_rate', 0.0)), 4),
                'Win_Rate_Pct' : (
                    f"{st.get('win_rate',0):.1%}"),
                'Profit_Factor': round(
                    float(pf), 4),
                'Sharpe'       : round(
                    float(st.get(
                        'sharpe', 0.0)), 4),
                'Max_DD'       : round(
                    float(st.get(
                        'max_dd', 0.0)), 4),
                'Max_DD_Pct'   : (
                    f"{st.get('max_dd',0):.1%}"),
                'Avg_Hold_Bars': round(
                    float(st.get(
                        'avg_hold', 0.0)), 1),
                'Total_PnL'    : round(
                    float(st.get(
                        'total_pnl_n', 0.0)),
                    4),
                'Profitable'   : (
                    "YES"
                    if pf >= MIN_BT_PF
                    else "NO"),
                'Hedge_Ratio'  : round(
                    float(row.get(
                        'Hedge_Ratio', 0.0)),
                    4),
                'Half_Life'    : float(
                    row.get('Half_Life', 0)),
            })
            print(f"  {pair}: "
                  f"PF={pf:.2f}  "
                  f"WR={st.get('win_rate',0):.1%}"
                  f"  {'PASS' if pf>=MIN_BT_PF else 'FAIL'}")

        except Exception as e:
            print(f"  [ERR] {pair}: {e}")

    if not rows:
        print("  [WARN] No backtest results")
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values(
        'Profit_Factor', ascending=False
    ).reset_index(drop=True)

    return df


# ─────────────────────────────────────────────
#  STEP 3: LIVE SIGNALS
# ─────────────────────────────────────────────
def compute_live_signals(
        backtest_df: pd.DataFrame,
        price_data:  dict
        ) -> pd.DataFrame:
    """
    Compute current z-scores for all
    profitable pairs. Returns Sheet 3 data.
    """
    if backtest_df.empty:
        print("\n[3-SIGNALS] No backtest "
              "results to compute signals for")
        return pd.DataFrame()

    profitable = backtest_df[
        backtest_df['Profitable'] == 'YES']

    if profitable.empty:
        print("\n[3-SIGNALS] No profitable "
              "pairs found")
        return pd.DataFrame()

    print(f"\n[3-SIGNALS] Computing signals "
          f"for {len(profitable)} pairs...")

    rows = []
    now  = datetime.now(timezone.utc)

    for _, row in profitable.iterrows():
        sym1 = row['Symbol1']
        sym2 = row['Symbol2']
        pair = row['Pair']

        if (sym1 not in price_data or
                sym2 not in price_data):
            continue

        try:
            s1, s2 = align_series(
                price_data[sym1],
                price_data[sym2])

            if len(s1) < TRAIN_BARS + 10:
                continue

            # OLS on last TRAIN_BARS
            t_s1 = s1.iloc[
                   -TRAIN_BARS:].values
            t_s2 = s2.iloc[
                   -TRAIN_BARS:].values

            beta, alpha, train_std = (
                estimate_ols(t_s1, t_s2))

            # Training spread stats
            log1_tr  = np.log(t_s1)
            log2_tr  = np.log(t_s2)
            train_sp = (log1_tr -
                        beta * log2_tr -
                        alpha)
            mu  = float(np.mean(train_sp))
            sig = float(np.std(
                train_sp, ddof=1))
            if sig < 1e-10:
                sig = 1.0

            # Current spread + z-score
            p1_now = float(s1.iloc[-1])
            p2_now = float(s2.iloc[-1])
            sp_now = (np.log(p1_now) -
                      beta * np.log(p2_now) -
                      alpha)
            z_now  = (sp_now - mu) / sig

            # Previous bar for direction
            p1_prev = float(s1.iloc[-2])
            p2_prev = float(s2.iloc[-2])
            sp_prev = (np.log(p1_prev) -
                       beta * np.log(p2_prev) -
                       alpha)
            z_prev   = (sp_prev - mu) / sig
            z_change = round(
                float(z_now - z_prev), 3)

            z_now = round(float(z_now), 3)

            # Signal logic
            if z_now <= -ENTRY_Z:
                signal  = "LONG"
                action  = (f"BUY {sym1} / "
                           f"SELL {sym2}")
                urgency = "ENTER NOW"
            elif z_now >= ENTRY_Z:
                signal  = "SHORT"
                action  = (f"SELL {sym1} / "
                           f"BUY {sym2}")
                urgency = "ENTER NOW"
            elif z_now <= -(ENTRY_Z - 0.3):
                signal  = "WATCH LONG"
                action  = "Approaching entry"
                urgency = "WATCH"
            elif z_now >= (ENTRY_Z - 0.3):
                signal  = "WATCH SHORT"
                action  = "Approaching entry"
                urgency = "WATCH"
            else:
                signal  = "FLAT"
                action  = "No trade"
                urgency = "WAIT"

            updated = now.strftime(
                '%Y-%m-%d %H:%M')

            rows.append({
                'Pair'        : pair,
                'Symbol1'     : sym1,
                'Symbol2'     : sym2,
                'Z_Score'     : z_now,
                'Z_Change'    : z_change,
                'Signal'      : signal,
                'Action'      : action,
                'Urgency'     : urgency,
                'Entry_Z'     : ENTRY_Z,
                'Exit_Z'      : EXIT_Z,
                'Stop_Z'      : STOP_Z,
                'Price1'      : round(
                    p1_now, 5),
                'Price2'      : round(
                    p2_now, 5),
                'Beta'        : round(
                    float(beta), 4),
                'Spread_Now'  : round(
                    float(sp_now), 6),
                'Spread_Std'  : round(
                    float(train_std), 6),
                'PF_Backtest' : float(
                    row.get(
                        'Profit_Factor', 0)),
                'WR_Backtest' : row.get(
                    'Win_Rate_Pct', '0%'),
                'Sharpe'      : float(
                    row.get('Sharpe', 0)),
                'Updated_UTC' : updated,
            })

            print(f"  {pair}: "
                  f"z={z_now:+.3f}  "
                  f"-> {signal}")

        except Exception as e:
            print(f"  [ERR] {pair}: {e}")

    if not rows:
        print("  [WARN] No signals computed")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Sort: active signals first
    order = {
        "LONG"        : 0,
        "SHORT"       : 1,
        "WATCH LONG"  : 2,
        "WATCH SHORT" : 3,
        "FLAT"        : 4,
    }
    df['_sort'] = df['Signal'].map(
        order).fillna(5)
    df = (df.sort_values('_sort')
            .drop('_sort', axis=1)
            .reset_index(drop=True))

    return df


# ─────────────────────────────────────────────
#  STEP 4: RISK SIZING
# ─────────────────────────────────────────────
def compute_risk_sizing(
        signals_df:      pd.DataFrame,
        account_balance: float = 10000.0,
        risk_pct:        float = 0.01,
        ) -> pd.DataFrame:
    """
    Compute lot sizes for active signals.
    Returns Sheet 4 data.
    """
    if signals_df.empty:
        print("\n[4-SIZING] No signals "
              "to size")
        return pd.DataFrame()

    active = signals_df[
        signals_df['Signal'].isin(
            ['LONG', 'SHORT'])].copy()

    if active.empty:
        print("\n[4-SIZING] No active "
              "signals (all FLAT/WATCH)")
        return pd.DataFrame()

    print(f"\n[4-SIZING] Sizing "
          f"{len(active)} active signals...")

    risk_amount = account_balance * risk_pct
    rows = []

    for _, row in active.iterrows():
        try:
            beta   = float(row['Beta'])
            z_now  = float(row['Z_Score'])
            std    = float(row['Spread_Std'])
            signal = row['Signal']

            # Stop distance
            stop_z = (
                -STOP_Z
                if signal == 'LONG'
                else STOP_Z)
            stop_dist_z = abs(stop_z - z_now)
            stop_dist   = stop_dist_z * std
            pips_risk   = max(
                stop_dist * 10000, 1.0)

            # Target (1.5 sigma move)
            target_pips = std * 1.5 * 10000
            rr_ratio    = round(
                target_pips / pips_risk, 2)

            # Lot sizes
            pip_value = 10.0
            lot1 = risk_amount / (
                pips_risk * pip_value)
            lot1 = round(float(
                np.clip(lot1, 0.01, 5.0)), 2)
            lot2 = round(float(
                np.clip(
                    lot1 * abs(beta),
                    0.01, 5.0)), 2)

            if signal == 'LONG':
                leg1_dir = 'BUY'
                leg2_dir = 'SELL'
            else:
                leg1_dir = 'SELL'
                leg2_dir = 'BUY'

            rows.append({
                'Pair'           : row['Pair'],
                'Signal'         : signal,
                'Z_Score'        : z_now,
                'Account_Bal'    : account_balance,
                'Risk_Pct'       : (
                    f"{risk_pct:.1%}"),
                'Risk_Amount_USD': round(
                    risk_amount, 2),
                'Leg1_Symbol'    : row[
                    'Symbol1'],
                'Leg1_Direction' : leg1_dir,
                'Leg1_Lots'      : lot1,
                'Leg2_Symbol'    : row[
                    'Symbol2'],
                'Leg2_Direction' : leg2_dir,
                'Leg2_Lots'      : lot2,
                'Stop_Pips'      : round(
                    pips_risk, 1),
                'Target_Pips'    : round(
                    target_pips, 1),
                'RR_Ratio'       : rr_ratio,
                'Beta'           : round(
                    beta, 4),
            })

            print(f"  {row['Pair']}: "
                  f"{signal}  "
                  f"Leg1={leg1_dir} {lot1}L  "
                  f"Leg2={leg2_dir} {lot2}L  "
                  f"RR={rr_ratio}")

        except Exception as e:
            print(f"  [ERR] {row['Pair']}: "
                  f"{e}")

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
#  STEP 5: SUMMARY
# ─────────────────────────────────────────────
def build_summary(
        screener_df:     pd.DataFrame,
        backtest_df:     pd.DataFrame,
        signals_df:      pd.DataFrame,
        sizing_df:       pd.DataFrame,
        account_balance: float,
        risk_pct:        float,
        ) -> pd.DataFrame:
    """
    Build KPI summary table.
    Returns Sheet 5 data.
    """
    now = datetime.now(
        timezone.utc).strftime(
        '%Y-%m-%d %H:%M UTC')

    n_tested = len(screener_df)
    n_valid  = (
        int((screener_df['Valid'] == 'YES')
            .sum())
        if not screener_df.empty else 0)
    n_profit = (
        int((backtest_df['Profitable'] == 'YES')
            .sum())
        if not backtest_df.empty else 0)
    n_active = (
        len(signals_df[
            signals_df['Signal'].isin(
                ['LONG', 'SHORT'])])
        if not signals_df.empty else 0)

    rows = [
        ["STATARB LIVE DASHBOARD", ""],
        ["Last Updated",       now],
        ["", ""],
        ["-- PIPELINE --",     ""],
        ["Pairs Tested",       n_tested],
        ["Valid (Coint)",      n_valid],
        ["Profitable OOS",     n_profit],
        ["Active Signals",     n_active],
        ["", ""],
        ["-- ACCOUNT --",      ""],
        ["Balance",
         f"${account_balance:,.2f}"],
        ["Risk Per Trade",
         f"{risk_pct:.1%}"],
        ["Risk Amount",
         f"${account_balance*risk_pct:,.2f}"],
        ["", ""],
    ]

    # Backtest averages
    if not backtest_df.empty:
        prof = backtest_df[
            backtest_df['Profitable'] == 'YES']
        if not prof.empty:
            rows += [
                ["-- BACKTEST AVERAGES --",
                 "(profitable pairs only)"],
                ["Avg Win Rate",
                 f"{prof['Win_Rate'].mean():.1%}"],
                ["Avg Profit Factor",
                 f"{prof['Profit_Factor'].mean():.2f}"],
                ["Avg Sharpe",
                 f"{prof['Sharpe'].mean():.2f}"],
                ["Avg Max DD",
                 f"{prof['Max_DD'].mean():.1%}"],
                ["Avg Hold (H1 bars)",
                 f"{prof['Avg_Hold_Bars'].mean():.0f}"],
                ["", ""],
            ]

    # Active signals detail
    if not signals_df.empty:
        active = signals_df[
            signals_df['Signal'].isin(
                ['LONG', 'SHORT'])]
        if not active.empty:
            rows.append(
                ["-- ACTIVE SIGNALS --", ""])
            for _, r in active.iterrows():
                rows.append([
                    r['Pair'],
                    f"{r['Signal']}  "
                    f"z={r['Z_Score']:+.3f}  "
                    f"-> {r['Action']}"
                ])

    return pd.DataFrame(
        rows, columns=['Metric', 'Value'])


# ─────────────────────────────────────────────
#  FULL PIPELINE
# ─────────────────────────────────────────────
def run_full_pipeline(
        account_balance: float = 10000.0,
        risk_pct:        float = 0.01,
        symbols:         list  = None,
        ) -> dict:
    """
    Run complete pipeline.
    Returns all DataFrames for Google Sheets.
    """
    if symbols is None:
        symbols = UNIVERSE

    print(f"\n{'='*55}")
    print(f"  STATARB PIPELINE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Balance: ${account_balance:,.2f}")
    print(f"  Risk:    {risk_pct:.1%}")
    print(f"{'='*55}")

    # Load prices once — used by all steps
    price_data = load_all_price_data(symbols)

    if len(price_data) < 2:
        print("[ERR] Insufficient price data "
              "loaded")
        return {}

    # Run pipeline steps
    screener_df = run_screener(price_data)
    backtest_df = run_backtests(
        screener_df, price_data)
    signals_df  = compute_live_signals(
        backtest_df, price_data)
    sizing_df   = compute_risk_sizing(
        signals_df,
        account_balance,
        risk_pct)
    summary_df  = build_summary(
        screener_df, backtest_df,
        signals_df,  sizing_df,
        account_balance, risk_pct)

    # Console summary
    n_act = len(sizing_df)
    print(f"\n{'='*55}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Pairs tested  : "
          f"{len(screener_df)}")
    print(f"  Valid pairs   : "
          f"{(screener_df['Valid']=='YES').sum() if not screener_df.empty else 0}")
    print(f"  Profitable    : "
          f"{(backtest_df['Profitable']=='YES').sum() if not backtest_df.empty else 0}")
    print(f"  Active signals: {n_act}")

    if n_act > 0:
        for _, r in sizing_df.iterrows():
            print(f"    {r['Pair']}: "
                  f"{r['Signal']}  "
                  f"{r['Leg1_Symbol']} "
                  f"{r['Leg1_Direction']} "
                  f"{r['Leg1_Lots']}L  "
                  f"{r['Leg2_Symbol']} "
                  f"{r['Leg2_Direction']} "
                  f"{r['Leg2_Lots']}L")

    print(f"{'='*55}")

    return {
        'screener': screener_df,
        'backtest': backtest_df,
        'signals' : signals_df,
        'sizing'  : sizing_df,
        'summary' : summary_df,
    }
