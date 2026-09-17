"""
dashboard/screener.py
=====================
Full pipeline engine.
Runs research -> backtest -> signals -> sizing.

Fix: Removed dependency on pairs_research imports
     that were failing silently due to path issues.
     All required functions defined locally.
"""

import numpy as np
import pandas as pd
import sys, os, traceback
from datetime import datetime, timezone
from itertools import combinations

# ── Path setup ────────────────────────────────
ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Import from root-level files ──────────────
try:
    from pairs_research import (
        align_series,
        compute_half_life,
        test_window,
        _run_adf,
        _run_johansen,
        _hurst_on_diffs,
    )
    print("[IMPORT] pairs_research: OK")
except Exception as e:
    print(f"[IMPORT ERR] pairs_research: {e}")
    traceback.print_exc()
    raise

try:
    from pairs_backtest import (
        estimate_ols,
        apply_fixed_spread,
        compute_test_zscore,
        backtest_pair,
        compute_stats,
    )
    print("[IMPORT] pairs_backtest: OK")
except Exception as e:
    print(f"[IMPORT ERR] pairs_backtest: {e}")
    traceback.print_exc()
    raise

try:
    from dashboard.live_feed import (
        load_all_price_data,
    )
    print("[IMPORT] live_feed: OK")
except Exception as e:
    print(f"[IMPORT ERR] live_feed: {e}")
    traceback.print_exc()
    raise

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

    avail = list(price_data.keys())
    pairs = list(combinations(avail, 2))
    print(f"\n[1-SCREENER] Testing "
          f"{len(pairs)} pairs...")

    # Debug first series structure
    if price_data:
        first_k = list(price_data.keys())[0]
        first_v = price_data[first_k]
        print(f"  [DEBUG] First series: "
              f"{first_k} "
              f"type={type(first_v).__name__} "
              f"ndim={getattr(first_v,'ndim','?')} "
              f"len={len(first_v)} "
              f"index_type="
              f"{type(first_v.index).__name__}")

    rows = []
    n_ok  = 0
    n_err = 0

    for sym1, sym2 in pairs:
        try:
            s1_raw = price_data[sym1]
            s2_raw = price_data[sym2]

            # Defensive: ensure both are
            # clean Series before aligning
            if not isinstance(
                    s1_raw, pd.Series):
                raise ValueError(
                    f"{sym1} is "
                    f"{type(s1_raw)}, "
                    f"not Series")
            if not isinstance(
                    s2_raw, pd.Series):
                raise ValueError(
                    f"{sym2} is "
                    f"{type(s2_raw)}, "
                    f"not Series")

            # Align on common timestamps
            # Use simple inner join
            df = pd.concat(
                [s1_raw.rename(sym1),
                 s2_raw.rename(sym2)],
                axis=1,
                join='inner').dropna()

            if len(df) < 500:
                continue

            s1 = df[sym1]
            s2 = df[sym2]

            n = len(s1)
            w = min(TRAIN_BARS, n)

            result = test_window(
                s1.iloc[-w:],
                s2.iloc[-w:])

            if not isinstance(result, dict):
                continue

            hl    = result.get(
                'half_life', np.inf)
            score = float(
                result.get('score', 0.0))
            if np.isnan(score):
                score = 0.0

            rows.append({
                'Pair'         : (
                    f"{sym1}/{sym2}"),
                'Symbol1'      : sym1,
                'Symbol2'      : sym2,
                'EG_pval'      : round(float(
                    result.get(
                        'eg_pval', 1.0)), 4),
                'ADF_pval'     : round(float(
                    result.get(
                        'adf_pval', 1.0)), 4),
                'Johansen_pval': round(float(
                    result.get(
                        'johansen_pval',
                        1.0)), 4),
                'Half_Life'    : (
                    round(float(hl), 1)
                    if np.isfinite(hl)
                    else 999.0),
                'Hurst'        : round(float(
                    result.get(
                        'hurst', 0.5)), 3),
                'Hedge_Ratio'  : round(float(
                    result.get(
                        'hedge_ratio',
                        0.0)), 4),
                'Valid'        : (
                    "YES"
                    if result.get(
                        'cointegrated',
                        False)
                    else "NO"),
                'Score'        : round(
                    score, 4),
            })
            n_ok += 1

        except Exception as e:
            n_err += 1
            if n_err <= 3:
                print(f"  [ERR] "
                      f"{sym1}/{sym2}: {e}")
                traceback.print_exc()
            elif n_err == 4:
                print(f"  [ERR] "
                      f"(further errors "
                      f"suppressed)")

    print(f"  Processed: {n_ok} OK, "
          f"{n_err} errors")

    if not rows:
        print("  [WARN] No rows collected")
        return pd.DataFrame(columns=[
            'Pair', 'Symbol1', 'Symbol2',
            'EG_pval', 'ADF_pval',
            'Johansen_pval', 'Half_Life',
            'Hurst', 'Hedge_Ratio',
            'Valid', 'Score'])

    df = pd.DataFrame(rows)
    if 'Score' not in df.columns:
        df['Score'] = 0.0

    df = df.sort_values(
        'Score', ascending=False
    ).reset_index(drop=True)

    n_valid = (df['Valid'] == 'YES').sum()
    print(f"  Result: {n_valid} valid "
          f"/ {len(df)} total")
    return df

# ─────────────────────────────────────────────
#  STEP 2: BACKTEST
# ─────────────────────────────────────────────
def run_backtests(
        screener_df: pd.DataFrame,
        price_data:  dict
        ) -> pd.DataFrame:
    """Backtest all valid pairs."""
    if screener_df.empty:
        print("\n[2-BACKTEST] No pairs "
              "to backtest")
        return pd.DataFrame()

    valid = screener_df[
        screener_df['Valid'] == 'YES']

    if valid.empty:
        print("\n[2-BACKTEST] No valid pairs")
        return pd.DataFrame()

    print(f"\n[2-BACKTEST] Running "
          f"{len(valid)} pairs...")

    rows = []
    for _, row in valid.iterrows():
        sym1 = row['Symbol1']
        sym2 = row['Symbol2']
        pair = row['Pair']

        try:
            if (sym1 not in price_data or
                    sym2 not in price_data):
                print(f"  [SKIP] {pair}: "
                      f"missing price data")
                continue

            combined = pd.concat(
                [price_data[sym1],
                 price_data[sym2]],
                axis=1).dropna()
            combined.columns = [sym1, sym2]

            s1 = combined[sym1]
            s2 = combined[sym2]

            min_needed = (
                TRAIN_BARS + TEST_BARS)
            if len(s1) < min_needed:
                print(f"  [SKIP] {pair}: "
                      f"need {min_needed}, "
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

            st = compute_stats(
                trades, equity)
            pf = float(st.get(
                'profit_factor', 0.0))

            rows.append({
                'Pair'         : pair,
                'Symbol1'      : sym1,
                'Symbol2'      : sym2,
                'Trades'       : int(st.get(
                    'n_trades', 0)),
                'Win_Rate'     : round(float(
                    st.get('win_rate', 0.0)),
                    4),
                'Win_Rate_Pct' : (
                    f"{st.get('win_rate',0):.1%}"),
                'Profit_Factor': round(pf, 4),
                'Sharpe'       : round(float(
                    st.get('sharpe', 0.0)),
                    4),
                'Max_DD'       : round(float(
                    st.get('max_dd', 0.0)),
                    4),
                'Max_DD_Pct'   : (
                    f"{st.get('max_dd',0):.1%}"),
                'Avg_Hold_Bars': round(float(
                    st.get('avg_hold', 0.0)),
                    1),
                'Total_PnL'    : round(float(
                    st.get(
                        'total_pnl_n', 0.0)),
                    4),
                'Profitable'   : (
                    "YES"
                    if pf >= MIN_BT_PF
                    else "NO"),
                'Hedge_Ratio'  : round(float(
                    row.get(
                        'Hedge_Ratio', 0.0)),
                    4),
                'Half_Life'    : float(
                    row.get('Half_Life', 0)),
            })
            print(f"  {pair}: "
                  f"PF={pf:.2f}  "
                  f"WR={st.get('win_rate',0):.1%}"
                  f"  "
                  f"{'PASS' if pf>=MIN_BT_PF else 'FAIL'}")

        except Exception as e:
            print(f"  [ERR] {pair}: {e}")
            traceback.print_exc()

    if not rows:
        print("  [WARN] No backtest results")
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values(
        'Profit_Factor',
        ascending=False
    ).reset_index(drop=True)

    return df


# ─────────────────────────────────────────────
#  STEP 3: LIVE SIGNALS
# ─────────────────────────────────────────────
def compute_live_signals(
        backtest_df: pd.DataFrame,
        price_data:  dict
        ) -> pd.DataFrame:
    """Compute current z-scores."""
    if backtest_df.empty:
        print("\n[3-SIGNALS] No backtest "
              "results")
        return pd.DataFrame()

    profitable = backtest_df[
        backtest_df['Profitable'] == 'YES']

    if profitable.empty:
        print("\n[3-SIGNALS] No profitable "
              "pairs")
        return pd.DataFrame()

    print(f"\n[3-SIGNALS] Computing for "
          f"{len(profitable)} pairs...")

    rows = []
    now  = datetime.now(timezone.utc)

    for _, row in profitable.iterrows():
        sym1 = row['Symbol1']
        sym2 = row['Symbol2']
        pair = row['Pair']

        try:
            if (sym1 not in price_data or
                    sym2 not in price_data):
                continue

            combined = pd.concat(
                [price_data[sym1],
                 price_data[sym2]],
                axis=1).dropna()
            combined.columns = [sym1, sym2]

            s1 = combined[sym1]
            s2 = combined[sym2]

            if len(s1) < TRAIN_BARS + 10:
                continue

            t_s1 = s1.iloc[
                   -TRAIN_BARS:].values
            t_s2 = s2.iloc[
                   -TRAIN_BARS:].values

            beta, alpha, train_std = (
                estimate_ols(t_s1, t_s2))

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

            p1_now  = float(s1.iloc[-1])
            p2_now  = float(s2.iloc[-1])
            p1_prev = float(s1.iloc[-2])
            p2_prev = float(s2.iloc[-2])

            sp_now  = (np.log(p1_now) -
                       beta * np.log(p2_now) -
                       alpha)
            sp_prev = (np.log(p1_prev) -
                       beta * np.log(p2_prev) -
                       alpha)

            z_now   = round(
                float((sp_now - mu) / sig), 3)
            z_prev  = float(
                (sp_prev - mu) / sig)
            z_change = round(
                float(z_now - z_prev), 3)

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
                'WR_Backtest' : str(
                    row.get(
                        'Win_Rate_Pct',
                        '0%')),
                'Sharpe'      : float(
                    row.get('Sharpe', 0)),
                'Updated_UTC' : now.strftime(
                    '%Y-%m-%d %H:%M'),
            })

            print(f"  {pair}: "
                  f"z={z_now:+.3f}  "
                  f"-> {signal}")

        except Exception as e:
            print(f"  [ERR] {pair}: {e}")
            traceback.print_exc()

    if not rows:
        print("  [WARN] No signals computed")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
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
    """Compute lot sizes for active signals."""
    if signals_df.empty:
        print("\n[4-SIZING] No signals")
        return pd.DataFrame()

    active = signals_df[
        signals_df['Signal'].isin(
            ['LONG', 'SHORT'])].copy()

    if active.empty:
        print("\n[4-SIZING] No active signals")
        return pd.DataFrame()

    print(f"\n[4-SIZING] Sizing "
          f"{len(active)} signals...")

    risk_amount = account_balance * risk_pct
    rows = []

    for _, row in active.iterrows():
        try:
            beta   = float(row['Beta'])
            z_now  = float(row['Z_Score'])
            std    = float(row['Spread_Std'])
            signal = row['Signal']

            stop_z = (
                -STOP_Z
                if signal == 'LONG'
                else STOP_Z)
            stop_dist_z = abs(
                stop_z - z_now)
            stop_dist   = stop_dist_z * std
            pips_risk   = max(
                stop_dist * 10000, 1.0)

            target_pips = std * 1.5 * 10000
            rr_ratio    = round(
                target_pips / pips_risk, 2)

            pip_value = 10.0
            lot1 = risk_amount / (
                pips_risk * pip_value)
            lot1 = round(float(
                np.clip(lot1, 0.01, 5.0)),
                2)
            lot2 = round(float(
                np.clip(
                    lot1 * abs(beta),
                    0.01, 5.0)),
                2)

            leg1_dir = (
                'BUY' if signal == 'LONG'
                else 'SELL')
            leg2_dir = (
                'SELL' if signal == 'LONG'
                else 'BUY')

            rows.append({
                'Pair'           : row['Pair'],
                'Signal'         : signal,
                'Z_Score'        : z_now,
                'Account_Bal'    : (
                    account_balance),
                'Risk_Pct'       : (
                    f"{risk_pct:.1%}"),
                'Risk_Amount_USD': round(
                    risk_amount, 2),
                'Leg1_Symbol'    : (
                    row['Symbol1']),
                'Leg1_Direction' : leg1_dir,
                'Leg1_Lots'      : lot1,
                'Leg2_Symbol'    : (
                    row['Symbol2']),
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
                  f"{leg1_dir} {lot1}L / "
                  f"{leg2_dir} {lot2}L  "
                  f"RR={rr_ratio}")

        except Exception as e:
            print(f"  [ERR] "
                  f"{row.get('Pair','?')}: "
                  f"{e}")

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
    """Build KPI summary table."""
    now = datetime.now(
        timezone.utc).strftime(
        '%Y-%m-%d %H:%M UTC')

    n_tested = len(screener_df)
    n_valid  = (
        int((screener_df['Valid'] == 'YES')
            .sum())
        if not screener_df.empty else 0)
    n_profit = (
        int((backtest_df['Profitable']
             == 'YES').sum())
        if not backtest_df.empty else 0)
    n_active = (
        len(signals_df[
            signals_df['Signal'].isin(
                ['LONG', 'SHORT'])])
        if not signals_df.empty else 0)

    rows = [
        ["STATARB LIVE DASHBOARD", ""],
        ["Last Updated",   now],
        ["", ""],
        ["-- PIPELINE --", ""],
        ["Pairs Tested",   n_tested],
        ["Valid (Coint)",  n_valid],
        ["Profitable OOS", n_profit],
        ["Active Signals", n_active],
        ["", ""],
        ["-- ACCOUNT --",  ""],
        ["Balance",
         f"${account_balance:,.2f}"],
        ["Risk Per Trade",
         f"{risk_pct:.1%}"],
        ["Risk Amount",
         f"${account_balance*risk_pct:,.2f}"],
        ["", ""],
    ]

    if not backtest_df.empty:
        prof = backtest_df[
            backtest_df['Profitable']
            == 'YES']
        if not prof.empty:
            rows += [
                ["-- BACKTEST AVERAGES --",
                 "(profitable pairs)"],
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
        rows,
        columns=['Metric', 'Value'])


# ─────────────────────────────────────────────
#  FULL PIPELINE
# ─────────────────────────────────────────────
def run_full_pipeline(
        account_balance: float = 10000.0,
        risk_pct:        float = 0.01,
        symbols:         list  = None,
        ) -> dict:
    """Run complete pipeline."""
    if symbols is None:
        symbols = UNIVERSE

    now_str = datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S')

    print(f"\n{'='*55}")
    print(f"  STATARB PIPELINE")
    print(f"  {now_str}")
    print(f"  Balance: ${account_balance:,.2f}")
    print(f"  Risk:    {risk_pct:.1%}")
    print(f"{'='*55}")

    price_data = load_all_price_data(symbols)

    if len(price_data) < 2:
        print("[ERR] Insufficient price data")
        return {}

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
    print(f"{'='*55}")

    return {
        'screener': screener_df,
        'backtest': backtest_df,
        'signals' : signals_df,
        'sizing'  : sizing_df,
        'summary' : summary_df,
    }
