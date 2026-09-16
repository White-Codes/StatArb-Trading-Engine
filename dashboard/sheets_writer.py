"""
sheets_writer.py
================
Writes all pipeline DataFrames to
Google Sheets using gspread.

Sheet tabs:
  1_SCREENER     — all pairs tested
  2_BACKTEST     — backtest results
  3_SIGNALS      — live z-scores
  4_SIZING       — position sizes
  5_SUMMARY      — portfolio KPIs
"""

import gspread
import pandas as pd
import numpy as np
import time
from google.oauth2.service_account import (
    Credentials)


# ── Google API Scopes ─────────────────────────
SCOPES = [
    'https://spreadsheets.google.com/feeds',
    'https://www.googleapis.com/auth/'
    'drive',
]

# ── Colour constants (RGB for gspread) ────────
# gspread uses 0-1 float RGB
def rgb(r, g, b):
    return {
        "red"  : r / 255,
        "green": g / 255,
        "blue" : b / 255,
    }

COL_BLUE_DARK   = rgb(0,   70,  127)
COL_GREEN_DARK  = rgb(0,   176, 80)
COL_GREEN_LIGHT = rgb(198, 239, 206)
COL_RED_DARK    = rgb(192, 0,   0)
COL_RED_LIGHT   = rgb(255, 199, 206)
COL_ORANGE      = rgb(255, 235, 156)
COL_GREY        = rgb(242, 242, 242)
COL_WHITE       = rgb(255, 255, 255)
COL_GOLD        = rgb(255, 192, 0)


class GoogleSheetsWriter:
    def __init__(self,
                 credentials_json: dict,
                 sheet_id:         str):
        """
        Parameters
        ----------
        credentials_json : dict
            Parsed JSON from Google service
            account key file
        sheet_id : str
            Google Sheet ID from URL
        """
        creds = Credentials.from_service_account_info(
            credentials_json,
            scopes=SCOPES)
        self.gc       = gspread.authorize(creds)
        self.sheet_id = sheet_id
        self.wb       = self.gc.open_by_key(
            sheet_id)
        print(f"  [SHEETS] Connected: "
              f"{self.wb.title}")

    def _get_or_create_tab(
            self, name: str
            ) -> gspread.Worksheet:
        """Get tab by name or create it."""
        try:
            return self.wb.worksheet(name)
        except gspread.WorksheetNotFound:
            ws = self.wb.add_worksheet(
                title=name,
                rows=500,
                cols=30)
            print(f"  [SHEETS] Created tab: "
                  f"{name}")
            return ws

    def _clear_tab(self,
                   ws: gspread.Worksheet):
        """Clear all content."""
        ws.clear()
        time.sleep(0.5)  # API rate limit

    def _df_to_values(
            self,
            df: pd.DataFrame
            ) -> list:
        """
        Convert DataFrame to list of lists
        for gspread batch update.
        Handles NaN, inf, None safely.
        """
        # Headers
        rows = [df.columns.tolist()]

        # Data rows
        for _, row in df.iterrows():
            clean = []
            for v in row.values:
                if isinstance(v, float):
                    if np.isnan(v) or np.isinf(v):
                        clean.append("")
                    else:
                        clean.append(
                            round(v, 6))
                elif v is None:
                    clean.append("")
                else:
                    clean.append(str(v)
                                 if not isinstance(
                                     v, (int, str,
                                         bool))
                                 else v)
            rows.append(clean)

        return rows

    def _write_values(
            self,
            ws:     gspread.Worksheet,
            values: list):
        """Write values with rate limit guard."""
        try:
            ws.update(values,
                      value_input_option='RAW')
            time.sleep(1.0)
        except gspread.exceptions.APIError as e:
            print(f"  [API ERR] {e}")
            time.sleep(30)
            ws.update(values,
                      value_input_option='RAW')

    def _format_header_row(
            self,
            ws:      gspread.Worksheet,
            row:     int,
            n_cols:  int,
            bg_color: dict = None,
            ):
        """Format a header row."""
        if bg_color is None:
            bg_color = COL_BLUE_DARK

        end_col = chr(ord('A') + n_cols - 1)
        cell_range = f"A{row}:{end_col}{row}"

        fmt = {
            "backgroundColor": bg_color,
            "textFormat": {
                "bold"            : True,
                "foregroundColor" : COL_WHITE,
                "fontSize"        : 10,
            },
            "horizontalAlignment": "CENTER",
        }
        try:
            ws.format(cell_range, fmt)
            time.sleep(0.5)
        except Exception as e:
            print(f"  [FMT ERR] {e}")

    def _colour_rows_by_column(
            self,
            ws:          gspread.Worksheet,
            df:          pd.DataFrame,
            col_name:    str,
            start_row:   int,
            colour_map:  dict,
            ):
        """
        Colour entire rows based on value
        in a specific column.

        colour_map: {"YES": COL_GREEN_LIGHT, ...}
        """
        if col_name not in df.columns:
            return

        col_idx = df.columns.tolist().index(
            col_name)
        n_cols  = len(df.columns)
        end_col = chr(ord('A') + n_cols - 1)

        requests = []
        for ri, (_, row) in enumerate(
                df.iterrows()):
            val    = str(row[col_name])
            colour = colour_map.get(val)
            if colour is None:
                continue

            row_excel = start_row + ri
            requests.append({
                "repeatCell": {
                    "range": {
                        "sheetId"         :
                            ws._properties[
                                'sheetId'],
                        "startRowIndex"   :
                            row_excel - 1,
                        "endRowIndex"     :
                            row_excel,
                        "startColumnIndex": 0,
                        "endColumnIndex"  :
                            n_cols,
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "backgroundColor":
                                colour,
                        }
                    },
                    "fields":
                        "userEnteredFormat"
                        ".backgroundColor",
                }
            })

        if requests:
            try:
                self.wb.batch_update(
                    {"requests": requests})
                time.sleep(1.0)
            except Exception as e:
                print(f"  [COLOUR ERR] {e}")

    def _freeze_row(self,
                    ws:  gspread.Worksheet,
                    row: int = 1):
        """Freeze top N rows."""
        try:
            ws.freeze(rows=row)
            time.sleep(0.3)
        except Exception:
            pass

    # ──────────────────────────────────────────
    #  TAB 1: SCREENER
    # ──────────────────────────────────────────
    def write_screener(self,
                       df: pd.DataFrame):
        print("  [SHEETS] Writing screener...")
        ws = self._get_or_create_tab(
            "1_SCREENER")
        self._clear_tab(ws)

        if df.empty:
            ws.update([["No data"]])
            return

        # Title row
        ws.update(
            "A1",
            [["PAIR SCREENER — "
              "Cointegration Research"]])
        ws.update(
            "A2",
            [["GREEN = Valid | "
              "RED = Rejected | "
              "Score = quality composite"]])

        # Data from row 4
        cols = [
            'Pair', 'EG_pval', 'ADF_pval',
            'Johansen_pval', 'Half_Life',
            'Hurst', 'Hedge_Ratio',
            'Valid', 'Score']
        disp   = df[cols].copy()
        values = self._df_to_values(disp)
        ws.update(f"A4", values)
        time.sleep(1)

        # Format header
        self._format_header_row(
            ws, 4, len(cols))
        self._freeze_row(ws, 1)

        # Colour by Valid
        self._colour_rows_by_column(
            ws, disp, 'Valid',
            start_row=5,
            colour_map={
                "YES": COL_GREEN_LIGHT,
                "NO" : COL_RED_LIGHT,
            })

        print(f"    Done: {len(df)} pairs")

    # ──────────────────────────────────────────
    #  TAB 2: BACKTEST
    # ──────────────────────────────────────────
    def write_backtest(self,
                       df: pd.DataFrame):
        print("  [SHEETS] Writing backtest...")
        ws = self._get_or_create_tab(
            "2_BACKTEST")
        self._clear_tab(ws)

        if df.empty:
            ws.update(
                [["No backtest results"]])
            return

        ws.update(
            "A1",
            [["BACKTEST RESULTS — "
              "Walk-Forward OOS"]])
        ws.update(
            "A2",
            [["GREEN = Profitable OOS | "
              "RED = Unprofitable | "
              "PF > 1.0 = pass"]])

        cols = [
            'Pair', 'Trades',
            'Win_Rate_Pct',
            'Profit_Factor', 'Sharpe',
            'Max_DD_Pct', 'Avg_Hold_Bars',
            'Profitable', 'Half_Life']
        disp   = df[cols].copy()
        values = self._df_to_values(disp)
        ws.update("A4", values)
        time.sleep(1)

        self._format_header_row(
            ws, 4, len(cols))
        self._freeze_row(ws, 1)

        self._colour_rows_by_column(
            ws, disp, 'Profitable',
            start_row=5,
            colour_map={
                "YES": COL_GREEN_LIGHT,
                "NO" : COL_RED_LIGHT,
            })

        print(f"    Done: {len(df)} pairs")

    # ──────────────────────────────────────────
    #  TAB 3: LIVE SIGNALS
    # ──────────────────────────────────────────
    def write_signals(self,
                      df: pd.DataFrame):
        print("  [SHEETS] Writing signals...")
        ws = self._get_or_create_tab(
            "3_SIGNALS")
        self._clear_tab(ws)

        if df.empty:
            ws.update(
                [["No profitable pairs"]])
            return

        ws.update(
            "A1",
            [["LIVE SIGNALS — "
              "Current Z-Scores & Actions"]])
        ws.update(
            "A2",
            [["GREEN=LONG | RED=SHORT | "
              "ORANGE=WATCH | WHITE=FLAT"]])

        cols = [
            'Pair', 'Z_Score', 'Z_Change',
            'Signal', 'Action', 'Urgency',
            'Price1', 'Price2', 'Beta',
            'PF_Backtest', 'WR_Backtest',
            'Sharpe', 'Updated_UTC']
        disp   = df[cols].copy()
        values = self._df_to_values(disp)
        ws.update("A4", values)
        time.sleep(1)

        self._format_header_row(
            ws, 4, len(cols))
        self._freeze_row(ws, 1)

        self._colour_rows_by_column(
            ws, disp, 'Signal',
            start_row=5,
            colour_map={
                "LONG"        : COL_GREEN_LIGHT,
                "SHORT"       : COL_RED_LIGHT,
                "WATCH LONG"  : COL_ORANGE,
                "WATCH SHORT" : COL_ORANGE,
                "FLAT"        : COL_WHITE,
            })

        print(f"    Done: {len(df)} signals")

    # ──────────────────────────────────────────
    #  TAB 4: RISK SIZING
    # ──────────────────────────────────────────
    def write_sizing(self,
                     df: pd.DataFrame):
        print("  [SHEETS] Writing sizing...")
        ws = self._get_or_create_tab(
            "4_SIZING")
        self._clear_tab(ws)

        ws.update(
            "A1",
            [["POSITION SIZING — "
              "Active Signals Only"]])

        if df.empty:
            ws.update(
                "A3",
                [["No active signals "
                  "at this time."]])
            return

        values = self._df_to_values(df)
        ws.update("A3", values)
        time.sleep(1)

        self._format_header_row(
            ws, 3, len(df.columns))
        self._freeze_row(ws, 1)

        self._colour_rows_by_column(
            ws, df, 'Signal',
            start_row=4,
            colour_map={
                "LONG" : COL_GREEN_LIGHT,
                "SHORT": COL_RED_LIGHT,
            })

        print(f"    Done: {len(df)} positions")

    # ──────────────────────────────────────────
    #  TAB 5: SUMMARY
    # ──────────────────────────────────────────
    def write_summary(self,
                      df: pd.DataFrame):
        print("  [SHEETS] Writing summary...")
        ws = self._get_or_create_tab(
            "5_SUMMARY")
        self._clear_tab(ws)

        if df.empty:
            ws.update([["No summary data"]])
            return

        values = self._df_to_values(df)
        ws.update("A1", values)
        time.sleep(1)

        # Format section headers
        for ri, (_, row) in enumerate(
                df.iterrows()):
            val = str(row['Metric'])
            if val.startswith("──"):
                row_excel = ri + 2
                ws.format(
                    f"A{row_excel}:B{row_excel}",
                    {"backgroundColor":
                         COL_BLUE_DARK,
                     "textFormat": {
                         "bold"           : True,
                         "foregroundColor":
                             COL_WHITE}})
                time.sleep(0.2)

        # Bold the title row
        ws.format("A2:B2", {
            "textFormat": {"bold": True,
                           "fontSize": 12}})

        print(f"    Done: summary written")

    # ──────────────────────────────────────────
    #  WRITE ALL TABS
    # ──────────────────────────────────────────
    def write_all(self, results: dict):
        """Write all tabs from pipeline output."""
        print("\n[SHEETS] Writing all tabs...")

        self.write_screener(
            results['screener'])
        self.write_backtest(
            results['backtest'])
        self.write_signals(
            results['signals'])
        self.write_sizing(
            results['sizing'])
        self.write_summary(
            results['summary'])

        print("[SHEETS] All tabs updated")
        print(f"  View at: "
              f"https://docs.google.com/"
              f"spreadsheets/d/"
              f"{self.sheet_id}")
