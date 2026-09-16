"""
scheduler.py
============
Entry point for GitHub Actions.
Reads secrets from environment variables.
Runs full pipeline and updates Google Sheets.
"""

import os
import json
import argparse
from datetime import datetime, timezone

from dashboard.screener import (
    run_full_pipeline, UNIVERSE)
from dashboard.sheets_writer import (
    GoogleSheetsWriter)


def run_once(
        account_balance: float = 10000.0,
        risk_pct:        float = 0.01,
        ):
    """
    Run pipeline once and update Sheets.
    Called by GitHub Actions every hour.
    """
    print(f"\n{'='*55}")
    print(f"  StatArb → Google Sheets")
    print(f"  {datetime.now(timezone.utc)"
          f".strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*55}")

    # ── Load credentials from env ─────────────
    creds_raw = os.environ.get(
        'GOOGLE_CREDENTIALS', '')
    sheet_id  = os.environ.get(
        'GOOGLE_SHEET_ID', '')

    if not creds_raw:
        raise ValueError(
            "GOOGLE_CREDENTIALS env var not set.\n"
            "Add it to GitHub Secrets.")
    if not sheet_id:
        raise ValueError(
            "GOOGLE_SHEET_ID env var not set.\n"
            "Add it to GitHub Secrets.")

    try:
        creds_dict = json.loads(creds_raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"GOOGLE_CREDENTIALS is not valid "
            f"JSON: {e}")

    # ── Run pipeline ──────────────────────────
    results = run_full_pipeline(
        account_balance = account_balance,
        risk_pct        = risk_pct,
        symbols         = UNIVERSE,
    )

    if not results:
        print("[WARN] Pipeline returned "
              "no results")
        return

    # ── Write to Google Sheets ────────────────
    writer = GoogleSheetsWriter(
        credentials_json = creds_dict,
        sheet_id         = sheet_id,
    )
    writer.write_all(results)

    print(f"\n  [DONE] "
          f"{datetime.now(timezone.utc)"
          f".strftime('%Y-%m-%d %H:%M UTC')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="StatArb Google Sheets "
                    "Dashboard")
    parser.add_argument(
        "--balance",
        type    = float,
        default = 10000.0,
        help    = "Account balance (USD)")
    parser.add_argument(
        "--risk",
        type    = float,
        default = 0.01,
        help    = "Risk per trade (0.01=1%)")
    args = parser.parse_args()

    run_once(
        account_balance = args.balance,
        risk_pct        = args.risk,
    )
