#!/usr/bin/env python3
"""
Daily Active Users (DAU) for the Arize platform.

Uses BigQuery Pendo data (mkt-analytics-268801.pendo.event) to compute:
- DAU by visitor: distinct users (visitor_id) per day
- DAU by account: distinct accounts (account_id) per day

Run from id-pain directory:
  python scripts/daily_active_users.py

Optional env or edit: DAYS_BACK (default 90), output CSV path.

Requires: gcloud auth application-default login (for BigQuery access)
"""

from datetime import datetime, timedelta, timezone
from google.cloud import bigquery

PROJECT_ID = "mkt-analytics-268801"
DAYS_BACK = 90  # override with env DAU_DAYS_BACK if needed


def run_dau_query(days_back: int = DAYS_BACK):
    """Query BigQuery for daily active users (visitors and accounts) in the Arize platform."""
    import os
    days = int(os.environ.get("DAU_DAYS_BACK", days_back))
    start_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

    client = bigquery.Client(project=PROJECT_ID)

    query = f"""
    WITH daily_activity AS (
        SELECT
            DATE(timestamp) AS activity_date,
            visitor_id,
            account_id
        FROM `{PROJECT_ID}.pendo.event`
        WHERE timestamp >= TIMESTAMP('{start_date}')
    )
    SELECT
        activity_date,
        COUNT(DISTINCT visitor_id) AS dau_visitors,
        COUNT(DISTINCT account_id) AS dau_accounts
    FROM daily_activity
    GROUP BY activity_date
    ORDER BY activity_date DESC
    """

    print("=" * 60)
    print("Arize platform – Daily Active Users (BigQuery Pendo)")
    print(f"Project: {PROJECT_ID}  |  Since: {start_date}  ({days} days)")
    print("=" * 60)

    try:
        job = client.query(query)
        results = list(job.result())

        if not results:
            print("No rows returned.")
            return []

        # Summary stats
        dau_visitors_list = [r.dau_visitors for r in results]
        dau_accounts_list = [r.dau_accounts for r in results]
        avg_dau_visitors = sum(dau_visitors_list) / len(dau_visitors_list)
        avg_dau_accounts = sum(dau_accounts_list) / len(dau_accounts_list)
        max_dau_visitors = max(dau_visitors_list)
        max_dau_accounts = max(dau_accounts_list)

        print(f"\nSummary (last {len(results)} days):")
        print(f"  DAU (visitors):  avg = {avg_dau_visitors:.1f}, max = {max_dau_visitors}")
        print(f"  DAU (accounts):  avg = {avg_dau_accounts:.1f}, max = {max_dau_accounts}")

        print("\nDaily breakdown (most recent 14 days):")
        print(f"  {'Date':<12}  {'DAU (visitors)':>14}  {'DAU (accounts)':>14}")
        print("  " + "-" * 44)
        for row in results[:14]:
            print(f"  {str(row.activity_date):<12}  {row.dau_visitors:>14}  {row.dau_accounts:>14}")

        return results
    except Exception as e:
        print(f"Error running query: {e}")
        raise


def run_dau_to_csv(days_back: int = DAYS_BACK, csv_path: str | None = None):
    """Run DAU query and optionally write full results to CSV."""
    import os
    import csv
    days = int(os.environ.get("DAU_DAYS_BACK", days_back))
    start_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    client = bigquery.Client(project=PROJECT_ID)

    query = f"""
    WITH daily_activity AS (
        SELECT
            DATE(timestamp) AS activity_date,
            visitor_id,
            account_id
        FROM `{PROJECT_ID}.pendo.event`
        WHERE timestamp >= TIMESTAMP('{start_date}')
    )
    SELECT
        activity_date,
        COUNT(DISTINCT visitor_id) AS dau_visitors,
        COUNT(DISTINCT account_id) AS dau_accounts
    FROM daily_activity
    GROUP BY activity_date
    ORDER BY activity_date ASC
    """
    job = client.query(query)
    results = list(job.result())

    if csv_path:
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["activity_date", "dau_visitors", "dau_accounts"])
            for r in results:
                w.writerow([str(r.activity_date), r.dau_visitors, r.dau_accounts])
        print(f"\nWrote {len(results)} rows to {csv_path}")

    return results


if __name__ == "__main__":
    run_dau_query()
    # Uncomment to also write full history to CSV:
    # run_dau_to_csv(csv_path="dau_arize_platform.csv")
