#!/usr/bin/env python3
"""
Enterprise Usage Report: Time (hours) in Arize for Enterprise customers.
Uses market analytics BigQuery data - Pendo usage + Salesforce.
Breaks down by net new vs tenured customers within the past fiscal year.

Run from id-pain directory:
  python scripts/enterprise_usage_report.py

Requires: gcloud auth application-default login (for BigQuery access)
"""

from datetime import datetime
from google.cloud import bigquery

PROJECT_ID = "mkt-analytics-268801"

# Arize fiscal year: assumed Feb 1 - Jan 31 (common for SaaS)
# Past fiscal year = FY24: Feb 1, 2024 - Jan 31, 2025
FISCAL_YEAR_START = "2024-02-01"
FISCAL_YEAR_END = "2025-01-31"


def run_enterprise_usage_report():
    """Query BigQuery for Enterprise usage hours, net new vs tenured."""
    client = bigquery.Client(project=PROJECT_ID)

    # Arize for Enterprise = customer_success_tier_c = 'Enterprise' (96 accounts)
    # Alternative: product_tier_c = 'Enterprise' (9 accounts) - fewer, different set
    # Net new = first closed-won opportunity in fiscal year
    # Tenured = first closed-won opportunity before fiscal year
    query = f"""
    WITH enterprise_accounts AS (
        -- Get Enterprise accounts (customer_success_tier) with their first closed-won date
        SELECT 
            a.id as account_id,
            a.name as account_name,
            a.customer_success_tier_c as customer_tier,
            a.pendo_account_id_c as pendo_account_id,
            MIN(CASE WHEN o.is_won = TRUE AND o.close_date IS NOT NULL 
                THEN DATE(o.close_date) END) as first_close_date
        FROM `{PROJECT_ID}.salesforce.account` a
        LEFT JOIN `{PROJECT_ID}.salesforce.opportunity` o 
            ON a.id = o.account_id AND o.is_deleted = FALSE
        WHERE a.is_deleted = FALSE
          AND a.customer_success_tier_c = 'Enterprise'
        GROUP BY a.id, a.name, a.customer_success_tier_c, a.pendo_account_id_c
    ),
    accounts_with_segment AS (
        SELECT 
            ea.*,
            CASE 
                WHEN ea.first_close_date >= DATE('{FISCAL_YEAR_START}') 
                     AND ea.first_close_date <= DATE('{FISCAL_YEAR_END}')
                THEN 'net_new'
                WHEN ea.first_close_date < DATE('{FISCAL_YEAR_START}')
                THEN 'tenured'
                ELSE 'unknown'
            END as customer_segment
        FROM enterprise_accounts ea
        WHERE ea.first_close_date IS NOT NULL  -- Must have closed-won to classify
    ),
    pendo_usage_fy AS (
        -- Pendo usage within fiscal year for Enterprise accounts
        SELECT 
            e.account_id,
            e.pendo_account_id,
            e.customer_segment,
            COALESCE(SUM(p.num_minutes), 0) as total_minutes
        FROM accounts_with_segment e
        INNER JOIN `{PROJECT_ID}.pendo.event` p 
            ON e.pendo_account_id = p.account_id
        WHERE p.timestamp >= TIMESTAMP('{FISCAL_YEAR_START}')
          AND p.timestamp < TIMESTAMP_ADD(TIMESTAMP('{FISCAL_YEAR_END}'), INTERVAL 1 DAY)
        GROUP BY e.account_id, e.pendo_account_id, e.customer_segment
    ),
    -- All Enterprise accounts with Pendo link (include 0 usage for accounts with no FY events)
    all_segments AS (
        SELECT 
            e.account_id, 
            e.pendo_account_id, 
            e.customer_segment, 
            COALESCE(p.total_minutes, 0) as total_minutes
        FROM accounts_with_segment e
        LEFT JOIN pendo_usage_fy p 
            ON e.account_id = p.account_id 
            AND e.pendo_account_id = p.pendo_account_id 
            AND e.customer_segment = p.customer_segment
        WHERE e.pendo_account_id IS NOT NULL
    )
    SELECT 
        customer_segment,
        COUNT(DISTINCT account_id) as account_count,
        ROUND(SUM(total_minutes) / 60.0, 2) as total_hours,
        ROUND(AVG(total_minutes) / 60.0, 2) as avg_hours_per_account
    FROM all_segments
    GROUP BY customer_segment
    ORDER BY customer_segment
    """

    print("=" * 60)
    print("Arize for Enterprise – Usage Report (Past Fiscal Year)")
    print(f"Fiscal year: {FISCAL_YEAR_START} to {FISCAL_YEAR_END}")
    print("=" * 60)

    try:
        job = client.query(query)
        results = job.result()

        total_hours = 0
        rows = []
        for row in results:
            rows.append(row)
            total_hours += row.total_hours
            seg_label = "Net New" if row.customer_segment == "net_new" else "Tenured"
            print(f"\n{seg_label}:")
            print(f"  Accounts:   {row.account_count}")
            print(f"  Total hrs:  {row.total_hours:,.2f}")
            print(f"  Avg/account: {row.avg_hours_per_account:,.2f} hrs")

        if rows:
            print("\n" + "-" * 60)
            print(f"TOTAL HOURS (Enterprise): {total_hours:,.2f}")
            print("=" * 60)

    except Exception as e:
        print(f"Error running query: {e}")
        raise


def run_simple_totals():
    """Simpler query - total Enterprise hours (no segment) for validation."""
    client = bigquery.Client(project=PROJECT_ID)

    query = f"""
    SELECT 
        COUNT(DISTINCT a.id) as enterprise_account_count,
        COUNT(DISTINCT a.pendo_account_id_c) as with_pendo_id,
        ROUND(SUM(p.num_minutes) / 60.0, 2) as total_hours
    FROM `{PROJECT_ID}.salesforce.account` a
    JOIN `{PROJECT_ID}.pendo.event` p 
        ON a.pendo_account_id_c = p.account_id
    WHERE a.is_deleted = FALSE
      AND (LOWER(COALESCE(a.product_tier_c, '')) LIKE '%enterprise%'
           OR LOWER(COALESCE(a.product_tier_c, '')) = 'arize for enterprise')
      AND p.timestamp >= TIMESTAMP('{FISCAL_YEAR_START}')
      AND p.timestamp < TIMESTAMP_ADD(TIMESTAMP('{FISCAL_YEAR_END}'), INTERVAL 1 DAY)
    """
    job = client.query(query)
    for row in job.result():
        print("\nQuick totals (all Enterprise, FY):")
        print(f"  Enterprise accounts: {row.enterprise_account_count}")
        print(f"  With Pendo link: {row.with_pendo_id}")
        print(f"  Total hours: {row.total_hours:,.2f}")


def run_all_time_summary():
    """All-time Enterprise usage (no FY filter) for context."""
    client = bigquery.Client(project=PROJECT_ID)
    q = f"""
    SELECT a.name, ROUND(SUM(p.num_minutes)/60.0, 2) as hours
    FROM `{PROJECT_ID}.salesforce.account` a
    JOIN `{PROJECT_ID}.pendo.event` p ON a.pendo_account_id_c = p.account_id
    WHERE a.is_deleted = FALSE 
      AND a.customer_success_tier_c = 'Enterprise'
      AND a.pendo_account_id_c IS NOT NULL
    GROUP BY a.name
    """
    print("\n" + "=" * 60)
    print("All-time usage (3 Enterprise accts with direct Pendo link)")
    print("(Last activity for each was before Feb 2024)")
    print("=" * 60)
    total = 0
    for r in client.query(q).result():
        print(f"  {r.name}: {r.hours} hrs")
        total += r.hours
    print(f"  TOTAL: {total} hrs")


if __name__ == "__main__":
    run_enterprise_usage_report()
    run_all_time_summary()
