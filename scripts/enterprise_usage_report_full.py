#!/usr/bin/env python3
"""
Enterprise Usage Report: Time (hours) in Arize for Enterprise customers.
Uses market analytics BigQuery - Salesforce + Pendo.
Includes both direct pendo_account_id link and name-based matching.

Run: python scripts/enterprise_usage_report_full.py
Requires: gcloud auth application-default login
"""

from google.cloud import bigquery

PROJECT_ID = "mkt-analytics-268801"
FISCAL_YEAR_START = "2024-02-01"
FISCAL_YEAR_END = "2025-01-31"


def main():
    client = bigquery.Client(project=PROJECT_ID)

    # Enterprise = customer_success_tier_c = 'Enterprise'
    # Match to Pendo via: (1) pendo_account_id_c OR (2) name match to pendo.account_history.agent_name
    # Net new = first closed-won in FY; Tenured = first closed-won before FY
    query = f"""
    WITH enterprise_accounts AS (
        SELECT 
            a.id as account_id,
            a.name as account_name,
            a.pendo_account_id_c as pendo_id_direct,
            MIN(CASE WHEN o.is_won = TRUE AND o.close_date IS NOT NULL 
                THEN DATE(o.close_date) END) as first_close_date
        FROM `{PROJECT_ID}.salesforce.account` a
        LEFT JOIN `{PROJECT_ID}.salesforce.opportunity` o 
            ON a.id = o.account_id AND o.is_deleted = FALSE
        WHERE a.is_deleted = FALSE AND a.customer_success_tier_c = 'Enterprise'
        GROUP BY a.id, a.name, a.pendo_account_id_c
    ),
    -- Resolve Pendo ID: direct link or name match
    pendo_ids AS (
        SELECT 
            e.account_id,
            e.account_name,
            e.first_close_date,
            COALESCE(e.pendo_id_direct, ph.id) as pendo_account_id
        FROM enterprise_accounts e
        LEFT JOIN (
            SELECT id, agent_name,
                   ROW_NUMBER() OVER (PARTITION BY LOWER(TRIM(agent_name)) ORDER BY last_visit DESC) rn
            FROM `{PROJECT_ID}.pendo.account_history`
            WHERE agent_name IS NOT NULL
        ) ph ON LOWER(TRIM(e.account_name)) = LOWER(TRIM(ph.agent_name)) AND ph.rn = 1
        WHERE COALESCE(e.pendo_id_direct, ph.id) IS NOT NULL
    ),
    accounts_with_segment AS (
        SELECT 
            p.*,
            CASE 
                WHEN p.first_close_date >= DATE('{FISCAL_YEAR_START}') 
                     AND p.first_close_date <= DATE('{FISCAL_YEAR_END}') THEN 'net_new'
                WHEN p.first_close_date < DATE('{FISCAL_YEAR_START}') THEN 'tenured'
                ELSE 'unknown'
            END as customer_segment
        FROM pendo_ids p
        WHERE p.first_close_date IS NOT NULL
    ),
    pendo_usage_fy AS (
        SELECT 
            e.account_id,
            e.customer_segment,
            COALESCE(SUM(p.num_minutes), 0) as total_minutes
        FROM accounts_with_segment e
        JOIN `{PROJECT_ID}.pendo.event` p ON e.pendo_account_id = p.account_id
        WHERE p.timestamp >= TIMESTAMP('{FISCAL_YEAR_START}')
          AND p.timestamp < TIMESTAMP_ADD(TIMESTAMP('{FISCAL_YEAR_END}'), INTERVAL 1 DAY)
        GROUP BY e.account_id, e.customer_segment
    ),
    all_segments AS (
        SELECT 
            e.account_id,
            e.account_name,
            e.customer_segment,
            COALESCE(p.total_minutes, 0) as total_minutes
        FROM accounts_with_segment e
        LEFT JOIN pendo_usage_fy p ON e.account_id = p.account_id AND e.customer_segment = p.customer_segment
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

    print("=" * 65)
    print("Arize for Enterprise – Usage Report (Past Fiscal Year)")
    print(f"Fiscal year: {FISCAL_YEAR_START} to {FISCAL_YEAR_END}")
    print("Enterprise = customer_success_tier_c = 'Enterprise'")
    print("Pendo match: direct pendo_account_id OR name match to Pendo account")
    print("=" * 65)

    job = client.query(query)
    results = list(job.result())
    total_hours = sum(r.total_hours for r in results)

    for row in results:
        seg = "Net New" if row.customer_segment == "net_new" else "Tenured"
        if row.customer_segment == "unknown":
            seg = "Unknown (no closed-won)"
        print(f"\n{seg}:")
        print(f"  Accounts:      {row.account_count}")
        print(f"  Total hours:   {row.total_hours:,.2f}")
        print(f"  Avg/account:   {row.avg_hours_per_account:,.2f} hrs")

    if results:
        print("\n" + "-" * 65)
        print(f"TOTAL HOURS (Enterprise, past fiscal year): {total_hours:,.2f}")
        print("=" * 65)


if __name__ == "__main__":
    main()
