#!/usr/bin/env python3
"""Check Pendo account_id matching."""
from google.cloud import bigquery

PROJECT_ID = "mkt-analytics-268801"
client = bigquery.Client(project=PROJECT_ID)

# Enterprise accounts (customer_success_tier) with their pendo IDs
print("=== Enterprise accounts with pendo_account_id ===")
q1 = f"""
SELECT a.id, a.name, a.pendo_account_id_c
FROM `{PROJECT_ID}.salesforce.account` a
WHERE a.is_deleted = FALSE AND a.customer_success_tier_c = 'Enterprise'
  AND a.pendo_account_id_c IS NOT NULL
"""
ents = list(client.query(q1).result())
print(f"  Count: {len(ents)}")
for r in ents[:5]:
    print(f"    {r.name!r} -> pendo_id={r.pendo_account_id_c!r}")

# Do these pendo IDs exist in pendo.event?
if ents:
    pendo_ids = [r.pendo_account_id_c for r in ents]
    print("\n=== Do these exist in pendo.event? ===")
    # Sample one
    pid = pendo_ids[0]
    q2 = f"""
    SELECT account_id, SUM(num_minutes) as total_min, COUNT(*) as cnt
    FROM `{PROJECT_ID}.pendo.event`
    WHERE account_id = @pid
    GROUP BY account_id
    """
    job = client.query(q2, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("pid", "STRING", pid)]
    ))
    for r in job.result():
        print(f"  account_id={pid!r}: {r.total_min} min, {r.cnt} events")
    # Also check pendo.event distinct account_id format
    q3 = f"""
    SELECT account_id FROM `{PROJECT_ID}.pendo.event` LIMIT 5
    """
    print("\n  Sample pendo.event account_id values:")
    for r in client.query(q3).result():
        print(f"    {r.account_id!r}")
