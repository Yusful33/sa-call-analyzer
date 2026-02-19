#!/usr/bin/env python3
"""Check Pendo events in FY range for Enterprise accounts."""
from google.cloud import bigquery

PROJECT_ID = "mkt-analytics-268801"
client = bigquery.Client(project=PROJECT_ID)

q = f"""
SELECT 
  a.name,
  a.pendo_account_id_c,
  MIN(p.timestamp) as first_evt,
  MAX(p.timestamp) as last_evt,
  SUM(p.num_minutes) as total_min
FROM `{PROJECT_ID}.salesforce.account` a
JOIN `{PROJECT_ID}.pendo.event` p ON a.pendo_account_id_c = p.account_id
WHERE a.is_deleted = FALSE 
  AND a.customer_success_tier_c = 'Enterprise'
  AND a.pendo_account_id_c IS NOT NULL
  AND p.timestamp >= TIMESTAMP('2024-02-01')
  AND p.timestamp < TIMESTAMP('2025-02-01')
GROUP BY a.name, a.pendo_account_id_c
"""
for r in client.query(q).result():
    print(f"{r.name}: {r.total_min} min ({r.total_min/60:.1f} hrs)")
    print(f"  Range: {r.first_evt} to {r.last_evt}")

# Also try without FY filter to see all Enterprise usage
print("\n=== Same query WITHOUT fiscal year filter ===")
q2 = f"""
SELECT 
  a.name,
  SUM(p.num_minutes) as total_min
FROM `{PROJECT_ID}.salesforce.account` a
JOIN `{PROJECT_ID}.pendo.event` p ON a.pendo_account_id_c = p.account_id
WHERE a.is_deleted = FALSE 
  AND a.customer_success_tier_c = 'Enterprise'
  AND a.pendo_account_id_c IS NOT NULL
GROUP BY a.name
"""
for r in client.query(q2).result():
    print(f"  {r.name}: {r.total_min} min ({r.total_min/60:.1f} hrs)")
