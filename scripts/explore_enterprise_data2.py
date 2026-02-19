#!/usr/bin/env python3
"""Explore Enterprise identification and lifecycle."""
from google.cloud import bigquery

PROJECT_ID = "mkt-analytics-268801"
client = bigquery.Client(project=PROJECT_ID)

# Lifecycle stages
print("=== Lifecycle stage values ===")
q = f"""
SELECT lifecycle_stage_c, COUNT(*) as cnt
FROM `{PROJECT_ID}.salesforce.account`
WHERE is_deleted = FALSE
GROUP BY lifecycle_stage_c
ORDER BY cnt DESC
"""
for r in client.query(q).result():
    print(f"  {r.lifecycle_stage_c!r}: {r.cnt}")

# Customer success tier
print("\n=== Customer success tier ===")
q2 = f"""
SELECT customer_success_tier_c, COUNT(*) as cnt
FROM `{PROJECT_ID}.salesforce.account`
WHERE is_deleted = FALSE
GROUP BY customer_success_tier_c
ORDER BY cnt DESC
"""
for r in client.query(q2).result():
    print(f"  {r.customer_success_tier_c!r}: {r.cnt}")

# The 9 Enterprise accounts - details
print("\n=== 9 Enterprise accounts (product_tier_c = 'Enterprise') ===")
q3 = f"""
SELECT id, name, pendo_account_id_c, lifecycle_stage_c, customer_success_tier_c
FROM `{PROJECT_ID}.salesforce.account`
WHERE is_deleted = FALSE AND product_tier_c = 'Enterprise'
"""
for r in client.query(q3).result():
    print(f"  {r.name[:40]!r}: pendo={r.pendo_account_id_c!r} lifecycle={r.lifecycle_stage_c!r}")

# Total Pendo usage for any account in FY24
print("\n=== Total Pendo hours (all accounts) in FY24 ===")
q4 = f"""
SELECT ROUND(SUM(num_minutes)/60.0, 2) as total_hrs, COUNT(DISTINCT account_id) as accts
FROM `{PROJECT_ID}.pendo.event`
WHERE timestamp >= TIMESTAMP('2024-02-01')
  AND timestamp < TIMESTAMP('2025-02-01')
"""
for r in client.query(q4).result():
    print(f"  Total hours: {r.total_hrs}, Unique accounts: {r.accts}")

# Maybe "customer" lifecycle = paid/enterprise?
print("\n=== Accounts with lifecycle 'Customer' (possible paid) ===")
q5 = f"""
SELECT COUNT(*) as cnt,
       COUNT(CASE WHEN pendo_account_id_c IS NOT NULL THEN 1 END) as with_pendo
FROM `{PROJECT_ID}.salesforce.account`
WHERE is_deleted = FALSE
  AND LOWER(COALESCE(lifecycle_stage_c, '')) = 'customer'
"""
for r in client.query(q5).result():
    print(f"  Total: {r.cnt}, With Pendo: {r.with_pendo}")
