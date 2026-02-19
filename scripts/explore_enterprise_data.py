#!/usr/bin/env python3
"""Explore BigQuery schema to debug Enterprise usage report."""
from google.cloud import bigquery

PROJECT_ID = "mkt-analytics-268801"
client = bigquery.Client(project=PROJECT_ID)

# 1. What product_tier values exist?
print("=== Product tier values (salesforce.account) ===")
q1 = f"""
SELECT product_tier_c, COUNT(*) as cnt
FROM `{PROJECT_ID}.salesforce.account`
WHERE is_deleted = FALSE
GROUP BY product_tier_c
ORDER BY cnt DESC
"""
for r in client.query(q1).result():
    print(f"  {r.product_tier_c!r}: {r.cnt}")

# 2. Enterprise-like counts
print("\n=== Accounts with 'enterprise' in product_tier ===")
q2 = f"""
SELECT COUNT(*) as cnt
FROM `{PROJECT_ID}.salesforce.account`
WHERE is_deleted = FALSE
  AND LOWER(COALESCE(product_tier_c, '')) LIKE '%enterprise%'
"""
for r in client.query(q2).result():
    print(f"  Count: {r.cnt}")

# 3. Enterprise with pendo_account_id
print("\n=== Enterprise accounts with pendo_account_id ===")
q3 = f"""
SELECT COUNT(*) as cnt
FROM `{PROJECT_ID}.salesforce.account`
WHERE is_deleted = FALSE
  AND LOWER(COALESCE(product_tier_c, '')) LIKE '%enterprise%'
  AND pendo_account_id_c IS NOT NULL
"""
for r in client.query(q3).result():
    print(f"  Count: {r.cnt}")

# 4. Pendo event date range and sample
print("\n=== Pendo event sample (date range, num_minutes) ===")
q4 = f"""
SELECT 
  MIN(timestamp) as min_ts,
  MAX(timestamp) as max_ts,
  SUM(num_minutes) as total_min,
  COUNT(*) as row_count
FROM `{PROJECT_ID}.pendo.event`
LIMIT 1
"""
for r in client.query(q4).result():
    print(f"  Min: {r.min_ts}, Max: {r.max_ts}")
    print(f"  Total minutes: {r.total_min}, Rows: {r.row_count}")

# 5. Join Enterprise + Pendo in FY24 range
print("\n=== Enterprise + Pendo usage in FY24 (Feb 2024 - Jan 2025) ===")
q5 = f"""
SELECT 
  COUNT(DISTINCT a.id) as acct_count,
  ROUND(SUM(p.num_minutes)/60.0, 2) as total_hrs
FROM `{PROJECT_ID}.salesforce.account` a
JOIN `{PROJECT_ID}.pendo.event` p ON a.pendo_account_id_c = p.account_id
WHERE a.is_deleted = FALSE
  AND LOWER(COALESCE(a.product_tier_c, '')) LIKE '%enterprise%'
  AND a.pendo_account_id_c IS NOT NULL
  AND p.timestamp >= TIMESTAMP('2024-02-01')
  AND p.timestamp < TIMESTAMP('2025-02-01')
"""
for r in client.query(q5).result():
    print(f"  Accounts: {r.acct_count}, Total hours: {r.total_hrs}")
