#!/usr/bin/env python3
"""Check timestamp range for Enterprise Pendo events."""
from google.cloud import bigquery

PROJECT_ID = "mkt-analytics-268801"
client = bigquery.Client(project=PROJECT_ID)

# Get min/max timestamps for each Enterprise account's events
q = """
SELECT 
  a.name,
  MIN(p.timestamp) as first_evt,
  MAX(p.timestamp) as last_evt,
  SUM(p.num_minutes) as total_min
FROM `{}.salesforce.account` a
JOIN `{}.pendo.event` p ON a.pendo_account_id_c = p.account_id
WHERE a.is_deleted = FALSE 
  AND a.customer_success_tier_c = 'Enterprise'
  AND a.pendo_account_id_c IS NOT NULL
GROUP BY a.name
""".format(PROJECT_ID, PROJECT_ID)

for r in client.query(q).result():
    print(f"{r.name}:")
    print(f"  First: {r.first_evt}, Last: {r.last_evt}")
    print(f"  Total: {r.total_min} min")
