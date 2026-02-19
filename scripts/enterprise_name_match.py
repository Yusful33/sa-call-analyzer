#!/usr/bin/env python3
"""Try matching Enterprise accounts to Pendo by name for FY24 usage."""
from google.cloud import bigquery

PROJECT_ID = "mkt-analytics-268801"
client = bigquery.Client(project=PROJECT_ID)

# Join salesforce (Enterprise) to pendo.account_history by name, then to pendo.event
# Use agent_name from pendo.account_history - may match salesforce account name
q = """
WITH enterprise_sf AS (
  SELECT id, name
  FROM `mkt-analytics-268801.salesforce.account`
  WHERE is_deleted = FALSE AND customer_success_tier_c = 'Enterprise'
),
pendo_by_name AS (
  SELECT DISTINCT id as pendo_id, agent_name
  FROM `mkt-analytics-268801.pendo.account_history`
  WHERE agent_name IS NOT NULL
),
matched AS (
  SELECT e.id, e.name, p.pendo_id
  FROM enterprise_sf e
  JOIN pendo_by_name p ON LOWER(TRIM(e.name)) = LOWER(TRIM(p.agent_name))
)
SELECT 
  m.name,
  ROUND(SUM(ev.num_minutes)/60.0, 2) as fy24_hours
FROM matched m
JOIN `mkt-analytics-268801.pendo.event` ev 
  ON m.pendo_id = ev.account_id
WHERE ev.timestamp >= TIMESTAMP('2024-02-01')
  AND ev.timestamp < TIMESTAMP('2025-02-01')
GROUP BY m.name
ORDER BY fy24_hours DESC
"""
for r in client.query(q).result():
    print(f"  {r.name}: {r.fy24_hours} hrs (FY24)")
