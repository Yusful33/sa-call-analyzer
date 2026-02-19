"""BigQuery client for Salesforce and Gong data access.

Note: Salesforce data is in US region, Gong data is in us-central1 region.
Cross-region joins are not supported, so we query each separately.
"""

from datetime import datetime
from typing import Any
from google.cloud import bigquery
from pydantic import BaseModel


class WonOpportunity(BaseModel):
    """A won opportunity from Salesforce."""

    opp_id: str
    opp_name: str
    account_id: str
    account_name: str
    industry: str | None
    amount: float
    close_date: datetime | None
    employee_count: int | None = None


class GongCallInsight(BaseModel):
    """Gong call insight linked to an opportunity."""

    opp_id: str
    call_title: str
    spotlight_brief: str | None
    key_points: list[str]
    outcome: str | None
    next_steps: str | None
    call_date: datetime | None


class IndustryStats(BaseModel):
    """Statistics for an industry."""

    industry: str
    deal_count: int
    total_revenue: float
    avg_deal_size: float


class SimilarCustomer(BaseModel):
    """A similar customer for proof points."""

    account_name: str
    industry: str
    amount: float
    employee_count: int | None
    close_date: datetime | None


class BigQueryClient:
    """Client for Salesforce + Gong queries via BigQuery.

    Handles both US region (Salesforce) and us-central1 region (Gong) queries.
    """

    def __init__(self, project_id: str = "mkt-analytics-268801"):
        self.project_id = project_id
        self.client = bigquery.Client(project=project_id)

    def _run_query(self, query: str, location: str = "US") -> list[dict[str, Any]]:
        """Run a BigQuery query and return results as list of dicts."""
        job_config = bigquery.QueryJobConfig()
        job = self.client.query(query, job_config=job_config, location=location)
        results = job.result()
        return [dict(row) for row in results]

    # =========================================================================
    # Salesforce queries (US region)
    # =========================================================================

    def get_won_opportunities(
        self,
        industry: str | None = None,
        limit: int = 100,
        min_amount: float = 0,
    ) -> list[WonOpportunity]:
        """Query closed-won opportunities with account info.

        Args:
            industry: Optional filter by industry
            limit: Maximum number of results
            min_amount: Minimum deal amount

        Returns:
            List of won opportunities
        """
        industry_filter = ""
        if industry:
            # Escape single quotes in industry name
            safe_industry = industry.replace("'", "\\'")
            industry_filter = f"AND a.industry = '{safe_industry}'"

        query = f"""
        SELECT
            o.id as opp_id,
            o.name as opp_name,
            o.account_id,
            a.name as account_name,
            a.industry,
            o.amount,
            o.close_date,
            a.number_of_employees as employee_count
        FROM `{self.project_id}.salesforce.opportunity` o
        JOIN `{self.project_id}.salesforce.account` a ON o.account_id = a.id
        WHERE o.is_won = true
            AND o.amount >= {min_amount}
            {industry_filter}
        ORDER BY o.close_date DESC
        LIMIT {limit}
        """

        results = self._run_query(query, location="US")

        return [
            WonOpportunity(
                opp_id=r["opp_id"],
                opp_name=r["opp_name"],
                account_id=r["account_id"],
                account_name=r["account_name"],
                industry=r.get("industry"),
                amount=float(r["amount"]) if r["amount"] else 0,
                close_date=r.get("close_date"),
                employee_count=r.get("employee_count"),
            )
            for r in results
        ]

    def get_account_by_name(self, account_name: str) -> dict[str, Any] | None:
        """Lookup account details by name (fuzzy match).

        Args:
            account_name: Company name to search for

        Returns:
            Account details or None if not found
        """
        safe_name = account_name.replace("'", "\\'")

        query = f"""
        SELECT
            a.id,
            a.name,
            a.industry,
            a.number_of_employees,
            a.website,
            a.billing_city,
            a.billing_state,
            a.billing_country
        FROM `{self.project_id}.salesforce.account` a
        WHERE LOWER(a.name) LIKE LOWER('%{safe_name}%')
        LIMIT 1
        """

        results = self._run_query(query, location="US")
        return results[0] if results else None

    def get_similar_customers(
        self,
        industry: str,
        employee_count: int | None = None,
        limit: int = 3,
    ) -> list[SimilarCustomer]:
        """Find similar won customers for proof points.

        Args:
            industry: Industry to match
            employee_count: Optional employee count for size matching
            limit: Maximum number of results

        Returns:
            List of similar customers
        """
        safe_industry = industry.replace("'", "\\'")

        # If we have employee count, try to match similar size
        size_filter = ""
        if employee_count:
            min_size = int(employee_count * 0.5)
            max_size = int(employee_count * 2)
            size_filter = f"AND a.number_of_employees BETWEEN {min_size} AND {max_size}"

        query = f"""
        SELECT
            a.name as account_name,
            a.industry,
            o.amount,
            a.number_of_employees as employee_count,
            o.close_date
        FROM `{self.project_id}.salesforce.account` a
        JOIN `{self.project_id}.salesforce.opportunity` o ON a.id = o.account_id
        WHERE o.is_won = true
            AND a.industry = '{safe_industry}'
            AND o.amount > 0
            {size_filter}
        ORDER BY o.close_date DESC
        LIMIT {limit}
        """

        results = self._run_query(query, location="US")

        return [
            SimilarCustomer(
                account_name=r["account_name"],
                industry=r["industry"],
                amount=float(r["amount"]) if r["amount"] else 0,
                employee_count=r.get("employee_count"),
                close_date=r.get("close_date"),
            )
            for r in results
        ]

    def get_industry_stats(self) -> list[IndustryStats]:
        """Get deal counts and revenue by industry.

        Returns:
            List of industry statistics, sorted by deal count descending
        """
        query = f"""
        SELECT
            a.industry,
            COUNT(DISTINCT o.id) as deal_count,
            SUM(o.amount) as total_revenue,
            AVG(o.amount) as avg_deal_size
        FROM `{self.project_id}.salesforce.opportunity` o
        JOIN `{self.project_id}.salesforce.account` a ON o.account_id = a.id
        WHERE o.is_won = true
            AND o.amount > 0
            AND a.industry IS NOT NULL
        GROUP BY a.industry
        HAVING deal_count >= 3
        ORDER BY deal_count DESC
        """

        results = self._run_query(query, location="US")

        return [
            IndustryStats(
                industry=r["industry"],
                deal_count=r["deal_count"],
                total_revenue=float(r["total_revenue"]) if r["total_revenue"] else 0,
                avg_deal_size=float(r["avg_deal_size"]) if r["avg_deal_size"] else 0,
            )
            for r in results
        ]

    # =========================================================================
    # Gong queries (us-central1 region)
    # =========================================================================

    def get_call_insights_for_opportunities(
        self,
        opp_ids: list[str],
    ) -> list[GongCallInsight]:
        """Fetch Gong call insights for given opportunity IDs.

        Args:
            opp_ids: List of Salesforce opportunity IDs

        Returns:
            List of call insights with CALL_SPOTLIGHT data
        """
        if not opp_ids:
            return []

        # Format IDs for SQL IN clause
        ids_str = ", ".join(f"'{oid}'" for oid in opp_ids)

        query = f"""
        SELECT
            ctx.OBJECT_ID as opp_id,
            c.TITLE as call_title,
            c.CALL_SPOTLIGHT_BRIEF as spotlight_brief,
            c.CALL_SPOTLIGHT_KEY_POINTS as key_points,
            c.CALL_SPOTLIGHT_OUTCOME as outcome,
            c.CALL_SPOTLIGHT_NEXT_STEPS as next_steps,
            c.EFFECTIVE_START_DATETIME as call_date
        FROM `{self.project_id}.gong.CONVERSATION_CONTEXTS` ctx
        JOIN `{self.project_id}.gong.CALLS` c ON c.CONVERSATION_KEY = ctx.CONVERSATION_KEY
        WHERE ctx.OBJECT_ID IN ({ids_str})
            AND ctx.OBJECT_TYPE = 'opportunity'
            AND c.CALL_SPOTLIGHT_BRIEF IS NOT NULL
        ORDER BY c.EFFECTIVE_START_DATETIME DESC
        """

        results = self._run_query(query, location="us-central1")

        insights = []
        for r in results:
            # Parse key_points from JSON if present
            key_points = []
            if r.get("key_points"):
                # key_points is already parsed by BigQuery if it's JSON type
                kp = r["key_points"]
                if isinstance(kp, list):
                    key_points = kp
                elif isinstance(kp, str):
                    # Try to parse as JSON string
                    import json

                    try:
                        key_points = json.loads(kp)
                    except json.JSONDecodeError:
                        key_points = [kp]

            insights.append(
                GongCallInsight(
                    opp_id=r["opp_id"],
                    call_title=r["call_title"],
                    spotlight_brief=r.get("spotlight_brief"),
                    key_points=key_points,
                    outcome=r.get("outcome"),
                    next_steps=r.get("next_steps"),
                    call_date=r.get("call_date"),
                )
            )

        return insights

    def get_calls_for_industry(
        self,
        industry: str,
        min_deals: int = 10,
        calls_per_deal: int = 3,
    ) -> tuple[list[WonOpportunity], list[GongCallInsight]]:
        """Get won deals and their Gong call insights for an industry.

        This is a convenience method that:
        1. Queries Salesforce for won deals in the industry
        2. Queries Gong for call insights linked to those deals

        Args:
            industry: Industry to query
            min_deals: Minimum number of deals required
            calls_per_deal: Target number of calls per deal

        Returns:
            Tuple of (won opportunities, call insights)
        """
        # Get won opportunities
        opps = self.get_won_opportunities(
            industry=industry,
            limit=min_deals * 2,  # Get extra to account for missing Gong data
            min_amount=1000,  # Skip tiny deals
        )

        if len(opps) < min_deals:
            print(
                f"Warning: Only {len(opps)} deals found for {industry} "
                f"(minimum {min_deals} requested)"
            )

        # Get call insights
        opp_ids = [o.opp_id for o in opps]
        insights = self.get_call_insights_for_opportunities(opp_ids)

        return opps, insights

    def get_value_drivers_by_industry(
        self,
        industry: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get value drivers (pain, why_arize, urgency) from won deals.

        Args:
            industry: Optional industry filter
            limit: Maximum results

        Returns:
            List of value driver records
        """
        industry_filter = ""
        if industry:
            safe_industry = industry.replace("'", "\\'")
            industry_filter = f"AND a.industry = '{safe_industry}'"

        query = f"""
        SELECT 
            a.industry,
            o.identify_pain_c as pain,
            o.why_arize_c as why_arize,
            o.why_now_c as urgency,
            o.metrics_c as success_metrics,
            o.amount
        FROM `{self.project_id}.salesforce.opportunity` o
        JOIN `{self.project_id}.salesforce.account` a ON o.account_id = a.id
        WHERE o.is_won = TRUE 
            AND (o.identify_pain_c IS NOT NULL OR o.why_arize_c IS NOT NULL)
            {industry_filter}
        ORDER BY o.close_date DESC
        LIMIT {limit}
        """

        return self._run_query(query, location="US")

    def get_common_pains_by_industry(self) -> dict[str, list[str]]:
        """Get aggregated pain points by industry from won deals.

        Returns:
            Dict mapping industry to list of pain points
        """
        query = f"""
        SELECT 
            a.industry,
            ARRAY_AGG(DISTINCT o.identify_pain_c IGNORE NULLS LIMIT 10) as pains
        FROM `{self.project_id}.salesforce.opportunity` o
        JOIN `{self.project_id}.salesforce.account` a ON o.account_id = a.id
        WHERE o.is_won = TRUE 
            AND o.identify_pain_c IS NOT NULL
            AND a.industry IS NOT NULL
        GROUP BY a.industry
        HAVING COUNT(*) >= 2
        """

        results = self._run_query(query, location="US")
        return {r["industry"]: r["pains"] for r in results if r["pains"]}
