"""Prospect overview and account-related routes."""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from opentelemetry.trace import Status, StatusCode

from models import (
    ProspectOverviewRequest,
    ProspectOverview,
    AccountSuggestionsRequest,
    AccountSuggestionsResponse,
)
from observability import api_span

logger = logging.getLogger("sa-call-analyzer")

router = APIRouter(prefix="/api", tags=["prospect"])


def create_prospect_routes(bq_client, gong_enrichment_fn):
    """
    Factory function to create prospect routes with injected dependencies.
    
    Args:
        bq_client: BigQuery client instance
        gong_enrichment_fn: Function to enrich prospect overview with Gong data
    
    Returns:
        APIRouter with prospect-related routes
    """
    
    @router.post("/prospect-overview", response_model=ProspectOverview)
    async def get_prospect_overview(request: ProspectOverviewRequest):
        """
        Get comprehensive prospect overview from BigQuery data warehouse.
        
        Aggregates data from:
        - Salesforce (account, opportunities, activities)
        - Gong (call analytics, deal summaries)
        - Pendo (product usage, feature adoption)
        - FullStory (user behavior, friction events)
        """
        with api_span(
            "get_prospect_overview",
            lookup_method="name" if request.account_name else ("domain" if request.domain else "sfdc_id"),
            lookup_value=request.account_name or request.domain or request.sfdc_account_id,
        ):
            if not bq_client:
                raise HTTPException(
                    status_code=503,
                    detail="BigQuery client not available. Check GCP credentials configuration.",
                )

            try:
                def _fetch():
                    return bq_client.get_prospect_overview(
                        account_name=request.account_name,
                        domain=request.domain,
                        sfdc_account_id=request.sfdc_account_id,
                    )

                overview = await asyncio.wait_for(asyncio.to_thread(_fetch), timeout=60.0)
                
                if gong_enrichment_fn:
                    overview = gong_enrichment_fn(overview)
                
                return overview

            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=504,
                    detail="Request timed out while fetching prospect data. Try again.",
                )
            except HTTPException:
                raise
            except Exception as e:
                logger.error("Prospect overview failed: %s", e)
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to fetch prospect overview: {str(e)[:200]}",
                )

    @router.post("/account-suggestions", response_model=AccountSuggestionsResponse)
    async def account_suggestions(request: AccountSuggestionsRequest):
        """
        Resolve a typed account/company name against Salesforce for disambiguation.
        
        Helps when spacing or punctuation differs from the CRM record
        (e.g. "Alliance Bernstein" vs "AllianceBernstein").
        """
        if not bq_client:
            raise HTTPException(
                status_code=503,
                detail="BigQuery client not available.",
            )
        data = bq_client.suggest_salesforce_account_names(
            account_name=request.account_name.strip(),
            domain=(request.domain or "").strip() or None,
        )
        return AccountSuggestionsResponse.model_validate(data)

    return router
