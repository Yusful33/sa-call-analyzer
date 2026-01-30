"""
BigQuery Client for Prospect Overview

Fetches comprehensive organization data from BigQuery's Market Analytics project,
including Salesforce, Gong, Pendo, and FullStory data.
"""
import os
import json
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from google.cloud import bigquery
from google.cloud.exceptions import GoogleCloudError
import litellm

from models import (
    SalesforceAccountData,
    OpportunityData,
    GongCallData,
    GongParticipant,
    GongSummaryData,
    PendoUsageData,
    PendoVisitorActivity,
    PendoFeatureUsage,
    PendoPageUsage,
    DealSummary,
    UserIssueEvent,
    AdoptionMilestone,
    UserBehaviorAnalysis,
    SalesEngagementSummary,
    ProductUsageSummary,
    ProspectOverview,
)


class BigQueryClient:
    """Client for fetching prospect data from BigQuery."""
    
    PROJECT_ID = "mkt-analytics-268801"
    
    def __init__(self):
        """
        Initialize the BigQuery client using Application Default Credentials.
        
        For local development, run:
            gcloud auth application-default login
            
        For production/Docker, set GOOGLE_APPLICATION_CREDENTIALS to a service account JSON file.
        """
        try:
            self.client = bigquery.Client(project=self.PROJECT_ID)
            # Test the connection by listing datasets (lightweight operation)
            list(self.client.list_datasets(max_results=1))
            print(f"✅ BigQuery client initialized for project: {self.PROJECT_ID}")
        except Exception as e:
            error_msg = str(e)
            print(f"❌ BigQuery initialization failed: {error_msg}")
            print("\n📋 To fix this, run one of the following:")
            print("   For local development:")
            print("     gcloud auth application-default login")
            print("\n   For production/Docker:")
            print("     Set GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json")
            raise RuntimeError(
                f"BigQuery authentication failed. Run 'gcloud auth application-default login' "
                f"for local development. Error: {error_msg}"
            )
    
    def get_prospect_overview(
        self,
        account_name: Optional[str] = None,
        domain: Optional[str] = None,
        sfdc_account_id: Optional[str] = None,
        manual_competitors: Optional[List[str]] = None
    ) -> ProspectOverview:
        """
        Get comprehensive prospect overview from all data sources.
        
        Args:
            account_name: Account name to search (fuzzy match)
            domain: Email domain to search (e.g., "acme.com")
            sfdc_account_id: Salesforce Account ID (exact match)
            manual_competitors: List of known competitors to include in recommendations
                               (e.g., ["BrainTrust", "LangSmith"])
            
        Returns:
            ProspectOverview with data from all available sources
        """
        # Determine lookup method
        if sfdc_account_id:
            lookup_method = "sfdc_id"
            lookup_value = sfdc_account_id
        elif domain:
            lookup_method = "domain"
            lookup_value = domain
        elif account_name:
            lookup_method = "name"
            lookup_value = account_name
        else:
            raise ValueError("At least one lookup parameter is required")
        
        errors = []
        data_sources = []
        
        # 1. Fetch Salesforce account
        salesforce_account = None
        try:
            salesforce_account = self._get_salesforce_account(
                account_name=account_name,
                domain=domain,
                sfdc_account_id=sfdc_account_id
            )
            if salesforce_account:
                data_sources.append("salesforce")
        except Exception as e:
            errors.append(f"Salesforce error: {str(e)}")
        
        # 2. Fetch opportunities with full details
        all_opportunities = []
        latest_opportunity = None
        if salesforce_account:
            try:
                all_opportunities = self._get_opportunities(salesforce_account.id)
                if all_opportunities:
                    data_sources.append("salesforce_opportunities")
                    # Smart opportunity selection logic
                    latest_opportunity = self._select_most_relevant_opportunity(all_opportunities)
            except Exception as e:
                errors.append(f"Opportunities error: {str(e)}")
        
        # 3. Fetch Gong call data with transcripts and participants
        gong_summary = None
        if salesforce_account:
            try:
                gong_summary = self._get_gong_summary(salesforce_account.id)
                if gong_summary and gong_summary.total_calls > 0:
                    data_sources.append("gong")
            except Exception as e:
                errors.append(f"Gong error: {str(e)}")
        
        # 4. Fetch enhanced Pendo usage
        pendo_usage = None
        pendo_account_id = None
        
        # Try to get Pendo account ID from Salesforce first
        if salesforce_account and salesforce_account.pendo_account_id:
            pendo_account_id = salesforce_account.pendo_account_id
        # Otherwise, try to find it by account name in Pendo
        elif salesforce_account and salesforce_account.name:
            try:
                pendo_account_id = self._lookup_pendo_account_by_name(salesforce_account.name)
                if pendo_account_id:
                    data_sources.append("pendo_name_match")
            except Exception as e:
                errors.append(f"Pendo lookup error: {str(e)}")
        
        if pendo_account_id:
            try:
                pendo_usage = self._get_pendo_usage(pendo_account_id)
                if pendo_usage and pendo_usage.total_events > 0:
                    data_sources.append("pendo")
            except Exception as e:
                errors.append(f"Pendo error: {str(e)}")
        
        # 5. Build deal summary from Gong transcripts
        deal_summary = None
        if gong_summary and gong_summary.total_calls > 0:
            try:
                deal_summary = self._build_deal_summary(gong_summary)
            except Exception as e:
                errors.append(f"Deal summary error: {str(e)}")
        
        # 6. Fetch FullStory user issues (errors, dead clicks, frustrated moments)
        fullstory_issues = []
        search_domain = domain
        if not search_domain and salesforce_account and salesforce_account.website:
            website = salesforce_account.website
            if website:
                search_domain = website.replace("https://", "").replace("http://", "").replace("www.", "").split("/")[0]
        
        if search_domain:
            try:
                fullstory_issues = self._get_fullstory_issues(search_domain)
                if fullstory_issues:
                    data_sources.append("fullstory_issues")
            except Exception as e:
                errors.append(f"FullStory issues fetch error: {str(e)}")
        
        # 7. Fetch adoption milestones (projects, traces, experiments, evals, prompts)
        adoption_milestones = []
        if pendo_account_id:
            try:
                adoption_milestones = self._get_adoption_milestones(pendo_account_id)
                if adoption_milestones:
                    data_sources.append("adoption_milestones")
            except Exception as e:
                errors.append(f"Adoption milestones fetch error: {str(e)}")
        
        # 8. Build user behavior analysis from Pendo data + issues + milestones + Gong + account
        user_behavior = None
        if pendo_usage or fullstory_issues or adoption_milestones:
            try:
                user_behavior = self._build_user_behavior_analysis(
                    pendo=pendo_usage, 
                    fullstory_issues=fullstory_issues, 
                    adoption_milestones=adoption_milestones,
                    gong=gong_summary,
                    account=salesforce_account,
                    manual_competitors=manual_competitors
                )
                if user_behavior:
                    data_sources.append("user_behavior_analysis")
            except Exception as e:
                errors.append(f"User behavior analysis error: {str(e)}")
        
        # 8. Build sales engagement summary (based on selected opportunity)
        sales_engagement = self._build_sales_engagement_summary(
            salesforce_account, latest_opportunity, gong_summary, deal_summary
        )
        
        # 9. Build product usage summary
        product_usage = self._build_product_usage_summary(pendo_usage)
        
        return ProspectOverview(
            lookup_method=lookup_method,
            lookup_value=lookup_value,
            salesforce=salesforce_account,
            latest_opportunity=latest_opportunity,
            all_opportunities=all_opportunities,
            gong_summary=gong_summary,
            sales_engagement=sales_engagement,
            pendo_usage=pendo_usage,
            user_behavior=user_behavior,
            product_usage=product_usage,
            data_sources_available=data_sources,
            errors=errors
        )
    
    def _get_salesforce_account(
        self,
        account_name: Optional[str] = None,
        domain: Optional[str] = None,
        sfdc_account_id: Optional[str] = None
    ) -> Optional[SalesforceAccountData]:
        """Fetch Salesforce account data with enhanced fields and resolved user names."""
        
        conditions = []
        params = []
        
        if sfdc_account_id:
            conditions.append("a.id = @sfdc_id")
            params.append(bigquery.ScalarQueryParameter("sfdc_id", "STRING", sfdc_account_id))
        
        if domain:
            conditions.append("LOWER(a.website) LIKE @domain_pattern")
            params.append(bigquery.ScalarQueryParameter("domain_pattern", "STRING", f"%{domain.lower()}%"))
        
        if account_name:
            conditions.append("LOWER(a.name) LIKE @name_pattern")
            params.append(bigquery.ScalarQueryParameter("name_pattern", "STRING", f"%{account_name.lower()}%"))
        
        if not conditions:
            return None
        
        where_clause = " OR ".join(conditions)
        
        # Query with JOINs to resolve all user IDs to names
        query = f"""
        SELECT 
            a.id,
            a.name,
            a.website,
            a.industry,
            CAST(a.annual_revenue AS FLOAT64) as annual_revenue,
            a.number_of_employees,
            CAST(a.total_active_arr_c AS FLOAT64) as total_active_arr,
            a.lifecycle_stage_c as lifecycle_stage,
            a.product_tier_c as product_tier,
            a.status_c as status,
            a.customer_success_tier_c as customer_success_tier,
            sa_user.name as assigned_sa,
            cse_user.name as assigned_cse,
            csm_user.name as assigned_csm,
            ai_se_user.name as assigned_ai_se,
            owner_user.name as owner_name,
            a.pendo_account_id_c as pendo_account_id,
            a.pendo_time_on_site_c as pendo_time_on_site,
            a.of_models_c as num_models,
            a.is_using_arize_for_llms_c as is_using_llms,
            a.deployment_types_c as deployment_types,
            CAST(a.created_date AS STRING) as created_date,
            CAST(a.last_activity_date AS STRING) as last_activity_date,
            a.next_steps_c as next_steps,
            a.customer_notes_c as customer_notes,
            a.description
        FROM `{self.PROJECT_ID}.salesforce.account` a
        LEFT JOIN `{self.PROJECT_ID}.salesforce.user` owner_user ON a.owner_id = owner_user.id
        LEFT JOIN `{self.PROJECT_ID}.salesforce.user` sa_user ON a.assigned_sa_c = sa_user.id
        LEFT JOIN `{self.PROJECT_ID}.salesforce.user` cse_user ON a.assigned_cse_c = cse_user.id
        LEFT JOIN `{self.PROJECT_ID}.salesforce.user` csm_user ON a.assigned_csm_c = csm_user.id
        LEFT JOIN `{self.PROJECT_ID}.salesforce.user` ai_se_user ON a.assigned_ai_se_c = ai_se_user.id
        WHERE ({where_clause})
          AND a.is_deleted = FALSE
        ORDER BY 
            CASE WHEN a.total_active_arr_c IS NOT NULL THEN 0 ELSE 1 END,
            a.total_active_arr_c DESC
        LIMIT 1
        """
        
        job_config = bigquery.QueryJobConfig(query_parameters=params)
        results = self.client.query(query, job_config=job_config).result()
        
        for row in results:
            return SalesforceAccountData(
                id=row.id,
                name=row.name,
                website=row.website,
                industry=row.industry,
                annual_revenue=row.annual_revenue,
                number_of_employees=row.number_of_employees,
                total_active_arr=row.total_active_arr,
                lifecycle_stage=row.lifecycle_stage,
                product_tier=row.product_tier,
                status=row.status,
                customer_success_tier=row.customer_success_tier,
                assigned_sa=row.assigned_sa,
                assigned_cse=row.assigned_cse,
                assigned_csm=row.assigned_csm,
                assigned_ai_se=row.assigned_ai_se,
                owner_name=row.owner_name,
                pendo_account_id=row.pendo_account_id,
                pendo_time_on_site=row.pendo_time_on_site,
                num_models=row.num_models,
                is_using_llms=row.is_using_llms,
                deployment_types=row.deployment_types,
                created_date=row.created_date,
                last_activity_date=row.last_activity_date,
                next_steps=row.next_steps,
                customer_notes=row.customer_notes,
                description=row.description
            )
        
        return None
    
    def _get_opportunities(self, account_id: str) -> List[OpportunityData]:
        """Fetch opportunities with enhanced details."""
        
        query = f"""
        SELECT 
            op.id,
            op.name,
            op.stage_name,
            CAST(op.amount AS FLOAT64) as amount,
            CAST(op.close_date AS STRING) as close_date,
            op.probability,
            op.is_closed,
            op.is_won,
            op.forecast_category_name as forecast_category,
            CAST(op.created_date AS STRING) as created_date,
            CAST(op.last_modified_date AS STRING) as last_modified_date,
            op.age_in_days,
            op.next_step,
            op.description,
            op.lead_source,
            op.type,
            u.name as owner_name
        FROM `{self.PROJECT_ID}.salesforce.opportunity` op
        LEFT JOIN `{self.PROJECT_ID}.salesforce.user` u ON op.owner_id = u.id
        WHERE op.account_id = @account_id
          AND op.is_deleted = FALSE
        ORDER BY 
            CASE WHEN op.is_closed THEN 1 ELSE 0 END,
            op.close_date DESC
        LIMIT 10
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("account_id", "STRING", account_id)
            ]
        )
        
        results = self.client.query(query, job_config=job_config).result()
        
        opportunities = []
        for row in results:
            opportunities.append(OpportunityData(
                id=row.id,
                name=row.name,
                stage_name=row.stage_name,
                amount=row.amount,
                close_date=row.close_date,
                probability=row.probability,
                is_closed=row.is_closed or False,
                is_won=row.is_won or False,
                forecast_category=row.forecast_category,
                created_date=row.created_date,
                last_modified_date=row.last_modified_date,
                age_in_days=row.age_in_days,
                next_step=row.next_step,
                description=row.description,
                lead_source=row.lead_source,
                type=row.type,
                owner_name=row.owner_name
            ))
        
        return opportunities
    
    def _get_gong_summary(self, sfdc_account_id: str) -> Optional[GongSummaryData]:
        """Fetch comprehensive Gong call analytics with transcripts and participants."""
        
        # Get recent calls with full details (deduplicated by conversation_key)
        calls_query = f"""
        WITH ranked_calls AS (
            SELECT 
                c.CONVERSATION_KEY as conversation_key,
                c.CALL_URL as call_url,
                c.TITLE as call_title,
                CAST(c.EFFECTIVE_START_DATETIME AS STRING) as call_date,
                CAST(c.BROWSER_DURATION_SEC / 60.0 AS FLOAT64) as duration_minutes,
                c.CALL_SPOTLIGHT_BRIEF as spotlight_brief,
                c.CALL_SPOTLIGHT_KEY_POINTS as spotlight_key_points,
                c.CALL_SPOTLIGHT_NEXT_STEPS as spotlight_next_steps,
                c.CALL_SPOTLIGHT_OUTCOME as spotlight_outcome,
                c.CALL_SPOTLIGHT_TYPE as spotlight_type,
                CAST(s.TALK_RATIO AS FLOAT64) as talk_ratio,
                s.INTERACTIVITY as interactivity,
                s.PATIENCE as patience,
                CAST(s.QUESTION_RATE AS FLOAT64) as question_rate,
                s.LONGEST_MONOLOGUE as longest_monologue,
                s.LONGEST_CUSTOMER_STORY as longest_customer_story,
                ROW_NUMBER() OVER (PARTITION BY c.CONVERSATION_KEY ORDER BY c.EFFECTIVE_START_DATETIME DESC) as rn
            FROM `{self.PROJECT_ID}.gong.CALLS` c
            JOIN `{self.PROJECT_ID}.gong.CONVERSATION_CONTEXTS` ctx 
                ON c.CONVERSATION_KEY = ctx.CONVERSATION_KEY
            LEFT JOIN `{self.PROJECT_ID}.gong.INTERACTION_STATS` s 
                ON c.CONVERSATION_KEY = s.CONVERSATION_KEY
            WHERE LOWER(ctx.OBJECT_TYPE) = 'account' 
              AND ctx.OBJECT_ID = @account_id
              AND (c.IS_DELETED = false OR c.IS_DELETED IS NULL)
        )
        SELECT * EXCEPT(rn)
        FROM ranked_calls
        WHERE rn = 1
        ORDER BY call_date DESC
        LIMIT 20
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("account_id", "STRING", sfdc_account_id)
            ]
        )
        
        results = self.client.query(calls_query, job_config=job_config).result()
        
        calls = []
        talk_ratios = []
        interactivities = []
        patiences = []
        question_rates = []
        total_duration = 0
        first_call_date = None
        last_call_date = None
        all_briefs = []
        
        conversation_keys = []
        
        for row in results:
            # Track conversation keys for participant lookup
            if row.conversation_key:
                conversation_keys.append(row.conversation_key)
            
            # Parse key points from JSON - ensure all items are strings
            key_points = None
            if row.spotlight_key_points:
                try:
                    if isinstance(row.spotlight_key_points, str):
                        parsed = json.loads(row.spotlight_key_points)
                    else:
                        parsed = list(row.spotlight_key_points) if row.spotlight_key_points else []
                    
                    # Ensure all items are strings
                    key_points = [str(kp) for kp in parsed if kp] if parsed else None
                except (json.JSONDecodeError, TypeError):
                    key_points = None
            
            call = GongCallData(
                conversation_key=row.conversation_key,
                call_url=row.call_url,
                call_title=row.call_title,
                call_date=row.call_date,
                duration_minutes=row.duration_minutes,
                spotlight_brief=row.spotlight_brief,
                spotlight_key_points=key_points,
                spotlight_next_steps=row.spotlight_next_steps,
                spotlight_outcome=row.spotlight_outcome,
                spotlight_type=row.spotlight_type,
                talk_ratio=row.talk_ratio,
                interactivity=row.interactivity,
                patience=row.patience,
                question_rate=row.question_rate,
                longest_monologue=row.longest_monologue,
                longest_customer_story=row.longest_customer_story,
                participants=[]  # Will be populated below
            )
            calls.append(call)
            
            # Collect metrics for averaging
            if row.talk_ratio is not None:
                talk_ratios.append(row.talk_ratio)
            if row.interactivity is not None:
                interactivities.append(row.interactivity)
            if row.patience is not None:
                patiences.append(row.patience)
            if row.question_rate is not None:
                question_rates.append(row.question_rate)
            if row.duration_minutes:
                total_duration += row.duration_minutes
            
            # Track dates
            if row.call_date:
                if not last_call_date:
                    last_call_date = row.call_date
                first_call_date = row.call_date
            
            # Collect briefs for theme extraction
            if row.spotlight_brief:
                all_briefs.append(row.spotlight_brief)
        
        # Fetch participants for calls
        if conversation_keys:
            participants_map = self._get_gong_participants(conversation_keys[:10])  # Top 10 recent calls
            for call in calls[:10]:
                if call.conversation_key and call.conversation_key in participants_map:
                    call.participants = participants_map[call.conversation_key]
        
        # Fetch transcript snippets for top 5 calls
        for call in calls[:5]:
            if call.conversation_key:
                snippet = self._get_transcript_snippet(call.conversation_key)
                if snippet:
                    call.transcript_snippet = snippet
        
        # Calculate averages
        avg_talk_ratio = sum(talk_ratios) / len(talk_ratios) if talk_ratios else None
        avg_interactivity = sum(interactivities) / len(interactivities) if interactivities else None
        avg_patience = sum(patiences) / len(patiences) if patiences else None
        avg_question_rate = sum(question_rates) / len(question_rates) if question_rates else None
        
        # Calculate days since last call
        days_since_last = None
        if last_call_date:
            try:
                last_date = datetime.fromisoformat(last_call_date.replace('Z', '+00:00').split('+')[0])
                days_since_last = (datetime.now() - last_date).days
            except:
                pass
        
        # Extract key themes from briefs
        key_themes = self._extract_themes(all_briefs)
        
        return GongSummaryData(
            total_calls=len(calls),
            total_duration_minutes=total_duration,
            avg_talk_ratio=avg_talk_ratio,
            avg_interactivity=avg_interactivity,
            avg_patience=avg_patience,
            avg_question_rate=avg_question_rate,
            first_call_date=first_call_date,
            last_call_date=last_call_date,
            days_since_last_call=days_since_last,
            key_themes=key_themes,
            recent_calls=calls[:10]
        )
    
    def _get_gong_participants(self, conversation_keys: List[str]) -> Dict[str, List[GongParticipant]]:
        """Fetch participants for multiple conversations."""
        
        if not conversation_keys:
            return {}
        
        keys_str = ", ".join([f"'{k}'" for k in conversation_keys])
        
        query = f"""
        SELECT 
            CONVERSATION_KEY as conversation_key,
            NAME as name,
            EMAIL_ADDRESS as email,
            AFFILIATION as affiliation,
            SPEAKER_ID as speaker_id
        FROM `{self.PROJECT_ID}.gong.CONVERSATION_PARTICIPANTS`
        WHERE CONVERSATION_KEY IN ({keys_str})
          AND IS_DELETED = FALSE
        """
        
        results = self.client.query(query).result()
        
        participants_map: Dict[str, List[GongParticipant]] = {}
        for row in results:
            key = row.conversation_key
            if key not in participants_map:
                participants_map[key] = []
            
            participants_map[key].append(GongParticipant(
                name=row.name,
                email=row.email,
                affiliation=row.affiliation,
                speaker_id=str(row.speaker_id) if row.speaker_id else None
            ))
        
        return participants_map
    
    def _get_transcript_snippet(self, conversation_key: str) -> Optional[str]:
        """Get first portion of transcript for a call."""
        
        query = f"""
        SELECT TRANSCRIPT
        FROM `{self.PROJECT_ID}.gong.CALL_TRANSCRIPTS`
        WHERE CONVERSATION_KEY = @conversation_key
          AND IS_DELETED = FALSE
        LIMIT 1
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("conversation_key", "STRING", conversation_key)
            ]
        )
        
        results = self.client.query(query, job_config=job_config).result()
        
        for row in results:
            if row.TRANSCRIPT:
                # Return first 1000 characters of transcript
                return row.TRANSCRIPT[:1000] + "..." if len(row.TRANSCRIPT) > 1000 else row.TRANSCRIPT
        
        return None
    
    def _extract_themes(self, briefs: List[str]) -> List[str]:
        """Extract common themes from call spotlight briefs."""
        # Simple keyword extraction - in production could use NLP
        themes = []
        theme_keywords = {
            "POC": ["poc", "proof of concept", "pilot", "trial"],
            "Integration": ["integration", "implement", "setup", "configure"],
            "Pricing": ["pricing", "cost", "budget", "contract"],
            "Technical Review": ["technical", "architecture", "demo", "walkthrough"],
            "Discovery": ["discovery", "requirements", "needs", "use case"],
            "Follow-up": ["follow up", "next steps", "action items"],
            "Onboarding": ["onboarding", "training", "getting started"],
            "Support": ["support", "issue", "problem", "bug"],
        }
        
        all_text = " ".join(briefs).lower()
        
        for theme, keywords in theme_keywords.items():
            if any(kw in all_text for kw in keywords):
                themes.append(theme)
        
        return themes[:5]  # Return top 5 themes
    
    def _select_most_relevant_opportunity(self, opportunities: List[OpportunityData]) -> Optional[OpportunityData]:
        """
        Select the most relevant opportunity based on recency and timing.
        
        Priority logic:
        1. If there's a recently closed (won) opportunity (within 60 days), show that - 
           it represents the current deal context
        2. If there's an open opportunity closing within 90 days, show that - 
           it's the active focus
        3. If the next open opportunity is far out (>90 days) but there's a recent 
           closed one, show the closed one
        4. Otherwise, show the nearest open opportunity
        """
        if not opportunities:
            return None
        
        today = datetime.now().date()
        
        # Parse close dates and categorize opportunities
        recently_closed_won = []
        upcoming_open = []
        future_open = []
        
        for opp in opportunities:
            if not opp.close_date:
                continue
                
            try:
                close_date_str = opp.close_date.split('T')[0].split('+')[0].split(' ')[0]
                close_date = datetime.fromisoformat(close_date_str).date()
                days_from_today = (close_date - today).days
            except Exception:
                continue
            
            if opp.is_won and opp.is_closed:
                # Recently closed won (within past 60 days)
                if -60 <= days_from_today <= 0:
                    recently_closed_won.append((opp, days_from_today))
            elif not opp.is_closed:
                # Open opportunities
                if days_from_today <= 90:
                    upcoming_open.append((opp, days_from_today))
                else:
                    future_open.append((opp, days_from_today))
        
        # Priority 1: Open opportunity closing within 90 days (most urgent)
        if upcoming_open:
            # Sort by close date ascending (soonest first)
            upcoming_open.sort(key=lambda x: x[1])
            return upcoming_open[0][0]
        
        # Priority 2: Recently closed won opportunity
        if recently_closed_won:
            # Sort by close date descending (most recent first)
            recently_closed_won.sort(key=lambda x: x[1], reverse=True)
            return recently_closed_won[0][0]
        
        # Priority 3: Future open opportunity
        if future_open:
            # Sort by close date ascending (soonest first)
            future_open.sort(key=lambda x: x[1])
            return future_open[0][0]
        
        # Fallback: return the first opportunity
        return opportunities[0] if opportunities else None
    
    def _lookup_pendo_account_by_name(self, account_name: str) -> Optional[str]:
        """Look up Pendo account ID by matching account name in agent_name field."""
        
        # 1. Try exact match first
        query = f"""
        SELECT id as pendo_account_id
        FROM `{self.PROJECT_ID}.pendo.account_history`
        WHERE LOWER(agent_name) = LOWER(@account_name)
        ORDER BY last_visit DESC
        LIMIT 1
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("account_name", "STRING", account_name)
            ]
        )
        
        results = self.client.query(query, job_config=job_config).result()
        
        for row in results:
            return row.pendo_account_id
        
        # 2. Try fuzzy match - Pendo name contains Salesforce name
        fuzzy_query = f"""
        SELECT id as pendo_account_id, agent_name
        FROM `{self.PROJECT_ID}.pendo.account_history`
        WHERE LOWER(agent_name) LIKE LOWER(@name_pattern)
          AND last_visit >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 365 DAY)
        ORDER BY last_visit DESC
        LIMIT 1
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("name_pattern", "STRING", f"%{account_name}%")
            ]
        )
        
        results = self.client.query(fuzzy_query, job_config=job_config).result()
        
        for row in results:
            return row.pendo_account_id
        
        # 3. Try reverse fuzzy match - Salesforce name contains Pendo name
        # This handles cases like "Disney Parks, Experiences and Products" -> "Disney"
        # Extract first word or key part of the name
        first_word = account_name.split()[0] if account_name else ""
        if first_word and len(first_word) >= 3:
            reverse_query = f"""
            SELECT id as pendo_account_id, agent_name
            FROM `{self.PROJECT_ID}.pendo.account_history`
            WHERE (LOWER(agent_name) = LOWER(@first_word) 
                   OR LOWER(agent_name) LIKE CONCAT('acct-%', LOWER(@first_word), '%'))
              AND last_visit >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 365 DAY)
            ORDER BY last_visit DESC
            LIMIT 1
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("first_word", "STRING", first_word)
                ]
            )
            
            results = self.client.query(reverse_query, job_config=job_config).result()
            
            for row in results:
                return row.pendo_account_id
        
        return None
    
    def _get_pendo_usage(self, pendo_account_id: str) -> Optional[PendoUsageData]:
        """Fetch comprehensive Pendo product usage data."""
        
        # Get overall usage stats
        usage_query = f"""
        SELECT 
            SUM(num_events) as total_events,
            SUM(num_minutes) as total_minutes,
            COUNT(DISTINCT visitor_id) as unique_visitors,
            MIN(timestamp) as first_activity,
            MAX(timestamp) as last_activity,
            COUNT(DISTINCT DATE(timestamp)) as total_active_days
        FROM `{self.PROJECT_ID}.pendo.event`
        WHERE account_id = @account_id
          AND timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("account_id", "STRING", pendo_account_id)
            ]
        )
        
        results = self.client.query(usage_query, job_config=job_config).result()
        
        base_data = None
        for row in results:
            base_data = row
            break
        
        if not base_data or not base_data.total_events:
            return None
        
        # Get active days in last 7 and 30 days
        active_days_query = f"""
        SELECT 
            COUNT(DISTINCT CASE WHEN timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY) 
                  THEN DATE(timestamp) END) as active_days_7,
            COUNT(DISTINCT CASE WHEN timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY) 
                  THEN DATE(timestamp) END) as active_days_30
        FROM `{self.PROJECT_ID}.pendo.event`
        WHERE account_id = @account_id
        """
        
        active_results = self.client.query(active_days_query, job_config=job_config).result()
        active_days_7 = 0
        active_days_30 = 0
        for row in active_results:
            active_days_7 = row.active_days_7 or 0
            active_days_30 = row.active_days_30 or 0
        
        # Get recent visitors (who last used it)
        recent_visitors = self._get_recent_pendo_visitors(pendo_account_id)
        
        # Get top features
        top_features = self._get_top_features(pendo_account_id)
        
        # Get top pages
        top_pages = self._get_top_pages(pendo_account_id)
        
        # Get weekly trend
        weekly_trend = self._get_weekly_trend(pendo_account_id)
        
        # Calculate days since last activity
        days_since_last = None
        if base_data.last_activity:
            try:
                last_date = base_data.last_activity
                if hasattr(last_date, 'date'):
                    days_since_last = (datetime.now().date() - last_date.date()).days
            except:
                pass
        
        # Calculate averages
        avg_daily = None
        if base_data.total_active_days and base_data.total_active_days > 0:
            avg_daily = base_data.total_minutes / base_data.total_active_days
        
        return PendoUsageData(
            total_events=base_data.total_events or 0,
            total_minutes=base_data.total_minutes or 0,
            unique_visitors=base_data.unique_visitors or 0,
            first_activity=str(base_data.first_activity) if base_data.first_activity else None,
            last_activity=str(base_data.last_activity) if base_data.last_activity else None,
            days_since_last_activity=days_since_last,
            active_days_last_30=active_days_30,
            active_days_last_7=active_days_7,
            avg_daily_minutes=avg_daily,
            recent_visitors=recent_visitors,
            top_features=top_features,
            top_pages=top_pages,
            weekly_trend=weekly_trend
        )
    
    def _get_recent_pendo_visitors(self, pendo_account_id: str) -> List[PendoVisitorActivity]:
        """Get most recent visitors to the platform with their details."""
        
        # Use subquery to aggregate events first, then join with deduplicated visitor info
        # This prevents row multiplication from visitor_history having multiple rows per visitor
        query = f"""
        WITH visitor_stats AS (
            SELECT 
                visitor_id,
                SUM(num_events) as total_events,
                SUM(num_minutes) as total_minutes,
                MAX(timestamp) as last_visit,
                MIN(timestamp) as first_visit,
                COUNT(DISTINCT DATE(timestamp)) as visit_count
            FROM `{self.PROJECT_ID}.pendo.event`
            WHERE account_id = @account_id
              AND timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
            GROUP BY visitor_id
        ),
        visitor_info AS (
            SELECT 
                id,
                agent_email,
                agent_full_name,
                ROW_NUMBER() OVER (PARTITION BY id ORDER BY last_updated_at DESC) as rn
            FROM `{self.PROJECT_ID}.pendo.visitor_history`
        )
        SELECT 
            vs.visitor_id,
            vi.agent_email as email,
            vi.agent_full_name as display_name,
            vs.total_events,
            vs.total_minutes,
            vs.last_visit,
            vs.first_visit,
            vs.visit_count
        FROM visitor_stats vs
        LEFT JOIN visitor_info vi ON vs.visitor_id = vi.id AND vi.rn = 1
        ORDER BY vs.last_visit DESC
        LIMIT 10
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("account_id", "STRING", pendo_account_id)
            ]
        )
        
        results = self.client.query(query, job_config=job_config).result()
        
        visitors = []
        for row in results:
            visitors.append(PendoVisitorActivity(
                visitor_id=row.visitor_id,
                email=row.email,
                display_name=row.display_name,
                total_events=row.total_events or 0,
                total_minutes=row.total_minutes or 0,
                last_visit=str(row.last_visit) if row.last_visit else None,
                first_visit=str(row.first_visit) if row.first_visit else None,
                visit_count=row.visit_count or 0
            ))
        
        return visitors
    
    def _get_top_features(self, pendo_account_id: str) -> List[PendoFeatureUsage]:
        """Get top features used by an account with feature names."""
        
        query = f"""
        SELECT 
            fe.feature_id,
            fh.name as feature_name,
            SUM(fe.num_events) as event_count,
            COUNT(DISTINCT fe.visitor_id) as unique_users,
            MAX(fe.timestamp) as last_used
        FROM `{self.PROJECT_ID}.pendo.feature_event` fe
        LEFT JOIN `{self.PROJECT_ID}.pendo.feature_history` fh ON fe.feature_id = fh.id
        WHERE fe.account_id = @account_id
          AND fe.timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
        GROUP BY fe.feature_id, fh.name
        ORDER BY event_count DESC
        LIMIT 15
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("account_id", "STRING", pendo_account_id)
            ]
        )
        
        results = self.client.query(query, job_config=job_config).result()
        
        features = []
        for row in results:
            features.append(PendoFeatureUsage(
                feature_id=row.feature_id,
                feature_name=row.feature_name,
                event_count=row.event_count or 0,
                unique_users=row.unique_users or 0,
                last_used=str(row.last_used) if row.last_used else None
            ))
        
        return features
    
    def _get_top_pages(self, pendo_account_id: str) -> List[PendoPageUsage]:
        """Get top pages viewed by an account."""
        
        query = f"""
        SELECT 
            pe.page_id,
            ph.name as page_name,
            SUM(pe.num_events) as view_count,
            COUNT(DISTINCT pe.visitor_id) as unique_viewers,
            SUM(pe.num_minutes) as total_minutes
        FROM `{self.PROJECT_ID}.pendo.page_event` pe
        LEFT JOIN `{self.PROJECT_ID}.pendo.page_history` ph ON pe.page_id = ph.id
        WHERE pe.account_id = @account_id
          AND pe.timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
        GROUP BY pe.page_id, ph.name
        ORDER BY view_count DESC
        LIMIT 10
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("account_id", "STRING", pendo_account_id)
            ]
        )
        
        results = self.client.query(query, job_config=job_config).result()
        
        pages = []
        for row in results:
            pages.append(PendoPageUsage(
                page_id=row.page_id,
                page_name=row.page_name,
                view_count=row.view_count or 0,
                unique_viewers=row.unique_viewers or 0,
                total_minutes=row.total_minutes or 0
            ))
        
        return pages
    
    def _get_weekly_trend(self, pendo_account_id: str) -> List[Dict[str, Any]]:
        """Get weekly usage trend for the last 12 weeks."""
        
        query = f"""
        SELECT 
            DATE_TRUNC(DATE(timestamp), WEEK) as week_start,
            SUM(num_events) as events,
            SUM(num_minutes) as minutes,
            COUNT(DISTINCT visitor_id) as unique_visitors
        FROM `{self.PROJECT_ID}.pendo.event`
        WHERE account_id = @account_id
          AND timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 84 DAY)
        GROUP BY week_start
        ORDER BY week_start DESC
        LIMIT 12
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("account_id", "STRING", pendo_account_id)
            ]
        )
        
        results = self.client.query(query, job_config=job_config).result()
        
        trend = []
        for row in results:
            trend.append({
                "week_start": str(row.week_start) if row.week_start else None,
                "events": row.events or 0,
                "minutes": row.minutes or 0,
                "unique_visitors": row.unique_visitors or 0
            })
        
        return trend
    
    def _get_fullstory_issues(self, domain: str) -> List[UserIssueEvent]:
        """Fetch user issue events from FullStory (errors, exceptions, frustration signals)."""
        
        # FullStory org ID for constructing recording URLs
        FULLSTORY_ORG_ID = "o-1HS6XF-na1"
        
        # FullStory session recording retention is typically 14-30 days depending on plan.
        # We use 14 days to ensure "View Recording" links actually work.
        # Older sessions may still have event data but the recordings are no longer available.
        FULLSTORY_RECORDING_RETENTION_DAYS = 14
        
        # FullStory URL format: https://app.fullstory.com/ui/ORG_ID/session/DEVICE_ID:SESSION_ID
        # indv_id is the individual/device identifier, session_id is the session
        # Target event types that indicate issues:
        # - exception: JavaScript errors
        # - thrash: Mouse thrashing (frustration)
        # - highlight_error: Error highlights
        # - abandon: Page abandonment
        # - change_error: Form change errors
        # - console_message_error: Console errors
        query = f"""
        SELECT 
            se.session_id,
            se.user_id,
            se.indv_id,
            se.user_email as email,
            se.user_display_name as display_name,
            se.event_type,
            se.event_var_error_kind,
            SUBSTR(se.page_url, 1, 300) as page_url,
            se.event_start,
            COUNT(*) as issue_count
        FROM `{self.PROJECT_ID}.fullstory.segment_event` se
        WHERE se.event_type IN ('exception', 'thrash', 'highlight_error', 'abandon', 'change_error', 'console_message_error')
          AND se.event_start > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {FULLSTORY_RECORDING_RETENTION_DAYS} DAY)
          AND LOWER(se.user_email) LIKE @domain_pattern
        GROUP BY se.session_id, se.user_id, se.indv_id, se.user_email, se.user_display_name, 
                 se.event_type, se.event_var_error_kind, page_url, se.event_start
        ORDER BY se.event_start DESC
        LIMIT 30
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("domain_pattern", "STRING", f"%@{domain.lower()}%")
            ]
        )
        
        try:
            results = self.client.query(query, job_config=job_config).result()
            
            issues = []
            for row in results:
                # Map event_type to user-friendly issue type
                event_type = row.event_type or ""
                if event_type == "exception":
                    issue_type = "error"
                elif event_type == "thrash":
                    issue_type = "frustrated"
                elif event_type == "highlight_error":
                    issue_type = "error"
                elif event_type == "abandon":
                    issue_type = "abandoned"
                elif event_type in ("change_error", "console_message_error"):
                    issue_type = "error"
                else:
                    issue_type = "unknown"
                
                # Extract page context from URL
                page_context = None
                if row.page_url:
                    url = row.page_url.lower()
                    if "tasks/create" in url:
                        if "template_evaluation" in url:
                            page_context = "Creating evaluation task"
                        else:
                            page_context = "Creating task"
                    elif "models/" in url:
                        page_context = "Viewing model"
                    elif "prompt-hub" in url or "playground" in url:
                        page_context = "Prompt playground"
                    elif "home" in url:
                        page_context = "Home page"
                    elif "settings" in url:
                        page_context = "Account settings"
                    elif "datasets" in url:
                        page_context = "Datasets"
                    elif "evals" in url:
                        page_context = "Evaluations"
                    elif "traces" in url:
                        page_context = "Tracing"
                
                # Construct recording URL using FullStory format: /ui/ORG_ID/session/DEVICE_ID:SESSION_ID
                # indv_id is the individual/device identifier used in FullStory URLs
                recording_url = None
                if row.session_id and row.indv_id:
                    recording_url = f"https://app.fullstory.com/ui/{FULLSTORY_ORG_ID}/session/{row.indv_id}:{row.session_id}"
                elif row.session_id and row.user_id:
                    # Fallback to user_id if indv_id not available
                    recording_url = f"https://app.fullstory.com/ui/{FULLSTORY_ORG_ID}/session/{row.user_id}:{row.session_id}"
                elif row.session_id:
                    # Last resort - just session_id
                    recording_url = f"https://app.fullstory.com/ui/{FULLSTORY_ORG_ID}/session/{row.session_id}"
                
                issues.append(UserIssueEvent(
                    issue_type=issue_type,
                    error_kind=row.event_var_error_kind,
                    page_url=row.page_url,
                    page_context=page_context,
                    user_email=row.email,
                    user_name=row.display_name,
                    session_id=str(row.session_id) if row.session_id else None,
                    recording_url=recording_url,
                    timestamp=str(row.event_start) if row.event_start else None,
                    count=row.issue_count or 1
                ))
            
            return issues
        except Exception:
            return []
    
    def _get_adoption_milestones(self, pendo_account_id: str) -> List[AdoptionMilestone]:
        """Fetch adoption milestones (projects, traces, experiments, evals, prompts) for an account."""
        
        if not self.client or not pendo_account_id:
            return []
        
        # Define milestones and the features/pages that indicate completion
        # Note: These names are based on actual Pendo feature/page names in the data
        milestone_definitions = {
            "created_project": {
                "display": "Created Projects",
                "features": ["Create New Space - Finish Button", "Dashboard - create dashboard", "NEW: Create Blank Dashboard - Dashboard Listing"],
                "pages": ["Space - Create New", "Project - Space Listing", "Project - Specific Page", "Space Overview [Left Nav]"]
            },
            "sent_traces": {
                "display": "Sent Trace Data",
                "features": ["Send in Data", "Send in Data Button - Landing Page", "Click Send in Data"],
                "pages": ["LLM Tracing Tab", "Performance Tracing - Model Tab", "llamatrace-app-phoenix-all", "llamatrace-app-phoenix"]
            },
            "ran_experiment": {
                "display": "Ran Experiments",
                "features": ["Datasets-&-Experiments"],
                "pages": ["Experiments", "Datasets"]
            },
            "created_evals": {
                "display": "Created Evaluations",
                "features": ["Evals & Tasks Link", "Eval Builder - button", "Suggest Eval (Generative)", "Write a Custom Eval (Generative)"],
                "pages": ["Online Evals", "Online Evals Task Tab", "Trevor Tasks Test"]
            },
            "used_prompts": {
                "display": "Used Prompt Playground",
                "features": [],
                "pages": ["Prompt Playground", "arize-demo-generative-llm-search-and-retrieval-debug-prompts"]
            }
        }
        
        milestones = []
        
        try:
            # Query feature events for this account
            feature_query = f"""
            SELECT 
                fh.name as feature_name,
                COUNT(*) as event_count,
                MIN(fe.day) as first_use,
                MAX(fe.day) as last_use
            FROM `{self.PROJECT_ID}.pendo.feature_event` fe
            JOIN `{self.PROJECT_ID}.pendo.feature_history` fh ON fe.feature_id = fh.id
            WHERE fe.account_id = @account_id
            GROUP BY fh.name
            """
            
            feature_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("account_id", "STRING", pendo_account_id)
                ]
            )
            
            feature_results = {row.feature_name: {
                "count": row.event_count,
                "first": str(row.first_use) if row.first_use else None,
                "last": str(row.last_use) if row.last_use else None
            } for row in self.client.query(feature_query, job_config=feature_config).result()}
            
            # Query page events for this account
            page_query = f"""
            SELECT 
                ph.name as page_name,
                COUNT(*) as visit_count,
                MIN(pe.day) as first_visit,
                MAX(pe.day) as last_visit
            FROM `{self.PROJECT_ID}.pendo.page_event` pe
            JOIN `{self.PROJECT_ID}.pendo.page_history` ph ON pe.page_id = ph.id
            WHERE pe.account_id = @account_id
            GROUP BY ph.name
            """
            
            page_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("account_id", "STRING", pendo_account_id)
                ]
            )
            
            page_results = {row.page_name: {
                "count": row.visit_count,
                "first": str(row.first_visit) if row.first_visit else None,
                "last": str(row.last_visit) if row.last_visit else None
            } for row in self.client.query(page_query, job_config=page_config).result()}
            
            # Check each milestone
            for milestone_id, definition in milestone_definitions.items():
                total_count = 0
                first_date = None
                last_date = None
                
                # Check feature usage
                for feature_name in definition["features"]:
                    if feature_name in feature_results:
                        data = feature_results[feature_name]
                        total_count += data["count"]
                        if data["first"] and (not first_date or data["first"] < first_date):
                            first_date = data["first"]
                        if data["last"] and (not last_date or data["last"] > last_date):
                            last_date = data["last"]
                
                # Check page visits
                for page_name in definition["pages"]:
                    if page_name in page_results:
                        data = page_results[page_name]
                        total_count += data["count"]
                        if data["first"] and (not first_date or data["first"] < first_date):
                            first_date = data["first"]
                        if data["last"] and (not last_date or data["last"] > last_date):
                            last_date = data["last"]
                
                milestones.append(AdoptionMilestone(
                    name=definition["display"],
                    completed=total_count > 0,
                    count=total_count,
                    first_date=first_date[:10] if first_date else None,
                    last_date=last_date[:10] if last_date else None
                ))
            
            return milestones
        except Exception:
            return []
    
    def _build_deal_summary(self, gong: GongSummaryData) -> Optional[DealSummary]:
        """Build a deal summary by analyzing Gong call data."""
        
        if not gong or gong.total_calls == 0:
            return None
        
        # Analyze call spotlight data to build summary
        all_briefs = []
        all_next_steps = []
        all_outcomes = []
        all_key_points = []
        
        for call in gong.recent_calls:
            if call.spotlight_brief:
                all_briefs.append(call.spotlight_brief)
            if call.spotlight_next_steps:
                # Parse next_steps which might be a JSON array string
                try:
                    if call.spotlight_next_steps.startswith('['):
                        parsed_steps = json.loads(call.spotlight_next_steps)
                        if isinstance(parsed_steps, list):
                            all_next_steps.extend([str(s) for s in parsed_steps if s])
                        else:
                            all_next_steps.append(str(parsed_steps))
                    else:
                        all_next_steps.append(call.spotlight_next_steps)
                except (json.JSONDecodeError, TypeError, AttributeError):
                    all_next_steps.append(call.spotlight_next_steps)
            if call.spotlight_outcome:
                all_outcomes.append(call.spotlight_outcome)
            if call.spotlight_key_points:
                # Ensure we only extend with string items
                for kp in call.spotlight_key_points:
                    if isinstance(kp, str):
                        all_key_points.append(kp)
                    elif isinstance(kp, list):
                        all_key_points.extend([str(item) for item in kp if item])
        
        # Build current state from most recent calls
        current_state = ""
        if all_briefs:
            # Use most recent call brief as primary state, with context from others
            current_state = all_briefs[0]
            if len(all_briefs) > 1:
                current_state = f"{all_briefs[0]} Previously: {all_briefs[1][:200]}..."
        
        # Identify blockers from call content
        blockers = []
        blocker_keywords = ["blocker", "block", "concern", "issue", "problem", "challenge", 
                          "delay", "risk", "obstacle", "waiting", "pending", "security review",
                          "legal", "procurement", "budget", "approval", "timeline"]
        
        # Ensure all items are strings before joining
        all_text_items = []
        for item in all_briefs + all_key_points:
            if isinstance(item, str):
                all_text_items.append(item)
            elif isinstance(item, list):
                all_text_items.extend([str(x) for x in item if x])
        combined_text = " ".join(all_text_items).lower()
        for call in gong.recent_calls:
            if call.transcript_snippet:
                # Handle transcript_snippet as either string or list of transcript objects
                if isinstance(call.transcript_snippet, str):
                    combined_text += " " + call.transcript_snippet.lower()
                elif isinstance(call.transcript_snippet, list):
                    # Extract text from transcript objects
                    for segment in call.transcript_snippet:
                        if isinstance(segment, dict):
                            for sentence in segment.get('sentences', []):
                                if isinstance(sentence, dict) and sentence.get('text'):
                                    combined_text += " " + sentence['text'].lower()
        
        for keyword in blocker_keywords:
            if keyword in combined_text:
                # Extract sentences containing blocker keywords
                for text_item in all_text_items:
                    if keyword in text_item.lower():
                        blockers.append(text_item[:200])
                        break
        
        blockers = list(set(blockers))[:5]  # Dedupe and limit
        
        # Risk factors
        risks = []
        if gong.days_since_last_call and gong.days_since_last_call > 30:
            risks.append(f"No calls in {gong.days_since_last_call} days - engagement may be cooling")
        if gong.avg_talk_ratio and gong.avg_talk_ratio > 0.7:
            risks.append("High talk ratio - may need more discovery/listening")
        
        # Sentiment analysis (simple heuristic)
        sentiment = "neutral"
        positive_words = ["excited", "great", "love", "perfect", "excellent", "impressed", "happy"]
        negative_words = ["concerned", "worried", "frustrated", "disappointed", "issue", "problem"]
        
        pos_count = sum(1 for w in positive_words if w in combined_text)
        neg_count = sum(1 for w in negative_words if w in combined_text)
        
        if pos_count > neg_count + 2:
            sentiment = "positive"
        elif neg_count > pos_count + 1:
            sentiment = "concerned"
        
        return DealSummary(
            current_state=current_state[:500] if current_state else "No recent call summaries available",
            key_topics_discussed=gong.key_themes,
            blockers_identified=blockers,
            next_steps_from_calls=all_next_steps[:8],  # Limit to 8 most recent next steps
            champion_sentiment=sentiment,
            risk_factors=risks
        )
    
    def _build_user_behavior_analysis(
        self, 
        pendo: Optional[PendoUsageData],
        fullstory_issues: List[UserIssueEvent] = None,
        adoption_milestones: List[AdoptionMilestone] = None,
        gong: Optional[GongSummaryData] = None,
        account: Optional[SalesforceAccountData] = None,
        manual_competitors: Optional[List[str]] = None
    ) -> Optional[UserBehaviorAnalysis]:
        """Analyze user behavior patterns from Pendo data and FullStory issues."""
        
        if not pendo and not fullstory_issues:
            return None
        
        fullstory_issues = fullstory_issues or []
        
        # Build summary of what users are doing
        summary_parts = []
        
        if pendo and pendo.unique_visitors:
            summary_parts.append(f"{pendo.unique_visitors} users have accessed the platform")
        
        if pendo and pendo.total_minutes:
            avg_per_user = pendo.total_minutes / max(pendo.unique_visitors, 1)
            summary_parts.append(f"averaging {avg_per_user:.0f} minutes each over the last 90 days")
        
        if pendo and pendo.active_days_last_7 > 0:
            summary_parts.append(f"with activity on {pendo.active_days_last_7} of the last 7 days")
        
        summary = ". ".join(summary_parts) + "." if summary_parts else "Limited usage data available."
        
        # Build hypothesis based on top features
        hypothesis = ""
        key_workflows = []
        
        if pendo and pendo.top_features:
            feature_names = [f.feature_name or f.feature_id for f in pendo.top_features[:5]]
            key_workflows = feature_names
            
            # Create hypothesis based on feature usage patterns
            if any("trace" in (f.feature_name or "").lower() for f in pendo.top_features[:3]):
                hypothesis = "Users appear focused on tracing and debugging LLM/ML workflows."
            elif any("eval" in (f.feature_name or "").lower() for f in pendo.top_features[:3]):
                hypothesis = "Users are primarily using evaluation features, suggesting active model testing."
            elif any("dashboard" in (f.feature_name or "").lower() for f in pendo.top_features[:3]):
                hypothesis = "Users are monitoring dashboards, indicating production usage tracking."
            else:
                hypothesis = f"Users are exploring various features including {', '.join(feature_names[:3])}."
        
        # Determine engagement level
        engagement_level = "unknown"
        if pendo:
            if pendo.active_days_last_7 >= 5:
                engagement_level = "high"
            elif pendo.active_days_last_7 >= 2:
                engagement_level = "medium"
            elif pendo.active_days_last_30 >= 5:
                engagement_level = "medium"
            elif pendo.total_events > 0:
                engagement_level = "low"
        
        # Build prescriptive recommendations
        recommendations = self._build_prescriptive_recommendations(
            pendo=pendo,
            engagement_level=engagement_level,
            gong=gong,
            account=account,
            fullstory_issues=fullstory_issues,
            key_workflows=key_workflows,
            manual_competitors=manual_competitors
        )
        
        # Process user issues (errors, dead clicks, frustrated moments)
        critical_issues = []
        issues_summary = None
        
        if pendo and pendo.days_since_last_activity and pendo.days_since_last_activity > 30:
            critical_issues.append({
                "type": "Disengagement",
                "description": f"No platform activity in {pendo.days_since_last_activity} days"
            })
        
        if fullstory_issues:
            # Group issues by type and context
            issue_by_type = {"dead_click": 0, "error": 0, "frustrated": 0}
            issue_by_context = {}
            affected_users = set()
            
            for issue in fullstory_issues:
                issue_by_type[issue.issue_type] = issue_by_type.get(issue.issue_type, 0) + issue.count
                if issue.page_context:
                    issue_by_context[issue.page_context] = issue_by_context.get(issue.page_context, 0) + issue.count
                if issue.user_email:
                    affected_users.add(issue.user_email)
            
            # Build detailed issues summary
            total_issues = sum(issue_by_type.values())
            summary_parts = []
            
            if total_issues > 0:
                summary_parts.append(f"{total_issues} user experience issues detected in the last 90 days")
                
                # Break down by type
                type_details = []
                if issue_by_type.get("dead_click", 0) > 0:
                    type_details.append(f"{issue_by_type['dead_click']} dead clicks (users clicking non-interactive elements)")
                if issue_by_type.get("error", 0) > 0:
                    type_details.append(f"{issue_by_type['error']} errors")
                if issue_by_type.get("frustrated", 0) > 0:
                    type_details.append(f"{issue_by_type['frustrated']} frustrated moments")
                
                if type_details:
                    summary_parts.append(f"Including: {'; '.join(type_details)}")
                
                # Add context about where issues occur
                if issue_by_context:
                    top_contexts = sorted(issue_by_context.items(), key=lambda x: x[1], reverse=True)[:3]
                    context_str = ", ".join([f"{ctx} ({cnt})" for ctx, cnt in top_contexts])
                    summary_parts.append(f"Most issues occur in: {context_str}")
                
                # Add affected users
                if affected_users:
                    summary_parts.append(f"Affecting {len(affected_users)} user(s): {', '.join(list(affected_users)[:3])}")
                
                issues_summary = ". ".join(summary_parts) + "."
                
                # Add to critical issues if significant
                if issue_by_type.get("dead_click", 0) > 5:
                    top_context = max(issue_by_context.items(), key=lambda x: x[1])[0] if issue_by_context else "unknown area"
                    critical_issues.append({
                        "type": "UX Confusion",
                        "description": f"Multiple dead clicks detected - users may be confused in {top_context}"
                    })
                
                if issue_by_type.get("error", 0) > 3:
                    critical_issues.append({
                        "type": "Platform Errors",
                        "description": f"{issue_by_type['error']} errors encountered by users"
                    })
        
        return UserBehaviorAnalysis(
            summary=summary,
            hypothesis=hypothesis,
            key_workflows_used=key_workflows,
            adoption_milestones=adoption_milestones or [],
            critical_issues=critical_issues,
            user_issues=fullstory_issues[:15],
            issues_summary=issues_summary,
            engagement_level=engagement_level,
            recommendations=recommendations
        )
    
    def _build_prescriptive_recommendations(
        self,
        pendo: Optional[PendoUsageData],
        engagement_level: str,
        gong: Optional[GongSummaryData],
        account: Optional[SalesforceAccountData],
        fullstory_issues: List[UserIssueEvent],
        key_workflows: List[str],
        manual_competitors: Optional[List[str]] = None
    ) -> List[str]:
        """Build prescriptive recommendations with competitive insights, next steps, and internal contacts."""
        
        recommendations = []
        
        # --- ENGAGEMENT-BASED RECOMMENDATIONS ---
        if engagement_level == "low":
            recommendations.append({
                "category": "Re-engagement",
                "title": "Schedule adoption check-in",
                "description": "User engagement is low. Schedule a 30-min call to understand adoption blockers and demonstrate value."
            })
        
        if pendo and pendo.days_since_last_activity and pendo.days_since_last_activity > 14:
            recommendations.append({
                "category": "Re-engagement",
                "title": f"Re-engage after {pendo.days_since_last_activity} days of inactivity",
                "description": "Send a personalized check-in email highlighting new features or offering a quick sync to troubleshoot any issues."
            })
        
        if pendo and pendo.unique_visitors == 1 and pendo.total_events and pendo.total_events > 100:
            recommendations.append({
                "category": "Expansion",
                "title": "Expand beyond single power user",
                "description": "One user is highly engaged. Propose a team workshop or lunch-and-learn to drive broader adoption across the org."
            })
        
        # --- WORKFLOW-BASED RECOMMENDATIONS ---
        if key_workflows:
            workflow_lower = " ".join(key_workflows).lower()
            if "trace" in workflow_lower or "tracing" in workflow_lower:
                recommendations.append({
                    "category": "Deepening",
                    "title": "Expand from tracing to evaluations",
                    "description": "Users are focused on tracing. Introduce our evaluation capabilities to help them measure LLM quality systematically."
                })
            elif "dashboard" in workflow_lower:
                recommendations.append({
                    "category": "Deepening", 
                    "title": "Move from monitoring to proactive alerting",
                    "description": "Users are monitoring dashboards. Set up custom monitors and alerts to catch issues before they impact production."
                })
        
        # --- COMPETITIVE DIFFERENTIATORS (from call mentions + manual input) ---
        # Analyze Gong call content for actual competitor mentions
        mentioned_competitors = self._detect_competitors_in_calls(gong)
        
        # Add manual competitors (user-specified known competitors for this deal)
        if manual_competitors:
            for comp in manual_competitors:
                # Normalize competitor name to match our known competitors
                comp_normalized = comp.strip()
                if comp_normalized and comp_normalized not in mentioned_competitors:
                    mentioned_competitors[comp_normalized] = [{
                        "call": "Manually flagged",
                        "context": "Known competitor in this deal (manually added)"
                    }]
        
        if mentioned_competitors:
            # Build competitive messaging for all competitors (detected + manual)
            competitive_messaging = self._build_competitive_messaging(
                mentioned_competitors=mentioned_competitors,
                key_workflows=key_workflows,
                gong=gong
            )
            
            if competitive_messaging:
                title = "Competitive positioning"
                if manual_competitors:
                    title += " (includes manually flagged competitors)"
                recommendations.append({
                    "category": "Competitive",
                    "title": title,
                    "competitive_messaging": competitive_messaging
                })
        
        # --- INTERNAL CONTACTS FROM GONG ---
        internal_contacts = []
        if gong and gong.recent_calls:
            arize_participants = set()
            for call in gong.recent_calls[:10]:
                if call.participants:
                    for p in call.participants:
                        # "company" affiliation means internal Arize team member
                        if p.affiliation and p.affiliation.lower() == "company" and p.email:
                            arize_participants.add((p.name or "Unknown", p.email))
            
            if arize_participants:
                internal_contacts = [{"name": name, "email": email} for name, email in list(arize_participants)[:5]]
                recommendations.append({
                    "category": "Internal Resources",
                    "title": "Arize team members who've engaged with this account",
                    "contacts": internal_contacts,
                    "description": "Reach out to these team members for context on prior conversations and relationship history."
                })
        
        # --- NEXT STEPS ---
        next_steps = []
        
        if pendo and pendo.days_since_last_activity and pendo.days_since_last_activity > 7:
            next_steps.append("Send re-engagement email with recent platform updates or case study")
        
        if pendo and pendo.unique_visitors and pendo.unique_visitors < 3:
            next_steps.append("Propose team training session to expand adoption")
        
        if fullstory_issues and len(fullstory_issues) > 5:
            next_steps.append("Review FullStory recordings and proactively address UX friction")
        
        if engagement_level == "high":
            next_steps.append("Identify expansion opportunities - additional teams or use cases")
            next_steps.append("Discuss case study or reference opportunity")
        
        if not next_steps:
            next_steps.append("Schedule regular check-in to maintain relationship")
        
        recommendations.append({
            "category": "Next Steps",
            "title": "Recommended actions",
            "steps": next_steps
        })
        
        return recommendations
    
    def _detect_competitors_in_calls(self, gong: Optional[GongSummaryData]) -> Dict[str, List[str]]:
        """
        Analyze Gong call transcripts and briefs to detect competitor mentions.
        Returns a dict of competitor name -> list of contexts where mentioned.
        """
        if not gong or not gong.recent_calls:
            return {}
        
        # Known competitors in the LLM observability/ML monitoring space
        # NOTE: LangChain/LangGraph are frameworks, not competitors - but prospects often
        # say "LangChain" when they mean "LangSmith" (the observability platform)
        competitor_patterns = {
            "Datadog": ["datadog", "data dog", "dd-trace"],
            "Weights & Biases": ["weights and biases", "weights & biases", "wandb", "w&b"],
            "LangSmith": ["langsmith", "lang smith"],
            "LangFuse": ["langfuse", "lang fuse"],  # Recently acquired by ClickHouse
            "LangChain (Framework)": ["langchain", "lang chain", "langgraph", "lang graph"],  # Framework, not competitor - but may indicate LangSmith
            "BrainTrust": ["braintrust", "brain trust", "braintrust.dev", "brain-trust", "braintrust ai", "braintrustdata"],
            "MLflow": ["mlflow", "ml flow", "databricks mlflow"],
            "Neptune": ["neptune.ai", "neptune ai"],
            "Comet": ["comet ml", "comet.ml", "cometml"],
            "Evidently": ["evidently", "evidently ai"],
            "Fiddler": ["fiddler", "fiddler ai"],
            "WhyLabs": ["whylabs", "why labs"],
            "Galileo": ["galileo ai", "rungalileo"],
            "Honeycomb": ["honeycomb", "honeycomb.io"],
            "New Relic": ["new relic", "newrelic"],
            "Splunk": ["splunk"],
            "Dynatrace": ["dynatrace"],
        }
        
        mentioned = {}
        
        for call in gong.recent_calls:
            # Combine all text content from the call
            call_text = ""
            
            if call.spotlight_brief:
                call_text += " " + call.spotlight_brief.lower()
            
            if call.spotlight_key_points:
                if isinstance(call.spotlight_key_points, list):
                    call_text += " " + " ".join([str(kp).lower() for kp in call.spotlight_key_points if kp])
                elif isinstance(call.spotlight_key_points, str):
                    call_text += " " + call.spotlight_key_points.lower()
            
            if call.transcript_snippet:
                if isinstance(call.transcript_snippet, str):
                    call_text += " " + call.transcript_snippet.lower()
                elif isinstance(call.transcript_snippet, list):
                    for item in call.transcript_snippet:
                        if isinstance(item, dict):
                            call_text += " " + str(item.get("text", "")).lower()
                        elif isinstance(item, str):
                            call_text += " " + item.lower()
            
            # Check for competitor mentions
            for competitor, patterns in competitor_patterns.items():
                for pattern in patterns:
                    if pattern in call_text:
                        if competitor not in mentioned:
                            mentioned[competitor] = []
                        
                        # Extract context around the mention
                        idx = call_text.find(pattern)
                        start = max(0, idx - 100)
                        end = min(len(call_text), idx + len(pattern) + 100)
                        context = call_text[start:end].strip()
                        
                        # Add call title for reference
                        call_ref = call.call_title or call.call_date or "recent call"
                        mentioned[competitor].append({
                            "call": call_ref,
                            "context": f"...{context}..."
                        })
                        break  # Only count once per call per competitor
        
        return mentioned
    
    def _generate_targeted_competitive_response(
        self,
        competitor: str,
        contexts: List[Dict],
        focus_areas: List[str],
        base_differentiator: str,
        base_talking_point: str
    ) -> Dict[str, str]:
        """
        Use LLM to generate a targeted competitive response based on what the prospect
        actually said about the competitor.
        
        Args:
            competitor: Name of the competitor
            contexts: List of dicts with 'call' and 'context' keys showing what was said
            focus_areas: List of focus areas the prospect cares about (tracing, evaluation, etc.)
            base_differentiator: The generic differentiator to use as baseline
            base_talking_point: The generic talking point to use as baseline
            
        Returns:
            Dict with 'targeted_response' and 'what_they_said' keys
        """
        # Extract the actual context snippets
        context_snippets = []
        for ctx in contexts[:5]:  # Limit to 5 most relevant contexts
            if isinstance(ctx, dict) and ctx.get("context"):
                context_snippets.append(f"- From {ctx.get('call', 'unknown call')}: \"{ctx['context']}\"")
        
        if not context_snippets:
            # No context available, return base messaging
            return {
                "what_they_said": "No specific mentions captured",
                "targeted_response": base_talking_point
            }
        
        what_they_said = "\n".join(context_snippets)
        
        # Build the prompt for targeted response generation
        prompt = f"""You are a sales enablement expert for Arize AI, an AI/LLM observability platform.

A prospect has mentioned {competitor} in their conversations. Based on what they specifically said, generate a targeted competitive response that directly addresses their comments.

## What the prospect said about {competitor}:
{what_they_said}

## Prospect's focus areas:
{', '.join(focus_areas) if focus_areas else 'General LLM observability'}

## Arize's baseline differentiator vs {competitor}:
{base_differentiator}

## Instructions:
1. Analyze what the prospect specifically said or implied about {competitor}
2. Identify what features, capabilities, or benefits they seem to value in {competitor}
3. Generate a targeted response that:
   - Acknowledges the specific capability they mentioned (don't dismiss it)
   - Positions how Arize addresses that same need, ideally better
   - Highlights a unique Arize capability that {competitor} lacks and is relevant to their use case
   - Is conversational and actionable (something an SA can actually say)

## Response format:
Return ONLY a JSON object with these fields:
- "summary_of_interest": A brief 1-sentence summary of what they seem interested in regarding {competitor}
- "targeted_response": A 2-3 sentence response the SA can use that directly addresses their interest while positioning Arize

Example response:
{{"summary_of_interest": "They're interested in Braintrust's evaluation scoring capabilities for testing prompts before production.", "targeted_response": "Arize absolutely supports evaluation scoring for pre-production testing—we have built-in LLM judges for relevance, toxicity, and custom criteria. Where we go further is tying those same evaluators to your production traffic, so you can see if your test scores hold up with real users. That production feedback loop is something Braintrust doesn't offer today."}}

Return only valid JSON, no markdown formatting or code blocks."""

        try:
            # Use Claude Haiku for fast, cost-effective responses
            model = os.environ.get("COMPETITIVE_ANALYSIS_MODEL", "claude-3-5-haiku-20241022")
            
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3  # Lower temperature for more consistent responses
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse the JSON response
            # Handle potential markdown code blocks
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            result = json.loads(response_text)
            
            return {
                "what_they_said": result.get("summary_of_interest", "See context below"),
                "targeted_response": result.get("targeted_response", base_talking_point),
                "contexts": context_snippets[:3]  # Include raw contexts for reference
            }
            
        except Exception as e:
            print(f"⚠️ Failed to generate targeted competitive response: {e}")
            # Fall back to base messaging
            return {
                "what_they_said": "See context below",
                "targeted_response": base_talking_point,
                "contexts": context_snippets[:3]
            }
    
    def _build_competitive_messaging(
        self,
        mentioned_competitors: Dict[str, List[str]],
        key_workflows: List[str],
        gong: Optional[GongSummaryData]
    ) -> List[Dict]:
        """
        Build competitive messaging tailored to mentioned competitors and prospect's use case.
        """
        if not mentioned_competitors:
            return []
        
        # Detect prospect's focus area from their usage/calls
        focus_areas = []
        workflow_text = " ".join(key_workflows).lower() if key_workflows else ""
        
        if "trace" in workflow_text or "tracing" in workflow_text:
            focus_areas.append("tracing")
        if "eval" in workflow_text:
            focus_areas.append("evaluation")
        if "experiment" in workflow_text:
            focus_areas.append("experimentation")
        if "prompt" in workflow_text:
            focus_areas.append("prompt_management")
        if "dashboard" in workflow_text or "monitor" in workflow_text:
            focus_areas.append("monitoring")
        
        # Also check Gong topics
        if gong and gong.key_themes:
            themes_text = " ".join(gong.key_themes).lower()
            if "production" in themes_text:
                focus_areas.append("production")
            if "debug" in themes_text:
                focus_areas.append("debugging")
            if "compliance" in themes_text or "governance" in themes_text:
                focus_areas.append("compliance")
        
        competitive_messaging = []
        
        # Build tailored messaging for each mentioned competitor
        competitor_differentiators = {
            "Datadog": {
                "default": {
                    "differentiator": "Datadog's LLM observability is an add-on to their APM platform. Arize is purpose-built for AI/ML with native evaluation, drift detection, and LLM-specific metrics.",
                    "talking_point": "While Datadog captures logs and traces, Arize provides semantic understanding of LLM outputs - detecting hallucinations, relevance issues, and quality degradation."
                },
                "tracing": {
                    "differentiator": "Datadog traces show request/response, but lack semantic analysis. Arize traces include automatic evaluation of response quality, relevance, and safety.",
                    "talking_point": "Your team can see not just that a request happened, but whether the LLM response was actually good."
                },
                "monitoring": {
                    "differentiator": "Datadog monitors latency and errors. Arize monitors model quality - catching when your LLM starts giving worse answers before users complain.",
                    "talking_point": "A fast response that's wrong is worse than a slow response that's right. Arize catches quality issues, not just infrastructure issues."
                }
            },
            "Weights & Biases": {
                "default": {
                    "differentiator": "W&B excels at experiment tracking during training. Arize extends observability into production with real-time monitoring, drift detection, and live evaluation.",
                    "talking_point": "W&B helps you train better models; Arize helps you catch when those models misbehave in production."
                },
                "experimentation": {
                    "differentiator": "W&B tracks training experiments offline. Arize enables online experimentation with A/B testing, canary deployments, and production-grade evaluation.",
                    "talking_point": "Move beyond offline experiments - test prompts and models in production with real user traffic and immediate feedback."
                },
                "compliance": {
                    "differentiator": "W&B focuses on data scientists. Arize provides production audit trails, explainability reports, and governance dashboards that compliance teams need.",
                    "talking_point": "For regulated industries, you need production-grade observability with audit trails - not just experiment logs."
                }
            },
            "LangSmith": {
                "default": {
                    "differentiator": "LangSmith is tightly coupled to LangChain. Arize is framework-agnostic - works with LangChain, LlamaIndex, custom code, or any combination.",
                    "talking_point": "Don't lock your observability into one framework. Arize works with your entire stack, now and as it evolves."
                },
                "tracing": {
                    "differentiator": "LangSmith provides LangChain-specific tracing. Arize captures traces from any framework with automatic evaluation and quality scoring.",
                    "talking_point": "If you're using multiple frameworks or considering migration, Arize provides consistent observability across your entire LLM stack."
                },
                "evaluation": {
                    "differentiator": "LangSmith evals are LangChain-native. Arize's evaluation engine is framework-agnostic with pre-built and custom evaluators that work with any LLM.",
                    "talking_point": "Build your evaluation suite once and use it everywhere - not tied to a single framework's ecosystem."
                }
            },
            "LangFuse": {
                "default": {
                    "differentiator": "LangFuse was recently acquired by ClickHouse. Arize is an independent, well-funded company 100% focused on AI observability with enterprise-grade support and roadmap stability.",
                    "talking_point": "With ClickHouse acquiring LangFuse, their roadmap priority may shift. Arize is dedicated to AI observability with no competing priorities."
                },
                "tracing": {
                    "differentiator": "LangFuse provides basic open-source tracing. Arize offers enterprise-grade tracing with automatic quality evaluation, production SLAs, and dedicated support.",
                    "talking_point": "Open-source is great for getting started. For production at scale, you need enterprise observability with guaranteed support."
                }
            },
            # NOTE: LangChain (Framework) is handled specially in the loop below - 
            # it shows LangSmith messaging with a clarifying note
            "BrainTrust": {
                "default": {
                    "differentiator": "BrainTrust focuses primarily on evaluation/evals. Arize provides end-to-end LLM observability including tracing, evaluation, monitoring, and production debugging - a complete platform vs. point solution.",
                    "talking_point": "Evals are critical, but they're just one piece. Arize gives you the full picture - from traces to evals to production monitoring - in one platform."
                },
                "evaluation": {
                    "differentiator": "BrainTrust is strong on offline evals. Arize provides both offline evaluation AND production evaluation with real user traffic, plus automatic quality scoring on every trace.",
                    "talking_point": "Pre-production evals are important, but what happens when real users interact differently than your test cases? Arize evaluates in production too."
                },
                "tracing": {
                    "differentiator": "BrainTrust has limited tracing capabilities. Arize provides full distributed tracing with automatic span detection, latency analysis, and cost tracking across your entire LLM pipeline.",
                    "talking_point": "To debug production issues, you need deep tracing - not just eval scores. Arize shows you exactly what happened in each request."
                },
                "monitoring": {
                    "differentiator": "BrainTrust lacks production monitoring. Arize provides real-time dashboards, alerting, drift detection, and SLA tracking for production LLM applications.",
                    "talking_point": "Evaluations tell you quality at a point in time. Monitoring tells you when quality degrades - before your users notice."
                }
            },
            "MLflow": {
                "default": {
                    "differentiator": "MLflow tracks experiments and models. Arize adds production-grade LLM observability with real-time evaluation, drift detection, and quality monitoring.",
                    "talking_point": "MLflow gets your model to production; Arize tells you if it's working well once it's there."
                }
            },
            "Honeycomb": {
                "default": {
                    "differentiator": "Honeycomb is excellent for general observability. Arize adds AI-specific capabilities like semantic evaluation, hallucination detection, and LLM quality metrics.",
                    "talking_point": "Honeycomb shows you what's happening in your system. Arize shows you whether your AI is giving good answers."
                }
            },
            "New Relic": {
                "default": {
                    "differentiator": "New Relic monitors application performance. Arize monitors AI quality - the difference between 'the API responded' and 'the response was helpful'.",
                    "talking_point": "Traditional APM can't tell you if your LLM is hallucinating. Arize's AI-native observability can."
                }
            }
        }
        
        for competitor, mentions in mentioned_competitors.items():
            # Special handling: If LangChain (Framework) is mentioned, show LangSmith messaging
            # since prospects often say "LangChain" when they mean "LangSmith" (the observability platform)
            if competitor == "LangChain (Framework)":
                # Use LangSmith messaging with a clarifying note
                comp_data = competitor_differentiators.get("LangSmith", {})
                
                messaging = comp_data.get("default", {})
                for focus in focus_areas:
                    if focus in comp_data:
                        messaging = comp_data[focus]
                        break
                
                if messaging:
                    # Generate targeted response based on what they actually said
                    targeted = self._generate_targeted_competitive_response(
                        competitor="LangSmith",
                        contexts=mentions,
                        focus_areas=focus_areas,
                        base_differentiator=messaging.get("differentiator", ""),
                        base_talking_point=messaging.get("talking_point", "")
                    )
                    
                    competitive_messaging.append({
                        "competitor": "LangSmith (prospect mentioned 'LangChain')",
                        "mention_count": len(mentions),
                        "mentioned_in": [m["call"] for m in mentions[:3]],
                        "note": "⚠️ Prospect mentioned 'LangChain' which is an orchestration framework. They likely mean 'LangSmith' - LangChain's observability platform. Position Arize vs LangSmith:",
                        "what_they_said": targeted.get("what_they_said", ""),
                        "targeted_response": targeted.get("targeted_response", ""),
                        "raw_contexts": targeted.get("contexts", []),
                        "differentiator": messaging.get("differentiator", ""),
                        "talking_point": messaging.get("talking_point", "")
                    })
            elif competitor in competitor_differentiators:
                comp_data = competitor_differentiators[competitor]
                
                # Find the best messaging based on prospect's focus
                messaging = comp_data.get("default", {})
                for focus in focus_areas:
                    if focus in comp_data:
                        messaging = comp_data[focus]
                        break
                
                if messaging:
                    # Generate targeted response based on what they actually said
                    targeted = self._generate_targeted_competitive_response(
                        competitor=competitor,
                        contexts=mentions,
                        focus_areas=focus_areas,
                        base_differentiator=messaging.get("differentiator", ""),
                        base_talking_point=messaging.get("talking_point", "")
                    )
                    
                    competitive_messaging.append({
                        "competitor": competitor,
                        "mention_count": len(mentions),
                        "mentioned_in": [m["call"] for m in mentions[:3]],  # First 3 calls
                        "what_they_said": targeted.get("what_they_said", ""),
                        "targeted_response": targeted.get("targeted_response", ""),
                        "raw_contexts": targeted.get("contexts", []),
                        "differentiator": messaging.get("differentiator", ""),
                        "talking_point": messaging.get("talking_point", "")
                    })
            else:
                # Unknown competitor - still try to generate a response
                targeted = self._generate_targeted_competitive_response(
                    competitor=competitor,
                    contexts=mentions,
                    focus_areas=focus_areas,
                    base_differentiator=f"Arize provides comprehensive AI/LLM observability that may offer capabilities beyond {competitor}.",
                    base_talking_point=f"Let's discuss what specific capabilities you're looking for in {competitor} - Arize may provide those plus additional value."
                )
                
                competitive_messaging.append({
                    "competitor": competitor,
                    "mention_count": len(mentions),
                    "mentioned_in": [m["call"] for m in mentions[:3]],
                    "what_they_said": targeted.get("what_they_said", ""),
                    "targeted_response": targeted.get("targeted_response", ""),
                    "raw_contexts": targeted.get("contexts", []),
                    "differentiator": f"Unknown competitor - research recommended for {competitor}",
                    "talking_point": f"Ask the prospect what specific capabilities they're evaluating in {competitor}."
                })
        
        return competitive_messaging
    
    def _build_sales_engagement_summary(
        self,
        account: Optional[SalesforceAccountData],
        selected_opportunity: Optional[OpportunityData],
        gong: Optional[GongSummaryData],
        deal_summary: Optional[DealSummary]
    ) -> Optional[SalesEngagementSummary]:
        """Build a summary of sales engagement based on selected opportunity."""
        
        if not account:
            return None
        
        # Get data from the selected opportunity
        first_touch = None
        days_in_cycle = None
        current_stage = account.lifecycle_stage
        total_calls = 0
        
        if selected_opportunity:
            # First touch = opportunity created date
            first_touch = selected_opportunity.created_date
            current_stage = selected_opportunity.stage_name
            
            # Calculate days in cycle based on the selected opportunity
            if selected_opportunity.created_date:
                try:
                    created_str = selected_opportunity.created_date.split('T')[0].split('+')[0].split(' ')[0]
                    created_date = datetime.fromisoformat(created_str)
                    
                    if selected_opportunity.is_closed and selected_opportunity.close_date:
                        # Closed deal: days from created to close
                        close_str = selected_opportunity.close_date.split('T')[0].split('+')[0].split(' ')[0]
                        close_date = datetime.fromisoformat(close_str)
                        days_in_cycle = (close_date.date() - created_date.date()).days
                    else:
                        # Open deal: days from created to now
                        days_in_cycle = (datetime.now().date() - created_date.date()).days
                except Exception:
                    pass
            
            # Count Gong calls within the opportunity timeframe
            if gong and gong.recent_calls and selected_opportunity.created_date:
                try:
                    opp_created_str = selected_opportunity.created_date.split('T')[0].split('+')[0].split(' ')[0]
                    opp_created = datetime.fromisoformat(opp_created_str).date()
                    
                    opp_close_date = None
                    if selected_opportunity.close_date:
                        opp_close_str = selected_opportunity.close_date.split('T')[0].split('+')[0].split(' ')[0]
                        opp_close_date = datetime.fromisoformat(opp_close_str).date()
                    
                    for call in gong.recent_calls:
                        if call.call_date:
                            call_date_str = call.call_date.split('T')[0].split('+')[0].split(' ')[0]
                            call_date = datetime.fromisoformat(call_date_str).date()
                            # Count calls during the opportunity lifecycle
                            if call_date >= opp_created:
                                if opp_close_date is None or call_date <= opp_close_date:
                                    total_calls += 1
                except Exception as e:
                    # Fallback to total calls on error
                    total_calls = gong.total_calls if gong else 0
        else:
            # Fallback: use all Gong calls if no opportunity selected
            total_calls = gong.total_calls if gong else 0
        
        # Last activity
        last_activity = account.last_activity_date
        if gong and gong.last_call_date:
            last_activity = gong.last_call_date
        
        days_since = None
        if last_activity:
            try:
                last_date_str = last_activity.split('T')[0].split('+')[0].split(' ')[0]
                last_date = datetime.fromisoformat(last_date_str)
                days_since = (datetime.now().date() - last_date.date()).days
            except Exception:
                pass
        
        return SalesEngagementSummary(
            first_touch_date=first_touch,
            days_in_sales_cycle=days_in_cycle,
            current_stage=current_stage,
            total_calls=total_calls,
            total_emails=0,
            total_meetings=0,
            total_tasks=0,
            last_sales_activity_date=last_activity,
            days_since_last_activity=days_since,
            deal_summary=deal_summary
        )
    
    def _build_product_usage_summary(
        self,
        pendo: Optional[PendoUsageData]
    ) -> Optional[ProductUsageSummary]:
        """Build a summary of product usage patterns."""
        
        if not pendo:
            return None
        
        # Determine adoption status
        adoption_status = "not_started"
        if pendo.total_events > 0:
            if pendo.active_days_last_7 >= 5:
                adoption_status = "power_user"
            elif pendo.active_days_last_7 >= 2:
                adoption_status = "active"
            elif pendo.active_days_last_30 >= 5:
                adoption_status = "exploring"
            elif pendo.days_since_last_activity and pendo.days_since_last_activity > 30:
                adoption_status = "churning"
            else:
                adoption_status = "exploring"
        
        # Determine trend
        trend = "stable"
        if pendo.weekly_trend and len(pendo.weekly_trend) >= 2:
            recent = pendo.weekly_trend[0].get("events", 0)
            previous = pendo.weekly_trend[1].get("events", 0)
            if previous > 0:
                change = (recent - previous) / previous
                if change > 0.2:
                    trend = "growing"
                elif change < -0.2:
                    trend = "declining"
        
        # Find last active user
        last_active_user = None
        if pendo.recent_visitors:
            visitor = pendo.recent_visitors[0]
            last_active_user = visitor.display_name or visitor.email or visitor.visitor_id
        
        # Count active users
        active_7 = sum(1 for v in pendo.recent_visitors if v.last_visit and 
                      self._is_within_days(v.last_visit, 7))
        active_30 = sum(1 for v in pendo.recent_visitors if v.last_visit and 
                       self._is_within_days(v.last_visit, 30))
        
        return ProductUsageSummary(
            adoption_status=adoption_status,
            total_users=pendo.unique_visitors,
            active_users_last_7_days=active_7,
            active_users_last_30_days=active_30,
            total_time_minutes=pendo.total_minutes,
            avg_session_minutes=pendo.avg_daily_minutes,
            last_platform_activity=pendo.last_activity,
            last_active_user=last_active_user,
            days_since_last_activity=pendo.days_since_last_activity,
            trend=trend
        )
    
    def _is_within_days(self, date_str: str, days: int) -> bool:
        """Check if a date string is within the last N days."""
        try:
            date = datetime.fromisoformat(date_str.replace('Z', '+00:00').split('+')[0])
            return (datetime.now() - date).days <= days
        except:
            return False
