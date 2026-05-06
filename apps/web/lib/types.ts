/**
 * TypeScript types for API responses.
 * These types mirror the Pydantic models in apps/api/models.py.
 */

// Re-export account types for convenience
export type {
  AccountSuggestionMatch,
  AccountSuggestionsResponse,
  AccountResolveInput,
  AccountResolveResult,
  ResolveAccountFn,
} from "./accountResolve";

// ============================================================================
// SALESFORCE DATA
// ============================================================================

export interface SalesforceAccountData {
  id: string;
  name: string;
  website?: string | null;
  industry?: string | null;
  annual_revenue?: number | null;
  number_of_employees?: number | null;
  total_active_arr?: number | null;
  lifecycle_stage?: string | null;
  product_tier?: string | null;
  status?: string | null;
  customer_success_tier?: string | null;
  assigned_sa?: string | null;
  assigned_cse?: string | null;
  assigned_csm?: string | null;
  assigned_ai_se?: string | null;
  owner_name?: string | null;
  pendo_account_id?: string | null;
  pendo_time_on_site?: number | null;
  num_models?: number | null;
  is_using_llms?: string | null;
  deployment_types?: string | null;
  created_date?: string | null;
  last_activity_date?: string | null;
  next_steps?: string | null;
  customer_notes?: string | null;
  description?: string | null;
}

export interface OpportunityData {
  id: string;
  name: string;
  stage_name: string;
  amount?: number | null;
  close_date?: string | null;
  probability?: number | null;
  is_closed: boolean;
  is_won: boolean;
  forecast_category?: string | null;
  created_date?: string | null;
  last_modified_date?: string | null;
  age_in_days?: number | null;
  next_step?: string | null;
  description?: string | null;
  lead_source?: string | null;
  type?: string | null;
  owner_name?: string | null;
}

// ============================================================================
// GONG DATA
// ============================================================================

export interface GongParticipant {
  name?: string | null;
  email?: string | null;
  affiliation?: string | null;
  speaker_id?: string | null;
}

export interface GongCallData {
  conversation_key?: string | null;
  call_url?: string | null;
  call_title?: string | null;
  call_date?: string | null;
  duration_minutes?: number | null;
  spotlight_brief?: string | null;
  spotlight_key_points?: string[] | null;
  spotlight_next_steps?: string | null;
  spotlight_outcome?: string | null;
  spotlight_type?: string | null;
  talk_ratio?: number | null;
  interactivity?: number | null;
  patience?: number | null;
  question_rate?: number | null;
  longest_monologue?: number | null;
  longest_customer_story?: number | null;
  participants: GongParticipant[];
  transcript_snippet?: string | null;
}

export interface GongSummaryData {
  total_calls: number;
  total_duration_minutes: number;
  avg_talk_ratio?: number | null;
  avg_interactivity?: number | null;
  avg_patience?: number | null;
  avg_question_rate?: number | null;
  first_call_date?: string | null;
  last_call_date?: string | null;
  days_since_last_call?: number | null;
  key_themes: string[];
  recent_calls: GongCallData[];
}

// ============================================================================
// PENDO DATA
// ============================================================================

export interface PendoVisitorActivity {
  visitor_id: string;
  email?: string | null;
  display_name?: string | null;
  total_events: number;
  total_minutes: number;
  last_visit?: string | null;
  first_visit?: string | null;
  visit_count: number;
}

export interface PendoFeatureUsage {
  feature_id: string;
  feature_name?: string | null;
  event_count: number;
  unique_users: number;
  last_used?: string | null;
}

export interface PendoPageUsage {
  page_id: string;
  page_name?: string | null;
  view_count: number;
  unique_viewers: number;
  total_minutes: number;
}

export interface PendoUsageData {
  total_events: number;
  total_minutes: number;
  unique_visitors: number;
  first_activity?: string | null;
  last_activity?: string | null;
  days_since_last_activity?: number | null;
  active_days_last_30: number;
  active_days_last_7: number;
  avg_daily_minutes?: number | null;
  avg_session_duration_minutes?: number | null;
  recent_visitors: PendoVisitorActivity[];
  top_features: PendoFeatureUsage[];
  top_pages: PendoPageUsage[];
  weekly_trend: Array<{ week_start?: string; events?: number }>;
}

// ============================================================================
// FULLSTORY DATA
// ============================================================================

export interface UserIssueEvent {
  issue_type: string;
  error_kind?: string | null;
  page_url?: string | null;
  page_context?: string | null;
  user_email?: string | null;
  user_name?: string | null;
  session_id?: string | null;
  recording_url?: string | null;
  timestamp?: string | null;
  count: number;
  fullstory_user_id?: string | null;
  fullstory_indv_id?: string | null;
}

export interface UserLast24hSurface {
  id: string;
  name?: string | null;
  count: number;
}

export interface UserLast24hActivity {
  visitor_id?: string | null;
  email?: string | null;
  display_name?: string | null;
  total_events_24h: number;
  total_minutes_24h: number;
  first_activity_24h?: string | null;
  last_activity_24h?: string | null;
  top_features_24h: UserLast24hSurface[];
  top_pages_24h: UserLast24hSurface[];
  summary: string;
  fullstory_issues_24h: UserIssueEvent[];
  fullstory_behavior_summary?: string | null;
}

export interface AccountLast24hActivity {
  account_summary?: string | null;
  total_active_users_24h: number;
  pendo_total_events_24h: number;
  active_users: UserLast24hActivity[];
}

// ============================================================================
// ANALYSIS SUMMARIES
// ============================================================================

export interface DealSummary {
  current_state: string;
  key_topics_discussed: string[];
  blockers_identified: string[];
  next_steps_from_calls: string[];
  champion_sentiment?: string | null;
  risk_factors: string[];
}

export interface AdoptionMilestone {
  name: string;
  completed: boolean;
  count: number;
  first_date?: string | null;
  last_date?: string | null;
}

export interface UserBehaviorAnalysis {
  summary: string;
  hypothesis: string;
  key_workflows_used: string[];
  adoption_milestones: AdoptionMilestone[];
  critical_issues: Array<{ type: string; description: string }>;
  user_issues: UserIssueEvent[];
  issues_summary?: string | null;
  engagement_level: "high" | "medium" | "low" | "unknown";
  recommendations: Recommendation[];
}

export interface Recommendation {
  category: string;
  title?: string;
  description?: string;
  steps?: string[];
  // Competitive recommendations
  competitive_messaging?: CompetitiveMessaging[];
  // Internal resources
  contacts?: Array<{ name: string; email: string }>;
}

export interface CompetitiveMessaging {
  competitor: string;
  mentioned_in?: string[];
  mention_count?: number;
  note?: string;
  what_they_said?: string;
  targeted_response?: string;
  raw_contexts?: string[];
  differentiator?: string;
  talking_point?: string;
}

export interface SalesEngagementSummary {
  first_touch_date?: string | null;
  days_in_sales_cycle?: number | null;
  current_stage?: string | null;
  total_calls: number;
  total_emails: number;
  total_meetings: number;
  total_tasks: number;
  last_sales_activity_date?: string | null;
  days_since_last_activity?: number | null;
  deal_summary?: DealSummary | null;
}

export interface ProductUsageSummary {
  adoption_status: "not_started" | "exploring" | "active" | "power_user" | "churning";
  total_users: number;
  active_users_last_7_days: number;
  active_users_last_30_days: number;
  total_time_minutes: number;
  avg_session_minutes?: number | null;
  last_platform_activity?: string | null;
  last_active_user?: string | null;
  days_since_last_activity?: number | null;
  trend: "growing" | "stable" | "declining";
}

// ============================================================================
// PROSPECT OVERVIEW (main API response)
// ============================================================================

export interface ProspectOverview {
  lookup_method: "name" | "domain" | "sfdc_id";
  lookup_value: string;
  
  // Sales engagement context
  salesforce?: SalesforceAccountData | null;
  latest_opportunity?: OpportunityData | null;
  all_opportunities: OpportunityData[];
  gong_summary?: GongSummaryData | null;
  sales_engagement?: SalesEngagementSummary | null;
  
  // Product usage patterns
  pendo_usage?: PendoUsageData | null;
  user_behavior?: UserBehaviorAnalysis | null;
  product_usage?: ProductUsageSummary | null;
  last_24h_activity?: AccountLast24hActivity | null;
  
  // Metadata
  data_sources_available: string[];
  errors: string[];
}

// ============================================================================
// DEMO INSIGHTS
// ============================================================================

export interface DemoInsightsResponse {
  account_name: string;
  industry_or_use_case?: string | null;
  suggested_framework?: string | null;
  suggested_agent_architecture?: string | null;
  suggested_tools?: string | null;
  additional_context?: string | null;
  gong_calls_analyzed: number;
  data_sources_note?: string | null;
  insights_summary?: string | null;
}

// ============================================================================
// HYPOTHESIS RESEARCH
// ============================================================================

export interface HypothesisResearchRequest {
  company_name: string;
  company_domain?: string | null;
  known_competitive_situation?: string | null;
}

export interface HypothesisResearchResponse {
  result: Record<string, unknown>;
  agent_reasoning: string;
}
