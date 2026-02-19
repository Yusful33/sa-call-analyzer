"""LangGraph agent for prospect research and hypothesis generation.

This agent autonomously:
1. Plans what research to conduct
2. Executes research in parallel where possible
3. Evaluates if enough information has been gathered
4. Loops back for deeper research if needed
5. Generates and validates hypotheses
"""

import json
import operator
import contextvars
import asyncio
from datetime import datetime
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ..clients.bigquery_client import BigQueryClient
from ..clients.brave_client import BraveSearchClient, SearchResult
from ..analyzers.signal_extractor import SignalExtractor
from ..models.hypothesis import (
    HypothesisResult,
    Hypothesis,
    HypothesisConfidence,
    DiscoveryQuestion,
    SimilarCustomer,
    SupportingSignal,
    CompetitiveBattlecard,
    CompanyResearch,
    CompetitiveSituation,
    SignalConfidence,
    AIMLSignal,
    CompetitorEvidence,
    ResearchQuality,
    ValueCategory,
)
from ..models.playbook import IndustryPlaybook
from ..config import get_settings
from ..errors import (
    CompanyNotFoundError,
    SearchAPIError,
    LLMError,
    InsufficientDataError,
)


class AgentState(TypedDict):
    """State maintained across the agent's execution."""

    # Input
    company_name: str
    company_domain: str | None

    # Research results (accumulated)
    ai_ml_results: list[SearchResult]
    competitor_results: dict[str, list[SearchResult]]
    job_results: list[SearchResult]
    news_results: list[SearchResult]
    crm_data: dict | None

    # Sub-agent research results (new specialized research)
    linkedin_team_results: list[SearchResult]
    job_postings_detailed_results: list[SearchResult]
    website_blog_results: list[SearchResult]
    news_funding_results: list[SearchResult]
    pain_points_results: list[SearchResult]

    # Extracted insights
    signals: list[AIMLSignal]
    competitor_evidence: list[CompetitorEvidence]
    competitive_situation: CompetitiveSituation
    detected_tools: list[str]
    company_summary: str | None
    industry: str | None

    # Sub-agent extracted insights (for hypothesis prompts)
    team_insights: str | None  # LinkedIn team analysis
    job_insights: str | None   # Job posting analysis
    blog_insights: str | None  # Website/blog analysis
    news_insights: str | None  # News/funding analysis
    pain_insights: str | None  # Pain points analysis

    # Playbook
    playbook: IndustryPlaybook | None

    # Hypotheses
    hypotheses: list[Hypothesis]
    competitive_context: CompetitiveBattlecard | None

    # Agent reasoning
    research_plan: str
    research_complete: bool
    confidence_score: float
    needs_more_research: bool
    research_iteration: int
    agent_reasoning: list[str]  # Track agent's thinking

    # Error tracking
    errors: list[str]  # Critical errors that occurred
    warnings: list[str]  # Non-critical warnings
    search_api_failed: bool  # Track if Brave API failed
    llm_failed: bool  # Track if LLM failed

    # Final output
    final_result: HypothesisResult | None

    # Timing
    start_time: datetime
    processing_time: float | None


class ResearchAgent:
    """LangGraph agent for prospect research."""

    def __init__(
        self,
        bq_client: BigQueryClient | None = None,
        brave_client: BraveSearchClient | None = None,
        signal_extractor: SignalExtractor | None = None,
    ):
        settings = get_settings()

        self.llm = ChatAnthropic(
            model=settings.llm_model,
            anthropic_api_key=settings.anthropic_api_key,
            max_tokens=4096,
        )

        self.bq_client = bq_client
        self.brave_client = brave_client or BraveSearchClient()
        self.signal_extractor = signal_extractor or SignalExtractor()

        # Load battlecards
        self._load_battlecards()

        # Build the graph
        self.graph = self._build_graph()

    def _load_battlecards(self):
        """Load competitive battlecards."""
        from pathlib import Path

        battlecard_path = Path(__file__).parent.parent / "data" / "battlecards.json"
        if battlecard_path.exists():
            with open(battlecard_path) as f:
                self.battlecards = json.load(f)
        else:
            self.battlecards = {}

    def _load_value_drivers(self) -> dict:
        """Load value drivers from JSON file."""
        from pathlib import Path

        value_drivers_path = Path(__file__).parent.parent / "data" / "value_drivers.json"
        if value_drivers_path.exists():
            with open(value_drivers_path) as f:
                return json.load(f)
        return {}

    def _get_relevant_value_drivers(self, industry: str | None) -> str:
        """Get value driver context for hypothesis generation.
        
        Args:
            industry: Prospect industry if known
            
        Returns:
            Formatted string of relevant value drivers for the LLM prompt
        """
        value_drivers = self._load_value_drivers()
        if not value_drivers:
            return ""
        
        # Build value driver context
        context_parts = []
        
        # Add category summaries
        for category, data in value_drivers.get("value_categories", {}).items():
            category_name = category.replace("_", " ").title()
            patterns = data.get("patterns", [])[:3]  # Top 3 per category
            
            if patterns:
                pattern_texts = []
                for p in patterns:
                    pain = p.get("pain", "")
                    solution = p.get("arize_solution", "")
                    quote = p.get("quote")
                    industries = p.get("industries", [])
                    
                    # Check if industry matches
                    industry_match = not industry or industry in industries or "All" in str(industries)
                    
                    if industry_match or not industry:
                        text = f"  - Pain: {pain}\n    Solution: {solution}"
                        if quote:
                            text += f'\n    Customer quote: "{quote}"'
                        pattern_texts.append(text)
                
                if pattern_texts:
                    context_parts.append(f"**{category_name}:**\n" + "\n".join(pattern_texts[:2]))
        
        # Add LLM-specific patterns (high priority)
        llm_patterns = value_drivers.get("llm_specific_patterns", [])[:4]
        if llm_patterns:
            llm_texts = []
            for p in llm_patterns:
                llm_texts.append(f"  - {p.get('pain', '')} → {p.get('solution', '')}")
            context_parts.append("**LLM/GenAI Specific:**\n" + "\n".join(llm_texts))
        
        # Add why Arize reasons
        why_reasons = value_drivers.get("why_arize_reasons", [])[:4]
        if why_reasons:
            context_parts.append("**Why Customers Choose Arize:**\n  - " + "\n  - ".join(why_reasons))
        
        return "\n\n".join(context_parts)

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""

        # Create the graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("plan_research", self._plan_research)
        workflow.add_node("execute_research", self._execute_research)
        workflow.add_node("execute_subagent_research", self._execute_subagent_research)
        workflow.add_node("analyze_subagent_insights", self._analyze_subagent_insights)
        workflow.add_node("check_crm", self._check_crm)
        workflow.add_node("analyze_signals", self._analyze_signals)
        workflow.add_node("evaluate_confidence", self._evaluate_confidence)
        # deep_research replaced by subagent research flow
        workflow.add_node("load_playbook", self._load_playbook)
        workflow.add_node("generate_hypotheses", self._generate_hypotheses)
        workflow.add_node("validate_hypotheses", self._validate_hypotheses)
        workflow.add_node("finalize_result", self._finalize_result)

        # Set entry point
        workflow.set_entry_point("plan_research")

        # Add edges - sub-agent research is now conditional
        workflow.add_edge("plan_research", "execute_research")
        workflow.add_edge("execute_research", "check_crm")
        workflow.add_edge("check_crm", "analyze_signals")
        workflow.add_edge("analyze_signals", "evaluate_confidence")

        # Conditional edge: need more research?
        # If confidence is low, run sub-agent research instead of simple deep research
        workflow.add_conditional_edges(
            "evaluate_confidence",
            self._should_research_more,
            {
                "subagent_research": "execute_subagent_research",
                "continue": "load_playbook",
            },
        )

        # Sub-agent research flow
        workflow.add_edge("execute_subagent_research", "analyze_subagent_insights")
        workflow.add_edge("analyze_subagent_insights", "analyze_signals")
        workflow.add_edge("load_playbook", "generate_hypotheses")
        workflow.add_edge("generate_hypotheses", "validate_hypotheses")

        # Conditional edge: hypotheses good enough?
        workflow.add_conditional_edges(
            "validate_hypotheses",
            self._hypotheses_valid,
            {
                "refine": "generate_hypotheses",
                "finalize": "finalize_result",
            },
        )

        workflow.add_edge("finalize_result", END)

        return workflow.compile()

    # =========================================================================
    # Node implementations
    # =========================================================================

    async def _plan_research(self, state: AgentState) -> dict:
        """Plan what research to conduct based on company name."""

        prompt = f"""You are a sales research assistant. Plan what research to conduct for: {state['company_name']}

Consider:
1. What AI/ML signals should we look for?
2. What competitors might they be using?
3. What job postings would indicate ML maturity?
4. What recent news is relevant?

Output a brief research plan (2-3 sentences)."""

        response = await self.llm.ainvoke([HumanMessage(content=prompt)])

        return {
            "research_plan": response.content,
            "agent_reasoning": [f"Planning research: {response.content}"],
            "research_iteration": 0,
        }

    async def _execute_research(self, state: AgentState) -> dict:
        """Execute web research in parallel for speed."""
        import asyncio

        company = state["company_name"]
        domain = state.get("company_domain")
        reasoning = list(state.get("agent_reasoning", []))
        warnings = list(state.get("warnings", []))
        errors = list(state.get("errors", []))
        search_api_failed = False

        # Use domain in search if provided for better accuracy
        search_term = f"{company} {domain}" if domain else company
        reasoning.append(f"Executing web research for {search_term} (parallel)")

        # Execute all 4 search types in PARALLEL for speed
        async def safe_search(coro, name):
            try:
                return await coro
            except SearchAPIError as e:
                errors.append(f"{name} search failed: {e.message}")
                return [] if name != "Competitor" else {}

        results = await asyncio.gather(
            safe_search(self.brave_client.search_company_ai_ml(company), "AI/ML"),
            safe_search(self.brave_client.search_company_competitors(company), "Competitor"),
            safe_search(self.brave_client.search_company_jobs(company), "Job"),
            safe_search(self.brave_client.search_company_news(company), "News"),
            return_exceptions=True
        )

        ai_ml_results = results[0] if not isinstance(results[0], Exception) else []
        competitor_results = results[1] if not isinstance(results[1], Exception) else {}
        job_results = results[2] if not isinstance(results[2], Exception) else []
        news_results = results[3] if not isinstance(results[3], Exception) else []

        # Check if all searches failed
        if not ai_ml_results and not competitor_results and not job_results and not news_results:
            search_api_failed = True

        # Calculate total results
        total_results = (
            len(ai_ml_results) +
            sum(len(v) for v in competitor_results.values()) +
            len(job_results) +
            len(news_results)
        )

        reasoning.append(
            f"Found {len(ai_ml_results)} AI/ML results, "
            f"{sum(len(v) for v in competitor_results.values())} competitor results, "
            f"{len(job_results)} job postings, {len(news_results)} news items"
        )

        # Check if we found anything meaningful
        # Only consider search_api_failed if ALL searches returned nothing
        all_searches_failed = (
            search_api_failed and 
            total_results == 0
        )
        
        if total_results == 0 and not search_api_failed:
            warnings.append(
                f"No search results found for '{company}'. "
                "The company name may be misspelled or too generic."
            )
            reasoning.append("WARNING: No search results found - company may not exist or name is ambiguous")
        elif total_results == 0 and search_api_failed:
            # All searches failed due to API issues
            pass  # Error already recorded
        elif total_results < 5 and total_results > 0:
            warnings.append(
                f"Limited information found for '{company}' ({total_results} results). "
                "Hypotheses may be less specific."
            )

        return {
            "ai_ml_results": ai_ml_results,
            "competitor_results": competitor_results,
            "job_results": job_results,
            "news_results": news_results,
            "agent_reasoning": reasoning,
            "warnings": warnings,
            "errors": errors,
            "search_api_failed": all_searches_failed,  # Only true if ALL searches failed
        }

    async def _execute_subagent_research(self, state: AgentState) -> dict:
        """Execute specialized sub-agent research in PARALLEL for speed.
        
        This runs 5 specialized research sub-agents concurrently:
        1. LinkedIn Team Research - team size, roles, maturity
        2. Job Postings Detailed - tools, frameworks, problems
        3. Website/Blog Research - AI features, tech posts
        4. News/Funding Research - announcements, strategy
        5. Pain Points Research - challenges, issues
        """
        import asyncio
        
        company = state["company_name"]
        domain = state.get("company_domain")
        reasoning = list(state.get("agent_reasoning", []))
        warnings = list(state.get("warnings", []))
        
        reasoning.append(f"Running 5 sub-agent searches in PARALLEL for {company}")
        
        # Helper for safe execution
        async def safe_search(coro, name):
            try:
                return await coro
            except SearchAPIError as e:
                warnings.append(f"{name} research limited: {e.message}")
                return []
        
        # Execute ALL 5 sub-agents in PARALLEL
        results = await asyncio.gather(
            safe_search(self.brave_client.search_linkedin_team(company), "LinkedIn"),
            safe_search(self.brave_client.search_job_postings_detailed(company), "Job postings"),
            safe_search(self.brave_client.search_website_blog(company, domain), "Website/blog"),
            safe_search(self.brave_client.search_news_funding(company), "News/funding"),
            safe_search(self.brave_client.search_pain_points(company), "Pain points"),
            return_exceptions=True
        )
        
        linkedin_results = results[0] if not isinstance(results[0], Exception) else []
        job_detailed_results = results[1] if not isinstance(results[1], Exception) else []
        website_results = results[2] if not isinstance(results[2], Exception) else []
        news_funding_results = results[3] if not isinstance(results[3], Exception) else []
        pain_results = results[4] if not isinstance(results[4], Exception) else []
        
        total_subagent_results = (
            len(linkedin_results) + len(job_detailed_results) + 
            len(website_results) + len(news_funding_results) + len(pain_results)
        )
        reasoning.append(f"Sub-agent research complete: {total_subagent_results} results (parallel)")
        
        return {
            "linkedin_team_results": linkedin_results,
            "job_postings_detailed_results": job_detailed_results,
            "website_blog_results": website_results,
            "news_funding_results": news_funding_results,
            "pain_points_results": pain_results,
            "agent_reasoning": reasoning,
            "warnings": warnings,
        }

    async def _analyze_subagent_insights(self, state: AgentState) -> dict:
        """Analyze sub-agent research results to extract structured insights.
        
        Uses a SINGLE LLM call to extract insights from all research dimensions
        for better performance.
        """
        reasoning = list(state.get("agent_reasoning", []))
        company = state["company_name"]
        
        # Helper to format results for LLM
        def format_results(results: list[SearchResult], max_items: int = 5) -> str:
            if not results:
                return "No results found"
            items = []
            for r in results[:max_items]:
                items.append(f"- {r.title}: {r.description[:150]}...")
            return "\n".join(items)
        
        # Build combined prompt for single LLM call
        sections = []
        
        if state.get("linkedin_team_results"):
            sections.append(f"""## LinkedIn/Team Research
{format_results(state['linkedin_team_results'])}""")
        
        if state.get("job_postings_detailed_results"):
            sections.append(f"""## Job Postings Research
{format_results(state['job_postings_detailed_results'])}""")
        
        if state.get("website_blog_results"):
            sections.append(f"""## Website/Blog Research
{format_results(state['website_blog_results'])}""")
        
        if state.get("news_funding_results"):
            sections.append(f"""## News/Funding Research
{format_results(state['news_funding_results'])}""")
        
        if state.get("pain_points_results"):
            sections.append(f"""## Pain Points Research
{format_results(state['pain_points_results'])}""")
        
        if not sections:
            reasoning.append("No sub-agent results to analyze")
            return {"agent_reasoning": reasoning}
        
        combined_results = "\n\n".join(sections)
        
        # Single LLM call for all insights
        prompt = f"""Analyze this research about {company}'s AI/ML capabilities and needs.

{combined_results}

Provide a brief analysis for each category found above. For each, extract 2-3 key bullet points:

**Team Insights**: (team size, leadership, hiring patterns, maturity)
**Job Insights**: (tools/frameworks, production vs research, scale indicators)  
**Blog Insights**: (AI features, technical challenges, use cases)
**News Insights**: (announcements, funding, partnerships, strategy)
**Pain Insights**: (challenges, monitoring gaps, cost concerns)

Be concise. Say "Unknown" for categories with no clear signals. Format as bullet points under each heading."""

        insights = {
            "team": None,
            "jobs": None,
            "blog": None,
            "news": None,
            "pain": None,
        }
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            full_analysis = response.content
            
            # Parse the response into sections (simple parsing)
            if "**Team Insights**" in full_analysis:
                insights["team"] = full_analysis
            if "**Job Insights**" in full_analysis:
                insights["jobs"] = full_analysis
            if "**Blog Insights**" in full_analysis:
                insights["blog"] = full_analysis
            if "**News Insights**" in full_analysis:
                insights["news"] = full_analysis
            if "**Pain Insights**" in full_analysis:
                insights["pain"] = full_analysis
            
            # Store the full analysis (all insights combined)
            reasoning.append(f"Analyzed sub-agent research in single LLM call")
            
            # Return the full analysis as all insights (they're combined anyway)
            return {
                "team_insights": full_analysis if insights["team"] else None,
                "job_insights": full_analysis if insights["jobs"] else None,
                "blog_insights": full_analysis if insights["blog"] else None,
                "news_insights": full_analysis if insights["news"] else None,
                "pain_insights": full_analysis if insights["pain"] else None,
                "agent_reasoning": reasoning,
            }
            
        except Exception as e:
            reasoning.append(f"Sub-agent analysis failed: {str(e)}")
            return {
                "team_insights": None,
                "job_insights": None,
                "blog_insights": None,
                "news_insights": None,
                "pain_insights": None,
                "agent_reasoning": reasoning,
            }

    async def _check_crm(self, state: AgentState) -> dict:
        """Check CRM for existing account data."""

        reasoning = list(state.get("agent_reasoning", []))

        crm_data = None
        if self.bq_client:
            try:
                crm_data = self.bq_client.get_account_by_name(state["company_name"])
                if crm_data:
                    reasoning.append(
                        f"Found in CRM: {crm_data.get('industry', 'Unknown industry')}"
                    )
                else:
                    reasoning.append("Not found in CRM - net new prospect")
            except Exception as e:
                reasoning.append(f"CRM check failed: {str(e)}")
        else:
            reasoning.append("CRM not available")

        return {
            "crm_data": crm_data,
            "industry": crm_data.get("industry") if crm_data else None,
            "agent_reasoning": reasoning,
        }

    async def _analyze_signals(self, state: AgentState) -> dict:
        """Analyze research results to extract signals."""

        reasoning = list(state.get("agent_reasoning", []))

        # Extract AI/ML signals
        signals = self.signal_extractor.extract_signals_from_search_results(
            state["ai_ml_results"]
        )
        job_signals = self.signal_extractor.extract_signals_from_search_results(
            state["job_results"]
        )
        signals.extend(job_signals)

        # Detect competitors
        all_competitor_results = []
        for results in state["competitor_results"].values():
            all_competitor_results.extend(results)

        competitive_situation, detected_tools, competitor_evidence = (
            self.signal_extractor.detect_competitors(all_competitor_results)
        )

        # If no competitors but strong signals, likely greenfield
        ai_confidence = self.signal_extractor.calculate_overall_confidence(signals)
        if competitive_situation == CompetitiveSituation.UNKNOWN:
            if ai_confidence in [SignalConfidence.HIGH, SignalConfidence.MEDIUM]:
                competitive_situation = CompetitiveSituation.GREENFIELD

        # Generate summary
        summary = await self.signal_extractor.analyze_with_llm(
            state["company_name"],
            state["ai_ml_results"],
            state["job_results"],
            state["news_results"],
        )

        reasoning.append(
            f"Extracted {len(signals)} signals, "
            f"AI/ML confidence: {ai_confidence.value}, "
            f"Competitive: {competitive_situation.value}"
        )

        if detected_tools:
            reasoning.append(f"Detected tools: {', '.join(detected_tools)}")

        return {
            "signals": signals,
            "competitor_evidence": competitor_evidence,
            "competitive_situation": competitive_situation,
            "detected_tools": detected_tools,
            "company_summary": summary,
            "agent_reasoning": reasoning,
        }

    async def _evaluate_confidence(self, state: AgentState) -> dict:
        """Evaluate if we have enough information to proceed."""

        reasoning = list(state.get("agent_reasoning", []))
        signals = state.get("signals", [])
        iteration = state.get("research_iteration", 0)

        # Calculate confidence score
        high_signals = sum(1 for s in signals if s.confidence == SignalConfidence.HIGH)
        medium_signals = sum(
            1 for s in signals if s.confidence == SignalConfidence.MEDIUM
        )

        confidence_score = min(
            1.0, (high_signals * 0.3 + medium_signals * 0.15 + 0.2)
        )

        # Decide if we need more research
        needs_more = confidence_score < 0.5 and iteration < 2

        if needs_more:
            reasoning.append(
                f"Confidence {confidence_score:.2f} is low, need deeper research (iteration {iteration + 1})"
            )
        else:
            reasoning.append(
                f"Confidence {confidence_score:.2f} is sufficient, proceeding to hypothesis generation"
            )

        return {
            "confidence_score": confidence_score,
            "needs_more_research": needs_more,
            "research_iteration": iteration + 1,
            "agent_reasoning": reasoning,
        }

    def _should_research_more(self, state: AgentState) -> str:
        """Routing function: should we do more research?
        
        If confidence is low and we haven't run sub-agents yet, run them.
        Sub-agents only run once (when iteration == 1 after first evaluation).
        """
        iteration = state.get("research_iteration", 0)
        needs_more = state.get("needs_more_research", False)
        
        # Only run sub-agent research on first iteration when confidence is low
        if needs_more and iteration == 1:
            return "subagent_research"
        return "continue"

    async def _deep_research(self, state: AgentState) -> dict:
        """Conduct deeper, more targeted research."""

        reasoning = list(state.get("agent_reasoning", []))
        company = state["company_name"]

        # Ask LLM what to search for
        prompt = f"""Based on our initial research for {company}, we need more information.

Current findings:
- AI/ML Confidence: {state.get('confidence_score', 0):.2f}
- Signals found: {len(state.get('signals', []))}
- Summary: {state.get('company_summary', 'None')}

Suggest 2 specific search queries that would help us understand:
1. Their ML/AI infrastructure maturity
2. Specific pain points they might have

Output just the search queries, one per line."""

        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        queries = response.content.strip().split("\n")[:2]

        reasoning.append(f"Deep research queries: {queries}")

        # Execute additional searches
        additional_results = []
        for query in queries:
            results = await self.brave_client.search(f"{company} {query}", count=5)
            additional_results.extend(results)

        # Merge with existing results
        existing = list(state.get("ai_ml_results", []))
        existing.extend(additional_results)

        reasoning.append(f"Deep research found {len(additional_results)} additional results")

        return {
            "ai_ml_results": existing,
            "agent_reasoning": reasoning,
        }

    async def _load_playbook(self, state: AgentState) -> dict:
        """Load the appropriate industry playbook with fuzzy matching."""

        reasoning = list(state.get("agent_reasoning", []))
        industry = state.get("industry")
        company_name = state.get("company_name", "").lower()

        # Industry aliases for fuzzy matching when CRM data is generic
        INDUSTRY_ALIASES = {
            # Financial sector keywords -> Financial Services playbook
            "organizations": {
                "keywords": ["finra", "sec", "fdic", "occ", "regulatory", "regulator", 
                            "federal reserve", "treasury", "financial industry"],
                "fallback": "financial_services"
            },
            # Generic tech -> Software playbook
            "technology": {
                "keywords": [],
                "fallback": "software"
            },
            "information technology": {
                "keywords": [],
                "fallback": "software"
            },
        }

        playbook = None
        if industry:
            from pathlib import Path

            playbook_dir = Path(__file__).parent.parent / "data" / "playbooks"
            filename = industry.lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
            filepath = playbook_dir / f"{filename}.json"

            # Try exact match first
            if filepath.exists():
                with open(filepath) as f:
                    data = json.load(f)
                playbook = IndustryPlaybook(**data)
                reasoning.append(
                    f"Loaded playbook for {industry} ({playbook.deal_count} deals)"
                )
            else:
                # Try fuzzy matching based on industry aliases
                industry_lower = industry.lower()
                fallback_used = None
                
                if industry_lower in INDUSTRY_ALIASES:
                    alias_config = INDUSTRY_ALIASES[industry_lower]
                    # Check if company name matches any keywords for this alias
                    for keyword in alias_config["keywords"]:
                        if keyword in company_name:
                            fallback_file = playbook_dir / f"{alias_config['fallback']}.json"
                            if fallback_file.exists():
                                with open(fallback_file) as f:
                                    data = json.load(f)
                                playbook = IndustryPlaybook(**data)
                                fallback_used = alias_config['fallback'].replace("_", " ").title()
                                reasoning.append(
                                    f"Industry '{industry}' mapped to '{fallback_used}' playbook "
                                    f"based on company name ('{keyword}' detected)"
                                )
                                break
                    
                    # If no keyword match but has a fallback, use it
                    if not playbook and alias_config.get("fallback"):
                        fallback_file = playbook_dir / f"{alias_config['fallback']}.json"
                        if fallback_file.exists():
                            with open(fallback_file) as f:
                                data = json.load(f)
                            playbook = IndustryPlaybook(**data)
                            fallback_used = alias_config['fallback'].replace("_", " ").title()
                            reasoning.append(
                                f"Industry '{industry}' mapped to '{fallback_used}' playbook (fallback)"
                            )
                
                # Try partial match on existing playbooks
                if not playbook:
                    for pb_file in playbook_dir.glob("*.json"):
                        pb_name = pb_file.stem.replace("_", " ")
                        if industry_lower in pb_name or pb_name in industry_lower:
                            with open(pb_file) as f:
                                data = json.load(f)
                            playbook = IndustryPlaybook(**data)
                            reasoning.append(
                                f"Partial match: loaded '{pb_name}' playbook for '{industry}'"
                            )
                            break
                
                if not playbook:
                    reasoning.append(f"No playbook found for {industry}")
        else:
            reasoning.append("No industry identified, using general approach")

        return {
            "playbook": playbook,
            "agent_reasoning": reasoning,
        }

    async def _generate_hypotheses(self, state: AgentState) -> dict:
        """Generate hypotheses using LLM."""

        reasoning = list(state.get("agent_reasoning", []))
        warnings = list(state.get("warnings", []))
        errors = list(state.get("errors", []))
        llm_failed = False

        # Check if we have enough data to generate meaningful hypotheses
        signals = state.get("signals", [])[:10]
        total_results = (
            len(state.get("ai_ml_results", [])) +
            len(state.get("job_results", [])) +
            len(state.get("news_results", []))
        )
        
        # If we have almost no data, add a strong warning
        if len(signals) == 0 and total_results < 3:
            warnings.append(
                f"CAUTION: Very limited data found for '{state['company_name']}'. "
                "The following hypotheses are highly speculative and based on general "
                "ML observability patterns rather than company-specific signals. "
                "Consider verifying the company name or providing the domain."
            )
            reasoning.append("WARNING: Generating hypotheses with minimal data - results will be generic")

        signals_text = "\n".join(
            f"- [{s.confidence.value}] {s.evidence} | URL: {s.source_url or 'N/A'}"
            for s in signals
        )
        
        # Build a lookup map for signals by evidence text
        self._signal_url_map = {
            s.evidence: s.source_url for s in signals if s.source_url
        }

        playbook = state.get("playbook")
        if playbook:
            playbook_text = f"""Industry: {playbook.industry} ({playbook.deal_count} won deals)
Top pain points:
{chr(10).join(f'- {pp.pain} (mentioned {pp.frequency}x)' for pp in playbook.top_pain_points[:5])}"""
        else:
            playbook_text = "No industry playbook available - using general ML observability patterns"

        competitive_text = f"Situation: {state.get('competitive_situation', 'unknown').value}"
        if state.get("detected_tools"):
            competitive_text += f"\nDetected tools: {', '.join(state['detected_tools'])}"

        # Get value drivers for context
        value_driver_context = self._get_relevant_value_drivers(state.get("industry"))
        
        # Build sub-agent insights section (they're combined in one analysis now)
        # Just use team_insights since they all contain the full combined analysis
        subagent_text = state.get("team_insights") or "No detailed sub-agent research performed (initial research was sufficient)"
        
        prompt = f"""Generate 2-3 hypotheses about where Arize could help {state['company_name']}.

ARIZE FOCUS AREAS (prioritize LLM/GenAI use cases):
1. LLM Observability - tracing, prompt/response monitoring, token usage, latency
2. LLM Evaluation - automated quality scoring, hallucination detection, response relevance
3. RAG Systems - retrieval quality, context relevance, chunking effectiveness  
4. Agent Workflows - multi-step tracing, tool call monitoring, agent debugging
5. Fine-tuning - dataset curation, model comparison, regression detection
6. Traditional ML - model monitoring, drift detection (secondary focus)

PROVEN VALUE DRIVERS FROM WON DEALS:
{value_driver_context or 'No specific value drivers available'}

==============================================================================
DETAILED RESEARCH FINDINGS (from 5 specialized sub-agents)
==============================================================================

{subagent_text}

==============================================================================

COMPANY SUMMARY:
{state.get('company_summary', 'No summary available')}

SIGNALS FOUND:
{signals_text or 'No strong signals found'}

INDUSTRY PLAYBOOK:
{playbook_text}

COMPETITIVE SITUATION:
{competitive_text}

For each hypothesis, use the Command of the Message framework:

1. **Hypothesis**: A clear statement of where Arize could help (focus on LLM/GenAI if signals support it)
2. **Current State**: What they're likely doing today and the challenges they face
3. **Future State**: What they could achieve with proper observability (1-2 sentences)
4. **Required Capabilities**: 2-3 specific capabilities they need (map to Arize features)
5. **Negative Consequences**: What happens if they don't address this (1 sentence, business impact)
6. **Value Category**: One of: reduce_risk, increase_efficiency, increase_revenue, reduce_cost
7. **Confidence**: high/medium/low with reasoning
8. **Supporting Signals**: Evidence from research
9. **Discovery Questions**: 2-3 questions to validate (with rationale)

IMPORTANT: Ground hypotheses in the PROVEN VALUE DRIVERS above when possible. Reference real customer pain points and outcomes.

Respond in JSON format:
{{
    "hypotheses": [
        {{
            "hypothesis": "statement focusing on LLM/GenAI observability where supported",
            "current_state": "What they're doing now and challenges",
            "future_state": "What they could achieve",
            "required_capabilities": ["capability1", "capability2"],
            "negative_consequences": "Business risk of inaction",
            "value_category": "reduce_risk|increase_efficiency|increase_revenue|reduce_cost",
            "confidence": "high|medium|low",
            "confidence_reasoning": "why",
            "supporting_signals": [
                {{"description": "brief signal description", "source_url": "URL from signals above or null", "confidence": "high|medium|low"}}
            ],
            "discovery_questions": [
                {{"question": "q", "rationale": "why ask this"}}
            ]
        }}
    ]
}}

IMPORTANT: For supporting_signals, include the source_url from the SIGNALS list above so users can verify.

CRITICAL: If there are NO signals found or the company summary is empty/generic, you MUST:
1. Set confidence to "low" for all hypotheses
2. Clearly state in confidence_reasoning that this is based on general patterns, not company-specific data
3. Focus discovery questions on validating whether the company even has ML/AI initiatives"""

        # Call LLM with error handling
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            response_text = response.content
        except Exception as e:
            llm_failed = True
            error_msg = str(e)
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                errors.append("LLM rate limit exceeded. Please try again in a few moments.")
            elif "authentication" in error_msg.lower() or "401" in error_msg:
                errors.append("LLM authentication failed. Please check API configuration.")
            else:
                errors.append(f"AI analysis failed: {error_msg}")
            reasoning.append(f"ERROR: LLM call failed - {error_msg}")
            
            # Return empty hypotheses with error state
            return {
                "hypotheses": [],
                "competitive_context": None,
                "agent_reasoning": reasoning,
                "warnings": warnings,
                "errors": errors,
                "llm_failed": True,
            }

        # Parse response
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end]
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end]

        try:
            data = json.loads(response_text.strip())
        except json.JSONDecodeError as e:
            errors.append(f"Failed to parse AI response. Please try again.")
            reasoning.append(f"ERROR: JSON parsing failed - {str(e)}")
            data = {"hypotheses": []}

        # Convert to Hypothesis objects
        hypotheses = []
        for h in data.get("hypotheses", []):
            conf_str = h.get("confidence", "low").lower()
            confidence = {
                "high": HypothesisConfidence.HIGH,
                "medium": HypothesisConfidence.MEDIUM,
                "low": HypothesisConfidence.LOW,
            }.get(conf_str, HypothesisConfidence.LOW)

            questions = [
                DiscoveryQuestion(
                    question=q.get("question", ""),
                    rationale=q.get("rationale", ""),
                )
                for q in h.get("discovery_questions", [])
            ]
            
            # Parse supporting signals - handle both string and object formats
            raw_signals = h.get("supporting_signals", [])
            supporting_signals = []
            for sig in raw_signals:
                if isinstance(sig, dict):
                    # New format with URL
                    supporting_signals.append(
                        SupportingSignal(
                            description=sig.get("description", ""),
                            source_url=sig.get("source_url"),
                            confidence=sig.get("confidence", "medium"),
                        )
                    )
                elif isinstance(sig, str):
                    # Old format - just a string, try to find URL from signal map
                    url = self._signal_url_map.get(sig) if hasattr(self, '_signal_url_map') else None
                    supporting_signals.append(
                        SupportingSignal(
                            description=sig,
                            source_url=url,
                            confidence="medium",
                        )
                    )

            # Parse value category
            value_cat_str = h.get("value_category", "").lower().replace(" ", "_")
            value_category = {
                "reduce_risk": ValueCategory.REDUCE_RISK,
                "increase_efficiency": ValueCategory.INCREASE_EFFICIENCY,
                "increase_revenue": ValueCategory.INCREASE_REVENUE,
                "reduce_cost": ValueCategory.REDUCE_COST,
            }.get(value_cat_str)

            hypotheses.append(
                Hypothesis(
                    hypothesis=h.get("hypothesis", ""),
                    confidence=confidence,
                    confidence_reasoning=h.get("confidence_reasoning", ""),
                    # Command of the Message fields
                    current_state=h.get("current_state"),
                    future_state=h.get("future_state"),
                    required_capabilities=h.get("required_capabilities", []),
                    negative_consequences=h.get("negative_consequences"),
                    value_category=value_category,
                    # Evidence and actions
                    supporting_signals=supporting_signals,
                    discovery_questions=questions,
                )
            )

        # Get competitive battlecard
        competitive_context = self._get_battlecard(
            state.get("competitive_situation", CompetitiveSituation.UNKNOWN),
            state.get("detected_tools", []),
        )

        reasoning.append(f"Generated {len(hypotheses)} hypotheses")
        
        # If no hypotheses were generated, add an error
        if len(hypotheses) == 0:
            errors.append(
                "Could not generate any hypotheses. This may indicate insufficient data "
                "or an issue with AI processing."
            )

        return {
            "hypotheses": hypotheses,
            "competitive_context": competitive_context,
            "agent_reasoning": reasoning,
            "warnings": warnings,
            "errors": errors,
            "llm_failed": llm_failed,
        }

    def _get_battlecard(
        self,
        situation: CompetitiveSituation,
        detected_tools: list[str],
    ) -> CompetitiveBattlecard:
        """Get appropriate battlecard."""

        if situation == CompetitiveSituation.SWITCHING and detected_tools:
            tool_key = detected_tools[0].lower().replace(" ", "_")
            if tool_key in self.battlecards:
                bc = self.battlecards[tool_key]
                return CompetitiveBattlecard(
                    situation=situation,
                    detected_competitor=bc.get("name", tool_key),
                    positioning=f"Switching from {bc.get('name', tool_key)}",
                    key_questions=bc.get("displacement_questions", []),
                    advantages=bc.get("arize_advantages", []),
                    watch_outs=bc.get("watch_outs", []),
                )

        if situation == CompetitiveSituation.BUILD_VS_BUY and "homegrown" in self.battlecards:
            bc = self.battlecards["homegrown"]
            return CompetitiveBattlecard(
                situation=situation,
                detected_competitor="Homegrown",
                positioning="Build vs Buy",
                key_questions=bc.get("displacement_questions", []),
                advantages=bc.get("arize_advantages", []),
            )

        if situation == CompetitiveSituation.GREENFIELD and "greenfield" in self.battlecards:
            bc = self.battlecards["greenfield"]
            return CompetitiveBattlecard(
                situation=situation,
                positioning="Build it right from the start",
                key_questions=bc.get("displacement_questions", []),
                advantages=bc.get("arize_advantages", []),
            )

        return CompetitiveBattlecard(
            situation=CompetitiveSituation.UNKNOWN,
            positioning="Gather more information",
            key_questions=["What's your current ML monitoring approach?"],
        )

    async def _validate_hypotheses(self, state: AgentState) -> dict:
        """Validate hypothesis quality."""

        reasoning = list(state.get("agent_reasoning", []))
        hypotheses = state.get("hypotheses", [])

        # Simple validation: do we have hypotheses with questions?
        valid = len(hypotheses) > 0 and all(
            len(h.discovery_questions) > 0 for h in hypotheses
        )

        if valid:
            reasoning.append("Hypotheses validated successfully")
        else:
            reasoning.append("Hypotheses need refinement")

        return {
            "research_complete": valid,
            "agent_reasoning": reasoning,
        }

    def _hypotheses_valid(self, state: AgentState) -> str:
        """Routing function: are hypotheses good enough?"""
        # For now, always finalize (could add refinement loop)
        return "finalize"

    async def _finalize_result(self, state: AgentState) -> dict:
        """Finalize and package the result."""

        processing_time = (datetime.utcnow() - state["start_time"]).total_seconds()
        warnings = list(state.get("warnings", []))
        errors = list(state.get("errors", []))

        # Build CompanyResearch
        research = CompanyResearch(
            company_name=state["company_name"],
            domain=state.get("company_domain"),
            industry=state.get("industry"),
            employee_count=state["crm_data"].get("number_of_employees") if state.get("crm_data") else None,
            ai_ml_signals=state.get("signals", []),
            ai_ml_confidence=self.signal_extractor.calculate_overall_confidence(
                state.get("signals", [])
            ),
            competitive_situation=state.get("competitive_situation", CompetitiveSituation.UNKNOWN),
            detected_tools=state.get("detected_tools", []),
            competitor_evidence=state.get("competitor_evidence", []),
            exists_in_crm=state.get("crm_data") is not None,
            company_summary=state.get("company_summary"),
        )

        # Add similar customers to first hypothesis
        hypotheses = list(state.get("hypotheses", []))
        if hypotheses and self.bq_client and state.get("industry"):
            try:
                similar = self.bq_client.get_similar_customers(
                    industry=state["industry"],
                    employee_count=state["crm_data"].get("number_of_employees") if state.get("crm_data") else None,
                )
                hypotheses[0].similar_customers = [
                    SimilarCustomer(
                        customer_name=s.account_name,
                        industry=s.industry,
                        deal_size=s.amount,
                    )
                    for s in similar
                ]
            except Exception:
                pass

        # Build confidence note
        playbook = state.get("playbook")
        confidence_note = None
        if not playbook:
            confidence_note = f"Limited data: No playbook available for {state.get('industry', 'this industry')}."
        elif playbook.deal_count < 10:
            confidence_note = f"Limited data: Only {playbook.deal_count} deals in {playbook.industry}."

        # Calculate research quality based on signals found, not just errors
        signals = state.get("signals", [])
        total_results = (
            len(state.get("ai_ml_results", [])) +
            len(state.get("job_results", [])) +
            len(state.get("news_results", []))
        )
        high_confidence_signals = sum(1 for s in signals if s.confidence == SignalConfidence.HIGH)
        
        # LLM failure is critical - can't generate hypotheses
        if state.get("llm_failed"):
            research_quality = ResearchQuality.INSUFFICIENT
            if not any("AI" in e for e in errors):
                errors.append("AI analysis failed. Could not generate hypotheses.")
        # If ALL searches failed AND we have no signals, insufficient
        elif state.get("search_api_failed") and len(signals) == 0:
            research_quality = ResearchQuality.INSUFFICIENT
        # No signals at all is insufficient
        elif len(signals) == 0 and total_results < 3:
            research_quality = ResearchQuality.INSUFFICIENT
            if not any("Very limited" in w or "CAUTION" in w for w in warnings):
                warnings.append(
                    f"Insufficient data found for '{state['company_name']}'. "
                    "Results are based on general patterns rather than company-specific signals."
                )
        # Good number of high-confidence signals = high quality
        elif high_confidence_signals >= 2 and len(signals) >= 4:
            research_quality = ResearchQuality.HIGH
        # Some signals = medium quality
        elif len(signals) >= 2:
            research_quality = ResearchQuality.MEDIUM
        # Few signals = low quality
        else:
            research_quality = ResearchQuality.LOW
            if not any("limited" in w.lower() for w in warnings):
                warnings.append(
                    "Limited signals found. Hypotheses may be generic."
                )

        # Build final result
        result = HypothesisResult(
            company_name=state["company_name"],
            research=research,
            hypotheses=hypotheses,
            competitive_context=state.get("competitive_context"),
            playbook_used=playbook.industry if playbook else None,
            playbook_deal_count=playbook.deal_count if playbook else None,
            confidence_note=confidence_note,
            research_quality=research_quality,
            warnings=warnings,
            errors=errors,
            processing_time_seconds=processing_time,
        )

        return {
            "final_result": result,
            "processing_time": processing_time,
            "agent_reasoning": state.get("agent_reasoning", []) + [
                f"Research complete in {processing_time:.1f}s (quality: {research_quality.value})"
            ],
        }

    # =========================================================================
    # Public interface
    # =========================================================================

    async def research(
        self,
        company_name: str,
        company_domain: str | None = None,
    ) -> tuple[HypothesisResult, list[str]]:
        """
        Research a company and generate hypotheses.

        Returns:
            Tuple of (HypothesisResult, agent_reasoning)
        """

        initial_state: AgentState = {
            "company_name": company_name,
            "company_domain": company_domain,
            "ai_ml_results": [],
            "competitor_results": {},
            "job_results": [],
            "news_results": [],
            "crm_data": None,
            # Sub-agent research results
            "linkedin_team_results": [],
            "job_postings_detailed_results": [],
            "website_blog_results": [],
            "news_funding_results": [],
            "pain_points_results": [],
            # Extracted insights
            "signals": [],
            "competitor_evidence": [],
            "competitive_situation": CompetitiveSituation.UNKNOWN,
            "detected_tools": [],
            "company_summary": None,
            "industry": None,
            # Sub-agent insights
            "team_insights": None,
            "job_insights": None,
            "blog_insights": None,
            "news_insights": None,
            "pain_insights": None,
            # Rest
            "playbook": None,
            "hypotheses": [],
            "competitive_context": None,
            "research_plan": "",
            "research_complete": False,
            "confidence_score": 0.0,
            "needs_more_research": False,
            "research_iteration": 0,
            "agent_reasoning": [],
            "errors": [],
            "warnings": [],
            "search_api_failed": False,
            "llm_failed": False,
            "final_result": None,
            "start_time": datetime.utcnow(),
            "processing_time": None,
        }

        # Run the graph
        #
        # IMPORTANT: Prevent orphaned root spans.
        # LangGraph may schedule node execution in separate asyncio Tasks and/or executors.
        # Ensure the current OpenTelemetry context (parent span from the API request)
        # is explicitly propagated to the task running the graph.
        #
        ctx = contextvars.copy_context()
        task = asyncio.create_task(self.graph.ainvoke(initial_state), context=ctx)
        final_state = await task

        return final_state["final_result"], final_state.get("agent_reasoning", [])


def create_research_agent(
    bq_client: BigQueryClient | None = None,
) -> ResearchAgent:
    """Factory function to create a research agent."""
    return ResearchAgent(bq_client=bq_client)
