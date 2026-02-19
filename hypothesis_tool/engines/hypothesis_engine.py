"""Hypothesis engine - generates ranked hypotheses for prospects.

Combines industry playbooks, research signals, and competitive context
to generate actionable hypotheses with discovery questions.
"""

import json
from pathlib import Path

import anthropic

from ..models.playbook import IndustryPlaybook
from ..models.hypothesis import (
    Hypothesis,
    HypothesisConfidence,
    HypothesisResult,
    DiscoveryQuestion,
    SimilarCustomer,
    CompetitiveBattlecard,
    CompanyResearch,
    CompetitiveSituation,
)
from ..clients.bigquery_client import BigQueryClient
from ..config import get_settings


HYPOTHESIS_GENERATION_PROMPT = """You are helping a sales rep prepare for outreach to a prospect. Generate hypotheses about where Arize (an AI/ML observability platform) could add value.

## Company Research
{company_research}

## Industry Playbook (based on {deal_count} won deals in {industry})
Top Pain Points:
{pain_points}

Winning Value Props:
{value_props}

## Competitive Situation
{competitive_context}

## Task
Generate 2-3 ranked hypotheses about where Arize could help this company. For each hypothesis:

1. Write a clear hypothesis statement
2. Assign confidence (high/medium/low) with reasoning
3. List supporting signals from the research
4. List matching patterns from the playbook
5. Write 2-3 discovery questions to validate this hypothesis (make them specific to the signals found)

Respond in JSON:
{{
    "hypotheses": [
        {{
            "hypothesis": "statement",
            "confidence": "high|medium|low",
            "confidence_reasoning": "why this confidence level",
            "supporting_signals": ["signal 1", "signal 2"],
            "playbook_matches": ["pattern from won deals"],
            "discovery_questions": [
                {{
                    "question": "the question",
                    "rationale": "why ask this",
                    "tied_to_signal": "what triggered this question"
                }}
            ]
        }}
    ]
}}

Make questions specific to what we found in research, not generic. Reference specific findings."""


class HypothesisEngine:
    """Generates ranked hypotheses for prospects."""

    def __init__(
        self,
        bq_client: BigQueryClient | None = None,
        llm_model: str | None = None,
        playbook_dir: str | Path | None = None,
        battlecard_path: str | Path | None = None,
    ):
        settings = get_settings()
        self.bq = bq_client or BigQueryClient(project_id=settings.bq_project_id)
        self.llm_model = llm_model or settings.llm_model
        self.anthropic = anthropic.Anthropic(api_key=settings.anthropic_api_key)

        # Set default paths
        base_path = Path(__file__).parent.parent / "data"
        self.playbook_dir = Path(playbook_dir) if playbook_dir else base_path / "playbooks"
        self.battlecard_path = (
            Path(battlecard_path) if battlecard_path else base_path / "battlecards.json"
        )

        # Load battlecards
        self.battlecards = self._load_battlecards()

    def _load_battlecards(self) -> dict:
        """Load competitive battlecards from JSON."""
        if self.battlecard_path.exists():
            with open(self.battlecard_path) as f:
                return json.load(f)
        return {}

    def _load_playbook(self, industry: str) -> IndustryPlaybook | None:
        """Load a playbook for an industry."""
        if not self.playbook_dir.exists():
            return None

        # Try exact match first
        filename = industry.lower().replace(" ", "_").replace("/", "_")
        filepath = self.playbook_dir / f"{filename}.json"

        if filepath.exists():
            with open(filepath) as f:
                data = json.load(f)
            return IndustryPlaybook(**data)

        # Try fuzzy match
        for file in self.playbook_dir.glob("*.json"):
            if industry.lower() in file.stem.lower().replace("_", " "):
                with open(file) as f:
                    data = json.load(f)
                return IndustryPlaybook(**data)

        return None

    def _get_battlecard(
        self,
        competitive_situation: CompetitiveSituation,
        detected_tools: list[str],
    ) -> CompetitiveBattlecard:
        """Get appropriate battlecard for competitive situation."""
        # Map situation to battlecard key
        if competitive_situation == CompetitiveSituation.SWITCHING and detected_tools:
            # Use first detected tool's battlecard
            tool_key = detected_tools[0].lower().replace(" ", "_")
            if tool_key in self.battlecards:
                bc = self.battlecards[tool_key]
                return CompetitiveBattlecard(
                    situation=competitive_situation,
                    detected_competitor=bc.get("name", tool_key),
                    positioning=f"Switching from {bc.get('name', tool_key)}",
                    key_questions=bc.get("displacement_questions", []),
                    advantages=bc.get("arize_advantages", []),
                    watch_outs=bc.get("watch_outs", []),
                )

        elif competitive_situation == CompetitiveSituation.BUILD_VS_BUY:
            if "homegrown" in self.battlecards:
                bc = self.battlecards["homegrown"]
                return CompetitiveBattlecard(
                    situation=competitive_situation,
                    detected_competitor="Homegrown Solution",
                    positioning="Build vs Buy",
                    key_questions=bc.get("displacement_questions", []),
                    advantages=bc.get("arize_advantages", []),
                    watch_outs=bc.get("watch_outs", []),
                )

        elif competitive_situation == CompetitiveSituation.GREENFIELD:
            if "greenfield" in self.battlecards:
                bc = self.battlecards["greenfield"]
                return CompetitiveBattlecard(
                    situation=competitive_situation,
                    positioning="Build it right from the start",
                    key_questions=bc.get("displacement_questions", []),
                    advantages=bc.get("arize_advantages", []),
                    watch_outs=bc.get("watch_outs", []),
                )

        # Default unknown situation
        return CompetitiveBattlecard(
            situation=CompetitiveSituation.UNKNOWN,
            positioning="Gather more information about current tooling",
            key_questions=[
                "What does your current ML monitoring stack look like?",
                "How do you currently track model performance in production?",
            ],
            advantages=[],
            watch_outs=["Need to understand their current situation better"],
        )

    def _format_research_for_prompt(self, research: CompanyResearch) -> str:
        """Format company research for the LLM prompt."""
        lines = [
            f"Company: {research.company_name}",
        ]

        if research.industry:
            lines.append(f"Industry: {research.industry}")
        if research.employee_count:
            lines.append(f"Employees: {research.employee_count:,}")

        lines.append(f"\nAI/ML Confidence: {research.ai_ml_confidence.value.upper()}")

        if research.ai_ml_signals:
            lines.append("\nAI/ML Signals Found:")
            for signal in research.ai_ml_signals[:5]:
                lines.append(f"  - [{signal.confidence.value}] {signal.evidence}")

        if research.company_summary:
            lines.append(f"\nSummary: {research.company_summary}")

        if research.exists_in_crm:
            lines.append("\nNote: This company exists in our CRM")

        return "\n".join(lines)

    def _format_playbook_for_prompt(self, playbook: IndustryPlaybook) -> tuple[str, str]:
        """Format playbook pain points and value props for prompt."""
        pain_lines = []
        for pp in playbook.top_pain_points[:5]:
            pain_lines.append(f"  - {pp.pain} (mentioned in {pp.frequency} deals)")

        value_lines = []
        for vp in playbook.winning_value_props[:5]:
            value_lines.append(f"  - {vp.value}")

        return "\n".join(pain_lines), "\n".join(value_lines)

    def _format_competitive_for_prompt(
        self,
        research: CompanyResearch,
        battlecard: CompetitiveBattlecard,
    ) -> str:
        """Format competitive context for prompt."""
        lines = [f"Situation: {research.competitive_situation.value}"]

        if research.detected_tools:
            lines.append(f"Detected tools: {', '.join(research.detected_tools)}")

        lines.append(f"Positioning: {battlecard.positioning}")

        if battlecard.advantages:
            lines.append("Arize advantages:")
            for adv in battlecard.advantages[:3]:
                lines.append(f"  - {adv}")

        return "\n".join(lines)

    async def generate_hypotheses(
        self,
        research: CompanyResearch,
        playbook: IndustryPlaybook | None = None,
    ) -> HypothesisResult:
        """Generate hypotheses for a prospect.

        Args:
            research: Company research results
            playbook: Optional industry playbook (will load from file if not provided)

        Returns:
            Complete hypothesis result
        """
        import time
        start_time = time.time()

        # Load playbook if not provided
        if playbook is None and research.industry:
            playbook = self._load_playbook(research.industry)

        # Get battlecard
        battlecard = self._get_battlecard(
            research.competitive_situation,
            research.detected_tools,
        )

        # Prepare prompt components
        research_text = self._format_research_for_prompt(research)

        if playbook:
            pain_text, value_text = self._format_playbook_for_prompt(playbook)
            industry = playbook.industry
            deal_count = playbook.deal_count
        else:
            pain_text = "No industry playbook available - using general patterns"
            value_text = "Real-time drift detection, faster debugging, better root cause analysis"
            industry = research.industry or "Unknown"
            deal_count = 0

        competitive_text = self._format_competitive_for_prompt(research, battlecard)

        # Generate with LLM
        prompt = HYPOTHESIS_GENERATION_PROMPT.format(
            company_research=research_text,
            deal_count=deal_count,
            industry=industry,
            pain_points=pain_text,
            value_props=value_text,
            competitive_context=competitive_text,
        )

        response = self.anthropic.messages.create(
            model=self.llm_model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = response.content[0].text

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
        except json.JSONDecodeError:
            data = {"hypotheses": []}

        # Build hypothesis objects
        hypotheses = []
        for h in data.get("hypotheses", []):
            # Map confidence string to enum
            conf_str = h.get("confidence", "low").lower()
            confidence = {
                "high": HypothesisConfidence.HIGH,
                "medium": HypothesisConfidence.MEDIUM,
                "low": HypothesisConfidence.LOW,
            }.get(conf_str, HypothesisConfidence.LOW)

            # Build discovery questions
            questions = []
            for q in h.get("discovery_questions", []):
                questions.append(
                    DiscoveryQuestion(
                        question=q.get("question", ""),
                        rationale=q.get("rationale", ""),
                        tied_to_signal=q.get("tied_to_signal"),
                    )
                )

            hypotheses.append(
                Hypothesis(
                    hypothesis=h.get("hypothesis", ""),
                    confidence=confidence,
                    confidence_reasoning=h.get("confidence_reasoning", ""),
                    supporting_signals=h.get("supporting_signals", []),
                    playbook_matches=h.get("playbook_matches", []),
                    discovery_questions=questions,
                )
            )

        # Get similar customers if we have industry
        similar_customers = []
        if research.industry:
            try:
                similar = self.bq.get_similar_customers(
                    industry=research.industry,
                    employee_count=research.employee_count,
                    limit=3,
                )
                for s in similar:
                    similar_customers.append(
                        SimilarCustomer(
                            customer_name=s.account_name,
                            industry=s.industry,
                            deal_size=s.amount,
                        )
                    )
            except Exception:
                pass  # Silently fail if we can't get similar customers

        # Attach similar customers to first hypothesis
        if hypotheses and similar_customers:
            hypotheses[0].similar_customers = similar_customers

        # Build confidence note
        confidence_note = None
        if not playbook:
            confidence_note = (
                f"Limited data: No playbook available for {industry}. "
                "Using general ML observability patterns."
            )
        elif playbook.deal_count < 10:
            confidence_note = (
                f"Limited data: Only {playbook.deal_count} deals in {industry}. "
                "Patterns may not be representative."
            )

        processing_time = time.time() - start_time

        return HypothesisResult(
            company_name=research.company_name,
            research=research,
            hypotheses=hypotheses,
            competitive_context=battlecard,
            playbook_used=playbook.industry if playbook else None,
            playbook_deal_count=playbook.deal_count if playbook else None,
            confidence_note=confidence_note,
            processing_time_seconds=processing_time,
        )
