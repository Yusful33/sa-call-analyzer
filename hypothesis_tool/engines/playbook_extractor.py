"""Playbook extractor - aggregates Gong AI insights into industry playbooks.

This module queries won deals from Salesforce, fetches linked Gong call insights,
and uses an LLM to aggregate patterns into structured playbooks.
"""

import json
from datetime import datetime
from pathlib import Path

import anthropic

from ..clients.bigquery_client import BigQueryClient, WonOpportunity, GongCallInsight
from ..models.playbook import (
    IndustryPlaybook,
    PainPoint,
    ValueProp,
    Objection,
    DiscoveryQuestionTemplate,
)
from ..config import get_settings


PLAYBOOK_EXTRACTION_PROMPT = """You are analyzing Gong call summaries from won deals to extract patterns for a sales playbook.

Industry: {industry}
Number of deals analyzed: {deal_count}

Below are the AI-generated call summaries (CALL_SPOTLIGHT_BRIEF) and key points from these won deals:

{call_summaries}

Based on these call summaries, extract the following patterns:

1. **Top Pain Points** (5-7 items): What problems did prospects mention that led them to buy? Include frequency count.

2. **Winning Value Propositions** (3-5 items): What Arize value props resonated most? What made them decide to buy?

3. **Common Objections** (3-5 items): What objections came up? Include when they typically appeared (early/mid/late stage) and effective responses.

4. **Discovery Questions** (5-7 items): Based on these patterns, what questions should AEs ask to uncover these pain points?

Respond in JSON format:
{{
    "top_pain_points": [
        {{"pain": "description", "frequency": number, "example_quote": "quote from call if available"}}
    ],
    "winning_value_props": [
        {{"value": "description", "frequency": number, "use_case": "specific use case"}}
    ],
    "common_objections": [
        {{"objection": "the objection", "stage_typically_appears": "early|mid|late", "effective_response": "what worked", "frequency": number}}
    ],
    "discovery_questions": [
        {{"question": "the question", "tied_to_signal": "what triggers this", "validates_hypothesis": "what this tests"}}
    ]
}}

Focus on patterns that appear multiple times across deals. Be specific, not generic."""


class PlaybookExtractor:
    """Aggregates Gong AI insights into industry playbooks."""

    def __init__(
        self,
        bq_client: BigQueryClient | None = None,
        llm_model: str | None = None,
    ):
        settings = get_settings()
        self.bq = bq_client or BigQueryClient(project_id=settings.bq_project_id)
        self.llm_model = llm_model or settings.llm_model
        self.anthropic = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def _format_call_summaries(
        self,
        opps: list[WonOpportunity],
        insights: list[GongCallInsight],
    ) -> str:
        """Format call summaries for the LLM prompt."""
        # Group insights by opportunity
        insights_by_opp = {}
        for insight in insights:
            if insight.opp_id not in insights_by_opp:
                insights_by_opp[insight.opp_id] = []
            insights_by_opp[insight.opp_id].append(insight)

        # Build formatted output
        lines = []
        for opp in opps:
            opp_insights = insights_by_opp.get(opp.opp_id, [])
            if not opp_insights:
                continue

            lines.append(f"\n{'='*60}")
            lines.append(f"DEAL: {opp.account_name} (${opp.amount:,.0f})")
            lines.append(f"{'='*60}")

            for insight in opp_insights[:3]:  # Limit to 3 calls per deal
                lines.append(f"\nCall: {insight.call_title}")
                if insight.spotlight_brief:
                    lines.append(f"Summary: {insight.spotlight_brief}")
                if insight.key_points:
                    lines.append("Key Points:")
                    for kp in insight.key_points[:5]:  # Limit key points
                        lines.append(f"  - {kp}")
                if insight.outcome:
                    lines.append(f"Outcome: {insight.outcome}")

        return "\n".join(lines)

    async def extract_playbook(
        self,
        industry: str,
        min_deals: int = 10,
    ) -> IndustryPlaybook:
        """Extract a playbook for an industry.

        Args:
            industry: Industry name to extract playbook for
            min_deals: Minimum number of deals required

        Returns:
            Structured industry playbook
        """
        # Get won deals and call insights
        opps, insights = self.bq.get_calls_for_industry(
            industry=industry,
            min_deals=min_deals,
        )

        if len(opps) < min_deals:
            print(
                f"Warning: Only {len(opps)} deals for {industry}, "
                f"playbook may have limited patterns"
            )

        # Format call summaries for LLM
        call_summaries = self._format_call_summaries(opps, insights)

        # Generate playbook with LLM
        prompt = PLAYBOOK_EXTRACTION_PROMPT.format(
            industry=industry,
            deal_count=len(opps),
            call_summaries=call_summaries,
        )

        response = self.anthropic.messages.create(
            model=self.llm_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        # Parse LLM response
        response_text = response.content[0].text

        # Extract JSON from response (handle markdown code blocks)
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
            print(f"Failed to parse LLM response: {e}")
            print(f"Response was: {response_text[:500]}")
            # Return empty playbook on parse failure
            data = {
                "top_pain_points": [],
                "winning_value_props": [],
                "common_objections": [],
                "discovery_questions": [],
            }

        # Calculate average deal size
        avg_deal_size = sum(o.amount for o in opps) / len(opps) if opps else 0

        # Build playbook
        playbook = IndustryPlaybook(
            industry=industry,
            deal_count=len(opps),
            generated_at=datetime.utcnow(),
            top_pain_points=[
                PainPoint(**pp) for pp in data.get("top_pain_points", [])
            ],
            winning_value_props=[
                ValueProp(**vp) for vp in data.get("winning_value_props", [])
            ],
            common_objections=[
                Objection(**obj) for obj in data.get("common_objections", [])
            ],
            discovery_questions=[
                DiscoveryQuestionTemplate(**dq)
                for dq in data.get("discovery_questions", [])
            ],
            sample_customers=[o.account_name for o in opps[:5]],
            avg_deal_size=avg_deal_size,
        )

        return playbook

    async def extract_all_playbooks(
        self,
        top_n_industries: int = 5,
        min_deals_per_industry: int = 10,
        output_dir: str | Path | None = None,
    ) -> list[IndustryPlaybook]:
        """Generate playbooks for top N industries by deal count.

        Args:
            top_n_industries: Number of top industries to generate playbooks for
            min_deals_per_industry: Minimum deals required per industry
            output_dir: Optional directory to save playbooks as JSON

        Returns:
            List of generated playbooks
        """
        # Get industry stats
        stats = self.bq.get_industry_stats()

        # Filter to industries with enough deals
        qualifying = [
            s for s in stats if s.deal_count >= min_deals_per_industry
        ][:top_n_industries]

        print(f"Generating playbooks for {len(qualifying)} industries:")
        for s in qualifying:
            print(f"  - {s.industry}: {s.deal_count} deals")

        # Generate playbooks
        playbooks = []
        for stat in qualifying:
            print(f"\nExtracting playbook for {stat.industry}...")
            playbook = await self.extract_playbook(
                industry=stat.industry,
                min_deals=min_deals_per_industry,
            )
            playbooks.append(playbook)

            # Save to file if output_dir specified
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)

                # Sanitize industry name for filename
                filename = stat.industry.lower().replace(" ", "_").replace("/", "_")
                filepath = output_path / f"{filename}.json"

                with open(filepath, "w") as f:
                    f.write(playbook.model_dump_json(indent=2))
                print(f"  Saved to {filepath}")

        return playbooks

    def load_playbook(self, industry: str, playbook_dir: str | Path) -> IndustryPlaybook | None:
        """Load a playbook from JSON file.

        Args:
            industry: Industry name
            playbook_dir: Directory containing playbook JSON files

        Returns:
            Loaded playbook or None if not found
        """
        playbook_path = Path(playbook_dir)

        # Try exact filename match
        filename = industry.lower().replace(" ", "_").replace("/", "_")
        filepath = playbook_path / f"{filename}.json"

        if filepath.exists():
            with open(filepath) as f:
                data = json.load(f)
            return IndustryPlaybook(**data)

        # Try fuzzy match
        for file in playbook_path.glob("*.json"):
            if industry.lower() in file.stem.lower():
                with open(file) as f:
                    data = json.load(f)
                return IndustryPlaybook(**data)

        return None
