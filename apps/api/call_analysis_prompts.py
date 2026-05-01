"""Prompt templates for LangGraph call analysis (ported from CrewAI task descriptions)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@lru_cache(maxsize=4)
def _load(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


def classification_prompt(transcript: str, prior_context: str) -> str:
    return _load("classification.txt").format(
        transcript=transcript,
        prior_context=prior_context or "",
    )


def technical_prompt(transcript: str, prior_context: str) -> str:
    return _load("technical.txt").format(
        transcript=transcript,
        prior_context=prior_context or "",
    )


def sales_methodology_prompt(transcript: str, prior_context: str) -> str:
    return _load("sales_methodology.txt").format(
        transcript=transcript,
        prior_context=prior_context or "",
    )


def compile_prompt(
    transcript: str,
    prior_context: str,
    tech_output: str,
    sales_output: str,
) -> str:
    base = _load("compile_base.txt").format(
        transcript=transcript,
        prior_context=prior_context or "",
    )
    return (
        f"{base}\n\n"
        "===== TECHNICAL EVALUATION OUTPUT =====\n"
        f"{tech_output}\n\n"
        "===== SALES METHODOLOGY OUTPUT =====\n"
        f"{sales_output}\n"
    )


def recap_slide_prompt(
    transcript: str,
    classification_context: str,
    call_summary: str,
    missing_elements_json: str,
    recommendations_json: str,
    suggested_questions_json: str,
) -> str:
    """Build recap slide prompt (same structure as former CrewAI recap task)."""
    return f"""Based on the following call transcript and analysis, generate a recap slide for the next call.

TRANSCRIPT:
{transcript[:10000]}

ANALYSIS CONTEXT:
{classification_context}

Call Summary: {call_summary}

=== MISSING ELEMENTS (gaps that need to be addressed) ===
{missing_elements_json if missing_elements_json else "None identified"}

=== RECOMMENDATIONS FOR NEXT CALL ===
{recommendations_json if recommendations_json else "None identified"}

=== SUGGESTED QUESTIONS FROM MISSED OPPORTUNITIES ===
{suggested_questions_json if suggested_questions_json else "None identified"}

FIRST, extract the following metadata from the transcript:
- CUSTOMER NAME: The company/organization name of the prospect or customer (e.g., "Fidelity", "Acme Corp", "Netflix").
  Look for introductions like "Hi, I'm [name] from [COMPANY]" or speaker labels like "[Company Name] John Smith:".
  If multiple customer attendees are present, use their company name.
- CALL DATE: The date of the call if mentioned in the transcript, otherwise leave empty.

Then generate a recap with these 4 sections:

1. KEY INITIATIVES - What the customer is trying to accomplish (their goals and projects)
   - Focus on their strategic objectives and what they want to build/achieve
   - Use specific details from the call (team names, project names, timelines mentioned)

2. CHALLENGES - Pain points and problems they're facing
   - What's blocking them or causing friction
   - Be specific about the impact (time wasted, manual processes, lack of visibility)

3. SOLUTION REQUIREMENTS - What they need from a solution
   - Technical requirements mentioned (integrations, features, capabilities)
   - Business requirements (security, scalability, ease of use)

4. FOLLOW-UP QUESTIONS - Probing questions to ask on the NEXT call
   - CRITICAL: Use the MISSING ELEMENTS, RECOMMENDATIONS, and SUGGESTED QUESTIONS above as your PRIMARY source
   - Convert each gap or recommendation into a specific, actionable question
   - Format as actual questions starting with "Can you help me understand...", "What would happen if...", "Who is responsible for...", etc.
   - Include 4-6 high-value questions that directly address the gaps identified
   - Questions should be specific to THIS customer's situation, not generic

Return as JSON:
{{
    "customer_name": "Company name of the prospect/customer",
    "call_date": "Date if mentioned, otherwise empty string",
    "key_initiatives": ["Initiative 1 with specific details", "Initiative 2"],
    "challenges": ["Challenge 1 with impact", "Challenge 2"],
    "solution_requirements": ["Requirement 1", "Requirement 2", "Requirement 3"],
    "follow_up_questions": [
        "Can you help me understand [specific gap from missing elements]?",
        "What would be the impact if [relates to a challenge]?",
        "Who on your team is responsible for [specific area]?",
        "What timeline are you working with for [specific initiative]?"
    ]
}}

IMPORTANT:
- The customer_name field is REQUIRED - extract the prospect/customer company name from the transcript.
- The follow_up_questions MUST be derived from the MISSING ELEMENTS and RECOMMENDATIONS sections above. Do NOT generate generic questions - each question should address a specific gap identified in the analysis.
"""
