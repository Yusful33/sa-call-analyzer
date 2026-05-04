"""
Maps app classification output to inputs for the **arize-synthetic-demo** Claude skill.

Skill source layout (Solutions team):
https://github.com/Arize-ai/solutions-resources/tree/main/.claude/skills/arize-synthetic-demo

Synthetic traces + generator.py workflows run in Claude / local Python — not inside this FastAPI bundle.
"""

from __future__ import annotations

import re
from typing import Any, Optional


SYNTHETIC_DEMO_SKILL = {
    "repository": "https://github.com/Arize-ai/solutions-resources",
    "skill_path": ".claude/skills/arize-synthetic-demo",
    "skill_document": ".claude/skills/arize-synthetic-demo/SKILL.md",
}

# Enumerations documented in SKILL.md ("Inputs you must gather" + optional scenarios).
SKILL_FRAMEWORK_VALUES: frozenset[str] = frozenset(
    {
        "openai",
        "anthropic",
        "bedrock",
        "vertex",
        "adk",
        "langchain",
        "langgraph",
        "crewai",
        "generic",
    }
)

SKILL_AGENT_ARCHITECTURE_VALUES: frozenset[str] = frozenset(
    {
        "single_agent",
        "multi_agent_coordinator",
        "retrieval_pipeline",
        "rag_rerank",
        "guarded_rag",
    }
)

SKILL_SCENARIO_VALUES: frozenset[str] = frozenset(
    {
        "happy_path",
        "tool_failure",
        "guardrail_denial",
        "ambiguity",
        "execution_failure",
        "retry",
        "poisoned_tokens",
        "no_llm_needed",
    }
)


def slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip().lower()).strip("_")
    return s or "demo"


def _use_case_to_agent_architecture(use_case: str) -> str:
    """Maps internal demo taxonomy → skill's ``agent_architecture`` enum."""
    m: dict[str, str] = {
        "text-to-sql-bi-agent": "retrieval_pipeline",
        "retrieval-augmented-search": "rag_rerank",
        "multi-agent-orchestration": "multi_agent_coordinator",
        "classification-routing": "single_agent",
        "multimodal-ai": "single_agent",
        "mcp-tool-use": "single_agent",
        "multiturn-chatbot-with-tools": "single_agent",
        "travel-agent": "single_agent",
        "guardrails": "guarded_rag",
        "generic": "single_agent",
    }
    return m.get(use_case, "single_agent")


def _internal_framework_to_skill_framework(framework: str) -> str:
    """Maps /api classify ``framework`` to skill ``framework`` list where possible."""
    m = {
        "langgraph": "langgraph",
        "langchain": "langchain",
        "crewai": "crewai",
        "adk": "adk",
    }
    return m.get((framework or "langgraph").lower(), "generic")


def build_industry_or_use_case(
    *,
    industry: Optional[str],
    use_case_internal: str,
    additional_context: Optional[str],
) -> str:
    parts = []
    if industry:
        parts.append(f"Industry: {industry}.")
    parts.append(f"Demo pattern (classification): {use_case_internal}")
    if (additional_context or "").strip():
        parts.append(f"Scenario context: {additional_context.strip()}")
    return " ".join(parts)


def build_synthetic_demo_skill_hints(
    *,
    company_name: str,
    industry: Optional[str],
    use_case: str,
    framework: str,
    reasoning: Optional[str],
    additional_context: Optional[str],
    skill_framework: Optional[str] = None,
    agent_architecture: Optional[str] = None,
    num_traces: Optional[int] = None,
    with_evals: Optional[bool] = None,
    with_dataset_and_experiments: Optional[bool] = None,
    scenarios: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    JSON-ready dict aligned with SKILL.md inputs (subset).
    Claude / SA fills the rest inside the IDE using the skill.
    """
    company = (company_name or "").strip() or "AcmeCorp"
    slug_company = slugify(company)
    slug_uc = slugify(use_case)
    suggested_dir = f"~/arize-repos/{slug_company}_{slug_uc}"
    proj = f"{slug_company}_{slug_uc}_synthetic"

    skill_fw = (
        (skill_framework or "").strip().lower()
        if (skill_framework or "").strip().lower() in SKILL_FRAMEWORK_VALUES
        else _internal_framework_to_skill_framework(framework)
    )
    arch_raw = (agent_architecture or "").strip()
    arch = (
        arch_raw
        if arch_raw in SKILL_AGENT_ARCHITECTURE_VALUES
        else _use_case_to_agent_architecture(use_case)
    )
    if isinstance(num_traces, int) and num_traces > 0:
        n_traces = min(max(num_traces, 50), 10_000)
    else:
        n_traces = 500
    we = True if with_evals is None else bool(with_evals)
    wde = True if with_dataset_and_experiments is None else bool(with_dataset_and_experiments)
    scen_list: list[str] = []
    if scenarios:
        scen_list = [s for s in scenarios if s in SKILL_SCENARIO_VALUES]

    suggested_prompt_for_claude = (
        "Create a synthetic Arize demo using the **arize-synthetic-demo** skill. "
        f"Use company `{company}`, industry/context `{build_industry_or_use_case(industry=industry, use_case_internal=use_case, additional_context=additional_context)}`, "
        f"skill framework `{skill_fw}`, agent_architecture `{arch}`, ~{n_traces} traces, output under `{suggested_dir}`. "
        f"Set with_evals={we}, with_dataset_and_experiments={wde}"
        + (f", scenarios={scen_list}" if scen_list else "")
        + ". "
        f"Our app classifier suggested use_case=`{use_case}`, internal orchestration framework=`{framework}`. Reasoning: {reasoning or 'n/a'}."
    )

    rec: dict[str, Any] = {
        "company_name": company,
        "industry_or_use_case": build_industry_or_use_case(
            industry=industry,
            use_case_internal=use_case,
            additional_context=additional_context,
        ),
        "framework": skill_fw,
        "agent_architecture": arch,
        "num_traces": n_traces,
        "output_dir_example": suggested_dir,
        "project_name_example": proj,
        "with_evals": we,
        "with_dataset_and_experiments": wde,
        "classification_from_sa_call_analyzer": {
            "use_case": use_case,
            "framework": framework,
            "reasoning": reasoning,
        },
    }
    if scen_list:
        rec["scenarios"] = scen_list

    return {
        "skill": SYNTHETIC_DEMO_SKILL,
        "suggested_prompt_for_claude": suggested_prompt_for_claude.strip(),
        "recommended_inputs": rec,
        "next_steps": [
            "Ensure the Claude **arize-synthetic-demo** skill is available (Solutions resources repo symlink or Cursor skills path).",
            "Paste ``suggested_prompt_for_claude`` into Claude Code / Cursor, or invoke the skill and fill **recommended_inputs**.",
            "Run the generated ``generator.py`` locally (see SKILL.md `--test`, then `--full` when credentials exist).",
        ],
    }


def static_skill_info() -> dict[str, Any]:
    """Returned by GET /api/custom-demo/skill for clients that only need links + checklist."""
    return {
        **SYNTHETIC_DEMO_SKILL,
        "summary": (
            "Custom synthetic trace demos are generated offline via the **arize-synthetic-demo** Claude skill "
            "(generator.py + notebooks), not by this HTTP API."
        ),
        "required_skill_inputs_checklist": [
            "company_name",
            "industry_or_use_case",
            "framework (openai|anthropic|bedrock|vertex|adk|langchain|langgraph|crewai|generic)",
            "agent_architecture (single_agent|multi_agent_coordinator|retrieval_pipeline|rag_rerank|guarded_rag)",
            "num_traces (default 500)",
            "output_dir (explicit path)",
        ],
        "optional_credentials_for_auto_run": ["ARIZE_SPACE_ID", "ARIZE_API_KEY", "ARIZE_PROJECT_NAME or project slug"],
        "references": "SKILL.md in the repo above lists scenarios, evaluations, datasets/experiments, and AX CLI flows.",
    }
