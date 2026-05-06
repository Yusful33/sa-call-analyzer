"""
Generate populated PoC / PoT Word documents from BigQuery-backed ProspectOverview.

Templates under templates/poc_pot/ use bracket placeholders like [Company Name] and
fixed boilerplate lines (trial dates, success examples). We discover those strings in
the document (body, tables, headers), ask an LLM for replacement text per key, then
substitute in-place (no appendix appended at the end).
"""

from __future__ import annotations

import io
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from openai_compat_completion import completion as llm_completion
from docx import Document
from docx.text.paragraph import Paragraph
from pydantic import BaseModel, Field, ValidationError

from models import ProspectOverview


class AppendixGenerationError(Exception):
    """Raised when the LLM output cannot be parsed into the expected JSON shape."""


BASE_DIR = Path(__file__).resolve().parent

def _find_template_dir() -> Path:
    """Find template directory - check api/ folder first (Vercel bundle), then root."""
    candidates = [
        BASE_DIR / "api" / "templates" / "poc_pot",  # Vercel bundles api/ folder
        BASE_DIR / "templates" / "poc_pot",          # Local dev / original location
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.docx")):
            return candidate
    return candidates[-1]  # Default to original location

TEMPLATE_DIR = _find_template_dir()

TEMPLATES: dict[str, Path] = {
    "poc_saas": TEMPLATE_DIR / "poc_saas.docx",
    "poc_vpc": TEMPLATE_DIR / "poc_vpc.docx",
    "pot": TEMPLATE_DIR / "pot.docx",
}

BRACKET_PLACEHOLDER_RE = re.compile(r"\[[^\]]{1,220}\]")

# Long literal lines present in PoC templates (not always in PoT). We only require keys
# that actually appear somewhere in the loaded document.
EXTRA_LITERAL_CANDIDATES_BY_TEMPLATE: dict[str, Tuple[str, ...]] = {
    "poc_saas": (
        "___________________________________________",
        "Start: _____ / End: _____ - [X] Weeks  ",
        "e.g., Mean time to root cause reduced from ___ days to < 4 hrs",
        "e.g., ___ prompt variants tested per sprint (target: 2-3x current)",
        "e.g., Drift alert configured for top 10 features; PSI threshold = ___",
    ),
    "poc_vpc": (
        "___________________________________________",
        "Start: _____ / End: _____ - [X] Weeks  ",
        "e.g., Mean time to root cause reduced from ___ days to < 4 hrs",
        "e.g., ___ prompt variants tested per sprint (target: 2-3x current)",
        "e.g., Drift alert configured for top 10 features; PSI threshold = ___",
    ),
    "pot": (
        "Start: _____ / End: _____ - [X] Weeks  ",
    ),
}


class IntegratedFillResult(BaseModel):
    """LLM output: map each template token to its replacement string."""

    replacements: Dict[str, str] = Field(default_factory=dict)


def _deep_truncate(obj: Any, max_str: int = 2200, max_list: int = 60) -> Any:
    if isinstance(obj, dict):
        return {k: _deep_truncate(v, max_str, max_list) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_truncate(i, max_str, max_list) for i in obj[:max_list]]
    if isinstance(obj, str) and len(obj) > max_str:
        return obj[:max_str] + "…"
    return obj


def _overview_to_prompt_json(overview: ProspectOverview) -> str:
    raw = overview.model_dump(mode="json", exclude_none=True)
    trimmed = _deep_truncate(raw)
    try:
        text = json.dumps(trimmed, indent=2, default=str, allow_nan=False)
    except (ValueError, TypeError):
        text = json.dumps(trimmed, indent=2, default=str)
    max_chars = int(os.environ.get("POC_DOC_MAX_PROMPT_CHARS", "75000"))
    if len(text) > max_chars:
        return (
            text[:max_chars]
            + "\n\n…(truncated for model context; prioritize non-null fields above when filling.)\n"
        )
    return text


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        parts = t.split("```")
        if len(parts) >= 2:
            inner = parts[1]
            if inner.lstrip().startswith("json"):
                inner = inner.lstrip()[4:]
            return inner.strip()
    return t


def _extract_json_object(raw: str) -> dict:
    """
    Parse a JSON object from an LLM message that may include prose or markdown fences.
    """
    s = raw.strip().lstrip("\ufeff")
    s = _strip_json_fence(s)
    try:
        data = json.loads(s)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    start = s.find("{")
    if start < 0:
        raise ValueError("No JSON object found in model response")
    decoder = json.JSONDecoder()
    try:
        obj, _end = decoder.raw_decode(s[start:])
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in model response: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError("Top-level JSON must be an object")
    return obj


def _overview_blob(overview: ProspectOverview) -> str:
    try:
        return json.dumps(overview.model_dump(mode="json", exclude_none=True), default=str).lower()
    except Exception:
        return ""


def _success_criteria_evidence_digest(overview: ProspectOverview) -> str:
    """Keyword scan over the full overview JSON to steer success-criteria lines
    away from generic ML/LLM filler when the opportunity never discussed those topics."""
    blob = _overview_blob(overview)
    if not blob.strip():
        return (
            "### Evidence scan for success criteria\n"
            "*No structured overview text — use only TBD / N/A style placeholders; do not invent ML metrics.*"
        )

    checks: list[tuple[str, tuple[str, ...]]] = [
        (
            "LLMs / GenAI / agents / prompts / RAG",
            (
                "llm",
                "large language",
                "genai",
                "generative ai",
                "gpt-",
                "claude",
                "openai",
                "anthropic",
                "copilot",
                "bedrock",
                "vertex ai",
                "agent",
                "tool call",
                "function calling",
                "rag",
                "retrieval augmented",
                "embedding",
                "vector",
                "pgvector",
                "pinecone",
                "weaviate",
                "langchain",
                "langgraph",
                "llamaindex",
                "prompt",
                "hallucination",
                "chatbot",
                "assistant",
                "token",
            ),
        ),
        (
            "Classical ML / tabular / batch models",
            (
                "xgboost",
                "sklearn",
                "scikit",
                "random forest",
                "tabular",
                "classification model",
                "regression model",
                "feature store",
                "training pipeline",
                "batch prediction",
                "spark ml",
                "sagemaker",
            ),
        ),
        (
            "Model monitoring / drift / production performance",
            (
                "drift",
                "psi",
                "data quality",
                "model performance",
                "serving",
                "inference latency",
                "shadow deployment",
                "champion",
                "challenger",
            ),
        ),
        (
            "Evaluations / benchmarks / golden sets",
            (
                "eval",
                "evaluation",
                "benchmark",
                "golden set",
                "offline eval",
                "online eval",
                "regression test",
            ),
        ),
        (
            "Tracing / observability for systems",
            (
                "trace",
                "span",
                "otel",
                "opentelemetry",
                "observability",
                "arize",
                "instrument",
            ),
        ),
    ]

    found = [label for label, kws in checks if any(kw in blob for kw in kws)]
    missing = [label for label, _ in checks if label not in found]

    lines = [
        "### Evidence scan for success criteria (keyword signal over full ProspectOverview JSON)",
        "**Buckets with at least weak textual support:** "
        + ("; ".join(found) if found else "*(none detected — be very conservative)*"),
        "**Buckets with no keyword hit (do not assume in-scope):** "
        + ("; ".join(missing) if missing else "—"),
    ]
    return "\n".join(lines)


def _one_line_account_summary(overview: ProspectOverview) -> str:
    bits: list[str] = []
    if overview.salesforce:
        sf = overview.salesforce
        if (sf.name or "").strip():
            bits.append(sf.name.strip())
        if (sf.industry or "").strip():
            bits.append(sf.industry.strip())
    if overview.gong_summary and getattr(overview.gong_summary, "total_calls", None):
        bits.append(f"{overview.gong_summary.total_calls} Gong calls on record")
    if overview.pendo_usage:
        bits.append("Pendo usage on file")
    if overview.product_usage and overview.product_usage.adoption_status:
        bits.append(f"Adoption: {overview.product_usage.adoption_status}")
    base = " · ".join(bits) if bits else (overview.lookup_value or "Account")
    return (base + " — scope and metrics to confirm in discovery.")[:300]


def _first_choice_from_bracket_key(key: str) -> str | None:
    if not (key.startswith("[") and key.endswith("]")):
        return None
    inner = key[1:-1].strip()
    if not inner or inner == "X":
        return None
    parts = [p.strip() for p in inner.split("/") if p.strip()]
    if not parts:
        return None
    return parts[0][:220]


def _keyword_pick_for_bracket(key: str, overview: ProspectOverview) -> str | None:
    """Pick a template option when the key lists vendors/frameworks and CRM text hints at one."""
    blob = _overview_blob(overview)
    if not blob:
        return None
    pairs = [
        ("[OpenAI / Anthropic / AWS Bedrock / GCP Vertex / Azure OAI]", "openai", "OpenAI"),
        ("[OpenAI / Anthropic / AWS Bedrock / GCP Vertex / Azure OAI]", "anthropic", "Anthropic"),
        ("[OpenAI / Anthropic / AWS Bedrock / GCP Vertex / Azure OAI]", "bedrock", "AWS Bedrock"),
        ("[OpenAI / Anthropic / AWS Bedrock / GCP Vertex / Azure OAI]", "vertex", "GCP Vertex"),
        ("[OpenAI / Anthropic / AWS Bedrock / GCP Vertex / Azure OAI]", "azure openai", "Azure OAI"),
        ("[LangChain / LangGraph / LlamaIndex / Custom / etc.]", "langgraph", "LangGraph"),
        ("[LangChain / LangGraph / LlamaIndex / Custom / etc.]", "langchain", "LangChain"),
        ("[LangChain / LangGraph / LlamaIndex / Custom / etc.]", "llamaindex", "LlamaIndex"),
        ("[LangChain / LlamaIndex / LangGraph / Custom / None]", "langgraph", "LangGraph"),
        ("[LangChain / LlamaIndex / LangGraph / Custom / None]", "langchain", "LangChain"),
        ("[LangChain / LlamaIndex / LangGraph / Custom / None]", "llamaindex", "LlamaIndex"),
        ("[DataDog / Splunk / Homegrown / None]", "datadog", "DataDog"),
        ("[DataDog / Splunk / Homegrown / None]", "splunk", "Splunk"),
        ("[Pinecone / Weaviate / pgvector / Internal / N/A]", "pinecone", "Pinecone"),
        ("[Pinecone / Weaviate / pgvector / Internal / N/A]", "weaviate", "Weaviate"),
        ("[Pinecone / Weaviate / pgvector / Internal / N/A]", "pgvector", "pgvector"),
        ("[SaaS / VPC / On-Prem]", "vpc", "VPC"),
        ("[SaaS / VPC / On-Prem]", "on-prem", "On-Prem"),
        ("[SaaS / VPC / On-Prem]", "saas", "SaaS"),
        ("[LLM Calls / RAG / Agentic (Tool Calls) / Multi-Agent]", "multi-agent", "Multi-Agent"),
        ("[LLM Calls / RAG / Agentic (Tool Calls) / Multi-Agent]", "agentic", "Agentic (Tool Calls)"),
        ("[LLM Calls / RAG / Agentic (Tool Calls) / Multi-Agent]", "rag", "RAG"),
        ("[LLM Calls / RAG / Agentic (Tool Calls) / Multi-Agent]", "llm calls", "LLM Calls"),
    ]
    for k_needle, needle, label in pairs:
        if key == k_needle and needle in blob:
            return label
    return None


def _fallback_fill_for_key(
    key: str,
    overview: ProspectOverview,
    document_template: str,
    manual_notes: str | None,
) -> str:
    """Deterministic fill when the LLM omits a key, echoes the placeholder, or returns whitespace."""
    if key == "[Company Name]":
        return _company_display_name(overview)

    if key == "[X]":
        if document_template == "pot":
            return "1"
        if document_template == "poc_saas":
            return "2"
        return "4"

    if key == "___________________________________________":
        return _one_line_account_summary(overview)

    if key == "Start: _____ / End: _____ - [X] Weeks  ":
        # Dates are unknown without SA input — still replace underscores so the line reads as filled.
        if document_template == "pot":
            return "Start: TBD / End: TBD — under 1 week (align in kickoff)  "
        if document_template == "poc_saas":
            return "Start: TBD / End: TBD — target under 2 weeks (align in kickoff)  "
        return "Start: TBD / End: TBD — target under 4 weeks (align in kickoff)  "

    if key.startswith("e.g.,") and ("___" in key or "<" in key):
        return (
            key.replace("___", "TBD")
            .replace("_____", "TBD")
            + " (Refine with customer during scoping.)"
        )[:320]

    picked = _keyword_pick_for_bracket(key, overview)
    if picked:
        return picked

    if key.startswith("[") and key.endswith("]"):
        first = _first_choice_from_bracket_key(key)
        if first:
            return first

    if (manual_notes or "").strip():
        return f"TBD — see SA notes: {(manual_notes or '')[:160]}"

    return "TBD — confirm with account team"


def _coerce_value_for_key(
    key: str,
    raw: Any,
    overview: ProspectOverview,
    document_template: str,
    manual_notes: str | None,
) -> str:
    if key == "[Company Name]":
        return _company_display_name(overview)
    if raw is None:
        return _fallback_fill_for_key(key, overview, document_template, manual_notes)
    s = str(raw).strip()
    if not s or s == key.strip():
        return _fallback_fill_for_key(key, overview, document_template, manual_notes)
    return s[:4000]


def _iter_paragraphs_in_cell(cell: Any) -> Iterable[Paragraph]:
    for p in cell.paragraphs:
        yield p
    for t in cell.tables:
        for row in t.rows:
            for c in row.cells:
                yield from _iter_paragraphs_in_cell(c)


def _iter_body_paragraphs(doc: Document) -> Iterable[Paragraph]:
    for p in doc.paragraphs:
        yield p
    for t in doc.tables:
        for row in t.rows:
            for cell in row.cells:
                yield from _iter_paragraphs_in_cell(cell)


def _iter_part_paragraphs(part: Any) -> Iterable[Paragraph]:
    if part is None:
        return
    for p in part.paragraphs:
        yield p
    for t in part.tables:
        for row in t.rows:
            for cell in row.cells:
                yield from _iter_paragraphs_in_cell(cell)


def iter_all_paragraphs(doc: Document) -> Iterable[Paragraph]:
    """Body, tables, and header/footer paragraphs."""
    yield from _iter_body_paragraphs(doc)
    for section in doc.sections:
        yield from _iter_part_paragraphs(section.header)
        yield from _iter_part_paragraphs(section.footer)
        if section.different_first_page_header_footer:
            yield from _iter_part_paragraphs(section.first_page_header)
            yield from _iter_part_paragraphs(section.first_page_footer)


def discover_replacement_keys(doc: Document, document_template: str) -> List[str]:
    """Collect bracket tokens plus any configured literal line that exists in this file."""
    found: set[str] = set()
    paras = list(iter_all_paragraphs(doc))
    for p in paras:
        found.update(BRACKET_PLACEHOLDER_RE.findall(p.text))
    for literal in EXTRA_LITERAL_CANDIDATES_BY_TEMPLATE.get(document_template, ()):
        for p in paras:
            if literal in p.text:
                found.add(literal)
                break
    # Longest keys first so shorter tokens (e.g. [X]) do not break longer literals prematurely.
    return sorted(found, key=len, reverse=True)


def _set_paragraph_text_collapsed(paragraph: Paragraph, new_text: str) -> None:
    runs = paragraph.runs
    if runs:
        runs[0].text = new_text
        for r in runs[1:]:
            r.text = ""
    else:
        paragraph.add_run(new_text)


def apply_replacements(doc: Document, mapping: List[Tuple[str, str]]) -> None:
    """Apply (old, new) pairs to every paragraph; longest keys should appear first in the list."""
    for paragraph in iter_all_paragraphs(doc):
        text = paragraph.text
        orig = text
        for old, new in mapping:
            if old in text:
                text = text.replace(old, new)
        if text != orig:
            _set_paragraph_text_collapsed(paragraph, text)


def _set_cell_text(cell: Any, text: str) -> None:
    """Write plain text into the first paragraph of a table cell."""
    if cell.paragraphs:
        _set_paragraph_text_collapsed(cell.paragraphs[0], text)
    else:
        cell.add_paragraph(text)


def _cell_is_effectively_empty(cell: Any) -> bool:
    return not any(p.text.strip() for p in cell.paragraphs)


def _row_data_cells_empty(row: Any) -> bool:
    return all(_cell_is_effectively_empty(c) for c in row.cells)


def _table_header_cells(row: Any) -> List[str]:
    return [c.text.strip() for c in row.cells]


def _rank_participants_across_calls(
    overview: ProspectOverview,
    affiliation_filter: str,
) -> List[Tuple[str, str, int]]:
    """Rank participants by frequency + recency across ALL Gong calls.

    affiliation_filter: "customer" to get external/prospect participants,
                        "arize" to get internal/company participants.

    Returns list of (name, email, call_count) sorted by call_count desc, then
    by most-recent appearance.
    """
    gong = overview.gong_summary
    if not gong:
        return []

    is_customer = affiliation_filter == "customer"
    counts: Dict[str, Dict[str, Any]] = {}

    for idx, call in enumerate(gong.recent_calls):
        for p in call.participants:
            name = (p.name or "").strip()
            if not name:
                continue
            aff = (p.affiliation or "").strip().lower()

            if is_customer:
                if aff in ("company", "internal"):
                    continue
                if aff not in ("external", "customer", "prospect", "non_company", ""):
                    continue
            else:
                if aff != "company":
                    continue

            key = name.lower()
            if key not in counts:
                counts[key] = {
                    "name": name,
                    "email": (p.email or "").strip(),
                    "call_count": 0,
                    "first_seen_idx": idx,
                }
            counts[key]["call_count"] += 1
            if (p.email or "").strip() and not counts[key]["email"]:
                counts[key]["email"] = (p.email or "").strip()

    ranked = sorted(
        counts.values(),
        key=lambda x: (-x["call_count"], x["first_seen_idx"]),
    )
    return [(r["name"], r["email"], r["call_count"]) for r in ranked]


def _ranked_customer_participants(overview: ProspectOverview) -> List[Tuple[str, str, int]]:
    return _rank_participants_across_calls(overview, "customer")


def _ranked_arize_participants(overview: ProspectOverview) -> List[Tuple[str, str, int]]:
    return _rank_participants_across_calls(overview, "arize")


def _arize_roster_lines(overview: ProspectOverview) -> List[Tuple[str, str, str, str]]:
    """Build ordered list of (name, title, role, email) for Arize team members.

    Sources in priority order:
    1. Salesforce team assignments (SA, CSE, AI SE, owner)
    2. Arize participants ranked across ALL Gong calls
    Deduplicates by lowered name.
    """
    seen: set[str] = set()
    lines: List[Tuple[str, str, str, str]] = []

    if overview.salesforce:
        sf = overview.salesforce
        assignments = [
            ((getattr(sf, "assigned_sa", None) or "").strip(), "Solution Architect"),
            ((getattr(sf, "assigned_cse", None) or "").strip(), "Customer Success Engineer"),
            ((getattr(sf, "assigned_ai_se", None) or "").strip(), "AI Sales Engineer"),
            ((getattr(sf, "assigned_csm", None) or "").strip(), "Customer Success Manager"),
            ((getattr(sf, "owner_name", None) or "").strip(), "Account Owner"),
        ]
        for name, title in assignments:
            if name and name.lower() not in seen:
                seen.add(name.lower())
                lines.append((name, f"{title} (Arize)", "Arize", ""))

    for name, email, _count in _ranked_arize_participants(overview):
        if name.lower() not in seen:
            seen.add(name.lower())
            lines.append((name, "Arize (from Gong calls)", "Arize", email))

    return lines


def _fill_empty_data_rows(
    table: Any,
    header_check: List[str],
    rows_data: List[List[str]],
) -> None:
    """Fill empty data rows in a Word table, skipping the header row."""
    if _table_header_cells(table.rows[0]) != header_check:
        return
    data_row_idx = 1
    for values in rows_data:
        if data_row_idx >= len(table.rows):
            break
        if _row_data_cells_empty(table.rows[data_row_idx]):
            for ci, val in enumerate(values):
                if ci < len(table.rows[data_row_idx].cells):
                    _set_cell_text(table.rows[data_row_idx].cells[ci], val[:240])
            data_row_idx += 1
        else:
            data_row_idx += 1


def fill_structured_roster_cells(
    doc: Document, overview: ProspectOverview, document_template: str
) -> None:
    """Fill roster and kickoff tables using ranked participants from ALL Gong calls
    plus Salesforce team assignments.

    Customer roster: people ranked by how many calls they appeared in (most active first).
    Arize roster: Salesforce assignments first, then Gong-ranked Arize participants.
    """
    if document_template not in ("poc_saas", "poc_vpc") or len(doc.tables) < 4:
        return

    # --- Customer roster (table 1): Name | Role / Team | % Time Committed | Responsibilities ---
    customer_ranked = _ranked_customer_participants(overview)
    if customer_ranked:
        customer_rows = []
        for name, email, call_count in customer_ranked[:6]:
            role_hint = f"Customer ({call_count} call{'s' if call_count != 1 else ''})"
            customer_rows.append([name, role_hint, "TBD", email or "—"])
        _fill_empty_data_rows(
            doc.tables[1],
            ["Name", "Role / Team", "% Time Committed", "Responsibilities"],
            customer_rows,
        )

    # --- Arize roster (table 2): Name | Title | Email | Role ---
    arize_lines = _arize_roster_lines(overview)
    if arize_lines:
        arize_rows = [[nm, title, email, role] for nm, title, role, email in arize_lines[:6]]
        _fill_empty_data_rows(
            doc.tables[2],
            ["Name", "Title", "Email", "Role"],
            arize_rows,
        )

    # --- Kickoff table (table 3): Primary Escalation Contact ---
    t3 = doc.tables[3]
    if _table_header_cells(t3.rows[0]) == ["Item", "Details"]:
        escalation_name = arize_lines[0][0] if arize_lines else None
        if escalation_name:
            for ri in range(1, len(t3.rows)):
                label = t3.rows[ri].cells[0].text.strip()
                if "Primary Escalation Contact (Arize)" in label and _cell_is_effectively_empty(
                    t3.rows[ri].cells[1]
                ):
                    _set_cell_text(t3.rows[ri].cells[1], escalation_name[:240])
                    break


def _build_account_context_summary(overview: ProspectOverview) -> str:
    """Distill the account's specific context from ALL Gong calls + opportunity
    data into a concise summary the LLM can use to tailor success criteria
    and other template fields."""
    sections: list[str] = []

    # Gong: BigQuery only stores what Gong synced (Spotlight columns may be NULL).
    # We surface title/date for every call, plus Spotlight and transcript snippets when present,
    # so PoC/PoT is closer to Gong UI even when CALL_SPOTLIGHT_* is empty.
    gong = overview.gong_summary
    if gong and gong.recent_calls:
        timeline_lines: list[str] = []
        transcript_excerpts: list[str] = []
        themes = []
        next_steps_mentioned = []
        use_cases_discussed = []
        pain_points = []
        for call in gong.recent_calls:
            title = (call.call_title or "Untitled").strip()
            date = (call.call_date or "")[:10] or "?"
            has_spotlight = bool(
                call.spotlight_brief
                or call.spotlight_key_points
                or call.spotlight_next_steps
                or call.spotlight_outcome
            )
            status = "Spotlight + details in warehouse" if has_spotlight else "title/metadata only (no CALL_SPOTLIGHT_* in export)"
            tl = f"{date} — **{title}** — {status}"
            if call.spotlight_brief:
                snip = call.spotlight_brief[:320]
                if len(call.spotlight_brief) > 320:
                    snip += "…"
                tl += f" — Brief: {snip}"
            timeline_lines.append(tl)

            if call.transcript_snippet and len(call.transcript_snippet.strip()) > 30:
                excerpt = (call.transcript_snippet or "")[:700]
                if len(call.transcript_snippet or "") > 700:
                    excerpt += "…"
                transcript_excerpts.append(f"- **{date} {title}:** {excerpt}")

            if call.spotlight_brief:
                themes.append(call.spotlight_brief)
            if call.spotlight_key_points:
                for kp in call.spotlight_key_points:
                    use_cases_discussed.append(str(kp))
            if call.spotlight_next_steps:
                next_steps_mentioned.append(call.spotlight_next_steps)
            if call.spotlight_outcome:
                pain_points.append(call.spotlight_outcome)

        sections.append(
            "**Gong calls (from BigQuery `gong.CALLS`; Gong UI may show more if Spotlight did not sync):**\n"
            + "\n".join(f"- {line}" for line in timeline_lines[:15])
        )
        if transcript_excerpts:
            sections.append(
                "**Transcript excerpts (warehouse `CALL_TRANSCRIPTS`, truncated; not all calls loaded):**\n"
                + "\n".join(transcript_excerpts[:5])
            )
        if themes:
            sections.append(
                f"**Gong Spotlight briefs only ({len(themes)} with text):** "
                + " | ".join(themes[:8])
            )
        if use_cases_discussed:
            sections.append(
                "**Key discussion points (Spotlight):** " + " | ".join(use_cases_discussed[:15])
            )
        if next_steps_mentioned:
            sections.append(
                "**Next steps (Spotlight):** " + " | ".join(next_steps_mentioned[:5])
            )
        if gong.key_themes:
            sections.append("**Extracted themes:** " + ", ".join(gong.key_themes[:10]))

    # Opportunity context
    opp = overview.latest_opportunity
    if opp:
        bits = []
        if opp.name:
            bits.append(f"Opportunity: {opp.name}")
        if opp.stage_name:
            bits.append(f"Stage: {opp.stage_name}")
        if opp.description:
            bits.append(f"Description: {opp.description[:500]}")
        if opp.next_step:
            bits.append(f"Next step: {opp.next_step}")
        if bits:
            sections.append("**Latest opportunity:** " + " · ".join(bits))

    # Salesforce notes
    sf = overview.salesforce
    if sf:
        if sf.next_steps:
            sections.append(f"**SFDC next steps:** {sf.next_steps[:400]}")
        if sf.customer_notes:
            sections.append(f"**Customer notes:** {sf.customer_notes[:400]}")
        if sf.is_using_llms:
            sections.append(f"**Using Arize for LLMs:** {sf.is_using_llms}")
        if sf.deployment_types:
            sections.append(f"**Deployment type:** {sf.deployment_types}")

    # Product usage hints
    if overview.product_usage and overview.product_usage.adoption_status:
        sections.append(f"**Adoption status:** {overview.product_usage.adoption_status}")

    # Key participants for context
    customer_folks = _ranked_customer_participants(overview)
    if customer_folks:
        top_names = [f"{n} ({c} calls)" for n, _e, c in customer_folks[:5]]
        sections.append("**Top customer contacts (by call frequency):** " + ", ".join(top_names))

    return "\n".join(sections) if sections else "No additional context available."


def _template_instructions(document_template: str) -> str:
    if document_template == "poc_saas":
        return (
            "You are filling an **Arize AX Proof of Concept (PoC) — SaaS** Word template. "
            "Choose concise values that fit table cells. Prefer SaaS deployment language where relevant."
        )
    if document_template == "poc_vpc":
        return (
            "You are filling an **Arize AX Proof of Concept (PoC) — VPC** Word template. "
            "Emphasize customer VPC / private environment, infrastructure alignment, and longer trial windows when appropriate."
        )
    return (
        "You are filling an **Arize AX Proof of Technology (PoT)** Word template. "
        "Emphasize short workshop-style validation, use-case mapping, and discovery outcomes."
    )


def _company_display_name(overview: ProspectOverview) -> str:
    if overview.salesforce and (overview.salesforce.name or "").strip():
        return overview.salesforce.name.strip()
    if (overview.lookup_value or "").strip():
        return overview.lookup_value.strip()
    return "Customer"


def generate_integrated_fills(
    *,
    keys: List[str],
    overview: ProspectOverview,
    document_template: str,
    manual_notes: str | None,
    llm_model: str | None = None,
) -> Dict[str, str]:
    model = llm_model or os.environ.get("POC_DOC_LLM_MODEL", "claude-haiku-4-5")
    payload = _overview_to_prompt_json(overview)
    keys_json = json.dumps(keys, indent=2, ensure_ascii=False)
    notes_block = ""
    if (manual_notes or "").strip():
        notes_block = (
            "\n## Solution Architect notes (use when supported by data; do not invent contradictions)\n"
            f"{manual_notes.strip()}\n"
        )

    account_context = _build_account_context_summary(overview)
    success_digest = _success_criteria_evidence_digest(overview)
    eg_keys = [k for k in keys if k.startswith("e.g.,")]
    eg_keys_json = json.dumps(eg_keys, indent=2, ensure_ascii=False)

    prompt = f"""{_template_instructions(document_template)}

You will receive JSON: a **ProspectOverview** from our warehouse (Salesforce, Gong, Pendo, FullStory, etc.).

## Account context (synthesized from ALL Gong calls + CRM data)
{account_context}

{success_digest}

## Your task
Return **only** a JSON object with a single property `"replacements"` whose keys are **exactly** the strings listed
in KEYS below (same spelling, punctuation, and spacing). Each value replaces that substring wherever it appears in the Word file.

### Rules
- Use only facts grounded in the ProspectOverview JSON (and SA notes when consistent). If unknown, use a short honest placeholder like "TBD — confirm with account team" rather than fabricating metrics.
- **Every key must have a non-empty string value** (never `""`, never null). Do not echo the placeholder key back as the value.
- For keys that look like multiple-choice prompts (e.g. "[OpenAI / Anthropic / ...]"), pick **one** realistic option or a short comma-separated subset that matches the account.
- For bracket keys, keep values **brief** (typically under ~200 characters) so they fit Word table cells.
- For the long `e.g., ...` lines and the `Start: _____ / End: _____ - [X] Weeks  ` line, return a **complete finished line** of similar length (replace the entire template line).
- For `___________________________________________`, replace with **one** short headline-style sentence summarizing the opportunity (no underscores).

### Success criteria — opportunity-first (mandatory)
These rules apply especially to KEYS whose text starts with `e.g.,` (success-metric example lines). Use **Account context**, **Evidence scan**, and the latest opportunity together.

1. **No invented ML/LLM scope** — If **"LLMs / GenAI / agents / prompts / RAG"** does *not* appear under *Buckets with at least weak textual support* above, then:
   - Do **not** populate prompt-/LLM-specific outcomes (prompt variant counts, LLM eval cadence, chat token SLOs, etc.). For any `e.g.,` template line that is clearly about prompts or LLM iteration, replace the **entire line** with something like: `N/A — not in scope for this opportunity: no LLM/GenAI discussion in Gong, CRM, or product signals.`
   - Likewise, if **"Classical ML / tabular / batch models"** is unsupported, do not invent tabular-model metrics; use N/A or reframed ops criteria that *are* supported.

2. **Match the deal, not the template boilerplate** — Each `e.g.,` value should read like a **credible success check for this specific opportunity** (stage, opp description, call themes, Pendo surfaces). Prefer **one** concrete, testable sentence tied to evidence. If the template example topic (drift, prompts, MTTR, etc.) does not fit the evidence scan, **replace the line** with N/A *or* **repurpose** the slot to a different measurable outcome that *does* match the evidence (do not keep the original example wording).

3. **Add value where data supports it** — If the evidence scan or narratives show strong themes that no `e.g.,` line captures, you may **repurpose** a loosely related `e.g.,` slot to propose **one additional** measurable trial outcome (still one sentence, no fabricated numbers). If nothing fits, use N/A rather than generic filler.

4. **Quantification** — Use numeric targets only when similar numbers appear in the JSON or SA notes; otherwise use "TBD — baseline in kickoff" or qualitative acceptance language.

## Success-criteria template keys in this file (each needs a full replacement line)
{eg_keys_json}

## KEYS (required JSON keys under "replacements")
{keys_json}

## ProspectOverview JSON
{payload}
{notes_block}
"""

    response = llm_completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=int(os.environ.get("POC_DOC_MAX_TOKENS", "12288")),
        temperature=0.25,
    )
    raw = (response.choices[0].message.content or "").strip()
    if os.environ.get("POC_DOC_DEBUG", "").strip().lower() in ("1", "true", "yes"):
        print(f"[poc_document_generator] LLM raw (first 800 chars): {raw[:800]!r}")

    try:
        data = _extract_json_object(raw)
        parsed = IntegratedFillResult.model_validate(data)
    except (ValueError, json.JSONDecodeError, ValidationError) as e:
        raise AppendixGenerationError(str(e)) from e

    llm_map = parsed.replacements or {}
    merged: Dict[str, str] = {}
    for k in keys:
        merged[k] = _coerce_value_for_key(
            k,
            llm_map.get(k),
            overview,
            document_template,
            manual_notes,
        )

    return merged


def _safe_filename_part(name: str) -> str:
    s = "".join(c for c in name if c.isalnum() or c in (" ", "-", "_")).strip()
    return s.replace(" ", "_")[:80] or "Account"


# Human-readable segment for download filenames (filesystem-safe, no slashes).
DOCUMENT_EXPORT_TYPE_LABELS: dict[str, str] = {
    "poc_saas": "PoC-SaaS",
    "poc_vpc": "PoC-VPC",
    "pot": "PoT",
}


def _export_type_filename_part(document_template: str) -> str:
    label = DOCUMENT_EXPORT_TYPE_LABELS.get(document_template, document_template)
    return _safe_filename_part(label)


def build_poc_document(
    overview: ProspectOverview,
    document_template: str,
    manual_notes: str | None = None,
    llm_model: str | None = None,
) -> tuple[bytes, str]:
    path = TEMPLATES.get(document_template)
    if not path or not path.is_file():
        raise FileNotFoundError(f"Missing template for {document_template}: {path}")

    doc = Document(str(path))
    keys = discover_replacement_keys(doc, document_template)
    if not keys:
        raise ValueError("No replaceable placeholders found in template")

    fill_map = generate_integrated_fills(
        keys=keys,
        overview=overview,
        document_template=document_template,
        manual_notes=manual_notes,
        llm_model=llm_model,
    )
    ordered = sorted(fill_map.items(), key=lambda kv: len(kv[0]), reverse=True)
    apply_replacements(doc, ordered)
    fill_structured_roster_cells(doc, overview, document_template)

    buf = io.BytesIO()
    doc.save(buf)
    date_part = datetime.now(timezone.utc).strftime("%Y%m%d")
    account_safe = _safe_filename_part(_company_display_name(overview))
    type_safe = _export_type_filename_part(document_template)
    # Account name + document type first so downloads sort clearly by customer.
    fname = f"Arize_AX_{account_safe}_{type_safe}_{date_part}.docx"
    return buf.getvalue(), fname
