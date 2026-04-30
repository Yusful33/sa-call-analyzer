"""Client for fetching and rendering Arize Prompt Hub prompts via GraphQL (app.arize.com/graphql)."""

import logging
import re
from typing import Any

import httpx

from ..config import get_settings

logger = logging.getLogger(__name__)

# Prompt (container) ID for "Sales Research Assistant". GraphQL node(id) returns prompt with messages.
PROMPT_ID_PLAN_RESEARCH = "UHJvbXB0OjMwNTI3Om9BbWo="
DEFAULT_PLAN_RESEARCH_PROMPT_ID = PROMPT_ID_PLAN_RESEARCH

# Fallback when GraphQL fails (e.g. no key, network error)
DEFAULT_PLAN_RESEARCH_PROMPT_TEMPLATE = (
    "You are a sales research assistant. Plan what research to conduct for: {{company_name}}. "
    "Output a brief research plan (2-3 sentences) covering company overview, key products, and relevant news."
)

GRAPHQL_ENDPOINT = "https://app.arize.com/graphql"


async def _fetch_prompt_via_graphql(
    api_key: str,
    prompt_id: str,
    *,
    space_id: str | None = None,
) -> dict[str, Any] | None:
    """Fetch a prompt by ID via Arize GraphQL API (app.arize.com/graphql). Use prompt (container) ID; returns normalized prompt data or None."""
    # Prompt type has messages (JSON array) and inputVariableFormat; PromptVersion has same but node(id) with version ID may be null
    query = """
    query GetPrompt($id: ID!) {
      node(id: $id) {
        ... on Prompt {
          id
          name
          inputVariableFormat
          messages
        }
        ... on PromptVersion {
          id
          inputVariableFormat
          messages
        }
      }
    }
    """
    payload: dict[str, Any] = {"query": query, "variables": {"id": prompt_id}}
    headers = {"Content-Type": "application/json", "x-api-key": api_key}
    if space_id:
        headers["arize-space-id"] = space_id
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(GRAPHQL_ENDPOINT, json=payload, headers=headers)
        if not resp.is_success:
            logger.debug("GraphQL prompt fetch status %s: %s", resp.status_code, resp.text[:200])
            return None
        body = resp.json()
        if body.get("errors"):
            logger.debug("GraphQL errors: %s", body["errors"])
            return None
        node = (body.get("data") or {}).get("node")
        if not node or not isinstance(node, dict):
            return None
        messages = node.get("messages")
        if not messages:
            return None
        template_format = (node.get("inputVariableFormat") or "NONE").upper()
        return {
            "template": {"type": "chat", "messages": messages},
            "template_type": "CHAT",
            "template_format": template_format,
        }


def _render_mustache(template: str, variables: dict[str, Any]) -> str:
    """Replace {{variable}} and {{{variable}}} with values. Keys normalized to match common casing."""
    result = template
    # Normalize keys: support both "company_name" and "Company Name"
    expanded: dict[str, Any] = {}
    for k, v in variables.items():
        expanded[k] = v
        if k == "company_name":
            expanded["Company Name"] = v
        elif k == "Company Name":
            expanded["company_name"] = v
    for key, value in expanded.items():
        if value is None:
            value = ""
        # {{key}} and {{{key}}} (raw)
        for pattern in (r"\{\{\{\s*" + re.escape(key) + r"\s*\}\}\}", r"\{\{\s*" + re.escape(key) + r"\s*\}\}"):
            result = re.sub(pattern, str(value), result, flags=re.IGNORECASE)
    return result


async def get_prompt_version(prompt_id: str) -> dict[str, Any]:
    """
    Fetch a prompt by ID via Arize GraphQL API (app.arize.com/graphql).
    Use prompt (container) ID. Returns dict with template, template_type, template_format.
    """
    settings = get_settings()
    if not settings.arize_api_key:
        raise ValueError("ARIZE_API_KEY is required to fetch Arize prompts")
    for try_id in (prompt_id, PROMPT_ID_PLAN_RESEARCH):
        data = await _fetch_prompt_via_graphql(
            settings.arize_api_key,
            try_id,
            space_id=settings.arize_space_id or None,
        )
        if data is not None:
            return data
    raise ValueError(
        "Could not fetch prompt from Arize GraphQL. Check ARIZE_API_KEY and ARIZE_SPACE_ID; use prompt (container) ID."
    )


def render_prompt_template(prompt_data: dict[str, Any], variables: dict[str, Any]) -> str:
    """
    Render the prompt template with the given variables.
    Supports string template with MUSTACHE or plain text.
    Returns the final user message string to send to the LLM.
    """
    template = prompt_data.get("template")
    if not template:
        raise ValueError("Prompt version has no template")
    template_type = prompt_data.get("template_type", "STR")
    template_format = (prompt_data.get("template_format") or "NONE").upper()

    if isinstance(template, dict):
        type_ = template.get("type", "string")
        if type_ == "chat":
            # Chat: list of messages; take the first user message or concatenate
            messages = template.get("messages", [])
            parts = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        p.get("text", "") if isinstance(p, dict) else str(p) for p in content if isinstance(p, dict) and p.get("type") == "text"
                    )
                if role == "user" and content:
                    if template_format == "MUSTACHE":
                        content = _render_mustache(content, variables)
                    elif template_format == "F_STRING":
                        try:
                            content = content.format(**{k: v or "" for k, v in variables.items()})
                        except KeyError:
                            content = _render_mustache(content, variables)
                    parts.append(content)
            return "\n\n".join(parts) if parts else ""
        # string template
        template_str = template.get("template", "")
    else:
        template_str = str(template)

    if template_format == "MUSTACHE":
        return _render_mustache(template_str, variables)
    if template_format == "F_STRING":
        try:
            return template_str.format(**variables)
        except KeyError:
            return _render_mustache(template_str, variables)
    return template_str


async def get_plan_research_prompt(company_name: str, prompt_id: str | None = None) -> str:
    """
    Fetch the Sales Research Assistant (plan research) prompt from Arize via GraphQL and render it
    with the given company name. Falls back to built-in template if GraphQL fails.
    """
    settings = get_settings()
    variables = {"company_name": company_name, "Company Name": company_name}
    if not settings.arize_api_key:
        logger.debug("ARIZE_API_KEY not set, using built-in plan-research prompt")
        return _render_mustache(DEFAULT_PLAN_RESEARCH_PROMPT_TEMPLATE, variables)
    pid = prompt_id or getattr(settings, "arize_plan_research_prompt_id", None) or DEFAULT_PLAN_RESEARCH_PROMPT_ID
    try:
        prompt_data = await get_prompt_version(pid)
        return render_prompt_template(prompt_data, variables)
    except (ValueError, httpx.HTTPStatusError) as e:
        logger.warning(
            "Arize GraphQL prompt fetch failed (%s), using built-in Sales Research Assistant fallback",
            e,
        )
        return _render_mustache(DEFAULT_PLAN_RESEARCH_PROMPT_TEMPLATE, variables)
