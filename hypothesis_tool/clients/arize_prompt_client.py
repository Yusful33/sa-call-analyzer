"""Client for fetching and rendering Arize Prompt Hub prompts by ID."""

import logging
import re
from typing import Any

import httpx

from ..config import get_settings

logger = logging.getLogger(__name__)

# Default prompt version ID for "Sales Research Assistant" (plan research step)
DEFAULT_PLAN_RESEARCH_PROMPT_ID = "UHJvbXB0OjMwNTI3Om9BbWo="


def _normalize_prompt_data(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize Arize v2 or Phoenix v1 response to our expected shape (template, template_format, etc.)."""
    # v1/Phoenix: { "template", "template_type", "template_format", "model_provider", ... }
    if data.get("template") is not None:
        return data
    # v2: may have prompt.versions[0] or prompt.latest_version or messages
    if "versions" in data and data["versions"]:
        v = data["versions"][0]
        return _normalize_prompt_data(v) if isinstance(v, dict) else data
    if "latest_version" in data:
        return _normalize_prompt_data(data["latest_version"])
    # v2 shape: messages array for chat
    if "messages" in data:
        return {
            "template": {"type": "chat", "messages": data["messages"]},
            "template_type": "CHAT",
            "template_format": (data.get("template_format") or "NONE").upper(),
        }
    if "template" in data:
        return data
    raise ValueError(f"Unrecognized prompt response shape: no template, versions, or messages in {list(data.keys())}")


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


async def get_prompt_version(prompt_version_id: str) -> dict[str, Any]:
    """
    Fetch a prompt (or prompt version) by ID from the Arize API.
    Tries v2 /v2/prompts/{id} first, then v1 /v1/prompt_versions/{id}.
    Returns normalized data with template, template_format, etc.
    """
    settings = get_settings()
    if not settings.arize_api_key:
        raise ValueError("ARIZE_API_KEY is required to fetch Arize prompts")
    base_url = (settings.arize_prompt_api_base_url or "https://api.arize.com").rstrip("/")
    headers = {
        "Authorization": f"Bearer {settings.arize_api_key}",
        "Content-Type": "application/json",
    }
    if settings.arize_space_id:
        headers["arize-space-id"] = settings.arize_space_id

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Try Arize v2 Prompt Hub first (GET /v2/prompts/{id})
        url_v2 = f"{base_url}/v2/prompts/{prompt_version_id}"
        params = {}
        if settings.arize_space_id:
            params["space_id"] = settings.arize_space_id
        try:
            resp = await client.get(url_v2, headers=headers, params=params or None)
            if resp.is_success:
                data = resp.json()
                # v2 may wrap in "data" or return prompt directly
                payload = data.get("data", data)
                if isinstance(payload, dict):
                    return _normalize_prompt_data(payload)
            elif resp.status_code != 404:
                resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                pass
            else:
                raise

        # Fallback: Phoenix-style v1 prompt_versions
        url_v1 = f"{base_url}/v1/prompt_versions/{prompt_version_id}"
        resp = await client.get(url_v1, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    payload = data.get("data", data)
    if isinstance(payload, dict):
        return _normalize_prompt_data(payload)
    return _normalize_prompt_data(data)


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
                    parts.append(_render_mustache(content, variables) if template_format == "MUSTACHE" else content)
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


async def get_plan_research_prompt(company_name: str, prompt_version_id: str | None = None) -> str:
    """
    Fetch the Sales Research Assistant (plan research) prompt from Arize and render it
    with the given company name. Returns the prompt text to send to the LLM.
    """
    settings = get_settings()
    pid = prompt_version_id or getattr(settings, "arize_plan_research_prompt_id", None) or DEFAULT_PLAN_RESEARCH_PROMPT_ID
    variables = {"company_name": company_name, "Company Name": company_name}
    prompt_data = await get_prompt_version(pid)
    return render_prompt_template(prompt_data, variables)
