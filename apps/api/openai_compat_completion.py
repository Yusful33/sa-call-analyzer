"""OpenAI-chat-style completions without LiteLLM (Anthropic + OpenAI SDKs only).

Used on Vercel and anywhere the full LiteLLM stack is not installed. Returns a
minimal object with ``choices[0].message.content`` matching ``litellm.completion``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Optional, TypedDict


class _ChatMessage(TypedDict):
    role: str
    content: str


@dataclass
class _Msg:
    content: Optional[str]


@dataclass
class _Choice:
    message: _Msg


@dataclass
class _Completion:
    choices: List[_Choice]


_MODEL_ALIASES = {
    "claude-3-5-haiku-20241022": "claude-haiku-4-5",
    "claude-3-5-sonnet-20241022": "claude-sonnet-4-20250514",
}


def _resolve_model_id(model: str) -> str:
    m = (model or "").strip()
    if not m:
        return "claude-haiku-4-5"
    return _MODEL_ALIASES.get(m, m)


def _split_system_and_rest(
    messages: List[_ChatMessage],
) -> tuple[Optional[str], List[dict[str, Any]]]:
    system_chunks: List[str] = []
    rest: List[dict[str, Any]] = []
    for msg in messages:
        role = (msg.get("role") or "user").strip()
        content = msg.get("content") or ""
        if role == "system":
            system_chunks.append(content)
        else:
            rest.append({"role": role, "content": content})
    system = "\n\n".join(system_chunks).strip() if system_chunks else None
    return system, rest


def _uses_openai(model: str) -> bool:
    m = model.lower()
    return m.startswith("gpt") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4")


def completion(
    *,
    model: str,
    messages: List[_ChatMessage],
    max_tokens: int = 1024,
    temperature: float = 0.0,
    **_: Any,
) -> _Completion:
    """Return a LiteLLM-shaped completion for a single-turn chat-style prompt."""
    resolved = _resolve_model_id(model)

    if _uses_openai(resolved):
        from openai import OpenAI

        base = (
            os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
            or ""
        ).strip() or None
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=base)
        out = client.chat.completions.create(
            model=resolved,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = (out.choices[0].message.content or "").strip() if out.choices else ""
        return _Completion(choices=[_Choice(message=_Msg(content=content))])

    from anthropic import Anthropic

    system, anthropic_messages = _split_system_and_rest(messages)
    if not anthropic_messages:
        anthropic_messages = [{"role": "user", "content": "(empty)"}]

    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    kwargs: dict[str, Any] = {
        "model": resolved,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": anthropic_messages,
    }
    if system:
        kwargs["system"] = system
    out = client.messages.create(**kwargs)
    blocks = getattr(out.content[0], "text", None) if out.content else None
    content = (blocks or "").strip() if isinstance(blocks, str) else ""
    if not content and out.content:
        first = out.content[0]
        content = getattr(first, "text", "") or ""
    return _Completion(choices=[_Choice(message=_Msg(content=content.strip()))])
