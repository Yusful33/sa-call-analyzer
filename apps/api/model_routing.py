"""Shared model ID normalization for LLM requests."""

from __future__ import annotations

from typing import Optional

MODEL_ID_ALIASES = {
    "claude-3-5-haiku-20241022": "claude-haiku-4-5",
    "claude-3-5-sonnet-20241022": "claude-sonnet-4-20250514",
}


def resolve_model_id(model: Optional[str]) -> Optional[str]:
    """Return current model ID; map retired IDs so requests do not 404."""
    if not model:
        return None
    return MODEL_ID_ALIASES.get(model.strip(), model.strip())
