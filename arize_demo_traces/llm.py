"""
Return the appropriate LangChain chat model (Anthropic or OpenAI).
Uses Anthropic directly when model is Claude; otherwise OpenAI (optionally via LiteLLM).
"""

import os
from typing import Any


def get_chat_llm(model: str, temperature: float = 0, **kwargs: Any):
    """
    Return a LangChain chat model. Uses ChatAnthropic for claude-* models (direct API),
    otherwise ChatOpenAI (with LiteLLM base_url if USE_LITELLM=true).
    """
    if model.startswith("claude-"):
        from langchain_anthropic import ChatAnthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for Claude models")
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            anthropic_api_key=api_key,
            **kwargs,
        )

    # Prefer direct OpenAI API when key is available (avoids LiteLLM proxy issues)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature, **kwargs)

    # Fall back to LiteLLM proxy
    use_litellm = os.getenv("USE_LITELLM", "false").lower() == "true"
    litellm_base_url = os.getenv("LITELLM_BASE_URL", "http://litellm:4000")
    llm_kwargs = {}
    if use_litellm:
        llm_kwargs["base_url"] = litellm_base_url
        llm_kwargs["api_key"] = os.getenv("LITELLM_API_KEY", "dummy")
        model = f"openai/{model}"

    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, temperature=temperature, **llm_kwargs, **kwargs)
