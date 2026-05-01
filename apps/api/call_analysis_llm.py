"""LangChain chat models for call analysis (replaces CrewAI LLM wrapper)."""

from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel


def build_chat_model(model_name: str, *, temperature: float = 0.7) -> BaseChatModel:
    """Return a LangChain chat model for the given model id (Anthropic or OpenAI, or LiteLLM proxy)."""
    use_litellm = os.getenv("USE_LITELLM", "false").lower() == "true"
    if use_litellm:
        from langchain_openai import ChatOpenAI

        litellm_base_url = os.getenv("LITELLM_BASE_URL", "http://litellm:4000").rstrip("/")
        print(f"🔧 LiteLLM proxy: model={model_name}, base_url={litellm_base_url}")
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=8192,
            api_key=os.getenv("LITELLM_API_KEY", "dummy"),
            base_url=f"{litellm_base_url}/v1",
        )

    m = model_name.lower()
    if "claude" in m:
        from langchain_anthropic import ChatAnthropic

        print(f"🔧 ChatAnthropic: model={model_name}")
        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
            max_tokens=8192,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )

    from langchain_openai import ChatOpenAI

    print(f"🔧 ChatOpenAI: model={model_name}")
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=8192,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def chat_invoke_text(chat: BaseChatModel, prompt: str) -> str:
    """Invoke chat model with a single user message; return string content."""
    from langchain_core.messages import HumanMessage

    msg = chat.invoke([HumanMessage(content=prompt)])
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)
