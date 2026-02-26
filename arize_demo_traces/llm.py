"""
Return the appropriate LangChain chat model (Anthropic, OpenAI, or AWS Bedrock).
Uses Anthropic directly when model is Claude; Bedrock when model is bedrock/<id>;
otherwise OpenAI (optionally via LiteLLM).
"""

import os
from typing import Any

# Prefix for Bedrock model IDs (e.g. bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0)
BEDROCK_MODEL_PREFIX = "bedrock/"


def get_chat_llm(model: str, temperature: float = 0, **kwargs: Any):
    """
    Return a LangChain chat model.
    - bedrock/<model_id> -> ChatBedrock (requires langchain-aws, AWS credentials).
    - claude-* -> ChatAnthropic (direct API).
    - Otherwise -> ChatOpenAI or LiteLLM proxy.
    """
    if model.startswith(BEDROCK_MODEL_PREFIX):
        model_id = model[len(BEDROCK_MODEL_PREFIX) :].strip()
        if not model_id:
            raise ValueError("Bedrock model ID required, e.g. bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0")
        try:
            from langchain_aws import ChatBedrock
        except ImportError as e:
            raise ImportError(
                "langchain-aws is required for Bedrock. Install with: pip install langchain-aws"
            ) from e
        model_kwargs = {"temperature": temperature, **kwargs.pop("model_kwargs", {})}
        bedrock_kwargs = {k: v for k, v in kwargs.items() if k != "temperature"}
        region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
        if region:
            bedrock_kwargs["region_name"] = region
        return ChatBedrock(
            model_id=model_id,
            model_kwargs=model_kwargs,
            **bedrock_kwargs,
        )

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
