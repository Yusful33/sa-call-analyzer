#!/usr/bin/env python3
"""
Example: Run the Guardrails use case with AWS Bedrock as the LLM for safety checks.

This script runs the same guardrail pipeline (content safety, jailbreak, toxicity, PII,
plus a rule-based length check) but uses a Claude model on AWS Bedrock instead of
OpenAI/Anthropic direct. Traces can be sent to Arize if ARIZE_SPACE_ID and ARIZE_API_KEY
are set.

Requirements:
  - pip install langchain-aws  (or install project deps: uv sync / pip install -e .)
  - AWS credentials (AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY, or AWS_PROFILE, or
    default profile)
  - Bedrock model access enabled in your AWS account for the chosen model

Usage:
  # Default Bedrock model (Claude 3.5 Sonnet in us-east-1)
  python examples/guardrails_bedrock.py

  # Custom model and query
  BEDROCK_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0 python examples/guardrails_bedrock.py --query "What is our refund policy?"

  # Send traces to Arize
  ARIZE_SPACE_ID=... ARIZE_API_KEY=... python examples/guardrails_bedrock.py
"""

import argparse
import os
import sys

# Ensure project root is on path when running from repo
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Bedrock model ID (full ID as in AWS Bedrock console)
DEFAULT_BEDROCK_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"


def main():
    parser = argparse.ArgumentParser(description="Run guardrails pipeline with AWS Bedrock")
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Input text to run through guardrails (default: sample from use case)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("BEDROCK_MODEL_ID", DEFAULT_BEDROCK_MODEL_ID),
        help="Bedrock model ID (e.g. anthropic.claude-3-5-sonnet-20240620-v1:0). Override with BEDROCK_MODEL_ID.",
    )
    parser.add_argument(
        "--no-arize",
        action="store_true",
        help="Do not send traces to Arize even if ARIZE_SPACE_ID/ARIZE_API_KEY are set",
    )
    args = parser.parse_args()

    # Model must be passed to get_chat_llm with the "bedrock/" prefix
    from arize_demo_traces.llm import BEDROCK_MODEL_PREFIX

    model_with_prefix = f"{BEDROCK_MODEL_PREFIX}{args.model}"

    # Optional: set up Arize tracer so traces appear in the app
    tracer_provider = None
    if not args.no_arize and os.getenv("ARIZE_SPACE_ID") and os.getenv("ARIZE_API_KEY"):
        try:
            from arize.otel import register as arize_register

            tracer_provider = arize_register(
                space_id=os.getenv("ARIZE_SPACE_ID"),
                api_key=os.getenv("ARIZE_API_KEY"),
                project_name=os.getenv("ARIZE_PROJECT_NAME", "guardrails-bedrock-example"),
            )
            print("Sending traces to Arize (project: guardrails-bedrock-example)")
        except Exception as e:
            print(f"Warning: Could not register Arize tracer: {e}")

    # Run the guardrails pipeline (same as Custom Demo Builder "Guardrails" use case)
    from arize_demo_traces.runners.langgraph.guardrails import run_guardrails

    print(f"Running guardrails with Bedrock model: {args.model}")
    print(f"Query: {args.query or '(sampled from use case)'}\n")

    result = run_guardrails(
        query=args.query,
        model=model_with_prefix,
        tracer_provider=tracer_provider,
    )

    print("--- Result ---")
    print(f"Query:  {result['query']}")
    print(f"Result: {result['answer']}")
    print("\nDone.")

    if tracer_provider:
        try:
            tracer_provider.force_flush()
        except Exception:
            pass


if __name__ == "__main__":
    main()
