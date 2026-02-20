"""
ADK-style multimodal / vision agent with manual OpenTelemetry spans.
Mimics Google Agent Development Kit trace patterns without requiring ADK installation.
Uses real LLM calls inside manually created OTel spans with OpenInference attributes.

Trace structure:
  vision_agent.run (AGENT)
    -> guardrails (GUARDRAIL group)
    -> generate_content [classify_image] (LLM)
    -> generate_content [analyze] (LLM)
    -> extract_data (TOOL)
    -> generate_content [summarize] (LLM)
"""

import random

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ...cost_guard import CostGuard
from ...llm import get_chat_llm
from ...trace_enrichment import (
    run_guardrail,
    run_local_guardrail,
    run_tool_call,
)
from ...use_cases.multimodal import (
    QUERIES,
    GUARDRAILS,
    LOCAL_GUARDRAILS,
    SYSTEM_PROMPT_CLASSIFY_IMAGE,
    SYSTEM_PROMPT_VISION,
    SYSTEM_PROMPT_EXTRACT,
    SYSTEM_PROMPT_SUMMARIZE_FINDINGS,
    get_random_query,
    analyze_image_content,
)


def run_multimodal(
    query: dict | None = None,
    model: str = "gpt-4o-mini",
    guard: CostGuard | None = None,
    tracer_provider=None,
    prospect_context=None,
    degraded_output=None,
    trace_quality="good",
    **kwargs,
) -> dict:
    """Execute an ADK-style multimodal agent: guardrails -> classify -> analyze -> extract -> summarize."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.multimodal.adk")

    if not query:
        query = get_random_query()

    text_query = query["text"]
    image_description = query["image_description"]
    image_type = query["image_type"]
    combined_input = f"{text_query}\n\n[Image: {image_description}]"

    llm = get_chat_llm(model, temperature=0)

    with tracer.start_as_current_span(
        "vision_agent.run",
        attributes={
            "openinference.span.kind": "AGENT",
            "input.value": combined_input,
            "input.mime_type": "text/plain",
            "metadata.framework": "adk",
            "metadata.use_case": "multimodal-ai",
        },
    ) as agent_span:

        # ---- Guardrails (LLM-based) ----
        for g in GUARDRAILS:
            run_guardrail(
                tracer, g["name"], combined_input, llm, guard,
                system_prompt=g["system_prompt"],
            )

        # ---- Local Guardrails (rule-based) ----
        for lg in LOCAL_GUARDRAILS:
            run_local_guardrail(
                tracer, lg["name"], combined_input,
                passed=lg["passed"],
                detail=lg.get("detail", ""),
            )

        # ---- generate_content: classify_image ----
        with tracer.start_as_current_span(
            "generate_content",
            attributes={
                "openinference.span.kind": "LLM",
                "input.value": combined_input,
                "input.mime_type": "text/plain",
            },
        ) as classify_span:
            classify_prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT_CLASSIFY_IMAGE),
                ("human", "Image description: {image_description}\nImage type hint: {image_type}"),
            ])
            classify_chain = classify_prompt | llm | StrOutputParser()
            if guard:
                guard.check()
            classification = classify_chain.invoke({
                "image_description": image_description,
                "image_type": image_type,
            })
            classify_span.set_attribute("output.value", classification)
            classify_span.set_attribute("output.mime_type", "text/plain")
            classify_span.set_status(Status(StatusCode.OK))

        # ---- Tool call: analyze image content ----
        run_tool_call(tracer, "analyze_image_content", image_description[:200], analyze_image_content, guard=guard, image_description=image_description)

        # ---- generate_content: analyze ----
        with tracer.start_as_current_span(
            "generate_content",
            attributes={
                "openinference.span.kind": "LLM",
                "input.value": combined_input,
                "input.mime_type": "text/plain",
            },
        ) as analyze_span:
            analyze_prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT_VISION.format(image_description=image_description)),
                ("human", "{input}"),
            ])
            analyze_chain = analyze_prompt | llm | StrOutputParser()
            if guard:
                guard.check()
            analysis = analyze_chain.invoke({"input": text_query})
            analyze_span.set_attribute("output.value", analysis)
            analyze_span.set_attribute("output.mime_type", "text/plain")
            analyze_span.set_status(Status(StatusCode.OK))

        # ---- extract_data (TOOL span) ----
        with tracer.start_as_current_span(
            "extract_data",
            attributes={
                "openinference.span.kind": "TOOL",
                "input.value": image_description,
                "input.mime_type": "text/plain",
            },
        ) as extract_span:
            extract_prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT_EXTRACT),
                ("human", "Image description: {image_description}"),
            ])
            extract_chain = extract_prompt | llm | StrOutputParser()
            if guard:
                guard.check()
            extracted_data = extract_chain.invoke({"image_description": image_description})
            extract_span.set_attribute("output.value", extracted_data)
            extract_span.set_attribute("output.mime_type", "text/plain")
            extract_span.set_status(Status(StatusCode.OK))

        # ---- generate_content: summarize ----
        with tracer.start_as_current_span(
            "generate_content",
            attributes={
                "openinference.span.kind": "LLM",
                "input.value": combined_input,
                "input.mime_type": "text/plain",
            },
        ) as summarize_span:
            summarize_prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT_SUMMARIZE_FINDINGS.format(
                    classification=classification,
                    analysis=analysis,
                )),
                ("human", "Extracted data:\n{extracted_data}\n\nOriginal question: {input}"),
            ])
            summarize_chain = summarize_prompt | llm | StrOutputParser()
            if guard:
                guard.check()
            summary = summarize_chain.invoke({
                "extracted_data": extracted_data,
                "input": text_query,
            })
            summarize_span.set_attribute("output.value", summary)
            summarize_span.set_attribute("output.mime_type", "text/plain")
            summarize_span.set_status(Status(StatusCode.OK))

        answer = summary

        agent_span.set_attribute("output.value", answer)
        agent_span.set_attribute("output.mime_type", "text/plain")
        agent_span.set_status(Status(StatusCode.OK))

    return {
        "query": query,
        "answer": answer,
    }
