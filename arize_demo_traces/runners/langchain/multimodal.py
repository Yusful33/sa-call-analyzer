"""
LangChain LCEL multimodal/vision pipeline with guardrails.
Uses manual OTel spans with LCEL chains (prompt | llm | StrOutputParser) for each step.
Auto-instrumented by LangChainInstrumentor for authentic LangChain trace patterns.
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
    SYSTEM_PROMPT_VISION,
    SYSTEM_PROMPT_EXTRACT,
    SYSTEM_PROMPT_CLASSIFY_IMAGE,
    SYSTEM_PROMPT_SUMMARIZE_FINDINGS,
    get_random_query,
    analyze_image_content,
    extract_structured_data,
)


def run_multimodal(
    query: str | None = None,
    model: str = "gpt-4o-mini",
    guard: CostGuard | None = None,
    tracer_provider=None,
    prospect_context=None,
) -> dict:
    """Execute a LangChain LCEL multimodal pipeline with guardrails."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.multimodal")

    # Handle query: None -> random dict, string -> wrap with default image description
    if query is None:
        q = get_random_query()
        query_text = q["text"]
        image_description = q["image_description"]
    elif isinstance(query, dict):
        query_text = query.get("text", str(query))
        image_description = query.get("image_description", "No image description provided.")
    else:
        query_text = str(query)
        image_description = "A general-purpose image relevant to the user query."

    combined_input = f"{query_text}\n\nImage: {image_description}"
    llm = get_chat_llm(model, temperature=0)

    with tracer.start_as_current_span(
        "multimodal_pipeline",
        attributes={
            "openinference.span.kind": "CHAIN",
            "input.value": combined_input,
            "input.mime_type": "text/plain",
            "metadata.framework": "langchain",
            "metadata.use_case": "multimodal-ai",
        },
    ) as pipeline_span:

        # === GUARDRAILS ===
        with tracer.start_as_current_span(
            "validate_input",
            attributes={"openinference.span.kind": "GUARDRAIL", "input.value": combined_input},
        ) as guard_span:
            for g in GUARDRAILS:
                run_guardrail(
                    tracer, g["name"], combined_input, llm, guard,
                    system_prompt=g["system_prompt"],
                )
            for lg in LOCAL_GUARDRAILS:
                run_local_guardrail(
                    tracer, lg["name"], combined_input,
                    passed=lg["passed"],
                    detail=lg.get("detail", ""),
                )
            guard_span.set_attribute("output.value", "All checks passed")
            guard_span.set_attribute("guardrail.passed", True)
            guard_span.set_status(Status(StatusCode.OK))

        # === CLASSIFY IMAGE ===
        with tracer.start_as_current_span(
            "classify_image",
            attributes={"openinference.span.kind": "CHAIN", "input.value": combined_input},
        ) as step:
            classify_prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT_CLASSIFY_IMAGE),
                ("human", "Image description: {image_description}\n\nQuery: {query}"),
            ])
            classify_chain = classify_prompt | llm | StrOutputParser()
            if guard:
                guard.check()
            classification = classify_chain.invoke({
                "image_description": image_description,
                "query": query_text,
            })
            step.set_attribute("output.value", classification)
            step.set_status(Status(StatusCode.OK))

        # === ANALYZE CONTENT ===
        with tracer.start_as_current_span(
            "analyze_content",
            attributes={"openinference.span.kind": "CHAIN", "input.value": combined_input},
        ) as step:
            run_tool_call(tracer, "analyze_image_content", image_description[:200], analyze_image_content, guard=guard, image_description=image_description)
            analyze_prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT_VISION),
                ("human", "{query}"),
            ])
            analyze_chain = analyze_prompt | llm | StrOutputParser()
            if guard:
                guard.check()
            analysis = analyze_chain.invoke({
                "image_description": image_description,
                "query": query_text,
            })
            step.set_attribute("output.value", analysis)
            step.set_status(Status(StatusCode.OK))

        # === EXTRACT DATA ===
        with tracer.start_as_current_span(
            "extract_data",
            attributes={"openinference.span.kind": "CHAIN", "input.value": combined_input},
        ) as step:
            extract_prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT_EXTRACT),
                ("human", "Image description: {image_description}\n\nAnalysis: {analysis}"),
            ])
            extract_chain = extract_prompt | llm | StrOutputParser()
            if guard:
                guard.check()
            extraction = extract_chain.invoke({
                "image_description": image_description,
                "analysis": analysis,
            })
            run_tool_call(tracer, "extract_structured_data", extraction[:200], extract_structured_data, guard=guard, text=extraction[:500])
            step.set_attribute("output.value", extraction)
            step.set_status(Status(StatusCode.OK))

        # === SUMMARIZE FINDINGS ===
        with tracer.start_as_current_span(
            "summarize_findings",
            attributes={"openinference.span.kind": "CHAIN", "input.value": combined_input},
        ) as step:
            summarize_prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT_SUMMARIZE_FINDINGS),
                ("human", "Original query: {query}\n\nExtracted data:\n{extraction}"),
            ])
            summarize_chain = summarize_prompt | llm | StrOutputParser()
            if guard:
                guard.check()
            summary = summarize_chain.invoke({
                "query": query_text,
                "classification": classification,
                "analysis": analysis,
                "extraction": extraction,
            })
            step.set_attribute("output.value", summary)
            step.set_status(Status(StatusCode.OK))

        pipeline_span.set_attribute("output.value", summary)
        pipeline_span.set_attribute("output.mime_type", "text/plain")
        pipeline_span.set_status(Status(StatusCode.OK))

    return {
        "query": query_text,
        "image_description": image_description,
        "classification": classification,
        "analysis": analysis,
        "extraction": extraction,
        "summary": summary,
    }
