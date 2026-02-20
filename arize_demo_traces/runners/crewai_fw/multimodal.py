"""
CrewAI multimodal / vision pipeline with vision analyst and summarizer agents.
Uses real CrewAI Crew, Agent, Task objects to generate authentic trace patterns
when instrumented by CrewAIInstrumentor.
"""

import random

from crewai import Agent, Crew, Task, Process

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
    extract_structured_data,
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
    """Execute a CrewAI multimodal pipeline: guardrails -> vision+summarizer crew."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.multimodal.crewai")
    if not query:
        query = get_random_query()

    text_query = query["text"]
    image_description = query["image_description"]
    image_type = query["image_type"]
    combined_input = f"{text_query}\n\n[Image: {image_description}]"

    llm = get_chat_llm(model, temperature=0)

    with tracer.start_as_current_span(
        "multimodal_pipeline",
        attributes={
            "openinference.span.kind": "CHAIN",
            "input.value": combined_input,
            "input.mime_type": "text/plain",
            "metadata.framework": "crewai",
            "metadata.use_case": "multimodal-ai",
        },
    ) as root_span:

        # === GUARDRAILS (LLM-based) ===
        for g in GUARDRAILS:
            run_guardrail(
                tracer, g["name"], combined_input, llm, guard,
                system_prompt=g["system_prompt"],
            )

        # === LOCAL GUARDRAILS (rule-based) ===
        for lg in LOCAL_GUARDRAILS:
            run_local_guardrail(
                tracer, lg["name"], combined_input,
                passed=lg["passed"],
                detail=lg.get("detail", ""),
            )

        # === TOOL CALLS ===
        run_tool_call(tracer, "analyze_image_content", image_description[:200], analyze_image_content, guard=guard, image_description=image_description)
        run_tool_call(tracer, "extract_structured_data", "extraction", extract_structured_data, guard=guard, text="document content")

        # === CREWAI AGENTS ===
        vision_analyst = Agent(
            role="Vision Analyst",
            goal="Classify and deeply analyze visual content to extract meaningful observations",
            backstory=(
                "You are an expert in computer vision and image analysis. You can "
                "classify images into categories, identify objects and defects, "
                "and provide detailed visual assessments."
            ),
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )

        summarizer = Agent(
            role="Multimodal Summarizer",
            goal="Extract structured data from visual content and produce concise executive summaries",
            backstory=(
                "You are a specialist in combining visual analysis with structured data "
                "extraction. You produce clear, actionable summaries that synthesize "
                "image classification, extracted data, and detailed analysis."
            ),
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )

        # === CREWAI TASKS ===
        classify_and_analyze_task = Task(
            description=(
                f"Perform two analyses on the following image:\n\n"
                f"Image description: {image_description}\n"
                f"Image type: {image_type}\n"
                f"User question: {text_query}\n\n"
                f"1. CLASSIFY the image:\n{SYSTEM_PROMPT_CLASSIFY_IMAGE}\n\n"
                f"2. ANALYZE the image in detail:\n"
                f"{SYSTEM_PROMPT_VISION.format(image_description=image_description)}\n\n"
                f"Provide both the classification and detailed analysis."
            ),
            expected_output=(
                "A structured report containing: (1) image classification with category "
                "and confidence, (2) detailed analysis addressing the user's question"
            ),
            agent=vision_analyst,
        )

        extract_and_summarize_task = Task(
            description=(
                f"Based on the vision analysis, extract structured data and produce a summary.\n\n"
                f"Image description: {image_description}\n"
                f"User question: {text_query}\n\n"
                f"1. EXTRACT structured data:\n{SYSTEM_PROMPT_EXTRACT}\n\n"
                f"2. SUMMARIZE all findings into an executive summary using the "
                f"classification and analysis from the previous task."
            ),
            expected_output=(
                "A JSON object with extracted data followed by a concise executive "
                "summary combining classification, analysis, and extracted information"
            ),
            agent=summarizer,
        )

        # === CREWAI CREW ===
        crew = Crew(
            agents=[vision_analyst, summarizer],
            tasks=[classify_and_analyze_task, extract_and_summarize_task],
            process=Process.sequential,
            verbose=False,
        )

        if guard:
            guard.check()
        result = crew.kickoff()
        answer = str(result)

        root_span.set_attribute("output.value", answer)
        root_span.set_attribute("output.mime_type", "text/plain")
        root_span.set_status(Status(StatusCode.OK))

    return {
        "query": query,
        "answer": answer,
    }
