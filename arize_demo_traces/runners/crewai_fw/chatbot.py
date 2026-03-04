"""
CrewAI chatbot pipeline with routing and support agents.
Uses common_runner_utils for query sampling.
"""

from crewai import Agent, Crew, Task, Process

from ...cost_guard import CostGuard
from ...llm import get_chat_llm
from ...trace_enrichment import run_guardrail, tool_definitions_json
from ...use_cases.chatbot import GUARDRAILS, get_system_prompt, get_tools
from ..common_runner_utils import get_query_for_run


def run_chatbot(
    query: str | None = None,
    model: str = "gpt-4o-mini",
    guard: CostGuard | None = None,
    tracer_provider=None,
    prospect_context=None,
    degraded_output=None,
    trace_quality="good",
    **kwargs,
) -> dict:
    """Execute a CrewAI chatbot pipeline with routing agent, support agent, guardrails, and evaluation."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    from ...use_cases import chatbot as chatbot_use_case

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.chatbot.crewai")
    if not query:
        rng = kwargs.get("rng")
        _kw = {k: v for k, v in kwargs.items() if k != "rng"}
        query_spec = get_query_for_run(chatbot_use_case, prospect_context=prospect_context, rng=rng, **_kw)
        query = query_spec.text
    else:
        query_spec = None

    llm = get_chat_llm(model, temperature=0)

    attrs = {
        "openinference.span.kind": "AGENT",
        "input.value": query,
        "input.mime_type": "text/plain",
        "metadata.framework": "crewai",
        "metadata.use_case": "multiturn-chatbot-with-tools",
        "metadata.tool_definitions": tool_definitions_json(tools),
    }
    if trace_quality:
        attrs["metadata.trace_quality"] = trace_quality
    if query_spec:
        attrs.update(query_spec.to_span_attributes())
    tools = get_tools(prospect_context)
    system_prompt = get_system_prompt(prospect_context)
    with tracer.start_as_current_span("chatbot_pipeline", attributes=attrs) as pipeline_span:

        # === GUARDRAILS ===
        for g in GUARDRAILS:
            run_guardrail(
                tracer, g["name"], query, llm, guard,
                system_prompt=g["system_prompt"],
            )

        # === CREWAI AGENTS ===
        routing_agent = Agent(
            role="Query Router",
            goal="Classify incoming queries and create an action plan for the support team",
            backstory=(
                "You are an experienced customer support lead who excels at triaging "
                "incoming requests. You quickly categorize queries by type (billing, "
                "technical, general) and determine what tools or resources are needed."
            ),
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )

        support_agent = Agent(
            role="Customer Support Specialist",
            goal="Provide accurate, helpful answers to customer questions using available tools",
            backstory=(
                "You are a knowledgeable customer support specialist with access to "
                "the knowledge base, account lookup, and cost calculation tools. "
                "You always use the right tool for the job and provide clear, actionable answers."
            ),
            llm=llm,
            verbose=False,
            allow_delegation=False,
            tools=tools,
        )

        # === CREWAI TASKS ===
        routing_task = Task(
            description=(
                f"Analyze the following customer query and classify it. Determine what "
                f"category it falls into (billing, technical, account, general) and what "
                f"tools or actions are needed to answer it.\n\n"
                f"Customer query: {query}"
            ),
            expected_output=(
                "A brief classification of the query type and a plan for how to answer it, "
                "including which tools should be used"
            ),
            agent=routing_agent,
        )

        response_task = Task(
            description=(
                f"Using the routing analysis and available tools, provide a comprehensive "
                f"answer to the customer's question.\n\n"
                f"Customer query: {query}\n\n"
                f"System context: {system_prompt}"
            ),
            expected_output="A helpful, accurate, and complete response to the customer's question",
            agent=support_agent,
        )

        # === CREWAI CREW ===
        crew = Crew(
            agents=[routing_agent, support_agent],
            tasks=[routing_task, response_task],
            process=Process.sequential,
            verbose=False,
        )

        if guard:
            guard.check()
        result = crew.kickoff()
        answer = str(result)
        if degraded_output:
            answer = degraded_output
        pipeline_span.set_attribute("output.value", answer[:5000] if len(answer) > 5000 else answer)
        pipeline_span.set_attribute("output.mime_type", "text/plain")
        pipeline_span.set_status(Status(StatusCode.OK))

    return {
        "query": query,
        "answer": answer,
        "tools_used": [],  # CrewAI doesn't expose per-tool list easily
    }
