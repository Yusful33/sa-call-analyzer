"""
LangChain LCEL chatbot/agent pipeline with tool calling and guardrails.
Uses prompt | llm chains with tool binding, auto-instrumented by LangChainInstrumentor.
"""

import json as _json
import random

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

from ...cost_guard import CostGuard
from ...llm import get_chat_llm
from ...trace_enrichment import run_guardrail
from ...use_cases.chatbot import QUERIES, GUARDRAILS, SYSTEM_PROMPT, TOOLS


def run_chatbot(
    query: str | None = None,
    model: str = "gpt-4o-mini",
    guard: CostGuard | None = None,
    tracer_provider=None,
) -> dict:
    """Execute a LangChain LCEL chatbot with tool calling and guardrails."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.chatbot")

    if not query:
        query = random.choice(QUERIES)

    llm = get_chat_llm(model, temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("messages"),
    ])
    chain = prompt | llm_with_tools

    with tracer.start_as_current_span(
        "customer_support_agent",
        attributes={
            "openinference.span.kind": "AGENT",
            "input.value": query,
            "input.mime_type": "text/plain",
        },
    ) as agent_span:

        # === GUARDRAIL ===
        with tracer.start_as_current_span(
            "validate_input",
            attributes={"openinference.span.kind": "GUARDRAIL", "input.value": query},
        ) as guard_span:
            for g in GUARDRAILS:
                run_guardrail(
                    tracer, g["name"], query, llm, guard,
                    system_prompt=g["system_prompt"],
                )
            guard_span.set_attribute("output.value", "All checks passed")
            guard_span.set_attribute("guardrail.passed", True)
            guard_span.set_status(Status(StatusCode.OK))

        # === QUERY ROUTING ===
        with tracer.start_as_current_span(
            "routing.analyze_query",
            attributes={"openinference.span.kind": "CHAIN", "input.value": query},
        ) as route_span:
            needs_tools = any(kw in query.lower() for kw in ["account", "cost", "calculate", "look up", "check"])
            route_result = "TOOL_REQUIRED" if needs_tools else "DIRECT_RESPONSE"
            route_span.set_attribute("output.value", route_result)
            route_span.set_status(Status(StatusCode.OK))

        # === AGENT REASONING ===
        with tracer.start_as_current_span(
            "plan_action",
            attributes={"openinference.span.kind": "CHAIN", "input.value": query},
        ) as step:
            if guard:
                guard.check()
            messages = [HumanMessage(content=query)]
            response = chain.invoke({"messages": messages})
            messages.append(response)
            tool_names = [tc["name"] for tc in response.tool_calls] if response.tool_calls else []
            step.set_attribute("output.value", f"Tools to call: {tool_names}" if tool_names else response.content)
            step.set_status(Status(StatusCode.OK))

        tools_used = []

        # === TOOL EXECUTION ===
        if response.tool_calls:
            tool_map = {t.name: t for t in TOOLS}
            for tc in response.tool_calls:
                with tracer.start_as_current_span(
                    f"tool.{tc['name']}",
                    attributes={
                        "openinference.span.kind": "TOOL",
                        "tool.name": tc["name"],
                        "input.value": _json.dumps(tc["args"]),
                        "input.mime_type": "application/json",
                    },
                ) as tool_span:
                    if guard:
                        guard.check()
                    tool_fn = tool_map.get(tc["name"])
                    if tool_fn:
                        result = tool_fn.invoke(tc["args"])
                        tools_used.append({"tool": tc["name"], "args": tc["args"], "result": result})
                        from langchain_core.messages import ToolMessage
                        messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
                        tool_span.set_attribute("output.value", str(result))
                    tool_span.set_status(Status(StatusCode.OK))

            # === FINAL SYNTHESIS ===
            with tracer.start_as_current_span(
                "synthesize_answer",
                attributes={"openinference.span.kind": "CHAIN", "input.value": query},
            ) as step:
                if guard:
                    guard.check()
                final_response = chain.invoke({"messages": messages})
                answer = final_response.content
                step.set_attribute("output.value", answer)
                step.set_status(Status(StatusCode.OK))
        else:
            answer = response.content

        agent_span.set_attribute("output.value", answer)
        agent_span.set_attribute("output.mime_type", "text/plain")
        agent_span.set_attribute("tools.count", len(tools_used))
        agent_span.set_status(Status(StatusCode.OK))

    return {
        "query": query,
        "answer": answer,
        "tools_used": tools_used,
    }
