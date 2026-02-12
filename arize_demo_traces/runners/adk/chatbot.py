"""
ADK-style chatbot agent with manual OpenTelemetry spans.
Mimics Google Agent Development Kit trace patterns without requiring ADK installation.
Uses real LLM calls inside manually created OTel spans with OpenInference attributes.

Trace structure:
  support_agent.run (AGENT)
    -> guardrails (GUARDRAIL group)
    -> generate_content (LLM) - analyze query and plan tool calls
    -> tool execution (TOOL spans) - execute relevant tools
    -> generate_content (LLM) - synthesize final answer
"""

import random

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
    """Execute an ADK-style chatbot agent: guardrails -> plan -> tools -> synthesize."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.chatbot.adk")

    if not query:
        query = random.choice(QUERIES)

    llm = get_chat_llm(model, temperature=0)
    tool_map = {t.name: t for t in TOOLS}

    with tracer.start_as_current_span(
        "support_agent.run",
        attributes={
            "openinference.span.kind": "AGENT",
            "input.value": query,
            "input.mime_type": "text/plain",
        },
    ) as agent_span:

        # ---- Guardrails ----
        for g in GUARDRAILS:
            run_guardrail(
                tracer, g["name"], query, llm, guard,
                system_prompt=g["system_prompt"],
            )

        # ---- generate_content: analyze query and plan tool calls ----
        with tracer.start_as_current_span(
            "generate_content",
            attributes={
                "openinference.span.kind": "LLM",
                "input.value": query,
                "input.mime_type": "text/plain",
            },
        ) as plan_span:
            tool_descriptions = "\n".join(
                f"- {t.name}: {t.description}" for t in TOOLS
            )
            plan_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 f"You are a support agent planner. Given the user query, decide which tools to use.\n\n"
                 f"Available tools:\n{tool_descriptions}\n\n"
                 "List the tool names and arguments you would call, one per line. "
                 "Format: tool_name(arg_value)\n"
                 "If no tools are needed, say 'NO_TOOLS'."),
                ("human", "{input}"),
            ])
            plan_chain = plan_prompt | llm | StrOutputParser()
            if guard:
                guard.check()
            plan = plan_chain.invoke({"input": query})
            plan_span.set_attribute("output.value", plan)
            plan_span.set_attribute("output.mime_type", "text/plain")
            plan_span.set_status(Status(StatusCode.OK))

        # ---- Tool execution: parse plan and call relevant tools ----
        tool_results = []
        query_lower = query.lower()

        # Determine which tools to call based on query keywords
        tools_to_call = []
        if any(kw in query_lower for kw in ["refund", "sso", "compliance", "monitoring", "knowledge", "search"]):
            tools_to_call.append(("search_knowledge_base", {"query": query}))
        if "account" in query_lower:
            # Extract account ID from query or use a default
            acc_id = "ACC-12345"
            for word in query.split():
                if word.upper().startswith("ACC-"):
                    acc_id = word.strip(".,!?")
                    break
            tools_to_call.append(("get_account_info", {"account_id": acc_id}))
        if any(kw in query_lower for kw in ["cost", "token", "price", "calculate"]):
            tokens = 1000000
            for word in query.split():
                if word.isdigit():
                    tokens = int(word)
                    break
            tools_to_call.append(("calculate_cost", {"tokens": tokens, "model": "gpt-4o"}))

        # If no keywords matched, default to searching knowledge base
        if not tools_to_call:
            tools_to_call.append(("search_knowledge_base", {"query": query}))

        for tool_name, tool_args in tools_to_call:
            with tracer.start_as_current_span(
                tool_name,
                attributes={
                    "openinference.span.kind": "TOOL",
                    "input.value": str(tool_args),
                    "input.mime_type": "text/plain",
                    "tool.name": tool_name,
                },
            ) as tool_span:
                tool_fn = tool_map.get(tool_name)
                if tool_fn:
                    if guard:
                        guard.check()
                    result = tool_fn.invoke(tool_args)
                    tool_results.append({"tool": tool_name, "args": tool_args, "result": result})
                    tool_span.set_attribute("output.value", str(result))
                    tool_span.set_attribute("output.mime_type", "text/plain")
                    tool_span.set_status(Status(StatusCode.OK))
                else:
                    tool_span.set_attribute("output.value", f"Tool '{tool_name}' not found")
                    tool_span.set_status(Status(StatusCode.ERROR, "Tool not found"))

        # ---- generate_content: synthesize final answer ----
        with tracer.start_as_current_span(
            "generate_content",
            attributes={
                "openinference.span.kind": "LLM",
                "input.value": query,
                "input.mime_type": "text/plain",
            },
        ) as synth_span:
            tool_output_text = "\n".join(
                f"[{r['tool']}]: {r['result']}" for r in tool_results
            )
            synth_prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("human",
                 "User question: {question}\n\n"
                 "Tool results:\n{tool_results}\n\n"
                 "Synthesize a helpful response based on the tool results above."),
            ])
            synth_chain = synth_prompt | llm | StrOutputParser()
            if guard:
                guard.check()
            answer = synth_chain.invoke({
                "question": query,
                "tool_results": tool_output_text,
            })
            synth_span.set_attribute("output.value", answer)
            synth_span.set_attribute("output.mime_type", "text/plain")
            synth_span.set_status(Status(StatusCode.OK))

        agent_span.set_attribute("output.value", answer)
        agent_span.set_attribute("output.mime_type", "text/plain")
        agent_span.set_attribute("tools.count", len(tool_results))
        agent_span.set_status(Status(StatusCode.OK))

    return {
        "query": query,
        "answer": answer,
        "tools_used": tool_results,
    }
