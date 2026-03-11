"""
ADK-style chatbot agent with manual OpenTelemetry spans.
Uses common_runner_utils for query sampling.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ...cost_guard import CostGuard
from ...llm import get_chat_llm
from ...trace_enrichment import invoke_chain_in_context, run_guardrail, run_in_context, tool_definitions_json
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
    """Execute an ADK-style chatbot agent: guardrails -> plan -> tools -> synthesize."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    from ...use_cases import chatbot as chatbot_use_case

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.chatbot.adk")

    if not query:
        rng = kwargs.get("rng")
        _kw = {k: v for k, v in kwargs.items() if k != "rng"}
        query_spec = get_query_for_run(chatbot_use_case, prospect_context=prospect_context, rng=rng, **_kw)
        query = query_spec.text
    else:
        query_spec = None

    tools = get_tools(prospect_context)
    llm = get_chat_llm(model, temperature=0)
    tool_map = {t.name: t for t in tools}
    system_prompt = get_system_prompt(prospect_context)

    attrs = {
        "openinference.span.kind": "AGENT",
        "input.value": query,
        "input.mime_type": "text/plain",
        "metadata.framework": "adk",
        "metadata.use_case": "multiturn-chatbot-with-tools",
        "metadata.tool_definitions": tool_definitions_json(tools),
    }
    if trace_quality:
        attrs["metadata.trace_quality"] = trace_quality
    if query_spec:
        attrs.update(query_spec.to_span_attributes())
    with tracer.start_as_current_span("support_agent.run", attributes=attrs) as agent_span:

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
                f"- {t.name}: {t.description}" for t in tools
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
            plan = invoke_chain_in_context(plan_chain, {"input": query})
            plan_span.set_attribute("output.value", plan)
            plan_span.set_attribute("output.mime_type", "text/plain")
            plan_span.set_status(Status(StatusCode.OK))

        # ---- Tool execution: parse plan and call relevant tools ----
        tool_results = []
        query_lower = query.lower()
        tool_names = set(tool_map)

        # Dispatch by tool set: HR 1:1 tools vs default support tools
        tools_to_call = []
        if "get_my_goals" in tool_names:
            # HR / 1:1 prep tools
            if any(kw in query_lower for kw in ["goal", "okr", "objective", "priority", "focus"]):
                tools_to_call.append(("get_my_goals", {}))
            if any(kw in query_lower for kw in ["feedback", "review", "improve", "strength"]):
                tools_to_call.append(("get_recent_feedback", {}))
            if any(kw in query_lower for kw in ["agenda", "structure", "schedule", "time"]):
                tools_to_call.append(("get_1_1_agenda_template", {}))
            if any(kw in query_lower for kw in ["deadline", "due", "upcoming", "project"]):
                tools_to_call.append(("get_upcoming_deadlines", {}))
            if any(kw in query_lower for kw in ["accomplish", "win", "highlight", "share", "talk"]):
                tools_to_call.append(("get_recent_accomplishments", {}))
            if not tools_to_call:
                tools_to_call.append(("get_my_goals", {}))
                tools_to_call.append(("get_recent_accomplishments", {}))
        else:
            # Default support tools
            if any(kw in query_lower for kw in ["refund", "sso", "compliance", "monitoring", "knowledge", "search", "policy"]):
                if "search_knowledge_base" in tool_names:
                    tools_to_call.append(("search_knowledge_base", {"query": query}))
                if "lookup_policy" in tool_names and any(kw in query_lower for kw in ["policy", "expense", "pto", "remote"]):
                    tools_to_call.append(("lookup_policy", {"policy_name": "expense" if "expense" in query_lower else "PTO"}))
            if "account" in query_lower and "get_account_info" in tool_names:
                acc_id = "ACC-12345"
                for word in query.split():
                    if word.upper().startswith("ACC-"):
                        acc_id = word.strip(".,!?")
                        break
                tools_to_call.append(("get_account_info", {"account_id": acc_id}))
            if any(kw in query_lower for kw in ["cost", "token", "price", "calculate"]) and "calculate_cost" in tool_names:
                tokens = 1000000
                for word in query.split():
                    if word.isdigit():
                        tokens = int(word)
                        break
                tools_to_call.append(("calculate_cost", {"tokens": tokens, "model": "gpt-4o"}))
            if not tools_to_call and "search_knowledge_base" in tool_names:
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
                    result = run_in_context(tool_fn.invoke, tool_args)
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
                ("system", system_prompt),
                ("human",
                 "User question: {question}\n\n"
                 "Tool results:\n{tool_results}\n\n"
                 "Synthesize a helpful response based on the tool results above."),
            ])
            synth_chain = synth_prompt | llm | StrOutputParser()
            if guard:
                guard.check()
            answer = invoke_chain_in_context(synth_chain, {
                "question": query,
                "tool_results": tool_output_text,
            })
            synth_span.set_attribute("output.value", answer)
            synth_span.set_attribute("output.mime_type", "text/plain")
            synth_span.set_status(Status(StatusCode.OK))

        if degraded_output:
            answer = degraded_output
        agent_span.set_attribute("output.value", answer[:5000] if len(answer) > 5000 else answer)
        agent_span.set_attribute("output.mime_type", "text/plain")
        agent_span.set_attribute("tools.count", len(tool_results))
        if tool_results:
            agent_span.set_attribute("metadata.tools_used", ",".join(r["tool"] for r in tool_results))
            agent_span.set_attribute("context.tool_results", tool_output_text[:2000])
        agent_span.set_attribute("context.plan", plan[:1000] if plan else "")
        agent_span.set_status(Status(StatusCode.OK))

    return {
        "query": query,
        "answer": answer,
        "tools_used": tool_results,
    }
