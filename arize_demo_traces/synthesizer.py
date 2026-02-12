import json
import random
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .models import GenerateRequest, SpanPreview, UseCaseEnum, USE_CASE_LABELS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_ns() -> int:
    return int(time.time() * 1_000_000_000)


def _fmt_iso(ts_ns: int) -> str:
    return datetime.fromtimestamp(ts_ns / 1_000_000_000, tz=timezone.utc).isoformat()


def _ms_to_ns(ms: float) -> int:
    return int(ms * 1_000_000)


def _rand_latency(lo: float, hi: float) -> float:
    return random.uniform(lo, hi)


# ---------------------------------------------------------------------------
# Sample content pools
# ---------------------------------------------------------------------------

_USER_QUERIES = [
    "What is the refund policy for enterprise accounts?",
    "How do I configure SSO with Okta?",
    "Can you summarize last quarter's revenue performance?",
    "What are the best practices for prompt engineering?",
    "Explain the difference between fine-tuning and RAG.",
    "How do I set up monitoring for my LLM application?",
    "What compliance certifications does the platform support?",
    "Show me examples of multi-agent architectures.",
]

_ASSISTANT_RESPONSES = [
    "Based on the retrieved documents, enterprise accounts are eligible for a full refund within 30 days of purchase. After 30 days, a prorated refund may be issued depending on usage.",
    "To configure SSO with Okta, navigate to Settings > Authentication > SAML 2.0. You'll need your Okta metadata URL and entity ID.",
    "Last quarter's revenue was $42.3M, representing a 23% year-over-year increase. Key drivers included enterprise expansion and new product adoption.",
    "Best practices for prompt engineering include: 1) Be specific, 2) Use few-shot examples, 3) Break complex tasks into steps, 4) Set clear output formats.",
    "Fine-tuning modifies model weights for specialized tasks, while RAG retrieves documents at inference time. RAG is better for dynamic knowledge.",
    "To set up monitoring, integrate OpenTelemetry tracing, configure span exporters to Arize, and set up evaluators for relevance and hallucination.",
    "The platform supports SOC 2 Type II, HIPAA, GDPR, and ISO 27001 certifications.",
    "Multi-agent architectures typically use a supervisor agent that delegates tasks to specialized worker agents.",
]

_RETRIEVED_DOCS = [
    {"document_id": "doc-001", "score": 0.94, "content": "Enterprise refund policy: Full refund within 30 days..."},
    {"document_id": "doc-002", "score": 0.91, "content": "SSO Configuration Guide: Navigate to Settings > Authentication..."},
    {"document_id": "doc-003", "score": 0.88, "content": "Q3 2025 Financial Summary: Revenue $42.3M (+23% YoY)..."},
    {"document_id": "doc-004", "score": 0.85, "content": "Security & Compliance: SOC 2 Type II certified since 2022..."},
]

_GUARDRAIL_TOPICS = ["jailbreak_detection", "toxicity_check", "pii_detection", "topic_relevance"]

_LLM_MODELS = {
    "OpenAI": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
    "Anthropic": ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
    "Amazon Bedrock": ["anthropic.claude-3-sonnet", "amazon.titan-text-express"],
    "Vertex AI": ["gemini-1.5-pro", "gemini-1.5-flash"],
    "Mistral": ["mistral-large-latest", "mistral-medium-latest"],
}

_EMBEDDING_MODELS = ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]

_MCP_TOOLS = [
    ("analyze_stock", {"ticker": "AAPL", "period": "1Y"}),
    ("search_documents", {"query": "compliance policy", "limit": 10}),
    ("get_weather", {"location": "San Francisco", "units": "imperial"}),
    ("run_query", {"sql": "SELECT * FROM metrics LIMIT 100"}),
]

_AGENT_NAMES = [
    ("Research_Agent", "Gathers and synthesizes information from multiple sources"),
    ("Financial_Data_Agent", "Analyzes financial data and market trends"),
    ("Compliance_Agent", "Checks regulatory requirements and compliance"),
    ("Writing_Agent", "Drafts and refines written content"),
]


def _pick_model(providers: List[str]) -> str:
    if providers:
        provider = random.choice(providers)
        models = _LLM_MODELS.get(provider, ["gpt-4o"])
        return random.choice(models)
    return "gpt-4o"


def _pick_embedding_model() -> str:
    return random.choice(_EMBEDDING_MODELS)


def _llm_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return round((prompt_tokens * 0.000003) + (completion_tokens * 0.000015), 6)


# ---------------------------------------------------------------------------
# Span node builder
# ---------------------------------------------------------------------------

class SpanNode:
    """Tree node that builds a SpanPreview with children."""

    def __init__(
        self,
        name: str,
        span_kind: str,
        duration_ms: float,
        attributes: Dict[str, Any] | None = None,
        events: List[Dict] | None = None,
        status: str = "OK",
    ):
        self.name = name
        self.span_kind = span_kind
        self.duration_ms = duration_ms
        self.attributes = attributes or {}
        self.events = events or []
        self.status = status
        self.children: List["SpanNode"] = []

    def add_child(self, child: "SpanNode") -> "SpanNode":
        self.children.append(child)
        return child

    def to_spans(
        self,
        trace_id: str,
        parent_span_id: str | None,
        start_ns: int,
        base_attrs: Dict[str, Any],
    ) -> List[SpanPreview]:
        span_id = uuid.uuid4().hex[:16]
        end_ns = start_ns + _ms_to_ns(self.duration_ms)

        merged_attrs = {**base_attrs, **self.attributes, "openinference.span.kind": self.span_kind}
        events = list(self.events)

        if self.status == "ERROR":
            merged_attrs["error.type"] = "SyntheticFailure"
            merged_attrs["error.message"] = "Injected failure path"
            events.append({
                "name": "exception",
                "timestamp": _fmt_iso(start_ns + _ms_to_ns(self.duration_ms) // 2),
                "attributes": {"message": "Injected failure for demo"},
            })

        span = SpanPreview(
            span_id=span_id,
            parent_span_id=parent_span_id,
            name=self.name,
            status=self.status,
            start_time=_fmt_iso(start_ns),
            end_time=_fmt_iso(end_ns),
            attributes=merged_attrs,
            events=events,
        )

        result = [span]
        child_offset = start_ns + _ms_to_ns(self.duration_ms * 0.05)
        for child in self.children:
            child_spans = child.to_spans(trace_id, span_id, child_offset, base_attrs)
            result.extend(child_spans)
            child_offset += _ms_to_ns(child.duration_ms) + _ms_to_ns(random.uniform(5, 20))

        return result


# ---------------------------------------------------------------------------
# Shared sub-tree helpers
# ---------------------------------------------------------------------------

def _llm_span(name: str, model: str, prompt_tokens: int, completion_tokens: int,
              input_msgs: list, output_msgs: list, duration_ms: float | None = None,
              status: str = "OK") -> SpanNode:
    total = prompt_tokens + completion_tokens
    cost = _llm_cost(prompt_tokens, completion_tokens)
    return SpanNode(name, "LLM", duration_ms or _rand_latency(300, 3000), attributes={
        "llm.model_name": model,
        "llm.token_count.prompt": prompt_tokens,
        "llm.token_count.completion": completion_tokens,
        "llm.token_count.total": total,
        "llm.cost": cost,
        "llm.input_messages": json.dumps(input_msgs),
        "llm.output_messages": json.dumps(output_msgs),
    }, status=status)


def _guardrail_span(name: str, model: str, query: str, result: str = "pass") -> SpanNode:
    guard = SpanNode(name, "GUARDRAIL", _rand_latency(400, 1800), attributes={
        "guardrail.result": result,
    })
    pt = random.randint(100, 400)
    ct = random.randint(20, 80)
    guard.add_child(_llm_span("ChatCompletion", model, pt, ct,
        [{"role": "system", "content": f"You are a {name} classifier. Return PASS or FAIL."},
         {"role": "user", "content": query}],
        [{"role": "assistant", "content": result.upper()}],
    ))
    return guard


def _langgraph_agent_span(agent_name: str, model: str, query: str, response: str,
                          tools: list | None = None, failure: bool = False) -> SpanNode:
    """LangGraph-style: agent_name -> LangGraph -> agent -> call_model -> RunnableSequence -> Prompt + ChatOpenAI"""
    agent = SpanNode(agent_name, "AGENT", _rand_latency(5000, 25000), attributes={
        "input.value": query,
    })

    lg = SpanNode("LangGraph", "CHAIN", _rand_latency(4500, 24000))
    agent.add_child(lg)

    # First agent loop
    agent_inner = SpanNode("agent", "AGENT", _rand_latency(1000, 3000))
    lg.add_child(agent_inner)

    call_model = SpanNode("call_model", "CHAIN", _rand_latency(900, 2800))
    agent_inner.add_child(call_model)

    runnable = SpanNode("RunnableSequence", "CHAIN", _rand_latency(850, 2700))
    call_model.add_child(runnable)

    runnable.add_child(SpanNode("Prompt", "CHAIN", _rand_latency(1, 5)))

    pt1 = random.randint(200, 600)
    ct1 = random.randint(50, 200)
    runnable.add_child(_llm_span("ChatOpenAI", model, pt1, ct1,
        [{"role": "system", "content": f"You are {agent_name}. Analyze the request and use tools if needed."},
         {"role": "user", "content": query}],
        [{"role": "assistant", "content": f"I'll use the available tools to help with: {query}"}],
    ))

    agent_inner.add_child(SpanNode("should_continue", "CHAIN", _rand_latency(1, 10)))

    # Tool calls
    if tools:
        tools_span = SpanNode("tools", "CHAIN", _rand_latency(50, 500))
        lg.add_child(tools_span)
        for tool_name, tool_params in tools:
            tool_outer = SpanNode(tool_name, "TOOL", _rand_latency(20, 400), attributes={
                "tool.name": tool_name,
                "tool.parameters": json.dumps(tool_params),
            })
            tools_span.add_child(tool_outer)
            tool_inner = SpanNode(tool_name, "TOOL", _rand_latency(15, 380), attributes={
                "tool.name": tool_name,
                "output.value": json.dumps({"result": f"Data from {tool_name}"}),
            })
            tool_outer.add_child(tool_inner)

    # Second agent loop (with tool results)
    agent2 = SpanNode("agent", "AGENT", _rand_latency(2000, 5000))
    lg.add_child(agent2)

    call_model2 = SpanNode("call_model", "CHAIN", _rand_latency(1800, 4800))
    agent2.add_child(call_model2)

    runnable2 = SpanNode("RunnableSequence", "CHAIN", _rand_latency(1700, 4600))
    call_model2.add_child(runnable2)
    runnable2.add_child(SpanNode("Prompt", "CHAIN", _rand_latency(1, 5)))

    pt2 = random.randint(400, 900)
    ct2 = random.randint(100, 400)
    runnable2.add_child(_llm_span("ChatOpenAI", model, pt2, ct2,
        [{"role": "system", "content": f"You are {agent_name}. Synthesize the tool results."},
         {"role": "user", "content": query},
         {"role": "tool", "content": f"Tool results: analysis complete for {query}"}],
        [{"role": "assistant", "content": response}],
        status="ERROR" if failure else "OK",
    ))

    agent2.add_child(SpanNode("should_continue", "CHAIN", _rand_latency(1, 10)))

    if tools:
        for tool_name, _ in tools:
            lg.add_child(SpanNode(f"tool:{tool_name}", "TOOL", _rand_latency(5, 50)))

    return agent


def _crewai_agent_span(agent_name: str, model: str, query: str, response: str,
                       failure: bool = False) -> SpanNode:
    """CrewAI-style: Task.execute -> Agent.execute -> ChatCompletion"""
    task = SpanNode(f"Task.execute [{agent_name}]", "CHAIN", _rand_latency(5000, 20000), attributes={
        "input.value": query,
    })

    agent_exec = SpanNode(f"Agent.execute [{agent_name}]", "AGENT", _rand_latency(4000, 18000), attributes={
        "agent.name": agent_name,
    })
    task.add_child(agent_exec)

    pt = random.randint(300, 800)
    ct = random.randint(100, 400)
    agent_exec.add_child(_llm_span("ChatCompletion", model, pt, ct,
        [{"role": "system", "content": f"You are {agent_name}. Complete the assigned task."},
         {"role": "user", "content": query}],
        [{"role": "assistant", "content": response}],
        status="ERROR" if failure else "OK",
    ))

    return task


def _google_adk_agent_span(agent_name: str, model: str, query: str, response: str,
                           tools: list | None = None, failure: bool = False) -> SpanNode:
    """Google ADK-style: SubAgent.run -> generate_content + tool calls"""
    agent = SpanNode(f"{agent_name}.run", "AGENT", _rand_latency(5000, 22000), attributes={
        "input.value": query,
    })

    pt1 = random.randint(200, 500)
    ct1 = random.randint(50, 200)
    agent.add_child(_llm_span("generate_content", model, pt1, ct1,
        [{"role": "user", "content": query}],
        [{"role": "model", "content": f"I need to use tools to answer: {query}"}],
    ))

    if tools:
        for tool_name, tool_params in tools:
            tool = SpanNode(tool_name, "TOOL", _rand_latency(100, 2000), attributes={
                "tool.name": tool_name,
                "tool.parameters": json.dumps(tool_params),
                "output.value": json.dumps({"result": f"Data from {tool_name}"}),
            })
            agent.add_child(tool)

    pt2 = random.randint(400, 1000)
    ct2 = random.randint(150, 500)
    agent.add_child(_llm_span("generate_content", model, pt2, ct2,
        [{"role": "user", "content": query},
         {"role": "tool", "content": "Tool results available"}],
        [{"role": "model", "content": response}],
        status="ERROR" if failure else "OK",
    ))

    return agent


def _mcp_tool_span(tool_name: str, tool_params: dict) -> SpanNode:
    """MCP-style tool call: tool_name -> MCP.tool_name"""
    outer = SpanNode(tool_name, "TOOL", _rand_latency(1000, 18000), attributes={
        "tool.name": tool_name,
        "tool.parameters": json.dumps(tool_params),
    })
    inner = SpanNode(f"MCP.{tool_name}", "TOOL", _rand_latency(900, 17500), attributes={
        "tool.name": f"MCP.{tool_name}",
        "output.value": json.dumps({"result": f"MCP result from {tool_name}", "status": "success"}),
    })
    outer.add_child(inner)
    return outer


def _a2a_handler_chain(agent_label: str) -> SpanNode:
    """A2A protocol handler spans wrapping agent execution."""
    handler = SpanNode(f"{agent_label}:handle_request", "AGENT", _rand_latency(10000, 30000))

    handler.add_child(SpanNode(
        "a2a.server.request_handlers.jsonrpc_handler.JSONRPCHandler.on_message_send",
        "CHAIN", _rand_latency(9000, 29000)))

    for h in ["on_message_send", "setup_message_execution", "register_producer", "run_event_stream"]:
        handler.add_child(SpanNode(
            f"a2a.server.request_handlers.default_request_handler.DefaultRequestHandler.{h}",
            "CHAIN", _rand_latency(1, 50) if h != "run_event_stream" else _rand_latency(8000, 28000)))

    return handler


# ---------------------------------------------------------------------------
# Use-case trace tree builders
# ---------------------------------------------------------------------------

def _build_rag_tree(req: GenerateRequest, failure: bool) -> SpanNode:
    providers = req.tech_stack.providers
    model = _pick_model(providers)
    embed_model = _pick_embedding_model()
    query = random.choice(_USER_QUERIES)
    response = random.choice(_ASSISTANT_RESPONSES)
    docs = random.sample(_RETRIEVED_DOCS, k=min(3, len(_RETRIEVED_DOCS)))

    root = SpanNode("user_interaction", "CHAIN", _rand_latency(2800, 8500), attributes={
        "input.value": query,
        "output.value": response,
        "session.id": uuid.uuid4().hex[:12],
    }, status="ERROR" if failure else "OK")

    validate = SpanNode("validate_interaction", "CHAIN", _rand_latency(800, 2500))
    root.add_child(validate)
    for gn in random.sample(_GUARDRAIL_TOPICS, k=2):
        validate.add_child(_guardrail_span(gn, model, query,
            "fail" if (failure and random.random() < 0.3) else "pass"))

    pt_plan = random.randint(150, 300)
    ct_plan = random.randint(50, 150)
    root.add_child(_llm_span("ChatCompletion", model, pt_plan, ct_plan,
        [{"role": "system", "content": "You are a helpful assistant. Use retrieved context to answer."},
         {"role": "user", "content": query}],
        [{"role": "assistant", "content": "Let me search for relevant information..."}],
    ))

    retriever = SpanNode("VectorIndexRetriever.retrieve", "RETRIEVER", _rand_latency(400, 1200), attributes={
        "retrieval.documents": json.dumps(docs), "retrieval.top_k": 3,
    })
    root.add_child(retriever)
    inner_ret = SpanNode("VectorIndexRetriever._retrieve", "RETRIEVER", _rand_latency(350, 1100))
    retriever.add_child(inner_ret)
    embed_outer = SpanNode("OpenAIEmbedding.get_query_embedding", "EMBEDDING", _rand_latency(300, 900), attributes={
        "embedding.model_name": embed_model,
    })
    inner_ret.add_child(embed_outer)
    embed_outer.add_child(SpanNode("OpenAIEmbedding._get_query_embedding", "EMBEDDING",
        _rand_latency(250, 850), attributes={"embedding.model_name": embed_model}))

    et = random.randint(8, 24)
    root.add_child(SpanNode("CreateEmbeddingResponse", "EMBEDDING", _rand_latency(200, 700), attributes={
        "embedding.model_name": embed_model, "llm.token_count.prompt": et, "llm.token_count.total": et,
    }))

    pt_final = random.randint(180, 450)
    ct_final = random.randint(80, 350)
    root.add_child(_llm_span("ChatCompletion", model, pt_final, ct_final,
        [{"role": "system", "content": "Use the following context to answer."},
         {"role": "user", "content": f"Context: {json.dumps(docs)}\n\nQuestion: {query}"}],
        [{"role": "assistant", "content": response}],
        status="ERROR" if failure else "OK",
    ))

    return root


def _build_multiagent_chatbot_tree(req: GenerateRequest, failure: bool) -> SpanNode:
    """Multi-agent chatbot: adapts nesting to selected framework."""
    providers = req.tech_stack.providers
    frameworks = req.tech_stack.frameworks
    model = _pick_model(providers)
    query = random.choice(_USER_QUERIES)
    response = random.choice(_ASSISTANT_RESPONSES)
    agents = random.sample(_AGENT_NAMES, k=2)
    tools_pool = random.sample(_MCP_TOOLS, k=2)

    # Determine framework-specific root and agent builders
    framework = frameworks[0] if frameworks else "LangGraph"

    if framework == "LangGraph":
        root = SpanNode("LangGraph", "CHAIN", _rand_latency(15000, 45000), attributes={
            "input.value": query, "output.value": response, "session.id": uuid.uuid4().hex[:12],
        }, status="ERROR" if failure else "OK")

        # Supervisor
        sup = SpanNode("Supervisor_Agent", "AGENT", _rand_latency(500, 1500))
        root.add_child(sup)
        rs = SpanNode("RunnableSequence", "CHAIN", _rand_latency(450, 1400))
        sup.add_child(rs)
        rs.add_child(SpanNode("ChatPromptTemplate", "CHAIN", _rand_latency(1, 5)))
        pt_sup = random.randint(200, 500)
        ct_sup = random.randint(30, 100)
        rs.add_child(_llm_span("ChatOpenAI", model, pt_sup, ct_sup,
            [{"role": "system", "content": "You are a supervisor. Route to the best agent."},
             {"role": "user", "content": query}],
            [{"role": "assistant", "content": f"Routing to {agents[0][0]} and {agents[1][0]}"}],
        ))
        rs.add_child(SpanNode("RunnableLambda", "CHAIN", _rand_latency(1, 5)))
        sup.add_child(SpanNode("Unnamed", "CHAIN", _rand_latency(1, 5)))

        # Sub-agents
        for i, (agent_name, _desc) in enumerate(agents):
            agent_tools = [tools_pool[i]] if i < len(tools_pool) else None
            root.add_child(_langgraph_agent_span(
                agent_name, model, query, response,
                tools=agent_tools,
                failure=failure and i == 0,
            ))

    elif framework == "CrewAI":
        root = SpanNode("CrewAI", "CHAIN", _rand_latency(15000, 40000), attributes={
            "input.value": query, "output.value": response, "session.id": uuid.uuid4().hex[:12],
        }, status="ERROR" if failure else "OK")

        kickoff = SpanNode("Crew.kickoff", "CHAIN", _rand_latency(14000, 38000))
        root.add_child(kickoff)

        for i, (agent_name, _desc) in enumerate(agents):
            kickoff.add_child(_crewai_agent_span(
                agent_name, model, query, response,
                failure=failure and i == 0,
            ))

    elif framework == "Google AI SDK":
        root = SpanNode("ADK", "CHAIN", _rand_latency(15000, 40000), attributes={
            "input.value": query, "output.value": response, "session.id": uuid.uuid4().hex[:12],
        }, status="ERROR" if failure else "OK")

        orchestrator = SpanNode("Orchestrator", "AGENT", _rand_latency(14000, 38000))
        root.add_child(orchestrator)

        pt_orch = random.randint(200, 400)
        ct_orch = random.randint(30, 100)
        orchestrator.add_child(_llm_span("generate_content", model, pt_orch, ct_orch,
            [{"role": "user", "content": query}],
            [{"role": "model", "content": f"Delegating to {agents[0][0]} and {agents[1][0]}"}],
        ))

        for i, (agent_name, _desc) in enumerate(agents):
            agent_tools = [tools_pool[i]] if i < len(tools_pool) else None
            orchestrator.add_child(_google_adk_agent_span(
                agent_name, model, query, response,
                tools=agent_tools,
                failure=failure and i == 0,
            ))

    else:
        # Generic multi-agent
        root = SpanNode("multi_agent_run", "CHAIN", _rand_latency(10000, 35000), attributes={
            "input.value": query, "output.value": response, "session.id": uuid.uuid4().hex[:12],
        }, status="ERROR" if failure else "OK")
        for i, (agent_name, _desc) in enumerate(agents):
            agent = SpanNode(agent_name, "AGENT", _rand_latency(3000, 15000))
            root.add_child(agent)
            pt = random.randint(200, 600)
            ct = random.randint(80, 300)
            agent.add_child(_llm_span("ChatCompletion", model, pt, ct,
                [{"role": "system", "content": f"You are {agent_name}."},
                 {"role": "user", "content": query}],
                [{"role": "assistant", "content": response}],
                status="ERROR" if (failure and i == 0) else "OK",
            ))

    return root


def _build_single_agent_chatbot_tree(req: GenerateRequest, failure: bool) -> SpanNode:
    """Single-agent chatbot with tools and guardrails."""
    providers = req.tech_stack.providers
    model = _pick_model(providers)
    query = random.choice(_USER_QUERIES)
    response = random.choice(_ASSISTANT_RESPONSES)

    root = SpanNode("agent_interaction", "CHAIN", _rand_latency(3000, 12000), attributes={
        "input.value": query, "output.value": response, "session.id": uuid.uuid4().hex[:12],
    }, status="ERROR" if failure else "OK")

    pt_r = random.randint(100, 250)
    ct_r = random.randint(20, 60)
    router = SpanNode("intent_router", "CHAIN", _rand_latency(200, 600), attributes={
        "input.value": query, "output.value": "tool_call: search_knowledge_base",
    })
    root.add_child(router)
    router.add_child(_llm_span("ChatCompletion", model, pt_r, ct_r,
        [{"role": "system", "content": "Classify intent and decide which tool to call."},
         {"role": "user", "content": query}],
        [{"role": "assistant", "content": '{"tool": "search_knowledge_base"}'}],
    ))

    root.add_child(SpanNode("search_knowledge_base", "TOOL", _rand_latency(500, 1500), attributes={
        "tool.name": "search_knowledge_base",
        "tool.parameters": json.dumps({"query": query, "top_k": 5}),
        "output.value": json.dumps(random.sample(_RETRIEVED_DOCS, k=2)),
    }))

    for gn in random.sample(_GUARDRAIL_TOPICS, k=1):
        root.add_child(_guardrail_span(gn, model, response))

    pt_f = random.randint(200, 600)
    ct_f = random.randint(100, 400)
    root.add_child(_llm_span("ChatCompletion", model, pt_f, ct_f,
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": query}],
        [{"role": "assistant", "content": response}],
        status="ERROR" if failure else "OK",
    ))

    return root


def _build_chatbot_tree(req: GenerateRequest, failure: bool) -> SpanNode:
    """Routes to multi-agent or single-agent chatbot based on architecture selection."""
    arch = req.tech_stack.architecture
    if "multi-agent" in arch:
        return _build_multiagent_chatbot_tree(req, failure)
    return _build_single_agent_chatbot_tree(req, failure)


def _build_mcp_agent_tree(req: GenerateRequest, failure: bool) -> SpanNode:
    """MCP agent pattern: root -> Agent workflow -> Assistant -> mcp_tools + response + MCP.tool + response"""
    providers = req.tech_stack.providers
    model = _pick_model(providers)
    query = random.choice(_USER_QUERIES)
    response = random.choice(_ASSISTANT_RESPONSES)
    mcp_tool = random.choice(_MCP_TOOLS)

    root = SpanNode("financial-analysis", "CHAIN", _rand_latency(20000, 55000), attributes={
        "input.value": query, "output.value": response, "session.id": uuid.uuid4().hex[:12],
    }, status="ERROR" if failure else "OK")

    workflow = SpanNode("Agent workflow", "AGENT", _rand_latency(18000, 50000))
    root.add_child(workflow)

    assistant = SpanNode("Assistant", "AGENT", _rand_latency(17000, 48000))
    workflow.add_child(assistant)

    assistant.add_child(SpanNode("mcp_tools", "CHAIN", _rand_latency(5, 20), attributes={
        "output.value": json.dumps({"tools_discovered": [t[0] for t in _MCP_TOOLS]}),
    }))

    pt1 = random.randint(200, 500)
    ct1 = random.randint(50, 200)
    assistant.add_child(_llm_span("response", model, pt1, ct1,
        [{"role": "system", "content": "You are a financial analyst assistant with MCP tools."},
         {"role": "user", "content": query}],
        [{"role": "assistant", "content": f"I'll use {mcp_tool[0]} to analyze this."}],
    ))

    assistant.add_child(_mcp_tool_span(mcp_tool[0], mcp_tool[1]))

    pt2 = random.randint(500, 1200)
    ct2 = random.randint(200, 600)
    assistant.add_child(_llm_span("response", model, pt2, ct2,
        [{"role": "system", "content": "Synthesize the MCP tool results."},
         {"role": "user", "content": query},
         {"role": "tool", "content": f"MCP result from {mcp_tool[0]}"}],
        [{"role": "assistant", "content": response}],
        status="ERROR" if failure else "OK",
    ))

    return root


def _build_a2a_tree(req: GenerateRequest, failure: bool) -> SpanNode:
    """A2A pattern: orchestrator -> decide_routing -> call_remote_agent -> A2A handlers -> framework agent"""
    providers = req.tech_stack.providers
    frameworks = req.tech_stack.frameworks
    model = _pick_model(providers)
    query = random.choice(_USER_QUERIES)
    response = random.choice(_ASSISTANT_RESPONSES)
    agents = random.sample(_AGENT_NAMES, k=2)
    tools_pool = random.sample(_MCP_TOOLS, k=2)

    root = SpanNode("cl:analyze_request", "CHAIN", _rand_latency(15000, 50000), attributes={
        "input.value": query, "output.value": response,
    }, status="ERROR" if failure else "OK")

    orch_handler = SpanNode("orchestrator:handle_request", "AGENT", _rand_latency(14000, 48000))
    root.add_child(orch_handler)

    # A2A server handler chain
    orch_handler.add_child(SpanNode(
        "a2a.server.request_handlers.jsonrpc_handler.JSONRPCHandler.on_message_send",
        "CHAIN", _rand_latency(13000, 46000)))

    for h in ["on_message_send", "setup_message_execution", "register_producer", "run_event_stream"]:
        orch_handler.add_child(SpanNode(
            f"a2a.server.request_handlers.default_request_handler.DefaultRequestHandler.{h}",
            "CHAIN", _rand_latency(1, 30) if h != "run_event_stream" else _rand_latency(12000, 44000)))

    # Orchestrator execute
    orch_exec = SpanNode("orchestrator.execute", "CHAIN", _rand_latency(12000, 42000))
    orch_handler.add_child(orch_exec)

    # Decide routing
    routing = SpanNode("orchestrator.decide_routing", "AGENT", _rand_latency(800, 2500))
    orch_exec.add_child(routing)

    invocation = SpanNode("invocation [trading_strategy_orchestrator]", "CHAIN", _rand_latency(700, 2200))
    routing.add_child(invocation)
    agent_run = SpanNode("agent_run [trading_strategy_orchestrator]", "AGENT", _rand_latency(600, 2000))
    invocation.add_child(agent_run)
    pt_route = random.randint(200, 500)
    ct_route = random.randint(30, 100)
    agent_run.add_child(_llm_span("call_llm", model, pt_route, ct_route,
        [{"role": "system", "content": "Route to the best agent."},
         {"role": "user", "content": query}],
        [{"role": "assistant", "content": f"Route to {agents[0][0]} and {agents[1][0]}"}],
    ))

    # Remote agent calls
    framework = frameworks[0] if frameworks else "LangGraph"

    for i, (agent_name, agent_desc) in enumerate(agents):
        fw_label = "ADK + MCP" if (i == 0 and framework == "Google AI SDK") else "LangGraph + MCP"
        remote = SpanNode(f"call_remote_agent:{agent_name} ({fw_label})", "CHAIN",
            _rand_latency(5000, 25000))
        orch_exec.add_child(remote)

        agent_handler = _a2a_handler_chain(agent_name.lower().replace(" ", "_"))
        remote.add_child(agent_handler)

        # Framework-specific inner agent
        agent_tools = [tools_pool[i]] if i < len(tools_pool) else None
        if "LangGraph" in fw_label:
            inner = _langgraph_agent_span(agent_name.lower().replace(" ", "_"), model, query, response,
                tools=agent_tools, failure=failure and i == 0)
        else:
            inner = _google_adk_agent_span(agent_name, model, query, response,
                tools=agent_tools, failure=failure and i == 0)
        agent_handler.add_child(inner)

        # A2A cleanup handlers
        for cleanup in ["cleanup_producer", "validate_task_id_match", "send_push_notification_if_needed"]:
            agent_handler.add_child(SpanNode(
                f"a2a.server.request_handlers.default_request_handler.DefaultRequestHandler.{cleanup}",
                "CHAIN", _rand_latency(1, 10)))

    # Orchestrator cleanup
    for cleanup in ["cleanup_producer", "validate_task_id_match", "send_push_notification_if_needed"]:
        root.add_child(SpanNode(
            f"a2a.server.request_handlers.default_request_handler.DefaultRequestHandler.{cleanup}",
            "CHAIN", _rand_latency(1, 10)))

    return root


def _build_text_to_sql_tree(req: GenerateRequest, failure: bool) -> SpanNode:
    providers = req.tech_stack.providers
    model = _pick_model(providers)
    query = "What were total sales by region last quarter?"
    sql = "SELECT region, SUM(amount) as total_sales FROM sales WHERE quarter = 'Q3-2025' GROUP BY region ORDER BY total_sales DESC"

    root = SpanNode("nl2sql_interaction", "CHAIN", _rand_latency(3000, 9000), attributes={
        "input.value": query,
        "output.value": json.dumps({"sql": sql, "rows": 5, "execution_time_ms": 342}),
        "session.id": uuid.uuid4().hex[:12],
    }, status="ERROR" if failure else "OK")

    root.add_child(SpanNode("authorize_user", "CHAIN", _rand_latency(50, 200), attributes={
        "auth.user_id": "user-12345", "auth.permissions": json.dumps(["read:sales", "read:analytics"]),
    }))

    root.add_child(SpanNode("fetch_schema", "TOOL", _rand_latency(100, 400), attributes={
        "tool.name": "fetch_schema", "output.value": json.dumps({"tables": ["sales", "regions", "products"], "columns": 42}),
    }))

    sql_gen = SpanNode("generate_sql", "CHAIN", _rand_latency(800, 2500))
    root.add_child(sql_gen)
    pt_sql = random.randint(300, 800)
    ct_sql = random.randint(50, 200)
    sql_gen.add_child(_llm_span("ChatCompletion", model, pt_sql, ct_sql,
        [{"role": "system", "content": "Generate SQL. Schema: sales(id, region, amount, quarter), regions(id, name)"},
         {"role": "user", "content": query}],
        [{"role": "assistant", "content": sql}],
    ))

    root.add_child(SpanNode("validate_sql", "GUARDRAIL", _rand_latency(100, 400), attributes={
        "guardrail.result": "pass", "input.value": sql,
    }))

    root.add_child(SpanNode("execute_query", "TOOL", _rand_latency(200, 800), attributes={
        "tool.name": "execute_query", "input.value": sql,
        "output.value": json.dumps({"rows": 5, "execution_time_ms": 342}),
    }, status="ERROR" if failure else "OK"))

    pt_sum = random.randint(200, 500)
    ct_sum = random.randint(80, 250)
    root.add_child(_llm_span("ChatCompletion", model, pt_sum, ct_sum,
        [{"role": "system", "content": "Summarize SQL results in natural language."},
         {"role": "user", "content": f"Query: {query}\nSQL: {sql}\nResults: 5 rows"}],
        [{"role": "assistant", "content": "Last quarter, North America led with $15.2M, followed by EMEA at $12.8M."}],
    ))

    return root


def _build_generic_tree(req: GenerateRequest, failure: bool) -> SpanNode:
    providers = req.tech_stack.providers
    model = _pick_model(providers)
    query = random.choice(_USER_QUERIES)
    response = random.choice(_ASSISTANT_RESPONSES)

    root = SpanNode("agent_run", "CHAIN", _rand_latency(2000, 7000), attributes={
        "input.value": query, "output.value": response,
    }, status="ERROR" if failure else "OK")

    pt1 = random.randint(100, 400)
    ct1 = random.randint(50, 200)
    root.add_child(_llm_span("ChatCompletion", model, pt1, ct1,
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": query}],
        [{"role": "assistant", "content": "I'll process this request..."}],
    ))

    root.add_child(SpanNode("process_request", "TOOL", _rand_latency(300, 1500), attributes={
        "tool.name": "process_request", "input.value": query, "output.value": "Processing complete.",
    }))

    pt2 = random.randint(200, 600)
    ct2 = random.randint(80, 300)
    root.add_child(_llm_span("ChatCompletion", model, pt2, ct2,
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": query}],
        [{"role": "assistant", "content": response}],
        status="ERROR" if failure else "OK",
    ))

    return root


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_TREE_BUILDERS = {
    UseCaseEnum.rag_search: _build_rag_tree,
    UseCaseEnum.multiturn_chatbot: _build_chatbot_tree,
    UseCaseEnum.text_to_sql: _build_text_to_sql_tree,
}


def _select_builder(req: GenerateRequest):
    """Select the right builder based on use case + architecture + framework."""
    # Explicit use-case builders take priority
    if req.use_case in _TREE_BUILDERS:
        return _TREE_BUILDERS[req.use_case]

    # If multi-agent is selected, use the multi-agent chatbot builder (framework-aware)
    if "multi-agent" in req.tech_stack.architecture:
        return _build_multiagent_chatbot_tree

    return _build_generic_tree


def _base_attributes(req: GenerateRequest) -> Dict:
    return {
        "use_case": USE_CASE_LABELS.get(req.use_case, req.use_case.value),
        "architecture": req.tech_stack.architecture,
        "frameworks": req.tech_stack.frameworks,
        "providers": req.tech_stack.providers,
        "languages": req.tech_stack.languages,
    }


def synthesize_traces(req: GenerateRequest) -> List[List[SpanPreview]]:
    base_attrs = _base_attributes(req)
    builder = _select_builder(req)
    all_traces: List[List[SpanPreview]] = []

    traces_to_make = max(1, req.traces_per_variant)
    for _ in range(traces_to_make):
        for failure in [False, True]:
            trace_id = uuid.uuid4().hex[:32]
            tree = builder(req, failure)
            start_ns = _now_ns()
            spans = tree.to_spans(trace_id, None, start_ns, base_attrs)
            all_traces.append(spans)

    return all_traces


def synthesize_preview(req: GenerateRequest) -> List[SpanPreview]:
    traces = synthesize_traces(req)
    return [span for trace in traces for span in trace]
