"""
Shared guardrail and evaluator spans for demo trace pipelines.
Adds realistic GUARDRAIL and EVALUATOR spans matching production Arize demo patterns.
"""

import contextvars
import json
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def tool_definitions_json(tools) -> str:
    """
    Serialize a list of LangChain tools to a JSON array of tool definitions
    (name, description, parameters schema) for attachment to a parent span.
    Enables evals to use tool definitions without inferring from child spans.
    """
    definitions = []
    for t in tools:
        entry = {
            "name": getattr(t, "name", str(t)),
            "description": getattr(t, "description", "") or "",
        }
        args_schema = getattr(t, "args_schema", None)
        if args_schema is not None and hasattr(args_schema, "model_json_schema"):
            try:
                entry["parameters"] = args_schema.model_json_schema()
            except Exception:
                entry["parameters"] = {}
        else:
            entry["parameters"] = {}
        definitions.append(entry)
    return json.dumps(definitions)


def run_in_context(fn, *args, **kwargs):
    """
    Run fn(*args, **kwargs) in the current trace context so any spans created
    (e.g. by LangChain/LangGraph instrumentors) are parented to the current span.
    Prevents orphaned spans when runners are executed in worker threads.
    """
    return contextvars.copy_context().run(lambda: fn(*args, **kwargs))


def invoke_chain_in_context(chain, input_dict):
    """
    Run chain.invoke in the current trace context so LangChain-instrumented spans
    (RunnableSequence, ChatOpenAI, etc.) are parented to the current span.
    Prevents orphaned spans when runners are executed in worker threads.
    """
    return run_in_context(chain.invoke, input_dict)


def invoke_llm_in_context(llm, messages):
    """
    Run llm.invoke in the current trace context so LLM spans are parented correctly.
    Prevents orphaned spans when runners are executed in worker threads.
    """
    return run_in_context(llm.invoke, messages)


def run_guardrail(tracer, name, input_text, llm, guard=None, system_prompt=None):
    """Create a GUARDRAIL span with an LLM safety check."""
    from opentelemetry.trace import Status, StatusCode

    default_prompt = (
        "You are a security guardrail. Analyze the input for safety issues. "
        "Respond with ONLY 'PASS' or 'FAIL: <brief reason>'."
    )

    with tracer.start_as_current_span(
        name,
        attributes={
            "openinference.span.kind": "GUARDRAIL",
            "input.value": input_text[:500],
            "input.mime_type": "text/plain",
        },
    ) as span:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt or default_prompt),
            ("human", "{input}"),
        ])
        chain = prompt | llm | StrOutputParser()
        if guard:
            guard.check()
        result = invoke_chain_in_context(chain, {"input": input_text})
        passed = not result.strip().upper().startswith("FAIL")
        span.set_attribute("output.value", result)
        span.set_attribute("output.mime_type", "text/plain")
        span.set_attribute("guardrail.passed", passed)
        span.set_status(Status(StatusCode.OK))
        return passed


def run_local_guardrail(tracer, name, input_text, passed=True, detail=""):
    """Create a GUARDRAIL span without an LLM call (rule-based check)."""
    from opentelemetry.trace import Status, StatusCode

    with tracer.start_as_current_span(
        name,
        attributes={
            "openinference.span.kind": "GUARDRAIL",
            "input.value": input_text[:500],
        },
    ) as span:
        result = "PASS" if passed else f"FAIL: {detail}"
        span.set_attribute("output.value", result)
        span.set_attribute("guardrail.passed", passed)
        span.set_status(Status(StatusCode.OK))
        return passed


def run_evaluator(tracer, name, question, response, llm, guard=None, criteria="quality and relevance"):
    """Create an EVALUATOR span with an LLM-based evaluation."""
    from opentelemetry.trace import Status, StatusCode

    system_prompt = (
        f"Evaluate the {criteria} of the response. "
        f'Return ONLY JSON: {{{{"score": <0.0-1.0>, "label": "good"|"ok"|"bad", "explanation": "<1 sentence>"}}}}'
    )

    with tracer.start_as_current_span(
        name,
        attributes={
            "openinference.span.kind": "EVALUATOR",
            "input.value": question[:500],
            "input.mime_type": "text/plain",
        },
    ) as span:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Question: {question}\n\nResponse: {response}"),
        ])
        chain = prompt | llm | StrOutputParser()
        if guard:
            guard.check()
        result = invoke_chain_in_context(chain, {"question": question, "response": response[:1000]})
        span.set_attribute("output.value", result)
        try:
            parsed = json.loads(result.strip())
            if "score" in parsed:
                span.set_attribute("eval.score", float(parsed["score"]))
            if "label" in parsed:
                span.set_attribute("eval.label", parsed["label"])
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
        span.set_status(Status(StatusCode.OK))
        return result


def run_tool_call(tracer, name, input_value, tool_fn, guard=None, **kwargs):
    """Create a TOOL span and execute a tool function."""
    from opentelemetry.trace import Status, StatusCode

    with tracer.start_as_current_span(
        name,
        attributes={
            "openinference.span.kind": "TOOL",
            "tool.name": name,
            "input.value": str(input_value)[:500],
            "input.mime_type": "text/plain",
        },
    ) as span:
        if guard:
            guard.check()
        result = tool_fn(**kwargs)
        span.set_attribute("output.value", str(result)[:2000])
        span.set_attribute("output.mime_type", "text/plain")
        span.set_status(Status(StatusCode.OK))
        return result


def run_local_evaluator(tracer, name, score, label="", input_value="", explanation=""):
    """Create an EVALUATOR span for computed (non-LLM) metrics like response time."""
    from opentelemetry.trace import Status, StatusCode

    with tracer.start_as_current_span(
        name,
        attributes={
            "openinference.span.kind": "EVALUATOR",
            "input.value": input_value or "metric computation",
            "input.mime_type": "text/plain",
        },
    ) as span:
        output = explanation if explanation else f"{score}"
        span.set_attribute("output.value", output)
        span.set_attribute("output.mime_type", "text/plain")
        span.set_attribute("eval.score", float(score) if isinstance(score, (int, float)) else 0.0)
        if label:
            span.set_attribute("eval.label", label)
        if explanation:
            span.set_attribute("eval.explanation", explanation)
        span.set_status(Status(StatusCode.OK))
        return score
