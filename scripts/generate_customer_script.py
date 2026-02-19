"""
Generate a standalone customer demo script from the current codebase.

Given a (use_case, framework, model, project_name) tuple, reads the relevant
source files at runtime and assembles a single self-contained .py file that
customers can run to reproduce demo traces in their own Arize project.
"""

import re
import textwrap
from pathlib import Path

# Root of the arize_demo_traces package
_TRACES_ROOT = Path(__file__).resolve().parent.parent / "arize_demo_traces"

# Use-case slug -> filename stem mapping
_USE_CASE_FILE_MAP = {
    "retrieval-augmented-search": "rag",
    "multiturn-chatbot-with-tools": "chatbot",
    "text-to-sql-bi-agent": "text_to_sql",
    "multi-agent-orchestration": "multi_agent",
    "classification-routing": "classification",
    "multimodal-ai": "multimodal",
    "mcp-tool-use": "mcp",
    "travel-agent": "travel_agent",
    "generic": "generic",
    "generic-llm-chain": "generic",
}

# Framework slug -> runner subdirectory
_FRAMEWORK_DIR_MAP = {
    "langgraph": "langgraph",
    "langchain": "langchain",
    "crewai": "crewai_fw",
    "adk": "adk",
}

# Base deps needed by all scripts
_BASE_DEPS = [
    "arize-otel",
    "openinference-instrumentation-openai",
    "openinference-instrumentation-langchain",
    "opentelemetry-api",
    "opentelemetry-sdk",
    "langchain-core",
    "langchain-openai",
    "litellm",
]

# Framework-specific additional deps
_FRAMEWORK_DEPS = {
    "langgraph": ["langgraph"],
    "langchain": [],
    "crewai": ["crewai", "openinference-instrumentation-crewai"],
    "adk": ["google-adk"],
}

# Use-case-specific additional deps
_USE_CASE_DEPS = {
    "retrieval-augmented-search": ["langchain-chroma", "chromadb"],
}


def _resolve_use_case_file(use_case: str) -> Path:
    """Resolve the use_cases/*.py file for a given use-case slug."""
    stem = _USE_CASE_FILE_MAP.get(use_case, "generic")
    path = _TRACES_ROOT / "use_cases" / f"{stem}.py"
    if path.exists():
        return path
    return _TRACES_ROOT / "use_cases" / "generic.py"


def _resolve_runner_file(framework: str, use_case: str) -> Path:
    """Resolve the runner file, mimicking the registry's fallback chain."""
    stem = _USE_CASE_FILE_MAP.get(use_case, "generic")
    fw_dir = _FRAMEWORK_DIR_MAP.get(framework, "langgraph")

    # 1. Exact match
    path = _TRACES_ROOT / "runners" / fw_dir / f"{stem}.py"
    if path.exists():
        return path

    # 2. Same framework, generic
    path = _TRACES_ROOT / "runners" / fw_dir / "generic.py"
    if path.exists():
        return path

    # 3. LangGraph with same use case
    if framework != "langgraph":
        path = _TRACES_ROOT / "runners" / "langgraph" / f"{stem}.py"
        if path.exists():
            return path

    # 4. Final fallback: langgraph generic
    return _TRACES_ROOT / "runners" / "langgraph" / "generic.py"


def _read_and_strip(path: Path) -> str:
    """Read a Python source file and strip relative imports."""
    source = path.read_text()

    # Remove lines with relative imports (from ...foo import bar, from ..foo import bar, from .foo import bar)
    lines = source.split("\n")
    cleaned = []
    for line in lines:
        # Match relative imports: from .something or from ..something
        if re.match(r"^\s*from\s+\.+\w*\s+import\s+", line):
            cleaned.append(f"# [inlined] {line.strip()}")
        else:
            cleaned.append(line)
    return "\n".join(cleaned)


def _build_deps_list(framework: str, use_case: str, model: str) -> list[str]:
    """Build the complete pip install dependency list."""
    deps = list(_BASE_DEPS)
    deps.extend(_FRAMEWORK_DEPS.get(framework, []))
    deps.extend(_USE_CASE_DEPS.get(use_case, []))
    if model.startswith("claude-"):
        deps.append("langchain-anthropic")
    return sorted(set(deps))


def generate_script(
    use_case: str,
    framework: str,
    model: str = "gpt-4o-mini",
    project_name: str = "customer-demo",
) -> str:
    """Generate a standalone customer demo script.

    Reads the relevant source files, strips relative imports, and assembles
    them into a single runnable Python file.

    Args:
        use_case: Use case slug (e.g., "retrieval-augmented-search").
        framework: Framework name (e.g., "langgraph").
        model: LLM model name.
        project_name: Arize project name for trace export.

    Returns:
        The complete script as a string.
    """
    # Resolve source files
    use_case_file = _resolve_use_case_file(use_case)
    runner_file = _resolve_runner_file(framework, use_case)
    trace_enrichment_file = _TRACES_ROOT / "trace_enrichment.py"
    llm_file = _TRACES_ROOT / "llm.py"
    cost_guard_file = _TRACES_ROOT / "cost_guard.py"
    eval_wrapper_file = _TRACES_ROOT / "eval_wrapper.py"
    online_evals_file = _TRACES_ROOT / "online_evals.py"

    # Read and strip all modules
    use_case_source = _read_and_strip(use_case_file)
    runner_source = _read_and_strip(runner_file)
    trace_enrichment_source = _read_and_strip(trace_enrichment_file)
    llm_source = _read_and_strip(llm_file)
    cost_guard_source = _read_and_strip(cost_guard_file)
    eval_wrapper_source = _read_and_strip(eval_wrapper_file)
    online_evals_source = _read_and_strip(online_evals_file)

    # Build dependency list
    deps = _build_deps_list(framework, use_case, model)
    pip_install = f"pip install {' '.join(deps)}"

    # Detect the runner function name from the runner source
    runner_func_match = re.search(r"^def (run_\w+)\(", runner_source, re.MULTILINE)
    runner_func_name = runner_func_match.group(1) if runner_func_match else "run_generic"

    # Build the script
    script = _SCRIPT_TEMPLATE.format(
        pip_install=pip_install,
        project_name=project_name,
        use_case=use_case,
        framework=framework,
        model=model,
        use_case_filename=use_case_file.name,
        runner_filename=runner_file.name,
        runner_framework_dir=runner_file.parent.name,
        cost_guard_source=cost_guard_source,
        llm_source=llm_source,
        trace_enrichment_source=trace_enrichment_source,
        use_case_source=use_case_source,
        runner_source=runner_source,
        eval_wrapper_source=eval_wrapper_source,
        online_evals_source=online_evals_source,
        runner_func_name=runner_func_name,
    )

    return script


# ── Script template ──────────────────────────────────────────────────────────
_SCRIPT_TEMPLATE = '''#!/usr/bin/env python3
"""
Arize Demo Script — How trace data is sent to the platform
===========================================================
Project:   {project_name}
Use Case:  {use_case}
Framework: {framework}
Model:     {model}

This script is for prospective customers to see how trace data flows into Arize.
When you run it interactively, each question you type triggers the same pipeline
that would run in production; OpenTelemetry captures the run as a trace, and
the script exports that trace to Arize. You can then open the Arize app and
see your request, the agent's response, and the full span hierarchy — so you
understand exactly how data is sent into the platform.

  python {project_name}_demo.py     # Interactive: try a few turns, then view traces in Arize

To generate 10 synthetic traces without interaction (e.g. for eval demos):

  python {project_name}_demo.py --batch

Setup:
  1. Install dependencies:
     {pip_install}

  2. Set environment variables (or pass via --space-id / --api-key, or enter when prompted):
     export ARIZE_API_KEY="your-arize-api-key"   # Your own Arize API key — traces go to your project
     export ARIZE_SPACE_ID="your-arize-space-id" # Your own Arize Space ID
     export OPENAI_API_KEY="your-openai-api-key"
     # If using Claude models:
     # export ANTHROPIC_API_KEY="your-anthropic-api-key"

  3. Run the script. Traces are sent to Arize using your API key and Space ID. Open the URL printed below.
"""

import os
import sys
import json
import time
import random
import threading
import re
import requests
from typing import Any, Callable, List, TypedDict
from pathlib import Path
from datetime import datetime, timezone, timedelta

# ── Verify LLM API key (required for running the pipeline) ─────────────────────
if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
    print("ERROR: Set OPENAI_API_KEY (or ANTHROPIC_API_KEY for Claude) so the pipeline can call the LLM.")
    sys.exit(1)


# ============================================================================
# Module: cost_guard
# ============================================================================
{cost_guard_source}


# ============================================================================
# Module: llm  (from arize_demo_traces/llm.py)
# ============================================================================
{llm_source}


# ============================================================================
# Module: trace_enrichment  (from arize_demo_traces/trace_enrichment.py)
# ============================================================================
{trace_enrichment_source}


# ============================================================================
# Module: use_case  (from arize_demo_traces/use_cases/{use_case_filename})
# ============================================================================
{use_case_source}


# ============================================================================
# Module: runner  (from arize_demo_traces/runners/{runner_framework_dir}/{runner_filename})
# ============================================================================
{runner_source}


# ============================================================================
# Module: eval_wrapper  (from arize_demo_traces/eval_wrapper.py)
# ============================================================================
{eval_wrapper_source}


# ============================================================================
# Module: online_evals  (from arize_demo_traces/online_evals.py)
# ============================================================================
{online_evals_source}


# ============================================================================
# Main: Interactive agent or batch trace generation
# ============================================================================
PROJECT_NAME = "{project_name}"
USE_CASE = "{use_case}"
FRAMEWORK = "{framework}"
MODEL = "{model}"
NUM_TRACES = 10
POISONED_RATIO = 0.30


def _print_result(result: dict) -> None:
    """Print the main output from a runner result."""
    for key in ("answer", "response", "final_output", "generated_sql", "summary"):
        if key in result and result[key]:
            print(str(result[key])[:2000])
            return
    print(result)


def _run_batch(tracer_provider, guard, space_id: str, api_key: str) -> None:
    """Generate NUM_TRACES synthetic traces and create online eval."""
    bad_indices = set(random.sample(range(NUM_TRACES), k=max(1, int(NUM_TRACES * POISONED_RATIO))))
    print(f"Generating {{NUM_TRACES}} traces ({{len(bad_indices)}} poisoned)...")
    results = []
    for i in range(NUM_TRACES):
        quality = "poisoned" if i in bad_indices else "good"
        print(f"  Trace {{i + 1}}/{{NUM_TRACES}} ({{quality}})...", end=" ", flush=True)
        try:
            result = run_with_evals(
                runner={runner_func_name},
                use_case=USE_CASE,
                framework=FRAMEWORK,
                model=MODEL,
                guard=guard,
                tracer_provider=tracer_provider,
                force_bad=(i in bad_indices),
            )
            results.append(result)
            print("done")
        except Exception as e:
            print(f"error: {{e}}")
            break
        try:
            tracer_provider.force_flush(timeout_millis=5000)
        except Exception:
            pass
    try:
        tracer_provider.force_flush(timeout_millis=30000)
    except Exception:
        pass
    print(f"Successfully generated {{len(results)}}/{{NUM_TRACES}} traces.")
    print("Creating online evaluation task...")
    try:
        eval_result = create_and_run_online_eval(
            project_name=PROJECT_NAME,
            use_case=USE_CASE,
            space_id=space_id,
            api_key=api_key,
        )
        if eval_result.get("success"):
            print(f"Online eval '{{eval_result.get('eval_name', 'unknown')}}' created and running.")
        elif eval_result.get("error"):
            print(f"Online eval setup note: {{eval_result['error']}}")
    except Exception as e:
        print(f"Online eval setup skipped: {{e}}")
    arize_url = f"https://app.arize.com/?space_id={{space_id}}&project_id={{PROJECT_NAME}}"
    print(f"\\nDone! View your traces at:\\n  {{arize_url}}")


def _run_interactive(tracer_provider, guard, space_id: str) -> None:
    """Interactive demo: each turn is traced and sent to Arize so the user can see how trace data flows."""
    arize_url = f"https://app.arize.com/?space_id={{space_id}}&project_id={{PROJECT_NAME}}" if space_id else None
    print("\\n" + "=" * 60)
    print("HOW TRACE DATA IS SENT TO ARIZE")
    print("=" * 60)
    print("Each message you send below runs the {{USE_CASE}} pipeline. OpenTelemetry")
    print("records the run as a trace (spans for each step), and this script exports")
    print("that trace to your Arize project. After a few turns, open the link below")
    print("to see your requests and responses in the Arize app — same data you see here.")
    if arize_url:
        print(f"\\n  View traces in Arize: {{arize_url}}")
    print("\\nType a message and press Enter. Empty line or 'quit'/'exit' to stop.")
    print("=" * 60 + "\\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\\nExiting.")
            break
        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break
        try:
            result = run_with_evals(
                runner={runner_func_name},
                use_case=USE_CASE,
                framework=FRAMEWORK,
                model=MODEL,
                guard=guard,
                tracer_provider=tracer_provider,
                force_bad=False,
                query=user_input,
            )
            print("Agent: ", end="")
            _print_result(result)
            if arize_url:
                print("  \\n  → Trace sent to Arize. Open the link above to see this turn in the platform.")
        except Exception as e:
            print(f"Error: {{e}}")
        try:
            tracer_provider.force_flush(timeout_millis=5000)
        except Exception:
            pass
        print()


def _get_arize_credentials() -> tuple[str, str]:
    """Get Arize Space ID and API key from CLI, env, or prompt. Traces are sent using the user's own credentials."""
    import argparse
    p = argparse.ArgumentParser(description="Run Arize trace demo (interactive or --batch).")
    p.add_argument("--batch", action="store_true", help="Generate 10 synthetic traces instead of interactive mode.")
    p.add_argument("--space-id", type=str, default=os.getenv("ARIZE_SPACE_ID"), help="Your Arize Space ID (or set ARIZE_SPACE_ID).")
    p.add_argument("--api-key", type=str, default=os.getenv("ARIZE_API_KEY"), help="Your Arize API key (or set ARIZE_API_KEY).")
    args, _ = p.parse_known_args()
    space_id = (args.space_id or "").strip()
    api_key = (args.api_key or "").strip()
    if not space_id:
        try:
            space_id = input("Enter your Arize Space ID (traces will be sent to this space): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\\nArize Space ID is required. Set ARIZE_SPACE_ID or pass --space-id.")
            sys.exit(1)
    if not api_key:
        try:
            api_key = input("Enter your Arize API key: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\\nArize API key is required. Set ARIZE_API_KEY or pass --api-key.")
            sys.exit(1)
    return space_id, api_key


def main():
    from arize.otel import register as arize_register
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.instrumentation.openai import OpenAIInstrumentor

    space_id, api_key = _get_arize_credentials()

    print(f"Setting up tracing for project '{{PROJECT_NAME}}' (traces will be sent to Arize using your Space ID and API key).")
    tracer_provider = arize_register(
        space_id=space_id,
        api_key=api_key,
        project_name=PROJECT_NAME,
    )

    LangChainInstrumentor().instrument(tracer_provider=tracer_provider, skip_dep_check=True)
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider, skip_dep_check=True)

    batch_mode = "--batch" in sys.argv  # argparse in _get_arize_credentials also parses --batch
    guard = CostGuard(max_calls=500 if batch_mode else 200)

    if batch_mode:
        _run_batch(tracer_provider, guard, space_id, api_key)
    else:
        _run_interactive(tracer_provider, guard, space_id)

    try:
        tracer_provider.force_flush(timeout_millis=5000)
    except Exception:
        pass


if __name__ == "__main__":
    main()
'''
