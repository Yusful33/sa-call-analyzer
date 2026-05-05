"""
Generates synthetic demo files by executing the arize-synthetic-demo skill logic.

This module implements the skill's workflow:
1. Generate domain-specific content (queries, tools, prompts) using an LLM
2. Fill in the generator skeleton template
3. Package all files into a ZIP archive
"""

import io
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Any, Optional

# Path to the skill templates - check local bundled templates first, then user's home directory
_API_DIR = Path(__file__).parent
_LOCAL_TEMPLATES = _API_DIR / "demo_templates"
_USER_SKILL_PATH = Path.home() / ".claude" / "skills" / "arize-synthetic-demo"
_USER_TEMPLATES = _USER_SKILL_PATH / "templates"

# Use local bundled templates if available (for Vercel deployment), otherwise user's skill
if _LOCAL_TEMPLATES.exists():
    SKILL_PATH = _API_DIR
    TEMPLATES_PATH = _LOCAL_TEMPLATES
    SNIPPETS_PATH = _LOCAL_TEMPLATES / "snippets"
else:
    SKILL_PATH = _USER_SKILL_PATH
    TEMPLATES_PATH = _USER_TEMPLATES
    SNIPPETS_PATH = _USER_TEMPLATES / "snippets"


def slugify(text: str) -> str:
    """Convert text to a URL-safe slug."""
    s = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip().lower()).strip("_")
    return s or "demo"


# Framework to LLM metadata mapping (from references/frameworks.md)
FRAMEWORK_METADATA = {
    "openai": {
        "model": "gpt-4o-mini",
        "provider": "openai",
        "system": "openai",
        "span_name": "ChatCompletion",
    },
    "anthropic": {
        "model": "claude-3-7-sonnet-latest",
        "provider": "anthropic",
        "system": "anthropic",
        "span_name": "Messages",
    },
    "bedrock": {
        "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "provider": "aws",
        "system": "bedrock",
        "span_name": "InvokeModel",
    },
    "vertex": {
        "model": "gemini-2.0-flash",
        "provider": "google",
        "system": "vertexai",
        "span_name": "AsyncGenerateContent",
    },
    "adk": {
        "model": "gemini-2.0-flash",
        "provider": "google",
        "system": "vertexai",
        "span_name": "AsyncGenerateContent",
    },
    "langchain": {
        "model": "gpt-4o-mini",
        "provider": "openai",
        "system": "openai",
        "span_name": "ChatOpenAI",
    },
    "langgraph": {
        "model": "gpt-4o-mini",
        "provider": "openai",
        "system": "openai",
        "span_name": "ChatOpenAI",
    },
    "crewai": {
        "model": "gpt-4o",
        "provider": "openai",
        "system": "openai",
        "span_name": "crew_run",
    },
    "generic": {
        "model": "synthetic-llm-v1",
        "provider": "synthetic",
        "system": "generic",
        "span_name": "LLM",
    },
}

# Architecture to snippet file mapping
ARCHITECTURE_SNIPPETS = {
    "single_agent": "single_agent.py.snippet",
    "multi_agent_coordinator": "multi_agent.py.snippet",
    "retrieval_pipeline": "rag.py.snippet",
    "rag_rerank": "rag_rerank.py.snippet",
    "guarded_rag": "guarded_rag.py.snippet",
}


def _generate_domain_content_prompt(
    company_name: str,
    industry_or_use_case: str,
    framework: str,
    agent_architecture: str,
    tools: Optional[list[dict[str, str]]] = None,
    additional_context: Optional[str] = None,
) -> str:
    """Generate a prompt to create domain-specific content using an LLM."""
    tools_section = ""
    if tools:
        tools_lines = [f"- {t.get('name', 'unnamed')}: {t.get('description', '')}" for t in tools]
        tools_section = f"\n\nUser-specified tools to include:\n" + "\n".join(tools_lines)

    return f"""Generate domain-specific content for a synthetic Arize demo for the following company and use case.

Company: {company_name}
Industry/Use Case: {industry_or_use_case}
Framework: {framework}
Architecture: {agent_architecture}
{f"Additional Context: {additional_context}" if additional_context else ""}
{tools_section}

Generate the following as a JSON object:

1. "query_bank": An array of 10-15 realistic user queries with expected answers. Each entry should have:
   - "question": The user's question/request
   - "answer": The expected agent response
   - "tool_calls": Array of tool calls this query would trigger, each with "name", "input" (object), and "output" (object)
   - "complexity": "simple" | "aggregation" | "multi_hop"

2. "ambiguous_queries": Array of 2-3 ambiguous queries with:
   - "question": The ambiguous question
   - "clarification_question": What the agent would ask to clarify
   - "clarified_question": The user's clarified question
   - "ambiguity_reason": Why it was ambiguous
   - "answer": The final answer

3. "no_llm_queries": Array of 2-3 trivial queries that don't need LLM processing:
   - "question": The simple question
   - "answer": The direct answer

4. "tools": Array of 4-6 tools with:
   - "name": Tool function name (snake_case)
   - "description": One-line description
   - "parameters": JSON schema for parameters

5. "prompt_templates": Object with 1-2 named prompt templates:
   - Key is template name (e.g., "main_prompt", "synthesis_prompt")
   - Value has "template", "system_message", "version"

6. "agent_name": A descriptive agent name in snake_case (e.g., "fraud_triage_agent", "claims_processor")

Make the content realistic and specific to the {industry_or_use_case} domain. Use industry-specific terminology, realistic data formats, and believable scenarios.

Return ONLY valid JSON, no markdown formatting or explanation."""


def _parse_llm_json_response(response_text: str) -> dict:
    """Parse JSON from LLM response, handling common formatting issues."""
    text = response_text.strip()
    
    # Remove markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Find the first line that's just ``` or ```json
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("```"):
                start_idx = i + 1
                break
        # Find the closing ```
        end_idx = len(lines)
        for i in range(len(lines) - 1, start_idx, -1):
            if lines[i].strip() == "```":
                end_idx = i
                break
        text = "\n".join(lines[start_idx:end_idx])
    
    return json.loads(text)


def generate_domain_content(
    company_name: str,
    industry_or_use_case: str,
    framework: str,
    agent_architecture: str,
    tools: Optional[list[dict[str, str]]] = None,
    additional_context: Optional[str] = None,
) -> dict[str, Any]:
    """Use an LLM to generate domain-specific content for the demo."""
    from openai_compat_completion import completion as llm_completion

    prompt = _generate_domain_content_prompt(
        company_name=company_name,
        industry_or_use_case=industry_or_use_case,
        framework=framework,
        agent_architecture=agent_architecture,
        tools=tools,
        additional_context=additional_context,
    )

    model = os.environ.get("DEMO_GENERATOR_MODEL", "claude-sonnet-4-20250514")
    
    response = llm_completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
        temperature=0.7,
    )

    response_text = response.choices[0].message.content.strip()
    return _parse_llm_json_response(response_text)


def _format_query_bank(query_bank: list[dict]) -> str:
    """Format query bank as Python code."""
    if not query_bank:
        return "[]"
    
    lines = ["["]
    for q in query_bank:
        lines.append("    {")
        lines.append(f'        "question": {json.dumps(q.get("question", ""))},')
        lines.append(f'        "answer": {json.dumps(q.get("answer", ""))},')
        
        tool_calls = q.get("tool_calls", [])
        if tool_calls:
            lines.append('        "tool_calls": [')
            for tc in tool_calls:
                tc_str = json.dumps(tc, indent=12)
                # Fix indentation
                tc_lines = tc_str.split("\n")
                tc_lines = [tc_lines[0]] + ["            " + l.strip() for l in tc_lines[1:]]
                lines.append("            " + ",\n".join(tc_lines).strip() + ",")
            lines.append("        ],")
        else:
            lines.append('        "tool_calls": [],')
        
        lines.append(f'        "complexity": {json.dumps(q.get("complexity", "simple"))},')
        lines.append("    },")
    lines.append("]")
    return "\n".join(lines)


def _format_ambiguous_queries(queries: list[dict]) -> str:
    """Format ambiguous queries as Python code."""
    if not queries:
        return "[]"
    
    lines = ["["]
    for q in queries:
        lines.append("    {")
        lines.append(f'        "question": {json.dumps(q.get("question", ""))},')
        lines.append(f'        "clarification_question": {json.dumps(q.get("clarification_question", ""))},')
        lines.append(f'        "clarified_question": {json.dumps(q.get("clarified_question", ""))},')
        lines.append(f'        "ambiguity_reason": {json.dumps(q.get("ambiguity_reason", ""))},')
        lines.append(f'        "answer": {json.dumps(q.get("answer", ""))},')
        lines.append("    },")
    lines.append("]")
    return "\n".join(lines)


def _format_no_llm_queries(queries: list[dict]) -> str:
    """Format no-LLM queries as Python code."""
    if not queries:
        return "[]"
    
    lines = ["["]
    for q in queries:
        lines.append("    {")
        lines.append(f'        "question": {json.dumps(q.get("question", ""))},')
        lines.append(f'        "answer": {json.dumps(q.get("answer", ""))},')
        lines.append("    },")
    lines.append("]")
    return "\n".join(lines)


def _format_tools(tools: list[dict]) -> str:
    """Format tools as Python code."""
    if not tools:
        return "{}"
    
    lines = ["{"]
    for t in tools:
        name = t.get("name", "unnamed_tool")
        lines.append(f'    "{name}": {{')
        lines.append(f'        "name": {json.dumps(name)},')
        lines.append(f'        "description": {json.dumps(t.get("description", ""))},')
        
        params = t.get("parameters", {})
        params_str = json.dumps(params, indent=8)
        params_lines = params_str.split("\n")
        params_formatted = params_lines[0]
        if len(params_lines) > 1:
            params_formatted += "\n" + "\n".join("        " + l.strip() for l in params_lines[1:])
        lines.append(f'        "parameters": {params_formatted},')
        lines.append("    },")
    lines.append("}")
    return "\n".join(lines)


def _format_prompt_templates(templates: dict) -> str:
    """Format prompt templates as Python code."""
    if not templates:
        return "{}"
    
    lines = ["{"]
    for name, t in templates.items():
        lines.append(f'    "{name}": {{')
        lines.append(f'        "template": {json.dumps(t.get("template", ""))},')
        lines.append(f'        "system_message": {json.dumps(t.get("system_message", ""))},')
        lines.append(f'        "version": {json.dumps(t.get("version", "v1.0"))},')
        lines.append("    },")
    lines.append("}")
    return "\n".join(lines)


def _read_template_file(filename: str, subdir: Optional[str] = None) -> str:
    """Read a template file from the skill templates directory."""
    if subdir:
        path = TEMPLATES_PATH / subdir / filename
    else:
        path = TEMPLATES_PATH / filename
    
    if not path.exists():
        raise FileNotFoundError(f"Template file not found: {path}")
    
    return path.read_text()


def _get_architecture_snippet(architecture: str) -> str:
    """Get the appropriate architecture snippet."""
    snippet_file = ARCHITECTURE_SNIPPETS.get(architecture, "single_agent.py.snippet")
    return _read_template_file(snippet_file, subdir="snippets")


def generate_demo_files(
    company_name: str,
    industry_or_use_case: str,
    framework: str,
    agent_architecture: str,
    num_traces: int = 500,
    tools: Optional[list[dict[str, str]]] = None,
    additional_context: Optional[str] = None,
    with_evals: bool = True,
    with_dataset_and_experiments: bool = True,
    scenarios: Optional[list[str]] = None,
) -> tuple[bytes, dict[str, Any]]:
    """
    Generate all demo files and return as a ZIP archive.
    
    Returns:
        Tuple of (zip_bytes, metadata) where metadata includes file list and generation info.
    """
    # Generate domain content using LLM
    domain_content = generate_domain_content(
        company_name=company_name,
        industry_or_use_case=industry_or_use_case,
        framework=framework,
        agent_architecture=agent_architecture,
        tools=tools,
        additional_context=additional_context,
    )
    
    # Get framework metadata
    fw_meta = FRAMEWORK_METADATA.get(framework, FRAMEWORK_METADATA["generic"])
    
    # Generate slugs and names
    company_slug = slugify(company_name)
    use_case_slug = slugify(industry_or_use_case)[:30]
    project_name = f"{company_slug}_{use_case_slug}_synthetic"
    dataset_name = f"{company_slug}_eval_set"
    agent_name = domain_content.get("agent_name", f"{company_slug}_agent")
    
    # Get the first prompt template name for prompt hub
    prompt_templates = domain_content.get("prompt_templates", {})
    first_template_name = next(iter(prompt_templates.keys()), "main_prompt")
    prompt_name = f"{company_slug}_{first_template_name}"
    
    # Read the skeleton template
    skeleton = _read_template_file("generator_skeleton.py")
    
    # Get the architecture snippet
    arch_snippet = _get_architecture_snippet(agent_architecture)
    
    # Replace template variables
    generator_py = skeleton
    
    # Replace metadata placeholders
    generator_py = generator_py.replace("{{COMPANY_NAME}}", company_name)
    generator_py = generator_py.replace("{{USE_CASE}}", industry_or_use_case)
    generator_py = generator_py.replace("{{ARCHITECTURE}}", agent_architecture)
    generator_py = generator_py.replace("{{FRAMEWORK}}", framework)
    generator_py = generator_py.replace("{{LLM_MODEL}}", fw_meta["model"])
    generator_py = generator_py.replace("{{LLM_PROVIDER}}", fw_meta["provider"])
    generator_py = generator_py.replace("{{LLM_SYSTEM}}", fw_meta["system"])
    generator_py = generator_py.replace("{{LLM_SPAN_NAME}}", fw_meta["span_name"])
    generator_py = generator_py.replace("{{PROJECT_NAME}}", project_name)
    generator_py = generator_py.replace("{{DATASET_NAME}}", dataset_name)
    generator_py = generator_py.replace("{{PROMPT_NAME}}", prompt_name)
    
    # Replace snippet placeholders in arch snippet
    arch_snippet = arch_snippet.replace("{{AGENT_NAME}}", agent_name)
    arch_snippet = arch_snippet.replace("{{COMPANY_SLUG}}", company_slug)
    
    # Format domain content as Python code
    query_bank_code = _format_query_bank(domain_content.get("query_bank", []))
    ambiguous_code = _format_ambiguous_queries(domain_content.get("ambiguous_queries", []))
    no_llm_code = _format_no_llm_queries(domain_content.get("no_llm_queries", []))
    tools_code = _format_tools(domain_content.get("tools", []))
    prompts_code = _format_prompt_templates(domain_content.get("prompt_templates", {}))
    
    # Replace content placeholders
    generator_py = re.sub(
        r"QUERY_BANK = \[\n.*?\n\]",
        f"QUERY_BANK = {query_bank_code}",
        generator_py,
        flags=re.DOTALL
    )
    
    generator_py = re.sub(
        r"AMBIGUOUS_QUERIES = \[\n.*?\n\]",
        f"AMBIGUOUS_QUERIES = {ambiguous_code}",
        generator_py,
        flags=re.DOTALL
    )
    
    generator_py = re.sub(
        r"NO_LLM_QUERIES = \[\n.*?\n\]",
        f"NO_LLM_QUERIES = {no_llm_code}",
        generator_py,
        flags=re.DOTALL
    )
    
    generator_py = re.sub(
        r"TOOLS = \{\n.*?\n\}",
        f"TOOLS = {tools_code}",
        generator_py,
        flags=re.DOTALL
    )
    
    generator_py = re.sub(
        r"PROMPT_TEMPLATES = \{\n.*?\n\}",
        f"PROMPT_TEMPLATES = {prompts_code}",
        generator_py,
        flags=re.DOTALL
    )
    
    # Insert architecture snippet (replace the placeholder comment)
    generator_py = re.sub(
        r"# ─── Architecture-specific span emitters ─────────────────────────────────────\n# <<<FILL:.*?>>>",
        f"# ─── Architecture-specific span emitters ({agent_architecture}) ───────────────────────────\n\n{arch_snippet}",
        generator_py,
        flags=re.DOTALL
    )
    
    # Fix remaining <<<FILL:...>>> placeholders
    generator_py = re.sub(r"<<<FILL:.*?>>>", fw_meta["model"], generator_py)
    
    # Generate README
    readme = f"""# {company_name} — {industry_or_use_case} Synthetic Arize Demo

Generated by the arize-synthetic-demo skill via the SA Call Analyzer Demo Builder.

Architecture: {agent_architecture}
Framework: {framework}

## Quick Start

```bash
# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure credentials
cp .env.example .env
# Edit .env with your ARIZE_SPACE_ID and ARIZE_API_KEY

# Export environment variables
export $(grep -v '^#' .env | xargs)

# Run a test trace
python generator.py --test

# Generate full batch
python generator.py --count {num_traces} {"--with-evals" if with_evals else ""}

# OR: Full pipeline (traces + evals + dataset + experiments + prompts)
python generator.py --full --count {num_traces}
```

## What to Explore in Arize

**Project (`{project_name}`):**
- Trace tree visualization
- Filter by `llm.prompt_template.version` to compare v1.0 vs v2.0
- Session view for multi-turn conversations
- Error spans (tool failures, execution failures)

**Datasets (`{dataset_name}`):**
- 3-column evaluation set built from query bank
- `user_input` / `prompt_variables` / `expected_output`

**Experiments:**
- 4-cell grid: 2 models × 2 prompt versions
- Sort by `eval.correctness.score` to compare

**Prompts (`{prompt_name}`):**
- Editable prompt templates in Prompt Hub
- Attach dataset for Playground testing

## Regenerating

```bash
# Different count or seed
python generator.py --count 1000 --seed 42

# Different project name
python generator.py --count 500 --project-name {company_slug}_v2
```

## Files

- `generator.py` - Main trace generator
- `requirements.txt` - Python dependencies
- `.env.example` - Environment variable template
- `.gitignore` - Git ignore rules

## Agent Details

- **Agent Name:** {agent_name}
- **Architecture:** {agent_architecture}
- **Framework:** {framework}
- **LLM Model:** {fw_meta["model"]}
"""

    # Read static template files
    requirements_txt = _read_template_file("requirements.txt")
    env_example = _read_template_file("env.example")
    gitignore = _read_template_file("gitignore")
    
    # Create ZIP archive
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        folder_name = f"{company_slug}_{use_case_slug}_demo"
        
        zf.writestr(f"{folder_name}/generator.py", generator_py)
        zf.writestr(f"{folder_name}/README.md", readme)
        zf.writestr(f"{folder_name}/requirements.txt", requirements_txt)
        zf.writestr(f"{folder_name}/.env.example", env_example)
        zf.writestr(f"{folder_name}/.gitignore", gitignore)
    
    zip_bytes = zip_buffer.getvalue()
    
    metadata = {
        "folder_name": folder_name,
        "project_name": project_name,
        "dataset_name": dataset_name,
        "prompt_name": prompt_name,
        "agent_name": agent_name,
        "framework": framework,
        "architecture": agent_architecture,
        "llm_model": fw_meta["model"],
        "files": [
            f"{folder_name}/generator.py",
            f"{folder_name}/README.md",
            f"{folder_name}/requirements.txt",
            f"{folder_name}/.env.example",
            f"{folder_name}/.gitignore",
        ],
        "domain_content": {
            "query_count": len(domain_content.get("query_bank", [])),
            "tool_count": len(domain_content.get("tools", [])),
            "template_count": len(domain_content.get("prompt_templates", {})),
        },
    }
    
    return zip_bytes, metadata
