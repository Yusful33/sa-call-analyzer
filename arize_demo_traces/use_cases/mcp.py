"""MCP (Model Context Protocol) use-case: shared prompts, queries, tools, guardrails."""

import random

QUERIES = [
    "Find all Python files in the project that import requests and summarize their purpose.",
    "Query the customer database for accounts with revenue over $1M and draft a summary report.",
    "Search our Slack channels for discussions about the Q4 launch and compile key decisions.",
    "Read the latest PR comments on the authentication feature branch and summarize feedback.",
    "Check our monitoring dashboard for any alerts in the last 24 hours and create a status report.",
    "List all open GitHub issues labeled 'bug' and prioritize them by severity.",
    "Pull the latest sales data from our CRM and generate a pipeline forecast.",
    "Search the internal wiki for our deployment runbook and summarize the rollback procedure.",
]

# Simulated MCP servers and their tools
MCP_SERVERS = [
    {
        "name": "filesystem",
        "description": "File system access server",
        "tools": ["list_files", "read_file", "search_files"],
    },
    {
        "name": "database",
        "description": "Database query server",
        "tools": ["query", "list_tables", "describe_table"],
    },
    {
        "name": "slack",
        "description": "Slack workspace server",
        "tools": ["search_messages", "list_channels", "get_channel_history"],
    },
    {
        "name": "github",
        "description": "GitHub repository server",
        "tools": ["list_prs", "get_pr_comments", "list_issues", "get_file_contents"],
    },
]

# Simulated tool call results
TOOL_RESULTS = {
    "list_files": [
        {"path": "src/api/client.py", "size": 4200},
        {"path": "src/utils/http.py", "size": 1800},
        {"path": "tests/test_client.py", "size": 3100},
    ],
    "read_file": "import requests\n\nclass APIClient:\n    def __init__(self, base_url):\n        self.session = requests.Session()\n    ...",
    "search_files": [
        {"path": "src/api/client.py", "matches": 3},
        {"path": "src/utils/http.py", "matches": 1},
    ],
    "query": [
        {"account": "Acme Corp", "revenue": 2400000, "tier": "Enterprise"},
        {"account": "TechStart Inc", "revenue": 1200000, "tier": "Growth"},
        {"account": "DataFlow AI", "revenue": 1800000, "tier": "Enterprise"},
    ],
    "list_tables": ["accounts", "opportunities", "contacts", "activities"],
    "search_messages": [
        {"channel": "#q4-launch", "author": "PM Lead", "text": "Final launch date confirmed for Dec 15..."},
        {"channel": "#q4-launch", "author": "Eng Lead", "text": "All feature branches merged, starting QA..."},
    ],
    "list_prs": [
        {"number": 142, "title": "Add OAuth2 support", "status": "open"},
        {"number": 138, "title": "Fix session timeout", "status": "merged"},
    ],
    "get_pr_comments": [
        {"author": "reviewer1", "body": "Consider adding rate limiting to the token refresh endpoint."},
        {"author": "reviewer2", "body": "LGTM, but please add integration tests for the OAuth flow."},
    ],
    "list_issues": [
        {"number": 87, "title": "Login fails on mobile", "labels": ["bug", "P1"]},
        {"number": 92, "title": "Slow dashboard load", "labels": ["bug", "P2"]},
    ],
}

SYSTEM_PROMPT_DISCOVER = (
    "You are an MCP (Model Context Protocol) agent. Given the user's request, "
    "determine which MCP servers and tools are relevant. List the servers and "
    "specific tools you would connect to. Be concise."
)

SYSTEM_PROMPT_PLAN = (
    "You are an MCP agent planning tool execution. Given the user's request and "
    "the available MCP tools, create a step-by-step execution plan. Specify which "
    "tools to call, in what order, and what parameters to pass. Be specific and concise."
)

SYSTEM_PROMPT_SYNTHESIZE = (
    "You are an MCP agent synthesizing results. Given the user's original request "
    "and the results from MCP tool calls, produce a clear, well-structured response "
    "that directly answers the user's question. Include relevant details from the "
    "tool results."
)

GUARDRAILS = [
    {
        "name": "Authorization Check",
        "system_prompt": (
            "Check if this request requires elevated permissions or accesses "
            "sensitive resources that need authorization. "
            "Respond ONLY 'PASS' or 'FAIL: <reason>'."
        ),
    },
]

EVALUATORS = [
    {
        "name": "tool_selection_evaluation",
        "criteria": "tool selection accuracy — whether the correct MCP servers and tools were identified",
    },
    {
        "name": "synthesis_quality_evaluation",
        "criteria": "synthesis quality — whether the response accurately reflects the tool results",
    },
]


def get_simulated_tool_results(num_tools: int = 2) -> list[dict]:
    """Return a random set of simulated MCP tool call results."""
    available_tools = list(TOOL_RESULTS.keys())
    selected = random.sample(available_tools, min(num_tools, len(available_tools)))
    return [
        {"tool": tool, "server": _tool_to_server(tool), "result": TOOL_RESULTS[tool]}
        for tool in selected
    ]


def _tool_to_server(tool_name: str) -> str:
    """Map a tool name to its MCP server."""
    for server in MCP_SERVERS:
        if tool_name in server["tools"]:
            return server["name"]
    return "unknown"
