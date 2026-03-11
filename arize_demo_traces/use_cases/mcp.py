"""MCP (Model Context Protocol) use-case: shared prompts, queries, tools, guardrails."""

import random

from .common import industry_key

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

# Industry-tailored MCP-style tasks for prospect-aware demos
INDUSTRY_QUERIES: dict[str, list[str]] = {
    "default": QUERIES,
    "technology": QUERIES,
    "financial": [
        "Query the trading activity database for accounts with margin usage over threshold and draft a risk summary.",
        "Search our Slack channels for discussions about the Q4 compliance audit and compile key decisions.",
        "Pull the latest transaction and fee data from our CRM and generate a revenue forecast.",
        "List all open support tickets labeled 'dispute' and prioritize them by amount.",
        "Check our fraud monitoring dashboard for any alerts in the last 24 hours and create a status report.",
        "Search the internal wiki for our regulatory reporting runbook and summarize the submission procedure.",
        "Query the customer database for accounts with high cash balance and draft an engagement summary.",
        "Find all accounts that opted in to options trading in the last 30 days and summarize adoption metrics.",
    ],
    "travel": [
        "Query the reservations database for flights with delays over 2 hours and draft a compensation summary.",
        "Search our Slack channels for discussions about the holiday schedule and compile crew coverage decisions.",
        "Pull the latest booking and cancellation data and generate a demand forecast.",
        "List all open customer complaints labeled 'baggage' and prioritize by loyalty tier.",
        "Check our operations dashboard for any irregular ops in the last 24 hours and create a status report.",
        "Search the internal wiki for our rebooking runbook and summarize the waiver procedure.",
        "Query the loyalty database for members nearing tier status and draft an engagement summary.",
        "Find all reward bookings in the next 90 days and summarize availability by route.",
    ],
    "retail": [
        "Query the inventory database for SKUs with stockouts and draft a replenishment summary.",
        "Search our Slack channels for discussions about the holiday promotion and compile key decisions.",
        "Pull the latest sales data from our POS and generate a store performance forecast.",
        "List all open orders with fulfillment delays and prioritize by customer tier.",
        "Check our supply chain dashboard for any disruptions and create a status report.",
        "Search the internal wiki for our return policy runbook and summarize the exception procedure.",
    ],
    "healthcare": [
        "Query the patient records system for pending data requests and draft a compliance summary.",
        "Search our Slack channels for discussions about the clinical rollout and compile key decisions.",
        "Pull the latest quality metrics and generate a readmission forecast.",
        "List all open incidents labeled 'critical' and prioritize by facility.",
        "Check our clinical dashboard for any alerts in the last 24 hours and create a status report.",
        "Search the internal wiki for our HIPAA breach runbook and summarize the notification procedure.",
    ],
}


def get_queries_for_prospect(prospect_context: dict | None) -> list[str]:
    """Return MCP-style tasks tailored to the prospect's industry."""
    if not prospect_context:
        return INDUSTRY_QUERIES["default"]
    key = industry_key(prospect_context.get("industry"))
    return INDUSTRY_QUERIES.get(key, INDUSTRY_QUERIES["default"])


def sample_query(prospect_context=None, rng=None, **kwargs):
    """Contract: sample_query(prospect_context, rng) -> str. Industry-aware sampling."""
    queries = get_queries_for_prospect(prospect_context)
    if rng is not None:
        return rng.choice(queries)
    return random.choice(queries)

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


def get_tool_results(num_tools: int = 2) -> list[dict]:
    """Return a random set of MCP tool call results."""
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
