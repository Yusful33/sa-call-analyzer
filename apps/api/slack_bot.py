"""Slack bot integration for Stillness.

Provides command parsing, account resolution, and result formatting
for all Stillness capabilities accessible via Slack.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Coroutine

import httpx

if TYPE_CHECKING:
    from slack_sdk.web.async_client import AsyncWebClient

logger = logging.getLogger("stillness.slack")

STILLNESS_ANALYSIS_API_URL = os.getenv("STILLNESS_ANALYSIS_API_URL", "").rstrip("/")


class CommandType(Enum):
    """Supported Slack bot commands."""

    ANALYZE = "analyze"
    PROSPECT_OVERVIEW = "prospect_overview"
    PROSPECT_TIMELINE = "prospect_timeline"
    HYPOTHESIS = "hypothesis"
    CLASSIFY_DEMO = "classify_demo"
    GENERATE_POC = "generate_poc"
    GENERATE_RECAP = "generate_recap"
    CALLS = "calls"
    HELP = "help"
    UNKNOWN = "unknown"


@dataclass
class ParsedCommand:
    """Parsed user command with extracted arguments."""

    command_type: CommandType
    raw_text: str
    args: dict[str, Any]
    error: str | None = None


@dataclass
class SlackContext:
    """Context for a Slack interaction."""

    channel_id: str
    user_id: str
    thread_ts: str | None
    message_ts: str | None
    response_url: str | None = None


COMMAND_PATTERNS: list[tuple[re.Pattern, CommandType, list[str]]] = [
    (re.compile(r"^analyze\s+(.+)", re.IGNORECASE), CommandType.ANALYZE, ["input"]),
    (
        re.compile(r"^prospect\s+overview\s+(.+)", re.IGNORECASE),
        CommandType.PROSPECT_OVERVIEW,
        ["account"],
    ),
    (
        re.compile(r"^prospect\s+timeline\s+(.+)", re.IGNORECASE),
        CommandType.PROSPECT_TIMELINE,
        ["account"],
    ),
    (re.compile(r"^hypothesis\s+(.+)", re.IGNORECASE), CommandType.HYPOTHESIS, ["account"]),
    (
        re.compile(r"^classify\s+demo\s+(.+)", re.IGNORECASE),
        CommandType.CLASSIFY_DEMO,
        ["account"],
    ),
    (
        re.compile(r"^generate\s+poc\s+(saas|vpc|pot)\s+(.+)", re.IGNORECASE),
        CommandType.GENERATE_POC,
        ["poc_type", "account"],
    ),
    (re.compile(r"^generate\s+recap\s+(.+)", re.IGNORECASE), CommandType.GENERATE_RECAP, ["data"]),
    (re.compile(r"^calls\s+(.+)", re.IGNORECASE), CommandType.CALLS, ["account"]),
    (re.compile(r"^help\s*$", re.IGNORECASE), CommandType.HELP, []),
]


def parse_command(text: str) -> ParsedCommand:
    """Parse user input into a structured command."""
    text = text.strip()

    if not text:
        return ParsedCommand(
            command_type=CommandType.HELP,
            raw_text=text,
            args={},
        )

    for pattern, cmd_type, arg_names in COMMAND_PATTERNS:
        match = pattern.match(text)
        if match:
            groups = match.groups()
            args = {name: groups[i].strip() if i < len(groups) else None for i, name in enumerate(arg_names)}
            return ParsedCommand(command_type=cmd_type, raw_text=text, args=args)

    return ParsedCommand(
        command_type=CommandType.UNKNOWN,
        raw_text=text,
        args={},
        error=f"Unknown command: {text.split()[0] if text.split() else text}",
    )


HELP_TEXT = """*Stillness Bot Commands*

*Call Analysis*
• `analyze <gong_url>` - Analyze a Gong call
• `analyze <transcript>` - Analyze pasted transcript text

*Prospect Intelligence*
• `prospect overview <account>` - Get comprehensive prospect overview
• `prospect timeline <account>` - Analyze all calls for a prospect
• `hypothesis <account>` - Research hypotheses for an account
• `calls <account>` - List recent calls for an account

*Demo & Deliverables*
• `classify demo <account>` - Classify demo signals for synthetic demo generation
• `generate poc <saas|vpc|pot> <account>` - Generate PoC/PoT document

*Other*
• `help` - Show this help message

*Account Lookup*
You can specify accounts by name, domain, or Salesforce ID. The bot will fuzzy-match to find the best match.

*Examples*
• `/stillness analyze https://app.gong.io/call?id=123456`
• `/stillness prospect overview FINRA`
• `/stillness hypothesis acme.com`
• `@Stillness calls Snowflake`
"""


async def resolve_account(
    account_input: str,
    bq_client: Any,
) -> dict[str, Any] | None:
    """Resolve account input (name, domain, or SFDC ID) to account details.

    Uses the account-suggestions endpoint for fuzzy matching.
    """
    if not bq_client:
        return {"account_name": account_input, "resolved": False}

    try:
        suggestions = await asyncio.to_thread(
            bq_client.get_account_suggestions,
            account_input,
            limit=1,
        )
        if suggestions:
            top = suggestions[0]
            return {
                "account_name": top.get("account_name"),
                "domain": top.get("domain"),
                "sfdc_account_id": top.get("sfdc_account_id"),
                "resolved": True,
            }
    except Exception as e:
        logger.warning("Account resolution failed for %s: %s", account_input, e)

    return {"account_name": account_input, "resolved": False}


def format_prospect_overview(data: dict[str, Any]) -> str:
    """Format prospect overview response for Slack."""
    lines = [f"*Prospect Overview: {data.get('account_name', 'Unknown')}*\n"]

    if sfdc := data.get("salesforce"):
        lines.append("*Salesforce*")
        if acc := sfdc.get("account"):
            lines.append(f"• Industry: {acc.get('industry', 'N/A')}")
            lines.append(f"• Type: {acc.get('type', 'N/A')}")
            lines.append(f"• Owner: {acc.get('owner_name', 'N/A')}")
        if opps := sfdc.get("opportunities"):
            lines.append(f"• Open Opportunities: {len(opps)}")
        lines.append("")

    if gong := data.get("gong"):
        lines.append("*Gong Activity*")
        if calls := gong.get("recent_calls"):
            lines.append(f"• Recent Calls: {len(calls)}")
        if summary := gong.get("deal_summary"):
            lines.append(f"• Deal Stage: {summary.get('stage', 'N/A')}")
        lines.append("")

    if pendo := data.get("pendo"):
        lines.append("*Product Usage (Pendo)*")
        if usage := pendo.get("usage_summary"):
            lines.append(f"• Active Users: {usage.get('active_users', 'N/A')}")
            lines.append(f"• Sessions (30d): {usage.get('sessions_30d', 'N/A')}")
        lines.append("")

    return "\n".join(lines)


def format_calls_list(data: dict[str, Any]) -> str:
    """Format calls-by-account response for Slack."""
    calls = data.get("calls", [])
    account = data.get("account_name", "Unknown")

    if not calls:
        return f"No calls found for *{account}*"

    lines = [f"*Recent Calls for {account}* ({len(calls)} found)\n"]
    for call in calls[:10]:
        date = call.get("date", "Unknown date")
        title = call.get("title", "Untitled")
        duration = call.get("duration_minutes", "?")
        gong_url = call.get("gong_url", "")
        line = f"• {date}: {title} ({duration}m)"
        if gong_url:
            line += f" - <{gong_url}|View>"
        lines.append(line)

    if len(calls) > 10:
        lines.append(f"\n_...and {len(calls) - 10} more_")

    return "\n".join(lines)


def format_hypothesis_result(data: dict[str, Any]) -> str:
    """Format hypothesis research response for Slack."""
    lines = [f"*Hypothesis Research: {data.get('account_name', 'Unknown')}*\n"]

    if hypotheses := data.get("hypotheses", []):
        for i, h in enumerate(hypotheses[:5], 1):
            lines.append(f"*{i}. {h.get('title', 'Untitled')}*")
            if desc := h.get("description"):
                lines.append(f"   {desc[:200]}{'...' if len(desc) > 200 else ''}")
            if evidence := h.get("evidence"):
                lines.append(f"   _Evidence: {evidence[:150]}{'...' if len(evidence) > 150 else ''}_")
            lines.append("")

    if not hypotheses:
        lines.append("_No hypotheses generated. Try providing more context about the account._")

    return "\n".join(lines)


def format_classify_demo_result(data: dict[str, Any]) -> str:
    """Format demo classification response for Slack."""
    lines = [f"*Demo Classification: {data.get('account_name', 'Unknown')}*\n"]

    if classification := data.get("classification"):
        lines.append(f"*Primary Use Case:* {classification.get('primary_use_case', 'Unknown')}")
        lines.append(f"*Industry:* {classification.get('industry', 'Unknown')}")
        lines.append(f"*Persona:* {classification.get('persona', 'Unknown')}")
        lines.append(f"*Confidence:* {classification.get('confidence', 'N/A')}")

    if prompt := data.get("skill_prompt"):
        lines.append("\n*Generated Skill Prompt:*")
        lines.append(f"```{prompt[:500]}{'...' if len(prompt) > 500 else ''}```")

    return "\n".join(lines)


def format_analysis_result(data: dict[str, Any]) -> str:
    """Format call analysis response for Slack."""
    lines = ["*Call Analysis Complete*\n"]

    if summary := data.get("summary"):
        lines.append(f"*Summary:* {summary}\n")

    if score := data.get("overall_score"):
        lines.append(f"*Overall Score:* {score}/100\n")

    if strengths := data.get("strengths"):
        lines.append("*Strengths:*")
        for s in strengths[:3]:
            lines.append(f"• {s}")
        lines.append("")

    if improvements := data.get("improvements"):
        lines.append("*Areas for Improvement:*")
        for imp in improvements[:3]:
            lines.append(f"• {imp}")
        lines.append("")

    if insights := data.get("insights"):
        lines.append("*Key Insights:*")
        for ins in insights[:5]:
            ts = ins.get("timestamp", "")
            text = ins.get("insight", ins.get("text", ""))
            lines.append(f"• [{ts}] {text}")

    return "\n".join(lines)


def format_error(error: str, command: str | None = None) -> str:
    """Format an error message for Slack."""
    lines = [":warning: *Error*", f"_{error}_"]
    if command:
        lines.append(f"\nCommand: `{command}`")
    lines.append("\nType `help` for available commands.")
    return "\n".join(lines)


class SlackBot:
    """Slack bot for Stillness capabilities."""

    def __init__(
        self,
        bq_client: Any = None,
        gong_client: Any = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.bq_client = bq_client
        self.gong_client = gong_client
        self._http_client = http_client
        self._owned_http_client = False

    async def get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client for external API calls."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=600.0)
            self._owned_http_client = True
        return self._http_client

    async def close(self):
        """Clean up resources."""
        if self._owned_http_client and self._http_client:
            await self._http_client.aclose()

    async def handle_command(
        self,
        text: str,
        context: SlackContext,
        slack_client: "AsyncWebClient",
    ) -> str:
        """Handle a parsed command and return the response text."""
        parsed = parse_command(text)

        if parsed.command_type == CommandType.HELP:
            return HELP_TEXT

        if parsed.command_type == CommandType.UNKNOWN:
            return format_error(parsed.error or "Unknown command", parsed.raw_text)

        handler = self._get_handler(parsed.command_type)
        if not handler:
            return format_error(f"Handler not implemented for {parsed.command_type.value}")

        try:
            return await handler(parsed, context)
        except Exception as e:
            logger.exception("Command handler failed: %s", e)
            return format_error(str(e), parsed.raw_text)

    def _get_handler(
        self, cmd_type: CommandType
    ) -> Callable[[ParsedCommand, SlackContext], Coroutine[Any, Any, str]] | None:
        """Get the handler function for a command type."""
        handlers: dict[CommandType, Callable] = {
            CommandType.ANALYZE: self._handle_analyze,
            CommandType.PROSPECT_OVERVIEW: self._handle_prospect_overview,
            CommandType.PROSPECT_TIMELINE: self._handle_prospect_timeline,
            CommandType.HYPOTHESIS: self._handle_hypothesis,
            CommandType.CLASSIFY_DEMO: self._handle_classify_demo,
            CommandType.GENERATE_POC: self._handle_generate_poc,
            CommandType.CALLS: self._handle_calls,
        }
        return handlers.get(cmd_type)

    async def _handle_analyze(self, cmd: ParsedCommand, ctx: SlackContext) -> str:
        """Handle call analysis command."""
        input_text = cmd.args.get("input", "")

        if not STILLNESS_ANALYSIS_API_URL:
            return format_error(
                "Analysis API not configured. Set STILLNESS_ANALYSIS_API_URL.",
                cmd.raw_text,
            )

        http = await self.get_http_client()

        is_gong_url = "gong.io" in input_text.lower()
        payload = {"gong_url": input_text} if is_gong_url else {"transcript": input_text}

        try:
            resp = await http.post(
                f"{STILLNESS_ANALYSIS_API_URL}/api/analyze",
                json=payload,
                timeout=600.0,
            )
            resp.raise_for_status()
            return format_analysis_result(resp.json())
        except httpx.TimeoutException:
            return format_error("Analysis timed out. The call may be too long.", cmd.raw_text)
        except httpx.HTTPStatusError as e:
            return format_error(f"Analysis failed: {e.response.status_code}", cmd.raw_text)

    async def _handle_prospect_overview(self, cmd: ParsedCommand, ctx: SlackContext) -> str:
        """Handle prospect overview command."""
        account_input = cmd.args.get("account", "")
        resolved = await resolve_account(account_input, self.bq_client)

        if not self.bq_client:
            return format_error("BigQuery not configured for prospect overview.", cmd.raw_text)

        try:
            from bigquery_client import BigQueryClient

            overview = await asyncio.to_thread(
                self.bq_client.get_prospect_overview,
                account_name=resolved.get("account_name"),
                domain=resolved.get("domain"),
                sfdc_account_id=resolved.get("sfdc_account_id"),
            )
            return format_prospect_overview(overview)
        except Exception as e:
            logger.exception("Prospect overview failed: %s", e)
            return format_error(f"Failed to get prospect overview: {e}", cmd.raw_text)

    async def _handle_prospect_timeline(self, cmd: ParsedCommand, ctx: SlackContext) -> str:
        """Handle prospect timeline command."""
        account_input = cmd.args.get("account", "")

        if not STILLNESS_ANALYSIS_API_URL:
            return format_error(
                "Analysis API not configured. Set STILLNESS_ANALYSIS_API_URL.",
                cmd.raw_text,
            )

        resolved = await resolve_account(account_input, self.bq_client)
        http = await self.get_http_client()

        try:
            resp = await http.post(
                f"{STILLNESS_ANALYSIS_API_URL}/api/analyze-prospect",
                json={"account_name": resolved.get("account_name")},
                timeout=300.0,
            )
            resp.raise_for_status()
            data = resp.json()
            return f"*Prospect Timeline: {resolved.get('account_name')}*\n\n{data.get('summary', 'Analysis complete.')}"
        except httpx.TimeoutException:
            return format_error("Timeline analysis timed out.", cmd.raw_text)
        except httpx.HTTPStatusError as e:
            return format_error(f"Timeline analysis failed: {e.response.status_code}", cmd.raw_text)

    async def _handle_hypothesis(self, cmd: ParsedCommand, ctx: SlackContext) -> str:
        """Handle hypothesis research command."""
        account_input = cmd.args.get("account", "")
        resolved = await resolve_account(account_input, self.bq_client)

        http = await self.get_http_client()
        api_base = os.getenv("API_BASE_URL", "http://localhost:8080")

        try:
            resp = await http.post(
                f"{api_base}/api/hypothesis-research",
                json={"account_name": resolved.get("account_name")},
                timeout=120.0,
            )
            resp.raise_for_status()
            return format_hypothesis_result(resp.json())
        except Exception as e:
            logger.exception("Hypothesis research failed: %s", e)
            return format_error(f"Hypothesis research failed: {e}", cmd.raw_text)

    async def _handle_classify_demo(self, cmd: ParsedCommand, ctx: SlackContext) -> str:
        """Handle demo classification command."""
        account_input = cmd.args.get("account", "")
        resolved = await resolve_account(account_input, self.bq_client)

        http = await self.get_http_client()
        api_base = os.getenv("API_BASE_URL", "http://localhost:8080")

        try:
            resp = await http.post(
                f"{api_base}/api/classify-demo",
                json={"account_name": resolved.get("account_name")},
                timeout=60.0,
            )
            resp.raise_for_status()
            return format_classify_demo_result(resp.json())
        except Exception as e:
            logger.exception("Demo classification failed: %s", e)
            return format_error(f"Demo classification failed: {e}", cmd.raw_text)

    async def _handle_generate_poc(self, cmd: ParsedCommand, ctx: SlackContext) -> str:
        """Handle PoC/PoT document generation command."""
        poc_type = cmd.args.get("poc_type", "saas").lower()
        account_input = cmd.args.get("account", "")
        resolved = await resolve_account(account_input, self.bq_client)

        http = await self.get_http_client()
        api_base = os.getenv("API_BASE_URL", "http://localhost:8080")

        try:
            resp = await http.post(
                f"{api_base}/api/generate-poc-document",
                json={
                    "account_name": resolved.get("account_name"),
                    "poc_type": poc_type,
                },
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()

            if download_url := data.get("download_url"):
                return f"*PoC Document Generated*\n\nDownload: {download_url}"
            return f"*PoC Document Generated* for {resolved.get('account_name')}\n\n_Document generation complete. Check the web UI for download._"
        except Exception as e:
            logger.exception("PoC generation failed: %s", e)
            return format_error(f"PoC generation failed: {e}", cmd.raw_text)

    async def _handle_calls(self, cmd: ParsedCommand, ctx: SlackContext) -> str:
        """Handle calls-by-account command."""
        account_input = cmd.args.get("account", "")
        resolved = await resolve_account(account_input, self.bq_client)

        http = await self.get_http_client()
        api_base = os.getenv("API_BASE_URL", "http://localhost:8080")

        try:
            resp = await http.post(
                f"{api_base}/api/calls-by-account",
                json={"account_name": resolved.get("account_name")},
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            data["account_name"] = resolved.get("account_name")
            return format_calls_list(data)
        except Exception as e:
            logger.exception("Calls lookup failed: %s", e)
            return format_error(f"Failed to get calls: {e}", cmd.raw_text)
