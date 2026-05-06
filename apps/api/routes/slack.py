"""Slack event and command routes for FastAPI.

Handles:
- /api/slack/events - Slack Events API (mentions, DMs)
- /api/slack/commands - Slash command endpoint
- /api/slack/interactions - Interactive components (future)
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request, Response

if TYPE_CHECKING:
    from slack_bot import SlackBot

logger = logging.getLogger("stillness.slack.routes")

router = APIRouter(prefix="/api/slack", tags=["slack"])

_slack_bot: "SlackBot | None" = None
_slack_client: Any = None

SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET", "")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")

_processed_events: dict[str, float] = {}
_EVENT_DEDUP_TTL = 300


def _verify_slack_signature(
    body: bytes,
    timestamp: str,
    signature: str,
) -> bool:
    """Verify the request signature from Slack."""
    if not SLACK_SIGNING_SECRET:
        logger.warning("SLACK_SIGNING_SECRET not configured, skipping verification")
        return True

    if abs(time.time() - int(timestamp)) > 60 * 5:
        logger.warning("Slack request timestamp too old")
        return False

    sig_basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
    computed = "v0=" + hmac.new(
        SLACK_SIGNING_SECRET.encode(),
        sig_basestring.encode(),
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(computed, signature)


def _is_duplicate_event(event_id: str) -> bool:
    """Check if we've already processed this event (Slack retries)."""
    now = time.time()

    for eid in list(_processed_events.keys()):
        if now - _processed_events[eid] > _EVENT_DEDUP_TTL:
            del _processed_events[eid]

    if event_id in _processed_events:
        return True

    _processed_events[event_id] = now
    return False


def configure_slack_routes(slack_bot: "SlackBot") -> APIRouter:
    """Configure the Slack routes with the bot instance."""
    global _slack_bot, _slack_client
    _slack_bot = slack_bot

    try:
        from slack_sdk.web.async_client import AsyncWebClient

        if SLACK_BOT_TOKEN:
            _slack_client = AsyncWebClient(token=SLACK_BOT_TOKEN)
            logger.info("Slack client initialized")
        else:
            logger.warning("SLACK_BOT_TOKEN not set, Slack client not initialized")
    except ImportError:
        logger.warning("slack_sdk not installed, Slack routes will not function")

    return router


async def _post_message(channel: str, text: str, thread_ts: str | None = None) -> dict | None:
    """Post a message to Slack."""
    if not _slack_client:
        logger.error("Slack client not initialized")
        return None

    try:
        return await _slack_client.chat_postMessage(
            channel=channel,
            text=text,
            thread_ts=thread_ts,
        )
    except Exception as e:
        logger.exception("Failed to post Slack message: %s", e)
        return None


async def _update_message(channel: str, ts: str, text: str) -> dict | None:
    """Update an existing Slack message."""
    if not _slack_client:
        return None

    try:
        return await _slack_client.chat_update(
            channel=channel,
            ts=ts,
            text=text,
        )
    except Exception as e:
        logger.exception("Failed to update Slack message: %s", e)
        return None


async def _process_command_background(
    text: str,
    channel_id: str,
    user_id: str,
    thread_ts: str | None,
    working_msg_ts: str | None,
):
    """Process a command in the background and post results."""
    if not _slack_bot:
        logger.error("Slack bot not initialized")
        return

    from slack_bot import SlackContext

    ctx = SlackContext(
        channel_id=channel_id,
        user_id=user_id,
        thread_ts=thread_ts,
        message_ts=working_msg_ts,
    )

    try:
        result = await _slack_bot.handle_command(text, ctx, _slack_client)

        if working_msg_ts:
            await _update_message(channel_id, working_msg_ts, result)
        else:
            await _post_message(channel_id, result, thread_ts)

    except Exception as e:
        logger.exception("Background command processing failed: %s", e)
        error_msg = f":warning: An error occurred while processing your request: {e}"
        if working_msg_ts:
            await _update_message(channel_id, working_msg_ts, error_msg)
        else:
            await _post_message(channel_id, error_msg, thread_ts)


@router.post("/events")
async def slack_events(
    request: Request,
    background_tasks: BackgroundTasks,
    x_slack_signature: str = Header(None, alias="X-Slack-Signature"),
    x_slack_request_timestamp: str = Header(None, alias="X-Slack-Request-Timestamp"),
    x_slack_retry_num: str | None = Header(None, alias="X-Slack-Retry-Num"),
):
    """Handle Slack Events API requests.

    Supports:
    - URL verification challenge
    - app_mention events
    - message.im events (DMs)
    """
    body = await request.body()

    if not _verify_slack_signature(body, x_slack_request_timestamp or "", x_slack_signature or ""):
        raise HTTPException(status_code=401, detail="Invalid signature")

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    if payload.get("type") == "url_verification":
        return {"challenge": payload.get("challenge")}

    event = payload.get("event", {})
    event_id = payload.get("event_id", "")
    event_type = event.get("type", "")

    if x_slack_retry_num:
        logger.info("Slack retry #%s for event %s", x_slack_retry_num, event_id)

    if _is_duplicate_event(event_id):
        logger.info("Skipping duplicate event: %s", event_id)
        return Response(status_code=200)

    if event.get("bot_id"):
        return Response(status_code=200)

    if event_type in ("app_mention", "message"):
        if event_type == "message" and event.get("channel_type") != "im":
            return Response(status_code=200)

        text = event.get("text", "")

        if event_type == "app_mention":
            text = " ".join(text.split()[1:])

        channel_id = event.get("channel", "")
        user_id = event.get("user", "")
        thread_ts = event.get("thread_ts") or event.get("ts")

        working_msg = await _post_message(
            channel_id,
            ":hourglass_flowing_sand: Working on your request...",
            thread_ts,
        )
        working_ts = working_msg.get("ts") if working_msg else None

        background_tasks.add_task(
            _process_command_background,
            text,
            channel_id,
            user_id,
            thread_ts,
            working_ts,
        )

    return Response(status_code=200)


@router.post("/commands")
async def slack_commands(
    request: Request,
    background_tasks: BackgroundTasks,
    x_slack_signature: str = Header(None, alias="X-Slack-Signature"),
    x_slack_request_timestamp: str = Header(None, alias="X-Slack-Request-Timestamp"),
):
    """Handle Slack slash commands (/stillness)."""
    body = await request.body()

    if not _verify_slack_signature(body, x_slack_request_timestamp or "", x_slack_signature or ""):
        raise HTTPException(status_code=401, detail="Invalid signature")

    form_data = await request.form()
    command = form_data.get("command", "")
    text = form_data.get("text", "")
    channel_id = form_data.get("channel_id", "")
    user_id = form_data.get("user_id", "")
    response_url = form_data.get("response_url", "")

    logger.info("Slash command: %s %s from user %s", command, text, user_id)

    working_msg = await _post_message(
        channel_id,
        f":hourglass_flowing_sand: Processing `{command} {text}`...",
    )
    working_ts = working_msg.get("ts") if working_msg else None

    background_tasks.add_task(
        _process_command_background,
        text,
        channel_id,
        user_id,
        None,
        working_ts,
    )

    return Response(status_code=200)


@router.post("/interactions")
async def slack_interactions(
    request: Request,
    x_slack_signature: str = Header(None, alias="X-Slack-Signature"),
    x_slack_request_timestamp: str = Header(None, alias="X-Slack-Request-Timestamp"),
):
    """Handle Slack interactive components (buttons, modals, etc.).

    Reserved for future use.
    """
    body = await request.body()

    if not _verify_slack_signature(body, x_slack_request_timestamp or "", x_slack_signature or ""):
        raise HTTPException(status_code=401, detail="Invalid signature")

    form_data = await request.form()
    payload_str = form_data.get("payload", "{}")

    try:
        payload = json.loads(payload_str)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid payload")

    logger.info("Interaction: %s", payload.get("type"))

    return Response(status_code=200)
