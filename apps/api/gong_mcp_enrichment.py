"""
Supplement BigQuery-backed Gong context with live Gong MCP data when warehouse
Spotlight/snippets are thin.

Controlled by POC_GONG_MCP_ENRICH (default on). Uses existing GongMCPClient HTTP
paths (/calls, /transcript) against GONG_MCP_URL.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set, TYPE_CHECKING

from gong_mcp_client import GongMCPClient
from models import GongCallData, GongParticipant, GongSummaryData, ProspectOverview

if TYPE_CHECKING:
    from bigquery_client import BigQueryClient


def _enrich_enabled() -> bool:
    v = os.environ.get("POC_GONG_MCP_ENRICH", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _snip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _rich_call(c: GongCallData) -> bool:
    if (c.spotlight_brief or "").strip():
        return True
    sn = (c.transcript_snippet or "").strip()
    return len(sn) >= 120


def prospect_gong_context_is_sparse(overview: ProspectOverview) -> bool:
    """Heuristic: BigQuery Gong slice is missing enough narrative for PoC / overview."""
    if not _enrich_enabled():
        return False
    g = overview.gong_summary
    if not g or g.total_calls == 0:
        return True
    calls = g.recent_calls or []
    if not calls:
        return True
    rich = sum(1 for c in calls if _rich_call(c))
    if rich < 2:
        return True
    spotlight_hits = sum(1 for c in calls if (c.spotlight_brief or "").strip())
    if len(calls) >= 3 and spotlight_hits == 0:
        return True
    return False


def _resolve_account_search_name(overview: ProspectOverview) -> str:
    if overview.salesforce and (overview.salesforce.name or "").strip():
        return overview.salesforce.name.strip()
    if overview.lookup_method == "name" and (overview.lookup_value or "").strip():
        return overview.lookup_value.strip()
    if overview.lookup_method == "domain" and (overview.lookup_value or "").strip():
        raw = overview.lookup_value.strip().lower().replace("www.", "")
        parts = raw.split(".")
        if parts and len(parts[0]) >= 3:
            return parts[0]
        return overview.lookup_value.strip()
    return ""


def _gong_numeric_id_from_call(call: GongCallData) -> Optional[str]:
    if call.call_url:
        try:
            return GongMCPClient.extract_call_id_from_url(call.call_url)
        except ValueError:
            pass
    if call.conversation_key:
        ck = str(call.conversation_key).strip()
        if ck.isdigit():
            return ck
    return None


def _duration_minutes(raw: Dict) -> Optional[float]:
    d = raw.get("duration")
    if d is None:
        return None
    try:
        x = float(d)
    except (TypeError, ValueError):
        return None
    if x > 1_000_000:
        x = x / 1000.0
    if x > 172800:
        x = x / 60.0
    return max(x / 60.0, 0.0)


def _participants_from_mcp(raw: Dict) -> List[GongParticipant]:
    out: List[GongParticipant] = []
    for p in raw.get("parties") or []:
        if not isinstance(p, dict):
            continue
        aff = p.get("affiliation")
        out.append(
            GongParticipant(
                name=p.get("name") or p.get("speakerName"),
                email=p.get("emailAddress") or p.get("email"),
                affiliation=(str(aff).lower() if aff is not None else None),
                speaker_id=str(p.get("speakerId")) if p.get("speakerId") is not None else None,
            )
        )
    return out


def _scheduled_to_iso(scheduled) -> Optional[str]:
    if scheduled is None:
        return None
    if isinstance(scheduled, (int, float)):
        ts = scheduled / 1000.0 if scheduled > 1e12 else scheduled
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    return str(scheduled)


def _mcp_raw_to_call(raw: Dict, transcript: Optional[str], snippet_limit: int) -> GongCallData:
    gid = str(raw.get("id") or "")
    url = raw.get("url") or (f"https://app.gong.io/call?id={gid}" if gid else None)
    scheduled = raw.get("scheduled") or raw.get("started")
    call_date = _scheduled_to_iso(scheduled)
    snippet = _snip(transcript, snippet_limit) if transcript else None
    return GongCallData(
        conversation_key=gid or None,
        call_url=url,
        call_title=raw.get("title"),
        call_date=call_date,
        duration_minutes=_duration_minutes(raw),
        participants=_participants_from_mcp(raw),
        transcript_snippet=snippet,
    )


def _call_sort_key(c: GongCallData) -> str:
    return (c.call_date or "")[:26]


def maybe_enrich_overview_with_gong_mcp(
    overview: ProspectOverview,
    gong_client: Optional[GongMCPClient],
    bq_client: Optional["BigQueryClient"] = None,
) -> ProspectOverview:
    """
    When Gong warehouse context looks thin, match calls via Gong MCP and merge
    transcripts plus MCP-only calls into gong_summary, then refresh sales engagement.
    """
    if not _enrich_enabled() or not gong_client or not prospect_gong_context_is_sparse(overview):
        return overview

    search_name = _resolve_account_search_name(overview)
    if not search_name:
        return overview

    lookback = int(os.environ.get("POC_GONG_MCP_LOOKBACK_DAYS", "45"))
    # Caps Gong MCP /calls work; higher values risk read timeouts on the API→MCP HTTP hop.
    max_scan = int(os.environ.get("POC_GONG_MCP_MAX_CALLS_SCAN", "400"))
    max_tx = int(os.environ.get("POC_GONG_MCP_MAX_TRANSCRIPTS", "8"))
    snippet_limit = int(os.environ.get("POC_GONG_MCP_TRANSCRIPT_SNIPPET_CHARS", "8000"))

    now = datetime.now(timezone.utc)
    from_dt = (now - timedelta(days=lookback)).strftime("%Y-%m-%dT%H:%M:%SZ")
    to_dt = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        mcp_calls = gong_client.get_calls_by_prospect_name(
            search_name,
            from_date=from_dt,
            to_date=to_dt,
            max_calls_to_scan=max_scan,
        )
    except Exception as e:
        errs = list(overview.errors or [])
        errs.append(f"Gong MCP enrichment skipped: {e}")
        return overview.model_copy(update={"errors": errs})

    gs = overview.gong_summary
    work: List[GongCallData] = [c.model_copy(deep=True) for c in (gs.recent_calls or [])] if gs else []

    prior_gids: Set[str] = set()
    for c in work:
        gid = _gong_numeric_id_from_call(c)
        if gid:
            prior_gids.add(gid)

    added_new_calls = 0
    for raw in mcp_calls:
        if not isinstance(raw, dict):
            continue
        gid = str(raw.get("id") or "")
        if not gid or gid in prior_gids:
            continue
        work.append(_mcp_raw_to_call(raw, None, snippet_limit))
        prior_gids.add(gid)
        added_new_calls += 1

    work.sort(key=_call_sort_key, reverse=True)

    fetch_gids: List[str] = []
    for c in work:
        gid = _gong_numeric_id_from_call(c)
        if not gid:
            continue
        if (c.transcript_snippet or "").strip():
            continue
        fetch_gids.append(gid)
        if len(fetch_gids) >= max_tx:
            break

    n_transcripts_applied = 0
    if fetch_gids:
        tx_results = gong_client.get_transcripts_parallel(fetch_gids, max_workers=4)
        gid_to_idx: Dict[str, int] = {}
        for i, c in enumerate(work):
            g = _gong_numeric_id_from_call(c)
            if g:
                gid_to_idx[g] = i
        for gid, data in tx_results.items():
            if not data or gid not in gid_to_idx:
                continue
            idx = gid_to_idx[gid]
            try:
                txt = gong_client.format_transcript_for_analysis(data)
            except Exception:
                txt = ""
            sn = _snip(txt, snippet_limit)
            if sn:
                work[idx] = work[idx].model_copy(update={"transcript_snippet": sn})
                n_transcripts_applied += 1

    if added_new_calls == 0 and n_transcripts_applied == 0:
        return overview

    work.sort(key=_call_sort_key, reverse=True)
    recent_cap = int(os.environ.get("POC_GONG_MCP_RECENT_CALLS_CAP", "20"))
    recent = work[:recent_cap]

    first_cd = recent[-1].call_date if recent else None
    last_cd = recent[0].call_date if recent else None
    ds: Optional[int] = None
    if last_cd:
        try:
            clean = last_cd.replace("Z", "+00:00").split("+")[0]
            last_date = datetime.fromisoformat(clean)
            if last_date.tzinfo is None:
                last_date = last_date.replace(tzinfo=timezone.utc)
            ds = (datetime.now(timezone.utc) - last_date).days
        except Exception:
            pass

    total_dur = sum((c.duration_minutes or 0) for c in recent)
    base_total = gs.total_calls if gs else 0

    if gs:
        new_gs = gs.model_copy(
            update={
                "recent_calls": recent,
                "total_calls": max(base_total, len(work)),
                "total_duration_minutes": total_dur or gs.total_duration_minutes,
                "first_call_date": first_cd or gs.first_call_date,
                "last_call_date": last_cd or gs.last_call_date,
                "days_since_last_call": ds if ds is not None else gs.days_since_last_call,
            }
        )
    else:
        new_gs = GongSummaryData(
            total_calls=len(work),
            total_duration_minutes=total_dur,
            recent_calls=recent,
            first_call_date=first_cd,
            last_call_date=last_cd,
            days_since_last_call=ds,
        )

    sources = list(overview.data_sources_available or [])
    if "gong_mcp" not in sources:
        sources.append("gong_mcp")

    out = overview.model_copy(
        update={
            "gong_summary": new_gs,
            "data_sources_available": sources,
        }
    )
    if bq_client:
        out = bq_client.refresh_gong_dependent_fields(out)
    return out
