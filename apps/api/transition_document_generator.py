"""
Pre-sales -> post-sales (CS) transition document generator.

Pulls a populated ``ProspectOverview`` (Salesforce + Opportunities + Gong
analytics + Pendo + FullStory, plus Gong MCP enrichment in force mode) and
asks an LLM to fill in a fixed Knowledge Transfer template that the Arize
team uses to hand a closed/won account from pre-sales to Customer Success.

The structured data is broken into named, per-section blocks (account /
arize_team / contract / pipeline / deal_lifecycle / gong / product_usage /
user_behavior) plus a separate full-fat "GONG RECENT CALLS" block that
preserves Spotlight briefs, key points, next steps, outcomes, and several
KB of transcript per call so the LLM can quote them in the KT doc.

Returns markdown text suitable for rendering in the browser, copying into
Notion / Google Docs, or downloading as ``.md``.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from openai_compat_completion import completion as llm_completion

from models import GongCallData, OpportunityData, ProspectOverview


TRANSITION_TEMPLATE = """[Customer Name] - Internal Knowledge Transfer

Assigned Team:
CSE:
AE:

Executive Summary
- What do they do?
- How do they monetize?
- Strategic Initiatives?
- In general what do they care about as a company?
- Are there any big AI initiatives we should be aware of?
- Stock Price / Market Leader? / Industry Challenges?

Account Summary

Org & Stakeholders:
- Arize team
- Customer stakeholders - roles, involvement, etc.
- Identify EB, champions, promoters, detractors

Contract Entitlements:
- Current ARR
- Renewal Date
- Entitlements
- Is there a specific Case Study clause?

Use Cases:
- LLM:
- ML:
- Include names of apps if possible

Key Initiatives / Outcomes / Business Pain:

Expansion Opportunities:
- What growth opportunities have we identified?
- Do they have more Models / LLM apps in the future?
- Have we discussed growth with their executives?
- Have we asked for an introduction and/or started prospecting?
- Do they have an all DS meeting we could demo at?
- Have we positioned workshops?
- Do they have a Central AI team?
- What is our strategy to get wider? AE + SDR + CSE

POC Requirements / Info:
- Link in the POC scoping doc
- SA to add any color on the POC?
- Any roadblocks?
- Any sticking points CSE should know about (politics/how easy it was to work with the team)
- Features they really cared about, etc.

Current Tech Stack / Competition:
- Incumbent or competition details:
- Orchestration framework:
- Cloud provider:

Asks / Product Requests:
- Link to product feature tracking
- Are there any expectations on features that we need to consider in roadmap
- Which PM was involved in the sales cycle if any?

Risks to onboarding / initial value:
"""


_SYSTEM_PROMPT = """You are an Arize Solutions Architect drafting an internal Knowledge Transfer (KT) document that hands an account from pre-sales to Customer Success (CSE / CSM).

The reader is a CSE who has never spoken to this customer. Write with the confidence of an SA who lived the deal — short, declarative sentences, no hedging.

================ OUTPUT RULES ================
1. Output ONLY GitHub-flavored Markdown. No JSON, no preamble, no code fences around the whole thing.
2. Use the exact section headings from the template. Preserve their order. Each section must have content. When a value is genuinely not in the data, write "TBD".
3. Replace "[Customer Name]" in the H1 with the real account name from `account.name`.
4. Render top-level template sections as `## Section Name`. Render sub-blocks (Org & Stakeholders, Contract Entitlements, Use Cases, etc.) as `### Sub Heading`.
5. Convert each multi-question prompt in the template into its own bullet, in the form: `- **Prompt question?** Your answer.` (or `- **Short label:** your answer`).
6. Keep each bullet to 1-3 sentences. The whole document should be skimmable in under 3 minutes.
7. Quote facts verbatim when present in the structured data — names, dollar amounts, dates, stage names, opp names, feature names. NEVER fabricate stakeholder names, Salesforce IDs, dollar amounts, or dates.

================ TONE ================
Be confident. Drop hedging language ("appears to", "likely", "may be", "based on call participation"). State what the data says.
DO NOT use any soft tags such as `(public knowledge)`, `(inferred)`, `*(public knowledge)*`, or `*(inferred)*`. The CSE doesn't need provenance — they need the answer. If you genuinely don't know something, write "TBD". If you have to reason across multiple fields to arrive at an answer, just state the answer.
DO NOT put metadata in parentheses inline (no email addresses, no "(3 call appearances)", no "(stage: Commercials)"). Reserve parentheses for Gong call citations only, in the form `(Gong: "Call title", YYYY-MM-DD)` or `(Gong call on YYYY-MM-DD)`.
Use markdown links `[label](url)` when the data contains a URL (Gong call URL, link pasted into manual notes).

================ WHERE TO PULL EACH SECTION FROM ================
Executive Summary
- "What do they do? / How do they monetize? / Strategic Initiatives? / Stock Price / Market Leader? / Industry Challenges?": draw on general knowledge of `account.name` and `account.industry` plus the structured data. State the answer plainly — no `(public knowledge)` tag.
- "In general what do they care about as a company?": cross-reference `gong.key_themes` and `deal_lifecycle.deal_summary.key_topics_discussed`.
- "Are there any big AI initiatives we should be aware of?": prefer evidence from GONG RECENT CALLS (spotlight briefs, transcripts), `gong.key_themes`, `account.is_using_llms`, `account.num_models`, `account.deployment_types`. State the AI initiative directly. No `(inferred)` tag.

Assigned Team (top of doc)
- AE: `arize_team.account_owner`.
- SA: `arize_team.assigned_sa`. If null, write "TBD".
- AI SE: `arize_team.assigned_ai_se`. Only include if present.
- CSE: `arize_team.assigned_cse`. If null, write "TBD".
- CSM: `arize_team.assigned_csm`. Only include if present.
- DO NOT infer SA / AE / CSE from Gong call participation. The Gong attendee list is not authoritative for role assignment — only the SFDC fields are.

Org & Stakeholders
- "Arize team": list whoever is populated in `arize_team` (AE / SA / AI SE / CSE / CSM). Plain names, role labels, no metadata in parentheses.
- "Customer stakeholders": list each entry in `gong.stakeholders_seen_external` already sorted by `appearances`. Render each on its own bullet as `- **Name** — role/responsibility` if a role is implied by the transcripts, otherwise just `- **Name**`. NEVER include the email address, the appearance count, or any other metadata in parentheses. The data behind those names is for your context only — don't echo it.
- "Identify EB, champions, promoters, detractors": only assign a label when transcripts give clear evidence (someone explicitly pushed the deal, blocked it, or escalated). State the label and the one-sentence reason; no `(inferred)` tag. If no clear signal, write "TBD".

Contract Entitlements
- "Current ARR" — pick the FIRST non-null source from this priority chain and state the value plainly:
  1. `contract.total_active_arr_display` (active ARR on the SFDC account).
  2. `contract.summed_won_amount_display` (sum of closed-won opps).
  3. `contract.expected_arr_display` (largest non-zero open opportunity amount when there are no won opps yet — pre-contract pipeline).
  Always say which opp the figure ties to when ARR comes from #2 or #3, e.g. `**Current ARR:** $177,352 — open opportunity "Netapp-LLM-NEW", stage 5. Commercials, close 2026-05-29.`
- "Renewal Date": NOT in the structured data. If there is a closed-won opp, state `contract.latest_won_close_date` as the contract close date, then add: "Renewal: TBD — confirm with RevOps / Deal Desk." If there is no closed-won opp, write "Renewal: TBD — no contract closed yet."
- "Entitlements": derive from `account.product_tier`, `account.deployment_types`, `account.is_using_llms`, `account.num_models`. List the source opp name(s) from `contract.won_opportunities` or, if none, `contract.expected_arr_opportunity`.
- "Is there a specific Case Study clause?": "TBD — confirm with the AE."

Use Cases (LLM / ML / app names)
- LLM use cases: pull from GONG RECENT CALLS spotlights/transcripts mentioning agents, LLM apps, RAG, evals, prompts, guardrails, etc. Cross-reference `account.is_using_llms` and `gong.key_themes`.
- ML use cases: same approach for tabular / classical ML signals (model monitoring, drift, fraud, recsys).
- App names: ONLY include names that appear verbatim in `account.description`, `account.customer_notes`, opp `description`, or call transcripts.

Key Initiatives / Outcomes / Business Pain
- Pull from `deal_lifecycle.deal_summary.current_state`, `key_topics_discussed`, `blockers_identified`, `next_steps_from_calls`, plus quoted business pain from GONG transcripts.

Expansion Opportunities
- Open opps: every entry in `pipeline.open_opportunities` with stage / amount / probability / close_date / next_step. Render as `**$Amount – Opp Name (Stage X, close YYYY-MM-DD)**` then a one-sentence next-step description.
- "Do they have more Models / LLM apps in the future?": evidence from `gong.key_themes`, transcripts, `account.num_models`, `product_usage.adoption_status` / `trend`.
- "Have we discussed growth with their executives?": YES if any external stakeholder has a VP/Director/Head/Chief title in transcripts.
- "Central AI team?": evidence from transcripts.
- "Strategy to get wider": one short paragraph using AE/SA/CSE pivots.

POC Requirements / Info
- "Link in the POC scoping doc": include the link only if it appears in the manual SA notes; otherwise "TBD".
- "Any sticking points": pull from `deal_lifecycle.deal_summary.blockers_identified`, `risk_factors`, and transcript snippets.
- "Features they really cared about": pull from transcripts and `user_behavior.key_workflows_used` / `product_usage.top_features`.

Current Tech Stack / Competition
- ONLY include competitors / frameworks / cloud providers explicitly mentioned in: GONG transcripts, `account.description`, `account.customer_notes`, opp descriptions, or `deal_lifecycle.deal_summary.*`. Never invent. If no signal, write "TBD".

Asks / Product Requests
- Pull from `deal_lifecycle.deal_summary.next_steps_from_calls`, transcripts, and any feature-tracking links in the SA notes.
- "Which PM was involved?": only if a PM is named in transcripts or `gong.arize_attendees` with a clearly stated PM role.

Risks to onboarding / initial value
- Combine `deal_lifecycle.deal_summary.risk_factors` + `blockers_identified`, `user_behavior.critical_issues` / `issues_summary`, and `product_usage.adoption_status` / `days_since_last_activity`. If `product_usage.adoption_status` is `not_started` or `churning`, call it out as a real risk.

================ FINAL CHECK BEFORE YOU WRITE ================
- If you see at least one non-null Gong call in GONG RECENT CALLS, the "Customer stakeholders" and "Use Cases" sections MUST cite at least one of those calls. Don't fall back to "TBD" when call data exists.
- The "Assigned Team" header AND the "Arize team" bullet MUST come from `arize_team` (which already merges SFDC account fields with the most-relevant opportunity's `assigned_sa` / `assigned_ai_se` / owner). Do not pick names from Gong attendees for the Arize team.
- "Current ARR" MUST follow the three-step priority chain above. If `contract.total_active_arr_display` is set, use it. If not but `contract.summed_won_amount_display` is set, use that. If both are null but `contract.expected_arr_display` is set (open opp pipeline), use that and cite the opp.
- Do NOT use `(public knowledge)` or `(inferred)` anywhere. Do NOT put email addresses or call-appearance counts in parentheses next to stakeholder names.
- If `manual_notes` is non-empty, treat it as authoritative SA color and surface its contents (especially links and sticking points) in the relevant section.
"""


_USER_PROMPT = """Generate the Knowledge Transfer document for the account below.

============== TEMPLATE ==============
{template}
============== END TEMPLATE ==============

============== ACCOUNT (Salesforce) ==============
{account_json}
============== END ACCOUNT ==============

============== ARIZE TEAM ==============
{arize_team_json}
============== END ARIZE TEAM ==============

============== CONTRACT (closed-won opps + ARR) ==============
{contract_json}
============== END CONTRACT ==============

============== OPEN PIPELINE ==============
{pipeline_json}
============== END OPEN PIPELINE ==============

============== DEAL LIFECYCLE & DEAL SUMMARY ==============
{deal_lifecycle_json}
============== END DEAL LIFECYCLE ==============

============== GONG SUMMARY (counts + themes + stakeholders) ==============
{gong_json}
============== END GONG SUMMARY ==============

============== GONG RECENT CALLS (newest first; quote these) ==============
{gong_calls_block}
============== END GONG RECENT CALLS ==============

============== PRODUCT USAGE (Pendo + FullStory) ==============
{product_usage_json}
============== END PRODUCT USAGE ==============

============== USER BEHAVIOR ==============
{user_behavior_json}
============== END USER BEHAVIOR ==============

============== LAST 24H ACCOUNT TELEMETRY ==============
{last_24h_json}
============== END LAST 24H ==============

DATA SOURCES SURFACED: {data_sources}
ERRORS FROM PIPELINE (informational): {errors}

{manual_notes_block}

Today is {today}. Write the full Knowledge Transfer markdown document now, following every rule in the system prompt and the field-mapping guidance for each section."""


# ---------------------------------------------------------------------------
# Context-building helpers
# ---------------------------------------------------------------------------


def _money(v: Any) -> Optional[str]:
    if v is None:
        return None
    try:
        return f"${float(v):,.0f}"
    except (TypeError, ValueError):
        return None


def _opp_to_dict(
    o: OpportunityData,
    *,
    include_description: bool = False,
    max_desc: int = 800,
) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "id": o.id,
        "name": o.name,
        "stage_name": o.stage_name,
        "amount": o.amount,
        "amount_display": _money(o.amount),
        "close_date": o.close_date,
        "is_closed": o.is_closed,
        "is_won": o.is_won,
        "probability": o.probability,
        "type": o.type,
        "lead_source": o.lead_source,
        "owner_name": o.owner_name,
        "next_step": o.next_step,
        "age_in_days": o.age_in_days,
        "created_date": o.created_date,
        "last_modified_date": o.last_modified_date,
        "forecast_category": o.forecast_category,
    }
    if include_description and o.description:
        d["description"] = o.description[:max_desc]
    return {k: v for k, v in d.items() if v not in (None, "")}


def _strip_empty(d: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: v
        for k, v in d.items()
        if v not in (None, "", [], {}, "unknown")
    }


def _format_gong_calls_block(calls: List[GongCallData]) -> str:
    """Render Gong calls as a readable per-call block (NOT JSON) so the model
    can grep the body text. JSON dump truncates strings field-wise; this view
    is what gets the full transcript and spotlight content."""
    if not calls:
        return "(No Gong calls available for this account.)"
    cap = int(os.environ.get("TRANSITION_GONG_CALLS_IN_PROMPT", "8"))
    snippet_chars = int(os.environ.get("TRANSITION_GONG_TRANSCRIPT_CHARS", "4000"))
    spotlight_chars = int(os.environ.get("TRANSITION_GONG_SPOTLIGHT_CHARS", "1500"))

    sorted_calls = sorted(calls, key=lambda c: (c.call_date or ""), reverse=True)
    parts: List[str] = []
    for i, c in enumerate(sorted_calls[:cap], 1):
        ext_attendees = [
            f"{p.name or p.email or '?'}"
            + (f" <{p.email}>" if p.email and p.name else "")
            for p in (c.participants or [])
            if (p.affiliation or "").lower() == "external"
        ]
        int_attendees = [
            f"{p.name or p.email or '?'}"
            for p in (c.participants or [])
            if (p.affiliation or "").lower() != "external"
        ]
        block: List[str] = [
            f"--- Call #{i} ---",
            f"Date: {c.call_date or 'unknown'}",
            f"Title: {c.call_title or 'untitled'}",
        ]
        if c.duration_minutes is not None:
            try:
                block.append(f"Duration (min): {float(c.duration_minutes):.0f}")
            except (TypeError, ValueError):
                pass
        block.append(f"Customer attendees: {', '.join(ext_attendees) or '(none captured)'}")
        block.append(f"Arize attendees: {', '.join(int_attendees) or '(none captured)'}")
        if c.spotlight_brief:
            block.append(f"Spotlight brief: {c.spotlight_brief[:spotlight_chars]}")
        if c.spotlight_key_points:
            kp = "\n  - ".join(str(p) for p in c.spotlight_key_points[:6])
            block.append(f"Spotlight key points:\n  - {kp}")
        if c.spotlight_next_steps:
            block.append(f"Spotlight next steps: {c.spotlight_next_steps[:600]}")
        if c.spotlight_outcome:
            block.append(f"Spotlight outcome: {c.spotlight_outcome[:400]}")
        if c.transcript_snippet:
            block.append(f"Transcript snippet:\n{c.transcript_snippet[:snippet_chars]}")
        if c.call_url:
            block.append(f"Gong URL: {c.call_url}")
        parts.append("\n".join(block))
    if len(sorted_calls) > cap:
        parts.append(f"(…{len(sorted_calls) - cap} additional older calls truncated for context budget.)")
    return "\n\n".join(parts)


def _build_transition_context(
    overview: ProspectOverview,
) -> Tuple[Dict[str, Any], str]:
    """Build a denormalized, named-section view + a raw Gong calls block.

    The dict is what we serialize as JSON in the per-section prompt blocks.
    The string is the full-fat per-call rendering (kept out of JSON so it
    doesn't get truncated field-by-field)."""
    sf = overview.salesforce
    all_opps = list(overview.all_opportunities or [])
    won_opps = [o for o in all_opps if o.is_won]
    won_opps_sorted = sorted(won_opps, key=lambda o: (o.close_date or ""), reverse=True)
    open_opps = [o for o in all_opps if not o.is_closed]
    # Sort open opps by amount desc (largest first) for ARR fallback selection,
    # then by nearest close date for display ordering.
    open_opps_by_amount = sorted(
        open_opps, key=lambda o: (-(o.amount or 0), o.close_date or "9999-12-31")
    )
    open_opps_by_close = sorted(
        open_opps, key=lambda o: (o.close_date or "9999-12-31")
    )

    # Pick the most-relevant opportunity for *team assignment* fallback. Prefer
    # the largest open opp (the one currently driving the deal) so SA / AI SE
    # come from the live opp; if no open opp exists, use the most recent won.
    most_relevant_opp = None
    if open_opps_by_amount:
        most_relevant_opp = open_opps_by_amount[0]
    elif won_opps_sorted:
        most_relevant_opp = won_opps_sorted[0]
    elif all_opps:
        most_relevant_opp = sorted(
            all_opps, key=lambda o: (o.close_date or ""), reverse=True
        )[0]

    account: Dict[str, Any] = {}
    arize_team: Dict[str, Any] = {}
    if sf:
        account = _strip_empty(
            {
                "name": sf.name,
                "website": sf.website,
                "industry": sf.industry,
                "annual_revenue": sf.annual_revenue,
                "annual_revenue_display": _money(sf.annual_revenue),
                "number_of_employees": sf.number_of_employees,
                "lifecycle_stage": sf.lifecycle_stage,
                "product_tier": sf.product_tier,
                "customer_success_tier": sf.customer_success_tier,
                "deployment_types": sf.deployment_types,
                "is_using_llms": sf.is_using_llms,
                "num_models": sf.num_models,
                "last_activity_date": sf.last_activity_date,
                "created_date": sf.created_date,
                "description": (sf.description or "")[:1500] or None,
                "customer_notes": (sf.customer_notes or "")[:1500] or None,
                "next_steps": (sf.next_steps or "")[:800] or None,
                "total_active_arr": sf.total_active_arr,
                "total_active_arr_display": _money(sf.total_active_arr),
                "status": sf.status,
            }
        )

    # arize_team: prefer SFDC account-level fields, then fall back to the most
    # relevant opportunity (live deal). The account record often leaves these
    # null while the opp owns the assignment — see NetApp's Netapp-LLM-NEW.
    def _first_non_empty(*values: Any) -> Optional[str]:
        for v in values:
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    rel_owner = getattr(most_relevant_opp, "owner_name", None) if most_relevant_opp else None
    rel_sa = getattr(most_relevant_opp, "assigned_sa", None) if most_relevant_opp else None
    rel_ai_se = getattr(most_relevant_opp, "assigned_ai_se", None) if most_relevant_opp else None

    arize_team = _strip_empty(
        {
            "account_owner": _first_non_empty(
                getattr(sf, "owner_name", None), rel_owner
            ),
            "assigned_sa": _first_non_empty(
                getattr(sf, "assigned_sa", None), rel_sa
            ),
            "assigned_ai_se": _first_non_empty(
                getattr(sf, "assigned_ai_se", None), rel_ai_se
            ),
            "assigned_cse": getattr(sf, "assigned_cse", None) if sf else None,
            "assigned_csm": getattr(sf, "assigned_csm", None) if sf else None,
            "team_source": (
                "salesforce_account+opportunity"
                if (rel_sa or rel_ai_se or rel_owner)
                else ("salesforce_account" if sf else None)
            ),
            "team_source_opportunity": (
                most_relevant_opp.name
                if most_relevant_opp
                and (
                    (rel_sa and (not sf or not getattr(sf, "assigned_sa", None)))
                    or (rel_ai_se and (not sf or not getattr(sf, "assigned_ai_se", None)))
                    or (rel_owner and (not sf or not getattr(sf, "owner_name", None)))
                )
                else None
            ),
        }
    )

    if not account.get("name"):
        # fall back to the lookup label so the LLM has *something* for the H1.
        account.setdefault("name", overview.lookup_value)

    # ---- Contract: ARR with three-step fallback chain ----
    summed_won = sum((o.amount or 0) for o in won_opps) or None
    latest_won_close = won_opps_sorted[0].close_date if won_opps_sorted else None

    # Largest non-zero open opp drives "expected ARR" when no contract exists yet.
    expected_arr_opp = next(
        (o for o in open_opps_by_amount if (o.amount or 0) > 0), None
    )

    contract = _strip_empty(
        {
            "total_active_arr": sf.total_active_arr if sf else None,
            "total_active_arr_display": _money(sf.total_active_arr) if sf else None,
            "summed_won_amount": summed_won,
            "summed_won_amount_display": _money(summed_won),
            "latest_won_close_date": latest_won_close,
            "won_opportunity_count": len(won_opps),
            "won_opportunities": [
                _opp_to_dict(o, include_description=True) for o in won_opps_sorted[:5]
            ],
            # Pre-contract / open-opp fallback so the LLM can still report a
            # "Current ARR" figure when nothing has closed yet (NetApp case).
            "expected_arr": (expected_arr_opp.amount if expected_arr_opp else None),
            "expected_arr_display": _money(expected_arr_opp.amount) if expected_arr_opp else None,
            "expected_arr_opportunity": (
                _opp_to_dict(expected_arr_opp, include_description=True)
                if expected_arr_opp
                else None
            ),
            # Pre-computed source label so the LLM doesn't have to think about it.
            "arr_source": (
                "total_active_arr"
                if sf and sf.total_active_arr
                else (
                    "summed_won_amount"
                    if summed_won
                    else (
                        "expected_arr_open_opp"
                        if expected_arr_opp
                        else None
                    )
                )
            ),
        }
    )

    # ---- Pipeline: open opps drive expansion (ordered by close date) ----
    pipeline = _strip_empty(
        {
            "open_count": len(open_opps),
            "open_opportunities": [
                _opp_to_dict(o, include_description=True) for o in open_opps_by_close[:5]
            ],
        }
    )

    # ---- Deal lifecycle ----
    se = overview.sales_engagement
    deal_lifecycle: Dict[str, Any] = {}
    if se:
        deal_lifecycle = _strip_empty(
            {
                "first_touch_date": se.first_touch_date,
                "days_in_sales_cycle": se.days_in_sales_cycle,
                "current_stage": se.current_stage,
                "total_calls": se.total_calls,
                "total_emails": se.total_emails,
                "total_meetings": se.total_meetings,
                "total_tasks": se.total_tasks,
                "last_sales_activity_date": se.last_sales_activity_date,
                "days_since_last_activity": se.days_since_last_activity,
            }
        )
        if se.deal_summary:
            ds = _strip_empty(
                {
                    "current_state": se.deal_summary.current_state,
                    "key_topics_discussed": list(se.deal_summary.key_topics_discussed or []),
                    "blockers_identified": list(se.deal_summary.blockers_identified or []),
                    "next_steps_from_calls": list(se.deal_summary.next_steps_from_calls or []),
                    "champion_sentiment": se.deal_summary.champion_sentiment,
                    "risk_factors": list(se.deal_summary.risk_factors or []),
                }
            )
            if ds:
                deal_lifecycle["deal_summary"] = ds

    # ---- Gong: structured roll-up (separate from the full call block) ----
    gs = overview.gong_summary
    gong: Dict[str, Any] = {}
    gong_calls_block = ""
    if gs:
        ext_buckets: Dict[str, Dict[str, Any]] = {}
        int_buckets: Dict[str, Dict[str, Any]] = {}
        for call in gs.recent_calls or []:
            for p in call.participants or []:
                key = (p.email or p.name or "").strip().lower()
                if not key:
                    continue
                bucket = (
                    ext_buckets
                    if (p.affiliation or "").lower() == "external"
                    else int_buckets
                )
                if key not in bucket:
                    bucket[key] = {
                        "name": p.name,
                        "email": p.email,
                        "appearances": 0,
                    }
                bucket[key]["appearances"] += 1

        ext_sorted = sorted(
            ext_buckets.values(), key=lambda x: (-x["appearances"], x.get("name") or "")
        )
        int_sorted = sorted(
            int_buckets.values(), key=lambda x: (-x["appearances"], x.get("name") or "")
        )

        gong = _strip_empty(
            {
                "total_calls": gs.total_calls,
                "total_duration_minutes": gs.total_duration_minutes,
                "first_call_date": gs.first_call_date,
                "last_call_date": gs.last_call_date,
                "days_since_last_call": gs.days_since_last_call,
                "avg_talk_ratio": gs.avg_talk_ratio,
                "key_themes": list(gs.key_themes or []),
                "stakeholders_seen_external": ext_sorted[:25],
                "arize_attendees": int_sorted[:15],
            }
        )
        gong_calls_block = _format_gong_calls_block(gs.recent_calls or [])

    # ---- Product usage ----
    pu = overview.product_usage
    product_usage: Dict[str, Any] = {}
    if pu:
        product_usage = _strip_empty(
            {
                "adoption_status": pu.adoption_status if pu.adoption_status != "not_started" else None,
                "total_users": pu.total_users or None,
                "active_users_last_7_days": pu.active_users_last_7_days or None,
                "active_users_last_30_days": pu.active_users_last_30_days or None,
                "total_time_minutes": pu.total_time_minutes or None,
                "avg_session_minutes": pu.avg_session_minutes,
                "last_platform_activity": pu.last_platform_activity,
                "last_active_user": pu.last_active_user,
                "days_since_last_activity": pu.days_since_last_activity,
                "trend": pu.trend if pu.trend != "stable" else None,
            }
        )
    pendo = overview.pendo_usage
    if pendo:
        if pendo.top_features:
            product_usage["top_features"] = [
                _strip_empty(
                    {
                        "name": f.feature_name or f.feature_id,
                        "events": f.event_count,
                        "unique_users": f.unique_users,
                        "last_used": f.last_used,
                    }
                )
                for f in pendo.top_features[:8]
            ]
        if pendo.top_pages:
            product_usage["top_pages"] = [
                _strip_empty(
                    {
                        "name": p.page_name or p.page_id,
                        "views": p.view_count,
                        "unique_viewers": p.unique_viewers,
                        "minutes": p.total_minutes,
                    }
                )
                for p in pendo.top_pages[:8]
            ]

    # ---- User behavior ----
    ub = overview.user_behavior
    user_behavior: Dict[str, Any] = {}
    if ub:
        user_behavior = _strip_empty(
            {
                "summary": ub.summary,
                "hypothesis": ub.hypothesis,
                "engagement_level": ub.engagement_level if ub.engagement_level != "unknown" else None,
                "key_workflows_used": list(ub.key_workflows_used or []),
                "adoption_milestones_completed": [
                    m.name for m in (ub.adoption_milestones or []) if m.completed
                ][:20],
                "critical_issues": list(ub.critical_issues or [])[:8],
                "issues_summary": ub.issues_summary,
            }
        )

    # ---- Last 24h telemetry (high-level only) ----
    last_24h: Dict[str, Any] = {}
    if overview.last_24h_activity:
        last_24h = _strip_empty(
            {
                "account_summary": overview.last_24h_activity.account_summary,
                "total_active_users_24h": overview.last_24h_activity.total_active_users_24h or None,
                "pendo_total_events_24h": overview.last_24h_activity.pendo_total_events_24h or None,
            }
        )

    return (
        {
            "account": account,
            "arize_team": arize_team,
            "contract": contract,
            "pipeline": pipeline,
            "deal_lifecycle": deal_lifecycle,
            "gong": gong,
            "product_usage": product_usage,
            "user_behavior": user_behavior,
            "last_24h": last_24h,
            "data_sources_available": list(overview.data_sources_available or []),
            "errors": list(overview.errors or [])[:5],
        },
        gong_calls_block,
    )


def _to_pretty_json(obj: Any) -> str:
    if obj in (None, {}, []):
        return "{}"
    try:
        return json.dumps(obj, indent=2, default=str, allow_nan=False)
    except (ValueError, TypeError):
        return json.dumps(obj, indent=2, default=str)


def _resolve_account_label(
    overview: ProspectOverview, ctx: Optional[Dict[str, Any]] = None
) -> str:
    """Best-effort human label for the account, used in the document title."""
    if ctx:
        a = ctx.get("account") or {}
        n = a.get("name")
        if isinstance(n, str) and n.strip():
            return n.strip()
    sf = getattr(overview, "salesforce", None)
    if sf is not None:
        for attr in ("name", "account_name"):
            value = getattr(sf, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
    fallback = getattr(overview, "lookup_value", None)
    if isinstance(fallback, str) and fallback.strip():
        return fallback.strip()
    return "Unknown account"


def build_transition_document(
    *,
    overview: ProspectOverview,
    manual_notes: Optional[str] = None,
    llm_model: Optional[str] = None,
    max_tokens: int = 6000,
) -> Dict[str, Any]:
    """Build a CS transition Markdown document for a closed-won account.

    Returns a dict with:
      - ``markdown``: the full document text (str)
      - ``account_name``: best-effort label resolved from the overview (str)
      - ``model``: model id actually used (str)
      - ``data_sources``: data sources surfaced by the BQ pipeline (list[str])
      - ``stats``: counts of inputs that were fed to the LLM (for diagnostics)
    """
    ctx, gong_calls_block = _build_transition_context(overview)
    account_label = _resolve_account_label(overview, ctx)

    notes_clean = (manual_notes or "").strip()
    if notes_clean:
        manual_notes_block = (
            "============== ADDITIONAL NOTES FROM SA (treat as authoritative) ==============\n"
            f"{notes_clean}\n"
            "============== END ADDITIONAL NOTES ==============\n"
        )
    else:
        manual_notes_block = ""

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    user_msg = _USER_PROMPT.format(
        template=TRANSITION_TEMPLATE,
        account_json=_to_pretty_json(ctx["account"]),
        arize_team_json=_to_pretty_json(ctx["arize_team"]),
        contract_json=_to_pretty_json(ctx["contract"]),
        pipeline_json=_to_pretty_json(ctx["pipeline"]),
        deal_lifecycle_json=_to_pretty_json(ctx["deal_lifecycle"]),
        gong_json=_to_pretty_json(ctx["gong"]),
        gong_calls_block=gong_calls_block or "(No Gong calls available for this account.)",
        product_usage_json=_to_pretty_json(ctx["product_usage"]),
        user_behavior_json=_to_pretty_json(ctx["user_behavior"]),
        last_24h_json=_to_pretty_json(ctx["last_24h"]),
        data_sources=", ".join(ctx["data_sources_available"]) or "(none)",
        errors=", ".join(ctx["errors"]) or "(none)",
        manual_notes_block=manual_notes_block,
        today=today,
    )

    # Hard cap so we do not blow past model context windows on huge accounts.
    max_chars = int(os.environ.get("TRANSITION_DOC_MAX_PROMPT_CHARS", "120000"))
    if len(user_msg) > max_chars:
        user_msg = (
            user_msg[:max_chars]
            + "\n\n…(prompt truncated for model context budget; rely on what's above when filling.)\n"
        )

    model_id = (
        llm_model
        or os.environ.get("TRANSITION_DOC_MODEL")
        or os.environ.get("MODEL_NAME")
        or "claude-sonnet-4-20250514"
    )

    response = llm_completion(
        model=model_id,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    content = ""
    try:
        content = (response.choices[0].message.content or "").strip()
    except Exception:
        content = ""

    if not content:
        raise RuntimeError(
            "LLM returned an empty Knowledge Transfer document. Check model availability and try again."
        )

    if not content.lstrip().startswith("#"):
        content = f"# {account_label} - Internal Knowledge Transfer\n\n" + content

    gong_call_count = 0
    gong_with_transcripts = 0
    gs = overview.gong_summary
    if gs and gs.recent_calls:
        gong_call_count = len(gs.recent_calls)
        gong_with_transcripts = sum(
            1 for c in gs.recent_calls if (c.transcript_snippet or "").strip()
        )

    return {
        "markdown": content,
        "account_name": account_label,
        "model": model_id,
        "data_sources": list(getattr(overview, "data_sources_available", []) or []),
        "stats": {
            "won_opportunities": len(ctx.get("contract", {}).get("won_opportunities", []) or []),
            "open_opportunities": len(ctx.get("pipeline", {}).get("open_opportunities", []) or []),
            "gong_calls": gong_call_count,
            "gong_calls_with_transcripts": gong_with_transcripts,
            "external_stakeholders": len(
                (ctx.get("gong") or {}).get("stakeholders_seen_external", []) or []
            ),
            "prompt_chars": len(user_msg),
        },
    }
