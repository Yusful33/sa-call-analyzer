"""
Derive deal health, competitive mention rollups, and meeting-prep bullets from ProspectOverview.
"""

from __future__ import annotations

from typing import Any

from models import ProspectOverview


def compute_deal_health(overview: ProspectOverview) -> dict[str, Any]:
    """Heuristic 0–100 score from CRM, Gong, and product signals (no ML)."""
    score = 50.0
    factors: list[dict[str, str]] = []

    pu = overview.product_usage
    if pu:
        ad = (pu.adoption_status or "unknown").lower()
        if ad in ("power_user", "active"):
            score += 10
            factors.append({"signal": "product_adoption", "detail": f"Adoption: {ad}", "impact": "+10"})
        elif ad == "exploring":
            score += 4
            factors.append({"signal": "product_adoption", "detail": "Adoption: exploring", "impact": "+4"})
        elif ad == "churning":
            score -= 18
            factors.append({"signal": "product_adoption", "detail": "Adoption: churning risk", "impact": "-18"})

        dsla = pu.days_since_last_activity
        if dsla is not None:
            if dsla <= 7:
                score += 8
                factors.append({"signal": "recency", "detail": f"Last platform activity {dsla}d ago", "impact": "+8"})
            elif dsla > 30:
                score -= 10
                factors.append({"signal": "recency", "detail": f"No platform activity {dsla}d", "impact": "-10"})

    se = overview.sales_engagement
    ds = se.deal_summary if se else None
    if ds:
        sent = (ds.champion_sentiment or "").lower()
        if sent == "positive":
            score += 10
            factors.append({"signal": "gong_sentiment", "detail": "Champion sentiment positive", "impact": "+10"})
        elif sent == "concerned":
            score -= 15
            factors.append({"signal": "gong_sentiment", "detail": "Champion sentiment concerned", "impact": "-15"})
        if ds.risk_factors:
            score -= min(12, 4 * len(ds.risk_factors))
            factors.append({"signal": "risks", "detail": f"{len(ds.risk_factors)} risk factor(s) noted", "impact": "penalty"})

    lo = overview.latest_opportunity
    if lo:
        prob = lo.probability or 0
        if prob >= 60:
            score += 6
            factors.append({"signal": "opportunity", "detail": f"Opp probability {prob}%", "impact": "+6"})
        elif prob <= 25 and prob > 0:
            score -= 5
            factors.append({"signal": "opportunity", "detail": f"Opp probability {prob}%", "impact": "-5"})

    ub = overview.user_behavior
    if ub:
        if ub.engagement_level == "high":
            score += 6
            factors.append({"signal": "engagement", "detail": "High product engagement", "impact": "+6"})
        elif ub.engagement_level == "low":
            score -= 8
            factors.append({"signal": "engagement", "detail": "Low engagement", "impact": "-8"})
        if ub.critical_issues:
            score -= min(15, 5 * len(ub.critical_issues))
            factors.append({"signal": "issues", "detail": f"{len(ub.critical_issues)} critical issue(s)", "impact": "penalty"})

    gs = overview.gong_summary
    if gs and gs.days_since_last_call is not None:
        if gs.days_since_last_call > 21:
            score -= 6
            factors.append({"signal": "calls", "detail": f"No Gong call {gs.days_since_last_call}d", "impact": "-6"})

    score = max(0, min(100, int(round(score))))
    band = "strong" if score >= 75 else "steady" if score >= 50 else "at_risk"
    return {
        "score": score,
        "band": band,
        "factors": factors,
    }


def _rec_as_dict(rec: Any) -> dict[str, Any]:
    if isinstance(rec, dict):
        return rec
    if hasattr(rec, "model_dump"):
        return rec.model_dump()
    return {}


def compute_competitive_mentions(overview: ProspectOverview) -> list[dict[str, Any]]:
    """Flatten competitive recommendation objects from user_behavior."""
    rows: list[dict[str, Any]] = []
    ub = overview.user_behavior
    if not ub or not ub.recommendations:
        return rows

    for rec in ub.recommendations:
        rd = _rec_as_dict(rec)
        cat = rd.get("category")
        if cat != "Competitive":
            continue

        cms = rd.get("competitive_messaging") or []

        for cm in cms:
            if isinstance(cm, dict):
                competitor = cm.get("competitor") or "Unknown"
                rows.append(
                    {
                        "competitor": competitor,
                        "what_they_said": cm.get("what_they_said"),
                        "targeted_response": cm.get("targeted_response"),
                        "mentioned_in": cm.get("mentioned_in") or [],
                        "note": cm.get("note"),
                    }
                )
            else:
                rows.append(
                    {
                        "competitor": getattr(cm, "competitor", None) or "Unknown",
                        "what_they_said": getattr(cm, "what_they_said", None),
                        "targeted_response": getattr(cm, "targeted_response", None),
                        "mentioned_in": getattr(cm, "mentioned_in", None) or [],
                        "note": getattr(cm, "note", None),
                    }
                )

    return rows


def compute_meeting_prep(overview: ProspectOverview) -> dict[str, Any]:
    """Structured talking points for SA/AE prep."""
    bullets: list[str] = []
    sf = overview.salesforce
    if sf:
        nm = sf.name or "Account"
        arr = sf.total_active_arr
        bullets.append(f"{nm} — active ARR context recorded" + (f" ({arr})" if arr is not None else ""))
        team_bits = [x for x in (sf.assigned_sa, sf.assigned_csm, sf.owner_name) if x]
        if team_bits:
            bullets.append("Coverage: " + ", ".join(team_bits[:4]))

    lo = overview.latest_opportunity
    if lo:
        bullets.append(
            f"Primary opportunity: {lo.name} — stage {lo.stage_name or 'n/a'}"
            + (f", close {lo.close_date}" if lo.close_date else "")
        )
        if lo.next_step:
            bullets.append(f"CRM next step: {lo.next_step}")

    se = overview.sales_engagement
    ds = se.deal_summary if se else None
    if ds and ds.current_state:
        bullets.append(f"Deal narrative: {ds.current_state[:280]}{'…' if len(ds.current_state) > 280 else ''}")
    if ds and ds.next_steps_from_calls:
        for ns in ds.next_steps_from_calls[:3]:
            bullets.append(f"From calls: {ns}")

    gs = overview.gong_summary
    if gs and gs.key_themes:
        bullets.append("Gong themes: " + ", ".join(gs.key_themes[:6]))

    pu = overview.product_usage
    if pu and pu.days_since_last_activity is not None:
        bullets.append(f"Product: last activity ~{pu.days_since_last_activity} day(s) ago")

    ub = overview.user_behavior
    if ub:
        for r in ub.recommendations or []:
            rd = _rec_as_dict(r)
            if rd.get("category") == "Next Steps" and rd.get("steps"):
                for s in (rd.get("steps") or [])[:4]:
                    bullets.append(f"Recommended next step: {s}")

    return {"bullets": bullets[:20]}


def build_prospect_intelligence_bundle(overview: ProspectOverview) -> dict[str, Any]:
    return {
        "deal_health": compute_deal_health(overview),
        "competitive_mentions": compute_competitive_mentions(overview),
        "meeting_prep": compute_meeting_prep(overview),
    }
