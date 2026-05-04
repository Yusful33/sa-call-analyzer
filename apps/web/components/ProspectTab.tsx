"use client";

import { useState } from "react";
import { apiPost } from "@/lib/api";
import {
  escapeHtml,
  formatCurrency,
  formatPercent,
  formatDate,
  formatMinutes,
} from "@/lib/helpers";
import type { ResolveAccountFn } from "@/lib/accountResolve";

/* eslint-disable @typescript-eslint/no-explicit-any */

/* ---------- Feature definitions for Pendo feature tooltips ---------- */
const featureDefinitions: Record<string, string> = {
  "Query Filters": "Filtering and searching through trace/span data",
  "Query filters": "Filtering and searching through trace/span data",
  "Trace View": "Viewing detailed traces of LLM/ML operations",
  Dashboard: "Main analytics dashboard with metrics overview",
  Projects: "Organizing work into separate project spaces",
  Datasets: "Managing evaluation and training datasets",
  Experiments: "Running and comparing model experiments",
  Evaluations: "Testing model quality with eval tasks",
  Prompts: "Managing and versioning prompt templates",
  Spans: "Individual operation tracking within traces",
  Annotations: "Adding labels and feedback to data",
  Monitors: "Setting up alerts and monitoring rules",
  Embeddings: "Visualizing and analyzing vector embeddings",
  "Model Performance": "Tracking model accuracy and metrics",
  "Model Tabs": "Navigating between model configuration views",
  Config: "Model configuration and settings",
  Performance: "Analyzing model performance metrics",
  "worst performing": "Identifying low-performing data segments",
  slices: "Data segments grouped by attributes",
  baseline: "Reference model for performance comparison",
  "Cost Analysis": "Monitoring API costs and token usage",
  Comparisons: "Side-by-side model/version comparisons",
  Export: "Exporting data for external analysis",
  Inferences: "Viewing individual model predictions",
  Sessions: "User session and conversation tracking",
  "AI Search": "Natural language search across platform data",
  "Search Input": "Searching through traces, spans, or data",
  Configure: "Setting up model or project configuration",
  Drift: "Detecting changes in data distribution over time",
  "Data Quality": "Monitoring data integrity and completeness",
  Playground: "Interactive environment for testing prompts",
  Traces: "End-to-end request/response tracking",
  LLM: "Large Language Model monitoring features",
};

function getFeatureDescription(name: string | null): string | null {
  if (!name) return null;
  if (featureDefinitions[name]) return featureDefinitions[name];
  const lower = name.toLowerCase();
  for (const [key, desc] of Object.entries(featureDefinitions)) {
    if (lower.includes(key.toLowerCase()) || key.toLowerCase().includes(lower))
      return desc;
  }
  return null;
}

/* ---------- HTML builders ---------- */

function buildLast24hHtml(l24: any): string {
  if (!l24) return "";
  const users: any[] = l24.active_users || [];
  const userRows = users
    .map((u) => {
      const label = escapeHtml(u.display_name || u.email || u.visitor_id || "Unknown user");
      const emailLine = u.email && u.display_name ? `<div style="font-size:0.8em;color:#8ab4f8;">${escapeHtml(u.email)}</div>` : "";
      const sub = escapeHtml(u.summary || "");
      const feats = (u.top_features_24h || []).slice(0, 5).map((f: any) => `<li style="margin:2px 0;color:#d0d8e4;">${escapeHtml(f.name || f.id)} <span style="opacity:0.75">(${f.count})</span></li>`).join("");
      const pgs = (u.top_pages_24h || []).slice(0, 5).map((p: any) => `<li style="margin:2px 0;color:#d0d8e4;">${escapeHtml(p.name || p.id)} <span style="opacity:0.75">(${p.count})</span></li>`).join("");
      const issueItems = (u.fullstory_issues_24h || []).slice(0, 4).map((i: any) => {
        const href = i.recording_url && String(i.recording_url).startsWith("https://") ? i.recording_url : "";
        const link = href ? `<a href="${escapeHtml(href)}" target="_blank" rel="noopener" style="color:#f5b041;font-size:0.78em;">Recording</a>` : "";
        const ctx = escapeHtml(i.page_context || i.issue_type || "");
        return `<div style="display:flex;justify-content:space-between;align-items:center;gap:8px;margin-top:4px;font-size:0.78em;color:#e8c547;"><span>${ctx}</span>${link}</div>`;
      }).join("");
      const narrativeText = (u.fullstory_behavior_summary || "").trim();
      const hasFsIssues = (u.fullstory_issues_24h || []).length > 0;
      const fsNarrative = narrativeText || hasFsIssues
        ? `<div style="margin-top:10px;padding:10px 12px;background:rgba(126,214,255,0.08);border-radius:8px;border:1px solid rgba(126,214,255,0.25);">
            <div style="font-size:0.72em;text-transform:uppercase;letter-spacing:0.06em;color:#7dd3fc;margin-bottom:6px;">Session narrative (from FullStory data export)</div>
            <div style="font-size:0.86em;color:#e2edf5;line-height:1.55;white-space:pre-wrap;">${narrativeText ? escapeHtml(narrativeText) : '<span style="color:#9aa5b8;">No narrative returned yet.</span>'}</div>
            <div style="font-size:0.68em;color:#7a8fa6;margin-top:8px;">Built from warehouse events (URLs, clicks, loads), not video. Use Recording for the replay.</div>
           </div>`
        : "";
      return `<div style="margin-bottom:14px;padding:12px;background:rgba(0,0,0,0.22);border-radius:10px;border-left:3px solid #4facfe;">
        <div style="font-weight:600;color:#fff;">${label}</div>${emailLine}
        <div style="font-size:0.88em;color:#c8d0dc;margin-top:6px;line-height:1.45;">${sub}</div>
        ${fsNarrative}
        ${feats ? `<div style="margin-top:8px;font-size:0.78em;color:#a8b0bc;"><strong style="color:#9ee0ff;">Features</strong><ul style="margin:4px 0 0 18px;padding:0;">${feats}</ul></div>` : ""}
        ${pgs ? `<div style="margin-top:6px;font-size:0.78em;color:#a8b0bc;"><strong style="color:#9ee0ff;">Pages</strong><ul style="margin:4px 0 0 18px;padding:0;">${pgs}</ul></div>` : ""}
        ${issueItems ? `<div style="margin-top:8px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.08);"><span style="font-size:0.75em;color:#f5b041;">FullStory (friction)</span>${issueItems}</div>` : ""}
      </div>`;
    })
    .join("");
  const acct = l24.account_summary ? `<p style="margin:0 0 12px 0;line-height:1.5;color:#e8eef5;font-size:0.95em;">${escapeHtml(l24.account_summary)}</p>` : "";
  const meta = l24.total_active_users_24h != null ? `<div style="font-size:0.8em;color:#8fa3bf;margin-bottom:10px;">${l24.total_active_users_24h} user(s) in this view &middot; ${l24.pendo_total_events_24h || 0} Pendo events (24h)</div>` : "";
  return `<div style="margin-top:20px;padding:16px;background:rgba(255,255,255,0.07);border-radius:12px;border:1px solid rgba(255,255,255,0.12);">
    <div style="font-size:0.72em;text-transform:uppercase;letter-spacing:0.07em;color:#7dd3fc;margin-bottom:10px;">Last 24 hours (UTC) &mdash; by user</div>
    ${meta}${acct}${userRows || (l24.account_summary ? "" : '<p style="margin:0;color:#9aa5b8;font-size:0.9em;">No activity in this window.</p>')}
  </div>`;
}

function buildProspectHtml(overview: any): string {
  const sf = overview.salesforce;
  const se = overview.sales_engagement;
  const pu = overview.product_usage;

  /* ---- Sources / errors banner ---- */
  const sourcesHtml = (overview.data_sources_available ?? [])
    .map((src: string) => `<span style="background:#27ae60;color:white;padding:4px 10px;border-radius:12px;font-size:0.8em;margin-right:6px;">${escapeHtml(src)}</span>`)
    .join("");
  const errorsHtml = overview.errors?.length > 0
    ? `<div style="background:#fff3cd;border:1px solid #ffc107;padding:15px;border-radius:8px;margin-bottom:20px;"><strong>Note:</strong> ${overview.errors.map((e: string) => escapeHtml(e)).join("; ")}</div>`
    : "";

  /* ================================================================
     NEXT STEPS (shown at top)
     ================================================================ */
  let nextStepsHtml = "";
  const ub_ns = overview.user_behavior;
  if (ub_ns?.recommendations) {
    const nsRec = ub_ns.recommendations.find((r: any) => r.category === "Next Steps");
    if (nsRec?.steps?.length) {
      nextStepsHtml = `
        <div style="background:linear-gradient(135deg,#00bcd4 0%,#00acc1 100%);border-radius:16px;padding:25px;margin-bottom:30px;box-shadow:0 4px 20px rgba(0,188,212,0.3);">
          <h3 style="margin:0 0 15px 0;color:white;display:flex;align-items:center;gap:10px;"><span style="font-size:1.4em;">&#10004;</span><span>Recommended Actions</span></h3>
          <div style="background:rgba(255,255,255,0.95);border-radius:12px;padding:20px;">
            <ol style="margin:0;padding-left:25px;">
              ${nsRec.steps.map((s: string) => `<li style="margin-bottom:12px;color:#00695c;font-size:1.05em;line-height:1.5;">${escapeHtml(s)}</li>`).join("")}
            </ol>
          </div>
        </div>`;
    }
  }

  /* ================================================================
     EXECUTIVE SUMMARY (dark card)
     ================================================================ */
  let summaryHtml = "";
  if (sf) {
    const adoptionStatus = pu?.adoption_status || "unknown";
    const adoptionColors: Record<string, string> = { power_user: "#27ae60", active: "#3498db", exploring: "#f39c12", churning: "#e74c3c", not_started: "#95a5a6", unknown: "#7f8c8d" };
    const adoptionColor = adoptionColors[adoptionStatus] || "#7f8c8d";
    summaryHtml = `
      <div style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);border-radius:16px;padding:25px;margin-bottom:30px;color:white;">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:20px;">
          <div>
            <h2 style="margin:0 0 8px 0;font-size:1.8em;">${escapeHtml(sf.name)}</h2>
            <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center;">
              ${sf.lifecycle_stage ? `<span style="background:#667eea;padding:4px 12px;border-radius:20px;font-size:0.85em;">${escapeHtml(sf.lifecycle_stage)}</span>` : ""}
              ${sf.industry ? `<span style="color:#a0a0a0;">${escapeHtml(sf.industry)}</span>` : ""}
              ${sf.website ? `<a href="https://${sf.website.replace(/^https?:\/\//, "")}" target="_blank" style="color:#4facfe;text-decoration:none;">${escapeHtml(sf.website.replace(/^https?:\/\//, ""))}</a>` : ""}
            </div>
          </div>
          <div style="text-align:right;">
            <div style="font-size:2em;font-weight:bold;color:#27ae60;">${formatCurrency(sf.total_active_arr)}</div>
            <div style="font-size:0.85em;color:#a0a0a0;">Active ARR</div>
          </div>
        </div>
        ${buildLast24hHtml(overview.last_24h_activity)}
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:15px;margin-top:25px;">
          <div style="text-align:center;padding:15px;background:rgba(255,255,255,0.05);border-radius:12px;"><div style="font-size:1.8em;font-weight:bold;">${se?.total_calls || 0}</div><div style="font-size:0.85em;color:#a0a0a0;">Sales Calls</div></div>
          <div style="text-align:center;padding:15px;background:rgba(255,255,255,0.05);border-radius:12px;"><div style="font-size:1.8em;font-weight:bold;">${se?.days_in_sales_cycle || "N/A"}</div><div style="font-size:0.85em;color:#a0a0a0;">Days in Cycle</div></div>
          <div style="text-align:center;padding:15px;background:rgba(255,255,255,0.05);border-radius:12px;"><div style="font-size:1.8em;font-weight:bold;">${overview.pendo_usage?.unique_visitors || 0}</div><div style="font-size:0.85em;color:#a0a0a0;">Platform Users</div></div>
          <div style="text-align:center;padding:15px;background:rgba(255,255,255,0.05);border-radius:12px;"><div style="font-size:1.2em;font-weight:bold;color:${adoptionColor};">${adoptionStatus.replace("_", " ").toUpperCase()}</div><div style="font-size:0.85em;color:#a0a0a0;">Adoption Status</div></div>
        </div>
        ${sf.assigned_sa || sf.assigned_ai_se || sf.assigned_cse || sf.assigned_csm || sf.owner_name ? `
          <div style="margin-top:20px;padding-top:15px;border-top:1px solid rgba(255,255,255,0.1);">
            <div style="display:flex;flex-wrap:wrap;gap:10px;align-items:center;">
              <span style="color:#a0a0a0;font-size:0.85em;">Team:</span>
              ${sf.owner_name ? `<span style="background:#0176d3;padding:4px 10px;border-radius:12px;font-size:0.85em;">Owner: ${escapeHtml(sf.owner_name)}</span>` : ""}
              ${sf.assigned_sa ? `<span style="background:#667eea;padding:4px 10px;border-radius:12px;font-size:0.85em;">SA: ${escapeHtml(sf.assigned_sa)}</span>` : ""}
              ${sf.assigned_ai_se ? `<span style="background:#9b59b6;padding:4px 10px;border-radius:12px;font-size:0.85em;">AI SE: ${escapeHtml(sf.assigned_ai_se)}</span>` : ""}
              ${sf.assigned_cse ? `<span style="background:#11998e;padding:4px 10px;border-radius:12px;font-size:0.85em;">CSE: ${escapeHtml(sf.assigned_cse)}</span>` : ""}
              ${sf.assigned_csm ? `<span style="background:#4facfe;padding:4px 10px;border-radius:12px;font-size:0.85em;">CSM: ${escapeHtml(sf.assigned_csm)}</span>` : ""}
            </div>
          </div>` : ""}
      </div>`;
  } else {
    summaryHtml = `<div style="background:#f8f9fa;padding:30px;border-radius:12px;margin-bottom:25px;text-align:center;color:#666;"><h3>No Salesforce account found</h3><p>Try a different name, domain, or Salesforce ID.</p></div>`;
  }

  /* ================================================================
     SALES ENGAGEMENT CONTEXT
     ================================================================ */
  let salesEngagementHtml = "";
  if (sf) {
    const latestOpp = overview.latest_opportunity;
    let latestOppHtml = "";
    if (latestOpp) {
      latestOppHtml = `
        <div style="background:linear-gradient(135deg,#f8f9ff 0%,#e8f0fe 100%);border:2px solid #667eea;border-radius:12px;padding:20px;margin-bottom:25px;">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:15px;">
            <div><div style="font-size:0.85em;color:#667eea;font-weight:600;margin-bottom:5px;">CURRENT OPPORTUNITY</div>
              <h4 style="margin:0 0 8px 0;color:#333;">${escapeHtml(latestOpp.name)}</h4>
              <div style="display:flex;gap:15px;flex-wrap:wrap;align-items:center;">
                <span style="background:#667eea;color:white;padding:4px 12px;border-radius:15px;font-size:0.85em;">${escapeHtml(latestOpp.stage_name)}</span>
                <span style="color:#666;">Close: ${formatDate(latestOpp.close_date)}</span>
                ${latestOpp.owner_name ? `<span style="color:#666;">Owner: ${escapeHtml(latestOpp.owner_name)}</span>` : ""}
              </div></div>
            <div style="text-align:right;"><div style="font-size:1.8em;font-weight:bold;color:#27ae60;">${formatCurrency(latestOpp.amount)}</div><div style="font-size:0.85em;color:#666;">${latestOpp.probability || 0}% probability</div></div>
          </div>
          ${latestOpp.next_step ? `<div style="margin-top:15px;padding-top:15px;border-top:1px solid #d0d8e8;"><strong style="color:#555;">Next Step:</strong> <span style="color:#333;">${escapeHtml(latestOpp.next_step)}</span></div>` : ""}
        </div>`;
    }

    let allOppsHtml = "";
    if (overview.all_opportunities?.length) {
      const rows = overview.all_opportunities.map((opp: any) => {
        const sc = opp.is_won ? "#27ae60" : opp.is_closed ? "#e74c3c" : "#667eea";
        return `<tr style="border-bottom:1px solid #e9ecef;"><td style="padding:10px 14px;"><div style="font-weight:500;">${escapeHtml(opp.name)}</div>${opp.owner_name ? `<div style="font-size:0.8em;color:#888;">${escapeHtml(opp.owner_name)}</div>` : ""}</td><td style="padding:10px 14px;"><span style="background:${sc};color:white;padding:2px 8px;border-radius:10px;font-size:0.8em;">${escapeHtml(opp.stage_name)}</span></td><td style="padding:10px 14px;font-weight:500;">${formatCurrency(opp.amount)}</td><td style="padding:10px 14px;font-size:0.9em;">${formatDate(opp.close_date)}</td></tr>`;
      }).join("");
      allOppsHtml = `
        <details style="margin-bottom:25px;"><summary style="cursor:pointer;color:#667eea;font-weight:600;padding:10px;background:#f8f9fa;border-radius:8px;">View All Opportunities (${overview.all_opportunities.length})</summary>
        <div style="margin-top:10px;overflow-x:auto;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,0.05);">
          <table style="width:100%;border-collapse:collapse;background:white;font-size:0.9em;">
            <thead style="background:#f1f3f9;color:#555;"><tr><th style="padding:10px 14px;text-align:left;">Opportunity</th><th style="padding:10px 14px;text-align:left;">Stage</th><th style="padding:10px 14px;text-align:left;">Amount</th><th style="padding:10px 14px;text-align:left;">Close Date</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </div></details>`;
    }

    let dealSummaryHtml = "";
    const ds = se?.deal_summary;
    if (ds?.current_state) {
      const sentColors: Record<string, string> = { positive: "#27ae60", neutral: "#f39c12", concerned: "#e74c3c" };
      const sentColor = sentColors[ds.champion_sentiment] || "#7f8c8d";
      dealSummaryHtml = `
        <div style="background:#fafbfc;border-radius:12px;padding:20px;margin-bottom:25px;border-left:4px solid #667eea;">
          <h4 style="margin:0 0 15px 0;color:#333;display:flex;align-items:center;gap:10px;">Deal Summary ${ds.champion_sentiment ? `<span style="background:${sentColor};color:white;padding:2px 10px;border-radius:10px;font-size:0.75em;">${escapeHtml(ds.champion_sentiment)}</span>` : ""}</h4>
          <p style="margin:0 0 15px 0;color:#444;line-height:1.6;">${escapeHtml(ds.current_state)}</p>
          ${ds.blockers_identified?.length ? `<div style="background:#fff3cd;border-radius:8px;padding:12px;margin-bottom:12px;"><strong style="color:#856404;">Blockers/Concerns:</strong><ul style="margin:8px 0 0 0;padding-left:20px;color:#856404;">${ds.blockers_identified.map((b: string) => `<li style="margin-bottom:4px;">${escapeHtml(b)}</li>`).join("")}</ul></div>` : ""}
          ${ds.risk_factors?.length ? `<div style="background:#f8d7da;border-radius:8px;padding:12px;margin-bottom:12px;"><strong style="color:#721c24;">Risk Factors:</strong><ul style="margin:8px 0 0 0;padding-left:20px;color:#721c24;">${ds.risk_factors.map((r: string) => `<li style="margin-bottom:4px;">${escapeHtml(r)}</li>`).join("")}</ul></div>` : ""}
          ${ds.next_steps_from_calls?.length ? `<div style="background:#d4edda;border-radius:8px;padding:12px;"><strong style="color:#155724;">Next Steps from Calls:</strong><ul style="margin:8px 0 0 0;padding-left:20px;color:#155724;">${ds.next_steps_from_calls.slice(0, 3).map((n: string) => `<li style="margin-bottom:4px;">${escapeHtml(n)}</li>`).join("")}</ul></div>` : ""}
        </div>`;
    }

    let gongHtml = "";
    if (overview.gong_summary?.total_calls > 0) {
      const gs = overview.gong_summary;
      const themesHtml = gs.key_themes?.length ? `<div style="margin-bottom:15px;"><strong style="color:#555;margin-right:8px;">Topics:</strong>${gs.key_themes.map((t: string) => `<span style="background:#e8f0fe;color:#1967d2;padding:4px 12px;border-radius:15px;font-size:0.85em;margin-right:8px;margin-bottom:8px;display:inline-block;">${escapeHtml(t)}</span>`).join("")}</div>` : "";
      const rc = gs.recent_calls || [];
      const dates = rc.map((c: any) => c.call_date).filter(Boolean).sort();
      const dateRange = dates.length > 1 ? `${formatDate(dates[dates.length - 1])} - ${formatDate(dates[0])}` : dates.length === 1 ? formatDate(dates[0]) : "";
      gongHtml = `
        <div style="margin-bottom:20px;padding:15px;background:#f8f9fa;border-radius:10px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
            <h4 style="margin:0;color:#333;font-size:1em;">Engagement Metrics</h4>
            <span style="font-size:0.85em;color:#666;">${gs.total_calls} calls &bull; ${Math.round(gs.total_duration_minutes || 0)} min${dateRange ? ` &bull; ${dateRange}` : ""}</span>
          </div>
          <div style="display:flex;gap:20px;margin-bottom:15px;flex-wrap:wrap;">
            <div><span style="font-size:1.3em;font-weight:bold;color:#667eea;">${formatPercent(gs.avg_talk_ratio)}</span><span style="font-size:0.8em;color:#666;margin-left:4px;">talk ratio</span></div>
            <div><span style="font-size:1.3em;font-weight:bold;color:#11998e;">${gs.avg_interactivity?.toFixed(1) ?? "N/A"}</span><span style="font-size:0.8em;color:#666;margin-left:4px;">interactivity</span></div>
            <div><span style="font-size:1.3em;font-weight:bold;color:#4facfe;">${gs.days_since_last_call ?? "N/A"}</span><span style="font-size:0.8em;color:#666;margin-left:4px;">days since last call</span></div>
          </div>
          ${themesHtml}
        </div>`;
    }

    salesEngagementHtml = `
      <div style="background:white;border-radius:16px;padding:25px;margin-bottom:30px;box-shadow:0 2px 12px rgba(0,0,0,0.06);">
        <h3 style="margin:0 0 20px 0;padding-bottom:15px;border-bottom:2px solid #667eea;color:#333;"><span style="background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">Sales Engagement Context</span></h3>
        ${latestOppHtml}${dealSummaryHtml}${allOppsHtml}${gongHtml}
      </div>`;
  }

  /* ================================================================
     PRODUCT USAGE PATTERNS (Pendo)
     ================================================================ */
  let productUsageHtml = "";
  if (overview.pendo_usage?.total_events > 0) {
    const pendo = overview.pendo_usage;

    let visitorsHtml = "";
    if (pendo.recent_visitors?.length) {
      const totalUsers = pendo.unique_visitors || pendo.recent_visitors.length;
      const shown = pendo.recent_visitors.length;
      const vRows = pendo.recent_visitors.map((v: any) => `<tr style="border-bottom:1px solid #e9ecef;"><td style="padding:12px;"><div style="font-weight:500;">${escapeHtml(v.display_name || v.email || v.visitor_id)}</div></td><td style="padding:12px;">${formatDate(v.last_visit)}</td><td style="padding:12px;">${formatMinutes(v.total_minutes)}</td><td style="padding:12px;">${(v.total_events ?? 0).toLocaleString()} events</td><td style="padding:12px;">${v.visit_count} visits</td></tr>`).join("");
      const note = totalUsers > shown ? `<span style="font-size:13px;color:#666;font-weight:normal;">(showing ${shown} most recent of ${totalUsers} total)</span>` : "";
      visitorsHtml = `<div style="margin-bottom:25px;"><h4 style="margin-bottom:15px;color:#333;">Who's Using the Platform ${note}</h4><div style="overflow-x:auto;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,0.08);"><table style="width:100%;border-collapse:collapse;background:white;"><thead style="background:linear-gradient(135deg,#ff6b6b 0%,#ffa500 100%);color:white;"><tr><th style="padding:12px;text-align:left;">User</th><th style="padding:12px;text-align:left;">Last Active</th><th style="padding:12px;text-align:left;">Time on Platform</th><th style="padding:12px;text-align:left;">Activity</th><th style="padding:12px;text-align:left;">Visits</th></tr></thead><tbody>${vRows}</tbody></table></div></div>`;
    }

    let featuresHtml = "";
    if (pendo.top_features?.length) {
      const items = pendo.top_features.slice(0, 8).map((f: any) => {
        const name = f.feature_name || f.feature_id;
        const desc = getFeatureDescription(name);
        return `<div style="padding:12px 14px;background:#f8f9fa;border-radius:8px;margin-bottom:8px;"><div style="display:flex;justify-content:space-between;align-items:center;"><span style="font-weight:500;">${escapeHtml(name)}</span><div style="display:flex;gap:15px;font-size:0.9em;color:#666;"><span>${(f.event_count ?? 0).toLocaleString()} uses</span><span>${f.unique_users} users</span></div></div>${desc ? `<div style="font-size:0.85em;color:#888;margin-top:4px;">${escapeHtml(desc)}</div>` : ""}</div>`;
      }).join("");
      featuresHtml = `<div style="margin-bottom:25px;"><h4 style="margin-bottom:15px;color:#333;">Top Features Used</h4>${items}</div>`;
    }

    let trendHtml = "";
    if (pendo.weekly_trend?.length) {
      const maxE = Math.max(...pendo.weekly_trend.map((w: any) => w.events || 0));
      const bars = pendo.weekly_trend.slice(0, 8).reverse().map((w: any) => {
        const h = maxE > 0 ? Math.max(10, (w.events / maxE) * 100) : 10;
        const lbl = w.week_start ? new Date(w.week_start).toLocaleDateString("en-US", { month: "short", day: "numeric" }) : "";
        return `<div style="flex:1;display:flex;flex-direction:column;align-items:center;gap:8px;"><div style="width:100%;max-width:40px;height:${h}px;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);border-radius:4px 4px 0 0;"></div><div style="font-size:0.7em;color:#666;white-space:nowrap;">${lbl}</div></div>`;
      }).join("");
      trendHtml = `<div style="margin-bottom:25px;"><h4 style="margin-bottom:15px;color:#333;">Usage Trend (Last 8 Weeks)</h4><div style="display:flex;align-items:flex-end;height:120px;gap:8px;padding:10px;background:#f8f9fa;border-radius:12px;">${bars}</div></div>`;
    }

    productUsageHtml = `
      <div style="background:white;border-radius:16px;padding:25px;margin-bottom:30px;box-shadow:0 2px 12px rgba(0,0,0,0.06);">
        <h3 style="margin:0 0 20px 0;padding-bottom:15px;border-bottom:2px solid #ff6b6b;color:#333;"><span style="background:linear-gradient(135deg,#ff6b6b 0%,#ffa500 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">Product Usage Patterns</span></h3>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:15px;margin-bottom:25px;">
          <div style="background:linear-gradient(135deg,#ff6b6b 0%,#ffa500 100%);color:white;padding:20px;border-radius:12px;text-align:center;"><div style="font-size:2em;font-weight:bold;">${pendo.unique_visitors}</div><div style="font-size:0.85em;opacity:0.9;">Total Users</div></div>
          <div style="background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:20px;border-radius:12px;text-align:center;"><div style="font-size:2em;font-weight:bold;">${pendo.active_days_last_7 || 0}</div><div style="font-size:0.85em;opacity:0.9;">Active Days (7d)</div></div>
          <div style="background:linear-gradient(135deg,#11998e 0%,#38ef7d 100%);color:white;padding:20px;border-radius:12px;text-align:center;"><div style="font-size:2em;font-weight:bold;">${formatMinutes(pendo.total_minutes)}</div><div style="font-size:0.85em;opacity:0.9;">Time on Platform</div></div>
          <div style="background:linear-gradient(135deg,#4facfe 0%,#00f2fe 100%);color:white;padding:20px;border-radius:12px;text-align:center;"><div style="font-size:1.3em;font-weight:bold;">${pendo.days_since_last_activity ?? "N/A"}</div><div style="font-size:0.85em;opacity:0.9;">Days Since Last Use</div></div>
        </div>
        ${trendHtml}${visitorsHtml}${featuresHtml}
      </div>`;
  }

  /* ================================================================
     USER BEHAVIOR ANALYSIS
     ================================================================ */
  let userBehaviorHtml = "";
  const ub = overview.user_behavior;
  if (ub) {
    const engColors: Record<string, string> = { high: "#27ae60", medium: "#f39c12", low: "#e74c3c", unknown: "#7f8c8d" };
    const engColor = engColors[ub.engagement_level] || "#7f8c8d";

    let issuesHtml = "";
    if (ub.issues_summary || ub.user_issues?.length) {
      const typeIcons: Record<string, string> = { dead_click: "&#128433;", error: "&#9888;&#65039;", frustrated: "&#128548;", unknown: "&#10067;" };
      const items = (ub.user_issues || []).slice(0, 8).map((issue: any) => {
        const icon = typeIcons[issue.issue_type] || "&#10067;";
        const lbl = issue.issue_type === "dead_click" ? "Dead Click" : issue.issue_type === "error" ? "Error" : issue.issue_type === "frustrated" ? "Frustrated" : issue.issue_type;
        return `<div style="display:flex;align-items:flex-start;gap:12px;padding:12px;background:white;border-radius:8px;margin-bottom:8px;border:1px solid #f0f0f0;"><span style="font-size:1.3em;">${icon}</span><div style="flex:1;min-width:0;"><div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;"><span style="font-weight:600;color:#333;">${escapeHtml(lbl)}</span>${issue.recording_url ? `<a href="${escapeHtml(issue.recording_url)}" target="_blank" style="background:#6c5ce7;color:white;padding:4px 10px;border-radius:6px;font-size:0.8em;text-decoration:none;">View Recording</a>` : ""}</div><div style="font-size:0.9em;color:#666;margin-top:4px;">${issue.page_context ? `<strong>${escapeHtml(issue.page_context)}</strong> &bull; ` : ""}${escapeHtml(issue.user_email || "Unknown user")}</div>${issue.timestamp ? `<div style="font-size:0.8em;color:#999;margin-top:2px;">${new Date(issue.timestamp).toLocaleString()}</div>` : ""}</div></div>`;
      }).join("");
      issuesHtml = `<div style="background:#fff8e6;border-radius:12px;padding:20px;margin-bottom:20px;border-left:4px solid #f39c12;"><h4 style="margin:0 0 12px 0;color:#8a6914;font-size:1em;">User Experience Issues Detected</h4>${ub.issues_summary ? `<p style="margin:0 0 15px 0;color:#6b5210;line-height:1.5;font-size:0.95em;">${escapeHtml(ub.issues_summary)}</p>` : ""}${items ? `<details style="margin-top:10px;" open><summary style="cursor:pointer;color:#8a6914;font-weight:600;margin-bottom:10px;">View Issue Details with FullStory Recordings</summary>${items}</details>` : ""}</div>`;
    }

    let milestonesHtml = "";
    if (ub.adoption_milestones?.length) {
      const mItems = ub.adoption_milestones.map((m: any) => {
        const statusIcon = m.completed ? "&#10004;" : "&#11036;";
        const sc = m.completed ? "#27ae60" : "#bdc3c7";
        const ct = m.completed && m.count > 0 ? ` (${m.count}x)` : "";
        const dt = m.completed && m.last_date ? ` &bull; Last: ${escapeHtml(m.last_date)}` : "";
        return `<div style="display:flex;align-items:center;gap:10px;padding:10px 14px;background:${m.completed ? "#e8f8f0" : "#f8f9fa"};border-radius:8px;border-left:3px solid ${sc};"><span style="font-size:1.1em;">${statusIcon}</span><div style="flex:1;"><span style="font-weight:500;color:${m.completed ? "#27ae60" : "#7f8c8d"};">${escapeHtml(m.name)}</span>${m.completed ? `<span style="font-size:0.85em;color:#888;">${ct}${dt}</span>` : ""}</div></div>`;
      }).join("");
      milestonesHtml = `<div style="margin-bottom:20px;"><h4 style="margin:0 0 12px 0;color:#333;font-size:1em;">Adoption Milestones</h4><div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;">${mItems}</div></div>`;
    }

    let recsHtml = "";
    const filteredRecs = (ub.recommendations || []).filter((r: any) => r.category !== "Next Steps");
    if (filteredRecs.length) {
      const catColors: Record<string, { bg: string; border: string; icon: string }> = {
        "Re-engagement": { bg: "#fff3e0", border: "#ff9800", icon: "&#128260;" },
        Expansion: { bg: "#e3f2fd", border: "#2196f3", icon: "&#128200;" },
        Deepening: { bg: "#f3e5f5", border: "#9c27b0", icon: "&#127919;" },
        Competitive: { bg: "#fce4ec", border: "#e91e63", icon: "&#9876;&#65039;" },
        "Internal Resources": { bg: "#e8f5e9", border: "#4caf50", icon: "&#128101;" },
      };
      const recItems = filteredRecs.map((rec: any) => {
        if (typeof rec === "string") return `<div style="background:#e8f5e9;border-radius:8px;padding:12px;margin-bottom:10px;border-left:4px solid #27ae60;"><span style="color:#2e7d32;">${escapeHtml(rec)}</span></div>`;
        const style = catColors[rec.category] || { bg: "#f5f5f5", border: "#9e9e9e", icon: "&#128161;" };
        let content = "";
        if (rec.category === "Competitive" && rec.competitive_messaging) {
          content = rec.competitive_messaging.map((cm: any) => {
            const mentionedIn = cm.mentioned_in?.length ? `<div style="font-size:0.8em;color:#888;margin-bottom:8px;">Mentioned in: ${cm.mentioned_in.slice(0, 3).map((m: string) => escapeHtml(m)).join(", ")}${cm.mention_count > 3 ? ` (+${cm.mention_count - 3} more)` : ""}</div>` : "";
            const noteS = cm.note ? `<div style="background:#fff3cd;border:1px solid #ffc107;padding:10px;border-radius:6px;margin-bottom:10px;font-size:0.9em;">${escapeHtml(cm.note)}</div>` : "";
            const whatS = cm.what_they_said && cm.what_they_said !== "No specific mentions captured" && cm.what_they_said !== "See context below" ? `<div style="background:#e3f2fd;border-left:3px solid #1976d2;padding:10px;border-radius:0 6px 6px 0;margin-bottom:12px;"><div style="font-weight:600;color:#1565c0;font-size:0.85em;margin-bottom:4px;">What They're Interested In:</div><div style="color:#333;font-size:0.9em;">${escapeHtml(cm.what_they_said)}</div></div>` : "";
            const targS = cm.targeted_response ? `<div style="background:#e8f5e9;border:1px solid #4caf50;padding:12px;border-radius:6px;margin-bottom:12px;"><div style="font-weight:600;color:#2e7d32;font-size:0.85em;margin-bottom:6px;">Recommended Response:</div><div style="color:#1b5e20;font-size:0.95em;line-height:1.5;">${escapeHtml(cm.targeted_response)}</div></div>` : "";
            const rawS = cm.raw_contexts?.length ? `<details style="margin-bottom:10px;"><summary style="cursor:pointer;font-size:0.8em;color:#666;padding:4px;">See exact quotes from calls (${cm.raw_contexts.length})</summary><div style="background:#fafafa;padding:10px;border-radius:4px;margin-top:6px;font-size:0.8em;color:#555;">${cm.raw_contexts.map((ctx: string) => `<div style="margin-bottom:8px;padding-left:8px;border-left:2px solid #ddd;">${escapeHtml(ctx)}</div>`).join("")}</div></details>` : "";
            const genS = cm.differentiator ? `<details style="margin-top:8px;"><summary style="cursor:pointer;font-size:0.8em;color:#888;padding:4px;">Generic positioning (backup)</summary><div style="background:#f5f5f5;padding:10px;border-radius:4px;margin-top:6px;"><div style="font-size:0.85em;color:#555;margin-bottom:8px;">${escapeHtml(cm.differentiator)}</div><div style="background:#fff8e1;padding:8px;border-radius:4px;font-size:0.8em;"><strong>Generic Talking Point:</strong> ${escapeHtml(cm.talking_point)}</div></div></details>` : "";
            return `<div style="background:white;border-radius:6px;padding:14px;margin-top:10px;border:1px solid #e0e0e0;box-shadow:0 1px 3px rgba(0,0,0,0.08);"><div style="font-weight:600;color:#c62828;margin-bottom:6px;font-size:1.05em;">vs ${escapeHtml(cm.competitor)}</div>${mentionedIn}${noteS}${whatS}${targS}${rawS}${genS}</div>`;
          }).join("");
        } else if (rec.category === "Internal Resources" && rec.contacts) {
          content = `<div style="margin-top:8px;">${escapeHtml(rec.description || "")}</div><div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:10px;">${rec.contacts.map((c: any) => `<a href="mailto:${escapeHtml(c.email)}" style="background:white;border:1px solid #4caf50;padding:6px 12px;border-radius:20px;text-decoration:none;color:#2e7d32;font-size:0.9em;">${escapeHtml(c.name)} &#9993;&#65039;</a>`).join("")}</div>`;
        } else if (rec.description) {
          content = `<div style="margin-top:6px;color:#555;font-size:0.95em;">${escapeHtml(rec.description)}</div>`;
        }
        return `<div style="background:${style.bg};border-radius:10px;padding:15px;margin-bottom:12px;border-left:4px solid ${style.border};"><div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;"><span style="font-size:1.2em;">${style.icon}</span><span style="font-weight:600;color:#333;">${escapeHtml(rec.title || rec.category)}</span><span style="background:${style.border};color:white;padding:2px 8px;border-radius:10px;font-size:0.75em;">${escapeHtml(rec.category)}</span></div>${content}</div>`;
      }).join("");
      recsHtml = `<div style="margin-top:20px;"><h4 style="margin:0 0 15px 0;color:#333;font-size:1.1em;">Actionable Recommendations</h4>${recItems}</div>`;
    }

    userBehaviorHtml = `
      <div style="background:white;border-radius:16px;padding:25px;margin-bottom:30px;box-shadow:0 2px 12px rgba(0,0,0,0.06);">
        <h3 style="margin:0 0 20px 0;padding-bottom:15px;border-bottom:2px solid #6c5ce7;color:#333;">
          <span style="background:linear-gradient(135deg,#6c5ce7 0%,#a29bfe 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">User Behavior Analysis</span>
          <span style="background:${engColor};color:white;padding:3px 10px;border-radius:10px;font-size:0.6em;margin-left:10px;vertical-align:middle;">${ub.engagement_level.toUpperCase()} ENGAGEMENT</span>
        </h3>
        ${milestonesHtml}
        <div style="background:#f8f9fa;border-radius:12px;padding:20px;margin-bottom:20px;"><h4 style="margin:0 0 10px 0;color:#333;font-size:1em;">What Users Are Doing</h4><p style="margin:0;color:#555;line-height:1.6;">${escapeHtml(ub.summary)}</p></div>
        ${ub.hypothesis ? `<div style="background:#e8f4fd;border-radius:12px;padding:20px;margin-bottom:20px;border-left:4px solid #3498db;"><h4 style="margin:0 0 10px 0;color:#2980b9;font-size:1em;">Hypothesis: What They're Trying to Accomplish</h4><p style="margin:0;color:#34495e;line-height:1.6;">${escapeHtml(ub.hypothesis)}</p></div>` : ""}
        ${ub.key_workflows_used?.length ? `<div style="margin-bottom:20px;"><h4 style="margin:0 0 12px 0;color:#333;font-size:1em;">Key Workflows Used</h4><div style="display:flex;flex-wrap:wrap;gap:8px;">${ub.key_workflows_used.map((w: string) => `<span style="background:#e8f0fe;color:#1967d2;padding:6px 14px;border-radius:20px;font-size:0.9em;">${escapeHtml(w)}</span>`).join("")}</div></div>` : ""}
        ${issuesHtml}
        ${ub.critical_issues?.length ? `<div style="background:#ffeaea;border-radius:12px;padding:20px;margin-bottom:20px;border-left:4px solid #e74c3c;"><h4 style="margin:0 0 12px 0;color:#c0392b;font-size:1em;">Critical Issues</h4>${ub.critical_issues.map((i: any) => `<div style="background:white;padding:12px;border-radius:8px;margin-bottom:8px;"><strong style="color:#e74c3c;">${escapeHtml(i.type)}:</strong> <span style="color:#555;">${escapeHtml(i.description)}</span></div>`).join("")}</div>` : ""}
        ${recsHtml}
      </div>`;
  }

  /* ---- Assemble final output ---- */
  return `
    <div style="margin-bottom:20px;">
      <span style="color:#666;">Searched by ${escapeHtml(overview.lookup_method)}: </span>
      <strong>${escapeHtml(overview.lookup_value)}</strong>
      <span style="margin-left:20px;color:#666;">Sources: </span>${sourcesHtml}
    </div>
    ${errorsHtml}${nextStepsHtml}${summaryHtml}${salesEngagementHtml}${productUsageHtml}${userBehaviorHtml}`;
}

/* ---------- Component ---------- */

export default function ProspectTab({
  onLoading,
  onResult,
  resolveAccount,
}: {
  onLoading: (msg: string) => void;
  onResult: (html: string) => void;
  resolveAccount: ResolveAccountFn;
}) {
  const [accountName, setAccountName] = useState("");
  const [accountDomain, setAccountDomain] = useState("");
  const [sfdcAccountId, setSfdcAccountId] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  async function submit() {
    if (!accountName.trim() && !accountDomain.trim() && !sfdcAccountId.trim()) {
      alert("Please enter at least one search criteria");
      return;
    }
    if (isLoading) return;
    setIsLoading(true);
    onLoading("Resolving account...");

    let nm = accountName.trim();
    let dom = accountDomain.trim();
    let sid = sfdcAccountId.trim();
    try {
      if (nm && !sid) {
        const resolved = await resolveAccount({
          accountName: nm,
          accountDomain: dom,
          sfdcAccountId: sid,
        });
        if (!resolved.proceed) {
          setIsLoading(false);
          onResult("");
          return;
        }
        nm = (resolved.accountName || nm).trim();
        dom = (resolved.accountDomain ?? dom).trim();
        sid = (resolved.sfdcAccountId ?? sid).trim();
        if (nm !== accountName.trim()) setAccountName(nm);
        if (sid !== sfdcAccountId.trim()) setSfdcAccountId(sid);
      }
      onLoading("Fetching prospect data from BigQuery...\nThis may take 30-60 seconds");
      const body: Record<string, string> = {};
      if (nm) body.account_name = nm;
      if (dom) body.domain = dom;
      if (sid) body.sfdc_account_id = sid;
      const overview = await apiPost("/api/prospect-overview", body);
      onResult(buildProspectHtml(overview));
    } catch (err: any) {
      alert("Error: " + (err.message ?? String(err)));
      onResult("");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <>
      <div className="input-section">
        <label htmlFor="accountName">Account Name</label>
        <input type="text" id="accountName" placeholder="e.g., Acme Corp, OpenAI, Stripe" value={accountName} onChange={(e) => setAccountName(e.target.value)} />
        <p className="help-text">Search by account/company name (fuzzy matching supported).</p>
      </div>
      <div className="input-section" style={{ marginTop: 15 }}>
        <label htmlFor="accountDomain">Email Domain (Optional)</label>
        <input type="text" id="accountDomain" placeholder="e.g., acme.com, openai.com" value={accountDomain} onChange={(e) => setAccountDomain(e.target.value)} />
        <p className="help-text">When set with account name, <strong>both</strong> must match.</p>
      </div>
      <div className="input-section" style={{ marginTop: 15 }}>
        <label htmlFor="sfdcAccountId">Salesforce Account ID (Optional)</label>
        <input type="text" id="sfdcAccountId" placeholder="e.g., 001..." value={sfdcAccountId} onChange={(e) => setSfdcAccountId(e.target.value)} />
        <p className="help-text">Exact match on Salesforce Account ID (most precise).</p>
      </div>
      <p className="help-text" style={{ marginTop: 15, padding: 12, background: "#e3f2fd", borderRadius: 8 }}>
        {"📊 "}<strong>Prospect Overview</strong> pulls data from BigQuery including Salesforce, Gong, Pendo, FullStory.
      </p>
      <div className="button-group">
        <button className="btn-primary" onClick={submit} disabled={isLoading}>
          {isLoading ? "Loading..." : "Get Prospect Overview"}
        </button>
      </div>
    </>
  );
}
