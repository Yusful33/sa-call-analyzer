"use client";

import { useState } from "react";
import { apiPost } from "@/lib/api";
import { escapeHtml } from "@/lib/helpers";
import type { ResolveAccountFn } from "@/lib/accountResolve";

/* eslint-disable @typescript-eslint/no-explicit-any */

function renderHypothesisResults(data: any): string {
  if (!data || !data.result) return '<div style="color:red;padding:20px;">Invalid response: no result data.</div>';
  const r = data.result;
  const research = r.research || {};
  const hypotheses: any[] = r.hypotheses || [];
  const competitive = r.competitive_context || {};
  const reasoning: string[] = data.agent_reasoning || [];

  let html = "<h2>Hypothesis Research Results</h2>";

  const qualityColors: Record<string, string> = { high: "#4caf50", medium: "#ff9800", low: "#f44336", insufficient: "#9e9e9e" };
  const quality = r.research_quality || "unknown";
  html += `<div style="margin-bottom:15px;"><span style="background:${qualityColors[quality] || "#9e9e9e"};color:white;padding:4px 12px;border-radius:12px;font-size:13px;font-weight:600;">Research Quality: ${quality.toUpperCase()}</span></div>`;

  if (r.warnings?.length) {
    html += `<div style="padding:12px;background:#fff3e0;border-radius:8px;margin-bottom:15px;">`;
    r.warnings.forEach((w: string) => { html += `<p style="margin:4px 0;">&#9888;&#65039; ${escapeHtml(w)}</p>`; });
    html += `</div>`;
  }

  html += `<div style="padding:15px;background:#e3f2fd;border-radius:8px;margin-bottom:15px;">`;
  html += `<h3 style="margin-top:0;">Company Overview</h3>`;
  html += `<p><strong>Company:</strong> ${escapeHtml(research.company_name || r.company_name || "")}</p>`;
  if (research.industry) html += `<p><strong>Industry:</strong> ${escapeHtml(research.industry)}</p>`;
  if (research.employee_count) html += `<p><strong>Employees:</strong> ${Number(research.employee_count).toLocaleString()}</p>`;
  if (research.ai_ml_confidence) html += `<p><strong>AI/ML Confidence:</strong> ${escapeHtml(research.ai_ml_confidence)}</p>`;
  if (research.competitive_situation) html += `<p><strong>Competitive Situation:</strong> ${escapeHtml(research.competitive_situation)}</p>`;
  if (research.company_summary) html += `<p>${escapeHtml(research.company_summary)}</p>`;
  if (research.ai_ml_signals?.length) {
    html += `<p style="font-weight:600;margin-bottom:8px;">Top Signals:</p>`;
    research.ai_ml_signals.slice(0, 5).forEach((s: any) => {
      const confColor: Record<string, string> = { high: "#4caf50", medium: "#ff9800", low: "#f44336" };
      const cc = confColor[s.confidence] || "#9e9e9e";
      const typeLabel = (s.signal_type || "signal").replace(/_/g, " ");
      const sourceLink = s.source_url ? ` <a href="${escapeHtml(s.source_url)}" target="_blank" rel="noopener" style="color:#1565c0;font-size:12px;">View source &rarr;</a>` : "";
      html += `<div style="padding:10px 14px;background:white;border-left:4px solid ${cc};border-radius:0 6px 6px 0;margin-bottom:8px;"><div style="display:flex;gap:6px;align-items:center;margin-bottom:4px;"><span style="background:#e8eaf6;padding:2px 8px;border-radius:8px;font-size:11px;font-weight:600;text-transform:uppercase;">${escapeHtml(typeLabel)}</span><span style="background:${cc};color:white;padding:2px 8px;border-radius:8px;font-size:11px;">${escapeHtml(s.confidence)}</span></div><span style="font-size:14px;color:#333;">${escapeHtml(s.evidence || "")}</span>${sourceLink}</div>`;
    });
  }
  html += `</div>`;

  if (hypotheses.length) {
    html += `<h3>Hypotheses (${hypotheses.length})</h3>`;
    hypotheses.forEach((h: any) => {
      const confColor: Record<string, string> = { high: "#4caf50", medium: "#ff9800", low: "#f44336" };
      const cc = confColor[h.confidence] || "#9e9e9e";
      const valueIcons: Record<string, string> = { reduce_risk: "&#128737;&#65039;", increase_efficiency: "&#9889;", increase_revenue: "&#128200;", reduce_cost: "&#128176;" };
      const vi = valueIcons[h.value_category] || "&#128203;";
      html += `<div style="padding:15px;border:1px solid #e0e0e0;border-radius:8px;margin-bottom:12px;">`;
      html += `<div style="display:flex;gap:8px;margin-bottom:8px;"><span style="background:${cc};color:white;padding:2px 10px;border-radius:10px;font-size:12px;">${escapeHtml(h.confidence)}</span><span style="background:#e8eaf6;padding:2px 10px;border-radius:10px;font-size:12px;">${vi} ${escapeHtml((h.value_category || "").replace(/_/g, " "))}</span></div>`;
      html += `<p style="font-weight:600;font-size:15px;">${escapeHtml(h.hypothesis)}</p>`;

      if (h.current_state) html += `<p><strong>Current State:</strong> ${escapeHtml(h.current_state)}</p>`;
      if (h.future_state) html += `<p><strong>Future State:</strong> ${escapeHtml(h.future_state)}</p>`;
      if (h.required_capabilities?.length) html += `<p><strong>Required Capabilities:</strong> ${h.required_capabilities.map((c: string) => escapeHtml(c)).join(", ")}</p>`;
      if (h.negative_consequences) html += `<p><strong>Risk of Inaction:</strong> ${escapeHtml(h.negative_consequences)}</p>`;

      if (h.supporting_signals?.length) {
        html += `<div style="margin-top:10px;"><strong>Supporting Evidence:</strong></div>`;
        h.supporting_signals.forEach((sig: any) => {
          const sc: Record<string, string> = { high: "#4caf50", medium: "#ff9800", low: "#f44336" };
          const sigC = sc[sig.confidence] || "#9e9e9e";
          const sigLink = sig.source_url ? ` <a href="${escapeHtml(sig.source_url)}" target="_blank" rel="noopener" style="color:#1565c0;font-size:12px;">View source &rarr;</a>` : "";
          html += `<div style="padding:6px 10px;background:#f5f5f5;border-left:3px solid ${sigC};border-radius:0 4px 4px 0;margin:4px 0;font-size:13px;">${escapeHtml(sig.description || "")}${sigLink}</div>`;
        });
      }

      if (h.discovery_questions?.length) {
        html += `<details style="margin-top:8px;"><summary style="cursor:pointer;font-weight:600;">Discovery Questions (${h.discovery_questions.length})</summary><div style="margin-top:8px;">`;
        h.discovery_questions.forEach((q: any) => {
          if (typeof q === "string") {
            html += `<div style="padding:8px 12px;background:#f5f5f5;border-radius:6px;margin-bottom:6px;">${escapeHtml(q)}</div>`;
          } else {
            html += `<div style="padding:8px 12px;background:#f5f5f5;border-radius:6px;margin-bottom:6px;"><div style="font-weight:500;">${escapeHtml(q.question || JSON.stringify(q))}</div>${q.rationale ? `<div style="font-size:12px;color:#666;margin-top:4px;">${escapeHtml(q.rationale)}</div>` : ""}</div>`;
          }
        });
        html += `</div></details>`;
      }

      if (h.similar_customers?.length) {
        html += `<details style="margin-top:8px;"><summary style="cursor:pointer;font-weight:600;">Similar Customers (${h.similar_customers.length})</summary><div style="margin-top:8px;">`;
        h.similar_customers.forEach((c: any) => {
          if (typeof c === "string") {
            html += `<div style="padding:8px 12px;background:#f5f5f5;border-radius:6px;margin-bottom:6px;">${escapeHtml(c)}</div>`;
          } else {
            html += `<div style="padding:8px 12px;background:#f5f5f5;border-radius:6px;margin-bottom:6px;"><div style="font-weight:500;">${escapeHtml(c.customer_name || c.name || c.company || "Unknown")}</div>`;
            const details = [c.industry, c.use_case].filter(Boolean).join(" &middot; ");
            if (details) html += `<div style="font-size:12px;color:#666;margin-top:2px;">${escapeHtml(details)}</div>`;
            if (c.outcome) html += `<div style="font-size:12px;color:#4caf50;margin-top:2px;">${escapeHtml(c.outcome)}</div>`;
            html += `</div>`;
          }
        });
        html += `</div></details>`;
      }
      html += `</div>`;
    });
  }

  if (competitive?.situation) {
    html += `<div style="padding:15px;background:#fce4ec;border-radius:8px;margin-bottom:15px;">`;
    html += `<h3 style="margin-top:0;">Competitive Context</h3>`;
    html += `<p><strong>Situation:</strong> ${escapeHtml(competitive.situation.replace(/_/g, " "))}</p>`;
    if (competitive.detected_competitor) html += `<p><strong>Detected Competitor:</strong> ${escapeHtml(competitive.detected_competitor)}</p>`;
    if (competitive.positioning) html += `<p style="margin:10px 0;">${escapeHtml(competitive.positioning)}</p>`;
    if (competitive.advantages?.length) {
      html += `<p style="font-weight:600;margin-top:10px;">Advantages:</p><ul style="margin:4px 0 8px 20px;">`;
      competitive.advantages.forEach((a: string) => { html += `<li style="margin-bottom:4px;">${escapeHtml(a)}</li>`; });
      html += `</ul>`;
    }
    if (competitive.watch_outs?.length) {
      html += `<p style="font-weight:600;">Watch Outs:</p><ul style="margin:4px 0 8px 20px;">`;
      competitive.watch_outs.forEach((w: string) => { html += `<li style="margin-bottom:4px;">${escapeHtml(w)}</li>`; });
      html += `</ul>`;
    }
    if (competitive.key_questions?.length) {
      html += `<details style="margin-top:8px;"><summary style="cursor:pointer;font-weight:600;">Key Questions (${competitive.key_questions.length})</summary><ul style="margin-top:6px;">`;
      competitive.key_questions.forEach((q: string) => { html += `<li style="margin-bottom:4px;">${escapeHtml(q)}</li>`; });
      html += `</ul></details>`;
    }
    html += `</div>`;
  }

  if (reasoning.length) {
    html += `<details style="margin-top:15px;"><summary style="cursor:pointer;font-weight:600;">Agent Reasoning (${reasoning.length} steps)</summary><ol style="margin-top:8px;">`;
    reasoning.forEach((step: string) => { html += `<li style="margin-bottom:4px;color:#555;">${escapeHtml(step)}</li>`; });
    html += `</ol></details>`;
  }

  if (r.processing_time_seconds) html += `<p style="color:#999;margin-top:15px;font-size:13px;">Research completed in ${r.processing_time_seconds.toFixed(1)}s</p>`;

  return html;
}

export default function HypothesisTab({
  onLoading,
  onResult,
  resolveAccount,
}: {
  onLoading: (msg: string) => void;
  onResult: (html: string) => void;
  resolveAccount: ResolveAccountFn;
}) {
  const [companyName, setCompanyName] = useState("");
  const [domain, setDomain] = useState("");

  async function submit() {
    if (!companyName.trim()) { alert("Please enter a company name"); return; }
    const resolved = await resolveAccount({
      accountName: companyName.trim(),
      accountDomain: domain.trim(),
      sfdcAccountId: "",
    });
    if (!resolved.proceed) return;
    const cn = (resolved.accountName || companyName).trim();
    if (cn !== companyName.trim()) setCompanyName(cn);
    onLoading(`AI agent researching ${cn}... This may take 30-60 seconds.`);
    try {
      const body: Record<string, unknown> = { company_name: cn };
      if (domain.trim()) body.company_domain = domain.trim();
      const data = await apiPost("/api/hypothesis-research", body);
      onResult(renderHypothesisResults(data));
    } catch (err: any) {
      alert("Error: " + (err.message ?? String(err)));
      onResult("");
    }
  }

  return (
    <>
      <div className="input-section">
        <label htmlFor="hypCompanyName">Company Name</label>
        <input type="text" id="hypCompanyName" placeholder="e.g., Stripe, Datadog, Snowflake" value={companyName} onChange={(e) => setCompanyName(e.target.value)} />
        <p className="help-text">Enter a company name to research their AI/ML signals and generate data-driven hypotheses.</p>
      </div>
      <div className="input-section" style={{ marginTop: 15 }}>
        <label htmlFor="hypDomain">Domain (Optional)</label>
        <input type="text" id="hypDomain" placeholder="e.g., stripe.com" value={domain} onChange={(e) => setDomain(e.target.value)} />
        <p className="help-text">Company domain for more precise web research.</p>
      </div>
      <p className="help-text" style={{ marginTop: 15, padding: 12, background: "#ede7f6", borderRadius: 8 }}>
        {"🔬 "}<strong>Hypothesis Research</strong> uses an AI agent to research companies via web search and CRM data, then generates ranked hypotheses with discovery questions, competitive context, and proof points.
      </p>
      <div className="button-group">
        <button className="btn-primary" onClick={submit}>Research Company</button>
      </div>
    </>
  );
}
