"use client";

import { useState, useRef } from "react";
import { apiPost, apiPostBlob } from "@/lib/api";
import { escapeHtml, downloadBlob } from "@/lib/helpers";
import { useToast } from "@/components/Toast";

/* eslint-disable @typescript-eslint/no-explicit-any */

function buildEvidenceHtml(evidence: any[]): string {
  if (!evidence?.length) return "";
  return evidence.map(e => `<div class="evidence-item">${e.timestamp ? `<span class="timestamp-badge">${escapeHtml(e.timestamp)}</span>` : ""}<strong>${escapeHtml((e.criteria_name || "").replace(/_/g, " "))}</strong>${e.speaker ? ` - ${escapeHtml(e.speaker)}` : ""}${e.conversation_snippet ? `<div class="snippet">${escapeHtml(e.conversation_snippet)}</div>` : ""}</div>`).join("");
}

function buildMissedOpportunitiesHtml(opportunities: any[]): string {
  if (!opportunities?.length) return "";
  return opportunities.map(o => `<div class="missed-opportunity-item">${o.timestamp ? `<span class="timestamp-badge">${escapeHtml(o.timestamp)}</span>` : ""}<strong>Missed: ${escapeHtml((o.criteria_name || "").replace(/_/g, " "))}</strong><div style="margin-top:6px;">${escapeHtml(o.context)}</div><div class="suggested-question">&ldquo;${escapeHtml(o.suggested_question)}&rdquo;</div><div class="why-important"><strong>Why:</strong> ${escapeHtml(o.why_important)}</div></div>`).join("");
}

function countCriteriaStatus(obj: any) {
  let captured = 0, notCaptured = 0;
  const exclude = ["people_impact_understood", "process_impact_understood", "technology_impact_understood"];
  for (const [key, value] of Object.entries(obj)) {
    if (typeof value === "boolean" && !exclude.includes(key)) {
      if (value) captured++; else notCaptured++;
    }
  }
  return { captured, notCaptured };
}

function buildCriteriaSectionHtml(title: string, criteriaObj: any, checklistHtml: string): string {
  const { captured, notCaptured } = countCriteriaStatus(criteriaObj);
  const evidenceItems = criteriaObj.evidence?.length ?? 0;
  const missedItems = criteriaObj.missed_opportunities?.length ?? 0;
  return `<details><summary class="criteria-section-header"><span>${escapeHtml(title)}${captured > 0 ? `<span class="evidence-count">${captured} captured</span>` : ""}${notCaptured > 0 ? `<span class="opportunity-count">${notCaptured} gaps</span>` : ""}</span></summary><ul class="criteria-checklist">${checklistHtml}</ul>${evidenceItems > 0 ? `<div style="margin-top:12px;"><strong style="color:#27ae60;">Evidence Captured:</strong>${buildEvidenceHtml(criteriaObj.evidence)}</div>` : ""}${missedItems > 0 ? `<div style="margin-top:12px;"><strong style="color:#e74c3c;">Missed Opportunities:</strong>${buildMissedOpportunitiesHtml(criteriaObj.missed_opportunities)}</div>` : ""}</details>`;
}

function buildClassificationHtml(c: any): string {
  if (!c) return "";
  const callTypeLabels: Record<string, string> = { discovery: "Discovery Call", poc_scoping: "PoC Scoping Call", mixed: "Mixed Call", unclear: "Unclear" };
  const callTypeLabel = callTypeLabels[c.call_type] || c.call_type;
  const getPC = (s: number) => s >= 70 ? "high" : s >= 40 ? "medium" : "low";
  const ci = (v: any) => v === true ? '<span class="check-icon">&#10003;</span>' : '<span class="x-icon">&#10007;</span>';
  const ciOpt = (v: any) => v === true ? '<span class="check-icon">&#10003;</span>' : v === false ? '<span class="x-icon">&#10007;</span>' : '&#9675;';

  let discoveryCriteriaHtml = "";
  if (c.discovery_criteria) {
    const dc = c.discovery_criteria;
    const painChecklist = `${dc.pain_current_state.primary_use_case ? `<li style="margin-bottom:8px;opacity:0.9;"><strong>Focus:</strong> ${escapeHtml(dc.pain_current_state.primary_use_case)}</li>` : ""}
      <li>${ci(dc.pain_current_state.prompt_model_iteration_understood)} Prompt/model iteration understood (Dev)</li>
      <li>${ci(dc.pain_current_state.debugging_process_documented)} Debugging process documented (Prod)</li>
      <li>${ci(dc.pain_current_state.situation_understood)} Situation understood</li>
      <li>${ci(dc.pain_current_state.resolution_attempts_documented)} Resolution attempts documented</li>
      <li>${ci(dc.pain_current_state.outcomes_documented)} Outcomes documented</li>
      <li>${ci(dc.pain_current_state.frequency_quantified)} Frequency quantified</li>
      <li>${ci(dc.pain_current_state.duration_quantified)} Duration quantified</li>
      <li>${ci(dc.pain_current_state.impact_quantified)} Impact quantified</li>
      <li>${ci(dc.pain_current_state.mttd_mttr_quantified)} MTTD/MTTR quantified</li>
      <li>${ci(dc.pain_current_state.experiment_time_quantified)} Experiment time quantified</li>`;
    const stakeholderChecklist = `<li>${ci(dc.stakeholder_map.technical_champion_identified)} Technical champion identified</li>
      <li>${ci(dc.stakeholder_map.technical_champion_engaged)} Technical champion engaged</li>
      <li>${ci(dc.stakeholder_map.economic_buyer_identified)} Economic buyer identified</li>
      <li>${ci(dc.stakeholder_map.decision_maker_confirmed)} Decision maker confirmed</li>`;
    const rcChecklist = `<li>${ci(dc.required_capabilities.top_rcs_ranked)} Top RCs ranked</li>
      <li>${ci(dc.required_capabilities.must_have_vs_nice_to_have_distinguished)} Must-have vs nice-to-have</li>
      <li>${ci(dc.required_capabilities.deal_breakers_identified)} Deal-breakers identified</li>
      <li style="margin-top:10px;font-weight:600;opacity:0.8;">Core Capabilities:</li>
      <li style="padding-left:12px;">${ciOpt(dc.required_capabilities.llm_agent_tracing_important)} LLM/Agent Tracing</li>
      <li style="padding-left:12px;">${ciOpt(dc.required_capabilities.llm_evaluations_important)} LLM Evaluations</li>
      <li style="padding-left:12px;">${ciOpt(dc.required_capabilities.production_monitoring_important)} Production Monitoring</li>
      <li style="padding-left:12px;">${ciOpt(dc.required_capabilities.prompt_management_important)} Prompt Management</li>
      <li style="padding-left:12px;">${ciOpt(dc.required_capabilities.prompt_experimentation_important)} Prompt Experimentation</li>
      <li style="padding-left:12px;">${ciOpt(dc.required_capabilities.monitoring_important)} Monitoring</li>
      <li style="padding-left:12px;">${ciOpt(dc.required_capabilities.compliance_important)} Compliance (SOC2, SSO, GDPR)</li>`;
    const compChecklist = `<li>${ci(dc.competitive_landscape.current_tools_evaluated)} Current tools evaluated</li>
      <li>${ci(dc.competitive_landscape.why_looking_vs_staying)} Why looking vs staying</li>
      <li>${ci(dc.competitive_landscape.key_differentiators_identified)} Key differentiators identified</li>
      ${dc.competitive_landscape.tools_mentioned?.length ? `<li style="margin-top:8px;opacity:0.8;">Tools: ${dc.competitive_landscape.tools_mentioned.map((t: string) => escapeHtml(t)).join(", ")}</li>` : ""}`;

    discoveryCriteriaHtml = `<div class="criteria-card"><h4>Discovery Criteria</h4><div class="progress-bar"><div class="progress-fill ${getPC(c.discovery_completion_score)}" style="width:${c.discovery_completion_score}%"></div></div><div style="text-align:center;margin-bottom:15px;font-size:1.2em;">${c.discovery_completion_score.toFixed(0)}% Complete</div>${buildCriteriaSectionHtml("Pain & Current State", dc.pain_current_state, painChecklist)}${buildCriteriaSectionHtml("Stakeholder Map", dc.stakeholder_map, stakeholderChecklist)}${buildCriteriaSectionHtml("Required Capabilities", dc.required_capabilities, rcChecklist)}${buildCriteriaSectionHtml("Competitive Landscape", dc.competitive_landscape, compChecklist)}</div>`;
  }

  let pocCriteriaHtml = "";
  if (c.poc_scoping_criteria) {
    const pc = c.poc_scoping_criteria;
    const useCaseChecklist = `<li>${ci(pc.use_case_scoped.llm_applications_selected)} LLM application selected</li>
      <li>${ci(pc.use_case_scoped.environment_decided)} Environment decided ${pc.use_case_scoped.environment_type ? `(${escapeHtml(pc.use_case_scoped.environment_type)})` : ""}</li>
      <li>${ci(pc.use_case_scoped.trace_volume_estimated)} Trace volume estimated ${pc.use_case_scoped.estimated_volume ? `(${escapeHtml(pc.use_case_scoped.estimated_volume)})` : ""}</li>
      <li>${ci(pc.use_case_scoped.llm_provider_identified)} LLM Provider identified ${pc.use_case_scoped.llm_provider ? `(${escapeHtml(pc.use_case_scoped.llm_provider)})` : ""}</li>
      <li>${ci(pc.use_case_scoped.integration_complexity_assessed)} Integration complexity assessed</li>
      ${pc.use_case_scoped.applications_list?.length ? `<li style="margin-top:8px;opacity:0.8;">Apps: ${pc.use_case_scoped.applications_list.map((a: string) => escapeHtml(a)).join(", ")}</li>` : ""}`;
    const implChecklist = `<li>${ci(pc.implementation_requirements.data_residency_confirmed)} Data residency confirmed ${pc.implementation_requirements.deployment_model ? `(${escapeHtml(pc.implementation_requirements.deployment_model)})` : ""}</li>
      <li>${ci(pc.implementation_requirements.blockers_identified)} Blockers identified</li>
      ${pc.implementation_requirements.blockers_list?.length ? `<li style="margin-top:8px;opacity:0.8;">Blockers: ${pc.implementation_requirements.blockers_list.map((b: string) => escapeHtml(b)).join(", ")}</li>` : ""}`;
    const metricsChecklist = `<li>${ci(pc.metrics_success_criteria.specific_metrics_defined)} Specific metrics defined</li>
      <li>${ci(pc.metrics_success_criteria.baseline_captured)} Baseline captured</li>
      <li>${ci(pc.metrics_success_criteria.success_measurement_agreed)} Success measurement agreed</li>
      <li>${ci(pc.metrics_success_criteria.competitive_favorable_criteria)} Criteria favorable vs competitors</li>
      ${pc.metrics_success_criteria.example_metrics?.length ? `<li style="margin-top:8px;opacity:0.8;">Metrics: ${pc.metrics_success_criteria.example_metrics.map((m: string) => escapeHtml(m)).join(", ")}</li>` : ""}`;
    const timelineChecklist = `<li>${ci(pc.timeline_milestones.poc_duration_defined)} PoC duration defined ${pc.timeline_milestones.duration_weeks ? `(${pc.timeline_milestones.duration_weeks} weeks)` : ""}</li>
      <li>${ci(pc.timeline_milestones.key_milestones_with_dates)} Key milestones with dates</li>
      <li>${ci(pc.timeline_milestones.decision_date_committed)} Decision date committed ${pc.timeline_milestones.decision_date ? `(${escapeHtml(pc.timeline_milestones.decision_date)})` : ""}</li>
      <li>${ci(pc.timeline_milestones.next_steps_discussed)} Next steps discussed</li>`;
    const resourcesChecklist = `<li>${ci(pc.resources_committed.engineering_resources_allocated)} Engineering resources allocated</li>
      <li>${ci(pc.resources_committed.checkin_cadence_established)} Check-in cadence established ${pc.resources_committed.cadence ? `(${escapeHtml(pc.resources_committed.cadence)})` : ""}</li>
      <li>${ci(pc.resources_committed.communication_channel_created)} Communication channel created</li>
      ${pc.resources_committed.resource_names?.length ? `<li style="margin-top:8px;opacity:0.8;">Resources: ${pc.resources_committed.resource_names.map((r: string) => escapeHtml(r)).join(", ")}</li>` : ""}`;

    pocCriteriaHtml = `<div class="criteria-card"><h4>PoC Scoping Criteria</h4><div class="progress-bar"><div class="progress-fill ${getPC(c.poc_scoping_completion_score)}" style="width:${c.poc_scoping_completion_score}%"></div></div><div style="text-align:center;margin-bottom:15px;font-size:1.2em;">${c.poc_scoping_completion_score.toFixed(0)}% Complete</div>${buildCriteriaSectionHtml("Use Case Scoped", pc.use_case_scoped, useCaseChecklist)}${buildCriteriaSectionHtml("Implementation Requirements", pc.implementation_requirements, implChecklist)}${buildCriteriaSectionHtml("Metrics & Success Criteria", pc.metrics_success_criteria, metricsChecklist)}${buildCriteriaSectionHtml("Timeline & Milestones", pc.timeline_milestones, timelineChecklist)}${buildCriteriaSectionHtml("Resources Committed", pc.resources_committed, resourcesChecklist)}</div>`;
  }

  let missingHtml = "";
  const ct = c.call_type;
  const me = c.missing_elements || {};
  const showDisc = ct === "discovery" || ct === "mixed" || ct === "unclear";
  const showPoc = ct === "poc_scoping" || ct === "mixed" || ct === "unclear";
  const dm = me.discovery || [];
  const pm = me.poc_scoping || [];
  if ((showDisc && dm.length) || (showPoc && pm.length)) {
    missingHtml = `<div class="missing-elements"><h4>Missing Elements</h4>`;
    if (showDisc && dm.length) missingHtml += `<div><h5>Discovery Gaps</h5><ul>${dm.map((e: string) => `<li>${escapeHtml(e)}</li>`).join("")}</ul></div>`;
    if (showPoc && pm.length) missingHtml += `<div><h5>PoC Scoping Gaps</h5><ul>${pm.map((e: string) => `<li>${escapeHtml(e)}</li>`).join("")}</ul></div>`;
    missingHtml += `</div>`;
  }

  let recsHtml = "";
  if (c.recommendations?.length) {
    recsHtml = `<div class="recommendations-section"><h4>Recommendations for Next Call</h4><ul>${c.recommendations.map((r: string) => `<li>${escapeHtml(r)}</li>`).join("")}</ul></div>`;
  }

  return `<div class="classification-section"><div class="classification-header"><span class="call-type-badge ${c.call_type}">${escapeHtml(callTypeLabel)}</span><span class="confidence-badge">Confidence: ${escapeHtml(c.confidence)}</span></div><div class="classification-reasoning">${escapeHtml(c.reasoning)}</div><div class="criteria-grid">${discoveryCriteriaHtml}${pocCriteriaHtml}</div>${missingHtml}${recsHtml}</div>`;
}

function buildInsightsHtml(insights: any[]): string {
  return insights.map(i => `<div class="insight ${i.severity ?? ""}"><div class="insight-header"><span class="insight-category">${escapeHtml(i.category ?? "")}</span>${i.timestamp ? `<span class="insight-timestamp">${escapeHtml(i.timestamp)}</span>` : ""}</div>${i.conversation_snippet ? `<div class="conversation-snippet"><strong>Conversation:</strong><pre>${escapeHtml(i.conversation_snippet)}</pre></div>` : ""}<div class="insight-section"><strong>What Happened:</strong>${escapeHtml(i.what_happened ?? "")}</div><div class="insight-section"><strong>Why It Matters:</strong>${escapeHtml(i.why_it_matters ?? "")}</div><div class="insight-section"><strong>Better Approach:</strong>${escapeHtml(i.better_approach ?? "")}</div>${i.example_phrasing ? `<div class="example-phrasing"><strong>Example: </strong>&ldquo;${escapeHtml(i.example_phrasing)}&rdquo;</div>` : ""}</div>`).join("");
}

function displayResults(data: any): string {
  const classHtml = buildClassificationHtml(data.call_classification);
  const insightsHtml = buildInsightsHtml(data.top_insights ?? []);
  return `${data.recap_data ? `<div class="recap-section" style="margin-bottom:25px;"><h3>Recap Slide Deck</h3><p style="margin-bottom:15px;color:#666;">Download a <strong>two-slide PowerPoint deck</strong> with a recap and probing questions.</p><button class="btn-recap" id="generateRecapBtn">Download Recap Slides (.pptx)</button><p id="recapStatus" style="margin-top:10px;display:none;"></p></div>` : ""}${classHtml}<h2>${escapeHtml(data.call_summary ?? "")}</h2><h3 class="section-title">Top Actionable Insights</h3>${insightsHtml}<h3 class="section-title">Strengths</h3><ul class="list">${(data.strengths ?? []).map((s: string) => `<li>${escapeHtml(s)}</li>`).join("")}</ul><h3 class="section-title">Areas for Improvement</h3><ul class="list improvement-list">${(data.improvement_areas ?? []).map((a: string) => `<li>${escapeHtml(a)}</li>`).join("")}</ul><h3 class="section-title">Key Moments</h3><ul class="list">${(data.key_moments ?? []).map((m: any) => `<li><strong>${escapeHtml(m.timestamp ?? "")}</strong>: ${escapeHtml(m.description ?? "")}</li>`).join("")}</ul>`;
}

export default function GongTab({ onLoading, onResult }: { onLoading: (msg: string) => void; onResult: (html: string) => void }) {
  const toast = useToast();
  const [gongUrl, setGongUrl] = useState("");
  const [model, setModel] = useState("gpt-4o-mini");
  const recapRef = useRef<any>(null);

  async function submit() {
    if (!gongUrl.trim()) { toast.warning("Please paste a Gong URL first"); return; }
    onLoading("Fetching transcript from Gong API...");
    try {
      const data = await apiPost<any>("/api/analyze", { sa_name: null, model, gong_url: gongUrl.trim() }, { timeout: 240000 });
      recapRef.current = data.recap_data ?? null;
      onResult(displayResults(data));
      setTimeout(() => {
        const btn = document.getElementById("generateRecapBtn");
        if (btn) btn.onclick = async () => {
          if (!recapRef.current) { toast.warning("No recap data available"); return; }
          btn.textContent = "Generating...";
          (btn as HTMLButtonElement).disabled = true;
          const status = document.getElementById("recapStatus");
          if (status) { status.style.display = "block"; status.textContent = "Creating your PowerPoint presentation..."; status.style.color = "#666"; }
          try {
            const { blob, filename } = await apiPostBlob(
              "/api/generate-recap-slide",
              recapRef.current,
              "Recap_Slide.pptx"
            );
            downloadBlob(blob, filename || "Recap_Slide.pptx");
            if (status) { status.innerHTML = "PowerPoint downloaded!"; status.style.color = "#27ae60"; }
          } catch (e: any) {
            if (status) { status.textContent = "Error: " + (e.message ?? String(e)); status.style.color = "#e74c3c"; }
          } finally {
            btn.textContent = "Download Recap Slides (.pptx)";
            (btn as HTMLButtonElement).disabled = false;
          }
        };
      }, 100);
    } catch (err: any) {
      const msg = err.name === "AbortError" ? "Request timed out after 4 minutes." : (err.message ?? String(err));
      toast.error("Error: " + msg);
      onResult("");
    }
  }

  return (
    <>
      <div className="input-section">
        <label htmlFor="gongUrl">Gong Call URL</label>
        <input type="text" id="gongUrl" placeholder="https://app.gong.io/call?id=YOUR_CALL_ID" value={gongUrl} onChange={(e) => setGongUrl(e.target.value)} />
        <p className="help-text">Paste a Gong call URL and we&rsquo;ll automatically fetch and analyze the transcript.</p>
      </div>
      <div className="input-section" style={{ marginTop: 15 }}>
        <label htmlFor="modelSelect">AI Model</label>
        <select id="modelSelect" value={model} onChange={(e) => setModel(e.target.value)}>
          <option value="gpt-4o-mini">GPT-4o Mini (Fast)</option>
          <option value="gpt-5.2">GPT-5.2 (Latest)</option>
          <option value="gpt-4o">GPT-4o (Most Capable)</option>
          <option value="claude-haiku-4-5">Claude Haiku 4.5</option>
          <option value="claude-sonnet-4">Claude Sonnet 4</option>
        </select>
      </div>
      <div className="button-group">
        <button className="btn-primary" onClick={submit}>Analyze Call</button>
      </div>
    </>
  );
}
