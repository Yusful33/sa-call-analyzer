"use client";

import { useState, useRef } from "react";
import { apiPost, exportScriptUrl } from "@/lib/api";
import { escapeHtml } from "@/lib/helpers";
import type { ResolveAccountFn } from "@/lib/accountResolve";

/* eslint-disable @typescript-eslint/no-explicit-any */

const BASE = process.env.NEXT_PUBLIC_LEGACY_API_URL ?? "http://localhost:8080";

export default function DemoTab({
  onLoading,
  onResult,
  resolveAccount,
}: {
  onLoading: (msg: string) => void;
  onResult: (html: string) => void;
  resolveAccount: ResolveAccountFn;
}) {
  const [accountName, setAccountName] = useState("");
  const [modelSelect, setModelSelect] = useState("gpt-5.2");
  const [projectName, setProjectName] = useState("");
  const [spaceId, setSpaceId] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [additionalContext, setAdditionalContext] = useState("");
  const [phase, setPhase] = useState<"input" | "confirm">("input");
  const classifyRef = useRef<any>(null);

  async function classify() {
    if (!accountName.trim()) { alert("Please enter a prospect/account name."); return; }
    const resolved = await resolveAccount({
      accountName: accountName.trim(),
      accountDomain: "",
      sfdcAccountId: "",
    });
    if (!resolved.proceed) return;
    const useName = (resolved.accountName || accountName).trim();
    if (useName !== accountName.trim()) setAccountName(useName);
    onLoading("Classifying prospect use case...");
    try {
      const body: Record<string, unknown> = { account_name: useName };
      if (additionalContext.trim()) body.additional_context = additionalContext.trim();
      const data = await apiPost<any>("/api/classify-demo", body);
      classifyRef.current = data;
      setPhase("confirm");
      onResult("");
      onLoading("");
    } catch (err: any) {
      alert("Error: " + (err.message ?? String(err)));
      onLoading("");
    }
  }

  async function confirmAndGenerate() {
    const classification = classifyRef.current;
    if (!classification) return;
    setPhase("input");
    onLoading("Starting trace generation...");

    const pn = projectName.trim() || accountName.toLowerCase().replace(/\s+/g, "-") + "-demo";
    const payload: Record<string, unknown> = {
      account_name: accountName.trim(),
      project_name: pn,
      model: modelSelect,
      use_case: classification.use_case,
      framework: classification.framework,
    };
    if (additionalContext.trim()) payload.additional_context = additionalContext.trim();
    if (spaceId.trim()) payload.arize_space_id = spaceId.trim();
    if (apiKey.trim()) payload.arize_api_key = apiKey.trim();

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 600000);
      const res = await fetch(`${BASE}/api/generate-demo-stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail ?? `Request failed: ${res.status}`);
      }

      const reader = res.body?.getReader();
      if (!reader) throw new Error("No response body");
      const decoder = new TextDecoder();
      let data: any = null;
      let buf = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const events = buf.split(/\n\n+/);
        buf = events.pop() || "";
        for (const chunk of events) {
          const eventLine = chunk.match(/^event: (\w+)/m);
          const dataLine = chunk.match(/^data: (.+)/m);
          if (!eventLine || !dataLine) continue;
          const eventType = eventLine[1];
          try {
            const parsed = JSON.parse(dataLine[1].trim());
            if (eventType === "progress" && parsed.message) {
              onLoading(parsed.message);
            } else if (eventType === "done") {
              data = parsed;
              break;
            } else if (eventType === "error") {
              throw new Error(parsed.detail || "Pipeline failed");
            }
          } catch (e) {
            if (e instanceof SyntaxError) continue;
            throw e;
          }
        }
        if (data) break;
      }

      if (!data) { onResult(`<div style="color:red;padding:20px;"><strong>Error:</strong> No response received.</div>`); return; }

      let html = "<h2>Demo Traces Generated</h2>";
      html += `<div style="padding:15px;background:#e8f5e9;border-radius:8px;margin-bottom:15px;">`;
      html += `<p><strong>Prospect:</strong> ${escapeHtml(data.prospect_name ?? accountName)}</p>`;
      html += `<p><strong>Industry:</strong> ${escapeHtml(data.industry || "N/A")}</p>`;
      html += `<p><strong>Use Case:</strong> ${escapeHtml(data.use_case)}</p>`;
      if (data.framework) html += `<p><strong>Framework:</strong> ${escapeHtml(data.framework)}</p>`;
      if (data.use_case_reasoning) html += `<p><strong>Why:</strong> ${escapeHtml(data.use_case_reasoning)}</p>`;
      html += `<p><strong>Model:</strong> ${escapeHtml(data.model_used ?? modelSelect)}</p>`;
      html += `<p><strong>LLM Calls:</strong> ${data.llm_calls_made ?? "N/A"}</p>`;
      html += `<p><strong>Arize Project:</strong> ${escapeHtml(data.project_name ?? pn)}</p>`;
      if (data.arize_url) html += `<p><a href="${escapeHtml(data.arize_url)}" target="_blank" rel="noreferrer" style="color:#1976d2;font-weight:bold;">Open in Arize</a></p>`;
      if (data.eval_message != null && data.eval_message !== "") {
        const evalStyle = data.eval_created ? "color:#2e7d32;" : "color:#ed6c02;";
        html += `<p style="${evalStyle}margin-top:8px;"><strong>Online evals:</strong> ${escapeHtml(String(data.eval_message))}</p>`;
      }

      if (data.result?.traces_generated > 0) {
        const scriptParams = new URLSearchParams({
          use_case: data.use_case,
          framework: data.framework || "langgraph",
          model: data.model_used ?? modelSelect,
          project_name: data.project_name ?? pn,
        });
        html += `<p style="margin-top:10px;"><a href="${exportScriptUrl(Object.fromEntries(scriptParams))}" download style="display:inline-block;padding:10px 20px;background:#1976d2;color:white;border-radius:6px;text-decoration:none;font-weight:600;">Download Trace Demo Script</a></p>`;
        html += `<p style="color:#888;font-size:0.85em;margin-top:4px;">Standalone .py file for prospects: run it, try a few turns, then open Arize to see traces. Use <code>--batch</code> for 10 synthetic traces.</p>`;
      }
      html += `</div>`;

      if (data.result) {
        html += `<h3>Pipeline Result</h3><pre style="background:#f5f5f5;padding:15px;border-radius:8px;overflow-x:auto;max-height:400px;">${escapeHtml(JSON.stringify(data.result, null, 2))}</pre>`;
      }

      const errMsg = data.error_message || data.result?.error;
      const noTraces = data.result?.traces_generated === 0;
      if (noTraces && errMsg) {
        html += `<p style="color:#c62828;margin-top:15px;"><strong>No traces were sent.</strong> ${escapeHtml(String(errMsg))}</p>`;
      } else if (!noTraces) {
        html += `<p style="color:#666;margin-top:15px;">Traces were sent to Arize automatically. Open the link above to walk through the demo.</p>`;
      }
      onResult(html);
    } catch (err: any) {
      const msg = err.name === "AbortError" ? "Request timed out after 10 minutes." : (err.message ?? String(err));
      onResult(`<div style="color:red;padding:20px;"><strong>Error:</strong> ${escapeHtml(msg)}</div>`);
    }
  }

  if (phase === "confirm" && classifyRef.current) {
    const c = classifyRef.current;
    return (
      <div style={{ marginTop: 10 }}>
        <h3 style={{ marginBottom: 5 }}>Confirm Use Case</h3>
        <p style={{ color: "#666", fontSize: "0.9em", marginBottom: 8 }}>{c.reasoning}</p>
        {c.data_sources_note && <p style={{ color: "#888", fontSize: "0.8em", marginBottom: 15, fontStyle: "italic" }}>{c.data_sources_note}</p>}
        <div style={{ display: "flex", gap: 15 }}>
          <div style={{ flex: 1 }}>
            <label style={{ fontSize: "0.9em" }}>Use Case</label>
            <select defaultValue={c.use_case} onChange={(e) => { classifyRef.current = { ...classifyRef.current, use_case: e.target.value }; }}>
              {(c.available_use_cases ?? []).map((uc: any) => <option key={uc.value} value={uc.value}>{uc.label}</option>)}
            </select>
          </div>
          <div style={{ flex: 1 }}>
            <label style={{ fontSize: "0.9em" }}>Framework</label>
            <select defaultValue={c.framework} onChange={(e) => { classifyRef.current = { ...classifyRef.current, framework: e.target.value }; }}>
              {(c.available_frameworks ?? []).map((fw: any) => <option key={fw.value} value={fw.value}>{fw.label}</option>)}
            </select>
          </div>
        </div>
        {c.industry && <p style={{ color: "#888", fontSize: "0.85em", marginTop: 10 }}>Industry: {c.industry}</p>}
        <div style={{ marginTop: 15, display: "flex", gap: 10 }}>
          <button className="btn-primary" onClick={confirmAndGenerate}>Confirm &amp; Generate</button>
          <button className="btn-secondary" onClick={() => setPhase("input")}>Cancel</button>
        </div>
      </div>
    );
  }

  return (
    <>
      <div className="input-section">
        <label htmlFor="demoAccountName">Prospect / Account Name</label>
        <input type="text" id="demoAccountName" placeholder="e.g., Acme Corp, Stripe, Datadog" value={accountName} onChange={(e) => setAccountName(e.target.value)} />
        <p className="help-text">Enter a prospect name. We&rsquo;ll generate tailored demo traces.</p>
      </div>
      <div className="input-section" style={{ marginTop: 15 }}>
        <label htmlFor="demoModelSelect">LLM Model for Demo Traces</label>
        <select id="demoModelSelect" value={modelSelect} onChange={(e) => setModelSelect(e.target.value)}>
          <option value="gpt-5.2">GPT-5.2</option>
          <option value="claude-opus-4-6">Claude Opus 4.6</option>
          <option value="claude-sonnet-4-20250514">Claude Sonnet 4</option>
          <option value="claude-haiku-4-5">Claude Haiku 4.5</option>
          <option value="gpt-4.1">GPT-4.1</option>
          <option value="gpt-4.1-mini">GPT-4.1 Mini</option>
          <option value="gpt-4o">GPT-4o</option>
          <option value="gpt-4o-mini">GPT-4o Mini</option>
        </select>
      </div>
      <div className="input-section" style={{ marginTop: 15 }}>
        <label htmlFor="demoProjectName">Arize Project Name (optional)</label>
        <input type="text" id="demoProjectName" placeholder="e.g., acme-demo" value={projectName} onChange={(e) => setProjectName(e.target.value)} />
      </div>
      <div className="input-section" style={{ marginTop: 15 }}>
        <label htmlFor="demoArizeSpaceId">Arize Space ID (optional)</label>
        <input type="text" id="demoArizeSpaceId" placeholder="Leave blank to use app default" value={spaceId} onChange={(e) => setSpaceId(e.target.value)} />
      </div>
      <div className="input-section" style={{ marginTop: 15 }}>
        <label htmlFor="demoArizeApiKey">Arize API Key (optional)</label>
        <input type="password" id="demoArizeApiKey" placeholder="Leave blank to use app default" value={apiKey} onChange={(e) => setApiKey(e.target.value)} />
      </div>
      <div className="input-section" style={{ marginTop: 15 }}>
        <label htmlFor="demoAdditionalContext">Additional context (optional)</label>
        <textarea id="demoAdditionalContext" rows={3} placeholder="e.g., HR chatbot for employees..." value={additionalContext} onChange={(e) => setAdditionalContext(e.target.value)} />
      </div>
      <p className="help-text" style={{ marginTop: 15, padding: 12, background: "#fff3e0", borderRadius: 8 }}>
        {"🎯 "}<strong>Custom Demo Builder</strong> fetches the prospect profile, identifies their use case, then generates instrumented demo traces.
      </p>
      <div className="button-group">
        <button className="btn-primary" onClick={classify}>Generate Demo Traces</button>
      </div>
    </>
  );
}
