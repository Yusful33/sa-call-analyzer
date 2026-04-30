"use client";

import { useState } from "react";
import { apiPost } from "@/lib/api";
import { escapeHtml } from "@/lib/helpers";
import type { ResolveAccountFn } from "@/lib/accountResolve";

const SKILL_DOC =
  "https://github.com/Arize-ai/solutions-resources/blob/main/.claude/skills/arize-synthetic-demo/SKILL.md";

/* eslint-disable @typescript-eslint/no-explicit-any */

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
  const [additionalContext, setAdditionalContext] = useState("");
  const [phase, setPhase] = useState<"input" | "confirm">("input");
  const [classified, setClassified] = useState<any>(null);

  async function classify() {
    if (!accountName.trim()) {
      alert("Please enter a prospect/account name.");
      return;
    }
    const resolved = await resolveAccount({
      accountName: accountName.trim(),
      accountDomain: "",
      sfdcAccountId: "",
    });
    if (!resolved.proceed) return;
    const useName = (resolved.accountName || accountName).trim();
    if (useName !== accountName.trim()) setAccountName(useName);
    onLoading("Classifying prospect (for Claude skill inputs)...");
    try {
      const body: Record<string, unknown> = { account_name: useName };
      if (additionalContext.trim()) body.additional_context = additionalContext.trim();
      const data = await apiPost<any>("/api/classify-demo", body);
      setClassified(data);
      setPhase("confirm");
      onResult("");
      onLoading("");
    } catch (err: any) {
      alert("Error: " + (err.message ?? String(err)));
      onLoading("");
    }
  }

  function copyPrompt() {
    const t = classified?.synthetic_demo_skill?.suggested_prompt_for_claude;
    if (!t) return;
    void navigator.clipboard.writeText(t);
  }

  if (phase === "confirm" && classified) {
    const sk = classified.synthetic_demo_skill;
    return (
      <div style={{ marginTop: 10 }}>
        <h3 style={{ marginBottom: 8 }}>Suggested for the Arize synthetic demo skill</h3>
        <p style={{ color: "#666", fontSize: "0.92em", marginBottom: 12 }}>
          This app no longer runs in-browser trace synthesis. Use the{" "}
          <strong>arize-synthetic-demo</strong> Claude skill to scaffold <code>generator.py</code>, datasets, and AX
          uploads. Open the skill doc and paste the prompt below.
        </p>
        <p>
          <a href={SKILL_DOC} target="_blank" rel="noreferrer" style={{ fontWeight: 600 }}>
            View SKILL.md (Solutions resources)
          </a>
        </p>

        {classified.data_sources_note && (
          <p style={{ color: "#888", fontSize: "0.85em", fontStyle: "italic" }}>{classified.data_sources_note}</p>
        )}

        <div style={{ marginTop: 14, display: "flex", gap: 12, flexWrap: "wrap", alignItems: "flex-end" }}>
          <div style={{ flex: 1, minWidth: 200 }}>
            <label style={{ fontSize: "0.88em" }}>Use case</label>
            <select
              defaultValue={classified.use_case}
              onChange={(e) => setClassified({ ...classified, use_case: e.target.value })}
            >
              {(classified.available_use_cases ?? []).map((uc: any) => (
                <option key={uc.value} value={uc.value}>
                  {uc.label}
                </option>
              ))}
            </select>
          </div>
          <div style={{ flex: 1, minWidth: 200 }}>
            <label style={{ fontSize: "0.88em" }}>Framework</label>
            <select
              defaultValue={classified.framework}
              onChange={(e) => setClassified({ ...classified, framework: e.target.value })}
            >
              {(classified.available_frameworks ?? []).map((fw: any) => (
                <option key={fw.value} value={fw.value}>
                  {fw.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        {sk?.recommended_inputs && (
          <details style={{ marginTop: 14 }}>
            <summary style={{ cursor: "pointer", fontWeight: 600 }}>Skill input mapping (JSON)</summary>
            <pre
              style={{
                background: "#f5f5f5",
                padding: 12,
                borderRadius: 8,
                overflow: "auto",
                maxHeight: 280,
                fontSize: "0.82em",
              }}
            >
              {escapeHtml(JSON.stringify(sk.recommended_inputs, null, 2))}
            </pre>
          </details>
        )}

        <div style={{ marginTop: 16 }}>
          <label style={{ fontSize: "0.88em", fontWeight: 600 }}>Prompt to paste into Claude (invoke arize-synthetic-demo)</label>
          <textarea
            readOnly
            rows={6}
            style={{ width: "100%", marginTop: 6, fontFamily: "monospace", fontSize: "0.85em" }}
            value={sk?.suggested_prompt_for_claude ?? ""}
          />
          <div style={{ marginTop: 8, display: "flex", gap: 8, flexWrap: "wrap" }}>
            <button type="button" className="btn-primary" onClick={copyPrompt}>
              Copy prompt
            </button>
            <button type="button" className="btn-secondary" onClick={() => setPhase("input")}>
              Back
            </button>
          </div>
        </div>

        {sk?.next_steps && (
          <ul style={{ marginTop: 16, color: "#555", paddingLeft: 20 }}>
            {(sk.next_steps as string[]).map((s: string) => (
              <li key={s.slice(0, 40)} style={{ marginBottom: 6 }}>
                {s}
              </li>
            ))}
          </ul>
        )}
      </div>
    );
  }

  return (
    <>
      <div className="input-section">
        <label htmlFor="demoAccountName">Prospect / Account Name</label>
        <input
          type="text"
          id="demoAccountName"
          placeholder="e.g., Acme Corp"
          value={accountName}
          onChange={(e) => setAccountName(e.target.value)}
        />
      </div>
      <div className="input-section" style={{ marginTop: 15 }}>
        <label htmlFor="demoAdditionalContext">Additional context (optional)</label>
        <textarea
          id="demoAdditionalContext"
          rows={3}
          placeholder="e.g., retail banking fraud triage agent, HR 1:1 prep chatbot..."
          value={additionalContext}
          onChange={(e) => setAdditionalContext(e.target.value)}
        />
      </div>
      <p
        className="help-text"
        style={{ marginTop: 12, padding: 12, background: "#e3f2fd", borderRadius: 8, fontSize: "0.92em" }}
      >
        Custom synthetic traces are built with the <strong>arize-synthetic-demo</strong> skill (see{" "}
        <a href={SKILL_DOC} target="_blank" rel="noreferrer">
          SKILL.md
        </a>
        ). This tab classifies the account and builds a ready-to-paste prompt for Claude.
      </p>
      <div className="button-group">
        <button className="btn-primary" type="button" onClick={classify}>
          Get skill prompt
        </button>
      </div>
    </>
  );
}
