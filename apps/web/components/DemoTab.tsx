"use client";

import { useState } from "react";
import { apiPost } from "@/lib/api";
import { escapeHtml } from "@/lib/helpers";
import type { ResolveAccountFn } from "@/lib/accountResolve";

const SKILL_DOC =
  "https://github.com/Arize-ai/solutions-resources/blob/main/.claude/skills/arize-synthetic-demo/SKILL.md";

/** SKILL.md required `framework` — LLM / agent stack (no "auto"). */
const SKILL_FRAMEWORK_OPTIONS: { value: string; label: string }[] = [
  { value: "openai", label: "openai" },
  { value: "anthropic", label: "anthropic" },
  { value: "bedrock", label: "bedrock" },
  { value: "vertex", label: "vertex" },
  { value: "adk", label: "adk" },
  { value: "langchain", label: "langchain" },
  { value: "langgraph", label: "langgraph" },
  { value: "crewai", label: "crewai" },
  { value: "generic", label: "generic" },
];

/** SKILL.md required `agent_architecture`. */
const AGENT_ARCHITECTURE_OPTIONS: { value: string; label: string }[] = [
  { value: "single_agent", label: "single_agent" },
  { value: "multi_agent_coordinator", label: "multi_agent_coordinator" },
  { value: "retrieval_pipeline", label: "retrieval_pipeline" },
  { value: "rag_rerank", label: "rag_rerank" },
  { value: "guarded_rag", label: "guarded_rag" },
];

const NUM_TRACE_OPTIONS = [100, 250, 500, 1000, 2000] as const;

/** SKILL.md optional `scenarios` subset. */
const SCENARIO_OPTIONS: { value: string; label: string }[] = [
  { value: "happy_path", label: "happy_path" },
  { value: "tool_failure", label: "tool_failure" },
  { value: "guardrail_denial", label: "guardrail_denial" },
  { value: "ambiguity", label: "ambiguity" },
  { value: "execution_failure", label: "execution_failure" },
  { value: "retry", label: "retry" },
  { value: "poisoned_tokens", label: "poisoned_tokens" },
  { value: "no_llm_needed", label: "no_llm_needed" },
];

/* eslint-disable @typescript-eslint/no-explicit-any */

type SkillFormState = {
  industryOrUseCase: string;
  outputDir: string;
  skillFramework: string;
  agentArchitecture: string;
  numTraces: number;
  withEvals: boolean;
  withDatasetAndExperiments: boolean;
  scenarios: string[];
  toolsText: string;
  promptTemplateNames: string;
  sessionSizeMin: number;
  sessionSizeMax: number;
  promptVersionsJson: string;
  experimentGridModels: string;
};

const defaultSkillForm = (): SkillFormState => ({
  industryOrUseCase: "",
  outputDir: "",
  skillFramework: "langgraph",
  agentArchitecture: "single_agent",
  numTraces: 500,
  withEvals: true,
  withDatasetAndExperiments: true,
  scenarios: [],
  toolsText: "",
  promptTemplateNames: "",
  sessionSizeMin: 3,
  sessionSizeMax: 6,
  promptVersionsJson: "",
  experimentGridModels: "",
});

function buildClassifyBody(
  accountName: string,
  additionalContext: string,
  skill: SkillFormState
): Record<string, unknown> {
  const body: Record<string, unknown> = {
    account_name: accountName,
    industry_or_use_case: skill.industryOrUseCase.trim(),
    skill_framework: skill.skillFramework,
    agent_architecture: skill.agentArchitecture,
    num_traces: skill.numTraces,
    with_evals: skill.withEvals,
    with_dataset_and_experiments: skill.withDatasetAndExperiments,
  };
  const ctx = additionalContext.trim();
  if (ctx) body.additional_context = ctx;
  const od = skill.outputDir.trim();
  if (od) body.output_dir = od;
  if (skill.scenarios.length) body.scenarios = skill.scenarios;
  const tools = skill.toolsText.trim();
  if (tools) body.tools_text = tools;
  const ptn = skill.promptTemplateNames.trim();
  if (ptn) body.prompt_template_names = ptn;
  body.session_size_min = skill.sessionSizeMin;
  body.session_size_max = skill.sessionSizeMax;
  const pv = skill.promptVersionsJson.trim();
  if (pv) body.prompt_versions_json = pv;
  const egm = skill.experimentGridModels.trim();
  if (egm) body.experiment_grid_models = egm;
  return body;
}

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
  const [skillForm, setSkillForm] = useState<SkillFormState>(defaultSkillForm);

  const [phase, setPhase] = useState<"input" | "result">("input");
  const [classified, setClassified] = useState<any>(null);

  async function runClassify(body: Record<string, unknown>) {
    onLoading("Building skill prompt (CRM/Gong hints + SKILL.md inputs)...");
    try {
      const data = await apiPost<any>("/api/classify-demo", body);
      setClassified(data);
      setPhase("result");
      onResult("");
      onLoading("");
    } catch (err: any) {
      alert("Error: " + (err.message ?? String(err)));
      onLoading("");
    }
  }

  async function classifyFromInput() {
    if (!accountName.trim()) {
      alert("Please enter a prospect/account name (SKILL.md: company_name).");
      return;
    }
    if (!skillForm.industryOrUseCase.trim()) {
      alert("Please enter industry / use case (SKILL.md: industry_or_use_case).");
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

    const body = buildClassifyBody(useName, additionalContext, skillForm);
    await runClassify(body);
  }

  function copyPrompt() {
    const t = classified?.synthetic_demo_skill?.suggested_prompt_for_claude;
    if (!t) return;
    void navigator.clipboard.writeText(t);
  }

  if (phase === "result" && classified) {
    const sk = classified.synthetic_demo_skill;
    return (
      <div style={{ marginTop: 10 }}>
        <h3 style={{ marginBottom: 8 }}>Suggested for the Arize synthetic demo skill</h3>
        <p style={{ color: "#666", fontSize: "0.92em", marginBottom: 12 }}>
          This app does not synthesize traces. Use the <strong>arize-synthetic-demo</strong> Claude skill to scaffold{" "}
          <code>generator.py</code>, datasets, and AX uploads.
        </p>
        <p>
          <a href={SKILL_DOC} target="_blank" rel="noreferrer" style={{ fontWeight: 600 }}>
            View SKILL.md (Solutions resources)
          </a>
        </p>

        {classified.data_sources_note && (
          <p style={{ color: "#888", fontSize: "0.85em", fontStyle: "italic" }}>{classified.data_sources_note}</p>
        )}

        <p style={{ marginTop: 12, fontSize: "0.9em", color: "#555" }}>
          <strong>CRM/Gong classification (for context only):</strong> {classified.use_case} / {classified.framework}
          {classified.reasoning ? ` — ${classified.reasoning}` : ""}
        </p>

        {sk?.recommended_inputs && (
          <details style={{ marginTop: 14 }}>
            <summary style={{ cursor: "pointer", fontWeight: 600 }}>Skill input mapping (JSON)</summary>
            <pre
              style={{
                background: "#f5f5f5",
                padding: 12,
                borderRadius: 8,
                overflow: "auto",
                maxHeight: 320,
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
            rows={8}
            style={{ width: "100%", marginTop: 6, fontFamily: "monospace", fontSize: "0.85em" }}
            value={sk?.suggested_prompt_for_claude ?? ""}
          />
          <div style={{ marginTop: 8, display: "flex", gap: 8, flexWrap: "wrap" }}>
            <button type="button" className="btn-primary" onClick={copyPrompt}>
              Copy prompt
            </button>
            <button
              type="button"
              className="btn-secondary"
              onClick={() => {
                setPhase("input");
                setClassified(null);
              }}
            >
              Back
            </button>
          </div>
        </div>

        {sk?.next_steps && (
          <ul style={{ marginTop: 16, color: "#555", paddingLeft: 20 }}>
            {(sk.next_steps as string[]).map((s: string) => (
              <li key={s.slice(0, 48)} style={{ marginBottom: 6 }}>
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
        <label htmlFor="demoAccountName">company_name (SKILL.md)</label>
        <input
          type="text"
          id="demoAccountName"
          placeholder="e.g., Acme Bank"
          value={accountName}
          onChange={(e) => setAccountName(e.target.value)}
        />
        <p className="help-text" style={{ marginTop: 6 }}>
          Used for BigQuery Gong/Salesforce hints shown below the skill JSON. The skill inputs below follow SKILL.md
          exactly, except <strong>additional context</strong> at the bottom (app-only field).
        </p>
      </div>

      <div className="input-section" style={{ marginTop: 15 }}>
        <label htmlFor="demoIndustryUseCase">industry_or_use_case (SKILL.md — required)</label>
        <textarea
          id="demoIndustryUseCase"
          rows={3}
          placeholder='e.g., retail banking fraud triage agent, insurance claims adjudication'
          value={skillForm.industryOrUseCase}
          onChange={(e) => setSkillForm((s) => ({ ...s, industryOrUseCase: e.target.value }))}
        />
      </div>

      <div className="input-section" style={{ marginTop: 15 }}>
        <label htmlFor="demoSkillFramework">framework (SKILL.md — required)</label>
        <p className="help-text" style={{ marginTop: 4, marginBottom: 6 }}>
          One of: openai | anthropic | bedrock | vertex | adk | langchain | langgraph | crewai | generic
        </p>
        <select
          id="demoSkillFramework"
          style={{ width: "100%", marginTop: 4 }}
          value={skillForm.skillFramework}
          onChange={(e) => setSkillForm((s) => ({ ...s, skillFramework: e.target.value }))}
        >
          {SKILL_FRAMEWORK_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      </div>

      <div className="input-section" style={{ marginTop: 15 }}>
        <label htmlFor="demoAgentArch">agent_architecture (SKILL.md — required)</label>
        <select
          id="demoAgentArch"
          style={{ width: "100%", marginTop: 6 }}
          value={skillForm.agentArchitecture}
          onChange={(e) => setSkillForm((s) => ({ ...s, agentArchitecture: e.target.value }))}
        >
          {AGENT_ARCHITECTURE_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      </div>

      <div className="input-section" style={{ marginTop: 15 }}>
        <label htmlFor="demoNumTraces">num_traces (SKILL.md — default 500)</label>
        <select
          id="demoNumTraces"
          style={{ width: "100%", marginTop: 6 }}
          value={skillForm.numTraces}
          onChange={(e) => setSkillForm((s) => ({ ...s, numTraces: Number(e.target.value) }))}
        >
          {NUM_TRACE_OPTIONS.map((n) => (
            <option key={n} value={n}>
              {n}
            </option>
          ))}
        </select>
      </div>

      <div className="input-section" style={{ marginTop: 15 }}>
        <label htmlFor="demoOutputDir">output_dir (SKILL.md — optional)</label>
        <input
          type="text"
          id="demoOutputDir"
          placeholder="Leave blank for ~/arize-repos/&lt;company&gt;_&lt;slug&gt;/"
          value={skillForm.outputDir}
          onChange={(e) => setSkillForm((s) => ({ ...s, outputDir: e.target.value }))}
        />
      </div>

      <div className="input-section" style={{ marginTop: 15 }}>
        <label htmlFor="demoToolsText">tools (SKILL.md — optional)</label>
        <p className="help-text" style={{ marginTop: 4, marginBottom: 6 }}>
          One tool per line: <code>tool_name — one-line description</code> (also accepts <code> - </code> or{" "}
          <code>: </code>).
        </p>
        <textarea
          id="demoToolsText"
          rows={4}
          value={skillForm.toolsText}
          onChange={(e) => setSkillForm((s) => ({ ...s, toolsText: e.target.value }))}
        />
      </div>

      <div className="input-section" style={{ marginTop: 15 }}>
        <label htmlFor="demoPromptTemplateNames">prompt_template_names (SKILL.md — optional)</label>
        <input
          type="text"
          id="demoPromptTemplateNames"
          placeholder="Comma-separated names if you lock templates; else leave blank for auto-derive"
          value={skillForm.promptTemplateNames}
          onChange={(e) => setSkillForm((s) => ({ ...s, promptTemplateNames: e.target.value }))}
        />
      </div>

      <div className="input-section" style={{ marginTop: 15 }}>
        <label htmlFor="demoScenarios">scenarios (SKILL.md — optional)</label>
        <p className="help-text" style={{ marginTop: 4, marginBottom: 6 }}>
          Hold Cmd/Ctrl for multiple; empty uses skill defaults for the architecture.
        </p>
        <select
          id="demoScenarios"
          multiple
          size={6}
          style={{ width: "100%", marginTop: 4 }}
          value={skillForm.scenarios}
          onChange={(e) => {
            const next = Array.from(e.target.selectedOptions, (o) => o.value);
            setSkillForm((s) => ({ ...s, scenarios: next }));
          }}
        >
          {SCENARIO_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      </div>

      <div className="input-section" style={{ marginTop: 15 }}>
        <label htmlFor="demoWithEvals">with_evals (SKILL.md — optional, default true)</label>
        <select
          id="demoWithEvals"
          style={{ width: "100%", marginTop: 6 }}
          value={skillForm.withEvals ? "true" : "false"}
          onChange={(e) => setSkillForm((s) => ({ ...s, withEvals: e.target.value === "true" }))}
        >
          <option value="true">true</option>
          <option value="false">false</option>
        </select>
      </div>

      <div className="input-section" style={{ marginTop: 15 }}>
        <label>session_size_range (SKILL.md — optional, default (3, 6))</label>
        <div style={{ display: "flex", gap: 12, marginTop: 6, flexWrap: "wrap" }}>
          <div style={{ flex: 1, minWidth: 100 }}>
            <label htmlFor="demoSessionMin" style={{ fontSize: "0.85em" }}>
              min turns
            </label>
            <input
              id="demoSessionMin"
              type="number"
              min={1}
              max={50}
              style={{ width: "100%", marginTop: 4 }}
              value={skillForm.sessionSizeMin}
              onChange={(e) =>
                setSkillForm((s) => ({ ...s, sessionSizeMin: Math.min(50, Math.max(1, Number(e.target.value) || 3)) }))
              }
            />
          </div>
          <div style={{ flex: 1, minWidth: 100 }}>
            <label htmlFor="demoSessionMax" style={{ fontSize: "0.85em" }}>
              max turns
            </label>
            <input
              id="demoSessionMax"
              type="number"
              min={1}
              max={50}
              style={{ width: "100%", marginTop: 4 }}
              value={skillForm.sessionSizeMax}
              onChange={(e) =>
                setSkillForm((s) => ({ ...s, sessionSizeMax: Math.min(50, Math.max(1, Number(e.target.value) || 6)) }))
              }
            />
          </div>
        </div>
      </div>

      <div className="input-section" style={{ marginTop: 15 }}>
        <label htmlFor="demoPromptVersions">prompt_versions (SKILL.md — optional)</label>
        <textarea
          id="demoPromptVersions"
          rows={2}
          placeholder='JSON object, e.g. {"v1.0": 0.7, "v2.0": 0.3}'
          value={skillForm.promptVersionsJson}
          onChange={(e) => setSkillForm((s) => ({ ...s, promptVersionsJson: e.target.value }))}
        />
      </div>

      <div className="input-section" style={{ marginTop: 15 }}>
        <label htmlFor="demoWithDataset">with_dataset_and_experiments (SKILL.md — optional, default true)</label>
        <select
          id="demoWithDataset"
          style={{ width: "100%", marginTop: 6 }}
          value={skillForm.withDatasetAndExperiments ? "true" : "false"}
          onChange={(e) => setSkillForm((s) => ({ ...s, withDatasetAndExperiments: e.target.value === "true" }))}
        >
          <option value="true">true</option>
          <option value="false">false</option>
        </select>
      </div>

      <div className="input-section" style={{ marginTop: 15 }}>
        <label htmlFor="demoExperimentGrid">experiment_grid_models (SKILL.md — optional)</label>
        <input
          type="text"
          id="demoExperimentGrid"
          placeholder="Comma-separated model slugs, e.g. gpt-4o, gpt-4o-mini"
          value={skillForm.experimentGridModels}
          onChange={(e) => setSkillForm((s) => ({ ...s, experimentGridModels: e.target.value }))}
        />
      </div>

      <div className="input-section" style={{ marginTop: 15 }}>
        <label htmlFor="demoAdditionalContext">additional_context (app-only — not in SKILL.md)</label>
        <textarea
          id="demoAdditionalContext"
          rows={3}
          placeholder="Extra notes for the pasted Claude prompt and recommended_inputs JSON (deal context, stakeholder names, etc.)."
          value={additionalContext}
          onChange={(e) => setAdditionalContext(e.target.value)}
        />
      </div>

      <p
        className="help-text"
        style={{ marginTop: 12, padding: 12, background: "#e3f2fd", borderRadius: 8, fontSize: "0.92em" }}
      >
        Fields mirror{" "}
        <a href={SKILL_DOC} target="_blank" rel="noreferrer">
          arize-synthetic-demo SKILL.md
        </a>{" "}
        (&quot;Inputs you must gather&quot; + optional list). Auto-send credentials (<code>ARIZE_SPACE_ID</code>, etc.)
        stay in Claude / local env per the skill, not this form.
      </p>
      <div className="button-group">
        <button className="btn-primary" type="button" onClick={() => void classifyFromInput()}>
          Get skill prompt
        </button>
      </div>
    </>
  );
}
