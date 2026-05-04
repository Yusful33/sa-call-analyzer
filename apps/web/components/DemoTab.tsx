"use client";

import { useState } from "react";
import { apiPost } from "@/lib/api";
import { escapeHtml } from "@/lib/helpers";
import type { ResolveAccountFn } from "@/lib/accountResolve";

const SKILL_DOC =
  "https://github.com/Arize-ai/solutions-resources/blob/main/.claude/skills/arize-synthetic-demo/SKILL.md";

/** SKILL.md — `framework` (LLM / agent stack). Empty = derive from internal orchestration classifier. */
const SKILL_FRAMEWORK_OPTIONS: { value: string; label: string }[] = [
  { value: "", label: "Auto (from orchestration framework)" },
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

/** SKILL.md — `agent_architecture`. Empty = derive from demo pattern (use case). */
const AGENT_ARCHITECTURE_OPTIONS: { value: string; label: string }[] = [
  { value: "", label: "Auto (from demo pattern)" },
  { value: "single_agent", label: "single_agent" },
  { value: "multi_agent_coordinator", label: "multi_agent_coordinator" },
  { value: "retrieval_pipeline", label: "retrieval_pipeline" },
  { value: "rag_rerank", label: "rag_rerank" },
  { value: "guarded_rag", label: "guarded_rag" },
];

const NUM_TRACE_OPTIONS = [100, 250, 500, 1000, 2000] as const;

/** SKILL.md — optional `scenarios` subset. */
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
  skillFramework: string;
  agentArchitecture: string;
  numTraces: number;
  withEvals: boolean;
  withDatasetAndExperiments: boolean;
  scenarios: string[];
};

function buildClassifyBody(
  accountName: string,
  additionalContext: string,
  inputUseCaseOverride: string,
  inputFrameworkOverride: string,
  skill: SkillFormState,
  /** When set (confirm phase), always send these overrides for stable re-classify. */
  confirmLocks: { useCase: string; framework: string } | null
): Record<string, unknown> {
  const body: Record<string, unknown> = { account_name: accountName };
  const ctx = additionalContext.trim();
  if (ctx) body.additional_context = ctx;

  if (confirmLocks) {
    body.use_case_override = confirmLocks.useCase;
    body.framework_override = confirmLocks.framework;
  } else {
    if (inputUseCaseOverride.trim()) body.use_case_override = inputUseCaseOverride.trim();
    if (inputFrameworkOverride.trim()) body.framework_override = inputFrameworkOverride.trim();
  }

  if (skill.skillFramework.trim()) body.skill_framework = skill.skillFramework.trim();
  if (skill.agentArchitecture.trim()) body.agent_architecture = skill.agentArchitecture.trim();
  body.num_traces = skill.numTraces;
  body.with_evals = skill.withEvals;
  body.with_dataset_and_experiments = skill.withDatasetAndExperiments;
  if (skill.scenarios.length) body.scenarios = skill.scenarios;

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
  const [inputUseCaseOverride, setInputUseCaseOverride] = useState("");
  const [inputFrameworkOverride, setInputFrameworkOverride] = useState("");
  const [skillForm, setSkillForm] = useState<SkillFormState>({
    skillFramework: "",
    agentArchitecture: "",
    numTraces: 500,
    withEvals: true,
    withDatasetAndExperiments: true,
    scenarios: [],
  });

  const [phase, setPhase] = useState<"input" | "confirm">("input");
  const [classified, setClassified] = useState<any>(null);
  const [confirmUseCase, setConfirmUseCase] = useState("");
  const [confirmFramework, setConfirmFramework] = useState("");

  async function runClassify(body: Record<string, unknown>) {
    onLoading("Classifying prospect (for Claude skill inputs)...");
    try {
      const data = await apiPost<any>("/api/classify-demo", body);
      setClassified(data);
      setConfirmUseCase(data.use_case ?? "");
      setConfirmFramework(data.framework ?? "");
      setPhase("confirm");
      onResult("");
      onLoading("");
    } catch (err: any) {
      alert("Error: " + (err.message ?? String(err)));
      onLoading("");
    }
  }

  async function classifyFromInput() {
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

    const body = buildClassifyBody(
      useName,
      additionalContext,
      inputUseCaseOverride,
      inputFrameworkOverride,
      skillForm,
      null
    );
    await runClassify(body);
  }

  async function refetchWithConfirmLocks(next: { useCase?: string; framework?: string }) {
    if (!accountName.trim() || !classified) return;
    const useCase = next.useCase ?? confirmUseCase;
    const framework = next.framework ?? confirmFramework;
    setConfirmUseCase(useCase);
    setConfirmFramework(framework);

    const body = buildClassifyBody(
      accountName.trim(),
      additionalContext,
      inputUseCaseOverride,
      inputFrameworkOverride,
      skillForm,
      { useCase, framework }
    );
    onLoading("Updating skill prompt...");
    try {
      const data = await apiPost<any>("/api/classify-demo", body);
      setClassified(data);
      setConfirmUseCase(data.use_case ?? useCase);
      setConfirmFramework(data.framework ?? framework);
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
            <label style={{ fontSize: "0.88em" }}>Demo pattern (use case)</label>
            <select
              style={{ width: "100%", marginTop: 6 }}
              value={confirmUseCase}
              onChange={(e) => void refetchWithConfirmLocks({ useCase: e.target.value })}
            >
              {(classified.available_use_cases ?? []).map((uc: any) => (
                <option key={uc.value} value={uc.value}>
                  {uc.label}
                </option>
              ))}
            </select>
          </div>
          <div style={{ flex: 1, minWidth: 200 }}>
            <label style={{ fontSize: "0.88em" }}>Orchestration framework</label>
            <select
              style={{ width: "100%", marginTop: 6 }}
              value={confirmFramework}
              onChange={(e) => void refetchWithConfirmLocks({ framework: e.target.value })}
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
        <label htmlFor="demoUseCase">Demo pattern (optional)</label>
        <p className="help-text" style={{ marginTop: 4, marginBottom: 6 }}>
          Leave as <em>Infer from CRM and context</em> to classify from Gong/Salesforce and your notes below.
        </p>
        <select
          id="demoUseCase"
          style={{ width: "100%", marginTop: 4 }}
          value={inputUseCaseOverride}
          onChange={(e) => setInputUseCaseOverride(e.target.value)}
        >
          <option value="">Infer from CRM and context</option>
          {/* Populated from same taxonomy the API documents; values must match apps/api/main.py AVAILABLE_USE_CASES */}
          <option value="text-to-sql-bi-agent">Text-to-SQL / BI Agent</option>
          <option value="retrieval-augmented-search">RAG / Retrieval Search</option>
          <option value="multi-agent-orchestration">Multi-Agent Orchestration</option>
          <option value="classification-routing">Classification / Routing</option>
          <option value="multimodal-ai">Multimodal / Vision AI</option>
          <option value="mcp-tool-use">MCP Tool Use</option>
          <option value="multiturn-chatbot-with-tools">Chatbot with Tools</option>
          <option value="travel-agent">Travel Agent</option>
          <option value="guardrails">Guardrails</option>
          <option value="generic">Generic LLM Pipeline</option>
        </select>
      </div>

      <div className="input-section" style={{ marginTop: 15 }}>
        <label htmlFor="demoFramework">Orchestration framework (optional)</label>
        <p className="help-text" style={{ marginTop: 4, marginBottom: 6 }}>
          App classifier taxonomy (LangGraph, LangChain, CrewAI, ADK). Leave as auto unless you want to lock it before
          Gong/LLM classification.
        </p>
        <select
          id="demoFramework"
          style={{ width: "100%", marginTop: 4 }}
          value={inputFrameworkOverride}
          onChange={(e) => setInputFrameworkOverride(e.target.value)}
        >
          <option value="">Infer from CRM and context</option>
          <option value="langgraph">LangGraph</option>
          <option value="langchain">LangChain</option>
          <option value="crewai">CrewAI</option>
          <option value="adk">Google ADK</option>
        </select>
      </div>

      <div className="input-section" style={{ marginTop: 15 }}>
        <label htmlFor="demoSkillFramework">Skill LLM / provider framework</label>
        <p className="help-text" style={{ marginTop: 4, marginBottom: 6 }}>
          SKILL.md <code>framework</code> — spans and model attributes follow this provider.
        </p>
        <select
          id="demoSkillFramework"
          style={{ width: "100%", marginTop: 4 }}
          value={skillForm.skillFramework}
          onChange={(e) => setSkillForm((s) => ({ ...s, skillFramework: e.target.value }))}
        >
          {SKILL_FRAMEWORK_OPTIONS.map((o) => (
            <option key={o.value || "auto"} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      </div>

      <div className="input-section" style={{ marginTop: 15 }}>
        <label htmlFor="demoAgentArch">Agent architecture</label>
        <p className="help-text" style={{ marginTop: 4, marginBottom: 6 }}>
          SKILL.md <code>agent_architecture</code> — span tree shape for the generator.
        </p>
        <select
          id="demoAgentArch"
          style={{ width: "100%", marginTop: 4 }}
          value={skillForm.agentArchitecture}
          onChange={(e) => setSkillForm((s) => ({ ...s, agentArchitecture: e.target.value }))}
        >
          {AGENT_ARCHITECTURE_OPTIONS.map((o) => (
            <option key={o.value || "auto-arch"} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      </div>

      <div className="input-section" style={{ marginTop: 15 }}>
        <label htmlFor="demoNumTraces">Number of traces</label>
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
        <label htmlFor="demoWithEvals">Emit span/trace evals (with_evals)</label>
        <select
          id="demoWithEvals"
          style={{ width: "100%", marginTop: 6 }}
          value={skillForm.withEvals ? "true" : "false"}
          onChange={(e) => setSkillForm((s) => ({ ...s, withEvals: e.target.value === "true" }))}
        >
          <option value="true">Yes (default)</option>
          <option value="false">No</option>
        </select>
      </div>

      <div className="input-section" style={{ marginTop: 15 }}>
        <label htmlFor="demoWithDataset">Dataset + experiment grid (with_dataset_and_experiments)</label>
        <select
          id="demoWithDataset"
          style={{ width: "100%", marginTop: 6 }}
          value={skillForm.withDatasetAndExperiments ? "true" : "false"}
          onChange={(e) => setSkillForm((s) => ({ ...s, withDatasetAndExperiments: e.target.value === "true" }))}
        >
          <option value="true">Yes (default)</option>
          <option value="false">No (traces only)</option>
        </select>
      </div>

      <div className="input-section" style={{ marginTop: 15 }}>
        <label htmlFor="demoScenarios">Scenarios (optional)</label>
        <p className="help-text" style={{ marginTop: 4, marginBottom: 6 }}>
          SKILL.md optional <code>scenarios</code> — hold Cmd/Ctrl to pick multiple; leave empty for skill defaults.
        </p>
        <select
          id="demoScenarios"
          multiple
          size={5}
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
        <label htmlFor="demoAdditionalContext">Industry / scenario context (optional)</label>
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
        <button className="btn-primary" type="button" onClick={() => void classifyFromInput()}>
          Get skill prompt
        </button>
      </div>
    </>
  );
}
