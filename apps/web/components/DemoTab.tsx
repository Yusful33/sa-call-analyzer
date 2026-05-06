"use client";

import { useState, useEffect, useRef } from "react";
import { apiPost, apiPostBlob } from "@/lib/api";
import { escapeHtml } from "@/lib/helpers";
import { useToast } from "@/components/Toast";
import type { ResolveAccountFn } from "@/lib/accountResolve";
import type { ShareQuery } from "@/lib/shareableUrl";

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

type DemoInsightsResponse = {
  account_name: string;
  industry_or_use_case?: string;
  suggested_framework?: string;
  suggested_agent_architecture?: string;
  suggested_tools?: string;
  additional_context?: string;
  gong_calls_analyzed: number;
  data_sources_note?: string;
  insights_summary?: string;
};

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

type AutoFilledFields = {
  industryOrUseCase?: boolean;
  skillFramework?: boolean;
  agentArchitecture?: boolean;
  toolsText?: boolean;
  additionalContext?: boolean;
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
  urlQuery,
  onShareHintsChange,
}: {
  onLoading: (msg: string) => void;
  onResult: (html: string) => void;
  resolveAccount: ResolveAccountFn;
  urlQuery?: ShareQuery | null;
  onShareHintsChange?: (hints: Pick<ShareQuery, "demo_account">) => void;
}) {
  const toast = useToast();
  const [accountName, setAccountName] = useState("");
  const [additionalContext, setAdditionalContext] = useState("");
  const [skillForm, setSkillForm] = useState<SkillFormState>(defaultSkillForm);

  const [phase, setPhase] = useState<"lookup" | "input" | "result">("lookup");
  const [classified, setClassified] = useState<any>(null);
  const [insightsLoading, setInsightsLoading] = useState(false);
  const [autoFilledFields, setAutoFilledFields] = useState<AutoFilledFields>({});
  const [insightsSummary, setInsightsSummary] = useState<string | null>(null);
  const [dataSourcesNote, setDataSourcesNote] = useState<string | null>(null);
  const [gongCallsAnalyzed, setGongCallsAnalyzed] = useState<number>(0);
  const [generatingDemo, setGeneratingDemo] = useState(false);
  const [demoGenerated, setDemoGenerated] = useState(false);
  const urlHydrated = useRef(false);

  useEffect(() => {
    if (!urlQuery || urlHydrated.current) return;
    urlHydrated.current = true;
    const seed = urlQuery.demo_account || urlQuery.account_name;
    if (seed) setAccountName(seed);
  }, [urlQuery]);

  useEffect(() => {
    onShareHintsChange?.({
      demo_account: accountName.trim() || undefined,
    });
  }, [accountName, onShareHintsChange]);

  async function fetchInsights() {
    if (!accountName.trim()) {
      toast.warning("Please enter a prospect/account name to fetch insights.");
      return;
    }

    setInsightsLoading(true);
    onLoading("Resolving account...");

    try {
      const resolved = await resolveAccount({
        accountName: accountName.trim(),
        accountDomain: "",
        sfdcAccountId: "",
      });
      if (!resolved.proceed) {
        setInsightsLoading(false);
        onLoading("");
        return;
      }
      const useName = (resolved.accountName || accountName).trim();
      if (useName !== accountName.trim()) setAccountName(useName);

      onLoading("Fetching Gong call insights for " + useName + "...");

      const data = await apiPost<DemoInsightsResponse>("/api/demo-insights", {
        account_name: useName,
      });

      const newAutoFilled: AutoFilledFields = {};
      const newForm = { ...skillForm };

      if (data.industry_or_use_case) {
        newForm.industryOrUseCase = data.industry_or_use_case;
        newAutoFilled.industryOrUseCase = true;
      }

      if (data.suggested_framework) {
        newForm.skillFramework = data.suggested_framework;
        newAutoFilled.skillFramework = true;
      }

      if (data.suggested_agent_architecture) {
        newForm.agentArchitecture = data.suggested_agent_architecture;
        newAutoFilled.agentArchitecture = true;
      }

      if (data.suggested_tools) {
        newForm.toolsText = data.suggested_tools;
        newAutoFilled.toolsText = true;
      }

      if (data.additional_context) {
        setAdditionalContext(data.additional_context);
        newAutoFilled.additionalContext = true;
      }

      setSkillForm(newForm);
      setAutoFilledFields(newAutoFilled);
      setInsightsSummary(data.insights_summary || null);
      setDataSourcesNote(data.data_sources_note || null);
      setGongCallsAnalyzed(data.gong_calls_analyzed);
      setPhase("input");
      onLoading("");
    } catch (err: any) {
      toast.error("Error fetching insights: " + (err.message ?? String(err)));
      onLoading("");
    } finally {
      setInsightsLoading(false);
    }
  }

  function skipInsights() {
    setPhase("input");
    setAutoFilledFields({});
    setInsightsSummary(null);
    setDataSourcesNote(null);
    setGongCallsAnalyzed(0);
  }

  async function runClassify(body: Record<string, unknown>) {
    onLoading("Building skill prompt (CRM/Gong hints + SKILL.md inputs)...");
    try {
      const data = await apiPost<any>("/api/classify-demo", body);
      setClassified(data);
      setPhase("result");
      onResult("");
      onLoading("");
    } catch (err: any) {
      toast.error("Error: " + (err.message ?? String(err)));
      onLoading("");
    }
  }

  async function classifyFromInput() {
    if (!accountName.trim()) {
      toast.warning("Please enter a prospect/account name (SKILL.md: company_name).");
      return;
    }
    if (!skillForm.industryOrUseCase.trim()) {
      toast.warning("Please enter industry / use case (SKILL.md: industry_or_use_case).");
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

  async function executeSkill() {
    if (generatingDemo) return;
    
    setGeneratingDemo(true);
    onLoading("Generating demo files (this may take 30-60 seconds)...");

    try {
      const body = {
        account_name: accountName.trim(),
        industry_or_use_case: skillForm.industryOrUseCase.trim(),
        skill_framework: skillForm.skillFramework,
        agent_architecture: skillForm.agentArchitecture,
        num_traces: skillForm.numTraces,
        with_evals: skillForm.withEvals,
        with_dataset_and_experiments: skillForm.withDatasetAndExperiments,
        scenarios: skillForm.scenarios.length > 0 ? skillForm.scenarios : undefined,
        tools_text: skillForm.toolsText.trim() || undefined,
        additional_context: additionalContext.trim() || undefined,
      };

      const { blob, filename } = await apiPostBlob("/api/generate-demo", body, `${accountName.trim().toLowerCase().replace(/\s+/g, "_")}_demo.zip`);

      // Download the file
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      setDemoGenerated(true);
      onLoading("");
    } catch (err: any) {
      toast.error("Error generating demo: " + (err.message ?? String(err)));
      onLoading("");
    } finally {
      setGeneratingDemo(false);
    }
  }

  if (phase === "result" && classified) {
    const sk = classified.synthetic_demo_skill;
    return (
      <div style={{ marginTop: 10 }}>
        <h3 style={{ marginBottom: 8 }}>Arize Synthetic Demo Builder</h3>

        {/* Execute Skill Section - Primary Action */}
        <div
          style={{
            padding: 16,
            background: demoGenerated ? "#e8f5e9" : "#e3f2fd",
            borderRadius: 8,
            marginBottom: 16,
            border: demoGenerated ? "1px solid #a5d6a7" : "1px solid #90caf9",
          }}
        >
          <h4 style={{ margin: "0 0 8px 0", fontSize: "1em" }}>
            {demoGenerated ? "✓ Demo Generated Successfully" : "Generate Demo Files"}
          </h4>
          <p style={{ color: "#555", fontSize: "0.9em", margin: "0 0 12px 0" }}>
            {demoGenerated
              ? "Your demo has been downloaded. Extract the ZIP and follow the README to run the generator."
              : "Click below to generate a complete demo package including generator.py, requirements.txt, and documentation."}
          </p>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
            <button
              type="button"
              className="btn-primary"
              onClick={() => void executeSkill()}
              disabled={generatingDemo}
              style={{
                background: demoGenerated ? "#4caf50" : undefined,
                minWidth: 180,
              }}
            >
              {generatingDemo
                ? "Generating..."
                : demoGenerated
                ? "Download Again"
                : "Generate & Download Demo"}
            </button>
            {demoGenerated && (
              <span style={{ color: "#388e3c", fontSize: "0.85em" }}>
                Demo downloaded — check your downloads folder
              </span>
            )}
          </div>
        </div>

        {classified.data_sources_note && (
          <p style={{ color: "#888", fontSize: "0.85em", fontStyle: "italic", marginBottom: 12 }}>
            {classified.data_sources_note}
          </p>
        )}

        <p style={{ fontSize: "0.9em", color: "#555", marginBottom: 12 }}>
          <strong>Classification:</strong> {classified.use_case} / {classified.framework}
          {classified.reasoning ? ` — ${classified.reasoning}` : ""}
        </p>

        {sk?.recommended_inputs && (
          <details style={{ marginBottom: 14 }}>
            <summary style={{ cursor: "pointer", fontWeight: 600, fontSize: "0.9em" }}>Skill input mapping (JSON)</summary>
            <pre
              style={{
                background: "#f5f5f5",
                padding: 12,
                borderRadius: 8,
                overflow: "auto",
                maxHeight: 280,
                fontSize: "0.8em",
              }}
            >
              {escapeHtml(JSON.stringify(sk.recommended_inputs, null, 2))}
            </pre>
          </details>
        )}

        {/* Alternative: Manual Prompt Section */}
        <details style={{ marginTop: 12 }}>
          <summary style={{ cursor: "pointer", fontWeight: 600, fontSize: "0.9em", color: "#666" }}>
            Alternative: Copy prompt for Claude
          </summary>
          <div style={{ marginTop: 10 }}>
            <p style={{ color: "#666", fontSize: "0.85em", marginBottom: 8 }}>
              Use this prompt with the <strong>arize-synthetic-demo</strong> Claude skill for more customization.{" "}
              <a href={SKILL_DOC} target="_blank" rel="noreferrer" style={{ fontWeight: 500 }}>
                View SKILL.md
              </a>
            </p>
            <textarea
              readOnly
              rows={6}
              style={{ width: "100%", fontFamily: "monospace", fontSize: "0.8em" }}
              value={sk?.suggested_prompt_for_claude ?? ""}
            />
            <button
              type="button"
              className="btn-secondary"
              onClick={copyPrompt}
              style={{ marginTop: 8, fontSize: "0.85em" }}
            >
              Copy prompt
            </button>
          </div>
        </details>

        <div style={{ marginTop: 16, display: "flex", gap: 8 }}>
          <button
            type="button"
            className="btn-secondary"
            onClick={() => {
              setPhase("lookup");
              setClassified(null);
              setAutoFilledFields({});
              setInsightsSummary(null);
              setDataSourcesNote(null);
              setGongCallsAnalyzed(0);
              setDemoGenerated(false);
            }}
          >
            Start over
          </button>
        </div>

        {sk?.next_steps && !demoGenerated && (
          <div style={{ marginTop: 16 }}>
            <strong style={{ fontSize: "0.9em" }}>After generating:</strong>
            <ul style={{ color: "#555", paddingLeft: 20, marginTop: 6 }}>
              {(sk.next_steps as string[]).slice(1).map((s: string) => (
                <li key={s.slice(0, 48)} style={{ marginBottom: 4, fontSize: "0.85em" }}>
                  {s}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    );
  }

  if (phase === "lookup") {
    return (
      <>
        <div
          style={{
            padding: 16,
            background: "#e8f4fd",
            borderRadius: 8,
            marginBottom: 16,
            border: "1px solid #b3d9f2",
          }}
        >
          <h3 style={{ margin: "0 0 8px 0", fontSize: "1.1em" }}>Step 1: Fetch Insights from Gong Calls</h3>
          <p style={{ color: "#555", fontSize: "0.92em", margin: "0 0 12px 0" }}>
            Enter a prospect/account name to automatically populate the demo builder fields based on insights from
            Gong calls (use cases mentioned, pain points, industry context, features discussed).
          </p>

          <div className="input-section" style={{ marginBottom: 12 }}>
            <label htmlFor="lookupAccountName">Prospect/Account Name</label>
            <input
              type="text"
              id="lookupAccountName"
              placeholder="e.g., Acme Corp, Tesla, etc."
              value={accountName}
              onChange={(e) => setAccountName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !insightsLoading) {
                  e.preventDefault();
                  void fetchInsights();
                }
              }}
              disabled={insightsLoading}
            />
          </div>

          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            <button
              type="button"
              className="btn-primary"
              onClick={() => void fetchInsights()}
              disabled={insightsLoading || !accountName.trim()}
            >
              {insightsLoading ? "Fetching..." : "Fetch Insights"}
            </button>
            <button
              type="button"
              className="btn-secondary"
              onClick={skipInsights}
              disabled={insightsLoading}
            >
              Skip (fill manually)
            </button>
          </div>
        </div>

        <p className="help-text" style={{ fontSize: "0.9em", color: "#666" }}>
          After fetching, you can review and edit the auto-populated fields before generating the skill prompt.
        </p>
      </>
    );
  }

  const autoFilledStyle = (isAutoFilled: boolean | undefined) =>
    isAutoFilled
      ? {
          borderColor: "#4caf50",
          backgroundColor: "#f1f8e9",
        }
      : {};

  const autoFilledLabel = (isAutoFilled: boolean | undefined) =>
    isAutoFilled ? (
      <span style={{ color: "#4caf50", fontSize: "0.8em", marginLeft: 6 }}>
        (auto-filled from Gong)
      </span>
    ) : null;

  return (
    <>
      {(insightsSummary || dataSourcesNote) && (
        <div
          style={{
            padding: 12,
            background: "#e8f4fd",
            borderRadius: 8,
            marginBottom: 16,
            border: "1px solid #b3d9f2",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
            <div>
              <strong style={{ fontSize: "0.95em" }}>
                Insights from {gongCallsAnalyzed} Gong call{gongCallsAnalyzed !== 1 ? "s" : ""}
              </strong>
              {insightsSummary && (
                <p style={{ margin: "6px 0 0 0", fontSize: "0.9em", color: "#333" }}>{insightsSummary}</p>
              )}
              {dataSourcesNote && (
                <p style={{ margin: "4px 0 0 0", fontSize: "0.82em", color: "#666", fontStyle: "italic" }}>
                  {dataSourcesNote}
                </p>
              )}
            </div>
            <button
              type="button"
              style={{
                background: "none",
                border: "none",
                color: "#1976d2",
                cursor: "pointer",
                fontSize: "0.85em",
                padding: "4px 8px",
              }}
              onClick={() => setPhase("lookup")}
            >
              Change account
            </button>
          </div>
        </div>
      )}

      <div className="input-section">
        <label htmlFor="demoAccountName">
          company_name (SKILL.md)
        </label>
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
        {!insightsSummary && !dataSourcesNote && gongCallsAnalyzed === 0 && (
          <button
            type="button"
            style={{
              marginTop: 8,
              background: "none",
              border: "1px solid #1976d2",
              color: "#1976d2",
              cursor: "pointer",
              fontSize: "0.85em",
              padding: "6px 12px",
              borderRadius: 6,
            }}
            onClick={() => setPhase("lookup")}
          >
            Try auto-fill from Gong
          </button>
        )}
      </div>

      <div className="input-section" style={{ marginTop: 15 }}>
        <label htmlFor="demoIndustryUseCase">
          industry_or_use_case (SKILL.md — required)
          {autoFilledLabel(autoFilledFields.industryOrUseCase)}
        </label>
        <textarea
          id="demoIndustryUseCase"
          rows={3}
          placeholder='e.g., retail banking fraud triage agent, insurance claims adjudication'
          value={skillForm.industryOrUseCase}
          onChange={(e) => {
            setSkillForm((s) => ({ ...s, industryOrUseCase: e.target.value }));
            if (autoFilledFields.industryOrUseCase) {
              setAutoFilledFields((f) => ({ ...f, industryOrUseCase: false }));
            }
          }}
          style={autoFilledStyle(autoFilledFields.industryOrUseCase)}
        />
      </div>

      <div className="input-section" style={{ marginTop: 15 }}>
        <label htmlFor="demoSkillFramework">
          framework (SKILL.md — required)
          {autoFilledLabel(autoFilledFields.skillFramework)}
        </label>
        <p className="help-text" style={{ marginTop: 4, marginBottom: 6 }}>
          One of: openai | anthropic | bedrock | vertex | adk | langchain | langgraph | crewai | generic
        </p>
        <select
          id="demoSkillFramework"
          style={{ width: "100%", marginTop: 4, ...autoFilledStyle(autoFilledFields.skillFramework) }}
          value={skillForm.skillFramework}
          onChange={(e) => {
            setSkillForm((s) => ({ ...s, skillFramework: e.target.value }));
            if (autoFilledFields.skillFramework) {
              setAutoFilledFields((f) => ({ ...f, skillFramework: false }));
            }
          }}
        >
          {SKILL_FRAMEWORK_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      </div>

      <div className="input-section" style={{ marginTop: 15 }}>
        <label htmlFor="demoAgentArch">
          agent_architecture (SKILL.md — required)
          {autoFilledLabel(autoFilledFields.agentArchitecture)}
        </label>
        <select
          id="demoAgentArch"
          style={{ width: "100%", marginTop: 6, ...autoFilledStyle(autoFilledFields.agentArchitecture) }}
          value={skillForm.agentArchitecture}
          onChange={(e) => {
            setSkillForm((s) => ({ ...s, agentArchitecture: e.target.value }));
            if (autoFilledFields.agentArchitecture) {
              setAutoFilledFields((f) => ({ ...f, agentArchitecture: false }));
            }
          }}
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
        <label htmlFor="demoToolsText">
          tools (SKILL.md — optional)
          {autoFilledLabel(autoFilledFields.toolsText)}
        </label>
        <p className="help-text" style={{ marginTop: 4, marginBottom: 6 }}>
          One tool per line: <code>tool_name — one-line description</code> (also accepts <code> - </code> or{" "}
          <code>: </code>).
        </p>
        <textarea
          id="demoToolsText"
          rows={4}
          value={skillForm.toolsText}
          onChange={(e) => {
            setSkillForm((s) => ({ ...s, toolsText: e.target.value }));
            if (autoFilledFields.toolsText) {
              setAutoFilledFields((f) => ({ ...f, toolsText: false }));
            }
          }}
          style={autoFilledStyle(autoFilledFields.toolsText)}
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
        <label htmlFor="demoAdditionalContext">
          additional_context (app-only — not in SKILL.md)
          {autoFilledLabel(autoFilledFields.additionalContext)}
        </label>
        <textarea
          id="demoAdditionalContext"
          rows={3}
          placeholder="Extra notes for the pasted Claude prompt and recommended_inputs JSON (deal context, stakeholder names, etc.)."
          value={additionalContext}
          onChange={(e) => {
            setAdditionalContext(e.target.value);
            if (autoFilledFields.additionalContext) {
              setAutoFilledFields((f) => ({ ...f, additionalContext: false }));
            }
          }}
          style={autoFilledStyle(autoFilledFields.additionalContext)}
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
