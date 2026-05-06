"use client";

import { useState, useCallback, useEffect, useMemo } from "react";
import ProspectTab from "@/components/ProspectTab";
import HypothesisTab from "@/components/HypothesisTab";
import GongTab from "@/components/GongTab";
import DemoTab from "@/components/DemoTab";
import TransitionTab from "@/components/TransitionTab";
import PocTab from "@/components/PocTab";
import LoadingCard from "@/components/LoadingCard";
import ResultsCard from "@/components/ResultsCard";
import AccountSuggestModal from "@/components/AccountSuggestModal";
import {
  AchievementProvider,
  AchievementToast,
  AchievementBadges,
  useAchievements,
} from "@/components/UsageAchievements";
import { apiPost } from "@/lib/api";
import type {
  AccountResolveInput,
  AccountResolveResult,
  AccountSuggestionsResponse,
  AccountSuggestionMatch,
} from "@/lib/accountResolve";

type ToolId =
  | "hypothesis"
  | "demo"
  | "gong"
  | "poc"
  | "prospect"
  | "transition";

type StageId = "stage1" | "stage2" | "stage3" | "stage4" | "stage5";

type Tool = {
  id: ToolId;
  label: string;
  shortLabel: string;
};

type Stage = {
  id: StageId;
  number: 1 | 2 | 3 | 4 | 5;
  name: string;
  blurb: string;
  tools: Tool[];
};

const STAGES: Stage[] = [
  {
    id: "stage1",
    number: 1,
    name: "Engaged",
    blurb: "First touch. Pull AI/ML signals before the first call.",
    tools: [
      {
        id: "hypothesis",
        label: "\u{1F52C} Hypothesis Research",
        shortLabel: "Hypothesis Research",
      },
    ],
  },
  {
    id: "stage2",
    number: 2,
    name: "Qualification",
    blurb:
      "Validate fit. Build a tailored demo and dissect the discovery call.",
    tools: [
      {
        id: "demo",
        label: "\u{1F3AF} Custom Demo Builder",
        shortLabel: "Custom Demo Builder",
      },
      {
        id: "gong",
        label: "\u{1F4DE} Single Call Analysis",
        shortLabel: "Single Call Analysis",
      },
    ],
  },
  {
    id: "stage3",
    number: 3,
    name: "Pre-PoC",
    blurb:
      "Stand up the PoC scope. Draft the PoT / PoC doc and align on success criteria.",
    tools: [
      {
        id: "poc",
        label: "\u{1F4C4} PoC / PoT Document",
        shortLabel: "PoC / PoT Document",
      },
    ],
  },
  {
    id: "stage4",
    number: 4,
    name: "PoC",
    blurb: "PoC in flight. 360° view of the buyer and the deal as you push to close.",
    tools: [
      {
        id: "prospect",
        label: "\u{1F4CA} Prospect Overview",
        shortLabel: "Prospect Overview",
      },
    ],
  },
  {
    id: "stage5",
    number: 5,
    name: "Closed Won",
    blurb: "Ship a clean Knowledge Transfer doc into Customer Success.",
    tools: [
      {
        id: "transition",
        label: "\u{1F501} Transition to CS",
        shortLabel: "Transition to CS",
      },
    ],
  },
];

const TOOL_TO_STAGE: Record<ToolId, StageId> = STAGES.reduce(
  (acc, s) => {
    s.tools.forEach((t) => {
      acc[t.id] = s.id;
    });
    return acc;
  },
  {} as Record<ToolId, StageId>,
);

type SuggestUi = {
  reason: string;
  typedQuery: string;
  matches: AccountSuggestionMatch[];
  resolve: (c: "cancel" | "keep" | AccountSuggestionMatch) => void;
};

function HomeContent() {
  // Default landing tool: Stage 1 / Hypothesis Research — the start of the funnel.
  const [activeTool, setActiveTool] = useState<ToolId>("hypothesis");
  const [loading, setLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState("");
  const [resultHtml, setResultHtml] = useState("");
  const [resultOwner, setResultOwner] = useState<ToolId | null>(null);
  const [suggestUi, setSuggestUi] = useState<SuggestUi | null>(null);

  const {
    trackProspectRun,
    trackGongSuccess,
    trackDemoBuild,
    trackTabUsed,
  } = useAchievements();

  useEffect(() => {
    trackTabUsed(activeTool);
  }, [activeTool, trackTabUsed]);

  const activeStage = useMemo<Stage>(
    () =>
      STAGES.find((s) => s.id === TOOL_TO_STAGE[activeTool]) ??
      STAGES[0],
    [activeTool],
  );

  const onLoading = useCallback((msg: string) => {
    if (msg) {
      setLoading(true);
      setLoadingMessage(msg);
      setResultHtml("");
      setResultOwner(null);
    } else {
      setLoading(false);
    }
  }, []);

  const makeOnResult = useCallback(
    (tool: ToolId) => (html: string) => {
      setLoading(false);
      setResultHtml(html);
      setResultOwner(html ? tool : null);

      if (html) {
        if (tool === "prospect") {
          trackProspectRun();
        } else if (tool === "gong") {
          trackGongSuccess();
        } else if (tool === "demo") {
          trackDemoBuild();
        }
      }
    },
    [trackProspectRun, trackGongSuccess, trackDemoBuild],
  );

  const resolveAccount = useCallback(
    async (input: AccountResolveInput): Promise<AccountResolveResult> => {
      const sid = (input.sfdcAccountId || "").trim();
      const an = (input.accountName || "").trim();
      const dom = (input.accountDomain || "").trim();
      if (sid) {
        return {
          proceed: true,
          accountName: an,
          accountDomain: dom || undefined,
          sfdcAccountId: sid,
        };
      }
      if (!an) {
        return {
          proceed: true,
          accountName: an,
          accountDomain: dom || undefined,
        };
      }
      try {
        const r = await apiPost<AccountSuggestionsResponse>(
          "/api/account-suggestions",
          { account_name: an, domain: dom || null },
        );
        if (r.status === "ok" && r.matches?.length === 1) {
          const m = r.matches[0];
          return {
            proceed: true,
            accountName: m.name,
            accountDomain: dom || undefined,
            sfdcAccountId: m.id,
          };
        }
        if (r.status !== "suggest" || !r.matches?.length) {
          return {
            proceed: true,
            accountName: an,
            accountDomain: dom || undefined,
          };
        }
        return await new Promise((resolve) => {
          setSuggestUi({
            reason: r.reason || "Pick an account.",
            typedQuery: an,
            matches: r.matches,
            resolve: (choice) => {
              setSuggestUi(null);
              if (choice === "cancel") resolve({ proceed: false });
              else if (choice === "keep") {
                resolve({
                  proceed: true,
                  accountName: an,
                  accountDomain: dom || undefined,
                });
              } else {
                resolve({
                  proceed: true,
                  accountName: choice.name,
                  accountDomain: dom || undefined,
                  sfdcAccountId: choice.id,
                });
              }
            },
          });
        });
      } catch {
        return {
          proceed: true,
          accountName: an,
          accountDomain: dom || undefined,
        };
      }
    },
    [],
  );

  const showResults = resultOwner === activeTool && !!resultHtml;

  return (
    <div className="container">
      <div className="header">
        <div className="brand-mark">
          <svg viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M16 2L30 28H2L16 2Z" fill="url(#arize-grad)" />
            <defs>
              <linearGradient
                id="arize-grad"
                x1="2"
                y1="28"
                x2="30"
                y2="2"
                gradientUnits="userSpaceOnUse"
              >
                <stop stopColor="#E5117F" />
                <stop offset="1" stopColor="#8E5BD3" />
              </linearGradient>
            </defs>
          </svg>
          <span className="arize-wordmark">Arize</span>
          <span className="brand-divider" />
          <span>Solutions</span>
        </div>
        <h1>
          <span className="accent">Stillness</span>
          <AchievementBadges />
        </h1>
        <p className="app-summary">
          <strong>360° prospect intelligence:</strong> CRM data, call insights,
          product usage, and user behavior — unified in one place to prepare for
          your next customer conversation.
        </p>
      </div>

      <AchievementToast />

      <div className="stage-rail-wrapper">
        <div className="stage-rail-eyebrow">Sales Stage</div>
        <ol className="stage-rail" role="tablist" aria-label="Sales Stage">
          {STAGES.map((s, i) => {
            const stageActive = activeStage.id === s.id;
            const isMulti = s.tools.length > 1;
            const eyebrow = (
              <div className="stage-card-eyebrow">
                <span className="stage-card-number">Stage {s.number}</span>
                <span className="stage-card-divider" aria-hidden="true">
                  •
                </span>
                <span className="stage-card-name">{s.name}</span>
              </div>
            );

            return (
              <li key={s.id} className="stage-rail-item">
                {isMulti ? (
                  <div
                    className={`stage-card stage-card-multi${stageActive ? " active" : ""}`}
                  >
                    {eyebrow}
                    <div className="stage-card-tool-list" role="tablist">
                      {s.tools.map((t) => {
                        const toolActive = activeTool === t.id;
                        return (
                          <button
                            key={t.id}
                            type="button"
                            role="tab"
                            aria-selected={toolActive}
                            className={`stage-tool-button${toolActive ? " active" : ""}`}
                            onClick={() => setActiveTool(t.id)}
                          >
                            {t.label}
                          </button>
                        );
                      })}
                    </div>
                    <div className="stage-card-blurb">{s.blurb}</div>
                  </div>
                ) : (
                  <button
                    type="button"
                    role="tab"
                    aria-selected={stageActive}
                    className={`stage-card stage-card-single${stageActive ? " active" : ""}`}
                    onClick={() => setActiveTool(s.tools[0].id)}
                  >
                    {eyebrow}
                    <div className="stage-card-tool-name">
                      {s.tools[0].label}
                    </div>
                    <div className="stage-card-blurb">{s.blurb}</div>
                  </button>
                )}
                {i < STAGES.length - 1 ? (
                  <span className="stage-rail-arrow" aria-hidden="true">
                    →
                  </span>
                ) : null}
              </li>
            );
          })}
        </ol>
      </div>

      <div className="card stage-card-body">
        <div className="stage-body-header">
          <div>
            <div className="stage-body-eyebrow">
              Stage {activeStage.number} • {activeStage.name}
            </div>
            <div className="stage-body-tool-name">
              {activeStage.tools.find((t) => t.id === activeTool)?.label ?? ""}
            </div>
          </div>
        </div>

        <div className="tab-content-wrapper">
          <div className={`tab-content${activeTool === "hypothesis" ? " active" : ""}`}>
            <HypothesisTab
              onLoading={onLoading}
              onResult={makeOnResult("hypothesis")}
              resolveAccount={resolveAccount}
            />
          </div>
          <div className={`tab-content${activeTool === "demo" ? " active" : ""}`}>
            <DemoTab
              onLoading={onLoading}
              onResult={makeOnResult("demo")}
              resolveAccount={resolveAccount}
            />
          </div>
          <div className={`tab-content${activeTool === "gong" ? " active" : ""}`}>
            <GongTab onLoading={onLoading} onResult={makeOnResult("gong")} />
          </div>
          <div className={`tab-content${activeTool === "poc" ? " active" : ""}`}>
            <PocTab onLoading={onLoading} resolveAccount={resolveAccount} />
          </div>
          <div className={`tab-content${activeTool === "prospect" ? " active" : ""}`}>
            <ProspectTab
              onLoading={onLoading}
              onResult={makeOnResult("prospect")}
              resolveAccount={resolveAccount}
            />
          </div>
          <div className={`tab-content${activeTool === "transition" ? " active" : ""}`}>
            <TransitionTab
              onLoading={onLoading}
              resolveAccount={resolveAccount}
            />
          </div>
        </div>
      </div>

      <LoadingCard visible={loading} message={loadingMessage} />
      <ResultsCard visible={showResults} html={resultHtml} />

      {suggestUi ? (
        <AccountSuggestModal
          open
          reason={suggestUi.reason}
          typedQuery={suggestUi.typedQuery}
          matches={suggestUi.matches}
          onPick={(c) => suggestUi.resolve(c)}
        />
      ) : null}
    </div>
  );
}

export default function Home() {
  return (
    <AchievementProvider>
      <HomeContent />
    </AchievementProvider>
  );
}
