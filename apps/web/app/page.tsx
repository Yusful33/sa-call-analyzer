"use client";

import { useState, useCallback, useEffect } from "react";
import ProspectTab from "@/components/ProspectTab";
import HypothesisTab from "@/components/HypothesisTab";
import GongTab from "@/components/GongTab";
import DemoTab from "@/components/DemoTab";
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

type TabId = "hypothesis" | "prospect" | "demo" | "gong";

const TABS: { id: TabId; label: string }[] = [
  { id: "hypothesis", label: "\u{1F52C} Hypothesis Research" },
  { id: "prospect", label: "\u{1F4CA} Prospect Overview" },
  { id: "demo", label: "\u{1F3AF} Custom Demo Builder" },
  { id: "gong", label: "Single Call Analysis" },
];

type SuggestUi = {
  reason: string;
  typedQuery: string;
  matches: AccountSuggestionMatch[];
  resolve: (c: "cancel" | "keep" | AccountSuggestionMatch) => void;
};

function HomeContent() {
  const [activeTab, setActiveTab] = useState<TabId>("prospect");
  const [loading, setLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState("");
  const [resultHtml, setResultHtml] = useState("");
  const [resultOwner, setResultOwner] = useState<TabId | null>(null);
  const [suggestUi, setSuggestUi] = useState<SuggestUi | null>(null);

  const {
    trackProspectRun,
    trackGongSuccess,
    trackDemoBuild,
    trackTabUsed,
  } = useAchievements();

  useEffect(() => {
    trackTabUsed(activeTab);
  }, [activeTab, trackTabUsed]);

  const onLoading = useCallback(
    (msg: string) => {
      if (msg) {
        setLoading(true);
        setLoadingMessage(msg);
        setResultHtml("");
        setResultOwner(null);
      } else {
        setLoading(false);
      }
    },
    []
  );

  const makeOnResult = useCallback(
    (tab: TabId) => (html: string) => {
      setLoading(false);
      setResultHtml(html);
      setResultOwner(html ? tab : null);

      if (html) {
        if (tab === "prospect") {
          trackProspectRun();
        } else if (tab === "gong") {
          trackGongSuccess();
        } else if (tab === "demo") {
          trackDemoBuild();
        }
      }
    },
    [trackProspectRun, trackGongSuccess, trackDemoBuild]
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
          { account_name: an, domain: dom || null }
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
    []
  );

  const showResults = resultOwner === activeTab && !!resultHtml;

  return (
    <div className="container">
      <div className="header">
        <div className="brand-mark">
          <svg viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M16 2L30 28H2L16 2Z" fill="url(#arize-grad)" />
            <defs>
              <linearGradient id="arize-grad" x1="2" y1="28" x2="30" y2="2" gradientUnits="userSpaceOnUse">
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

      <div className="card">
        <div className="tabs-container">
          <div className="tabs">
            {TABS.map((t) => (
              <button
                key={t.id}
                className={`tab${activeTab === t.id ? " active" : ""}`}
                onClick={() => setActiveTab(t.id)}
              >
                {t.label}
              </button>
            ))}
          </div>

          <div className="tab-content-wrapper">
            <div className={`tab-content${activeTab === "prospect" ? " active" : ""}`}>
              <ProspectTab
                onLoading={onLoading}
                onResult={makeOnResult("prospect")}
                resolveAccount={resolveAccount}
              />
            </div>
            <div className={`tab-content${activeTab === "hypothesis" ? " active" : ""}`}>
              <HypothesisTab
                onLoading={onLoading}
                onResult={makeOnResult("hypothesis")}
                resolveAccount={resolveAccount}
              />
            </div>
            <div className={`tab-content${activeTab === "gong" ? " active" : ""}`}>
              <GongTab onLoading={onLoading} onResult={makeOnResult("gong")} />
            </div>
            <div className={`tab-content${activeTab === "demo" ? " active" : ""}`}>
              <DemoTab
                onLoading={onLoading}
                onResult={makeOnResult("demo")}
                resolveAccount={resolveAccount}
              />
            </div>
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
