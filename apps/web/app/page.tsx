"use client";

import dynamic from "next/dynamic";
import { Suspense, useState, useCallback, useEffect, useMemo, type ReactNode } from "react";
import { useSearchParams } from "next/navigation";
import type { ProspectShareHints } from "@/components/ProspectTab";
import SalesStageRail, { stageBodyCopy } from "@/components/SalesStageRail";
import LoadingCard from "@/components/LoadingCard";
import ResultsCard from "@/components/ResultsCard";
import AccountSuggestModal from "@/components/AccountSuggestModal";
import ShareToolbar from "@/components/ShareToolbar";
import { parseShareQuery, type ShareQuery, type ShareableTab } from "@/lib/shareableUrl";
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

const PipelineTab = dynamic(() => import("@/components/PipelineTab"), {
  loading: () => <TabPanelLoading />,
});

const HypothesisTab = dynamic(() => import("@/components/HypothesisTab"), {
  loading: () => <TabPanelLoading />,
});

const ProspectTab = dynamic(() => import("@/components/ProspectTab"), {
  loading: () => <TabPanelLoading />,
});

const DemoTab = dynamic(() => import("@/components/DemoTab"), {
  loading: () => <TabPanelLoading />,
});

const GongTab = dynamic(() => import("@/components/GongTab"), {
  loading: () => <TabPanelLoading />,
});

const PocTab = dynamic(() => import("@/components/PocTab"), {
  loading: () => <TabPanelLoading />,
});

const TransitionTab = dynamic(() => import("@/components/TransitionTab"), {
  loading: () => <TabPanelLoading />,
});

function TabPanelLoading() {
  return (
    <div className="tab-panel-loading" style={{ padding: 24, color: "var(--arize-text-muted, #5a5f6e)" }}>
      Loading…
    </div>
  );
}

function TabPanel({
  tab,
  activeTab,
  seenTabs,
  children,
}: {
  tab: ShareableTab;
  activeTab: ShareableTab;
  seenTabs: ReadonlySet<ShareableTab>;
  children: ReactNode;
}) {
  if (!seenTabs.has(tab)) return null;
  return (
    <div
      role="tabpanel"
      id={`stillness-panel-${tab}`}
      aria-hidden={activeTab !== tab}
      aria-label={tab}
      className={`tab-content${activeTab === tab ? " active" : ""}`}
    >
      {children}
    </div>
  );
}

type ResultTab = Exclude<ShareableTab, "pipeline" | "pocpot" | "transition">;

type SuggestUi = {
  reason: string;
  typedQuery: string;
  matches: AccountSuggestionMatch[];
  resolve: (c: "cancel" | "keep" | AccountSuggestionMatch) => void;
};

function HomeContent() {
  const searchParams = useSearchParams();
  /** Deferred from URL so server and client first paint match (useSearchParams differs on SSR). */
  const [shareQuery, setShareQuery] = useState<ShareQuery>({});
  const [activeTab, setActiveTab] = useState<ShareableTab>("prospect");
  /** Tabs stay mounted after first visit so in-tab form state is preserved. */
  const [seenTabs, setSeenTabs] = useState(() => new Set<ShareableTab>(["prospect"]));
  const [shareHints, setShareHints] = useState<Partial<ShareQuery>>({});
  const [loading, setLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState("");
  const [resultHtml, setResultHtml] = useState("");
  const [resultOwner, setResultOwner] = useState<ResultTab | null>(null);
  const [suggestUi, setSuggestUi] = useState<SuggestUi | null>(null);

  const stageHeader = useMemo(() => stageBodyCopy(activeTab), [activeTab]);

  const queryString = searchParams.toString();
  useEffect(() => {
    const q = parseShareQuery(new URLSearchParams(queryString));
    setShareQuery(q);
    if (q.tab) {
      const tab = q.tab;
      setActiveTab(tab);
      setSeenTabs((prev) => new Set(prev).add(tab));
    }
  }, [queryString]);

  useEffect(() => {
    setSeenTabs((prev) => {
      if (prev.has(activeTab)) return prev;
      const next = new Set(prev);
      next.add(activeTab);
      return next;
    });
  }, [activeTab]);

  const mergeShareHints = useCallback((patch: Partial<ShareQuery>) => {
    setShareHints((prev) => ({ ...prev, ...patch }));
  }, []);

  const onProspectShareHints = useCallback(
    (hints: ProspectShareHints) =>
      mergeShareHints({
        account_name: hints.account_name,
        domain: hints.domain,
        sfdc_account_id: hints.sfdc_account_id,
      }),
    [mergeShareHints],
  );

  const { trackProspectRun, trackGongSuccess, trackDemoBuild, trackTabUsed } = useAchievements();

  useEffect(() => {
    trackTabUsed(activeTab);
  }, [activeTab, trackTabUsed]);

  const onLoading = useCallback((msg: string) => {
    if (msg) {
      setLoading(true);
      setLoadingMessage(msg);
      setResultHtml("");
      setResultOwner(null);
    } else {
      setLoading(false);
      setLoadingMessage("");
    }
  }, []);

  /** Stops the global overlay without clearing results (e.g. while account pick modal is open). */
  const clearGlobalLoading = useCallback(() => {
    setLoading(false);
    setLoadingMessage("");
  }, []);

  const makeOnResult = useCallback(
    (tab: ResultTab) => (html: string) => {
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
    [trackProspectRun, trackGongSuccess, trackDemoBuild],
  );

  const resolveAccount = useCallback(async (input: AccountResolveInput): Promise<AccountResolveResult> => {
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
      const r = await apiPost<AccountSuggestionsResponse>("/api/account-suggestions", {
        account_name: an,
        domain: dom || null,
      });
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
        // Account-suggestions request is done; tabs already set "Resolving…". Hide spinner
        // so the user is not stuck on a loading state behind the pick-account dialog.
        clearGlobalLoading();
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
  }, [clearGlobalLoading]);

  const hideSharedResults =
    activeTab === "pipeline" || activeTab === "pocpot" || activeTab === "transition";
  const showResults = !hideSharedResults && resultOwner === activeTab && !!resultHtml;

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
        <ShareToolbar activeTab={activeTab} hints={shareHints} />
      </div>

      <SalesStageRail activeTab={activeTab} onSelectTab={setActiveTab} />

      <AchievementToast />

      <div className="card">
        <div className="tabs-container">
          <div className="stage-body-header">
            <div>
              <div className="stage-body-eyebrow">{stageHeader.eyebrow}</div>
              <div className="stage-body-tool-name">{stageHeader.toolName}</div>
            </div>
          </div>

          <div className="tab-content-wrapper">
            <TabPanel tab="pipeline" activeTab={activeTab} seenTabs={seenTabs}>
              <PipelineTab />
            </TabPanel>
            <TabPanel tab="hypothesis" activeTab={activeTab} seenTabs={seenTabs}>
              <HypothesisTab
                onLoading={onLoading}
                onResult={makeOnResult("hypothesis")}
                resolveAccount={resolveAccount}
                urlQuery={shareQuery}
                onShareHintsChange={(h) => mergeShareHints(h)}
              />
            </TabPanel>
            <TabPanel tab="prospect" activeTab={activeTab} seenTabs={seenTabs}>
              <ProspectTab
                onLoading={onLoading}
                onResult={makeOnResult("prospect")}
                resolveAccount={resolveAccount}
                urlQuery={shareQuery}
                onShareHintsChange={onProspectShareHints}
              />
            </TabPanel>
            <TabPanel tab="demo" activeTab={activeTab} seenTabs={seenTabs}>
              <DemoTab
                onLoading={onLoading}
                onResult={makeOnResult("demo")}
                resolveAccount={resolveAccount}
                urlQuery={shareQuery}
                onShareHintsChange={(h) => mergeShareHints(h)}
              />
            </TabPanel>
            <TabPanel tab="gong" activeTab={activeTab} seenTabs={seenTabs}>
              <GongTab onLoading={onLoading} onResult={makeOnResult("gong")} />
            </TabPanel>
            <TabPanel tab="pocpot" activeTab={activeTab} seenTabs={seenTabs}>
              <PocTab onLoading={onLoading} resolveAccount={resolveAccount} />
            </TabPanel>
            <TabPanel tab="transition" activeTab={activeTab} seenTabs={seenTabs}>
              <TransitionTab onLoading={onLoading} resolveAccount={resolveAccount} />
            </TabPanel>
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
      <Suspense
        fallback={
          <div className="container" style={{ padding: 32 }}>
            Loading…
          </div>
        }
      >
        <HomeContent />
      </Suspense>
    </AchievementProvider>
  );
}
