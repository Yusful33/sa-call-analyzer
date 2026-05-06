export type ShareableTab =
  | "hypothesis"
  | "prospect"
  | "demo"
  | "gong"
  | "pocpot"
  | "transition";

export type ShareQuery = {
  tab?: ShareableTab;
  /** Prospect + demo */
  account_name?: string;
  domain?: string;
  sfdc_account_id?: string;
  /** Hypothesis */
  company_name?: string;
  company_domain?: string;
  /** Demo builder only (same as account_name sometimes) */
  demo_account?: string;
};

const VALID_TABS = new Set<ShareableTab>([
  "hypothesis",
  "prospect",
  "demo",
  "gong",
  "pocpot",
  "transition",
]);

export function parseShareQuery(searchParams: URLSearchParams): ShareQuery {
  const tabRaw = searchParams.get("tab");
  const tab =
    tabRaw && VALID_TABS.has(tabRaw as ShareableTab)
      ? (tabRaw as ShareableTab)
      : undefined;
  return {
    tab,
    account_name: searchParams.get("account_name") || undefined,
    domain: searchParams.get("domain") || undefined,
    sfdc_account_id: searchParams.get("sfdc_account_id") || undefined,
    company_name: searchParams.get("company_name") || undefined,
    company_domain: searchParams.get("company_domain") || undefined,
    demo_account: searchParams.get("demo_account") || undefined,
  };
}

export function buildStillnessShareUrl(query: ShareQuery): string {
  if (typeof window === "undefined") {
    return "";
  }
  const u = new URL(window.location.href);
  u.search = "";
  if (query.tab) u.searchParams.set("tab", query.tab);
  const set = (k: keyof ShareQuery, param: string) => {
    const v = query[k];
    if (typeof v === "string" && v.trim()) u.searchParams.set(param, v.trim());
  };
  set("account_name", "account_name");
  set("domain", "domain");
  set("sfdc_account_id", "sfdc_account_id");
  set("company_name", "company_name");
  set("company_domain", "company_domain");
  set("demo_account", "demo_account");
  return u.toString();
}
