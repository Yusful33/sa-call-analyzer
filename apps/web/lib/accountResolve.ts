export type AccountSuggestionMatch = {
  id: string;
  name: string;
  website?: string | null;
  match_score?: number;
};

export type AccountSuggestionsResponse = {
  status: "ok" | "suggest" | "none";
  reason: string;
  query: string;
  domain?: string | null;
  matches: AccountSuggestionMatch[];
};

export type AccountResolveInput = {
  accountName: string;
  accountDomain?: string;
  sfdcAccountId?: string;
};

export type AccountResolveResult =
  | { proceed: true; accountName: string; accountDomain?: string; sfdcAccountId?: string }
  | { proceed: false };

export type ResolveAccountFn = (input: AccountResolveInput) => Promise<AccountResolveResult>;
