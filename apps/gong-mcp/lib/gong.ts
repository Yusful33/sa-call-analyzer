/**
 * Direct Gong API helpers (no MCP subprocess).
 *
 * Each Vercel Function imports this and calls Gong over HTTPS using
 * GONG_ACCESS_KEY + GONG_ACCESS_SECRET (or GONG_SECRET_KEY) env vars.
 */

const GONG_API_BASE = "https://api.gong.io/v2";

export class GongConfigError extends Error {}
export class GongApiError extends Error {
  public status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

function basicAuthHeader(): string {
  const accessKey = process.env.GONG_ACCESS_KEY;
  const secretKey = process.env.GONG_ACCESS_SECRET || process.env.GONG_SECRET_KEY;
  if (!accessKey || !secretKey) {
    throw new GongConfigError(
      "GONG_ACCESS_KEY and GONG_ACCESS_SECRET (or GONG_SECRET_KEY) must be set"
    );
  }
  const token = Buffer.from(`${accessKey}:${secretKey}`).toString("base64");
  return `Basic ${token}`;
}

export async function gongRequest<T = unknown>(
  endpoint: string,
  init: { method?: "GET" | "POST"; body?: unknown } = {}
): Promise<T> {
  const url = `${GONG_API_BASE}${endpoint}`;
  const res = await fetch(url, {
    method: init.method ?? "GET",
    headers: {
      Authorization: basicAuthHeader(),
      "Content-Type": "application/json",
    },
    body: init.body ? JSON.stringify(init.body) : undefined,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new GongApiError(res.status, `Gong API ${res.status} ${text}`);
  }
  return (await res.json()) as T;
}

export async function listCallIds(
  fromDateTime: string,
  toDateTime: string,
  maxIds?: number
): Promise<string[]> {
  const ids: string[] = [];
  let cursor: string | undefined;

  do {
    const qs = new URLSearchParams({ fromDateTime, toDateTime });
    if (cursor) qs.set("cursor", cursor);
    const page = (await gongRequest(`/calls?${qs.toString()}`)) as {
      calls?: Array<{ id?: string }>;
      records?: { cursor?: string };
    };
    if (Array.isArray(page.calls)) {
      ids.push(...page.calls.map((c) => c.id).filter((x): x is string => Boolean(x)));
    }
    if (maxIds !== undefined && maxIds > 0 && ids.length >= maxIds) {
      return ids.slice(0, maxIds);
    }
    cursor = page.records?.cursor;
  } while (cursor);

  return maxIds !== undefined && maxIds > 0 ? ids.slice(0, maxIds) : ids;
}

interface ExtensiveCall {
  metaData?: Record<string, unknown>;
  parties?: unknown[];
  accountName?: string;
  account?: { name?: string };
}

interface FormattedCall {
  id: unknown;
  title: unknown;
  scheduled: unknown;
  started: unknown;
  duration: unknown;
  url: unknown;
  direction: unknown;
  accountName: string | null;
  parties: unknown[];
}

function formatCall(call: ExtensiveCall): FormattedCall {
  const meta = (call.metaData ?? call) as Record<string, unknown>;
  const accountName =
    call.accountName ??
    call.account?.name ??
    (meta as { accountName?: string }).accountName ??
    (meta as { account?: { name?: string } }).account?.name ??
    null;
  return {
    id: meta.id,
    title: meta.title,
    scheduled: meta.scheduled,
    started: meta.started,
    duration: meta.duration,
    url: meta.url,
    direction: meta.direction,
    accountName: accountName ?? null,
    parties: call.parties ?? [],
  };
}

export async function listCallsExtensive(
  fromDateTime: string,
  toDateTime: string,
  maxCalls?: number
): Promise<{ calls: FormattedCall[]; total: number }> {
  const ids = await listCallIds(fromDateTime, toDateTime, maxCalls);
  if (ids.length === 0) return { calls: [], total: 0 };

  const batchSize = 50;
  const all: FormattedCall[] = [];
  for (let i = 0; i < ids.length; i += batchSize) {
    const batch = ids.slice(i, i + batchSize);
    const res = (await gongRequest("/calls/extensive", {
      method: "POST",
      body: {
        filter: { callIds: batch },
        contentSelector: { exposedFields: { parties: true, content: { structure: true } } },
      },
    })) as { calls?: ExtensiveCall[] };
    if (Array.isArray(res.calls)) {
      all.push(...res.calls.map(formatCall));
    }
  }
  return { calls: all, total: all.length };
}

export async function getCallInfo(callId: string): Promise<FormattedCall | { error: string }> {
  const res = (await gongRequest("/calls/extensive", {
    method: "POST",
    body: {
      filter: { callIds: [callId] },
      contentSelector: { exposedFields: { parties: true, content: { structure: true } } },
    },
  })) as { calls?: ExtensiveCall[] };
  const call = res.calls?.[0];
  if (!call) return { error: "Call not found" };
  return formatCall(call);
}

export async function retrieveTranscripts(callIds: string[]): Promise<unknown> {
  return gongRequest("/calls/transcript", {
    method: "POST",
    body: { filter: { callIds } },
  });
}

export function defaultDateRange(): { fromDateTime: string; toDateTime: string } {
  const now = new Date();
  const ninetyDaysAgo = new Date(now.getTime() - 90 * 24 * 60 * 60 * 1000);
  return { fromDateTime: ninetyDaysAgo.toISOString(), toDateTime: now.toISOString() };
}
