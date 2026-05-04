/** Trim; treat blank as unset. `""` would otherwise win over `??` and break URL resolution. */
function normalizeApiOrigin(raw: string | undefined): string | undefined {
  const u = raw?.trim();
  if (!u) return undefined;
  if (!u.startsWith("http://") && !u.startsWith("https://")) {
    return `https://${u}`.replace(/\/$/, "");
  }
  return u.replace(/\/$/, "");
}

const isDev = process.env.NODE_ENV === "development";

/**
 * In production on Vercel, use same-origin `/api/...` so `next.config.ts` **fallback rewrites**
 * proxy to FastAPI (avoids CORS and survives a blank `NEXT_PUBLIC_LEGACY_API_URL` in the client).
 * In development, call the local FastAPI origin directly.
 */
const BASE = isDev
  ? normalizeApiOrigin(process.env.NEXT_PUBLIC_LEGACY_API_URL) ?? "http://localhost:8080"
  : "";

/** CrewAI-heavy routes (second Vercel project when `NEXT_PUBLIC_CREW_API_URL` is set). */
const CREW_API_PREFIXES = [
  "/api/analyze-prospect-stream",
  "/api/analyze-prospect",
  "/api/generate-recap-slide",
  "/api/analyze",
] as const;

function baseUrlForPath(path: string): string {
  const crewBase = normalizeApiOrigin(process.env.NEXT_PUBLIC_CREW_API_URL);
  const legacyBase = BASE.replace(/\/$/, "");
  if (!crewBase || crewBase === legacyBase) return BASE;
  for (const prefix of CREW_API_PREFIXES) {
    if (
      path === prefix ||
      path.startsWith(`${prefix}?`) ||
      path.startsWith(`${prefix}/`)
    ) {
      return crewBase;
    }
  }
  return BASE;
}

/** Parse Content-Disposition for a download filename (handles quoted and RFC 5987 forms). */
function parseFilenameFromContentDisposition(cd: string | null): string | null {
  if (!cd) return null;
  const star = cd.match(/filename\*\s*=\s*(?:UTF-8''|utf-8'')([^;\s]+)/i);
  if (star?.[1]) {
    try {
      return decodeURIComponent(star[1].trim().replace(/["']/g, ""));
    } catch {
      /* fall through */
    }
  }
  const quoted = cd.match(/filename\s*=\s*"((?:\\.|[^"\\])*)"/i);
  if (quoted?.[1]) return quoted[1].replace(/\\"/g, '"');
  const plain = cd.match(/filename\s*=\s*([^;\s]+)/i);
  if (plain?.[1]) return plain[1].trim().replace(/^["']|["']$/g, "");
  return null;
}

export async function apiPost<T = unknown>(
  path: string,
  body: Record<string, unknown>,
  opts?: { signal?: AbortSignal; timeout?: number }
): Promise<T> {
  const controller = new AbortController();
  let tid: ReturnType<typeof setTimeout> | undefined;
  if (opts?.timeout) {
    tid = setTimeout(() => controller.abort(), opts.timeout);
  }
  const signal = opts?.signal ?? controller.signal;

  const res = await fetch(`${baseUrlForPath(path)}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (tid) clearTimeout(tid);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      if (j.detail) detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export async function apiPostBlob(
  path: string,
  body: Record<string, unknown>,
  /** Used when the browser cannot see Content-Disposition (CORS) or the header is missing. */
  filenameFallback?: string
): Promise<{ blob: Blob; filename: string }> {
  const res = await fetch(`${baseUrlForPath(path)}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      if (j.detail) detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  const blob = await res.blob();
  let filename =
    parseFilenameFromContentDisposition(res.headers.get("Content-Disposition")) ??
    filenameFallback?.trim() ??
    "";
  if (!filename || filename === "download") {
    filename = path.includes("recap") ? "Recap_Slide.pptx" : "document.docx";
  }
  if (!/\.(docx|pptx)$/i.test(filename)) {
    filename += path.includes("recap") ? ".pptx" : ".docx";
  }
  return { blob, filename };
}

export function apiStreamPost(
  path: string,
  body: Record<string, unknown>,
  opts?: { signal?: AbortSignal; timeout?: number }
) {
  const controller = new AbortController();
  let tid: ReturnType<typeof setTimeout> | undefined;
  if (opts?.timeout) {
    tid = setTimeout(() => controller.abort(), opts.timeout);
  }
  const signal = opts?.signal ?? controller.signal;

  const promise = fetch(`${baseUrlForPath(path)}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });

  return { promise, controller, clearTimeout: () => tid && clearTimeout(tid) };
}
