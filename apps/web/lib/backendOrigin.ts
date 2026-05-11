/** Duplicated from former `middleware.ts` — single source for FastAPI base URL + bypass env names. */
const DEFAULT_LEGACY_API = "https://arize-gtm-stillness-api.vercel.app";

/**
 * FastAPI origin for `/api/*` proxying (Route Handler + legacy callers).
 * Prefer **FASTAPI_ORIGIN** / **LEGACY_API_ORIGIN** at runtime on Vercel.
 * **NEXT_PUBLIC_LEGACY_API_URL** is inlined at build; use runtime vars to repoint without rebuild.
 */
export function backendOrigin(): string {
  const runtime =
    (process.env.FASTAPI_ORIGIN ?? "").trim() ||
    (process.env.LEGACY_API_ORIGIN ?? "").trim();
  const pub = (process.env.NEXT_PUBLIC_LEGACY_API_URL ?? "").trim();
  const raw = runtime || pub;

  if (process.env.NODE_ENV === "development") {
    if (!raw) return "http://localhost:8080";
    if (!raw.startsWith("http://") && !raw.startsWith("https://")) {
      return `https://${raw}`.replace(/\/$/, "");
    }
    return raw.replace(/\/$/, "");
  }
  if (!raw) {
    return DEFAULT_LEGACY_API;
  }
  if (!raw.startsWith("http://") && !raw.startsWith("https://")) {
    return `https://${raw}`.replace(/\/$/, "");
  }
  return raw.replace(/\/$/, "");
}

export function vercelProtectionBypass(): string {
  return (
    (process.env.FASTAPI_VERCEL_PROTECTION_BYPASS ?? "").trim() ||
    (process.env.VERCEL_AUTOMATION_BYPASS_SECRET ?? "").trim()
  );
}
