import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";

/** Must stay aligned with `next.config.ts` / ops default FastAPI host. */
const DEFAULT_LEGACY_API = "https://arize-gtm-stillness-api.vercel.app";

/**
 * FastAPI origin for `/api/*` rewrites.
 *
 * Prefer **FASTAPI_ORIGIN** (or **LEGACY_API_ORIGIN**) — read at **request time** on Vercel.
 * **`NEXT_PUBLIC_LEGACY_API_URL` is often inlined at `next build` time**; changing it in the
 * Vercel UI without a **redeploy** can leave middleware pointing at an old or broken API host
 * (symptom: 500/404 on every `/api/*` call). Set `FASTAPI_ORIGIN` to override without rebuild,
 * or trigger a new production deployment after changing `NEXT_PUBLIC_LEGACY_API_URL`.
 */
function backendOrigin(): string {
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

/**
 * Proxy browser `/api/*` to FastAPI (same-origin from the UI → no CORS).
 * `app/api/health` stays on Next; everything else under `/api/` goes to the backend.
 * 
 * In development with Docker (FASTAPI_ORIGIN set), we skip middleware and let
 * next.config.ts rewrites handle the proxy (better timeout support).
 */
export function middleware(request: NextRequest) {
  const { pathname, search } = request.nextUrl;
  
  // Skip health endpoint
  if (pathname === "/api/health" || pathname.startsWith("/api/health/")) {
    return NextResponse.next();
  }
  
  // In development with Docker, let rewrites handle the proxy for better timeout support
  if (process.env.NODE_ENV === "development" && process.env.FASTAPI_ORIGIN) {
    return NextResponse.next();
  }
  
  if (pathname.startsWith("/api/")) {
    const base = backendOrigin().replace(/\/$/, "");
    let dest: URL;
    try {
      dest = new URL(`${pathname}${search}`, `${base}/`);
    } catch {
      dest = new URL(`${pathname}${search}`, `${DEFAULT_LEGACY_API}/`);
    }
    return NextResponse.rewrite(dest);
  }
  return NextResponse.next();
}

export const config = {
  matcher: ["/api/:path*"],
};
