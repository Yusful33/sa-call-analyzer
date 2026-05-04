import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";

/** Must stay aligned with `next.config.ts` / ops default FastAPI host. */
const DEFAULT_LEGACY_API = "https://arize-gtm-stillness-api.vercel.app";

function backendOrigin(): string {
  const raw = process.env.NEXT_PUBLIC_LEGACY_API_URL?.trim();
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
 */
export function middleware(request: NextRequest) {
  const { pathname, search } = request.nextUrl;
  if (pathname === "/api/health" || pathname.startsWith("/api/health/")) {
    return NextResponse.next();
  }
  if (pathname.startsWith("/api/")) {
    const base = backendOrigin();
    const dest = new URL(`${pathname}${search}`, `${base}/`);
    return NextResponse.rewrite(dest);
  }
  return NextResponse.next();
}

export const config = {
  matcher: ["/api/:path*"],
};
