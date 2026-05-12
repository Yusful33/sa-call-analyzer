import type { NextRequest } from "next/server";
import { backendOrigin, vercelProtectionBypass } from "@/lib/backendOrigin";

/**
 * Server-side proxy to FastAPI (replaces Edge `middleware` external rewrites, which are flaky on Vercel).
 * Adds `x-vercel-protection-bypass` when configured. `app/api/health/route.ts` stays more specific than this catch-all.
 */
export const runtime = "nodejs";
export const maxDuration = 800;

/**
 * Endpoints that return stable data and can be cached.
 * Map of path suffix -> cache duration in seconds.
 */
const CACHEABLE_GET_ENDPOINTS: Record<string, number> = {
  "pipeline-user-options": 300, // 5 minutes - user list rarely changes
};

function buildTargetUrl(request: NextRequest, pathSegments: string[] | undefined): string {
  const u = new URL(request.url);
  const suffix =
    pathSegments && pathSegments.length > 0 ? `/${pathSegments.join("/")}` : "";
  const pathname = `/api${suffix}`;
  const base = backendOrigin().replace(/\/$/, "");
  return `${base}${pathname}${u.search}`;
}

function forwardRequestHeaders(request: NextRequest): Headers {
  const out = new Headers();
  const pass = [
    "content-type",
    "accept",
    "accept-encoding",
    "accept-language",
    "authorization",
    "user-agent",
    "x-request-id",
  ];
  for (const name of pass) {
    const v = request.headers.get(name);
    if (v) out.set(name, v);
  }
  const bypass = vercelProtectionBypass();
  if (bypass) {
    out.set("x-vercel-protection-bypass", bypass);
  }
  return out;
}

/**
 * Check if a path is cacheable and return the cache duration.
 */
function getCacheDuration(method: string, pathSegments: string[] | undefined): number {
  if (method !== "GET") return 0;
  const lastSegment = pathSegments?.[pathSegments.length - 1];
  if (!lastSegment) return 0;
  return CACHEABLE_GET_ENDPOINTS[lastSegment] ?? 0;
}

async function proxy(request: NextRequest, pathSegments: string[] | undefined): Promise<Response> {
  const target = buildTargetUrl(request, pathSegments);
  const headers = forwardRequestHeaders(request);
  const method = request.method.toUpperCase();

  const init: RequestInit & { duplex?: "half" } = {
    method,
    headers,
    redirect: "manual",
  };

  if (method !== "GET" && method !== "HEAD") {
    if (request.body) {
      init.body = request.body;
      init.duplex = "half";
    }
  }

  const upstream = await fetch(target, init);
  const contentType = upstream.headers.get("content-type") || "application/json";
  
  // Use arrayBuffer for binary content (ZIP files, images, etc.)
  const isBinary = contentType.includes("application/zip") || 
                   contentType.includes("application/octet-stream") ||
                   contentType.includes("image/");
  const body = isBinary ? await upstream.arrayBuffer() : await upstream.text();

  // Build response headers - forward important headers from upstream
  const responseHeaders: Record<string, string> = {
    "content-type": contentType,
  };

  // Forward headers needed for file downloads and custom API responses
  const forwardHeaders = [
    "content-disposition",
    "x-demo-arize-push",
    "x-demo-arize-push-detail",
    "x-demo-project-name",
  ];
  for (const name of forwardHeaders) {
    const value = upstream.headers.get(name);
    if (value) {
      responseHeaders[name] = value;
    }
  }

  // Add caching headers for cacheable GET endpoints
  const cacheDuration = getCacheDuration(method, pathSegments);
  if (cacheDuration > 0 && upstream.status === 200) {
    // Use stale-while-revalidate: serve stale content while fetching fresh in background
    responseHeaders["cache-control"] = `public, s-maxage=${cacheDuration}, stale-while-revalidate=${cacheDuration * 2}`;
  } else {
    // Don't cache errors or non-cacheable endpoints
    responseHeaders["cache-control"] = "no-store";
  }

  return new Response(body, {
    status: upstream.status,
    headers: responseHeaders,
  });
}

type RouteCtx = { params: Promise<{ path?: string[] }> };

export async function GET(request: NextRequest, ctx: RouteCtx) {
  const { path } = await ctx.params;
  return proxy(request, path);
}

export async function POST(request: NextRequest, ctx: RouteCtx) {
  const { path } = await ctx.params;
  return proxy(request, path);
}

export async function PUT(request: NextRequest, ctx: RouteCtx) {
  const { path } = await ctx.params;
  return proxy(request, path);
}

export async function PATCH(request: NextRequest, ctx: RouteCtx) {
  const { path } = await ctx.params;
  return proxy(request, path);
}

export async function DELETE(request: NextRequest, ctx: RouteCtx) {
  const { path } = await ctx.params;
  return proxy(request, path);
}

export async function HEAD(request: NextRequest, ctx: RouteCtx) {
  const { path } = await ctx.params;
  return proxy(request, path);
}

export async function OPTIONS(request: NextRequest, ctx: RouteCtx) {
  const { path } = await ctx.params;
  return proxy(request, path);
}
