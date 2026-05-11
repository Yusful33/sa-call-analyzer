import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";
import { backendOrigin, vercelProtectionBypass } from "@/lib/backendOrigin";

/**
 * Server-side proxy to FastAPI (replaces Edge `middleware` external rewrites, which are flaky on Vercel).
 * Adds `x-vercel-protection-bypass` when configured. `app/api/health/route.ts` stays more specific than this catch-all.
 */
export const runtime = "nodejs";
export const maxDuration = 800;

const HOP_BY_HOP = new Set([
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailers",
  "transfer-encoding",
  "upgrade",
]);

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

function sanitizeResponseHeaders(src: Headers): Headers {
  const out = new Headers();
  src.forEach((value, key) => {
    if (!HOP_BY_HOP.has(key.toLowerCase())) {
      out.set(key, value);
    }
  });
  return out;
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

  // Read the body as an ArrayBuffer to avoid streaming issues with chunked responses
  const bodyBuffer = await upstream.arrayBuffer();

  return new NextResponse(bodyBuffer, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers: sanitizeResponseHeaders(upstream.headers),
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
