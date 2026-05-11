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

  // Debug route: /api/_debug returns proxy config info
  if (pathSegments?.[0] === "_debug") {
    return NextResponse.json({
      target: backendOrigin(),
      bypassConfigured: vercelProtectionBypass().length > 0,
      bypassLength: vercelProtectionBypass().length,
      fullTarget: target,
      headersToSend: Object.fromEntries(headers.entries()),
    });
  }

  // Debug route: /api/_test-fetch does a test fetch and returns debug info
  if (pathSegments?.[0] === "_test-fetch") {
    try {
      const testTarget = `${backendOrigin()}/api/pipeline-user-options`;
      const testResp = await fetch(testTarget, { headers, method: "GET" });
      const testBody = await testResp.text();
      return NextResponse.json({
        target: testTarget,
        status: testResp.status,
        statusText: testResp.statusText,
        contentType: testResp.headers.get("content-type"),
        contentLength: testResp.headers.get("content-length"),
        bodyLength: testBody.length,
        bodyPreview: testBody.slice(0, 200),
      });
    } catch (error) {
      return NextResponse.json({
        error: String(error),
      }, { status: 500 });
    }
  }

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

  // Read the body as text first
  const bodyText = await upstream.text();

  return new NextResponse(bodyText, {
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
