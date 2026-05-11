import { NextResponse } from "next/server";
import { backendOrigin, vercelProtectionBypass } from "@/lib/backendOrigin";

/**
 * Warmup endpoint that pings the FastAPI backend to keep it warm.
 * Called by Vercel Cron to prevent cold starts.
 */
export const runtime = "nodejs";
export const maxDuration = 30;

export async function GET() {
  const start = Date.now();
  const results: Record<string, { ok: boolean; ms: number; error?: string }> = {};

  // Warm up the FastAPI backend health endpoint
  try {
    const headers: Record<string, string> = {};
    const bypass = vercelProtectionBypass();
    if (bypass) {
      headers["x-vercel-protection-bypass"] = bypass;
    }

    const apiStart = Date.now();
    const res = await fetch(`${backendOrigin()}/health`, {
      method: "GET",
      headers,
    });
    results.fastapi = {
      ok: res.ok,
      ms: Date.now() - apiStart,
    };
  } catch (e) {
    results.fastapi = {
      ok: false,
      ms: Date.now() - start,
      error: e instanceof Error ? e.message : String(e),
    };
  }

  // Also warm up the user options endpoint (commonly used)
  try {
    const headers: Record<string, string> = {};
    const bypass = vercelProtectionBypass();
    if (bypass) {
      headers["x-vercel-protection-bypass"] = bypass;
    }

    const usersStart = Date.now();
    const res = await fetch(`${backendOrigin()}/api/pipeline-user-options`, {
      method: "GET",
      headers,
    });
    results.users = {
      ok: res.ok,
      ms: Date.now() - usersStart,
    };
  } catch (e) {
    results.users = {
      ok: false,
      ms: Date.now() - start,
      error: e instanceof Error ? e.message : String(e),
    };
  }

  return NextResponse.json({
    ok: Object.values(results).every((r) => r.ok),
    totalMs: Date.now() - start,
    results,
    ts: new Date().toISOString(),
  });
}
