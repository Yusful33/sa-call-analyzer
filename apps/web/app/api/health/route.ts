import { NextResponse } from "next/server";
import { backendOrigin, vercelProtectionBypass } from "@/lib/backendOrigin";

export async function GET() {
  const bypass = vercelProtectionBypass();
  return NextResponse.json({
    ok: true,
    service: "stillness-web",
    ts: new Date().toISOString(),
    debug: {
      fastapiOrigin: backendOrigin(),
      bypassConfigured: bypass.length > 0,
      bypassLength: bypass.length,
    },
  });
}
