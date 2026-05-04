import { NextResponse } from "next/server";

export async function GET() {
  return NextResponse.json({
    ok: true,
    service: "stillness-web",
    ts: new Date().toISOString(),
  });
}
