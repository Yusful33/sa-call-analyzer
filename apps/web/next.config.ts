import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  serverExternalPackages: [
    "@opentelemetry/sdk-trace-node",
    "@opentelemetry/sdk-trace-base",
    "@opentelemetry/exporter-trace-otlp-proto",
    "@opentelemetry/resources",
    "@opentelemetry/semantic-conventions",
    "@opentelemetry/api",
  ],
  /**
   * `/api/*` (except `app/api/health`) is proxied to FastAPI by `app/api/[[...path]]/route.ts`
   * so we can attach `x-vercel-protection-bypass` in Node.js (Edge middleware external rewrites
   * were unreliable and could surface as 404 on Vercel).
   */
};

export default nextConfig;
