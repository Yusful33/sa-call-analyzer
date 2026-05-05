import type { NextConfig } from "next";

/** Duplicated in `middleware.ts` (Edge) for API proxy default. */
const PROD_DEFAULT_LEGACY_API = "https://arize-gtm-stillness-api.vercel.app";

function legacyApiOriginForRewrites(): string {
  const raw = process.env.NEXT_PUBLIC_LEGACY_API_URL?.trim();
  if (!raw) {
    return process.env.NODE_ENV === "development" ? "" : PROD_DEFAULT_LEGACY_API;
  }
  if (!raw.startsWith("http://") && !raw.startsWith("https://")) {
    return `https://${raw}`.replace(/\/$/, "");
  }
  return raw.replace(/\/$/, "");
}

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
  // Increase proxy timeout for long-running API calls (demo-insights, prospect-overview)
  experimental: {
    proxyTimeout: 300000, // 5 minutes
  },
  /**
   * Rewrites for API proxy. In development, uses FASTAPI_ORIGIN env var.
   * In production, middleware.ts handles the proxy for Vercel Edge.
   */
  async rewrites() {
    const devOrigin = process.env.FASTAPI_ORIGIN || process.env.NEXT_PUBLIC_LEGACY_API_URL;
    const origin = process.env.NODE_ENV === "development" 
      ? (devOrigin || "http://localhost:8080")
      : legacyApiOriginForRewrites();
    
    if (!origin) {
      return [];
    }
    return {
      beforeFiles: [
        {
          source: "/api/:path*",
          destination: `${origin}/api/:path*`,
          // Skip the health endpoint (handled by Next.js)
          has: [
            {
              type: "header",
              key: "x-skip-middleware",
              value: undefined,
            },
          ],
        },
      ],
      fallback: [
        {
          source: "/api/:path*",
          destination: `${origin}/api/:path*`,
        },
      ],
    };
  },
};

export default nextConfig;
