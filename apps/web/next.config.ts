import type { NextConfig } from "next";

/** Fallback when `NEXT_PUBLIC_LEGACY_API_URL` is unset at build (must match ops default). */
const PROD_DEFAULT_LEGACY_API = "https://arize-gtm-stillness-api-six.vercel.app";

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
  async rewrites() {
    const origin = legacyApiOriginForRewrites();
    if (!origin || process.env.NODE_ENV === "development") {
      return [];
    }
    return {
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
