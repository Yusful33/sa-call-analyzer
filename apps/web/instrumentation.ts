export async function register() {
  // Skip in Edge runtime (middleware) — OTel Node SDK requires Node.js APIs
  if (typeof globalThis.EdgeRuntime !== "undefined") {
    return;
  }

  const { NodeTracerProvider } = await import("@opentelemetry/sdk-trace-node");
  const { OTLPTraceExporter } = await import("@opentelemetry/exporter-trace-otlp-proto");
  const { BatchSpanProcessor, ConsoleSpanExporter } = await import("@opentelemetry/sdk-trace-base");
  const { ATTR_SERVICE_NAME } = await import("@opentelemetry/semantic-conventions");
  const { resourceFromAttributes } = await import("@opentelemetry/resources");
  const spaceId = process.env.ARIZE_SPACE_ID;
  const apiKey = process.env.ARIZE_API_KEY;

  if (!spaceId || !apiKey) {
    console.warn(
      "[instrumentation] ARIZE_SPACE_ID or ARIZE_API_KEY not set — telemetry disabled. " +
        "Set both in your .env or docker-compose environment to send traces to Arize."
    );
    return;
  }

  const projectName = process.env.ARIZE_PROJECT_NAME ?? "stillness-web";

  const resource = resourceFromAttributes({
    [ATTR_SERVICE_NAME]: projectName,
    "openinference.project.name": projectName,
  });

  const processors = [
    new BatchSpanProcessor(
      new OTLPTraceExporter({
        url: "https://otlp.arize.com/v1/traces",
        headers: {
          authorization: `Bearer ${apiKey}`,
          "arize-space-id": spaceId,
        },
      })
    ),
  ];

  if (process.env.OTEL_LOG_TO_CONSOLE === "true") {
    processors.push(new BatchSpanProcessor(new ConsoleSpanExporter()));
  }

  const provider = new NodeTracerProvider({
    resource,
    spanProcessors: processors,
  });

  provider.register();
  console.log(
    `[instrumentation] Arize OTEL tracing enabled — project="${projectName}"`
  );

  if (typeof process.on === "function") {
    process.on("SIGTERM", () => provider.shutdown());
  }
}
