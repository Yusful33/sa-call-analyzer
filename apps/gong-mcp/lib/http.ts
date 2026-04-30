import { GongApiError, GongConfigError } from "./gong";

export type JsonHandler = (body: unknown) => Promise<{ status: number; body: unknown }>;

/**
 * Wrap a Vercel Function so we always:
 *   - parse JSON body when present
 *   - return JSON content-type
 *   - translate Gong/auth errors to clean HTTP statuses
 */
export function jsonHandler(handler: JsonHandler) {
  return async function (req: Request): Promise<Response> {
    let body: unknown = undefined;
    if (req.method !== "GET" && req.method !== "HEAD") {
      try {
        body = req.headers.get("content-type")?.includes("application/json")
          ? await req.json()
          : undefined;
      } catch {
        return Response.json({ error: "Invalid JSON body" }, { status: 400 });
      }
    }
    try {
      const out = await handler(body);
      return Response.json(out.body, { status: out.status });
    } catch (err) {
      if (err instanceof GongConfigError) {
        return Response.json({ error: err.message }, { status: 500 });
      }
      if (err instanceof GongApiError) {
        return Response.json({ error: err.message }, { status: err.status });
      }
      const message = err instanceof Error ? err.message : String(err);
      return Response.json({ error: message }, { status: 500 });
    }
  };
}
