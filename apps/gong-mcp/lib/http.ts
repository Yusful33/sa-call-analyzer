import type { VercelApiHandler, VercelRequest, VercelResponse } from "@vercel/node";
import { GongApiError, GongConfigError } from "./gong.js";

export type JsonHandler = (body: unknown) => Promise<{ status: number; body: unknown }>;

function parseJsonBody(req: VercelRequest): unknown {
  const b = req.body;
  if (b === undefined || b === null) {
    return undefined;
  }
  if (typeof b === "string") {
    try {
      return JSON.parse(b);
    } catch {
      return undefined;
    }
  }
  return b;
}

/**
 * Classic Vercel Node handler (`VercelApiHandler`). The Web `{ fetch }` export
 * is not reliably invoked for `api/*.ts` on the **Other** framework preset.
 */
export function jsonHandler(handler: JsonHandler): VercelApiHandler {
  return async (req: VercelRequest, res: VercelResponse): Promise<void> => {
    let body: unknown = undefined;
    if (req.method !== "GET" && req.method !== "HEAD") {
      try {
        const ct = String(req.headers["content-type"] ?? "");
        if (ct.includes("application/json")) {
          body = parseJsonBody(req);
        }
      } catch {
        res.status(400).json({ error: "Invalid JSON body" });
        return;
      }
    }
    try {
      const out = await handler(body);
      res.status(out.status).json(out.body);
    } catch (err) {
      if (err instanceof GongConfigError) {
        res.status(500).json({ error: err.message });
        return;
      }
      if (err instanceof GongApiError) {
        res.status(err.status).json({ error: err.message });
        return;
      }
      const message = err instanceof Error ? err.message : String(err);
      res.status(500).json({ error: message });
    }
  };
}
