import { listCallsExtensive, defaultDateRange } from "../lib/gong.js";
import { jsonHandler } from "../lib/http.js";

export const config = { runtime: "nodejs", maxDuration: 300 } as const;

interface ListCallsBody {
  from_date?: string;
  to_date?: string;
  fromDateTime?: string;
  toDateTime?: string;
  /** Cap Gong list + extensive fetch (newest-first pages) to avoid multi-minute responses. */
  max_calls?: number;
}

export default jsonHandler(async (raw) => {
  const body = (raw ?? {}) as ListCallsBody;
  const defaults = defaultDateRange();
  const fromDateTime = body.fromDateTime ?? body.from_date ?? defaults.fromDateTime;
  const toDateTime = body.toDateTime ?? body.to_date ?? defaults.toDateTime;
  const maxCalls =
    typeof body.max_calls === "number" && body.max_calls > 0
      ? Math.min(body.max_calls, 2000)
      : undefined;
  const result = await listCallsExtensive(fromDateTime, toDateTime, maxCalls);
  return { status: 200, body: result };
});
