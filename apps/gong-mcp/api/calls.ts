import { listCallsExtensive, defaultDateRange } from "../lib/gong";
import { jsonHandler } from "../lib/http";

export const config = { runtime: "nodejs", maxDuration: 300 } as const;

interface ListCallsBody {
  from_date?: string;
  to_date?: string;
  fromDateTime?: string;
  toDateTime?: string;
}

export default jsonHandler(async (raw) => {
  const body = (raw ?? {}) as ListCallsBody;
  const defaults = defaultDateRange();
  const fromDateTime = body.fromDateTime ?? body.from_date ?? defaults.fromDateTime;
  const toDateTime = body.toDateTime ?? body.to_date ?? defaults.toDateTime;
  const result = await listCallsExtensive(fromDateTime, toDateTime);
  return { status: 200, body: result };
});
