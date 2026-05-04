import { getCallInfo } from "../lib/gong.js";
import { jsonHandler } from "../lib/http.js";

export const config = { runtime: "nodejs", maxDuration: 300 } as const;

interface CallInfoBody {
  call_id?: string;
  callId?: string;
}

export default jsonHandler(async (raw) => {
  const body = (raw ?? {}) as CallInfoBody;
  const callId = body.callId ?? body.call_id;
  if (!callId) {
    return { status: 400, body: { error: "Provide call_id" } };
  }
  const result = await getCallInfo(callId);
  return { status: 200, body: result };
});
