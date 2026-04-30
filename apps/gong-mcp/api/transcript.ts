import { retrieveTranscripts } from "../lib/gong";
import { jsonHandler } from "../lib/http";

export const config = { runtime: "nodejs", maxDuration: 300 } as const;

interface TranscriptBody {
  call_id?: string;
  callIds?: string[];
}

export default jsonHandler(async (raw) => {
  const body = (raw ?? {}) as TranscriptBody;
  const callIds = body.callIds ?? (body.call_id ? [body.call_id] : []);
  if (callIds.length === 0) {
    return { status: 400, body: { error: "Provide call_id or callIds" } };
  }
  const result = await retrieveTranscripts(callIds);
  return { status: 200, body: result };
});
