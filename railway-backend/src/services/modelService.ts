// railway-backend/src/services/modelService.ts
import { env } from "../config";
import { z } from "zod";

/** ---------- Request/Response Schemas ---------- */
export const PredictReqSchema = z.object({
  symbol: z.string().min(1),
  features: z.record(z.number()).default({}),
  timestamp: z.number().int().optional(), // ms epoch if provided
});

export const PredictRespSchema = z.object({
  signal: z.enum(["long", "short", "flat"]),
  prob_long: z.number().min(0).max(1),
  prob_short: z.number().min(0).max(1),
  confidence: z.number().min(0).max(1), // calibrated
  model_version: z.string().min(1),
  explain: z
    .object({
      regime: z.string().optional(),
      drivers: z.array(z.string()).optional(),
    })
    .optional(),
});

export type PredictReq = z.infer<typeof PredictReqSchema>;
export type PredictResp = z.infer<typeof PredictRespSchema>;

/** ---------- Error Types ---------- */
export class ModelServiceError extends Error {
  constructor(
    message: string,
    public readonly code:
      | "bad_request"
      | "timeout"
      | "unreachable"
      | "server_error"
      | "bad_response",
    public readonly status?: number,
  ) {
    super(message);
  }
}

/** ---------- Fetch utils (timeout + retry) ---------- */
const DEFAULT_TIMEOUT_MS = 3_000;
const MAX_RETRIES = 2; // total attempts = 1 + MAX_RETRIES
const JITTER_MS = 150;

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

function backoffDelay(attempt: number): number {
  // attempt: 0,1,2 -> 200, 400, 800 (+ jitter)
  const base = 200 * Math.pow(2, attempt);
  const jitter = Math.floor(Math.random() * JITTER_MS);
  return base + jitter;
}

async function fetchWithTimeout(
  url: string,
  opts: RequestInit,
  timeoutMs = DEFAULT_TIMEOUT_MS,
): Promise<Response> {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...opts, signal: controller.signal });
  } finally {
    clearTimeout(id);
  }
}

/** Classify if we should retry */
function isRetriable(status?: number, errCode?: string): boolean {
  if (errCode === "AbortError") return true; // timeout -> retry
  if (!status) return true; // likely network
  // 5xx server errors are retriable; 429 too (respect backoff if you add it)
  return status >= 500 || status === 429;
}

/** ---------- Main client ---------- */
export async function predict(
  req: PredictReq,
  {
    timeoutMs = DEFAULT_TIMEOUT_MS,
    traceId,
  }: { timeoutMs?: number; traceId?: string } = {},
): Promise<PredictResp> {
  // Validate request early (defensive)
  const parsed = PredictReqSchema.safeParse(req);
  if (!parsed.success) {
    throw new ModelServiceError(
      "Invalid predict() request payload",
      "bad_request",
    );
  }

  const url = `${env.MODEL_SERVICE_URL}/predict`;
  const body = JSON.stringify(parsed.data);
  const headers: Record<string, string> = {
    "content-type": "application/json",
  };
  if (traceId) headers["x-trace-id"] = traceId;

  let lastError: unknown;

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      const resp = await fetchWithTimeout(
        url,
        { method: "POST", headers, body },
        timeoutMs,
      );

      if (!resp.ok) {
        // Non-2xx response
        const text = await resp.text().catch(() => "");
        lastError = new ModelServiceError(
          `Model service HTTP ${resp.status}: ${text.slice(0, 300)}`,
          resp.status >= 500 || resp.status === 429
            ? "server_error"
            : "bad_request",
          resp.status,
        );
        if (isRetriable(resp.status)) {
          if (attempt < MAX_RETRIES) await sleep(backoffDelay(attempt));
          continue;
        }
        throw lastError; // non-retriable
      }

      // Parse + validate JSON
      const json = (await resp.json()) as unknown;
      const out = PredictRespSchema.safeParse(json);
      if (!out.success) {
        throw new ModelServiceError(
          `Model service returned invalid schema: ${out.error.message}`,
          "bad_response",
        );
      }

      // Optional sanity: if probs don't roughly sum to ~1, normalize
      const sum = out.data.prob_long + out.data.prob_short;
      if (sum > 1.0001 || sum < 0.9999) {
        // normalize softly, don't throw
        const pl = out.data.prob_long / sum;
        const ps = out.data.prob_short / sum;
        return { ...out.data, prob_long: pl, prob_short: ps };
      }

      return out.data;
    } catch (err: any) {
      lastError = err;
      const aborted = err?.name === "AbortError";
      const status = err?.status as number | undefined;
      if (!isRetriable(status, aborted ? "AbortError" : undefined)) throw err;
      if (attempt < MAX_RETRIES) await sleep(backoffDelay(attempt));
    }
  }

  // Ran out of retries
  if (lastError instanceof ModelServiceError) {
    // bubble up our typed error
    throw lastError;
  }
  // Unknown/network error
  throw new ModelServiceError(
    `Model service unreachable: ${(lastError as Error)?.message ?? "unknown"}`,
    "unreachable",
  );
}

/** Optional: a lightweight /health probe for startup checks */
export async function checkHealth(
  timeoutMs = 1_500,
): Promise<{ ok: boolean; model_version?: string }> {
  const url = `${env.MODEL_SERVICE_URL}/health`;
  try {
    const resp = await fetchWithTimeout(url, { method: "GET" }, timeoutMs);
    if (!resp.ok) return { ok: false };
    const j = (await resp.json()) as any;
    return { ok: Boolean(j?.ok), model_version: j?.model_version };
  } catch {
    return { ok: false };
  }
}
