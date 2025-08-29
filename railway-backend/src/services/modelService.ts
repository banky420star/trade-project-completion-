import { env } from "../config";
export type PredictReq = { symbol: string; features: Record<string, number>; timestamp?: number };
export type PredictResp = { signal: "long"|"short"|"flat"; prob_long: number; prob_short: number; confidence: number; model_version: string; };
export async function predict(req: PredictReq): Promise<PredictResp> {
const r = await fetch(`${env.MODEL_SERVICE_URL}/predict`, { method:"POST", headers:{ "content-type":"application/json" }, body: JSON.stringify(req) });
if (!r.ok) throw new Error(`modelService ${r.status}`);
return await r.json() as PredictResp;
}
