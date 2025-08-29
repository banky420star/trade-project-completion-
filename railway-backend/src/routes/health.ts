import { Router } from "express";
import { env } from "../config";
const r = Router();
r.get("/health", (_req,res) => res.json({ ok:true, env: env.NODE_ENV, commit: env.COMMIT_SHA ?? "dev", tradingMode: env.TRADING_MODE, modelService: env.MODEL_SERVICE_URL }));
export default r;
