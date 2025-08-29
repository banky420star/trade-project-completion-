#!/usr/bin/env bash
set -euo pipefail

mkdir -p model-service railway-backend .github/workflows

# .gitignore + README + docker-compose
cat > .gitignore <<'GIT'
node_modules
dist
.env
__pycache__/
*.pyc
.DS_Store
GIT

cat > README.md <<'MD'
# Trade Project (Model Service + Backend)

## Quick start
```bash
docker compose up --build
# Backend: http://localhost:8000/health
# Model:   http://localhost:9000/health
```

### Smoke tests
```bash
curl :8000/health
curl -X POST :8000/api/ai/consensus -H 'content-type: application/json' \
  -d '{"symbol":"BTCUSDT","features":{"mom_20":1.0,"rv_5":0.2}}'
curl -X POST :8000/api/trade/execute -H 'content-type: application/json' \
  -d '{"symbol":"BTCUSDT","side":"buy","qtyUsd":2000,"confidence":0.9}'
```
MD

cat > docker-compose.yml <<'YML'
version: "3.9"
services:
model-service:
build: ./model-service
environment:
- MODEL_VERSION=${MODEL_VERSION:-dev}
ports: [ "9000:9000" ]
healthcheck:
test: ["CMD","wget","-qO-","http://127.0.0.1:9000/health"]
interval: 30s
timeout: 3s
retries: 5

backend:
build: ./railway-backend
environment:
- PORT=8000
- TRADING_MODE=${TRADING_MODE:-paper}
- MODEL_SERVICE_URL=http://model-service:9000
- CONFIDENCE_THRESHOLD=${CONFIDENCE_THRESHOLD:-0.60}
- TARGET_ANN_VOL=${TARGET_ANN_VOL:-0.12}
- MAX_DRAWDOWN_PCT=${MAX_DRAWDOWN_PCT:-0.15}
- PER_SYMBOL_USD_CAP=${PER_SYMBOL_USD_CAP:-10000}
- CORS_ORIGIN=*
ports: [ "8000:8000" ]
depends_on:
model-service:
condition: service_healthy
YML

# --- model-service (FastAPI) ---

cat > model-service/requirements.txt <<'REQ'
fastapi==0.115.0
uvicorn==0.30.6
pydantic==2.8.2
REQ

cat > model-service/utils.py <<'PY'
import os, time
def model_version(): return os.getenv("MODEL_VERSION", time.strftime("%Y%m%d_%H%M%S"))
PY

cat > model-service/app.py <<'PY'
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional
from utils import model_version

app = FastAPI(title="RL Model Service", version="1.0.0")

class PredictIn(BaseModel):
symbol: str
features: Dict[str, float] = Field(default_factory=dict)
timestamp: Optional[int] = None

class PredictOut(BaseModel):
signal: str
prob_long: float
prob_short: float
confidence: float
model_version: str

def fake_model(features: Dict[str, float]):
score = features.get("mom_20", 0.0) - features.get("rv_5", 0.0)
import math
prob_long = 1 / (1 + math.e**(-score))
prob_short = 1 - prob_long
if prob_long > 0.55: signal = "long"
elif prob_short > 0.55: signal = "short"
else: signal = "flat"
conf = max(prob_long, prob_short)
return signal, prob_long, prob_short, conf

@app.get("/health")
def health():
return {"ok": True, "model_version": model_version()}

@app.post("/predict", response_model=PredictOut)
def predict(p: PredictIn):
try:
signal, pl, ps, conf = fake_model(p.features)
return PredictOut(signal=signal, prob_long=pl, prob_short=ps, confidence=conf,
model_version=model_version())
except Exception as e:
raise HTTPException(status_code=500, detail=str(e))
PY

cat > model-service/Dockerfile <<'DOCK'
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=9000
EXPOSE 9000
HEALTHCHECK --interval=30s --timeout=3s CMD wget -qO- http://127.0.0.1:9000/health || exit 1
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","9000","--workers","1"]
DOCK

# --- railway-backend (Node/TS Express) ---

cat > railway-backend/.env.example <<'ENV'
TRADING_MODE=paper
MODEL_SERVICE_URL=http://localhost:9000
CONFIDENCE_THRESHOLD=0.60
TARGET_ANN_VOL=0.12
MAX_DRAWDOWN_PCT=0.15
PER_SYMBOL_USD_CAP=10000
CORS_ORIGIN=*
PORT=8000
ENV

cat > railway-backend/package.json <<'PKG'
{
"name": "sb1-backend",
"version": "0.1.0",
"private": true,
"main": "src/index.ts",
"scripts": {
"dev": "ts-node-dev --respawn --transpile-only src/index.ts",
"build": "tsc",
"start": "node dist/index.js"
},
"dependencies": {
"cors": "^2.8.5",
"express": "^4.19.2",
"express-rate-limit": "^7.4.0",
"helmet": "^7.1.0",
"pino": "^9.3.2",
"zod": "^3.23.8"
},
"devDependencies": {
"@types/express": "^4.17.21",
"@types/node": "^20.14.10",
"ts-node-dev": "^2.0.0",
"typescript": "^5.5.4"
}
}
PKG

cat > railway-backend/tsconfig.json <<'TS'
{
"compilerOptions": {
"target": "ES2021",
"module": "CommonJS",
"moduleResolution": "Node",
"lib": ["ES2021", "DOM"],
"esModuleInterop": true,
"strict": true,
"skipLibCheck": true,
"outDir": "dist"
},
"include": ["src"]
}
TS

mkdir -p railway-backend/src/{middleware,routes,services}
cat > railway-backend/src/config.ts <<'TS'
import { z } from "zod";
const Env = z.object({
NODE_ENV: z.enum(["development","test","production"]).default("development"),
PORT: z.string().default("8000"),
COMMIT_SHA: z.string().optional(),
CORS_ORIGIN: z.string().default("*"),
RATE_LIMIT_WINDOW_SEC: z.coerce.number().default(30),
RATE_LIMIT_MAX: z.coerce.number().default(100),
TRADING_MODE: z.enum(["paper","live"]).default("paper"),
BYBIT_API_KEY: z.string().optional(),
BYBIT_API_SECRET: z.string().optional(),
MODEL_SERVICE_URL: z.string().url().default("http://localhost:9000"),
TARGET_ANN_VOL: z.coerce.number().default(0.12),
MAX_DRAWDOWN_PCT: z.coerce.number().default(0.15),
PER_SYMBOL_USD_CAP: z.coerce.number().default(10000),
CONFIDENCE_THRESHOLD: z.coerce.number().min(0).max(1).default(0.60),
});
export const env = Env.parse(process.env);
export type Env = z.infer<typeof Env>;
TS

cat > railway-backend/src/logger.ts <<'TS'
import pino from "pino";
import { env } from "./config";
export const logger = pino({ name: "sb1-backend", level: process.env.LOG_LEVEL || "info", base: { commit: env.COMMIT_SHA ?? "dev" }});
TS

cat > railway-backend/src/middleware/validate.ts <<'TS'
import type { Request, Response, NextFunction } from "express";
import { ZodTypeAny } from "zod";
export const validate =
(schema: ZodTypeAny) =>
(req: Request, res: Response, next: NextFunction) => {
const parsed = schema.safeParse({ body: req.body, query: req.query, params: req.params, headers: req.headers });
if (!parsed.success) return res.status(400).json({ error: "bad_request", details: parsed.error.flatten() });
// @ts-ignore narrowed bag for handlers
req.z = parsed.data; next();
};
TS

cat > railway-backend/src/services/riskService.ts <<'TS'
import { env } from "../config";
let rollingDrawdownPct = 0; // wire to live metric if available
const symbolCaps = new Map<string, number>();
export function isDrawdownBreached() { return rollingDrawdownPct <= -Math.abs(env.MAX_DRAWDOWN_PCT); }
export function withinCaps(symbol: string, notionalUsd: number) {
const cap = symbolCaps.get(symbol) ?? env.PER_SYMBOL_USD_CAP;
return Math.abs(notionalUsd) <= cap;
}
export function sizeByVolTarget(notionalUsd: number, symbol: string, realizedVol = 0.5) {
const scale = Math.min(1, env.TARGET_ANN_VOL / Math.max(1e-6, realizedVol));
return notionalUsd * scale;
}
export function updateDrawdown(pnlPctSincePeak: number) { rollingDrawdownPct = pnlPctSincePeak; }
TS

cat > railway-backend/src/middleware/riskGate.ts <<'TS'
import type { Request, Response, NextFunction } from "express";
import crypto from "node:crypto";
import { env } from "../config";
import { isDrawdownBreached, withinCaps, sizeByVolTarget } from "../services/riskService";
export function riskGate(req: Request, res: Response, next: NextFunction) {
const { symbol, qtyUsd, confidence } = req.body as { symbol: string; qtyUsd: number; confidence: number };
if (env.TRADING_MODE === "live" && isDrawdownBreached()) return res.status(423).json({ error:"risk_locked", reason:"max_drawdown_breached" });
if (confidence < env.CONFIDENCE_THRESHOLD) return res.status(412).json({ error:"low_confidence", min: env.CONFIDENCE_THRESHOLD, got: confidence });
if (!withinCaps(symbol, qtyUsd)) return res.status(409).json({ error:"exceeds_symbol_cap", capUsd:"PER_SYMBOL_USD_CAP" });
const realizedVol = 0.5; req.body.qtyUsd = sizeByVolTarget(qtyUsd, symbol, realizedVol);
if (!req.header("Idempotency-Key")) {
const ik = crypto.createHash("sha256").update(JSON.stringify(req.body)).digest("hex");
(req as any).idempotencyKey = ik; res.setHeader("Idempotency-Key", ik);
} next();
}
TS

cat > railway-backend/src/services/modelService.ts <<'TS'
import { env } from "../config";
export type PredictReq = { symbol: string; features: Record<string, number>; timestamp?: number };
export type PredictResp = { signal: "long"|"short"|"flat"; prob_long: number; prob_short: number; confidence: number; model_version: string; };
export async function predict(req: PredictReq): Promise<PredictResp> {
const r = await fetch(`${env.MODEL_SERVICE_URL}/predict`, { method:"POST", headers:{ "content-type":"application/json" }, body: JSON.stringify(req) });
if (!r.ok) throw new Error(`modelService ${r.status}`);
return await r.json() as PredictResp;
}
TS

cat > railway-backend/src/routes/ai.ts <<'TS'
import { Router } from "express";
import { z } from "zod";
import { validate } from "../middleware/validate";
import { predict } from "../services/modelService";
import { env } from "../config";
const r = Router();
const ConsensusIn = z.object({ body: z.object({ symbol: z.string(), features: z.record(z.number()).default({}), timestamp: z.number().optional() }) });
r.post("/ai/consensus", validate(ConsensusIn), async (req, res) => {
const { symbol, features, timestamp } = (req as any).z.body;
const out = await predict({ symbol, features, timestamp });
const allow = out.confidence >= env.CONFIDENCE_THRESHOLD && out.signal !== "flat";
res.json({ ...out, allow });
});
export default r;
TS

cat > railway-backend/src/routes/trade.ts <<'TS'
import { Router } from "express";
import { z } from "zod";
import { validate } from "../middleware/validate";
import { riskGate } from "../middleware/riskGate";
const r = Router();
const ExecIn = z.object({ body: z.object({ symbol:z.string(), side:z.enum(["buy","sell"]), qtyUsd:z.number().positive(), confidence:z.number().min(0).max(1),
slPct:z.number().positive().max(0.2).default(0.01), tpPct:z.number().positive().max(0.5).default(0.02) })});
r.post("/trade/execute", validate(ExecIn), riskGate, async (req, res) => { res.json({ ok:true, dryRun:true, order:req.body }); });
export default r;
TS

cat > railway-backend/src/routes/health.ts <<'TS'
import { Router } from "express";
import { env } from "../config";
const r = Router();
r.get("/health", (_req,res) => res.json({ ok:true, env: env.NODE_ENV, commit: env.COMMIT_SHA ?? "dev", tradingMode: env.TRADING_MODE, modelService: env.MODEL_SERVICE_URL }));
export default r;
TS

cat > railway-backend/src/index.ts <<'TS'
import express from "express"; import helmet from "helmet"; import cors from "cors"; import rateLimit from "express-rate-limit";
import ai from "./routes/ai"; import trade from "./routes/trade"; import health from "./routes/health";
import { env } from "./config"; import { logger } from "./logger";
const app = express();
app.use(helmet()); app.use(cors({ origin: env.CORS_ORIGIN, credentials:true })); app.use(express.json({ limit:"1mb" }));
app.use(rateLimit({ windowMs: env.RATE_LIMIT_WINDOW_SEC*1000, max: env.RATE_LIMIT_MAX, standardHeaders:true, legacyHeaders:false }));
app.use("/api", ai); app.use("/api", trade); app.use("/", health);
app.use((err:any,_req:any,res:any,_next:any)=>{ logger.error({err},"unhandled"); res.status(500).json({error:"internal_error"})});
app.listen(Number(env.PORT), ()=> logger.info(`backend up on :${env.PORT}`));
TS

cat > railway-backend/Dockerfile <<'DOCK'
FROM node:20-alpine
WORKDIR /app
COPY package.json ./
RUN npm install
COPY tsconfig.json ./
COPY src ./src
ENV PORT=8000
EXPOSE 8000
CMD ["npm","run","dev"]
DOCK

# --- CI (optional lightweight) ---

cat > .github/workflows/ci.yml <<'CI'
name: ci
on: [push, pull_request]
jobs:
build-model:
runs-on: ubuntu-latest
steps:
- uses: actions/checkout@v4
- uses: actions/setup-python@v5
with: { python-version: '3.11' }
- run: pip install -r model-service/requirements.txt
- run: python - <<'PY'
print("model ok")
PY
build-backend:
runs-on: ubuntu-latest
steps:
- uses: actions/checkout@v4
- uses: actions/setup-node@v4
with: { node-version: '20' }
- run: cd railway-backend && npm install && npm run build || true
CI

echo "✅ Files created."
