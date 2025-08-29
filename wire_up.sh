#!/usr/bin/env bash
set -euo pipefail

# --- model-service ---
mkdir -p model-service models
cat > model-service/requirements.txt <<'REQ'
fastapi==0.115.0
uvicorn==0.30.6
pydantic==2.8.2
torch==2.3.1
scikit-learn==1.5.1
joblib==1.4.2
numpy>=1.26
pandas>=2.2
REQ

cat > model-service/utils.py <<'PY'
import os, time
def model_version(): return os.getenv("MODEL_VERSION", time.strftime("%Y%m%d_%H%M%S"))
PY

cat > model-service/rl_agent.py <<'PY'
import torch, torch.nn as nn, numpy as np
from typing import Tuple
ACTIONS = ["long","short","flat"]
class DuelingLSTMDQN(nn.Module):
    def __init__(self, n_features:int, hidden:int=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden, num_layers=1, batch_first=True)
        self.adv  = nn.Sequential(nn.Linear(hidden,64), nn.ReLU(), nn.Linear(64,3))
        self.val  = nn.Sequential(nn.Linear(hidden,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self, x):
        h,_ = self.lstm(x); h = h[:,-1,:]
        adv = self.adv(h);  val = self.val(h)
        return val + adv - adv.mean(dim=1, keepdim=True)
class RLModel:
    def __init__(self, n_features:int, device:str="cpu"):
        self.device = torch.device(device)
        self.net = DuelingLSTMDQN(n_features).to(self.device)
        self.net.eval()
    def load_weights(self, path:str):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
    @torch.no_grad()
    def q_values(self, feat_seq:np.ndarray)->np.ndarray:
        import torch as t
        x = t.tensor(feat_seq, dtype=t.float32, device=self.device).unsqueeze(0)
        return self.net(x).squeeze(0).cpu().numpy()
PY

cat > model-service/features.py <<'PY'
import numpy as np, joblib
from typing import Dict
class FeaturePipe:
    def __init__(self, feature_order, scaler_path:str):
        self.feature_order = feature_order
        self.scaler = joblib.load(scaler_path)
    def make_window(self, features_live:Dict[str,float], window_store):
        row = np.array([float(features_live.get(k,0.0)) for k in self.feature_order], dtype=np.float32)
        window_store.append(row)
        X = np.stack(window_store, axis=0)
        return self.scaler.transform(X)
PY

cat > model-service/calibration.py <<'PY'
import joblib, numpy as np
from typing import Tuple
class BinaryCalibrator:
    def __init__(self, path_long:str, path_short:str):
        self.long_cal = joblib.load(path_long)
        self.short_cal= joblib.load(path_short)
    def calibrate(self, q:np.ndarray)->Tuple[float,float,float]:
        s_long = float(q[0] - max(q[1], q[2]))
        s_short= float(q[1] - max(q[0], q[2]))
        p_long = float(np.clip(self.long_cal.predict([s_long])[0], 0, 1))
        p_short= float(np.clip(self.short_cal.predict([s_short])[0],0, 1))
        p_flat = max(0.0, 1.0 - max(p_long, p_short))
        s = p_long+p_short+p_flat
        return p_long/s, p_short/s, p_flat/s
PY

cat > model-service/registry.py <<'PY'
import os
from collections import deque
def model_paths(base="models/current"):
    return (os.path.join(base,"weights.pth"),
            os.path.join(base,"scaler.pkl"),
            os.path.join(base,"cal_long.pkl"),
            os.path.join(base,"cal_short.pkl"),
            os.path.join(base,"feature_order.txt"))
def load_feature_order(path:str):
    with open(path) as f: return [ln.strip() for ln in f if ln.strip()]
def make_window_store(maxlen:int): return deque(maxlen=maxlen)
PY

cat > model-service/app.py <<'PY'
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional
import os, numpy as np
from utils import model_version
from rl_agent import RLModel, ACTIONS
from features import FeaturePipe
from calibration import BinaryCalibrator
from registry import model_paths, load_feature_order, make_window_store
app = FastAPI(title="RL Model Service", version="1.0.0")
WEIGHTS, SCALER, CAL_LONG, CAL_SHORT, FEAT_PATH = model_paths()
FEATURE_ORDER = load_feature_order(FEAT_PATH)
N_FEATURES = len(FEATURE_ORDER)
WINDOW = int(os.getenv("MODEL_WINDOW","32"))
MODEL = RLModel(n_features=N_FEATURES, device=os.getenv("MODEL_DEVICE","cpu"))
MODEL.load_weights(WEIGHTS)
PIPE = FeaturePipe(FEATURE_ORDER, SCALER)
CAL  = BinaryCalibrator(CAL_LONG, CAL_SHORT)
WINDOW_STORE = make_window_store(WINDOW)
class PredictIn(BaseModel):
    symbol: str
    features: Dict[str,float] = Field(default_factory=dict)
    timestamp: Optional[int] = None
class PredictOut(BaseModel):
    signal: str
    prob_long: float
    prob_short: float
    confidence: float
    model_version: str
@app.get("/health")
def health(): return {"ok": True, "model_version": model_version(), "window_len": len(WINDOW_STORE)}
@app.post("/predict", response_model=PredictOut)
def predict(p: PredictIn):
    try:
        Xs = PIPE.make_window(p.features, WINDOW_STORE)
        if Xs.shape[0] < 4:
            return PredictOut(signal="flat", prob_long=0.33, prob_short=0.33, confidence=0.34, model_version=model_version())
        q = MODEL.q_values(Xs)
        pl, ps, pf = CAL.calibrate(q)
        probs = np.array([pl, ps, pf], dtype=float)
        idx = int(probs.argmax())
        return PredictOut(signal=ACTIONS[idx], prob_long=float(pl), prob_short=float(ps), confidence=float(probs.max()), model_version=model_version())
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

cat > model-service/train.py <<'PY'
# skeleton exporter: builds scaler/calibrators + random weights and flips models/current
import os, time, numpy as np, joblib, torch
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from rl_agent import DuelingLSTMDQN
FEATURES=["mom_5","mom_20","rv_5","rv_20","rsi_14","atr_14","spread_bps","imbalance_1m","hour_sin","hour_cos"]
def main():
    N=len(FEATURES); rng=np.random.default_rng(17)
    X=rng.normal(size=(10000,N)).astype(np.float32)
    scaler=StandardScaler().fit(X)
    net=DuelingLSTMDQN(n_features=N)
    s_long=(X[:,0]-X[:,2]); s_short=(X[:,1]-X[:,2])
    y_long=(X[:,0]>0).astype(float); y_short=(X[:,1]>0).astype(float)
    cal_long=IsotonicRegression(out_of_bounds="clip").fit(s_long,y_long)
    cal_short=IsotonicRegression(out_of_bounds="clip").fit(s_short,y_short)
    ver=time.strftime("%Y%m%d_%H%M"); out=f"models/{ver}"; os.makedirs(out,exist_ok=True)
    torch.save(net.state_dict(), f"{out}/weights.pth")
    joblib.dump(scaler,   f"{out}/scaler.pkl")
    joblib.dump(cal_long, f"{out}/cal_long.pkl")
    joblib.dump(cal_short,f"{out}/cal_short.pkl")
    with open(f"{out}/feature_order.txt","w") as f: f.write("\n".join(FEATURES))
    cur="models/current"; 
    try: os.remove(cur)
    except: pass
    os.symlink(ver, cur)  # relative symlink
    print("Exported:", out, " -> models/current")
if __name__=="__main__": main()
PY

# --- backend with Bybit ---
mkdir -p railway-backend/src/{middleware,routes,services}
if [ ! -f railway-backend/package.json ]; then
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
    "bybit-api": "^3.8.6",
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
fi

cat > railway-backend/tsconfig.json <<'TS'
{ "compilerOptions": { "target":"ES2021","module":"CommonJS","moduleResolution":"Node","lib":["ES2021","DOM"],
  "esModuleInterop":true,"strict":true,"skipLibCheck":true,"outDir":"dist" }, "include":["src"] }
TS

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

cat > railway-backend/src/config.ts <<'TS'
import { z } from "zod";
const Env = z.object({
  NODE_ENV: z.enum(["development","test","production"]).default("development"),
  PORT: z.string().default("8000"),
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
TS

cat > railway-backend/src/logger.ts <<'TS'
import pino from "pino"; export const logger = pino({ name:"sb1-backend", level:process.env.LOG_LEVEL||"info" });
TS

cat > railway-backend/src/middleware/validate.ts <<'TS'
import type { Request, Response, NextFunction } from "express";
import { ZodTypeAny } from "zod";
export const validate=(schema:ZodTypeAny)=>(req:Request,res:Response,next:NextFunction)=>{
  const p=schema.safeParse({body:req.body,query:req.query,params:req.params,headers:req.headers});
  if(!p.success) return res.status(400).json({error:"bad_request",details:p.error.flatten()});
  // @ts-ignore
  req.z=p.data; next();
};
TS

cat > railway-backend/src/services/riskService.ts <<'TS'
import { env } from "../config";
let rollingDrawdownPct=0; const caps=new Map<string,number>();
export function isDrawdownBreached(){ return rollingDrawdownPct <= -Math.abs(env.MAX_DRAWDOWN_PCT); }
export function withinCaps(sym:string, usd:number){ const cap=caps.get(sym)??env.PER_SYMBOL_USD_CAP; return Math.abs(usd)<=cap; }
export function sizeByVolTarget(usd:number, _sym:string, rv=0.5){ const s=Math.min(1, env.TARGET_ANN_VOL/Math.max(1e-6,rv)); return usd*s; }
export function updateDrawdown(pct:number){ rollingDrawdownPct=pct; }
TS

cat > railway-backend/src/middleware/riskGate.ts <<'TS'
import type { Request, Response, NextFunction } from "express";
import crypto from "node:crypto";
import { env } from "../config";
import { isDrawdownBreached, withinCaps, sizeByVolTarget } from "../services/riskService";
export function riskGate(req:Request,res:Response,next:NextFunction){
  const { symbol, qtyUsd, confidence } = req.body as any;
  if(env.TRADING_MODE==="live" && isDrawdownBreached()) return res.status(423).json({error:"risk_locked"});
  if(confidence < env.CONFIDENCE_THRESHOLD) return res.status(412).json({error:"low_confidence",min:env.CONFIDENCE_THRESHOLD,got:confidence});
  if(!withinCaps(symbol, qtyUsd)) return res.status(409).json({error:"exceeds_symbol_cap"});
  req.body.qtyUsd = sizeByVolTarget(qtyUsd, symbol, 0.5);
  if(!req.header("Idempotency-Key")){
    const ik=crypto.createHash("sha256").update(JSON.stringify(req.body)).digest("hex");
    (req as any).idempotencyKey=ik; res.setHeader("Idempotency-Key", ik);
  }
  next();
}
TS

cat > railway-backend/src/services/modelService.ts <<'TS'
import { z } from "zod"; import { env } from "../config";
export const PredictReqSchema=z.object({symbol:z.string().min(1),features:z.record(z.number()).default({}),timestamp:z.number().int().optional()});
export const PredictRespSchema=z.object({signal:z.enum(["long","short","flat"]),prob_long:z.number().min(0).max(1),prob_short:z.number().min(0).max(1),confidence:z.number().min(0).max(1),model_version:z.string().min(1)});
export type PredictReq= z.infer<typeof PredictReqSchema>; export type PredictResp= z.infer<typeof PredictRespSchema>;
async function fetchWithTimeout(url:string,opts:RequestInit,ms=3000){ const c=new AbortController(); const id=setTimeout(()=>c.abort(),ms); try{ return await fetch(url,{...opts,signal:c.signal}); } finally{ clearTimeout(id);} }
export async function predict(req:PredictReq):Promise<PredictResp>{
  const p=PredictReqSchema.safeParse(req); if(!p.success) throw new Error("bad_request");
  const r=await fetchWithTimeout(`${env.MODEL_SERVICE_URL}/predict`,{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify(p.data)});
  if(!r.ok) throw new Error(`modelService ${r.status}`); const j=await r.json(); const v=PredictRespSchema.safeParse(j); if(!v.success) throw new Error("bad_response"); return v.data;
}
TS

cat > railway-backend/src/services/bybitClient.ts <<'TS'
import { RestClientV5 } from "bybit-api"; import { env } from "../config";
const client=new RestClientV5({ key:process.env.BYBIT_API_KEY||"", secret:process.env.BYBIT_API_SECRET||"", testnet: env.TRADING_MODE!=="live", recv_window:15000 });
export type PlaceOrderParams={symbol:string; side:"Buy"|"Sell"; qty:string; market:boolean; price?:string; idempotencyKey?:string; category?:"linear"|"inverse"|"option"|"spot";};
export async function placeOrder(p:PlaceOrderParams){
  const req:any={ category:p.category??"linear", symbol:p.symbol, side:p.side, orderType:p.market?"Market":"Limit", qty:p.qty, timeInForce:"GTC" };
  if(!p.market) req.price=p.price; if(p.idempotencyKey) req.orderLinkId=p.idempotencyKey.slice(0,36);
  return client.submitOrder(req);
}
TS

cat > railway-backend/src/services/quotes.ts <<'TS'
import { RestClientV5 } from "bybit-api"; import { env } from "../config";
const qc=new RestClientV5({ testnet: env.TRADING_MODE!=="live" });
export async function midPx(symbol:string, category:"linear"|"inverse"|"spot"="linear"):Promise<number>{
  const { result }=await qc.getTickers({ category, symbol }); const t:any=result.list?.[0]; const b=Number(t?.bid1Price||0), a=Number(t?.ask1Price||0);
  if(!b||!a) throw new Error("No quotes"); return (b+a)/2;
}
TS

cat > railway-backend/src/routes/ai.ts <<'TS'
import { Router } from "express"; import { z } from "zod"; import { validate } from "../middleware/validate";
import { predict, PredictReqSchema } from "../services/modelService"; import { env } from "../config";
const r=Router();
r.post("/ai/consensus", validate(z.object({ body: PredictReqSchema })), async (req,res)=>{
  const { symbol, features, timestamp } = (req as any).z.body;
  const out = await predict({ symbol, features, timestamp });
  const allow = out.confidence >= env.CONFIDENCE_THRESHOLD && out.signal!=="flat";
  res.json({ ...out, allow });
});
export default r;
TS

cat > railway-backend/src/routes/trade.ts <<'TS'
import { Router } from "express"; import { z } from "zod"; import { validate } from "../middleware/validate"; import { riskGate } from "../middleware/riskGate";
import { env } from "../config"; import { placeOrder } from "../services/bybitClient"; import { midPx } from "../services/quotes";
const r=Router();
const ExecIn=z.object({ body:z.object({ symbol:z.string(), side:z.enum(["buy","sell"]), qtyUsd:z.number().positive(), confidence:z.number().min(0).max(1), slPct:z.number().positive().max(0.2).default(0.01), tpPct:z.number().positive().max(0.5).default(0.02), category:z.enum(["linear","inverse","spot"]).default("linear") }) });
r.post("/trade/execute", validate(ExecIn), riskGate, async (req,res)=>{
  const { symbol, side, qtyUsd, slPct, tpPct, category } = (req as any).z.body; const idk=(req as any).idempotencyKey;
  if(env.TRADING_MODE!=="live") return res.json({ ok:true, mode:"paper", dryRun:true, order:req.body });
  try{
    const px=await midPx(symbol, category); const qty=(qtyUsd/px).toFixed(6);
    const out=await placeOrder({ symbol, side: side==="buy"?"Buy":"Sell", qty, market:true, idempotencyKey:idk, category });
    res.json({ ok:true, mode:"live", broker: out });
  }catch(e:any){ res.status(502).json({ ok:false, error:String(e?.message||e) }); }
});
export default r;
TS

cat > railway-backend/src/routes/health.ts <<'TS'
import { Router } from "express"; import { env } from "../config"; const r=Router();
r.get("/health", (_req,res)=>res.json({ ok:true, env:env.NODE_ENV, tradingMode:env.TRADING_MODE, modelService:env.MODEL_SERVICE_URL }));
export default r;
TS

cat > railway-backend/src/index.ts <<'TS'
import express from "express"; import helmet from "helmet"; import cors from "cors"; import rateLimit from "express-rate-limit";
import ai from "./routes/ai"; import trade from "./routes/trade"; import health from "./routes/health"; import { env } from "./config";
const app=express(); app.use(helmet()); app.use(cors({ origin:env.CORS_ORIGIN, credentials:true })); app.use(express.json({ limit:"1mb" }));
app.use(rateLimit({ windowMs: env.RATE_LIMIT_WINDOW_SEC*1000, max: env.RATE_LIMIT_MAX, standardHeaders:true, legacyHeaders:false }));
app.use("/api", ai); app.use("/api", trade); app.use("/", health);
app.use((err:any,_req:any,res:any,_next:any)=>{ res.status(500).json({ error:"internal_error", detail:String(err?.message||err) })});
app.listen(Number(env.PORT), ()=>console.log(`backend up :${env.PORT}`));
TS

cat > docker-compose.yml <<'YML'
version: "3.9"
services:
  model-service:
    build: ./model-service
    environment: [ "MODEL_VERSION=${MODEL_VERSION:-dev}" ]
    ports: [ "9000:9000" ]
    healthcheck: { test: ["CMD","wget","-qO-","http://127.0.0.1:9000/health"], interval: 30s, timeout: 3s, retries: 5 }
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
      - BYBIT_API_KEY
      - BYBIT_API_SECRET
    ports: [ "8000:8000" ]
    depends_on: { model-service: { condition: service_healthy } }
YML

echo "✅ Files written. Next: build model artifacts, then docker compose up."
