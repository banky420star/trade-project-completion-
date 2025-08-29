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
