import { Router } from "express";
import { z } from "zod";
import { validate } from "../middleware/validate";
import { riskGate } from "../middleware/riskGate";
const r = Router();
const ExecIn = z.object({ body: z.object({ symbol:z.string(), side:z.enum(["buy","sell"]), qtyUsd:z.number().positive(), confidence:z.number().min(0).max(1),
slPct:z.number().positive().max(0.2).default(0.01), tpPct:z.number().positive().max(0.5).default(0.02) })});
r.post("/trade/execute", validate(ExecIn), riskGate, async (req, res) => { res.json({ ok:true, dryRun:true, order:req.body }); });
export default r;
