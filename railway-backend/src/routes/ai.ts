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
