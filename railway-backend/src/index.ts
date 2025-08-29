import express from "express"; import helmet from "helmet"; import cors from "cors"; import rateLimit from "express-rate-limit";
import ai from "./routes/ai"; import trade from "./routes/trade"; import health from "./routes/health";
import { env } from "./config"; import { logger } from "./logger";
const app = express();
app.use(helmet()); app.use(cors({ origin: env.CORS_ORIGIN, credentials:true })); app.use(express.json({ limit:"1mb" }));
app.use(rateLimit({ windowMs: env.RATE_LIMIT_WINDOW_SEC*1000, max: env.RATE_LIMIT_MAX, standardHeaders:true, legacyHeaders:false }));
app.use("/api", ai); app.use("/api", trade); app.use("/", health);
app.use((err:any,_req:any,res:any,_next:any)=>{ logger.error({err},"unhandled"); res.status(500).json({error:"internal_error"})});
app.listen(Number(env.PORT), ()=> logger.info(`backend up on :${env.PORT}`));
