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
