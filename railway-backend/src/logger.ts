import pino from "pino";
import { env } from "./config";
export const logger = pino({ name: "sb1-backend", level: process.env.LOG_LEVEL || "info", base: { commit: env.COMMIT_SHA ?? "dev" }});
