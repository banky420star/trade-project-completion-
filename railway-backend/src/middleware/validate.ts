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
