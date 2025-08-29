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
