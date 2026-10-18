// URL tuning knobs. A missing, empty or non-numeric value falls back to the default, so a typo
// like ?power=abc can't push NaN into the shader, and a bare ?candles still means "cake mode".
export function numParam(params: URLSearchParams, key: string, def: number, min = -Infinity, max = Infinity): number {
  const raw = params.get(key);
  const v = raw === null || raw.trim() === '' ? NaN : Number(raw);
  return Number.isFinite(v) ? Math.min(max, Math.max(min, v)) : def;
}
