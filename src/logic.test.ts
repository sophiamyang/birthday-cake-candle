import { describe, expect, it } from 'vitest';
import { layoutCandles } from './cake';
import { numParam } from './params';

describe('numParam', () => {
  const p = (q: string) => new URLSearchParams(q);
  it('reads numbers and falls back to the default', () => {
    expect(numParam(p('power=0.02'), 'power', 0.012)).toBe(0.02);
    expect(numParam(p(''), 'power', 0.012)).toBe(0.012);
  });
  it('treats empty and non-numeric values as missing', () => {
    expect(numParam(p('candles'), 'candles', 12)).toBe(12); // bare ?candles means cake mode, not 0
    expect(numParam(p('candles='), 'candles', 12)).toBe(12);
    expect(numParam(p('power=abc'), 'power', 0.012)).toBe(0.012);
    expect(numParam(p('power=Infinity'), 'power', 0.012)).toBe(0.012);
  });
  it('clamps to the given range', () => {
    expect(numParam(p('candles=500'), 'candles', 12, 1, 64)).toBe(64);
    expect(numParam(p('depthRows=0'), 'depthRows', 196, 14)).toBe(14);
  });
});

describe('layoutCandles', () => {
  it.each([1, 3, 8, 12, 24, 38, 39, 50, 64])('places exactly %i candles', (n) => {
    expect(layoutCandles(n)).toHaveLength(n);
  });

  it('spreads large counts over the rings instead of piling them in the centre', () => {
    const pts = layoutCandles(64);
    // the innermost ring sits within 0.1 of the top radius (290 px) of the centre (x = 400)
    const centre = pts.filter((p) => Math.abs(p.x - 400) <= 290 * 0.1 + 1e-6 && Math.abs(p.v) <= 0.1 + 1e-6);
    expect(centre.length).toBeLessThanOrEqual(6);
  });

  it('keeps every candle on the top of the cake', () => {
    for (const n of [12, 64]) {
      for (const p of layoutCandles(n)) {
        const ex = (p.x - 400) / 290, ey = (p.y - 250) / 66;
        expect(ex * ex + ey * ey).toBeLessThanOrEqual(1);
      }
    }
  });
});
