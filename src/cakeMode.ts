import { buildCake, makeCandle, SPRITE_H, SPRITE_W, type CandleState, type Wick } from './cake';
import { MAX_GPU_CANDLES, type CandleGPU, Renderer } from './renderer';

export interface CakeInput {
  dt: number;
  t: number;
  tip: [number, number] | null;   // index fingertip, source uv (null = no hand)
  tipScreen: [number, number];
  fingerFlame: number;             // height of the drawn finger flame (screen-height units)
  fingerLit: boolean;              // only a burning fingertip can light a wick
  wind: number;                   // 0..1 blow strength
  mouthScreenX: number;           // where the blow comes from (screen uv x)
}

export interface CakeOutput {
  rect: [number, number, number, number];
  candles: CandleGPU[];
  shadowLight: [number, number, number, number];
  lit: number;
}

// How long the finger flame must stay on a wick to light it; meanwhile a small flame grows on it.
const CATCH_TIME = 0.15;
// Wind is aimed: candles in front of the mouth get about 1.5x, the cake's far ends about half.
// It acts above a small floor.
const WIND_FLOOR = 0.1;
const WIND_RATE = 5;
export const windReach = (mx: number) => 0.55 + Math.exp(-(mx * mx) / 0.03);

const flickerNoise = (t: number) => {
  const i = Math.floor(t), f = t - i, s = f * f * (3 - 2 * f);
  const h = (x: number) => { const v = Math.sin(x * 127.1 + 311.7) * 43758.5453; return v - Math.floor(v); };
  return h(i) * (1 - s) + h(i + 1) * s;
};

export class CakeMode {
  count = 0;
  candles: CandleState[] = [];
  wicks: Wick[] = [];
  litEver = false;
  onAllOut: (() => void) | null = null;
  onLit: ((i: number) => void) | null = null;
  power: number;
  freezeDemo = false; // screenshot helper: hold the blown state still

  constructor(private renderer: Renderer, n: number, private cakeD: number, power: number, litAll: boolean) {
    this.power = power;
    this.setCount(n, litAll);
  }

  setCount(n: number, litAll = false) {
    n = Math.max(1, Math.min(MAX_GPU_CANDLES, Math.round(n)));
    const sprite = buildCake(n, this.cakeD);
    this.renderer.uploadCake(sprite.color, sprite.geo);
    this.wicks = sprite.wicks;
    const prev = this.candles;
    this.candles = sprite.wicks.map((_, i) => {
      const c = makeCandle(i, litAll || (prev[i]?.lit ?? false));
      return c;
    });
    if (litAll) this.litEver = true;
    this.count = n;
  }

  lightAll() { this.candles.forEach((c) => { c.lit = true; c.exposure = 0; }); this.litEver = true; }

  get litCount() { return this.candles.filter((c) => c.lit).length; }

  /** Where the sprite sits in source uv: bottom centre of the visible frame. */
  rect(): [number, number, number, number] {
    const r = this.renderer;
    const va = r.videoW / r.videoH || 16 / 9;
    let w = 0.5 * r.uvScale[0];
    let h = w * (SPRITE_H / SPRITE_W) * va;
    const maxH = 0.52 * r.uvScale[1];
    if (h > maxH) { w *= maxH / h; h = maxH; }
    const bottom = 0.5 + 0.5 * r.uvScale[1] + h * 0.05;
    return [0.5 - w / 2, bottom - h, w, h];
  }

  update(inp: CakeInput): CakeOutput {
    const r = this.renderer;
    const rect = this.rect();
    const aspect = window.innerWidth / Math.max(1, window.innerHeight);
    const out: CandleGPU[] = [];
    let sx = 0, sy = 0, sz = 0, sw = 0, lit = 0;
    const wasAnyLit = this.candles.some((c) => c.lit);
    const placed = this.wicks.map((wk) => {
      const src: [number, number] = [rect[0] + (wk.x / SPRITE_W) * rect[2], rect[1] + (wk.y / SPRITE_H) * rect[3]];
      return { src, scr: r.srcToScreen(src[0], src[1]), flameSize: (50 / SPRITE_H) * (rect[3] / r.uvScale[1]) * wk.scale };
    });

    // ---- lighting: hold the finger flame to a wick ----
    // People hold the visible flame to the wick, and the flame is drawn above the fingertip, so
    // any point from the fingertip up to the flame's tip counts. Only the nearest unlit wick in
    // reach catches, so overlapping candles light one at a time. No relighting while someone is
    // blowing, nor for ~1 s after a wick went out.
    let catchIdx = -1, catchDist = Infinity;
    if (inp.tip && inp.fingerLit && inp.wind < 0.15) {
      placed.forEach(({ scr, flameSize }, i) => {
        const c = this.candles[i];
        if (c.lit || c.smoke >= 0.64) return;
        const wickY = scr[1] - flameSize * 0.2;
        const alongY = Math.min(inp.tipScreen[1], Math.max(inp.tipScreen[1] - inp.fingerFlame, wickY));
        const dx = (inp.tipScreen[0] - scr[0]) * aspect;
        if (Math.hypot(dx, alongY - wickY) >= Math.max(0.05, flameSize * 1.3)) return;
        // among wicks in reach, the one nearest the middle of the flame, so aiming higher picks
        // a candle further back when two overlap
        const rank = Math.hypot(dx, inp.tipScreen[1] - inp.fingerFlame * 0.4 - wickY);
        if (rank < catchDist) { catchIdx = i; catchDist = rank; }
      });
    }

    this.wicks.forEach((wk, i) => {
      const c = this.candles[i];
      const { src, scr, flameSize } = placed[i];
      if (i === catchIdx) {
        c.dwell += inp.dt;
        if (c.dwell >= CATCH_TIME) {
          c.lit = true; c.amt = Math.max(c.amt, 0.6); c.exposure = 0; c.dwell = 0; c.smoke = 0; this.litEver = true;
          this.onLit?.(i);
        }
      } else c.dwell = Math.max(0, c.dwell - inp.dt * 0.5);

      // ---- wind ----
      const mx = (scr[0] - inp.mouthScreenX) * aspect;
      const wind = inp.wind * windReach(mx);
      const dir = Math.abs(mx) > 0.01 ? Math.sign(mx) : (c.seed % 2 > 1 ? 1 : -1);
      const bendT = c.lit ? dir * wind * 0.9 : dir * wind * 0.4;
      c.bend += (bendT + Math.sin(inp.t * 9 + c.seed) * wind * 0.12 - c.bend) * Math.min(1, inp.dt * 12);
      c.stretch += (wind - c.stretch) * Math.min(1, inp.dt * 10);
      if (!this.freezeDemo) {
        if (c.lit && wind > WIND_FLOOR) c.exposure += (wind - WIND_FLOOR) * inp.dt * WIND_RATE;
        else c.exposure = Math.max(0, c.exposure - inp.dt * 0.6);
        if (c.lit && c.exposure > c.threshold) { c.lit = false; c.smoke = 1; c.exposure = 0; }
        c.smoke = Math.max(0, c.smoke - inp.dt / 2.8);
      }
      const target = c.lit ? 1 - 0.35 * Math.min(1, wind) : 0;
      c.amt += (target - c.amt) * Math.min(1, inp.dt * (c.lit ? 3 : 14));

      const flick = 0.88 + 0.08 * (flickerNoise(inp.t * 5 + c.seed) * 2 - 1) + 0.04 * (flickerNoise(inp.t * 13 + c.seed * 3) * 2 - 1);
      // light at the flame core: source uv plus the wick's normalised depth (the shader places it)
      const lp = [src[0] + c.bend * flameSize * 0.3 * r.uvScale[0] * -1, src[1] - flameSize * 0.32 * r.uvScale[1], wk.d + 0.004];
      const power = this.power * c.amt * flick;
      if (c.amt > 0.01) { sx += lp[0] * power; sy += lp[1] * power; sz += lp[2] * power; sw += power; }
      if (c.lit) lit++;
      // a wick that is catching shows a small flame growing toward full size
      const catching = c.lit ? 0 : 0.7 * Math.min(1, c.dwell / CATCH_TIME);
      out.push({
        screen: [scr[0], scr[1], flameSize * (1 + 0.25 * (flick - 0.9)) * (c.lit ? 1 : 0.5 + 0.5 * catching), Math.max(c.amt, catching) * (0.92 + 0.12 * (flick - 0.88) / 0.12)],
        light: [lp[0], lp[1], lp[2], power],
        anim: [c.bend, c.stretch, c.smoke, c.seed],
      });
    });

    if (wasAnyLit && lit === 0 && this.litEver) this.onAllOut?.();
    const shadowLight: [number, number, number, number] = sw > 0 ? [sx / sw, sy / sw, sz / sw, sw] : [0, 0, 1, 0];
    return { rect, candles: out, shadowLight, lit };
  }
}
