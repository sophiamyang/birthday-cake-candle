// Procedural birthday cake: a colour sprite (RGBA) plus a geometry sprite
// (R = normalised inverse depth, G/B = view-space normal x/y), both in sprite pixels.
// The shader composites them into the scene so the cake is lit, shadowed and depth-ordered.

export const SPRITE_W = 800;
export const SPRITE_H = 600;
export const MAX_CANDLES = 64;

export interface Wick {
  x: number; y: number;   // sprite px, top of wick
  d: number;              // normalised inverse depth at the wick
  scale: number;          // perspective scale (back candles slightly smaller)
}

export interface CakeSprite {
  color: ImageData;
  geo: ImageData;
  wicks: Wick[];
}

const PASTELS = ['#f7a8c4', '#9fd6f5', '#fbe38e', '#b8e6b0', '#d4b6f2', '#ffc49b'];

// cake geometry in sprite px
const CX = 400;
const RX = 290;
const TOP_Y = 250, TOP_RY = 66;
const BOT_Y = 470, BOT_RY = 66;
const PLATE_Y = 500, PLATE_RX = 370, PLATE_RY = 82;

function rng(seed: number) {
  let s = seed >>> 0;
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
}

/** Candle base positions on the top ellipse, arranged in concentric rings. */
export function layoutCandles(n: number): { x: number; y: number; v: number }[] {
  const out: { x: number; y: number; v: number }[] = [];
  if (n <= 0) return out;
  if (n === 1) return [{ x: CX, y: TOP_Y, v: 0 }];
  // ring radii (fraction of the top radius) and capacity ~ circumference; past the base capacity
  // (38) every ring grows in proportion, instead of piling the extra candles onto the centre ring
  const radii = [0.78, 0.55, 0.32, 0.1];
  const grow = Math.max(1, n / 38);
  const cap = radii.map((r) => Math.max(1, Math.round(r * 22 * grow)));
  let left = n;
  const counts: number[] = [];
  for (let i = 0; i < radii.length && left > 0; i++) {
    const total = cap.slice(i).reduce((a, b) => a + b, 0);
    const c = i === radii.length - 1 ? left : Math.min(cap[i], Math.ceil((left * cap[i]) / total));
    counts.push(c);
    left -= c;
  }
  if (n <= 8) { counts.length = 0; counts.push(n); }
  counts.forEach((c, ring) => {
    const r = n <= 8 ? (n <= 3 ? 0.4 : 0.6) : radii[ring];
    for (let k = 0; k < c; k++) {
      const a = (k / c) * Math.PI * 2 + ring * 0.6 + Math.PI / 2;
      out.push({ x: CX + Math.cos(a) * RX * r, y: TOP_Y + Math.sin(a) * TOP_RY * r, v: Math.sin(a) * r });
    }
  });
  return out;
}

export function buildCake(n: number, cakeD: number): CakeSprite {
  const cc = document.createElement('canvas'); cc.width = SPRITE_W; cc.height = SPRITE_H;
  const gc = document.createElement('canvas'); gc.width = SPRITE_W; gc.height = SPRITE_H;
  const c = cc.getContext('2d', { willReadFrequently: true })!;
  const g = gc.getContext('2d', { willReadFrequently: true })!;
  const dBack = cakeD - 0.035;
  const geo = (d: number, nx: number, ny: number) =>
    `rgb(${Math.round(Math.min(1, Math.max(0, d)) * 255)},${Math.round((nx * 0.5 + 0.5) * 255)},${Math.round((ny * 0.5 + 0.5) * 255)})`;
  // depth of a point on a horizontal surface: farther at the back (smaller y)
  const dAtY = (y: number) => dBack + (cakeD - dBack) * Math.min(1, Math.max(0, (y - (TOP_Y - TOP_RY)) / (PLATE_Y + PLATE_RY - (TOP_Y - TOP_RY))));
  const sideGrad = (ctx: CanvasRenderingContext2D, x0: number, x1: number, dMid: number, geoMode: boolean, colors?: [string, string]) => {
    const gr = ctx.createLinearGradient(x0, 0, x1, 0);
    for (let i = 0; i <= 10; i++) {
      const t = i / 10, nx = t * 2 - 1;
      if (geoMode) gr.addColorStop(t, geo(dMid - (1 - Math.sqrt(Math.max(0, 1 - nx * nx))) * 0.03, nx * 0.9, 0.3));
      else if (colors) gr.addColorStop(t, i % 10 === 0 ? colors[1] : colors[0]);
    }
    return gr;
  };
  const R = rng(1234 + n);
  const TILT = 0.95; // top surfaces face up and a little toward the camera

  // ---------- plate ----------
  {
    const pg = c.createRadialGradient(CX, PLATE_Y - 20, 40, CX, PLATE_Y, PLATE_RX);
    pg.addColorStop(0, '#d8d3cb'); pg.addColorStop(0.85, '#c9c3ba'); pg.addColorStop(1, '#a29b91');
    c.fillStyle = pg;
    c.beginPath(); c.ellipse(CX, PLATE_Y, PLATE_RX, PLATE_RY, 0, 0, Math.PI * 2); c.fill();
    c.fillStyle = '#d2cdc5';
    c.beginPath(); c.ellipse(CX, PLATE_Y + 10, PLATE_RX * 0.98, PLATE_RY * 0.9, 0, 0, Math.PI); c.fill();
    g.fillStyle = geo(dAtY(PLATE_Y), 0, TILT);
    g.beginPath(); g.ellipse(CX, PLATE_Y, PLATE_RX, PLATE_RY, 0, 0, Math.PI * 2); g.fill();
    g.beginPath(); g.ellipse(CX, PLATE_Y + 10, PLATE_RX * 0.98, PLATE_RY * 0.9, 0, 0, Math.PI); g.fill();
  }

  // ---------- cake side (cylinder) ----------
  const sidePath = (ctx: CanvasRenderingContext2D) => {
    ctx.beginPath();
    ctx.moveTo(CX - RX, TOP_Y);
    ctx.lineTo(CX - RX, BOT_Y);
    ctx.ellipse(CX, BOT_Y, RX, BOT_RY, 0, Math.PI, 0, true);
    ctx.lineTo(CX + RX, TOP_Y);
    ctx.ellipse(CX, TOP_Y, RX, TOP_RY, 0, 0, Math.PI, false);
    ctx.closePath();
  };
  {
    const sg = c.createLinearGradient(0, TOP_Y, 0, BOT_Y + BOT_RY);
    sg.addColorStop(0, '#f6cdd8'); sg.addColorStop(0.5, '#f2bccb'); sg.addColorStop(1, '#e9a9bb');
    c.fillStyle = sg; sidePath(c); c.fill();
    // a band of piped cream near the bottom
    c.fillStyle = '#fff4ef';
    for (let i = 0; i <= 40; i++) {
      const a = Math.PI * (i / 40);
      const x = CX + Math.cos(a) * RX * 0.995, y = BOT_Y + Math.sin(a) * BOT_RY;
      c.beginPath(); c.arc(x, y - 4, 13, 0, Math.PI * 2); c.fill();
    }
    g.fillStyle = sideGrad(g, CX - RX, CX + RX, cakeD - 0.006, true); sidePath(g); g.fill();
    for (let i = 0; i <= 40; i++) {
      const a = Math.PI * (i / 40);
      const x = CX + Math.cos(a) * RX * 0.995, y = BOT_Y + Math.sin(a) * BOT_RY;
      const nx = Math.cos(a);
      g.fillStyle = geo(cakeD - 0.004 - (1 - Math.sin(a)) * 0.03, nx * 0.9, 0.25);
      g.beginPath(); g.arc(x, y - 4, 13, 0, Math.PI * 2); g.fill();
    }
  }

  // ---------- white drip icing over the top edge ----------
  {
    // same random sequence on both canvases so colour and geometry line up
    const drawDrips = (ctx: CanvasRenderingContext2D, rnd: () => number) => {
      ctx.beginPath();
      ctx.moveTo(CX - RX, TOP_Y);
      const steps = 34;
      for (let i = 0; i <= steps; i++) {
        const a = Math.PI * (1 - i / steps);
        const x = CX + Math.cos(a) * RX;
        const yEdge = TOP_Y + Math.sin(a) * TOP_RY;
        const len = 10 + (i % 3 === 0 ? 34 + rnd() * 34 : 6 + rnd() * 12) * Math.sin(a);
        ctx.quadraticCurveTo(x - 4, yEdge + len, x, yEdge + len * 0.92);
      }
      ctx.lineTo(CX + RX, TOP_Y);
      ctx.ellipse(CX, TOP_Y, RX, TOP_RY, 0, 0, Math.PI, true);
      ctx.closePath();
    };
    const seed = Math.floor(R() * 1e9);
    c.fillStyle = '#fffaf3'; drawDrips(c, rng(seed)); c.fill();
    g.fillStyle = sideGrad(g, CX - RX, CX + RX, cakeD - 0.004, true); drawDrips(g, rng(seed)); g.fill();
  }

  // ---------- top ----------
  {
    const tg = c.createRadialGradient(CX - 60, TOP_Y - 20, 20, CX, TOP_Y, RX);
    tg.addColorStop(0, '#fffdf8'); tg.addColorStop(1, '#f6ece2');
    c.fillStyle = tg;
    c.beginPath(); c.ellipse(CX, TOP_Y, RX, TOP_RY, 0, 0, Math.PI * 2); c.fill();
    const tgg = g.createLinearGradient(0, TOP_Y - TOP_RY, 0, TOP_Y + TOP_RY);
    tgg.addColorStop(0, geo(dAtY(TOP_Y - TOP_RY), 0, TILT));
    tgg.addColorStop(1, geo(dAtY(TOP_Y + TOP_RY), 0, TILT));
    g.fillStyle = tgg;
    g.beginPath(); g.ellipse(CX, TOP_Y, RX, TOP_RY, 0, 0, Math.PI * 2); g.fill();
    // sprinkles
    const sp = ['#ff6f91', '#ffd166', '#4ecdc4', '#a78bfa', '#ffffff', '#7bd389'];
    for (let i = 0; i < 160; i++) {
      const a = R() * Math.PI * 2, r = Math.sqrt(R()) * 0.86;
      const x = CX + Math.cos(a) * RX * r, y = TOP_Y + Math.sin(a) * TOP_RY * r;
      c.save(); c.translate(x, y); c.rotate(R() * Math.PI);
      c.fillStyle = sp[i % sp.length]; c.fillRect(-4, -1.2, 8, 2.4); c.restore();
    }
  }

  // ---------- rosettes, candles ----------
  const rosette = (a: number) => {
    const x = CX + Math.cos(a) * RX * 0.9, y = TOP_Y + Math.sin(a) * TOP_RY * 0.9;
    const rg = c.createRadialGradient(x - 4, y - 6, 2, x, y, 17);
    rg.addColorStop(0, '#ffffff'); rg.addColorStop(1, '#f0d9df');
    c.fillStyle = rg; c.beginPath(); c.arc(x, y - 6, 16, 0, Math.PI * 2); c.fill();
    c.fillStyle = '#e04a6b'; c.beginPath(); c.arc(x, y - 14, 5.5, 0, Math.PI * 2); c.fill(); // berry
    g.fillStyle = geo(dAtY(y) + 0.006, Math.cos(a) * 0.4, 0.7);
    g.beginPath(); g.arc(x, y - 6, 16, 0, Math.PI * 2); g.fill();
    g.beginPath(); g.arc(x, y - 14, 5.5, 0, Math.PI * 2); g.fill();
  };
  const ROS = 22;
  const backRos: number[] = [], frontRos: number[] = [];
  for (let i = 0; i < ROS; i++) { const a = (i / ROS) * Math.PI * 2; (Math.sin(a) < 0 ? backRos : frontRos).push(a); }
  backRos.forEach(rosette);

  const bases = layoutCandles(Math.min(n, MAX_CANDLES)).sort((a, b) => a.y - b.y);
  const wicks: Wick[] = [];
  bases.forEach((b, i) => {
    const scale = 0.9 + 0.1 * ((b.v + 1) / 2);
    const h = (n > 20 ? 84 : 100) * scale, w = (n > 20 ? 15 : 18) * scale;
    const x0 = b.x - w / 2, yTop = b.y - h;
    const col = PASTELS[i % PASTELS.length];
    // body + gentle cylinder shading baked lightly (lighting adds the rest)
    c.fillStyle = col; c.fillRect(x0, yTop, w, h);
    c.save(); c.beginPath(); c.rect(x0, yTop, w, h); c.clip();
    c.strokeStyle = 'rgba(255,255,255,0.85)'; c.lineWidth = w * 0.32;
    for (let yy = yTop - w; yy < b.y + w; yy += w * 1.1) { c.beginPath(); c.moveTo(x0, yy + w); c.lineTo(x0 + w, yy); c.stroke(); }
    c.restore();
    c.fillStyle = 'rgba(255,255,255,0.35)'; c.beginPath(); c.ellipse(b.x, yTop, w / 2, w * 0.22, 0, 0, Math.PI * 2); c.fill();
    // wick
    c.strokeStyle = '#2b2118'; c.lineWidth = 2.2 * scale; c.lineCap = 'round';
    c.beginPath(); c.moveTo(b.x, yTop); c.quadraticCurveTo(b.x + 1.5, yTop - 6 * scale, b.x + 0.5, yTop - 10 * scale); c.stroke();
    const d = dAtY(b.y) + 0.01;
    g.fillStyle = sideGrad(g, x0, x0 + w, d, true); g.fillRect(x0, yTop - 10 * scale, w, h + 10 * scale);
    wicks.push({ x: b.x + 0.5, y: yTop - 10 * scale, d, scale });
  });
  frontRos.forEach(rosette);

  return {
    color: c.getImageData(0, 0, SPRITE_W, SPRITE_H),
    geo: g.getImageData(0, 0, SPRITE_W, SPRITE_H),
    wicks,
  };
}

// ---------------- candle state ----------------
export interface CandleState {
  lit: boolean;
  amt: number;        // 0..1 visible flame
  exposure: number;   // accumulated wind
  threshold: number;  // how much wind it can take (wind-seconds above the floor)
  bend: number;       // screen-space lateral bend (signed)
  stretch: number;
  smoke: number;      // 1 right after going out, decays
  dwell: number;      // seconds the fingertip has hovered over the wick
  seed: number;
}

export function makeCandle(i: number, lit: boolean): CandleState {
  const r = rng(i * 7919 + 17);
  r();
  return { lit, amt: lit ? 1 : 0, exposure: 0, threshold: 0.25 + r() * 0.6, bend: 0, stretch: 0, smoke: 0, dwell: 0, seed: r() * 100 };
}
