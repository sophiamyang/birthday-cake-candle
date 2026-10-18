// A procedural "webcam" used when no camera is available (headless tests, denied permission).
// It draws a simple head-and-shoulders scene plus a matching analytic depth map,
// and moves a synthetic fingertip around the face.
export class SyntheticScene {
  readonly canvas = document.createElement('canvas');
  private ctx: CanvasRenderingContext2D;
  private depthCanvas = document.createElement('canvas');
  private dctx: CanvasRenderingContext2D;
  tip: [number, number] = [0.5, 0.5];

  constructor(readonly w = 960, readonly h = 540) {
    this.canvas.width = w;
    this.canvas.height = h;
    this.ctx = this.canvas.getContext('2d')!;
    this.dctx = this.depthCanvas.getContext('2d', { willReadFrequently: true })!;
  }

  /** t in seconds; fixedTip overrides the animated fingertip path. */
  draw(t: number, fixedTip?: [number, number]) {
    const { ctx, w, h } = this;
    const cx = w * 0.5, cy = h * 0.47, rx = h * 0.17, ry = h * 0.23;
    // wall
    const g = ctx.createLinearGradient(0, 0, w, h);
    g.addColorStop(0, '#8d8679'); g.addColorStop(1, '#6f695f');
    ctx.fillStyle = g; ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = '#5b5348'; ctx.fillRect(w * 0.08, h * 0.12, w * 0.16, h * 0.5); // door/frame
    ctx.fillStyle = '#9a8f7d'; ctx.fillRect(w * 0.1, h * 0.15, w * 0.12, h * 0.44);
    ctx.fillStyle = '#4e4a44'; ctx.fillRect(w * 0.72, h * 0.18, w * 0.18, h * 0.22); // picture
    // shoulders
    ctx.fillStyle = '#2d3a4e';
    ctx.beginPath(); ctx.ellipse(cx, h * 1.02, h * 0.48, h * 0.3, 0, 0, Math.PI * 2); ctx.fill();
    // neck
    ctx.fillStyle = '#b98a6e'; ctx.fillRect(cx - rx * 0.45, cy + ry * 0.6, rx * 0.9, h * 0.2);
    // hair + face
    ctx.fillStyle = '#2a1d16';
    ctx.beginPath(); ctx.ellipse(cx, cy - ry * 0.12, rx * 1.12, ry * 1.08, 0, 0, Math.PI * 2); ctx.fill();
    const fg = ctx.createRadialGradient(cx - rx * 0.2, cy - ry * 0.2, rx * 0.1, cx, cy, rx * 1.1);
    fg.addColorStop(0, '#e2b69a'); fg.addColorStop(1, '#b4836a');
    ctx.fillStyle = fg;
    ctx.beginPath(); ctx.ellipse(cx, cy + ry * 0.05, rx, ry * 0.95, 0, 0, Math.PI * 2); ctx.fill();
    // eyes, brows, nose, mouth
    ctx.fillStyle = '#3a2a22';
    for (const s of [-1, 1]) {
      ctx.beginPath(); ctx.ellipse(cx + s * rx * 0.38, cy - ry * 0.12, rx * 0.12, ry * 0.045, 0, 0, Math.PI * 2); ctx.fill();
      ctx.fillRect(cx + s * rx * 0.38 - rx * 0.16, cy - ry * 0.26, rx * 0.32, ry * 0.035);
    }
    ctx.fillStyle = '#a8765e';
    ctx.beginPath(); ctx.moveTo(cx, cy - ry * 0.08); ctx.lineTo(cx - rx * 0.1, cy + ry * 0.2); ctx.lineTo(cx + rx * 0.1, cy + ry * 0.2); ctx.fill();
    ctx.fillStyle = '#8c4f45';
    ctx.beginPath(); ctx.ellipse(cx, cy + ry * 0.42, rx * 0.28, ry * 0.05, 0, 0, Math.PI * 2); ctx.fill();

    // fingertip path: loops past the nose, under the chin, past the eyes
    const tip: [number, number] = fixedTip ?? [
      0.5 + 0.13 * Math.sin(t * 0.7),
      0.55 + 0.12 * Math.sin(t * 1.1 + 1.0),
    ];
    this.tip = tip; // source uv (the canvas is un-mirrored, like a raw webcam frame)
    const tx = tip[0] * w, ty = tip[1] * h;
    ctx.strokeStyle = '#c8957a'; ctx.lineCap = 'round'; ctx.lineWidth = h * 0.035;
    ctx.beginPath(); ctx.moveTo(tx + h * 0.02, ty + h * 0.3); ctx.lineTo(tx, ty); ctx.stroke();
    ctx.fillStyle = '#c8957a';
    ctx.beginPath(); ctx.ellipse(tx + h * 0.04, ty + h * 0.36, h * 0.07, h * 0.08, 0, 0, Math.PI * 2); ctx.fill();
  }

  /** Silhouette of the synthetic finger + hand, for the cake occlusion mask. */
  drawMask(ctx: CanvasRenderingContext2D) {
    const W = ctx.canvas.width, H = ctx.canvas.height, k = H / this.h;
    const tx = this.tip[0] * W, ty = this.tip[1] * H, h = this.h * k;
    ctx.save(); ctx.filter = 'blur(1px)';
    ctx.strokeStyle = '#fff'; ctx.fillStyle = '#fff'; ctx.lineCap = 'round'; ctx.lineWidth = h * 0.045;
    ctx.beginPath(); ctx.moveTo(tx + h * 0.02, ty + h * 0.3); ctx.lineTo(tx, ty); ctx.stroke();
    ctx.beginPath(); ctx.ellipse(tx + h * 0.04, ty + h * 0.36, h * 0.08, h * 0.09, 0, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
  }

  depth(dw: number, dh: number): Float32Array {
    const { dctx } = this;
    if (this.depthCanvas.width !== dw || this.depthCanvas.height !== dh) { this.depthCanvas.width = dw; this.depthCanvas.height = dh; }
    const sx = dw / this.w, sy = dh / this.h;
    const W = this.w, H = this.h;
    const cx = W * 0.5, cy = H * 0.47, rx = H * 0.17, ry = H * 0.23;
    dctx.setTransform(sx, 0, 0, sy, 0, 0);
    const v = (d: number) => `rgb(${Math.round(d * 255)},0,0)`;
    dctx.fillStyle = v(0.08); dctx.fillRect(0, 0, W, H);
    dctx.fillStyle = v(0.45);
    dctx.beginPath(); dctx.ellipse(cx, H * 1.02, H * 0.48, H * 0.3, 0, 0, Math.PI * 2); dctx.fill();
    dctx.fillStyle = v(0.5); dctx.fillRect(cx - rx * 0.45, cy + ry * 0.6, rx * 0.9, H * 0.2);
    const fg = dctx.createRadialGradient(cx, cy, 0, cx, cy, rx * 1.15);
    fg.addColorStop(0, v(0.72)); fg.addColorStop(0.7, v(0.62)); fg.addColorStop(1, v(0.5));
    dctx.fillStyle = fg;
    dctx.beginPath(); dctx.ellipse(cx, cy, rx * 1.1, ry * 1.05, 0, 0, Math.PI * 2); dctx.fill();
    const ng = dctx.createRadialGradient(cx, cy + ry * 0.1, 0, cx, cy + ry * 0.1, rx * 0.25);
    ng.addColorStop(0, v(0.8)); ng.addColorStop(1, 'rgba(0,0,0,0)');
    dctx.fillStyle = ng; dctx.beginPath(); dctx.arc(cx, cy + ry * 0.1, rx * 0.25, 0, Math.PI * 2); dctx.fill();
    const tx = this.tip[0] * W, ty = this.tip[1] * H;
    dctx.strokeStyle = v(0.9); dctx.lineCap = 'round'; dctx.lineWidth = H * 0.035;
    dctx.beginPath(); dctx.moveTo(tx + H * 0.02, ty + H * 0.3); dctx.lineTo(tx, ty); dctx.stroke();
    dctx.fillStyle = v(0.88);
    dctx.beginPath(); dctx.ellipse(tx + H * 0.04, ty + H * 0.36, H * 0.07, H * 0.08, 0, 0, Math.PI * 2); dctx.fill();
    const px = dctx.getImageData(0, 0, dw, dh).data;
    const out = new Float32Array(dw * dh);
    for (let i = 0; i < out.length; i++) out[i] = px[i * 4] / 255;
    return out;
  }
}

// ---------- scripted synthetic hand (exercises the real gesture detector) ----------
export type SynthPose = {
  tip: [number, number]; pinch: number; snap: number; fist: number; shakeX: number;
  midCurl?: number;
  indexCurl?: number; // bend the index finger
  tuck?: number;      // fold the ring and little fingers into the palm, as in a snap
};

/** 21 MediaPipe-style landmarks (source uv) for an upright hand whose index tip is at pose.tip. */
export function synthLandmarks(pose: SynthPose, h = 0.11): [number, number][] {
  const [tx, ty] = pose.tip;
  const ox = pose.shakeX;
  const s = h / 4; // phalanx length
  const pts: [number, number][] = new Array(21);
  const wrist: [number, number] = [tx + 0.01 + ox, ty + s * 6.2];
  pts[0] = wrist;
  // finger bases spread to the screen-right of the index (source left), mirrored doesn't matter
  const bases = [[0, 0], [-0.022, 0.004], [-0.04, 0.012], [-0.055, 0.024]];
  const lens = [1, 1.08, 1.0, 0.8];
  [5, 9, 13, 17].forEach((mcp, f) => {
    const bx = tx + bases[f][0] + ox, by = ty + s * 3 + bases[f][1];
    const L = s * lens[f];
    const curl = Math.max(pose.fist, f === 0 ? pose.indexCurl ?? 0 : f === 1 ? pose.midCurl ?? 0 : pose.tuck ?? 0);
    // straight finger points up; curled folds the tip back down toward the palm
    pts[mcp] = [bx, by];
    pts[mcp + 1] = [bx, by - L * (1 - curl * 0.4)];
    pts[mcp + 2] = [bx, by - L * (2 - curl * 1.6)];
    pts[mcp + 3] = [bx, by - L * (3 - curl * 3.7)];
  });
  // thumb: rests out to the side; pinch moves its tip to the index tip, snap to the middle tip
  const thumbRest: [number, number] = [tx + 0.05 + ox, ty + s * 2.2];
  const target = pose.snap > pose.pinch ? pts[12] : pts[8];
  const k = Math.max(pose.pinch, pose.snap);
  pts[1] = [wrist[0] + 0.02, wrist[1] - s * 0.8];
  pts[2] = [wrist[0] + 0.035, wrist[1] - s * 1.8];
  pts[3] = [thumbRest[0] * (1 - k * 0.5) + target[0] * k * 0.5, thumbRest[1] * (1 - k * 0.5) + target[1] * k * 0.5 + s * 0.5];
  pts[4] = [thumbRest[0] * (1 - k) + target[0] * k, thumbRest[1] * (1 - k) + target[1] * k];
  return pts;
}
