// "Make a wish" moment: a short confetti burst and a caption, drawn on a 2D canvas
// that sits above the WebGPU canvas. It is part of the shot, so recording mode keeps it.
interface P { x: number; y: number; vx: number; vy: number; r: number; vr: number; c: string; w: number; h: number }
const COLORS = ['#ff6f91', '#ffd166', '#4ecdc4', '#a78bfa', '#ffffff', '#7bd389', '#ff9f6b'];

export class Confetti {
  private ctx: CanvasRenderingContext2D;
  private parts: P[] = [];
  private caption = 0; // seconds remaining
  private dirty = false; // something was drawn last frame and still needs clearing
  constructor(private canvas: HTMLCanvasElement) { this.ctx = canvas.getContext('2d')!; }

  burst() {
    const W = this.canvas.width, H = this.canvas.height;
    for (let i = 0; i < 220; i++) {
      const left = i % 2 === 0;
      this.parts.push({
        x: left ? W * 0.05 : W * 0.95, y: H * 0.9,
        vx: (left ? 1 : -1) * (H * (0.25 + Math.random() * 0.55)), vy: -H * (0.7 + Math.random() * 0.7),
        r: Math.random() * 6, vr: (Math.random() - 0.5) * 12, c: COLORS[i % COLORS.length],
        w: 6 + Math.random() * 6, h: 3 + Math.random() * 4,
      });
    }
    this.caption = 4.5;
  }

  cancel() { this.parts = []; this.caption = 0; }

  private sparks: { x: number; y: number; vx: number; vy: number; life: number }[] = [];
  /** Little ignition puff at screen uv (x, y). */
  spark(x: number, y: number) {
    const W = this.canvas.width, H = this.canvas.height;
    for (let i = 0; i < 26; i++) {
      const a = -Math.PI / 2 + (Math.random() - 0.5) * Math.PI * 1.6, v = H * (0.15 + Math.random() * 0.45);
      this.sparks.push({ x: x * W, y: y * H, vx: Math.cos(a) * v, vy: Math.sin(a) * v, life: 0.25 + Math.random() * 0.4 });
    }
  }

  get active() { return this.parts.length > 0 || this.caption > 0 || this.sparks.length > 0; }

  frame(dt: number) {
    const c = this.canvas, dpr = Math.min(window.devicePixelRatio || 1, 2);
    const W = Math.round(c.clientWidth * dpr), H = Math.round(c.clientHeight * dpr);
    if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
    const ctx = this.ctx;
    if (this.dirty) ctx.clearRect(0, 0, W, H);
    this.dirty = this.active;
    if (!this.active) return;
    const g = H * 1.1;
    this.parts = this.parts.filter((p) => p.y < H + 20);
    for (const p of this.parts) {
      p.vy += g * dt; p.vx *= 1 - dt * 0.8; p.vy *= 1 - dt * 0.6;
      p.x += p.vx * dt; p.y += p.vy * dt; p.r += p.vr * dt;
      ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(p.r);
      ctx.scale(1, Math.abs(Math.cos(p.r * 1.7)) + 0.15);
      ctx.fillStyle = p.c; ctx.globalAlpha = 0.9; ctx.fillRect(-p.w / 2, -p.h / 2, p.w, p.h); ctx.restore();
    }
    if (this.sparks.length) {
      ctx.save();
      ctx.globalCompositeOperation = 'lighter';
      for (const p of this.sparks) {
        p.vy += H * 0.9 * dt; p.vx *= 1 - dt * 3; p.vy *= 1 - dt * 2;
        p.x += p.vx * dt; p.y += p.vy * dt; p.life -= dt;
        const x0 = p.x - p.vx * 0.025, y0 = p.y - p.vy * 0.025; // short motion streak
        const a = Math.max(0, Math.min(1, p.life * 3));
        ctx.strokeStyle = `rgba(255,${170 + Math.round(60 * a)},90,${a})`;
        ctx.lineWidth = Math.max(1, H * 0.003);
        ctx.shadowColor = 'rgba(255,150,60,0.9)'; ctx.shadowBlur = H * 0.01;
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(p.x, p.y); ctx.stroke();
      }
      ctx.restore();
      this.sparks = this.sparks.filter((p) => p.life > 0);
    }
    if (this.caption > 0) {
      this.caption -= dt;
      const a = Math.min(1, this.caption / 0.8, (4.5 - this.caption) / 0.6);
      ctx.save();
      ctx.globalAlpha = Math.max(0, a);
      ctx.font = `600 ${Math.round(H * 0.07)}px ui-serif, Georgia, serif`;
      ctx.textAlign = 'center';
      ctx.shadowColor = 'rgba(255,170,90,0.8)'; ctx.shadowBlur = H * 0.03;
      ctx.fillStyle = '#ffe9c7';
      ctx.fillText('Make a wish ✨', W / 2, H * 0.2);
      ctx.restore();
    }
  }
}
