import { FilesetResolver, HandLandmarker, type NormalizedLandmark } from '@mediapipe/tasks-vision';

// keep the version in step with @mediapipe/tasks-vision in package.json (pinned exactly)
export const VISION_WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.1/wasm';
const MODEL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

export interface HandState {
  tip: [number, number];     // index fingertip, source uv (un-mirrored)
  size: number;              // wrist -> middle MCP distance, uv units
  points: [number, number][]; // all 21 landmarks, source uv
  world: [number, number, number][] | null; // the same in metres (true 3D, centred on the hand)
}

const dist = (a: NormalizedLandmark, b: NormalizedLandmark) => Math.hypot(a.x - b.x, a.y - b.y);

export class HandTracker {
  lastMs = 0;
  private constructor(private lm: HandLandmarker) {}

  static async create(): Promise<HandTracker> {
    const fileset = await FilesetResolver.forVisionTasks(VISION_WASM);
    const make = (delegate: 'GPU' | 'CPU') =>
      HandLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MODEL, delegate },
        runningMode: 'VIDEO',
        numHands: 1,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
    let lm: HandLandmarker;
    try { lm = await make('GPU'); } catch { lm = await make('CPU'); }
    return new HandTracker(lm);
  }

  detect(source: HTMLVideoElement, ts: number): HandState | null {
    const t0 = performance.now();
    const res = this.lm.detectForVideo(source, ts);
    this.lastMs = performance.now() - t0;
    const h = res.landmarks?.[0];
    if (!h) return null;
    const wrist = h[0];
    const size = dist(wrist, h[9]);
    const wl = res.worldLandmarks?.[0];
    return {
      tip: [h[8].x, h[8].y], size,
      points: h.map((p) => [p.x, p.y] as [number, number]),
      world: wl ? wl.map((p) => [p.x, p.y, p.z] as [number, number, number]) : null,
    };
  }
}

const BONES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];

/** Rasterise a soft hand silhouette (source uv) used to let the hand pass in front of the cake. */
export function drawHandMask(ctx: CanvasRenderingContext2D, pts: [number, number][], size: number) {
  const { width: W, height: H } = ctx.canvas;
  ctx.save();
  ctx.filter = 'blur(1.5px)';
  ctx.strokeStyle = '#fff'; ctx.fillStyle = '#fff'; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  ctx.lineWidth = Math.max(2, size * H * 0.3);
  ctx.beginPath();
  for (const [a, b] of BONES) { ctx.moveTo(pts[a][0] * W, pts[a][1] * H); ctx.lineTo(pts[b][0] * W, pts[b][1] * H); }
  ctx.stroke();
  ctx.beginPath();
  [0, 1, 5, 9, 13, 17].forEach((i, k) => (k ? ctx.lineTo : ctx.moveTo).call(ctx, pts[i][0] * W, pts[i][1] * H));
  ctx.closePath(); ctx.fill();
  // the forearm continues below the wrist
  const w = pts[0], m = pts[9];
  ctx.lineWidth = Math.max(3, size * H * 0.75);
  ctx.beginPath(); ctx.moveTo(w[0] * W, w[1] * H); ctx.lineTo((w[0] + (w[0] - m[0]) * 3) * W, (w[1] + (w[1] - m[1]) * 3) * H); ctx.stroke();
  ctx.restore();
}
