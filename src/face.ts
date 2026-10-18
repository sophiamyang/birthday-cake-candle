import { FilesetResolver, FaceLandmarker } from '@mediapipe/tasks-vision';
import { VISION_WASM } from './hands';

const MODEL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

export interface FaceState {
  mouth: [number, number]; // source uv
  blow: number;            // 0..1 "blowing" expression
  rounded: number;         // 0..1 how much the mouth looks like it's blowing (rounded lips, puffed cheeks)
}

/** Fallback blow detector: an open, rounded mouth (funnel/pucker) or puffed cheeks. */
export class FaceTracker {
  lastMs = 0;
  private constructor(private lm: FaceLandmarker) {}

  static async create(): Promise<FaceTracker> {
    const fileset = await FilesetResolver.forVisionTasks(VISION_WASM);
    const make = (delegate: 'GPU' | 'CPU') => FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL, delegate },
      runningMode: 'VIDEO',
      numFaces: 1,
      outputFaceBlendshapes: true,
    });
    let lm: FaceLandmarker;
    try { lm = await make('GPU'); } catch { lm = await make('CPU'); }
    return new FaceTracker(lm);
  }

  detect(video: HTMLVideoElement, ts: number): FaceState | null {
    const t0 = performance.now();
    const res = this.lm.detectForVideo(video, ts);
    this.lastMs = performance.now() - t0;
    const lms = res.faceLandmarks?.[0];
    if (!lms) return null;
    const bs: Record<string, number> = {};
    for (const c of res.faceBlendshapes?.[0]?.categories ?? []) bs[c.categoryName] = c.score;
    const jaw = bs.jawOpen ?? 0, funnel = bs.mouthFunnel ?? 0, pucker = bs.mouthPucker ?? 0, puff = bs.cheekPuff ?? 0;
    const rounded = Math.max(funnel, pucker * 0.8);
    let blow = 0;
    if (jaw > 0.08 && jaw < 0.6 && rounded > 0.35) blow = Math.min(1, (rounded - 0.35) * 2.2 + 0.3);
    if (puff > 0.4) blow = Math.max(blow, Math.min(1, (puff - 0.4) * 2 + 0.3));
    const m = lms[13];
    // rounded lips or puffed cheeks; a wide-open jaw means talking or singing, not blowing
    const blowShape = jaw < 0.6 ? Math.min(1, Math.max(0, (Math.max(rounded, puff) - 0.25) / 0.2)) : 0;
    return { mouth: [m.x, m.y], blow, rounded: blowShape };
  }
}
