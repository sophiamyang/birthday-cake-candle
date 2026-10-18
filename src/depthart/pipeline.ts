// The depth side of each frame, all on the GPU, after TypeGPU's "Monocular Light Injection"
// example: DepthART inference on the camera frame, a stable disparity range, a motion-aware
// temporal filter and a surface texture (slope xy, occlusion, depth), plus the depth of the
// finger holding the flame. Nothing is read back; the lighting shader samples the results.
import { d, std, tgpu } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import { parseDepthBundle } from './inference/bundle.ts';
import { DepthInferencePlan } from './inference/depthart.ts';
import { DepthDisparityRangeEstimator } from './inference/disparity-range.ts';
import { fetchModel, modelLabel, modelVariant, type ModelSize } from './model-store.ts';
import {
  DEPTH_WORKGROUP_SIZE,
  DepthParams,
  SURFACE_WORKGROUP_SIZE,
  depthPrepareKernel,
  depthPrepareLayout,
  rangeStabilityLayout,
  stabilizeRangeKernel,
  surfaceKernel,
  surfaceLayout,
} from './surface.ts';

/** depth size before a model is attached (synthetic scene, flat placeholder) */
export const DEPTH_SIZE = 448;

// ---- finger light: nearest depth around the fingertip, DIP and PIP joints ----
// A single sample at the fingertip often misses: the depth map blurs thin fingers. The result
// snaps nearer quickly and drifts away slowly, so a frame that misses the finger doesn't throw
// the light back to the wall.
const FINGER_APPROACH = 20;
const FINGER_RETREAT = 1.5;
const FINGER_RING = [-2, -1, 0, 1, 2] as const;

const FingerParams = d.struct({
  tipDip: d.vec4f, // fingertip xy, DIP joint zw (source uv)
  pip: d.vec2f,
  size: d.vec2u,
  dt: d.f32,
  reset: d.u32,
});

const fingerLayout = tgpu.bindGroupLayout({
  params: { uniform: FingerParams },
  depth: { storage: d.arrayOf(d.f32), access: 'readonly' },
  finger: { storage: d.vec4f, access: 'mutable' },
});

function windowMax(uv: d.v2f): number {
  'use gpu';
  const size = d.vec2i(fingerLayout.$.params.size);
  const center = d.vec2i(std.floor(std.mul(uv, d.vec2f(fingerLayout.$.params.size))));
  let best = d.f32(0);
  for (const dy of tgpu.unroll(FINGER_RING)) {
    for (const dx of tgpu.unroll(FINGER_RING)) {
      const c = std.clamp(std.add(center, d.vec2i(dx, dy)), d.vec2i(0), std.sub(size, d.vec2i(1)));
      best = std.max(best, fingerLayout.$.depth[d.u32(c.y) * d.u32(size.x) + d.u32(c.x)]);
    }
  }
  return best;
}

const fingerKernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
  'use gpu';
  const p = fingerLayout.$.params;
  const target = std.max(std.max(windowMax(p.tipDip.xy), windowMax(p.tipDip.zw)), windowMax(p.pip));
  let next = d.f32(target);
  if (p.reset === 0) {
    const previous = fingerLayout.$.finger.x;
    let rate = d.f32(FINGER_RETREAT);
    if (target > previous) {
      rate = d.f32(FINGER_APPROACH);
    }
    next = std.mix(previous, target, std.min(p.dt * rate, d.f32(1)));
  }
  fingerLayout.$.finger = d.vec4f(next, 0, 0, 0);
});

export interface FingerInput {
  /** fingertip, DIP and PIP joints in source uv (all three may be the tip) */
  points: [[number, number], [number, number], [number, number]];
  dt: number;
}

export async function loadDepthModel(root: TgpuRoot, size: ModelSize, onStatus: (msg: string) => void) {
  const variant = modelVariant(size, root.device.features.has('shader-f16')) ?? modelVariant('small', false)!;
  const label = modelLabel(size, variant);
  onStatus(`Downloading depth model (${label}, cached after the first run)…`);
  const bundle = parseDepthBundle(await fetchModel(variant, new AbortController().signal));
  onStatus(`Compiling ${bundle.model} pipelines…`);
  // invalid GPU objects don't throw on creation; catch them here instead of failing silently later
  root.device.pushErrorScope('validation');
  const plan = new DepthInferencePlan(root, bundle);
  await plan.initAsync();
  const gpuError = await root.device.popErrorScope();
  if (gpuError) throw new Error(`GPU rejected the depth model: ${gpuError.message}`);
  return { plan, label: `${label}, ${plan.outputSize[0]}×${plan.outputSize[1]}` };
}

export class DepthPipeline {
  width = 0;
  height = 0;
  /** bumped whenever `surface` is replaced, so the renderer knows to rebind it */
  version = 0;
  private pixels = 0;
  private readonly range: DepthDisparityRangeEstimator;
  private readonly frameRange;
  private readonly stableRange;
  private readonly depthParams;
  private readonly fingerParams;
  private history!: ReturnType<DepthPipeline['makeHistory']>;
  surface!: ReturnType<DepthPipeline['makeSurface']>;
  readonly finger;
  private readonly stabilizePipeline;
  private readonly preparePipeline;
  private readonly surfacePipeline;
  private readonly fingerPipeline;
  private readonly rangeBindGroup;
  private surfaceBindGroup!: ReturnType<DepthPipeline['makeSurfaceBindGroup']>;
  private fingerBindGroup!: ReturnType<DepthPipeline['makeFingerBindGroup']>;
  private plan: DepthInferencePlan | null = null;
  private prepareBindGroup: ReturnType<DepthPipeline['makePrepareBindGroup']> | null = null;
  private reset = true;
  private fingerReset = true;

  constructor(private readonly root: TgpuRoot) {
    this.range = new DepthDisparityRangeEstimator(root);
    this.frameRange = root.createBuffer(d.vec2f, d.vec2f(0, 1)).$usage('storage');
    this.stableRange = root.createBuffer(d.vec2f, d.vec2f(0, 1)).$usage('storage');
    this.depthParams = root.createBuffer(DepthParams, { outputSize: d.vec2u(DEPTH_SIZE), reset: 1 }).$usage('uniform');
    this.fingerParams = root.createBuffer(FingerParams).$usage('uniform');
    this.finger = root.createBuffer(d.vec4f, d.vec4f(0.9, 0, 0, 0)).$usage('storage');

    this.stabilizePipeline = root.createComputePipeline({ compute: stabilizeRangeKernel });
    this.preparePipeline = root.createComputePipeline({ compute: depthPrepareKernel });
    this.surfacePipeline = root.createComputePipeline({ compute: surfaceKernel });
    this.fingerPipeline = root.createComputePipeline({ compute: fingerKernel });
    this.rangeBindGroup = root.createBindGroup(rangeStabilityLayout, {
      params: this.depthParams, frameRange: this.frameRange, stableRange: this.stableRange,
    });
    this.allocate(DEPTH_SIZE, DEPTH_SIZE);
  }

  // filtered, normalised disparity (1 = nearest); starts as a flat mid-depth wall
  private makeHistory() {
    return this.root.createBuffer(d.arrayOf(d.f32, this.pixels), Array(this.pixels).fill(0.35)).$usage('storage');
  }
  private makeSurface() {
    return this.root.createTexture({ size: [this.width, this.height], format: 'rgba16float' }).$usage('storage', 'sampled');
  }
  private makeSurfaceBindGroup() {
    return this.root.createBindGroup(surfaceLayout, {
      params: this.depthParams,
      depth: this.history,
      surface: this.surface.createView(d.textureStorage2d('rgba16float', 'write-only')),
    });
  }
  private makeFingerBindGroup() {
    return this.root.createBindGroup(fingerLayout, { params: this.fingerParams, depth: this.history, finger: this.finger });
  }

  /** (Re)creates everything sized by the depth map: the model decides its output size. */
  private allocate(width: number, height: number) {
    if (width === this.width && height === this.height) return;
    this.history?.destroy();
    this.surface?.destroy();
    this.width = width; this.height = height; this.pixels = width * height;
    this.history = this.makeHistory();
    this.surface = this.makeSurface();
    this.surfaceBindGroup = this.makeSurfaceBindGroup();
    this.fingerBindGroup = this.makeFingerBindGroup();
    this.depthParams.write({ outputSize: d.vec2u(width, height), reset: 1 });
    this.prepareBindGroup = null;
    this.version++;
  }

  async initAsync() {
    await Promise.all([
      this.range.initAsync(),
      this.stabilizePipeline.initAsync(),
      this.preparePipeline.initAsync(),
      this.surfacePipeline.initAsync(),
      this.fingerPipeline.initAsync(),
    ]);
  }

  private makePrepareBindGroup(disparity: ReturnType<DepthPipeline['makeDisparity']>) {
    return this.root.createBindGroup(depthPrepareLayout, {
      params: this.depthParams, disparity, stableRange: this.stableRange, history: this.history,
    });
  }

  private makeDisparity(plan: DepthInferencePlan) {
    return this.root.createBuffer(d.arrayOf(d.vec4f, this.pixels), plan.outputBuffer).$usage('storage');
  }

  get hasModel() { return this.plan !== null; }

  /** Stop running the model (after an error); the last depth stays in place. */
  detach() { this.plan = null; this.prepareBindGroup = null; }

  attach(plan: DepthInferencePlan) {
    this.allocate(plan.outputSize[0], plan.outputSize[1]);
    const disparity = this.makeDisparity(plan);
    this.range.attach(disparity, this.frameRange, this.pixels);
    this.prepareBindGroup = this.makePrepareBindGroup(disparity);
    this.plan = plan;
    this.reset = true;
  }

  /** Replace the filtered depth directly (synthetic scene): width × height values, 1 = nearest. */
  writeDepth(data: Float32Array) {
    this.root.device.queue.writeBuffer(this.root.unwrap(this.history), 0, data as Float32Array<ArrayBuffer>);
  }

  /**
   * Encodes and submits one frame of depth work: the model on `frame` when given (a new camera
   * frame), then the surface texture, then the finger light. Submit before the lighting pass.
   */
  run(frame: HTMLVideoElement | VideoFrame | null, depthChanged: boolean, finger: FingerInput | null) {
    const encoder = this.root['~unstable'].createCommandEncoder();
    const pass = encoder.beginComputePass();
    const plan = this.plan, prepare = this.prepareBindGroup;
    if (frame && plan && prepare) {
      this.depthParams.patch({ reset: this.reset ? 1 : 0 });
      const external = this.root.device.importExternalTexture({ source: frame });
      plan.encodeFrame(pass, external, { uvTransform: d.mat2x2f.identity(), mirrorX: false, swapAxes: false, fullFrame: true });
      this.range.encode(pass);
      this.stabilizePipeline.with(pass).with(this.rangeBindGroup).dispatchWorkgroups(1);
      this.preparePipeline.with(pass).with(prepare).dispatchWorkgroups(Math.ceil(this.pixels / DEPTH_WORKGROUP_SIZE));
      this.reset = false;
      depthChanged = true;
    }
    if (depthChanged) {
      this.surfacePipeline.with(pass).with(this.surfaceBindGroup)
        .dispatchWorkgroups(Math.ceil(this.width / SURFACE_WORKGROUP_SIZE), Math.ceil(this.height / SURFACE_WORKGROUP_SIZE));
    }
    if (finger) {
      const [tip, dip, pip] = finger.points;
      this.fingerParams.write({
        tipDip: d.vec4f(tip[0], tip[1], dip[0], dip[1]),
        pip: d.vec2f(pip[0], pip[1]),
        size: d.vec2u(this.width, this.height),
        dt: finger.dt,
        reset: this.fingerReset ? 1 : 0,
      });
      this.fingerPipeline.with(pass).with(this.fingerBindGroup).dispatchWorkgroups(1);
      this.fingerReset = false;
    }
    pass.end();
    encoder.submit();
  }
}
