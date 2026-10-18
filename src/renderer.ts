import tgpu from 'typegpu';
import * as d from 'typegpu/data';
import { shaderTemplate } from './shader';
import { DepthPipeline } from './depthart/pipeline';

export const Candle = d.struct({
  screen: d.vec4f, // screen uv xy, flame size, visible amount
  light: d.vec4f,  // light at source uv xy, normalised depth z (1 = nearest), power w (0 = off)
  anim: d.vec4f,   // bend, stretch, smoke, seed
});
export const MAX_GPU_CANDLES = 64;
const CandleArray = d.arrayOf(Candle, MAX_GPU_CANDLES);
// Candle is three vec4f with no padding, so the buffer can be filled straight from a Float32Array.
const CANDLE_FLOATS = 12;
if (d.sizeOf(Candle) !== CANDLE_FLOATS * 4) throw new Error('Candle layout changed; update CANDLE_FLOATS');
export type CandleGPU = { screen: [number, number, number, number]; light: [number, number, number, number]; anim: [number, number, number, number] };

export const Uniforms = d.struct({
  cakeRect: d.vec4f,     // source-uv rect of the cake sprite: min xy, size xy
  shadowLight: d.vec4f,  // power-weighted centroid of lit candles (source uv, depth), w = total power
  resolution: d.vec2f,
  uvScale: d.vec2f,
  tipScreen: d.vec2f,
  tipSrc: d.vec2f,
  velocity: d.vec2f,
  time: d.f32,
  flameAmt: d.f32,
  flameSize: d.f32,
  flicker: d.f32,
  videoAspect: d.f32,
  intensity: d.f32,      // finger flame brightness
  lightRadius: d.f32,    // distance (frame heights) at which a light falls to half
  ambient: d.f32,        // how much of the room stays visible
  exposure: d.f32,
  relief: d.f32,         // strength of the depth-derived surface normals
  specular: d.f32,
  shadow: d.f32,         // shadow strength
  occlusion: d.f32,      // ambient occlusion strength
  shadows: d.f32,
  view: d.u32,
  cakeOn: d.f32,
  cakeDepth: d.f32,      // normalised depth of the cake front (1 = nearest)
  candleCount: d.u32,
});

export interface FrameParams {
  tipScreen: [number, number];
  tipSrc: [number, number];
  velocity: [number, number];
  time: number;
  flameAmt: number;
  flameSize: number;
  flicker: number;
  intensity: number;
  lightRadius: number;
  ambient: number;
  exposure: number;
  relief: number;
  specular: number;
  shadow: number;
  occlusion: number;
  shadows: boolean;
  view: number;
  cakeOn?: boolean;
  cakeDepth?: number;
  cakeRect?: [number, number, number, number];
  candles?: CandleGPU[];
  shadowLight?: [number, number, number, number];
}

export type Root = ReturnType<typeof tgpu.initFromDevice>;
// hold strong references so the adapter/device are never garbage-collected
const keepAlive: unknown[] = [];

export class Renderer {
  readonly root: Root;
  readonly device: GPUDevice;
  private ctx: GPUCanvasContext;
  private format: GPUTextureFormat;
  private pipeline: GPURenderPipeline;
  private layout: GPUBindGroupLayout;
  private uniforms;
  private sampler: GPUSampler;
  private camTex: GPUTexture | null = null;
  readonly depth: DepthPipeline;
  private cakeColTex: GPUTexture;
  private cakeGeoTex: GPUTexture;
  private maskTex: GPUTexture;
  private candleBuf;
  private candleData = new Float32Array(MAX_GPU_CANDLES * CANDLE_FLOATS);
  private bindGroup: GPUBindGroup | null = null;
  private depthVersion = -1;
  videoW = 0;
  videoH = 0;
  uvScale: [number, number] = [1, 1];

  private constructor(root: Root, private canvas: HTMLCanvasElement) {
    this.root = root;
    this.device = root.device;
    const ctx = canvas.getContext('webgpu');
    if (!ctx) throw new Error('Could not get a WebGPU canvas context');
    this.ctx = ctx;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    ctx.configure({ device: this.device, format: this.format, alphaMode: 'opaque' });

    this.uniforms = root.createBuffer(Uniforms).$usage('uniform');
    this.candleBuf = root.createBuffer(CandleArray).$usage('storage');

    // TypeGPU resolves the typed struct into WGSL and links it into the template.
    const code = tgpu.resolve({ template: shaderTemplate, externals: { Uniforms, Candle } });
    const module = this.device.createShaderModule({ code, label: 'finger-flame' });
    module.getCompilationInfo().then((info) => {
      for (const m of info.messages) console[m.type === 'error' ? 'error' : 'warn'](`WGSL ${m.type} ${m.lineNum}:${m.linePos} ${m.message}`);
    });

    this.layout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 7, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 8, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      ],
    });
    this.pipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list' },
    });
    this.sampler = this.device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    this.cakeColTex = this.makeRgbaTex(1, 1);
    this.cakeGeoTex = this.makeRgbaTex(1, 1);
    this.maskTex = this.makeRgbaTex(1, 1);
    this.depth = new DepthPipeline(root);
  }

  static async create(canvas: HTMLCanvasElement): Promise<Renderer> {
    if (!navigator.gpu) throw new Error('WebGPU is not available in this browser');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter found');
    // half-precision shaders halve the depth model's download and speed it up
    const f16: GPUFeatureName[] = adapter.features.has('shader-f16') ? ['shader-f16'] : [];
    const device = await adapter.requestDevice({ requiredFeatures: f16 });
    keepAlive.push(adapter, device);
    const root = tgpu.initFromDevice({ device });
    const renderer = new Renderer(root, canvas);
    await renderer.depth.initAsync();
    return renderer;
  }

  private makeRgbaTex(w: number, h: number) {
    return this.device.createTexture({
      size: [w, h], format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
  }

  private writeRgba(which: 'cakeColTex' | 'cakeGeoTex' | 'maskTex', img: ImageData) {
    let tex = this[which];
    if (tex.width !== img.width || tex.height !== img.height) {
      tex.destroy();
      tex = this[which] = this.makeRgbaTex(img.width, img.height);
      this.bindGroup = null;
    }
    this.device.queue.writeTexture({ texture: tex }, img.data, { bytesPerRow: img.width * 4 }, [img.width, img.height]);
  }

  uploadCake(color: ImageData, geo: ImageData) { this.writeRgba('cakeColTex', color); this.writeRgba('cakeGeoTex', geo); }
  uploadMask(mask: ImageData) { this.writeRgba('maskTex', mask); }

  /** Copy the current camera frame (video or canvas) into a GPU texture. */
  uploadFrame(source: HTMLVideoElement | HTMLCanvasElement, w: number, h: number) {
    // no decoded frame yet (or mid-seek): keep the last one rather than tripping the CPU fallback below
    if (source instanceof HTMLVideoElement && source.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
    if (!this.camTex || this.videoW !== w || this.videoH !== h) {
      this.camTex?.destroy();
      this.camTex = this.device.createTexture({
        size: [w, h],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      this.videoW = w;
      this.videoH = h;
      this.bindGroup = null;
    }
    if (!this.cpuUpload) {
      try {
        this.device.queue.copyExternalImageToTexture({ source }, { texture: this.camTex }, [w, h]);
        return;
      } catch (e) {
        console.warn('copyExternalImageToTexture failed; falling back to CPU upload', e);
        this.cpuUpload = true;
      }
    }
    // Fallback (some headless/software GPU setups): read pixels via 2D canvas.
    if (!this.scratch) this.scratch = document.createElement('canvas').getContext('2d', { willReadFrequently: true })!;
    const c = this.scratch.canvas;
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    this.scratch.drawImage(source, 0, 0, w, h);
    const img = this.scratch.getImageData(0, 0, w, h);
    this.device.queue.writeTexture({ texture: this.camTex }, img.data, { bytesPerRow: w * 4 }, [w, h]);
  }
  private cpuUpload = new URLSearchParams(location.search).has('cpuUpload');
  private scratch: CanvasRenderingContext2D | null = null;

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    let w = Math.round(this.canvas.clientWidth * dpr);
    let h = Math.round(this.canvas.clientHeight * dpr);
    const maxW = 2400;
    if (w > maxW) { h = Math.round((h * maxW) / w); w = maxW; }
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = Math.max(1, w);
      this.canvas.height = Math.max(1, h);
    }
    // object-fit: cover mapping from screen uv to source uv
    const ca = this.canvas.width / this.canvas.height;
    const va = this.videoW && this.videoH ? this.videoW / this.videoH : ca;
    this.uvScale = ca > va ? [1, va / ca] : [ca / va, 1];
  }

  /** Map a source-image uv (un-mirrored) to screen uv (mirrored, cover-fit). */
  srcToScreen(x: number, y: number): [number, number] {
    const mx = 1 - x;
    return [0.5 + (mx - 0.5) / this.uvScale[0], 0.5 + (y - 0.5) / this.uvScale[1]];
  }

  render(p: FrameParams) {
    if (!this.camTex) return;
    this.resize();
    if (this.depthVersion !== this.depth.version) { this.depthVersion = this.depth.version; this.bindGroup = null; }
    if (!this.bindGroup) {
      this.bindGroup = this.device.createBindGroup({
        layout: this.layout,
        entries: [
          { binding: 0, resource: { buffer: this.root.unwrap(this.uniforms) } },
          { binding: 1, resource: this.sampler },
          { binding: 2, resource: this.camTex.createView() },
          { binding: 3, resource: this.root.unwrap(this.depth.surface).createView() },
          { binding: 4, resource: { buffer: this.root.unwrap(this.candleBuf) } },
          { binding: 5, resource: this.cakeColTex.createView() },
          { binding: 6, resource: this.cakeGeoTex.createView() },
          { binding: 7, resource: this.maskTex.createView() },
          { binding: 8, resource: { buffer: this.root.unwrap(this.depth.finger) } },
        ],
      });
    }
    // the shader only reads the first candleCount entries, so only those are uploaded
    const n = p.candles ? Math.min(p.candles.length, MAX_GPU_CANDLES) : 0;
    this.uniforms.write({
      resolution: d.vec2f(this.canvas.width, this.canvas.height),
      uvScale: d.vec2f(...this.uvScale),
      tipScreen: d.vec2f(...p.tipScreen),
      tipSrc: d.vec2f(...p.tipSrc),
      velocity: d.vec2f(...p.velocity),
      time: p.time,
      flameAmt: p.flameAmt,
      flameSize: p.flameSize,
      flicker: p.flicker,
      videoAspect: this.videoW / this.videoH,
      intensity: p.intensity,
      lightRadius: p.lightRadius,
      ambient: p.ambient,
      exposure: p.exposure,
      relief: p.relief,
      specular: p.specular,
      shadow: p.shadow,
      occlusion: p.occlusion,
      shadows: p.shadows ? 1 : 0,
      view: p.view,
      cakeOn: p.cakeOn ? 1 : 0,
      cakeDepth: p.cakeDepth ?? 0.87,
      candleCount: n,
      cakeRect: d.vec4f(...(p.cakeRect ?? [0, 0, 1, 1])),
      shadowLight: d.vec4f(...(p.shadowLight ?? [0, 0, 1, 0])),
    });
    if (n) {
      const f = this.candleData;
      for (let i = 0; i < n; i++) {
        const c = p.candles![i];
        f.set(c.screen, i * CANDLE_FLOATS);
        f.set(c.light, i * CANDLE_FLOATS + 4);
        f.set(c.anim, i * CANDLE_FLOATS + 8);
      }
      this.device.queue.writeBuffer(this.root.unwrap(this.candleBuf), 0, f, 0, n * CANDLE_FLOATS);
    }
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: this.ctx.getCurrentTexture().createView(),
        loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(3);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }
}
