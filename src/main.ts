import { Renderer } from './renderer';
import { loadDepthModel, type FingerInput } from './depthart/pipeline';
import { MODEL_SIZES, type ModelSize } from './depthart/model-store';
import { HandTracker, type HandState } from './hands';
import { SyntheticScene, synthLandmarks, type SynthPose } from './synthetic';
import { GestureDetector } from './gestures';
import { drawHandMask } from './hands';
import { CakeMode, windReach } from './cakeMode';
import { BlowDetector } from './blow';
import { FaceTracker, type FaceState } from './face';
import { Confetti } from './confetti';
import { numParam } from './params';

const params = new URLSearchParams(location.search);
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const statusEl = $('status');
const statsEl = $('stats');
const overlayEl = $('overlay');

let statusTimer = 0;
function status(msg: string, sticky = false) {
  statusEl.textContent = msg;
  statusEl.classList.remove('hidden');
  clearTimeout(statusTimer);
  if (!sticky) statusTimer = window.setTimeout(() => statusEl.classList.add('hidden'), 3500);
}

// ---------- UI state ----------
const VIEWS = ['lit', 'depth', 'normals', 'light only'];
const ui = {
  overlay: params.get('overlay') !== '0',
  recording: params.has('record'),
  view: Math.max(0, Math.min(3, Number(params.get('view') ?? 0) || 0)),
  shadows: params.get('shadows') !== '0',
  // the cake is the point of the app: on unless ?cake=0 or ?mode=flame asks for the finger alone
  cake: params.get('cake') !== '0' && params.get('mode') !== 'flame',
};
function applyUi() {
  overlayEl.classList.toggle('off', !ui.overlay);
  $('cakePanel').classList.toggle('off', !ui.cake);
  document.body.classList.toggle('recording', ui.recording);
}
function toggleRecording() {
  ui.recording = !ui.recording;
  applyUi();
  if (!ui.recording) status('Recording mode off');
}
$('btnOverlay').addEventListener('click', () => { ui.overlay = !ui.overlay; applyUi(); });
$('btnRecord').addEventListener('click', toggleRecording);

// ---------- flame state ----------
const flame = { lit: true, amt: 0, flare: 0, smoke: 0, exposure: 0, bend: 0 };

window.addEventListener('keydown', (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (e.repeat && !['+', '=', '-', '_'].includes(k)) return; // holding O or K must not strobe
  if (k === 'o' || k === 'd') { ui.overlay = !ui.overlay; applyUi(); }
  else if (k === 'r') toggleRecording();
  else if (k === 'escape' && ui.recording) toggleRecording();
  else if (k === 'v') { ui.view = (ui.view + 1) % VIEWS.length; status(`View: ${VIEWS[ui.view]}`); }
  else if (k === 's') { ui.shadows = !ui.shadows; status(`Shadows ${ui.shadows ? 'on' : 'off'}`); }
  else if (k === 'f') hooks.setFinger(!flame.lit, 'key F');
  else if (k === 'k') hooks.toggleCake();
  else if (k === 'l') hooks.lightAll();
  else if (k === 'g') hooks.saveGestures();
  else if (k === 'b' && !keyBlow.down) { keyBlow.down = true; keyBlow.since = performance.now(); }
  else if ((k === '+' || k === '=') && ui.cake) hooks.setCandles(hooks.candles() + 1);
  else if ((k === '-' || k === '_') && ui.cake) hooks.setCandles(hooks.candles() - 1);
});
window.addEventListener('keyup', (e) => { if (e.key.toLowerCase() === 'b') keyBlow.down = false; });
const keyBlow = { down: false, since: 0 };
// filled in by main() once the renderer exists
const hooks = { toggleCake: () => {}, lightAll: () => {}, setCandles: (_n: number) => {}, candles: () => 0, setFinger: (_on: boolean, _why: string) => {}, saveGestures: () => {} };
applyUi();

// ---------- tuning (face-first defaults; override via URL for experiments) ----------
const num = (k: string, def: number, min?: number, max?: number) => numParam(params, k, def, min, max);
// Lighting model and defaults follow TypeGPU's "Monocular Light Injection" example, with a
// tighter light radius and a darker room so the flame reads as a candle.
const tune = {
  intensity: num('intensity', 3, 0),          // finger flame brightness
  lightRadius: num('radius', 0.6, 0.05, 5),   // frame heights at which a light falls to half
  ambient: num('ambient', 0.35, 0),           // how much of the room stays visible
  exposure: num('exposure', 1, 0),
  relief: num('relief', 0.85, 0, 3),          // strength of the depth-derived surface normals
  specular: num('specular', 0.22, 0),
  shadow: num('shadow', 0.7, 0, 1),
  occlusion: num('occlusion', 0.55, 0, 1),
  flameSize: num('flame', 0.085, 0.005, 1),   // flame height as a fraction of screen height
  candles: Math.round(num('candles', 12, 1, 64)),
  candlePower: num('candlePower', 0.3, 0),    // per birthday candle; more candles = brighter face
  cakeDepth: num('cakeDepth', 0.87, 0, 1),    // normalised depth of the cake front (1 = nearest)
};
const modelSize: ModelSize = (MODEL_SIZES as readonly string[]).includes(params.get('model') ?? '') ? (params.get('model') as ModelSize) : 'small';

// Smooth value noise for flicker: candle flames sway gently, with rare gutters.
const n1 = (t: number) => {
  const i = Math.floor(t), f = t - i, s = f * f * (3 - 2 * f);
  const h = (x: number) => { const v = Math.sin(x * 127.1 + 311.7) * 43758.5453; return v - Math.floor(v); };
  return h(i) * (1 - s) + h(i + 1) * s;
};
const flickerAt = (t: number) => 0.9 + 0.07 * (n1(t * 5.0) - 0.5) * 2 + 0.03 * (n1(t * 13.0 + 7) - 0.5) * 2 - 0.05 * Math.pow(n1(t * 0.7 + 3), 6);

async function startCamera(): Promise<HTMLVideoElement> {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user', frameRate: { ideal: 60 } },
    audio: false,
  });
  const video = document.createElement('video');
  video.playsInline = true;
  video.muted = true;
  video.srcObject = stream;
  await video.play();
  if (!video.videoWidth) await new Promise((r) => video.addEventListener('loadedmetadata', r, { once: true }));
  return video;
}

async function main() {
  let renderer: Renderer;
  try {
    renderer = await Renderer.create($<HTMLCanvasElement>('gfx'));
  } catch (e) {
    status(`WebGPU unavailable: ${(e as Error).message}. Use Chrome/Edge 113+ on a GPU-enabled machine.`, true);
    return;
  }
  renderer.device.lost.then((info) => { console.warn('device lost', info.reason, info.message, performance.now()); status(`GPU device lost: ${info.message}`, true); });
  let gpuError = '';
  renderer.device.addEventListener('uncapturederror', (e) => {
    gpuError = (e as GPUUncapturedErrorEvent).error.message;
    console.error('WebGPU error:', gpuError);
  });

  // Camera, or the synthetic scene when there is none (or ?synthetic).
  let video: HTMLVideoElement | null = null;
  let synth: SyntheticScene | null = null;
  if (!params.has('synthetic')) {
    try {
      status('Requesting camera…', true);
      video = await startCamera();
    } catch (e) {
      status(`No camera (${(e as Error).name}); showing a synthetic scene instead.`);
    }
  }
  if (!video) synth = new SyntheticScene();
  const srcW = () => (video ? video.videoWidth : synth!.w);
  const srcH = () => (video ? video.videoHeight : synth!.h);
  const source = () => (video ?? synth!.canvas);

  // Load ML models in the background; render meanwhile.
  let hands: HandTracker | null = null;
  let depthNote = 'loading';
  if (video) {
    HandTracker.create().then((h) => { hands = h; status(ui.cake ? 'Hand tracking ready. Snap or pinch to light your finger, then hold the flame on each wick.' : 'Hand tracking ready. Raise your index finger.'); })
      .catch((e) => { console.error(e); status(`Hand tracker failed: ${e.message}`); });
  }
  const useModel = video ? params.get('depth') !== 'off' : params.get('depth') === 'model';
  if (useModel) {
    loadDepthModel(renderer.root, modelSize, (m) => status(m, true))
      .then(({ plan, label }) => { renderer.depth.attach(plan); depthNote = `DepthART ${label}`; status(`Depth model ready (DepthART ${label})`); })
      .catch((e) => { console.error(e); depthNote = `failed: ${e.message}`; status(`Depth model failed: ${e.message}`, true); });
  } else {
    depthNote = 'synthetic';
  }

  // ---- birthday cake mode ----
  let cake: CakeMode | null = null;
  let blowMic: BlowDetector | null = null;
  let micNote = 'off';
  let face: FaceTracker | null = null;
  let faceState: FaceState | null = null;
  let faceLoading = false;
  const confetti = new Confetti($<HTMLCanvasElement>('fx'));
  const maskCanvas = document.createElement('canvas');
  maskCanvas.width = 384; maskCanvas.height = 216;
  const maskCtx = maskCanvas.getContext('2d', { willReadFrequently: true })!;
  const demoBlow = params.get('demo') === 'blow';
  const cRange = $<HTMLInputElement>('cRange');
  const micMeter = document.querySelector<HTMLElement>('#micMeter i')!;
  const syncPicker = () => { if (cake) { cRange.value = String(Math.min(40, cake.count)); $('cCount').textContent = String(cake.count); } };

  let micPending = false;
  async function enableMic() {
    if (blowMic?.live || micPending) return;
    micPending = true;
    micNote = 'requesting';
    try {
      const mic = await BlowDetector.create();
      blowMic?.dispose();
      blowMic = mic;
      micNote = 'on';
      $('cMic').classList.add('on');
    } catch (e) {
      micNote = `unavailable (${(e as Error).name})`;
      status(`Microphone unavailable: blow with an open, rounded mouth or hold B instead.`);
    } finally {
      micPending = false;
    }
  }
  function startCake() {
    if (!cake) {
      cake = new CakeMode(renderer, tune.candles, tune.cakeDepth, tune.candlePower, params.get('lit') === 'all');
      cake.freezeDemo = demoBlow;
      cake.onAllOut = () => { confetti.burst(); status('All out. Make a wish!'); };
      cake.onLit = () => confetti.cancel();
      if (demoBlow) {
        cake.lightAll();
        cake.candles.forEach((c, i) => { if (i % 3 === 1) { c.lit = false; c.amt = 0; c.smoke = 0.75; } });
      }
    }
    syncPicker();
    if (!params.has('nomic')) enableMic();
    if (video && !face && !faceLoading) {
      faceLoading = true;
      FaceTracker.create().then((f) => { face = f; }).catch((e) => console.warn('face tracker failed', e));
    }
  }
  hooks.toggleCake = () => { ui.cake = !ui.cake; if (ui.cake) startCake(); else { confetti.cancel(); } applyUi(); };
  hooks.lightAll = () => { if (cake) { cake.lightAll(); confetti.cancel(); } };
  hooks.setCandles = (n: number) => { if (cake) { cake.setCount(n); syncPicker(); } };
  hooks.candles = () => cake?.count ?? 0;
  $('btnCake').addEventListener('click', hooks.toggleCake);
  $('cLight').addEventListener('click', hooks.lightAll);
  $('cMic').addEventListener('click', enableMic);
  $('cMinus').addEventListener('click', () => hooks.setCandles(hooks.candles() - 1));
  $('cPlus').addEventListener('click', () => hooks.setCandles(hooks.candles() + 1));
  cRange.addEventListener('input', () => hooks.setCandles(Number(cRange.value)));
  if (ui.cake) startCake();
  applyUi();
  let roomDim = 1;

  // ---- finger flame on/off (gestures, F key, blowing) ----
  const gestures = new GestureDetector();
  const log: { t: number; ev: string; lit: number }[] = [];
  (window as unknown as { __ffLog: typeof log }).__ffLog = log;
  // the last 20 s of hand landmarks, saved with G, to tune the gestures on a real hand
  const handLog: { t: number; lit: boolean; ev: string | null; p: number[][] | null; w: number[][] | null }[] = [];
  const round = (pts: readonly (readonly number[])[] | null | undefined) => pts ? pts.map((q) => q.map((v) => +v.toFixed(4))) : null;
  hooks.saveGestures = () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify({ version: 1, frames: handLog })], { type: 'application/json' }));
    const a = Object.assign(document.createElement('a'), { href: url, download: `hand-landmarks-${Date.now()}.json` });
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    status(`Saved the last ${handLog.length ? ((handLog[handLog.length - 1].t - handLog[0].t) / 1000).toFixed(0) : 0} s of hand landmarks`);
  };
  // real camera: start unlit (snap / pinch-release to ignite); synthetic: lit unless ?finger=off or a script runs
  flame.lit = !!synth && params.get('finger') !== 'off' && !params.has('script');
  hooks.setFinger = (on: boolean, why: string) => {
    if (on === flame.lit) return;
    flame.lit = on;
    if (on) {
      flame.flare = 1; flame.smoke = 0; flame.exposure = 0;
      const [sx, sy] = renderer.srcToScreen(tip[0], tip[1]);
      confetti.spark(sx, sy - tune.flameSize * 0.3);
    } else {
      flame.smoke = 1;
    }
    const t = (performance.now() - t0) / 1000;
    log.push({ t: +(simClock ? simT : t).toFixed(2), ev: `${on ? 'ignite' : 'out'} (${why})`, lit: cake?.litCount ?? 0 });
    // blowing out the cake shows "Make a wish"; don't talk over it about the finger
    if (!(why === 'blown out' && cake && cake.litCount === 0)) status(on ? `Finger lit (${why})` : `Finger out (${why})`);
  };

  // ---- scripted synthetic gestures (?synthetic&script): exercises the real detector ----
  const simClock = !!synth && params.has('script');
  const simFps = num('simfps', 30, 1, 240);
  const simUntil = num('until', 1e9);
  let simT = 0;
  function scriptPose(t: number): SynthPose {
    const r = cake ? cake.rect() : [0.3, 0.5, 0.4, 0.4];
    const wickY = r[1] + r[3] * 0.22;
    const sweep = (u: number) => 0.5 + r[2] * 0.36 * Math.cos(u * Math.PI);
    const pose: SynthPose = { tip: [sweep(t / 2), wickY], pinch: 0, snap: 0, fist: 0, shakeX: 0 };
    if (t >= 2 && t < 2.6) { pose.tip = [sweep(1), wickY - 0.06]; pose.pinch = t >= 2.2 && t < 2.4 ? 1 : 0; }
    else if (t >= 2.6 && t < 8.6) pose.tip = [sweep(1 + (t - 2.6) / 3), wickY + r[3] * 0.1 * Math.sin((t - 2.6) * 5.2)];
    else if (t >= 8.6) {
      pose.tip = [0.62, wickY - 0.12];
      if (t >= 8.8 && t < 9.8) pose.shakeX = 0.05 * Math.sin((t - 8.8) * Math.PI * 2 * 2.5);
      if (t >= 10.2 && t < 10.5) pose.snap = 1;
      if (t >= 10.5 && t < 10.7) pose.midCurl = 1;
      if (t >= 11.2 && t < 11.8) pose.fist = 1;
    }
    return pose;
  }


  // ---- fingertip filtering ----
  let tip: [number, number] = [0.5, 0.62];
  let tipVel: [number, number] = [0, 0];
  let handSize = 0.12;
  let lastSeen = -1e9;
  let lastVideoTime = -1;
  let lastDepthVideoTime = -1;
  let depthCount = 0, depthFps = 0, depthWindow = performance.now();
  let handFrames = 0, handFps = 0, handWindow = performance.now();
  let hand: HandState | null = null;

  let frames = 0, fps = 0, fpsWindow = performance.now();
  let prev = performance.now();
  const t0 = prev;
  const fixedTip = params.get('tip')?.split(',').map(Number) as [number, number] | undefined;

  function frame(now: number) {
    const realNow = now;
    if (simClock) { if (simT < simUntil) simT += 1 / simFps; now = t0 + simT * 1000; }
    const dt = Math.min(0.1, (now - prev) / 1000);
    prev = now;
    const t = (now - t0) / 1000;

    let synthTip = fixedTip;
    let pose: SynthPose | null = null;
    if (simClock) { pose = scriptPose(simT); synthTip = pose.tip; }
    else if (synth && ui.cake && cake && !fixedTip) {
      const r = (cake as CakeMode).rect();
      synthTip = [0.5 + r[2] * 0.36 * Math.sin(t * 0.55), r[1] + r[3] * (0.2 + 0.06 * Math.sin(t * 1.9))];
    }
    if (synth) synth.draw(t, synthTip);
    const w = srcW(), h = srcH();
    if (w && h) renderer.uploadFrame(source(), w, h);

    // hand tracking on each new camera frame
    let target: [number, number] | null = null;
    let newHandFrame = false;
    if (video && hands && video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      newHandFrame = true;
      hand = (hands as HandTracker).detect(video, now);
      if (hand) { target = hand.tip; handSize = handSize * 0.8 + hand.size * 0.2; lastSeen = now; }
      if (ui.cake && face && (frames & 1) === 0) faceState = (face as FaceTracker).detect(video, now + 0.5);
    } else if (synth) {
      target = synth.tip;
      lastSeen = now;
    }
    // gestures toggle the finger flame (real hand, or the scripted synthetic hand)
    const gpts = hand ? hand.points : pose ? synthLandmarks(pose) : null;
    if (video ? newHandFrame : simClock) {
      const wasLit = flame.lit;
      const ev = gestures.update(gpts, now, flame.lit, hand?.world);
      handFrames++;
      handLog.push({ t: Math.round(now), lit: wasLit, ev, p: round(gpts), w: round(hand?.world) });
      while (handLog.length && now - handLog[0].t > 20000) handLog.shift();
      if (ev === 'ignite-pinch') hooks.setFinger(true, 'pinch-release');
      else if (ev === 'ignite-snap') hooks.setFinger(true, 'snap');
      else if (ev === 'out-fist') hooks.setFinger(false, 'fist');
      else if (ev === 'out-shake') hooks.setFinger(false, 'shake');
    }
    if (target) {
      // adaptive smoothing: steady when still, responsive when moving fast
      const dx = target[0] - tip[0], dy = target[1] - tip[1];
      const speed = Math.hypot(dx, dy) / Math.max(dt, 1e-3);
      const a = Math.min(1, 0.35 + speed * 0.6);
      const nx = tip[0] + dx * a, ny = tip[1] + dy * a;
      const vx = (nx - tip[0]) / Math.max(dt, 1e-3), vy = (ny - tip[1]) / Math.max(dt, 1e-3);
      tipVel = [tipVel[0] * 0.85 + vx * 0.15, tipVel[1] * 0.85 + vy * 0.15];
      tip = [nx, ny];
    } else {
      tipVel = [tipVel[0] * 0.9, tipVel[1] * 0.9];
    }

    const visible = now - lastSeen < 350;
    const targetAmt = flame.lit && visible ? 1 : 0;
    flame.amt += (targetAmt - flame.amt) * Math.min(1, dt * (targetAmt > flame.amt ? 4 : 9));
    flame.flare = Math.max(0, flame.flare - dt * 2.5);
    flame.smoke = Math.max(0, flame.smoke - dt / 2.2);

    renderer.resize();
    const tipScreen = renderer.srcToScreen(tip[0], tip[1]);

    // ---- blowing: microphone, then face gesture, then the B key ----
    blowMic?.update(dt);
    const micWind = blowMic ? (blowMic as BlowDetector).strength : 0;
    const faceWind = faceState ? faceState.blow * 0.8 : 0;
    // mic and camera together: a blowing mouth plus a bass-heavy rise in the mic
    const bothWind = faceState && blowMic ? (blowMic as BlowDetector).looseStrength * faceState.rounded : 0;
    const keyWind = keyBlow.down ? Math.min(1, 0.35 + ((now - keyBlow.since) / 1000) * 0.8) : 0;
    const wind = demoBlow ? 0.55 : Math.max(micWind, faceWind, bothWind, keyWind);
    const windSrc = demoBlow ? 'demo' : wind <= 0.02 ? '-' : wind === micWind ? 'mic' : wind === bothWind ? 'mic+face' : wind === faceWind ? 'face' : 'key';
    let cakeOut: ReturnType<CakeMode['update']> | null = null;
    const mouthX = faceState ? renderer.srcToScreen(faceState.mouth[0], faceState.mouth[1])[0] : 0.5;
    // the finger flame feels the same wind as a candle: it bends, and a sustained blow puts it out
    {
      const aspect = window.innerWidth / Math.max(1, window.innerHeight);
      const mx = (tipScreen[0] - mouthX) * aspect;
      const fw = wind * windReach(mx);
      flame.bend += ((Math.abs(mx) > 0.01 ? Math.sign(mx) : 1) * fw - flame.bend) * Math.min(1, dt * 12);
      if (flame.lit && fw > 0.1 && !demoBlow) flame.exposure += (fw - 0.1) * dt * 5;
      else flame.exposure = Math.max(0, flame.exposure - dt * 0.6);
      if (flame.lit && flame.exposure > 0.55) hooks.setFinger(false, 'blown out');
    }
    if (ui.cake && cake) {
      cakeOut = (cake as CakeMode).update({ dt, t, tip: visible ? tip : null, tipScreen, fingerFlame: tune.flameSize * Math.min(1.4, Math.max(0.7, handSize / 0.12)), fingerLit: flame.lit && flame.amt > 0.6, wind, mouthScreenX: mouthX });
      micMeter.style.width = `${Math.round(wind * 100)}%`;
    }
    // hand outline: lets the hand pass in front of the cake, and keeps the hand from shadowing itself
    maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    if (hand && visible) drawHandMask(maskCtx, hand.points, hand.size);
    else if (synth) synth.drawMask(maskCtx);
    renderer.uploadMask(maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height));
    confetti.frame(dt);
    // with every candle (and the finger) out, let the room sink further into darkness
    const dark = ui.cake && cakeOut && cakeOut.lit === 0 && flame.amt < 0.05 ? 0.55 : 1;
    roomDim += (dark - roomDim) * Math.min(1, dt * 1.5);
    // mirror x velocity for the screen-space flame
    const vel: [number, number] = [-tipVel[0] / renderer.uvScale[0] + flame.bend * 9, tipVel[1] / renderer.uvScale[1]];
    // finger smoke wisp rides along as an extra (unlit) entry in the candle buffer
    const candleList = cakeOut ? cakeOut.candles.slice() : [];
    if (flame.smoke > 0.002) {
      candleList.push({ screen: [tipScreen[0], tipScreen[1], tune.flameSize * 0.55, 0], light: [0, 0, 1, 0], anim: [flame.bend, 0, flame.smoke, 7.3] });
    }
    // ---- depth on the GPU, in step with this frame: the model runs on each new camera frame ----
    let depthFrame: HTMLVideoElement | VideoFrame | null = null;
    let depthChanged = false;
    if (renderer.depth.hasModel) {
      if (video && video.readyState >= 2 && video.currentTime !== lastDepthVideoTime) { lastDepthVideoTime = video.currentTime; depthFrame = video; }
      else if (synth) depthFrame = new VideoFrame(synth.canvas, { timestamp: Math.round(now * 1000) });
    } else if (synth && depthNote === 'synthetic') {
      renderer.depth.writeDepth(synth.depth(renderer.depth.width, renderer.depth.height));
      depthChanged = true;
    }
    if (depthFrame || depthChanged) depthCount++;
    // the flame's light sits just in front of the finger: its depth comes from the tip, DIP and PIP joints
    const finger: FingerInput | null = target
      ? { points: gpts ? [gpts[8], gpts[7], gpts[6]] : [tip, tip, tip], dt }
      : null;
    try {
      renderer.depth.run(depthFrame, depthChanged, finger);
    } catch (e) {
      // a failure while running the model: keep going with the last depth, and say why
      console.error(e);
      renderer.depth.detach();
      depthNote = `failed: ${(e as Error).message}`;
      status(`Depth model failed: ${(e as Error).message}`, true);
    }
    if (depthFrame instanceof VideoFrame) depthFrame.close();

    const sizeScale = Math.min(1.4, Math.max(0.7, handSize / 0.12));
    renderer.render({
      tipScreen, tipSrc: tip, velocity: vel, time: t,
      flameAmt: flame.amt * (1 + flame.flare * 0.5),
      flameSize: tune.flameSize * sizeScale * (1 + flame.flare * 0.4),
      flicker: flickerAt(t),
      intensity: tune.intensity, lightRadius: tune.lightRadius, ambient: tune.ambient * roomDim, exposure: tune.exposure,
      relief: tune.relief, specular: tune.specular, shadow: tune.shadow, occlusion: tune.occlusion,
      shadows: ui.shadows, view: ui.view,
      cakeOn: !!cakeOut,
      cakeDepth: tune.cakeDepth,
      cakeRect: cakeOut?.rect,
      candles: candleList,
      shadowLight: cakeOut?.shadowLight,
    });

    // stats
    frames++;
    if (realNow - fpsWindow > 500) {
      fps = (frames * 1000) / (realNow - fpsWindow); frames = 0; fpsWindow = realNow;
      depthFps = (depthCount * 1000) / (realNow - depthWindow); depthCount = 0; depthWindow = realNow;
      handFps = (handFrames * 1000) / (realNow - handWindow); handFrames = 0; handWindow = realNow;
      if (ui.overlay && !ui.recording) {
        statsEl.textContent =
          `FPS        ${fps.toFixed(0)}\n` +
          `depth      ${depthNote}  (${depthFps.toFixed(0)}/s, ${renderer.depth.width}×${renderer.depth.height}, on the GPU)\n` +
          (gpuError ? `gpu error  ${gpuError.slice(0, 120)}\n` : '') +
          `hand       ${video ? (hands ? `${(hands as HandTracker).lastMs.toFixed(1)} ms, ${handFps.toFixed(0)} frames/s` : 'loading') : 'synthetic'}\n` +
          `flame      ${flame.lit ? (visible ? 'lit' : 'lit (no hand)') : 'out'}${gestures.lastEvent ? `  last: ${gestures.lastEvent}` : ''}\n` +
          `gesture    ${gestures.pinch ? 'pinch ' : ''}${gestures.snapContact ? 'snap-contact ' : ''}${gestures.fist ? 'fist ' : ''}${gestures.shakeCount ? `shake×${gestures.shakeCount}` : ''}\n` +
          (hand ? `thumb to   index ${gestures.rp.toFixed(2)} · middle ${gestures.rs.toFixed(2)} (touch < 0.30) · curled ${gestures.curled}\n` : '') +
          `view       ${VIEWS[ui.view]}${ui.shadows ? '' : ' (no shadows)'}` +
          (ui.cake && cake ? `\ncandles    ${(cake as CakeMode).litCount}/${(cake as CakeMode).count} lit\n` +
            `blow       ${windSrc} ${(wind * 100).toFixed(0)}%\n` +
            `mic        ${micNote}${blowMic ? ` ${(blowMic as BlowDetector).lowDb.toFixed(0)} dB` : ''}\n` +
            `face       ${video ? (face ? `${(face as FaceTracker).lastMs.toFixed(1)} ms` : 'loading') : 'synthetic'}` : '');
      }
    }
    requestAnimationFrame(frame);
  }
  status(video ? 'Loading models…' : 'Synthetic scene', !!video);
  requestAnimationFrame(frame);
}


main();
