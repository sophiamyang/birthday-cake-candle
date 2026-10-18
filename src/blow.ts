// Microphone "blow" detector. Blowing on a laptop mic produces a sustained burst of noise-like
// energy that is strongest at low frequencies (wind buffeting the capsule). We track a slow
// baseline of low-band energy and report how far above it we are, gated three ways: noise-like
// within each octave (speech and music are harmonic), bass-heavy (low band above 1-4 kHz), and
// sustained for a quarter second (typing, footsteps and consonants come in short bursts).
// Thresholds were tuned on recorded exhales, room silence, speech and everyday sounds.
const OCTAVES = [[150, 300], [300, 600], [600, 1200], [1200, 2400], [2400, 4800]] as const;

export class BlowDetector {
  strength = 0;       // 0..1, smoothed
  /** 0..1 from loudness alone (bass-heavy rise over the room), for when the face shows a blow */
  looseStrength = 0;
  lowDb = -100;
  baseDb = -70;
  flatness = 0;
  tilt = 0;            // low band minus 1-4 kHz band, dB
  private sustained = 0; // seconds the breath-like sound has lasted
  private looseSustained = 0;
  private ctx: AudioContext;
  private analyser: AnalyserNode;
  private buf: Float32Array<ArrayBuffer>;
  private resume = () => { if (this.ctx.state === 'suspended') this.ctx.resume(); };

  private constructor(ctx: AudioContext, analyser: AnalyserNode, private stream: MediaStream) {
    this.ctx = ctx;
    this.analyser = analyser;
    this.buf = new Float32Array(analyser.frequencyBinCount);
    // autoplay policy: the context may start suspended until the next user gesture
    this.resume();
    window.addEventListener('pointerdown', this.resume);
    window.addEventListener('keydown', this.resume);
  }

  static async create(): Promise<BlowDetector> {
    const stream = await navigator.mediaDevices.getUserMedia({
      // processing would remove exactly the noise we are looking for
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      video: false,
    });
    const ctx = new AudioContext();
    const src = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 8192;
    analyser.smoothingTimeConstant = 0.15;
    src.connect(analyser);
    return new BlowDetector(ctx, analyser, stream);
  }

  get running() { return this.ctx.state === 'running'; }
  /** False once the mic track has ended (device unplugged, permission revoked). */
  get live() { return this.ctx.state !== 'closed' && this.stream.getAudioTracks().some((t) => t.readyState === 'live'); }

  /** Release the microphone and the audio graph. */
  dispose() {
    window.removeEventListener('pointerdown', this.resume);
    window.removeEventListener('keydown', this.resume);
    this.stream.getTracks().forEach((t) => t.stop());
    this.ctx.close();
  }

  update(dt: number) {
    this.analyser.getFloatFrequencyData(this.buf);
    const binHz = this.ctx.sampleRate / this.analyser.fftSize;
    const band = (lo: number, hi: number) => {
      let p = 0, c = 0;
      for (let i = Math.max(1, Math.floor(lo / binHz)); i <= Math.min(this.buf.length - 1, Math.ceil(hi / binHz)); i++) {
        p += Math.pow(10, this.buf[i] / 10); c++;
      }
      return c ? p / c : 1e-12;
    };
    const low = band(20, 400);
    this.lowDb = 10 * Math.log10(low + 1e-12);
    this.tilt = this.lowDb - 10 * Math.log10(band(1000, 4000) + 1e-12);
    // spectral flatness inside each octave from 150 Hz to 4.8 kHz, averaged (1 = noise, ~0 = a
    // few sharp peaks). Breath is strongest at low frequencies, which drags a single wide-band
    // flatness down; within an octave it is still noise, while voices and music show harmonics.
    let flat = 0;
    for (const [lo, hi] of OCTAVES) {
      let logSum = 0, sum = 0, n = 0;
      for (let i = Math.floor(lo / binHz); i < Math.ceil(hi / binHz); i++) {
        const p = Math.pow(10, this.buf[i] / 10) + 1e-12;
        logSum += Math.log(p); sum += p; n++;
      }
      flat += Math.exp(logSum / n) / (sum / n);
    }
    this.flatness = flat / OCTAVES.length;

    const rise = this.lowDb - this.baseDb;
    const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
    const breathy = clamp01((rise - 4) / 6) * clamp01((this.flatness - 0.4) / 0.15) * (this.tilt > 5 ? 1 : 0);
    this.sustained = breathy > 0.15 ? this.sustained + dt : 0;
    const raw = this.sustained >= 0.24 ? breathy : 0;
    // with the camera seeing a blowing mouth, the checks relax: a smaller bass-heavy rise, only
    // noise-like, but less strictly, and a shorter sustain
    const loose = clamp01((rise - 3) / 6) * clamp01((this.flatness - 0.35) / 0.12) * (this.tilt > 0 ? 1 : 0);
    this.looseSustained = loose > 0.15 ? this.looseSustained + dt : 0;
    const looseRaw = this.looseSustained >= 0.15 ? loose : 0;
    this.looseStrength += (looseRaw - this.looseStrength) * Math.min(1, dt * (looseRaw > this.looseStrength ? 20 : 6));
    // baseline follows the room slowly, and not while something breath-like is going on
    if (breathy < 0.05) this.baseDb += (this.lowDb - this.baseDb) * Math.min(1, dt / 2.5);
    const k = raw > this.strength ? Math.min(1, dt * 20) : Math.min(1, dt * 6);
    this.strength += (raw - this.strength) * k;
  }
}
