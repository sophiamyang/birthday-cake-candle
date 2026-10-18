import { beforeAll, describe, expect, it } from 'vitest';
import { BlowDetector } from './blow';

// Drives the real detector with synthetic spectra (dB per bin), the way an AnalyserNode
// reports them, at 60 frames a second.
const SR = 48000, FFT = 8192, BINS = FFT / 2, BIN_HZ = SR / FFT, FPS = 60;
type Spectrum = (hz: number) => number;

let seed = 1;
const jitter = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return (seed / 4294967296) * 10 - 5; }; // ±5 dB, noise-like

/** room hiss: quiet and flat */
const room: Spectrum = () => -95 + jitter();
/** a blow: loud, noisy, falling off with frequency (wind on the capsule) */
const blow: Spectrum = (hz) => -45 - 25 * Math.log10(Math.max(hz, 40) / 40) + jitter();
/** a voiced sound at 150 Hz: sharp harmonics over a quiet floor */
const voice: Spectrum = (hz) => {
  const nearest = Math.round(hz / 150) * 150;
  return Math.abs(hz - nearest) < BIN_HZ * 1.5 && nearest > 0 ? -35 - 10 * Math.log10(nearest / 150) : -95 + jitter();
};

function detector(schedule: (t: number) => Spectrum) {
  let t = 0;
  const analyser = {
    fftSize: FFT,
    frequencyBinCount: BINS,
    getFloatFrequencyData(buf: Float32Array) { const s = schedule(t); for (let i = 0; i < BINS; i++) buf[i] = s(i * BIN_HZ); },
  };
  const bd = new (BlowDetector as unknown as new (...a: unknown[]) => BlowDetector)(
    { sampleRate: SR, state: 'running', resume() {} }, analyser, { getAudioTracks: () => [] });
  return {
    /** runs until `until` seconds and returns the peak strength seen after `from` */
    peak(from: number, until: number) {
      let peak = 0;
      for (; t < until; t += 1 / FPS) { bd.update(1 / FPS); if (t >= from) peak = Math.max(peak, bd.strength); }
      return peak;
    },
  };
}

beforeAll(() => {
  (globalThis as { window?: unknown }).window ??= { addEventListener() {}, removeEventListener() {} };
});

describe('BlowDetector', () => {
  it('stays quiet in a quiet room', () => {
    expect(detector(() => room).peak(0, 4)).toBe(0);
  });

  it('hears a sustained blow', () => {
    expect(detector((t) => (t >= 3 && t < 4 ? blow : room)).peak(3, 4.2)).toBeGreaterThan(0.6);
  });

  it('ignores short bursts such as typing or footsteps', () => {
    // 0.1 s of blow-like noise every 0.3 s
    expect(detector((t) => (t >= 3 && (t - 3) % 0.3 < 0.1 ? blow : room)).peak(3, 6)).toBe(0);
  });

  it('ignores a loud voiced sound', () => {
    expect(detector((t) => (t >= 3 ? voice : room)).peak(3, 5)).toBe(0);
  });
});
