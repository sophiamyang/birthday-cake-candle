import { describe, expect, it } from 'vitest';
import { GestureDetector, type GestureEvent } from './gestures';
import { synthLandmarks, type SynthPose } from './synthetic';

const FPS = 30;
const rest = (): SynthPose => ({ tip: [0.5, 0.4], pinch: 0, snap: 0, fist: 0, shakeX: 0 });

/** Feed `poseAt(t)` (t in seconds, null = no hand) to a detector at 30 fps; returns every event fired. */
function run(poseAt: (t: number) => SynthPose | null, seconds: number, lit: boolean, g = new GestureDetector(), startMs = 1000) {
  const events: { t: number; ev: GestureEvent }[] = [];
  for (let f = 0; f <= seconds * FPS; f++) {
    const t = f / FPS;
    const pose = poseAt(t);
    const ev = g.update(pose ? synthLandmarks(pose) : null, startMs + t * 1000, lit);
    if (ev) events.push({ t, ev });
  }
  return events.map((e) => e.ev);
}

describe('GestureDetector', () => {
  it('ignites on a quick pinch-and-release', () => {
    const ev = run((t) => ({ ...rest(), pinch: t >= 0.5 && t < 0.7 ? 1 : 0 }), 1.2, false);
    expect(ev).toEqual(['ignite-pinch']);
  });

  it('ignites when the fingertip points stay a little apart, as real fingertips do', () => {
    // thumb travels 70% of the way: the landmarks end about a quarter palm length apart
    const ev = run((t) => ({ ...rest(), pinch: t >= 0.5 && t < 0.7 ? 0.7 : 0 }), 1.2, false);
    expect(ev).toEqual(['ignite-pinch']);
  });

  it('ignites on an unhurried pinch of about 0.7 s', () => {
    const ev = run((t) => ({ ...rest(), pinch: t >= 0.5 && t < 1.2 ? 1 : 0 }), 1.8, false);
    expect(ev).toEqual(['ignite-pinch']);
  });

  it('ignores a pinch held for more than 2 s', () => {
    const ev = run((t) => ({ ...rest(), pinch: t >= 0.5 && t < 2.7 ? 1 : 0 }), 3.2, false);
    expect(ev).toEqual([]);
  });

  it('ignites on a small flick that only opens the fingers a little', () => {
    const g = new GestureDetector();
    const ev = run((t) => ({ ...rest(), pinch: t >= 0.5 && t < 0.7 ? 1 : t >= 0.7 ? 0.54 : 0 }), 1.2, false, g);
    // the thumb ends between the new release point and the old one
    expect(g.rp).toBeGreaterThan(0.36);
    expect(g.rp).toBeLessThan(0.45);
    expect(ev).toEqual(['ignite-pinch']);
  });

  it('ignites on a snap that was set up slowly', () => {
    const ev = run((t) => ({ ...rest(), snap: t >= 0.5 && t < 2.1 ? 1 : 0, midCurl: t >= 2.1 && t < 2.3 ? 1 : 0 }), 2.6, false);
    expect(ev).toEqual(['ignite-snap']);
  });

  it('ignites when the snap blurs the hand out of tracking for a few frames', () => {
    // contact, then 3 frames with no hand, then the hand comes back open
    const ev = run((t) => (t >= 0.8 && t < 0.9 ? null : { ...rest(), snap: t >= 0.5 && t < 0.8 ? 1 : 0, midCurl: t >= 0.9 && t < 1.1 ? 1 : 0 }), 1.5, false);
    expect(ev).toEqual(['ignite-snap']);
  });

  it('forgets a touch when the hand is gone for long', () => {
    const ev = run((t) => (t >= 0.8 && t < 1.6 ? null : { ...rest(), pinch: t >= 0.5 && t < 0.8 ? 1 : 0 }), 2, false);
    expect(ev).toEqual([]);
  });

  it('keeps the flame a snap just lit while the fingers stay curled', () => {
    const g = new GestureDetector();
    const snapHand = (t: number): SynthPose => ({ ...rest(), tuck: 1, indexCurl: 0.8, snap: t >= 0.5 && t < 0.8 ? 1 : 0, midCurl: t >= 0.8 ? 1 : 0.6 });
    expect(run(snapHand, 1, false, g)).toEqual(['ignite-snap']);
    // the fingers fold the rest of the way only after the refractory period: still no fist
    expect(run(() => ({ ...rest(), fist: 1 }), 2, true, g, 2700)).toEqual([]);
    // opening the hand re-arms the fist
    expect(run((t) => ({ ...rest(), fist: t >= 0.3 ? 1 : 0 }), 1, true, g, 4800)).toEqual(['out-fist']);
  });

  it('does not ignite a flame that is already lit', () => {
    const ev = run((t) => ({ ...rest(), pinch: t >= 0.5 && t < 0.7 ? 1 : 0 }), 1.2, true);
    expect(ev).toEqual([]);
  });

  it('ignites on a snap (thumb + middle contact, then a fast release)', () => {
    const ev = run((t) => ({ ...rest(), snap: t >= 0.5 && t < 0.8 ? 1 : 0, midCurl: t >= 0.8 && t < 1.0 ? 1 : 0 }), 1.5, false);
    expect(ev).toEqual(['ignite-snap']);
  });

  it('reads a real snap hand as a snap, not a fist', () => {
    // ring and little fingers tucked, index bent, middle finger bent onto the thumb, then the
    // middle finger whips down: every fingertip is nearer the wrist than its middle knuckle
    const hand = (t: number): SynthPose => ({ ...rest(), tuck: 1, indexCurl: 0.8, snap: t >= 0.5 && t < 0.8 ? 1 : 0, midCurl: t >= 0.5 && t < 0.8 ? 0.6 : t >= 0.8 && t < 1.0 ? 1 : 0 });
    expect(run(hand, 1.5, false)).toEqual(['ignite-snap']);
    // and with the finger already lit, the same hand must not put it out
    expect(run(hand, 1.5, true)).toEqual([]);
  });

  it('reads a pinch hand with flattened depth as a pinch, not a fist', () => {
    // fingers toward the camera: the bent index and middle tips measure about as near the wrist
    // as their knuckles, the ring and little fingers are tucked, and the thumb is still on its way
    const hand = (t: number): SynthPose => ({ ...rest(), tuck: 1, indexCurl: 0.78, midCurl: 0.78, pinch: t >= 0.5 && t < 0.7 ? 1 : 0.2 });
    const g = new GestureDetector();
    expect(run(hand, 1.2, false, g)).toEqual(['ignite-pinch']);
    expect(g.curled).toBeLessThan(4);
    // and the same hand doesn't put a lit flame out
    expect(run(hand, 1.5, true)).toEqual([]);
  });

  it('puts the flame out once per held fist', () => {
    const ev = run((t) => ({ ...rest(), fist: t >= 0.5 && t < 2 ? 1 : 0 }), 2.5, true);
    expect(ev).toEqual(['out-fist']);
  });

  it('needs the fist held for about a quarter second', () => {
    const ev = run((t) => ({ ...rest(), fist: t >= 0.5 && t < 0.65 ? 1 : 0 }), 1.5, true);
    expect(ev).toEqual([]);
  });

  it('puts the flame out on a shake', () => {
    const ev = run((t) => ({ ...rest(), shakeX: t >= 0.5 && t < 1.5 ? 0.05 * Math.sin((t - 0.5) * Math.PI * 5) : 0 }), 2, true);
    expect(ev).toEqual(['out-shake']);
  });

  it('ignores a gesture inside the 800 ms refractory period', () => {
    const g = new GestureDetector();
    expect(run((t) => ({ ...rest(), pinch: t >= 0.5 && t < 0.7 ? 1 : 0 }), 0.8, false, g)).toEqual(['ignite-pinch']);
    // the fist starts right after ignition, so it completes inside the refractory window
    expect(run((t) => ({ ...rest(), fist: t < 1 ? 1 : 0 }), 1.2, true, g)).toEqual([]);
  });

  it('forgets a half-finished shake when the hand leaves the frame', () => {
    const g = new GestureDetector();
    // two reversals, not enough to fire
    run((t) => ({ ...rest(), shakeX: 0.05 * Math.sin(t * Math.PI * 5) }), 0.3, true, g);
    expect(g.shakeCount).toBeGreaterThan(0);
    g.update(null, 5000, true);
    expect(g.shakeCount).toBe(0);
    expect(g.fist).toBe(false);
  });
});
