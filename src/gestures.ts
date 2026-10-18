// Robust on/off gestures for the finger flame, from MediaPipe hand landmarks.
//   ignite:     a tap of the thumb on the index finger (pinch) or on the middle finger (snap):
//               touch for 40 ms to 2 s, then open
//   extinguish: a fist held ~0.25 s, or a quick shake (>= 3 fast direction reversals in 0.8 s)
// Touch and curl tests use MediaPipe's 3D world landmarks when given (metres, the same in every
// direction); the shake uses image positions, since world landmarks are centred on the hand.
// Landmarks sit in the middle of each fingertip pad, so touching fingertips still measure about a
// quarter of a palm length apart.
// Hysteresis on every contact/curl test and an 800 ms refractory period after any toggle keep
// the flame from flickering on and off. A snap's whip often blurs the hand out of a frame or two,
// so a contact survives a brief loss of the hand and resolves when the hand comes back.

export type GestureEvent = 'ignite-pinch' | 'ignite-snap' | 'out-fist' | 'out-shake' | null;

type Pt = [number, number];
type Pt3 = readonly number[];
const dist = (a: Pt3, b: Pt3) => Math.hypot(a[0] - b[0], a[1] - b[1], (a[2] ?? 0) - (b[2] ?? 0));

const TOUCH = 0.3;     // thumb-to-fingertip distance (palm lengths) that counts as touching
const RELEASE = 0.36;  // open again: at least this far apart...
const RELEASE_GAP = 0.12; // ...and this much farther than at the closest point of the touch
const TAP_MIN = 40, TAP_MAX = 2000; // ms a touch may last to count as a tap (a snap is often set up slowly)
const LOST_GRACE = 500; // ms a touch survives the hand dropping out of tracking
const CURLED = 0.9;     // a fingertip nearer the wrist than this times its base knuckle is curled

export class GestureDetector {
  // live state (exposed for the overlay)
  pinch = false;
  snapContact = false;
  fist = false;
  shakeCount = 0;
  lastEvent = '';
  /** thumb-to-index and thumb-to-middle distances (palm lengths) and curled fingers, last frame */
  rp = 0;
  rs = 0;
  curled = 0;

  private pinchSince = 0;
  private snapSince = 0;
  private fistSince = 0;
  private fistFired = false;
  // after an ignite, a fist counts only once the hand has opened: the curled fingers of a snap
  // must not put out the flame they just lit
  private fistArmed = true;
  private contactMin = 0;
  private lost: { kind: 'pinch' | 'snap'; since: number; at: number; min: number } | null = null;
  private refractoryUntil = 0;
  private prevPalm: Pt | null = null;
  private prevT = 0;
  private vx = 0;
  private reversals: number[] = [];
  private lastSign = 0;

  /**
   * p = 21 landmarks in image uv, world = the same in metres (optional); lit = current flame
   * state. Returns an event only when the flame should change.
   */
  update(p: Pt[] | null, now: number, lit: boolean, world?: Pt3[] | null): GestureEvent {
    if (!p) {
      // hand lost: forget half-finished gestures, so a hand that comes back half-curled or on the
      // other side of the frame can't complete a fist or a shake reversal it never made. A touch
      // is kept for a moment, since fast fingers blur the hand out of tracking
      if (!this.lost && (this.pinch || this.snapContact)) {
        this.lost = { kind: this.pinch ? 'pinch' : 'snap', since: this.pinch ? this.pinchSince : this.snapSince, at: now, min: this.contactMin };
      }
      this.prevPalm = null;
      this.pinch = this.snapContact = this.fist = false;
      this.vx = 0; this.lastSign = 0; this.reversals = []; this.shakeCount = 0;
      return null;
    }
    const size = Math.max(1e-3, dist(p[0], p[9]));
    const w: Pt3[] = world ?? p;
    const wsize = Math.max(1e-6, dist(w[0], w[9]));
    const dt = Math.max(1e-3, (now - this.prevT) / 1000);
    this.prevT = now;
    let ev: GestureEvent = null;
    const ready = now >= this.refractoryUntil;

    // the index and middle tips sit close together, so a touch belongs to whichever is nearer
    const rp = dist(w[4], w[8]) / wsize, rs = dist(w[4], w[12]) / wsize;
    this.rp = rp; this.rs = rs;
    const thumbTouching = rp < TOUCH || rs < TOUCH;
    const released = (r: number) => r > Math.max(RELEASE, this.contactMin + RELEASE_GAP);
    const tap = (held: number) => !lit && ready && !this.fist && held >= TAP_MIN && held <= TAP_MAX;

    // the hand is back after a brief loss: an open hand finishes the tap, a touching one resumes it
    if (this.lost) {
      const l = this.lost;
      this.lost = null;
      if (now - l.at <= LOST_GRACE) {
        const r = l.kind === 'pinch' ? rp : rs;
        this.contactMin = l.min;
        if (released(r)) {
          if (tap(l.at - l.since)) ev = l.kind === 'pinch' ? 'ignite-pinch' : 'ignite-snap';
        } else if (l.kind === 'pinch') { this.pinch = true; this.pinchSince = l.since; }
        else { this.snapContact = true; this.snapSince = l.since; }
      }
    }

    // ---- curl / fist (enter at 4 curled fingers, leave at <= 2) ----
    // Curled means folded into the palm: the tip well inside the base knuckle, as a fist puts it
    // (about 0.75-0.8 of the knuckle's distance from the wrist). Merely bent fingers don't count:
    // with the fingers pointing at the camera MediaPipe flattens their depth, so a finger bent onto
    // the thumb in a pinch or snap can measure as near the wrist as its knuckle. A hand whose
    // thumb is on a fingertip is tapping, not making a fist.
    const curled = [[8, 5], [12, 9], [16, 13], [20, 17]].filter(([t, k]) => dist(w[t], w[0]) < dist(w[k], w[0]) * CURLED).length;
    this.curled = curled;
    if (curled <= 2) this.fistArmed = true;
    if (!this.fist && this.fistArmed && curled >= 4 && !thumbTouching && !this.pinch && !this.snapContact) { this.fist = true; this.fistSince = now; this.fistFired = false; }
    else if (this.fist && curled <= 2) { this.fist = false; }
    if (this.fist && !this.fistFired && now - this.fistSince > 250) {
      this.fistFired = true;
      if (lit && ready) ev = 'out-fist';
    }

    // ---- quick pinch-and-release (thumb tip + index tip), with hysteresis ----
    if (!this.pinch && !this.snapContact && rp < TOUCH && rp <= rs && !this.fist) { this.pinch = true; this.pinchSince = now; this.contactMin = rp; }
    else if (this.pinch && released(rp)) {
      this.pinch = false;
      if (!ev && tap(now - this.pinchSince)) ev = 'ignite-pinch';
    }
    if (this.pinch) this.contactMin = Math.min(this.contactMin, rp);

    // ---- snap: thumb + middle fingertip touch, then release. The middle finger's whip is often a
    // single camera frame, so any quick touch-and-release counts; no speed is required ----
    if (!this.snapContact && !this.pinch && rs < TOUCH && rs < rp && !this.fist) { this.snapContact = true; this.snapSince = now; this.contactMin = rs; }
    else if (this.snapContact && released(rs)) {
      this.snapContact = false;
      if (!ev && tap(now - this.snapSince)) ev = 'ignite-snap';
    }
    if (this.snapContact) this.contactMin = Math.min(this.contactMin, rs);

    // ---- shake: fast horizontal direction reversals of the palm ----
    const palm: Pt = [(p[0][0] + p[5][0] + p[9][0] + p[13][0] + p[17][0]) / 5, (p[0][1] + p[5][1] + p[9][1] + p[13][1] + p[17][1]) / 5];
    if (this.prevPalm) {
      const v = (palm[0] - this.prevPalm[0]) / size / dt;
      this.vx = this.vx * 0.4 + v * 0.6;
      const sign = Math.abs(this.vx) > 2.2 ? Math.sign(this.vx) : 0;
      if (sign && this.lastSign && sign !== this.lastSign) this.reversals.push(now);
      if (sign) this.lastSign = sign;
    }
    this.prevPalm = palm;
    this.reversals = this.reversals.filter((t) => now - t < 800);
    this.shakeCount = this.reversals.length;
    if (!ev && lit && ready && this.reversals.length >= 3) ev = 'out-shake';

    if (ev) {
      this.refractoryUntil = now + 800;
      this.reversals = [];
      this.lastEvent = ev;
      if (ev === 'ignite-pinch' || ev === 'ignite-snap') this.fistArmed = false;
    }
    return ev;
  }
}
