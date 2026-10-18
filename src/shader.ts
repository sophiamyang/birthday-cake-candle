// WGSL for the single full-screen relighting + flame pass.
// `Uniforms` and `Candle` are TypeGPU structs; tgpu.resolve() injects their WGSL definitions.
// The lighting follows TypeGPU's "Monocular Light Injection" example (MIT): positions live in a
// normalised space (x, y in frame-height units with y down, z toward the camera, depth 0..1 ->
// z -0.7..0), lights fade gently as 1 / (1 + (d / radius)^2), normals come from the smoothed
// depth slopes in the surface texture, and shadows are ray-marched through the depth.
export const shaderTemplate = /* wgsl */ `
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var camSampler: sampler;
@group(0) @binding(2) var camTex: texture_2d<f32>;
@group(0) @binding(3) var surfaceTex: texture_2d<f32>; // slope xy, occlusion, depth (1 = nearest)
@group(0) @binding(4) var<storage, read> candles: array<Candle, 64>;
@group(0) @binding(5) var cakeColTex: texture_2d<f32>;
@group(0) @binding(6) var cakeGeoTex: texture_2d<f32>;
@group(0) @binding(7) var maskTex: texture_2d<f32>;
@group(0) @binding(8) var<storage, read> finger: vec4f; // x = depth of the finger holding the flame

const SURFACE_FAR_Z: f32 = -0.7;
const SHADOW_FAR_Z: f32 = -1.25;
const LIGHT_CLEARANCE: f32 = 0.04;
const LIGHT_WRAP: f32 = 0.25;
const RELIEF_SCALE: f32 = 200.0;
const SLOPE_COMPRESSION: f32 = 0.55;
const SPECULAR_POWER: f32 = 36.0;
const SPECULAR_F0: f32 = 0.06;
const WHITE_POINT: f32 = 2.6;
const HIGHLIGHT_BLEACH: f32 = 2.0;
const AMBIENT_FILL = vec3f(0.78, 0.86, 1.0);
const SHADOW_STEPS: i32 = 32;
const SHADOW_SPAN: f32 = 0.3;
const SHADOW_BASELINE: f32 = 0.005;
const SHADOW_BIAS: f32 = 0.014;
const SHADOW_SLOPE_BIAS: f32 = 0.02;
const SHADOW_THICKNESS: f32 = 0.7;
const SHADOW_THICKNESS_GROWTH: f32 = 2.6;
const SHADOW_SOFTNESS: f32 = 0.089;
const SHADOW_GAIN: f32 = 2.5;
const SHADOW_FRONT_FADE: f32 = 0.2; // occluders this far in front of the light stop casting

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
};

@vertex fn vs(@builtin(vertex_index) i: u32) -> VOut {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VOut;
  o.pos = vec4f(p[i], 0.0, 1.0);
  o.uv = vec2f(p[i].x * 0.5 + 0.5, 0.5 - p[i].y * 0.5);
  return o;
}


// ---------- depth (source-image uv space, un-mirrored) ----------
fn surfaceAt(uv: vec2f) -> vec4f { return textureSampleLevel(surfaceTex, camSampler, uv, 0.0); }
fn depthAt(uv: vec2f) -> f32 { return surfaceAt(uv).w; }
fn surfaceZ(d: f32) -> f32 { return mix(SURFACE_FAR_Z, 0.0, d); }
fn shadowZ(d: f32) -> f32 { return mix(SHADOW_FAR_Z, 0.0, d); }
// source uv <-> position plane (aspect-corrected, so distances are the same in x and y)
fn toPlane(uv: vec2f) -> vec2f { return (uv - 0.5) * vec2f(u.videoAspect, 1.0); }
fn toUv(p: vec2f) -> vec2f { return p / vec2f(u.videoAspect, 1.0) + 0.5; }

// ---------- birthday cake (sprite in source-uv rect) ----------
// returns (coverage, cake depth); coverage is 0 where a hand is in front of the cake
fn cakeHit(uv: vec2f, sceneD: f32) -> vec2f {
  if (u.cakeOn < 0.5) { return vec2f(0.0); }
  let c = (uv - u.cakeRect.xy) / u.cakeRect.zw;
  if (any(c < vec2f(0.0)) || any(c > vec2f(1.0))) { return vec2f(0.0); }
  let a = textureSampleLevel(cakeColTex, camSampler, c, 0.0).a;
  if (a < 0.003) { return vec2f(0.0); }
  let gd = textureSampleLevel(cakeGeoTex, camSampler, c, 0.0).r;
  let m = textureSampleLevel(maskTex, camSampler, uv, 0.0).r;
  let handFront = smoothstep(0.3, 0.7, m) * smoothstep(gd - 0.02, gd + 0.01, sceneD);
  return vec2f(a * (1.0 - handFront), gd);
}

// depth of whatever is visible: the camera depth, or the cake where it covers it
fn sceneDepth(uv: vec2f) -> f32 {
  let d = depthAt(uv);
  let ck = cakeHit(uv, d);
  return select(d, ck.y, ck.x > 0.5);
}

// ---------- noise ----------
fn hash21(p: vec2f) -> f32 {
  var q = fract(p * vec2f(123.34, 456.21));
  q += dot(q, q + 45.32);
  return fract(q.x * q.y);
}
fn vnoise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = p - i;
  let s = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash21(i), hash21(i + vec2f(1.0, 0.0)), s.x),
             mix(hash21(i + vec2f(0.0, 1.0)), hash21(i + vec2f(1.0, 1.0)), s.x), s.y);
}
fn fbm(p0: vec2f) -> f32 {
  var p = p0;
  var a = 0.5;
  var s = 0.0;
  for (var k = 0; k < 4; k++) {
    s += a * vnoise(p);
    p = p * 2.03 + vec2f(1.7, 9.2);
    a *= 0.5;
  }
  return s;
}
fn ign(fc: vec2f) -> f32 {
  return fract(52.9829189 * fract(dot(fc, vec2f(0.06711056, 0.00583715))));
}


// ---------- soft shadows ray-marched through the depth ----------
fn shadowFactor(origin: vec3f, dir: vec3f, reach: f32, jitter: f32, lightZ: f32) -> f32 {
  let stride = reach / f32(SHADOW_STEPS);
  let baselineTravel = reach * (SHADOW_BASELINE / SHADOW_SPAN);
  let trail = origin - dir * baselineTravel;
  let receiverRise = max(origin.z - shadowZ(sceneDepth(toUv(trail.xy))) - baselineTravel * dir.z, 0.0);
  let risePerTravel = receiverRise / baselineTravel;
  var occlusion = 0.0;
  for (var i = 0; i < SHADOW_STEPS; i++) {
    let travel = (f32(i) + jitter) * stride;
    let probe = origin + dir * travel;
    let sampleZ = shadowZ(sceneDepth(toUv(probe.xy)));
    let difference = sampleZ - probe.z;
    let bias = SHADOW_BIAS + travel * (SHADOW_SLOPE_BIAS + risePerTravel);
    let thickness = SHADOW_THICKNESS * (1.0 + (travel / SHADOW_SPAN) * SHADOW_THICKNESS_GROWTH);
    if (difference > bias && difference < thickness) {
      let behindLight = 1.0 - saturate((sampleZ - lightZ) / SHADOW_FRONT_FADE);
      occlusion += saturate((difference - bias) / SHADOW_SOFTNESS) * behindLight;
    }
  }
  return 1.0 - saturate(occlusion / f32(SHADOW_STEPS) * SHADOW_GAIN);
}

fn shadowToward(plane: vec2f, d: f32, light: vec3f, jitter: f32) -> f32 {
  let origin = vec3f(plane, shadowZ(d));
  let toLight = light - origin;
  let dist = max(length(toLight), 0.0001);
  let reach = dist * (SHADOW_SPAN / max(length(toLight.xy), SHADOW_SPAN));
  return shadowFactor(origin, toLight / dist, reach, jitter, light.z);
}

// ---------- candle flame (screen space, y up, units = flame height) ----------
fn candle(q0: vec2f, t: f32) -> vec4f {
  return flameShape(q0, t, u.velocity.x * 0.10, 0.0);
}

// bend: lateral lean of the tip (screen units), stretch: 0 = calm, 1 = blown long and thin
fn flameShape(q1: vec2f, t: f32, bend: f32, stretch: f32) -> vec4f {
  var q = q1;
  q.y = q.y / (1.0 + stretch * 0.9);
  q.x = q.x * (1.0 + stretch * 0.8);
  let y = q.y;
  q.x += bend * max(y, 0.0) * max(y, 0.0) * (1.0 + stretch);
  q.x += (fbm(vec2f(t * 1.3, 2.0)) - 0.5) * 0.10 * max(y, 0.0) * (1.0 + stretch * 3.0);
  let n = fbm(vec2f(q.x * 6.0, y * 4.0 - t * (5.0 + stretch * 8.0)));
  q.x += (n - 0.5) * 0.06 * smoothstep(0.1, 1.0, y);

  // teardrop: round bottom, tapering pointed top
  let p = vec2f(q.x, y - 0.28);
  let up = max(p.y, 0.0);
  let taper = 1.0 + up * 2.0;
  let e = length(vec2f(p.x * taper / 0.17, p.y / select(0.22, 0.66, p.y > 0.0)));
  let body = 1.0 - smoothstep(0.62, 1.0, e + (n - 0.5) * 0.25 * smoothstep(0.0, 1.0, y));
  let core = 1.0 - smoothstep(0.05, 0.62, e * 1.15);

  var col = mix(vec3f(1.0, 0.38, 0.06), vec3f(1.0, 0.68, 0.22), body);
  col = mix(col, vec3f(1.0, 0.93, 0.72), core);
  // faint blue base where the flame meets the "wick"
  let blue = smoothstep(0.22, 0.02, y) * smoothstep(-0.05, 0.08, y) * (1.0 - core) * body;
  col = mix(col, vec3f(0.22, 0.32, 0.95), blue * 0.7);
  let a = body * (1.0 - blue * 0.5);
  return vec4f(col * a * 1.35, a);
}

// a thin wisp of smoke rising from a snuffed wick (q in flame units, y up)
fn smokeWisp(q: vec2f, t: f32, seed: f32, bend: f32, amt: f32) -> f32 {
  let y = q.y;
  if (y < -0.1 || y > 7.0) { return 0.0; }
  let x = q.x - sin(y * 1.1 - t * 1.4 + seed) * 0.12 * y - bend * y * 0.5;
  let w = 0.05 + y * 0.07;
  let n = fbm(vec2f(x * 3.0 + seed, y * 1.3 - t * 1.6));
  return exp(-x * x / (w * w)) * smoothstep(-0.1, 0.4, y) * (1.0 - y / 7.0) * n * 1.4 * amt;
}


// one point light: rgb = diffuse weight (times albedo later), a = specular weight
fn lightTerm(N: vec3f, P: vec3f, light: vec3f) -> vec2f {
  let toLight = light - P;
  let dist = max(length(toLight), 0.0001);
  let Ld = toLight / dist;
  let spread = dist / u.lightRadius;
  let falloff = 1.0 / (1.0 + spread * spread);
  let wrapped = saturate((dot(N, Ld) + LIGHT_WRAP) / (1.0 + LIGHT_WRAP));
  let H = normalize(Ld + vec3f(0.0, 0.0, 1.0));
  let lobe = pow(saturate(dot(N, H)), SPECULAR_POWER);
  let grazing = pow(1.0 - saturate(N.z), 5.0);
  return vec2f(wrapped * wrapped, lobe * (SPECULAR_F0 + (1.0 - SPECULAR_F0) * grazing)) * falloff;
}

fn toLinear(c: vec3f) -> vec3f { return pow(c, vec3f(2.2)); }
fn toSrgb(c: vec3f) -> vec3f { return pow(max(c, vec3f(0.0)), vec3f(1.0 / 2.2)); }

fn compress(v: f32) -> f32 { return (v * (v / (WHITE_POINT * WHITE_POINT) + 1.0)) / (v + 1.0); }
fn tonemap(color: vec3f) -> vec3f {
  let luminance = max(dot(color, vec3f(0.2126, 0.7152, 0.0722)), 0.0001);
  let mapped = compress(luminance);
  let shoulder = color / vec3f(WHITE_POINT * WHITE_POINT) + vec3f(1.0);
  let perChannel = (color * shoulder) / (color + vec3f(1.0));
  let bleach = pow(saturate(mapped), HIGHLIGHT_BLEACH);
  return saturate(mix(color * (mapped / luminance), perChannel, bleach));
}

fn depthRamp(v: f32) -> vec3f {
  let cold = vec3f(0.03, 0.02, 0.12); let middle = vec3f(0.11, 0.45, 0.94);
  let warm = vec3f(0.85, 0.36, 0.96); let hot = vec3f(0.97, 0.97, 0.87);
  if (v < 0.4) { return mix(cold, middle, v / 0.4); }
  if (v < 0.75) { return mix(middle, warm, (v - 0.4) / 0.35); }
  return mix(warm, hot, (v - 0.75) / 0.25);
}

// finger flame + tiny halo, drawn in screen space (aspect-correct, y up); vis hides it behind the cake
fn fingerFlame(col0: vec3f, s: vec2f, aspect: f32, vis: f32) -> vec3f {
  var col = col0;
  let q = vec2f((s.x - u.tipScreen.x) * aspect, u.tipScreen.y - s.y) / u.flameSize;
  let hq = q - vec2f(0.0, 0.32);
  let halo = exp(-dot(hq, hq) * 1.6) * 0.07 + exp(-dot(hq, hq) * 12.0) * 0.10;
  col += vec3f(1.0, 0.55, 0.22) * halo * u.flameAmt * u.flicker * vis;
  if (u.flameAmt > 0.001 && abs(q.x) < 0.6 && q.y > -0.2 && q.y < 1.3) {
    let f = candle(q, u.time);
    let a = f.a * u.flameAmt * vis;
    col = col * (1.0 - a * 0.35) + f.rgb * u.flameAmt * vis;
  }
  return col;
}

@fragment fn fs(in: VOut) -> @location(0) vec4f {
  let s = in.uv;
  var src = 0.5 + (s - 0.5) * u.uvScale;
  src.x = 1.0 - src.x; // selfie mirror
  let camSrgb = textureSampleLevel(camTex, camSampler, src, 0.0).rgb;
  var albedo = toLinear(camSrgb);

  let surface = surfaceAt(src);
  var d = surface.w;
  // the hand holding the flame: depth on thin fingers is rough, so its shape is smoothed, it gets
  // little ambient occlusion, and it isn't shadowed from its own flame (it still casts shadows)
  let handOutline = smoothstep(0.3, 0.7, textureSampleLevel(maskTex, camSampler, src, 0.0).r);
  // the outline is drawn a little wide: only pixels near the finger's depth are really the hand
  let handHere = handOutline * (1.0 - smoothstep(0.1, 0.25, abs(d - finger.x)));
  // smoothed depth slopes -> normal; x is per stretched texel, so undo the stretch
  let slope = vec2f(surface.x / u.videoAspect, surface.y) * (u.relief * (1.0 - 0.85 * handHere) * RELIEF_SCALE);
  let tilt = -slope / (1.0 + length(slope) * SLOPE_COMPRESSION);
  var N = normalize(vec3f(tilt, 1.0));
  var occlusion = mix(mix(1.0, surface.z, u.occlusion), 1.0, 0.85 * handHere);

  // composite the virtual cake into the scene (colour, depth and normal)
  let ck = cakeHit(src, d);
  // hand outline and its depth here, to tell whether the hand passes in front of a candle flame
  let handMask = select(0.0, handOutline, u.cakeOn > 0.5);
  let handD = depthAt(src);
  if (ck.x > 0.001) {
    let cuv = (src - u.cakeRect.xy) / u.cakeRect.zw;
    let cc = textureSampleLevel(cakeColTex, camSampler, cuv, 0.0);
    let cg = textureSampleLevel(cakeGeoTex, camSampler, cuv, 0.0);
    let nxy = cg.gb * 2.0 - 1.0;
    // the sprite stores x right, y up; this space has y down and z toward the camera
    let cn = normalize(vec3f(nxy.x, -nxy.y, sqrt(max(0.02, 1.0 - dot(nxy, nxy)))));
    let a = ck.x;
    albedo = mix(albedo, toLinear(cc.rgb), a);
    d = mix(d, ck.y, a);
    N = normalize(mix(N, cn, a));
    occlusion = mix(occlusion, 1.0, a);
  }
  let plane = toPlane(src);
  // on the hand, light it at the finger's smoothed depth: per-pixel depth on fingers is too noisy
  // so close to the flame, where small distance errors become big brightness swings
  let P = vec3f(plane, surfaceZ(mix(d, finger.x, 0.8 * handHere)));
  let jitter = ign(in.pos.xy);

  // ---- finger flame: a point light just in front of the finger, never behind the surface it sits over ----
  var direct = vec3f(0.0);
  if (u.flameAmt > 0.001) {
    let lightUV = u.tipSrc + vec2f(0.0, -u.flameSize * 0.32 * u.uvScale.y);
    let lightZ = max(surfaceZ(finger.x), surfaceZ(sceneDepth(lightUV))) + LIGHT_CLEARANCE;
    let L = vec3f(toPlane(lightUV), lightZ);
    let lt = lightTerm(N, P, L);
    let sh = mix(select(1.0, mix(1.0, shadowToward(plane, d, L, jitter), u.shadow), u.shadows > 0.5), 1.0, handHere);
    let tint = vec3f(1.0, 0.62, 0.32) * mix(0.92, 1.06, u.flicker);
    direct += (albedo * lt.x + vec3f(lt.y * u.specular * occlusion)) * tint * (sh * u.intensity * u.flameAmt * u.flicker);
  }

  // ---- birthday candles: each one is a point light; one shared soft shadow toward their centroid ----
  if (u.candleCount > 0u && u.shadowLight.w > 1e-5) {
    var sum = vec3f(0.0);
    for (var i = 0u; i < u.candleCount; i++) {
      let c = candles[i];
      if (c.light.w <= 1e-5) { continue; }
      let L = vec3f(toPlane(c.light.xy), surfaceZ(c.light.z) + LIGHT_CLEARANCE);
      let lt = lightTerm(N, P, L);
      sum += (albedo * lt.x + vec3f(lt.y * u.specular * occlusion)) * c.light.w;
    }
    let centroid = vec3f(toPlane(u.shadowLight.xy), surfaceZ(u.shadowLight.z) + LIGHT_CLEARANCE);
    let sh = select(1.0, mix(1.0, shadowToward(plane, d, centroid, fract(jitter + 0.37)), u.shadow), u.shadows > 0.5);
    direct += sum * vec3f(1.0, 0.64, 0.34) * sh;
  }

  let ambient = albedo * AMBIENT_FILL * (u.ambient * occlusion);
  var col = ambient + direct;

  if (u.view == 1u) { col = toLinear(depthRamp(saturate(d))); }
  if (u.view == 2u) { col = toLinear(N * 0.5 + 0.5); }
  if (u.view == 3u) { col = direct / max(albedo, vec3f(0.05)) * 0.25; }

  col = tonemap(col * u.exposure);

  let aspect = u.resolution.x / u.resolution.y;
  if (u.view == 0u || u.view == 3u) {
    // a finger behind the cake: its flame goes under the candle flames and the cake body hides it
    let fingerBehindCake = u.cakeOn > 0.5 && finger.x < u.cakeDepth - 0.015;
    if (fingerBehindCake) { col = fingerFlame(col, s, aspect, 1.0 - saturate(ck.x)); }
    // candle flames, halos and smoke; a hand hides a flame only where it is nearer than that candle
    {
      for (var i = 0u; i < u.candleCount; i++) {
        let c = candles[i];
        let vis = 1.0 - handMask * smoothstep(c.light.z + 0.01, c.light.z + 0.04, handD);
        let size = c.screen.z;
        let q = vec2f((s.x - c.screen.x) * aspect, c.screen.y - s.y) / size;
        if (c.anim.z > 0.002 && abs(q.x) < 3.0 && q.y > -0.2 && q.y < 7.0) {
          let a = smokeWisp(q, u.time, c.anim.w, c.anim.x, c.anim.z);
          col = mix(col, vec3f(0.62, 0.60, 0.58) * (0.45 + 0.55 * min(1.0, dot(direct, vec3f(0.33)) * 0.6)), clamp(a * 0.85, 0.0, 0.75) * vis);
        }
        if (c.screen.w < 0.002) { continue; }
        let hq = q - vec2f(c.anim.x * 0.4, 0.32);
        if (dot(hq, hq) < 40.0) {
          let halo = exp(-dot(hq, hq) * 1.4) * 0.05 + exp(-dot(hq, hq) * 10.0) * 0.08;
          col += vec3f(1.0, 0.55, 0.22) * halo * c.screen.w * vis;
        }
        if (abs(q.x) < 1.6 && q.y > -0.2 && q.y < 2.6) {
          let f = flameShape(q, u.time + c.anim.w, c.anim.x, c.anim.y);
          let fa = f.a * c.screen.w * vis;
          col = col * (1.0 - fa * 0.35) + f.rgb * c.screen.w * vis;
        }
      }
    }
    if (!fingerBehindCake) { col = fingerFlame(col, s, aspect, 1.0); }
  }
  return vec4f(toSrgb(col), 1.0);
}
`;
