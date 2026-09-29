override ALGO: i32 = 23;
override CUBIC: bool = false;
override CUBIC_REF: bool = false;
override HAS_SAD: bool = false;
override LINEAR_LUMA: bool = true;
override DITHER: bool = false;

struct Params {
  algorithm: i32, width: i32, height: i32, x_ratio: i32, y_ratio: i32, pel: i32,
  block_w: i32, block_h: i32, origin_x: i32, origin_y: i32, phase: i32, has_sad: i32,
  linear_luma: i32, cubic: i32, cubic_ref: i32, offset_x: i32, offset_y: i32,
  sad_blend: f32, dither: i32, out_base: u32, out_stride: u32, pad: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var source_f: texture_2d<f32>;
@group(0) @binding(2) var source_b: texture_2d<f32>;
@group(0) @binding(3) var vectors: texture_2d<f32>;
@group(0) @binding(4) var vectors_ext: texture_2d<f32>;
@group(0) @binding(5) var masks: texture_2d<f32>;
@group(0) @binding(6) var<storage, read_write> output: array<atomic<u32>>;

fn lerp(a: f32, b: f32, t: f32) -> f32 { return a + (b - a) * t; }
fn lerp4(a: vec4f, b: vec4f, t: f32) -> vec4f { return a + (b - a) * t; }

fn median3(a: f32, b: f32, c: f32) -> f32 {
  let lo = min(a, b);
  let hi = a + b - lo;
  return max(lo, min(hi, c));
}

fn filter_weights(position: vec2f) -> vec4i {
  let grid = position - vec2f(0.5, 0.5);
  let k = vec2i(floor((grid - floor(grid)) * 256.0 + 0.5));
  let a = 256 - k;
  return vec4i((a.x * a.y + 128) >> 8u, (k.x * a.y + 127) >> 8u, (a.x * k.y + 127) >> 8u, (k.x * k.y + 128) >> 8u);
}

const UNORM8: i32 = 0;
const RAW16: i32 = 1;
const FLOAT: i32 = 2;

fn sample_at(image: texture_2d<f32>, position: vec2f, kind: i32) -> vec4f {
  let size = vec2i(textureDimensions(image));
  let base = vec2i(floor(position - vec2f(0.5, 0.5)));
  let i0 = clamp(base, vec2i(0), size - 1);
  let i1 = clamp(base + 1, vec2i(0), size - 1);
  let w = filter_weights(position);
  let t00 = textureLoad(image, i0, 0);
  let t10 = textureLoad(image, vec2i(i1.x, i0.y), 0);
  let t01 = textureLoad(image, vec2i(i0.x, i1.y), 0);
  let t11 = textureLoad(image, i1, 0);
  if (kind == FLOAT) {
    let wf = vec4f(w) / 256.0;
    return ((wf.x * t00 + wf.y * t10) + wf.z * t01) + wf.w * t11;
  }
  var scale = 1.0;
  var widen = 1;
  if (kind == UNORM8) {
    scale = 255.0;
    widen = 257;
  }
  let a = vec4i(round(t00 * scale)) * widen;
  let b = vec4i(round(t10 * scale)) * widen;
  let c = vec4i(round(t01 * scale)) * widen;
  let d = vec4i(round(t11 * scale)) * widen;
  let sum = vec4u(w.x * a + w.y * b + w.z * c + w.w * d);
  return vec4f((sum + 128u) >> vec4u(8u)) / 65535.0;
}

fn cubic_sample(image: texture_2d<f32>, position: vec2f, kind: i32) -> vec4f {
  let grid = position - vec2f(0.5, 0.5);
  let index = floor(grid);
  let f = grid - index;
  let r = 1.0 - f;
  let r2 = r * r;
  let f2 = f * f;
  let w0 = 1.0 / 6.0 * r2 * r;
  let w1 = 2.0 / 3.0 - 0.5 * f2 * (2.0 - f);
  let w2 = 2.0 / 3.0 - 0.5 * r2 * (2.0 - r);
  let w3 = 1.0 / 6.0 * f2 * f;
  let g0 = w0 + w1;
  let g1 = w2 + w3;
  let h0 = (w1 / g0) - 0.5 + index;
  let h1 = (w3 / g1) + 1.5 + index;
  var a = sample_at(image, h0, kind);
  var b = sample_at(image, vec2f(h1.x, h0.y), kind);
  let c = sample_at(image, vec2f(h0.x, h1.y), kind);
  let d = sample_at(image, h1, kind);
  a = lerp4(c, a, g0.y);
  b = lerp4(d, b, g0.y);
  return lerp4(b, a, g0.x);
}

fn field_sample(image: texture_2d<f32>, position: vec2f, kind: i32) -> vec4f {
  if (CUBIC) { return cubic_sample(image, position, kind); }
  return sample_at(image, position * vec2f(textureDimensions(image)), kind);
}

fn scale() -> f32 { return 255.0; }
fn source_kind() -> i32 { return select(UNORM8, FLOAT, p.linear_luma != 0); }

fn moved(gid: vec2f, vx: f32, vy: f32, time: i32) -> vec2f {
  let step = (vec2f(vx * 65535.0 - 1024.0, vy * 65535.0 - 1024.0) * f32(time))
    / (vec2f(f32(p.x_ratio * p.pel), f32(p.y_ratio * p.pel)) * 256.0);
  return clamp(gid + step, vec2f(0.0, 0.0), vec2f(f32(p.width - 1), f32(p.height - 1)));
}

fn source_sample(source: texture_2d<f32>, gid: vec2f, vx: f32, vy: f32, time: i32) -> f32 {
  let position = (vec2f(f32(p.offset_x), f32(p.offset_y)) + vec2f(0.5, 0.5)) + moved(gid, vx, vy, time);
  if (CUBIC_REF) { return scale() * cubic_sample(source, position, source_kind()).x; }
  return scale() * sample_at(source, position, source_kind()).x;
}

fn linear_source_sample(source: texture_2d<f32>, gid: vec2f, vx: f32, vy: f32, time: i32) -> f32 {
  let position = (vec2f(f32(p.offset_x), f32(p.offset_y)) + vec2f(0.5, 0.5)) + moved(gid, vx, vy, time);
  return scale() * sample_at(source, position, source_kind()).x;
}

fn base_sample(source: texture_2d<f32>, gid: vec2f) -> f32 {
  let position = (vec2f(f32(p.offset_x), f32(p.offset_y)) + vec2f(0.5, 0.5)) + gid;
  if (CUBIC_REF) { return scale() * cubic_sample(source, position, source_kind()).x; }
  return scale() * sample_at(source, position, source_kind()).x;
}

const bayer = array<i32, 64>(
  1, 49, 13, 61, 4, 52, 16, 64, 33, 17, 45, 29, 36, 20, 48, 32,
  9, 57, 5, 53, 12, 60, 8, 56, 41, 25, 37, 21, 44, 28, 40, 24,
  3, 51, 15, 63, 2, 50, 14, 62, 35, 19, 47, 31, 34, 18, 46, 30,
  11, 59, 7, 55, 10, 58, 6, 54, 43, 27, 39, 23, 42, 26, 38, 22);

fn round_half_up(value: f32) -> f32 {
  let whole = trunc(value);
  return select(whole, whole + 1.0, value - whole >= 0.5);
}

@compute @workgroup_size(16, 16)
fn render(@builtin(global_invocation_id) id: vec3u) {
  let x = i32(id.x);
  let y = i32(id.y);
  if (x >= p.width || y >= p.height) { return; }
  let gid = vec2f(f32(x), f32(y));

  let field_position = vec2f(f32(x * p.x_ratio - p.origin_x), f32(y * p.y_ratio - p.origin_y))
    / vec2f(f32(p.block_w), f32(p.block_h));
  let vector = field_sample(vectors, field_position, RAW16);
  var mask = vec4f(0.0);
  if (ALGO >= 21 || HAS_SAD) { mask = field_sample(masks, field_position, UNORM8); }
  let time = f32(p.phase) / 256.0;

  let ref_b = source_sample(source_b, gid, vector.z, vector.y, 256 - p.phase);
  let ref_f = source_sample(source_f, gid, vector.x, vector.w, p.phase);
  var ref_b0 = 0.0;
  var ref_f0 = 0.0;
  if (ALGO == 13 || ALGO == 22 || HAS_SAD) {
    ref_b0 = base_sample(source_b, gid);
    ref_f0 = base_sample(source_f, gid);
  }
  var result: f32;
  if (ALGO == 1) {
    result = ref_b + ref_f * 0.00001;
  } else if (ALGO == 2) {
    result = ref_f + ref_b * 0.00001;
  } else if (ALGO == 11) {
    result = lerp(ref_f, ref_b, time);
  } else if (ALGO == 13) {
    result = median3(ref_f, ref_b, lerp(ref_f0, ref_b0, p.sad_blend));
  } else if (ALGO == 21) {
    result = lerp(lerp(ref_f, ref_b, mask.y), lerp(ref_b, ref_f, mask.z), time);
  } else if (ALGO == 22) {
    result = median3(lerp(ref_f, ref_b, mask.y), lerp(ref_b, ref_f, mask.z), lerp(ref_f0, ref_b0, time));
  } else {
    let ext = field_sample(vectors_ext, field_position, RAW16);
    let ref_bb = linear_source_sample(source_b, gid, ext.z, ext.y, 256 - p.phase);
    let ref_ff = linear_source_sample(source_f, gid, ext.x, ext.w, p.phase);
    result = lerp(
      lerp(ref_f, median3(ref_b, ref_bb, ref_f), mask.y),
      lerp(ref_b, median3(ref_b, ref_ff, ref_f), mask.z),
      time);
  }
  if (HAS_SAD) {
    if (ALGO == 1) {
      result = lerp(result, ref_b0, mask.x);
    } else if (ALGO == 2) {
      result = lerp(result, ref_f0, mask.w);
    } else {
      result = lerp(result, lerp(ref_f0, ref_b0, p.sad_blend), max(mask.w, mask.x));
    }
  }
  if (LINEAR_LUMA && p.linear_luma != 0) {
    result = result / 255.0;
    if (result < 0.018) { result *= 4.5; } else { result = 1.099 * pow(result, 0.45) - 0.099; }
    result *= 255.0;
  }
  if (DITHER) { result += f32(bayer[(x % 8) * 8 + (y % 8)] - 32) / 65.0; }
  let value = u32(clamp(round_half_up(result), 0.0, 255.0));
  let index = p.out_base + u32(y) * p.out_stride + u32(x);
  atomicOr(&output[index >> 2u], value << ((index & 3u) * 8u));
}

@group(1) @binding(0) var packed_source: texture_2d<f32>;
@group(1) @binding(1) var linear_out: texture_storage_2d<r32float, write>;

@compute @workgroup_size(16, 16)
fn linear_luma(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(linear_out);
  if (id.x >= size.x || id.y >= size.y) { return; }
  var value = textureLoad(packed_source, vec2i(id.xy), 0).x;
  if (value < 0.081) { value = value / 4.5; } else { value = pow((value + 0.099) / 1.099, 1.0 / 0.45); }
  textureStore(linear_out, vec2i(id.xy), vec4f(value, 0.0, 0.0, 0.0));
}
