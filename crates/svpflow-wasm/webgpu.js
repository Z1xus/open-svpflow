const MODES = new Set([1, 2, 11, 13, 21, 22, 23]);

const shader = `
struct Config {
  width: u32, height: u32, block_w: i32, block_h: i32,
  origin_x: i32, origin_y: i32, grid_w: u32, grid_h: u32,
  chroma_y_div: i32, source_step: i32, mode: u32, interpolate: u32,
  threshold: i32, threshold_limit: i32, mask_planes: u32, output_len: u32,
  source_y_len: u32, source_chroma_len: u32, output_y_len: u32, output_chroma_len: u32,
};
@group(0) @binding(0) var<uniform> cfg: Config;
@group(0) @binding(1) var<storage, read> source0: array<u32>;
@group(0) @binding(2) var<storage, read> source1: array<u32>;
@group(0) @binding(3) var<storage, read> motion: array<u32>;
@group(0) @binding(4) var<storage, read> masks: array<u32>;
@group(0) @binding(5) var<storage, read_write> output: array<atomic<u32>>;

fn byte_at(data: ptr<storage, array<u32>, read>, index: u32) -> i32 {
  return i32(((*data)[index >> 2u] >> ((index & 3u) * 8u)) & 255u);
}
fn half_at(index: u32) -> i32 {
  return i32((motion[index >> 1u] >> ((index & 1u) * 16u)) & 65535u);
}
fn grid_index(x: i32, y: i32) -> u32 {
  if (x < 0 || y < 0 || x >= i32(cfg.grid_w) || y >= i32(cfg.grid_h)) { return 0xffffffffu; }
  return u32(y) * cfg.grid_w + u32(x);
}
fn mask_at(plane: u32, x: i32, y: i32) -> i32 {
  if (x < 0 || y < 0 || plane >= cfg.mask_planes) { return 0; }
  let index = u32(y) * cfg.grid_w + u32(x);
  if (index >= cfg.grid_w * cfg.grid_h) { return 0; }
  return byte_at(&masks, plane * cfg.grid_w * cfg.grid_h + index);
}
fn mask_value(plane: u32, x: i32, y: i32) -> i32 {
  if (plane == 3u) { return max(mask_at(0u, x, y), mask_at(1u, x, y)); }
  return mask_at(plane, x, y);
}
fn cpu_shift(value: i32) -> i32 {
  switch value {
    case 1: { return 0; } case 2: { return 1; } case 3, 4: { return 2; }
    case 6, 7, 8: { return 3; } case 12, 14, 16: { return 4; }
    case 25, 29, 32: { return 5; } default: { return -1; }
  }
}
fn interp_pair(a: i32, b: i32, total: i32, pos: i32, shift: i32) -> i32 {
  if (shift < 0) { return ((total - pos) * a + pos * b) / total; }
  let scale = 1 << u32(shift);
  let wa = (2 * (total - pos) * scale + total - 1) / (2 * total);
  let wb = (2 * pos * scale + total - 1) / (2 * total);
  return (wa * a + wb * b) >> u32(shift);
}
fn interp4(a: i32, b: i32, c: i32, d: i32, bw: i32, bh: i32, x: i32, y: i32) -> i32 {
  let xshift = cpu_shift(bw);
  let yshift = cpu_shift(select(bh, cfg.block_h / 2, bw != cfg.block_w));
  return interp_pair(interp_pair(a, b, bw, x, xshift), interp_pair(c, d, bw, x, xshift), bh, y, yshift);
}
fn tile_axis(p: i32, origin: i32, block: i32, count: i32) -> vec3<i32> {
  if (p < origin) { return vec3<i32>(0, 0, p); }
  let cell = min((p - origin) / block, count - 1);
  return vec3<i32>(cell, min(cell + 1, count - 1), p - origin - cell * block);
}
fn scaled_motion(plane: u32, index: u32, factor: i32, divisor: i32) -> i32 {
  let centered = min(half_at(plane * cfg.grid_w * cfg.grid_h + index), 2047) - 1024;
  return (centered * factor / 256) / divisor;
}
fn motion_at(plane: u32, factor: i32, divisor: i32, tx: vec3<i32>, ty: vec3<i32>, bw: i32, bh: i32) -> i32 {
  let i00 = u32(ty.x) * cfg.grid_w + u32(tx.x);
  if (cfg.interpolate == 0u) { return scaled_motion(plane, i00, factor, divisor); }
  let i01 = u32(ty.x) * cfg.grid_w + u32(tx.y);
  let i10 = u32(ty.y) * cfg.grid_w + u32(tx.x);
  let i11 = u32(ty.y) * cfg.grid_w + u32(tx.y);
  return interp4(
    scaled_motion(plane, i00, factor, divisor), scaled_motion(plane, i01, factor, divisor),
    scaled_motion(plane, i10, factor, divisor), scaled_motion(plane, i11, factor, divisor),
    bw, bh, tx.z, ty.z);
}
fn alpha_interpolated(plane: u32, tx: vec3<i32>, ty: vec3<i32>, bw: i32, bh: i32) -> i32 {
  return clamp(interp4(mask_value(plane, tx.x, ty.x), mask_value(plane, tx.y, ty.x),
    mask_value(plane, tx.x, ty.y), mask_value(plane, tx.y, ty.y), bw, bh, tx.z, ty.z), 0, 255);
}
fn alpha_absolute(plane: u32, px: i32, py: i32, origin_x: i32, origin_y: i32, bw: i32, bh: i32) -> i32 {
  var x0 = 0; var x1 = 0; var y0 = 0; var y1 = 0;
  if (px >= origin_x) { x0 = (px - origin_x) / bw; x1 = x0 + 1; }
  if (py >= origin_y) { y0 = (py - origin_y) / bh; y1 = y0 + 1; }
  var fx = 0; var fy = 0;
  if (px > origin_x) { fx = (px - origin_x) % bw; }
  if (py > origin_y) { fy = (py - origin_y) % bh; }
  let top = ((bw - fx) * mask_value(plane, x0, y0) + fx * mask_value(plane, x1, y0)) / bw;
  let bottom = ((bw - fx) * mask_value(plane, x0, y1) + fx * mask_value(plane, x1, y1)) / bw;
  return clamp(((bh - fy) * top + fy * bottom) / bh, 0, 255);
}
fn alpha(plane: u32, tx: vec3<i32>, ty: vec3<i32>, px: i32, py: i32,
         origin_x: i32, origin_y: i32, bw: i32, bh: i32) -> i32 {
  if (cfg.interpolate != 0u) { return alpha_interpolated(plane, tx, ty, bw, bh); }
  return alpha_absolute(plane, px, py, origin_x, origin_y, bw, bh);
}
fn sample(data: ptr<storage, array<u32>, read>, offset: u32, stride: i32,
          width: i32, height: i32, px: i32, py: i32, dx: i32, dy: i32) -> i32 {
  let max_x = width * cfg.source_step - 1;
  let max_y = height * cfg.source_step - 1;
  let x = clamp(px * cfg.source_step + dx, 0, max_x);
  let y = clamp(py * cfg.source_step + dy, 0, max_y);
  return byte_at(data, offset + u32(y * stride + x));
}
fn blend255(a: i32, b: i32, weight: i32) -> i32 { return clamp((a * (255-weight) + b*weight + 255) >> 8u, 0, 255); }
fn blend256(a: i32, b: i32, weight: i32) -> i32 { return clamp((a * (256-weight) + b*weight + 128) >> 8u, 0, 255); }
fn between(value: i32, a: i32, b: i32) -> i32 { return clamp(value, min(a,b), max(a,b)); }

@compute @workgroup_size(256)
fn render(@builtin(global_invocation_id) id: vec3<u32>) {
  let out_index = id.x;
  if (out_index >= cfg.output_len) { return; }
  var chroma = false;
  var plane_offset = 0u;
  var source_offset = 0u;
  var plane_index = out_index;
  var width = i32(cfg.width);
  var height = i32(cfg.height);
  if (out_index >= cfg.output_y_len) {
    chroma = true; width = i32(cfg.width) / 2; height = i32(cfg.height) / cfg.chroma_y_div;
    plane_index = out_index - cfg.output_y_len;
    if (plane_index >= cfg.output_chroma_len) {
      plane_index -= cfg.output_chroma_len; plane_offset = cfg.output_chroma_len;
      source_offset = cfg.source_y_len + cfg.source_chroma_len;
    } else { source_offset = cfg.source_y_len; }
  }
  let px = i32(plane_index % u32(width));
  let py = i32(plane_index / u32(width));
  let xdiv = select(1, 2, chroma);
  let ydiv = select(1, cfg.chroma_y_div, chroma);
  let bw = cfg.block_w / xdiv; let bh = cfg.block_h / ydiv;
  let real_ox = cfg.origin_x / xdiv; let real_oy = cfg.origin_y / ydiv;
  let ox = select(0, real_ox, cfg.interpolate != 0u);
  let oy = select(0, real_oy, cfg.interpolate != 0u);
  let tx = tile_axis(px, ox, bw, i32(cfg.grid_w));
  let ty = tile_axis(py, oy, bh, i32(cfg.grid_h));
  let inverse = 256 - cfg.threshold;
  let dx0 = motion_at(0u, cfg.threshold, xdiv, tx, ty, bw, bh);
  let dy0 = motion_at(1u, cfg.threshold, ydiv, tx, ty, bw, bh);
  let dx1 = motion_at(2u, inverse, xdiv, tx, ty, bw, bh);
  let dy1 = motion_at(3u, inverse, ydiv, tx, ty, bw, bh);
  let stride = width * cfg.source_step;
  let cur0 = sample(&source0, source_offset, stride, width, height, px, py, 0, 0);
  let cur1 = sample(&source1, source_offset, stride, width, height, px, py, 0, 0);
  let mv0 = sample(&source0, source_offset, stride, width, height, px, py, dx0, dy0);
  let mv1 = sample(&source1, source_offset, stride, width, height, px, py, dx1, dy1);
  var result = 0;
  if (cfg.mode == 1u || cfg.mode == 2u) {
    let use_second = cfg.mode == 2u;
    let factor = cfg.threshold;
    let plane_x = select(0u, 2u, use_second); let plane_y = plane_x + 1u;
    let dx = motion_at(plane_x, factor, xdiv, tx, ty, bw, bh);
    let dy = motion_at(plane_y, factor, ydiv, tx, ty, bw, bh);
    let warped = select(sample(&source0, source_offset, stride, width, height, px, py, dx, dy),
      sample(&source1, source_offset, stride, width, height, px, py, dx, dy), use_second);
    let base = select(cur0, cur1, use_second);
    if (cfg.mask_planes >= 2u) { result = blend255(warped, base, alpha(select(1u, 0u, use_second), tx, ty, px, py, real_ox, real_oy, bw, bh)); }
    else { result = warped; }
  } else if (cfg.mode == 11u || cfg.mode == 13u) {
    result = select(blend256(mv0, mv1, cfg.threshold), between(blend256(cur0, cur1, cfg.threshold), mv0, mv1), cfg.mode == 13u);
    if (cfg.mask_planes >= 2u) {
      let a = alpha(3u, tx, ty, px, py, real_ox, real_oy, bw, bh);
      result = blend255(result, blend256(cur0, cur1, cfg.threshold), a);
    }
  } else if (cfg.mode == 21u || cfg.mode == 22u) {
    let a0 = alpha(0u, tx, ty, px, py, real_ox, real_oy, bw, bh);
    let a1 = alpha(1u, tx, ty, px, py, real_ox, real_oy, bw, bh);
    let a = blend255(mv0, mv1, a0); let b = blend255(mv1, mv0, a1);
    result = select(blend256(a, b, cfg.threshold), between(blend256(cur0, cur1, cfg.threshold), a, b), cfg.mode == 22u);
    if (cfg.mask_planes == 3u) { result = blend255(result, blend256(cur0, cur1, cfg.threshold_limit), alpha(2u, tx, ty, px, py, real_ox, real_oy, bw, bh)); }
  } else {
    let dx2 = motion_at(4u, cfg.threshold, xdiv, tx, ty, bw, bh);
    let dy2 = motion_at(5u, cfg.threshold, ydiv, tx, ty, bw, bh);
    let dx3 = motion_at(6u, inverse, xdiv, tx, ty, bw, bh);
    let dy3 = motion_at(7u, inverse, ydiv, tx, ty, bw, bh);
    let c = sample(&source0, source_offset, stride, width, height, px, py, dx2, dy2);
    let d = sample(&source1, source_offset, stride, width, height, px, py, dx3, dy3);
    let i0 = blend255(mv0, between(c, mv0, mv1), alpha(0u, tx, ty, px, py, real_ox, real_oy, bw, bh));
    let i1 = blend255(mv1, between(d, mv0, mv1), alpha(1u, tx, ty, px, py, real_ox, real_oy, bw, bh));
    result = blend256(i0, i1, cfg.threshold);
    if (cfg.mask_planes == 3u) { result = blend255(result, blend256(cur0, cur1, cfg.threshold), alpha(2u, tx, ty, px, py, real_ox, real_oy, bw, bh)); }
  }
  atomicOr(&output[out_index >> 2u], u32(result) << ((out_index & 3u) * 8u));
}
`;

const blendShader = `
struct Params {
  output_len: u32, y_len: u32, chroma_len: u32, frame_words: u32,
  frame_count: u32, bright: u32, grading: u32, coring: u32,
  brightness: f32, contrast: f32, chroma_cos: f32, chroma_sin: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> frames: array<u32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> luma_lut: array<u32>;
@group(0) @binding(5) var<storage, read> chroma_lut: array<u32>;

fn byte_at(index: u32) -> f32 {
  return f32((frames[index >> 2u] >> ((index & 3u) * 8u)) & 255u);
}
fn blend_at(pixel: u32) -> f32 {
  var value = 0.0;
  var sum = 0.0;
  for (var frame = 0u; frame < p.frame_count; frame++) {
    let sample = byte_at(frame * p.frame_words * 4u + pixel);
    let weight = weights[frame];
    value += select(sample, pow(sample / 255.0, 2.2), p.bright != 0u && pixel < p.y_len) * weight;
    sum += weight;
  }
  value /= sum;
  if (p.bright != 0u && pixel < p.y_len) { value = pow(clamp(value, 0.0, 1.0), 1.0 / 2.2) * 255.0; }
  return value;
}

@compute @workgroup_size(256)
fn blend(@builtin(global_invocation_id) id: vec3<u32>) {
  let pixel = id.x;
  if (pixel >= p.output_len) { return; }
  var value = blend_at(pixel);
  if (p.grading != 0u) {
    value = floor(value + 0.5);
    if (pixel < p.y_len) {
      value = f32(luma_lut[u32(value)]);
    } else {
      let chroma = (pixel - p.y_len) % p.chroma_len;
      let u = u32(floor(blend_at(p.y_len + chroma) + 0.5));
      let v = u32(floor(blend_at(p.y_len + p.chroma_len + chroma) + 0.5));
      let pair = chroma_lut[u * 256u + v];
      value = f32(select(pair & 255u, pair >> 8u, pixel >= p.y_len + p.chroma_len));
    }
  }
  let result = u32(clamp(floor(value + 0.5), 0.0, 255.0));
  atomicOr(&output[pixel >> 2u], result << ((pixel & 3u) * 8u));
}
`;

const words = length => Math.max(1, Math.ceil(length / 4));
const packed = view => {
  const bytes = new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
  const copy = new Uint8Array(Math.ceil(bytes.length / 4) * 4);
  copy.set(bytes);
  return new Uint32Array(copy.buffer);
};

export class WebGpuRenderer {
  static async create(config) {
    if (!navigator.gpu) throw new Error("WebGPU is unavailable");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("no WebGPU adapter");
    const device = await adapter.requestDevice();
    const renderer = new WebGpuRenderer(device, config);
    renderer.adapterInfo = adapter.info;
    return renderer;
  }

  constructor(device, {width, height, blockWidth, blockHeight, originX = 0, originY = 0,
    gridWidth, gridHeight, chromaYDivisor = 2, sourceStep = 1, scale = 1}) {
    if (![width, height, blockWidth, blockHeight, gridWidth, gridHeight, chromaYDivisor, sourceStep].every(Number.isInteger) ||
        Math.min(width, height, blockWidth, blockHeight, gridWidth, gridHeight, chromaYDivisor, sourceStep) <= 0 ||
        blockWidth > 32 || blockHeight > 32 || !Number.isFinite(scale)) throw new Error("invalid renderer configuration");
    this.device = device;
    this.config = {width, height, blockWidth, blockHeight, originX, originY, gridWidth, gridHeight, chromaYDivisor, sourceStep, scale};
    this.threshold = 0;
    const chromaWidth = Math.floor(width / 2), chromaHeight = Math.floor(height / chromaYDivisor);
    this.sourceYLength = width * height * sourceStep * sourceStep;
    this.sourceChromaLength = chromaWidth * chromaHeight * sourceStep * sourceStep;
    this.sourceLength = this.sourceYLength + 2 * this.sourceChromaLength;
    this.outputYLength = width * height;
    this.outputChromaLength = chromaWidth * chromaHeight;
    this.outputLength = this.outputYLength + 2 * this.outputChromaLength;
    this.gridLength = gridWidth * gridHeight;
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    this.uniform = device.createBuffer({size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
    this.source0 = device.createBuffer({size: words(this.sourceLength) * 4, usage: storage});
    this.source1 = device.createBuffer({size: words(this.sourceLength) * 4, usage: storage});
    this.motion = device.createBuffer({size: words(this.gridLength * 8 * 2) * 4, usage: storage});
    this.masks = device.createBuffer({size: words(this.gridLength * 3) * 4, usage: storage});
    this.outputBytes = words(this.outputLength) * 4;
    this.output = device.createBuffer({size: this.outputBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST});
    this.readback = device.createBuffer({size: this.outputBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
    const module = device.createShaderModule({code: shader});
    this.pipeline = device.createComputePipeline({layout: "auto", compute: {module, entryPoint: "render"}});
    this.bindGroup = device.createBindGroup({layout: this.pipeline.getBindGroupLayout(0), entries: [
      [0, this.uniform], [1, this.source0], [2, this.source1], [3, this.motion], [4, this.masks], [5, this.output]
    ].map(([binding, buffer]) => ({binding, resource: {buffer}}))});
    const blendModule = device.createShaderModule({code: blendShader});
    this.blendPipeline = device.createComputePipeline({layout: "auto", compute: {module: blendModule, entryPoint: "blend"}});
    this.blendUniform = device.createBuffer({size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
    this.lumaLut = device.createBuffer({size: 256 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
    this.chromaLut = device.createBuffer({size: 65536 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
    this.batchCapacity = 0;
  }

  setThreshold(value) { if (!Number.isInteger(value)) throw new Error("invalid threshold"); this.threshold = value; }

  upload(source0, source1, motion, masks = new Uint8Array()) {
    this.uploadSources(source0, source1);
    this.uploadMotion(motion);
    this.uploadMasks(masks);
  }

  uploadSources(source0, source1) {
    if (!(source0 instanceof Uint8Array) || source0.length !== this.sourceLength || source1.length !== this.sourceLength) throw new Error("invalid source buffer length");
    this.device.queue.writeBuffer(this.source0, 0, packed(source0));
    this.device.queue.writeBuffer(this.source1, 0, packed(source1));
  }

  uploadMotion(motion) {
    if (!(motion instanceof Uint16Array) || ![this.gridLength * 4, this.gridLength * 8].includes(motion.length)) throw new Error("invalid motion buffer length");
    this.motionPlanes = motion.length / this.gridLength / 2;
    this.device.queue.writeBuffer(this.motion, 0, packed(motion));
  }

  uploadMasks(masks = new Uint8Array()) {
    if (!(masks instanceof Uint8Array) || ![0, this.gridLength * 2, this.gridLength * 3].includes(masks.length)) throw new Error("invalid mask buffer length");
    this.maskPlanes = masks.length / this.gridLength;
    if (masks.length) this.device.queue.writeBuffer(this.masks, 0, packed(masks));
  }

  beginBatch(frameCount) {
    if (!Number.isInteger(frameCount) || frameCount < 1) throw new Error("invalid batch size");
    this.batchCount = frameCount;
    if (frameCount <= this.batchCapacity) return;
    this.batchCapacity = frameCount;
    this.batchFrames?.destroy();
    this.batchWeights?.destroy();
    this.batchFrames = this.device.createBuffer({size: this.outputBytes * frameCount, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
    this.batchWeights = this.device.createBuffer({size: Math.ceil(frameCount / 4) * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
    this.blendBindGroup = this.device.createBindGroup({layout: this.blendPipeline.getBindGroupLayout(0), entries: [
      {binding: 0, resource: {buffer: this.blendUniform}},
      {binding: 1, resource: {buffer: this.batchFrames}},
      {binding: 2, resource: {buffer: this.batchWeights}},
      {binding: 3, resource: {buffer: this.output}},
      {binding: 4, resource: {buffer: this.lumaLut}},
      {binding: 5, resource: {buffer: this.chromaLut}},
    ]});
  }

  uploadFrame(slot, frame) {
    if (!(frame instanceof Uint8Array) || frame.length !== this.outputLength || slot < 0 || slot >= this.batchCount) throw new Error("invalid batch frame");
    this.device.queue.writeBuffer(this.batchFrames, slot * this.outputBytes, packed(frame));
  }

  renderToSlot(mode, interpolate, slot) {
    this.#prepare(mode, interpolate);
    if (slot < 0 || slot >= this.batchCount) throw new Error("invalid batch slot");
    const encoder = this.device.createCommandEncoder();
    this.#encodeRender(encoder);
    encoder.copyBufferToBuffer(this.output, 0, this.batchFrames, slot * this.outputBytes, this.outputBytes);
    this.device.queue.submit([encoder.finish()]);
  }

  async finishBatch(weights, bright, grading) {
    if (!(weights instanceof Float32Array) || weights.length !== this.batchCount) throw new Error("invalid blend weights");
    const hue = (grading.hue * Math.PI) / 180;
    const gradingKey = JSON.stringify(grading);
    if (grading.enabled && gradingKey !== this.gradingKey) {
      const luma = new Uint32Array(256), chroma = new Uint32Array(65536);
      const yLo = grading.coring ? 16 : 0, yHi = grading.coring ? 235 : 255;
      const cLo = grading.coring ? 16 : 0, cHi = grading.coring ? 240 : 255;
      for (let i = 0; i < 256; i++) luma[i] = Math.max(yLo, Math.min(yHi, Math.round(((i - 126) * grading.contrast + 126) * grading.brightness)));
      const cos = Math.cos(hue) * grading.saturation, sin = Math.sin(hue) * grading.saturation;
      for (let u = 0; u < 256; u++) for (let v = 0; v < 256; v++) {
        const cu = u - 128, cv = v - 128;
        const outU = Math.max(cLo, Math.min(cHi, Math.round(cu * cos - cv * sin + 128)));
        const outV = Math.max(cLo, Math.min(cHi, Math.round(cu * sin + cv * cos + 128)));
        chroma[u * 256 + v] = outU | (outV << 8);
      }
      this.device.queue.writeBuffer(this.lumaLut, 0, luma);
      this.device.queue.writeBuffer(this.chromaLut, 0, chroma);
      this.gradingKey = gradingKey;
    }
    const params = new ArrayBuffer(48), u32 = new Uint32Array(params), f32 = new Float32Array(params);
    u32.set([this.outputLength,this.outputYLength,this.outputChromaLength,this.outputBytes/4,this.batchCount,bright?1:0,grading.enabled?1:0,grading.coring?1:0]);
    f32.set([grading.brightness,grading.contrast,Math.cos(hue)*grading.saturation,Math.sin(hue)*grading.saturation], 8);
    this.device.queue.writeBuffer(this.blendUniform, 0, params);
    this.device.queue.writeBuffer(this.batchWeights, 0, weights);
    const encoder = this.device.createCommandEncoder();
    encoder.clearBuffer(this.output);
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.blendPipeline); pass.setBindGroup(0, this.blendBindGroup); pass.dispatchWorkgroups(Math.ceil(this.outputLength / 256)); pass.end();
    encoder.copyBufferToBuffer(this.output, 0, this.readback, 0, this.outputBytes);
    this.device.queue.submit([encoder.finish()]);
    await this.readback.mapAsync(GPUMapMode.READ);
    const result = new Uint8Array(this.readback.getMappedRange()).slice(0, this.outputLength);
    this.readback.unmap();
    return result;
  }

  #prepare(mode, interpolate) {
    if (!MODES.has(mode)) throw new Error("unsupported render mode");
    if (mode === 23 && this.motionPlanes !== 4) throw new Error("mode 23 requires four motion vectors");
    if ([21, 22, 23].includes(mode) && this.maskPlanes < 2) throw new Error("render mode requires masks");
    const c = this.config;
    const thresholdLimit = Math.trunc(this.threshold > 126 ? 256 - (256 - this.threshold) * c.scale : this.threshold * c.scale);
    const values = new Int32Array([c.width,c.height,c.blockWidth,c.blockHeight,c.originX,c.originY,c.gridWidth,c.gridHeight,
      c.chromaYDivisor,c.sourceStep,mode,interpolate ? 1 : 0,this.threshold,thresholdLimit,this.maskPlanes,this.outputLength,
      this.sourceYLength,this.sourceChromaLength,this.outputYLength,this.outputChromaLength]);
    this.device.queue.writeBuffer(this.uniform, 0, values);
  }

  #encodeRender(encoder) {
    encoder.clearBuffer(this.output);
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline); pass.setBindGroup(0, this.bindGroup); pass.dispatchWorkgroups(Math.ceil(this.outputLength / 256)); pass.end();
  }

  async render(mode, interpolate, readback = true) {
    this.#prepare(mode, interpolate);
    const encoder = this.device.createCommandEncoder();
    this.#encodeRender(encoder);
    if (readback) encoder.copyBufferToBuffer(this.output, 0, this.readback, 0, this.outputBytes);
    this.device.queue.submit([encoder.finish()]);
    if (!readback) { await this.device.queue.onSubmittedWorkDone(); return null; }
    await this.readback.mapAsync(GPUMapMode.READ);
    const result = new Uint8Array(this.readback.getMappedRange()).slice(0, this.outputLength);
    this.readback.unmap();
    return result;
  }
}
