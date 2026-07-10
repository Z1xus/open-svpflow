#[derive(Clone, Copy)]
pub struct VectorData {
    pub flags: i32,
    pub block: VectorShape,
    pub shape: VectorShape,
    pub grid: VectorShape,
    pub marker: i32,
    selector: i32,
    pub delta: i32,
    overlap_x: i32,
    overlap_y: i32,
}

#[derive(Clone, Copy)]
pub struct VectorShape {
    pub width: i32,
    pub height: i32,
}

pub enum VectorRecord {
    Missing,
    Invalid,
    Ready(VectorData),
}

pub struct SuperData {
    marker: i32,
    width: i32,
    height: i32,
    selector: i32,
    limit: i32,
}

#[derive(Clone, Copy, Default)]
pub struct DecodedVector {
    pub dx: i16,
    pub dy: i16,
    pub score: u32,
    pub luma: u8,
}

pub struct DecodedVectors {
    pub previous: Option<Vec<DecodedVector>>,
    pub current: Option<Vec<DecodedVector>>,
}

pub struct LumaStats {
    pub map: Vec<u8>,
    pub geomean: f64,
    pub max: f64,
}

pub struct SceneAnalysis {
    pub class: i32,
    pub luma: LumaStats,
}

pub struct AvglumaPayload {
    pub class: i32,
    pub geomean: f64,
    pub max: f64,
}

#[derive(Clone, Copy)]
pub struct SceneThresholds {
    pub blocks_pct: i32,
    pub blocks13_pct: i32,
    pub zero: i32,
    pub m1: i32,
    pub m2: i32,
    pub scene: i32,
    pub ignore: f64,
}

const VECTOR_MAGIC: i32 = 0xA0;
const FLAGS: usize = 0x04;
const BLOCK_WIDTH: usize = 0x08;
const BLOCK_HEIGHT: usize = 0x0C;
const MARKER: usize = 0x10;
const WIDTH: usize = 0x1C;
const HEIGHT: usize = 0x20;
const OVERLAP_X: usize = 0x24;
const OVERLAP_Y: usize = 0x28;
const GRID_WIDTH: usize = 0x2C;
const GRID_HEIGHT: usize = 0x30;
const SELECTOR: usize = 0x34;
const DELTA: usize = 0x38;
pub const VECTOR_DATA_LEN: usize = DELTA + 4;

pub fn vector_data(data: &[u8]) -> VectorRecord {
    if data.len() < VECTOR_DATA_LEN || read_i32_slice(data, 0) != Some(VECTOR_MAGIC) {
        return VectorRecord::Invalid;
    }
    VectorRecord::Ready(VectorData {
        flags: read_i32_slice(data, FLAGS).unwrap_or_default(),
        block: VectorShape {
            width: read_i32_slice(data, BLOCK_WIDTH).unwrap_or_default(),
            height: read_i32_slice(data, BLOCK_HEIGHT).unwrap_or_default(),
        },
        shape: VectorShape {
            width: read_i32_slice(data, WIDTH).unwrap_or_default(),
            height: read_i32_slice(data, HEIGHT).unwrap_or_default(),
        },
        grid: VectorShape {
            width: read_i32_slice(data, GRID_WIDTH).unwrap_or_default(),
            height: read_i32_slice(data, GRID_HEIGHT).unwrap_or_default(),
        },
        marker: read_i32_slice(data, MARKER).unwrap_or_default(),
        selector: read_i32_slice(data, SELECTOR).unwrap_or_default(),
        delta: read_i32_slice(data, DELTA).unwrap_or_default(),
        overlap_x: read_i32_slice(data, OVERLAP_X).unwrap_or_default(),
        overlap_y: read_i32_slice(data, OVERLAP_Y).unwrap_or_default(),
    })
}

pub fn super_data(handle: i64) -> SuperData {
    let raw = u64::from_ne_bytes(handle.to_ne_bytes());
    SuperData {
        marker: byte(raw, 32),
        width: word(raw, 16),
        height: word(raw, 0),
        selector: byte(raw, 48),
        limit: byte(raw, 56),
    }
}

impl SuperData {
    pub const fn scale(&self) -> i32 {
        self.marker
    }

    pub const fn matches_vectors(&self, vectors: &VectorData) -> bool {
        self.marker == vectors.marker
            && self.width == vectors.shape.width
            && self.height == vectors.shape.height
            && (self.limit >= 4 || self.selector >= 3 || self.selector == vectors.selector)
    }
}

impl VectorData {
    pub const fn generated_rife(width: i32, height: i32) -> Self {
        Self {
            flags: 2,
            block: VectorShape {
                width: 16,
                height: 16,
            },
            shape: VectorShape { width, height },
            grid: VectorShape {
                width: ceil_div(width, 16),
                height: ceil_div(height, 16),
            },
            marker: 1,
            selector: 1,
            delta: 1,
            overlap_x: 0,
            overlap_y: 0,
        }
    }

    pub const fn generated_source_8bit(width: i32, height: i32, scene_mode: i64) -> Self {
        Self {
            flags: if scene_mode == 1 { 2 } else { 3 },
            block: VectorShape {
                width: 4,
                height: 4,
            },
            shape: VectorShape { width, height },
            grid: VectorShape {
                width: ceil_div(width, 4),
                height: ceil_div(height, 4),
            },
            marker: 4,
            selector: 1,
            delta: 1,
            overlap_x: 0,
            overlap_y: 0,
        }
    }

    pub const fn marker_is_one(&self) -> bool {
        self.marker == 1
    }

    pub const fn render_mode_one(&self) -> bool {
        self.selector < 1
    }

    pub const fn rejects_cubic(&self) -> bool {
        self.selector == 0
    }

    pub const fn has_odd_overlap(&self) -> bool {
        self.overlap_x & 1 != 0 || self.overlap_y & 1 != 0
    }

    pub const fn effective_block(&self) -> VectorShape {
        VectorShape {
            width: positive(self.block.width - self.overlap_x),
            height: positive(self.block.height - self.overlap_y),
        }
    }

    pub const fn origin(&self) -> VectorShape {
        VectorShape {
            width: self.overlap_x,
            height: self.overlap_y,
        }
    }

    pub const fn motion_grid(&self) -> VectorShape {
        VectorShape {
            width: extended_grid(
                self.block.width,
                self.overlap_x,
                self.grid.width,
                self.shape.width,
            ),
            height: extended_grid(
                self.block.height,
                self.overlap_y,
                self.grid.height,
                self.shape.height,
            ),
        }
    }
}

const fn positive(value: i32) -> i32 {
    if value > 0 { value } else { 1 }
}

const fn ceil_div(value: i32, divisor: i32) -> i32 {
    if value <= 0 {
        0
    } else {
        (value + divisor - 1) / divisor
    }
}

const fn extended_grid(block: i32, overlap: i32, grid: i32, source: i32) -> i32 {
    let span = (block - overlap) * grid + overlap;
    grid + if span < source { 1 } else { 0 }
}

pub fn decode_vectors(
    payload: &[u8],
    mode_flags: i32,
    shape: VectorShape,
    flags: i32,
) -> Option<DecodedVectors> {
    if read_i32_slice(payload, 0)? != 16
        || read_i32_slice(payload, 4)? != VECTOR_MAGIC
        || read_i32_slice(payload, 8)? != mode_flags
    {
        return None;
    }

    let count = vector_count(shape)?;
    let first = 0x40;
    let second = first + 4 + count.checked_mul(8)?;
    let previous = if flags & 1 != 0 {
        Some(decode_region(payload, first, count)?)
    } else {
        None
    };
    let current = if flags & 2 != 0 {
        Some(decode_region(payload, second, count)?)
    } else {
        None
    };
    Some(DecodedVectors { previous, current })
}

pub fn luma_map(
    previous: &[DecodedVector],
    current: &[DecodedVector],
    marker: i32,
    gamma: f64,
) -> LumaStats {
    let count = previous.len().min(current.len());
    let denom = if marker == 3 { 510.0 } else { 255.0 };
    let mut map = Vec::with_capacity(count);
    let mut max = 0.0;
    let mut sum_log = 0.0;
    for i in 0..count {
        let value = (f64::from(previous[i].luma) + f64::from(current[i].luma)) / denom;
        max = max_f64(max, value);
        sum_log += value.ln();
        map.push(luma_byte(value, gamma));
    }
    LumaStats {
        map,
        geomean: (sum_log / usize_to_f64(count)).exp(),
        max,
    }
}

pub fn classify_scene(
    vectors: &[DecodedVector],
    luma: &[u8],
    shape: VectorShape,
    thresholds: SceneThresholds,
) -> i32 {
    let width = shape.width.max(0);
    let height = shape.height.max(0);
    let mut scene_count = 0;
    let mut m2_count = 0;
    let mut m1_count = 0;
    let mut considered = 0;
    for y in 0..height {
        for x in 0..width {
            let index =
                usize::try_from(y.saturating_mul(width).saturating_add(x)).unwrap_or(usize::MAX);
            let Some(vector) = vectors.get(index) else {
                continue;
            };
            let luma = i32::from(luma.get(index).copied().unwrap_or(1).max(1));
            let score = i32::try_from(vector.score)
                .unwrap_or(i32::MAX)
                .saturating_mul(255)
                / luma;
            if score < thresholds.zero {
                continue;
            }
            considered += 1;
            if score >= thresholds.scene {
                scene_count += 1;
            } else if score >= thresholds.m2 {
                m2_count += 1;
            } else if score >= thresholds.m1 {
                m1_count += 1;
            }
        }
    }

    let required = considered * thresholds.blocks_pct / 100;
    let high = scene_count + m2_count;
    let mid = high + m1_count;
    if scene_count >= required {
        3
    } else if high >= required {
        2
    } else {
        i32::from(mid >= required)
    }
}

pub fn analyze_scene(
    previous: &[DecodedVector],
    current: &[DecodedVector],
    marker: i32,
    shape: VectorShape,
    gamma: f64,
    thresholds: SceneThresholds,
) -> SceneAnalysis {
    let luma = luma_map(previous, current, marker, gamma);
    SceneAnalysis {
        class: classify_scene(current, &luma.map, shape, thresholds),
        luma,
    }
}

pub fn avgluma_payload(analysis: &SceneAnalysis) -> Vec<u8> {
    let area = analysis.luma.map.len();
    let mut payload = vec![0; area.saturating_add(12)];
    write_i32_slice(&mut payload, 0, analysis.class);
    let copy_len = area.min(payload.len().saturating_sub(4));
    payload[4..4 + copy_len].copy_from_slice(&analysis.luma.map[..copy_len]);
    write_i32_slice(&mut payload, area, scaled_metric(analysis.luma.geomean));
    write_i32_slice(
        &mut payload,
        area.saturating_add(4),
        scaled_metric(analysis.luma.max),
    );
    payload
}

pub fn decode_avgluma_payload(payload: &[u8], area: usize) -> Option<AvglumaPayload> {
    Some(AvglumaPayload {
        class: read_i32_slice(payload, 0)?,
        geomean: f64::from(read_i32_slice(payload, area)?) / 1_000_000.0,
        max: f64::from(read_i32_slice(payload, area.checked_add(4)?)?) / 1_000_000.0,
    })
}

pub fn hdr_weight_from_average(average: f64, luminance: f64) -> f64 {
    let scale = if average < 0.081 {
        average / 4.5
    } else {
        ((average + 0.099) / 1.099).powf(20.0 / 9.0)
    };
    scale * luminance
}

pub fn pack_nv12(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: usize,
    height: usize,
    y_stride: usize,
    uv_stride: usize,
) -> Vec<u8> {
    let mut out = vec![0; width.saturating_mul(height).saturating_mul(3) / 2];
    for row in 0..height {
        copy_row(y, y_stride, &mut out, width, row, width);
    }
    let uv_base = width.saturating_mul(height);
    for row in 0..height / 2 {
        for col in 0..width / 2 {
            let src = row.saturating_mul(uv_stride).saturating_add(col);
            let dst = uv_base
                .saturating_add(row.saturating_mul(width))
                .saturating_add(col.saturating_mul(2));
            if dst + 1 < out.len() {
                out[dst] = u.get(src).copied().unwrap_or(0);
                out[dst + 1] = v.get(src).copied().unwrap_or(0);
            }
        }
    }
    out
}

pub fn encode_vector_payload(
    data: VectorData,
    previous: Option<&[DecodedVector]>,
    current: &[DecodedVector],
) -> Vec<u8> {
    let count = vector_count(data.grid).unwrap_or(0);
    let mut payload = vec![0; 0x48usize.saturating_add(count.saturating_mul(16))];
    write_i32_slice(&mut payload, 0x00, 16);
    write_i32_slice(&mut payload, 0x04, VECTOR_MAGIC);
    write_i32_slice(&mut payload, 0x08, data.flags);
    write_i32_slice(&mut payload, 0x0C, data.block.width);
    write_i32_slice(&mut payload, 0x10, data.block.height);
    write_i32_slice(&mut payload, 0x14, data.marker);
    write_i32_slice(&mut payload, 0x20, data.shape.width);
    write_i32_slice(&mut payload, 0x24, data.shape.height);
    write_i32_slice(&mut payload, 0x28, data.overlap_x);
    write_i32_slice(&mut payload, 0x2C, data.overlap_y);
    write_i32_slice(&mut payload, 0x30, data.grid.width);
    write_i32_slice(&mut payload, 0x34, data.grid.height);
    write_i32_slice(&mut payload, 0x38, data.selector);
    write_i32_slice(&mut payload, 0x3C, data.delta);

    let first = 0x40;
    let second = first + 4 + count.saturating_mul(8);
    if let Some(vectors) = previous {
        encode_region(&mut payload, first, count, vectors);
    }
    encode_region(&mut payload, second, count, current);
    payload
}

fn vector_count(shape: VectorShape) -> Option<usize> {
    usize::try_from(shape.width.max(0).checked_mul(shape.height.max(0))?).ok()
}

fn decode_region(payload: &[u8], offset: usize, count: usize) -> Option<Vec<DecodedVector>> {
    let expected = i32::try_from(2usize.checked_mul(count)?.checked_add(1)?).ok()?;
    if read_i32_slice(payload, offset)? != expected {
        return None;
    }
    let mut vectors = Vec::with_capacity(count);
    let mut cursor = offset + 4;
    for _ in 0..count {
        let raw0 = read_u32_slice(payload, cursor)?;
        let raw1 = read_u32_slice(payload, cursor + 4)?;
        let rotated = raw0.rotate_right(16);
        let parts = rotated.to_le_bytes();
        vectors.push(DecodedVector {
            dx: i16::from_le_bytes([parts[0], parts[1]]),
            dy: i16::from_le_bytes([parts[2], parts[3]]),
            score: raw1 & 0x00FF_FFFF,
            luma: (raw1 >> 24) as u8,
        });
        cursor += 8;
    }
    Some(vectors)
}

fn encode_region(payload: &mut [u8], offset: usize, count: usize, vectors: &[DecodedVector]) {
    let header = i32::try_from(count.saturating_mul(2).saturating_add(1)).unwrap_or(i32::MAX);
    write_i32_slice(payload, offset, header);
    for (index, vector) in vectors.iter().take(count).enumerate() {
        let cursor = offset
            .saturating_add(4)
            .saturating_add(index.saturating_mul(8));
        let raw0 = u32::from(u16::from_ne_bytes(vector.dy.to_ne_bytes()))
            | (u32::from(u16::from_ne_bytes(vector.dx.to_ne_bytes())) << 16);
        let raw1 = (u32::from(vector.luma) << 24) | (vector.score & 0x00FF_FFFF);
        write_u32_slice(payload, cursor, raw0);
        write_u32_slice(payload, cursor.saturating_add(4), raw1);
    }
}

fn read_i32_slice(data: &[u8], offset: usize) -> Option<i32> {
    Some(i32::from_le_bytes(
        data.get(offset..offset + 4)?.try_into().ok()?,
    ))
}

fn write_i32_slice(data: &mut [u8], offset: usize, value: i32) {
    if let Some(dst) = data.get_mut(offset..offset.saturating_add(4)) {
        dst.copy_from_slice(&value.to_le_bytes());
    }
}

fn write_u32_slice(data: &mut [u8], offset: usize, value: u32) {
    if let Some(dst) = data.get_mut(offset..offset.saturating_add(4)) {
        dst.copy_from_slice(&value.to_le_bytes());
    }
}

fn read_u32_slice(data: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        data.get(offset..offset + 4)?.try_into().ok()?,
    ))
}

#[allow(clippy::cast_possible_truncation)]
fn luma_byte(value: f64, gamma: f64) -> u8 {
    let scaled = (value.powf(gamma) * 255.0) as i32;
    u8::try_from(scaled.max(20) & 0xFF).unwrap_or(20)
}

fn max_f64(left: f64, right: f64) -> f64 {
    if left < right { right } else { left }
}

#[allow(clippy::cast_possible_truncation)]
fn scaled_metric(value: f64) -> i32 {
    (value * 1_000_000.0) as i32
}

fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).unwrap_or(u32::MAX))
}

fn copy_row(src: &[u8], stride: usize, dst: &mut [u8], width: usize, row: usize, len: usize) {
    let src_start = row.saturating_mul(stride);
    let dst_start = row.saturating_mul(width);
    let src = src.get(src_start..src_start.saturating_add(len));
    let dst = dst.get_mut(dst_start..dst_start.saturating_add(len));
    if let (Some(src), Some(dst)) = (src, dst) {
        dst.copy_from_slice(src);
    }
}

fn byte(raw: u64, shift: u32) -> i32 {
    ((raw >> shift) & 0xFF) as i32
}

fn word(raw: u64, shift: u32) -> i32 {
    ((raw >> shift) & 0xFFFF) as i32
}
