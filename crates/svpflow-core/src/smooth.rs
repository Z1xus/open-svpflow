#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::cell::RefCell;

use crate::metadata::VectorData;

type BlockEntry = (i32, usize, [i16; 32], [i16; 32]);

#[derive(Default)]
struct RenderScratch {
    channels: [Vec<i16>; 11],
    lanes: Vec<i32>,
    blocks: Vec<BlockEntry>,
}

thread_local! {
    static SCRATCH: RefCell<RenderScratch> = RefCell::new(RenderScratch::default());
}

const LOG2_STEP: [i8; 32] = [
    0, 1, 2, 2, -1, 3, 3, 3, -1, -1, -1, 4, -1, 4, -1, 4, -1, -1, -1, -1, -1, -1, -1, 5, -1, -1,
    -1, 5, -1, -1, -1, 5,
];
const VECTOR_BIAS: i32 = 1024;
const NEUTRAL: u16 = 1024;
const MAX_STEP: usize = 64;

mod sealed {
    pub trait Sealed {}

    impl Sealed for u8 {}

    impl Sealed for u16 {}
}

pub trait Sample:
    sealed::Sealed + Copy + Default + PartialOrd + Into<i32> + Into<u32> + Send + Sync + 'static
{
    fn new(value: i32) -> Self;
}

impl Sample for u8 {
    fn new(value: i32) -> Self {
        value as Self
    }
}

impl Sample for u16 {
    fn new(value: i32) -> Self {
        value.clamp(0, Self::MAX.into()) as Self
    }
}

#[derive(Clone, Copy, Default)]
pub struct BlockVector {
    pub x: i16,
    pub y: i16,
    pub sad: i32,
    pub luma: u8,
}

#[derive(Clone, Copy)]
pub struct FieldShape {
    pub flags: i32,
    pub block_w: i32,
    pub block_h: i32,
    pub pel: i32,
    pub width: i32,
    pub height: i32,
    pub overlap_x: i32,
    pub overlap_y: i32,
    pub grid_w: i32,
    pub grid_h: i32,
    pub selector: i32,
}

impl FieldShape {
    pub const fn from_data(data: &VectorData) -> Self {
        let overlap = data.origin();
        Self {
            flags: data.flags,
            block_w: data.block.width,
            block_h: data.block.height,
            pel: data.marker,
            width: data.shape.width,
            height: data.shape.height,
            overlap_x: overlap.width,
            overlap_y: overlap.height,
            grid_w: data.grid.width,
            grid_h: data.grid.height,
            selector: data.selector(),
        }
    }

    pub const fn blocks(&self) -> usize {
        let count = self.grid_w * self.grid_h;
        if count > 0 { count as usize } else { 0 }
    }

    pub const fn packed_width(&self) -> i32 {
        let span = (self.block_w - self.overlap_x) * self.grid_w + self.overlap_x;
        self.grid_w + if span < self.width { 1 } else { 0 }
    }

    pub const fn packed_height(&self) -> i32 {
        let span = (self.block_h - self.overlap_y) * self.grid_h + self.overlap_y;
        self.grid_h + if span < self.height { 1 } else { 0 }
    }
}

#[derive(Clone)]
pub struct VectorField {
    shape: FieldShape,
    forward: Vec<BlockVector>,
    backward: Vec<BlockVector>,
}

pub struct SceneLimits {
    pub blocks: i32,
    pub zero: i32,
    pub m1: i32,
    pub m2: i32,
    pub scene: i32,
}

impl VectorField {
    pub fn new(shape: FieldShape) -> Self {
        let count = shape.blocks();
        Self {
            shape,
            forward: vec![BlockVector::default(); count],
            backward: vec![BlockVector::default(); count],
        }
    }

    pub const fn shape(&self) -> &FieldShape {
        &self.shape
    }

    pub fn is_static(&self) -> bool {
        let count = self.shape.blocks();
        if count == 0 {
            return false;
        }
        let area = self.shape.block_w.saturating_mul(self.shape.block_h).max(1);
        let check = |set: &[BlockVector]| {
            set.iter()
                .take(count)
                .all(|v| v.x == 0 && v.y == 0 && v.sad <= area)
        };
        let fwd_ok = self.shape.flags & 2 == 0 || check(&self.forward);
        let bwd_ok = self.shape.flags & 1 == 0 || check(&self.backward);
        fwd_ok && bwd_ok
    }

    const fn set(&self, backward: bool) -> &Vec<BlockVector> {
        if backward {
            &self.backward
        } else {
            &self.forward
        }
    }

    pub fn cover_layout(&self, cover: i32) -> CoverLayout {
        let s = self.shape;
        let area = s.block_w * s.block_h;
        let scale = f64::from(cover) / 100.0;
        CoverLayout {
            grid: (s.grid_w, s.grid_h),
            size: (s.packed_width(), s.packed_height()),
            step: (s.block_w - s.overlap_x, s.block_h - s.overlap_y),
            block: (s.block_w, s.block_h),
            denom: s.pel << 8,
            area,
            scale: scale as f32,
            table: cover_table(area, scale),
        }
    }

    pub fn cover_vectors(&self, out: &mut Vec<i16>) {
        out.clear();
        for vector in self.backward.iter().chain(&self.forward) {
            out.extend([vector.x, vector.y]);
        }
    }

    pub fn update(&mut self, payload: &[u8]) -> bool {
        let read = |offset: usize| -> Option<i32> {
            Some(i32::from_le_bytes(
                payload.get(offset..offset + 4)?.try_into().ok()?,
            ))
        };
        if read(0) != Some(16) || read(4) != Some(160) || read(8) != Some(self.shape.flags) {
            return false;
        }
        let count = self.shape.blocks();
        let expected = i32::try_from(count * 2 + 1).unwrap_or(-1);
        let mut offset = 68 + 8 * count;
        if self.shape.flags & 1 != 0
            && (read(64) != Some(expected) || !decode_set(payload, 68, &mut self.backward))
        {
            return false;
        }
        if self.shape.flags & 2 != 0 {
            if read(offset) != Some(expected) {
                return false;
            }
            offset += 4;
            return decode_set(payload, offset, &mut self.forward);
        }
        true
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn luma_table(&self, gamma: f64) -> [u8; 511] {
        let denom = if self.shape.flags == 3 { 510.0 } else { 255.0 };
        std::array::from_fn(|sum| {
            let scaled = ((f64::from(sum as u16) / denom).powf(gamma) * 255.0) as i32;
            if scaled < 20 { 20 } else { scaled as u8 }
        })
    }

    pub fn avg_luma(&self, out: &mut [u8], table: &[u8; 511]) {
        let count = self.shape.blocks();
        for ((out, back), fwd) in out
            .iter_mut()
            .zip(&self.backward)
            .zip(&self.forward)
            .take(count)
        {
            *out = table[usize::from(back.luma) + usize::from(fwd.luma)];
        }
    }

    pub fn luma_geomean(&self) -> f64 {
        let denom = if self.shape.flags == 3 { 510.0 } else { 255.0 };
        let count = self.shape.blocks();
        let ln: [f64; 511] = std::array::from_fn(|sum| (sum as f64 / denom).ln());
        let sum = self
            .backward
            .iter()
            .zip(&self.forward)
            .take(count)
            .fold(0.0, |sum, (back, fwd)| {
                sum + ln[usize::from(back.luma) + usize::from(fwd.luma)]
            });
        (sum / count as f64).exp()
    }

    pub fn quality_map(&self, backward: bool, limits: &SceneLimits, luma: &[u8], out: &mut [u8]) {
        let count = self.shape.blocks();
        out[..count].fill(3);
        for ((out, vector), &luma) in out.iter_mut().zip(self.set(backward)).zip(luma).take(count) {
            let score = vector.sad.wrapping_mul(255) / i32::from(luma.max(1));
            if score < limits.zero {
                *out = 0xFF;
            } else if score < limits.m1 {
                *out = 0;
            } else if score < limits.m2 {
                *out = 1;
            } else if score < limits.scene {
                *out = 2;
            }
        }
    }

    pub fn quality(&self, backward: bool, limits: &SceneLimits, luma: &[u8]) -> i32 {
        let count = self.shape.blocks();
        if count == 0 {
            return 3;
        }
        let (mut considered, mut scene, mut m2, mut m1) = (0i32, 0, 0, 0);
        for (vector, &luma) in self.set(backward).iter().zip(luma).take(count) {
            let score = vector.sad.wrapping_mul(255) / i32::from(luma.max(1));
            if score < limits.zero {
                continue;
            }
            considered += 1;
            if score >= limits.scene {
                scene += 1;
            } else if score >= limits.m2 {
                m2 += 1;
            } else if score >= limits.m1 {
                m1 += 1;
            }
        }
        let required = limits.blocks.wrapping_mul(considered) / 100;
        if required <= scene {
            3
        } else if required <= scene + m2 {
            2
        } else {
            i32::from(required <= scene + m2 + m1)
        }
    }

    #[allow(clippy::cast_sign_loss)]
    pub fn pack(&self, backward: bool, xs: &mut [u16], ys: &mut [u16], width: i32, height: i32) {
        let s = &self.shape;
        let set = self.set(backward);
        let step_x = s.block_w - s.overlap_x;
        let step_y = s.block_h - s.overlap_y;
        for y in 0..height {
            let pos_y = y * step_y;
            let row = y.min(s.grid_h - 1) * s.grid_w;
            for x in 0..width {
                let pos_x = x * step_x;
                let index = (y * width + x) as usize;
                let vector = set[(row + x.min(s.grid_w - 1)) as usize];
                let mut vx = i32::from(vector.x);
                let mut vy = i32::from(vector.y);
                if !(-1023..=1023).contains(&vx) || !(-1023..=1023).contains(&vy) {
                    xs[index] = NEUTRAL;
                    ys[index] = NEUTRAL;
                    continue;
                }
                if s.selector == 0 {
                    vx = clamp_pack(vx, pos_x, s.block_w, s.width);
                    vy = clamp_pack(vy, pos_y, s.block_h, s.height);
                }
                xs[index] = (vx + 1024) as u16;
                ys[index] = (vy + 1024) as u16;
            }
        }
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::too_many_arguments,
        clippy::too_many_lines
    )]
    pub fn cover_mask(
        &self,
        scratch: &mut Vec<i32>,
        forward: bool,
        out: &mut [u8],
        cover: i32,
        time: i32,
        width: i32,
        height: i32,
    ) {
        let s = self.shape;
        let stride = (width + 2) as usize;
        let len = stride * (height + 2) as usize;
        scratch.clear();
        scratch.resize(len + 1, 0);
        let area = s.block_w * s.block_h;
        let step_x = s.block_w - s.overlap_x;
        let step_y = s.block_h - s.overlap_y;
        let denom = s.pel << 8;
        let pow2 = denom > 0 && denom.count_ones() == 1;
        let shift = denom.trailing_zeros();
        let scale = |value: i32| {
            if pow2 {
                (value + ((value >> 31) & (denom - 1))) >> shift
            } else {
                value / denom
            }
        };
        let (floor_x, floor_y) = (FloorDiv::new(step_x), FloorDiv::new(step_y));
        let set = if forward {
            &self.forward
        } else {
            &self.backward
        };
        let points = &mut scratch[..];
        let geometry = Scatter {
            width,
            height,
            stride,
            trash: len,
            step_x,
            step_y,
            block_w: s.block_w,
            block_h: s.block_h,
            time,
        };
        let grid_w = s.grid_w.max(0) as usize;
        let rows = set
            .chunks_exact(grid_w.max(1))
            .take(s.grid_h.max(0) as usize);
        if pow2 && floor_x.magic != 0 && floor_y.magic != 0 && time.unsigned_abs() <= 256 {
            for (by, row) in rows.enumerate() {
                geometry.fast_row(points, row, by as i32, shift, floor_x, floor_y);
            }
        } else {
            for (by, row) in rows.enumerate() {
                for (bx, vector) in row.iter().enumerate() {
                    let tx = scale(time * i32::from(vector.x));
                    let ty = scale(time * i32::from(vector.y));
                    geometry.add(
                        points,
                        bx as i32,
                        by as i32,
                        (tx, ty),
                        (floor_div(tx, step_x), floor_div(ty, step_y)),
                    );
                }
            }
        }
        let scale = f64::from(cover) / 100.0;
        let level = |covered: i32| cover_level(area, scale, covered);
        let table = cover_table(area, scale);
        let top = table.len() as i32 - 1;
        let width = width as usize;
        let mut column = vec![0i32; stride];
        let mut covered = vec![0i32; width];
        for (y, out) in out
            .chunks_exact_mut(width)
            .take(height as usize)
            .enumerate()
        {
            let above = &points[y * stride..(y + 1) * stride];
            let middle = &points[(y + 1) * stride..(y + 2) * stride];
            let below = &points[(y + 2) * stride..(y + 3) * stride];
            for (((sum, &a), &m), &b) in column.iter_mut().zip(above).zip(middle).zip(below) {
                *sum = a.wrapping_add(m).wrapping_add(b);
            }
            for (((covered, &l), &c), &r) in covered
                .iter_mut()
                .zip(&column[..width])
                .zip(&column[1..=width])
                .zip(&column[2..width + 2])
            {
                *covered = (l.wrapping_add(c).wrapping_add(r) >> 3).min(area);
            }
            for (out, &covered) in out.iter_mut().zip(&covered) {
                *out = if covered < 0 {
                    level(covered)
                } else {
                    table[covered.min(top) as usize]
                };
            }
        }
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn sad_mask(
        &self,
        backward: bool,
        out: &mut [u8],
        scale: f64,
        sharp: f64,
        width: i32,
        height: i32,
    ) {
        let s = &self.shape;
        let set = self.set(backward);
        let area = f64::from(s.block_h * s.block_w);
        let w = width as usize;
        for x in 0..width {
            for y in 0..height {
                let index = y as usize * w + x as usize;
                out[index] = if x >= s.grid_w {
                    out[index - 1]
                } else if y < s.grid_h {
                    let sad = set[(x + y * s.grid_w) as usize].sad;
                    let value = (f64::from(4 * sad) * scale / area).powf(sharp) * 255.0;
                    if value > 255.0 {
                        255
                    } else {
                        value as i32 as u8
                    }
                } else {
                    out[index - w]
                };
            }
        }
    }
}

fn decode_set(payload: &[u8], offset: usize, set: &mut [BlockVector]) -> bool {
    let Some(bytes) = payload.get(offset..offset + set.len() * 8) else {
        return false;
    };
    for (vector, raw) in set.iter_mut().zip(bytes.as_chunks::<8>().0) {
        let packed = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
        let extra = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]);
        *vector = BlockVector {
            x: (packed >> 16) as u16 as i16,
            y: packed as u16 as i16,
            sad: (extra & 0x00FF_FFFF) as i32,
            luma: (extra >> 24) as u8,
        };
    }
    true
}

const fn clamp_pack(value: i32, pos: i32, block: i32, limit: i32) -> i32 {
    if value + pos < 0 {
        -pos
    } else if value + pos + block > limit {
        let clamped = limit - pos - block;
        if clamped < 0 { 0 } else { clamped }
    } else {
        value
    }
}

#[derive(Clone, Copy)]
struct FloorDiv {
    magic: u64,
    bias: i32,
    offset: i32,
}

impl FloorDiv {
    const LIMIT: i32 = 1 << 17;

    const fn new(divisor: i32) -> Self {
        if divisor <= 0 || divisor > 128 {
            return Self {
                magic: 0,
                bias: 0,
                offset: 0,
            };
        }
        let bias = (Self::LIMIT + divisor - 1) / divisor;
        Self {
            magic: (1u64 << 32).div_ceil(divisor as u64),
            bias,
            offset: bias * divisor,
        }
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    #[inline]
    const fn apply(self, value: i32) -> i32 {
        ((((value + self.offset) as u32) as u64 * self.magic) >> 32) as i32 - self.bias
    }
}

#[derive(Clone, Copy)]
struct Scatter {
    width: i32,
    height: i32,
    stride: usize,
    trash: usize,
    step_x: i32,
    step_y: i32,
    block_w: i32,
    block_h: i32,
    time: i32,
}

impl Scatter {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    #[inline]
    fn cells(
        self,
        bx: i32,
        by: i32,
        (tx, ty): (i32, i32),
        (qx, qy): (i32, i32),
    ) -> [(u32, i32); 4] {
        let cx = bx + qx;
        let cy = by + qy;
        let rx = self.step_x * (qx + 1) - tx;
        let ry = self.step_y * (qy + 1) - ty;
        let (lx, ly) = (self.block_w - rx, self.block_h - ry);
        let targets = [(cx, cy), (cx + 1, cy), (cx + 1, cy + 1), (cx, cy + 1)];
        let values = [ry * rx, ry * lx, ly * lx, rx * ly];
        let mut cells = [(0u32, 0i32); 4];
        for ((cell, (x, y)), value) in cells.iter_mut().zip(targets).zip(values) {
            let inside = (x as u32) < self.width as u32 && (y as u32) < self.height as u32;
            let index = ((y + 1) as u32)
                .wrapping_mul(self.stride as u32)
                .wrapping_add((x + 1) as u32);
            *cell = (if inside { index } else { self.trash as u32 }, value);
        }
        cells
    }

    #[inline]
    fn add(self, points: &mut [i32], bx: i32, by: i32, t: (i32, i32), q: (i32, i32)) {
        for (index, value) in self.cells(bx, by, t, q) {
            let cell = &mut points[index as usize];
            *cell = cell.wrapping_add(value);
        }
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn fast_row(
        self,
        points: &mut [i32],
        row: &[BlockVector],
        by: i32,
        shift: u32,
        floor_x: FloorDiv,
        floor_y: FloorDiv,
    ) {
        const LANES: usize = 16;
        let round = (1i32 << shift) - 1;
        let scale = |value: i32| (value + ((value >> 31) & round)) >> shift;
        let stride = self.stride as i32;
        let (last_x, last_y) = (
            (self.width.max(1) - 1) as u32,
            (self.height.max(1) - 1) as u32,
        );
        let add = |points: &mut [i32], (at, top, bottom): (usize, i32, i32)| {
            points[at] = points[at].wrapping_add(top);
            points[at + self.stride] = points[at + self.stride].wrapping_add(bottom);
        };
        let mut pending = (0, 0, 0);
        let lanes = row.len().div_ceil(LANES) * LANES;
        let mut vx = vec![0i32; lanes];
        let mut vy = vec![0i32; lanes];
        for ((x, y), vector) in vx.iter_mut().zip(vy.iter_mut()).zip(row) {
            *x = i32::from(vector.x);
            *y = i32::from(vector.y);
        }
        for (chunk_index, ((chunk, vx), vy)) in row
            .chunks(LANES)
            .zip(vx.as_chunks::<LANES>().0)
            .zip(vy.as_chunks::<LANES>().0)
            .enumerate()
        {
            let base = (chunk_index * LANES) as i32;
            let mut index = [0i32; LANES];
            let mut inside = [false; LANES];
            let mut values = [[0i32; LANES]; 4];
            for lane in 0..LANES {
                let tx = scale(self.time * vx[lane]);
                let ty = scale(self.time * vy[lane]);
                let (qx, qy) = (floor_x.apply(tx), floor_y.apply(ty));
                let cx = base + lane as i32 + qx;
                let cy = by + qy;
                let rx = self.step_x * (qx + 1) - tx;
                let ry = self.step_y * (qy + 1) - ty;
                let (lx, ly) = (self.block_w - rx, self.block_h - ry);
                values[0][lane] = ry * rx;
                values[1][lane] = ry * lx;
                values[2][lane] = ly * lx;
                values[3][lane] = rx * ly;
                inside[lane] = (cx as u32) < last_x && (cy as u32) < last_y;
                index[lane] = (cy + 1) * stride + cx + 1;
            }
            for lane in 0..chunk.len() {
                if inside[lane] {
                    let at = index[lane] as usize;
                    let (mut top, mut bottom) = (values[0][lane], values[3][lane]);
                    if at == pending.0 {
                        top = top.wrapping_add(pending.1);
                        bottom = bottom.wrapping_add(pending.2);
                    } else {
                        add(points, pending);
                    }
                    add(points, (at, top, bottom));
                    pending = (at + 1, values[1][lane], values[2][lane]);
                } else {
                    let tx = scale(self.time * vx[lane]);
                    let ty = scale(self.time * vy[lane]);
                    let q = (floor_x.apply(tx), floor_y.apply(ty));
                    self.add(points, base + lane as i32, by, (tx, ty), q);
                }
            }
        }
        add(points, pending);
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn cover_level(area: i32, scale: f64, covered: i32) -> u8 {
    let value = (f64::from(area - covered) * scale * 256.0 / f64::from(area)) as i32;
    if value > 255 { 255 } else { value as u8 }
}

fn cover_table(area: i32, scale: f64) -> Vec<u8> {
    (0..=area.max(0))
        .map(|covered| cover_level(area, scale, covered))
        .collect()
}

pub struct CoverLayout {
    pub grid: (i32, i32),
    pub size: (i32, i32),
    pub step: (i32, i32),
    pub block: (i32, i32),
    pub denom: i32,
    pub area: i32,
    pub scale: f32,
    pub table: Vec<u8>,
}

const fn floor_div(value: i32, divisor: i32) -> i32 {
    if value < 0 {
        (value - divisor + 1) / divisor
    } else {
        value / divisor
    }
}

fn log2_step(step: i32) -> i32 {
    usize::try_from(step - 1)
        .ok()
        .and_then(|index| LOG2_STEP.get(index))
        .map_or(-1, |&value| i32::from(value))
}

#[derive(Clone, Copy)]
pub struct RenderShape {
    pub width: i32,
    pub height: i32,
    pub step_x: i32,
    pub step_y: i32,
    pub grid_w: i32,
    pub grid_h: i32,
    pub origin_x: i32,
    pub origin_y: i32,
    pub pel: i32,
    pub blend: f64,
    pub chroma: (i32, i32),
    pub depth: u32,
}

pub struct Renderer {
    shape: RenderShape,
    shifts: [i32; 4],
    time: i32,
    time_blend: i32,
    fast_flags: u8,
    x_luma: Vec<[i16; 2]>,
    y_luma: Vec<i16>,
    x_chroma: Vec<[i16; 2]>,
    y_chroma: Vec<i16>,
}

#[derive(Clone, Copy)]
pub struct Plane<'a, T = u8> {
    pub data: &'a [T],
    pub pitch: usize,
    pub slack: usize,
}

pub struct PlaneMut<'a, T = u8> {
    pub data: &'a mut [T],
    pub pitch: usize,
}

#[derive(Clone, Copy)]
pub struct Frame<'a, T = u8> {
    pub y: Plane<'a, T>,
    pub u: Plane<'a, T>,
    pub v: Plane<'a, T>,
}

pub struct FrameMut<'a, T = u8> {
    pub y: PlaneMut<'a, T>,
    pub u: PlaneMut<'a, T>,
    pub v: PlaneMut<'a, T>,
}

#[derive(Clone, Copy)]
pub struct PackedVectors<'a> {
    pub fwd_x: &'a [u16],
    pub fwd_y: &'a [u16],
    pub bwd_x: &'a [u16],
    pub bwd_y: &'a [u16],
    pub next_fwd_x: &'a [u16],
    pub next_fwd_y: &'a [u16],
    pub prev_bwd_x: &'a [u16],
    pub prev_bwd_y: &'a [u16],
    pub cover_bwd: &'a [u8],
    pub cover_fwd: &'a [u8],
    pub sad: &'a [u8],
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Algo {
    Fast { next: bool },
    FastSad { next: bool },
    NoMask { median: bool },
    Normal { simple: bool },
    NoMaskSad { median: bool },
    NormalSad { simple: bool },
    Extended,
    ExtendedSad,
    Fill,
}

struct Pass<'a> {
    width: i32,
    height: i32,
    step_x: i32,
    step_y: i32,
    origin_x: i32,
    origin_y: i32,
    div_x: i32,
    div_y: i32,
    shift_x: u32,
    shift_y: u32,
    x_weights: &'a [[i16; 2]],
    y_weights: &'a [i16],
    pitch: usize,
    blocks: std::ops::Range<i32>,
    row_base: i32,
}

impl Pass<'_> {
    fn block_rows(&self, interp: bool, grid_h: i32, by: i32) -> (i32, i32) {
        let py0 = if by < 0 {
            0
        } else {
            self.origin_y + by * self.step_y
        };
        let mut rows = if py0 + self.step_y < self.height {
            if by == -1 { self.origin_y } else { self.step_y }
        } else {
            self.height - py0
        };
        if !interp && by == grid_h - 1 {
            rows = self.height - py0;
        }
        (py0, rows)
    }

    fn rows_of(&self, interp: bool, grid_h: i32) -> std::ops::Range<i32> {
        let edge = |by: i32| {
            if by >= grid_h {
                self.height
            } else {
                self.block_rows(interp, grid_h, by.max(-1))
                    .0
                    .min(self.height)
            }
        };
        let start = edge(self.blocks.start);
        start..edge(self.blocks.end).max(start)
    }
}

#[allow(clippy::cast_possible_truncation)]
fn weight_pairs(count: i32, total: i32, scale: f64) -> Vec<[i16; 2]> {
    (0..count.max(0))
        .map(|k| {
            let a = (f64::from(k) * scale - 0.5).ceil() as i32 as i16;
            let b = (f64::from(total - k) * scale - 0.5).ceil() as i32 as i16;
            [b, a]
        })
        .collect()
}

#[allow(clippy::cast_possible_truncation)]
fn weights(count: i32, scale: f64) -> Vec<i16> {
    (0..=count.max(0))
        .map(|k| (f64::from(k) * scale - 0.5).ceil() as i32 as i16)
        .collect()
}

impl Renderer {
    pub fn new(shape: RenderShape) -> Self {
        let (cx, cy) = shape.chroma;
        let shifts = [
            log2_step(shape.step_x),
            log2_step(shape.step_y),
            log2_step(shape.step_x / cx),
            log2_step(shape.step_y / cy),
        ];
        let pow = |shift: i32| 2f64.powf(f64::from(shift));
        let x_luma = weight_pairs(
            shape.step_x,
            shape.step_x,
            pow(shifts[0]) / f64::from(shape.step_x),
        );
        let y_luma = weights(shape.step_y, pow(shifts[1]) / f64::from(shape.step_y));
        let x_chroma = weight_pairs(
            shape.step_x / cx,
            shape.step_x / cx,
            pow(shifts[2]) * f64::from(cx) / f64::from(shape.step_x),
        );
        let y_chroma = weights(
            shape.step_y / cy,
            pow(shifts[3]) * f64::from(cy) / f64::from(shape.step_y),
        );
        let mut renderer = Self {
            shape,
            shifts,
            time: -1,
            time_blend: 0,
            fast_flags: 0,
            x_luma,
            y_luma,
            x_chroma,
            y_chroma,
        };
        renderer.set_time(0);
        renderer
    }

    pub const fn set_fast(&mut self, fast: u8, fast_cpu: bool) {
        self.fast_flags = if fast >= 1 { 1 } else { 0 } | if fast_cpu { 2 } else { 0 };
    }

    const fn fast_occ(&self) -> bool {
        self.fast_flags & 1 != 0
    }

    const fn fast_cpu(&self) -> bool {
        self.fast_flags & 2 != 0
    }

    pub const fn grid_h(&self) -> i32 {
        self.shape.grid_h
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    pub fn set_time(&mut self, time: i32) {
        if self.time == time {
            return;
        }
        self.time = time;
        let rev = 256 - time;
        self.time_blend = if time <= 126 {
            (f64::from(time) * self.shape.blend) as i32
        } else {
            (256.0 - f64::from(rev) * self.shape.blend) as i32
        };
    }

    #[allow(clippy::cast_sign_loss)]
    fn pass(
        &self,
        chroma: bool,
        interp: bool,
        pitch: usize,
        blocks: std::ops::Range<i32>,
    ) -> Pass<'_> {
        let s = &self.shape;
        let shift = |i: usize| self.shifts[i].clamp(0, 31) as u32;
        let mut pass = if chroma {
            let (cx, cy) = s.chroma;
            Pass {
                width: s.width / cx,
                height: s.height / cy,
                step_x: s.step_x / cx,
                step_y: s.step_y / cy,
                origin_x: if interp { s.origin_x / cx } else { 0 },
                origin_y: if interp { s.origin_y / cy } else { 0 },
                div_x: cx,
                div_y: cy,
                shift_x: shift(2),
                shift_y: shift(3),
                x_weights: &self.x_chroma,
                y_weights: &self.y_chroma,
                pitch,
                blocks,
                row_base: 0,
            }
        } else {
            Pass {
                width: s.width,
                height: s.height,
                step_x: s.step_x,
                step_y: s.step_y,
                origin_x: if interp { s.origin_x } else { 0 },
                origin_y: if interp { s.origin_y } else { 0 },
                div_x: 1,
                div_y: 1,
                shift_x: shift(0),
                shift_y: shift(1),
                x_weights: &self.x_luma,
                y_weights: &self.y_luma,
                pitch,
                blocks,
                row_base: 0,
            }
        };
        pass.row_base = pass.rows_of(interp, s.grid_h).start;
        pass
    }

    pub fn band_rows(
        &self,
        interp: bool,
        blocks: std::ops::Range<i32>,
    ) -> [std::ops::Range<usize>; 2] {
        [false, true].map(|chroma| {
            let rows = self
                .pass(chroma, interp, 0, blocks.clone())
                .rows_of(interp, self.shape.grid_h);
            usize::try_from(rows.start).unwrap_or(0)..usize::try_from(rows.end).unwrap_or(0)
        })
    }

    pub fn fill<T: Sample>(&self, dst: &mut FrameMut<'_, T>) {
        let h = self.shape.height;
        self.fill_rows(dst, [0..h as usize, 0..(h / self.shape.chroma.1) as usize]);
    }

    fn fill_rows<T: Sample>(&self, dst: &mut FrameMut<'_, T>, rows: [std::ops::Range<usize>; 2]) {
        let w = self.shape.width as usize;
        let cw = (self.shape.width / self.shape.chroma.0) as usize;
        let [luma, chroma] = rows;
        let shift = self.shape.depth - 8;
        let (gray, mid) = (T::new(0x7F << shift), T::new(0x80 << shift));
        for (plane, pw, ph, value) in [
            (&mut dst.y, w, luma.len(), gray),
            (&mut dst.u, cw, chroma.len(), mid),
            (&mut dst.v, cw, chroma.len(), mid),
        ] {
            for row in 0..ph {
                plane.data[row * plane.pitch..row * plane.pitch + pw].fill(value);
            }
        }
    }

    pub fn render<T: Sample>(
        &self,
        algo: Algo,
        interp: bool,
        dst: &mut FrameMut<'_, T>,
        next: Frame<'_, T>,
        current: Frame<'_, T>,
        vectors: &PackedVectors<'_>,
    ) {
        self.draw(algo, interp, dst, next, current, vectors);
        if size_of::<T>() > 1 && self.shape.depth < 16 {
            let top = T::new((1 << self.shape.depth) - 1);
            for plane in [&mut dst.y, &mut dst.u, &mut dst.v] {
                for sample in plane.data.iter_mut().filter(|sample| **sample > top) {
                    *sample = top;
                }
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    fn draw<T: Sample>(
        &self,
        algo: Algo,
        interp: bool,
        dst: &mut FrameMut<'_, T>,
        next: Frame<'_, T>,
        current: Frame<'_, T>,
        vectors: &PackedVectors<'_>,
    ) {
        let blocks = -1..self.shape.grid_h;
        if !self.split() {
            #[cfg(not(target_arch = "wasm32"))]
            if algo != Algo::Fill && self.shape.grid_h >= 16 {
                let (_guard, active) = crate::threads::enter_busy();
                let workers = crate::threads::workers();
                let bands = if active * 2 <= workers && workers >= 4 {
                    (workers / active.max(1))
                        .min(4)
                        .min((self.shape.grid_h as usize) / 8)
                } else {
                    1
                };
                if bands >= 2 {
                    let grid_h = self.shape.grid_h;
                    let (yp, up, vp) = (dst.y.pitch, dst.u.pitch, dst.v.pitch);
                    let mut y_rem = &mut *dst.y.data;
                    let mut u_rem = &mut *dst.u.data;
                    let mut v_rem = &mut *dst.v.data;
                    let mut items = Vec::with_capacity(bands);
                    for band in 0..bands {
                        let b0 = if band == 0 {
                            -1
                        } else {
                            (band as i32 * grid_h) / bands as i32
                        };
                        let b1 = if band + 1 == bands {
                            grid_h
                        } else {
                            ((band as i32 + 1) * grid_h) / bands as i32
                        };
                        let range = b0..b1;
                        let [y_rows, uv_rows] = self.band_rows(interp, range.clone());
                        let (y_slice, u_slice, v_slice) = if band + 1 == bands {
                            (
                                std::mem::take(&mut y_rem),
                                std::mem::take(&mut u_rem),
                                std::mem::take(&mut v_rem),
                            )
                        } else {
                            let y_take = (y_rows.len() * yp).min(y_rem.len());
                            let u_take = (uv_rows.len() * up).min(u_rem.len());
                            let v_take = (uv_rows.len() * vp).min(v_rem.len());
                            let (ys, yr) = y_rem.split_at_mut(y_take);
                            let (us, ur) = u_rem.split_at_mut(u_take);
                            let (vs, vr) = v_rem.split_at_mut(v_take);
                            y_rem = yr;
                            u_rem = ur;
                            v_rem = vr;
                            (ys, us, vs)
                        };
                        items.push((
                            range,
                            FrameMut {
                                y: PlaneMut {
                                    data: y_slice,
                                    pitch: yp,
                                },
                                u: PlaneMut {
                                    data: u_slice,
                                    pitch: up,
                                },
                                v: PlaneMut {
                                    data: v_slice,
                                    pitch: vp,
                                },
                            },
                        ));
                    }
                    crate::threads::scope(|scope| {
                        for (range, mut band) in items {
                            scope.spawn(move |_| {
                                self.render_band(
                                    algo, interp, &mut band, next, current, vectors, range,
                                );
                            });
                        }
                    });
                    return;
                }
                self.render_band(algo, interp, dst, next, current, vectors, blocks);
                return;
            }
            self.render_band(algo, interp, dst, next, current, vectors, blocks);
            return;
        }
        if algo == Algo::Fill {
            self.fill(dst);
            return;
        }
        let pass = self.pass(false, interp, next.y.pitch, blocks.clone());
        let mut planes = [(&mut dst.y, next.y, current.y)];
        self.render_algo(algo, interp, &pass, &mut planes, vectors, None);
        let wide = |plane: Plane<'_, T>| self.widen(plane, next.y);
        let (next_u, next_v, current_u, current_v) =
            (wide(next.u), wide(next.v), wide(current.u), wide(current.v));
        let full = dst.y.data.len();
        let (mut u, mut v) = (vec![T::default(); full], vec![T::default(); full]);
        let pitch = dst.y.pitch;
        let plane = |data| Plane {
            data,
            pitch: next.y.pitch,
            slack: 0,
        };
        {
            let mut u = PlaneMut {
                data: &mut u,
                pitch,
            };
            let mut v = PlaneMut {
                data: &mut v,
                pitch,
            };
            let pass = self.pass(false, interp, next.y.pitch, blocks);
            let mut planes = [
                (&mut u, plane(&next_u), plane(&current_u)),
                (&mut v, plane(&next_v), plane(&current_v)),
            ];
            let aux = Some((next.y, current.y));
            self.render_algo(algo, interp, &pass, &mut planes, vectors, aux);
        }
        self.narrow(&u, pitch, &mut dst.u);
        self.narrow(&v, pitch, &mut dst.v);
    }

    fn split(&self) -> bool {
        let s = &self.shape;
        let (cx, cy) = s.chroma;
        s.step_x % cx != 0 || s.step_y % cy != 0 || s.origin_x % cx != 0 || s.origin_y % cy != 0
    }

    #[allow(clippy::cast_sign_loss)]
    fn widen<T: Sample>(&self, plane: Plane<'_, T>, luma: Plane<'_, T>) -> Vec<T> {
        let (cx, cy) = (self.shape.chroma.0 as usize, self.shape.chroma.1 as usize);
        let rows = plane.data.len().div_ceil(plane.pitch.max(1)).max(1);
        let pitch = luma.pitch.max(1);
        let mut wide = vec![T::default(); luma.data.len()];
        for (r, rows_out) in wide.chunks_mut(pitch * cy).enumerate() {
            let start = r.min(rows - 1) * plane.pitch;
            let source = &plane.data[start..plane.data.len().min(start + plane.pitch)];
            let (first, rest) = rows_out.split_at_mut(pitch.min(rows_out.len()));
            let last = source.len() - 1;
            for (c, pixels) in first.chunks_mut(cx).enumerate() {
                pixels.fill(source[c.min(last)]);
            }
            for row in rest.chunks_mut(pitch) {
                row.copy_from_slice(&first[..row.len()]);
            }
        }
        wide
    }

    #[allow(clippy::cast_sign_loss)]
    fn narrow<T: Sample>(&self, wide: &[T], pitch: usize, out: &mut PlaneMut<'_, T>) {
        let (cx, cy) = (self.shape.chroma.0 as usize, self.shape.chroma.1 as usize);
        let (width, height) = (
            (self.shape.width / self.shape.chroma.0) as usize,
            (self.shape.height / self.shape.chroma.1) as usize,
        );
        let half = (cx * cy / 2) as u32;
        for r in 0..height {
            for c in 0..width {
                let mut total = half;
                for dy in 0..cy {
                    for dx in 0..cx {
                        let sample: u32 = wide[(r * cy + dy) * pitch + c * cx + dx].into();
                        total += sample;
                    }
                }
                out.data[r * out.pitch + c] = T::new((total / (cx * cy) as u32) as i32);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render_band<T: Sample>(
        &self,
        algo: Algo,
        interp: bool,
        dst: &mut FrameMut<'_, T>,
        next: Frame<'_, T>,
        current: Frame<'_, T>,
        vectors: &PackedVectors<'_>,
        blocks: std::ops::Range<i32>,
    ) {
        if algo == Algo::Fill {
            self.fill_rows(dst, self.band_rows(interp, blocks));
            return;
        }
        for chroma in [false, true] {
            let pitch = if chroma { next.u.pitch } else { next.y.pitch };
            let pass = self.pass(chroma, interp, pitch, blocks.clone());
            if chroma {
                let mut planes = [
                    (&mut dst.u, next.u, current.u),
                    (&mut dst.v, next.v, current.v),
                ];
                self.render_algo(
                    algo,
                    interp,
                    &pass,
                    &mut planes,
                    vectors,
                    Some((next.y, current.y)),
                );
            } else {
                let mut planes = [(&mut dst.y, next.y, current.y)];
                self.render_algo(algo, interp, &pass, &mut planes, vectors, None);
            }
        }
    }

    fn fill_channel(out: &mut Vec<i16>, raw: &[u16], time: i32, div: i32) {
        out.resize(raw.len(), 0);
        match div {
            1 => {
                for (dst, &value) in out.iter_mut().zip(raw) {
                    *dst = ((i32::from(value) - VECTOR_BIAS) * time / 256) as i16;
                }
            }
            2 => {
                for (dst, &value) in out.iter_mut().zip(raw) {
                    *dst = ((i32::from(value) - VECTOR_BIAS) * time / 256 / 2) as i16;
                }
            }
            _ => {
                for (dst, &value) in out.iter_mut().zip(raw) {
                    *dst = ((i32::from(value) - VECTOR_BIAS) * time / 256 / div) as i16;
                }
            }
        }
    }

    fn fill_mask(out: &mut Vec<i16>, raw: &[u8]) {
        out.resize(raw.len(), 0);
        for (dst, &value) in out.iter_mut().zip(raw) {
            *dst = i16::from(value);
        }
    }

    #[allow(clippy::too_many_lines)]
    fn render_algo<T: Sample>(
        &self,
        algo: Algo,
        interp: bool,
        pass: &Pass<'_>,
        planes: &mut [(&mut PlaneMut<'_, T>, Plane<'_, T>, Plane<'_, T>)],
        v: &PackedVectors<'_>,
        aux: Option<(Plane<'_, T>, Plane<'_, T>)>,
    ) {
        let t = self.time;
        let rt = 256 - t;
        let tb = self.time_blend;
        let rtb = 256 - tb;
        let (fwd, rev) = (t, rt);
        let weights = row::Weights {
            t,
            rt,
            tb,
            rtb,
            wrap: (256 << self.shape.depth) - 1,
        };
        let use_pair = self.fast_cpu() && interp && pass.width >= 4;
        SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            let RenderScratch {
                channels,
                lanes,
                blocks,
            } = &mut *scratch;
            let [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10] = channels;
            match algo {
                Algo::Fast { next } => {
                    if next {
                        Self::fill_channel(c0, v.fwd_x, fwd, pass.div_x);
                        Self::fill_channel(c1, v.fwd_y, fwd, pass.div_y);
                    } else {
                        Self::fill_channel(c0, v.bwd_x, fwd, pass.div_x);
                        Self::fill_channel(c1, v.bwd_y, fwd, pass.div_y);
                    }
                    if use_pair {
                        self.walk_pair(
                            pass,
                            [c0.as_slice(), c1.as_slice()],
                            false,
                            aux,
                            planes,
                            lanes,
                            blocks,
                            |out, p, base, n, l| row::fast_pair(out, p, next, base, n, l),
                        );
                    } else {
                        self.walk(
                            pass,
                            interp,
                            [c0.as_slice(), c1.as_slice()],
                            [None; 2],
                            false,
                            false,
                            aux,
                            planes,
                            lanes,
                            blocks,
                            |out, p, base, n, l| {
                                row::fast(out, p, next, base, n, l);
                            },
                        );
                    }
                }
                Algo::NoMask { median } => {
                    Self::fill_channel(c0, v.bwd_x, fwd, pass.div_x);
                    Self::fill_channel(c1, v.bwd_y, fwd, pass.div_y);
                    Self::fill_channel(c2, v.fwd_x, rev, pass.div_x);
                    Self::fill_channel(c3, v.fwd_y, rev, pass.div_y);
                    if use_pair {
                        self.walk_pair(
                            pass,
                            [c0.as_slice(), c1.as_slice(), c2.as_slice(), c3.as_slice()],
                            false,
                            aux,
                            planes,
                            lanes,
                            blocks,
                            |out, p, base, n, l| {
                                row::no_mask_pair(out, p, base, n, l, &weights, median);
                            },
                        );
                    } else {
                        self.walk(
                            pass,
                            interp,
                            [c0.as_slice(), c1.as_slice(), c2.as_slice(), c3.as_slice()],
                            [None; 4],
                            false,
                            false,
                            aux,
                            planes,
                            lanes,
                            blocks,
                            |out, p, base, n, l| {
                                row::no_mask(out, p, base, n, l, &weights, median);
                            },
                        );
                    }
                }
                Algo::FastSad { next } => {
                    if next {
                        Self::fill_channel(c0, v.fwd_x, fwd, pass.div_x);
                        Self::fill_channel(c1, v.fwd_y, fwd, pass.div_y);
                    } else {
                        Self::fill_channel(c0, v.bwd_x, fwd, pass.div_x);
                        Self::fill_channel(c1, v.bwd_y, fwd, pass.div_y);
                    }
                    Self::fill_mask(c2, v.sad);
                    if use_pair {
                        self.walk_pair(
                            pass,
                            [c0.as_slice(), c1.as_slice(), c2.as_slice()],
                            false,
                            aux,
                            planes,
                            lanes,
                            blocks,
                            |out, p, base, n, l| row::fast_sad_pair(out, p, next, base, n, l),
                        );
                    } else {
                        self.walk(
                            pass,
                            interp,
                            [c0.as_slice(), c1.as_slice(), c2.as_slice()],
                            [None, None, Some(v.sad)],
                            true,
                            false,
                            aux,
                            planes,
                            lanes,
                            blocks,
                            |out, p, base, n, l| {
                                row::fast_sad(out, p, next, base, n, l);
                            },
                        );
                    }
                }
                Algo::NoMaskSad { median } => {
                    Self::fill_channel(c0, v.bwd_x, fwd, pass.div_x);
                    Self::fill_channel(c1, v.bwd_y, fwd, pass.div_y);
                    Self::fill_channel(c2, v.fwd_x, rev, pass.div_x);
                    Self::fill_channel(c3, v.fwd_y, rev, pass.div_y);
                    Self::fill_mask(c4, v.sad);
                    if use_pair {
                        self.walk_pair(
                            pass,
                            [
                                c0.as_slice(),
                                c1.as_slice(),
                                c2.as_slice(),
                                c3.as_slice(),
                                c4.as_slice(),
                            ],
                            false,
                            aux,
                            planes,
                            lanes,
                            blocks,
                            |out, p, base, n, l| {
                                row::no_mask_sad_pair(out, p, base, n, l, &weights, median);
                            },
                        );
                    } else {
                        self.walk(
                            pass,
                            interp,
                            [
                                c0.as_slice(),
                                c1.as_slice(),
                                c2.as_slice(),
                                c3.as_slice(),
                                c4.as_slice(),
                            ],
                            [None, None, None, None, Some(v.sad)],
                            false,
                            false,
                            aux,
                            planes,
                            lanes,
                            blocks,
                            |out, p, base, n, l| {
                                row::no_mask_sad(out, p, base, n, l, &weights, median);
                            },
                        );
                    }
                }
                Algo::NormalSad { simple } => {
                    Self::fill_channel(c0, v.bwd_x, fwd, pass.div_x);
                    Self::fill_channel(c1, v.bwd_y, fwd, pass.div_y);
                    Self::fill_channel(c2, v.fwd_x, rev, pass.div_x);
                    Self::fill_channel(c3, v.fwd_y, rev, pass.div_y);
                    Self::fill_mask(c4, v.cover_fwd);
                    Self::fill_mask(c5, v.cover_bwd);
                    Self::fill_mask(c6, v.sad);
                    if use_pair {
                        self.walk_pair(
                            pass,
                            [
                                c0.as_slice(),
                                c1.as_slice(),
                                c2.as_slice(),
                                c3.as_slice(),
                                c4.as_slice(),
                                c5.as_slice(),
                                c6.as_slice(),
                            ],
                            false,
                            aux,
                            planes,
                            lanes,
                            blocks,
                            |out, p, base, n, l| {
                                row::normal_sad_pair(out, p, base, n, l, &weights, simple);
                            },
                        );
                    } else {
                        self.walk(
                            pass,
                            interp,
                            [
                                c0.as_slice(),
                                c1.as_slice(),
                                c2.as_slice(),
                                c3.as_slice(),
                                c4.as_slice(),
                                c5.as_slice(),
                                c6.as_slice(),
                            ],
                            [
                                None,
                                None,
                                None,
                                None,
                                Some(v.cover_fwd),
                                Some(v.cover_bwd),
                                Some(v.sad),
                            ],
                            false,
                            false,
                            aux,
                            planes,
                            lanes,
                            blocks,
                            |out, p, base, n, l| {
                                row::normal_sad(out, p, base, n, l, &weights, simple);
                            },
                        );
                    }
                }
                Algo::Normal { simple } => {
                    Self::fill_channel(c0, v.bwd_x, fwd, pass.div_x);
                    Self::fill_channel(c1, v.bwd_y, fwd, pass.div_y);
                    Self::fill_channel(c2, v.fwd_x, rev, pass.div_x);
                    Self::fill_channel(c3, v.fwd_y, rev, pass.div_y);
                    Self::fill_mask(c4, v.cover_fwd);
                    Self::fill_mask(c5, v.cover_bwd);
                    if use_pair {
                        self.walk_pair(
                            pass,
                            [
                                c0.as_slice(),
                                c1.as_slice(),
                                c2.as_slice(),
                                c3.as_slice(),
                                c4.as_slice(),
                                c5.as_slice(),
                            ],
                            false,
                            aux,
                            planes,
                            lanes,
                            blocks,
                            |out, p, base, n, l| {
                                row::normal_pair(out, p, base, n, l, &weights, simple);
                            },
                        );
                    } else {
                        self.walk(
                            pass,
                            interp,
                            [
                                c0.as_slice(),
                                c1.as_slice(),
                                c2.as_slice(),
                                c3.as_slice(),
                                c4.as_slice(),
                                c5.as_slice(),
                            ],
                            [None, None, None, None, Some(v.cover_fwd), Some(v.cover_bwd)],
                            false,
                            false,
                            aux,
                            planes,
                            lanes,
                            blocks,
                            |out, p, base, n, l| row::normal(out, p, base, n, l, &weights, simple),
                        );
                    }
                }
                Algo::Extended => {
                    Self::fill_channel(c0, v.bwd_x, fwd, pass.div_x);
                    Self::fill_channel(c1, v.bwd_y, fwd, pass.div_y);
                    Self::fill_channel(c2, v.fwd_x, rev, pass.div_x);
                    Self::fill_channel(c3, v.fwd_y, rev, pass.div_y);
                    Self::fill_channel(c4, v.prev_bwd_x, fwd, pass.div_x);
                    Self::fill_channel(c5, v.prev_bwd_y, fwd, pass.div_y);
                    Self::fill_channel(c6, v.next_fwd_x, rev, pass.div_x);
                    Self::fill_channel(c7, v.next_fwd_y, rev, pass.div_y);
                    Self::fill_mask(c8, v.cover_fwd);
                    Self::fill_mask(c9, v.cover_bwd);
                    if use_pair {
                        self.walk_pair(
                            pass,
                            [
                                c0.as_slice(),
                                c1.as_slice(),
                                c2.as_slice(),
                                c3.as_slice(),
                                c4.as_slice(),
                                c5.as_slice(),
                                c6.as_slice(),
                                c7.as_slice(),
                                c8.as_slice(),
                                c9.as_slice(),
                            ],
                            self.fast_occ(),
                            aux,
                            planes,
                            lanes,
                            blocks,
                            |out, p, base, n, l| row::extended_pair(out, p, base, n, l, &weights),
                        );
                    } else {
                        self.walk(
                            pass,
                            interp,
                            [
                                c0.as_slice(),
                                c1.as_slice(),
                                c2.as_slice(),
                                c3.as_slice(),
                                c4.as_slice(),
                                c5.as_slice(),
                                c6.as_slice(),
                                c7.as_slice(),
                                c8.as_slice(),
                                c9.as_slice(),
                            ],
                            [
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                Some(v.cover_fwd),
                                Some(v.cover_bwd),
                            ],
                            false,
                            self.fast_occ(),
                            aux,
                            planes,
                            lanes,
                            blocks,
                            |out, p, base, n, l| row::extended(out, p, base, n, l, &weights),
                        );
                    }
                }
                Algo::ExtendedSad => {
                    Self::fill_channel(c0, v.bwd_x, fwd, pass.div_x);
                    Self::fill_channel(c1, v.bwd_y, fwd, pass.div_y);
                    Self::fill_channel(c2, v.fwd_x, rev, pass.div_x);
                    Self::fill_channel(c3, v.fwd_y, rev, pass.div_y);
                    Self::fill_channel(c4, v.prev_bwd_x, fwd, pass.div_x);
                    Self::fill_channel(c5, v.prev_bwd_y, fwd, pass.div_y);
                    Self::fill_channel(c6, v.next_fwd_x, rev, pass.div_x);
                    Self::fill_channel(c7, v.next_fwd_y, rev, pass.div_y);
                    Self::fill_mask(c8, v.cover_fwd);
                    Self::fill_mask(c9, v.cover_bwd);
                    Self::fill_mask(c10, v.sad);
                    if use_pair {
                        self.walk_pair(
                            pass,
                            [
                                c0.as_slice(),
                                c1.as_slice(),
                                c2.as_slice(),
                                c3.as_slice(),
                                c4.as_slice(),
                                c5.as_slice(),
                                c6.as_slice(),
                                c7.as_slice(),
                                c8.as_slice(),
                                c9.as_slice(),
                                c10.as_slice(),
                            ],
                            self.fast_occ(),
                            aux,
                            planes,
                            lanes,
                            blocks,
                            |out, p, base, n, l| {
                                row::extended_sad_pair(out, p, base, n, l, &weights);
                            },
                        );
                    } else {
                        self.walk(
                            pass,
                            interp,
                            [
                                c0.as_slice(),
                                c1.as_slice(),
                                c2.as_slice(),
                                c3.as_slice(),
                                c4.as_slice(),
                                c5.as_slice(),
                                c6.as_slice(),
                                c7.as_slice(),
                                c8.as_slice(),
                                c9.as_slice(),
                                c10.as_slice(),
                            ],
                            [
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                Some(v.cover_fwd),
                                Some(v.cover_bwd),
                                Some(v.sad),
                            ],
                            false,
                            self.fast_occ(),
                            aux,
                            planes,
                            lanes,
                            blocks,
                            |out, p, base, n, l| row::extended_sad(out, p, base, n, l, &weights),
                        );
                    }
                }
                Algo::Fill => {}
            }
        });
    }

    fn bilinear(&self, buf: &[u8], skip: i32, x: i32, y: i32, pass: &Pass<'_>) -> u8 {
        let s = &self.shape;
        let (ox, oy, sx, sy) = (
            s.origin_x / pass.div_x,
            s.origin_y / pass.div_y,
            s.step_x / pass.div_x,
            s.step_y / pass.div_y,
        );
        let (bx, bx1) = if x >= ox {
            let b = (x - ox) / sx;
            (b, b + 1)
        } else {
            (0, 0)
        };
        let (by, by1) = if y >= oy {
            let b = (y - oy) / sy;
            (b, b + 1)
        } else {
            (0, 0)
        };
        let count = s.grid_w.wrapping_mul(s.grid_h);
        let at = |cx: i32, cy: i32| {
            let direct = cy.wrapping_mul(s.grid_w).wrapping_add(cx);
            let skipped = direct.wrapping_add(skip);
            let index = if skipped < count { skipped } else { direct };
            usize::try_from(index)
                .ok()
                .and_then(|index| buf.get(index))
                .map_or(0, |&v| i32::from(v))
        };
        let (tl, tr, bl, br) = (at(bx, by), at(bx1, by), at(bx, by1), at(bx1, by1));
        let fx = if x <= ox { 0 } else { (x - ox) % sx };
        let top = (fx * tr + (sx - fx) * tl) / sx;
        let (fy, bottom) = if y <= oy {
            (0, 0)
        } else {
            let fy = (y - oy) % sy;
            (fy, fy * ((fx * br + (sx - fx) * bl) / sx))
        };
        ((top * (sy - fy) + bottom) / sy) as u8
    }

    fn fill_interp_block<const N: usize>(
        setup: &row::Setup<'_>,
        lanes: &mut [i32],
        (span, at, cols): (usize, usize, usize),
        (top, bottom): (&[i16; 32], &[i16; 32]),
        (wt, wb, shift_y): (i16, i16, u32),
        skip_ext: bool,
    ) {
        let cover_zero = skip_ext
            && N >= 10
            && (top[16]
                | top[17]
                | bottom[16]
                | bottom[17]
                | top[18]
                | top[19]
                | bottom[18]
                | bottom[19])
                == 0;
        if cover_zero {
            row::fill_block::<4>(setup, lanes, (span, at, cols), (top, bottom), (wt, wb, shift_y));
            lanes[8 * span + at..8 * span + at + cols].fill(0);
            lanes[9 * span + at..9 * span + at + cols].fill(0);
            if N == 11 {
                let m0 = wt.wrapping_mul(top[20]).wrapping_add(wb.wrapping_mul(bottom[20])) >> shift_y;
                let m1 = wt.wrapping_mul(top[21]).wrapping_add(wb.wrapping_mul(bottom[21])) >> shift_y;
                let pair = i32::from(m0 as u16) | (i32::from(m1 as u16) << 16);
                row::fill(setup, &mut lanes[10 * span + at..], pair, cols, false);
            }
            return;
        }
        row::fill_block::<N>(setup, lanes, (span, at, cols), (top, bottom), (wt, wb, shift_y));
    }

    #[allow(clippy::too_many_lines, clippy::too_many_arguments)]
    fn walk<const N: usize, T: Sample, K>(
        &self,
        pass: &Pass<'_>,
        interp: bool,
        channels: [&[i16]; N],
        samplers: [Option<&[u8]>; N],
        skip_rows: bool,
        skip_ext: bool,
        aux: Option<(Plane<'_, T>, Plane<'_, T>)>,
        planes: &mut [(&mut PlaneMut<'_, T>, Plane<'_, T>, Plane<'_, T>)],
        lanes_buf: &mut Vec<i32>,
        blocks: &mut Vec<BlockEntry>,
        kernel: K,
    ) where
        K: Fn(&mut [T], &row::Prepared<'_, T>, i32, usize, &[&[i32]; N]),
    {
        let grid_w = self.shape.grid_w;
        let grid_h = self.shape.grid_h;
        let pel = self.shape.pel;
        let pitch = pass.pitch as i32;
        let unit = [1i32; MAX_STEP];
        let mut real = [0i32; MAX_STEP];
        for (pair, w) in real.iter_mut().zip(pass.x_weights) {
            *pair = i32::from(w[0] as u16) | (i32::from(w[1] as u16) << 16);
        }
        let (vector_weights, vector_shift) = if interp {
            (&real, pass.shift_x)
        } else {
            (&unit, 0)
        };
        let setup = row::Setup {
            pel,
            pitch,
            weights: &real,
            shift: pass.shift_x,
            vector_weights,
            vector_shift,
        };
        assert!(planes.len() <= 2);
        let mut prepared: [Option<row::Prepared<'_, T>>; 2] = [None, None];
        for (index, (_, a, b)) in planes.iter().enumerate() {
            prepared[index] = Some(row::Prepared::new(&setup, a, b, aux, index));
        }
        const { assert!(N <= 16) };
        let width = usize::try_from(pass.width).unwrap_or(0);
        let span = width + MAX_STEP + 16;
        let need = N * span;
        if lanes_buf.len() < need {
            lanes_buf.resize(need, 0);
        }
        let lanes = &mut lanes_buf[..need];
        let mut pairs = [0i32; N];
        for by in pass.blocks.start.max(-1)..pass.blocks.end.min(grid_h) {
            let (py0, rows) = pass.block_rows(interp, grid_h, by);
            if rows <= 0 {
                continue;
            }
            let upper = by.max(0);
            let lower = if by == -1 || by == grid_h - 1 {
                upper
            } else {
                upper + 1
            };
            blocks.clear();
            for bx in -1..grid_w {
                let px0 = if bx < 0 {
                    0
                } else {
                    pass.origin_x + bx * pass.step_x
                };
                let mut cols = if px0 + pass.step_x < pass.width {
                    if bx == -1 { pass.origin_x } else { pass.step_x }
                } else {
                    pass.width - px0
                };
                if !interp && bx == grid_w - 1 {
                    cols = pass.width - px0;
                }
                if cols <= 0 {
                    continue;
                }
                let left = bx.max(0);
                let right = if bx == -1 || bx == grid_w - 1 {
                    left
                } else {
                    left + 1
                };
                let tl = (upper * grid_w + left) as usize;
                let tr = (upper * grid_w + right) as usize;
                let bl = (lower * grid_w + left) as usize;
                let br = (lower * grid_w + right) as usize;
                blocks.push((px0, cols as usize, [0; 32], [0; 32]));
                let (_, _, top, bottom) = blocks.last_mut().expect("pushed block");
                for k in 0..N {
                    if !interp && let Some(buf) = samplers[k] {
                        let skip = if skip_rows { upper * grid_w } else { 0 };
                        let corner =
                            |x: i32, y: i32| i16::from(self.bilinear(buf, skip, x, y, pass));
                        top[2 * k] = corner(px0, py0);
                        top[2 * k + 1] = corner(px0 + cols, py0);
                        bottom[2 * k] = corner(px0, py0 + rows);
                        bottom[2 * k + 1] = corner(px0 + cols, py0 + rows);
                        continue;
                    }
                    let ch = channels[k];
                    top[2 * k] = ch[tl];
                    top[2 * k + 1] = ch[tr];
                    bottom[2 * k] = ch[bl];
                    bottom[2 * k + 1] = ch[br];
                }
            }
            for r in 0..rows {
                let py = py0 + r;
                let ry = r.min(pass.step_y);
                let wt = pass.y_weights[(pass.step_y - ry) as usize];
                let wb = pass.y_weights[ry as usize];
                let shift_y = pass.shift_y.min(15);
                for (px0, cols, top, bottom) in blocks.iter() {
                    if interp {
                        Self::fill_interp_block::<N>(
                            &setup,
                            lanes,
                            (span, *px0 as usize, *cols),
                            (top, bottom),
                            (wt, wb, shift_y),
                            skip_ext,
                        );
                        continue;
                    }
                    let mut mixed = [0i16; 32];
                    for ((m, &t), &b) in mixed.iter_mut().zip(top).zip(bottom) {
                        *m = wt.wrapping_mul(t).wrapping_add(wb.wrapping_mul(b)) >> shift_y;
                    }
                    for (k, pair) in pairs.iter_mut().enumerate() {
                        *pair = if interp || samplers[k].is_some() {
                            i32::from(mixed[2 * k] as u16)
                                | (i32::from(mixed[2 * k + 1] as u16) << 16)
                        } else {
                            i32::from(top[2 * k] as u16)
                        };
                    }
                    let at = *px0 as usize;
                    for (k, lane) in lanes.chunks_exact_mut(span).enumerate() {
                        row::fill(
                            &setup,
                            &mut lane[at..],
                            pairs[k],
                            *cols,
                            samplers[k].is_none(),
                        );
                    }
                }
                if skip_ext && N >= 10 {
                    lanes[8 * span + width..8 * span + width + 16].fill(0);
                    lanes[9 * span + width..9 * span + width + 16].fill(0);
                }
                let view: [&[i32]; N] = std::array::from_fn(|k| &lanes[k * span..(k + 1) * span]);
                let base = pel * py * pitch;
                for ((dst, _, _), prepared) in planes.iter_mut().zip(&prepared) {
                    if let Some(prepared) = prepared {
                        let row_start = (py - pass.row_base) as usize * dst.pitch;
                        let end = (row_start + width).min(dst.data.len());
                        kernel(&mut dst.data[row_start..end], prepared, base, width, &view);
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_lines, clippy::too_many_arguments)]
    fn walk_pair<const N: usize, T: Sample, K>(
        &self,
        pass: &Pass<'_>,
        channels: [&[i16]; N],
        skip_ext: bool,
        aux: Option<(Plane<'_, T>, Plane<'_, T>)>,
        planes: &mut [(&mut PlaneMut<'_, T>, Plane<'_, T>, Plane<'_, T>)],
        lanes_buf: &mut Vec<i32>,
        blocks: &mut Vec<BlockEntry>,
        kernel: K,
    ) where
        K: Fn(&mut [T], &row::Prepared<'_, T>, i32, usize, &[&[i32]; N]),
    {
        let grid_w = self.shape.grid_w;
        let grid_h = self.shape.grid_h;
        let pel = self.shape.pel;
        let pitch = pass.pitch as i32;
        let mut real = [0i32; MAX_STEP];
        for (pair, w) in real.iter_mut().zip(pass.x_weights) {
            *pair = i32::from(w[0] as u16) | (i32::from(w[1] as u16) << 16);
        }
        let mut real_even = [0i32; MAX_STEP];
        let mut real_odd = [0i32; MAX_STEP];
        for j in 0..MAX_STEP {
            real_even[j] = real[(2 * j).min(MAX_STEP - 1)];
            real_odd[j] = real[(2 * j + 1).min(MAX_STEP - 1)];
        }
        let setup_even = row::Setup {
            pel,
            pitch,
            weights: &real_even,
            shift: pass.shift_x,
            vector_weights: &real_even,
            vector_shift: pass.shift_x,
        };
        let setup_odd = row::Setup {
            pel,
            pitch,
            weights: &real_odd,
            shift: pass.shift_x,
            vector_weights: &real_odd,
            vector_shift: pass.shift_x,
        };
        assert!(planes.len() <= 2);
        let mut prepared: [Option<row::Prepared<'_, T>>; 2] = [None, None];
        for (index, (_, a, b)) in planes.iter().enumerate() {
            prepared[index] = Some(row::Prepared::new(&setup_even, a, b, aux, index));
        }
        const { assert!(N <= 16) };
        let width = usize::try_from(pass.width).unwrap_or(0);
        let pair_count = width.div_ceil(2);
        let span = pair_count + MAX_STEP + 16;
        let need = N * span;
        if lanes_buf.len() < need {
            lanes_buf.resize(need, 0);
        }
        let lanes = &mut lanes_buf[..need];
        for by in pass.blocks.start.max(-1)..pass.blocks.end.min(grid_h) {
            let (py0, rows) = pass.block_rows(true, grid_h, by);
            if rows <= 0 {
                continue;
            }
            let upper = by.max(0);
            let lower = if by == -1 || by == grid_h - 1 {
                upper
            } else {
                upper + 1
            };
            blocks.clear();
            for bx in -1..grid_w {
                let px0 = if bx < 0 {
                    0
                } else {
                    pass.origin_x + bx * pass.step_x
                };
                let cols = if px0 + pass.step_x < pass.width {
                    if bx == -1 { pass.origin_x } else { pass.step_x }
                } else {
                    pass.width - px0
                };
                if cols <= 0 {
                    continue;
                }
                let left = bx.max(0);
                let right = if bx == -1 || bx == grid_w - 1 {
                    left
                } else {
                    left + 1
                };
                let tl = (upper * grid_w + left) as usize;
                let tr = (upper * grid_w + right) as usize;
                let bl = (lower * grid_w + left) as usize;
                let br = (lower * grid_w + right) as usize;
                blocks.push((px0, cols as usize, [0; 32], [0; 32]));
                let (_, _, top, bottom) = blocks.last_mut().expect("pushed block");
                for k in 0..N {
                    let ch = channels[k];
                    top[2 * k] = ch[tl];
                    top[2 * k + 1] = ch[tr];
                    bottom[2 * k] = ch[bl];
                    bottom[2 * k + 1] = ch[br];
                }
            }
            for r in 0..rows {
                let py = py0 + r;
                let ry = r.min(pass.step_y);
                let wt = pass.y_weights[(pass.step_y - ry) as usize];
                let wb = pass.y_weights[ry as usize];
                let shift_y = pass.shift_y.min(15);
                for (px0, cols, top, bottom) in blocks.iter() {
                    let x0 = *px0 as usize;
                    let p0 = x0.div_ceil(2);
                    let p1 = (x0 + *cols).div_ceil(2);
                    if p1 <= p0 {
                        continue;
                    }
                    let setup_p = if 2 * p0 == x0 {
                        &setup_even
                    } else {
                        &setup_odd
                    };
                    Self::fill_interp_block::<N>(
                        setup_p,
                        lanes,
                        (span, p0, p1 - p0),
                        (top, bottom),
                        (wt, wb, shift_y),
                        skip_ext,
                    );
                }
                if skip_ext && N >= 10 {
                    lanes[8 * span + pair_count..8 * span + pair_count + 16].fill(0);
                    lanes[9 * span + pair_count..9 * span + pair_count + 16].fill(0);
                }
                let view: [&[i32]; N] = std::array::from_fn(|k| &lanes[k * span..(k + 1) * span]);
                let base = pel * py * pitch;
                for ((dst, _, _), prepared) in planes.iter_mut().zip(&prepared) {
                    if let Some(prepared) = prepared {
                        let row_start = (py - pass.row_base) as usize * dst.pitch;
                        let end = (row_start + width).min(dst.data.len());
                        kernel(&mut dst.data[row_start..end], prepared, base, width, &view);
                    }
                }
            }
        }
    }
}

mod row {
    use super::MAX_STEP;

    pub(super) struct Setup<'a> {
        pub(super) pel: i32,
        pub(super) pitch: i32,
        pub(super) weights: &'a [i32; MAX_STEP],
        pub(super) shift: u32,
        pub(super) vector_weights: &'a [i32; MAX_STEP],
        pub(super) vector_shift: u32,
    }

    pub(super) struct Weights {
        pub(super) t: i32,
        pub(super) rt: i32,
        pub(super) tb: i32,
        pub(super) rtb: i32,
        pub(super) wrap: i32,
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[allow(
        unsafe_code,
        unused_unsafe,
        unsafe_op_in_unsafe_fn,
        clippy::cast_ptr_alignment,
        clippy::inline_always,
        clippy::many_single_char_names,
        clippy::trivially_copy_pass_by_ref
    )]
    mod imp {
        use std::arch::x86_64::{
            __m128i, __m256i, __m512i, _mm_cvtsi32_si128, _mm_cvtsi128_si32, _mm_loadl_epi64,
            _mm_loadu_si128, _mm_packs_epi32, _mm_packus_epi16, _mm_packus_epi32, _mm_storel_epi64,
            _mm_storeu_si128, _mm_unpackhi_epi8, _mm_unpackhi_epi16, _mm_unpacklo_epi8,
            _mm_unpacklo_epi16, _mm256_add_epi16, _mm256_add_epi32, _mm256_and_si256,
            _mm256_broadcastsi128_si256, _mm256_castsi256_si128, _mm256_cmpeq_epi32,
            _mm256_cmpgt_epi32, _mm256_cvtepu8_epi32, _mm256_cvtepu16_epi32, _mm256_cvtsi256_si32,
            _mm256_extracti128_si256, _mm256_i32gather_epi32, _mm256_loadu_si256,
            _mm256_madd_epi16, _mm256_max_epi32, _mm256_min_epi32, _mm256_movemask_epi8,
            _mm256_mullo_epi16, _mm256_mullo_epi32, _mm256_or_si256, _mm256_permutevar8x32_epi32,
            _mm256_set1_epi16, _mm256_set1_epi32, _mm256_setr_epi32, _mm256_shuffle_epi8,
            _mm256_sll_epi32, _mm256_slli_epi32, _mm256_sra_epi16, _mm256_sra_epi32,
            _mm256_srai_epi32, _mm256_srli_epi32, _mm256_storeu_si256, _mm256_sub_epi32,
            _mm256_testz_si256, _mm512_add_epi32, _mm512_and_si512, _mm512_castsi512_si128,
            _mm512_cmpeq_epi32_mask, _mm512_cmpgt_epi32_mask, _mm512_cvtepu8_epi32,
            _mm512_cvtepu16_epi32, _mm512_cvtusepi32_epi8, _mm512_cvtusepi32_epi16,
            _mm512_i32gather_epi32, _mm512_loadu_si512, _mm512_madd_epi16,
            _mm512_maskz_permutex2var_epi8, _mm512_maskz_permutex2var_epi16,
            _mm512_maskz_permutexvar_epi8, _mm512_maskz_permutexvar_epi16, _mm512_max_epi32,
            _mm512_min_epi32, _mm512_mullo_epi16, _mm512_mullo_epi32, _mm512_set1_epi32,
            _mm512_setr_epi32, _mm512_setzero_si512, _mm512_slli_epi32, _mm512_srai_epi32,
            _mm512_srli_epi32, _mm512_srlv_epi32, _mm512_sub_epi32, _mm512_test_epi32_mask,
        };
        use std::marker::PhantomData;
        use std::sync::OnceLock;

        use super::super::{Plane, Sample};
        use super::{MAX_STEP, Setup, Weights};

        fn has_avx512() -> bool {
            static FLAG: OnceLock<bool> = OnceLock::new();
            *FLAG.get_or_init(|| {
                std::arch::is_x86_feature_detected!("avx512f")
                    && std::arch::is_x86_feature_detected!("avx512bw")
                    && std::arch::is_x86_feature_detected!("avx512vl")
                    && std::arch::is_x86_feature_detected!("avx512vbmi")
            })
        }

        #[derive(Clone, Copy)]
        struct Source<T> {
            ptr: *const i32,
            data: *const T,
            last: __m256i,
            safe: __m256i,
            offsets: __m256i,
            len: i32,
            valid: bool,
            padded: bool,
            readable: usize,
            pel: i32,
        }

        impl<T: Sample> Source<T> {
            const SIZE: usize = size_of::<T>();
            const MASK: i32 = (1 << (8 * Self::SIZE)) - 1;
            const WORD: i32 = (4 / Self::SIZE) as i32;

            #[inline(always)]
            fn new(plane: &Plane<'_, T>, pel: i32) -> Self {
                unsafe {
                    let len = i32::try_from(plane.data.len()).unwrap_or(i32::MAX);
                    Self {
                        ptr: plane.data.as_ptr().cast::<i32>(),
                        data: plane.data.as_ptr(),
                        last: _mm256_set1_epi32(len - 1),
                        safe: _mm256_set1_epi32(len - Self::WORD),
                        offsets: _mm256_mullo_epi32(
                            _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                            _mm256_set1_epi32(pel),
                        ),
                        len,
                        valid: len >= Self::WORD,
                        padded: plane.slack + 1 >= Self::WORD as usize,
                        readable: plane.data.len() + plane.slack,
                        pel,
                    }
                }
            }

            #[inline(always)]
            fn fetch(&self, index: __m256i) -> __m256i {
                unsafe {
                    let zero = _mm256_set1_epi32(0);
                    if !self.valid {
                        return zero;
                    }
                    let first = _mm256_cvtsi256_si32(index);
                    let run = _mm256_add_epi32(_mm256_set1_epi32(first), self.offsets);
                    if first >= 0
                        && _mm256_movemask_epi8(_mm256_cmpeq_epi32(index, run)) == -1
                        && let Some(value) = self.load(first)
                    {
                        return value;
                    }
                    let index = _mm256_min_epi32(_mm256_max_epi32(index, zero), self.last);
                    if Self::SIZE == 1 && self.pel == 1 {
                        let cfirst = _mm256_cvtsi256_si32(index);
                        let win_base = (cfirst - 2).max(0);
                        let rel = _mm256_sub_epi32(index, _mm256_set1_epi32(win_base));
                        if _mm256_testz_si256(rel, _mm256_set1_epi32(!15)) != 0
                            && (win_base as usize) + 16 <= self.readable
                        {
                            let win = _mm256_broadcastsi128_si256(_mm_loadu_si128(
                                self.data.add(win_base as usize).cast::<__m128i>(),
                            ));
                            return _mm256_shuffle_epi8(
                                win,
                                _mm256_or_si256(rel, _mm256_set1_epi32(0x8080_8000u32 as i32)),
                            );
                        }
                    }
                    if self.padded {
                        let words = self.gather(index);
                        return _mm256_and_si256(words, _mm256_set1_epi32(Self::MASK));
                    }
                    let base = _mm256_min_epi32(index, self.safe);
                    let offset = _mm256_sub_epi32(index, base);
                    let pick = if Self::SIZE == 1 {
                        let lanes = _mm256_setr_epi32(
                            0x8080_8000u32 as i32,
                            0x8080_8004u32 as i32,
                            0x8080_8008u32 as i32,
                            0x8080_800Cu32 as i32,
                            0x8080_8000u32 as i32,
                            0x8080_8004u32 as i32,
                            0x8080_8008u32 as i32,
                            0x8080_800Cu32 as i32,
                        );
                        _mm256_add_epi32(offset, lanes)
                    } else {
                        let lanes = _mm256_setr_epi32(
                            0x8080_0100u32 as i32,
                            0x8080_0504u32 as i32,
                            0x8080_0908u32 as i32,
                            0x8080_0D0Cu32 as i32,
                            0x8080_0100u32 as i32,
                            0x8080_0504u32 as i32,
                            0x8080_0908u32 as i32,
                            0x8080_0D0Cu32 as i32,
                        );
                        _mm256_add_epi32(_mm256_mullo_epi32(offset, splat(0x0202)), lanes)
                    };
                    _mm256_shuffle_epi8(self.gather(base), pick)
                }
            }

            #[inline(always)]
            fn gather(&self, index: __m256i) -> __m256i {
                unsafe {
                    if Self::SIZE == 1 {
                        _mm256_i32gather_epi32::<1>(self.ptr, index)
                    } else {
                        _mm256_i32gather_epi32::<2>(self.ptr, index)
                    }
                }
            }

            #[inline(always)]
            fn load(&self, start: i32) -> Option<__m256i> {
                unsafe {
                    let start = start as usize;
                    let span = 8 * self.pel as usize;
                    if start + span.max(16 / Self::SIZE) > self.readable {
                        return None;
                    }
                    let at = self.data.add(start);
                    if Self::SIZE == 2 {
                        return match self.pel {
                            1 => Some(_mm256_cvtepu16_epi32(_mm_loadu_si128(at.cast::<__m128i>()))),
                            2 => {
                                let words = _mm256_loadu_si256(at.cast::<__m256i>());
                                Some(_mm256_and_si256(words, _mm256_set1_epi32(0xFFFF)))
                            }
                            _ => None,
                        };
                    }
                    match self.pel {
                        1 => Some(_mm256_cvtepu8_epi32(_mm_loadl_epi64(at.cast::<__m128i>()))),
                        2 => {
                            let words =
                                _mm256_cvtepu16_epi32(_mm_loadu_si128(at.cast::<__m128i>()));
                            Some(_mm256_and_si256(words, _mm256_set1_epi32(0xFF)))
                        }
                        4 => {
                            let words = _mm256_loadu_si256(at.cast::<__m256i>());
                            Some(_mm256_and_si256(words, _mm256_set1_epi32(0xFF)))
                        }
                        _ => None,
                    }
                }
            }

            #[inline(always)]
            fn still(&self, start: i32, chunk: __m256i) -> __m256i {
                self.load(start).unwrap_or_else(|| self.fetch(chunk))
            }

            #[inline(always)]
            fn load_pair(&self, start: i32) -> Option<(__m256i, __m256i)> {
                unsafe {
                    let start = start as usize;
                    let span = 16 * self.pel as usize;
                    if start + span.max(16 / Self::SIZE) > self.readable {
                        return None;
                    }
                    let at = self.data.add(start);
                    if Self::SIZE == 2 {
                        return match self.pel {
                            1 => {
                                let w = _mm256_loadu_si256(at.cast::<__m256i>());
                                Some((
                                    _mm256_and_si256(w, _mm256_set1_epi32(0xFFFF)),
                                    _mm256_srli_epi32::<16>(w),
                                ))
                            }
                            _ => None,
                        };
                    }
                    match self.pel {
                        1 => {
                            let w = _mm256_cvtepu16_epi32(_mm_loadu_si128(at.cast::<__m128i>()));
                            Some((
                                _mm256_and_si256(w, _mm256_set1_epi32(0xFF)),
                                _mm256_srli_epi32::<8>(w),
                            ))
                        }
                        2 => {
                            let w = _mm256_loadu_si256(at.cast::<__m256i>());
                            let mask = _mm256_set1_epi32(0xFF);
                            Some((
                                _mm256_and_si256(w, mask),
                                _mm256_and_si256(_mm256_srli_epi32::<16>(w), mask),
                            ))
                        }
                        _ => None,
                    }
                }
            }

            #[inline(always)]
            fn fetch_pair(&self, index: __m256i, offsets2: __m256i) -> (__m256i, __m256i) {
                unsafe {
                    let zero = _mm256_set1_epi32(0);
                    if !self.valid {
                        return (zero, zero);
                    }
                    let first = _mm256_cvtsi256_si32(index);
                    let run = _mm256_add_epi32(_mm256_set1_epi32(first), offsets2);
                    if first >= 0
                        && _mm256_movemask_epi8(_mm256_cmpeq_epi32(index, run)) == -1
                        && let Some(pair) = self.load_pair(first)
                    {
                        return pair;
                    }
                    let clamped = _mm256_min_epi32(_mm256_max_epi32(index, zero), self.last);
                    let safe_limit = (self.readable as i32 - Self::WORD).min(self.len - 1 - self.pel);
                    if ((Self::SIZE == 1 && self.pel <= 2) || (Self::SIZE == 2 && self.pel == 1))
                        && safe_limit >= 0
                        && _mm256_movemask_epi8(_mm256_cmpgt_epi32(clamped, splat(safe_limit))) == 0
                    {
                        let words = self.gather(clamped);
                        let mask = _mm256_set1_epi32(Self::MASK);
                        let even = _mm256_and_si256(words, mask);
                        let odd = if Self::SIZE == 1 {
                            if self.pel == 1 {
                                _mm256_and_si256(_mm256_srli_epi32::<8>(words), mask)
                            } else {
                                _mm256_and_si256(_mm256_srli_epi32::<16>(words), mask)
                            }
                        } else {
                            _mm256_srli_epi32::<16>(words)
                        };
                        return (even, odd);
                    }
                    (
                        self.fetch(index),
                        self.fetch(_mm256_add_epi32(index, splat(self.pel))),
                    )
                }
            }

            #[inline(always)]
            fn still_pair(&self, start: i32, chunk: __m256i, offsets2: __m256i) -> (__m256i, __m256i) {
                self.load_pair(start)
                    .unwrap_or_else(|| self.fetch_pair(chunk, offsets2))
            }
        }

        pub(in super::super) struct Prepared<'a, T> {
            a: Source<T>,
            b: Source<T>,
            offsets: __m256i,
            step: __m256i,
            pel: i32,
            pitch: __m256i,
            pitch_i32: i32,
            avx512: bool,
            aux_a: Source<T>,
            aux_b: Source<T>,
            pub(in super::super) index: usize,
            _life: PhantomData<&'a [T]>,
        }

        impl<'a, T: Sample> Prepared<'a, T> {
            pub(in super::super) fn new(
                setup: &Setup<'a>,
                a: &Plane<'a, T>,
                b: &Plane<'a, T>,
                aux: Option<(Plane<'a, T>, Plane<'a, T>)>,
                index: usize,
            ) -> Self {
                unsafe {
                    let pel = setup.pel;
                    let (aux_a, aux_b) = aux.unwrap_or((*a, *b));
                    Self {
                        a: Source::new(a, pel),
                        b: Source::new(b, pel),
                        aux_a: Source::new(&aux_a, pel),
                        aux_b: Source::new(&aux_b, pel),
                        index,
                        offsets: _mm256_setr_epi32(
                            0,
                            pel,
                            2 * pel,
                            3 * pel,
                            4 * pel,
                            5 * pel,
                            6 * pel,
                            7 * pel,
                        ),
                        step: _mm256_set1_epi32(8 * pel),
                        pel,
                        pitch: _mm256_set1_epi32(setup.pitch),
                        pitch_i32: setup.pitch,
                        avx512: has_avx512(),
                        _life: PhantomData,
                    }
                }
            }

            #[allow(clippy::unused_self)]
            #[inline(always)]
            fn lane(&self, values: &[i32], c: usize) -> __m256i {
                assert!(c + 8 <= values.len());
                unsafe { _mm256_loadu_si256(values.as_ptr().add(c).cast::<__m256i>()) }
            }

            #[inline(always)]
            fn at(&self, chunk: __m256i, x: &[i32], y: &[i32], c: usize) -> __m256i {
                unsafe {
                    let x = self.lane(x, c);
                    let y = self.lane(y, c);
                    let dy = if (0..=32767).contains(&self.pitch_i32) {
                        _mm256_madd_epi16(y, self.pitch)
                    } else {
                        _mm256_mullo_epi32(y, self.pitch)
                    };
                    _mm256_add_epi32(_mm256_add_epi32(chunk, x), dy)
                }
            }

            #[inline(always)]
            fn run(
                &self,
                out: &mut [T],
                base: i32,
                n: usize,
                mut body: impl FnMut(usize, __m256i, i32) -> __m256i,
            ) {
                unsafe {
                    let mut chunk = _mm256_add_epi32(_mm256_set1_epi32(base), self.offsets);
                    let mut start = base;
                    let mut c = 0;
                    while c < n {
                        let value = body(c, chunk, start);
                        start += 8 * self.pel;
                        let lo = _mm256_castsi256_si128(value);
                        let hi = _mm256_extracti128_si256::<1>(value);
                        let store = |at: *mut T| {
                            if size_of::<T>() == 1 {
                                let words = _mm_packs_epi32(lo, hi);
                                let bytes = _mm_packus_epi16(words, words);
                                _mm_storel_epi64(at.cast::<__m128i>(), bytes);
                            } else {
                                _mm_storeu_si128(at.cast::<__m128i>(), _mm_packus_epi32(lo, hi));
                            }
                        };
                        if c + 8 <= out.len() {
                            store(out.as_mut_ptr().add(c));
                        } else {
                            let mut tmp = [T::default(); 8];
                            store(tmp.as_mut_ptr());
                            let take = (n - c).min(8);
                            out[c..c + take].copy_from_slice(&tmp[..take]);
                        }
                        chunk = _mm256_add_epi32(chunk, self.step);
                        c += 8;
                    }
                }
            }

            #[inline(always)]
            fn run_pair(
                &self,
                out: &mut [T],
                base: i32,
                n: usize,
                mut body: impl FnMut(usize, __m256i, i32, __m256i) -> (__m256i, __m256i),
            ) {
                unsafe {
                    let offsets2 = _mm256_add_epi32(self.offsets, self.offsets);
                    let step2 = _mm256_add_epi32(self.step, self.step);
                    let mut chunk = _mm256_add_epi32(_mm256_set1_epi32(base), offsets2);
                    let mut start = base;
                    let mut p = 0;
                    let mut c = 0;
                    while c < n {
                        let (even, odd) = body(p, chunk, start, offsets2);
                        start += 16 * self.pel;
                        let store = |at: *mut T| {
                            let elo = _mm256_castsi256_si128(even);
                            let ehi = _mm256_extracti128_si256::<1>(even);
                            let olo = _mm256_castsi256_si128(odd);
                            let ohi = _mm256_extracti128_si256::<1>(odd);
                            if size_of::<T>() == 1 {
                                let eb = _mm_packus_epi16(_mm_packs_epi32(elo, ehi), _mm_packs_epi32(elo, ehi));
                                let ob = _mm_packus_epi16(_mm_packs_epi32(olo, ohi), _mm_packs_epi32(olo, ohi));
                                _mm_storeu_si128(at.cast::<__m128i>(), _mm_unpacklo_epi8(eb, ob));
                            } else {
                                let ew = _mm_packus_epi32(elo, ehi);
                                let ow = _mm_packus_epi32(olo, ohi);
                                _mm_storeu_si128(at.cast::<__m128i>(), _mm_unpacklo_epi16(ew, ow));
                                _mm_storeu_si128(
                                    at.add(8).cast::<__m128i>(),
                                    _mm_unpackhi_epi16(ew, ow),
                                );
                            }
                        };
                        if c + 16 <= out.len() {
                            store(out.as_mut_ptr().add(c));
                        } else {
                            let mut tmp = [T::default(); 16];
                            store(tmp.as_mut_ptr());
                            let take = (n - c).min(out.len().saturating_sub(c)).min(16);
                            out[c..c + take].copy_from_slice(&tmp[..take]);
                        }
                        chunk = _mm256_add_epi32(chunk, step2);
                        p += 8;
                        c += 16;
                    }
                }
            }
        }

        #[inline(always)]
        fn splat(v: i32) -> __m256i {
            unsafe { _mm256_set1_epi32(v) }
        }

        #[inline(always)]
        pub(in super::super) fn fill(
            setup: &Setup<'_>,
            out: &mut [i32],
            pair: i32,
            cols: usize,
            vector: bool,
        ) {
            let (weights, shift) = if vector {
                (setup.vector_weights, setup.vector_shift)
            } else {
                (setup.weights, setup.shift)
            };
            let count = cols.next_multiple_of(8);
            assert!(count <= MAX_STEP && count <= out.len());
            unsafe {
                let shift = _mm_cvtsi32_si128(shift as i32);
                let pair = _mm256_set1_epi32(pair);
                let mut c = 0;
                while c < count {
                    let w = _mm256_loadu_si256(weights.as_ptr().add(c).cast::<__m256i>());
                    let sum = _mm256_sra_epi32(_mm256_madd_epi16(pair, w), shift);
                    let value = _mm256_srai_epi32::<16>(_mm256_slli_epi32::<16>(sum));
                    _mm256_storeu_si256(out.as_mut_ptr().add(c).cast::<__m256i>(), value);
                    c += 8;
                }
            }
        }

        #[inline(always)]
        pub(in super::super) fn fill_block<const N: usize>(
            setup: &Setup<'_>,
            lanes: &mut [i32],
            (span, at, cols): (usize, usize, usize),
            (top, bottom): (&[i16; 32], &[i16; 32]),
            (wt, wb, shift_y): (i16, i16, u32),
        ) {
            let count = cols.next_multiple_of(8);
            assert!(count <= MAX_STEP && at + count <= span && N * span <= lanes.len());
            unsafe {
                let (wt, wb) = (_mm256_set1_epi16(wt), _mm256_set1_epi16(wb));
                let shift_y = _mm_cvtsi32_si128(shift_y as i32);
                let mix = |half: usize| {
                    let t = _mm256_loadu_si256(top.as_ptr().add(16 * half).cast::<__m256i>());
                    let b = _mm256_loadu_si256(bottom.as_ptr().add(16 * half).cast::<__m256i>());
                    let sum =
                        _mm256_add_epi16(_mm256_mullo_epi16(wt, t), _mm256_mullo_epi16(wb, b));
                    _mm256_sra_epi16(sum, shift_y)
                };
                let halves = [mix(0), if N > 8 { mix(1) } else { mix(0) }];
                let pairs: [__m256i; N] = std::array::from_fn(|k| {
                    _mm256_permutevar8x32_epi32(halves[k / 8], _mm256_set1_epi32((k % 8) as i32))
                });
                let near = setup.shift <= 16;
                let shift = _mm_cvtsi32_si128(setup.shift as i32);
                let up = _mm_cvtsi32_si128(16 - setup.shift.min(16) as i32);
                let out = lanes.as_mut_ptr().add(at);
                let mut c = 0;
                while c < count {
                    let w = _mm256_loadu_si256(setup.weights.as_ptr().add(c).cast::<__m256i>());
                    for (k, &pair) in pairs.iter().enumerate() {
                        let sum = _mm256_madd_epi16(pair, w);
                        let value = if near {
                            _mm256_srai_epi32::<16>(_mm256_sll_epi32(sum, up))
                        } else {
                            let sum = _mm256_sra_epi32(sum, shift);
                            _mm256_srai_epi32::<16>(_mm256_slli_epi32::<16>(sum))
                        };
                        _mm256_storeu_si256(out.add(k * span + c).cast::<__m256i>(), value);
                    }
                    c += 8;
                }
            }
        }

        #[inline(always)]
        fn mul(a: __m256i, b: __m256i) -> __m256i {
            unsafe { _mm256_mullo_epi32(a, b) }
        }

        #[inline(always)]
        fn add(a: __m256i, b: __m256i) -> __m256i {
            unsafe { _mm256_add_epi32(a, b) }
        }

        #[inline(always)]
        fn sub(a: __m256i, b: __m256i) -> __m256i {
            unsafe { _mm256_sub_epi32(a, b) }
        }

        #[inline(always)]
        fn min(a: __m256i, b: __m256i) -> __m256i {
            unsafe { _mm256_min_epi32(a, b) }
        }

        #[inline(always)]
        fn max(a: __m256i, b: __m256i) -> __m256i {
            unsafe { _mm256_max_epi32(a, b) }
        }

        #[inline(always)]
        fn shr8(a: __m256i) -> __m256i {
            unsafe { _mm256_srai_epi32::<8>(a) }
        }

        #[inline(always)]
        fn shr8u(a: __m256i) -> __m256i {
            unsafe { _mm256_srli_epi32::<8>(a) }
        }

        #[inline(always)]
        fn blend255_256<T: Sample>(a: __m256i, b: __m256i, m: __m256i, full: __m256i) -> __m256i {
            unsafe {
                if size_of::<T>() == 1 {
                    let p0 = _mm256_mullo_epi16(_mm256_sub_epi32(full, m), a);
                    let p1 = _mm256_mullo_epi16(m, b);
                    _mm256_srli_epi32::<8>(_mm256_add_epi32(_mm256_add_epi32(p0, p1), full))
                } else {
                    let a255 = _mm256_sub_epi32(_mm256_slli_epi32::<8>(a), a);
                    _mm256_srai_epi32::<8>(_mm256_add_epi32(
                        _mm256_add_epi32(a255, _mm256_mullo_epi32(m, _mm256_sub_epi32(b, a))),
                        full,
                    ))
                }
            }
        }

        #[inline(always)]
        fn blend256_256<T: Sample>(a: __m256i, b: __m256i, rt: __m256i, t: __m256i) -> __m256i {
            unsafe {
                if size_of::<T>() == 1 {
                    let p0 = _mm256_mullo_epi16(rt, a);
                    let p1 = _mm256_mullo_epi16(t, b);
                    _mm256_srli_epi32::<8>(_mm256_add_epi32(p0, p1))
                } else {
                    _mm256_srai_epi32::<8>(_mm256_add_epi32(
                        _mm256_slli_epi32::<8>(a),
                        _mm256_mullo_epi32(t, _mm256_sub_epi32(b, a)),
                    ))
                }
            }
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
        #[inline]
        unsafe fn blend255_512<T: Sample>(
            a: __m512i,
            b: __m512i,
            m: __m512i,
            full: __m512i,
        ) -> __m512i {
            if size_of::<T>() == 1 {
                let p0 = _mm512_mullo_epi16(_mm512_sub_epi32(full, m), a);
                let p1 = _mm512_mullo_epi16(m, b);
                _mm512_srli_epi32::<8>(_mm512_add_epi32(_mm512_add_epi32(p0, p1), full))
            } else {
                let a255 = _mm512_sub_epi32(_mm512_slli_epi32::<8>(a), a);
                _mm512_srai_epi32::<8>(_mm512_add_epi32(
                    _mm512_add_epi32(a255, _mm512_mullo_epi32(m, _mm512_sub_epi32(b, a))),
                    full,
                ))
            }
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
        #[inline]
        unsafe fn blend256_512<T: Sample>(
            a: __m512i,
            b: __m512i,
            rt: __m512i,
            t: __m512i,
        ) -> __m512i {
            if size_of::<T>() == 1 {
                let p0 = _mm512_mullo_epi16(rt, a);
                let p1 = _mm512_mullo_epi16(t, b);
                _mm512_srli_epi32::<8>(_mm512_add_epi32(p0, p1))
            } else {
                _mm512_srai_epi32::<8>(_mm512_add_epi32(
                    _mm512_slli_epi32::<8>(a),
                    _mm512_mullo_epi32(t, _mm512_sub_epi32(b, a)),
                ))
            }
        }

        #[derive(Clone, Copy)]
        struct Pitch512 {
            vec: __m512i,
            short: bool,
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
        #[inline]
        unsafe fn pitch512(pitch: i32) -> Pitch512 {
            Pitch512 {
                vec: _mm512_set1_epi32(pitch),
                short: (0..=32767).contains(&pitch),
            }
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
        #[inline]
        unsafe fn load512<T: Sample>(src: &Source<T>, start: i32) -> Option<__m512i> {
            let start = start as usize;
            let span = 16 * src.pel as usize;
            if start + span.max(16 / Source::<T>::SIZE) > src.readable {
                return None;
            }
            let at = src.data.add(start);
            if Source::<T>::SIZE == 2 {
                return match src.pel {
                    1 => Some(_mm512_cvtepu16_epi32(_mm256_loadu_si256(
                        at.cast::<__m256i>(),
                    ))),
                    2 => Some(_mm512_and_si512(
                        _mm512_loadu_si512(at.cast::<__m512i>()),
                        _mm512_set1_epi32(0xFFFF),
                    )),
                    _ => None,
                };
            }
            match src.pel {
                1 => Some(_mm512_cvtepu8_epi32(_mm_loadu_si128(at.cast::<__m128i>()))),
                2 => Some(_mm512_and_si512(
                    _mm512_cvtepu16_epi32(_mm256_loadu_si256(at.cast::<__m256i>())),
                    _mm512_set1_epi32(0xFF),
                )),
                4 => Some(_mm512_and_si512(
                    _mm512_loadu_si512(at.cast::<__m512i>()),
                    _mm512_set1_epi32(0xFF),
                )),
                _ => None,
            }
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
        #[inline]
        unsafe fn gather512<T: Sample>(src: &Source<T>, index: __m512i) -> __m512i {
            if Source::<T>::SIZE == 1 {
                _mm512_i32gather_epi32::<1>(index, src.ptr)
            } else {
                _mm512_i32gather_epi32::<2>(index, src.ptr)
            }
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
        #[inline]
        unsafe fn fetch512<T: Sample>(src: &Source<T>, index: __m512i, offsets: __m512i) -> __m512i {
            let zero = _mm512_setzero_si512();
            if !src.valid {
                return zero;
            }
            let first = _mm_cvtsi128_si32(_mm512_castsi512_si128(index));
            let run = _mm512_add_epi32(_mm512_set1_epi32(first), offsets);
            if first >= 0
                && _mm512_cmpeq_epi32_mask(index, run) == 0xFFFF
                && let Some(value) = load512(src, first)
            {
                return value;
            }
            let last = _mm512_set1_epi32(src.len - 1);
            let index = _mm512_min_epi32(_mm512_max_epi32(index, zero), last);
            let cfirst = _mm_cvtsi128_si32(_mm512_castsi512_si128(index));
            if Source::<T>::SIZE == 1 {
                const LOW_BYTES: u64 = 0x1111_1111_1111_1111;
                let win_base = if src.pel == 4 {
                    cfirst.max(0)
                } else {
                    (cfirst - 4).max(0)
                };
                let rel = _mm512_sub_epi32(index, _mm512_set1_epi32(win_base));
                if _mm512_test_epi32_mask(rel, _mm512_set1_epi32(!63)) == 0
                    && (win_base as usize) + 64 <= src.readable
                {
                    let win = _mm512_loadu_si512(src.data.add(win_base as usize).cast::<__m512i>());
                    return _mm512_maskz_permutexvar_epi8(LOW_BYTES, rel, win);
                }
                if _mm512_test_epi32_mask(rel, _mm512_set1_epi32(!127)) == 0
                    && (win_base as usize) + 128 <= src.readable
                {
                    let base_ptr = src.data.add(win_base as usize);
                    let win0 = _mm512_loadu_si512(base_ptr.cast::<__m512i>());
                    let win1 = _mm512_loadu_si512(base_ptr.add(64).cast::<__m512i>());
                    return _mm512_maskz_permutex2var_epi8(LOW_BYTES, win0, rel, win1);
                }
            } else {
                const LOW_WORDS: u32 = 0x5555_5555;
                let win_base = (cfirst - 2).max(0);
                let rel = _mm512_sub_epi32(index, _mm512_set1_epi32(win_base));
                if _mm512_test_epi32_mask(rel, _mm512_set1_epi32(!31)) == 0
                    && (win_base as usize) + 32 <= src.readable
                {
                    let win = _mm512_loadu_si512(src.data.add(win_base as usize).cast::<__m512i>());
                    return _mm512_maskz_permutexvar_epi16(LOW_WORDS, rel, win);
                }
                if _mm512_test_epi32_mask(rel, _mm512_set1_epi32(!63)) == 0
                    && (win_base as usize) + 64 <= src.readable
                {
                    let base_ptr = src.data.add(win_base as usize);
                    let win0 = _mm512_loadu_si512(base_ptr.cast::<__m512i>());
                    let win1 = _mm512_loadu_si512(base_ptr.add(32).cast::<__m512i>());
                    return _mm512_maskz_permutex2var_epi16(LOW_WORDS, win0, rel, win1);
                }
            }
            if src.padded {
                return _mm512_and_si512(
                    gather512(src, index),
                    _mm512_set1_epi32(Source::<T>::MASK),
                );
            }
            let safe = _mm512_set1_epi32(src.len - Source::<T>::WORD);
            let base = _mm512_min_epi32(index, safe);
            let offset = _mm512_sub_epi32(index, base);
            let shift = if Source::<T>::SIZE == 1 {
                _mm512_slli_epi32::<3>(offset)
            } else {
                _mm512_slli_epi32::<4>(offset)
            };
            _mm512_and_si512(
                _mm512_srlv_epi32(gather512(src, base), shift),
                _mm512_set1_epi32(Source::<T>::MASK),
            )
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
        #[inline]
        unsafe fn still512<T: Sample>(
            src: &Source<T>,
            start: i32,
            chunk: __m512i,
            offsets: __m512i,
        ) -> __m512i {
            load512(src, start).unwrap_or_else(|| fetch512(src, chunk, offsets))
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
        #[inline]
        unsafe fn lane512(values: &[i32], c: usize) -> __m512i {
            assert!(c + 16 <= values.len());
            _mm512_loadu_si512(values.as_ptr().add(c).cast::<__m512i>())
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
        #[inline]
        unsafe fn at512(
            chunk: __m512i,
            pitch: Pitch512,
            x: &[i32],
            y: &[i32],
            c: usize,
        ) -> __m512i {
            let xv = lane512(x, c);
            let yv = lane512(y, c);
            let dy = if pitch.short {
                _mm512_madd_epi16(yv, pitch.vec)
            } else {
                _mm512_mullo_epi32(yv, pitch.vec)
            };
            _mm512_add_epi32(_mm512_add_epi32(chunk, xv), dy)
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
        #[inline]
        unsafe fn offsets512(pel: i32) -> __m512i {
            _mm512_mullo_epi32(
                _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
                _mm512_set1_epi32(pel),
            )
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
        #[inline]
        unsafe fn store512<T: Sample>(out: &mut [T], c: usize, n: usize, value: __m512i) {
            let clamped = if size_of::<T>() == 1 {
                value
            } else {
                _mm512_max_epi32(value, _mm512_setzero_si512())
            };
            if c + 16 <= out.len() {
                let ptr = out.as_mut_ptr().add(c);
                if size_of::<T>() == 1 {
                    _mm_storeu_si128(ptr.cast::<__m128i>(), _mm512_cvtusepi32_epi8(clamped));
                } else {
                    _mm256_storeu_si256(ptr.cast::<__m256i>(), _mm512_cvtusepi32_epi16(clamped));
                }
            } else {
                let mut tmp = [T::default(); 16];
                if size_of::<T>() == 1 {
                    _mm_storeu_si128(
                        tmp.as_mut_ptr().cast::<__m128i>(),
                        _mm512_cvtusepi32_epi8(clamped),
                    );
                } else {
                    _mm256_storeu_si256(
                        tmp.as_mut_ptr().cast::<__m256i>(),
                        _mm512_cvtusepi32_epi16(clamped),
                    );
                }
                let take = (n - c).min(out.len().saturating_sub(c)).min(16);
                out[c..c + take].copy_from_slice(&tmp[..take]);
            }
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
        #[inline]
        unsafe fn load_pair512<T: Sample>(src: &Source<T>, start: i32) -> Option<(__m512i, __m512i)> {
            let start = start as usize;
            let span = 32 * src.pel as usize;
            if start + span.max(16 / Source::<T>::SIZE) > src.readable {
                return None;
            }
            let at = src.data.add(start);
            if Source::<T>::SIZE == 2 {
                return match src.pel {
                    1 => {
                        let w = _mm512_loadu_si512(at.cast::<__m512i>());
                        Some((
                            _mm512_and_si512(w, _mm512_set1_epi32(0xFFFF)),
                            _mm512_srli_epi32::<16>(w),
                        ))
                    }
                    _ => None,
                };
            }
            match src.pel {
                1 => {
                    let w = _mm512_cvtepu16_epi32(_mm256_loadu_si256(at.cast::<__m256i>()));
                    Some((
                        _mm512_and_si512(w, _mm512_set1_epi32(0xFF)),
                        _mm512_srli_epi32::<8>(w),
                    ))
                }
                2 => {
                    let w = _mm512_loadu_si512(at.cast::<__m512i>());
                    let mask = _mm512_set1_epi32(0xFF);
                    Some((
                        _mm512_and_si512(w, mask),
                        _mm512_and_si512(_mm512_srli_epi32::<16>(w), mask),
                    ))
                }
                _ => None,
            }
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
        #[inline]
        unsafe fn fetch_pair512<T: Sample>(
            src: &Source<T>,
            index: __m512i,
            offsets2: __m512i,
            offsets1: __m512i,
        ) -> (__m512i, __m512i) {
            let zero = _mm512_setzero_si512();
            if !src.valid {
                return (zero, zero);
            }
            let first = _mm_cvtsi128_si32(_mm512_castsi512_si128(index));
            let run = _mm512_add_epi32(_mm512_set1_epi32(first), offsets2);
            if first >= 0
                && _mm512_cmpeq_epi32_mask(index, run) == 0xFFFF
                && let Some(pair) = load_pair512(src, first)
            {
                return pair;
            }
            let last = _mm512_set1_epi32(src.len - 1);
            let clamped = _mm512_min_epi32(_mm512_max_epi32(index, zero), last);
            let in_last = _mm512_cmpgt_epi32_mask(clamped, _mm512_set1_epi32(src.len - 1 - src.pel))
                == 0;
            if Source::<T>::SIZE == 1 && src.pel <= 2 && in_last {
                const LOW_BYTES: u64 = 0x1111_1111_1111_1111;
                let cfirst = _mm_cvtsi128_si32(_mm512_castsi512_si128(clamped));
                let win_base = (cfirst - 4).max(0);
                let rel = _mm512_sub_epi32(clamped, _mm512_set1_epi32(win_base));
                let rel_odd = _mm512_add_epi32(rel, _mm512_set1_epi32(src.pel));
                if _mm512_test_epi32_mask(rel, _mm512_set1_epi32(!63)) == 0
                    && _mm512_test_epi32_mask(rel_odd, _mm512_set1_epi32(!63)) == 0
                    && (win_base as usize) + 64 <= src.readable
                {
                    let win = _mm512_loadu_si512(src.data.add(win_base as usize).cast::<__m512i>());
                    return (
                        _mm512_maskz_permutexvar_epi8(LOW_BYTES, rel, win),
                        _mm512_maskz_permutexvar_epi8(LOW_BYTES, rel_odd, win),
                    );
                }
                if _mm512_test_epi32_mask(rel, _mm512_set1_epi32(!127)) == 0
                    && _mm512_test_epi32_mask(rel_odd, _mm512_set1_epi32(!127)) == 0
                    && (win_base as usize) + 128 <= src.readable
                {
                    let base_ptr = src.data.add(win_base as usize);
                    let win0 = _mm512_loadu_si512(base_ptr.cast::<__m512i>());
                    let win1 = _mm512_loadu_si512(base_ptr.add(64).cast::<__m512i>());
                    return (
                        _mm512_maskz_permutex2var_epi8(LOW_BYTES, win0, rel, win1),
                        _mm512_maskz_permutex2var_epi8(LOW_BYTES, win0, rel_odd, win1),
                    );
                }
            }
            let safe_limit = (src.readable as i32 - Source::<T>::WORD).min(src.len - 1 - src.pel);
            if ((Source::<T>::SIZE == 1 && src.pel <= 2)
                || (Source::<T>::SIZE == 2 && src.pel == 1))
                && safe_limit >= 0
                && _mm512_cmpgt_epi32_mask(clamped, _mm512_set1_epi32(safe_limit)) == 0
            {
                let words = gather512(src, clamped);
                let mask = _mm512_set1_epi32(Source::<T>::MASK);
                let even = _mm512_and_si512(words, mask);
                let odd = if Source::<T>::SIZE == 1 {
                    if src.pel == 1 {
                        _mm512_and_si512(_mm512_srli_epi32::<8>(words), mask)
                    } else {
                        _mm512_and_si512(_mm512_srli_epi32::<16>(words), mask)
                    }
                } else {
                    _mm512_srli_epi32::<16>(words)
                };
                return (even, odd);
            }
            (
                fetch512(src, index, offsets1),
                fetch512(
                    src,
                    _mm512_add_epi32(index, _mm512_set1_epi32(src.pel)),
                    offsets1,
                ),
            )
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
        #[inline]
        unsafe fn still_pair512<T: Sample>(
            src: &Source<T>,
            start: i32,
            chunk: __m512i,
            offsets2: __m512i,
            offsets1: __m512i,
        ) -> (__m512i, __m512i) {
            load_pair512(src, start)
                .unwrap_or_else(|| fetch_pair512(src, chunk, offsets2, offsets1))
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
        #[inline]
        unsafe fn store_pair512<T: Sample>(
            out: &mut [T],
            c: usize,
            n: usize,
            even: __m512i,
            odd: __m512i,
        ) {
            let zero = _mm512_setzero_si512();
            let e = _mm512_max_epi32(even, zero);
            let o = _mm512_max_epi32(odd, zero);
            let write = |ptr: *mut T| {
                if size_of::<T>() == 1 {
                    let eb = _mm512_cvtusepi32_epi8(e);
                    let ob = _mm512_cvtusepi32_epi8(o);
                    _mm_storeu_si128(ptr.cast::<__m128i>(), _mm_unpacklo_epi8(eb, ob));
                    _mm_storeu_si128(ptr.add(16).cast::<__m128i>(), _mm_unpackhi_epi8(eb, ob));
                } else {
                    let ew = _mm512_cvtusepi32_epi16(e);
                    let ow = _mm512_cvtusepi32_epi16(o);
                    let e_lo = _mm256_castsi256_si128(ew);
                    let e_hi = _mm256_extracti128_si256::<1>(ew);
                    let o_lo = _mm256_castsi256_si128(ow);
                    let o_hi = _mm256_extracti128_si256::<1>(ow);
                    _mm_storeu_si128(ptr.cast::<__m128i>(), _mm_unpacklo_epi16(e_lo, o_lo));
                    _mm_storeu_si128(ptr.add(8).cast::<__m128i>(), _mm_unpackhi_epi16(e_lo, o_lo));
                    _mm_storeu_si128(ptr.add(16).cast::<__m128i>(), _mm_unpacklo_epi16(e_hi, o_hi));
                    _mm_storeu_si128(ptr.add(24).cast::<__m128i>(), _mm_unpackhi_epi16(e_hi, o_hi));
                }
            };
            if c + 32 <= out.len() {
                write(out.as_mut_ptr().add(c));
            } else {
                let mut tmp = [T::default(); 32];
                write(tmp.as_mut_ptr());
                let take = (n - c).min(out.len().saturating_sub(c)).min(32);
                out[c..c + take].copy_from_slice(&tmp[..take]);
            }
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
        unsafe fn fast_512<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            next: bool,
            base: i32,
            n: usize,
            l: &[&[i32]; 2],
        ) {
            let src = if next { &p.a } else { &p.b };
            let offsets = offsets512(p.pel);
            let step = _mm512_set1_epi32(16 * p.pel);
            let pitch = pitch512(p.pitch_i32);
            let mut chunk = _mm512_add_epi32(_mm512_set1_epi32(base), offsets);
            let mut c = 0;
            while c < n {
                let value = fetch512(src, at512(chunk, pitch, l[0], l[1], c), offsets);
                store512(out, c, n, value);
                chunk = _mm512_add_epi32(chunk, step);
                c += 16;
            }
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
        unsafe fn no_mask_512<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 4],
            w: &Weights,
            median: bool,
        ) {
            let offsets = offsets512(p.pel);
            let step = _mm512_set1_epi32(16 * p.pel);
            let pitch = pitch512(p.pitch_i32);
            let (t, rt, tb, rtb) = (
                _mm512_set1_epi32(w.t),
                _mm512_set1_epi32(w.rt),
                _mm512_set1_epi32(w.tb),
                _mm512_set1_epi32(w.rtb),
            );
            let mut chunk = _mm512_add_epi32(_mm512_set1_epi32(base), offsets);
            let mut start = base;
            let mut c = 0;
            while c < n {
                let pb = fetch512(&p.b, at512(chunk, pitch, l[0], l[1], c), offsets);
                let pa = fetch512(&p.a, at512(chunk, pitch, l[2], l[3], c), offsets);
                let value = if median {
                    let b0 = still512(&p.b, start, chunk, offsets);
                    let a0 = still512(&p.a, start, chunk, offsets);
                    let mid = blend256_512::<T>(b0, a0, rtb, tb);
                    _mm512_max_epi32(
                        _mm512_min_epi32(mid, _mm512_max_epi32(pb, pa)),
                        _mm512_min_epi32(pb, pa),
                    )
                } else {
                    blend256_512::<T>(pb, pa, rt, t)
                };
                store512(out, c, n, value);
                start += 16 * p.pel;
                chunk = _mm512_add_epi32(chunk, step);
                c += 16;
            }
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
        unsafe fn normal_512<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 6],
            w: &Weights,
            simple: bool,
        ) {
            let offsets = offsets512(p.pel);
            let step = _mm512_set1_epi32(16 * p.pel);
            let pitch = pitch512(p.pitch_i32);
            let (t, rt, full) = (
                _mm512_set1_epi32(w.t),
                _mm512_set1_epi32(w.rt),
                _mm512_set1_epi32(255),
            );
            let mut chunk = _mm512_add_epi32(_mm512_set1_epi32(base), offsets);
            let mut start = base;
            let mut c = 0;
            while c < n {
                let pb = fetch512(&p.b, at512(chunk, pitch, l[0], l[1], c), offsets);
                let pa = fetch512(&p.a, at512(chunk, pitch, l[2], l[3], c), offsets);
                let m9 = lane512(l[4], c);
                let m8 = lane512(l[5], c);
                let (first, second) = if simple {
                    let f = if size_of::<T>() == 1 && _mm512_test_epi32_mask(m9, m9) == 0 {
                        pb
                    } else {
                        blend255_512::<T>(pb, pa, m9, full)
                    };
                    let s = if size_of::<T>() == 1 && _mm512_test_epi32_mask(m8, m8) == 0 {
                        pa
                    } else {
                        blend255_512::<T>(pa, pb, m8, full)
                    };
                    (f, s)
                } else {
                    let bb = _mm512_mullo_epi32(_mm512_sub_epi32(full, m9), pb);
                    let aa = _mm512_mullo_epi32(_mm512_sub_epi32(full, m8), pa);
                    let b0 = still512(&p.b, start, chunk, offsets);
                    let a0 = still512(&p.a, start, chunk, offsets);
                    let inner1 = _mm512_srli_epi32::<8>(_mm512_add_epi32(
                        _mm512_mullo_epi32(m9, _mm512_add_epi32(aa, _mm512_mullo_epi32(m8, b0))),
                        full,
                    ));
                    let first = _mm512_srai_epi32::<8>(_mm512_add_epi32(
                        _mm512_add_epi32(bb, inner1),
                        full,
                    ));
                    let inner2 = _mm512_srli_epi32::<8>(_mm512_add_epi32(
                        _mm512_mullo_epi32(_mm512_add_epi32(bb, _mm512_mullo_epi32(a0, m9)), m8),
                        full,
                    ));
                    let second = _mm512_srai_epi32::<8>(_mm512_add_epi32(
                        _mm512_add_epi32(aa, inner2),
                        full,
                    ));
                    (first, second)
                };
                let value = blend256_512::<T>(first, second, rt, t);
                store512(out, c, n, value);
                start += 16 * p.pel;
                chunk = _mm512_add_epi32(chunk, step);
                c += 16;
            }
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
        unsafe fn extended_512<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 10],
            w: &Weights,
        ) {
            let offsets = offsets512(p.pel);
            let step = _mm512_set1_epi32(16 * p.pel);
            let pitch = pitch512(p.pitch_i32);
            let (t, rt, full) = (
                _mm512_set1_epi32(w.t),
                _mm512_set1_epi32(w.rt),
                _mm512_set1_epi32(255),
            );
            let mut chunk = _mm512_add_epi32(_mm512_set1_epi32(base), offsets);
            let mut c = 0;
            while c < n {
                let c0 = fetch512(&p.b, at512(chunk, pitch, l[0], l[1], c), offsets);
                let n0 = fetch512(&p.a, at512(chunk, pitch, l[2], l[3], c), offsets);
                let m9 = lane512(l[8], c);
                let m8 = lane512(l[9], c);
                let has_m9 = _mm512_test_epi32_mask(m9, m9) != 0;
                let has_m8 = _mm512_test_epi32_mask(m8, m8) != 0;
                let value = if !has_m9 && !has_m8 {
                    let first = if size_of::<T>() == 1 {
                        c0
                    } else {
                        let c0_255 = _mm512_sub_epi32(_mm512_slli_epi32::<8>(c0), c0);
                        _mm512_srai_epi32::<8>(_mm512_add_epi32(c0_255, full))
                    };
                    let second = if size_of::<T>() == 1 {
                        n0
                    } else {
                        let n0_255 = _mm512_sub_epi32(_mm512_slli_epi32::<8>(n0), n0);
                        _mm512_srai_epi32::<8>(_mm512_add_epi32(n0_255, full))
                    };
                    blend256_512::<T>(first, second, rt, t)
                } else {
                    let lo = _mm512_min_epi32(c0, n0);
                    let hi = _mm512_max_epi32(c0, n0);
                    let first = if has_m9 {
                        let n1 = fetch512(&p.a, at512(chunk, pitch, l[6], l[7], c), offsets);
                        let cn = _mm512_max_epi32(lo, _mm512_min_epi32(n1, hi));
                        blend255_512::<T>(c0, cn, m9, full)
                    } else if size_of::<T>() == 1 {
                        c0
                    } else {
                        let c0_255 = _mm512_sub_epi32(_mm512_slli_epi32::<8>(c0), c0);
                        _mm512_srai_epi32::<8>(_mm512_add_epi32(c0_255, full))
                    };
                    let second = if has_m8 {
                        let c1 = fetch512(&p.b, at512(chunk, pitch, l[4], l[5], c), offsets);
                        let cc = _mm512_max_epi32(lo, _mm512_min_epi32(c1, hi));
                        blend255_512::<T>(n0, cc, m8, full)
                    } else if size_of::<T>() == 1 {
                        n0
                    } else {
                        let n0_255 = _mm512_sub_epi32(_mm512_slli_epi32::<8>(n0), n0);
                        _mm512_srai_epi32::<8>(_mm512_add_epi32(n0_255, full))
                    };
                    blend256_512::<T>(first, second, rt, t)
                };
                store512(out, c, n, value);
                chunk = _mm512_add_epi32(chunk, step);
                c += 16;
            }
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
        unsafe fn fast_sad_512<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            next: bool,
            base: i32,
            n: usize,
            l: &[&[i32]; 3],
        ) {
            let src = if next { &p.a } else { &p.b };
            let offsets = offsets512(p.pel);
            let step = _mm512_set1_epi32(16 * p.pel);
            let pitch = pitch512(p.pitch_i32);
            let full = _mm512_set1_epi32(255);
            let mut chunk = _mm512_add_epi32(_mm512_set1_epi32(base), offsets);
            let mut start = base;
            let mut c = 0;
            while c < n {
                let moved = fetch512(src, at512(chunk, pitch, l[0], l[1], c), offsets);
                let s = lane512(l[2], c);
                let value = if _mm512_test_epi32_mask(s, s) == 0 {
                    if size_of::<T>() == 1 {
                        moved
                    } else {
                        let m255 = _mm512_sub_epi32(_mm512_slli_epi32::<8>(moved), moved);
                        _mm512_srai_epi32::<8>(_mm512_add_epi32(m255, full))
                    }
                } else {
                    let still = still512(src, start, chunk, offsets);
                    blend255_512::<T>(moved, still, s, full)
                };
                store512(out, c, n, value);
                start += 16 * p.pel;
                chunk = _mm512_add_epi32(chunk, step);
                c += 16;
            }
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
        unsafe fn no_mask_sad_512<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 5],
            w: &Weights,
            median: bool,
        ) {
            let offsets = offsets512(p.pel);
            let step = _mm512_set1_epi32(16 * p.pel);
            let pitch = pitch512(p.pitch_i32);
            let (t, rt, tb, rtb, full) = (
                _mm512_set1_epi32(w.t),
                _mm512_set1_epi32(w.rt),
                _mm512_set1_epi32(w.tb),
                _mm512_set1_epi32(w.rtb),
                _mm512_set1_epi32(255),
            );
            let mut chunk = _mm512_add_epi32(_mm512_set1_epi32(base), offsets);
            let mut start = base;
            let mut c = 0;
            while c < n {
                let pb = fetch512(&p.b, at512(chunk, pitch, l[0], l[1], c), offsets);
                let pa = fetch512(&p.a, at512(chunk, pitch, l[2], l[3], c), offsets);
                let s = lane512(l[4], c);
                let has_s = _mm512_test_epi32_mask(s, s) != 0;
                let res = if !median && !has_s {
                    let value = blend256_512::<T>(pb, pa, rt, t);
                    if size_of::<T>() == 1 {
                        value
                    } else {
                        let v255 = _mm512_sub_epi32(_mm512_slli_epi32::<8>(value), value);
                        _mm512_srai_epi32::<8>(_mm512_add_epi32(v255, full))
                    }
                } else {
                    let b0 = still512(&p.b, start, chunk, offsets);
                    let a0 = still512(&p.a, start, chunk, offsets);
                    let mid = blend256_512::<T>(b0, a0, rtb, tb);
                    let value = if median {
                        _mm512_max_epi32(
                            _mm512_min_epi32(mid, _mm512_max_epi32(pb, pa)),
                            _mm512_min_epi32(pb, pa),
                        )
                    } else {
                        blend256_512::<T>(pb, pa, rt, t)
                    };
                    if has_s {
                        blend255_512::<T>(value, mid, s, full)
                    } else if size_of::<T>() == 1 {
                        value
                    } else {
                        let v255 = _mm512_sub_epi32(_mm512_slli_epi32::<8>(value), value);
                        _mm512_srai_epi32::<8>(_mm512_add_epi32(v255, full))
                    }
                };
                store512(out, c, n, res);
                start += 16 * p.pel;
                chunk = _mm512_add_epi32(chunk, step);
                c += 16;
            }
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
        unsafe fn normal_sad_512<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 7],
            w: &Weights,
            simple: bool,
        ) {
            let offsets = offsets512(p.pel);
            let step = _mm512_set1_epi32(16 * p.pel);
            let pitch = pitch512(p.pitch_i32);
            let (t, rt, tb, rtb, full, wrap) = (
                _mm512_set1_epi32(w.t),
                _mm512_set1_epi32(w.rt),
                _mm512_set1_epi32(w.tb),
                _mm512_set1_epi32(w.rtb),
                _mm512_set1_epi32(255),
                _mm512_set1_epi32(w.wrap),
            );
            let odd = _mm512_set1_epi32(265 - w.tb);
            let mut chunk = _mm512_add_epi32(_mm512_set1_epi32(base), offsets);
            let mut start = base;
            let mut c = 0;
            while c < n {
                let pb = fetch512(&p.b, at512(chunk, pitch, l[0], l[1], c), offsets);
                let pa = fetch512(&p.a, at512(chunk, pitch, l[2], l[3], c), offsets);
                let m9 = lane512(l[4], c);
                let m8 = lane512(l[5], c);
                let s = lane512(l[6], c);
                let first = if size_of::<T>() == 1 && _mm512_test_epi32_mask(m9, m9) == 0 {
                    pb
                } else {
                    blend255_512::<T>(pb, pa, m9, full)
                };
                let second = if size_of::<T>() == 1 && _mm512_test_epi32_mask(m8, m8) == 0 {
                    pa
                } else {
                    blend255_512::<T>(pa, pb, m8, full)
                };
                let has_s = _mm512_test_epi32_mask(s, s) != 0;
                let value = if simple {
                    blend256_512::<T>(first, second, rt, t)
                } else {
                    let ab0 = still512(&p.aux_b, start, chunk, offsets);
                    let aa0 = still512(&p.aux_a, start, chunk, offsets);
                    let time = blend256_512::<T>(ab0, aa0, rt, t);
                    _mm512_max_epi32(
                        _mm512_min_epi32(time, _mm512_max_epi32(first, second)),
                        _mm512_min_epi32(first, second),
                    )
                };
                let res = if has_s {
                    let b0 = still512(&p.b, start, chunk, offsets);
                    let a0 = still512(&p.a, start, chunk, offsets);
                    let mid = if simple {
                        let weight = if p.index == 1 { odd } else { rtb };
                        let mixed = _mm512_and_si512(
                            _mm512_add_epi32(
                                _mm512_mullo_epi32(weight, b0),
                                _mm512_mullo_epi32(tb, a0),
                            ),
                            wrap,
                        );
                        _mm512_srai_epi32::<8>(mixed)
                    } else {
                        blend256_512::<T>(b0, a0, rtb, tb)
                    };
                    blend255_512::<T>(value, mid, s, full)
                } else if size_of::<T>() == 1 {
                    value
                } else {
                    let v255 = _mm512_sub_epi32(_mm512_slli_epi32::<8>(value), value);
                    _mm512_srai_epi32::<8>(_mm512_add_epi32(v255, full))
                };
                store512(out, c, n, res);
                start += 16 * p.pel;
                chunk = _mm512_add_epi32(chunk, step);
                c += 16;
            }
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
        unsafe fn extended_sad_512<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 11],
            w: &Weights,
        ) {
            let offsets = offsets512(p.pel);
            let step = _mm512_set1_epi32(16 * p.pel);
            let pitch = pitch512(p.pitch_i32);
            let (t, rt, tb, rtb, full) = (
                _mm512_set1_epi32(w.t),
                _mm512_set1_epi32(w.rt),
                _mm512_set1_epi32(w.tb),
                _mm512_set1_epi32(w.rtb),
                _mm512_set1_epi32(255),
            );
            let mut chunk = _mm512_add_epi32(_mm512_set1_epi32(base), offsets);
            let mut start = base;
            let mut c = 0;
            while c < n {
                let c0 = fetch512(&p.b, at512(chunk, pitch, l[0], l[1], c), offsets);
                let n0 = fetch512(&p.a, at512(chunk, pitch, l[2], l[3], c), offsets);
                let m9 = lane512(l[8], c);
                let m8 = lane512(l[9], c);
                let s = lane512(l[10], c);
                let has_m9 = _mm512_test_epi32_mask(m9, m9) != 0;
                let has_m8 = _mm512_test_epi32_mask(m8, m8) != 0;
                let value = if !has_m9 && !has_m8 {
                    let first = if size_of::<T>() == 1 {
                        c0
                    } else {
                        let c0_255 = _mm512_sub_epi32(_mm512_slli_epi32::<8>(c0), c0);
                        _mm512_srai_epi32::<8>(_mm512_add_epi32(c0_255, full))
                    };
                    let second = if size_of::<T>() == 1 {
                        n0
                    } else {
                        let n0_255 = _mm512_sub_epi32(_mm512_slli_epi32::<8>(n0), n0);
                        _mm512_srai_epi32::<8>(_mm512_add_epi32(n0_255, full))
                    };
                    blend256_512::<T>(first, second, rt, t)
                } else {
                    let lo = _mm512_min_epi32(c0, n0);
                    let hi = _mm512_max_epi32(c0, n0);
                    let first = if has_m9 {
                        let n1 = fetch512(&p.a, at512(chunk, pitch, l[6], l[7], c), offsets);
                        let cn = _mm512_max_epi32(lo, _mm512_min_epi32(n1, hi));
                        blend255_512::<T>(c0, cn, m9, full)
                    } else if size_of::<T>() == 1 {
                        c0
                    } else {
                        let c0_255 = _mm512_sub_epi32(_mm512_slli_epi32::<8>(c0), c0);
                        _mm512_srai_epi32::<8>(_mm512_add_epi32(c0_255, full))
                    };
                    let second = if has_m8 {
                        let c1 = fetch512(&p.b, at512(chunk, pitch, l[4], l[5], c), offsets);
                        let cc = _mm512_max_epi32(lo, _mm512_min_epi32(c1, hi));
                        blend255_512::<T>(n0, cc, m8, full)
                    } else if size_of::<T>() == 1 {
                        n0
                    } else {
                        let n0_255 = _mm512_sub_epi32(_mm512_slli_epi32::<8>(n0), n0);
                        _mm512_srai_epi32::<8>(_mm512_add_epi32(n0_255, full))
                    };
                    blend256_512::<T>(first, second, rt, t)
                };
                let res = if _mm512_test_epi32_mask(s, s) == 0 {
                    if size_of::<T>() == 1 {
                        value
                    } else {
                        let v255 = _mm512_sub_epi32(_mm512_slli_epi32::<8>(value), value);
                        _mm512_srai_epi32::<8>(_mm512_add_epi32(v255, full))
                    }
                } else {
                    let b0 = still512(&p.b, start, chunk, offsets);
                    let a0 = still512(&p.a, start, chunk, offsets);
                    let mid = blend256_512::<T>(b0, a0, rtb, tb);
                    blend255_512::<T>(value, mid, s, full)
                };
                store512(out, c, n, res);
                start += 16 * p.pel;
                chunk = _mm512_add_epi32(chunk, step);
                c += 16;
            }
        }

        pub(in super::super) fn fast<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            next: bool,
            base: i32,
            n: usize,
            l: &[&[i32]; 2],
        ) {
            if p.avx512 {
                unsafe { fast_512(out, p, next, base, n, l) };
                return;
            }
            let src = if next { p.a } else { p.b };
            p.run(out, base, n, |c, chunk, _| {
                src.fetch(p.at(chunk, l[0], l[1], c))
            });
        }

        pub(in super::super) fn no_mask<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 4],
            w: &Weights,
            median: bool,
        ) {
            if p.avx512 {
                unsafe { no_mask_512(out, p, base, n, l, w, median) };
                return;
            }
            let (t, rt, tb, rtb) = (splat(w.t), splat(w.rt), splat(w.tb), splat(w.rtb));
            p.run(out, base, n, |c, chunk, start| {
                let pb = p.b.fetch(p.at(chunk, l[0], l[1], c));
                let pa = p.a.fetch(p.at(chunk, l[2], l[3], c));
                if median {
                    let b0 = p.b.still(start, chunk);
                    let a0 = p.a.still(start, chunk);
                    let mid = shr8(add(mul(rtb, b0), mul(tb, a0)));
                    max(min(mid, max(pb, pa)), min(pb, pa))
                } else {
                    shr8(add(mul(rt, pb), mul(t, pa)))
                }
            });
        }

        pub(in super::super) fn normal<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 6],
            w: &Weights,
            simple: bool,
        ) {
            if p.avx512 {
                unsafe { normal_512(out, p, base, n, l, w, simple) };
                return;
            }
            let (t, rt, full) = (splat(w.t), splat(w.rt), splat(255));
            p.run(out, base, n, |c, chunk, start| {
                let pb = p.b.fetch(p.at(chunk, l[0], l[1], c));
                let pa = p.a.fetch(p.at(chunk, l[2], l[3], c));
                let m9 = p.lane(l[4], c);
                let m8 = p.lane(l[5], c);
                let (first, second) = if simple {
                    (
                        shr8(add(add(mul(sub(full, m9), pb), mul(m9, pa)), full)),
                        shr8(add(add(mul(sub(full, m8), pa), mul(m8, pb)), full)),
                    )
                } else {
                    let bb = mul(sub(full, m9), pb);
                    let aa = mul(sub(full, m8), pa);
                    let b0 = p.b.still(start, chunk);
                    let a0 = p.a.still(start, chunk);
                    let inner = shr8u(add(mul(m9, add(aa, mul(m8, b0))), full));
                    let first = shr8(add(add(bb, inner), full));
                    let inner = shr8u(add(mul(add(bb, mul(a0, m9)), m8), full));
                    (first, shr8(add(add(aa, inner), full)))
                };
                shr8(add(mul(rt, first), mul(t, second)))
            });
        }

        pub(in super::super) fn extended<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 10],
            w: &Weights,
        ) {
            if p.avx512 {
                unsafe { extended_512(out, p, base, n, l, w) };
                return;
            }
            let (t, rt, full) = (splat(w.t), splat(w.rt), splat(255));
            p.run(out, base, n, |c, chunk, _| {
                let c0 = p.b.fetch(p.at(chunk, l[0], l[1], c));
                let n0 = p.a.fetch(p.at(chunk, l[2], l[3], c));
                let m9 = p.lane(l[8], c);
                let m8 = p.lane(l[9], c);
                let has_m9 = unsafe { _mm256_testz_si256(m9, m9) } == 0;
                let has_m8 = unsafe { _mm256_testz_si256(m8, m8) } == 0;
                if !has_m9 && !has_m8 {
                    let first = if size_of::<T>() == 1 {
                        c0
                    } else {
                        shr8(add(sub(unsafe { _mm256_slli_epi32::<8>(c0) }, c0), full))
                    };
                    let second = if size_of::<T>() == 1 {
                        n0
                    } else {
                        shr8(add(sub(unsafe { _mm256_slli_epi32::<8>(n0) }, n0), full))
                    };
                    return blend256_256::<T>(first, second, rt, t);
                }
                let lo = min(c0, n0);
                let hi = max(c0, n0);
                let first = if has_m9 {
                    let n1 = p.a.fetch(p.at(chunk, l[6], l[7], c));
                    let cn = max(lo, min(n1, hi));
                    blend255_256::<T>(c0, cn, m9, full)
                } else if size_of::<T>() == 1 {
                    c0
                } else {
                    shr8(add(sub(unsafe { _mm256_slli_epi32::<8>(c0) }, c0), full))
                };
                let second = if has_m8 {
                    let c1 = p.b.fetch(p.at(chunk, l[4], l[5], c));
                    let cc = max(lo, min(c1, hi));
                    blend255_256::<T>(n0, cc, m8, full)
                } else if size_of::<T>() == 1 {
                    n0
                } else {
                    shr8(add(sub(unsafe { _mm256_slli_epi32::<8>(n0) }, n0), full))
                };
                blend256_256::<T>(first, second, rt, t)
            });
        }

        pub(in super::super) fn fast_sad<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            next: bool,
            base: i32,
            n: usize,
            l: &[&[i32]; 3],
        ) {
            if p.avx512 {
                unsafe { fast_sad_512(out, p, next, base, n, l) };
                return;
            }
            let src = if next { p.a } else { p.b };
            let full = splat(255);
            p.run(out, base, n, |c, chunk, start| {
                let moved = src.fetch(p.at(chunk, l[0], l[1], c));
                let s = p.lane(l[2], c);
                if unsafe { _mm256_testz_si256(s, s) } != 0 {
                    if size_of::<T>() == 1 {
                        moved
                    } else {
                        shr8(add(sub(unsafe { _mm256_slli_epi32::<8>(moved) }, moved), full))
                    }
                } else {
                    let still = src.still(start, chunk);
                    blend255_256::<T>(moved, still, s, full)
                }
            });
        }

        pub(in super::super) fn no_mask_sad<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 5],
            w: &Weights,
            median: bool,
        ) {
            if p.avx512 {
                unsafe { no_mask_sad_512(out, p, base, n, l, w, median) };
                return;
            }
            let (t, rt, tb, rtb, full) = (
                splat(w.t),
                splat(w.rt),
                splat(w.tb),
                splat(w.rtb),
                splat(255),
            );
            p.run(out, base, n, |c, chunk, start| {
                let pb = p.b.fetch(p.at(chunk, l[0], l[1], c));
                let pa = p.a.fetch(p.at(chunk, l[2], l[3], c));
                let s = p.lane(l[4], c);
                let has_s = unsafe { _mm256_testz_si256(s, s) } == 0;
                if !median && !has_s {
                    let value = blend256_256::<T>(pb, pa, rt, t);
                    return if size_of::<T>() == 1 {
                        value
                    } else {
                        shr8(add(sub(unsafe { _mm256_slli_epi32::<8>(value) }, value), full))
                    };
                }
                let b0 = p.b.still(start, chunk);
                let a0 = p.a.still(start, chunk);
                let mid = blend256_256::<T>(b0, a0, rtb, tb);
                let value = if median {
                    max(min(mid, max(pb, pa)), min(pb, pa))
                } else {
                    blend256_256::<T>(pb, pa, rt, t)
                };
                if has_s {
                    blend255_256::<T>(value, mid, s, full)
                } else if size_of::<T>() == 1 {
                    value
                } else {
                    shr8(add(sub(unsafe { _mm256_slli_epi32::<8>(value) }, value), full))
                }
            });
        }

        pub(in super::super) fn normal_sad<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 7],
            w: &Weights,
            simple: bool,
        ) {
            if p.avx512 {
                unsafe { normal_sad_512(out, p, base, n, l, w, simple) };
                return;
            }
            let (t, rt, tb, rtb, full) = (
                splat(w.t),
                splat(w.rt),
                splat(w.tb),
                splat(w.rtb),
                splat(255),
            );
            let odd = splat(265 - w.tb);
            p.run(out, base, n, |c, chunk, start| {
                let pb = p.b.fetch(p.at(chunk, l[0], l[1], c));
                let pa = p.a.fetch(p.at(chunk, l[2], l[3], c));
                let m9 = p.lane(l[4], c);
                let m8 = p.lane(l[5], c);
                let s = p.lane(l[6], c);
                let first = if size_of::<T>() == 1 && unsafe { _mm256_testz_si256(m9, m9) } != 0 {
                    pb
                } else {
                    blend255_256::<T>(pb, pa, m9, full)
                };
                let second = if size_of::<T>() == 1 && unsafe { _mm256_testz_si256(m8, m8) } != 0 {
                    pa
                } else {
                    blend255_256::<T>(pa, pb, m8, full)
                };
                let value = if simple {
                    blend256_256::<T>(first, second, rt, t)
                } else {
                    let ab0 = p.aux_b.still(start, chunk);
                    let aa0 = p.aux_a.still(start, chunk);
                    let time = blend256_256::<T>(ab0, aa0, rt, t);
                    max(min(time, max(first, second)), min(first, second))
                };
                if unsafe { _mm256_testz_si256(s, s) } != 0 {
                    if size_of::<T>() == 1 {
                        value
                    } else {
                        shr8(add(sub(unsafe { _mm256_slli_epi32::<8>(value) }, value), full))
                    }
                } else {
                    let b0 = p.b.still(start, chunk);
                    let a0 = p.a.still(start, chunk);
                    let mid = if simple {
                        let weight = if p.index == 1 { odd } else { rtb };
                        let mixed = add(mul(weight, b0), mul(tb, a0));
                        let mixed = unsafe { _mm256_and_si256(mixed, splat(w.wrap)) };
                        shr8(mixed)
                    } else {
                        blend256_256::<T>(b0, a0, rtb, tb)
                    };
                    blend255_256::<T>(value, mid, s, full)
                }
            });
        }

        pub(in super::super) fn extended_sad<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 11],
            w: &Weights,
        ) {
            if p.avx512 {
                unsafe { extended_sad_512(out, p, base, n, l, w) };
                return;
            }
            let (t, rt, tb, rtb, full) = (
                splat(w.t),
                splat(w.rt),
                splat(w.tb),
                splat(w.rtb),
                splat(255),
            );
            p.run(out, base, n, |c, chunk, start| {
                let c0 = p.b.fetch(p.at(chunk, l[0], l[1], c));
                let n0 = p.a.fetch(p.at(chunk, l[2], l[3], c));
                let m9 = p.lane(l[8], c);
                let m8 = p.lane(l[9], c);
                let s = p.lane(l[10], c);
                let has_m9 = unsafe { _mm256_testz_si256(m9, m9) } == 0;
                let has_m8 = unsafe { _mm256_testz_si256(m8, m8) } == 0;
                let value = if !has_m9 && !has_m8 {
                    let first = if size_of::<T>() == 1 {
                        c0
                    } else {
                        shr8(add(mul(c0, full), full))
                    };
                    let second = if size_of::<T>() == 1 {
                        n0
                    } else {
                        shr8(add(mul(n0, full), full))
                    };
                    blend256_256::<T>(first, second, rt, t)
                } else {
                    let lo = min(c0, n0);
                    let hi = max(c0, n0);
                    let first = if has_m9 {
                        let n1 = p.a.fetch(p.at(chunk, l[6], l[7], c));
                        let cn = max(lo, min(n1, hi));
                        blend255_256::<T>(c0, cn, m9, full)
                    } else if size_of::<T>() == 1 {
                        c0
                    } else {
                        shr8(add(mul(c0, full), full))
                    };
                    let second = if has_m8 {
                        let c1 = p.b.fetch(p.at(chunk, l[4], l[5], c));
                        let cc = max(lo, min(c1, hi));
                        blend255_256::<T>(n0, cc, m8, full)
                    } else if size_of::<T>() == 1 {
                        n0
                    } else {
                        shr8(add(mul(n0, full), full))
                    };
                    blend256_256::<T>(first, second, rt, t)
                };
                if unsafe { _mm256_testz_si256(s, s) } != 0 {
                    if size_of::<T>() == 1 {
                        value
                    } else {
                        shr8(add(sub(unsafe { _mm256_slli_epi32::<8>(value) }, value), full))
                    }
                } else {
                    let b0 = p.b.still(start, chunk);
                    let a0 = p.a.still(start, chunk);
                    let mid = blend256_256::<T>(b0, a0, rtb, tb);
                    blend255_256::<T>(value, mid, s, full)
                }
            });
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
        #[inline]
        #[allow(clippy::too_many_arguments)]
        unsafe fn blend_ext_512<T: Sample>(
            c0: __m512i,
            n0: __m512i,
            c1: Option<__m512i>,
            n1: Option<__m512i>,
            m9: __m512i,
            m8: __m512i,
            t: __m512i,
            full: __m512i,
        ) -> __m512i {
            let first = match n1 {
                Some(n1) => {
                    let lo = _mm512_min_epi32(c0, n0);
                    let hi = _mm512_max_epi32(c0, n0);
                    let cn = _mm512_max_epi32(lo, _mm512_min_epi32(n1, hi));
                    let c0_255 = _mm512_sub_epi32(_mm512_slli_epi32::<8>(c0), c0);
                    _mm512_srai_epi32::<8>(_mm512_add_epi32(
                        _mm512_add_epi32(c0_255, _mm512_mullo_epi32(m9, _mm512_sub_epi32(cn, c0))),
                        full,
                    ))
                }
                None if size_of::<T>() == 1 => c0,
                None => {
                    let c0_255 = _mm512_sub_epi32(_mm512_slli_epi32::<8>(c0), c0);
                    _mm512_srai_epi32::<8>(_mm512_add_epi32(c0_255, full))
                }
            };
            let second = match c1 {
                Some(c1) => {
                    let lo = _mm512_min_epi32(c0, n0);
                    let hi = _mm512_max_epi32(c0, n0);
                    let cc = _mm512_max_epi32(lo, _mm512_min_epi32(c1, hi));
                    let n0_255 = _mm512_sub_epi32(_mm512_slli_epi32::<8>(n0), n0);
                    _mm512_srai_epi32::<8>(_mm512_add_epi32(
                        _mm512_add_epi32(n0_255, _mm512_mullo_epi32(m8, _mm512_sub_epi32(cc, n0))),
                        full,
                    ))
                }
                None if size_of::<T>() == 1 => n0,
                None => {
                    let n0_255 = _mm512_sub_epi32(_mm512_slli_epi32::<8>(n0), n0);
                    _mm512_srai_epi32::<8>(_mm512_add_epi32(n0_255, full))
                }
            };
            _mm512_srai_epi32::<8>(_mm512_add_epi32(
                _mm512_slli_epi32::<8>(first),
                _mm512_mullo_epi32(t, _mm512_sub_epi32(second, first)),
            ))
        }

        #[inline(always)]
        #[allow(clippy::too_many_arguments)]
        fn blend_ext_256<T: Sample>(
            c0: __m256i,
            n0: __m256i,
            c1: Option<__m256i>,
            n1: Option<__m256i>,
            m9: __m256i,
            m8: __m256i,
            t: __m256i,
            full: __m256i,
        ) -> __m256i {
            let first = match n1 {
                Some(n1) => {
                    let lo = min(c0, n0);
                    let hi = max(c0, n0);
                    let cn = max(lo, min(n1, hi));
                    let c0_255 = sub(unsafe { _mm256_slli_epi32::<8>(c0) }, c0);
                    shr8(add(add(c0_255, mul(m9, sub(cn, c0))), full))
                }
                None if size_of::<T>() == 1 => c0,
                None => shr8(add(sub(unsafe { _mm256_slli_epi32::<8>(c0) }, c0), full)),
            };
            let second = match c1 {
                Some(c1) => {
                    let lo = min(c0, n0);
                    let hi = max(c0, n0);
                    let cc = max(lo, min(c1, hi));
                    let n0_255 = sub(unsafe { _mm256_slli_epi32::<8>(n0) }, n0);
                    shr8(add(add(n0_255, mul(m8, sub(cc, n0))), full))
                }
                None if size_of::<T>() == 1 => n0,
                None => shr8(add(sub(unsafe { _mm256_slli_epi32::<8>(n0) }, n0), full)),
            };
            shr8(add(
                unsafe { _mm256_slli_epi32::<8>(first) },
                mul(t, sub(second, first)),
            ))
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
        unsafe fn fast_pair_512<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            next: bool,
            base: i32,
            n: usize,
            l: &[&[i32]; 2],
        ) {
            let src = if next { &p.a } else { &p.b };
            let offsets1 = offsets512(p.pel);
            let offsets2 = offsets512(2 * p.pel);
            let step = _mm512_set1_epi32(32 * p.pel);
            let pitch = pitch512(p.pitch_i32);
            let mut chunk = _mm512_add_epi32(_mm512_set1_epi32(base), offsets2);
            let (mut pi, mut c) = (0, 0);
            while c < n {
                let (e, o) = fetch_pair512(src, at512(chunk, pitch, l[0], l[1], pi), offsets2, offsets1);
                store_pair512(out, c, n, e, o);
                chunk = _mm512_add_epi32(chunk, step);
                pi += 16;
                c += 32;
            }
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
        unsafe fn no_mask_pair_512<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 4],
            w: &Weights,
            median: bool,
        ) {
            let offsets1 = offsets512(p.pel);
            let offsets2 = offsets512(2 * p.pel);
            let step = _mm512_set1_epi32(32 * p.pel);
            let pitch = pitch512(p.pitch_i32);
            let (t, rt, tb, rtb) = (
                _mm512_set1_epi32(w.t),
                _mm512_set1_epi32(w.rt),
                _mm512_set1_epi32(w.tb),
                _mm512_set1_epi32(w.rtb),
            );
            let mut chunk = _mm512_add_epi32(_mm512_set1_epi32(base), offsets2);
            let mut start = base;
            let (mut pi, mut c) = (0, 0);
            while c < n {
                let (pb_e, pb_o) =
                    fetch_pair512(&p.b, at512(chunk, pitch, l[0], l[1], pi), offsets2, offsets1);
                let (pa_e, pa_o) =
                    fetch_pair512(&p.a, at512(chunk, pitch, l[2], l[3], pi), offsets2, offsets1);
                let (ve, vo) = if median {
                    let (b0_e, b0_o) = still_pair512(&p.b, start, chunk, offsets2, offsets1);
                    let (a0_e, a0_o) = still_pair512(&p.a, start, chunk, offsets2, offsets1);
                    let med = |pb, pa, b0, a0| {
                        let mid = _mm512_srai_epi32::<8>(_mm512_add_epi32(
                            _mm512_mullo_epi32(rtb, b0),
                            _mm512_mullo_epi32(tb, a0),
                        ));
                        _mm512_max_epi32(
                            _mm512_min_epi32(mid, _mm512_max_epi32(pb, pa)),
                            _mm512_min_epi32(pb, pa),
                        )
                    };
                    (med(pb_e, pa_e, b0_e, a0_e), med(pb_o, pa_o, b0_o, a0_o))
                } else {
                    let mix = |pb, pa| {
                        _mm512_srai_epi32::<8>(_mm512_add_epi32(
                            _mm512_mullo_epi32(rt, pb),
                            _mm512_mullo_epi32(t, pa),
                        ))
                    };
                    (mix(pb_e, pa_e), mix(pb_o, pa_o))
                };
                store_pair512(out, c, n, ve, vo);
                start += 32 * p.pel;
                chunk = _mm512_add_epi32(chunk, step);
                pi += 16;
                c += 32;
            }
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
        unsafe fn normal_pair_512<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 6],
            w: &Weights,
            simple: bool,
        ) {
            let offsets1 = offsets512(p.pel);
            let offsets2 = offsets512(2 * p.pel);
            let step = _mm512_set1_epi32(32 * p.pel);
            let pitch = pitch512(p.pitch_i32);
            let (t, rt, full) = (
                _mm512_set1_epi32(w.t),
                _mm512_set1_epi32(w.rt),
                _mm512_set1_epi32(255),
            );
            let mut chunk = _mm512_add_epi32(_mm512_set1_epi32(base), offsets2);
            let mut start = base;
            let (mut pi, mut c) = (0, 0);
            while c < n {
                let (pb_e, pb_o) =
                    fetch_pair512(&p.b, at512(chunk, pitch, l[0], l[1], pi), offsets2, offsets1);
                let (pa_e, pa_o) =
                    fetch_pair512(&p.a, at512(chunk, pitch, l[2], l[3], pi), offsets2, offsets1);
                let m9 = lane512(l[4], pi);
                let m8 = lane512(l[5], pi);
                let inv9 = _mm512_sub_epi32(full, m9);
                let inv8 = _mm512_sub_epi32(full, m8);
                let (ve, vo) = if simple {
                    let blend = |pb, pa| {
                        let f = _mm512_srai_epi32::<8>(_mm512_add_epi32(
                            _mm512_add_epi32(_mm512_mullo_epi32(inv9, pb), _mm512_mullo_epi32(m9, pa)),
                            full,
                        ));
                        let s = _mm512_srai_epi32::<8>(_mm512_add_epi32(
                            _mm512_add_epi32(_mm512_mullo_epi32(inv8, pa), _mm512_mullo_epi32(m8, pb)),
                            full,
                        ));
                        _mm512_srai_epi32::<8>(_mm512_add_epi32(
                            _mm512_mullo_epi32(rt, f),
                            _mm512_mullo_epi32(t, s),
                        ))
                    };
                    (blend(pb_e, pa_e), blend(pb_o, pa_o))
                } else {
                    let (b0_e, b0_o) = still_pair512(&p.b, start, chunk, offsets2, offsets1);
                    let (a0_e, a0_o) = still_pair512(&p.a, start, chunk, offsets2, offsets1);
                    let blend = |pb, pa, b0, a0| {
                        let bb = _mm512_mullo_epi32(inv9, pb);
                        let aa = _mm512_mullo_epi32(inv8, pa);
                        let i1 = _mm512_srli_epi32::<8>(_mm512_add_epi32(
                            _mm512_mullo_epi32(m9, _mm512_add_epi32(aa, _mm512_mullo_epi32(m8, b0))),
                            full,
                        ));
                        let f = _mm512_srai_epi32::<8>(_mm512_add_epi32(_mm512_add_epi32(bb, i1), full));
                        let i2 = _mm512_srli_epi32::<8>(_mm512_add_epi32(
                            _mm512_mullo_epi32(_mm512_add_epi32(bb, _mm512_mullo_epi32(a0, m9)), m8),
                            full,
                        ));
                        let s = _mm512_srai_epi32::<8>(_mm512_add_epi32(_mm512_add_epi32(aa, i2), full));
                        _mm512_srai_epi32::<8>(_mm512_add_epi32(
                            _mm512_mullo_epi32(rt, f),
                            _mm512_mullo_epi32(t, s),
                        ))
                    };
                    (blend(pb_e, pa_e, b0_e, a0_e), blend(pb_o, pa_o, b0_o, a0_o))
                };
                store_pair512(out, c, n, ve, vo);
                start += 32 * p.pel;
                chunk = _mm512_add_epi32(chunk, step);
                pi += 16;
                c += 32;
            }
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
        unsafe fn extended_pair_512<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 10],
            w: &Weights,
        ) {
            let offsets1 = offsets512(p.pel);
            let offsets2 = offsets512(2 * p.pel);
            let step = _mm512_set1_epi32(32 * p.pel);
            let pitch = pitch512(p.pitch_i32);
            let (t, full) = (_mm512_set1_epi32(w.t), _mm512_set1_epi32(255));
            let mut chunk = _mm512_add_epi32(_mm512_set1_epi32(base), offsets2);
            let (mut pi, mut c) = (0, 0);
            while c < n {
                let (c0_e, c0_o) =
                    fetch_pair512(&p.b, at512(chunk, pitch, l[0], l[1], pi), offsets2, offsets1);
                let (n0_e, n0_o) =
                    fetch_pair512(&p.a, at512(chunk, pitch, l[2], l[3], pi), offsets2, offsets1);
                let m9 = lane512(l[8], pi);
                let m8 = lane512(l[9], pi);
                let has_m9 = _mm512_test_epi32_mask(m9, m9) != 0;
                let has_m8 = _mm512_test_epi32_mask(m8, m8) != 0;
                let (c1_e, c1_o) = if has_m8 {
                    let pair = fetch_pair512(&p.b, at512(chunk, pitch, l[4], l[5], pi), offsets2, offsets1);
                    (Some(pair.0), Some(pair.1))
                } else {
                    (None, None)
                };
                let (n1_e, n1_o) = if has_m9 {
                    let pair = fetch_pair512(&p.a, at512(chunk, pitch, l[6], l[7], pi), offsets2, offsets1);
                    (Some(pair.0), Some(pair.1))
                } else {
                    (None, None)
                };
                let ve = blend_ext_512::<T>(c0_e, n0_e, c1_e, n1_e, m9, m8, t, full);
                let vo = blend_ext_512::<T>(c0_o, n0_o, c1_o, n1_o, m9, m8, t, full);
                store_pair512(out, c, n, ve, vo);
                chunk = _mm512_add_epi32(chunk, step);
                pi += 16;
                c += 32;
            }
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
        unsafe fn fast_sad_pair_512<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            next: bool,
            base: i32,
            n: usize,
            l: &[&[i32]; 3],
        ) {
            let src = if next { &p.a } else { &p.b };
            let offsets1 = offsets512(p.pel);
            let offsets2 = offsets512(2 * p.pel);
            let step = _mm512_set1_epi32(32 * p.pel);
            let pitch = pitch512(p.pitch_i32);
            let full = _mm512_set1_epi32(255);
            let mut chunk = _mm512_add_epi32(_mm512_set1_epi32(base), offsets2);
            let mut start = base;
            let (mut pi, mut c) = (0, 0);
            while c < n {
                let (me, mo) =
                    fetch_pair512(src, at512(chunk, pitch, l[0], l[1], pi), offsets2, offsets1);
                let (se, so) = still_pair512(src, start, chunk, offsets2, offsets1);
                let s = lane512(l[2], pi);
                let inv = _mm512_sub_epi32(full, s);
                let mix = |m, st| {
                    _mm512_srai_epi32::<8>(_mm512_add_epi32(
                        _mm512_add_epi32(_mm512_mullo_epi32(inv, m), _mm512_mullo_epi32(s, st)),
                        full,
                    ))
                };
                store_pair512(out, c, n, mix(me, se), mix(mo, so));
                start += 32 * p.pel;
                chunk = _mm512_add_epi32(chunk, step);
                pi += 16;
                c += 32;
            }
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
        unsafe fn no_mask_sad_pair_512<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 5],
            w: &Weights,
            median: bool,
        ) {
            let offsets1 = offsets512(p.pel);
            let offsets2 = offsets512(2 * p.pel);
            let step = _mm512_set1_epi32(32 * p.pel);
            let pitch = pitch512(p.pitch_i32);
            let (t, rt, tb, rtb, full) = (
                _mm512_set1_epi32(w.t),
                _mm512_set1_epi32(w.rt),
                _mm512_set1_epi32(w.tb),
                _mm512_set1_epi32(w.rtb),
                _mm512_set1_epi32(255),
            );
            let mut chunk = _mm512_add_epi32(_mm512_set1_epi32(base), offsets2);
            let mut start = base;
            let (mut pi, mut c) = (0, 0);
            while c < n {
                let (pb_e, pb_o) =
                    fetch_pair512(&p.b, at512(chunk, pitch, l[0], l[1], pi), offsets2, offsets1);
                let (pa_e, pa_o) =
                    fetch_pair512(&p.a, at512(chunk, pitch, l[2], l[3], pi), offsets2, offsets1);
                let s = lane512(l[4], pi);
                let inv = _mm512_sub_epi32(full, s);
                let (b0_e, b0_o) = still_pair512(&p.b, start, chunk, offsets2, offsets1);
                let (a0_e, a0_o) = still_pair512(&p.a, start, chunk, offsets2, offsets1);
                let mix = |pb, pa, b0, a0| {
                    let mid = _mm512_srai_epi32::<8>(_mm512_add_epi32(
                        _mm512_mullo_epi32(rtb, b0),
                        _mm512_mullo_epi32(tb, a0),
                    ));
                    let value = if median {
                        _mm512_max_epi32(
                            _mm512_min_epi32(mid, _mm512_max_epi32(pb, pa)),
                            _mm512_min_epi32(pb, pa),
                        )
                    } else {
                        _mm512_srai_epi32::<8>(_mm512_add_epi32(
                            _mm512_mullo_epi32(rt, pb),
                            _mm512_mullo_epi32(t, pa),
                        ))
                    };
                    _mm512_srai_epi32::<8>(_mm512_add_epi32(
                        _mm512_add_epi32(_mm512_mullo_epi32(inv, value), _mm512_mullo_epi32(s, mid)),
                        full,
                    ))
                };
                store_pair512(out, c, n, mix(pb_e, pa_e, b0_e, a0_e), mix(pb_o, pa_o, b0_o, a0_o));
                start += 32 * p.pel;
                chunk = _mm512_add_epi32(chunk, step);
                pi += 16;
                c += 32;
            }
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
        unsafe fn normal_sad_pair_512<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 7],
            w: &Weights,
            simple: bool,
        ) {
            let offsets1 = offsets512(p.pel);
            let offsets2 = offsets512(2 * p.pel);
            let step = _mm512_set1_epi32(32 * p.pel);
            let pitch = pitch512(p.pitch_i32);
            let (t, rt, tb, rtb, full, wrap) = (
                _mm512_set1_epi32(w.t),
                _mm512_set1_epi32(w.rt),
                _mm512_set1_epi32(w.tb),
                _mm512_set1_epi32(w.rtb),
                _mm512_set1_epi32(255),
                _mm512_set1_epi32(w.wrap),
            );
            let odd_w = _mm512_set1_epi32(265 - w.tb);
            let mut chunk = _mm512_add_epi32(_mm512_set1_epi32(base), offsets2);
            let mut start = base;
            let (mut pi, mut c) = (0, 0);
            while c < n {
                let (pb_e, pb_o) =
                    fetch_pair512(&p.b, at512(chunk, pitch, l[0], l[1], pi), offsets2, offsets1);
                let (pa_e, pa_o) =
                    fetch_pair512(&p.a, at512(chunk, pitch, l[2], l[3], pi), offsets2, offsets1);
                let m9 = lane512(l[4], pi);
                let m8 = lane512(l[5], pi);
                let s = lane512(l[6], pi);
                let inv9 = _mm512_sub_epi32(full, m9);
                let inv8 = _mm512_sub_epi32(full, m8);
                let inv_s = _mm512_sub_epi32(full, s);
                let (b0_e, b0_o) = still_pair512(&p.b, start, chunk, offsets2, offsets1);
                let (a0_e, a0_o) = still_pair512(&p.a, start, chunk, offsets2, offsets1);
                let (ab0_e, ab0_o, aa0_e, aa0_o) = if simple {
                    (b0_e, b0_o, a0_e, a0_o)
                } else {
                    let (be, bo) = still_pair512(&p.aux_b, start, chunk, offsets2, offsets1);
                    let (ae, ao) = still_pair512(&p.aux_a, start, chunk, offsets2, offsets1);
                    (be, bo, ae, ao)
                };
                let mix = |pb, pa, b0, a0, ab0, aa0| {
                    let first = _mm512_srai_epi32::<8>(_mm512_add_epi32(
                        _mm512_add_epi32(_mm512_mullo_epi32(inv9, pb), _mm512_mullo_epi32(m9, pa)),
                        full,
                    ));
                    let second = _mm512_srai_epi32::<8>(_mm512_add_epi32(
                        _mm512_add_epi32(_mm512_mullo_epi32(inv8, pa), _mm512_mullo_epi32(m8, pb)),
                        full,
                    ));
                    let (value, mid) = if simple {
                        let value = _mm512_srai_epi32::<8>(_mm512_add_epi32(
                            _mm512_mullo_epi32(rt, first),
                            _mm512_mullo_epi32(t, second),
                        ));
                        let weight = if p.index == 1 { odd_w } else { rtb };
                        let mixed = _mm512_and_si512(
                            _mm512_add_epi32(_mm512_mullo_epi32(weight, b0), _mm512_mullo_epi32(tb, a0)),
                            wrap,
                        );
                        (value, _mm512_srai_epi32::<8>(mixed))
                    } else {
                        let time = _mm512_srai_epi32::<8>(_mm512_add_epi32(
                            _mm512_mullo_epi32(rt, ab0),
                            _mm512_mullo_epi32(t, aa0),
                        ));
                        let value = _mm512_max_epi32(
                            _mm512_min_epi32(time, _mm512_max_epi32(first, second)),
                            _mm512_min_epi32(first, second),
                        );
                        let mid = _mm512_srai_epi32::<8>(_mm512_add_epi32(
                            _mm512_mullo_epi32(rtb, b0),
                            _mm512_mullo_epi32(tb, a0),
                        ));
                        (value, mid)
                    };
                    _mm512_srai_epi32::<8>(_mm512_add_epi32(
                        _mm512_add_epi32(_mm512_mullo_epi32(inv_s, value), _mm512_mullo_epi32(s, mid)),
                        full,
                    ))
                };
                store_pair512(
                    out,
                    c,
                    n,
                    mix(pb_e, pa_e, b0_e, a0_e, ab0_e, aa0_e),
                    mix(pb_o, pa_o, b0_o, a0_o, ab0_o, aa0_o),
                );
                start += 32 * p.pel;
                chunk = _mm512_add_epi32(chunk, step);
                pi += 16;
                c += 32;
            }
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
        unsafe fn extended_sad_pair_512<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 11],
            w: &Weights,
        ) {
            let offsets1 = offsets512(p.pel);
            let offsets2 = offsets512(2 * p.pel);
            let step = _mm512_set1_epi32(32 * p.pel);
            let pitch = pitch512(p.pitch_i32);
            let (t, tb, rtb, full) = (
                _mm512_set1_epi32(w.t),
                _mm512_set1_epi32(w.tb),
                _mm512_set1_epi32(w.rtb),
                _mm512_set1_epi32(255),
            );
            let mut chunk = _mm512_add_epi32(_mm512_set1_epi32(base), offsets2);
            let mut start = base;
            let (mut pi, mut c) = (0, 0);
            while c < n {
                let (c0_e, c0_o) =
                    fetch_pair512(&p.b, at512(chunk, pitch, l[0], l[1], pi), offsets2, offsets1);
                let (n0_e, n0_o) =
                    fetch_pair512(&p.a, at512(chunk, pitch, l[2], l[3], pi), offsets2, offsets1);
                let m9 = lane512(l[8], pi);
                let m8 = lane512(l[9], pi);
                let s = lane512(l[10], pi);
                let inv_s = _mm512_sub_epi32(full, s);
                let has_m9 = _mm512_test_epi32_mask(m9, m9) != 0;
                let has_m8 = _mm512_test_epi32_mask(m8, m8) != 0;
                let (c1_e, c1_o) = if has_m8 {
                    let pair = fetch_pair512(&p.b, at512(chunk, pitch, l[4], l[5], pi), offsets2, offsets1);
                    (Some(pair.0), Some(pair.1))
                } else {
                    (None, None)
                };
                let (n1_e, n1_o) = if has_m9 {
                    let pair = fetch_pair512(&p.a, at512(chunk, pitch, l[6], l[7], pi), offsets2, offsets1);
                    (Some(pair.0), Some(pair.1))
                } else {
                    (None, None)
                };
                let ve = blend_ext_512::<T>(c0_e, n0_e, c1_e, n1_e, m9, m8, t, full);
                let vo = blend_ext_512::<T>(c0_o, n0_o, c1_o, n1_o, m9, m8, t, full);
                let (b0_e, b0_o) = still_pair512(&p.b, start, chunk, offsets2, offsets1);
                let (a0_e, a0_o) = still_pair512(&p.a, start, chunk, offsets2, offsets1);
                let mix = |value, b0, a0| {
                    let mid = _mm512_srai_epi32::<8>(_mm512_add_epi32(
                        _mm512_mullo_epi32(rtb, b0),
                        _mm512_mullo_epi32(tb, a0),
                    ));
                    _mm512_srai_epi32::<8>(_mm512_add_epi32(
                        _mm512_add_epi32(_mm512_mullo_epi32(inv_s, value), _mm512_mullo_epi32(s, mid)),
                        full,
                    ))
                };
                store_pair512(out, c, n, mix(ve, b0_e, a0_e), mix(vo, b0_o, a0_o));
                start += 32 * p.pel;
                chunk = _mm512_add_epi32(chunk, step);
                pi += 16;
                c += 32;
            }
        }

        pub(in super::super) fn fast_pair<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            next: bool,
            base: i32,
            n: usize,
            l: &[&[i32]; 2],
        ) {
            if p.avx512 {
                unsafe { fast_pair_512(out, p, next, base, n, l) };
                return;
            }
            let src = if next { p.a } else { p.b };
            p.run_pair(out, base, n, |pi, chunk, _, off2| {
                src.fetch_pair(p.at(chunk, l[0], l[1], pi), off2)
            });
        }

        pub(in super::super) fn no_mask_pair<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 4],
            w: &Weights,
            median: bool,
        ) {
            if p.avx512 {
                unsafe { no_mask_pair_512(out, p, base, n, l, w, median) };
                return;
            }
            let (t, rt, tb, rtb) = (splat(w.t), splat(w.rt), splat(w.tb), splat(w.rtb));
            p.run_pair(out, base, n, |pi, chunk, start, off2| {
                let (pb_e, pb_o) = p.b.fetch_pair(p.at(chunk, l[0], l[1], pi), off2);
                let (pa_e, pa_o) = p.a.fetch_pair(p.at(chunk, l[2], l[3], pi), off2);
                if median {
                    let (b0_e, b0_o) = p.b.still_pair(start, chunk, off2);
                    let (a0_e, a0_o) = p.a.still_pair(start, chunk, off2);
                    let med = |pb, pa, b0, a0| {
                        let mid = shr8(add(mul(rtb, b0), mul(tb, a0)));
                        max(min(mid, max(pb, pa)), min(pb, pa))
                    };
                    (med(pb_e, pa_e, b0_e, a0_e), med(pb_o, pa_o, b0_o, a0_o))
                } else {
                    (
                        shr8(add(mul(rt, pb_e), mul(t, pa_e))),
                        shr8(add(mul(rt, pb_o), mul(t, pa_o))),
                    )
                }
            });
        }

        pub(in super::super) fn normal_pair<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 6],
            w: &Weights,
            simple: bool,
        ) {
            if p.avx512 {
                unsafe { normal_pair_512(out, p, base, n, l, w, simple) };
                return;
            }
            let (t, rt, full) = (splat(w.t), splat(w.rt), splat(255));
            p.run_pair(out, base, n, |pi, chunk, start, off2| {
                let (pb_e, pb_o) = p.b.fetch_pair(p.at(chunk, l[0], l[1], pi), off2);
                let (pa_e, pa_o) = p.a.fetch_pair(p.at(chunk, l[2], l[3], pi), off2);
                let m9 = p.lane(l[4], pi);
                let m8 = p.lane(l[5], pi);
                let inv9 = sub(full, m9);
                let inv8 = sub(full, m8);
                if simple {
                    let blend = |pb, pa| {
                        let f = shr8(add(add(mul(inv9, pb), mul(m9, pa)), full));
                        let s = shr8(add(add(mul(inv8, pa), mul(m8, pb)), full));
                        shr8(add(mul(rt, f), mul(t, s)))
                    };
                    (blend(pb_e, pa_e), blend(pb_o, pa_o))
                } else {
                    let (b0_e, b0_o) = p.b.still_pair(start, chunk, off2);
                    let (a0_e, a0_o) = p.a.still_pair(start, chunk, off2);
                    let blend = |pb, pa, b0, a0| {
                        let bb = mul(inv9, pb);
                        let aa = mul(inv8, pa);
                        let i1 = shr8u(add(mul(m9, add(aa, mul(m8, b0))), full));
                        let f = shr8(add(add(bb, i1), full));
                        let i2 = shr8u(add(mul(add(bb, mul(a0, m9)), m8), full));
                        let s = shr8(add(add(aa, i2), full));
                        shr8(add(mul(rt, f), mul(t, s)))
                    };
                    (blend(pb_e, pa_e, b0_e, a0_e), blend(pb_o, pa_o, b0_o, a0_o))
                }
            });
        }

        pub(in super::super) fn extended_pair<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 10],
            w: &Weights,
        ) {
            if p.avx512 {
                unsafe { extended_pair_512(out, p, base, n, l, w) };
                return;
            }
            let (t, full) = (splat(w.t), splat(255));
            p.run_pair(out, base, n, |pi, chunk, _, off2| {
                let (c0_e, c0_o) = p.b.fetch_pair(p.at(chunk, l[0], l[1], pi), off2);
                let (n0_e, n0_o) = p.a.fetch_pair(p.at(chunk, l[2], l[3], pi), off2);
                let m9 = p.lane(l[8], pi);
                let m8 = p.lane(l[9], pi);
                let has_m9 = unsafe { _mm256_testz_si256(m9, m9) } == 0;
                let has_m8 = unsafe { _mm256_testz_si256(m8, m8) } == 0;
                let (c1_e, c1_o) = if has_m8 {
                    let pair = p.b.fetch_pair(p.at(chunk, l[4], l[5], pi), off2);
                    (Some(pair.0), Some(pair.1))
                } else {
                    (None, None)
                };
                let (n1_e, n1_o) = if has_m9 {
                    let pair = p.a.fetch_pair(p.at(chunk, l[6], l[7], pi), off2);
                    (Some(pair.0), Some(pair.1))
                } else {
                    (None, None)
                };
                (
                    blend_ext_256::<T>(c0_e, n0_e, c1_e, n1_e, m9, m8, t, full),
                    blend_ext_256::<T>(c0_o, n0_o, c1_o, n1_o, m9, m8, t, full),
                )
            });
        }

        pub(in super::super) fn fast_sad_pair<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            next: bool,
            base: i32,
            n: usize,
            l: &[&[i32]; 3],
        ) {
            if p.avx512 {
                unsafe { fast_sad_pair_512(out, p, next, base, n, l) };
                return;
            }
            let src = if next { p.a } else { p.b };
            let full = splat(255);
            p.run_pair(out, base, n, |pi, chunk, start, off2| {
                let (me, mo) = src.fetch_pair(p.at(chunk, l[0], l[1], pi), off2);
                let (se, so) = src.still_pair(start, chunk, off2);
                let s = p.lane(l[2], pi);
                let inv = sub(full, s);
                (
                    shr8(add(add(mul(inv, me), mul(s, se)), full)),
                    shr8(add(add(mul(inv, mo), mul(s, so)), full)),
                )
            });
        }

        pub(in super::super) fn no_mask_sad_pair<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 5],
            w: &Weights,
            median: bool,
        ) {
            if p.avx512 {
                unsafe { no_mask_sad_pair_512(out, p, base, n, l, w, median) };
                return;
            }
            let (t, rt, tb, rtb, full) = (
                splat(w.t),
                splat(w.rt),
                splat(w.tb),
                splat(w.rtb),
                splat(255),
            );
            p.run_pair(out, base, n, |pi, chunk, start, off2| {
                let (pb_e, pb_o) = p.b.fetch_pair(p.at(chunk, l[0], l[1], pi), off2);
                let (pa_e, pa_o) = p.a.fetch_pair(p.at(chunk, l[2], l[3], pi), off2);
                let s = p.lane(l[4], pi);
                let inv = sub(full, s);
                let (b0_e, b0_o) = p.b.still_pair(start, chunk, off2);
                let (a0_e, a0_o) = p.a.still_pair(start, chunk, off2);
                let mix = |pb, pa, b0, a0| {
                    let mid = shr8(add(mul(rtb, b0), mul(tb, a0)));
                    let value = if median {
                        max(min(mid, max(pb, pa)), min(pb, pa))
                    } else {
                        shr8(add(mul(rt, pb), mul(t, pa)))
                    };
                    shr8(add(add(mul(inv, value), mul(s, mid)), full))
                };
                (mix(pb_e, pa_e, b0_e, a0_e), mix(pb_o, pa_o, b0_o, a0_o))
            });
        }

        pub(in super::super) fn normal_sad_pair<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 7],
            w: &Weights,
            simple: bool,
        ) {
            if p.avx512 {
                unsafe { normal_sad_pair_512(out, p, base, n, l, w, simple) };
                return;
            }
            let (t, rt, tb, rtb, full) = (
                splat(w.t),
                splat(w.rt),
                splat(w.tb),
                splat(w.rtb),
                splat(255),
            );
            let odd_w = splat(265 - w.tb);
            p.run_pair(out, base, n, |pi, chunk, start, off2| {
                let (pb_e, pb_o) = p.b.fetch_pair(p.at(chunk, l[0], l[1], pi), off2);
                let (pa_e, pa_o) = p.a.fetch_pair(p.at(chunk, l[2], l[3], pi), off2);
                let m9 = p.lane(l[4], pi);
                let m8 = p.lane(l[5], pi);
                let s = p.lane(l[6], pi);
                let inv9 = sub(full, m9);
                let inv8 = sub(full, m8);
                let inv_s = sub(full, s);
                let (b0_e, b0_o) = p.b.still_pair(start, chunk, off2);
                let (a0_e, a0_o) = p.a.still_pair(start, chunk, off2);
                let (ab0_e, ab0_o, aa0_e, aa0_o) = if simple {
                    (b0_e, b0_o, a0_e, a0_o)
                } else {
                    let (be, bo) = p.aux_b.still_pair(start, chunk, off2);
                    let (ae, ao) = p.aux_a.still_pair(start, chunk, off2);
                    (be, bo, ae, ao)
                };
                let mix = |pb, pa, b0, a0, ab0, aa0| {
                    let first = shr8(add(add(mul(inv9, pb), mul(m9, pa)), full));
                    let second = shr8(add(add(mul(inv8, pa), mul(m8, pb)), full));
                    let (value, mid) = if simple {
                        let value = shr8(add(mul(rt, first), mul(t, second)));
                        let weight = if p.index == 1 { odd_w } else { rtb };
                        let mixed = unsafe {
                            _mm256_and_si256(add(mul(weight, b0), mul(tb, a0)), splat(w.wrap))
                        };
                        (value, shr8(mixed))
                    } else {
                        let time = shr8(add(mul(rt, ab0), mul(t, aa0)));
                        let value = max(min(time, max(first, second)), min(first, second));
                        (value, shr8(add(mul(rtb, b0), mul(tb, a0))))
                    };
                    shr8(add(add(mul(inv_s, value), mul(s, mid)), full))
                };
                (
                    mix(pb_e, pa_e, b0_e, a0_e, ab0_e, aa0_e),
                    mix(pb_o, pa_o, b0_o, a0_o, ab0_o, aa0_o),
                )
            });
        }

        pub(in super::super) fn extended_sad_pair<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 11],
            w: &Weights,
        ) {
            if p.avx512 {
                unsafe { extended_sad_pair_512(out, p, base, n, l, w) };
                return;
            }
            let (t, tb, rtb, full) = (splat(w.t), splat(w.tb), splat(w.rtb), splat(255));
            p.run_pair(out, base, n, |pi, chunk, start, off2| {
                let (c0_e, c0_o) = p.b.fetch_pair(p.at(chunk, l[0], l[1], pi), off2);
                let (n0_e, n0_o) = p.a.fetch_pair(p.at(chunk, l[2], l[3], pi), off2);
                let m9 = p.lane(l[8], pi);
                let m8 = p.lane(l[9], pi);
                let s = p.lane(l[10], pi);
                let inv_s = sub(full, s);
                let has_m9 = unsafe { _mm256_testz_si256(m9, m9) } == 0;
                let has_m8 = unsafe { _mm256_testz_si256(m8, m8) } == 0;
                let (c1_e, c1_o) = if has_m8 {
                    let pair = p.b.fetch_pair(p.at(chunk, l[4], l[5], pi), off2);
                    (Some(pair.0), Some(pair.1))
                } else {
                    (None, None)
                };
                let (n1_e, n1_o) = if has_m9 {
                    let pair = p.a.fetch_pair(p.at(chunk, l[6], l[7], pi), off2);
                    (Some(pair.0), Some(pair.1))
                } else {
                    (None, None)
                };
                let ve = blend_ext_256::<T>(c0_e, n0_e, c1_e, n1_e, m9, m8, t, full);
                let vo = blend_ext_256::<T>(c0_o, n0_o, c1_o, n1_o, m9, m8, t, full);
                let (b0_e, b0_o) = p.b.still_pair(start, chunk, off2);
                let (a0_e, a0_o) = p.a.still_pair(start, chunk, off2);
                let mix = |value, b0, a0| {
                    let mid = shr8(add(mul(rtb, b0), mul(tb, a0)));
                    shr8(add(add(mul(inv_s, value), mul(s, mid)), full))
                };
                (mix(ve, b0_e, a0_e), mix(vo, b0_o, a0_o))
            });
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[allow(
        unsafe_code,
        unused_unsafe,
        clippy::cast_ptr_alignment,
        clippy::inline_always,
        clippy::many_single_char_names,
        clippy::trivially_copy_pass_by_ref
    )]
    mod imp {
        use std::arch::aarch64::{
            int32x4_t, vaddq_s32, vandq_s32, vceqq_s32, vcombine_s16, vdupq_n_s32, vget_low_u16,
            vgetq_lane_s32, vld1_u8, vld1_u16, vld1q_s32, vld1q_u8_x4, vmaxq_s32, vmaxvq_u32,
            vminq_s32, vminvq_u32, vmovl_u8, vmovl_u16, vmulq_s32, vqmovn_s32, vqmovun_s16,
            vqmovun_s32, vqtbl4q_u8, vreinterpretq_s32_u8, vreinterpretq_s32_u32,
            vreinterpretq_u8_s32, vreinterpretq_u32_s32, vshlq_n_s32, vshrq_n_s32, vshrq_n_u32,
            vst1_u8, vst1_u16, vsubq_s32, vzip1_u8, vzip1_u16, vzip2_u16,
        };
        use std::marker::PhantomData;

        use super::super::{Plane, Sample};
        use super::{Setup, Weights};

        #[derive(Clone, Copy)]
        struct Source<T> {
            data: *const T,
            last: int32x4_t,
            offsets: int32x4_t,
            len: usize,
            readable: usize,
            pel: i32,
        }

        impl<T: Sample> Source<T> {
            const SIZE: usize = size_of::<T>();

            #[inline(always)]
            fn new(plane: &Plane<'_, T>, pel: i32) -> Self {
                unsafe {
                    let len = plane.data.len();
                    let last = i32::try_from(len.saturating_sub(1)).unwrap_or(i32::MAX);
                    let offs = [0, pel, 2 * pel, 3 * pel];
                    Self {
                        data: plane.data.as_ptr(),
                        last: vdupq_n_s32(last),
                        offsets: vld1q_s32(offs.as_ptr()),
                        len,
                        readable: len + plane.slack,
                        pel,
                    }
                }
            }

            #[inline(always)]
            fn load(&self, start: i32) -> Option<int32x4_t> {
                unsafe {
                    let start = start as usize;
                    let span = 4 * self.pel as usize;
                    if start + span.max(8 / Self::SIZE) > self.readable {
                        return None;
                    }
                    let at = self.data.add(start);
                    if Self::SIZE == 2 {
                        return match self.pel {
                            1 => {
                                let w = vld1_u16(at.cast::<u16>());
                                Some(vreinterpretq_s32_u32(vmovl_u16(w)))
                            }
                            _ => None,
                        };
                    }
                    match self.pel {
                        1 => {
                            let bytes = vld1_u8(at.cast::<u8>());
                            let words = vget_low_u16(vmovl_u8(bytes));
                            Some(vreinterpretq_s32_u32(vmovl_u16(words)))
                        }
                        2 => {
                            let words = vld1_u16(at.cast::<u16>());
                            Some(vandq_s32(
                                vreinterpretq_s32_u32(vmovl_u16(words)),
                                vdupq_n_s32(0xFF),
                            ))
                        }
                        _ => None,
                    }
                }
            }

            #[inline(always)]
            fn fetch(&self, index: int32x4_t) -> int32x4_t {
                unsafe {
                    let zero = vdupq_n_s32(0);
                    if self.len == 0 {
                        return zero;
                    }
                    let first = vgetq_lane_s32::<0>(index);
                    let run = vaddq_s32(vdupq_n_s32(first), self.offsets);
                    if first >= 0
                        && vminvq_u32(vceqq_s32(index, run)) == u32::MAX
                        && let Some(value) = self.load(first)
                    {
                        return value;
                    }
                    let index = vminq_s32(vmaxq_s32(index, zero), self.last);
                    if Self::SIZE == 1 {
                        let cfirst = vgetq_lane_s32::<0>(index);
                        let win_base = (cfirst - 4).max(0);
                        let rel = vsubq_s32(index, vdupq_n_s32(win_base));
                        if vmaxvq_u32(vreinterpretq_u32_s32(rel)) < 64
                            && (win_base as usize) + 64 <= self.readable
                        {
                            let tbl = vld1q_u8_x4(self.data.add(win_base as usize).cast::<u8>());
                            let picked = vqtbl4q_u8(tbl, vreinterpretq_u8_s32(rel));
                            return vandq_s32(vreinterpretq_s32_u8(picked), vdupq_n_s32(0xFF));
                        }
                    }
                    let i0 = vgetq_lane_s32::<0>(index) as usize;
                    let i1 = vgetq_lane_s32::<1>(index) as usize;
                    let i2 = vgetq_lane_s32::<2>(index) as usize;
                    let i3 = vgetq_lane_s32::<3>(index) as usize;
                    let vals = [
                        (*self.data.add(i0)).into(),
                        (*self.data.add(i1)).into(),
                        (*self.data.add(i2)).into(),
                        (*self.data.add(i3)).into(),
                    ];
                    vld1q_s32(vals.as_ptr())
                }
            }

            #[inline(always)]
            fn still(&self, start: i32, chunk: int32x4_t) -> int32x4_t {
                self.load(start).unwrap_or_else(|| self.fetch(chunk))
            }

            #[inline(always)]
            fn fetch_pair(&self, index: int32x4_t) -> (int32x4_t, int32x4_t) {
                unsafe {
                    let zero = vdupq_n_s32(0);
                    if self.len == 0 {
                        return (zero, zero);
                    }
                    let clamped = vminq_s32(vmaxq_s32(index, zero), self.last);
                    let pel_v = vdupq_n_s32(self.pel);
                    if Self::SIZE == 1 && self.pel <= 2 {
                        let cfirst = vgetq_lane_s32::<0>(clamped);
                        let win_base = (cfirst - 4).max(0);
                        let rel = vsubq_s32(clamped, vdupq_n_s32(win_base));
                        let rel_odd = vaddq_s32(rel, pel_v);
                        let max_clamped = vmaxvq_u32(vreinterpretq_u32_s32(clamped)) as usize;
                        if max_clamped + (self.pel as usize) < self.len
                            && vmaxvq_u32(vreinterpretq_u32_s32(rel)) < 64
                            && vmaxvq_u32(vreinterpretq_u32_s32(rel_odd)) < 64
                            && (win_base as usize) + 64 <= self.readable
                        {
                            let tbl = vld1q_u8_x4(self.data.add(win_base as usize).cast::<u8>());
                            let mask = vdupq_n_s32(0xFF);
                            let e = vandq_s32(
                                vreinterpretq_s32_u8(vqtbl4q_u8(tbl, vreinterpretq_u8_s32(rel))),
                                mask,
                            );
                            let o = vandq_s32(
                                vreinterpretq_s32_u8(vqtbl4q_u8(tbl, vreinterpretq_u8_s32(rel_odd))),
                                mask,
                            );
                            return (e, o);
                        }
                    }
                    let last = self.len - 1;
                    let pel = self.pel as usize;
                    let i0 = vgetq_lane_s32::<0>(clamped) as usize;
                    let i1 = vgetq_lane_s32::<1>(clamped) as usize;
                    let i2 = vgetq_lane_s32::<2>(clamped) as usize;
                    let i3 = vgetq_lane_s32::<3>(clamped) as usize;
                    let ev = [
                        (*self.data.add(i0)).into(),
                        (*self.data.add(i1)).into(),
                        (*self.data.add(i2)).into(),
                        (*self.data.add(i3)).into(),
                    ];
                    let od = [
                        (*self.data.add((i0 + pel).min(last))).into(),
                        (*self.data.add((i1 + pel).min(last))).into(),
                        (*self.data.add((i2 + pel).min(last))).into(),
                        (*self.data.add((i3 + pel).min(last))).into(),
                    ];
                    (vld1q_s32(ev.as_ptr()), vld1q_s32(od.as_ptr()))
                }
            }
        }

        pub(in super::super) struct Prepared<'a, T> {
            a: Source<T>,
            b: Source<T>,
            offsets: int32x4_t,
            step: int32x4_t,
            pel: i32,
            pitch: int32x4_t,
            aux_a: Source<T>,
            aux_b: Source<T>,
            pub(in super::super) index: usize,
            _life: PhantomData<&'a [T]>,
        }

        impl<'a, T: Sample> Prepared<'a, T> {
            pub(in super::super) fn new(
                setup: &Setup<'a>,
                a: &Plane<'a, T>,
                b: &Plane<'a, T>,
                aux: Option<(Plane<'a, T>, Plane<'a, T>)>,
                index: usize,
            ) -> Self {
                unsafe {
                    let pel = setup.pel;
                    let (aux_a, aux_b) = aux.unwrap_or((*a, *b));
                    let offs = [0, pel, 2 * pel, 3 * pel];
                    Self {
                        a: Source::new(a, pel),
                        b: Source::new(b, pel),
                        aux_a: Source::new(&aux_a, pel),
                        aux_b: Source::new(&aux_b, pel),
                        index,
                        offsets: vld1q_s32(offs.as_ptr()),
                        step: vdupq_n_s32(4 * pel),
                        pel,
                        pitch: vdupq_n_s32(setup.pitch),
                        _life: PhantomData,
                    }
                }
            }

            #[allow(clippy::unused_self)]
            #[inline(always)]
            fn lane(&self, values: &[i32], c: usize) -> int32x4_t {
                assert!(c + 4 <= values.len());
                unsafe { vld1q_s32(values.as_ptr().add(c)) }
            }

            #[inline(always)]
            fn at(&self, chunk: int32x4_t, x: &[i32], y: &[i32], c: usize) -> int32x4_t {
                unsafe {
                    let xv = self.lane(x, c);
                    let yv = self.lane(y, c);
                    vaddq_s32(vaddq_s32(chunk, xv), vmulq_s32(yv, self.pitch))
                }
            }

            #[inline(always)]
            fn run(
                &self,
                out: &mut [T],
                base: i32,
                n: usize,
                mut body: impl FnMut(usize, int32x4_t, i32) -> int32x4_t,
            ) {
                unsafe {
                    let mut chunk = vaddq_s32(vdupq_n_s32(base), self.offsets);
                    let mut start = base;
                    let mut c = 0;
                    while c < n {
                        let value = body(c, chunk, start);
                        start += 4 * self.pel;
                        if c + 4 <= out.len() {
                            let ptr = out.as_mut_ptr().add(c);
                            if size_of::<T>() == 1 {
                                let s16 = vqmovn_s32(value);
                                let bytes = vqmovun_s16(vcombine_s16(s16, s16));
                                let packed = std::mem::transmute::<std::arch::aarch64::uint8x8_t, u64>(bytes) as u32;
                                std::ptr::write_unaligned(ptr.cast::<u32>(), packed);
                            } else {
                                vst1_u16(ptr.cast::<u16>(), vqmovun_s32(value));
                            }
                        } else {
                            let mut tmp = [T::default(); 4];
                            if size_of::<T>() == 1 {
                                let s16 = vqmovn_s32(value);
                                let bytes = vqmovun_s16(vcombine_s16(s16, s16));
                                let packed = std::mem::transmute::<std::arch::aarch64::uint8x8_t, u64>(bytes) as u32;
                                std::ptr::write_unaligned(tmp.as_mut_ptr().cast::<u32>(), packed);
                            } else {
                                vst1_u16(tmp.as_mut_ptr().cast::<u16>(), vqmovun_s32(value));
                            }
                            let take = (n - c).min(out.len().saturating_sub(c)).min(4);
                            out[c..c + take].copy_from_slice(&tmp[..take]);
                        }
                        chunk = vaddq_s32(chunk, self.step);
                        c += 4;
                    }
                }
            }

            #[inline(always)]
            fn run_pair(
                &self,
                out: &mut [T],
                base: i32,
                n: usize,
                mut body: impl FnMut(usize, int32x4_t) -> (int32x4_t, int32x4_t),
            ) {
                unsafe {
                    let offsets2 = vaddq_s32(self.offsets, self.offsets);
                    let step2 = vaddq_s32(self.step, self.step);
                    let mut chunk = vaddq_s32(vdupq_n_s32(base), offsets2);
                    let (mut pi, mut c) = (0, 0);
                    while c < n {
                        let (even, odd) = body(pi, chunk);
                        let write = |ptr: *mut T| {
                            if size_of::<T>() == 1 {
                                let es = vqmovn_s32(even);
                                let os = vqmovn_s32(odd);
                                let eb = vqmovun_s16(vcombine_s16(es, es));
                                let ob = vqmovun_s16(vcombine_s16(os, os));
                                vst1_u8(ptr.cast::<u8>(), vzip1_u8(eb, ob));
                            } else {
                                let ew = vqmovun_s32(even);
                                let ow = vqmovun_s32(odd);
                                vst1_u16(ptr.cast::<u16>(), vzip1_u16(ew, ow));
                                vst1_u16(ptr.add(4).cast::<u16>(), vzip2_u16(ew, ow));
                            }
                        };
                        if c + 8 <= out.len() {
                            write(out.as_mut_ptr().add(c));
                        } else {
                            let mut tmp = [T::default(); 8];
                            write(tmp.as_mut_ptr());
                            let take = (n - c).min(out.len().saturating_sub(c)).min(8);
                            out[c..c + take].copy_from_slice(&tmp[..take]);
                        }
                        chunk = vaddq_s32(chunk, step2);
                        pi += 4;
                        c += 8;
                    }
                }
            }
        }

        #[inline(always)]
        fn splat(v: i32) -> int32x4_t {
            unsafe { vdupq_n_s32(v) }
        }

        #[inline(always)]
        fn mul(a: int32x4_t, b: int32x4_t) -> int32x4_t {
            unsafe { vmulq_s32(a, b) }
        }

        #[inline(always)]
        fn add(a: int32x4_t, b: int32x4_t) -> int32x4_t {
            unsafe { vaddq_s32(a, b) }
        }

        #[inline(always)]
        fn sub(a: int32x4_t, b: int32x4_t) -> int32x4_t {
            unsafe { vsubq_s32(a, b) }
        }

        #[inline(always)]
        fn min(a: int32x4_t, b: int32x4_t) -> int32x4_t {
            unsafe { vminq_s32(a, b) }
        }

        #[inline(always)]
        fn max(a: int32x4_t, b: int32x4_t) -> int32x4_t {
            unsafe { vmaxq_s32(a, b) }
        }

        #[inline(always)]
        fn shr8(a: int32x4_t) -> int32x4_t {
            unsafe { vshrq_n_s32::<8>(a) }
        }

        #[inline(always)]
        fn shr8u(a: int32x4_t) -> int32x4_t {
            unsafe { vreinterpretq_s32_u32(vshrq_n_u32::<8>(vreinterpretq_u32_s32(a))) }
        }

        pub(in super::super) fn fill(
            setup: &Setup<'_>,
            out: &mut [i32],
            pair: i32,
            cols: usize,
            vector: bool,
        ) {
            let (weights, shift) = if vector {
                (setup.vector_weights, setup.vector_shift)
            } else {
                (setup.weights, setup.shift)
            };
            let count = cols.next_multiple_of(8);
            let (lo, hi, shift) = (i32::from(pair as i16), pair >> 16, shift.min(31));
            for (value, &w) in out[..count].iter_mut().zip(&weights[..count]) {
                let sum = lo * i32::from(w as i16) + hi * (w >> 16);
                *value = i32::from((sum >> shift) as i16);
            }
        }

        pub(in super::super) fn fill_block<const N: usize>(
            setup: &Setup<'_>,
            lanes: &mut [i32],
            (span, at, cols): (usize, usize, usize),
            (top, bottom): (&[i16; 32], &[i16; 32]),
            (wt, wb, shift_y): (i16, i16, u32),
        ) {
            let mix = |i: usize| {
                wt.wrapping_mul(top[i])
                    .wrapping_add(wb.wrapping_mul(bottom[i]))
                    >> shift_y
            };
            for (k, lane) in lanes.chunks_exact_mut(span).take(N).enumerate() {
                let pair = i32::from(mix(2 * k) as u16) | (i32::from(mix(2 * k + 1) as u16) << 16);
                fill(setup, &mut lane[at..], pair, cols, false);
            }
        }

        pub(in super::super) fn fast<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            next: bool,
            base: i32,
            n: usize,
            l: &[&[i32]; 2],
        ) {
            let src = if next { p.a } else { p.b };
            p.run(out, base, n, |c, chunk, _| {
                src.fetch(p.at(chunk, l[0], l[1], c))
            });
        }

        pub(in super::super) fn no_mask<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 4],
            w: &Weights,
            median: bool,
        ) {
            let (t, rt, tb, rtb) = (splat(w.t), splat(w.rt), splat(w.tb), splat(w.rtb));
            p.run(out, base, n, |c, chunk, start| {
                let pb = p.b.fetch(p.at(chunk, l[0], l[1], c));
                let pa = p.a.fetch(p.at(chunk, l[2], l[3], c));
                if median {
                    let b0 = p.b.still(start, chunk);
                    let a0 = p.a.still(start, chunk);
                    let mid = shr8(add(mul(rtb, b0), mul(tb, a0)));
                    max(min(mid, max(pb, pa)), min(pb, pa))
                } else {
                    shr8(add(mul(rt, pb), mul(t, pa)))
                }
            });
        }

        pub(in super::super) fn normal<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 6],
            w: &Weights,
            simple: bool,
        ) {
            let (t, rt, full) = (splat(w.t), splat(w.rt), splat(255));
            p.run(out, base, n, |c, chunk, start| {
                let pb = p.b.fetch(p.at(chunk, l[0], l[1], c));
                let pa = p.a.fetch(p.at(chunk, l[2], l[3], c));
                let m9 = p.lane(l[4], c);
                let m8 = p.lane(l[5], c);
                let (first, second) = if simple {
                    (
                        shr8(add(add(mul(sub(full, m9), pb), mul(m9, pa)), full)),
                        shr8(add(add(mul(sub(full, m8), pa), mul(m8, pb)), full)),
                    )
                } else {
                    let bb = mul(sub(full, m9), pb);
                    let aa = mul(sub(full, m8), pa);
                    let b0 = p.b.still(start, chunk);
                    let a0 = p.a.still(start, chunk);
                    let inner = shr8u(add(mul(m9, add(aa, mul(m8, b0))), full));
                    let first = shr8(add(add(bb, inner), full));
                    let inner = shr8u(add(mul(add(bb, mul(a0, m9)), m8), full));
                    (first, shr8(add(add(aa, inner), full)))
                };
                shr8(add(mul(rt, first), mul(t, second)))
            });
        }

        pub(in super::super) fn extended<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 10],
            w: &Weights,
        ) {
            let (t, full) = (splat(w.t), splat(255));
            p.run(out, base, n, |c, chunk, _| {
                let c0 = p.b.fetch(p.at(chunk, l[0], l[1], c));
                let n0 = p.a.fetch(p.at(chunk, l[2], l[3], c));
                let m9 = p.lane(l[8], c);
                let m8 = p.lane(l[9], c);
                let has_m9 = unsafe { vmaxvq_u32(vreinterpretq_u32_s32(m9)) } != 0;
                let has_m8 = unsafe { vmaxvq_u32(vreinterpretq_u32_s32(m8)) } != 0;
                let first = if has_m9 {
                    let n1 = p.a.fetch(p.at(chunk, l[6], l[7], c));
                    let lo = min(c0, n0);
                    let hi = max(c0, n0);
                    let cn = max(lo, min(n1, hi));
                    let c0_255 = sub(unsafe { vshlq_n_s32::<8>(c0) }, c0);
                    shr8(add(add(c0_255, mul(m9, sub(cn, c0))), full))
                } else if size_of::<T>() == 1 {
                    c0
                } else {
                    shr8(add(sub(unsafe { vshlq_n_s32::<8>(c0) }, c0), full))
                };
                let second = if has_m8 {
                    let c1 = p.b.fetch(p.at(chunk, l[4], l[5], c));
                    let lo = min(c0, n0);
                    let hi = max(c0, n0);
                    let cc = max(lo, min(c1, hi));
                    let n0_255 = sub(unsafe { vshlq_n_s32::<8>(n0) }, n0);
                    shr8(add(add(n0_255, mul(m8, sub(cc, n0))), full))
                } else if size_of::<T>() == 1 {
                    n0
                } else {
                    shr8(add(sub(unsafe { vshlq_n_s32::<8>(n0) }, n0), full))
                };
                shr8(add(
                    unsafe { vshlq_n_s32::<8>(first) },
                    mul(t, sub(second, first)),
                ))
            });
        }

        pub(in super::super) fn fast_sad<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            next: bool,
            base: i32,
            n: usize,
            l: &[&[i32]; 3],
        ) {
            let src = if next { p.a } else { p.b };
            let full = splat(255);
            p.run(out, base, n, |c, chunk, start| {
                let moved = src.fetch(p.at(chunk, l[0], l[1], c));
                let still = src.still(start, chunk);
                let s = p.lane(l[2], c);
                shr8(add(add(mul(sub(full, s), moved), mul(s, still)), full))
            });
        }

        pub(in super::super) fn no_mask_sad<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 5],
            w: &Weights,
            median: bool,
        ) {
            let (t, rt, tb, rtb, full) = (
                splat(w.t),
                splat(w.rt),
                splat(w.tb),
                splat(w.rtb),
                splat(255),
            );
            p.run(out, base, n, |c, chunk, start| {
                let pb = p.b.fetch(p.at(chunk, l[0], l[1], c));
                let pa = p.a.fetch(p.at(chunk, l[2], l[3], c));
                let s = p.lane(l[4], c);
                let b0 = p.b.still(start, chunk);
                let a0 = p.a.still(start, chunk);
                let mid = shr8(add(mul(rtb, b0), mul(tb, a0)));
                let value = if median {
                    max(min(mid, max(pb, pa)), min(pb, pa))
                } else {
                    shr8(add(mul(rt, pb), mul(t, pa)))
                };
                shr8(add(add(mul(sub(full, s), value), mul(s, mid)), full))
            });
        }

        pub(in super::super) fn normal_sad<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 7],
            w: &Weights,
            simple: bool,
        ) {
            let (t, rt, tb, rtb, full) = (
                splat(w.t),
                splat(w.rt),
                splat(w.tb),
                splat(w.rtb),
                splat(255),
            );
            let odd = splat(265 - w.tb);
            p.run(out, base, n, |c, chunk, start| {
                let pb = p.b.fetch(p.at(chunk, l[0], l[1], c));
                let pa = p.a.fetch(p.at(chunk, l[2], l[3], c));
                let m9 = p.lane(l[4], c);
                let m8 = p.lane(l[5], c);
                let s = p.lane(l[6], c);
                let first = shr8(add(add(mul(sub(full, m9), pb), mul(m9, pa)), full));
                let second = shr8(add(add(mul(sub(full, m8), pa), mul(m8, pb)), full));
                let b0 = p.b.still(start, chunk);
                let a0 = p.a.still(start, chunk);
                let (value, mid) = if simple {
                    let value = shr8(add(mul(rt, first), mul(t, second)));
                    let weight = if p.index == 1 { odd } else { rtb };
                    let mixed = unsafe { vandq_s32(add(mul(weight, b0), mul(tb, a0)), splat(w.wrap)) };
                    (value, shr8(mixed))
                } else {
                    let ab0 = p.aux_b.still(start, chunk);
                    let aa0 = p.aux_a.still(start, chunk);
                    let time = shr8(add(mul(rt, ab0), mul(t, aa0)));
                    let value = max(min(time, max(first, second)), min(first, second));
                    (value, shr8(add(mul(rtb, b0), mul(tb, a0))))
                };
                shr8(add(add(mul(sub(full, s), value), mul(s, mid)), full))
            });
        }

        pub(in super::super) fn extended_sad<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 11],
            w: &Weights,
        ) {
            let (t, rt, tb, rtb, full) = (
                splat(w.t),
                splat(w.rt),
                splat(w.tb),
                splat(w.rtb),
                splat(255),
            );
            p.run(out, base, n, |c, chunk, start| {
                let c0 = p.b.fetch(p.at(chunk, l[0], l[1], c));
                let n0 = p.a.fetch(p.at(chunk, l[2], l[3], c));
                let m9 = p.lane(l[8], c);
                let m8 = p.lane(l[9], c);
                let s = p.lane(l[10], c);
                let has_m9 = unsafe { vmaxvq_u32(vreinterpretq_u32_s32(m9)) } != 0;
                let has_m8 = unsafe { vmaxvq_u32(vreinterpretq_u32_s32(m8)) } != 0;
                let first = if has_m9 {
                    let n1 = p.a.fetch(p.at(chunk, l[6], l[7], c));
                    let lo = min(c0, n0);
                    let hi = max(c0, n0);
                    let cn = max(lo, min(n1, hi));
                    shr8(add(add(mul(m9, cn), mul(c0, sub(full, m9))), full))
                } else if size_of::<T>() == 1 {
                    c0
                } else {
                    shr8(add(mul(c0, full), full))
                };
                let second = if has_m8 {
                    let c1 = p.b.fetch(p.at(chunk, l[4], l[5], c));
                    let lo = min(c0, n0);
                    let hi = max(c0, n0);
                    let cc = max(lo, min(c1, hi));
                    shr8(add(add(mul(m8, cc), mul(n0, sub(full, m8))), full))
                } else if size_of::<T>() == 1 {
                    n0
                } else {
                    shr8(add(mul(n0, full), full))
                };
                let value = shr8(add(mul(rt, first), mul(t, second)));
                if unsafe { vmaxvq_u32(vreinterpretq_u32_s32(s)) } == 0 {
                    if size_of::<T>() == 1 {
                        value
                    } else {
                        shr8(add(mul(value, full), full))
                    }
                } else {
                    let b0 = p.b.still(start, chunk);
                    let a0 = p.a.still(start, chunk);
                    let mid = shr8(add(mul(rtb, b0), mul(tb, a0)));
                    shr8(add(add(mul(sub(full, s), value), mul(s, mid)), full))
                }
            });
        }

        #[inline(always)]
        #[allow(clippy::too_many_arguments)]
        fn blend_ext_neon<T: Sample>(
            c0: int32x4_t,
            n0: int32x4_t,
            c1: Option<int32x4_t>,
            n1: Option<int32x4_t>,
            m9: int32x4_t,
            m8: int32x4_t,
            t: int32x4_t,
            full: int32x4_t,
        ) -> int32x4_t {
            let first = match n1 {
                Some(n1) => {
                    let lo = min(c0, n0);
                    let hi = max(c0, n0);
                    let cn = max(lo, min(n1, hi));
                    let c0_255 = sub(unsafe { vshlq_n_s32::<8>(c0) }, c0);
                    shr8(add(add(c0_255, mul(m9, sub(cn, c0))), full))
                }
                None if size_of::<T>() == 1 => c0,
                None => shr8(add(sub(unsafe { vshlq_n_s32::<8>(c0) }, c0), full)),
            };
            let second = match c1 {
                Some(c1) => {
                    let lo = min(c0, n0);
                    let hi = max(c0, n0);
                    let cc = max(lo, min(c1, hi));
                    let n0_255 = sub(unsafe { vshlq_n_s32::<8>(n0) }, n0);
                    shr8(add(add(n0_255, mul(m8, sub(cc, n0))), full))
                }
                None if size_of::<T>() == 1 => n0,
                None => shr8(add(sub(unsafe { vshlq_n_s32::<8>(n0) }, n0), full)),
            };
            shr8(add(
                unsafe { vshlq_n_s32::<8>(first) },
                mul(t, sub(second, first)),
            ))
        }

        pub(in super::super) fn fast_pair<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            next: bool,
            base: i32,
            n: usize,
            l: &[&[i32]; 2],
        ) {
            let src = if next { p.a } else { p.b };
            p.run_pair(out, base, n, |pi, chunk| {
                src.fetch_pair(p.at(chunk, l[0], l[1], pi))
            });
        }

        pub(in super::super) fn no_mask_pair<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 4],
            w: &Weights,
            median: bool,
        ) {
            let (t, rt, tb, rtb) = (splat(w.t), splat(w.rt), splat(w.tb), splat(w.rtb));
            p.run_pair(out, base, n, |pi, chunk| {
                let (pb_e, pb_o) = p.b.fetch_pair(p.at(chunk, l[0], l[1], pi));
                let (pa_e, pa_o) = p.a.fetch_pair(p.at(chunk, l[2], l[3], pi));
                if median {
                    let (b0_e, b0_o) = p.b.fetch_pair(chunk);
                    let (a0_e, a0_o) = p.a.fetch_pair(chunk);
                    let med = |pb, pa, b0, a0| {
                        let mid = shr8(add(mul(rtb, b0), mul(tb, a0)));
                        max(min(mid, max(pb, pa)), min(pb, pa))
                    };
                    (med(pb_e, pa_e, b0_e, a0_e), med(pb_o, pa_o, b0_o, a0_o))
                } else {
                    (
                        shr8(add(mul(rt, pb_e), mul(t, pa_e))),
                        shr8(add(mul(rt, pb_o), mul(t, pa_o))),
                    )
                }
            });
        }

        pub(in super::super) fn normal_pair<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 6],
            w: &Weights,
            simple: bool,
        ) {
            let (t, rt, full) = (splat(w.t), splat(w.rt), splat(255));
            p.run_pair(out, base, n, |pi, chunk| {
                let (pb_e, pb_o) = p.b.fetch_pair(p.at(chunk, l[0], l[1], pi));
                let (pa_e, pa_o) = p.a.fetch_pair(p.at(chunk, l[2], l[3], pi));
                let m9 = p.lane(l[4], pi);
                let m8 = p.lane(l[5], pi);
                let inv9 = sub(full, m9);
                let inv8 = sub(full, m8);
                if simple {
                    let blend = |pb, pa| {
                        let f = shr8(add(add(mul(inv9, pb), mul(m9, pa)), full));
                        let s = shr8(add(add(mul(inv8, pa), mul(m8, pb)), full));
                        shr8(add(mul(rt, f), mul(t, s)))
                    };
                    (blend(pb_e, pa_e), blend(pb_o, pa_o))
                } else {
                    let (b0_e, b0_o) = p.b.fetch_pair(chunk);
                    let (a0_e, a0_o) = p.a.fetch_pair(chunk);
                    let blend = |pb, pa, b0, a0| {
                        let bb = mul(inv9, pb);
                        let aa = mul(inv8, pa);
                        let i1 = shr8u(add(mul(m9, add(aa, mul(m8, b0))), full));
                        let f = shr8(add(add(bb, i1), full));
                        let i2 = shr8u(add(mul(add(bb, mul(a0, m9)), m8), full));
                        let s = shr8(add(add(aa, i2), full));
                        shr8(add(mul(rt, f), mul(t, s)))
                    };
                    (blend(pb_e, pa_e, b0_e, a0_e), blend(pb_o, pa_o, b0_o, a0_o))
                }
            });
        }

        pub(in super::super) fn extended_pair<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 10],
            w: &Weights,
        ) {
            let (t, full) = (splat(w.t), splat(255));
            p.run_pair(out, base, n, |pi, chunk| {
                let (c0_e, c0_o) = p.b.fetch_pair(p.at(chunk, l[0], l[1], pi));
                let (n0_e, n0_o) = p.a.fetch_pair(p.at(chunk, l[2], l[3], pi));
                let m9 = p.lane(l[8], pi);
                let m8 = p.lane(l[9], pi);
                let has_m9 = unsafe { vmaxvq_u32(vreinterpretq_u32_s32(m9)) } != 0;
                let has_m8 = unsafe { vmaxvq_u32(vreinterpretq_u32_s32(m8)) } != 0;
                let (c1_e, c1_o) = if has_m8 {
                    let pair = p.b.fetch_pair(p.at(chunk, l[4], l[5], pi));
                    (Some(pair.0), Some(pair.1))
                } else {
                    (None, None)
                };
                let (n1_e, n1_o) = if has_m9 {
                    let pair = p.a.fetch_pair(p.at(chunk, l[6], l[7], pi));
                    (Some(pair.0), Some(pair.1))
                } else {
                    (None, None)
                };
                (
                    blend_ext_neon::<T>(c0_e, n0_e, c1_e, n1_e, m9, m8, t, full),
                    blend_ext_neon::<T>(c0_o, n0_o, c1_o, n1_o, m9, m8, t, full),
                )
            });
        }

        pub(in super::super) fn fast_sad_pair<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            next: bool,
            base: i32,
            n: usize,
            l: &[&[i32]; 3],
        ) {
            let src = if next { p.a } else { p.b };
            let full = splat(255);
            p.run_pair(out, base, n, |pi, chunk| {
                let (me, mo) = src.fetch_pair(p.at(chunk, l[0], l[1], pi));
                let (se, so) = src.fetch_pair(chunk);
                let s = p.lane(l[2], pi);
                let inv = sub(full, s);
                (
                    shr8(add(add(mul(inv, me), mul(s, se)), full)),
                    shr8(add(add(mul(inv, mo), mul(s, so)), full)),
                )
            });
        }

        pub(in super::super) fn no_mask_sad_pair<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 5],
            w: &Weights,
            median: bool,
        ) {
            let (t, rt, tb, rtb, full) = (
                splat(w.t),
                splat(w.rt),
                splat(w.tb),
                splat(w.rtb),
                splat(255),
            );
            p.run_pair(out, base, n, |pi, chunk| {
                let (pb_e, pb_o) = p.b.fetch_pair(p.at(chunk, l[0], l[1], pi));
                let (pa_e, pa_o) = p.a.fetch_pair(p.at(chunk, l[2], l[3], pi));
                let s = p.lane(l[4], pi);
                let inv = sub(full, s);
                let (b0_e, b0_o) = p.b.fetch_pair(chunk);
                let (a0_e, a0_o) = p.a.fetch_pair(chunk);
                let mix = |pb, pa, b0, a0| {
                    let mid = shr8(add(mul(rtb, b0), mul(tb, a0)));
                    let value = if median {
                        max(min(mid, max(pb, pa)), min(pb, pa))
                    } else {
                        shr8(add(mul(rt, pb), mul(t, pa)))
                    };
                    shr8(add(add(mul(inv, value), mul(s, mid)), full))
                };
                (mix(pb_e, pa_e, b0_e, a0_e), mix(pb_o, pa_o, b0_o, a0_o))
            });
        }

        pub(in super::super) fn normal_sad_pair<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 7],
            w: &Weights,
            simple: bool,
        ) {
            let (t, rt, tb, rtb, full) = (
                splat(w.t),
                splat(w.rt),
                splat(w.tb),
                splat(w.rtb),
                splat(255),
            );
            let odd_w = splat(265 - w.tb);
            p.run_pair(out, base, n, |pi, chunk| {
                let (pb_e, pb_o) = p.b.fetch_pair(p.at(chunk, l[0], l[1], pi));
                let (pa_e, pa_o) = p.a.fetch_pair(p.at(chunk, l[2], l[3], pi));
                let m9 = p.lane(l[4], pi);
                let m8 = p.lane(l[5], pi);
                let s = p.lane(l[6], pi);
                let inv9 = sub(full, m9);
                let inv8 = sub(full, m8);
                let inv_s = sub(full, s);
                let (b0_e, b0_o) = p.b.fetch_pair(chunk);
                let (a0_e, a0_o) = p.a.fetch_pair(chunk);
                let (ab0_e, ab0_o, aa0_e, aa0_o) = if simple {
                    (b0_e, b0_o, a0_e, a0_o)
                } else {
                    let (be, bo) = p.aux_b.fetch_pair(chunk);
                    let (ae, ao) = p.aux_a.fetch_pair(chunk);
                    (be, bo, ae, ao)
                };
                let mix = |pb, pa, b0, a0, ab0, aa0| {
                    let first = shr8(add(add(mul(inv9, pb), mul(m9, pa)), full));
                    let second = shr8(add(add(mul(inv8, pa), mul(m8, pb)), full));
                    let (value, mid) = if simple {
                        let value = shr8(add(mul(rt, first), mul(t, second)));
                        let weight = if p.index == 1 { odd_w } else { rtb };
                        let mixed = unsafe { vandq_s32(add(mul(weight, b0), mul(tb, a0)), splat(w.wrap)) };
                        (value, shr8(mixed))
                    } else {
                        let time = shr8(add(mul(rt, ab0), mul(t, aa0)));
                        let value = max(min(time, max(first, second)), min(first, second));
                        (value, shr8(add(mul(rtb, b0), mul(tb, a0))))
                    };
                    shr8(add(add(mul(inv_s, value), mul(s, mid)), full))
                };
                (
                    mix(pb_e, pa_e, b0_e, a0_e, ab0_e, aa0_e),
                    mix(pb_o, pa_o, b0_o, a0_o, ab0_o, aa0_o),
                )
            });
        }

        pub(in super::super) fn extended_sad_pair<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 11],
            w: &Weights,
        ) {
            let (t, tb, rtb, full) = (splat(w.t), splat(w.tb), splat(w.rtb), splat(255));
            p.run_pair(out, base, n, |pi, chunk| {
                let (c0_e, c0_o) = p.b.fetch_pair(p.at(chunk, l[0], l[1], pi));
                let (n0_e, n0_o) = p.a.fetch_pair(p.at(chunk, l[2], l[3], pi));
                let m9 = p.lane(l[8], pi);
                let m8 = p.lane(l[9], pi);
                let s = p.lane(l[10], pi);
                let inv_s = sub(full, s);
                let has_m9 = unsafe { vmaxvq_u32(vreinterpretq_u32_s32(m9)) } != 0;
                let has_m8 = unsafe { vmaxvq_u32(vreinterpretq_u32_s32(m8)) } != 0;
                let (c1_e, c1_o) = if has_m8 {
                    let pair = p.b.fetch_pair(p.at(chunk, l[4], l[5], pi));
                    (Some(pair.0), Some(pair.1))
                } else {
                    (None, None)
                };
                let (n1_e, n1_o) = if has_m9 {
                    let pair = p.a.fetch_pair(p.at(chunk, l[6], l[7], pi));
                    (Some(pair.0), Some(pair.1))
                } else {
                    (None, None)
                };
                let ve = blend_ext_neon::<T>(c0_e, n0_e, c1_e, n1_e, m9, m8, t, full);
                let vo = blend_ext_neon::<T>(c0_o, n0_o, c1_o, n1_o, m9, m8, t, full);
                let (b0_e, b0_o) = p.b.fetch_pair(chunk);
                let (a0_e, a0_o) = p.a.fetch_pair(chunk);
                let mix = |value, b0, a0| {
                    let mid = shr8(add(mul(rtb, b0), mul(tb, a0)));
                    shr8(add(add(mul(inv_s, value), mul(s, mid)), full))
                };
                (mix(ve, b0_e, a0_e), mix(vo, b0_o, a0_o))
            });
        }
    }

    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "avx2"),
        target_arch = "aarch64"
    )))]
    #[allow(
        clippy::inline_always,
        clippy::many_single_char_names,
        clippy::trivially_copy_pass_by_ref
    )]
    mod imp {
        use super::super::{Plane, Sample};
        use super::{Setup, Weights};

        pub(in super::super) struct Prepared<'a, T> {
            a: &'a [T],
            b: &'a [T],
            pel: i32,
            pitch: i32,
            aux_a: &'a [T],
            aux_b: &'a [T],
            pub(in super::super) index: usize,
        }

        impl<'a, T: Sample> Prepared<'a, T> {
            pub(in super::super) fn new(
                setup: &Setup<'a>,
                a: &Plane<'a, T>,
                b: &Plane<'a, T>,
                aux: Option<(Plane<'a, T>, Plane<'a, T>)>,
                index: usize,
            ) -> Self {
                let (aux_a, aux_b) = aux.unwrap_or((*a, *b));
                Self {
                    aux_a: aux_a.data,
                    aux_b: aux_b.data,
                    index,
                    a: a.data,
                    b: b.data,
                    pel: setup.pel,
                    pitch: setup.pitch,
                }
            }

            #[inline(always)]
            #[allow(clippy::unused_self)]
            fn lane(&self, values: &[i32], c: usize) -> i32 {
                values[c]
            }

            #[inline(always)]
            fn at(&self, plane: &[T], base: i32, x: &[i32], y: &[i32], c: usize) -> i32 {
                px(plane, base + self.pel * c as i32 + x[c] + self.pitch * y[c])
            }

            #[inline(always)]
            fn still(&self, plane: &[T], base: i32, c: usize) -> i32 {
                px(plane, base + self.pel * c as i32)
            }
        }

        pub(in super::super) fn fill(
            setup: &Setup<'_>,
            out: &mut [i32],
            pair: i32,
            cols: usize,
            vector: bool,
        ) {
            let (weights, shift) = if vector {
                (setup.vector_weights, setup.vector_shift)
            } else {
                (setup.weights, setup.shift)
            };
            let count = cols.next_multiple_of(8);
            let (lo, hi, shift) = (i32::from(pair as i16), pair >> 16, shift.min(31));
            for (value, &w) in out[..count].iter_mut().zip(&weights[..count]) {
                let sum = lo * i32::from(w as i16) + hi * (w >> 16);
                *value = i32::from((sum >> shift) as i16);
            }
        }

        pub(in super::super) fn fill_block<const N: usize>(
            setup: &Setup<'_>,
            lanes: &mut [i32],
            (span, at, cols): (usize, usize, usize),
            (top, bottom): (&[i16; 32], &[i16; 32]),
            (wt, wb, shift_y): (i16, i16, u32),
        ) {
            let mix = |i: usize| {
                wt.wrapping_mul(top[i])
                    .wrapping_add(wb.wrapping_mul(bottom[i]))
                    >> shift_y
            };
            for (k, lane) in lanes.chunks_exact_mut(span).take(N).enumerate() {
                let pair = i32::from(mix(2 * k) as u16) | (i32::from(mix(2 * k + 1) as u16) << 16);
                fill(setup, &mut lane[at..], pair, cols, false);
            }
        }

        #[inline(always)]
        fn px<T: Sample>(plane: &[T], index: i32) -> i32 {
            let last = plane.len().saturating_sub(1);
            plane
                .get((index.max(0) as usize).min(last))
                .map_or(0, |&v| v.into())
        }

        pub(in super::super) fn fast<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            next: bool,
            base: i32,
            n: usize,
            l: &[&[i32]; 2],
        ) {
            let src = if next { p.a } else { p.b };
            for (c, o) in out.iter_mut().enumerate().take(n) {
                *o = T::new(p.at(src, base, l[0], l[1], c));
            }
        }

        pub(in super::super) fn no_mask<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 4],
            w: &Weights,
            median: bool,
        ) {
            for (c, o) in out.iter_mut().enumerate().take(n) {
                let pb = p.at(p.b, base, l[0], l[1], c);
                let pa = p.at(p.a, base, l[2], l[3], c);
                *o = T::new(if median {
                    let mid = (w.rtb * p.still(p.b, base, c) + w.tb * p.still(p.a, base, c)) >> 8;
                    mid.min(pb.max(pa)).max(pb.min(pa))
                } else {
                    (w.rt * pb + w.t * pa) >> 8
                });
            }
        }

        pub(in super::super) fn normal<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 6],
            w: &Weights,
            simple: bool,
        ) {
            for (c, o) in out.iter_mut().enumerate().take(n) {
                let pb = p.at(p.b, base, l[0], l[1], c);
                let pa = p.at(p.a, base, l[2], l[3], c);
                let (m9, m8) = (p.lane(l[4], c), p.lane(l[5], c));
                let (first, second) = if simple {
                    (
                        ((255 - m9) * pb + m9 * pa + 255) >> 8,
                        ((255 - m8) * pa + m8 * pb + 255) >> 8,
                    )
                } else {
                    let bb = (255 - m9) * pb;
                    let aa = (255 - m8) * pa;
                    let b0 = p.still(p.b, base, c);
                    let a0 = p.still(p.a, base, c);
                    let inner = |value: i32| ((value as u32).wrapping_add(255) >> 8) as i32;
                    (
                        (bb + inner(m9.wrapping_mul(aa + m8 * b0)) + 255) >> 8,
                        (aa + inner((bb + a0 * m9).wrapping_mul(m8)) + 255) >> 8,
                    )
                };
                *o = T::new((w.rt * first + w.t * second) >> 8);
            }
        }

        pub(in super::super) fn extended<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 10],
            w: &Weights,
        ) {
            for (c, o) in out.iter_mut().enumerate().take(n) {
                let c0 = p.at(p.b, base, l[0], l[1], c);
                let n0 = p.at(p.a, base, l[2], l[3], c);
                let (m9, m8) = (p.lane(l[8], c), p.lane(l[9], c));
                if (m9 | m8) == 0 {
                    let first = if size_of::<T>() == 1 {
                        c0
                    } else {
                        (c0 * 255 + 255) >> 8
                    };
                    let second = if size_of::<T>() == 1 {
                        n0
                    } else {
                        (n0 * 255 + 255) >> 8
                    };
                    *o = T::new((w.rt * first + w.t * second) >> 8);
                    continue;
                }
                let lo = c0.min(n0);
                let hi = c0.max(n0);
                let first = if m9 != 0 {
                    let n1 = p.at(p.a, base, l[6], l[7], c);
                    let cn = lo.max(n1.min(hi));
                    (m9 * cn + c0 * (255 - m9) + 255) >> 8
                } else if size_of::<T>() == 1 {
                    c0
                } else {
                    (c0 * 255 + 255) >> 8
                };
                let second = if m8 != 0 {
                    let c1 = p.at(p.b, base, l[4], l[5], c);
                    let cc = lo.max(c1.min(hi));
                    (m8 * cc + n0 * (255 - m8) + 255) >> 8
                } else if size_of::<T>() == 1 {
                    n0
                } else {
                    (n0 * 255 + 255) >> 8
                };
                *o = T::new((w.rt * first + w.t * second) >> 8);
            }
        }

        pub(in super::super) fn fast_sad<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            next: bool,
            base: i32,
            n: usize,
            l: &[&[i32]; 3],
        ) {
            let src = if next { p.a } else { p.b };
            for (c, o) in out.iter_mut().enumerate().take(n) {
                let moved = p.at(src, base, l[0], l[1], c);
                let still = p.still(src, base, c);
                let s = p.lane(l[2], c);
                *o = T::new(((255 - s) * moved + s * still + 255) >> 8);
            }
        }

        pub(in super::super) fn no_mask_sad<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 5],
            w: &Weights,
            median: bool,
        ) {
            for (c, o) in out.iter_mut().enumerate().take(n) {
                let pb = p.at(p.b, base, l[0], l[1], c);
                let pa = p.at(p.a, base, l[2], l[3], c);
                let s = p.lane(l[4], c);
                let mid = (w.rtb * p.still(p.b, base, c) + w.tb * p.still(p.a, base, c)) >> 8;
                let value = if median {
                    mid.min(pb.max(pa)).max(pb.min(pa))
                } else {
                    (w.rt * pb + w.t * pa) >> 8
                };
                *o = T::new(((255 - s) * value + s * mid + 255) >> 8);
            }
        }

        pub(in super::super) fn normal_sad<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 7],
            w: &Weights,
            simple: bool,
        ) {
            for (c, o) in out.iter_mut().enumerate().take(n) {
                let pb = p.at(p.b, base, l[0], l[1], c);
                let pa = p.at(p.a, base, l[2], l[3], c);
                let (m9, m8, s) = (p.lane(l[4], c), p.lane(l[5], c), p.lane(l[6], c));
                let first = ((255 - m9) * pb + m9 * pa + 255) >> 8;
                let second = ((255 - m8) * pa + m8 * pb + 255) >> 8;
                let b0 = p.still(p.b, base, c);
                let a0 = p.still(p.a, base, c);
                let (value, mid) = if simple {
                    let weight = if p.index == 1 { 265 - w.tb } else { w.rtb };
                    (
                        (w.rt * first + w.t * second) >> 8,
                        ((weight * b0 + w.tb * a0) & w.wrap) >> 8,
                    )
                } else {
                    let time =
                        (w.rt * p.still(p.aux_b, base, c) + w.t * p.still(p.aux_a, base, c)) >> 8;
                    (
                        time.min(first.max(second)).max(first.min(second)),
                        (w.rtb * b0 + w.tb * a0) >> 8,
                    )
                };
                *o = T::new(((255 - s) * value + s * mid + 255) >> 8);
            }
        }

        pub(in super::super) fn extended_sad<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 11],
            w: &Weights,
        ) {
            for (c, o) in out.iter_mut().enumerate().take(n) {
                let c0 = p.at(p.b, base, l[0], l[1], c);
                let n0 = p.at(p.a, base, l[2], l[3], c);
                let (m9, m8, s) = (p.lane(l[8], c), p.lane(l[9], c), p.lane(l[10], c));
                let value = if (m9 | m8) == 0 {
                    let first = if size_of::<T>() == 1 {
                        c0
                    } else {
                        (c0 * 255 + 255) >> 8
                    };
                    let second = if size_of::<T>() == 1 {
                        n0
                    } else {
                        (n0 * 255 + 255) >> 8
                    };
                    (w.rt * first + w.t * second) >> 8
                } else {
                    let lo = c0.min(n0);
                    let hi = c0.max(n0);
                    let first = if m9 != 0 {
                        let n1 = p.at(p.a, base, l[6], l[7], c);
                        let cn = lo.max(n1.min(hi));
                        (m9 * cn + c0 * (255 - m9) + 255) >> 8
                    } else if size_of::<T>() == 1 {
                        c0
                    } else {
                        (c0 * 255 + 255) >> 8
                    };
                    let second = if m8 != 0 {
                        let c1 = p.at(p.b, base, l[4], l[5], c);
                        let cc = lo.max(c1.min(hi));
                        (m8 * cc + n0 * (255 - m8) + 255) >> 8
                    } else if size_of::<T>() == 1 {
                        n0
                    } else {
                        (n0 * 255 + 255) >> 8
                    };
                    (w.rt * first + w.t * second) >> 8
                };
                let b0 = p.still(p.b, base, c);
                let a0 = p.still(p.a, base, c);
                let mid = (w.rtb * b0 + w.tb * a0) >> 8;
                *o = T::new(((255 - s) * value + s * mid + 255) >> 8);
            }
        }

        #[inline(always)]
        fn pair_at<T: Sample>(
            p: &Prepared<'_, T>,
            plane: &[T],
            base: i32,
            x: &[i32],
            y: &[i32],
            pi: usize,
            c: usize,
        ) -> (i32, i32) {
            let idx = base + p.pel * c as i32 + x[pi] + p.pitch * y[pi];
            (px(plane, idx), px(plane, idx + p.pel))
        }

        #[inline(always)]
        fn pair_still<T: Sample>(p: &Prepared<'_, T>, plane: &[T], base: i32, c: usize) -> (i32, i32) {
            let idx = base + p.pel * c as i32;
            (px(plane, idx), px(plane, idx + p.pel))
        }

        #[inline(always)]
        fn write_pair<T: Sample>(out: &mut [T], c: usize, n: usize, e: i32, o: i32) {
            if c < out.len() {
                out[c] = T::new(e);
            }
            if c + 1 < n && c + 1 < out.len() {
                out[c + 1] = T::new(o);
            }
        }

        pub(in super::super) fn fast_pair<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            next: bool,
            base: i32,
            n: usize,
            l: &[&[i32]; 2],
        ) {
            let src = if next { p.a } else { p.b };
            let pairs = n.div_ceil(2);
            for pi in 0..pairs {
                let c = 2 * pi;
                let (e, o) = pair_at(p, src, base, l[0], l[1], pi, c);
                write_pair(out, c, n, e, o);
            }
        }

        pub(in super::super) fn no_mask_pair<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 4],
            w: &Weights,
            median: bool,
        ) {
            let pairs = n.div_ceil(2);
            for pi in 0..pairs {
                let c = 2 * pi;
                let (pb_e, pb_o) = pair_at(p, p.b, base, l[0], l[1], pi, c);
                let (pa_e, pa_o) = pair_at(p, p.a, base, l[2], l[3], pi, c);
                let (ve, vo) = if median {
                    let (b0_e, b0_o) = pair_still(p, p.b, base, c);
                    let (a0_e, a0_o) = pair_still(p, p.a, base, c);
                    let med = |pb: i32, pa: i32, b0: i32, a0: i32| {
                        let mid = (w.rtb * b0 + w.tb * a0) >> 8;
                        mid.min(pb.max(pa)).max(pb.min(pa))
                    };
                    (med(pb_e, pa_e, b0_e, a0_e), med(pb_o, pa_o, b0_o, a0_o))
                } else {
                    ((w.rt * pb_e + w.t * pa_e) >> 8, (w.rt * pb_o + w.t * pa_o) >> 8)
                };
                write_pair(out, c, n, ve, vo);
            }
        }

        pub(in super::super) fn normal_pair<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 6],
            w: &Weights,
            simple: bool,
        ) {
            let pairs = n.div_ceil(2);
            for pi in 0..pairs {
                let c = 2 * pi;
                let (pb_e, pb_o) = pair_at(p, p.b, base, l[0], l[1], pi, c);
                let (pa_e, pa_o) = pair_at(p, p.a, base, l[2], l[3], pi, c);
                let (m9, m8) = (p.lane(l[4], pi), p.lane(l[5], pi));
                let (ve, vo) = if simple {
                    let mix = |pb: i32, pa: i32| {
                        let f = ((255 - m9) * pb + m9 * pa + 255) >> 8;
                        let s = ((255 - m8) * pa + m8 * pb + 255) >> 8;
                        (w.rt * f + w.t * s) >> 8
                    };
                    (mix(pb_e, pa_e), mix(pb_o, pa_o))
                } else {
                    let (b0_e, b0_o) = pair_still(p, p.b, base, c);
                    let (a0_e, a0_o) = pair_still(p, p.a, base, c);
                    let mix = |pb: i32, pa: i32, b0: i32, a0: i32| {
                        let bb = (255 - m9) * pb;
                        let aa = (255 - m8) * pa;
                        let inner = |v: i32| ((v as u32).wrapping_add(255) >> 8) as i32;
                        let f = (bb + inner(m9.wrapping_mul(aa + m8 * b0)) + 255) >> 8;
                        let s = (aa + inner((bb + a0 * m9).wrapping_mul(m8)) + 255) >> 8;
                        (w.rt * f + w.t * s) >> 8
                    };
                    (mix(pb_e, pa_e, b0_e, a0_e), mix(pb_o, pa_o, b0_o, a0_o))
                };
                write_pair(out, c, n, ve, vo);
            }
        }

        pub(in super::super) fn extended_pair<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 10],
            w: &Weights,
        ) {
            let pairs = n.div_ceil(2);
            for pi in 0..pairs {
                let c = 2 * pi;
                let (c0_e, c0_o) = pair_at(p, p.b, base, l[0], l[1], pi, c);
                let (n0_e, n0_o) = pair_at(p, p.a, base, l[2], l[3], pi, c);
                let (m9, m8) = (p.lane(l[8], pi), p.lane(l[9], pi));
                let (c1_e, c1_o) = if m8 != 0 {
                    let pair = pair_at(p, p.b, base, l[4], l[5], pi, c);
                    (Some(pair.0), Some(pair.1))
                } else {
                    (None, None)
                };
                let (n1_e, n1_o) = if m9 != 0 {
                    let pair = pair_at(p, p.a, base, l[6], l[7], pi, c);
                    (Some(pair.0), Some(pair.1))
                } else {
                    (None, None)
                };
                let mix = |c0: i32, n0: i32, c1: Option<i32>, n1: Option<i32>| {
                    let lo = c0.min(n0);
                    let hi = c0.max(n0);
                    let first = match n1 {
                        Some(n1) => (m9 * lo.max(n1.min(hi)) + c0 * (255 - m9) + 255) >> 8,
                        None if size_of::<T>() == 1 => c0,
                        None => (c0 * 255 + 255) >> 8,
                    };
                    let second = match c1 {
                        Some(c1) => (m8 * lo.max(c1.min(hi)) + n0 * (255 - m8) + 255) >> 8,
                        None if size_of::<T>() == 1 => n0,
                        None => (n0 * 255 + 255) >> 8,
                    };
                    (w.rt * first + w.t * second) >> 8
                };
                write_pair(
                    out,
                    c,
                    n,
                    mix(c0_e, n0_e, c1_e, n1_e),
                    mix(c0_o, n0_o, c1_o, n1_o),
                );
            }
        }

        pub(in super::super) fn fast_sad_pair<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            next: bool,
            base: i32,
            n: usize,
            l: &[&[i32]; 3],
        ) {
            let src = if next { p.a } else { p.b };
            let pairs = n.div_ceil(2);
            for pi in 0..pairs {
                let c = 2 * pi;
                let (me, mo) = pair_at(p, src, base, l[0], l[1], pi, c);
                let (se, so) = pair_still(p, src, base, c);
                let s = p.lane(l[2], pi);
                write_pair(
                    out,
                    c,
                    n,
                    ((255 - s) * me + s * se + 255) >> 8,
                    ((255 - s) * mo + s * so + 255) >> 8,
                );
            }
        }

        pub(in super::super) fn no_mask_sad_pair<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 5],
            w: &Weights,
            median: bool,
        ) {
            let pairs = n.div_ceil(2);
            for pi in 0..pairs {
                let c = 2 * pi;
                let (pb_e, pb_o) = pair_at(p, p.b, base, l[0], l[1], pi, c);
                let (pa_e, pa_o) = pair_at(p, p.a, base, l[2], l[3], pi, c);
                let s = p.lane(l[4], pi);
                let (b0_e, b0_o) = pair_still(p, p.b, base, c);
                let (a0_e, a0_o) = pair_still(p, p.a, base, c);
                let mix = |pb: i32, pa: i32, b0: i32, a0: i32| {
                    let mid = (w.rtb * b0 + w.tb * a0) >> 8;
                    let value = if median {
                        mid.min(pb.max(pa)).max(pb.min(pa))
                    } else {
                        (w.rt * pb + w.t * pa) >> 8
                    };
                    ((255 - s) * value + s * mid + 255) >> 8
                };
                write_pair(
                    out,
                    c,
                    n,
                    mix(pb_e, pa_e, b0_e, a0_e),
                    mix(pb_o, pa_o, b0_o, a0_o),
                );
            }
        }

        pub(in super::super) fn normal_sad_pair<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 7],
            w: &Weights,
            simple: bool,
        ) {
            let pairs = n.div_ceil(2);
            for pi in 0..pairs {
                let c = 2 * pi;
                let (pb_e, pb_o) = pair_at(p, p.b, base, l[0], l[1], pi, c);
                let (pa_e, pa_o) = pair_at(p, p.a, base, l[2], l[3], pi, c);
                let (m9, m8, s) = (p.lane(l[4], pi), p.lane(l[5], pi), p.lane(l[6], pi));
                let (b0_e, b0_o) = pair_still(p, p.b, base, c);
                let (a0_e, a0_o) = pair_still(p, p.a, base, c);
                let (ab0_e, ab0_o, aa0_e, aa0_o) = if simple {
                    (b0_e, b0_o, a0_e, a0_o)
                } else {
                    let (be, bo) = pair_still(p, p.aux_b, base, c);
                    let (ae, ao) = pair_still(p, p.aux_a, base, c);
                    (be, bo, ae, ao)
                };
                let mix = |pb: i32, pa: i32, b0: i32, a0: i32, ab0: i32, aa0: i32| {
                    let first = ((255 - m9) * pb + m9 * pa + 255) >> 8;
                    let second = ((255 - m8) * pa + m8 * pb + 255) >> 8;
                    let (value, mid) = if simple {
                        let weight = if p.index == 1 { 265 - w.tb } else { w.rtb };
                        (
                            (w.rt * first + w.t * second) >> 8,
                            ((weight * b0 + w.tb * a0) & w.wrap) >> 8,
                        )
                    } else {
                        let time = (w.rt * ab0 + w.t * aa0) >> 8;
                        (
                            time.min(first.max(second)).max(first.min(second)),
                            (w.rtb * b0 + w.tb * a0) >> 8,
                        )
                    };
                    ((255 - s) * value + s * mid + 255) >> 8
                };
                write_pair(
                    out,
                    c,
                    n,
                    mix(pb_e, pa_e, b0_e, a0_e, ab0_e, aa0_e),
                    mix(pb_o, pa_o, b0_o, a0_o, ab0_o, aa0_o),
                );
            }
        }

        pub(in super::super) fn extended_sad_pair<T: Sample>(
            out: &mut [T],
            p: &Prepared<'_, T>,
            base: i32,
            n: usize,
            l: &[&[i32]; 11],
            w: &Weights,
        ) {
            let pairs = n.div_ceil(2);
            for pi in 0..pairs {
                let c = 2 * pi;
                let (c0_e, c0_o) = pair_at(p, p.b, base, l[0], l[1], pi, c);
                let (n0_e, n0_o) = pair_at(p, p.a, base, l[2], l[3], pi, c);
                let (m9, m8, s) = (p.lane(l[8], pi), p.lane(l[9], pi), p.lane(l[10], pi));
                let (c1_e, c1_o) = if m8 != 0 {
                    let pair = pair_at(p, p.b, base, l[4], l[5], pi, c);
                    (Some(pair.0), Some(pair.1))
                } else {
                    (None, None)
                };
                let (n1_e, n1_o) = if m9 != 0 {
                    let pair = pair_at(p, p.a, base, l[6], l[7], pi, c);
                    (Some(pair.0), Some(pair.1))
                } else {
                    (None, None)
                };
                let (b0_e, b0_o) = pair_still(p, p.b, base, c);
                let (a0_e, a0_o) = pair_still(p, p.a, base, c);
                let mix = |c0: i32, n0: i32, c1: Option<i32>, n1: Option<i32>, b0: i32, a0: i32| {
                    let lo = c0.min(n0);
                    let hi = c0.max(n0);
                    let first = match n1 {
                        Some(n1) => (m9 * lo.max(n1.min(hi)) + c0 * (255 - m9) + 255) >> 8,
                        None if size_of::<T>() == 1 => c0,
                        None => (c0 * 255 + 255) >> 8,
                    };
                    let second = match c1 {
                        Some(c1) => (m8 * lo.max(c1.min(hi)) + n0 * (255 - m8) + 255) >> 8,
                        None if size_of::<T>() == 1 => n0,
                        None => (n0 * 255 + 255) >> 8,
                    };
                    let value = (w.rt * first + w.t * second) >> 8;
                    let mid = (w.rtb * b0 + w.tb * a0) >> 8;
                    ((255 - s) * value + s * mid + 255) >> 8
                };
                write_pair(
                    out,
                    c,
                    n,
                    mix(c0_e, n0_e, c1_e, n1_e, b0_e, a0_e),
                    mix(c0_o, n0_o, c1_o, n1_o, b0_o, a0_o),
                );
            }
        }
    }

    pub(super) use imp::{
        Prepared, extended, extended_pair, extended_sad, extended_sad_pair, fast, fast_pair,
        fast_sad, fast_sad_pair, fill, fill_block, no_mask, no_mask_pair, no_mask_sad,
        no_mask_sad_pair, normal, normal_pair, normal_sad, normal_sad_pair,
    };

    const _: () = assert!(MAX_STEP.is_multiple_of(8));
}
