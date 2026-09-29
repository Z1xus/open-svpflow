#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use crate::metadata::VectorData;

const LOG2_STEP: [i8; 32] = [
    0, 1, 2, 2, -1, 3, 3, 3, -1, -1, -1, 4, -1, 4, -1, 4, -1, -1, -1, -1, -1, -1, -1, 5, -1, -1,
    -1, 5, -1, -1, -1, 5,
];
const LUT_LEN: usize = 2048;
const NEUTRAL: u16 = 1024;
const MAX_STEP: usize = 64;

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
    cover: Vec<i32>,
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
            cover: Vec::new(),
        }
    }

    pub const fn shape(&self) -> &FieldShape {
        &self.shape
    }

    const fn set(&self, backward: bool) -> &Vec<BlockVector> {
        if backward {
            &self.backward
        } else {
            &self.forward
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
        let sum =
            self.backward
                .iter()
                .zip(&self.forward)
                .take(count)
                .fold(0.0, |sum, (back, fwd)| {
                    sum + (f64::from(u16::from(back.luma) + u16::from(fwd.luma)) / denom).ln()
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
        clippy::too_many_lines
    )]
    pub fn cover_mask(
        &mut self,
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
        self.cover.clear();
        self.cover.resize(len + 1, 0);
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
        let points = &mut self.cover[..];
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
        let level = |covered: i32| {
            let value = (f64::from(area - covered) * scale * 256.0 / f64::from(area)) as i32;
            if value > 255 { 255 } else { value as u8 }
        };
        let table: Vec<u8> = (0..=area.max(0)).map(level).collect();
        let top = table.len() as i32 - 1;
        let width = width as usize;
        let mut column = vec![0i32; stride];
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
            for (out, window) in out.iter_mut().zip(column.windows(3)) {
                let total = window[0].wrapping_add(window[1]).wrapping_add(window[2]);
                let covered = (total >> 3).min(area);
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
        for (chunk_index, chunk) in row.chunks(LANES).enumerate() {
            let base = (chunk_index * LANES) as i32;
            let mut vx = [0i32; LANES];
            let mut vy = [0i32; LANES];
            for ((x, y), vector) in vx.iter_mut().zip(vy.iter_mut()).zip(chunk) {
                *x = i32::from(vector.x);
                *y = i32::from(vector.y);
            }
            let mut cells = [[(0u32, 0i32); 4]; LANES];
            for (lane, cell) in cells.iter_mut().enumerate() {
                let tx = scale(self.time * vx[lane]);
                let ty = scale(self.time * vy[lane]);
                *cell = self.cells(
                    base + lane as i32,
                    by,
                    (tx, ty),
                    (floor_x.apply(tx), floor_y.apply(ty)),
                );
            }
            for cell in &cells[..chunk.len()] {
                for &(index, value) in cell {
                    let target = &mut points[index as usize];
                    *target = target.wrapping_add(value);
                }
            }
        }
    }
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
}

pub struct Renderer {
    shape: RenderShape,
    chroma_div: i32,
    shifts: [i32; 4],
    time: i32,
    time_blend: i32,
    lut_rev: Vec<i16>,
    lut_fwd: Vec<i16>,
    x_luma: Vec<[i16; 2]>,
    y_luma: Vec<i16>,
    x_chroma: Vec<[i16; 2]>,
    y_chroma: Vec<i16>,
}

#[derive(Clone, Copy)]
pub struct Plane<'a> {
    pub data: &'a [u8],
    pub pitch: usize,
    pub slack: usize,
}

pub struct PlaneMut<'a> {
    pub data: &'a mut [u8],
    pub pitch: usize,
}

#[derive(Clone, Copy)]
pub struct Frame<'a> {
    pub y: Plane<'a>,
    pub u: Plane<'a>,
    pub v: Plane<'a>,
}

pub struct FrameMut<'a> {
    pub y: PlaneMut<'a>,
    pub u: PlaneMut<'a>,
    pub v: PlaneMut<'a>,
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
        let chroma_div = 2;
        let shifts = [
            log2_step(shape.step_x),
            log2_step(shape.step_y),
            log2_step(shape.step_x / 2),
            log2_step(shape.step_y / 2),
        ];
        let pow = |shift: i32| 2f64.powf(f64::from(shift));
        let x_luma = weight_pairs(
            shape.step_x,
            shape.step_x,
            pow(shifts[0]) / f64::from(shape.step_x),
        );
        let y_luma = weights(shape.step_y, pow(shifts[1]) / f64::from(shape.step_y));
        let x_chroma = weight_pairs(
            shape.step_x / 2,
            shape.step_x / 2,
            (pow(shifts[2]) + pow(shifts[2])) / f64::from(shape.step_x),
        );
        let y_chroma = weights(
            shape.step_y / 2,
            (pow(shifts[3]) + pow(shifts[3])) / f64::from(shape.step_y),
        );
        let mut renderer = Self {
            shape,
            chroma_div,
            shifts,
            time: -1,
            time_blend: 0,
            lut_rev: vec![0; LUT_LEN],
            lut_fwd: vec![0; LUT_LEN],
            x_luma,
            y_luma,
            x_chroma,
            y_chroma,
        };
        renderer.set_time(0);
        renderer
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
        for (i, (r, f)) in self.lut_rev.iter_mut().zip(&mut self.lut_fwd).enumerate() {
            let v = i as i32 - 1024;
            *r = (v * rev / 256) as i16;
            *f = (v * time / 256) as i16;
        }
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
            let d = self.chroma_div;
            Pass {
                width: s.width / 2,
                height: s.height / d,
                step_x: s.step_x / 2,
                step_y: s.step_y / d,
                origin_x: if interp { s.origin_x / 2 } else { 0 },
                origin_y: if interp { s.origin_y / d } else { 0 },
                div_x: 2,
                div_y: d,
                shift_x: shift(2),
                shift_y: shift(3),
                x_weights: &self.x_chroma,
                y_weights: if d == 2 { &self.y_chroma } else { &self.y_luma },
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

    pub fn fill(&self, dst: &mut FrameMut<'_>) {
        let h = self.shape.height as usize;
        self.fill_rows(dst, [0..h, 0..h / 2]);
    }

    fn fill_rows(&self, dst: &mut FrameMut<'_>, rows: [std::ops::Range<usize>; 2]) {
        let w = self.shape.width as usize;
        let [luma, chroma] = rows;
        for (plane, pw, ph, value) in [
            (&mut dst.y, w, luma.len(), 0x7F),
            (&mut dst.u, w / 2, chroma.len(), 0x80),
            (&mut dst.v, w / 2, chroma.len(), 0x80),
        ] {
            for row in 0..ph {
                plane.data[row * plane.pitch..row * plane.pitch + pw].fill(value);
            }
        }
    }

    pub fn render(
        &self,
        algo: Algo,
        interp: bool,
        dst: &mut FrameMut<'_>,
        next: Frame<'_>,
        current: Frame<'_>,
        vectors: &PackedVectors<'_>,
    ) {
        self.render_band(
            algo,
            interp,
            dst,
            next,
            current,
            vectors,
            -1..self.shape.grid_h,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render_band(
        &self,
        algo: Algo,
        interp: bool,
        dst: &mut FrameMut<'_>,
        next: Frame<'_>,
        current: Frame<'_>,
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
            let mut planes: Vec<(&mut PlaneMut<'_>, Plane<'_>, Plane<'_>)> = if chroma {
                vec![
                    (&mut dst.u, next.u, current.u),
                    (&mut dst.v, next.v, current.v),
                ]
            } else {
                vec![(&mut dst.y, next.y, current.y)]
            };
            let aux = chroma.then_some((next.y, current.y));
            self.render_algo(algo, interp, &pass, &mut planes, vectors, aux);
        }
    }

    fn channel(raw: &[u16], lut: &[i16], div: i32) -> Vec<i16> {
        raw.iter()
            .map(|&value| (i32::from(lut[usize::from(value)]) / div) as i16)
            .collect()
    }

    #[allow(clippy::too_many_lines)]
    fn render_algo(
        &self,
        algo: Algo,
        interp: bool,
        pass: &Pass<'_>,
        planes: &mut [(&mut PlaneMut<'_>, Plane<'_>, Plane<'_>)],
        v: &PackedVectors<'_>,
        aux: Option<(Plane<'_>, Plane<'_>)>,
    ) {
        let t = self.time;
        let rt = 256 - t;
        let tb = self.time_blend;
        let rtb = 256 - tb;
        let fwd = &self.lut_fwd;
        let rev = &self.lut_rev;
        let vec2 = |x: &[u16], y: &[u16], lut: &[i16]| {
            (
                Self::channel(x, lut, pass.div_x),
                Self::channel(y, lut, pass.div_y),
            )
        };
        let mask = |m: &[u8]| m.iter().map(|&x| i16::from(x)).collect::<Vec<i16>>();
        let weights = row::Weights { t, rt, tb, rtb };
        match algo {
            Algo::Fast { next } => {
                let (x, y) = if next {
                    vec2(v.fwd_x, v.fwd_y, fwd)
                } else {
                    vec2(v.bwd_x, v.bwd_y, fwd)
                };
                self.walk(
                    pass,
                    interp,
                    [&x, &y],
                    [None; 2],
                    false,
                    aux,
                    planes,
                    |out, p, base, n, l| {
                        row::fast(out, p, next, base, n, l);
                    },
                );
            }
            Algo::NoMask { median } => {
                let (bx, by) = vec2(v.bwd_x, v.bwd_y, fwd);
                let (fx, fy) = vec2(v.fwd_x, v.fwd_y, rev);
                self.walk(
                    pass,
                    interp,
                    [&bx, &by, &fx, &fy],
                    [None; 4],
                    false,
                    aux,
                    planes,
                    |out, p, base, n, l| {
                        row::no_mask(out, p, base, n, l, &weights, median);
                    },
                );
            }
            Algo::FastSad { next } => {
                let (x, y) = if next {
                    vec2(v.fwd_x, v.fwd_y, fwd)
                } else {
                    vec2(v.bwd_x, v.bwd_y, fwd)
                };
                let sad = mask(v.sad);
                self.walk(
                    pass,
                    interp,
                    [&x, &y, &sad],
                    [None, None, Some(v.sad)],
                    true,
                    aux,
                    planes,
                    |out, p, base, n, l| {
                        row::fast_sad(out, p, next, base, n, l);
                    },
                );
            }
            Algo::NoMaskSad { median } => {
                let (bx, by) = vec2(v.bwd_x, v.bwd_y, fwd);
                let (fx, fy) = vec2(v.fwd_x, v.fwd_y, rev);
                let sad = mask(v.sad);
                self.walk(
                    pass,
                    interp,
                    [&bx, &by, &fx, &fy, &sad],
                    [None, None, None, None, Some(v.sad)],
                    false,
                    aux,
                    planes,
                    |out, p, base, n, l| {
                        row::no_mask_sad(out, p, base, n, l, &weights, median);
                    },
                );
            }
            Algo::NormalSad { simple } => {
                let (bx, by) = vec2(v.bwd_x, v.bwd_y, fwd);
                let (fx, fy) = vec2(v.fwd_x, v.fwd_y, rev);
                let (m9, m8) = (mask(v.cover_fwd), mask(v.cover_bwd));
                let sad = mask(v.sad);
                self.walk(
                    pass,
                    interp,
                    [&bx, &by, &fx, &fy, &m9, &m8, &sad],
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
                    aux,
                    planes,
                    |out, p, base, n, l| row::normal_sad(out, p, base, n, l, &weights, simple),
                );
            }
            Algo::Normal { simple } => {
                let (bx, by) = vec2(v.bwd_x, v.bwd_y, fwd);
                let (fx, fy) = vec2(v.fwd_x, v.fwd_y, rev);
                let (m9, m8) = (mask(v.cover_fwd), mask(v.cover_bwd));
                self.walk(
                    pass,
                    interp,
                    [&bx, &by, &fx, &fy, &m9, &m8],
                    [None, None, None, None, Some(v.cover_fwd), Some(v.cover_bwd)],
                    false,
                    aux,
                    planes,
                    |out, p, base, n, l| row::normal(out, p, base, n, l, &weights, simple),
                );
            }
            Algo::Extended => {
                let (bx, by) = vec2(v.bwd_x, v.bwd_y, fwd);
                let (fx, fy) = vec2(v.fwd_x, v.fwd_y, rev);
                let (px, py) = vec2(v.prev_bwd_x, v.prev_bwd_y, fwd);
                let (nx, ny) = vec2(v.next_fwd_x, v.next_fwd_y, rev);
                let (m9, m8) = (mask(v.cover_fwd), mask(v.cover_bwd));
                self.walk(
                    pass,
                    interp,
                    [&bx, &by, &fx, &fy, &px, &py, &nx, &ny, &m9, &m8],
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
                    aux,
                    planes,
                    |out, p, base, n, l| row::extended(out, p, base, n, l, &weights),
                );
            }
            Algo::ExtendedSad => {
                let (bx, by) = vec2(v.bwd_x, v.bwd_y, fwd);
                let (fx, fy) = vec2(v.fwd_x, v.fwd_y, rev);
                let (px, py) = vec2(v.prev_bwd_x, v.prev_bwd_y, fwd);
                let (nx, ny) = vec2(v.next_fwd_x, v.next_fwd_y, rev);
                let (m9, m8) = (mask(v.cover_fwd), mask(v.cover_bwd));
                let sad = mask(v.sad);
                self.walk(
                    pass,
                    interp,
                    [&bx, &by, &fx, &fy, &px, &py, &nx, &ny, &m9, &m8, &sad],
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
                    aux,
                    planes,
                    |out, p, base, n, l| row::extended_sad(out, p, base, n, l, &weights),
                );
            }
            Algo::Fill => {}
        }
    }

    fn bilinear(&self, buf: &[u8], skip: i32, x: i32, y: i32, pass: &Pass<'_>) -> u8 {
        let s = &self.shape;
        let (ox, oy, sx, sy) = if pass.div_x == 2 {
            (
                s.origin_x / 2,
                s.origin_y / self.chroma_div,
                s.step_x / 2,
                s.step_y / self.chroma_div,
            )
        } else {
            (s.origin_x, s.origin_y, s.step_x, s.step_y)
        };
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

    #[allow(clippy::too_many_lines, clippy::too_many_arguments)]
    fn walk<const N: usize, K>(
        &self,
        pass: &Pass<'_>,
        interp: bool,
        channels: [&[i16]; N],
        samplers: [Option<&[u8]>; N],
        skip_rows: bool,
        aux: Option<(Plane<'_>, Plane<'_>)>,
        planes: &mut [(&mut PlaneMut<'_>, Plane<'_>, Plane<'_>)],
        kernel: K,
    ) where
        K: Fn(&mut [u8], &row::Prepared<'_>, i32, usize, &[&[i32]; N]),
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
        let sources: Vec<(Plane<'_>, Plane<'_>)> =
            planes.iter().map(|(_, a, b)| (*a, *b)).collect();
        let prepared: Vec<row::Prepared<'_>> = sources
            .iter()
            .enumerate()
            .map(|(index, (a, b))| row::Prepared::new(&setup, a, b, aux, index))
            .collect();
        const { assert!(N <= 16) };
        let width = usize::try_from(pass.width).unwrap_or(0);
        let span = width + MAX_STEP + 8;
        let mut lanes = vec![0i32; N * span];
        let mut blocks: Vec<(i32, usize, [i16; 32], [i16; 32])> =
            Vec::with_capacity(usize::try_from(grid_w).unwrap_or(0) + 1);
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
                let mut top = [0i16; 32];
                let mut bottom = [0i16; 32];
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
                blocks.push((px0, cols as usize, top, bottom));
            }
            for r in 0..rows {
                let py = py0 + r;
                let wt = pass.y_weights[(pass.step_y - r) as usize];
                let wb = pass.y_weights[r as usize];
                let shift_y = pass.shift_y.min(15);
                for (px0, cols, top, bottom) in &blocks {
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
                let view: [&[i32]; N] = std::array::from_fn(|k| &lanes[k * span..(k + 1) * span]);
                let base = pel * py * pitch;
                for ((dst, _, _), prepared) in planes.iter_mut().zip(&prepared) {
                    let row_start = (py - pass.row_base) as usize * dst.pitch;
                    let end = (row_start + width).min(dst.data.len());
                    kernel(&mut dst.data[row_start..end], prepared, base, width, &view);
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
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[allow(
        unsafe_code,
        unused_unsafe,
        clippy::cast_ptr_alignment,
        clippy::inline_always,
        clippy::many_single_char_names,
        clippy::trivially_copy_pass_by_ref
    )]
    mod imp {
        use std::arch::x86_64::{
            __m128i, __m256i, _mm_cvtsi32_si128, _mm_loadl_epi64, _mm_loadu_si128, _mm_packs_epi32,
            _mm_packus_epi16, _mm_storel_epi64, _mm256_add_epi32, _mm256_and_si256,
            _mm256_castsi256_si128, _mm256_cvtepu8_epi32, _mm256_cvtepu16_epi32,
            _mm256_extracti128_si256, _mm256_i32gather_epi32, _mm256_loadu_si256,
            _mm256_madd_epi16, _mm256_max_epi32, _mm256_min_epi32, _mm256_mullo_epi32,
            _mm256_set1_epi32, _mm256_setr_epi32, _mm256_slli_epi32, _mm256_sra_epi32,
            _mm256_srai_epi32, _mm256_srlv_epi32, _mm256_storeu_si256, _mm256_sub_epi32,
        };
        use std::marker::PhantomData;

        use super::super::Plane;
        use super::{MAX_STEP, Setup, Weights};

        #[derive(Clone, Copy)]
        struct Source {
            ptr: *const i32,
            bytes: *const u8,
            last: __m256i,
            safe: __m256i,
            valid: bool,
            padded: bool,
            readable: usize,
            pel: i32,
        }

        impl Source {
            #[inline(always)]
            fn new(plane: &Plane<'_>, pel: i32) -> Self {
                unsafe {
                    let len = i32::try_from(plane.data.len()).unwrap_or(i32::MAX);
                    Self {
                        ptr: plane.data.as_ptr().cast::<i32>(),
                        bytes: plane.data.as_ptr(),
                        last: _mm256_set1_epi32(len - 1),
                        safe: _mm256_set1_epi32(len - 4),
                        valid: len >= 4,
                        padded: plane.slack >= 3,
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
                    let index = _mm256_min_epi32(_mm256_max_epi32(index, zero), self.last);
                    if self.padded {
                        let words = _mm256_i32gather_epi32::<1>(self.ptr, index);
                        return _mm256_and_si256(words, _mm256_set1_epi32(0xFF));
                    }
                    let base = _mm256_min_epi32(index, self.safe);
                    let shift = _mm256_slli_epi32::<3>(_mm256_sub_epi32(index, base));
                    let words = _mm256_i32gather_epi32::<1>(self.ptr, base);
                    _mm256_and_si256(_mm256_srlv_epi32(words, shift), _mm256_set1_epi32(0xFF))
                }
            }

            #[inline(always)]
            fn still(&self, start: i32, chunk: __m256i) -> __m256i {
                unsafe {
                    let start = start as usize;
                    let span = 8 * self.pel as usize;
                    if start + span.max(16) <= self.readable {
                        let at = self.bytes.add(start);
                        match self.pel {
                            1 => {
                                return _mm256_cvtepu8_epi32(_mm_loadl_epi64(at.cast::<__m128i>()));
                            }
                            2 => {
                                let words =
                                    _mm256_cvtepu16_epi32(_mm_loadu_si128(at.cast::<__m128i>()));
                                return _mm256_and_si256(words, _mm256_set1_epi32(0xFF));
                            }
                            4 => {
                                let words = _mm256_loadu_si256(at.cast::<__m256i>());
                                return _mm256_and_si256(words, _mm256_set1_epi32(0xFF));
                            }
                            _ => {}
                        }
                    }
                    self.fetch(chunk)
                }
            }
        }

        pub(in super::super) struct Prepared<'a> {
            a: Source,
            b: Source,
            offsets: __m256i,
            step: __m256i,
            pel: i32,
            pitch: __m256i,
            aux_a: Source,
            aux_b: Source,
            pub(in super::super) index: usize,
            _life: PhantomData<&'a [u8]>,
        }

        impl<'a> Prepared<'a> {
            pub(in super::super) fn new(
                setup: &Setup<'a>,
                a: &Plane<'a>,
                b: &Plane<'a>,
                aux: Option<(Plane<'a>, Plane<'a>)>,
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
                    _mm256_add_epi32(
                        _mm256_add_epi32(chunk, x),
                        _mm256_mullo_epi32(y, self.pitch),
                    )
                }
            }

            #[inline(always)]
            fn run(
                &self,
                out: &mut [u8],
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
                        let words = _mm_packs_epi32(lo, hi);
                        let bytes = _mm_packus_epi16(words, words);
                        if c + 8 <= out.len() {
                            _mm_storel_epi64(out.as_mut_ptr().add(c).cast::<__m128i>(), bytes);
                        } else {
                            let mut tmp = [0u8; 16];
                            _mm_storel_epi64(tmp.as_mut_ptr().cast::<__m128i>(), bytes);
                            let take = (n - c).min(8);
                            out[c..c + take].copy_from_slice(&tmp[..take]);
                        }
                        chunk = _mm256_add_epi32(chunk, self.step);
                        c += 8;
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

        pub(in super::super) fn fast(
            out: &mut [u8],
            p: &Prepared<'_>,
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

        pub(in super::super) fn no_mask(
            out: &mut [u8],
            p: &Prepared<'_>,
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

        pub(in super::super) fn normal(
            out: &mut [u8],
            p: &Prepared<'_>,
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
                    let inner = shr8(add(mul(m9, add(aa, mul(m8, b0))), full));
                    let first = shr8(add(add(bb, inner), full));
                    let inner = shr8(add(mul(add(bb, mul(a0, m9)), m8), full));
                    (first, shr8(add(add(aa, inner), full)))
                };
                shr8(add(mul(rt, first), mul(t, second)))
            });
        }

        pub(in super::super) fn extended(
            out: &mut [u8],
            p: &Prepared<'_>,
            base: i32,
            n: usize,
            l: &[&[i32]; 10],
            w: &Weights,
        ) {
            let (t, rt, full) = (splat(w.t), splat(w.rt), splat(255));
            p.run(out, base, n, |c, chunk, _| {
                let c0 = p.b.fetch(p.at(chunk, l[0], l[1], c));
                let n0 = p.a.fetch(p.at(chunk, l[2], l[3], c));
                let c1 = p.b.fetch(p.at(chunk, l[4], l[5], c));
                let n1 = p.a.fetch(p.at(chunk, l[6], l[7], c));
                let m9 = p.lane(l[8], c);
                let m8 = p.lane(l[9], c);
                let lo = min(c0, n0);
                let hi = max(c0, n0);
                let cn = max(lo, min(n1, hi));
                let cc = max(lo, min(c1, hi));
                let first = shr8(add(add(mul(m9, cn), mul(c0, sub(full, m9))), full));
                let second = shr8(add(add(mul(m8, cc), mul(n0, sub(full, m8))), full));
                shr8(add(mul(rt, first), mul(t, second)))
            });
        }

        pub(in super::super) fn fast_sad(
            out: &mut [u8],
            p: &Prepared<'_>,
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

        pub(in super::super) fn no_mask_sad(
            out: &mut [u8],
            p: &Prepared<'_>,
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

        pub(in super::super) fn normal_sad(
            out: &mut [u8],
            p: &Prepared<'_>,
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
                    let mixed = add(mul(weight, b0), mul(tb, a0));
                    let mixed = unsafe { _mm256_and_si256(mixed, _mm256_set1_epi32(0xFFFF)) };
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

        pub(in super::super) fn extended_sad(
            out: &mut [u8],
            p: &Prepared<'_>,
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
                let c1 = p.b.fetch(p.at(chunk, l[4], l[5], c));
                let n1 = p.a.fetch(p.at(chunk, l[6], l[7], c));
                let m9 = p.lane(l[8], c);
                let m8 = p.lane(l[9], c);
                let s = p.lane(l[10], c);
                let lo = min(c0, n0);
                let hi = max(c0, n0);
                let cn = max(lo, min(n1, hi));
                let cc = max(lo, min(c1, hi));
                let first = shr8(add(add(mul(m9, cn), mul(c0, sub(full, m9))), full));
                let second = shr8(add(add(mul(m8, cc), mul(n0, sub(full, m8))), full));
                let value = shr8(add(mul(rt, first), mul(t, second)));
                let b0 = p.b.still(start, chunk);
                let a0 = p.a.still(start, chunk);
                let mid = shr8(add(mul(rtb, b0), mul(tb, a0)));
                shr8(add(add(mul(sub(full, s), value), mul(s, mid)), full))
            });
        }
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    #[allow(
        clippy::inline_always,
        clippy::many_single_char_names,
        clippy::trivially_copy_pass_by_ref
    )]
    mod imp {
        use super::super::Plane;
        use super::{Setup, Weights};

        pub(in super::super) struct Prepared<'a> {
            a: &'a [u8],
            b: &'a [u8],
            pel: i32,
            pitch: i32,
            aux_a: &'a [u8],
            aux_b: &'a [u8],
            pub(in super::super) index: usize,
        }

        impl<'a> Prepared<'a> {
            pub(in super::super) fn new(
                setup: &Setup<'a>,
                a: &Plane<'a>,
                b: &Plane<'a>,
                aux: Option<(Plane<'a>, Plane<'a>)>,
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
            fn lane(&self, values: &[i32], c: usize) -> i32 {
                values[c]
            }

            #[inline(always)]
            fn at(&self, plane: &[u8], base: i32, x: &[i32], y: &[i32], c: usize) -> i32 {
                px(plane, base + self.pel * c as i32 + x[c] + self.pitch * y[c])
            }

            #[inline(always)]
            fn still(&self, plane: &[u8], base: i32, c: usize) -> i32 {
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
            for (c, value) in out.iter_mut().enumerate().take(cols.next_multiple_of(8)) {
                let w = weights[c];
                let sum = i32::from(pair as i16) * i32::from(w as i16) + (pair >> 16) * (w >> 16);
                *value = i32::from((sum >> shift.min(31)) as i16);
            }
        }

        #[inline(always)]
        fn px(plane: &[u8], index: i32) -> i32 {
            let last = plane.len().saturating_sub(1);
            plane
                .get((index.max(0) as usize).min(last))
                .map_or(0, |&v| i32::from(v))
        }

        pub(in super::super) fn fast(
            out: &mut [u8],
            p: &Prepared<'_>,
            next: bool,
            base: i32,
            n: usize,
            l: &[&[i32]; 2],
        ) {
            let src = if next { p.a } else { p.b };
            for (c, o) in out.iter_mut().enumerate().take(n) {
                *o = p.at(src, base, l[0], l[1], c) as u8;
            }
        }

        pub(in super::super) fn no_mask(
            out: &mut [u8],
            p: &Prepared<'_>,
            base: i32,
            n: usize,
            l: &[&[i32]; 4],
            w: &Weights,
            median: bool,
        ) {
            for (c, o) in out.iter_mut().enumerate().take(n) {
                let pb = p.at(p.b, base, l[0], l[1], c);
                let pa = p.at(p.a, base, l[2], l[3], c);
                *o = if median {
                    let mid = (w.rtb * p.still(p.b, base, c) + w.tb * p.still(p.a, base, c)) >> 8;
                    mid.min(pb.max(pa)).max(pb.min(pa)) as u8
                } else {
                    ((w.rt * pb + w.t * pa) >> 8) as u8
                };
            }
        }

        pub(in super::super) fn normal(
            out: &mut [u8],
            p: &Prepared<'_>,
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
                    (
                        (bb + ((m9 * (aa + m8 * b0) + 255) >> 8) + 255) >> 8,
                        (aa + (((bb + a0 * m9) * m8 + 255) >> 8) + 255) >> 8,
                    )
                };
                *o = ((w.rt * first + w.t * second) >> 8) as u8;
            }
        }

        pub(in super::super) fn extended(
            out: &mut [u8],
            p: &Prepared<'_>,
            base: i32,
            n: usize,
            l: &[&[i32]; 10],
            w: &Weights,
        ) {
            for (c, o) in out.iter_mut().enumerate().take(n) {
                let c0 = p.at(p.b, base, l[0], l[1], c);
                let n0 = p.at(p.a, base, l[2], l[3], c);
                let c1 = p.at(p.b, base, l[4], l[5], c);
                let n1 = p.at(p.a, base, l[6], l[7], c);
                let (m9, m8) = (p.lane(l[8], c), p.lane(l[9], c));
                let lo = c0.min(n0);
                let hi = c0.max(n0);
                let cn = lo.max(n1.min(hi));
                let cc = lo.max(c1.min(hi));
                let first = (m9 * cn + c0 * (255 - m9) + 255) >> 8;
                let second = (m8 * cc + n0 * (255 - m8) + 255) >> 8;
                *o = ((w.rt * first + w.t * second) >> 8) as u8;
            }
        }

        pub(in super::super) fn fast_sad(
            out: &mut [u8],
            p: &Prepared<'_>,
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
                *o = (((255 - s) * moved + s * still + 255) >> 8) as u8;
            }
        }

        pub(in super::super) fn no_mask_sad(
            out: &mut [u8],
            p: &Prepared<'_>,
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
                *o = (((255 - s) * value + s * mid + 255) >> 8) as u8;
            }
        }

        pub(in super::super) fn normal_sad(
            out: &mut [u8],
            p: &Prepared<'_>,
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
                        ((weight * b0 + w.tb * a0) & 0xFFFF) >> 8,
                    )
                } else {
                    let time =
                        (w.rt * p.still(p.aux_b, base, c) + w.t * p.still(p.aux_a, base, c)) >> 8;
                    (
                        time.min(first.max(second)).max(first.min(second)),
                        (w.rtb * b0 + w.tb * a0) >> 8,
                    )
                };
                *o = (((255 - s) * value + s * mid + 255) >> 8) as u8;
            }
        }

        pub(in super::super) fn extended_sad(
            out: &mut [u8],
            p: &Prepared<'_>,
            base: i32,
            n: usize,
            l: &[&[i32]; 11],
            w: &Weights,
        ) {
            for (c, o) in out.iter_mut().enumerate().take(n) {
                let c0 = p.at(p.b, base, l[0], l[1], c);
                let n0 = p.at(p.a, base, l[2], l[3], c);
                let c1 = p.at(p.b, base, l[4], l[5], c);
                let n1 = p.at(p.a, base, l[6], l[7], c);
                let (m9, m8, s) = (p.lane(l[8], c), p.lane(l[9], c), p.lane(l[10], c));
                let lo = c0.min(n0);
                let hi = c0.max(n0);
                let cn = lo.max(n1.min(hi));
                let cc = lo.max(c1.min(hi));
                let first = (m9 * cn + c0 * (255 - m9) + 255) >> 8;
                let second = (m8 * cc + n0 * (255 - m8) + 255) >> 8;
                let value = (w.rt * first + w.t * second) >> 8;
                let b0 = p.still(p.b, base, c);
                let a0 = p.still(p.a, base, c);
                let mid = (w.rtb * b0 + w.tb * a0) >> 8;
                *o = (((255 - s) * value + s * mid + 255) >> 8) as u8;
            }
        }
    }

    pub(super) use imp::{
        Prepared, extended, extended_sad, fast, fast_sad, fill, no_mask, no_mask_sad, normal,
        normal_sad,
    };

    const _: () = assert!(MAX_STEP.is_multiple_of(8));
}
