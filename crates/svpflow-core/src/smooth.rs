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

pub struct AvgLuma {
    pub geomean: f64,
    pub max: f64,
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
    pub fn avg_luma(&self, out: &mut [u8], gamma: f64) -> AvgLuma {
        let denom = if self.shape.flags == 3 { 510.0 } else { 255.0 };
        let count = self.shape.blocks();
        let mut max = 0.0f64;
        let mut log_sum = 0.0;
        for ((out, back), fwd) in out
            .iter_mut()
            .zip(&self.backward)
            .zip(&self.forward)
            .take(count)
        {
            let x = f64::from(u16::from(back.luma) + u16::from(fwd.luma)) / denom;
            let scaled = (x.powf(gamma) * 255.0) as i32;
            max = x.max(max);
            *out = if scaled < 20 { 20 } else { scaled as u8 };
            log_sum += x.ln();
        }
        AvgLuma {
            geomean: (log_sum / count as f64).exp(),
            max,
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
        self.cover.resize(len, 0);
        let area = s.block_w * s.block_h;
        let step_x = s.block_w - s.overlap_x;
        let step_y = s.block_h - s.overlap_y;
        let denom = s.pel << 8;
        let set = if forward {
            &self.forward
        } else {
            &self.backward
        };
        let buf = &mut self.cover;
        let mut add = |cx: i32, cy: i32, value: i32| {
            let base = cy as usize * stride + cx as usize;
            for row in 0..3 {
                for cell in &mut buf[base + row * stride..base + row * stride + 3] {
                    *cell = cell.wrapping_add(value);
                }
            }
        };
        for by in 0..s.grid_h {
            for bx in 0..s.grid_w {
                let vector = set[(bx + by * s.grid_w) as usize];
                let x = time * i32::from(vector.x) / denom + bx * step_x;
                let y = time * i32::from(vector.y) / denom + by * step_y;
                let cx = floor_div(x, step_x);
                let cy = floor_div(y, step_y);
                let rx = step_x * (cx + 1) - x;
                let ry = step_y * (cy + 1) - y;
                let cx1 = cx + 1;
                let cy1 = cy + 1;
                let col_in = |c: i32| c >= 0 && c < width;
                let row_in = |r: i32| r >= 0 && r < height;
                let row_ok = row_in(cy);
                if col_in(cx) && row_ok {
                    add(cx, cy, ry * rx);
                }
                if row_ok && cx1 >= 0 {
                    if cx1 < width {
                        add(cx1, cy, ry * (s.block_w - rx));
                        if row_in(cy1) {
                            add(cx1, cy1, (s.block_h - ry) * (s.block_w - rx));
                            if col_in(cx) {
                                add(cx, cy1, rx * (s.block_h - ry));
                            }
                        }
                    } else if row_in(cy1) && col_in(cx) {
                        add(cx, cy1, rx * (s.block_h - ry));
                    }
                } else {
                    let below = row_in(cy1);
                    if cx1 >= 0 && below && cx1 < width {
                        add(cx1, cy1, (s.block_h - ry) * (s.block_w - rx));
                    }
                    if col_in(cx) && below {
                        add(cx, cy1, rx * (s.block_h - ry));
                    }
                }
            }
        }
        let scale = f64::from(cover) / 100.0;
        for y in 0..height as usize {
            let row = &self.cover[(y + 1) * stride + 1..(y + 1) * stride + 1 + width as usize];
            for (out, &sum) in out[y * width as usize..].iter_mut().zip(row) {
                let covered = (sum >> 3).min(area);
                let value = (f64::from(area - covered) * scale * 256.0 / f64::from(area)) as i32;
                *out = if value > 255 { 255 } else { value as u8 };
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
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Algo {
    Fast { next: bool },
    NoMask { median: bool },
    Normal { simple: bool },
    Extended,
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

fn sra16(value: i16, shift: u32) -> i16 {
    value >> shift.min(15)
}

fn sra32(value: i32, shift: u32) -> i32 {
    value >> shift.min(31)
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
    fn pass(&self, chroma: bool, interp: bool, pitch: usize) -> Pass<'_> {
        let s = &self.shape;
        let shift = |i: usize| self.shifts[i].clamp(0, 31) as u32;
        if chroma {
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
            }
        }
    }

    pub fn fill(&self, dst: &mut FrameMut<'_>) {
        let s = &self.shape;
        let w = s.width as usize;
        let h = s.height as usize;
        for (plane, pw, ph, value) in [
            (&mut dst.y, w, h, 0x7F),
            (&mut dst.u, w / 2, h / 2, 0x80),
            (&mut dst.v, w / 2, h / 2, 0x80),
        ] {
            for row in 0..ph {
                plane.data[row * plane.pitch..row * plane.pitch + pw].fill(value);
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    pub fn render(
        &self,
        algo: Algo,
        interp: bool,
        dst: &mut FrameMut<'_>,
        next: Frame<'_>,
        current: Frame<'_>,
        vectors: &PackedVectors<'_>,
    ) {
        if algo == Algo::Fill {
            self.fill(dst);
            return;
        }
        for chroma in [false, true] {
            let pitch = if chroma { next.u.pitch } else { next.y.pitch };
            let pass = self.pass(chroma, interp, pitch);
            let planes: &mut [(&mut PlaneMut<'_>, Plane<'_>, Plane<'_>)] = if chroma {
                &mut [
                    (&mut dst.u, next.u, current.u),
                    (&mut dst.v, next.v, current.v),
                ]
            } else {
                &mut [(&mut dst.y, next.y, current.y)]
            };
            self.render_pass(algo, interp, &pass, planes, vectors);
        }
    }

    #[allow(
        clippy::too_many_lines,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation
    )]
    fn render_pass(
        &self,
        algo: Algo,
        interp: bool,
        pass: &Pass<'_>,
        planes: &mut [(&mut PlaneMut<'_>, Plane<'_>, Plane<'_>)],
        v: &PackedVectors<'_>,
    ) {
        let grid_w = self.shape.grid_w;
        let grid_h = self.shape.grid_h;
        let pel = self.shape.pel as isize;
        let pitch = pass.pitch as isize;
        let t = self.time;
        let rt = 256 - t;
        let tb = self.time_blend;
        let rtb = 256 - tb;
        let fwd = |lut: &[i16], raw: u16, div: i32| -> i16 {
            (i32::from(lut[usize::from(raw)]) / div) as i16
        };
        let mut channels: Vec<[i16; 4]> = Vec::with_capacity(10);
        let mut row_values: Vec<[i16; 2]> = Vec::with_capacity(10);
        let mut lanes = [0i32; 10];
        for by in -1..grid_h {
            let py0 = if by < 0 {
                0
            } else {
                pass.origin_y + by * pass.step_y
            };
            let rows = if py0 + pass.step_y < pass.height {
                if by == -1 { pass.origin_y } else { pass.step_y }
            } else {
                pass.height - py0
            };
            let rows = if !interp && by == grid_h - 1 {
                pass.height - py0
            } else {
                rows
            };
            let top = by.max(0);
            let bottom = if by == -1 || by == grid_h - 1 {
                top
            } else {
                top + 1
            };
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
                if rows <= 0 || cols <= 0 {
                    continue;
                }
                let left = bx.max(0);
                let right = if bx == -1 || bx == grid_w - 1 {
                    left
                } else {
                    left + 1
                };
                let idx = |row: i32, col: i32| (row * grid_w + col) as usize;
                let corners = [
                    idx(top, left),
                    idx(top, right),
                    idx(bottom, left),
                    idx(bottom, right),
                ];
                channels.clear();
                let mut push_vec = |xs: &[u16], ys: &[u16], lut: &[i16]| {
                    channels.push(corners.map(|c| fwd(lut, xs[c], pass.div_x)));
                    channels.push(corners.map(|c| fwd(lut, ys[c], pass.div_y)));
                };
                match algo {
                    Algo::Fast { next: true } => push_vec(v.fwd_x, v.fwd_y, &self.lut_fwd),
                    Algo::Fast { next: false } => push_vec(v.bwd_x, v.bwd_y, &self.lut_fwd),
                    Algo::NoMask { .. } | Algo::Normal { .. } => {
                        push_vec(v.bwd_x, v.bwd_y, &self.lut_fwd);
                        push_vec(v.fwd_x, v.fwd_y, &self.lut_rev);
                    }
                    Algo::Extended => {
                        push_vec(v.bwd_x, v.bwd_y, &self.lut_fwd);
                        push_vec(v.fwd_x, v.fwd_y, &self.lut_rev);
                        push_vec(v.prev_bwd_x, v.prev_bwd_y, &self.lut_fwd);
                        push_vec(v.next_fwd_x, v.next_fwd_y, &self.lut_rev);
                    }
                    Algo::Fill => {}
                }
                if matches!(algo, Algo::Normal { .. } | Algo::Extended) {
                    channels.push(corners.map(|c| i16::from(v.cover_fwd[c])));
                    channels.push(corners.map(|c| i16::from(v.cover_bwd[c])));
                }
                for r in 0..rows {
                    let py = py0 + r;
                    let wt = pass.y_weights[(pass.step_y - r) as usize];
                    let wb = pass.y_weights[r as usize];
                    row_values.clear();
                    for ch in &channels {
                        let mix = |a: i16, b: i16| {
                            sra16(
                                wt.wrapping_mul(a).wrapping_add(wb.wrapping_mul(b)),
                                pass.shift_y,
                            )
                        };
                        row_values.push([mix(ch[0], ch[2]), mix(ch[1], ch[3])]);
                    }
                    for c in 0..cols {
                        let px = px0 + c;
                        if interp {
                            let [wl, wr] = pass.x_weights[c as usize];
                            for (lane, rv) in lanes.iter_mut().zip(&row_values) {
                                let sum = i32::from(rv[0]) * i32::from(wl)
                                    + i32::from(rv[1]) * i32::from(wr);
                                *lane = i32::from(sra32(sum, pass.shift_x) as i16);
                            }
                        } else {
                            for (lane, ch) in lanes.iter_mut().zip(&channels) {
                                *lane = i32::from(ch[0]);
                            }
                        }
                        let base = pel * (px as isize + py as isize * pitch);
                        let at = |dx: i32, dy: i32| base + dx as isize + pitch * dy as isize;
                        for (dst, a, b) in planes.iter_mut() {
                            let get = |p: &Plane<'_>, i: isize| -> i32 {
                                usize::try_from(i)
                                    .ok()
                                    .and_then(|i| p.data.get(i))
                                    .map_or(0, |&x| i32::from(x))
                            };
                            let value = match algo {
                                Algo::Fast { next } => {
                                    let src = if next { *a } else { *b };
                                    get(&src, at(lanes[0], lanes[1]))
                                }
                                Algo::NoMask { median: false } => {
                                    let pb = get(b, at(lanes[0], lanes[1]));
                                    let pa = get(a, at(lanes[2], lanes[3]));
                                    i32::from(((rt * pb + t * pa) as u16) >> 8)
                                }
                                Algo::NoMask { median: true } => {
                                    let pb = get(b, at(lanes[0], lanes[1]));
                                    let pa = get(a, at(lanes[2], lanes[3]));
                                    let lo = pb.min(pa);
                                    let hi = pb.max(pa);
                                    let mid = (rtb * get(b, base) + tb * get(a, base)) >> 8;
                                    mid.min(hi).max(lo)
                                }
                                Algo::Normal { simple } => {
                                    let pb = get(b, at(lanes[0], lanes[1]));
                                    let pa = get(a, at(lanes[2], lanes[3]));
                                    let m9 = lanes[4];
                                    let m8 = lanes[5];
                                    if simple {
                                        let first = ((255 - m9) * pb + m9 * pa + 255) >> 8;
                                        let second = ((255 - m8) * pa + m8 * pb + 255) >> 8;
                                        i32::from(((rt * first + t * second) as u16) >> 8)
                                    } else {
                                        let bb = (255 - m9) * pb;
                                        let aa = (255 - m8) * pa;
                                        let first = (bb
                                            + ((m9 * (aa + m8 * get(b, base)) + 255) >> 8)
                                            + 255)
                                            >> 8;
                                        let second = (aa
                                            + (((bb + get(a, base) * m9) * m8 + 255) >> 8)
                                            + 255)
                                            >> 8;
                                        i32::from(((rt * first + t * second) as u16) >> 8)
                                    }
                                }
                                Algo::Extended => {
                                    let c0 = get(b, at(lanes[0], lanes[1]));
                                    let n0 = get(a, at(lanes[2], lanes[3]));
                                    let c1 = get(b, at(lanes[4], lanes[5]));
                                    let n1 = get(a, at(lanes[6], lanes[7]));
                                    let m9 = lanes[8];
                                    let m8 = lanes[9];
                                    let lo = c0.min(n0);
                                    let hi = c0.max(n0);
                                    let clamp = |x: i32| if x > lo { x.min(hi) } else { lo };
                                    let first = (m9 * clamp(n1) + c0 * (255 - m9) + 255) >> 8;
                                    let second = (m8 * clamp(c1) + n0 * (255 - m8) + 255) >> 8;
                                    i32::from(((rt * first + t * second) as u16) >> 8)
                                }
                                Algo::Fill => 0,
                            };
                            let di = py as usize * dst.pitch + px as usize;
                            dst.data[di] = value as u8;
                        }
                    }
                }
            }
        }
    }
}
