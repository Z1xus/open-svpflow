use super::metric::{self, Kernel, Shape};
use super::planes::{LevelFrame, MAX_BLOCK_AREA};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Mv {
    pub(crate) x: i16,
    pub(crate) y: i16,
    pub(crate) sad: i32,
}

impl Mv {
    pub(crate) const UNSET: Self = Self {
        x: 0,
        y: 0,
        sad: -1,
    };

    const fn new(x: i32, y: i32, sad: i32) -> Self {
        Self {
            x: x as i16,
            y: y as i16,
            sad,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum SearchType {
    Hex2,
    Umh,
    Exhaustive,
}

impl SearchType {
    pub(crate) fn from_option(value: i64) -> Self {
        match value {
            2 => Self::Hex2,
            3 => Self::Umh,
            _ => Self::Exhaustive,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct BlockLayout {
    pub(crate) width: i32,
    pub(crate) height: i32,
    pub(crate) overlap_x: i32,
    pub(crate) overlap_y: i32,
}

impl BlockLayout {
    pub(crate) const fn step_x(&self) -> i32 {
        self.width - self.overlap_x
    }

    pub(crate) const fn step_y(&self) -> i32 {
        self.height - self.overlap_y
    }
}

#[derive(Clone, Copy)]
pub(crate) struct SearchSettings {
    pub(crate) search_type: SearchType,
    pub(crate) search_param: i32,
    pub(crate) lambda: i32,
    pub(crate) lsad: i32,
    pub(crate) pnew: i32,
    pub(crate) plevel: f64,
    pub(crate) pzero: i32,
    pub(crate) pglobal: i32,
    pub(crate) pnbour: i32,
    pub(crate) preverse: i32,
    pub(crate) bad_sad: i32,
    pub(crate) bad_range: bool,
    pub(crate) try_many: bool,
    pub(crate) flags: i32,
}

pub(crate) type VectorOutput<'o> = &'o mut [u32];

const CHROMA_MUL: i32 = 1;
const CHROMA_DIV: i32 = 3;
const INT_MAX_OVER_100: f64 = i32::MAX as f64 / 100.0;
const CONTRAST_FLAG: i32 = 2;
const NO_CONTRAST_BLUR_FLAG: i32 = 0x10;

#[derive(Clone, Copy)]
struct Kernels {
    luma: Kernel,
    chroma: Kernel,
}

pub(crate) struct Field {
    pub(crate) blk_x: i32,
    pub(crate) blk_y: i32,
    layout: BlockLayout,
    pel: i32,
    level: i32,
    levels_count: i32,
    smallest: bool,
    chroma_shift: (u32, u32),
    full: Kernels,
    half: Kernels,
    pub(crate) vectors: Vec<Mv>,
    pub(crate) reverse: Vec<Mv>,
    border: Vec<Mv>,
    order: Vec<(i16, i16)>,
}

impl Field {
    #[allow(clippy::fn_params_excessive_bools)]
    pub(crate) fn new(
        blk_x: i32,
        blk_y: i32,
        layout: BlockLayout,
        pel: i32,
        level: i32,
        levels_count: i32,
        smallest: bool,
        satd: bool,
        chroma_shift: (u32, u32),
    ) -> Self {
        let count = (blk_x * blk_y) as usize;
        let extended = ((blk_x + 1) * (blk_y + 1)) as usize;
        let shape = Shape::new(layout.width as usize, layout.height as usize);
        let chroma = if chroma_shift == (0, 0) {
            metric::luma_kernel(shape, satd)
        } else {
            metric::chroma420_kernel(shape, satd)
        };
        Self {
            blk_x,
            blk_y,
            layout,
            pel,
            level,
            levels_count,
            smallest,
            chroma_shift,
            full: Kernels {
                luma: metric::luma_kernel(shape, satd),
                chroma,
            },
            half: Kernels {
                luma: metric::chroma420_kernel(shape, satd),
                chroma: metric::null_kernel,
            },
            vectors: vec![Mv::default(); count],
            reverse: vec![Mv::default(); count],
            border: vec![Mv::default(); (blk_x + blk_y + 1) as usize],
            order: vec![(0, 0); if level > 0 { extended } else { count }],
        }
    }

    fn count(&self) -> usize {
        self.vectors.len()
    }

    pub(crate) fn estimate_global_doubled(&self, global: &mut Mv) {
        let freq_size = 8192 * self.pel * 2;
        let mode = |component: fn(&Mv) -> i16| {
            let mut freq = vec![0i32; freq_size as usize];
            let (mut low, mut high) = (freq_size - 1, 0);
            for v in &self.vectors {
                let index = (freq_size >> 1) + i32::from(component(v));
                if (0..freq_size).contains(&index) {
                    freq[index as usize] += 1;
                    high = high.max(index);
                    low = low.min(index);
                }
            }
            let mut best = low;
            for index in low + 1..=high {
                if freq[index as usize] > freq[best as usize] {
                    best = index;
                }
            }
            best - (freq_size >> 1)
        };
        let median_x = mode(|v| v.x);
        let median_y = mode(|v| v.y);
        let (mut sum_x, mut sum_y, mut count) = (0, 0, 0);
        for v in &self.vectors {
            let (x, y) = (i32::from(v.x), i32::from(v.y));
            if (x - median_x).abs() < 6 && (y - median_y).abs() < 6 {
                sum_x += x;
                sum_y += y;
                count += 1;
            }
        }
        let (x, y) = if count > 0 {
            (2 * sum_x / count, 2 * sum_y / count)
        } else {
            (2 * median_x, 2 * median_y)
        };
        global.x = x as i16;
        global.y = y as i16;
    }

    pub(crate) fn sort_blocks(&mut self) {
        let area = self.layout.width * self.layout.height;
        let shift = ilog2(area * 256 / 2048) + 1;
        let buckets = ((area * 255) >> ilog2(area * 256 / 2048)) as usize;
        let mut keyed: Vec<(usize, (i16, i16))> = Vec::with_capacity(self.count());
        for by in 0..self.blk_y {
            for bx in 0..self.blk_x {
                let sad = self.vectors[(by * self.blk_x + bx) as usize].sad;
                let bucket = ((sad >> shift) as usize).min(buckets - 1);
                keyed.push((bucket, (bx as i16, by as i16)));
            }
        }
        keyed.sort_by_key(|&(bucket, _)| bucket);
        for (slot, (_, block)) in self.order.iter_mut().zip(keyed) {
            *slot = block;
        }
    }

    pub(crate) fn set_order(&mut self, coarse: Option<&Self>) {
        let (nx, ny) = (self.blk_x, self.blk_y);
        let mut order = Vec::with_capacity(self.order.len());
        match coarse {
            None => {
                for by in 1..ny - 1 {
                    let (start, dir) = if by % 2 == 0 { (0, 1) } else { (nx - 1, -1) };
                    order.extend((1..nx - 1).map(|i| (start + i * dir, by)));
                }
                for bx in 0..nx {
                    order.push((bx, 0));
                    if ny > 1 {
                        order.push((bx, ny - 1));
                    }
                }
                for by in 1..ny - 1 {
                    order.push((0, by));
                    if nx > 1 {
                        order.push((nx - 1, by));
                    }
                }
            }
            Some(coarse) => self.children_order(coarse, &mut order),
        }
        if self.level > 0 {
            order.extend((0..=nx).map(|bx| (bx, ny)));
            order.extend((0..ny).map(|by| (nx, by)));
        }
        self.order.fill((0, 0));
        for (slot, (x, y)) in self.order.iter_mut().zip(order) {
            *slot = (x as i16, y as i16);
        }
    }

    fn children_order(&self, coarse: &Self, order: &mut Vec<(i32, i32)>) {
        let (nx, ny) = (self.blk_x, self.blk_y);
        let add_x = nx - coarse.blk_x * 2;
        let add_y = ny - coarse.blk_y * 2;
        let ring = (nx * 2 + ny * 2 - 4).max(0) as usize;
        let mut border = Vec::with_capacity(ring);
        let mut push = |interior: bool, x: i32, y: i32| {
            if interior {
                order.push((x, y));
            } else {
                border.push((x, y));
            }
        };
        for &(cx, cy) in &coarse.order {
            let (cx, cy) = (i32::from(cx), i32::from(cy));
            if cx == coarse.blk_x || cy == coarse.blk_y {
                continue;
            }
            let (x, y) = (cx * 2, cy * 2);
            let on_edge = x == 0 || y == 0 || cx == coarse.blk_x - 1 || cy == coarse.blk_y - 1;
            if !on_edge {
                for (dx, dy) in [(0, 0), (1, 0), (1, 1), (0, 1)] {
                    push(true, x + dx, y + dy);
                }
                continue;
            }
            push(x > 0 && y > 0, x, y);
            push(x + 1 < nx - 1 && y > 0, x + 1, y);
            push(x + 1 < nx - 1 && y + 1 < ny - 1, x + 1, y + 1);
            push(x > 0 && y + 1 < ny - 1, x, y + 1);
            let last_x = add_x != 0 && cx == coarse.blk_x - 1;
            let last_y = add_y != 0 && cy == coarse.blk_y - 1;
            if last_x {
                for i in 0..add_x {
                    push(i < add_x - 1 && y > 0, x + 2 + i, y);
                    push(i < add_x - 1 && y + 1 < ny - 1, x + 2 + i, y + 1);
                    if last_y {
                        for k in 0..add_y {
                            push(i < add_x - 1 && k < add_y - 1, x + 2 + i, y + 2 + k);
                        }
                    }
                }
            }
            if last_y {
                for k in 0..add_y {
                    push(x > 0 && k < add_y - 1, x, y + 2 + k);
                    push(x + 1 < nx - 1 && k < add_y - 1, x + 1, y + 2 + k);
                }
            }
        }
        border.resize(ring, (0, 0));
        order.extend(border);
    }

    pub(crate) fn interpolate_prediction(&mut self, coarse: &Self, for_recalc: bool) {
        let norm = if for_recalc { 4 } else { 3 } - ilog2(self.pel) + ilog2(coarse.pel);
        if !for_recalc {
            self.reverse.fill(Mv::UNSET);
        }
        let l = self.layout;
        let norm_area = l.step_x() * l.step_y();
        let (odd_x, even_x) = (l.width * 3 - l.overlap_x * 2, l.width * 3 - l.overlap_x * 4);
        let (odd_y, even_y) = (
            l.height * 3 - l.overlap_y * 2,
            l.height * 3 - l.overlap_y * 4,
        );
        let (cx, cy) = (coarse.blk_x, coarse.blk_y);
        let at = |i: i32, j: i32| coarse.vectors[(i + j * cx) as usize];
        for l_y in 0..self.blk_y {
            for k_x in 0..self.blk_x {
                let index = (l_y * self.blk_x + k_x) as usize;
                let (mut i, mut j) = (k_x, l_y);
                if for_recalc {
                    i = i.min(2 * cx - 1);
                    j = j.min(2 * cy - 1);
                } else if i >= 2 * cx || j >= 2 * cy {
                    let mut v = coarse.border[((i / 2).min(cx) + cy - (j / 2).min(cy)) as usize];
                    v.x = (i32::from(v.x) << (4 - norm)) as i16;
                    v.y = (i32::from(v.y) << (4 - norm)) as i16;
                    self.vectors[index] = v;
                    continue;
                }
                let off_x = -1 + 2 * (i % 2);
                let off_y = -1 + 2 * (j % 2);
                let edge_x = i == 0 || i >= 2 * cx - 1;
                let edge_y = j == 0 || j >= 2 * cy - 1;
                let v1 = at(i / 2, j / 2);
                let (v2, v3, v4) = if !for_recalc {
                    (v1, v1, v1)
                } else if edge_x && edge_y {
                    (v1, v1, v1)
                } else if edge_x {
                    let v = at(i / 2, j / 2 + off_y);
                    (v1, v, v)
                } else if edge_y {
                    let v = at(i / 2 + off_x, j / 2);
                    (v1, v, v)
                } else {
                    (
                        at(i / 2 + off_x, j / 2),
                        at(i / 2, j / 2 + off_y),
                        at(i / 2 + off_x, j / 2 + off_y),
                    )
                };
                let (mut tx, mut ty, mut sad);
                if l.overlap_x == 0 && l.overlap_y == 0 {
                    let blend = |a: i32, b: i32, c: i32, d: i32| 9 * a + 3 * b + 3 * c + d;
                    tx = blend(v1.x.into(), v2.x.into(), v3.x.into(), v4.x.into());
                    ty = blend(v1.y.into(), v2.y.into(), v3.y.into(), v4.y.into());
                    sad = blend(v1.sad, v2.sad, v3.sad, v4.sad) + 8;
                } else {
                    let ax1 = if off_x > 0 { odd_x } else { even_x };
                    let ax2 = l.step_x() * 4 - ax1;
                    let ay1 = if off_y > 0 { odd_y } else { even_y };
                    let ay2 = l.step_y() * 4 - ay1;
                    let (a11, a12, a21, a22) = (ax1 * ay1, ax1 * ay2, ax2 * ay1, ax2 * ay2);
                    let blend = |a: i32, b: i32, c: i32, d: i32| {
                        (a11 * a + a21 * b + a12 * c + a22 * d) / norm_area
                    };
                    tx = blend(v1.x.into(), v2.x.into(), v3.x.into(), v4.x.into());
                    ty = blend(v1.y.into(), v2.y.into(), v3.y.into(), v4.y.into());
                    sad = blend(v1.sad, v2.sad, v3.sad, v4.sad);
                }
                tx = i32::from(tx as i16) >> norm;
                ty = i32::from(ty as i16) >> norm;
                sad >>= if for_recalc { 6 } else { 4 };
                let target = Mv::new(tx, ty, sad);
                self.vectors[index] = target;
                if !for_recalc {
                    let rbx = ((i * l.step_x() + (l.width >> 1) + tx) / l.step_x())
                        .clamp(0, self.blk_x - 1);
                    let rby = ((j * l.step_y() + (l.height >> 1) + ty) / l.step_y())
                        .clamp(0, self.blk_y - 1);
                    let reverse = &mut self.reverse[(rby * self.blk_x + rbx) as usize];
                    if reverse.sad < 0 || sad < reverse.sad {
                        *reverse = Mv::new(-tx, -ty, sad);
                    }
                }
            }
        }
    }

    pub(crate) fn search(
        &mut self,
        src: &LevelFrame<'_>,
        reference: &LevelFrame<'_>,
        settings: &SearchSettings,
        global: Mv,
        lambda_out: Option<&mut i32>,
        output: Option<VectorOutput<'_>>,
    ) {
        let mut settings = *settings;
        let levels_ok = |mask: i32, shift: i32| {
            let min = match (settings.flags & mask) >> shift {
                0 => 3,
                m => m,
            };
            self.level >= min
        };
        settings.try_many = settings.try_many && levels_ok(0x60, 5);
        settings.bad_range = settings.bad_range && levels_ok(0x0C, 2);
        if self.smallest {
            self.vectors.fill(Mv::default());
            self.set_order(None);
            settings.pnew = 0;
            settings.lambda = 0;
        }
        let global = Mv::new(
            self.pel * i32::from(global.x),
            self.pel * i32::from(global.y),
            global.sad,
        );
        self.run(true, src, reference, &settings, global, lambda_out, output);
    }

    pub(crate) fn recalculate(
        &mut self,
        src: &LevelFrame<'_>,
        reference: &LevelFrame<'_>,
        settings: &SearchSettings,
        lambda_out: Option<&mut i32>,
        output: Option<VectorOutput<'_>>,
    ) {
        self.run(
            false,
            src,
            reference,
            settings,
            Mv::default(),
            lambda_out,
            output,
        );
    }

    #[allow(clippy::too_many_lines)]
    fn run(
        &mut self,
        searching: bool,
        src: &LevelFrame<'_>,
        reference: &LevelFrame<'_>,
        settings: &SearchSettings,
        global: Mv,
        lambda_out: Option<&mut i32>,
        mut output: Option<VectorOutput<'_>>,
    ) {
        let stp = settings.search_param;
        let search_param = if stp != 0 {
            stp.abs()
        } else if self.level != 0 {
            10
        } else {
            0
        };
        let lambda = if searching {
            let scale = settings.plevel.powi(self.levels_count - self.level - 1);
            let scaled = f64::from(settings.lambda) / scale / f64::from(self.pel * self.pel);
            scaled.min(INT_MAX_OVER_100) as i32
        } else {
            settings.lambda >> 2
        };
        if let Some(out) = lambda_out {
            *out = lambda;
        }
        let border_negative = self.smallest && self.layout.width > 4;
        let (w, h) = (src.y.width, src.y.height);
        let diagonal = self.pel * f64::from(w * w + h * h).sqrt() as i32;
        let mut ctx = BlockSearch {
            src: *src,
            reference: *reference,
            pel: self.pel,
            level: self.level,
            chroma_shift: self.chroma_shift,
            settings: *settings,
            adaptive_radius: search_param > 0 && stp <= 0,
            global,
            kernels: self.full,
            half: false,
            shape: Shape::new(self.layout.width as usize, self.layout.height as usize),
            state: BlockState::default(),
            buffers: Default::default(),
            active: 0,
            block_luma: 0,
            scratch: [0; MAX_BLOCK_AREA],
            flat: [0; MAX_BLOCK_AREA],
            blurred: [0; MAX_BLOCK_AREA],
        };
        let mut processed = vec![false; ((self.blk_x + 1) * (self.blk_y + 1)) as usize];
        let l = self.layout;
        for position in 0..self.order.len() {
            let (bx, by) = self.order[position];
            let (bx, by) = (i32::from(bx), i32::from(by));
            let is_border = bx == self.blk_x || by == self.blk_y;
            let index = (!is_border).then(|| (by * self.blk_x + bx) as usize);
            let negative = border_negative
                && (bx == 0 || bx >= self.blk_x - 1 || by == 0 || by >= self.blk_y - 1);
            let mut x0 = if bx < self.blk_x {
                bx * l.step_x()
            } else {
                w - l.width
            };
            let mut y0 = if by < self.blk_y {
                by * l.step_y()
            } else {
                h - l.height
            };
            if negative {
                x0 += l.width >> 2;
                y0 += l.height >> 2;
            }
            let covered_w = if negative { l.width >> 1 } else { l.width };
            let covered_h = if negative { l.height >> 1 } else { l.height };
            let bounds = Bounds {
                min_x: -self.pel * x0,
                min_y: -self.pel * y0,
                max_x: self.pel * (w - x0 - covered_w),
                max_y: self.pel * (h - y0 - covered_h),
            };
            let predictor_index = index.unwrap_or_else(|| {
                let px = if bx == self.blk_x { bx - 1 } else { bx };
                let py = if by == self.blk_y { by - 1 } else { by };
                (px + py * self.blk_x) as usize
            });
            let mut state = BlockState {
                index,
                origin: [x0, x0 >> ctx.chroma_shift.0, x0 >> ctx.chroma_shift.0],
                origin_y: [y0, y0 >> ctx.chroma_shift.1, y0 >> ctx.chroma_shift.1],
                pos: [0; 3],
                pos_y: [0; 3],
                bounds0: bounds,
                bounds,
                best: Mv::default(),
                min_cost: 0,
                predictor: self.vectors[predictor_index],
                lambda,
                sad_limit: settings.lsad,
                bad_sad: settings.bad_sad,
                search_param,
            };
            state.pos = state.origin;
            state.pos_y = state.origin_y;
            if border_negative && negative {
                state.lambda = state.lambda * CHROMA_MUL / CHROMA_DIV;
                state.bad_sad = state.bad_sad * CHROMA_MUL / CHROMA_DIV;
                state.sad_limit = state.sad_limit * CHROMA_MUL / CHROMA_DIV;
            }
            ctx.set_half(negative, self.full, self.half, self.layout);
            ctx.state = state;
            if searching {
                let (neighbours, count) = self.neighbours(bx, by, &processed);
                let reverse = index.map_or(Mv::UNSET, |i| self.reverse[i]);
                ctx.search_block(negative, reverse, &neighbours[..count]);
            } else {
                ctx.recalculate_block(output.is_some());
            }
            processed[(by * (self.blk_x + 1) + bx) as usize] = true;
            let mut best = ctx.state.best;
            if negative {
                best.sad = best.sad * CHROMA_DIV / CHROMA_MUL;
            }
            match index {
                Some(i) => {
                    self.vectors[i] = best;
                    if let Some(out) = output.as_deref_mut() {
                        out[2 * i] = ((i32::from(best.x) << 16) + i32::from(best.y)) as u32;
                        out[2 * i + 1] = packed_score(best, diagonal, ctx.block_luma);
                    }
                }
                None => self.border[(bx + self.blk_y - by) as usize] = best,
            }
        }
    }

    fn neighbours(&self, bx: i32, by: i32, processed: &[bool]) -> ([Mv; 9], usize) {
        let x1 = (bx - 1).max(0);
        let x2 = if bx < self.blk_x {
            (self.blk_x - 1).min(bx + 1)
        } else {
            bx - 1
        };
        let y1 = (by - 1).max(0);
        let y2 = if by < self.blk_y {
            (self.blk_y - 1).min(by + 1)
        } else {
            by - 1
        };
        let mut found = [Mv::default(); 9];
        let mut count = 0;
        for y in y1..=y2 {
            for x in x1..=x2 {
                let index = y * self.blk_x + x;
                if processed[(index + y) as usize] {
                    found[count] = self.vectors[index as usize];
                    count += 1;
                }
            }
        }
        (found, count)
    }
}

fn packed_score(best: Mv, diagonal: i32, block_luma: i32) -> u32 {
    let mut sad = best.sad;
    let relative = (i32::from(best.x).abs() + i32::from(best.y).abs()) * 100 / diagonal;
    if relative > 50 {
        sad = sad.wrapping_mul(20);
    } else if relative > 15 {
        sad = sad.wrapping_add(best.sad.wrapping_mul(relative - 15) / 35 * 9);
    }
    ((sad & 0x00FF_FFFF) + (block_luma << 24)) as u32
}

fn ilog2(mut value: i32) -> i32 {
    let mut result = 0;
    while value > 1 {
        value /= 2;
        result += 1;
    }
    result
}

#[derive(Clone, Copy, Default)]
struct Bounds {
    min_x: i32,
    min_y: i32,
    max_x: i32,
    max_y: i32,
}

impl Bounds {
    const fn contains(&self, x: i32, y: i32) -> bool {
        !(x < self.min_x || y < self.min_y || x >= self.max_x || y >= self.max_y)
    }
}

#[derive(Clone, Copy, Default)]
struct BlockState {
    index: Option<usize>,
    origin: [i32; 3],
    origin_y: [i32; 3],
    pos: [i32; 3],
    pos_y: [i32; 3],
    bounds0: Bounds,
    bounds: Bounds,
    best: Mv,
    min_cost: i32,
    predictor: Mv,
    lambda: i32,
    sad_limit: i32,
    bad_sad: i32,
    search_param: i32,
}

#[derive(Clone)]
struct SourceBlock {
    y: [u8; MAX_BLOCK_AREA],
    u: [u8; MAX_BLOCK_AREA],
    v: [u8; MAX_BLOCK_AREA],
}

impl Default for SourceBlock {
    fn default() -> Self {
        Self {
            y: [0; MAX_BLOCK_AREA],
            u: [0; MAX_BLOCK_AREA],
            v: [0; MAX_BLOCK_AREA],
        }
    }
}

struct BlockSearch<'a> {
    src: LevelFrame<'a>,
    reference: LevelFrame<'a>,
    pel: i32,
    level: i32,
    chroma_shift: (u32, u32),
    settings: SearchSettings,
    adaptive_radius: bool,
    global: Mv,
    kernels: Kernels,
    half: bool,
    shape: Shape,
    state: BlockState,
    buffers: [SourceBlock; 2],
    active: usize,
    block_luma: i32,
    scratch: [u8; MAX_BLOCK_AREA],
    flat: [u8; MAX_BLOCK_AREA],
    blurred: [u8; MAX_BLOCK_AREA],
}

const HEX2: [(i32, i32); 8] = [
    (-1, -2),
    (-2, 0),
    (-1, 2),
    (1, 2),
    (2, 0),
    (1, -2),
    (-1, -2),
    (-2, 0),
];
const MOD6_MINUS1: [i32; 8] = [5, 0, 1, 2, 3, 4, 5, 0];
const HEX4: [(i32, i32); 16] = [
    (-4, 2),
    (-4, 1),
    (-4, 0),
    (-4, -1),
    (-4, -2),
    (4, -2),
    (4, -1),
    (4, 0),
    (4, 1),
    (4, 2),
    (2, 3),
    (0, 4),
    (-2, 3),
    (-2, -3),
    (0, -4),
    (2, -3),
];
const MAX_CANDIDATES: usize = 4 + 9;

fn cmin(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

fn truncate(value: f64) -> i32 {
    if value.is_nan() || value >= 2_147_483_648.0 || value < -2_147_483_648.0 {
        i32::MIN
    } else {
        value as i32
    }
}

impl BlockSearch<'_> {
    fn set_half(&mut self, half: bool, full: Kernels, half_kernels: Kernels, layout: BlockLayout) {
        let shape = Shape::new(layout.width as usize, layout.height as usize);
        self.half = half;
        (self.kernels, self.shape) = if half {
            (half_kernels, shape.half())
        } else {
            (full, shape)
        };
    }

    fn chroma_shape(&self) -> Shape {
        Shape::new(
            self.shape.width >> self.chroma_shift.0,
            self.shape.height >> self.chroma_shift.1,
        )
    }

    fn copy_luma(&mut self, set: usize) {
        let offset = self.src.y.pel1(self.state.pos[0], self.state.pos_y[0]);
        self.src
            .y
            .copy_block(offset, self.shape, &mut self.buffers[set].y);
    }

    fn copy_chroma(&mut self, set: usize) {
        if self.half {
            return;
        }
        let shape = self.chroma_shape();
        let (s, buffer) = (self.state, &mut self.buffers[set]);
        self.src
            .u
            .copy_block(self.src.u.pel1(s.pos[1], s.pos_y[1]), shape, &mut buffer.u);
        self.src
            .v
            .copy_block(self.src.v.pel1(s.pos[2], s.pos_y[2]), shape, &mut buffer.v);
    }

    fn copy_source(&mut self, set: usize) {
        self.copy_luma(set);
        self.copy_chroma(set);
        self.active = set;
    }

    fn move_block(&mut self, mut vx: i32, mut vy: i32, margin: i32) -> bool {
        if margin > 0 {
            vx = if vx < 0 { vx - margin } else { vx + margin };
            vy = if vy < 0 { vy - margin } else { vy + margin };
        }
        let b0 = self.state.bounds0;
        let outside = |v: i32, min: i32, max: i32| {
            if v < min {
                v - min
            } else if v >= max {
                v - max + 1
            } else {
                0
            }
        };
        let dx = outside(vx, b0.min_x, b0.max_x);
        let dy = outside(vy, b0.min_y, b0.max_y);
        let s = &mut self.state;
        if dx == 0 && dy == 0 {
            self.active = 0;
            s.pos = s.origin;
            s.pos_y = s.origin_y;
            return false;
        }
        let x = (s.origin[0] - dx)
            .max(0)
            .min(self.src.y.width - self.shape.width as i32);
        let y = (s.origin_y[0] - dy)
            .max(0)
            .min(self.src.y.height - self.shape.height as i32);
        let (dx, dy) = (s.origin[0] - x, s.origin_y[0] - y);
        let (cdx, cdy) = (dx >> self.chroma_shift.0, dy >> self.chroma_shift.1);
        s.pos = [x, s.origin[1] - cdx, s.origin[2] - cdx];
        s.pos_y = [y, s.origin_y[1] - cdy, s.origin_y[2] - cdy];
        s.bounds.max_x = b0.max_x - dx.max(0);
        s.bounds.max_y = b0.max_y - dy.max(0);
        s.bounds.min_x = b0.min_x - dx.min(0);
        s.bounds.min_y = b0.min_y - dy.min(0);
        self.copy_source(1);
        true
    }

    fn luma_sad(&mut self, vx: i32, vy: i32) -> i32 {
        let plane = self.reference.y;
        let offset = plane.subpel(
            self.state.pos[0] * self.pel + vx,
            self.state.pos_y[0] * self.pel + vy,
        );
        let (block, pitch) = plane.block(offset, self.shape, &mut self.scratch);
        (self.kernels.luma)(&self.buffers[self.active].y, block, pitch) as i32
    }

    fn chroma_sad(&mut self, vx: i32, vy: i32) -> i32 {
        if self.half {
            return 0;
        }
        let shape = self.chroma_shape();
        let cx = self.state.pos[1] * self.pel + (vx >> self.chroma_shift.0);
        let cy = self.state.pos_y[1] * self.pel + (vy >> self.chroma_shift.1);
        let (u, v) = (self.reference.u, self.reference.v);
        let offset = u.subpel(cx, cy);
        let buffer = &self.buffers[self.active];
        let (block, pitch) = u.block(offset, shape, &mut self.scratch);
        let mut sum = (self.kernels.chroma)(&buffer.u, block, pitch);
        let (block, pitch) = v.block(offset, shape, &mut self.scratch);
        sum += (self.kernels.chroma)(&buffer.v, block, pitch);
        (sum as i32) << (self.chroma_shift.0 + self.chroma_shift.1)
    }

    fn full_sad(&mut self, vx: i32, vy: i32) -> i32 {
        self.luma_sad(vx, vy) + self.chroma_sad(vx, vy)
    }

    fn motion_cost(&self, vx: i32, vy: i32) -> i32 {
        let s = &self.state;
        let dx = i32::from(s.predictor.x) - vx;
        let dy = i32::from(s.predictor.y) - vy;
        s.lambda
            .wrapping_mul(dx.wrapping_mul(dx).wrapping_add(dy.wrapping_mul(dy)))
            >> 8
    }

    fn new_penalty(&self, sad: i32) -> i32 {
        sad.wrapping_add(self.settings.pnew.wrapping_mul(sad) >> 8)
    }

    fn try_candidate(&mut self, vx: i32, vy: i32) -> bool {
        self.try_candidate_at(vx, vy, vx)
    }

    fn try_candidate_at(&mut self, vx: i32, vy: i32, luma_x: i32) -> bool {
        let mut cost = self.motion_cost(vx, vy);
        if cost >= self.state.min_cost {
            return false;
        }
        let sad = self.luma_sad(luma_x, vy);
        cost = cost.wrapping_add(self.new_penalty(sad));
        if cost >= self.state.min_cost {
            return false;
        }
        let sad_uv = self.chroma_sad(vx, vy);
        cost = cost.wrapping_add(self.new_penalty(sad_uv));
        if cost >= self.state.min_cost {
            return false;
        }
        self.state.best = Mv::new(vx, vy, sad + sad_uv);
        self.state.min_cost = cost;
        true
    }

    fn check_mv(&mut self, vx: i32, vy: i32) -> bool {
        self.state.bounds.contains(vx, vy) && self.try_candidate(vx, vy)
    }

    fn exhaustive(&mut self, radius: i32) {
        let (b, best) = (self.state.bounds, self.state.best);
        let (mx, my) = (i32::from(best.x), i32::from(best.y));
        let (min_x, max_x) = ((mx - radius).max(b.min_x), (mx + radius).min(b.max_x - 1));
        for vy in (my - radius).max(b.min_y)..=(my + radius).min(b.max_y - 1) {
            let mut luma_x = min_x;
            for vx in min_x..=max_x {
                if self.pel == 1 {
                    if self.motion_cost(vx, vy) < self.state.min_cost {
                        self.try_candidate_at(vx, vy, luma_x);
                        luma_x += 1;
                    }
                } else {
                    self.try_candidate(vx, vy);
                }
            }
        }
    }

    fn hex2(&mut self, range: i32) {
        if range > 1 {
            let (mut bx, mut by) = (i32::from(self.state.best.x), i32::from(self.state.best.y));
            let mut dir = -2;
            for (value, &(dx, dy)) in HEX2[1..7].iter().enumerate() {
                if self.check_mv(bx + dx, by + dy) {
                    dir = value as i32;
                }
            }
            if dir != -2 {
                bx += HEX2[(dir + 1) as usize].0;
                by += HEX2[(dir + 1) as usize].1;
                let mut step = 1;
                while step < range / 2 && self.state.bounds.contains(bx, by) {
                    let odir = MOD6_MINUS1[(dir + 1) as usize];
                    dir = -2;
                    for k in 0..3 {
                        let (dx, dy) = HEX2[(odir + k) as usize];
                        if self.check_mv(bx + dx, by + dy) {
                            dir = odir - 1 + k;
                        }
                    }
                    if dir == -2 {
                        break;
                    }
                    bx += HEX2[(dir + 1) as usize].0;
                    by += HEX2[(dir + 1) as usize].1;
                    step += 1;
                }
            }
            self.state.best.x = bx as i16;
            self.state.best.y = by as i16;
        }
        self.exhaustive(1);
    }

    fn umh(&mut self, range: i32, cx: i32, cy: i32) {
        for i in (1..range).step_by(2) {
            self.check_mv(cx - i, cy);
            self.check_mv(cx + i, cy);
        }
        for j in (1..range).step_by(2) {
            self.check_mv(cx, cy - j);
            self.check_mv(cx, cy + j);
        }
        let mut scale = 1;
        loop {
            for (dx, dy) in HEX4 {
                self.check_mv(cx + dx * scale, cy + dy * scale);
            }
            scale += 1;
            if scale > range / 4 {
                break;
            }
        }
        self.hex2(range);
    }

    fn refine(&mut self, moved: bool) {
        if !moved {
            let best = self.state.best;
            self.move_block(best.x.into(), best.y.into(), 2);
        }
        let radius = self.state.search_param;
        match self.settings.search_type {
            SearchType::Exhaustive => self.exhaustive(radius),
            SearchType::Hex2 => self.hex2(radius),
            SearchType::Umh => {
                let best = self.state.best;
                self.umh(radius, best.x.into(), best.y.into());
            }
        }
    }

    fn check_predictor(
        &mut self,
        candidate: Mv,
        penalty: i32,
        absolute: bool,
        slot: &mut (Mv, i32),
    ) -> bool {
        let (vx, vy) = (i32::from(candidate.x), i32::from(candidate.y));
        self.move_block(vx, vy, 0);
        let sad = self.full_sad(vx, vy);
        let penalised = |sad: i32| {
            sad.wrapping_add(if absolute {
                penalty
            } else {
                penalty.wrapping_mul(sad) >> 8
            })
        };
        let accepted = penalised(sad) < self.state.min_cost || self.settings.try_many;
        if accepted {
            self.state.best = Mv::new(vx, vy, sad);
            self.state.min_cost = penalised(sad);
        }
        if self.settings.try_many {
            self.refine(false);
            *slot = (self.state.best, penalised(self.state.best.sad));
        }
        accepted
    }

    fn contrast(
        block: &[u8],
        shape: Shape,
        kernel: Kernel,
        flat: &mut [u8; MAX_BLOCK_AREA],
    ) -> (i32, i32) {
        let average = metric::block_average(block, shape);
        let area = shape.area();
        flat[..area].fill(average as u8);
        let deviation = kernel(block, &flat[..], shape.width) as i32;
        (average, deviation / area as i32)
    }

    fn block_contrast(&mut self, negative: bool, need: bool) -> i32 {
        let shape = self.shape;
        let area = shape.area();
        let source = &self.buffers[0];
        let luma: &[u8] =
            if need && self.level > 0 && self.settings.flags & NO_CONTRAST_BLUR_FLAG == 0 {
                let shift = shape.width + 1;
                let (head, rest) = self.blurred[..area].split_at_mut(area - shift);
                let pixels = &source.y[..area];
                for ((value, &a), &b) in head.iter_mut().zip(&pixels[shift..]).zip(pixels) {
                    *value = u16::midpoint(u16::from(a), u16::from(b)) as u8;
                }
                for ((value, &a), &b) in rest.iter_mut().zip(pixels).zip(&pixels[area - shift..]) {
                    *value = u16::midpoint(u16::from(a), u16::from(b)) as u8;
                }
                &self.blurred
            } else {
                &source.y
            };
        if !need {
            self.block_luma = metric::block_average(luma, shape);
            return 0;
        }
        let (average, mut contrast) =
            Self::contrast(luma, shape, self.kernels.luma, &mut self.flat);
        self.block_luma = average;
        if negative {
            contrast *= 2;
        } else {
            let chroma = self.chroma_shape();
            contrast += Self::contrast(&source.u, chroma, self.kernels.chroma, &mut self.flat).1;
            contrast += Self::contrast(&source.v, chroma, self.kernels.chroma, &mut self.flat).1;
        }
        contrast.min(255)
    }

    fn search_block(&mut self, negative: bool, reverse: Mv, neighbours: &[Mv]) {
        let s = &mut self.state;
        let ratio = s.sad_limit as f32 / (s.sad_limit + (s.predictor.sad >> 1)) as f32;
        let k = f64::from(ratio) * f64::from(ratio);
        s.lambda = truncate(cmin(f64::from(s.lambda) * k, INT_MAX_OVER_100));
        let dont_search = s.search_param == 0 && self.level == 0;
        let flags = self.settings.flags;
        self.copy_source(0);
        let need_contrast = !dont_search
            && (self.adaptive_radius || (flags & CONTRAST_FLAG != 0 && self.level > 0));
        let contrast = self.block_contrast(negative, need_contrast);
        if self.adaptive_radius {
            let s = &mut self.state;
            s.search_param = if contrast < 2 {
                0
            } else {
                1 + ((s.search_param * (contrast * 255 / 150).min(255)) >> 8)
            };
        }
        self.state.min_cost = i32::MAX;
        let mut many = [(Mv::UNSET, 0); MAX_CANDIDATES];
        let predictor = self.state.predictor;
        if predictor.sad >= 0 {
            self.check_predictor(predictor, 0, true, &mut many[0]);
        }
        if dont_search {
            return;
        }
        if self.state.index.is_some() && reverse.sad > 0 {
            if self.check_predictor(reverse, self.settings.preverse, false, &mut many[1]) {
                self.state.predictor = reverse;
            }
        }
        self.check_predictor(self.global, self.settings.pglobal, false, &mut many[2]);
        self.check_predictor(Mv::UNSET, self.settings.pzero, false, &mut many[3]);
        for (i, &neighbour) in neighbours.iter().enumerate() {
            self.check_predictor(neighbour, self.settings.pnbour, false, &mut many[4 + i]);
        }
        self.state.min_cost = self.state.best.sad;
        if flags & CONTRAST_FLAG != 0 && self.level > 0 {
            self.state.min_cost = self.state.min_cost * (60 + contrast.min(120) / 2) / 120;
        }
        if self.settings.try_many {
            self.state.min_cost = i32::MAX;
            for &(candidate, cost) in &many[..4 + neighbours.len()] {
                if candidate.sad >= 0 && cost < self.state.min_cost {
                    self.state.best = candidate;
                    self.state.min_cost = cost;
                }
            }
        } else {
            self.refine(false);
        }
        let found = self.state.best.sad;
        let wide = self.state.index.is_some_and(|i| i > 1);
        if self.settings.bad_range && found > self.state.bad_sad && wide {
            let best = self.state.best;
            self.move_block(best.x.into(), best.y.into(), 2);
            self.umh(1, best.x.into(), best.y.into());
            self.exhaustive(1);
        }
    }

    fn recalculate_block(&mut self, need_luma: bool) {
        self.state.best = self.state.predictor;
        let bad = self.state.best.sad > self.state.bad_sad;
        if !bad && !need_luma {
            return;
        }
        self.copy_luma(self.active);
        self.block_luma = metric::block_average(&self.buffers[self.active].y, self.shape);
        if !bad {
            return;
        }
        self.copy_chroma(self.active);
        let predictor = self.state.predictor;
        let (vx, vy) = (i32::from(predictor.x), i32::from(predictor.y));
        self.move_block(vx, vy, 2);
        let sad = self.full_sad(vx, vy);
        self.state.best.sad = sad;
        self.state.min_cost = sad;
        self.refine(true);
    }
}
