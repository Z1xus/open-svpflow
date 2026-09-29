use super::{Field, Mv, SearchSettings, SearchType, VectorOutput, packed_score};
use crate::analyse::planes::LevelFrame;

pub(crate) const NONE: i32 = i32::MIN;
pub(crate) const RECORD: usize = 12;

const BAD: i32 = 1;
const NEGATIVE: i32 = 2;

#[derive(Clone, Copy)]
struct Record {
    full: i32,
    overlay: i32,
    u: i32,
    v: i32,
    average: i32,
    pos: (i32, i32),
    cpos: (i32, i32),
    predictor: (i32, i32),
    window: (i32, i32, i32, i32),
    sad: i32,
    lambda: i32,
    flags: i32,
}

fn pack(a: i32, b: i32) -> Option<i32> {
    let (a, b) = (i16::try_from(a).ok()?, i16::try_from(b).ok()?);
    Some((u32::from(a as u16) | (u32::from(b as u16) << 16)) as i32)
}

fn unpack(value: i32) -> (i32, i32) {
    ((value << 16) >> 16, value >> 16)
}

impl Record {
    const fn new(predictor: (i32, i32), sad: i32, lambda: i32, flags: i32) -> Self {
        Self {
            full: NONE,
            overlay: NONE,
            u: NONE,
            v: NONE,
            average: NONE,
            pos: (0, 0),
            cpos: (0, 0),
            predictor,
            window: (0, -1, 0, -1),
            sad,
            lambda,
            flags,
        }
    }

    fn pack(&self) -> Option<[i32; RECORD]> {
        if !(-(1 << 28)..1 << 28).contains(&self.lambda) {
            return None;
        }
        Some([
            self.full,
            self.overlay,
            self.u,
            self.v,
            self.average,
            pack(self.pos.0, self.pos.1)?,
            pack(self.cpos.0, self.cpos.1)?,
            pack(self.predictor.0, self.predictor.1)?,
            pack(self.window.0, self.window.1)?,
            pack(self.window.2, self.window.3)?,
            self.sad,
            (self.lambda << 2) | self.flags,
        ])
    }

    fn unpack(r: &[i32]) -> Self {
        let (min_x, max_x) = unpack(r[8]);
        let (min_y, max_y) = unpack(r[9]);
        Self {
            full: r[0],
            overlay: r[1],
            u: r[2],
            v: r[3],
            average: r[4],
            pos: unpack(r[5]),
            cpos: unpack(r[6]),
            predictor: unpack(r[7]),
            window: (min_x, max_x, min_y, max_y),
            sad: r[10],
            lambda: r[11] >> 2,
            flags: r[11] & 3,
        }
    }
}

#[derive(Clone, Copy, Default)]
struct Source {
    full: i32,
    half: i32,
    u: i32,
    v: i32,
}

impl Source {
    const EMPTY: Self = Self {
        full: NONE,
        half: NONE,
        u: NONE,
        v: NONE,
    };
}

#[derive(Default)]
pub(crate) struct RecalcPlan {
    pub(crate) records: Vec<i32>,
    indices: Vec<u32>,
    pub(crate) slots: usize,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) satd: bool,
    diagonal: i32,
    pub(crate) pnew: i32,
    pub(crate) pel: i32,
}

impl RecalcPlan {
    pub(crate) fn blocks(&self) -> usize {
        self.indices.len()
    }
}

impl Field {
    pub(crate) fn plan_recalculate(
        &self,
        src: &LevelFrame<'_>,
        settings: &SearchSettings,
        need_luma: bool,
        plan: &mut RecalcPlan,
    ) -> bool {
        let layout = self.layout;
        if self.chroma_shift != (1, 1)
            || settings.search_type != SearchType::Exhaustive
            || !matches!((layout.width, layout.height), (8, 8) | (16, 8 | 16))
            || self.order.len() != self.count()
            || self.blk_x < 2
            || self.blk_y < 2
        {
            return false;
        }
        let stp = settings.search_param;
        let radius = stp.abs();
        if radius > 3 {
            return false;
        }
        let lambda = settings.lambda >> 2;
        let (frame_w, frame_h) = (src.y.width, src.y.height);
        let border_negative = self.smallest && layout.width > 4;
        let pel = self.pel;
        let limit = (i32::MAX / 2) as usize;
        if [src.y, src.u, src.v].iter().any(|p| p.data().len() > limit) {
            return false;
        }
        plan.records.clear();
        plan.indices.clear();
        plan.records.reserve(self.order.len() * RECORD);
        plan.indices.reserve(self.order.len());
        plan.slots = 1 + ((2 * radius + 1) * (2 * radius + 1)) as usize;
        plan.width = layout.width as usize;
        plan.height = layout.height as usize;
        plan.satd = self.satd;
        plan.diagonal = pel * f64::from(frame_w * frame_w + frame_h * frame_h).sqrt() as i32;
        plan.pnew = settings.pnew;
        plan.pel = pel;
        let mut sources = [Source::EMPTY; 2];
        let mut active = 0usize;
        for &(bx, by) in &self.order {
            let (bx, by) = (i32::from(bx), i32::from(by));
            let index = (by * self.blk_x + bx) as usize;
            let negative = border_negative
                && (bx == 0 || bx >= self.blk_x - 1 || by == 0 || by >= self.blk_y - 1);
            let (mut x0, mut y0) = (bx * layout.step_x(), by * layout.step_y());
            let (bw, bh) = if negative {
                x0 += layout.width >> 2;
                y0 += layout.height >> 2;
                (layout.width >> 1, layout.height >> 1)
            } else {
                (layout.width, layout.height)
            };
            let (mut block_lambda, mut bad_sad) = (lambda, settings.bad_sad);
            if negative {
                block_lambda /= 3;
                bad_sad /= 3;
            }
            let predictor = self.vectors[index];
            let bad = predictor.sad > bad_sad;
            plan.indices.push(index as u32);
            let (px, py) = (i32::from(predictor.x), i32::from(predictor.y));
            let flags = if negative { NEGATIVE } else { 0 };
            let mut record = Record::new((px, py), predictor.sad, block_lambda, flags);
            if !bad && !need_luma {
                let Some(packed) = record.pack() else {
                    return false;
                };
                plan.records.extend_from_slice(&packed);
                continue;
            }
            let origin = src.y.pel1(x0, y0) as i32;
            let write_luma = |source: &mut Source, offset: i32| {
                if negative {
                    source.half = offset;
                } else {
                    source.full = offset;
                    source.half = NONE;
                }
            };
            write_luma(&mut sources[active], origin);
            record.average = origin;
            if !bad {
                let Some(packed) = record.pack() else {
                    return false;
                };
                plan.records.extend_from_slice(&packed);
                continue;
            }
            let (sx, sy) = self.chroma_shift;
            let chroma_at = |x: i32, y: i32| (src.u.pel1(x, y) as i32, src.v.pel1(x, y) as i32);
            if !negative {
                (sources[active].u, sources[active].v) = chroma_at(x0 >> sx, y0 >> sy);
            }
            let margin = |v: i32| if v < 0 { v - 2 } else { v + 2 };
            let outside = |v: i32, min: i32, max: i32| {
                if v < min {
                    v - min
                } else if v >= max {
                    v - max + 1
                } else {
                    0
                }
            };
            let (min_x0, min_y0) = (-pel * x0, -pel * y0);
            let (max_x0, max_y0) = (pel * (frame_w - x0 - bw), pel * (frame_h - y0 - bh));
            let dx = outside(margin(px), min_x0, max_x0);
            let dy = outside(margin(py), min_y0, max_y0);
            let (pos, cpos, bounds) = if dx == 0 && dy == 0 {
                active = 0;
                (
                    (x0, y0),
                    (x0 >> sx, y0 >> sy),
                    (min_x0, max_x0, min_y0, max_y0),
                )
            } else {
                let moved_x = (x0 - dx).max(0).min(frame_w - bw);
                let moved_y = (y0 - dy).max(0).min(frame_h - bh);
                let (dx, dy) = (x0 - moved_x, y0 - moved_y);
                let cpos = ((x0 >> sx) - (dx >> sx), (y0 >> sy) - (dy >> sy));
                write_luma(&mut sources[1], src.y.pel1(moved_x, moved_y) as i32);
                if !negative {
                    (sources[1].u, sources[1].v) = chroma_at(cpos.0, cpos.1);
                }
                active = 1;
                let bounds = (
                    min_x0 - dx.min(0),
                    max_x0 - dx.max(0),
                    min_y0 - dy.min(0),
                    max_y0 - dy.max(0),
                );
                ((moved_x, moved_y), cpos, bounds)
            };
            let source = sources[active];
            let (min_x, max_x, min_y, max_y) = bounds;
            record.full = source.full;
            record.overlay = source.half;
            record.u = source.u;
            record.v = source.v;
            record.flags |= BAD;
            record.pos = (pos.0 * pel, pos.1 * pel);
            record.cpos = (cpos.0 * pel, cpos.1 * pel);
            record.window = (
                (px - radius).max(min_x),
                (px + radius).min(max_x - 1),
                (py - radius).max(min_y),
                (py + radius).min(max_y - 1),
            );
            let Some(packed) = record.pack() else {
                return false;
            };
            plan.records.extend_from_slice(&packed);
        }
        true
    }

    pub(crate) fn evaluate_recalc_cpu(
        plan: &RecalcPlan,
        src: &LevelFrame<'_>,
        reference: &LevelFrame<'_>,
    ) -> Vec<[i32; 4]> {
        let blocks = plan.blocks();
        let slots = plan.slots;
        let mut luma = vec![0u32; blocks * slots];
        let mut chroma = vec![0u32; blocks * slots];
        let mut average = vec![0i32; blocks];
        let (fw, fh) = (plan.width, plan.height);
        let fetch = |data: &[u8], index: i64| -> i32 {
            usize::try_from(index)
                .ok()
                .and_then(|i| data.get(i))
                .map_or(0, |&v| i32::from(v))
        };
        for block in 0..blocks {
            let record = Record::unpack(&plan.records[block * RECORD..][..RECORD]);
            let negative = record.flags & NEGATIVE != 0;
            let (bw, bh) = if negative { (fw / 2, fh / 2) } else { (fw, fh) };
            if record.average != NONE {
                let mut sum = 0i64;
                for y in 0..bh {
                    for x in 0..bw {
                        let index = i64::from(record.average) + (y * src.y.pitch + x) as i64;
                        sum += i64::from(fetch(src.y.data(), index));
                    }
                }
                average[block] = if bh < 4 {
                    0
                } else {
                    (sum >> (bw * bh).trailing_zeros()) as i32
                };
            }
            if record.flags & BAD == 0 {
                continue;
            }
            let source_luma = |b: usize| -> i32 {
                let quarter = (fw / 2) * (fh / 2);
                if record.overlay != NONE && b < quarter {
                    let (r, c) = (b / (fw / 2), b % (fw / 2));
                    fetch(
                        src.y.data(),
                        i64::from(record.overlay) + (r * src.y.pitch + c) as i64,
                    )
                } else if record.full != NONE {
                    let (r, c) = (b / fw, b % fw);
                    fetch(
                        src.y.data(),
                        i64::from(record.full) + (r * src.y.pitch + c) as i64,
                    )
                } else {
                    0
                }
            };
            let (min_x, max_x, min_y, max_y) = record.window;
            let (ww, wh) = (max_x - min_x + 1, max_y - min_y + 1);
            for slot in 0..slots {
                let (vx, vy) = if slot == 0 {
                    record.predictor
                } else {
                    let c = (slot - 1) as i32;
                    if ww <= 0 || wh <= 0 || c >= ww * wh {
                        continue;
                    }
                    (min_x + c % ww, min_y + c / ww)
                };
                let offset = reference.y.subpel(record.pos.0 + vx, record.pos.1 + vy) as i64;
                let reference_luma = |y: usize, x: usize| {
                    fetch(
                        reference.y.data(),
                        offset + (y * reference.y.pitch + x) as i64,
                    )
                };
                luma[block * slots + slot] = cost(
                    plan.satd,
                    bw,
                    bh,
                    |y, x| source_luma(y * bw + x),
                    reference_luma,
                );
                if negative {
                    continue;
                }
                let (cw, ch) = (fw / 2, fh / 2);
                let coffset = reference
                    .u
                    .subpel(record.cpos.0 + (vx >> 1), record.cpos.1 + (vy >> 1))
                    as i64;
                let mut sum = 0;
                for (plane, source, own) in [
                    (reference.u, record.u, src.u),
                    (reference.v, record.v, src.v),
                ] {
                    let s = |y: usize, x: usize| {
                        if source == NONE {
                            0
                        } else {
                            fetch(own.data(), i64::from(source) + (y * own.pitch + x) as i64)
                        }
                    };
                    let r = |y: usize, x: usize| {
                        fetch(plane.data(), coffset + (y * plane.pitch + x) as i64)
                    };
                    sum += cost(plan.satd, cw, ch, s, r);
                }
                chroma[block * slots + slot] = sum;
            }
        }
        (0..blocks)
            .map(|block| select(plan, block, &luma, &chroma, average[block]))
            .collect()
    }

    pub(crate) fn finish_recalculate(
        &mut self,
        plan: &RecalcPlan,
        bests: &[[i32; 4]],
        mut output: Option<VectorOutput<'_>>,
    ) {
        for (&index, &[x, y, sad, average]) in plan.indices.iter().zip(bests) {
            let best = Mv::new(x, y, sad);
            let i = index as usize;
            self.vectors[i] = best;
            if let Some(out) = output.as_deref_mut() {
                out[2 * i] = ((i32::from(best.x) << 16) + i32::from(best.y)) as u32;
                out[2 * i + 1] = packed_score(best, plan.diagonal, average);
            }
        }
    }
}

fn select(plan: &RecalcPlan, block: usize, luma: &[u32], chroma: &[u32], average: i32) -> [i32; 4] {
    let record = Record::unpack(&plan.records[block * RECORD..][..RECORD]);
    let pnew = plan.pnew;
    let penalty = |sad: i32| sad.wrapping_add(pnew.wrapping_mul(sad) >> 8);
    let (px, py) = record.predictor;
    let (mut bx, mut by, mut bsad) = (px, py, record.sad);
    if record.flags & BAD != 0 {
        let base = block * plan.slots;
        let value = |slot: usize| (luma[base + slot] as i32, (chroma[base + slot] as i32) << 2);
        let (l0, c0) = value(0);
        bsad = l0 + c0;
        let mut min_cost = bsad;
        let (min_x, max_x, min_y, max_y) = record.window;
        let ww = max_x - min_x + 1;
        let lambda = record.lambda;
        if min_x <= max_x && min_y <= max_y {
            for vy in min_y..=max_y {
                let row = 1 + ((vy - min_y) * ww) as usize;
                let dy = py - vy;
                let dy2 = dy.wrapping_mul(dy);
                let mut luma_x = min_x;
                for vx in min_x..=max_x {
                    let dx = px - vx;
                    let mut cost = lambda.wrapping_mul(dx.wrapping_mul(dx).wrapping_add(dy2)) >> 8;
                    if cost >= min_cost {
                        continue;
                    }
                    let at = if plan.pel == 1 { luma_x } else { vx };
                    if plan.pel == 1 {
                        luma_x += 1;
                    }
                    let sad = value(row + (at - min_x) as usize).0;
                    cost = cost.wrapping_add(penalty(sad));
                    if cost >= min_cost {
                        continue;
                    }
                    let sad_uv = value(row + (vx - min_x) as usize).1;
                    cost = cost.wrapping_add(penalty(sad_uv));
                    if cost >= min_cost {
                        continue;
                    }
                    (bx, by, bsad) = (vx, vy, sad + sad_uv);
                    min_cost = cost;
                }
            }
        }
    }
    if record.flags & NEGATIVE != 0 {
        bsad = bsad.wrapping_mul(3);
    }
    [bx, by, bsad, average]
}

fn cost(
    satd: bool,
    width: usize,
    height: usize,
    src: impl Fn(usize, usize) -> i32,
    reference: impl Fn(usize, usize) -> i32,
) -> u32 {
    if !satd {
        let mut total = 0;
        for y in 0..height {
            for x in 0..width {
                total += (src(y, x) - reference(y, x)).unsigned_abs();
            }
        }
        return total;
    }
    let mut total = 0;
    for ty in (0..height).step_by(4) {
        for tx in (0..width).step_by(4) {
            let mut d = [[0i32; 4]; 4];
            for (y, row) in d.iter_mut().enumerate() {
                for (x, value) in row.iter_mut().enumerate() {
                    *value = src(ty + y, tx + x) - reference(ty + y, tx + x);
                }
            }
            for row in &mut d {
                let [p0, p1, p2, p3] = *row;
                let (s0, s1, d0, d1) = (p0 + p1, p2 + p3, p0 - p1, p2 - p3);
                *row = [s0 + s1, s0 - s1, d0 + d1, d0 - d1];
            }
            for x in 0..4 {
                let (s0, s1) = (d[0][x] + d[1][x], d[2][x] + d[3][x]);
                let (d0, d1) = (d[0][x] - d[1][x], d[2][x] - d[3][x]);
                total += (s0 + s1).unsigned_abs() + (s0 - s1).unsigned_abs();
                total += (d0 + d1).unsigned_abs() + (d0 - d1).unsigned_abs();
            }
        }
    }
    total / 2
}
