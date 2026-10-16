#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::many_single_char_names
)]

pub const CHUNK: i32 = 12;
const MIN: f32 = 0.2;
const GROUP: usize = 3;
const FILL: f32 = 4.0;
const SAME: f32 = 2.5;
const GAIN: f32 = 4096.0;
const OFFSET: f32 = 64.0;
const CELL: usize = 8;
const WORKERS: usize = 6;

#[must_use]
pub fn limit(limit: Option<f64>) -> f32 {
    limit.unwrap_or(6.0) as f32
}

#[must_use]
pub fn pairs(chunk: i32, last: i32) -> std::ops::RangeInclusive<i32> {
    chunk * CHUNK..=(chunk * CHUNK + CHUNK - 1).min(last)
}

#[derive(Clone, Copy)]
pub struct Plane<'a> {
    pub data: &'a [u8],
    pub pitch: usize,
}

impl Plane<'_> {
    fn at(&self, x: usize, y: usize) -> f32 {
        f32::from(self.data[y * self.pitch + x])
    }

    fn sample(&self, x: f32, y: f32, (width, height): (usize, usize)) -> f32 {
        let (left, top) = (x as usize, y as usize);
        let (right, below) = ((left + 1).min(width - 1), (top + 1).min(height - 1));
        let (fx, fy) = (x - left as f32, y - top as f32);
        let upper = mix(self.at(left, top), self.at(right, top), fx);
        let lower = mix(self.at(left, below), self.at(right, below), fx);
        mix(upper, lower, fy)
    }
}

#[derive(Clone, Copy)]
pub struct Motion<'a> {
    pub forward: (&'a [u16], &'a [u16]),
    pub backward: (&'a [u16], &'a [u16]),
    pub grid: (usize, usize),
    pub block: (i32, i32),
    pub origin: (i32, i32),
    pub pel: i32,
}

struct Steps {
    cells: Vec<[f32; 4]>,
    block: f32,
    origin: f32,
    ratio: f32,
}

impl Motion<'_> {
    fn far(&self) -> f32 {
        let fields = [
            self.forward.0,
            self.forward.1,
            self.backward.0,
            self.backward.1,
        ];
        let far = fields
            .iter()
            .flat_map(|field| field.iter())
            .map(|&value| value.abs_diff(1024))
            .max()
            .unwrap_or(0);
        f32::from(far) / self.pel as f32
    }

    fn steps(&self, y: usize, ratio: usize) -> Steps {
        let (width, height) = self.grid;
        let at = (y * ratio) as f32 - self.origin.1 as f32;
        let (top, below, f) = taps(at / self.block.1 as f32, height);
        let scale = 1.0 / (ratio as f32 * self.pel as f32);
        let fields = [
            self.backward.0,
            self.backward.1,
            self.forward.0,
            self.forward.1,
        ];
        let cells = (0..width)
            .map(|x| {
                fields.map(|field| {
                    let value = |row: usize| f32::from(field[row * width + x]);
                    (mix(value(top), value(below), f) - 1024.0) * scale
                })
            })
            .collect();
        Steps {
            cells,
            block: self.block.0 as f32,
            origin: self.origin.0 as f32,
            ratio: ratio as f32,
        }
    }
}

impl Steps {
    fn at(&self, x: usize) -> [f32; 4] {
        let at = (x as f32 * self.ratio - self.origin) / self.block;
        let (left, right, f) = taps(at, self.cells.len());
        std::array::from_fn(|i| mix(self.cells[left][i], self.cells[right][i], f))
    }
}

fn taps(at: f32, size: usize) -> (usize, usize, f32) {
    let at = (at - 0.5).clamp(0.0, (size - 1) as f32);
    let first = at as usize;
    (first, (first + 1).min(size - 1), at - first as f32)
}

fn mix(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn median(a: f32, b: f32, c: f32) -> f32 {
    a.min(b).max(a.max(b).min(c))
}

fn pixel(value: f32) -> u8 {
    value.round().clamp(0.0, 255.0) as u8
}

pub struct Pair<'a> {
    pub first: [Plane<'a>; 3],
    pub second: [Plane<'a>; 3],
    pub motion: Motion<'a>,
}

pub struct Stats {
    luma: Vec<[f32; 5]>,
    chroma: [Vec<[f32; 2]>; 2],
    count: f32,
}

fn gather<const N: usize>(
    pairs: &[Pair],
    plane: usize,
    (width, height): (usize, usize),
    chroma: usize,
) -> Vec<[f32; N]> {
    let ratio = if plane == 0 { 1 } else { chroma };
    let size = (width / ratio, height / ratio);
    let top = ((size.0 - 1) as f32, (size.1 - 1) as f32);
    let mut sums = vec![[0.0; N]; size.0 * size.1];
    let band = size.1.div_ceil(WORKERS).max(1);
    std::thread::scope(|scope| {
        for (index, rows) in sums.chunks_mut(band * size.0).enumerate() {
            scope.spawn(move || {
                for pair in pairs {
                    let (first, second) = (pair.first[plane], pair.second[plane]);
                    for (y, row) in rows.chunks_exact_mut(size.0).enumerate() {
                        let y = index * band + y;
                        let steps = pair.motion.steps(y, ratio);
                        for (x, sum) in row.iter_mut().enumerate() {
                            let [fx, fy, bx, by] = steps.at(x);
                            let (ax, ay) = (x as f32, y as f32);
                            let (ia, ib) = (first.at(x, y), second.at(x, y));
                            let moved = |raw: Plane, dx: f32, dy: f32| {
                                let x = (ax + dx).clamp(0.0, top.0);
                                raw.sample(x, (ay + dy).clamp(0.0, top.1), size)
                            };
                            let (pb, pa) = (moved(first, fx, fy), moved(second, bx, by));
                            let terms = [
                                pb + pa,
                                ib + ia,
                                pb * pb + pa * pa,
                                pb * ib + pa * ia,
                                ib * ib + ia * ia,
                            ];
                            for (sum, term) in sum.iter_mut().zip(terms) {
                                *sum += term;
                            }
                        }
                    }
                }
            });
        }
    });
    sums
}

// chroma is 2 when the color planes are half the size of the picture, as in 4:2:0, and 1 in 4:4:4
#[must_use]
pub fn stats(pairs: &[Pair], size: (usize, usize), chroma: usize) -> Stats {
    Stats {
        luma: gather(pairs, 0, size, 1),
        chroma: [
            gather(pairs, 1, size, chroma),
            gather(pairs, 2, size, chroma),
        ],
        count: (pairs.len() * 2) as f32,
    }
}

fn model(t: &[f32; 5], n: f32, limit: f32, strict: bool) -> (f32, f32) {
    let (mp, mi) = (t[0] / n, t[1] / n);
    let (vp, vi, cov) = (t[2] / n - mp * mp, t[4] / n - mi * mi, t[3] / n - mp * mi);
    let a = cov / (vp + 4.0);
    let b = mi - a * mp;
    let same = vi + vp - 2.0 * cov + (mi - mp) * (mi - mp);
    let left = (vi - 2.0 * a * cov + a * a * vp).max(0.0);
    let g = ((1.0 - left / same.max(1e-3) - 0.5) / 0.2).clamp(0.0, 1.0)
        * ((same - limit) / limit).clamp(0.0, 1.0);
    let g = if strict {
        g * ((limit / 2.0 - left) / (limit / 4.0)).clamp(0.0, 1.0)
    } else {
        g
    };
    (1.0 + g * (a - 1.0), g * b)
}

fn error(t: &[f32; 5], n: f32, (a, b): (f32, f32)) -> f32 {
    (t[4] - 2.0 * a * t[3] - 2.0 * b * t[1] + a * a * t[2] + 2.0 * a * b * t[0]) / n + b * b
}

fn set((a, b): (f32, f32)) -> bool {
    (a - 1.0).abs() > 0.02 || b.abs() > 1.0
}

fn distances(cells: &mut [u8], width: usize) {
    let near = |cells: &[u8], at: usize, others: [Option<usize>; 4]| {
        others.into_iter().flatten().fold(cells[at], |best, other| {
            best.min(cells[other].saturating_add(1))
        })
    };
    let height = cells.len() / width;
    for y in 0..height {
        for x in 0..width {
            let at = y * width + x;
            let (left, right, up) = (x > 0, x + 1 < width, y > 0);
            cells[at] = near(
                cells,
                at,
                [
                    left.then(|| at - 1),
                    up.then(|| at - width),
                    (up && left).then(|| at - width - 1),
                    (up && right).then(|| at - width + 1),
                ],
            );
        }
    }
    for y in (0..height).rev() {
        for x in (0..width).rev() {
            let at = y * width + x;
            let (left, right, down) = (x > 0, x + 1 < width, y + 1 < height);
            cells[at] = near(
                cells,
                at,
                [
                    right.then(|| at + 1),
                    down.then(|| at + width),
                    (down && right).then(|| at + width + 1),
                    (down && left).then(|| at + width - 1),
                ],
            );
        }
    }
}

fn window(
    parts: [Option<&Stats>; 3],
    at: usize,
    limit: f32,
) -> Option<((f32, f32), std::ops::Range<usize>)> {
    let own = parts[1]?;
    let sums = &own.luma[at];
    let same = (sums[4] - 2.0 * sums[3] + sums[2]) / own.count;
    let pick = |strict| {
        let mut best = ((1.0, 0.0), 0.0, 0..3);
        for (window, range) in [0..3, 0..2, 1..3].into_iter().enumerate() {
            let mut total = [0.0; 5];
            let mut n = 0.0;
            for part in parts[range.clone()].iter().flatten() {
                for (total, sum) in total.iter_mut().zip(part.luma[at]) {
                    *total += sum;
                }
                n += part.count;
            }
            let found = model(&total, n, limit, strict);
            let e = error(sums, own.count, found);
            if window == 0 || e * 1.15 + 2.0 < best.1 {
                best = (found, e, range);
            }
        }
        let (found, e, range) = best;
        (e < 1.7 * same + 4.0 && found != (1.0, 0.0)).then_some((found, range))
    };
    // an overlay that changes here, like chat that scrolls, is clean in a near chunk
    let near = [parts[0], parts[2]]
        .into_iter()
        .flatten()
        .any(|part| set(model(&part.luma[at], part.count, limit, true)));
    pick(true).or_else(|| near.then(|| pick(false)).flatten())
}

pub struct Fit {
    pub luma: Vec<i16>,
    pub chroma: [Vec<i16>; 2],
}

#[must_use]
pub fn fit(
    parts: [Option<&Stats>; 3],
    (width, height): (usize, usize),
    limit: f32,
    ratio: usize,
) -> Option<Fit> {
    parts[1]?;
    let half = width / ratio;
    let limit = limit * limit;
    let mut rough = vec![(1.0, 0.0); width * height];
    let mut offsets = vec![[0.0f32; 2]; width * height];
    let band = height.div_ceil(WORKERS).max(1) * width;
    std::thread::scope(|scope| {
        let bands = rough.chunks_mut(band).zip(offsets.chunks_mut(band));
        for (index, (rough, offsets)) in bands.enumerate() {
            scope.spawn(move || {
                for (at, (rough, offsets)) in rough.iter_mut().zip(offsets).enumerate() {
                    let at = index * band + at;
                    let Some(((a, b), range)) = window(parts, at, limit) else {
                        continue;
                    };
                    let at = (at / width / ratio) * half + (at % width) / ratio;
                    let mut total = [0.0; 4];
                    let mut n = 0.0;
                    for part in parts[range].iter().flatten() {
                        let terms = [part.chroma[0][at], part.chroma[1][at]];
                        for (total, sum) in total.iter_mut().zip(terms.as_flattened()) {
                            *total += sum;
                        }
                        n += part.count;
                    }
                    *rough = (a, b);
                    *offsets = [(total[1] - a * total[0]) / n, (total[3] - a * total[2]) / n];
                }
            });
        }
    });

    let fixed = |value: f32, scale: f32| (value * scale).round().clamp(-32767.0, 32767.0) as i16;
    let mut luma = vec![0; width * height * 2];
    // the color planes hold the offsets, and below them in the second plane the distances to the layer
    let rows = height / ratio * 2;
    let mut chroma = [vec![0; half * rows], vec![0; half * rows]];
    let cells = (width.div_ceil(CELL), height.div_ceil(CELL));
    let mut cover = vec![255; cells.0 * cells.1];
    let (gains, shifts) = luma.split_at_mut(width * height);
    gains.fill(GAIN as i16);
    for y in 0..height {
        for x in 0..width {
            let at = y * width + x;
            if rough[at] == (1.0, 0.0) {
                continue;
            }
            if set(rough[at]) {
                let near = (0..9)
                    .filter(|side| {
                        let x = (x + side % 3).saturating_sub(1).min(width - 1);
                        let y = (y + side / 3).saturating_sub(1).min(height - 1);
                        set(rough[y * width + x])
                    })
                    .count();
                if near < GROUP {
                    continue;
                }
                cover[y / CELL * cells.0 + x / CELL] = 0;
            }
            gains[at] = fixed(rough[at].0, GAIN);
            shifts[at] = fixed(rough[at].1, OFFSET);
            let at = (y / ratio) * half + x / ratio;
            let share = (ratio * ratio) as f32;
            for (plane, offset) in chroma.iter_mut().zip(offsets[y * width + x]) {
                plane[at] += fixed(offset / share, OFFSET);
            }
        }
    }
    distances(&mut cover, cells.0);
    for (row, cover) in cover.chunks_exact(cells.0).enumerate() {
        let at = (height / ratio + row) * half;
        for (cell, &distance) in chroma[1][at..].iter_mut().zip(cover) {
            *cell = i16::from(distance);
        }
    }
    Some(Fit { luma, chroma })
}

impl Fit {
    #[must_use]
    pub fn layer(&self, (width, height): (usize, usize), ratio: usize) -> Layer<'_> {
        Layer {
            luma: &self.luma,
            chroma: [&self.chroma[0], &self.chroma[1]],
            pitch: (width, width / ratio),
            size: (width, height),
            ratio,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Layer<'a> {
    pub luma: &'a [i16],
    pub chroma: [&'a [i16]; 2],
    pub pitch: (usize, usize),
    pub size: (usize, usize),
    // 2 when the color planes are half the size of the picture, 1 when they are as large
    pub ratio: usize,
}

pub struct Frames<'a> {
    pub first: [Plane<'a>; 3],
    pub second: [Plane<'a>; 3],
    pub time: f32,
}

impl Layer<'_> {
    fn ratio(&self, plane: usize) -> usize {
        if plane == 0 { 1 } else { self.ratio }
    }

    fn plane(&self, plane: usize) -> (usize, usize) {
        let ratio = self.ratio(plane);
        (self.size.0 / ratio, self.size.1 / ratio)
    }

    fn gap(&self, plane: usize, x: usize, y: usize) -> f32 {
        let cell = CELL / self.ratio(plane);
        let at = (self.size.1 / self.ratio + y / cell) * self.pitch.1 + x / cell;
        f32::from(self.chroma[1][at]) * 8.0 - 8.0
    }

    #[must_use]
    pub fn empty(&self) -> bool {
        let (width, height) = (self.size.0.div_ceil(CELL), self.size.1.div_ceil(CELL));
        let rows = self.chroma[1][self.size.1 / self.ratio * self.pitch.1..].chunks(self.pitch.1);
        rows.take(height)
            .all(|row| row[..width].iter().all(|&distance| distance != 0))
    }

    fn texel(&self, plane: usize, x: usize, y: usize) -> (f32, f32) {
        let gain = |x: usize, y: usize| f32::from(self.luma[y * self.pitch.0 + x]);
        if plane == 0 {
            let offset = self.luma[(self.size.1 + y) * self.pitch.0 + x];
            return (gain(x, y) / GAIN, f32::from(offset) / OFFSET);
        }
        let ratio = self.ratio;
        let mut total = 0.0;
        for dy in 0..ratio {
            for dx in 0..ratio {
                total += gain(x * ratio + dx, y * ratio + dy);
            }
        }
        let offset = self.chroma[plane - 1][y * self.pitch.1 + x];
        (
            total / ((ratio * ratio) as f32 * GAIN),
            f32::from(offset) / OFFSET,
        )
    }

    fn model(&self, plane: usize, x: f32, y: f32) -> (f32, f32) {
        let (width, height) = self.plane(plane);
        let (left, top) = (x as usize, y as usize);
        let (fx, fy) = (x - left as f32, y - top as f32);
        if fx == 0.0 && fy == 0.0 {
            return self.texel(plane, left, top);
        }
        let (right, below) = ((left + 1).min(width - 1), (top + 1).min(height - 1));
        let row = |y: usize| {
            let (a, b) = (self.texel(plane, left, y), self.texel(plane, right, y));
            (mix(a.0, b.0, fx), mix(a.1, b.1, fx))
        };
        let (upper, lower) = (row(top), row(below));
        (mix(upper.0, lower.0, fy), mix(upper.1, lower.1, fy))
    }

    fn scene(&self, plane: usize, raw: Plane, x: f32, y: f32, fill: bool) -> (f32, bool) {
        let size = self.plane(plane);
        let under = |x: f32, y: f32| {
            let (a, b) = self.model(plane, x, y);
            (a.abs() >= MIN).then(|| ((raw.sample(x, y, size) - b) / a).clamp(0.0, 255.0))
        };
        if let Some(scene) = under(x, y) {
            return (scene, true);
        }
        let radius = FILL / self.ratio(plane) as f32;
        let top = ((size.0 - 1) as f32, (size.1 - 1) as f32);
        for ring in 1..=3 {
            if !fill {
                break;
            }
            let radius = radius * ring as f32;
            let (mut total, mut count) = (0.0, 0.0);
            for side in (0..9).filter(|&side| side != 4) {
                let near = (x + (side % 3 - 1) as f32 * radius).clamp(0.0, top.0);
                let scene = under(near, (y + (side / 3 - 1) as f32 * radius).clamp(0.0, top.1));
                if let Some(scene) = scene {
                    total += scene;
                    count += 1.0;
                }
            }
            if count > 0.0 {
                return (total / count, true);
            }
        }
        (raw.sample(x, y, size), false)
    }

    pub fn peel(&self, plane: usize, output: &mut [u8], pitch: usize, raw: Plane) {
        let (width, height) = self.plane(plane);
        for y in 0..height {
            for x in 0..width {
                if self.gap(plane, x, y) <= 0.0 && self.texel(plane, x, y) != (1.0, 0.0) {
                    let scene = self.scene(plane, raw, x as f32, y as f32, true).0;
                    output[y * pitch + x] = pixel(scene);
                }
            }
        }
    }

    pub fn unpeel(&self, plane: usize, output: &mut [u8], pitch: usize, frames: &Frames) {
        let (width, height) = self.plane(plane);
        for y in 0..height {
            for x in 0..width {
                let (a, b) = self.texel(plane, x, y);
                if self.gap(plane, x, y) > 0.0 || (a, b) == (1.0, 0.0) {
                    continue;
                }
                let own = (frames.first[plane].at(x, y), frames.second[plane].at(x, y));
                let at = y * pitch + x;
                if a.abs() < MIN || (own.0 - own.1).abs() <= SAME {
                    output[at] = pixel(mix(own.0, own.1, frames.time));
                    continue;
                }
                let under = |own: f32| ((own - b) / a).clamp(0.0, 255.0);
                let scene = median(f32::from(output[at]), under(own.0), under(own.1));
                output[at] = pixel(a * scene + b);
            }
        }
    }

    pub fn render(
        &self,
        plane: usize,
        output: &mut [u8],
        pitch: usize,
        frames: &Frames,
        motion: &Motion,
        algo: i32,
    ) {
        let (width, height) = self.plane(plane);
        let ratio = self.ratio(plane);
        let top = ((width - 1) as f32, (height - 1) as f32);
        let (first, second, time) = (frames.first[plane], frames.second[plane], frames.time);
        let far = (motion.far() + 1.0).max(16.0);
        for y in 0..height {
            let mut steps = None;
            for x in 0..width {
                let gap = self.gap(plane, x, y);
                if gap > far {
                    continue;
                }
                let steps = steps.get_or_insert_with(|| motion.steps(y, ratio));
                let [fx, fy, bx, by] = steps.at(x);
                let (fx, fy, bx, by) = (fx * time, fy * time, bx * (1.0 - time), by * (1.0 - time));
                let reach = fx.abs().max(fy.abs()).max(bx.abs()).max(by.abs()) * ratio as f32;
                if gap > (reach + 1.0).max(16.0) {
                    continue;
                }
                let (ax, ay) = (x as f32, y as f32);
                let moved = |raw: Plane, dx: f32, dy: f32| {
                    let x = (ax + dx).clamp(0.0, top.0);
                    self.scene(plane, raw, x, (ay + dy).clamp(0.0, top.1), true)
                };
                let (moved_f, moved_b) = (moved(first, fx, fy), moved(second, bx, by));
                let base = mix(
                    self.scene(plane, first, ax, ay, false).0,
                    self.scene(plane, second, ax, ay, false).0,
                    time,
                );
                let scene = match (algo, moved_f.1, moved_b.1) {
                    (1, _, true) | (_, false, true) => moved_b.0,
                    (1 | 2, true, _) | (_, true, false) => moved_f.0,
                    (_, true, true) => median(moved_f.0, moved_b.0, base),
                    _ => base,
                };
                let (a, b) = self.texel(plane, x, y);
                let own = (first.at(x, y), second.at(x, y));
                let layered = gap <= 0.0 && (a, b) != (1.0, 0.0);
                let kept = a.abs() < MIN || (layered && (own.0 - own.1).abs() <= SAME);
                output[y * pitch + x] = pixel(if kept {
                    mix(own.0, own.1, time)
                } else {
                    a * scene + b
                });
            }
        }
    }
}
