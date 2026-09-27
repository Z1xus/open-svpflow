use crate::super_opts::{SuperOpts, reduce_dim};

const SSE_CHUNK: usize = 16;

#[derive(Clone, Copy, PartialEq, Eq)]
enum ReduceFilter {
    Average,
    Bilinear,
    Quadratic,
    Cubic,
}

impl ReduceFilter {
    fn from_option(value: i32) -> Option<Self> {
        match value {
            0 | 1 => Some(Self::Average),
            2 => Some(Self::Bilinear),
            3 => Some(Self::Quadratic),
            4 => Some(Self::Cubic),
            _ => None,
        }
    }

    const fn rows(self) -> std::ops::RangeInclusive<isize> {
        match self {
            Self::Average => -1..=1,
            Self::Bilinear => -1..=2,
            Self::Quadratic | Self::Cubic => -2..=3,
        }
    }

    fn vertical_edge(self, y: usize, height: usize) -> bool {
        y == 0 || (self != Self::Average && y + 1 >= height.max(2))
    }

    fn apply(self, t: [u32; 6]) -> u32 {
        let [m2, m1, c0, p1, p2, p3] = t;
        match self {
            Self::Average => (m1 + 2 * c0 + p1 + 2) >> 2,
            Self::Bilinear => (m1 + 3 * (c0 + p1) + p2 + 4) >> 3,
            Self::Quadratic => (m2 + 9 * (m1 + p2) + 22 * (c0 + p1) + p3 + 32) >> 6,
            Self::Cubic => (m2 + 5 * (m1 + p2) + 10 * (c0 + p1) + p3 + 16) >> 5,
        }
    }
}

#[derive(Clone, Copy)]
enum Upscale {
    Bilinear,
    Bicubic,
    Wiener,
}

#[derive(Clone, Copy)]
pub(crate) struct Region {
    pub(crate) offset: usize,
    width: usize,
    height: usize,
}

pub(crate) fn level_regions(
    pitch: usize,
    luma_w: usize,
    luma_h: usize,
    (x_shift, y_shift): (u32, u32),
    opts: &SuperOpts,
) -> Vec<Region> {
    let pel = opts.pel as usize;
    let mut offset = 0;
    (0..opts.levels.max(0))
        .map(|level| {
            let width = reduce_dim(luma_w as i32, level) as usize >> x_shift;
            let height = reduce_dim(luma_h as i32, level) as usize >> y_shift;
            let region = Region {
                offset,
                width,
                height,
            };
            let stored = match (level, opts.full) {
                (0, true) => pel * pel,
                (0, false) => 0,
                _ => 1,
            };
            offset += stored * height * pitch;
            region
        })
        .collect()
}

pub(crate) fn build_plane(
    dst: &mut [u8],
    pitch: usize,
    src: &[u8],
    src_pitch: usize,
    luma_w: usize,
    luma_h: usize,
    shift: (u32, u32),
    opts: &SuperOpts,
) {
    let regions = level_regions(pitch, luma_w, luma_h, shift, opts);
    let Some(&level0) = regions.first() else {
        return;
    };
    let mut canvas = Canvas {
        frame: dst,
        spill: Vec::new(),
    };
    if opts.full {
        for row in 0..level0.height {
            canvas.store(
                level0.offset + row * pitch,
                &src[row * src_pitch..][..level0.width],
            );
        }
    }
    if let Some(filter) = ReduceFilter::from_option(opts.scale_down) {
        for (level, pair) in regions.windows(2).enumerate() {
            let source = if level == 0 && !opts.full {
                Source::Separate(src, src_pitch)
            } else {
                Source::Within(pair[0].offset, pitch)
            };
            let compact = matches!(source, Source::Separate(..));
            reduce(filter, source, &mut canvas, pair[1], pitch, compact);
        }
    }
    if opts.pel > 1 && opts.gpu < 2 && opts.full {
        let upscale = match opts.scale_up {
            0 => Upscale::Bilinear,
            1 => Upscale::Bicubic,
            _ => Upscale::Wiener,
        };
        refine(&mut canvas, level0, pitch, opts.pel as usize, upscale);
    }
}

#[derive(Clone, Copy)]
enum Source<'a> {
    Separate(&'a [u8], usize),
    Within(usize, usize),
}

struct Canvas<'a> {
    frame: &'a mut [u8],
    spill: Vec<u8>,
}

impl Canvas<'_> {
    fn fetch(&self, start: isize, out: &mut [u8]) {
        if let Ok(begin) = usize::try_from(start)
            && let Some(bytes) = self.frame.get(begin..begin + out.len())
        {
            out.copy_from_slice(bytes);
            return;
        }
        let len = self.frame.len() as isize;
        for (i, value) in out.iter_mut().enumerate() {
            let at = start + i as isize;
            *value = match at {
                ..0 => 0,
                _ if at < len => self.frame[at as usize],
                _ => self.spill.get((at - len) as usize).copied().unwrap_or(0),
            };
        }
    }

    fn store(&mut self, start: usize, data: &[u8]) {
        let len = self.frame.len();
        if let Some(bytes) = self.frame.get_mut(start..start + data.len()) {
            bytes.copy_from_slice(data);
            return;
        }
        for (i, &value) in data.iter().enumerate() {
            match (start + i).checked_sub(len) {
                None => self.frame[start + i] = value,
                Some(beyond) => {
                    if beyond >= self.spill.len() {
                        self.spill.resize(beyond + 1, 0);
                    }
                    self.spill[beyond] = value;
                }
            }
        }
    }

    fn fetch_source(&self, source: Source<'_>, offset: isize, out: &mut [u8]) {
        match source {
            Source::Within(base, _) => self.fetch(base as isize + offset, out),
            Source::Separate(src, _) => {
                for (i, value) in out.iter_mut().enumerate() {
                    let at = usize::try_from(offset + i as isize).ok();
                    *value = at.and_then(|at| src.get(at)).copied().unwrap_or(0);
                }
            }
        }
    }
}

fn reduce(
    filter: ReduceFilter,
    source: Source<'_>,
    canvas: &mut Canvas<'_>,
    target: Region,
    pitch: usize,
    compact: bool,
) {
    let src_pitch = match source {
        Source::Separate(_, p) | Source::Within(_, p) => p as isize,
    };
    let row_step = if compact { 2 * pitch } else { pitch };
    let (width2, height) = (2 * target.width, target.height);
    let exact_end = match filter {
        ReduceFilter::Cubic if width2 >= 8 => width2 / 8 * 8,
        ReduceFilter::Cubic => 0,
        _ => width2,
    };
    let mut rows = [const { Vec::new() }; 6];
    let mut halves = [const { Vec::new() }; 3];
    let mut out = vec![0u8; width2];
    for y in 0..height {
        let center = (2 * y) as isize * src_pitch;
        let edge = filter.vertical_edge(y, height);
        let needed = if edge { 0..=1 } else { filter.rows() };
        for k in needed {
            let row = &mut rows[(k + 2) as usize];
            row.resize(width2, 0);
            canvas.fetch_source(source, center + k * src_pitch, row);
        }
        let computed = if edge { width2 } else { exact_end };
        let [m2, m1, c0, p1, p2, p3] = &rows;
        if edge {
            vertical_average(&mut out[..computed], c0, p1);
        } else {
            vertical_filter(filter, &mut out[..computed], [m2, m1, c0, p1, p2, p3]);
        }
        let tap = |k: isize, x: usize| u32::from(rows[(k + 2) as usize][x]);
        if computed < width2 {
            let tail = width2 - computed;
            for (half, k) in halves.iter_mut().zip([-1, 1, 3]) {
                half.resize(tail, 0);
                canvas.fetch_source(
                    source,
                    center + computed as isize + k * (src_pitch / 2),
                    half,
                );
            }
            for i in 0..tail {
                let x = computed + i;
                let h = |n: usize| u32::from(halves[n][i]);
                let value =
                    h(0) + 5 * tap(-1, x) + 10 * (tap(0, x) + tap(1, x)) + 5 * h(1) + h(2) + 16;
                out[x] = (value >> 5) as u8;
            }
        }
        canvas.store(target.offset + y * row_step, &out);
    }
    let mut row = vec![0u8; width2];
    for y in 0..height {
        let start = target.offset + y * row_step;
        canvas.fetch(start as isize, &mut row);
        reduce_row_in_place(filter, &mut row, target.width);
        canvas.store(start, &row[..target.width]);
    }
    if compact {
        let mut line = vec![0u8; pitch];
        for y in 1..height {
            canvas.fetch((target.offset + 2 * y * pitch) as isize, &mut line);
            canvas.store(target.offset + y * pitch, &line);
        }
    }
}

fn vertical_average(out: &mut [u8], a: &[u8], b: &[u8]) {
    for ((value, &a), &b) in out.iter_mut().zip(a).zip(b) {
        *value = ((u16::from(a) + u16::from(b) + 1) >> 1) as u8;
    }
}

fn vertical_filter(filter: ReduceFilter, out: &mut [u8], rows: [&Vec<u8>; 6]) {
    let n = out.len();
    let used = filter.rows();
    let row = |k: isize| -> &[u8] {
        if used.contains(&k) {
            &rows[(k + 2) as usize][..n]
        } else {
            &rows[2][..n]
        }
    };
    let (m2, m1, c0, p1, p2, p3) = (row(-2), row(-1), row(0), row(1), row(2), row(3));
    for x in 0..n {
        let t = |r: &[u8]| u32::from(r[x]);
        out[x] = filter.apply([t(m2), t(m1), t(c0), t(p1), t(p2), t(p3)]) as u8;
    }
}

fn reduce_row_in_place(filter: ReduceFilter, row: &mut [u8], width: usize) {
    if width == 0 || row.len() < 2 {
        return;
    }
    let at = |row: &[u8], i: isize| u32::from(row.get(i as usize).copied().unwrap_or(0));
    row[0] = ((at(row, 0) + at(row, 1) + 1) >> 1) as u8;
    let tail = match filter {
        ReduceFilter::Average => width,
        _ => (width - 1).max(1),
    };
    for x in 1..tail.min(row.len()) {
        let base = 2 * x as isize;
        let value = match filter {
            ReduceFilter::Average => {
                (at(row, base - 1) + 2 * at(row, base) + at(row, base + 1) + 2) >> 2
            }
            _ => filter.apply(std::array::from_fn(|i| at(row, base + i as isize - 2))),
        };
        row[x] = value as u8;
    }
    for x in tail..width.min(row.len()) {
        row[x] = ((at(row, 2 * x as isize) + at(row, 2 * x as isize + 1) + 1) >> 1) as u8;
    }
}

fn refine(canvas: &mut Canvas<'_>, level0: Region, pitch: usize, pel: usize, upscale: Upscale) {
    let plane = |k: usize| level0.offset + k * pitch * level0.height;
    let (w, h) = (level0.width, level0.height);
    let full = Interp {
        pitch,
        width: w,
        height: h,
    };
    let (horizontal, vertical, diagonal) = if pel == 2 { (1, 2, 3) } else { (2, 8, 10) };
    match upscale {
        Upscale::Bilinear => {
            full.pair(canvas, plane(horizontal), plane(0), plane(0) + 1);
            full.pair(canvas, plane(vertical), plane(0), plane(0) + pitch);
            full.copy_first_row(canvas, plane(vertical), plane(0));
            full.pair(canvas, plane(diagonal), plane(0), plane(0) + pitch + 1);
            full.copy_first_row(canvas, plane(diagonal), plane(0) + 1);
        }
        Upscale::Bicubic => {
            full.horizontal(canvas, plane(horizontal), plane(0), HalfPel::CatmullRom);
            full.vertical(canvas, plane(vertical), plane(0), HalfPel::CatmullRom);
            full.horizontal(
                canvas,
                plane(diagonal),
                plane(vertical),
                HalfPel::CatmullRom,
            );
        }
        Upscale::Wiener => {
            full.horizontal(canvas, plane(horizontal), plane(0), HalfPel::Wiener);
            full.vertical(canvas, plane(vertical), plane(0), HalfPel::Wiener);
            full.horizontal(canvas, plane(diagonal), plane(vertical), HalfPel::Wiener);
        }
    }
    if pel == 2 {
        return;
    }
    let short_row = Interp {
        width: w - 1,
        ..full
    };
    let short_col = Interp {
        height: h - 1,
        ..full
    };
    full.pair(canvas, plane(1), plane(0), plane(2));
    full.pair(canvas, plane(9), plane(8), plane(10));
    full.pair(canvas, plane(4), plane(0), plane(8));
    full.pair(canvas, plane(6), plane(2), plane(10));
    full.pair(canvas, plane(5), plane(4), plane(6));
    short_row.pair(canvas, plane(3), plane(0) + 1, plane(2));
    short_row.pair(canvas, plane(11), plane(8) + 1, plane(10));
    short_col.pair(canvas, plane(12), plane(0) + pitch, plane(8));
    short_col.pair(canvas, plane(14), plane(2) + pitch, plane(10));
    full.pair(canvas, plane(13), plane(12), plane(14));
    short_row.pair(canvas, plane(7), plane(4) + 1, plane(6));
    short_row.pair(canvas, plane(15), plane(12) + 1, plane(14));
}

#[derive(Clone, Copy)]
enum HalfPel {
    CatmullRom,
    Wiener,
}

impl HalfPel {
    const fn edges(self) -> (usize, usize) {
        match self {
            Self::CatmullRom => (1, 3),
            Self::Wiener => (2, 4),
        }
    }

    fn apply(self, t: [i32; 6]) -> u8 {
        let [m2, m1, c0, p1, p2, p3] = t;
        let value = match self {
            Self::CatmullRom => (-(m1 + p2) + (c0 + p1) * 9 + 8) >> 4,
            Self::Wiener => (m2 + (-m1 + (c0 << 2) + (p1 << 2) - p2) * 5 + p3 + 16) >> 5,
        };
        value.clamp(0, 255) as u8
    }
}

#[derive(Clone, Copy)]
struct Interp {
    pitch: usize,
    width: usize,
    height: usize,
}

impl Interp {
    fn pair(self, canvas: &mut Canvas<'_>, dst: usize, first: usize, second: usize) {
        let span = self.width.div_ceil(SSE_CHUNK) * SSE_CHUNK;
        let (mut a, mut b) = (vec![0u8; span], vec![0u8; span]);
        for y in 0..self.height {
            let row = y * self.pitch;
            canvas.fetch((first + row) as isize, &mut a);
            canvas.fetch((second + row) as isize, &mut b);
            for (x, y) in a.iter_mut().zip(&b) {
                *x = ((u16::from(*x) + u16::from(*y) + 1) >> 1) as u8;
            }
            canvas.store(dst + row, &a);
        }
    }

    fn copy_first_row(self, canvas: &mut Canvas<'_>, dst: usize, src: usize) {
        let mut row = vec![0u8; self.width];
        canvas.fetch(src as isize, &mut row);
        canvas.store(dst, &row);
    }

    fn horizontal(self, canvas: &mut Canvas<'_>, dst: usize, src: usize, kernel: HalfPel) {
        let (lead, tail) = kernel.edges();
        let w = self.width;
        let inner = w.saturating_sub(tail).saturating_sub(lead);
        let mut padded = vec![0u8; w + 5];
        let mut out = vec![0u8; w];
        for y in 0..self.height {
            let row = y * self.pitch;
            canvas.fetch((src + row) as isize - 2, &mut padded);
            let taps: [&[u8]; 6] = std::array::from_fn(|k| &padded[lead + k..][..inner]);
            apply_half_pel(kernel, &mut out[lead..lead + inner], taps);
            let pixel = &padded[2..];
            for i in (0..lead).chain(lead + inner..w.saturating_sub(1)) {
                out[i] = ((u16::from(pixel[i]) + u16::from(pixel[i + 1]) + 1) >> 1) as u8;
            }
            out[w - 1] = pixel[w - 1];
            canvas.store(dst + row, &out);
        }
    }

    fn vertical(self, canvas: &mut Canvas<'_>, dst: usize, src: usize, kernel: HalfPel) {
        let (lead, tail) = kernel.edges();
        let (w, h, p) = (self.width, self.height, self.pitch as isize);
        let mut rows = [const { Vec::new() }; 6];
        let mut out = vec![0u8; w];
        for j in 0..h {
            let base = (src + j * self.pitch) as isize;
            for (k, row) in rows.iter_mut().enumerate() {
                row.resize(w, 0);
                canvas.fetch(base + (k as isize - 2) * p, row);
            }
            if j + 1 == h {
                out.copy_from_slice(&rows[2]);
            } else if j < lead || j >= h.saturating_sub(tail) {
                vertical_average(&mut out, &rows[2], &rows[3]);
            } else {
                apply_half_pel(kernel, &mut out, std::array::from_fn(|k| &rows[k][..w]));
            }
            canvas.store(dst + j * self.pitch, &out);
        }
    }
}

fn apply_half_pel(kernel: HalfPel, out: &mut [u8], taps: [&[u8]; 6]) {
    let n = out.len();
    let [m2, m1, c0, p1, p2, p3] = taps.map(|t| &t[..n]);
    for i in 0..n {
        let t = |r: &[u8]| i32::from(r[i]);
        out[i] = kernel.apply([t(m2), t(m1), t(c0), t(p1), t(p2), t(p3)]);
    }
}
