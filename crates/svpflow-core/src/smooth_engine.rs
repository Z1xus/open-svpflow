#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::fn_params_excessive_bools,
    clippy::too_many_arguments
)]

use std::ops::{Deref, Range};
use std::sync::Arc;

use crate::frame_math::{Timing, gcd};
use crate::metadata::{self, VectorData, VectorRecord};
use crate::params;
use crate::smooth::{
    Algo, FieldShape, Frame, FrameMut, PackedVectors, RenderShape, Renderer, SceneLimits,
    VectorField,
};
use crate::smooth_options::{Options, ReferenceParams, SourceInfo};

pub type MotionSet = [Vec<u16>; 8];

pub struct Rates {
    num: u64,
    den: u64,
}

pub struct Step {
    pub n: i32,
    pub raw: f64,
    pub phase: i32,
}

impl Rates {
    pub fn new(params: &ReferenceParams, source: &SourceInfo) -> Self {
        let src_num = u64::try_from(source.fps_num.max(1)).unwrap_or(1);
        let src_den = u64::try_from(source.fps_den.max(1)).unwrap_or(1);
        let (out_num, out_den) = if params.absolute {
            (params.rate_num, params.rate_den)
        } else {
            (src_num * params.rate_num, src_den * params.rate_den)
        };
        let (num, den) = (out_num * src_den, out_den * src_num);
        let common = gcd(num, den).max(1);
        Self {
            num: num / common,
            den: den / common,
        }
    }

    pub fn source_frame(&self, frame: i32) -> i32 {
        let frame = u64::try_from(frame.max(0)).unwrap_or(0);
        i32::try_from(frame * self.den / self.num).unwrap_or(i32::MAX)
    }

    pub fn raw_phase(&self, frame: i32, source: i32) -> f64 {
        256.0
            * (f64::from(frame) * f64::from(self.den as i32) / self.num as f64 - f64::from(source))
    }

    pub fn ratio(&self) -> f64 {
        self.num as f64 / self.den as f64
    }

    pub fn time_to_fixed(&self, raw: f64, mode: i32) -> i32 {
        let step = f64::from(self.den as i32) * 256.0 / self.num as f64;
        let mut left = (raw / step).floor() as i32;
        let rem = raw - f64::from(left) * step;
        let mut right = ((256.0 - raw - 0.001) / step).floor() as i32;
        let whole = if mode <= 1 {
            left += i32::from(rem as i32 >= (rem - step).abs() as i32);
            let tail = 256.0 - (f64::from(right) * step + raw);
            if tail as i32 > (tail - step).abs() as i32 {
                right += 1;
            }
            left << 8
        } else {
            let mut shifted = left << 8;
            if rem as i32 <= 0 {
                if left <= 0 {
                    shifted = 0;
                    left = 0;
                } else {
                    left -= 1;
                    shifted = left << 8;
                }
            }
            if (256.0 - (f64::from(right) * step + raw) - step).abs() < 0.1 {
                right += 1;
            }
            shifted
        };
        let total = left + right;
        if total == 0 {
            return 0;
        }
        let result = whole / total;
        if mode & !2 != 0 {
            if right >= left {
                result / 2
            } else {
                256 - (right << 7) / total
            }
        } else {
            result
        }
    }

    pub fn step(&self, params: &ReferenceParams, frame: i32) -> Step {
        let n = self.source_frame(frame);
        let raw = self.raw_phase(frame, n);
        let mut phase = (raw + 0.5) as i32;
        if matches!(params.scene_mode, 1 | 2) {
            phase = self.time_to_fixed(raw, if params.scene_mode == 1 { 0 } else { 2 });
        }
        Step { n, raw, phase }
    }
}

pub const fn needs_vectors(phase: i32) -> bool {
    phase & !0x100 != 0
}

pub const fn vector_reach(params: &ReferenceParams) -> i32 {
    if params.level >= 2 { 2 } else { 1 }
}

pub const fn limits(params: &ReferenceParams) -> SceneLimits {
    SceneLimits {
        blocks: params.blocks,
        zero: params.zero,
        m1: params.m1,
        m2: params.m2,
        scene: params.scene,
    }
}

pub fn quality(
    field: &VectorField,
    params: &ReferenceParams,
    limits: &SceneLimits,
    table: &mut Option<[u8; 511]>,
    luma: &mut [u8],
) -> (i32, Vec<u8>) {
    let table = table.get_or_insert_with(|| field.luma_table(params.luma));
    field.avg_luma(luma, table);
    let class = field.quality(false, limits, luma);
    let mut cached = luma.to_vec();
    if let Some(tail) = cached.len().checked_sub(4) {
        let mean = (field.luma_geomean() * 1_000_000.0) as i32;
        cached[tail..].copy_from_slice(&mean.to_le_bytes());
    }
    (class, cached)
}

pub fn classify(
    params: &ReferenceParams,
    n: i32,
    phase: i32,
    mut quality: impl FnMut(i32) -> i32,
) -> (i32, bool) {
    if !needs_vectors(phase) {
        return (3, false);
    }
    let algo = params.algo;
    let level = params.level;
    let (prev, next, prev2, next2) = if (level == 0 && algo != 23 && algo <= 89) || n <= 1 {
        (3, 3, -1, -1)
    } else {
        let prev = quality(n - 1);
        let next = quality(n + 1);
        if level <= 1 || n == 2 {
            (prev, next, -1, -1)
        } else {
            let prev2 = quality(n - 2);
            let next2 = if level > 2 { quality(n + 2) } else { -1 };
            if prev2 == 3 || next2 == 3 {
                (prev, next, -1, -1)
            } else {
                (prev, next, prev2, next2)
            }
        }
    };
    let current = quality(n);
    let mut class = current;
    if level != 0 && prev != 3 && next != 3 && current != 3 {
        let sum = next + prev + current;
        let average = if prev2 < 0 {
            f64::from(sum as f32) / 3.0
        } else if next2 < 0 {
            f64::from((sum + prev2) as f32) * 0.25
        } else {
            f64::from((sum + prev2 + next2) as f32) / 5.0
        };
        class = (f64::from(average as f32) + 0.5).floor() as i32;
    }
    (class, prev <= 2 && next <= 2)
}

pub fn adapt_phase(
    params: &ReferenceParams,
    rates: &Rates,
    raw: f64,
    class: i32,
    phase: i32,
) -> i32 {
    if params.scene_mode == 3 && (0..=2).contains(&class) {
        let adaptive = params.adaptive[class as usize];
        if adaptive >= 0 {
            return if rates.ratio() >= 2.0 {
                rates.time_to_fixed(raw, adaptive)
            } else {
                0
            };
        }
    }
    phase
}

pub fn copy_choice(
    params: &ReferenceParams,
    phase: i32,
    class: i32,
    debug_vectors: bool,
) -> Option<bool> {
    if phase == 0 {
        Some(false)
    } else if phase == 256 {
        Some(true)
    } else if (class == 3 && !params.blend) || debug_vectors {
        Some(phase > 127)
    } else {
        None
    }
}

pub fn directions(params: &ReferenceParams, phase: i32, class: i32) -> (bool, bool, bool) {
    let algo = params.algo;
    let (fwd, bwd) = if class == 3 {
        (false, false)
    } else if algo == 2 {
        (phase > 127, phase <= 127)
    } else {
        (true, algo > 10)
    };
    (fwd, bwd, class > 0 && params.force13)
}

pub const fn render_time(phase: i32, use_fwd: bool, use_bwd: bool) -> i32 {
    if use_fwd && !use_bwd {
        256 - phase
    } else {
        phase
    }
}

pub struct Masks {
    pub cover_bwd: Vec<u8>,
    pub cover_fwd: Vec<u8>,
    pub sad: Option<(Vec<u8>, Vec<u8>)>,
}

pub fn masks(field: &VectorField, params: &ReferenceParams, time: i32, neither: bool) -> Masks {
    let shape = *field.shape();
    let (width, height) = (shape.packed_width(), shape.packed_height());
    let count = usize::try_from(width * height).unwrap_or(0);
    let mut cover_bwd = vec![0u8; count];
    let mut cover_fwd = vec![0u8; count];
    if params.algo > 20 && !neither {
        let mut scratch = Vec::new();
        field.cover_mask(
            &mut scratch,
            false,
            &mut cover_bwd,
            params.cover,
            256 - time,
            width,
            height,
        );
        field.cover_mask(
            &mut scratch,
            true,
            &mut cover_fwd,
            params.cover,
            time,
            width,
            height,
        );
    }
    let sad = match params.sad {
        Some((scale, sharp)) if !neither => {
            let mut first = vec![0u8; count];
            let mut second = vec![0u8; count];
            field.sad_mask(false, &mut first, scale, sharp, width, height);
            field.sad_mask(true, &mut second, scale, sharp, width, height);
            Some((first, second))
        }
        _ => None,
    };
    Masks {
        cover_bwd,
        cover_fwd,
        sad,
    }
}

pub fn motion_key(n: i32, use_fwd: bool, use_bwd: bool, extended: bool, zero: (bool, bool)) -> i64 {
    (i64::from(n) << 8)
        | i64::from(use_fwd)
        | i64::from(use_bwd) << 1
        | i64::from(extended) << 2
        | i64::from(zero.0) << 3
        | i64::from(zero.1) << 4
}

pub fn motions<F: Deref<Target = VectorField>>(
    field: &VectorField,
    gpu: bool,
    use_fwd: bool,
    use_bwd: bool,
    extended: bool,
    (zerox, zeroy): (bool, bool),
    mut load: impl FnMut(i32) -> Option<F>,
    n: i32,
) -> MotionSet {
    let shape = *field.shape();
    let (width, height) = (shape.packed_width(), shape.packed_height());
    let count = usize::try_from(width * height).unwrap_or(0);
    let unset = u16::from(!gpu) * 1024;
    let mut arrays: MotionSet = std::array::from_fn(|_| vec![unset; count]);
    let [fwd_x, fwd_y, bwd_x, bwd_y, next_x, next_y, prev_x, prev_y] = &mut arrays;
    if use_fwd {
        field.pack(false, fwd_x, fwd_y, width, height);
    }
    if use_bwd {
        field.pack(true, bwd_x, bwd_y, width, height);
    }
    if !use_fwd && !use_bwd {
        for buf in [&mut *fwd_x, &mut *fwd_y, &mut *bwd_x, &mut *bwd_y] {
            buf.fill(1024);
        }
    }
    if extended {
        let prev = load(n - 1);
        let last = prev.as_deref().unwrap_or(field);
        last.pack(true, prev_x, prev_y, width, height);
        let next = load(n + 1);
        next.as_deref()
            .unwrap_or(last)
            .pack(false, next_x, next_y, width, height);
    }
    if zerox {
        for buf in [&mut *fwd_x, &mut *bwd_x, &mut *next_x, &mut *prev_x] {
            buf.fill(1024);
        }
    }
    if zeroy {
        for buf in [&mut *fwd_y, &mut *bwd_y, &mut *next_y, &mut *prev_y] {
            buf.fill(1024);
        }
    }
    arrays
}

pub const fn select(algo: i32, use_fwd: bool, use_bwd: bool, extended: bool, force13: bool) -> i32 {
    let mut selected = algo;
    if selected == 2 {
        if use_fwd && !use_bwd {
            selected = 1;
        }
    } else if selected == 23 || selected > 10 {
        if selected == 23 && !extended {
            selected = 21;
        }
        if force13 {
            selected = 13;
        }
    }
    if !use_fwd && !use_bwd {
        selected = 11;
    }
    selected
}

pub fn merged_sad(selected: i32, sad: Option<&(Vec<u8>, Vec<u8>)>) -> Option<Vec<u8>> {
    sad.map(|(first, second)| match selected {
        1 => first.clone(),
        2 => second.clone(),
        _ => first.iter().zip(second).map(|(a, b)| *a.max(b)).collect(),
    })
}

pub const fn kind(selected: i32, sad_on: bool) -> Algo {
    match selected {
        1 if sad_on => Algo::FastSad { next: true },
        2 if sad_on => Algo::FastSad { next: false },
        11 if sad_on => Algo::NoMaskSad { median: false },
        13 if sad_on => Algo::NoMaskSad { median: true },
        21 if sad_on => Algo::NormalSad { simple: true },
        22 if sad_on => Algo::NormalSad { simple: false },
        23 if sad_on => Algo::ExtendedSad,
        1 => Algo::Fast { next: true },
        2 => Algo::Fast { next: false },
        11 => Algo::NoMask { median: false },
        13 => Algo::NoMask { median: true },
        21 => Algo::Normal { simple: true },
        22 => Algo::Normal { simple: false },
        23 => Algo::Extended,
        _ => Algo::Fill,
    }
}

pub fn render_shape(data: &VectorData, width: i32, height: i32, blend: f64) -> RenderShape {
    let shape = FieldShape::from_data(data);
    let block = data.effective_block();
    RenderShape {
        width,
        height,
        step_x: block.width,
        step_y: block.height,
        grid_w: shape.packed_width(),
        grid_h: shape.packed_height(),
        origin_x: shape.block_w / 2,
        origin_y: shape.block_h / 2,
        pel: shape.pel,
        blend,
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuParams {
    pub algorithm: i32,
    pub width: i32,
    pub height: i32,
    pub x_ratio: i32,
    pub y_ratio: i32,
    pub pel: i32,
    pub block_w: i32,
    pub block_h: i32,
    pub origin_x: i32,
    pub origin_y: i32,
    pub phase: i32,
    pub has_sad: i32,
    pub linear_luma: i32,
    pub cubic: i32,
    pub cubic_ref: i32,
    pub offset_x: i32,
    pub offset_y: i32,
    pub sad_blend: f32,
    pub dither: i32,
}

pub const fn gpu_linear(params: &ReferenceParams, width: i32) -> bool {
    params.linear && width < 3001
}

pub fn gpu_params(
    params: &ReferenceParams,
    data: &VectorData,
    (width, height): (i32, i32),
    selected: i32,
    time: i32,
    has_sad: bool,
    dither: bool,
) -> [GpuParams; 3] {
    let shape = FieldShape::from_data(data);
    let (grid_w, grid_h) = (shape.packed_width(), shape.packed_height());
    let gpu_time = if selected == 1 { 256 - time } else { time };
    let fraction = (f64::from(gpu_time) * 0.003_906_25) as f32;
    let blend = params.area_blend as f32;
    let sad_blend = if gpu_time > 126 {
        (1.0 - f64::from(blend * (1.0 - f64::from(fraction)) as f32)) as f32
    } else {
        fraction * blend
    };
    let cubic = params.cubic & 1 != 0;
    let cubic_ref = params.cubic & 2 != 0;
    let linear = gpu_linear(params, width);
    let block = data.effective_block();
    let make = |chroma: bool, offset_x: i32, offset_y: i32| GpuParams {
        algorithm: selected,
        width: if chroma { width / 2 } else { width },
        height: if chroma { height / 2 } else { height },
        x_ratio: if chroma { 2 } else { 1 },
        y_ratio: if chroma { 2 } else { 1 },
        pel: shape.pel,
        block_w: if cubic {
            block.width
        } else {
            block.width * grid_w
        },
        block_h: if cubic {
            block.height
        } else {
            block.height * grid_h
        },
        origin_x: shape.overlap_x / 2,
        origin_y: shape.overlap_y / 2,
        phase: gpu_time,
        has_sad: i32::from(has_sad),
        linear_luma: i32::from(!chroma && linear),
        cubic: i32::from(cubic),
        cubic_ref: i32::from(cubic_ref),
        offset_x,
        offset_y,
        sad_blend,
        dither: i32::from(dither),
    };
    [
        make(false, 0, 0),
        make(true, 0, height),
        make(true, width / 2, height),
    ]
}

pub fn interleave_pel(
    subplanes: &[u8],
    width: usize,
    height: usize,
    pel: usize,
) -> Option<Vec<u8>> {
    if !matches!(pel, 2 | 4) {
        return None;
    }
    let span = width.checked_mul(height)?;
    let out_stride = width * pel;
    let mut out = vec![0u8; span.checked_mul(pel * pel)?];
    for sy in 0..pel {
        for row in 0..height {
            let at = (row * pel + sy) * out_stride;
            let dst = out.get_mut(at..at + out_stride)?;
            for sx in 0..pel {
                let base = (sy * pel + sx) * span + row * width;
                let src = subplanes.get(base..base + width)?;
                for (chunk, &value) in dst.chunks_exact_mut(pel).zip(src) {
                    chunk[sx] = value;
                }
            }
        }
    }
    Some(out)
}

pub struct Engine {
    params: ReferenceParams,
    rates: Rates,
    timing: Timing,
    limits: SceneLimits,
    data: VectorData,
    shape: FieldShape,
    render: RenderShape,
    zero: (bool, bool),
    dither: bool,
    width: i32,
    height: i32,
    table: Option<[u8; 511]>,
    quality: Vec<(i32, i32, Vec<u8>)>,
    fields: Vec<(i32, Arc<VectorField>)>,
}

pub enum Output {
    Copy(i32),
    Render(Box<Job>),
    Gpu(Box<GpuJob>),
}

pub struct GpuJob {
    pub n: i32,
    pub params: [GpuParams; 3],
    pub linear: bool,
    pub grid: (usize, usize),
    pub base: Vec<u16>,
    pub ext: Option<Vec<u16>>,
    pub mask: Option<Vec<u8>>,
}

struct Planned {
    n: i32,
    time: i32,
    neither: bool,
    selected: i32,
    motions: MotionSet,
    masks: Masks,
}

pub struct Job {
    renderer: Renderer,
    algo: Algo,
    interp: bool,
    motions: MotionSet,
    masks: Masks,
    sad: Option<Vec<u8>>,
    pub n: i32,
}

impl Engine {
    pub fn new(options: &str, source: &SourceInfo, header: &[u8]) -> Result<Self, String> {
        if source.width <= 0
            || source.height <= 0
            || source.width % 2 != 0
            || source.height % 2 != 0
        {
            return Err("dimensions must be positive and even".into());
        }
        let value = params::parse(options.as_bytes())?;
        let mut options = Options::from_value(&value, 0);
        if options.validate(0).is_err() || options.validate_timing(0, source).is_err() {
            return Err("invalid smooth options".into());
        }
        options.normalize_scene_mode(0, source);
        options.apply_source_depth(0);
        let VectorRecord::Ready(data) = metadata::vector_data(header) else {
            return Err("invalid vector header".into());
        };
        options.scale_scene_limits(data.block);
        options.apply_mask_area_scale(false);
        options.apply_debug_mask_scale();
        options.apply_mask_cover_algo(0);
        options.apply_cubic_default(!data.rejects_cubic());
        if !options.reference_supported() {
            return Err("unsupported smooth options".into());
        }
        if options.padding(source) != (0, 0) {
            return Err("light and aspect padding are not supported".into());
        }
        let params = options.reference();
        let render = render_shape(&data, source.width, source.height, params.area_blend);
        Ok(Self {
            rates: Rates::new(&params, source),
            timing: options.timing(source),
            limits: limits(&params),
            shape: FieldShape::from_data(&data),
            zero: (options.debug_zerox(), options.debug_zeroy()),
            dither: options.dither(),
            width: source.width,
            height: source.height,
            params,
            data,
            render,
            table: None,
            quality: Vec::new(),
            fields: Vec::new(),
        })
    }

    pub fn output_frames(&self, source_frames: i32) -> i32 {
        self.timing.scale_frame_count(source_frames)
    }

    pub const fn cpu_supported(&self) -> bool {
        !self.data.has_odd_overlap()
    }

    pub fn uses_super(&self) -> bool {
        self.shape.pel > 1 && !self.data.marker_is_one()
    }

    pub const fn pel(&self) -> i32 {
        self.shape.pel
    }

    pub fn plan(&self, frame: i32) -> (i32, Range<i32>) {
        let step = self.rates.step(&self.params, frame);
        let n = step.n;
        if !needs_vectors(step.phase) {
            return (n, n..n);
        }
        let reach = vector_reach(&self.params);
        (n, (n - reach).max(0)..n + reach + 1)
    }

    pub fn prepare<'a>(&mut self, frame: i32, vectors: impl Fn(i32) -> Option<&'a [u8]>) -> Output {
        let planned = match self.plan_frame(frame, vectors, false) {
            Ok(planned) => planned,
            Err(k) => return Output::Copy(k),
        };
        let sad = merged_sad(planned.selected, planned.masks.sad.as_ref());
        let mut renderer = Renderer::new(self.render);
        renderer.set_time(planned.time);
        Output::Render(Box::new(Job {
            renderer,
            algo: kind(planned.selected, sad.is_some()),
            interp: !planned.neither && !self.params.block,
            motions: planned.motions,
            masks: planned.masks,
            sad,
            n: planned.n,
        }))
    }

    pub fn prepare_gpu<'a>(
        &mut self,
        frame: i32,
        vectors: impl Fn(i32) -> Option<&'a [u8]>,
    ) -> Output {
        let planned = match self.plan_frame(frame, vectors, true) {
            Ok(planned) => planned,
            Err(k) => return Output::Copy(k),
        };
        let (grid_w, grid_h) = (self.shape.packed_width(), self.shape.packed_height());
        let count = usize::try_from(grid_w * grid_h).unwrap_or(0);
        let [fwd_x, fwd_y, bwd_x, bwd_y, next_x, next_y, prev_x, prev_y] = &planned.motions;
        let pack = |a: &[u16], b: &[u16], c: &[u16], d: &[u16]| {
            (0..count)
                .flat_map(|i| [a[i], b[i], c[i], d[i]])
                .collect::<Vec<u16>>()
        };
        let base = pack(bwd_x, fwd_y, fwd_x, bwd_y);
        let ext = (planned.selected == 23).then(|| pack(prev_x, next_y, next_x, prev_y));
        let masks = &planned.masks;
        let coverage = planned.selected >= 21;
        let mask = (coverage || masks.sad.is_some()).then(|| {
            (0..count)
                .flat_map(|i| {
                    let (first, second) = masks
                        .sad
                        .as_ref()
                        .map_or((0, 0), |(first, second)| (first[i], second[i]));
                    let (fwd, bwd) = if coverage {
                        (masks.cover_fwd[i], masks.cover_bwd[i])
                    } else {
                        (0, 0)
                    };
                    [first, fwd, bwd, second]
                })
                .collect()
        });
        Output::Gpu(Box::new(GpuJob {
            n: planned.n,
            params: gpu_params(
                &self.params,
                &self.data,
                (self.width, self.height),
                planned.selected,
                planned.time,
                masks.sad.is_some(),
                self.dither,
            ),
            linear: gpu_linear(&self.params, self.width),
            grid: (
                usize::try_from(grid_w).unwrap_or(0),
                usize::try_from(grid_h).unwrap_or(0),
            ),
            base,
            ext,
            mask,
        }))
    }

    fn field<'a>(
        &mut self,
        k: i32,
        vectors: &impl Fn(i32) -> Option<&'a [u8]>,
    ) -> Option<Arc<VectorField>> {
        if let Some((_, field)) = self.fields.iter().find(|(key, _)| *key == k) {
            return Some(Arc::clone(field));
        }
        let mut field = VectorField::new(self.shape);
        if !vectors(k).is_some_and(|p| field.update(p)) {
            return None;
        }
        let field = Arc::new(field);
        if self.fields.len() >= 8 {
            self.fields.remove(0);
        }
        self.fields.push((k, Arc::clone(&field)));
        Some(field)
    }

    fn plan_frame<'a>(
        &mut self,
        frame: i32,
        vectors: impl Fn(i32) -> Option<&'a [u8]>,
        gpu: bool,
    ) -> Result<Planned, i32> {
        let params = self.params;
        let step = self.rates.step(&params, frame);
        let (n, raw) = (step.n, step.raw);
        let mut phase = step.phase;
        let mut luma = vec![0u8; self.shape.blocks()];
        let (class, neighbors_ok) = classify(&params, n, phase, |k| {
            if let Some((_, class, _)) = self.quality.iter().find(|(key, _, _)| *key == k) {
                return *class;
            }
            let Some(field) = self.field(k, &vectors) else {
                return 3;
            };
            let (class, cached) =
                quality(&field, &params, &self.limits, &mut self.table, &mut luma);
            if self.quality.len() >= 64 {
                self.quality.remove(0);
            }
            self.quality.push((k, class, cached));
            class
        });
        phase = adapt_phase(&params, &self.rates, raw, class, phase);
        if let Some(next) = copy_choice(&params, phase, class, false) {
            return Err(n + i32::from(next));
        }
        let (use_fwd, use_bwd, force13) = directions(&params, phase, class);
        let field = self
            .field(n, &vectors)
            .unwrap_or_else(|| Arc::new(VectorField::new(self.shape)));
        let neither = !use_fwd && !use_bwd;
        let time = render_time(phase, use_fwd, use_bwd);
        let masks = masks(&field, &params, time, neither);
        let extended = neighbors_ok && params.algo == 23 && !neither;
        let motions = motions(
            &field,
            gpu,
            use_fwd,
            use_bwd,
            extended,
            self.zero,
            |k| self.field(k, &vectors),
            n,
        );
        Ok(Planned {
            n,
            time,
            neither,
            selected: select(params.algo, use_fwd, use_bwd, extended, force13),
            motions,
            masks,
        })
    }
}

impl Job {
    pub const fn blocks(&self) -> Range<i32> {
        -1..self.renderer.grid_h()
    }

    pub fn band_rows(&self, blocks: Range<i32>) -> [Range<usize>; 2] {
        self.renderer.band_rows(self.interp, blocks)
    }

    pub fn render_band(
        &self,
        dst: &mut FrameMut<'_>,
        current: Frame<'_>,
        next: Frame<'_>,
        blocks: Range<i32>,
    ) {
        let [fwd_x, fwd_y, bwd_x, bwd_y, next_x, next_y, prev_x, prev_y] =
            self.motions.each_ref().map(Vec::as_slice);
        let vectors = PackedVectors {
            fwd_x,
            fwd_y,
            bwd_x,
            bwd_y,
            next_fwd_x: next_x,
            next_fwd_y: next_y,
            prev_bwd_x: prev_x,
            prev_bwd_y: prev_y,
            cover_bwd: &self.masks.cover_bwd,
            cover_fwd: &self.masks.cover_fwd,
            sad: self.sad.as_deref().unwrap_or(&[]),
        };
        self.renderer
            .render_band(self.algo, self.interp, dst, next, current, &vectors, blocks);
    }
}
