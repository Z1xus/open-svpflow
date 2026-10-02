use crate::{metadata, params::Value};

#[derive(Clone, Copy, Debug)]
pub struct SourceInfo {
    pub fps_num: i64,
    pub fps_den: i64,
    pub width: i32,
    pub height: i32,
}

pub use crate::frame_math::Timing;

#[derive(Default)]
#[allow(dead_code)]
pub struct Options {
    rate: Rate,
    light: Light,
    scene: Scene,
    mask: Mask,
    debug: DebugOptions,
    render: RenderOptions,
    hdr: HdrOptions,
    nvof: NvofOptions,
    gpu: GpuOptions,
    algo: Option<i64>,
    block: bool,
    cubic: Option<i64>,
    fallback: bool,
    mt: i64,
}

#[derive(Default)]
struct Rate {
    absolute: bool,
    num: Option<i64>,
    den: Option<i64>,
}

#[derive(Default)]
struct Light {
    sar: Option<f64>,
    aspect: Option<f64>,
    zoom: Option<f64>,
    border: Option<i64>,
    lights: Option<i64>,
    length: Option<i64>,
    cell: Option<f64>,
}

#[allow(dead_code, clippy::struct_excessive_bools)]
#[derive(Clone, Copy, Default)]
pub struct DebugOptions {
    pub flags: i64,
    pub vectors: bool,
    pub qmap: bool,
    pub qmode: bool,
    pub zerox: bool,
    pub zeroy: bool,
    pub tt: bool,
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
pub struct RenderOptions {
    pub linear: bool,
    pub dither: bool,
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
pub struct HdrOptions {
    pub mluminance: f64,
    pub contrast: f64,
    pub adaptive: bool,
    pub dovi: bool,
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
pub struct NvofOptions {
    pub q: i64,
    pub gpuid: i64,
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
pub struct GpuOptions {
    pub render_cpu: bool,
    pub id: i64,
    pub qn: i64,
    pub api: i64,
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
pub struct Mask {
    pub cover: i64,
    pub area: i64,
    pub area_enabled: bool,
    pub area_scale: f64,
    pub area_sharp: f64,
    pub area_blend: f64,
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
pub struct Scene {
    pub mode: i64,
    pub blend: bool,
    pub adaptive: [i32; 3],
    pub force13: bool,
    pub limits: SceneLimits,
    pub qmap_limits: SceneLimits,
    pub luma: f64,
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
pub struct SceneLimits {
    pub blocks: i64,
    pub blocks13: i64,
    pub ignore: f64,
    pub zero: i64,
    pub m1: i64,
    pub m2: i64,
    pub scene: i64,
}

#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Copy)]
pub struct ReferenceParams {
    pub absolute: bool,
    pub rate_num: u64,
    pub rate_den: u64,
    pub scene_mode: i64,
    pub adaptive: [i32; 3],
    pub blocks: i32,
    pub zero: i32,
    pub m1: i32,
    pub m2: i32,
    pub scene: i32,
    pub luma: f64,
    pub level: i32,
    pub force13: bool,
    pub blend: bool,
    pub cover: i32,
    pub area_blend: f64,
    pub algo: i32,
    pub linear: bool,
    pub cubic: i64,
    pub block: bool,
    pub sad: Option<(f64, f64)>,
}

#[derive(Clone, Copy)]
pub struct LightParams {
    pub border: i32,
    pub lights: i32,
    pub length: i32,
    pub cell: f64,
}

impl Options {
    pub fn for_mode(mode: i32) -> Self {
        Self {
            scene: Scene::for_mode(mode),
            ..Self::default()
        }
    }

    pub fn from_value(value: &Value, mode: i32) -> Self {
        let hdr = HdrOptions::from_value(value);
        let render = RenderOptions::from_value(value);
        Self {
            rate: Rate {
                absolute: value.bool_at(&["rate", "abs"]).unwrap_or(false),
                num: value.int_at(&["rate", "num"]),
                den: value.int_at(&["rate", "den"]),
            },
            light: Light {
                sar: value.float_at(&["light", "sar"]),
                aspect: value.float_at(&["light", "aspect"]),
                zoom: value.float_at(&["light", "zoom"]),
                border: value.int_at(&["light", "border"]),
                lights: value.int_at(&["light", "lights"]),
                length: value.int_at(&["light", "length"]),
                cell: value.float_at(&["light", "cell"]),
            },
            scene: Scene::from_value(value, mode),
            mask: Mask::from_value(value),
            debug: DebugOptions::from_value(value),
            render,
            hdr,
            nvof: NvofOptions::from_value(value),
            gpu: GpuOptions::from_value(value),
            algo: value.int_at(&["algo"]),
            block: value.bool_at(&["block"]).unwrap_or(false),
            cubic: value.int_at(&["cubic"]),
            fallback: value.bool_at(&["fallback"]).unwrap_or(false),
            mt: value.int_at(&["mt"]).unwrap_or(0),
        }
    }

    pub fn validate(&self, mode: i32) -> Result<(), ValidateError> {
        if mode <= 1 && !valid_algo(self.algo.unwrap_or(21)) {
            return Err(ValidateError::Algo);
        }
        if mode <= 1 && !matches!(self.scene.mode, 0..=3) {
            return Err(ValidateError::SceneMode);
        }
        Ok(())
    }

    pub fn validate_timing(&self, mode: i32, source: &SourceInfo) -> Result<(), ValidateError> {
        if mode > 1 {
            return Ok(());
        }
        let timing = self.timing(source);
        match self.scene.mode {
            2 if below_ratio(&timing, 2) => Err(ValidateError::SceneModeRate),
            1 if below_ratio(&timing, 1) => Err(ValidateError::SceneModeRate),
            _ => Ok(()),
        }
    }

    pub fn normalize_scene_mode(&mut self, mode: i32, source: &SourceInfo) {
        if mode > 1 {
            self.scene.mode = 0;
            return;
        }
        let timing = self.timing(source);
        if self.scene.mode == 3 && below_ratio(&timing, 1) {
            self.scene.mode = 0;
            self.scene.adaptive = default_adaptive();
        }
    }

    pub fn scale_scene_limits(&mut self, block: metadata::VectorShape) {
        self.scene.limits.scale(block);
    }

    pub const fn dither(&self) -> bool {
        self.render.dither
    }

    pub fn apply_source_depth(&mut self, depth: i32) {
        if depth != 0 {
            self.render.dither = false;
        }
    }

    pub fn apply_core_threads(&mut self, threads: i32) {
        self.mt = i64::from(threads);
    }

    pub fn apply_sar(&mut self, sar: f64) {
        self.light.sar = Some(sar);
    }

    pub fn apply_mask_area_scale(&mut self, source_8bit: bool) {
        if self.mask.area > 0 {
            let area = i32::try_from(self.mask.area).unwrap_or(i32::MAX);
            self.mask.area_scale = f64::from(area) / if source_8bit { 10000.0 } else { 15000.0 };
        }
    }

    pub fn apply_debug_mask_scale(&mut self) {
        if self.debug.vectors && !self.mask.area_enabled {
            self.mask.area_scale = 0.01;
        }
    }

    pub fn apply_mask_cover_algo(&mut self, mode: i32) {
        if mode > 1 || self.mask.cover != 0 {
            return;
        }
        let algo = self.algo.unwrap_or(21);
        if (21..=31).contains(&algo) {
            self.algo = Some(if algo == 22 { 13 } else { 11 });
        }
    }

    pub fn timing(&self, source: &SourceInfo) -> Timing {
        let num = non_zero(
            self.rate
                .num
                .unwrap_or(if self.rate.absolute { 60 } else { 2 }),
        );
        let den = non_zero(self.rate.den.unwrap_or(1));
        Timing::new(source.fps_num, source.fps_den, self.rate.absolute, num, den)
    }

    pub fn padding(&self, source: &SourceInfo) -> (i32, i32) {
        if source.width <= 0 || source.height <= 0 {
            return (0, 0);
        }
        let width = f64::from(source.width);
        let height = f64::from(source.height);
        let source_aspect = width / height;
        let sar = self.light.sar.unwrap_or(1.0);
        let aspect = self.light.aspect.unwrap_or(0.0) / sar;
        let aspect = if aspect < 0.01 { source_aspect } else { aspect };
        let zoom = self.light.zoom.unwrap_or(0.0).max(0.0);
        let (target_w, target_h) = if (aspect - source_aspect).abs() <= 0.01 {
            (source.width, source.height)
        } else if source_aspect <= aspect {
            (trunc_i32(aspect * height), source.height)
        } else {
            (source.width, trunc_i32(width / aspect))
        };
        let x = trunc_i32(f64::from(target_w) * zoom / 100.0) + target_w - source.width;
        let y = trunc_i32(f64::from(target_h) * zoom / 100.0) + target_h - source.height;
        (pad_x(x), pad_y(y))
    }

    pub fn reference(&self) -> ReferenceParams {
        let limits = self.scene.limits;
        let positive = |value: Option<i64>, default: u64| {
            u64::try_from(value.unwrap_or(0))
                .ok()
                .filter(|value| *value != 0)
                .unwrap_or(default)
        };
        ReferenceParams {
            absolute: self.rate.absolute,
            rate_num: positive(self.rate.num, if self.rate.absolute { 60 } else { 2 }),
            rate_den: positive(self.rate.den, 1),
            scene_mode: self.scene.mode,
            adaptive: self.scene.adaptive,
            blocks: i32_saturating(limits.blocks),
            zero: i32_saturating(limits.zero),
            m1: i32_saturating(limits.m1),
            m2: i32_saturating(limits.m2),
            scene: i32_saturating(limits.scene),
            luma: self.scene.luma,
            level: i32::try_from((self.debug.flags >> 3) & 3).unwrap_or(0),
            force13: self.scene.force13,
            blend: self.scene.blend,
            cover: self.mask_cover(),
            area_blend: self.mask.area_blend,
            algo: i32_saturating(self.algo_for_mode(0)),
            linear: self.render.linear,
            cubic: self.cubic.unwrap_or(0),
            block: self.block,
            sad: (self.mask.area > 0).then(|| {
                let area = i32::try_from(self.mask.area).unwrap_or(i32::MAX);
                (f64::from(area) / 15000.0, self.mask.area_sharp)
            }),
        }
    }

    pub fn reference_supported(&self) -> bool {
        let debug = self.debug;
        debug.flags.trailing_zeros() >= 3
            && matches!(self.algo_for_mode(0), 1 | 2 | 11 | 13 | 21 | 22 | 23)
    }

    pub fn light_params(&self) -> LightParams {
        LightParams {
            border: i32_saturating(self.light.border.unwrap_or(12)),
            lights: i32_saturating(self.light.lights.unwrap_or(16)).max(1),
            length: i32_saturating(self.light.length.unwrap_or(100)),
            cell: self.light.cell.unwrap_or(1.0).max(0.1),
        }
    }

    pub const fn cpu_render(&self) -> bool {
        self.gpu.render_cpu
    }

    pub const fn gpu_id(&self) -> i64 {
        self.gpu.id
    }

    pub const fn gpu_qn(&self) -> i64 {
        self.gpu.qn
    }

    pub fn cubic_positive(&self) -> bool {
        self.cubic.unwrap_or(0) > 0
    }

    pub fn apply_cubic_default(&mut self, enabled: bool) {
        if self.cubic.is_none() {
            self.cubic = Some(i64::from(enabled));
        }
    }

    pub fn request_source_plus_two(&self, mode: i32) -> bool {
        let algo = self.algo_for_mode(mode);
        ((self.debug.flags >> 3) & 3) != 0 || algo == 23 || algo >= 90
    }

    pub fn vector_neighbor_level(&self, mode: i32, source_frame: i32) -> i32 {
        let level = i32::try_from((self.debug.flags >> 3) & 3).unwrap_or(0);
        let level_limit = if source_frame >= 3 { 3 } else { 1 };
        if level != 0 {
            return level.min(level_limit);
        }
        let algo = self.algo_for_mode(mode);
        if self.scene.blend || algo == 23 || algo >= 90 {
            level_limit
        } else {
            0
        }
    }

    pub fn request_scene_mode(&self, mode: i32) -> i64 {
        if mode <= 1 { self.scene.mode } else { 0 }
    }

    pub fn scene_adaptive(&self, class: i32) -> Option<i32> {
        usize::try_from(class)
            .ok()
            .and_then(|index| self.scene.adaptive.get(index).copied())
            .filter(|value| *value >= 0)
    }

    pub fn algorithm(&self, mode: i32) -> i64 {
        self.algo_for_mode(mode)
    }

    pub const fn scene_force13(&self) -> bool {
        self.scene.force13
    }

    pub const fn scene_blend(&self) -> bool {
        self.scene.blend
    }

    pub fn scene_luma(&self) -> f64 {
        self.scene.luma
    }

    pub fn scene_thresholds(&self) -> metadata::SceneThresholds {
        scene_thresholds(self.scene.limits)
    }

    pub fn qmap_thresholds(&self) -> metadata::SceneThresholds {
        let mut thresholds = scene_thresholds(self.scene.limits);
        thresholds.zero = i32_saturating(self.scene.qmap_limits.zero);
        thresholds
    }

    pub fn dovi_enabled(&self) -> bool {
        self.hdr.dovi
    }

    pub fn mask_cover(&self) -> i32 {
        i32::try_from(self.mask.cover).unwrap_or(i32::MAX)
    }

    pub const fn mask_area_enabled(&self) -> bool {
        self.mask.area_enabled
    }

    pub const fn fallback_enabled(&self) -> bool {
        self.fallback
    }

    pub const fn gpu_api(&self) -> i64 {
        self.gpu.api
    }

    pub const fn nvof_quality(&self) -> i64 {
        self.nvof.q
    }

    pub const fn nvof_gpu_id(&self) -> i64 {
        self.nvof.gpuid
    }

    pub const fn mask_area_scale(&self) -> f64 {
        self.mask.area_scale
    }

    pub const fn mask_area_sharp(&self) -> f64 {
        self.mask.area_sharp
    }

    pub const fn block_enabled(&self) -> bool {
        self.block
    }

    pub const fn debug_zerox(&self) -> bool {
        self.debug.zerox
    }

    pub const fn debug_zeroy(&self) -> bool {
        self.debug.zeroy
    }

    pub const fn debug_qmode(&self) -> bool {
        self.debug.qmode
    }

    pub const fn debug_qmap(&self) -> bool {
        self.debug.qmap
    }

    pub const fn debug_vectors(&self) -> bool {
        self.debug.vectors
    }

    pub const fn debug_tt(&self) -> bool {
        self.debug.tt
    }

    pub fn request_hdr_vectors(&self) -> bool {
        self.hdr.mluminance > 0.0 && self.hdr.adaptive
    }

    pub fn hdr_enabled(&self) -> bool {
        self.hdr.mluminance > 0.0
    }

    pub fn fast_output_enabled(&self) -> bool {
        self.hdr.mluminance <= 0.0
    }

    pub fn disables_render_for_identity_rate(&self, source: &SourceInfo) -> bool {
        let timing = self.timing(source);
        if timing.frame_den == 0 {
            return false;
        }
        (f64_i64(timing.frame_num) / f64_i64(timing.frame_den) - 1.0).abs() < 0.001
            && self.hdr.mluminance < 0.0
    }

    fn algo_for_mode(&self, mode: i32) -> i64 {
        if mode <= 1 {
            self.algo.unwrap_or(21)
        } else {
            1
        }
    }
}

fn scene_thresholds(limits: SceneLimits) -> metadata::SceneThresholds {
    metadata::SceneThresholds {
        blocks_pct: i32_saturating(limits.blocks),
        blocks13_pct: i32_saturating(limits.blocks13),
        zero: i32_saturating(limits.zero),
        m1: i32_saturating(limits.m1),
        m2: i32_saturating(limits.m2),
        scene: i32_saturating(limits.scene),
        ignore: limits.ignore,
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::for_mode(0)
    }
}

impl Default for Mask {
    fn default() -> Self {
        Self {
            cover: 100,
            area: 0,
            area_enabled: false,
            area_scale: 1.0,
            area_sharp: 1.0,
            area_blend: 0.4,
        }
    }
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            linear: true,
            dither: false,
        }
    }
}

impl Default for HdrOptions {
    fn default() -> Self {
        Self {
            mluminance: -1.0,
            contrast: -1.0,
            adaptive: true,
            dovi: true,
        }
    }
}

impl Default for NvofOptions {
    fn default() -> Self {
        Self { q: 2, gpuid: 0 }
    }
}

impl Default for GpuOptions {
    fn default() -> Self {
        Self {
            render_cpu: false,
            id: 0,
            qn: 2,
            api: 0,
        }
    }
}

impl Default for SceneLimits {
    fn default() -> Self {
        Self::for_mode(0)
    }
}

impl Scene {
    fn for_mode(mode: i32) -> Self {
        let scene_mode = if mode > 1 { 0 } else { 3 };
        Self {
            mode: scene_mode,
            blend: false,
            adaptive: if scene_mode == 3 {
                decode_adaptive(210)
            } else {
                default_adaptive()
            },
            force13: true,
            limits: SceneLimits::for_mode(mode),
            qmap_limits: SceneLimits::for_mode(mode),
            luma: 1.5,
        }
    }

    fn from_value(value: &Value, mode: i32) -> Self {
        let scene_mode = if mode > 1 {
            0
        } else {
            value.int_at(&["scene", "mode"]).unwrap_or(3)
        };
        let limits = SceneLimits::from_value(value, mode);
        Self {
            mode: scene_mode,
            blend: value.bool_at(&["scene", "blend"]).unwrap_or(false),
            adaptive: if scene_mode == 3 {
                decode_adaptive(value.int_at(&["scene", "adaptive"]).unwrap_or(210))
            } else {
                default_adaptive()
            },
            force13: value.bool_at(&["scene", "force13"]).unwrap_or(true),
            limits,
            qmap_limits: limits,
            luma: value
                .float_at(&["scene", "luma"])
                .unwrap_or(1.5)
                .clamp(0.0, 5.0),
        }
    }
}

impl Mask {
    fn from_value(value: &Value) -> Self {
        let area = value.int_at(&["mask", "area"]).unwrap_or(0);
        Self {
            cover: value.int_at(&["mask", "cover"]).unwrap_or(100).max(0),
            area,
            area_enabled: area > 0,
            area_scale: 1.0,
            area_sharp: value.float_at(&["mask", "area_sharp"]).unwrap_or(1.0),
            area_blend: value.float_at(&["mask", "area_blend"]).unwrap_or(0.4),
        }
    }
}

impl DebugOptions {
    fn from_value(value: &Value) -> Self {
        Self {
            flags: value.int_at(&["debug", "flags"]).unwrap_or(0),
            vectors: value.bool_at(&["debug", "vectors"]).unwrap_or(false),
            qmap: value.bool_at(&["debug", "qmap"]).unwrap_or(false),
            qmode: value.bool_at(&["debug", "qmode"]).unwrap_or(false),
            zerox: value.bool_at(&["debug", "zerox"]).unwrap_or(false),
            zeroy: value.bool_at(&["debug", "zeroy"]).unwrap_or(false),
            tt: value.bool_at(&["debug", "tt"]).unwrap_or(false),
        }
    }
}

impl RenderOptions {
    fn from_value(value: &Value) -> Self {
        Self {
            linear: value.bool_at(&["linear"]).unwrap_or(true),
            dither: value.bool_at(&["dither"]).unwrap_or(false),
        }
    }
}

impl HdrOptions {
    fn from_value(value: &Value) -> Self {
        let mluminance = value.int_at(&["hdr", "mluminance"]).unwrap_or(0);
        let enabled = mluminance >= 11;
        Self {
            mluminance: if enabled {
                f64::from(i32::try_from(mluminance).unwrap_or(i32::MAX)) / 100.0
            } else {
                -1.0
            },
            contrast: if enabled {
                value.float_at(&["hdr", "contrast"]).unwrap_or(2.0)
            } else {
                -1.0
            },
            adaptive: enabled && value.bool_at(&["hdr", "adaptive"]).unwrap_or(true),
            dovi: value.bool_at(&["hdr", "dovi"]).unwrap_or(true),
        }
    }
}

impl NvofOptions {
    fn from_value(value: &Value) -> Self {
        Self {
            q: value.int_at(&["nvof", "q"]).unwrap_or(2).min(2),
            gpuid: value.int_at(&["nvof", "gpuid"]).unwrap_or(0),
        }
    }
}

impl GpuOptions {
    fn from_value(value: &Value) -> Self {
        Self {
            render_cpu: value.string_at(&["render"]) == Some("null"),
            id: value.int_at(&["gpuid"]).unwrap_or(0),
            qn: value.int_at(&["gpu_qn"]).unwrap_or(2).max(1),
            api: gpu_api(value.int_at(&["api"]).unwrap_or(0)),
        }
    }
}

impl SceneLimits {
    fn for_mode(mode: i32) -> Self {
        let blocks = if mode < 2 { 20 } else { 50 };
        Self {
            blocks,
            blocks13: 0,
            ignore: 0.04,
            zero: 200,
            m1: 1600,
            m2: 2800,
            scene: if mode < 2 { 4000 } else { 8000 },
        }
    }

    fn from_value(value: &Value, mode: i32) -> Self {
        let defaults = Self::for_mode(mode);
        let blocks = int_below_or(value, &["scene", "limits", "blocks"], defaults.blocks, 100);
        Self {
            blocks,
            blocks13: int_below_or(
                value,
                &["scene", "limits", "blocks13"],
                defaults.blocks13,
                blocks,
            ),
            ignore: f64::from(
                i32::try_from(int_below_or(value, &["scene", "limits", "ignore"], 4, 30))
                    .unwrap_or(30),
            ) / 100.0,
            zero: value
                .int_at(&["scene", "limits", "zero"])
                .unwrap_or(defaults.zero),
            m1: value
                .int_at(&["scene", "limits", "m1"])
                .unwrap_or(defaults.m1),
            m2: value
                .int_at(&["scene", "limits", "m2"])
                .unwrap_or(defaults.m2),
            scene: value
                .int_at(&["scene", "limits", "scene"])
                .unwrap_or(defaults.scene),
        }
    }

    fn scale(&mut self, block: metadata::VectorShape) {
        self.zero = scale_scene_limit(self.zero, block);
        self.m1 = scale_scene_limit(self.m1, block);
        self.m2 = scale_scene_limit(self.m2, block);
        self.scene = scale_scene_limit(self.scene, block);
    }
}

pub enum ValidateError {
    Algo,
    SceneMode,
    SceneModeRate,
}

fn valid_algo(value: i64) -> bool {
    matches!(value, 1 | 2 | 11 | 13 | 21 | 22 | 23 | 90..=100)
}

fn non_zero(value: i64) -> i64 {
    if value == 0 { 1 } else { value }
}

fn below_ratio(timing: &Timing, limit: i64) -> bool {
    i128::from(timing.frame_num) < i128::from(limit) * i128::from(timing.frame_den)
}

fn decode_adaptive(value: i64) -> [i32; 3] {
    [
        adaptive_digit(value, 1),
        adaptive_digit(value, 10),
        adaptive_digit(value, 100),
    ]
}

const fn default_adaptive() -> [i32; 3] {
    [-1, -1, 1]
}

fn adaptive_digit(value: i64, divisor: i64) -> i32 {
    i32::try_from(((value / divisor) % 10).min(3) - 1).unwrap_or(i32::MIN)
}

fn int_below_or(value: &Value, path: &[&str], default: i64, maximum: i64) -> i64 {
    let value = value.int_at(path).unwrap_or(default);
    if value >= 0 && value < maximum {
        value
    } else {
        maximum
    }
}

fn gpu_api(value: i64) -> i64 {
    if (0..3).contains(&value) { value } else { 0 }
}

fn i32_saturating(value: i64) -> i32 {
    i32::try_from(value).unwrap_or(if value.is_negative() {
        i32::MIN
    } else {
        i32::MAX
    })
}

#[allow(clippy::cast_possible_truncation)]
fn trunc_i32(value: f64) -> i32 {
    if value.is_nan() {
        0
    } else if value >= f64::from(i32::MAX) {
        i32::MAX
    } else if value <= f64::from(i32::MIN) {
        i32::MIN
    } else {
        value.trunc() as i32
    }
}

#[allow(clippy::cast_precision_loss)]
fn scale_scene_limit(value: i64, block: metadata::VectorShape) -> i64 {
    i64::from(trunc_i32(
        value as f64 * f64::from(block.width) * f64::from(block.height) / 32.0,
    ))
}

#[allow(clippy::cast_precision_loss)]
fn f64_i64(value: i64) -> f64 {
    value as f64
}

fn pad_x(extra: i32) -> i32 {
    let half = extra / 2;
    (half + 3 * i32::from(half + 1 < 0) + 1) & !3
}

fn pad_y(extra: i32) -> i32 {
    let half = extra / 2;
    (half + i32::from(half + 1 < 0) + 1) & !1
}
