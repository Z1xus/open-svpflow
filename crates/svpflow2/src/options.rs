use crate::{metadata, params::Value, vs};

pub(crate) use svpflow_core::frame_math::Timing;

#[derive(Default)]
#[allow(dead_code)]
pub(crate) struct Options {
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
pub(crate) struct DebugOptions {
    pub(crate) flags: i64,
    pub(crate) vectors: bool,
    pub(crate) qmap: bool,
    pub(crate) qmode: bool,
    pub(crate) zerox: bool,
    pub(crate) zeroy: bool,
    pub(crate) tt: bool,
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
pub(crate) struct RenderOptions {
    pub(crate) linear: bool,
    pub(crate) dither: bool,
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
pub(crate) struct HdrOptions {
    pub(crate) mluminance: f64,
    pub(crate) contrast: f64,
    pub(crate) adaptive: bool,
    pub(crate) dovi: bool,
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
pub(crate) struct NvofOptions {
    pub(crate) q: i64,
    pub(crate) gpuid: i64,
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
pub(crate) struct GpuOptions {
    pub(crate) render_cpu: bool,
    pub(crate) id: i64,
    pub(crate) qn: i64,
    pub(crate) api: i64,
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
pub(crate) struct Mask {
    pub(crate) cover: i64,
    pub(crate) area: i64,
    pub(crate) area_enabled: bool,
    pub(crate) area_scale: f64,
    pub(crate) area_sharp: f64,
    pub(crate) area_blend: f64,
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
pub(crate) struct Scene {
    pub(crate) mode: i64,
    pub(crate) blend: bool,
    pub(crate) adaptive: [i32; 3],
    pub(crate) force13: bool,
    pub(crate) limits: SceneLimits,
    pub(crate) qmap_limits: SceneLimits,
    pub(crate) luma: f64,
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
pub(crate) struct SceneLimits {
    pub(crate) blocks: i64,
    pub(crate) blocks13: i64,
    pub(crate) ignore: f64,
    pub(crate) zero: i64,
    pub(crate) m1: i64,
    pub(crate) m2: i64,
    pub(crate) scene: i64,
}

#[derive(Clone, Copy)]
pub(crate) struct LightParams {
    pub(crate) border: i32,
    pub(crate) lights: i32,
    pub(crate) length: i32,
    pub(crate) cell: f64,
}

impl Options {
    pub(crate) fn for_mode(mode: i32) -> Self {
        Self {
            scene: Scene::for_mode(mode),
            ..Self::default()
        }
    }

    pub(crate) fn from_value(value: &Value, mode: i32) -> Self {
        let hdr = HdrOptions::from_value(value);
        let mut render = RenderOptions::from_value(value);
        render.linear &= hdr.mluminance <= 0.0;
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

    pub(crate) fn validate(&self, mode: i32) -> Result<(), ValidateError> {
        if mode <= 1 && !valid_algo(self.algo.unwrap_or(21)) {
            return Err(ValidateError::Algo);
        }
        if mode <= 1 && !matches!(self.scene.mode, 0..=3) {
            return Err(ValidateError::SceneMode);
        }
        Ok(())
    }

    pub(crate) fn validate_timing(
        &self,
        mode: i32,
        source: &vs::VideoInfo,
    ) -> Result<(), ValidateError> {
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

    pub(crate) fn normalize_scene_mode(&mut self, mode: i32, source: &vs::VideoInfo) {
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

    pub(crate) fn scale_scene_limits(&mut self, block: metadata::VectorShape) {
        self.scene.limits.scale(block);
    }

    pub(crate) fn apply_source_depth(&mut self, depth: i32) {
        if depth != 0 {
            self.render.dither = false;
        }
    }

    pub(crate) fn apply_core_threads(&mut self, threads: i32) {
        self.mt = i64::from(threads);
    }

    pub(crate) fn apply_mask_area_scale(&mut self, source_8bit: bool) {
        if self.mask.area > 0 {
            let area = i32::try_from(self.mask.area).unwrap_or(i32::MAX);
            self.mask.area_scale = f64::from(area) / if source_8bit { 10000.0 } else { 15000.0 };
        }
    }

    pub(crate) fn apply_debug_mask_scale(&mut self) {
        if self.debug.vectors && !self.mask.area_enabled {
            self.mask.area_scale = 0.01;
        }
    }

    pub(crate) fn apply_mask_cover_algo(&mut self, mode: i32) {
        if mode > 1 || self.mask.cover != 0 {
            return;
        }
        let algo = self.algo.unwrap_or(21);
        if (21..=31).contains(&algo) {
            self.algo = Some(if algo == 22 { 13 } else { 11 });
        }
    }

    pub(crate) fn timing(&self, source: &vs::VideoInfo) -> Timing {
        let num = non_zero(
            self.rate
                .num
                .unwrap_or(if self.rate.absolute { 60 } else { 2 }),
        );
        let den = non_zero(self.rate.den.unwrap_or(1));
        Timing::new(source.fps_num, source.fps_den, self.rate.absolute, num, den)
    }

    pub(crate) fn padding(&self, source: &vs::VideoInfo) -> (i32, i32) {
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

    pub(crate) fn light_params(&self) -> LightParams {
        LightParams {
            border: i32_saturating(self.light.border.unwrap_or(12)),
            lights: i32_saturating(self.light.lights.unwrap_or(16)).max(1),
            length: i32_saturating(self.light.length.unwrap_or(100)),
            cell: self.light.cell.unwrap_or(1.0).max(0.1),
        }
    }

    pub(crate) const fn cpu_render(&self) -> bool {
        self.gpu.render_cpu
    }

    pub(crate) const fn gpu_id(&self) -> i64 {
        self.gpu.id
    }

    pub(crate) const fn gpu_qn(&self) -> i64 {
        self.gpu.qn
    }

    pub(crate) fn cubic_positive(&self) -> bool {
        self.cubic.unwrap_or(0) > 0
    }

    pub(crate) fn apply_cubic_default(&mut self, enabled: bool) {
        if self.cubic.is_none() {
            self.cubic = Some(i64::from(enabled));
        }
    }

    pub(crate) fn request_source_plus_two(&self, mode: i32) -> bool {
        let algo = self.algo_for_mode(mode);
        ((self.debug.flags >> 3) & 3) != 0 || algo == 23 || algo >= 90
    }

    pub(crate) fn vector_neighbor_level(&self, mode: i32, source_frame: i32) -> i32 {
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

    pub(crate) fn request_scene_mode(&self, mode: i32) -> i64 {
        if mode <= 1 { self.scene.mode } else { 0 }
    }

    pub(crate) fn scene_adaptive(&self, class: i32) -> Option<i32> {
        usize::try_from(class)
            .ok()
            .and_then(|index| self.scene.adaptive.get(index).copied())
            .filter(|value| *value >= 0)
    }

    pub(crate) fn algorithm(&self, mode: i32) -> i64 {
        self.algo_for_mode(mode)
    }

    pub(crate) const fn scene_force13(&self) -> bool {
        self.scene.force13
    }

    pub(crate) const fn scene_blend(&self) -> bool {
        self.scene.blend
    }

    pub(crate) fn scene_luma(&self) -> f64 {
        self.scene.luma
    }

    pub(crate) fn scene_thresholds(&self) -> metadata::SceneThresholds {
        scene_thresholds(self.scene.limits)
    }

    pub(crate) fn qmap_thresholds(&self) -> metadata::SceneThresholds {
        let mut thresholds = scene_thresholds(self.scene.limits);
        thresholds.zero = i32_saturating(self.scene.qmap_limits.zero);
        thresholds
    }

    pub(crate) fn dovi_enabled(&self) -> bool {
        self.hdr.dovi
    }

    pub(crate) fn mask_cover(&self) -> i32 {
        i32::try_from(self.mask.cover).unwrap_or(i32::MAX)
    }

    pub(crate) const fn mask_area_enabled(&self) -> bool {
        self.mask.area_enabled
    }

    pub(crate) const fn fallback_enabled(&self) -> bool {
        self.fallback
    }

    pub(crate) const fn gpu_api(&self) -> i64 {
        self.gpu.api
    }

    pub(crate) const fn nvof_quality(&self) -> i64 {
        self.nvof.q
    }

    pub(crate) const fn nvof_gpu_id(&self) -> i64 {
        self.nvof.gpuid
    }

    pub(crate) const fn mask_area_scale(&self) -> f64 {
        self.mask.area_scale
    }

    pub(crate) const fn mask_area_sharp(&self) -> f64 {
        self.mask.area_sharp
    }

    pub(crate) const fn block_enabled(&self) -> bool {
        self.block
    }

    pub(crate) const fn debug_zerox(&self) -> bool {
        self.debug.zerox
    }

    pub(crate) const fn debug_zeroy(&self) -> bool {
        self.debug.zeroy
    }

    pub(crate) const fn debug_qmode(&self) -> bool {
        self.debug.qmode
    }

    pub(crate) const fn debug_qmap(&self) -> bool {
        self.debug.qmap
    }

    pub(crate) const fn debug_vectors(&self) -> bool {
        self.debug.vectors
    }

    pub(crate) const fn debug_tt(&self) -> bool {
        self.debug.tt
    }

    pub(crate) fn request_hdr_vectors(&self) -> bool {
        self.hdr.mluminance > 0.0 && self.hdr.adaptive
    }

    pub(crate) fn hdr_enabled(&self) -> bool {
        self.hdr.mluminance > 0.0
    }

    pub(crate) fn fast_output_enabled(&self) -> bool {
        self.hdr.mluminance <= 0.0
    }

    pub(crate) fn disables_render_for_identity_rate(&self, source: &vs::VideoInfo) -> bool {
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

pub(crate) enum ValidateError {
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
