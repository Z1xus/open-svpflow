pub const NEUTRAL: u16 = 1024;
const MAX_VECTOR: i32 = 1023;
const CPU_MAP: [i32; 32] = [
    0, 1, 2, 2, -1, 3, 3, 3, -1, -1, -1, 4, -1, 4, -1, 4, -1, -1, -1, -1, -1, -1, -1, -1, 5, -1,
    -1, -1, 5, -1, -1, 5,
];
const LUT_CENTER: i32 = 1024;
const LUT_LEN: usize = 2048;
const MAX_BLOCK_SIZE: usize = 32;

#[derive(Clone, Copy, Default)]
pub struct Vector {
    pub dx: i16,
    pub dy: i16,
    pub magnitude: u32,
}

#[derive(Clone, Copy)]
pub struct VectorContext<'a> {
    pub block_w: i32,
    pub block_h: i32,
    pub scale_shift_base: i32,
    pub frame_w: i32,
    pub frame_h: i32,
    pub origin_x: i32,
    pub origin_y: i32,
    pub grid_w: usize,
    pub grid_h: usize,
    pub raw: bool,
    pub a: &'a [Vector],
    pub b: &'a [Vector],
}

pub struct CpuConfig {
    pub width: i32,
    pub height: i32,
    pub block_w: i32,
    pub block_h: i32,
    pub origin_x: i32,
    pub origin_y: i32,
    pub grid_w: usize,
    pub grid_h: usize,
    pub chroma_y_div: i32,
    pub source_step: i32,
    pub scale: f64,
}

#[derive(Clone, Copy)]
pub struct GpuPlaneParams {
    pub width: i32,
    pub height: i32,
    pub block_w: i32,
    pub block_h: i32,
    pub origin_x: i32,
    pub origin_y: i32,
    pub x_shift: i32,
    pub y_shift: i32,
    pub x_div: i32,
    pub y_div: i32,
    pub source_step: i32,
}

pub struct CpuRenderer {
    config: CpuConfig,
    chroma_factor: i32,
    width_map: i32,
    height_map: i32,
    half_width_map: i32,
    half_height_map: i32,
    threshold: i32,
    threshold_limit: i32,
    luts_ready: bool,
    inverse_lut: [i16; LUT_LEN],
    threshold_lut: [i16; LUT_LEN],
}

#[derive(Clone, Copy)]
pub struct Mode23Sample {
    pub a: u8,
    pub b: u8,
    pub c: u8,
    pub d: u8,
    pub base_a: u8,
    pub base_b: u8,
    pub t0: u8,
    pub t1: u8,
    pub mask: Option<u8>,
}

#[derive(Clone, Copy)]
pub struct Plane<'a> {
    pub data: &'a [u8],
    pub stride: usize,
    super_span: usize,
    super_shift: u32,
}

impl<'a> Plane<'a> {
    pub const fn linear(data: &'a [u8], stride: usize) -> Self {
        Self {
            data,
            stride,
            super_span: 0,
            super_shift: 0,
        }
    }

    pub const fn super_plane(
        data: &'a [u8],
        stride: usize,
        span: usize,
        shift: u32,
    ) -> Option<Self> {
        if shift == 0 || shift >= usize::BITS {
            return None;
        }
        Some(Self {
            data,
            stride,
            super_span: span,
            super_shift: shift,
        })
    }
}

pub struct PlaneMut<'a> {
    pub data: &'a mut [u8],
    pub stride: usize,
}

#[derive(Clone, Copy)]
enum BlendOrder {
    Forward,
}

#[derive(Clone, Copy)]
pub struct FramePlanes<'a> {
    pub y: Plane<'a>,
    pub u: Plane<'a>,
    pub v: Plane<'a>,
}

pub struct FramePlanesMut<'a> {
    pub y: PlaneMut<'a>,
    pub u: PlaneMut<'a>,
    pub v: PlaneMut<'a>,
}

#[derive(Clone, Copy)]
pub struct MotionPlanes<'a> {
    pub x: &'a [u16],
    pub y: &'a [u16],
}

#[derive(Clone, Copy)]
pub struct MaskPlanes<'a> {
    pub a: &'a [u8],
    pub b: &'a [u8],
}

#[derive(Clone, Copy)]
pub struct PlaneRenderInput<'a> {
    pub source0: Plane<'a>,
    pub source1: Plane<'a>,
    pub motion0: MotionPlanes<'a>,
    pub motion1: MotionPlanes<'a>,
    pub motion2: Option<MotionPlanes<'a>>,
    pub motion3: Option<MotionPlanes<'a>>,
    pub masks: Option<MaskPlanes<'a>>,
    pub final_mask: Option<&'a [u8]>,
}

#[derive(Clone, Copy)]
struct Mode23PlaneInput<'a> {
    source0: Plane<'a>,
    source1: Plane<'a>,
    base0: MotionPlanes<'a>,
    base1: MotionPlanes<'a>,
    next0: MotionPlanes<'a>,
    prev1: MotionPlanes<'a>,
    masks: MaskPlanes<'a>,
    mask: Option<&'a [u8]>,
    chroma: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RenderError {
    UnsupportedMode,
    MissingMasks,
    MissingMotion,
}

impl<'a> VectorContext<'a> {
    fn set(&self, which: usize) -> &'a [Vector] {
        if which == 0 { self.a } else { self.b }
    }

    fn opposite_set(&self, which: usize) -> &'a [Vector] {
        if which == 0 { self.b } else { self.a }
    }

    fn area(&self) -> i32 {
        self.block_w.saturating_mul(self.block_h).max(1)
    }
}

impl CpuRenderer {
    pub fn new(config: CpuConfig) -> Self {
        let mut renderer = Self::new_deferred(config);
        renderer.set_threshold(0);
        renderer
    }

    pub fn new_deferred(config: CpuConfig) -> Self {
        let width_map = cpu_map(config.block_w);
        let height_map = cpu_map(config.block_h);
        let half_width_map = cpu_map(config.block_w / 2);
        let half_height_map = cpu_map(config.block_h / 2);
        Self {
            config,
            chroma_factor: 2,
            width_map,
            height_map,
            half_width_map,
            half_height_map,
            threshold: 0,
            threshold_limit: 0,
            luts_ready: false,
            inverse_lut: [0; LUT_LEN],
            threshold_lut: [0; LUT_LEN],
        }
    }

    pub fn set_threshold(&mut self, threshold: i32) {
        if self.luts_ready && self.threshold == threshold {
            return;
        }
        self.set_threshold_value(threshold);

        let inverse = 256i32.saturating_sub(threshold);
        for (index, sample) in (-LUT_CENTER..LUT_CENTER).enumerate() {
            self.inverse_lut[index] = trunc_div_256(sample.saturating_mul(inverse));
            self.threshold_lut[index] = trunc_div_256(sample.saturating_mul(threshold));
        }
        self.luts_ready = true;
    }

    pub fn set_gpu_threshold(&mut self, threshold: i32) {
        self.set_threshold_value(threshold);
    }

    fn set_threshold_value(&mut self, threshold: i32) {
        self.threshold = threshold;
        let inverse = 256i32.saturating_sub(threshold);
        self.threshold_limit = if threshold > 126 {
            trunc_f64_to_i32(256.0 - f64::from(inverse) * self.config.scale)
        } else {
            trunc_f64_to_i32(f64::from(threshold) * self.config.scale)
        };
    }

    pub fn gpu_plane_params(&self, chroma: bool) -> GpuPlaneParams {
        let p = self.plane_params(false, false);
        let x_ratio = if chroma { 2 } else { 1 };
        let y_ratio = if chroma { self.config.chroma_y_div } else { 1 };
        GpuPlaneParams {
            width: self.config.width / x_ratio,
            height: self.config.height / y_ratio,
            block_w: p.block_w,
            block_h: p.block_h,
            origin_x: p.origin_x,
            origin_y: p.origin_y,
            x_shift: p.x_shift,
            y_shift: p.y_shift,
            x_div: x_ratio,
            y_div: y_ratio,
            source_step: 1,
        }
    }

    pub fn gpu_luts(&self) -> (&[i16; LUT_LEN], &[i16; LUT_LEN]) {
        (&self.inverse_lut, &self.threshold_lut)
    }

    pub fn gpu_thresholds(&self) -> (i32, i32) {
        (self.threshold, self.threshold_limit)
    }

    pub fn gpu_grid(&self) -> (usize, usize) {
        (self.config.grid_w, self.config.grid_h)
    }

    pub fn fill_default(&self, y: &mut [u8], u: &mut [u8], v: &mut [u8]) {
        let area = usize::try_from(self.config.width.saturating_mul(self.config.height).max(0))
            .unwrap_or(0);
        let chroma =
            area / usize::try_from(self.chroma_factor.saturating_mul(2).max(1)).unwrap_or(1);
        fill_prefix(y, area, 0x7F);
        fill_prefix(u, chroma, 0x7F);
        fill_prefix(v, chroma, 0x7F);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render_mode1_or_2(
        &self,
        mode1: bool,
        interp: bool,
        dst: FramePlanesMut<'_>,
        source0: FramePlanes<'_>,
        source1: FramePlanes<'_>,
        motion0: MotionPlanes<'_>,
        motion1: MotionPlanes<'_>,
        masks: Option<MaskPlanes<'_>>,
    ) {
        let (source, motion, mask) = if mode1 {
            (source1, motion1, masks.map(|m| m.a))
        } else {
            (source0, motion0, masks.map(|m| m.b))
        };
        self.render_selected_warp(dst.y, source.y, motion, mask, false, interp);
        self.render_selected_warp(dst.u, source.u, motion, mask, true, interp);
        self.render_selected_warp(dst.v, source.v, motion, mask, true, interp);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render_mode11_or_13(
        &self,
        mode13: bool,
        interp: bool,
        dst: FramePlanesMut<'_>,
        source0: FramePlanes<'_>,
        source1: FramePlanes<'_>,
        motion0: MotionPlanes<'_>,
        motion1: MotionPlanes<'_>,
        masks: Option<MaskPlanes<'_>>,
    ) {
        self.render_mode11_or_13_with_origin(
            mode13,
            interp,
            !interp,
            BlendOrder::Forward,
            dst,
            source0,
            source1,
            motion0,
            motion1,
            masks,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render_mode11_or_13_zero_origin(
        &self,
        mode13: bool,
        interp: bool,
        dst: FramePlanesMut<'_>,
        source0: FramePlanes<'_>,
        source1: FramePlanes<'_>,
        motion0: MotionPlanes<'_>,
        motion1: MotionPlanes<'_>,
        masks: Option<MaskPlanes<'_>>,
    ) {
        self.render_mode11_or_13_with_origin(
            mode13,
            interp,
            true,
            BlendOrder::Forward,
            dst,
            source0,
            source1,
            motion0,
            motion1,
            masks,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn render_mode11_or_13_with_origin(
        &self,
        mode13: bool,
        interp: bool,
        zero_origin: bool,
        blend_order: BlendOrder,
        dst: FramePlanesMut<'_>,
        source0: FramePlanes<'_>,
        source1: FramePlanes<'_>,
        motion0: MotionPlanes<'_>,
        motion1: MotionPlanes<'_>,
        masks: Option<MaskPlanes<'_>>,
    ) {
        let max_mask = masks.map(max_mask);
        self.render_dual_warp(
            dst.y,
            source0.y,
            source1.y,
            motion0,
            motion1,
            max_mask.as_deref(),
            None,
            None,
            false,
            interp,
            zero_origin,
            |cur0, cur1, mv0, mv1, alpha, _, _| {
                mode11_or_13_pixel(
                    mode13,
                    cur0,
                    cur1,
                    mv0,
                    mv1,
                    self.threshold,
                    self.threshold_limit,
                    alpha,
                    blend_order,
                )
            },
        );
        self.render_dual_warp(
            dst.u,
            source0.u,
            source1.u,
            motion0,
            motion1,
            max_mask.as_deref(),
            None,
            None,
            true,
            interp,
            zero_origin,
            |cur0, cur1, mv0, mv1, alpha, _, _| {
                mode11_or_13_pixel(
                    mode13,
                    cur0,
                    cur1,
                    mv0,
                    mv1,
                    self.threshold,
                    self.threshold_limit,
                    alpha,
                    blend_order,
                )
            },
        );
        self.render_dual_warp(
            dst.v,
            source0.v,
            source1.v,
            motion0,
            motion1,
            max_mask.as_deref(),
            None,
            None,
            true,
            interp,
            zero_origin,
            |cur0, cur1, mv0, mv1, alpha, _, _| {
                mode11_or_13_pixel(
                    mode13,
                    cur0,
                    cur1,
                    mv0,
                    mv1,
                    self.threshold,
                    self.threshold_limit,
                    alpha,
                    blend_order,
                )
            },
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render_mode21_or_22(
        &self,
        mode21: bool,
        interp: bool,
        dst: FramePlanesMut<'_>,
        source0: FramePlanes<'_>,
        source1: FramePlanes<'_>,
        motion0: MotionPlanes<'_>,
        motion1: MotionPlanes<'_>,
        masks: MaskPlanes<'_>,
        final_mask: Option<&[u8]>,
    ) {
        self.render_dual_warp(
            dst.y,
            source0.y,
            source1.y,
            motion0,
            motion1,
            Some(masks.a),
            Some(masks.b),
            final_mask,
            false,
            interp,
            !interp,
            |cur0, cur1, mv0, mv1, alpha0, alpha1, final_alpha| {
                self.mode21_or_22_pixel(mode21, cur0, cur1, mv0, mv1, alpha0, alpha1, final_alpha)
            },
        );
        self.render_dual_warp(
            dst.u,
            source0.u,
            source1.u,
            motion0,
            motion1,
            Some(masks.a),
            Some(masks.b),
            final_mask,
            true,
            interp,
            !interp,
            |cur0, cur1, mv0, mv1, alpha0, alpha1, final_alpha| {
                self.mode21_or_22_pixel(mode21, cur0, cur1, mv0, mv1, alpha0, alpha1, final_alpha)
            },
        );
        self.render_dual_warp(
            dst.v,
            source0.v,
            source1.v,
            motion0,
            motion1,
            Some(masks.a),
            Some(masks.b),
            final_mask,
            true,
            interp,
            !interp,
            |cur0, cur1, mv0, mv1, alpha0, alpha1, final_alpha| {
                self.mode21_or_22_pixel(mode21, cur0, cur1, mv0, mv1, alpha0, alpha1, final_alpha)
            },
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn mode21_or_22_pixel(
        &self,
        mode21: bool,
        cur0: u8,
        cur1: u8,
        mv0: u8,
        mv1: u8,
        alpha0: Option<u8>,
        alpha1: Option<u8>,
        final_alpha: Option<u8>,
    ) -> u8 {
        let alpha0 = alpha0.unwrap_or(0);
        let alpha1 = alpha1.unwrap_or(0);
        let motion = if mode21 {
            mode21_pixel(mv0, mv1, alpha0, alpha1, self.threshold)
        } else {
            mode22_pixel(mv0, mv1, cur0, cur1, alpha0, alpha1, self.threshold)
        };
        if let Some(final_alpha) = final_alpha {
            let base = blend_256(cur0, cur1, self.threshold_limit);
            blend_255(motion, base, i32::from(final_alpha))
        } else {
            motion
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render_mode23(
        &self,
        interp: bool,
        dst: FramePlanesMut<'_>,
        source0: FramePlanes<'_>,
        source1: FramePlanes<'_>,
        base0: MotionPlanes<'_>,
        base1: MotionPlanes<'_>,
        next0: MotionPlanes<'_>,
        prev1: MotionPlanes<'_>,
        masks: MaskPlanes<'_>,
        mask: Option<&[u8]>,
    ) {
        self.render_mode23_plane(
            dst.y, source0.y, source1.y, base0, base1, next0, prev1, masks, mask, false, interp,
        );
        self.render_mode23_plane(
            dst.u, source0.u, source1.u, base0, base1, next0, prev1, masks, mask, true, interp,
        );
        self.render_mode23_plane(
            dst.v, source0.v, source1.v, base0, base1, next0, prev1, masks, mask, true, interp,
        );
    }

    pub fn render_plane(
        &self,
        mode: u32,
        interp: bool,
        dst: PlaneMut<'_>,
        input: PlaneRenderInput<'_>,
        chroma: bool,
    ) -> Result<(), RenderError> {
        let height = if chroma {
            self.config.height / self.config.chroma_y_div
        } else {
            self.config.height
        };
        self.render_plane_rows(mode, interp, dst, input, chroma, 0..height)
    }

    pub fn render_plane_rows(
        &self,
        mode: u32,
        interp: bool,
        dst: PlaneMut<'_>,
        input: PlaneRenderInput<'_>,
        chroma: bool,
        rows: Range<i32>,
    ) -> Result<(), RenderError> {
        match mode {
            1 | 2 => {
                let (source, motion, mask) = if mode == 1 {
                    (input.source0, input.motion0, input.masks.map(|m| m.b))
                } else {
                    (input.source1, input.motion1, input.masks.map(|m| m.a))
                };
                self.render_selected_warp_rows(
                    dst,
                    source,
                    motion,
                    mask,
                    chroma,
                    interp,
                    rows.clone(),
                );
            }
            11 | 13 => {
                let max_mask = input.masks.map(max_mask);
                self.render_dual_warp_rows(
                    dst,
                    input.source0,
                    input.source1,
                    input.motion0,
                    input.motion1,
                    max_mask.as_deref(),
                    None,
                    None,
                    chroma,
                    interp,
                    !interp,
                    rows.clone(),
                    |cur0, cur1, mv0, mv1, alpha, _, _| {
                        mode11_or_13_pixel(
                            mode == 13,
                            cur0,
                            cur1,
                            mv0,
                            mv1,
                            self.threshold,
                            self.threshold_limit,
                            alpha,
                            BlendOrder::Forward,
                        )
                    },
                );
            }
            21 | 22 => {
                let masks = input.masks.ok_or(RenderError::MissingMasks)?;
                self.render_dual_warp_rows(
                    dst,
                    input.source0,
                    input.source1,
                    input.motion0,
                    input.motion1,
                    Some(masks.a),
                    Some(masks.b),
                    input.final_mask,
                    chroma,
                    interp,
                    !interp,
                    rows.clone(),
                    |cur0, cur1, mv0, mv1, alpha0, alpha1, final_alpha| {
                        self.mode21_or_22_pixel(
                            mode == 21,
                            cur0,
                            cur1,
                            mv0,
                            mv1,
                            alpha0,
                            alpha1,
                            final_alpha,
                        )
                    },
                );
            }
            23 => self.render_mode23_plane_rows(
                dst,
                input.source0,
                input.source1,
                input.motion0,
                input.motion1,
                input.motion2.ok_or(RenderError::MissingMotion)?,
                input.motion3.ok_or(RenderError::MissingMotion)?,
                input.masks.ok_or(RenderError::MissingMasks)?,
                input.final_mask,
                chroma,
                interp,
                rows,
            ),
            _ => return Err(RenderError::UnsupportedMode),
        }
        Ok(())
    }

    #[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
    fn render_selected_warp(
        &self,
        dst: PlaneMut<'_>,
        source: Plane<'_>,
        motion: MotionPlanes<'_>,
        mask: Option<&[u8]>,
        chroma: bool,
        interp: bool,
    ) {
        let height = if chroma {
            self.config.height / self.config.chroma_y_div
        } else {
            self.config.height
        };
        self.render_selected_warp_rows(dst, source, motion, mask, chroma, interp, 0..height);
    }

    #[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
    fn render_selected_warp_rows(
        &self,
        dst: PlaneMut<'_>,
        source: Plane<'_>,
        motion: MotionPlanes<'_>,
        mask: Option<&[u8]>,
        chroma: bool,
        interp: bool,
        rows: Range<i32>,
    ) {
        let zero_origin = !interp;
        let params = self.plane_params(chroma, zero_origin);
        let mask_params = zero_origin.then(|| self.plane_params(chroma, false));
        let row_start = rows.start;
        render_tiles(
            &params,
            self.config.grid_w,
            self.config.grid_h,
            interp,
            rows,
            |tile| {
                let motion_samples =
                    Self::motion_samples(motion, &self.threshold_lut, &params, &tile, interp);
                let direct = tile_interior(&params, &tile, &[motion_samples])
                    && plane_covers(source, &params);
                for y in 0..tile.height {
                    let local_y = tile.local_y + y;
                    let source_y = (tile.y + y) * params.source_step;
                    let Some(output) = output_row(dst.data, dst.stride, &tile, y, row_start) else {
                        continue;
                    };
                    let motion_row =
                        interpolate_motion_row::<false>(motion_samples, &params, local_y, interp);
                    let mask_row = mask.map_or([0; 2], |mask| {
                        alpha_row::<false>(mask, &params, &tile, local_y, interp)
                    });
                    for (x, output) in (0..tile.width).zip(output) {
                        let px = tile.x + x;
                        let py = tile.y + y;
                        let source_x = px * params.source_step;
                        let weights = params.x_weights
                            [usize::try_from(x).unwrap_or(0).min(MAX_BLOCK_SIZE - 1)];
                        let (dx, dy) =
                            interpolate_motion::<false>(motion_row, &params, weights, interp);
                        let base = sample_plane(source, &params, source_x, source_y, 0, 0, direct);
                        let warped =
                            sample_plane(source, &params, source_x, source_y, dx, dy, direct);
                        let value = if let Some(mask) = mask {
                            let alpha = if let Some(mask_params) = &mask_params {
                                alpha_at_absolute(mask, mask_params, self.config.grid_w, px, py)
                            } else {
                                alpha_at_row::<false>(mask_row, &params, weights, interp)
                            };
                            mode1_pixel(base, warped, alpha)
                        } else {
                            warped
                        };
                        *output = value;
                    }
                }
            },
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_pass_by_value)]
    fn render_dual_warp(
        &self,
        dst: PlaneMut<'_>,
        source0: Plane<'_>,
        source1: Plane<'_>,
        motion0: MotionPlanes<'_>,
        motion1: MotionPlanes<'_>,
        mask0: Option<&[u8]>,
        mask1: Option<&[u8]>,
        mask2: Option<&[u8]>,
        chroma: bool,
        interp: bool,
        zero_origin: bool,
        pixel: impl FnMut(u8, u8, u8, u8, Option<u8>, Option<u8>, Option<u8>) -> u8,
    ) {
        let height = if chroma {
            self.config.height / self.config.chroma_y_div
        } else {
            self.config.height
        };
        self.render_dual_warp_rows(
            dst,
            source0,
            source1,
            motion0,
            motion1,
            mask0,
            mask1,
            mask2,
            chroma,
            interp,
            zero_origin,
            0..height,
            pixel,
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_pass_by_value)]
    fn render_dual_warp_rows(
        &self,
        dst: PlaneMut<'_>,
        source0: Plane<'_>,
        source1: Plane<'_>,
        motion0: MotionPlanes<'_>,
        motion1: MotionPlanes<'_>,
        mask0: Option<&[u8]>,
        mask1: Option<&[u8]>,
        mask2: Option<&[u8]>,
        chroma: bool,
        interp: bool,
        zero_origin: bool,
        rows: Range<i32>,
        mut pixel: impl FnMut(u8, u8, u8, u8, Option<u8>, Option<u8>, Option<u8>) -> u8,
    ) {
        let params = self.plane_params(chroma, zero_origin);
        let mask_params = zero_origin.then(|| self.plane_params(chroma, false));
        let row_start = rows.start;
        render_tiles(
            &params,
            self.config.grid_w,
            self.config.grid_h,
            interp,
            rows,
            |tile| {
                let motion0_samples =
                    Self::motion_samples(motion0, &self.threshold_lut, &params, &tile, interp);
                let motion1_samples =
                    Self::motion_samples(motion1, &self.inverse_lut, &params, &tile, interp);
                let direct = tile_interior(&params, &tile, &[motion0_samples, motion1_samples])
                    && plane_covers(source0, &params)
                    && plane_covers(source1, &params);
                for y in 0..tile.height {
                    let local_y = tile.local_y + y;
                    let source_y = (tile.y + y) * params.source_step;
                    let Some(output) = output_row(dst.data, dst.stride, &tile, y, row_start) else {
                        continue;
                    };
                    let motion0_row =
                        interpolate_motion_row::<false>(motion0_samples, &params, local_y, interp);
                    let motion1_row =
                        interpolate_motion_row::<false>(motion1_samples, &params, local_y, interp);
                    let mask0_row = mask0.map_or([0; 2], |mask| {
                        alpha_row::<false>(mask, &params, &tile, local_y, interp)
                    });
                    let mask1_row = mask1.map_or([0; 2], |mask| {
                        alpha_row::<false>(mask, &params, &tile, local_y, interp)
                    });
                    let mask2_row = mask2.map_or([0; 2], |mask| {
                        alpha_row::<false>(mask, &params, &tile, local_y, interp)
                    });
                    for (x, output) in (0..tile.width).zip(output) {
                        let px = tile.x + x;
                        let py = tile.y + y;
                        let source_x = px * params.source_step;
                        let weights = params.x_weights
                            [usize::try_from(x).unwrap_or(0).min(MAX_BLOCK_SIZE - 1)];
                        let (dx0, dy0) =
                            interpolate_motion::<false>(motion0_row, &params, weights, interp);
                        let (dx1, dy1) =
                            interpolate_motion::<false>(motion1_row, &params, weights, interp);
                        let cur0 = sample_plane(source0, &params, source_x, source_y, 0, 0, direct);
                        let cur1 = sample_plane(source1, &params, source_x, source_y, 0, 0, direct);
                        let mv0 =
                            sample_plane(source0, &params, source_x, source_y, dx0, dy0, direct);
                        let mv1 =
                            sample_plane(source1, &params, source_x, source_y, dx1, dy1, direct);
                        let alpha0 = mask0.map(|mask| {
                            if let Some(mask_params) = &mask_params {
                                alpha_at_absolute(mask, mask_params, self.config.grid_w, px, py)
                            } else {
                                alpha_at_row::<false>(mask0_row, &params, weights, interp)
                            }
                        });
                        let alpha1 = mask1.map(|mask| {
                            if let Some(mask_params) = &mask_params {
                                alpha_at_absolute(mask, mask_params, self.config.grid_w, px, py)
                            } else {
                                alpha_at_row::<false>(mask1_row, &params, weights, interp)
                            }
                        });
                        let alpha2 = mask2.map(|mask| {
                            if let Some(mask_params) = &mask_params {
                                alpha_at_absolute(mask, mask_params, self.config.grid_w, px, py)
                            } else {
                                alpha_at_row::<false>(mask2_row, &params, weights, interp)
                            }
                        });
                        *output = pixel(cur0, cur1, mv0, mv1, alpha0, alpha1, alpha2);
                    }
                }
            },
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_pass_by_value)]
    fn render_mode23_plane(
        &self,
        dst: PlaneMut<'_>,
        source0: Plane<'_>,
        source1: Plane<'_>,
        base0: MotionPlanes<'_>,
        base1: MotionPlanes<'_>,
        next0: MotionPlanes<'_>,
        prev1: MotionPlanes<'_>,
        masks: MaskPlanes<'_>,
        mask: Option<&[u8]>,
        chroma: bool,
        interp: bool,
    ) {
        let height = if chroma {
            self.config.height / self.config.chroma_y_div
        } else {
            self.config.height
        };
        self.render_mode23_plane_rows(
            dst,
            source0,
            source1,
            base0,
            base1,
            next0,
            prev1,
            masks,
            mask,
            chroma,
            interp,
            0..height,
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_pass_by_value)]
    fn render_mode23_plane_rows(
        &self,
        dst: PlaneMut<'_>,
        source0: Plane<'_>,
        source1: Plane<'_>,
        base0: MotionPlanes<'_>,
        base1: MotionPlanes<'_>,
        next0: MotionPlanes<'_>,
        prev1: MotionPlanes<'_>,
        masks: MaskPlanes<'_>,
        mask: Option<&[u8]>,
        chroma: bool,
        interp: bool,
        rows: Range<i32>,
    ) {
        let input = Mode23PlaneInput {
            source0,
            source1,
            base0,
            base1,
            next0,
            prev1,
            masks,
            mask,
            chroma,
        };
        match (interp, mask.is_some()) {
            (true, false) => {
                self.render_mode23_plane_rows_variant::<true, false>(dst, &input, rows);
            }
            (true, true) => self.render_mode23_plane_rows_variant::<true, true>(dst, &input, rows),
            (false, false) => {
                self.render_mode23_plane_rows_variant::<false, false>(dst, &input, rows);
            }
            (false, true) => {
                self.render_mode23_plane_rows_variant::<false, true>(dst, &input, rows);
            }
        }
    }

    fn render_mode23_plane_rows_variant<const INTERP: bool, const MASK: bool>(
        &self,
        dst: PlaneMut<'_>,
        input: &Mode23PlaneInput<'_>,
        rows: Range<i32>,
    ) {
        let shifted = if input.chroma {
            self.half_width_map >= 0 && self.half_height_map >= 0
        } else {
            self.width_map >= 0 && self.height_map >= 0
        };
        if shifted {
            self.render_mode23_plane_rows_inner::<INTERP, MASK, true>(dst, input, rows);
        } else {
            self.render_mode23_plane_rows_inner::<INTERP, MASK, false>(dst, input, rows);
        }
    }

    #[allow(clippy::needless_pass_by_value, clippy::too_many_lines)]
    fn render_mode23_plane_rows_inner<const INTERP: bool, const MASK: bool, const SHIFTED: bool>(
        &self,
        dst: PlaneMut<'_>,
        input: &Mode23PlaneInput<'_>,
        rows: Range<i32>,
    ) {
        let Mode23PlaneInput {
            source0,
            source1,
            base0,
            base1,
            next0,
            prev1,
            masks,
            mask,
            chroma,
        } = *input;
        let interp = INTERP;
        let mask = if MASK { mask } else { None };
        let zero_origin = !interp;
        let params = self.plane_params(chroma, zero_origin);
        let mask_params = zero_origin.then(|| self.plane_params(chroma, false));
        let row_start = rows.start;
        render_tiles(
            &params,
            self.config.grid_w,
            self.config.grid_h,
            interp,
            rows.clone(),
            |tile| {
                let motion0_samples =
                    Self::motion_samples(base0, &self.threshold_lut, &params, &tile, interp);
                let motion1_samples =
                    Self::motion_samples(base1, &self.inverse_lut, &params, &tile, interp);
                let motion2_samples =
                    Self::motion_samples(next0, &self.threshold_lut, &params, &tile, interp);
                let motion3_samples =
                    Self::motion_samples(prev1, &self.inverse_lut, &params, &tile, interp);
                let direct = tile_interior(
                    &params,
                    &tile,
                    &[
                        motion0_samples,
                        motion1_samples,
                        motion2_samples,
                        motion3_samples,
                    ],
                ) && plane_covers(source0, &params)
                    && plane_covers(source1, &params);
                for y in 0..tile.height {
                    let local_y = tile.local_y + y;
                    let source_y = (tile.y + y) * params.source_step;
                    let Some(output) = output_row(dst.data, dst.stride, &tile, y, row_start) else {
                        continue;
                    };
                    let motion0_row = interpolate_motion_row::<SHIFTED>(
                        motion0_samples,
                        &params,
                        local_y,
                        interp,
                    );
                    let motion1_row = interpolate_motion_row::<SHIFTED>(
                        motion1_samples,
                        &params,
                        local_y,
                        interp,
                    );
                    let motion2_row = interpolate_motion_row::<SHIFTED>(
                        motion2_samples,
                        &params,
                        local_y,
                        interp,
                    );
                    let motion3_row = interpolate_motion_row::<SHIFTED>(
                        motion3_samples,
                        &params,
                        local_y,
                        interp,
                    );
                    let mask0_row = alpha_row::<SHIFTED>(masks.a, &params, &tile, local_y, interp);
                    let mask1_row = alpha_row::<SHIFTED>(masks.b, &params, &tile, local_y, interp);
                    let mask_row = mask.map_or([0; 2], |mask| {
                        alpha_row::<SHIFTED>(mask, &params, &tile, local_y, interp)
                    });
                    for (x, output) in (0..tile.width).zip(output) {
                        let px = tile.x + x;
                        let py = tile.y + y;
                        let source_x = px * params.source_step;
                        let pos = AlphaPos {
                            plane_x: px,
                            plane_y: py,
                        };
                        let weights = params.x_weights
                            [usize::try_from(x).unwrap_or(0).min(MAX_BLOCK_SIZE - 1)];
                        let (dx0, dy0) =
                            interpolate_motion::<SHIFTED>(motion0_row, &params, weights, interp);
                        let (dx1, dy1) =
                            interpolate_motion::<SHIFTED>(motion1_row, &params, weights, interp);
                        let (dx2, dy2) =
                            interpolate_motion::<SHIFTED>(motion2_row, &params, weights, interp);
                        let (dx3, dy3) =
                            interpolate_motion::<SHIFTED>(motion3_row, &params, weights, interp);
                        let mask = mask.map(|mask| {
                            alpha_for_origin::<SHIFTED>(
                                mask,
                                mask_row,
                                &params,
                                mask_params.as_ref(),
                                self.config.grid_w,
                                pos,
                                weights,
                            )
                        });
                        let (base_a, base_b) = if mask.is_some() {
                            (
                                sample_plane(source0, &params, source_x, source_y, 0, 0, direct),
                                sample_plane(source1, &params, source_x, source_y, 0, 0, direct),
                            )
                        } else {
                            (0, 0)
                        };
                        let sample = Mode23Sample {
                            a: sample_plane(source0, &params, source_x, source_y, dx0, dy0, direct),
                            b: sample_plane(source1, &params, source_x, source_y, dx1, dy1, direct),
                            c: sample_plane(source0, &params, source_x, source_y, dx2, dy2, direct),
                            d: sample_plane(source1, &params, source_x, source_y, dx3, dy3, direct),
                            base_a,
                            base_b,
                            t0: alpha_for_origin::<SHIFTED>(
                                masks.a,
                                mask0_row,
                                &params,
                                mask_params.as_ref(),
                                self.config.grid_w,
                                pos,
                                weights,
                            ),
                            t1: alpha_for_origin::<SHIFTED>(
                                masks.b,
                                mask1_row,
                                &params,
                                mask_params.as_ref(),
                                self.config.grid_w,
                                pos,
                                weights,
                            ),
                            mask,
                        };
                        *output = mode23_pixel(sample, self.threshold, self.threshold_limit);
                    }
                }
            },
        );
    }

    fn plane_params(&self, chroma: bool, zero_origin: bool) -> PlaneParams {
        let (origin_x, origin_y) = if zero_origin {
            (0, 0)
        } else if chroma {
            (
                self.config.origin_x / 2,
                self.config.origin_y / self.config.chroma_y_div,
            )
        } else {
            (self.config.origin_x, self.config.origin_y)
        };
        if chroma {
            PlaneParams {
                width: self.config.width / 2,
                height: self.config.height / self.config.chroma_y_div,
                block_w: self.config.block_w / 2,
                block_h: self.config.block_h / self.config.chroma_y_div,
                origin_x,
                origin_y,
                x_shift: self.half_width_map,
                y_shift: self.half_height_map,
                x_div: 2,
                y_div: self.config.chroma_y_div,
                source_step: self.config.source_step,
                max_x: (self.config.width / 2)
                    .saturating_mul(self.config.source_step)
                    .saturating_sub(1)
                    .max(0),
                max_y: (self.config.height / self.config.chroma_y_div)
                    .saturating_mul(self.config.source_step)
                    .saturating_sub(1)
                    .max(0),
                x_weights: interpolation_weights(self.config.block_w / 2, self.half_width_map),
                y_weights: interpolation_weights(
                    self.config.block_h / self.config.chroma_y_div,
                    self.half_height_map,
                ),
            }
        } else {
            PlaneParams {
                width: self.config.width,
                height: self.config.height,
                block_w: self.config.block_w,
                block_h: self.config.block_h,
                origin_x,
                origin_y,
                x_shift: self.width_map,
                y_shift: self.height_map,
                x_div: 1,
                y_div: 1,
                source_step: self.config.source_step,
                max_x: self
                    .config
                    .width
                    .saturating_mul(self.config.source_step)
                    .saturating_sub(1)
                    .max(0),
                max_y: self
                    .config
                    .height
                    .saturating_mul(self.config.source_step)
                    .saturating_sub(1)
                    .max(0),
                x_weights: interpolation_weights(self.config.block_w, self.width_map),
                y_weights: interpolation_weights(self.config.block_h, self.height_map),
            }
        }
    }

    fn motion_samples(
        motion: MotionPlanes<'_>,
        lut: &[i16; LUT_LEN],
        params: &PlaneParams,
        tile: &Tile,
        interp: bool,
    ) -> MotionSamples {
        let x0 = motion_value(lut, motion.x, tile.i00) / params.x_div;
        let y0 = motion_value(lut, motion.y, tile.i00) / params.y_div;
        if !interp {
            return MotionSamples {
                x: [x0; 4],
                y: [y0; 4],
            };
        }
        MotionSamples {
            x: [
                x0,
                motion_value(lut, motion.x, tile.i01) / params.x_div,
                motion_value(lut, motion.x, tile.i10) / params.x_div,
                motion_value(lut, motion.x, tile.i11) / params.x_div,
            ],
            y: [
                y0,
                motion_value(lut, motion.y, tile.i01) / params.y_div,
                motion_value(lut, motion.y, tile.i10) / params.y_div,
                motion_value(lut, motion.y, tile.i11) / params.y_div,
            ],
        }
    }
}

pub fn fill_neutral(left: &mut [u16], right: Option<&mut [u16]>) {
    left.fill(NEUTRAL);
    if let Some(right) = right {
        right.fill(NEUTRAL);
    }
}

pub fn magnitude_mask(
    ctx: &VectorContext<'_>,
    which: usize,
    out: &mut [u8],
    width: usize,
    height: usize,
    scale: f64,
    exponent: f64,
) {
    let vectors = ctx.set(which);
    for y in 0..height {
        for x in 0..width {
            let index = y.saturating_mul(width).saturating_add(x);
            if index >= out.len() {
                return;
            }
            out[index] = if x < ctx.grid_w && y < ctx.grid_h {
                let vector = vectors
                    .get(y.saturating_mul(ctx.grid_w).saturating_add(x))
                    .copied()
                    .unwrap_or_default();
                scale_magnitude(vector.magnitude, ctx.area(), scale, exponent)
            } else if x > 0 {
                out[index - 1]
            } else if y > 0 {
                out[index - width]
            } else {
                0
            };
        }
    }
}

pub fn vector_planes(
    ctx: &VectorContext<'_>,
    which: usize,
    dst_x: &mut [u16],
    dst_y: &mut [u16],
    width: usize,
    height: usize,
) {
    let vectors = ctx.set(which);
    for y in 0..height {
        for x in 0..width {
            let out = y.saturating_mul(width).saturating_add(x);
            if out >= dst_x.len() || out >= dst_y.len() {
                return;
            }
            let vector = vectors
                .get(sample_index(ctx, x, y))
                .copied()
                .unwrap_or_default();
            let (dx, dy) = corrected_vector(ctx, vector, x, y);
            dst_x[out] = centered(dx);
            dst_y[out] = centered(dy);
        }
    }
}

pub fn coverage_mask(
    ctx: &VectorContext<'_>,
    which: usize,
    out: &mut [u8],
    width: usize,
    height: usize,
    strength_percent: i32,
    threshold: i32,
) {
    let stride = width.saturating_add(2);
    let mut accum = vec![0i32; stride.saturating_mul(height.saturating_add(2))];
    let vectors = ctx.opposite_set(which);
    splat_coverage(ctx, vectors, &mut accum, width, height, threshold);
    finish_coverage(ctx, &accum, out, width, height, strength_percent);
}

pub fn sample_mask(
    ctx: &VectorContext<'_>,
    mask: &[u8],
    x: i32,
    y: i32,
    chroma_y_div: i32,
    chroma: bool,
) -> u8 {
    let x_div = if chroma { 2 } else { 1 };
    let y_div = if chroma { chroma_y_div } else { 1 };
    if x_div == 0 || y_div == 0 {
        return 0;
    }
    let origin_x = ctx.origin_x / x_div;
    let origin_y = ctx.origin_y / y_div;
    let step_x = ctx.block_w / x_div;
    let step_y = ctx.block_h / y_div;
    if step_x == 0 || step_y == 0 || ctx.grid_w == 0 {
        return 0;
    }

    let ix0 = if x >= origin_x {
        (x - origin_x) / step_x
    } else {
        0
    };
    let iy0 = if y >= origin_y {
        (y - origin_y) / step_y
    } else {
        0
    };
    let ix1 = if x >= origin_x { ix0 + 1 } else { 0 };
    let iy1 = if y >= origin_y { iy0 + 1 } else { 0 };
    let fx = if x > origin_x {
        (x - origin_x) % step_x
    } else {
        0
    };
    let fy = if y > origin_y {
        (y - origin_y) % step_y
    } else {
        0
    };

    let m00 = i32::from(mask_at(mask, ctx.grid_w, ix0, iy0));
    let m01 = i32::from(mask_at(mask, ctx.grid_w, ix1, iy0));
    let m10 = i32::from(mask_at(mask, ctx.grid_w, ix0, iy1));
    let m11 = i32::from(mask_at(mask, ctx.grid_w, ix1, iy1));
    let top = ((step_x - fx) * m00 + fx * m01) / step_x;
    let bottom = ((step_x - fx) * m10 + fx * m11) / step_x;
    u8::try_from((((step_y - fy) * top + fy * bottom) / step_y) & 0xFF).unwrap_or(0)
}

pub fn mode1_pixel(base: u8, warped: u8, alpha: u8) -> u8 {
    blend_255(warped, base, i32::from(alpha))
}

pub fn mode11_pixel(mv0: u8, mv1: u8, weight: i32) -> u8 {
    blend_256(mv0, mv1, weight)
}

pub fn mode13_pixel(cur0: u8, cur1: u8, mv0: u8, mv1: u8, weight: i32) -> u8 {
    clamp_between(blend_256(cur0, cur1, weight), mv0, mv1)
}

pub fn mode21_pixel(mv0: u8, mv1: u8, alpha0: u8, alpha1: u8, weight: i32) -> u8 {
    let a = blend_255(mv0, mv1, i32::from(alpha0));
    let b = blend_255(mv1, mv0, i32::from(alpha1));
    blend_256(a, b, weight)
}

pub fn mode22_pixel(
    mv0: u8,
    mv1: u8,
    base0: u8,
    base1: u8,
    alpha0: u8,
    alpha1: u8,
    weight: i32,
) -> u8 {
    let a = blend_255(mv0, mv1, i32::from(alpha0));
    let b = blend_255(mv1, mv0, i32::from(alpha1));
    clamp_between(blend_256(base0, base1, weight), a, b)
}

pub fn mode23_pixel(sample: Mode23Sample, q0: i32, _q1: i32) -> u8 {
    let i0 = blend_255(
        sample.a,
        clamp_between(sample.c, sample.a, sample.b),
        i32::from(sample.t0),
    );
    let i1 = blend_255(
        sample.b,
        clamp_between(sample.d, sample.a, sample.b),
        i32::from(sample.t1),
    );
    let motion = blend_256(i0, i1, q0);
    if let Some(mask) = sample.mask {
        let base = blend_256(sample.base_a, sample.base_b, q0);
        blend_255(motion, base, i32::from(mask))
    } else {
        motion
    }
}

fn sample_index(ctx: &VectorContext<'_>, x: usize, y: usize) -> usize {
    let sx = x.min(ctx.grid_w.saturating_sub(1));
    let sy = y.min(ctx.grid_h.saturating_sub(1));
    sy.saturating_mul(ctx.grid_w).saturating_add(sx)
}

struct PlaneParams {
    width: i32,
    height: i32,
    block_w: i32,
    block_h: i32,
    origin_x: i32,
    origin_y: i32,
    x_shift: i32,
    y_shift: i32,
    x_div: i32,
    y_div: i32,
    source_step: i32,
    max_x: i32,
    max_y: i32,
    x_weights: [(i32, i32); MAX_BLOCK_SIZE],
    y_weights: [(i32, i32); MAX_BLOCK_SIZE],
}

struct Tile {
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    local_y: i32,
    i00: usize,
    i01: usize,
    i10: usize,
    i11: usize,
}

#[derive(Clone, Copy)]
struct MotionSamples {
    x: [i32; 4],
    y: [i32; 4],
}

#[derive(Clone, Copy)]
struct MotionRow {
    x: [i32; 2],
    y: [i32; 2],
}

#[derive(Clone, Copy)]
struct AlphaPos {
    plane_x: i32,
    plane_y: i32,
}

#[allow(clippy::similar_names)]
fn render_tiles(
    params: &PlaneParams,
    grid_w: usize,
    grid_h: usize,
    _interp: bool,
    rows: Range<i32>,
    mut render: impl FnMut(Tile),
) {
    let Ok(grid_w_i32) = i32::try_from(grid_w) else {
        return;
    };
    let Ok(grid_h_i32) = i32::try_from(grid_h) else {
        return;
    };
    if params.width <= 0
        || params.height <= 0
        || params.block_w <= 0
        || params.block_h <= 0
        || grid_w == 0
        || grid_h == 0
    {
        return;
    }

    for gy in -1..grid_h_i32 {
        let base_y = if gy < 0 {
            0
        } else {
            params.origin_y + params.block_h * gy
        };
        let mut height = if gy < 0 {
            params.origin_y
        } else {
            params.block_h
        };

        if base_y + params.block_h >= params.height || gy == grid_h_i32 - 1 {
            height = params.height - base_y;
        }
        let end_y = (base_y + height).min(rows.end);
        let clipped_y = base_y.max(rows.start);
        height = end_y - clipped_y;
        if height <= 0 {
            continue;
        }
        for gx in -1..grid_w_i32 {
            let base_x = if gx < 0 {
                0
            } else {
                params.origin_x + params.block_w * gx
            };
            let mut width = if gx < 0 {
                params.origin_x
            } else {
                params.block_w
            };
            if base_x + params.block_w >= params.width || gx == grid_w_i32 - 1 {
                width = params.width - base_x;
            }
            if width <= 0 {
                continue;
            }

            let x0 = usize::try_from(gx.max(0)).unwrap_or(0);
            let y0 = usize::try_from(gy.max(0)).unwrap_or(0);
            let next_x = usize::from(gx >= 0 && gx != grid_w_i32 - 1);
            let next_y = if gy >= 0 && gy != grid_h_i32 - 1 {
                grid_w
            } else {
                0
            };
            let i00 = y0.saturating_mul(grid_w).saturating_add(x0);
            render(Tile {
                x: base_x,
                y: clipped_y,
                width,
                height,
                local_y: clipped_y - base_y,
                i00,
                i01: i00.saturating_add(next_x),
                i10: i00.saturating_add(next_y),
                i11: i00.saturating_add(next_y).saturating_add(next_x),
            });
        }
    }
}

fn motion_value(lut: &[i16; LUT_LEN], plane: &[u16], index: usize) -> i32 {
    let lut_index = usize::from(plane.get(index).copied().unwrap_or(NEUTRAL)).min(LUT_LEN - 1);
    i32::from(lut[lut_index])
}

fn interpolate_row<const SHIFTED: bool>(
    samples: [i32; 4],
    params: &PlaneParams,
    y: i32,
    interp: bool,
) -> [i32; 2] {
    if !interp {
        return [samples[0]; 2];
    }
    let y = usize::try_from(y).unwrap_or(0).min(MAX_BLOCK_SIZE - 1);
    let weights = params.y_weights[y];
    [
        interp_pair::<SHIFTED>(
            samples[0],
            samples[2],
            params.block_h,
            params.y_shift,
            weights,
        ),
        interp_pair::<SHIFTED>(
            samples[1],
            samples[3],
            params.block_h,
            params.y_shift,
            weights,
        ),
    ]
}

fn interpolate_motion_row<const SHIFTED: bool>(
    samples: MotionSamples,
    params: &PlaneParams,
    y: i32,
    interp: bool,
) -> MotionRow {
    MotionRow {
        x: interpolate_row::<SHIFTED>(samples.x, params, y, interp),
        y: interpolate_row::<SHIFTED>(samples.y, params, y, interp),
    }
}

fn interpolate_motion<const SHIFTED: bool>(
    row: MotionRow,
    params: &PlaneParams,
    weights: (i32, i32),
    interp: bool,
) -> (i32, i32) {
    if !interp {
        return (row.x[0], row.y[0]);
    }
    (
        interp_pair::<SHIFTED>(row.x[0], row.x[1], params.block_w, params.x_shift, weights),
        interp_pair::<SHIFTED>(row.y[0], row.y[1], params.block_w, params.x_shift, weights),
    )
}

#[allow(clippy::many_single_char_names)]
fn alpha_row<const SHIFTED: bool>(
    mask: &[u8],
    params: &PlaneParams,
    tile: &Tile,
    y: i32,
    interp: bool,
) -> [i32; 2] {
    let a = i32::from(mask.get(tile.i00).copied().unwrap_or(0));
    let b = i32::from(mask.get(tile.i01).copied().unwrap_or(0));
    let c = i32::from(mask.get(tile.i10).copied().unwrap_or(0));
    let d = i32::from(mask.get(tile.i11).copied().unwrap_or(0));
    interpolate_row::<SHIFTED>([a, b, c, d], params, y, interp)
}

fn alpha_at_row<const SHIFTED: bool>(
    row: [i32; 2],
    params: &PlaneParams,
    weights: (i32, i32),
    interp: bool,
) -> u8 {
    if !interp {
        return byte_from_i32(row[0]);
    }
    byte_from_i32(interp_pair::<SHIFTED>(
        row[0],
        row[1],
        params.block_w,
        params.x_shift,
        weights,
    ))
}

fn alpha_for_origin<const SHIFTED: bool>(
    mask: &[u8],
    row: [i32; 2],
    params: &PlaneParams,
    mask_params: Option<&PlaneParams>,
    stride: usize,
    pos: AlphaPos,
    weights: (i32, i32),
) -> u8 {
    if let Some(mask_params) = mask_params {
        alpha_at_absolute(mask, mask_params, stride, pos.plane_x, pos.plane_y)
    } else {
        alpha_at_row::<SHIFTED>(row, params, weights, true)
    }
}

fn alpha_at_absolute(mask: &[u8], params: &PlaneParams, stride: usize, x: i32, y: i32) -> u8 {
    if params.block_w <= 0 || params.block_h <= 0 {
        return 0;
    }
    let mut ix0 = 0;
    let mut ix1 = 0;
    if params.origin_x <= x {
        ix0 = (x - params.origin_x) / params.block_w;
        ix1 = ix0 + 1;
    }
    let mut iy0 = 0;
    let mut iy1 = 0;
    if params.origin_y <= y {
        iy0 = (y - params.origin_y) / params.block_h;
        iy1 = iy0 + 1;
    }

    let fx = if params.origin_x < x {
        (x - params.origin_x) % params.block_w
    } else {
        0
    };
    let fy = if params.origin_y < y {
        (y - params.origin_y) % params.block_h
    } else {
        0
    };

    let top_left = i32::from(mask_at(mask, stride, ix0, iy0));
    let top_right = i32::from(mask_at(mask, stride, ix1, iy0));
    let bottom_left = i32::from(mask_at(mask, stride, ix0, iy1));
    let bottom_right = i32::from(mask_at(mask, stride, ix1, iy1));
    let top = ((params.block_w - fx) * top_left + fx * top_right) / params.block_w;
    let bottom = ((params.block_w - fx) * bottom_left + fx * bottom_right) / params.block_w;
    byte_from_i32(((params.block_h - fy) * top + fy * bottom) / params.block_h)
}

fn interp_pair<const SHIFTED: bool>(
    a: i32,
    b: i32,
    total: i32,
    shift: i32,
    weights: (i32, i32),
) -> i32 {
    if SHIFTED {
        return (weights.0 * a + weights.1 * b) >> shift;
    }
    if shift < 0 {
        return (weights.0 * a + weights.1 * b) / total;
    }
    (weights.0 * a + weights.1 * b) >> shift
}

fn interpolation_weights(total: i32, shift: i32) -> [(i32, i32); MAX_BLOCK_SIZE] {
    let mut weights = [(0, 0); MAX_BLOCK_SIZE];
    if total <= 0 {
        return weights;
    }
    let scale = 1i32
        .checked_shl(u32::try_from(shift.max(0)).unwrap_or(0))
        .unwrap_or(0);
    for (position, weights) in weights
        .iter_mut()
        .enumerate()
        .take(usize::try_from(total).unwrap_or(0).min(MAX_BLOCK_SIZE))
    {
        let position = i32::try_from(position).unwrap_or(0);
        *weights = if shift < 0 {
            (total - position, position)
        } else if scale == 0 {
            (1, 0)
        } else {
            (
                coeff(total - position, total, scale),
                coeff(position, total, scale),
            )
        };
    }
    weights
}

fn coeff(value: i32, total: i32, scale: i32) -> i32 {
    trunc_f64_to_i32((f64::from(value * scale) / f64::from(total) - 0.5).ceil())
}

fn plane_covers(plane: Plane<'_>, params: &PlaneParams) -> bool {
    if plane.super_shift != 0 && 1i32.checked_shl(plane.super_shift) != Some(params.source_step) {
        return false;
    }
    let (Ok(x), Ok(y)) = (usize::try_from(params.max_x), usize::try_from(params.max_y)) else {
        return false;
    };
    plane_index_checked(plane, x, y).is_some_and(|last| last < plane.data.len())
}

fn plane_index(plane: Plane<'_>, x: usize, y: usize) -> usize {
    if plane.super_shift == 0 {
        return y * plane.stride + x;
    }
    let mask = (1usize << plane.super_shift) - 1;
    let subplane = ((y & mask) << plane.super_shift) | (x & mask);
    subplane * plane.super_span + (y >> plane.super_shift) * plane.stride + (x >> plane.super_shift)
}

fn plane_index_checked(plane: Plane<'_>, x: usize, y: usize) -> Option<usize> {
    if plane.super_shift == 0 {
        return y.checked_mul(plane.stride)?.checked_add(x);
    }
    let mask = (1usize << plane.super_shift) - 1;
    let subplane = ((y & mask) << plane.super_shift) | (x & mask);
    subplane
        .checked_mul(plane.super_span)?
        .checked_add((y >> plane.super_shift).checked_mul(plane.stride)?)?
        .checked_add(x >> plane.super_shift)
}

fn tile_interior(params: &PlaneParams, tile: &Tile, motions: &[MotionSamples]) -> bool {
    if params.source_step <= 0 {
        return false;
    }
    let (mut min_x, mut max_x, mut min_y, mut max_y) = (i32::MAX, i32::MIN, i32::MAX, i32::MIN);
    for motion in motions {
        for value in motion.x {
            min_x = min_x.min(value);
            max_x = max_x.max(value);
        }
        for value in motion.y {
            min_y = min_y.min(value);
            max_y = max_y.max(value);
        }
    }
    let step = i64::from(params.source_step);
    let x0 = i64::from(tile.x) * step;
    let y0 = i64::from(tile.y) * step;
    let x1 = i64::from(tile.x + tile.width - 1) * step;
    let y1 = i64::from(tile.y + tile.height - 1) * step;
    x0 + i64::from(min_x) >= 0
        && y0 + i64::from(min_y) >= 0
        && x1 + i64::from(max_x) <= i64::from(params.max_x)
        && y1 + i64::from(max_y) <= i64::from(params.max_y)
}

#[allow(clippy::cast_sign_loss, clippy::inline_always, unsafe_code)]
#[inline(always)]
fn sample_plane(
    plane: Plane<'_>,
    params: &PlaneParams,
    source_x: i32,
    source_y: i32,
    dx: i32,
    dy: i32,
    direct: bool,
) -> u8 {
    let x = source_x + dx;
    let y = source_y + dy;
    if direct {
        let index = plane_index(plane, x as usize, y as usize);
        return unsafe { *plane.data.get_unchecked(index) };
    }
    let x = usize::try_from(x.clamp(0, params.max_x)).unwrap_or(0);
    let y = usize::try_from(y.clamp(0, params.max_y)).unwrap_or(0);
    plane
        .data
        .get(plane_index(plane, x, y))
        .copied()
        .unwrap_or(0)
}

fn output_row<'a>(
    dst: &'a mut [u8],
    stride: usize,
    tile: &Tile,
    y: i32,
    row_start: i32,
) -> Option<&'a mut [u8]> {
    let row = usize::try_from(tile.y + y - row_start).ok()?;
    let x = usize::try_from(tile.x).ok()?;
    let width = usize::try_from(tile.width).ok()?;
    let start = row.checked_mul(stride)?.checked_add(x)?;
    dst.get_mut(start..start.checked_add(width)?)
}

pub fn max_mask(masks: MaskPlanes<'_>) -> Vec<u8> {
    masks
        .a
        .iter()
        .zip(masks.b.iter())
        .map(|(a, b)| (*a).max(*b))
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn mode11_or_13_pixel(
    mode13: bool,
    cur0: u8,
    cur1: u8,
    mv0: u8,
    mv1: u8,
    q0: i32,
    _q1: i32,
    alpha: Option<u8>,
    blend_order: BlendOrder,
) -> u8 {
    let motion = if mode13 {
        mode13_pixel(cur0, cur1, mv0, mv1, q0)
    } else {
        match blend_order {
            BlendOrder::Forward => blend_256(mv0, mv1, q0),
        }
    };
    if let Some(alpha) = alpha {
        let base = blend_256(cur0, cur1, q0);
        blend_255(motion, base, i32::from(alpha))
    } else {
        motion
    }
}

fn splat_coverage(
    ctx: &VectorContext<'_>,
    vectors: &[Vector],
    accum: &mut [i32],
    width: usize,
    height: usize,
    threshold: i32,
) {
    let grid_w = i32::try_from(ctx.grid_w).unwrap_or(i32::MAX);
    let grid_h = i32::try_from(ctx.grid_h).unwrap_or(i32::MAX);
    let out_w = i32::try_from(width).unwrap_or(i32::MAX);
    let out_h = i32::try_from(height).unwrap_or(i32::MAX);
    if grid_w <= 0 || grid_h <= 0 || out_w <= 0 || out_h <= 0 {
        return;
    }

    let x_step = ctx.block_w.saturating_sub(ctx.origin_x);
    let y_step = ctx.block_h.saturating_sub(ctx.origin_y);
    let denom = ctx.scale_shift_base.checked_shl(8).unwrap_or(0);
    if x_step == 0 || y_step == 0 || denom == 0 {
        return;
    }

    let x_bias = 1i32.saturating_sub(x_step);
    let y_bias = 1i32.saturating_sub(y_step);
    let mut y = 0;
    while y < grid_h {
        let mut x = 0;
        let mut x_base: i32 = 0;
        let mut neg_x_base: i32 = 0;
        while x < grid_w {
            let vector = vectors
                .get(
                    usize::try_from(y)
                        .unwrap_or(usize::MAX)
                        .saturating_mul(ctx.grid_w)
                        .saturating_add(usize::try_from(x).unwrap_or(usize::MAX)),
                )
                .copied()
                .unwrap_or_default();

            let dx = threshold.saturating_mul(i32::from(vector.dx)) / denom;
            let dy = threshold.saturating_mul(i32::from(vector.dy)) / denom;
            let shifted_y = y_step.saturating_mul(y).saturating_add(dy);
            let left = div_toward_zero(
                x_base
                    .saturating_add(dx)
                    .saturating_add(x_bias & ((x_base.saturating_add(dx)) >> 31)),
                x_step,
            );
            let top = div_toward_zero(shifted_y.saturating_add(y_bias & (shifted_y >> 31)), y_step);
            let right = left.saturating_add(1);
            let bottom = top.saturating_add(1);
            let next_x = right.saturating_mul(x_step);
            let left_weight = neg_x_base.saturating_add(next_x).saturating_sub(dx);
            let top_weight = y_step.saturating_mul(bottom).saturating_sub(shifted_y);

            if top >= 0 && top < out_h && left >= 0 && left < out_w {
                add_3x3(
                    accum,
                    width,
                    left,
                    top,
                    left_weight.saturating_mul(top_weight),
                );
            }
            if left >= -1 && top >= 0 && top < out_h && right < out_w {
                let weight = top_weight.saturating_mul(
                    x_base
                        .saturating_add(dx)
                        .saturating_add(ctx.block_w)
                        .saturating_sub(next_x),
                );
                add_3x3(accum, width, right, top, weight);
            }
            if top >= -1 && bottom < out_h && left >= -1 && right < out_w {
                let weight = x_base
                    .saturating_add(ctx.block_w)
                    .saturating_add(dx)
                    .saturating_sub(next_x)
                    .saturating_mul(ctx.block_h.saturating_sub(top_weight));
                add_3x3(accum, width, right, bottom, weight);
            }
            if left >= 0 && top >= -1 && bottom < out_h && left < out_w {
                let weight = left_weight.saturating_mul(ctx.block_h.saturating_sub(top_weight));
                add_3x3(accum, width, left, bottom, weight);
            }

            x += 1;
            x_base = x_base.saturating_add(x_step);
            neg_x_base = neg_x_base.saturating_sub(x_step);
        }
        y += 1;
    }
}

fn finish_coverage(
    ctx: &VectorContext<'_>,
    accum: &[i32],
    out: &mut [u8],
    width: usize,
    height: usize,
    strength_percent: i32,
) {
    let stride = width.saturating_add(2);
    let area = ctx.area();
    let strength = f64::from(strength_percent) / 100.0;
    let area_f = f64::from(area);
    for y in 0..height {
        for x in 0..width {
            let out_index = y.saturating_mul(width).saturating_add(x);
            let accum_index = y
                .saturating_add(1)
                .saturating_mul(stride)
                .saturating_add(x.saturating_add(1));
            if out_index >= out.len() || accum_index >= accum.len() {
                return;
            }
            let covered = accum[accum_index] >> 3;
            let remaining = if area <= covered { 0 } else { area - covered };
            let value = f64::from(remaining) * strength * 256.0 / area_f;
            out[out_index] = coverage_byte(value);
        }
    }
}

fn add_3x3(accum: &mut [i32], width: usize, x: i32, y: i32, weight: i32) {
    let stride = width.saturating_add(2);
    let Ok(x) = usize::try_from(x) else {
        return;
    };
    let Ok(y) = usize::try_from(y) else {
        return;
    };
    let base = y.saturating_mul(stride).saturating_add(x);
    for row in 0..3 {
        let index = base.saturating_add(row * stride);
        for col in 0..3 {
            if let Some(cell) = accum.get_mut(index.saturating_add(col)) {
                *cell = cell.saturating_add(weight);
            }
        }
    }
}

fn div_toward_zero(value: i32, divisor: i32) -> i32 {
    value / divisor
}

fn cpu_map(value: i32) -> i32 {
    let index = value.saturating_sub(1);
    if (0..32).contains(&index) {
        CPU_MAP[usize::try_from(index).unwrap_or(0)]
    } else {
        -1
    }
}

fn trunc_div_256(value: i32) -> i16 {
    i16::try_from(value / 256).unwrap_or(if value < 0 { i16::MIN } else { i16::MAX })
}

#[allow(clippy::cast_possible_truncation)]
fn trunc_f64_to_i32(value: f64) -> i32 {
    value as i32
}

fn fill_prefix(dst: &mut [u8], len: usize, value: u8) {
    let len = len.min(dst.len());
    dst[..len].fill(value);
}

fn mask_at(mask: &[u8], stride: usize, x: i32, y: i32) -> u8 {
    let Ok(x) = usize::try_from(x) else {
        return 0;
    };
    let Ok(y) = usize::try_from(y) else {
        return 0;
    };
    mask.get(y.saturating_mul(stride).saturating_add(x))
        .copied()
        .unwrap_or(0)
}

fn blend_255(a: u8, b: u8, weight: i32) -> u8 {
    byte_from_i32((i32::from(a) * (255 - weight) + i32::from(b) * weight + 255) >> 8)
}

fn blend_256(a: u8, b: u8, weight: i32) -> u8 {
    byte_from_i32((i32::from(a) * (256 - weight) + i32::from(b) * weight + 128) >> 8)
}

fn clamp_between(value: u8, a: u8, b: u8) -> u8 {
    value.clamp(a.min(b), a.max(b))
}

fn byte_from_i32(value: i32) -> u8 {
    u8::try_from(value.clamp(0, 255)).unwrap_or(0)
}

fn corrected_vector(ctx: &VectorContext<'_>, vector: Vector, _x: usize, _y: usize) -> (i32, i32) {
    let raw_dx = i32::from(vector.dx);
    let raw_dy = i32::from(vector.dy);
    if ctx.raw || raw_dx.abs() > MAX_VECTOR || raw_dy.abs() > MAX_VECTOR {
        return (raw_dx, raw_dy);
    }
    let scale = ctx.scale_shift_base.max(1);
    (raw_dx / scale, raw_dy / scale)
}

fn centered(value: i32) -> u16 {
    if value.abs() > MAX_VECTOR {
        NEUTRAL
    } else {
        u16::try_from(value.saturating_add(i32::from(NEUTRAL))).unwrap_or(NEUTRAL)
    }
}

fn scale_magnitude(magnitude: u32, area: i32, scale: f64, exponent: f64) -> u8 {
    let value = (4.0 * f64::from(magnitude) * scale / f64::from(area)).powf(exponent) * 255.0;
    clamp_u8(value)
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn clamp_u8(value: f64) -> u8 {
    if !value.is_finite() || value <= 0.0 {
        0
    } else if value >= 255.0 {
        255
    } else {
        value as u8
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn coverage_byte(value: f64) -> u8 {
    if !value.is_finite() {
        return 0;
    }
    let value = value as i32;
    if value >= 255 {
        255
    } else {
        (value & 0xFF) as u8
    }
}
use std::ops::Range;
