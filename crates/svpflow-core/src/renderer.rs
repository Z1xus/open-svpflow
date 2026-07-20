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

    pub const fn layout(&self) -> (usize, usize, u32) {
        (self.stride, self.super_span, self.super_shift)
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
    source0: [Plane<'a>; 2],
    source1: [Plane<'a>; 2],
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
        let (source, motion, lut, mask) = if mode1 {
            (source1, motion1, &self.inverse_lut, masks.map(|m| m.a))
        } else {
            (source0, motion0, &self.threshold_lut, masks.map(|m| m.b))
        };
        self.render_selected_warp(dst.y, source.y, motion, lut, mask, false, interp);
        self.render_selected_warp(dst.u, source.u, motion, lut, mask, true, interp);
        self.render_selected_warp(dst.v, source.v, motion, lut, mask, true, interp);
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
        let mut pixel = |cur0, cur1, mv0, mv1, alpha: Option<u8>, _: Option<u8>, _: Option<u8>| {
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
        };
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
            mode13 || max_mask.is_some(),
            &mut pixel,
        );
        self.render_dual_warp_uv(
            [dst.u, dst.v],
            [source0.u, source0.v],
            [source1.u, source1.v],
            motion0,
            motion1,
            [max_mask.as_deref(), None, None],
            interp,
            zero_origin,
            mode13 || max_mask.is_some(),
            0..self.config.height / self.config.chroma_y_div,
            &mut pixel,
        );
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn render_dual_warp_uv(
        &self,
        dst: [PlaneMut<'_>; 2],
        sources0: [Plane<'_>; 2],
        sources1: [Plane<'_>; 2],
        motion0: MotionPlanes<'_>,
        motion1: MotionPlanes<'_>,
        masks: [Option<&[u8]>; 3],
        interp: bool,
        zero_origin: bool,
        needs_bases: bool,
        rows: Range<i32>,
        mut pixel: impl FnMut(u8, u8, u8, u8, Option<u8>, Option<u8>, Option<u8>) -> u8,
    ) {
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        {
            let params = self.plane_params(true, zero_origin);
            let [dst_u, dst_v] = &dst;
            let fused = interp
                && !zero_origin
                && params.x_shift >= 0
                && params.y_shift >= 0
                && dst_u.stride == dst_v.stride
                && dst_u.data.len() == dst_v.data.len()
                && sources0
                    .iter()
                    .chain(sources1.iter())
                    .all(|plane| plane_covers(*plane, &params))
                && mode23_fast::layout_matches(sources0[0], sources0[1])
                && mode23_fast::layout_matches(sources1[0], sources1[1]);
            if fused {
                let [dst_u, dst_v] = dst;
                #[allow(unsafe_code)]
                let xw = unsafe { mode23_fast::x_weight_table(&params) };
                render_tiles(
                    &params,
                    self.config.grid_w,
                    self.config.grid_h,
                    interp,
                    rows.clone(),
                    |tile| {
                        let motion0_samples = Self::motion_samples(
                            motion0,
                            &self.threshold_lut,
                            &params,
                            &tile,
                            interp,
                        );
                        let motion1_samples =
                            Self::motion_samples(motion1, &self.inverse_lut, &params, &tile, interp);
                        let direct = tile_interior(
                            &params,
                            &tile,
                            &[motion0_samples, motion1_samples],
                        );
                        let corners = |m: &[u8]| {
                            [tile.i00, tile.i01, tile.i10, tile.i11]
                                .map(|i| i32::from(m.get(i).copied().unwrap_or(0)))
                        };
                        macro_rules! fast_tile {
                            ($direct:literal, $bases:literal) => {
                                mode23_fast::warp_tile::<true, $direct, true, $bases>(
                                    dst_u.data,
                                    dst_u.stride,
                                    dst_v.data,
                                    dst_v.stride,
                                    sources0,
                                    sources1,
                                    [motion0_samples, motion1_samples],
                                    masks.map(|mask| mask.map_or([0; 4], corners)),
                                    masks.map(|mask| mask.is_some()),
                                    &params,
                                    &tile,
                                    rows.start,
                                    &xw,
                                    &mut pixel,
                                )
                            };
                        }
                        #[allow(unsafe_code)]
                        unsafe {
                            match (direct, needs_bases) {
                                (true, true) => fast_tile!(true, true),
                                (true, false) => fast_tile!(true, false),
                                (false, true) => fast_tile!(false, true),
                                (false, false) => fast_tile!(false, false),
                            }
                        }
                    },
                );
                return;
            }
        }
        let [dst_u, dst_v] = dst;
        self.render_dual_warp_rows(
            dst_u,
            sources0[0],
            sources1[0],
            motion0,
            motion1,
            masks[0],
            masks[1],
            masks[2],
            true,
            interp,
            zero_origin,
            needs_bases,
            rows.clone(),
            &mut pixel,
        );
        self.render_dual_warp_rows(
            dst_v,
            sources0[1],
            sources1[1],
            motion0,
            motion1,
            masks[0],
            masks[1],
            masks[2],
            true,
            interp,
            zero_origin,
            needs_bases,
            rows,
            &mut pixel,
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
        let mut pixel = |cur0, cur1, mv0, mv1, alpha0, alpha1, final_alpha| {
            self.mode21_or_22_pixel(mode21, cur0, cur1, mv0, mv1, alpha0, alpha1, final_alpha)
        };
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
            !mode21 || final_mask.is_some(),
            &mut pixel,
        );
        self.render_dual_warp_uv(
            [dst.u, dst.v],
            [source0.u, source0.v],
            [source1.u, source1.v],
            motion0,
            motion1,
            [Some(masks.a), Some(masks.b), final_mask],
            interp,
            !interp,
            !mode21 || final_mask.is_some(),
            0..self.config.height / self.config.chroma_y_div,
            &mut pixel,
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
                let (source, motion, lut, mask) = if mode == 1 {
                    (
                        input.source0,
                        input.motion0,
                        &self.threshold_lut,
                        input.masks.map(|m| m.b),
                    )
                } else {
                    (
                        input.source1,
                        input.motion1,
                        &self.inverse_lut,
                        input.masks.map(|m| m.a),
                    )
                };
                self.render_selected_warp_rows(
                    dst,
                    source,
                    motion,
                    lut,
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
                    mode == 13 || max_mask.is_some(),
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
                    mode == 22 || input.final_mask.is_some(),
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

    pub fn render_mode21_or_22_uv_rows(
        &self,
        mode21: bool,
        interp: bool,
        dst: [PlaneMut<'_>; 2],
        input: PlaneRenderInput<'_>,
        second: [Plane<'_>; 2],
        rows: Range<i32>,
    ) -> Result<(), RenderError> {
        let masks = input.masks.ok_or(RenderError::MissingMasks)?;
        let [second_source0, second_source1] = second;
        let mut pixel = |cur0, cur1, mv0, mv1, alpha0, alpha1, final_alpha| {
            self.mode21_or_22_pixel(mode21, cur0, cur1, mv0, mv1, alpha0, alpha1, final_alpha)
        };
        self.render_dual_warp_uv(
            dst,
            [input.source0, second_source0],
            [input.source1, second_source1],
            input.motion0,
            input.motion1,
            [Some(masks.a), Some(masks.b), input.final_mask],
            interp,
            !interp,
            !mode21 || input.final_mask.is_some(),
            rows,
            &mut pixel,
        );
        Ok(())
    }

    pub fn render_mode23_uv_rows(
        &self,
        interp: bool,
        dst: [PlaneMut<'_>; 2],
        input: PlaneRenderInput<'_>,
        second: [Plane<'_>; 2],
        rows: Range<i32>,
    ) -> Result<(), RenderError> {
        let [dst, second_dst] = dst;
        let [second_source0, second_source1] = second;
        let input = Mode23PlaneInput {
            source0: [input.source0, second_source0],
            source1: [input.source1, second_source1],
            base0: input.motion0,
            base1: input.motion1,
            next0: input.motion2.ok_or(RenderError::MissingMotion)?,
            prev1: input.motion3.ok_or(RenderError::MissingMotion)?,
            masks: input.masks.ok_or(RenderError::MissingMasks)?,
            mask: input.final_mask,
            chroma: true,
        };
        self.render_mode23_plane_rows_dispatch::<true>([dst, second_dst], &input, interp, rows);
        Ok(())
    }

    #[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
    fn render_selected_warp(
        &self,
        dst: PlaneMut<'_>,
        source: Plane<'_>,
        motion: MotionPlanes<'_>,
        lut: &[i16; LUT_LEN],
        mask: Option<&[u8]>,
        chroma: bool,
        interp: bool,
    ) {
        let height = if chroma {
            self.config.height / self.config.chroma_y_div
        } else {
            self.config.height
        };
        self.render_selected_warp_rows(dst, source, motion, lut, mask, chroma, interp, 0..height);
    }

    #[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
    fn render_selected_warp_rows(
        &self,
        dst: PlaneMut<'_>,
        source: Plane<'_>,
        motion: MotionPlanes<'_>,
        lut: &[i16; LUT_LEN],
        mask: Option<&[u8]>,
        chroma: bool,
        interp: bool,
        rows: Range<i32>,
    ) {
        let zero_origin = !interp;
        let params = self.plane_params(chroma, zero_origin);
        let mask_params = zero_origin.then(|| self.plane_params(chroma, false));
        let row_start = rows.start;
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        #[allow(unsafe_code)]
        let xw = unsafe { mode23_fast::x_weight_table(&params) };
        render_tiles(
            &params,
            self.config.grid_w,
            self.config.grid_h,
            interp,
            rows,
            |tile| {
                let motion_samples = Self::motion_samples(motion, lut, &params, &tile, interp);
                let covers = plane_covers(source, &params);
                let direct = covers && tile_interior(&params, &tile, &[motion_samples]);
                #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
                if interp && params.x_shift >= 0 && params.y_shift >= 0 && covers {
                    let corners = |m: &[u8]| {
                        [tile.i00, tile.i01, tile.i10, tile.i11]
                            .map(|i| i32::from(m.get(i).copied().unwrap_or(0)))
                    };
                    let mut pixel = |base, _, warped, _, alpha: Option<u8>, _, _| {
                        alpha.map_or(warped, |alpha| mode1_pixel(base, warped, alpha))
                    };
    let mut unused = [0u8; 0];
                    macro_rules! fast_tile {
                        ($direct:literal, $bases:literal) => {
                            mode23_fast::warp_tile::<false, $direct, false, $bases>(
                                dst.data,
                                dst.stride,
                                &mut unused,
                                0,
                                [source; 2],
                                [source; 2],
                                [motion_samples; 2],
                                [mask.map_or([0; 4], corners), [0; 4], [0; 4]],
                                [mask.is_some(), false, false],
                                &params,
                                &tile,
                                row_start,
                                &xw,
                                &mut pixel,
                            )
                        };
                    }
                    #[allow(unsafe_code)]
                    unsafe {
                        match (direct, mask.is_some()) {
                            (true, true) => fast_tile!(true, true),
                            (true, false) => fast_tile!(true, false),
                            (false, true) => fast_tile!(false, true),
                            (false, false) => fast_tile!(false, false),
                        }
                    }
                    return;
                }
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

    #[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
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
        needs_bases: bool,
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
            needs_bases,
            0..height,
            pixel,
        );
    }

    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::fn_params_excessive_bools
    )]
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
        needs_bases: bool,
        rows: Range<i32>,
        mut pixel: impl FnMut(u8, u8, u8, u8, Option<u8>, Option<u8>, Option<u8>) -> u8,
    ) {
        let params = self.plane_params(chroma, zero_origin);
        let mask_params = zero_origin.then(|| self.plane_params(chroma, false));
        let row_start = rows.start;
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        #[allow(unsafe_code)]
        let xw = unsafe { mode23_fast::x_weight_table(&params) };
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
                let covers = plane_covers(source0, &params) && plane_covers(source1, &params);
                let direct =
                    covers && tile_interior(&params, &tile, &[motion0_samples, motion1_samples]);
                #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
                if interp && !zero_origin && params.x_shift >= 0 && params.y_shift >= 0 && covers {
                    let corners = |m: &[u8]| {
                        [tile.i00, tile.i01, tile.i10, tile.i11]
                            .map(|i| i32::from(m.get(i).copied().unwrap_or(0)))
                    };
    let mut unused = [0u8; 0];
                    macro_rules! fast_tile {
                        ($direct:literal, $bases:literal) => {
                            mode23_fast::warp_tile::<true, $direct, false, $bases>(
                                dst.data,
                                dst.stride,
                                &mut unused,
                                0,
                                [source0; 2],
                                [source1; 2],
                                [motion0_samples, motion1_samples],
                                [
                                    mask0.map_or([0; 4], corners),
                                    mask1.map_or([0; 4], corners),
                                    mask2.map_or([0; 4], corners),
                                ],
                                [mask0.is_some(), mask1.is_some(), mask2.is_some()],
                                &params,
                                &tile,
                                row_start,
                                &xw,
                                &mut pixel,
                            )
                        };
                    }
                    #[allow(unsafe_code)]
                    unsafe {
                        match (direct, needs_bases) {
                            (true, true) => fast_tile!(true, true),
                            (true, false) => fast_tile!(true, false),
                            (false, true) => fast_tile!(false, true),
                            (false, false) => fast_tile!(false, false),
                        }
                    }
                    return;
                }
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
            source0: [source0; 2],
            source1: [source1; 2],
            base0,
            base1,
            next0,
            prev1,
            masks,
            mask,
            chroma,
        };
        self.render_mode23_plane_rows_dispatch::<false>(
            [
                dst,
                PlaneMut {
                    data: &mut [],
                    stride: 0,
                },
            ],
            &input,
            interp,
            rows,
        );
    }

    fn render_mode23_plane_rows_dispatch<const DUAL: bool>(
        &self,
        dst: [PlaneMut<'_>; 2],
        input: &Mode23PlaneInput<'_>,
        interp: bool,
        rows: Range<i32>,
    ) {
        match (interp, input.mask.is_some()) {
            (true, false) => {
                self.render_mode23_plane_rows_variant::<true, false, DUAL>(dst, input, rows);
            }
            (true, true) => {
                self.render_mode23_plane_rows_variant::<true, true, DUAL>(dst, input, rows);
            }
            (false, false) => {
                self.render_mode23_plane_rows_variant::<false, false, DUAL>(dst, input, rows);
            }
            (false, true) => {
                self.render_mode23_plane_rows_variant::<false, true, DUAL>(dst, input, rows);
            }
        }
    }

    fn render_mode23_plane_rows_variant<const INTERP: bool, const MASK: bool, const DUAL: bool>(
        &self,
        dst: [PlaneMut<'_>; 2],
        input: &Mode23PlaneInput<'_>,
        rows: Range<i32>,
    ) {
        let shifted = if input.chroma {
            self.half_width_map >= 0 && self.half_height_map >= 0
        } else {
            self.width_map >= 0 && self.height_map >= 0
        };
        if shifted {
            self.render_mode23_plane_rows_inner::<INTERP, MASK, true, DUAL>(dst, input, rows);
        } else {
            self.render_mode23_plane_rows_inner::<INTERP, MASK, false, DUAL>(dst, input, rows);
        }
    }

    #[allow(clippy::needless_pass_by_value, clippy::too_many_lines)]
    fn render_mode23_plane_rows_inner<
        const INTERP: bool,
        const MASK: bool,
        const SHIFTED: bool,
        const DUAL: bool,
    >(
        &self,
        dst: [PlaneMut<'_>; 2],
        input: &Mode23PlaneInput<'_>,
        rows: Range<i32>,
    ) {
        let [dst, second_dst] = dst;
        let Mode23PlaneInput {
            source0: sources0,
            source1: sources1,
            base0,
            base1,
            next0,
            prev1,
            masks,
            mask,
            chroma,
        } = *input;
        let [source0, second_source0] = sources0;
        let [source1, second_source1] = sources1;
        let interp = INTERP;
        let mask = if MASK { mask } else { None };
        let zero_origin = !interp;
        let params = self.plane_params(chroma, zero_origin);
        let mask_params = zero_origin.then(|| self.plane_params(chroma, false));
        let row_start = rows.start;
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        #[allow(unsafe_code)]
        let xw = unsafe { mode23_fast::x_weight_table(&params) };
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
                let covers = plane_covers(source0, &params)
                    && plane_covers(source1, &params)
                    && (!DUAL
                        || plane_covers(second_source0, &params)
                            && plane_covers(second_source1, &params));
                let direct = covers
                    && tile_interior(
                        &params,
                        &tile,
                        &[
                            motion0_samples,
                            motion1_samples,
                            motion2_samples,
                            motion3_samples,
                        ],
                    );
                #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
                if INTERP && SHIFTED && covers {
                    let corners = |m: &[u8]| {
                        [tile.i00, tile.i01, tile.i10, tile.i11]
                            .map(|i| i32::from(m.get(i).copied().unwrap_or(0)))
                    };
                    macro_rules! fast_tile {
                        ($direct:literal) => {
                            mode23_fast::render_tile::<MASK, DUAL, $direct>(
                                dst.data,
                                dst.stride,
                                second_dst.data,
                                second_dst.stride,
                                [source0, second_source0],
                                [source1, second_source1],
                                [
                                    motion0_samples,
                                    motion1_samples,
                                    motion2_samples,
                                    motion3_samples,
                                ],
                                [corners(masks.a), corners(masks.b), mask.map_or([0; 4], corners)],
                                (self.threshold, self.threshold_limit),
                                &params,
                                &tile,
                                row_start,
                                &xw,
                            )
                        };
                    }
                    #[allow(unsafe_code)]
                    let handled = unsafe {
                        if direct {
                            fast_tile!(true)
                        } else {
                            fast_tile!(false)
                        }
                    };
                    if handled {
                        return;
                    }
                }
                for y in 0..tile.height {
                    let local_y = tile.local_y + y;
                    let source_y = (tile.y + y) * params.source_step;
                    let Some(output) = output_row(dst.data, dst.stride, &tile, y, row_start) else {
                        continue;
                    };
                    let second_output = if DUAL {
                        output_row(second_dst.data, second_dst.stride, &tile, y, row_start)
                    } else {
                        None
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
                    let render = |x: i32, output: &mut u8, second_output: Option<&mut u8>| {
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
                        let final_mask = mask.map(|mask| {
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
                        let gather = |planes, dx, dy| {
                            sample_plane_pair::<DUAL>(
                                planes, &params, source_x, source_y, dx, dy, direct,
                            )
                        };
                        let sources0 = [source0, second_source0];
                        let sources1 = [source1, second_source1];
                        let base_a = final_mask.map_or([0; 2], |_| gather(sources0, 0, 0));
                        let base_b = final_mask.map_or([0; 2], |_| gather(sources1, 0, 0));
                        let a = gather(sources0, dx0, dy0);
                        let b = gather(sources1, dx1, dy1);
                        let c = gather(sources0, dx2, dy2);
                        let d = gather(sources1, dx3, dy3);
                        let t0 = alpha_for_origin::<SHIFTED>(
                            masks.a,
                            mask0_row,
                            &params,
                            mask_params.as_ref(),
                            self.config.grid_w,
                            pos,
                            weights,
                        );
                        let t1 = alpha_for_origin::<SHIFTED>(
                            masks.b,
                            mask1_row,
                            &params,
                            mask_params.as_ref(),
                            self.config.grid_w,
                            pos,
                            weights,
                        );
                        let sample = |lane: usize| Mode23Sample {
                            a: a[lane],
                            b: b[lane],
                            c: c[lane],
                            d: d[lane],
                            base_a: base_a[lane],
                            base_b: base_b[lane],
                            t0,
                            t1,
                            mask: final_mask,
                        };
                        *output = mode23_pixel(sample(0), self.threshold, self.threshold_limit);
                        if let Some(output) = second_output {
                            *output = mode23_pixel(sample(1), self.threshold, self.threshold_limit);
                        }
                    };
                    if DUAL {
                        if let Some(second_output) = second_output {
                            for (x, (output, second_output)) in
                                (0..tile.width).zip(output.iter_mut().zip(second_output))
                            {
                                render(x, output, Some(second_output));
                            }
                        }
                    } else {
                        for (x, output) in (0..tile.width).zip(output) {
                            render(x, output, None);
                        }
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
    let step_x = ctx.block_w - ctx.origin_x;
    let step_y = ctx.block_h - ctx.origin_y;
    let block_x = ctx.block_w;
    let block_y = ctx.block_h;
    let frame_x = ctx.frame_w;
    let frame_y = ctx.frame_h;
    for y in 0..height {
        let pos_y = i32::try_from(y).unwrap_or(0).saturating_mul(step_y);
        for x in 0..width {
            let out = y.saturating_mul(width).saturating_add(x);
            if out >= dst_x.len() || out >= dst_y.len() {
                return;
            }
            let vector = vectors
                .get(sample_index(ctx, x, y))
                .copied()
                .unwrap_or_default();
            let dx = i32::from(vector.dx);
            let dy = i32::from(vector.dy);
            if dx.abs() > MAX_VECTOR || dy.abs() > MAX_VECTOR {
                dst_x[out] = NEUTRAL;
                dst_y[out] = NEUTRAL;
                continue;
            }
            let (dx, dy) = if ctx.raw {
                (dx, dy)
            } else {
                let pos_x = i32::try_from(x).unwrap_or(0).saturating_mul(step_x);
                (
                    clamp_to_frame(dx, pos_x, block_x, frame_x),
                    clamp_to_frame(dy, pos_y, block_y, frame_y),
                )
            };
            dst_x[out] = centered(dx);
            dst_y[out] = centered(dy);
        }
    }
}

fn clamp_to_frame(value: i32, pos: i32, block: i32, frame: i32) -> i32 {
    if value + pos < 0 {
        -pos
    } else if value + block + pos > frame {
        (frame - block - pos).max(0)
    } else {
        value
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
    thread_local! {
        static SCRATCH: RefCell<(Vec<i32>, Vec<i32>)> =
            const { RefCell::new((Vec::new(), Vec::new())) };
    }
    let stride = width.saturating_add(2);
    let len = stride.saturating_mul(height.saturating_add(2));
    SCRATCH.with(|scratch| {
        let (points, sums) = &mut *scratch.borrow_mut();
        points.clear();
        points.resize(len, 0);
        sums.resize(len, 0);
        let vectors = ctx.opposite_set(which);
        splat_coverage(ctx, vectors, points, width, height, threshold);
        window_3x3(points, sums, stride);
        finish_coverage(ctx, sums, out, width, height, strength_percent);
    });
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
        clamp_between(sample.d, sample.a, sample.b),
        i32::from(sample.t0),
    );
    let i1 = blend_255(
        sample.b,
        clamp_between(sample.c, sample.a, sample.b),
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

#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
mod mode23_fast {
    use core::arch::x86_64::{
        __m128i, _mm_add_epi16, _mm_cvtsi32_si128, _mm_extract_epi16, _mm_madd_epi16,
        _mm_mullo_epi16, _mm_set_epi16, _mm_set1_epi16, _mm_setzero_si128, _mm_sra_epi16,
        _mm_sra_epi32,
    };
    #[cfg(target_feature = "sse4.1")]
    use core::arch::x86_64::{
        _mm_add_epi32, _mm_and_si128, _mm_castps_si128, _mm_castsi128_ps, _mm_extract_epi32,
        _mm_max_epi32, _mm_min_epi32, _mm_mullo_epi32, _mm_or_si128, _mm_set1_epi32,
        _mm_shuffle_epi32, _mm_shuffle_ps, _mm_sll_epi32,
    };

    use super::{
        MAX_BLOCK_SIZE, Mode23Sample, MotionSamples, Plane, PlaneParams, Tile, byte_from_i32,
        load_plane, mode23_pixel, output_row, plane_index, sample_plane_pair,
    };

    #[target_feature(enable = "sse2")]
    #[allow(clippy::cast_possible_truncation)]
    fn pack8(v: [i32; 8]) -> __m128i {
        _mm_set_epi16(
            v[7] as i16,
            v[6] as i16,
            v[5] as i16,
            v[4] as i16,
            v[3] as i16,
            v[2] as i16,
            v[1] as i16,
            v[0] as i16,
        )
    }

    #[target_feature(enable = "sse2")]
    pub(super) fn x_weight_table(params: &PlaneParams) -> [__m128i; MAX_BLOCK_SIZE] {
        let mut table = [_mm_setzero_si128(); MAX_BLOCK_SIZE];
        for (entry, &(left, right)) in table.iter_mut().zip(&params.x_weights) {
            *entry = pack8([left, right, left, right, left, right, left, right]);
        }
        table
    }

    #[target_feature(enable = "sse2")]
    fn corner_pair(a: MotionSamples, b: MotionSamples, lo: usize, hi: usize) -> __m128i {
        pack8([
            a.x[lo], a.x[hi], a.y[lo], a.y[hi], b.x[lo], b.x[hi], b.y[lo], b.y[hi],
        ])
    }

    pub(super) fn layout_matches(a: Plane<'_>, b: Plane<'_>) -> bool {
        a.stride == b.stride && a.super_span == b.super_span && a.super_shift == b.super_shift
    }

    #[target_feature(enable = "sse2")]
    #[allow(clippy::cast_possible_truncation)]
    fn lane(v: __m128i, index: usize) -> i32 {
        i32::from(match index {
            0 => _mm_extract_epi16::<0>(v),
            1 => _mm_extract_epi16::<2>(v),
            2 => _mm_extract_epi16::<4>(v),
            _ => _mm_extract_epi16::<6>(v),
        } as i16)
    }

    #[allow(clippy::cast_sign_loss)]
    fn gather<const DUAL: bool>(planes: [Plane<'_>; 2], x: i32, y: i32) -> [u8; 2] {
        let index = plane_index(planes[0], x as usize, y as usize);
        load2::<DUAL>(planes, index)
    }

    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::cast_possible_truncation
    )]
    #[cfg_attr(not(target_feature = "sse4.1"), target_feature(enable = "sse2"))]
    #[cfg_attr(target_feature = "sse4.1", target_feature(enable = "sse2,sse4.1"))]
    pub(super) fn warp_tile<
        const TWO_FIELDS: bool,
        const DIRECT: bool,
        const DUAL: bool,
        const BASES: bool,
    >(
        output: &mut [u8],
        output_stride: usize,
        second: &mut [u8],
        second_stride: usize,
        sources0: [Plane<'_>; 2],
        sources1: [Plane<'_>; 2],
        motion: [MotionSamples; 2],
        mask_corners: [[i32; 4]; 3],
        mask_present: [bool; 3],
        params: &PlaneParams,
        tile: &Tile,
        row_start: i32,
        xw: &[__m128i; MAX_BLOCK_SIZE],
        pixel: &mut impl FnMut(u8, u8, u8, u8, Option<u8>, Option<u8>, Option<u8>) -> u8,
    ) {
        let top01 = corner_pair(motion[0], motion[1], 0, 1);
        let bot01 = corner_pair(motion[0], motion[1], 2, 3);
        let [m0, m1, m2] = mask_corners;
        let topm = pack8([m0[0], m0[1], m1[0], m1[1], m2[0], m2[1], 0, 0]);
        let botm = pack8([m0[2], m0[3], m1[2], m1[3], m2[2], m2[3], 0, 0]);
        let y_shift = _mm_cvtsi32_si128(params.y_shift);
        let x_shift = _mm_cvtsi32_si128(params.x_shift);
        let any_mask = mask_present.iter().any(|present| *present);
        #[cfg(target_feature = "sse4.1")]
        let shared = layout_matches(sources0[0], sources1[0]);
        for y in 0..tile.height {
            let Some(row) = output_row(output, output_stride, tile, y, row_start) else {
                continue;
            };
            let second_row = if DUAL {
                match output_row(second, second_stride, tile, y, row_start) {
                    Some(row) => Some(row),
                    None => continue,
                }
            } else {
                None
            };
            let local_y = usize::try_from(tile.local_y + y)
                .unwrap_or(0)
                .min(MAX_BLOCK_SIZE - 1);
            let (wy0, wy1) = params.y_weights[local_y];
            let (wy0, wy1) = (_mm_set1_epi16(wy0 as i16), _mm_set1_epi16(wy1 as i16));
            let vertical = |top, bottom| {
                _mm_sra_epi16(
                    _mm_add_epi16(_mm_mullo_epi16(top, wy0), _mm_mullo_epi16(bottom, wy1)),
                    y_shift,
                )
            };
            let row01 = vertical(top01, bot01);
            let rowm = vertical(topm, botm);
            let source_y = (tile.y + y) * params.source_step;
            let base_step = |plane: Plane<'_>| {
                if plane.super_shift == 0 {
                    params.source_step
                } else {
                    1
                }
            };
            #[allow(clippy::cast_sign_loss)]
            let row_base0 = if BASES {
                plane_index(sources0[0], 0, source_y as usize)
            } else {
                0
            };
            let base_step0 = base_step(sources0[0]);
            #[allow(clippy::cast_sign_loss)]
            let row_base1 = if BASES && TWO_FIELDS {
                plane_index(sources1[0], 0, source_y as usize)
            } else {
                row_base0
            };
            let base_step1 = if TWO_FIELDS {
                base_step(sources1[0])
            } else {
                base_step0
            };
            let weight = |x: i32| xw[usize::try_from(x).unwrap_or(0).min(MAX_BLOCK_SIZE - 1)];
            let alphas = |wv: __m128i| {
                if any_mask {
                    _mm_sra_epi32(_mm_madd_epi16(wv, rowm), x_shift)
                } else {
                    rowm
                }
            };
            #[allow(unused_mut)]
            let mut x_start = 0i32;
            #[allow(unused_mut)]
            let mut second_row = second_row;
            #[cfg(target_feature = "sse4.1")]
            if BASES && DIRECT && shared && sources0[0].super_shift == 0 {
                #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                let stride = sources0[0].stride as i32;
                let row_off = source_y * stride;
                #[allow(clippy::cast_sign_loss)]
                let take = usize::try_from(tile.width)
                    .unwrap_or(0)
                    .min(MAX_BLOCK_SIZE);
                let mut sx = tile.x * params.source_step;
                #[allow(clippy::cast_sign_loss)]
                let mut base_off0 = row_base0 + (tile.x * base_step0) as usize;
                #[allow(clippy::cast_sign_loss)]
                let mut base_off1 = row_base1 + (tile.x * base_step1) as usize;
                let mut tight = |wv: __m128i, out: &mut u8, second_out: Option<&mut u8>| {
                    let m01 = _mm_sra_epi32(_mm_madd_epi16(wv, row01), x_shift);
                    #[allow(clippy::cast_sign_loss)]
                    let index0 = (row_off + sx + lane(m01, 0) + lane(m01, 1) * stride) as usize;
                    let mv0 = load2::<DUAL>(sources0, index0);
                    let mv1 = if TWO_FIELDS {
                        #[allow(clippy::cast_sign_loss)]
                        let index1 =
                            (row_off + sx + lane(m01, 2) + lane(m01, 3) * stride) as usize;
                        load2::<DUAL>(sources1, index1)
                    } else {
                        mv0
                    };
                    let cur0 = if BASES {
                        load2::<DUAL>(sources0, base_off0)
                    } else {
                        [0; 2]
                    };
                    let cur1 = if BASES && TWO_FIELDS {
                        load2::<DUAL>(sources1, base_off1)
                    } else {
                        cur0
                    };
                    let am = if any_mask {
                        _mm_sra_epi32(_mm_madd_epi16(wv, rowm), x_shift)
                    } else {
                        rowm
                    };
                    let alpha =
                        |index: usize| mask_present[index].then(|| byte_from_i32(lane(am, index)));
                    *out = pixel(cur0[0], cur1[0], mv0[0], mv1[0], alpha(0), alpha(1), alpha(2));
                    if let Some(second_out) = second_out {
                        *second_out =
                            pixel(cur0[1], cur1[1], mv0[1], mv1[1], alpha(0), alpha(1), alpha(2));
                    }
                    sx += params.source_step;
                    #[allow(clippy::cast_sign_loss)]
                    {
                        base_off0 += base_step0 as usize;
                        base_off1 += base_step1 as usize;
                    }
                };
                match second_row.as_deref_mut() {
                    Some(second) => {
                        for ((&wv, out), second_out) in xw[..take]
                            .iter()
                            .zip(&mut row[..take])
                            .zip(&mut second[..take])
                        {
                            tight(wv, out, Some(second_out));
                        }
                    }
                    None => {
                        for (&wv, out) in xw[..take].iter().zip(&mut row[..take]) {
                            tight(wv, out, None);
                        }
                    }
                }
                #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                {
                    x_start = take as i32;
                }
            }
            let mut pixel_at = |x: i32, out: &mut u8, second_out: Option<&mut u8>| {
                let wv = weight(x);
                let m01 = _mm_sra_epi32(_mm_madd_epi16(wv, row01), x_shift);
                let am = alphas(wv);
                let px = tile.x + x;
                let source_x = px * params.source_step;
                #[allow(clippy::cast_sign_loss)]
                let cur0 = if BASES {
                    load2::<DUAL>(sources0, row_base0 + (px * base_step0) as usize)
                } else {
                    [0; 2]
                };
                #[allow(clippy::cast_sign_loss)]
                let cur1 = if BASES && TWO_FIELDS {
                    load2::<DUAL>(sources1, row_base1 + (px * base_step1) as usize)
                } else {
                    cur0
                };
                let fetch = |planes: [Plane<'_>; 2], dx: i32, dy: i32| {
                    if DIRECT {
                        gather::<DUAL>(planes, source_x + dx, source_y + dy)
                    } else {
                        sample_plane_pair::<DUAL>(planes, params, source_x, source_y, dx, dy, false)
                    }
                };
                let scattered = || {
                    let mv0 = fetch(sources0, lane(m01, 0), lane(m01, 1));
                    let mv1 = if TWO_FIELDS {
                        fetch(sources1, lane(m01, 2), lane(m01, 3))
                    } else {
                        mv0
                    };
                    (mv0, mv1)
                };
                #[cfg(target_feature = "sse4.1")]
                let (mv0, mv1) = if DIRECT && shared {
                    let dxs = _mm_shuffle_epi32::<0b1000>(m01);
                    let dys = _mm_shuffle_epi32::<0b1101>(m01);
                    let xs = _mm_add_epi32(dxs, _mm_set1_epi32(source_x));
                    let ys = _mm_add_epi32(dys, _mm_set1_epi32(source_y));
                    let index = gather_indices(sources0[0], xs, ys);
                    (
                        load2::<DUAL>(sources0, index[0]),
                        if TWO_FIELDS {
                            load2::<DUAL>(sources1, index[1])
                        } else {
                            load2::<DUAL>(sources0, index[0])
                        },
                    )
                } else {
                    scattered()
                };
                #[cfg(not(target_feature = "sse4.1"))]
                let (mv0, mv1) = scattered();
                let alpha =
                    |index: usize| mask_present[index].then(|| byte_from_i32(lane(am, index)));
                *out = pixel(cur0[0], cur1[0], mv0[0], mv1[0], alpha(0), alpha(1), alpha(2));
                if let Some(second_out) = second_out {
                    *second_out = pixel(cur0[1], cur1[1], mv0[1], mv1[1], alpha(0), alpha(1), alpha(2));
                }
            };
            if let Some(second_row) = second_row {
                for (x, (out, second_out)) in (x_start..tile.width).zip(
                    row[usize::try_from(x_start).unwrap_or(0)..]
                        .iter_mut()
                        .zip(&mut second_row[usize::try_from(x_start).unwrap_or(0)..]),
                ) {
                    pixel_at(x, out, Some(second_out));
                }
            } else {
                for (x, out) in
                    (x_start..tile.width).zip(&mut row[usize::try_from(x_start).unwrap_or(0)..])
                {
                    pixel_at(x, out, None);
                }
            }
        }
    }

    fn load2<const DUAL: bool>(planes: [Plane<'_>; 2], index: usize) -> [u8; 2] {
        [
            load_plane(planes[0], index),
            if DUAL { load_plane(planes[1], index) } else { 0 },
        ]
    }

    #[cfg(target_feature = "sse4.1")]
    #[target_feature(enable = "sse4.1")]
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    fn gather_indices(plane: Plane<'_>, xs: __m128i, ys: __m128i) -> [usize; 4] {
        let stride = _mm_set1_epi32(plane.stride as i32);
        let index = if plane.super_shift == 0 {
            _mm_add_epi32(_mm_mullo_epi32(ys, stride), xs)
        } else {
            let shift = _mm_cvtsi32_si128(plane.super_shift as i32);
            let mask = _mm_set1_epi32((1i32 << plane.super_shift) - 1);
            let sub = _mm_or_si128(
                _mm_sll_epi32(_mm_and_si128(ys, mask), shift),
                _mm_and_si128(xs, mask),
            );
            _mm_add_epi32(
                _mm_add_epi32(
                    _mm_mullo_epi32(sub, _mm_set1_epi32(plane.super_span as i32)),
                    _mm_mullo_epi32(_mm_sra_epi32(ys, shift), stride),
                ),
                _mm_sra_epi32(xs, shift),
            )
        };
        [
            _mm_extract_epi32::<0>(index) as usize,
            _mm_extract_epi32::<1>(index) as usize,
            _mm_extract_epi32::<2>(index) as usize,
            _mm_extract_epi32::<3>(index) as usize,
        ]
    }

    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::cast_possible_truncation
    )]
    #[cfg_attr(not(target_feature = "sse4.1"), target_feature(enable = "sse2"))]
    #[cfg_attr(target_feature = "sse4.1", target_feature(enable = "sse2,sse4.1"))]
    pub(super) fn render_tile<const MASK: bool, const DUAL: bool, const DIRECT: bool>(
        output: &mut [u8],
        output_stride: usize,
        second: &mut [u8],
        second_stride: usize,
        sources0: [Plane<'_>; 2],
        sources1: [Plane<'_>; 2],
        motion: [MotionSamples; 4],
        mask_corners: [[i32; 4]; 3],
        thresholds: (i32, i32),
        params: &PlaneParams,
        tile: &Tile,
        row_start: i32,
        xw: &[__m128i; MAX_BLOCK_SIZE],
    ) -> bool {
        if DUAL
            && !(layout_matches(sources0[0], sources0[1])
                && layout_matches(sources1[0], sources1[1]))
        {
            return false;
        }
        #[cfg(target_feature = "sse4.1")]
        let shared = layout_matches(sources0[0], sources1[0]);
        let top01 = corner_pair(motion[0], motion[1], 0, 1);
        let bot01 = corner_pair(motion[0], motion[1], 2, 3);
        let top23 = corner_pair(motion[2], motion[3], 0, 1);
        let bot23 = corner_pair(motion[2], motion[3], 2, 3);
        let [m0, m1, mf] = mask_corners;
        let topm = pack8([m0[0], m0[1], m1[0], m1[1], mf[0], mf[1], 0, 0]);
        let botm = pack8([m0[2], m0[3], m1[2], m1[3], mf[2], mf[3], 0, 0]);
        let y_shift = _mm_cvtsi32_si128(params.y_shift);
        let x_shift = _mm_cvtsi32_si128(params.x_shift);
        for y in 0..tile.height {
            let Some(row) = output_row(output, output_stride, tile, y, row_start) else {
                continue;
            };
            let second_row = if DUAL {
                match output_row(second, second_stride, tile, y, row_start) {
                    Some(row) => Some(row),
                    None => continue,
                }
            } else {
                None
            };
            let local_y = usize::try_from(tile.local_y + y)
                .unwrap_or(0)
                .min(MAX_BLOCK_SIZE - 1);
            let (wy0, wy1) = params.y_weights[local_y];
            let (wy0, wy1) = (_mm_set1_epi16(wy0 as i16), _mm_set1_epi16(wy1 as i16));
            let vertical = |top, bottom| {
                _mm_sra_epi16(
                    _mm_add_epi16(_mm_mullo_epi16(top, wy0), _mm_mullo_epi16(bottom, wy1)),
                    y_shift,
                )
            };
            let row01 = vertical(top01, bot01);
            let row23 = vertical(top23, bot23);
            let rowm = vertical(topm, botm);
            let source_y = (tile.y + y) * params.source_step;
            let pixel = |x: i32, out: &mut u8, second_out: Option<&mut u8>| {
                let wv = xw[usize::try_from(x).unwrap_or(0).min(MAX_BLOCK_SIZE - 1)];
                let m01 = _mm_sra_epi32(_mm_madd_epi16(wv, row01), x_shift);
                let m23 = _mm_sra_epi32(_mm_madd_epi16(wv, row23), x_shift);
                let mm = _mm_sra_epi32(_mm_madd_epi16(wv, rowm), x_shift);
                let t0 = byte_from_i32(lane(mm, 0));
                let t1 = byte_from_i32(lane(mm, 1));
                let final_mask = MASK.then(|| byte_from_i32(lane(mm, 2)));
                let source_x = (tile.x + x) * params.source_step;
                let scattered = |m01: __m128i, m23: __m128i| {
                    let fetch = |planes, dx, dy| {
                        if DIRECT {
                            gather::<DUAL>(planes, source_x + dx, source_y + dy)
                        } else {
                            sample_plane_pair::<DUAL>(
                                planes, params, source_x, source_y, dx, dy, false,
                            )
                        }
                    };
                    [
                        fetch(sources0, lane(m01, 0), lane(m01, 1)),
                        fetch(sources1, lane(m01, 2), lane(m01, 3)),
                        fetch(sources0, lane(m23, 0), lane(m23, 1)),
                        fetch(sources1, lane(m23, 2), lane(m23, 3)),
                    ]
                };
                #[cfg(target_feature = "sse4.1")]
                let [a, b, c, d] = if shared {
                    let dxs = _mm_castps_si128(_mm_shuffle_ps::<0x88>(
                        _mm_castsi128_ps(m01),
                        _mm_castsi128_ps(m23),
                    ));
                    let dys = _mm_castps_si128(_mm_shuffle_ps::<0xDD>(
                        _mm_castsi128_ps(m01),
                        _mm_castsi128_ps(m23),
                    ));
                    let mut xs = _mm_add_epi32(dxs, _mm_set1_epi32(source_x));
                    let mut ys = _mm_add_epi32(dys, _mm_set1_epi32(source_y));
                    if !DIRECT {
                        xs = _mm_min_epi32(
                            _mm_max_epi32(xs, _mm_setzero_si128()),
                            _mm_set1_epi32(params.max_x),
                        );
                        ys = _mm_min_epi32(
                            _mm_max_epi32(ys, _mm_setzero_si128()),
                            _mm_set1_epi32(params.max_y),
                        );
                    }
                    let index = gather_indices(sources0[0], xs, ys);
                    [
                        load2::<DUAL>(sources0, index[0]),
                        load2::<DUAL>(sources1, index[1]),
                        load2::<DUAL>(sources0, index[2]),
                        load2::<DUAL>(sources1, index[3]),
                    ]
                } else {
                    scattered(m01, m23)
                };
                #[cfg(not(target_feature = "sse4.1"))]
                let [a, b, c, d] = scattered(m01, m23);
                let (base_a, base_b) = if MASK {
                    (
                        gather::<DUAL>(sources0, source_x, source_y),
                        gather::<DUAL>(sources1, source_x, source_y),
                    )
                } else {
                    ([0; 2], [0; 2])
                };
                let sample = |lane: usize| Mode23Sample {
                    a: a[lane],
                    b: b[lane],
                    c: c[lane],
                    d: d[lane],
                    base_a: base_a[lane],
                    base_b: base_b[lane],
                    t0,
                    t1,
                    mask: final_mask,
                };
                *out = mode23_pixel(sample(0), thresholds.0, thresholds.1);
                if let Some(second_out) = second_out {
                    *second_out = mode23_pixel(sample(1), thresholds.0, thresholds.1);
                }
            };
            if let Some(second_row) = second_row {
                for (x, (out, second_out)) in (0..tile.width).zip(row.iter_mut().zip(second_row)) {
                    pixel(x, out, Some(second_out));
                }
            } else {
                for (x, out) in (0..tile.width).zip(row) {
                    pixel(x, out, None);
                }
            }
        }
        true
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

#[allow(clippy::inline_always, unsafe_code)]
#[inline(always)]
fn load_plane(plane: Plane<'_>, index: usize) -> u8 {
    unsafe { *plane.data.get_unchecked(index) }
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
        return load_plane(plane, index);
    }
    let x = usize::try_from(x.clamp(0, params.max_x)).unwrap_or(0);
    let y = usize::try_from(y.clamp(0, params.max_y)).unwrap_or(0);
    plane
        .data
        .get(plane_index(plane, x, y))
        .copied()
        .unwrap_or(0)
}

#[allow(clippy::cast_sign_loss, clippy::inline_always)]
#[inline(always)]
fn sample_plane_pair<const DUAL: bool>(
    planes: [Plane<'_>; 2],
    params: &PlaneParams,
    source_x: i32,
    source_y: i32,
    dx: i32,
    dy: i32,
    direct: bool,
) -> [u8; 2] {
    if !DUAL {
        return [
            sample_plane(planes[0], params, source_x, source_y, dx, dy, direct),
            0,
        ];
    }
    let x = source_x + dx;
    let y = source_y + dy;
    if direct
        && planes[0].stride == planes[1].stride
        && planes[0].super_span == planes[1].super_span
        && planes[0].super_shift == planes[1].super_shift
    {
        let index = plane_index(planes[0], x as usize, y as usize);
        return [load_plane(planes[0], index), load_plane(planes[1], index)];
    }
    [
        sample_plane(planes[0], params, source_x, source_y, dx, dy, direct),
        sample_plane(planes[1], params, source_x, source_y, dx, dy, direct),
    ]
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

#[allow(clippy::too_many_lines)]
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
    let stride = width.saturating_add(2);
    let (denom_div, x_div, y_div) = (
        TruncDiv::new(denom),
        TruncDiv::new(x_step),
        TruncDiv::new(y_step),
    );
    let mut y = 0;
    while y < grid_h {
        let vector_row = usize::try_from(y)
            .ok()
            .and_then(|y| vectors.get(y.saturating_mul(ctx.grid_w)..))
            .unwrap_or_default();
        let mut x = 0;
        let mut x_base: i32 = 0;
        while x < grid_w {
            let vector = usize::try_from(x)
                .ok()
                .and_then(|x| vector_row.get(x))
                .copied()
                .unwrap_or_default();

            let dx = denom_div.div(threshold * i32::from(vector.dx));
            let dy = denom_div.div(threshold * i32::from(vector.dy));
            let shifted_x = x_base.saturating_add(dx);
            let shifted_y = y_step.saturating_mul(y).saturating_add(dy);
            let left = x_div.div(shifted_x + (x_bias & (shifted_x >> 31)));
            let top = y_div.div(shifted_y + (y_bias & (shifted_y >> 31)));
            let right = left + 1;
            let bottom = top + 1;
            let next_x = right.saturating_mul(x_step);
            let left_weight = next_x - shifted_x;
            let top_weight = y_step.saturating_mul(bottom) - shifted_y;
            let right_weight = shifted_x + ctx.block_w - next_x;
            let bottom_weight = ctx.block_h - top_weight;
            let interior = left >= 0 && right < out_w && top >= 0 && bottom < out_h;
            #[allow(clippy::cast_sign_loss)]
            if interior
                && let base = (top as usize)
                    .saturating_mul(stride)
                    .saturating_add(left as usize)
                && let Some(rows) = accum.get_mut(base..base.saturating_add(stride).saturating_add(2))
            {
                let (top_row, bottom_row) = rows.split_at_mut(stride);
                top_row[0] += left_weight * top_weight;
                top_row[1] += top_weight * right_weight;
                bottom_row[0] += left_weight * bottom_weight;
                bottom_row[1] += right_weight * bottom_weight;
            } else {
                if top >= 0 && top < out_h && left >= 0 && left < out_w {
                    add_point(
                        accum,
                        width,
                        left,
                        top,
                        left_weight.saturating_mul(top_weight),
                    );
                }
                if left >= -1 && top >= 0 && top < out_h && right < out_w {
                    add_point(accum, width, right, top, top_weight.saturating_mul(right_weight));
                }
                if top >= -1 && bottom < out_h && left >= -1 && right < out_w {
                    add_point(
                        accum,
                        width,
                        right,
                        bottom,
                        right_weight.saturating_mul(bottom_weight),
                    );
                }
                if left >= 0 && top >= -1 && bottom < out_h && left < out_w {
                    add_point(
                        accum,
                        width,
                        left,
                        bottom,
                        left_weight.saturating_mul(bottom_weight),
                    );
                }
            }

            x += 1;
            x_base = x_base.saturating_add(x_step);
        }
        y += 1;
    }
}

fn window_3x3(points: &[i32], sums: &mut [i32], stride: usize) {
    if stride < 2 {
        sums.copy_from_slice(points);
        return;
    }
    for (source, row) in points
        .chunks_exact(stride)
        .zip(sums.chunks_exact_mut(stride))
    {
        row[0] = source[0];
        row[1] = source[1] + source[0];
        for (cell, ((a, b), c)) in row[2..]
            .iter_mut()
            .zip(source[2..].iter().zip(&source[1..]).zip(&source[..stride - 2]))
        {
            *cell = a + b + c;
        }
    }
    let rows = sums.len() / stride;
    for y in (1..rows).rev() {
        let (above, current) = sums.split_at_mut(y * stride);
        let current = &mut current[..stride];
        let row1 = &above[(y - 1) * stride..y * stride];
        if y >= 2 {
            let row2 = &above[(y - 2) * stride..(y - 1) * stride];
            for ((cell, a), b) in current.iter_mut().zip(row1).zip(row2) {
                *cell += a + b;
            }
        } else {
            for (cell, a) in current.iter_mut().zip(row1) {
                *cell += a;
            }
        }
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
    let byte = |remaining: i32| coverage_byte(f64::from(remaining) * strength * 256.0 / area_f);
    let lut: Vec<u8> = (0..=area.clamp(0, 4096)).map(byte).collect();
    for y in 0..height {
        let out_start = y.saturating_mul(width);
        let accum_start = y.saturating_add(1).saturating_mul(stride).saturating_add(1);
        let (Some(out_row), Some(accum_row)) = (
            out.get_mut(out_start..out_start.saturating_add(width)),
            accum.get(accum_start..accum_start.saturating_add(width)),
        ) else {
            return;
        };
        for (cell, &value) in out_row.iter_mut().zip(accum_row) {
            let covered = value >> 3;
            let remaining = if area <= covered { 0 } else { area - covered };
            *cell = lut
                .get(usize::try_from(remaining).unwrap_or(usize::MAX))
                .copied()
                .unwrap_or_else(|| byte(remaining));
        }
    }
}

fn add_point(accum: &mut [i32], width: usize, x: i32, y: i32, weight: i32) {
    let stride = width.saturating_add(2);
    let Ok(x) = usize::try_from(x) else {
        return;
    };
    let Ok(y) = usize::try_from(y) else {
        return;
    };
    if let Some(cell) = accum.get_mut(y.saturating_mul(stride).saturating_add(x)) {
        *cell = cell.saturating_add(weight);
    }
}

struct TruncDiv {
    divisor: i32,
    magic: u64,
    bound: u64,
}

impl TruncDiv {
    fn new(divisor: i32) -> Self {
        let (magic, bound) = if divisor > 0 {
            let d = u64::from(divisor.unsigned_abs());
            let magic = (1u64 << 40).div_ceil(d);
            let bound = ((1u64 << 40) / d).min(u64::MAX / magic).min(1u64 << 31);
            (magic, bound)
        } else {
            (0, 0)
        };
        Self {
            divisor,
            magic,
            bound,
        }
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn div(&self, value: i32) -> i32 {
        let abs = u64::from(value.unsigned_abs());
        if abs < self.bound {
            let quotient = ((abs * self.magic) >> 40) as i32;
            if value < 0 { -quotient } else { quotient }
        } else if self.divisor == 0 {
            0
        } else {
            value / self.divisor
        }
    }
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
    byte_from_i32((i32::from(a) * (256 - weight) + i32::from(b) * weight) >> 8)
}

fn clamp_between(value: u8, a: u8, b: u8) -> u8 {
    value.clamp(a.min(b), a.max(b))
}

fn byte_from_i32(value: i32) -> u8 {
    u8::try_from(value.clamp(0, 255)).unwrap_or(0)
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
use std::cell::RefCell;
use std::ops::Range;
