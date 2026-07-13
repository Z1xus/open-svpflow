#![forbid(unsafe_code)]
#![allow(
    clippy::missing_errors_doc,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]

use rayon::prelude::*;
use svpflow_core::metadata::{self, DecodedVector, DecodedVectors, VectorData, VectorRecord};
use svpflow_core::renderer::{
    self, CpuConfig, CpuRenderer, FramePlanes, MaskPlanes, MotionPlanes, Plane, PlaneMut,
    PlaneRenderInput, Vector, VectorContext,
};
#[cfg(target_arch = "wasm32")]
use svpflow1_vs::{Analyser, SuperBuilder, SuperFrame};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
pub use wasm_bindgen_rayon::init_thread_pool;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmSuper {
    builder: SuperBuilder,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmSuperFrame {
    frame: SuperFrame,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmAnalyser {
    analyser: Analyser,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmSuper {
    #[wasm_bindgen(constructor)]
    pub fn new(width: i32, height: i32, pel: i32) -> Result<Self, String> {
        Ok(Self {
            builder: SuperBuilder::new(width, height, pel)?,
        })
    }

    #[must_use]
    pub fn width(&self) -> i32 {
        self.builder.width()
    }

    #[must_use]
    pub fn height(&self) -> i32 {
        self.builder.height()
    }

    #[must_use]
    pub fn pel(&self) -> i32 {
        self.builder.pel()
    }

    #[must_use]
    pub fn levels(&self) -> i32 {
        self.builder.levels()
    }

    #[must_use]
    pub fn source_len(&self) -> usize {
        self.builder.source_len()
    }

    #[must_use]
    pub fn output_len(&self) -> usize {
        self.builder.output_len()
    }

    pub fn build(&self, source: &[u8]) -> Result<WasmSuperFrame, String> {
        Ok(WasmSuperFrame {
            frame: self.builder.build(source)?,
        })
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmSuperFrame {
    #[must_use]
    pub fn len(&self) -> usize {
        self.frame.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frame.is_empty()
    }

    #[must_use]
    pub fn bytes(&self) -> Vec<u8> {
        self.frame.bytes()
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmAnalyser {
    #[wasm_bindgen(constructor)]
    pub fn new(
        super_builder: &WasmSuper,
        block_width: i32,
        block_height: i32,
        overlap_mode: i32,
        vectors: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            analyser: Analyser::new(
                &super_builder.builder,
                block_width,
                block_height,
                overlap_mode,
                vectors,
            )?,
        })
    }

    #[must_use]
    pub fn vector_header(&self) -> Vec<i32> {
        self.analyser.vector_header()
    }

    pub fn analyse(
        &self,
        current: &WasmSuperFrame,
        reference: &WasmSuperFrame,
    ) -> Result<Vec<u8>, String> {
        self.analyser.analyse(&current.frame, &reference.frame)
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct WasmVectorPrep {
    data: VectorData,
    frame_w: i32,
    frame_h: i32,
    motion_w: usize,
    motion_h: usize,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct WasmFrameBlender {
    frame_len: usize,
    luma_len: usize,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl WasmFrameBlender {
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(constructor))]
    pub fn new(width: usize, height: usize) -> Result<Self, String> {
        if width == 0 || height == 0 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return Err("dimensions must be positive and even".into());
        }
        let luma_len = width.checked_mul(height).ok_or("frame size overflow")?;
        let frame_len = luma_len
            .checked_add(luma_len / 2)
            .ok_or("frame size overflow")?;
        Ok(Self {
            frame_len,
            luma_len,
        })
    }

    #[must_use]
    pub fn frame_len(&self) -> usize {
        self.frame_len
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn blend(&self, frames: &[u8], weights: &[f64], bright: bool) -> Result<Vec<u8>, String> {
        if weights.is_empty() || frames.len() != self.frame_len * weights.len() {
            return Err("invalid blend inputs".into());
        }
        if weights.iter().any(|weight| !weight.is_finite()) {
            return Err("invalid blend weight".into());
        }
        let sum: f64 = weights.iter().sum();
        if sum.abs() < f64::EPSILON {
            return Err("blend weights sum to zero".into());
        }

        let linear: Vec<f64> = if bright {
            {
                (0..=255)
                    .map(|value| (f64::from(value) / 255.0).powf(2.2))
                    .collect()
            }
        } else {
            Vec::new()
        };
        let mut output = vec![0u8; self.frame_len];
        output.par_iter_mut().enumerate().for_each(|(pixel, dst)| {
            let value = weights
                .iter()
                .enumerate()
                .map(|(frame, weight)| {
                    let sample = frames[frame * self.frame_len + pixel];
                    if bright && pixel < self.luma_len {
                        linear[usize::from(sample)] * weight
                    } else {
                        f64::from(sample) * weight
                    }
                })
                .sum::<f64>()
                / sum;
            *dst = if bright && pixel < self.luma_len {
                (value.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8
            } else {
                value.clamp(0.0, 255.0).round() as u8
            };
        });
        Ok(output)
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl WasmVectorPrep {
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(constructor))]
    pub fn new(header: &[i32], frame_width: i32, frame_height: i32) -> Result<Self, String> {
        if frame_width <= 0 || frame_height <= 0 {
            return Err("invalid frame dimensions".into());
        }
        let mut bytes = Vec::with_capacity(header.len() * 4);
        for value in header {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let VectorRecord::Ready(data) = metadata::vector_data(&bytes) else {
            return Err("invalid vector header".into());
        };
        let motion_grid = data.motion_grid();
        let motion_w = usize::try_from(motion_grid.width.max(0)).map_err(|_| "invalid grid")?;
        let motion_h = usize::try_from(motion_grid.height.max(0)).map_err(|_| "invalid grid")?;
        if motion_w == 0 || motion_h == 0 {
            return Err("empty motion grid".into());
        }
        Ok(Self {
            data,
            frame_w: frame_width,
            frame_h: frame_height,
            motion_w,
            motion_h,
        })
    }

    #[must_use]
    pub fn renderer_args(&self) -> Vec<i32> {
        let block = self.data.effective_block();
        let origin = self.data.origin();
        vec![
            block.width,
            block.height,
            origin.width,
            origin.height,
            i32::try_from(self.motion_w).unwrap_or(0),
            i32::try_from(self.motion_h).unwrap_or(0),
        ]
    }

    #[must_use]
    pub fn grid_len(&self) -> usize {
        self.motion_w * self.motion_h
    }

    pub fn motion(&self, payload: &[u8]) -> Result<Vec<u16>, String> {
        let decoded = self.decode(payload)?;
        let (a, b) = pair_sets(&decoded)?;
        let ctx = self.context(&a, &b);
        let count = self.grid_len();
        let mut out = vec![0u16; count * 4];
        let (m0, m1) = out.split_at_mut(count * 2);
        let (x0, y0) = m0.split_at_mut(count);
        let (x1, y1) = m1.split_at_mut(count);
        renderer::vector_planes(&ctx, 0, x0, y0, self.motion_w, self.motion_h);
        renderer::vector_planes(&ctx, 1, x1, y1, self.motion_w, self.motion_h);
        Ok(out)
    }

    pub fn motion23(
        &self,
        payload: &[u8],
        prev_payload: &[u8],
        next_payload: &[u8],
    ) -> Result<Vec<u16>, String> {
        let decoded = self.decode(payload)?;
        let (a, b) = pair_sets(&decoded)?;
        let prev = self.decode(prev_payload)?;
        let next = self.decode(next_payload)?;
        let prev_side = prev
            .current
            .as_deref()
            .or(prev.previous.as_deref())
            .ok_or("previous pair carries no vectors")?;
        let next_side = next
            .previous
            .as_deref()
            .or(next.current.as_deref())
            .ok_or("next pair carries no vectors")?;
        let prev_side = renderer_vectors(prev_side);
        let next_side = renderer_vectors(next_side);

        let ctx = self.context(&a, &b);
        let count = self.grid_len();
        let mut out = vec![0u16; count * 8];
        {
            let (main, side) = out.split_at_mut(count * 4);
            let (m0, m1) = main.split_at_mut(count * 2);
            let (x0, y0) = m0.split_at_mut(count);
            let (x1, y1) = m1.split_at_mut(count);
            renderer::vector_planes(&ctx, 0, x0, y0, self.motion_w, self.motion_h);
            renderer::vector_planes(&ctx, 1, x1, y1, self.motion_w, self.motion_h);

            let (m2, m3) = side.split_at_mut(count * 2);
            let (x2, y2) = m2.split_at_mut(count);
            let (x3, y3) = m3.split_at_mut(count);
            let next_ctx = VectorContext {
                a: &next_side,
                b: &b,
                ..ctx
            };
            renderer::vector_planes(&next_ctx, 0, x2, y2, self.motion_w, self.motion_h);
            let prev_ctx = VectorContext {
                a: &a,
                b: &prev_side,
                ..ctx
            };
            renderer::vector_planes(&prev_ctx, 1, x3, y3, self.motion_w, self.motion_h);
        }
        Ok(out)
    }

    pub fn masks(&self, payload: &[u8], phase: i32, cover: i32) -> Result<Vec<u8>, String> {
        let decoded = self.decode(payload)?;
        let (a, b) = pair_sets(&decoded)?;
        let ctx = self.context(&a, &b);
        let count = self.grid_len();
        let mut out = vec![0u8; count * 2];
        let (mask_a, mask_b) = out.split_at_mut(count);
        renderer::coverage_mask(
            &ctx,
            0,
            mask_a,
            self.motion_w,
            self.motion_h,
            cover,
            256 - phase,
        );
        renderer::coverage_mask(&ctx, 1, mask_b, self.motion_w, self.motion_h, cover, phase);
        Ok(out)
    }
}

impl WasmVectorPrep {
    fn decode(&self, payload: &[u8]) -> Result<DecodedVectors, String> {
        metadata::decode_vectors(payload, self.data.flags, self.data.grid, self.data.flags)
            .ok_or_else(|| "invalid vector payload".into())
    }

    fn context<'a>(&self, a: &'a [Vector], b: &'a [Vector]) -> VectorContext<'a> {
        VectorContext {
            block_w: self.data.block.width.max(1),
            block_h: self.data.block.height.max(1),
            scale_shift_base: self.data.marker.max(1),
            frame_w: self.frame_w,
            frame_h: self.frame_h,
            origin_x: self.data.origin().width,
            origin_y: self.data.origin().height,
            grid_w: usize::try_from(self.data.grid.width.max(0)).unwrap_or(0),
            grid_h: usize::try_from(self.data.grid.height.max(0)).unwrap_or(0),
            raw: false,
            a,
            b,
        }
    }
}

fn pair_sets(decoded: &DecodedVectors) -> Result<(Vec<Vector>, Vec<Vector>), String> {
    let previous = decoded
        .previous
        .as_deref()
        .or(decoded.current.as_deref())
        .ok_or("payload carries no vectors")?;
    let current = decoded.current.as_deref().unwrap_or(previous);
    Ok((renderer_vectors(previous), renderer_vectors(current)))
}

fn renderer_vectors(vectors: &[DecodedVector]) -> Vec<Vector> {
    vectors
        .iter()
        .map(|vector| Vector {
            dx: vector.dx,
            dy: vector.dy,
            magnitude: vector.score,
        })
        .collect()
}

#[derive(Clone, Copy)]
struct Layout {
    width: usize,
    chroma_width: usize,
    source_step: usize,
    source_y_len: usize,
    source_chroma_len: usize,
    source_len: usize,
    output_y_len: usize,
    output_chroma_len: usize,
    output_len: usize,
    grid_len: usize,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct WasmRenderer {
    renderer: CpuRenderer,
    layout: Layout,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct WasmRenderBuffers {
    source0: Vec<u8>,
    source1: Vec<u8>,
    motion: Vec<u16>,
    masks: Vec<u8>,
    output: Vec<u8>,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl WasmRenderBuffers {
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(constructor))]
    #[must_use]
    pub fn new(renderer: &WasmRenderer) -> Self {
        Self {
            source0: vec![0; renderer.source_len()],
            source1: vec![0; renderer.source_len()],
            motion: vec![0; renderer.motion_len(23)],
            masks: vec![0; renderer.mask_len(true)],
            output: vec![0; renderer.output_len()],
        }
    }

    #[must_use]
    pub fn source0_ptr(&mut self) -> usize {
        self.source0.as_mut_ptr() as usize
    }

    #[must_use]
    pub fn source1_ptr(&mut self) -> usize {
        self.source1.as_mut_ptr() as usize
    }

    #[must_use]
    pub fn motion_ptr(&mut self) -> usize {
        self.motion.as_mut_ptr() as usize
    }

    #[must_use]
    pub fn masks_ptr(&mut self) -> usize {
        self.masks.as_mut_ptr() as usize
    }

    #[must_use]
    pub fn output_ptr(&self) -> usize {
        self.output.as_ptr() as usize
    }

    pub fn render(
        &mut self,
        renderer: &WasmRenderer,
        mode: u32,
        interpolate: bool,
        mask_planes: u32,
    ) -> Result<(), String> {
        if !matches!(mask_planes, 0 | 2 | 3) {
            return Err("mask plane count must be 0, 2 or 3".into());
        }
        let motion_len = renderer.motion_len(mode);
        let mask_len = renderer
            .layout
            .grid_len
            .checked_mul(mask_planes as usize)
            .ok_or("mask size overflow")?;
        renderer.render_into(
            mode,
            interpolate,
            &self.source0,
            &self.source1,
            &self.motion[..motion_len],
            &self.masks[..mask_len],
            &mut self.output,
        )
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl WasmRenderer {
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(constructor))]
    pub fn new(
        width: i32,
        height: i32,
        block_width: i32,
        block_height: i32,
        origin_x: i32,
        origin_y: i32,
        grid_width: u32,
        grid_height: u32,
        chroma_y_divisor: i32,
        source_step: i32,
        scale: f64,
    ) -> Result<Self, String> {
        if width <= 0
            || height <= 0
            || block_width <= 0
            || block_height <= 0
            || block_width > 32
            || block_height > 32
            || chroma_y_divisor <= 0
            || source_step <= 0
            || grid_width == 0
            || grid_height == 0
            || !scale.is_finite()
        {
            return Err("invalid renderer configuration".into());
        }
        let width_usize = usize::try_from(width).map_err(|_| "invalid width")?;
        let height_usize = usize::try_from(height).map_err(|_| "invalid height")?;
        let chroma_width = width_usize / 2;
        let chroma_height = height_usize
            / usize::try_from(chroma_y_divisor).map_err(|_| "invalid chroma divisor")?;
        let step = usize::try_from(source_step).map_err(|_| "invalid source step")?;
        let step_squared = step.checked_mul(step).ok_or("source size overflow")?;
        let source_y_len = width_usize
            .checked_mul(height_usize)
            .and_then(|len| len.checked_mul(step_squared))
            .ok_or("source size overflow")?;
        let source_chroma_len = chroma_width
            .checked_mul(chroma_height)
            .and_then(|len| len.checked_mul(step_squared))
            .ok_or("source size overflow")?;
        let output_y_len = width_usize
            .checked_mul(height_usize)
            .ok_or("output size overflow")?;
        let output_chroma_len = chroma_width
            .checked_mul(chroma_height)
            .ok_or("output size overflow")?;
        let grid_len = usize::try_from(grid_width)
            .ok()
            .and_then(|width| {
                usize::try_from(grid_height)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .ok_or("grid size overflow")?;
        let source_len = source_chroma_len
            .checked_mul(2)
            .and_then(|chroma| source_y_len.checked_add(chroma))
            .ok_or("source size overflow")?;
        let output_len = output_chroma_len
            .checked_mul(2)
            .and_then(|chroma| output_y_len.checked_add(chroma))
            .ok_or("output size overflow")?;
        grid_len.checked_mul(8).ok_or("motion size overflow")?;
        grid_len.checked_mul(3).ok_or("mask size overflow")?;
        Ok(Self {
            renderer: CpuRenderer::new(CpuConfig {
                width,
                height,
                block_w: block_width,
                block_h: block_height,
                origin_x,
                origin_y,
                grid_w: usize::try_from(grid_width).map_err(|_| "invalid grid width")?,
                grid_h: usize::try_from(grid_height).map_err(|_| "invalid grid height")?,
                chroma_y_div: chroma_y_divisor,
                source_step,
                scale,
            }),
            layout: Layout {
                width: width_usize,
                chroma_width,
                source_step: step,
                source_y_len,
                source_chroma_len,
                source_len,
                output_y_len,
                output_chroma_len,
                output_len,
                grid_len,
            },
        })
    }

    pub fn set_threshold(&mut self, threshold: i32) {
        self.renderer.set_threshold(threshold);
    }

    #[must_use]
    pub fn source_len(&self) -> usize {
        self.layout.source_len
    }

    #[must_use]
    pub fn output_len(&self) -> usize {
        self.layout.output_len
    }

    #[must_use]
    pub fn motion_len(&self, mode: u32) -> usize {
        self.layout.grid_len * if mode == 23 { 8 } else { 4 }
    }

    #[must_use]
    pub fn mask_len(&self, final_mask: bool) -> usize {
        self.layout.grid_len * if final_mask { 3 } else { 2 }
    }

    pub fn render(
        &self,
        mode: u32,
        interpolate: bool,
        source0: &[u8],
        source1: &[u8],
        motion: &[u16],
        masks: &[u8],
    ) -> Result<Vec<u8>, String> {
        self.render_inner(mode, interpolate, source0, source1, motion, masks)
    }
}

impl WasmRenderer {
    fn render_inner(
        &self,
        mode: u32,
        interpolate: bool,
        source0: &[u8],
        source1: &[u8],
        motion: &[u16],
        masks: &[u8],
    ) -> Result<Vec<u8>, String> {
        let mut output = vec![0; self.output_len()];
        self.render_into(
            mode,
            interpolate,
            source0,
            source1,
            motion,
            masks,
            &mut output,
        )?;
        Ok(output)
    }

    fn render_into(
        &self,
        mode: u32,
        interpolate: bool,
        source0: &[u8],
        source1: &[u8],
        motion: &[u16],
        masks: &[u8],
        output: &mut [u8],
    ) -> Result<(), String> {
        if !matches!(mode, 1 | 2 | 11 | 13 | 21 | 22 | 23) {
            return Err("unsupported render mode".into());
        }
        if output.len() != self.output_len() {
            return Err("invalid output buffer length".into());
        }
        let source0 = self.source_planes(source0)?;
        let source1 = self.source_planes(source1)?;
        let required_motion = self.motion_len(mode);
        if motion.len() != required_motion {
            return Err("invalid motion buffer length".into());
        }
        if !masks.is_empty()
            && masks.len() != self.mask_len(false)
            && masks.len() != self.mask_len(true)
        {
            return Err("invalid mask buffer length".into());
        }
        let mut motion_chunks = motion.chunks_exact(self.layout.grid_len);
        let motion0 = MotionPlanes {
            x: motion_chunks.next().ok_or("missing motion plane")?,
            y: motion_chunks.next().ok_or("missing motion plane")?,
        };
        let motion1 = MotionPlanes {
            x: motion_chunks.next().ok_or("missing motion plane")?,
            y: motion_chunks.next().ok_or("missing motion plane")?,
        };
        let motion2 = if mode == 23 {
            Some(MotionPlanes {
                x: motion_chunks.next().ok_or("missing motion plane")?,
                y: motion_chunks.next().ok_or("missing motion plane")?,
            })
        } else {
            None
        };
        let motion3 = if mode == 23 {
            Some(MotionPlanes {
                x: motion_chunks.next().ok_or("missing motion plane")?,
                y: motion_chunks.next().ok_or("missing motion plane")?,
            })
        } else {
            None
        };
        let mut mask_chunks = masks.chunks_exact(self.layout.grid_len);
        let coverage = match (mask_chunks.next(), mask_chunks.next()) {
            (Some(a), Some(b)) => Some(MaskPlanes { a, b }),
            _ => None,
        };
        let final_mask = mask_chunks.next();
        let (output_y, chroma) = output.split_at_mut(self.layout.output_y_len);
        let (output_u, output_v) = chroma.split_at_mut(self.layout.output_chroma_len);
        let input_y = PlaneRenderInput {
            source0: source0.y,
            source1: source1.y,
            motion0,
            motion1,
            motion2,
            motion3,
            masks: coverage,
            final_mask,
        };
        let input_u = PlaneRenderInput {
            source0: source0.u,
            source1: source1.u,
            ..input_y
        };
        let input_v = PlaneRenderInput {
            source0: source0.v,
            source1: source1.v,
            ..input_y
        };
        let results = {
            let (y, (u, v)) = rayon::join(
                || self.render_plane_banded(mode, interpolate, output_y, self.layout.width, input_y, false),
                || {
                    rayon::join(
                        || {
                            self.render_plane_banded(
                                mode,
                                interpolate,
                                output_u,
                                self.layout.chroma_width,
                                input_u,
                                true,
                            )
                        },
                        || {
                            self.render_plane_banded(
                                mode,
                                interpolate,
                                output_v,
                                self.layout.chroma_width,
                                input_v,
                                true,
                            )
                        },
                    )
                },
            );
            [y, u, v]
        };
        for result in results {
            result?;
        }
        Ok(())
    }

    fn render_plane_banded(
        &self,
        mode: u32,
        interpolate: bool,
        output: &mut [u8],
        stride: usize,
        input: PlaneRenderInput<'_>,
        chroma: bool,
    ) -> Result<(), String> {
        const BAND_ROWS: usize = 64;
        let band_bytes = stride * BAND_ROWS;
        if output.len() > band_bytes && rayon::current_num_threads() > 1 {
            output
                .par_chunks_mut(band_bytes)
                .enumerate()
                .try_for_each(|(band, output)| {
                    let start = band * BAND_ROWS;
                    let end = start + output.len() / stride;
                    self.render_plane_rows(mode, interpolate, output, stride, input, chroma, start..end)
                })
        } else {
            self.render_plane(mode, interpolate, output, stride, input, chroma)
        }
    }

    fn render_plane(
        &self,
        mode: u32,
        interpolate: bool,
        output: &mut [u8],
        stride: usize,
        input: PlaneRenderInput<'_>,
        chroma: bool,
    ) -> Result<(), String> {
        self.renderer
            .render_plane(
                mode,
                interpolate,
                PlaneMut {
                    data: output,
                    stride,
                },
                input,
                chroma,
            )
            .map_err(|_| "invalid render inputs".into())
    }

    fn render_plane_rows(
        &self,
        mode: u32,
        interpolate: bool,
        output: &mut [u8],
        stride: usize,
        input: PlaneRenderInput<'_>,
        chroma: bool,
        rows: std::ops::Range<usize>,
    ) -> Result<(), String> {
        let rows = i32::try_from(rows.start).map_err(|_| "invalid row range")?
            ..i32::try_from(rows.end).map_err(|_| "invalid row range")?;
        self.renderer
            .render_plane_rows(
                mode,
                interpolate,
                PlaneMut {
                    data: output,
                    stride,
                },
                input,
                chroma,
                rows,
            )
            .map_err(|_| "invalid render inputs".into())
    }

    fn source_planes<'a>(&self, source: &'a [u8]) -> Result<FramePlanes<'a>, String> {
        if source.len() != self.source_len() {
            return Err("invalid source buffer length".into());
        }
        let (y, chroma) = source.split_at(self.layout.source_y_len);
        let (u, v) = chroma.split_at(self.layout.source_chroma_len);
        Ok(FramePlanes {
            y: Plane::linear(y, self.layout.width * self.layout.source_step),
            u: Plane::linear(u, self.layout.chroma_width * self.layout.source_step),
            v: Plane::linear(v, self.layout.chroma_width * self.layout.source_step),
        })
    }
}
