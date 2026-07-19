use std::{borrow::Cow, slice};

use crate::{frame, hdr, light, metadata, options, renderer, video_format, vs};

#[derive(Clone, Copy)]
pub(crate) enum Mode {
    SmoothFps,
    Nvof,
    Rife,
}

impl Mode {
    pub(crate) const fn from_raw(raw: i32) -> Self {
        match raw {
            1 => Self::Nvof,
            2 => Self::Rife,
            _ => Self::SmoothFps,
        }
    }

    pub(crate) const fn raw(self) -> i32 {
        match self {
            Self::SmoothFps => 0,
            Self::Nvof => 1,
            Self::Rife => 2,
        }
    }
}

pub(crate) struct Clips {
    pub(crate) source: vs::Raw,
    pub(crate) super_clip: vs::Raw,
    pub(crate) vectors: vs::Raw,
    pub(crate) vec_src: vs::Raw,
    pub(crate) rife_out: vs::Raw,
    pub(crate) src: vs::Raw,
}

impl Clips {
    pub(crate) const fn empty() -> Self {
        Self {
            source: std::ptr::null_mut(),
            super_clip: std::ptr::null_mut(),
            vectors: std::ptr::null_mut(),
            vec_src: std::ptr::null_mut(),
            rife_out: std::ptr::null_mut(),
            src: std::ptr::null_mut(),
        }
    }

    pub(crate) const fn nodes(&self) -> [vs::Raw; 6] {
        [
            self.source,
            self.super_clip,
            self.vectors,
            self.vec_src,
            self.rife_out,
            self.src,
        ]
    }
}

pub(crate) struct FilterState {
    pub(crate) mode: Mode,
    pub(crate) clips: Clips,
    pub(crate) sdata: i64,
    pub(crate) vdata: i64,
    pub(crate) generated_vdata: Option<metadata::VectorData>,
    pub(crate) source_8bit_mode: bool,
    pub(crate) render_mode: i32,
    pub(crate) request_super: bool,
    pub(crate) options: options::Options,
    pub(crate) light: light::LightState,
    pub(crate) video_info: vs::VideoInfo,
    pub(crate) super_info: Option<vs::VideoInfo>,

    pub(crate) gpu: Option<crate::gpu::GpuContext>,

    pub(crate) nvof: Option<crate::nvof::NvofContext>,

    pub(crate) prep_cache: PrepCache,
    pub(crate) decode_cache: DecodeCache,
    pub(crate) expand_cache: ExpandCache,
}

pub(crate) struct SuperExpand {
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    y_stride: usize,
    uv_stride: usize,
}

type ExpandCell = std::sync::Arc<std::sync::OnceLock<Option<std::sync::Arc<SuperExpand>>>>;
pub(crate) type ExpandCache = std::sync::Mutex<Vec<(i64, ExpandCell)>>;

const EXPAND_CACHE_CAP: usize = 8;

type PrepCell = std::sync::Arc<std::sync::OnceLock<Option<std::sync::Arc<FramePrep>>>>;
type PrepCache = std::sync::Mutex<Vec<(i64, PrepCell)>>;

pub(crate) struct DecodedEntry {
    decoded: metadata::DecodedVectors,
    scene_class: i32,
    vectors_prev: Vec<renderer::Vector>,
    vectors_cur: Vec<renderer::Vector>,
    mvx0: Vec<u16>,
    mvy0: Vec<u16>,
    mvx1: Vec<u16>,
    mvy1: Vec<u16>,
}

type DecodeCell = std::sync::Arc<std::sync::OnceLock<Option<std::sync::Arc<DecodedEntry>>>>;
pub(crate) type DecodeCache = std::sync::Mutex<Vec<(i64, DecodeCell)>>;

const PREP_CACHE_CAP: usize = 24;
const DECODE_CACHE_CAP: usize = 16;

#[derive(Clone, Copy)]
struct SceneBlend {
    class: i32,
    dir0: bool,
    dir1: bool,
}

#[derive(Clone, Copy)]
enum ConsumedInput {
    None,
    Source,
    Next,
}

struct RenderedFrame {
    frame: vs::ConstRaw,
    consumed: ConsumedInput,
}

#[derive(Clone, Copy)]
enum VectorInput<'a> {
    Frame(vs::ConstRaw),
    Payload(&'a [u8]),
}

impl VectorInput<'_> {
    const fn is_missing(self) -> bool {
        matches!(self, Self::Frame(frame) if frame.is_null())
    }
}

impl FilterState {
    pub(crate) unsafe fn request_frame(&self, frame: i32, frame_ctx: vs::Raw, vsapi: vs::ConstRaw) {
        if let Some(request_frame) =
            unsafe { vs::table_fn::<vs::RequestFrameFilter>(vsapi, vs::REQUEST_FRAME_FILTER) }
        {
            let source_frame = self.source_frame(frame);
            request_node(request_frame, self.clips.src, source_frame, frame_ctx);
            request_node(request_frame, self.clips.source, source_frame, frame_ctx);
            request_node(
                request_frame,
                self.clips.source,
                source_frame.saturating_add(1),
                frame_ctx,
            );
            if self.options.request_source_plus_two(self.mode.raw()) {
                request_node(
                    request_frame,
                    self.clips.source,
                    source_frame.saturating_add(2),
                    frame_ctx,
                );
            }
            match self.mode {
                Mode::SmoothFps => {
                    self.request_vectors(request_frame, frame, source_frame, frame_ctx);
                    if self.request_super {
                        request_node(
                            request_frame,
                            self.clips.super_clip,
                            source_frame,
                            frame_ctx,
                        );
                        request_node(
                            request_frame,
                            self.clips.super_clip,
                            source_frame.saturating_add(1),
                            frame_ctx,
                        );
                    }
                }
                Mode::Nvof => {
                    self.request_source_8bit(request_frame, source_frame, frame_ctx);
                }
                Mode::Rife => {
                    request_node(request_frame, self.clips.rife_out, frame, frame_ctx);
                    if self.source_8bit_mode {
                        self.request_source_8bit(request_frame, source_frame, frame_ctx);
                    } else {
                        self.request_vectors(request_frame, frame, source_frame, frame_ctx);
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    pub(crate) unsafe fn get_frame(
        &self,
        frame: i32,
        frame_ctx: vs::Raw,
        core: vs::Raw,
        vsapi: vs::ConstRaw,
    ) -> vs::ConstRaw {
        let Some(get_frame) =
            (unsafe { vs::table_fn::<vs::GetFrameFilter>(vsapi, vs::GET_FRAME_FILTER) })
        else {
            return std::ptr::null();
        };
        let free_frame = unsafe { vs::table_fn::<vs::FreeFrame>(vsapi, vs::FREE_FRAME) };
        let source_frame = self.source_frame(frame);

        drop_frame(
            get_node(get_frame, self.clips.src, source_frame, frame_ctx),
            free_frame,
        );
        let source = get_node(get_frame, self.clips.source, source_frame, frame_ctx);
        let next_source = get_node(
            get_frame,
            self.clips.source,
            source_frame.saturating_add(1),
            frame_ctx,
        );
        if self.options.request_source_plus_two(self.mode.raw()) {
            fetch_node(
                get_frame,
                free_frame,
                self.clips.source,
                source_frame.saturating_add(2),
                frame_ctx,
            );
        }
        let rife = if matches!(self.mode, Mode::Rife)
            && (self.options.fast_output_enabled()
                || self.options.cpu_render()
                || self.options.hdr_enabled())
            && !self.clips.rife_out.is_null()
        {
            get_node(get_frame, self.clips.rife_out, frame, frame_ctx)
        } else {
            std::ptr::null()
        };
        match self.mode {
            Mode::SmoothFps => {
                let vectors = get_node(get_frame, self.clips.vectors, source_frame, frame_ctx);
                if self.options.request_hdr_vectors() {
                    self.fetch_hdr_vectors(get_frame, free_frame, source_frame, frame_ctx);
                }
                let algo = self.options.algorithm(self.mode.raw());
                let phase = self.phase_256(frame, source_frame);
                if algo != 23 && (1..=255).contains(&phase) && !self.options.scene_blend() {
                    self.fetch_vector_neighbors(get_frame, free_frame, source_frame, frame_ctx);
                }
                let use_merged_source = self.request_super;
                let merged_source = if use_merged_source {
                    get_node(get_frame, self.clips.super_clip, source_frame, frame_ctx)
                } else {
                    std::ptr::null()
                };
                let next_merged_source = if use_merged_source {
                    get_node(
                        get_frame,
                        self.clips.super_clip,
                        source_frame.saturating_add(1),
                        frame_ctx,
                    )
                } else {
                    std::ptr::null()
                };
                let neighbors =
                    self.vector_neighbors(get_frame, source_frame, phase, algo, frame_ctx);
                if let Some(rendered) = unsafe {
                    self.try_cpu_render(
                        source,
                        next_source,
                        vectors,
                        neighbors,
                        merged_source,
                        next_merged_source,
                        frame,
                        source_frame,
                        core,
                        vsapi,
                    )
                } {
                    if !matches!(rendered.consumed, ConsumedInput::Source) {
                        drop_frame(source, free_frame);
                    }
                    if !matches!(rendered.consumed, ConsumedInput::Next) {
                        drop_frame(next_source, free_frame);
                    }
                    drop_frame(vectors, free_frame);
                    drop_frame(merged_source, free_frame);
                    drop_frame(next_merged_source, free_frame);
                    drop_frames(neighbors, free_frame);
                    return rendered.frame;
                }
                drop_frames(neighbors, free_frame);
                drop_frame(merged_source, free_frame);
                drop_frame(next_merged_source, free_frame);
                let output = if self.options.fast_output_enabled() && phase == 0x100 {
                    let output = unsafe {
                        self.padded_output(
                            next_source,
                            source,
                            next_source,
                            frame,
                            source_frame,
                            vectors,
                            core,
                            vsapi,
                        )
                    };
                    drop_frame(source, free_frame);
                    output
                } else {
                    let output = unsafe {
                        self.padded_output(
                            source,
                            source,
                            next_source,
                            frame,
                            source_frame,
                            vectors,
                            core,
                            vsapi,
                        )
                    };
                    drop_frame(next_source, free_frame);
                    output
                };
                drop_frame(vectors, free_frame);
                return output;
            }
            Mode::Nvof => {
                let source_8bit = get_node(get_frame, self.clips.vec_src, source_frame, frame_ctx);
                let next_source_8bit = get_node(
                    get_frame,
                    self.clips.vec_src,
                    source_frame.saturating_add(1),
                    frame_ctx,
                );
                if let (Some(nvof), metadata::VectorRecord::Ready(vector_data)) =
                    (self.nvof.as_ref(), self.vector_data())
                {
                    let payload = if let Some(payload) = nvof.cached(source_frame) {
                        Some(payload)
                    } else {
                        let api = unsafe { frame::PlaneApi::load(vsapi) };
                        api.and_then(|api| {
                            let (width, height) = nvof.dimensions();
                            let current =
                                unsafe { pack_nv12_frame(&api, source_8bit, width, height) }?;
                            let next =
                                unsafe { pack_nv12_frame(&api, next_source_8bit, width, height) }?;
                            nvof.generate(source_frame, &current, &next, vector_data)
                                .ok()
                        })
                    };
                    if let Some(payload) = payload
                        && let Some(rendered) = unsafe {
                            self.try_nvof_render(
                                source,
                                next_source,
                                &payload,
                                frame,
                                source_frame,
                                core,
                                vsapi,
                            )
                        }
                    {
                        if !matches!(rendered.consumed, ConsumedInput::Source) {
                            drop_frame(source, free_frame);
                        }
                        if !matches!(rendered.consumed, ConsumedInput::Next) {
                            drop_frame(next_source, free_frame);
                        }
                        drop_frame(source_8bit, free_frame);
                        drop_frame(next_source_8bit, free_frame);
                        return rendered.frame;
                    }
                }
                drop_frame(source_8bit, free_frame);
                drop_frame(next_source_8bit, free_frame);
            }
            Mode::Rife => {
                if self.options.hdr_enabled()
                    && !self.options.cpu_render()
                    && !rife.is_null()
                    && let Some(output) = unsafe {
                        self.try_rife_hdr_render(
                            rife,
                            source,
                            next_source,
                            frame,
                            source_frame,
                            core,
                            vsapi,
                        )
                    }
                {
                    drop_frame(source, free_frame);
                    drop_frame(next_source, free_frame);
                    drop_frame(rife, free_frame);
                    return output;
                }
                if !rife.is_null() {
                    let output = unsafe {
                        self.padded_output(
                            rife,
                            source,
                            next_source,
                            frame,
                            source_frame,
                            std::ptr::null(),
                            core,
                            vsapi,
                        )
                    };
                    drop_frame(source, free_frame);
                    drop_frame(next_source, free_frame);
                    return output;
                }
                if self.source_8bit_mode {
                    self.fetch_source_8bit(get_frame, free_frame, source_frame, frame_ctx);
                } else {
                    let vectors = get_node(get_frame, self.clips.vectors, source_frame, frame_ctx);
                    if self.options.request_hdr_vectors() {
                        self.fetch_hdr_vectors(get_frame, free_frame, source_frame, frame_ctx);
                    }
                    if (1..=255).contains(&self.phase_256(frame, source_frame)) {
                        self.fetch_vector_neighbors(get_frame, free_frame, source_frame, frame_ctx);
                    }
                    if let Some(rendered) = unsafe {
                        self.try_cpu_render(
                            source,
                            next_source,
                            vectors,
                            [std::ptr::null(); 4],
                            std::ptr::null(),
                            std::ptr::null(),
                            frame,
                            source_frame,
                            core,
                            vsapi,
                        )
                    } {
                        if !matches!(rendered.consumed, ConsumedInput::Source) {
                            drop_frame(source, free_frame);
                        }
                        if !matches!(rendered.consumed, ConsumedInput::Next) {
                            drop_frame(next_source, free_frame);
                        }
                        drop_frame(vectors, free_frame);
                        return rendered.frame;
                    }
                    drop_frame(vectors, free_frame);
                }
            }
        }
        match self.mode {
            _ if self.options.fast_output_enabled()
                && self.phase_256(frame, source_frame) == 0x100 =>
            {
                let output = unsafe {
                    self.padded_output(
                        next_source,
                        source,
                        next_source,
                        frame,
                        source_frame,
                        std::ptr::null(),
                        core,
                        vsapi,
                    )
                };
                drop_frame(source, free_frame);
                output
            }
            _ => {
                let output = unsafe {
                    self.padded_output(
                        source,
                        source,
                        next_source,
                        frame,
                        source_frame,
                        std::ptr::null(),
                        core,
                        vsapi,
                    )
                };
                drop_frame(next_source, free_frame);
                output
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn try_rife_hdr_render(
        &self,
        rife: vs::ConstRaw,
        source: vs::ConstRaw,
        next_source: vs::ConstRaw,
        frame: i32,
        source_frame: i32,
        core: vs::Raw,
        vsapi: vs::ConstRaw,
    ) -> Option<vs::ConstRaw> {
        let api = unsafe { frame::PlaneApi::load(vsapi) }?;
        let selected = if frame != 0 && self.phase_256(frame, source_frame) == 0 {
            source
        } else {
            rife
        };
        let output_info = self.output_info();
        let output = unsafe {
            api.new_frame(
                self.video_info.format,
                output_info.width,
                output_info.height,
                selected,
                core,
            )
        }?;
        if unsafe { self.render_hdr_frame(&api, output, selected) }.is_none() {
            unsafe { api.free(output.cast_const()) };
            return None;
        }
        let timing = self.options.timing(&self.video_info);
        let ratio = f64_i64(timing.frame_num) / f64_i64(timing.frame_den);
        let raw_phase = timing.raw_phase_256(frame, source_frame);
        unsafe { api.copy_interpolated_timing(source, next_source, output, raw_phase, ratio) };
        Some(output.cast_const())
    }

    #[allow(clippy::similar_names)]
    unsafe fn render_hdr_frame(
        &self,
        api: &frame::PlaneApi,
        output: vs::Raw,
        selected: vs::ConstRaw,
    ) -> Option<()> {
        let width = self.video_info.width;
        let height = self.video_info.height;
        let chroma_h = usize_height(height / 2);
        let (src_y, src_y_stride, src_y_len) =
            unsafe { api.read_plane(selected, 0, usize_height(height)) }?;
        let (src_u, src_u_stride, src_u_len) = unsafe { api.read_plane(selected, 1, chroma_h) }?;
        let (src_v, src_v_stride, src_v_len) = unsafe { api.read_plane(selected, 2, chroma_h) }?;
        let (dst_y, dst_y_stride, dst_y_len) =
            unsafe { api.write_plane(output, 0, usize_height(self.output_info().height)) }?;
        let (dst_u, dst_u_stride, dst_u_len) =
            unsafe { api.write_plane(output, 1, usize_height(self.output_info().height / 2)) }?;
        let (dst_v, dst_v_stride, dst_v_len) =
            unsafe { api.write_plane(output, 2, usize_height(self.output_info().height / 2)) }?;
        let src = unsafe {
            renderer::FramePlanes {
                y: plane(src_y, src_y_stride, src_y_len),
                u: plane(src_u, src_u_stride, src_u_len),
                v: plane(src_v, src_v_stride, src_v_len),
            }
        };
        let padding = self.options.padding(&self.video_info);
        let dst = unsafe {
            renderer::FramePlanesMut {
                y: offset_plane_mut(dst_y, dst_y_stride, dst_y_len, padding.0, padding.1, 1),
                u: offset_plane_mut(
                    dst_u,
                    dst_u_stride,
                    dst_u_len,
                    padding.0 / 2,
                    padding.1 / 2,
                    1,
                ),
                v: offset_plane_mut(
                    dst_v,
                    dst_v_stride,
                    dst_v_len,
                    padding.0 / 2,
                    padding.1 / 2,
                    1,
                ),
            }
        };
        hdr::transform_420(width, height, src, dst);
        Some(())
    }

    pub(crate) unsafe fn free(self, vsapi: vs::ConstRaw) {
        unsafe { vs::free_nodes(self.clips.nodes(), vsapi) };
    }

    pub(crate) fn output_info(&self) -> vs::VideoInfo {
        let timing = self.options.timing(&self.video_info);
        let mut info = self.video_info;
        info.fps_num = timing.fps_num;
        info.fps_den = timing.fps_den;
        info.num_frames = timing.scale_frame_count(info.num_frames);
        let (pad_x, pad_y) = self.options.padding(&self.video_info);
        info.width = info.width.saturating_add(pad_x.saturating_mul(2));
        info.height = info.height.saturating_add(pad_y.saturating_mul(2));
        info
    }

    pub(crate) fn mask_area_uses_source_8bit_scale(&self) -> bool {
        self.source_8bit_mode
    }

    pub(crate) fn vector_data(&self) -> metadata::VectorRecord {
        if let Some(vectors) = self.generated_vdata {
            metadata::VectorRecord::Ready(vectors)
        } else if self.vdata != 0 {
            unsafe { metadata::vector_data(self.vdata) }
        } else {
            metadata::VectorRecord::Missing
        }
    }

    pub(crate) const fn requires_cpu_source(&self) -> bool {
        self.render_mode <= 1
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn padded_output(
        &self,
        source: vs::ConstRaw,
        timing_source: vs::ConstRaw,
        next_source: vs::ConstRaw,
        frame: i32,
        source_frame: i32,
        vectors: vs::ConstRaw,
        core: vs::Raw,
        vsapi: vs::ConstRaw,
    ) -> vs::ConstRaw {
        let padding = self.options.padding(&self.video_info);
        let Some(api) = (unsafe { frame::PlaneApi::load(vsapi) }) else {
            return source;
        };
        let output_info = self.output_info();
        let timing = self.options.timing(&self.video_info);
        let ratio = f64_i64(timing.frame_num) / f64_i64(timing.frame_den);
        let raw_phase = timing.raw_phase_256(frame, source_frame);
        let output_phase = timing.output_phase_256(frame);
        let output = unsafe {
            api.selected_copy(
                source,
                timing_source,
                next_source,
                core,
                &self.video_info,
                &output_info,
                padding,
                bytes_per_sample(&self.video_info),
                raw_phase,
                ratio,
            )
        };
        if !output.is_null() && output != source {
            let _ = unsafe { self.apply_light_border(&api, output.cast_mut(), frame) };
        }
        if self.options.debug_qmap()
            && (1..=255).contains(&output_phase)
            && !output.is_null()
            && output != source
        {
            let _ =
                unsafe { self.apply_qmap_overlay(&api, output.cast_mut(), vectors, output_phase) };
        }
        if self.options.debug_vectors()
            && (1..=255).contains(&output_phase)
            && !output.is_null()
            && output != source
        {
            let _ = unsafe {
                self.apply_vector_overlay(&api, output.cast_mut(), vectors, output_phase)
            };
        }
        if self.options.debug_tt() && !output.is_null() && output != source {
            let _ = unsafe { self.apply_timing_bar(&api, output.cast_mut(), frame) };
        }
        output
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn try_cpu_render(
        &self,
        source: vs::ConstRaw,
        next_source: vs::ConstRaw,
        vectors: vs::ConstRaw,
        neighbors: [vs::ConstRaw; 4],
        merged_source: vs::ConstRaw,
        next_merged_source: vs::ConstRaw,
        frame: i32,
        source_frame: i32,
        core: vs::Raw,
        vsapi: vs::ConstRaw,
    ) -> Option<RenderedFrame> {
        unsafe {
            self.try_render(
                source,
                next_source,
                VectorInput::Frame(vectors),
                neighbors,
                merged_source,
                next_merged_source,
                frame,
                source_frame,
                core,
                vsapi,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn try_nvof_render(
        &self,
        source: vs::ConstRaw,
        next_source: vs::ConstRaw,
        vectors: &[u8],
        frame: i32,
        source_frame: i32,
        core: vs::Raw,
        vsapi: vs::ConstRaw,
    ) -> Option<RenderedFrame> {
        unsafe {
            self.try_render(
                source,
                next_source,
                VectorInput::Payload(vectors),
                [std::ptr::null(); 4],
                std::ptr::null(),
                std::ptr::null(),
                frame,
                source_frame,
                core,
                vsapi,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn try_render(
        &self,
        source: vs::ConstRaw,
        next_source: vs::ConstRaw,
        vectors: VectorInput<'_>,
        neighbors: [vs::ConstRaw; 4],
        merged_source: vs::ConstRaw,
        next_merged_source: vs::ConstRaw,
        frame: i32,
        source_frame: i32,
        core: vs::Raw,
        vsapi: vs::ConstRaw,
    ) -> Option<RenderedFrame> {
        if !matches!(self.mode, Mode::SmoothFps | Mode::Nvof | Mode::Rife)
            || self.render_mode > 2
            || source.is_null()
            || next_source.is_null()
            || vectors.is_missing()
        {
            return None;
        }
        if matches!(self.mode, Mode::SmoothFps | Mode::Nvof)
            && (frame == 0 || matches!(self.phase_256(frame, source_frame), 0 | 256))
        {
            return None;
        }
        if matches!(self.mode, Mode::SmoothFps | Mode::Nvof)
            && !self.options.scene_blend()
            && self.options.request_scene_mode(self.mode.raw()) == 3
        {
            let timing = self.options.timing(&self.video_info);
            if timing.frame_den != 0 && f64_i64(timing.frame_num) / f64_i64(timing.frame_den) < 2.0
            {
                return None;
            }
        }
        let requested_algo = i32::try_from(self.options.algorithm(self.mode.raw())).ok()?;
        if !matches!(requested_algo, 1 | 2 | 11 | 13 | 21 | 22 | 23) {
            return None;
        }
        let metadata::VectorRecord::Ready(vector_data) = self.vector_data() else {
            return None;
        };
        let api = unsafe { frame::PlaneApi::load(vsapi) }?;
        let phase = self.phase_256(frame, source_frame).clamp(0, 256);
        if matches!(self.mode, Mode::SmoothFps | Mode::Nvof) && !self.options.scene_blend() {
            let count = usize::try_from(
                vector_data
                    .grid
                    .width
                    .max(0)
                    .saturating_mul(vector_data.grid.height.max(0)),
            )
            .ok()?;
            let vector_len = 0x48usize.saturating_add(count.saturating_mul(16));
            let prep = unsafe {
                self.cached_prep(
                    &api,
                    source,
                    vectors,
                    neighbors,
                    vector_data,
                    vector_len,
                    requested_algo,
                    source_frame,
                )
            }?;
            if prep.raw_scene_class >= 3 {
                let (selected, consumed) = if phase < 128 {
                    (source, ConsumedInput::Source)
                } else {
                    (next_source, ConsumedInput::Next)
                };
                let vector_frame = match vectors {
                    VectorInput::Frame(frame) => frame,
                    VectorInput::Payload(_) => std::ptr::null(),
                };
                let frame = unsafe {
                    self.padded_output(
                        selected,
                        source,
                        next_source,
                        frame,
                        source_frame,
                        vector_frame,
                        core,
                        vsapi,
                    )
                };
                return (!frame.is_null()).then_some(RenderedFrame { frame, consumed });
            }
        }
        let output_info = self.output_info();
        let output = unsafe {
            api.new_frame(
                self.video_info.format,
                output_info.width,
                output_info.height,
                source,
                core,
            )
        }?;
        if unsafe {
            self.render_into_output(
                &api,
                output,
                source,
                next_source,
                vectors,
                neighbors,
                frame,
                source_frame,
                vector_data,
                requested_algo,
                merged_source,
                next_merged_source,
            )
        }
        .is_none()
        {
            unsafe { api.free(output.cast_const()) };
            return None;
        }
        let timing = self.options.timing(&self.video_info);
        let ratio = f64_i64(timing.frame_num) / f64_i64(timing.frame_den);
        let raw_phase = timing.raw_phase_256(frame, source_frame);
        unsafe { api.copy_interpolated_timing(source, next_source, output, raw_phase, ratio) };
        Some(RenderedFrame {
            frame: output.cast_const(),
            consumed: ConsumedInput::None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn cached_prep(
        &self,
        api: &frame::PlaneApi,
        source: vs::ConstRaw,
        vectors: VectorInput<'_>,
        neighbors: [vs::ConstRaw; 4],
        vector_data: metadata::VectorData,
        vector_len: usize,
        requested_algo: i32,
        source_frame: i32,
    ) -> Option<std::sync::Arc<FramePrep>> {
        let key = i64::from(source_frame);
        if std::env::var_os("SVP_NO_PREP").is_some() {
            return unsafe {
                self.build_frame_prep(
                    api,
                    source,
                    vectors,
                    neighbors,
                    vector_data,
                    vector_len,
                    requested_algo,
                    source_frame,
                )
            }
            .map(std::sync::Arc::new);
        }

        if self.options.scene_blend() || requested_algo == 23 {
            let keys = neighbor_keys(source_frame);
            if let VectorInput::Frame(frame) = vectors {
                let _ = unsafe {
                    self.cached_decode(api, i64::from(source_frame), frame, vector_data, vector_len)
                };
            }
            for (key, frame) in keys.into_iter().zip(neighbors) {
                if !frame.is_null() {
                    let _ =
                        unsafe { self.cached_decode(api, key, frame, vector_data, vector_len) };
                }
            }
        }
        let cell: PrepCell = {
            let mut cache = self.prep_cache.lock().ok()?;
            if let Some(pos) = cache.iter().position(|(k, _)| *k == key) {
                let entry = cache.remove(pos);
                let cell = std::sync::Arc::clone(&entry.1);
                cache.push(entry);
                cell
            } else {
                let cell: PrepCell = std::sync::Arc::new(std::sync::OnceLock::new());
                if cache.len() >= PREP_CACHE_CAP {
                    cache.remove(0);
                }
                cache.push((key, std::sync::Arc::clone(&cell)));
                cell
            }
        };

        cell.get_or_init(|| {
            unsafe {
                self.build_frame_prep(
                    api,
                    source,
                    vectors,
                    neighbors,
                    vector_data,
                    vector_len,
                    requested_algo,
                    source_frame,
                )
            }
            .map(std::sync::Arc::new)
        })
        .clone()
    }

    fn decoded_entry(
        &self,
        decoded: metadata::DecodedVectors,
        vector_data: metadata::VectorData,
    ) -> DecodedEntry {
        let scene_class = self.scene_class(&decoded, vector_data);
        let previous = decoded.previous.as_deref().or(decoded.current.as_deref());
        let current = decoded.current.as_deref().or(previous);
        let vectors_prev = previous.map(renderer_vectors).unwrap_or_default();
        let vectors_cur = current.map(renderer_vectors).unwrap_or_default();
        let origin = vector_data.origin();
        let motion_grid = vector_data.motion_grid();
        let motion_w = usize::try_from(motion_grid.width.max(0)).unwrap_or(0);
        let motion_h = usize::try_from(motion_grid.height.max(0)).unwrap_or(0);
        let motion_count = motion_w.saturating_mul(motion_h);
        let mut mvx0 = vec![0u16; motion_count];
        let mut mvy0 = vec![0u16; motion_count];
        let mut mvx1 = vec![0u16; motion_count];
        let mut mvy1 = vec![0u16; motion_count];
        if let (Ok(grid_w), Ok(grid_h)) = (
            usize::try_from(vector_data.grid.width.max(0)),
            usize::try_from(vector_data.grid.height.max(0)),
        ) {
            let ctx = renderer::VectorContext {
                block_w: vector_data.block.width.max(1),
                block_h: vector_data.block.height.max(1),
                scale_shift_base: vector_data.marker.max(1),
                frame_w: self.video_info.width,
                frame_h: self.video_info.height,
                origin_x: origin.width,
                origin_y: origin.height,
                grid_w,
                grid_h,
                raw: false,
                a: &vectors_prev,
                b: &vectors_cur,
            };
            renderer::vector_planes(&ctx, 0, &mut mvx0, &mut mvy0, motion_w, motion_h);
            renderer::vector_planes(&ctx, 1, &mut mvx1, &mut mvy1, motion_w, motion_h);
        }
        DecodedEntry {
            decoded,
            scene_class,
            vectors_prev,
            vectors_cur,
            mvx0,
            mvy0,
            mvx1,
            mvy1,
        }
    }

    unsafe fn cached_decode(
        &self,
        api: &frame::PlaneApi,
        key: i64,
        frame: vs::ConstRaw,
        vector_data: metadata::VectorData,
        vector_len: usize,
    ) -> Option<std::sync::Arc<DecodedEntry>> {
        if frame.is_null() {
            return None;
        }
        let decode = || {
            let decoded = unsafe { decode_frame_vectors(api, frame, vector_data, vector_len) }?;
            Some(std::sync::Arc::new(self.decoded_entry(decoded, vector_data)))
        };
        if std::env::var_os("SVP_NO_PREP").is_some() {
            return decode();
        }
        let cell: DecodeCell = {
            let mut cache = self.decode_cache.lock().ok()?;
            if let Some(pos) = cache.iter().position(|(k, _)| *k == key) {
                let entry = cache.remove(pos);
                let cell = std::sync::Arc::clone(&entry.1);
                cache.push(entry);
                cell
            } else {
                let cell: DecodeCell = std::sync::Arc::new(std::sync::OnceLock::new());
                if cache.len() >= DECODE_CACHE_CAP {
                    cache.remove(0);
                }
                cache.push((key, std::sync::Arc::clone(&cell)));
                cell
            }
        };
        cell.get_or_init(decode).clone()
    }

    fn cached_expand(
        &self,
        key: i64,
        planes: &renderer::FramePlanes<'_>,
    ) -> Option<std::sync::Arc<SuperExpand>> {
        if std::env::var_os("SVP_NO_EXPAND").is_some() {
            return None;
        }
        let width = usize::try_from(self.video_info.width.max(0)).ok()?;
        let height = usize::try_from(self.video_info.height.max(0)).ok()?;
        let cell: ExpandCell = {
            let mut cache = self.expand_cache.lock().ok()?;
            if let Some(pos) = cache.iter().position(|(k, _)| *k == key) {
                let entry = cache.remove(pos);
                let cell = std::sync::Arc::clone(&entry.1);
                cache.push(entry);
                cell
            } else {
                let cell: ExpandCell = std::sync::Arc::new(std::sync::OnceLock::new());
                if cache.len() >= EXPAND_CACHE_CAP {
                    cache.remove(0);
                }
                cache.push((key, std::sync::Arc::clone(&cell)));
                cell
            }
        };
        cell.get_or_init(|| {
            let (y, y_stride) = expand_plane(planes.y, width, height)?;
            let (u, uv_stride) = expand_plane(planes.u, width / 2, height / 2)?;
            let (v, _) = expand_plane(planes.v, width / 2, height / 2)?;
            Some(std::sync::Arc::new(SuperExpand {
                y,
                u,
                v,
                y_stride,
                uv_stride,
            }))
        })
        .clone()
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    unsafe fn build_frame_prep(
        &self,
        api: &frame::PlaneApi,
        source: vs::ConstRaw,
        vectors: VectorInput<'_>,
        neighbors: [vs::ConstRaw; 4],
        vector_data: metadata::VectorData,
        vector_len: usize,
        requested_algo: i32,
        source_frame: i32,
    ) -> Option<FramePrep> {
        let self_entry = match vectors {
            VectorInput::Frame(frame) => unsafe {
                self.cached_decode(api, i64::from(source_frame), frame, vector_data, vector_len)
            }?,
            VectorInput::Payload(_) => {
                let decoded =
                    unsafe { decode_vectors_input(api, vectors, vector_data, vector_len) }?;
                std::sync::Arc::new(self.decoded_entry(decoded, vector_data))
            }
        };
        let decoded = &self_entry.decoded;
        let previous = decoded.previous.as_deref().or(decoded.current.as_deref())?;
        let current = decoded.current.as_deref().or(Some(previous))?;
        let prop_scene_change = unsafe { api.scene_change_next(source) };
        let raw_scene_class = if prop_scene_change {
            3
        } else {
            self_entry.scene_class
        };
        let neighbor_classes = unsafe {
            self.neighbor_scene_classes(api, neighbors, vector_data, vector_len, source_frame)
        };
        let _ = (previous, current);
        let origin = vector_data.origin();
        let motion_grid = vector_data.motion_grid();
        let motion_w = usize::try_from(motion_grid.width.max(0)).ok()?;
        let motion_h = usize::try_from(motion_grid.height.max(0)).ok()?;
        let motion_count = motion_w.checked_mul(motion_h)?;
        let mut mvx2 = if requested_algo == 23 {
            vec![0u16; motion_count]
        } else {
            Vec::new()
        };
        let mut mvy2 = if requested_algo == 23 {
            vec![0u16; motion_count]
        } else {
            Vec::new()
        };
        let mut mvx3 = if requested_algo == 23 {
            vec![0u16; motion_count]
        } else {
            Vec::new()
        };
        let mut mvy3 = if requested_algo == 23 {
            vec![0u16; motion_count]
        } else {
            Vec::new()
        };
        let mut magnitude0 = if self.options.mask_area_enabled() {
            vec![0u8; motion_count]
        } else {
            Vec::new()
        };
        let mut magnitude1 = if self.options.mask_area_enabled() {
            vec![0u8; motion_count]
        } else {
            Vec::new()
        };
        let side_ready = {
            let ctx = renderer::VectorContext {
                block_w: vector_data.block.width.max(1),
                block_h: vector_data.block.height.max(1),
                scale_shift_base: vector_data.marker.max(1),
                frame_w: self.video_info.width,
                frame_h: self.video_info.height,
                origin_x: origin.width,
                origin_y: origin.height,
                grid_w: usize::try_from(vector_data.grid.width.max(0)).ok()?,
                grid_h: usize::try_from(vector_data.grid.height.max(0)).ok()?,
                raw: false,
                a: &self_entry.vectors_prev,
                b: &self_entry.vectors_cur,
            };
            if self.options.mask_area_enabled() {
                renderer::magnitude_mask(
                    &ctx,
                    0,
                    &mut magnitude0,
                    motion_w,
                    motion_h,
                    self.options.mask_area_scale(),
                    self.options.mask_area_sharp(),
                );
                renderer::magnitude_mask(
                    &ctx,
                    1,
                    &mut magnitude1,
                    motion_w,
                    motion_h,
                    self.options.mask_area_scale(),
                    self.options.mask_area_sharp(),
                );
            }

            if requested_algo == 23 && !neighbors[1].is_null() && !neighbors[2].is_null() {
                let keys = neighbor_keys(source_frame);
                let prev_decoded = unsafe {
                    self.cached_decode(api, keys[1], neighbors[1], vector_data, vector_len)
                };
                let next_decoded = unsafe {
                    self.cached_decode(api, keys[2], neighbors[2], vector_data, vector_len)
                };
                if let (Some(prev_decoded), Some(next_decoded)) = (prev_decoded, next_decoded) {
                    let next_scene = next_decoded.scene_class;
                    let prev_side = prev_decoded
                        .decoded
                        .current
                        .as_deref()
                        .or(prev_decoded.decoded.previous.as_deref());
                    let next_side = next_decoded
                        .decoded
                        .previous
                        .as_deref()
                        .or(next_decoded.decoded.current.as_deref());
                    if next_scene < 3
                        && prev_side.is_some()
                        && next_side.is_some()
                    {
                        let prev_ctx = renderer::VectorContext {
                            a: &self_entry.vectors_prev,
                            b: &prev_decoded.vectors_cur,
                            ..ctx
                        };
                        let next_ctx = renderer::VectorContext {
                            a: &next_decoded.vectors_prev,
                            b: &self_entry.vectors_cur,
                            ..ctx
                        };
                        renderer::vector_planes(
                            &next_ctx, 0, &mut mvx2, &mut mvy2, motion_w, motion_h,
                        );
                        renderer::vector_planes(
                            &prev_ctx, 1, &mut mvx3, &mut mvy3, motion_w, motion_h,
                        );
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                false
            }
        };
        Some(FramePrep {
            decoded: self_entry,
            mvx2,
            mvy2,
            mvx3,
            mvy3,
            magnitude0,
            magnitude1,
            side_ready,
            raw_scene_class,
            neighbor_classes,
        })
    }

    unsafe fn render_into_output(
        &self,
        api: &frame::PlaneApi,
        output: vs::Raw,
        source: vs::ConstRaw,
        next_source: vs::ConstRaw,
        vectors: VectorInput<'_>,
        neighbors: [vs::ConstRaw; 4],
        frame: i32,
        source_frame: i32,
        vector_data: metadata::VectorData,
        requested_algo: i32,
        merged_source: vs::ConstRaw,
        next_merged_source: vs::ConstRaw,
    ) -> Option<()> {
        let Some((src0_y, src0_y_stride, src0_y_len)) =
            (unsafe { api.read_plane(source, 0, usize_height(self.video_info.height)) })
        else {
            return None;
        };
        let Some((src1_y, src1_y_stride, src1_y_len)) =
            (unsafe { api.read_plane(next_source, 0, usize_height(self.video_info.height)) })
        else {
            return None;
        };
        let chroma_h = usize_height(self.video_info.height / 2);
        let Some((src0_u, src0_u_stride, src0_u_len)) =
            (unsafe { api.read_plane(source, 1, chroma_h) })
        else {
            return None;
        };
        let Some((src0_v, src0_v_stride, src0_v_len)) =
            (unsafe { api.read_plane(source, 2, chroma_h) })
        else {
            return None;
        };
        let Some((src1_u, src1_u_stride, src1_u_len)) =
            (unsafe { api.read_plane(next_source, 1, chroma_h) })
        else {
            return None;
        };
        let Some((src1_v, src1_v_stride, src1_v_len)) =
            (unsafe { api.read_plane(next_source, 2, chroma_h) })
        else {
            return None;
        };
        let output_chroma_h = usize_height(self.output_info().height / 2);
        let Some((dst_y, dst_y_stride, dst_y_len)) =
            (unsafe { api.write_plane(output, 0, usize_height(self.output_info().height)) })
        else {
            return None;
        };
        let Some((dst_u, dst_u_stride, dst_u_len)) =
            (unsafe { api.write_plane(output, 1, output_chroma_h) })
        else {
            return None;
        };
        let Some((dst_v, dst_v_stride, dst_v_len)) =
            (unsafe { api.write_plane(output, 2, output_chroma_h) })
        else {
            return None;
        };
        let count = usize::try_from(
            vector_data
                .grid
                .width
                .max(0)
                .saturating_mul(vector_data.grid.height.max(0)),
        )
        .ok()?;
        let vector_len = 0x48usize.saturating_add(count.saturating_mul(16));
        let prep = unsafe {
            self.cached_prep(
                api,
                source,
                vectors,
                neighbors,
                vector_data,
                vector_len,
                requested_algo,
                source_frame,
            )
        }?;
        let previous = prep
            .decoded
            .decoded
            .previous
            .as_deref()
            .or(prep.decoded.decoded.current.as_deref())?;
        let current = prep.decoded.decoded.current.as_deref().or(Some(previous))?;
        let phase = self.phase_256(frame, source_frame).clamp(0, 256);
        let raw_scene_class = prep.raw_scene_class;
        let neighbor_classes = prep.neighbor_classes;
        let raw_scene_blend_class3 = matches!(self.mode, Mode::SmoothFps | Mode::Nvof)
            && self.options.scene_blend()
            && self.options.request_scene_mode(self.mode.raw()) == 3
            && raw_scene_class >= 3;
        let scene = if raw_scene_blend_class3 {
            SceneBlend {
                class: raw_scene_class,
                dir0: false,
                dir1: false,
            }
        } else {
            self.effective_scene(raw_scene_class, phase, neighbor_classes)
        };
        let scene_class = scene.class;
        let phase = if raw_scene_blend_class3 {
            phase
        } else {
            self.adaptive_phase(frame, source_frame, scene_class, phase)
        };
        if matches!(self.mode, Mode::SmoothFps | Mode::Nvof)
            && matches!(phase, 0 | 256)
            && !(self.options.scene_blend()
                && self.options.request_scene_mode(self.mode.raw()) == 3)
        {
            return None;
        }
        let mut algo = self.effective_algorithm(requested_algo, scene_class);
        if algo == 2 && phase > 128 {
            algo = 1;
        }
        let dovi_changed =
            self.options.dovi_enabled() && unsafe { api.dovi_changed(source, next_source) };
        let block = vector_data.effective_block();
        let origin = vector_data.origin();
        let motion_grid = vector_data.motion_grid();
        let motion_w = usize::try_from(motion_grid.width.max(0)).ok()?;
        let motion_h = usize::try_from(motion_grid.height.max(0)).ok()?;
        let motion_count = motion_w.checked_mul(motion_h)?;
        let ctx = renderer::VectorContext {
            block_w: vector_data.block.width.max(1),
            block_h: vector_data.block.height.max(1),
            scale_shift_base: vector_data.marker.max(1),
            frame_w: self.video_info.width,
            frame_h: self.video_info.height,
            origin_x: origin.width,
            origin_y: origin.height,
            grid_w: usize::try_from(vector_data.grid.width.max(0)).ok()?,
            grid_h: usize::try_from(vector_data.grid.height.max(0)).ok()?,
            raw: false,
            a: &prep.decoded.vectors_prev,
            b: &prep.decoded.vectors_cur,
        };

        let mut mvx0 = Cow::Borrowed(prep.decoded.mvx0.as_slice());
        let mut mvy0 = Cow::Borrowed(prep.decoded.mvy0.as_slice());
        let mut mvx1 = Cow::Borrowed(prep.decoded.mvx1.as_slice());
        let mut mvy1 = Cow::Borrowed(prep.decoded.mvy1.as_slice());
        let mut mvx2 = Cow::Borrowed(prep.mvx2.as_slice());
        let mut mvy2 = Cow::Borrowed(prep.mvy2.as_slice());
        let mut mvx3 = Cow::Borrowed(prep.mvx3.as_slice());
        let mut mvy3 = Cow::Borrowed(prep.mvy3.as_slice());
        let mode23_ready = prep.side_ready
            && algo == 23
            && scene_class < 3
            && !dovi_changed
            && !neighbors[1].is_null()
            && !neighbors[2].is_null();
        let scene_blend_invert_threshold = self.scene_blend_invert_threshold(scene);
        let scene_blend_zero_origin =
            if matches!(self.mode, Mode::SmoothFps | Mode::Nvof) && self.options.scene_blend() {
                self.apply_scene_blend_direction(scene, &mut mvx0, &mut mvy0, &mut mvx1, &mut mvy1)
            } else {
                false
            };
        let interp = !self.options.block_enabled();
        let gpu_candidate = interp && !scene_blend_zero_origin && self.gpu.is_some();
        let (super0, super1) = if gpu_candidate {
            (None, None)
        } else {
            (
                unsafe {
                    super_frame_planes(
                        api,
                        merged_source,
                        &self.video_info,
                        self.super_info.as_ref(),
                        self.sdata,
                    )
                },
                unsafe {
                    super_frame_planes(
                        api,
                        next_merged_source,
                        &self.video_info,
                        self.super_info.as_ref(),
                        self.sdata,
                    )
                },
            )
        };
        if self.options.debug_zerox() {
            mvx0.to_mut().fill(renderer::NEUTRAL);
            mvx1.to_mut().fill(renderer::NEUTRAL);
            mvx2.to_mut().fill(renderer::NEUTRAL);
            mvx3.to_mut().fill(renderer::NEUTRAL);
        }
        if self.options.debug_zeroy() {
            mvy0.to_mut().fill(renderer::NEUTRAL);
            mvy1.to_mut().fill(renderer::NEUTRAL);
            mvy2.to_mut().fill(renderer::NEUTRAL);
            mvy3.to_mut().fill(renderer::NEUTRAL);
        }
        let inverse_phase = 256 - phase;
        let needs_coverage = algo >= 21;
        let mut mask0 = if needs_coverage {
            vec![0u8; motion_count]
        } else {
            Vec::new()
        };
        let mut mask1 = if needs_coverage {
            vec![0u8; motion_count]
        } else {
            Vec::new()
        };

        let magnitude0: &[u8] = &prep.magnitude0;
        let magnitude1: &[u8] = &prep.magnitude1;
        if needs_coverage {
            renderer::coverage_mask(
                &ctx,
                0,
                &mut mask0,
                motion_w,
                motion_h,
                self.options.mask_cover(),
                inverse_phase,
            );
            renderer::coverage_mask(
                &ctx,
                1,
                &mut mask1,
                motion_w,
                motion_h,
                self.options.mask_cover(),
                phase,
            );
        }
        let source_step = if super0.is_some() || super1.is_some() {
            metadata::super_data(self.sdata).scale().max(1)
        } else {
            1
        };
        let config = renderer::CpuConfig {
            width: self.video_info.width,
            height: self.video_info.height,
            block_w: block.width,
            block_h: block.height,
            origin_x: origin.width,
            origin_y: origin.height,
            grid_w: motion_w,
            grid_h: motion_h,
            chroma_y_div: 2,
            source_step,
            scale: self.options.mask_area_scale(),
        };
        let mut cpu = if gpu_candidate {
            renderer::CpuRenderer::new_deferred(config)
        } else {
            renderer::CpuRenderer::new(config)
        };
        let render_threshold = if scene_blend_invert_threshold {
            256 - phase
        } else {
            phase
        };
        if gpu_candidate {
            cpu.set_gpu_threshold(render_threshold);
        } else {
            cpu.set_threshold(render_threshold);
        }
        let raw_sources0 = unsafe {
            renderer::FramePlanes {
                y: plane(src0_y, src0_y_stride, src0_y_len),
                u: plane(src0_u, src0_u_stride, src0_u_len),
                v: plane(src0_v, src0_v_stride, src0_v_len),
            }
        };
        let raw_sources1 = unsafe {
            renderer::FramePlanes {
                y: plane(src1_y, src1_y_stride, src1_y_len),
                u: plane(src1_u, src1_u_stride, src1_u_len),
                v: plane(src1_v, src1_v_stride, src1_v_len),
            }
        };
        let expand0 = super0
            .as_ref()
            .and_then(|planes| self.cached_expand(i64::from(source_frame), planes));
        let expand1 = super1
            .as_ref()
            .and_then(|planes| self.cached_expand(i64::from(source_frame) + 1, planes));
        let expanded = |expand: &Option<std::sync::Arc<SuperExpand>>| {
            expand.as_ref().map(|expand| renderer::FramePlanes {
                y: unsafe { plane(expand.y.as_ptr(), expand.y_stride, expand.y.len()) },
                u: unsafe { plane(expand.u.as_ptr(), expand.uv_stride, expand.u.len()) },
                v: unsafe { plane(expand.v.as_ptr(), expand.uv_stride, expand.v.len()) },
            })
        };
        let sources0 = expanded(&expand0).or(super0).unwrap_or(raw_sources0);
        let sources1 = expanded(&expand1).or(super1).unwrap_or(raw_sources1);
        let padding = self.options.padding(&self.video_info);
        let dst = unsafe {
            renderer::FramePlanesMut {
                y: offset_plane_mut(dst_y, dst_y_stride, dst_y_len, padding.0, padding.1, 1),
                u: offset_plane_mut(
                    dst_u,
                    dst_u_stride,
                    dst_u_len,
                    padding.0 / 2,
                    padding.1 / 2,
                    1,
                ),
                v: offset_plane_mut(
                    dst_v,
                    dst_v_stride,
                    dst_v_len,
                    padding.0 / 2,
                    padding.1 / 2,
                    1,
                ),
            }
        };
        #[allow(unused_mut)]
        let mut dst = dst;
        let motion0 = renderer::MotionPlanes {
            x: mvx0.as_ref(),
            y: mvy0.as_ref(),
        };
        let motion1 = renderer::MotionPlanes {
            x: mvx1.as_ref(),
            y: mvy1.as_ref(),
        };
        let motion2 = renderer::MotionPlanes {
            x: mvx2.as_ref(),
            y: mvy2.as_ref(),
        };
        let motion3 = renderer::MotionPlanes {
            x: mvx3.as_ref(),
            y: mvy3.as_ref(),
        };
        let coverage = renderer::MaskPlanes {
            a: &mask0,
            b: &mask1,
        };
        let magnitude = renderer::MaskPlanes {
            a: magnitude0,
            b: magnitude1,
        };
        let area_mask = self.options.mask_area_enabled().then_some(magnitude);
        let mode21_mask = ((matches!(algo, 21 | 22) || (algo == 23 && !mode23_ready))
            && self.options.mask_area_enabled())
        .then(|| renderer::max_mask(magnitude));
        let mode23_mask = (mode23_ready && self.options.mask_area_enabled())
            .then(|| renderer::max_mask(magnitude));
        let qmap = self.options.debug_qmap().then(|| {
            qmap_classes(
                if phase > 128 { current } else { previous },
                &metadata::luma_map(
                    previous,
                    current,
                    vector_data.marker,
                    self.options.scene_luma(),
                )
                .map,
                vector_data.grid,
                self.options.qmap_thresholds(),
            )
        });
        let gpu_algorithm = if algo == 23 && !mode23_ready {
            21
        } else {
            algo
        };
        let used_gpu = gpu_candidate
            && self.gpu.as_ref().is_some_and(|gpu| {
                gpu_render_frame(
                    gpu,
                    &cpu,
                    &mut dst,
                    raw_sources0,
                    raw_sources1,
                    motion0,
                    motion1,
                    motion2,
                    motion3,
                    coverage,
                    area_mask,
                    gpu_algorithm,
                    source_frame,
                )
            });
        if !used_gpu {
            cpu.set_threshold(render_threshold);
            let render_parallel = |mode,
                                   dst: renderer::FramePlanesMut<'_>,
                                   motion2,
                                   motion3,
                                   masks,
                                   final_mask| {
                let input = |source0, source1| renderer::PlaneRenderInput {
                    source0,
                    source1,
                    motion0,
                    motion1,
                    motion2,
                    motion3,
                    masks: Some(masks),
                    final_mask,
                };
                let height = self.video_info.height;
                let middle = height / 2;
                let _ = cpu.render_plane_rows(
                    mode,
                    interp,
                    dst.y,
                    input(sources0.y, sources1.y),
                    false,
                    0..height,
                );
                if mode == 23 {
                    let _ = cpu.render_mode23_uv_rows(
                        interp,
                        [dst.u, dst.v],
                        input(sources0.u, sources1.u),
                        [sources0.v, sources1.v],
                        0..middle,
                    );
                } else if matches!(mode, 21 | 22) {
                    let _ = cpu.render_mode21_or_22_uv_rows(
                        mode == 21,
                        interp,
                        [dst.u, dst.v],
                        input(sources0.u, sources1.u),
                        [sources0.v, sources1.v],
                        0..middle,
                    );
                } else {
                    let _ = cpu.render_plane_rows(
                        mode,
                        interp,
                        dst.u,
                        input(sources0.u, sources1.u),
                        true,
                        0..middle,
                    );
                    let _ = cpu.render_plane_rows(
                        mode,
                        interp,
                        dst.v,
                        input(sources0.v, sources1.v),
                        true,
                        0..middle,
                    );
                }
            };
            match algo {
                1 | 2 => cpu.render_mode1_or_2(
                    algo == 1,
                    interp,
                    dst,
                    sources0,
                    sources1,
                    motion0,
                    motion1,
                    area_mask,
                ),
                11 | 13 if scene_blend_zero_origin => cpu.render_mode11_or_13_zero_origin(
                    algo == 13,
                    interp,
                    dst,
                    sources0,
                    sources1,
                    motion0,
                    motion1,
                    None,
                ),
                11 | 13 => cpu.render_mode11_or_13(
                    algo == 13,
                    interp,
                    dst,
                    sources0,
                    sources1,
                    motion0,
                    motion1,
                    area_mask,
                ),
                21 | 22 => {
                    render_parallel(
                        u32::try_from(algo).unwrap_or(21),
                        dst,
                        None,
                        None,
                        coverage,
                        mode21_mask.as_deref(),
                    );
                }
                23 if mode23_ready => {
                    render_parallel(
                        23,
                        dst,
                        Some(motion2),
                        Some(motion3),
                        coverage,
                        mode23_mask.as_deref(),
                    );
                }
                23 => {
                    render_parallel(21, dst, None, None, coverage, mode21_mask.as_deref());
                }
                _ => return None,
            }
        }
        let _ = unsafe { self.apply_light_border(api, output, frame) };
        if let Some(qmap) = qmap {
            let output_info = self.output_info();
            let dst = unsafe {
                renderer::FramePlanesMut {
                    y: plane_mut(dst_y, dst_y_stride, dst_y_len),
                    u: plane_mut(dst_u, dst_u_stride, dst_u_len),
                    v: plane_mut(dst_v, dst_v_stride, dst_v_len),
                }
            };
            qmap_overlay(
                &qmap,
                vector_data.effective_block(),
                vector_data.grid,
                &output_info,
                video_format::source_depth(&self.video_info),
                dst,
            );
        }
        if self.options.debug_vectors() {
            let output_info = self.output_info();
            let alpha = if phase > 128 { magnitude1 } else { magnitude0 };
            let selected_vectors = if phase > 128 { current } else { previous };
            let (debug_mvx, debug_mvy) = debug_motion_planes(selected_vectors);
            let dst = unsafe {
                renderer::FramePlanesMut {
                    y: plane_mut(dst_y, dst_y_stride, dst_y_len),
                    u: plane_mut(dst_u, dst_u_stride, dst_u_len),
                    v: plane_mut(dst_v, dst_v_stride, dst_v_len),
                }
            };
            vector_overlay(
                renderer::MotionPlanes {
                    x: &debug_mvx,
                    y: &debug_mvy,
                },
                alpha,
                vector_data.block,
                vector_data.origin(),
                vector_data.grid,
                &output_info,
                video_format::source_depth(&self.video_info),
                self.options.debug_zerox(),
                self.options.debug_zeroy(),
                dst,
            );
        }
        if self.options.debug_qmode() {
            let output_info = self.output_info();
            let dst = unsafe {
                renderer::FramePlanesMut {
                    y: plane_mut(dst_y, dst_y_stride, dst_y_len),
                    u: plane_mut(dst_u, dst_u_stride, dst_u_len),
                    v: plane_mut(dst_v, dst_v_stride, dst_v_len),
                }
            };
            qmode_overlay(scene_class, output_info.width, output_info.height, dst);
        }
        if self.options.debug_tt() {
            let _ = unsafe { self.apply_timing_bar(api, output, frame) };
        }
        Some(())
    }

    #[allow(clippy::similar_names)]
    unsafe fn apply_timing_bar(
        &self,
        api: &frame::PlaneApi,
        output: vs::Raw,
        frame: i32,
    ) -> Option<()> {
        let output_info = self.output_info();
        let chroma_h = usize_height(output_info.height / 2);
        let (dst_y, dst_y_stride, dst_y_len) =
            unsafe { api.write_plane(output, 0, usize_height(output_info.height)) }?;
        let (dst_u, dst_u_stride, dst_u_len) = unsafe { api.write_plane(output, 1, chroma_h) }?;
        let (dst_v, dst_v_stride, dst_v_len) = unsafe { api.write_plane(output, 2, chroma_h) }?;
        let dst = unsafe {
            renderer::FramePlanesMut {
                y: plane_mut(dst_y, dst_y_stride, dst_y_len),
                u: plane_mut(dst_u, dst_u_stride, dst_u_len),
                v: plane_mut(dst_v, dst_v_stride, dst_v_len),
            }
        };
        timing_bar(
            frame,
            output_info.width,
            output_info.height,
            video_format::source_depth(&self.video_info),
            dst,
        );
        Some(())
    }

    #[allow(clippy::similar_names)]
    unsafe fn apply_light_border(
        &self,
        api: &frame::PlaneApi,
        output: vs::Raw,
        frame: i32,
    ) -> Option<()> {
        let padding = self.options.padding(&self.video_info);
        if padding == (0, 0) {
            return Some(());
        }
        let output_info = self.output_info();
        let chroma_h = usize_height(output_info.height / 2);
        let (dst_y, dst_y_stride, dst_y_len) =
            unsafe { api.write_plane(output, 0, usize_height(output_info.height)) }?;
        let (dst_u, dst_u_stride, dst_u_len) = unsafe { api.write_plane(output, 1, chroma_h) }?;
        let (dst_v, dst_v_stride, dst_v_len) = unsafe { api.write_plane(output, 2, chroma_h) }?;
        let mut dst = unsafe {
            renderer::FramePlanesMut {
                y: plane_mut(dst_y, dst_y_stride, dst_y_len),
                u: plane_mut(dst_u, dst_u_stride, dst_u_len),
                v: plane_mut(dst_v, dst_v_stride, dst_v_len),
            }
        };
        self.light.apply(
            &self.video_info,
            padding,
            self.options.light_params(),
            frame,
            &mut dst,
        );
        Some(())
    }

    #[allow(clippy::similar_names)]
    unsafe fn apply_qmap_overlay(
        &self,
        api: &frame::PlaneApi,
        output: vs::Raw,
        vectors: vs::ConstRaw,
        phase: i32,
    ) -> Option<()> {
        if vectors.is_null() {
            return None;
        }
        let metadata::VectorRecord::Ready(vector_data) = self.vector_data() else {
            return None;
        };
        let count = usize::try_from(
            vector_data
                .grid
                .width
                .max(0)
                .saturating_mul(vector_data.grid.height.max(0)),
        )
        .ok()?;
        let vector_len = 0x48usize.saturating_add(count.saturating_mul(16));
        let decoded = unsafe { decode_frame_vectors(api, vectors, vector_data, vector_len) }?;
        let previous = decoded.previous.as_deref().or(decoded.current.as_deref())?;
        let current = decoded.current.as_deref().or(Some(previous))?;
        let qmap = qmap_classes(
            if phase > 128 { current } else { previous },
            &metadata::luma_map(
                previous,
                current,
                vector_data.marker,
                self.options.scene_luma(),
            )
            .map,
            vector_data.grid,
            self.options.qmap_thresholds(),
        );
        let output_info = self.output_info();
        let chroma_h = usize_height(output_info.height / 2);
        let (dst_y, dst_y_stride, dst_y_len) =
            unsafe { api.write_plane(output, 0, usize_height(output_info.height)) }?;
        let (dst_u, dst_u_stride, dst_u_len) = unsafe { api.write_plane(output, 1, chroma_h) }?;
        let (dst_v, dst_v_stride, dst_v_len) = unsafe { api.write_plane(output, 2, chroma_h) }?;
        let dst = unsafe {
            renderer::FramePlanesMut {
                y: plane_mut(dst_y, dst_y_stride, dst_y_len),
                u: plane_mut(dst_u, dst_u_stride, dst_u_len),
                v: plane_mut(dst_v, dst_v_stride, dst_v_len),
            }
        };
        qmap_overlay(
            &qmap,
            vector_data.effective_block(),
            vector_data.grid,
            &output_info,
            video_format::source_depth(&self.video_info),
            dst,
        );
        Some(())
    }

    #[allow(clippy::similar_names)]
    unsafe fn apply_vector_overlay(
        &self,
        api: &frame::PlaneApi,
        output: vs::Raw,
        vectors: vs::ConstRaw,
        phase: i32,
    ) -> Option<()> {
        if vectors.is_null() {
            return None;
        }
        let metadata::VectorRecord::Ready(vector_data) = self.vector_data() else {
            return None;
        };
        let count = usize::try_from(
            vector_data
                .grid
                .width
                .max(0)
                .saturating_mul(vector_data.grid.height.max(0)),
        )
        .ok()?;
        let vector_len = 0x48usize.saturating_add(count.saturating_mul(16));
        let decoded = unsafe { decode_frame_vectors(api, vectors, vector_data, vector_len) }?;
        let previous = decoded.previous.as_deref().or(decoded.current.as_deref())?;
        let current = decoded.current.as_deref().or(Some(previous))?;
        let selected = if phase > 128 { current } else { previous };
        let selected = renderer_vectors(selected);
        let origin = vector_data.origin();
        let motion_grid = vector_data.motion_grid();
        let motion_w = usize::try_from(motion_grid.width.max(0)).ok()?;
        let motion_h = usize::try_from(motion_grid.height.max(0)).ok()?;
        let motion_count = motion_w.checked_mul(motion_h)?;
        let ctx = renderer::VectorContext {
            block_w: vector_data.block.width.max(1),
            block_h: vector_data.block.height.max(1),
            scale_shift_base: 1,
            frame_w: self.video_info.width,
            frame_h: self.video_info.height,
            origin_x: origin.width,
            origin_y: origin.height,
            grid_w: usize::try_from(vector_data.grid.width.max(0)).ok()?,
            grid_h: usize::try_from(vector_data.grid.height.max(0)).ok()?,
            raw: false,
            a: &selected,
            b: &selected,
        };
        let mut mvx = vec![0u16; motion_count];
        let mut mvy = vec![0u16; motion_count];
        let mut alpha = vec![0u8; motion_count];
        let (debug_mvx, debug_mvy) =
            debug_motion_planes(if phase > 128 { current } else { previous });
        renderer::vector_planes(&ctx, 0, &mut mvx, &mut mvy, motion_w, motion_h);
        renderer::magnitude_mask(
            &ctx,
            0,
            &mut alpha,
            motion_w,
            motion_h,
            self.options.mask_area_scale(),
            self.options.mask_area_sharp(),
        );
        let output_info = self.output_info();
        let chroma_h = usize_height(output_info.height / 2);
        let (dst_y, dst_y_stride, dst_y_len) =
            unsafe { api.write_plane(output, 0, usize_height(output_info.height)) }?;
        let (dst_u, dst_u_stride, dst_u_len) = unsafe { api.write_plane(output, 1, chroma_h) }?;
        let (dst_v, dst_v_stride, dst_v_len) = unsafe { api.write_plane(output, 2, chroma_h) }?;
        let dst = unsafe {
            renderer::FramePlanesMut {
                y: plane_mut(dst_y, dst_y_stride, dst_y_len),
                u: plane_mut(dst_u, dst_u_stride, dst_u_len),
                v: plane_mut(dst_v, dst_v_stride, dst_v_len),
            }
        };
        vector_overlay(
            renderer::MotionPlanes {
                x: &debug_mvx,
                y: &debug_mvy,
            },
            &alpha,
            vector_data.block,
            vector_data.origin(),
            vector_data.grid,
            &output_info,
            video_format::source_depth(&self.video_info),
            self.options.debug_zerox(),
            self.options.debug_zeroy(),
            dst,
        );
        Some(())
    }

    fn source_frame(&self, frame: i32) -> i32 {
        let timing = self.options.timing(&self.video_info);
        let round_nearest = matches!(self.mode, Mode::SmoothFps | Mode::Nvof)
            && frame > 0
            && !self.options.scene_blend()
            && self.options.request_scene_mode(self.mode.raw()) != 0
            && self.options.request_scene_mode(self.mode.raw()) != 2;
        timing.source_frame(frame, round_nearest)
    }

    fn phase_256(&self, frame: i32, source_frame: i32) -> i32 {
        let timing = self.options.timing(&self.video_info);
        if timing.frame_num <= 0 {
            return 0;
        }
        let scene_mode = self.options.request_scene_mode(self.mode.raw());
        if scene_mode == 1 || scene_mode == 2 {
            return timing.scene_phase_256(
                timing.raw_phase_256(frame, source_frame),
                if scene_mode == 1 { 0 } else { 2 },
            );
        }
        timing.phase_256(frame, source_frame)
    }

    fn scene_class(
        &self,
        decoded: &metadata::DecodedVectors,
        vector_data: metadata::VectorData,
    ) -> i32 {
        let Some(previous) = decoded.previous.as_deref().or(decoded.current.as_deref()) else {
            return 3;
        };
        let Some(current) = decoded.current.as_deref().or(Some(previous)) else {
            return 3;
        };
        metadata::analyze_scene(
            previous,
            current,
            vector_data.marker,
            vector_data.grid,
            self.options.scene_luma(),
            self.options.scene_thresholds(),
        )
        .class
        .max(0)
    }

    fn effective_algorithm(&self, requested: i32, scene_class: i32) -> i32 {
        if matches!(self.mode, Mode::SmoothFps | Mode::Nvof)
            && (1..3).contains(&scene_class)
            && requested >= 11
            && self.options.scene_force13()
        {
            13
        } else {
            requested
        }
    }

    fn effective_scene(&self, raw: i32, phase: i32, neighbors: [Option<i32>; 4]) -> SceneBlend {
        if phase == 0 || phase == 256 {
            SceneBlend {
                class: 0,
                dir0: false,
                dir1: false,
            }
        } else if self.options.scene_blend() {
            blended_scene(raw, neighbors)
        } else {
            SceneBlend {
                class: raw,
                dir0: false,
                dir1: false,
            }
        }
    }

    fn adaptive_phase(&self, frame: i32, source_frame: i32, scene_class: i32, phase: i32) -> i32 {
        if self.options.request_scene_mode(self.mode.raw()) != 3 || scene_class >= 3 {
            return phase;
        }
        let timing = self.options.timing(&self.video_info);
        if timing.frame_den == 0 {
            return phase;
        }
        if f64_i64(timing.frame_num) / f64_i64(timing.frame_den) < 2.0 {
            return 0;
        }
        let Some(mode) = self.options.scene_adaptive(scene_class) else {
            return phase;
        };
        timing.scene_phase_256(timing.raw_phase_256(frame, source_frame), i64::from(mode))
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_scene_blend_direction(
        &self,
        scene: SceneBlend,
        x0: &mut Cow<'_, [u16]>,
        y0: &mut Cow<'_, [u16]>,
        x1: &mut Cow<'_, [u16]>,
        y1: &mut Cow<'_, [u16]>,
    ) -> bool {
        if !matches!(self.mode, Mode::SmoothFps | Mode::Nvof) || !self.options.scene_blend() {
            return false;
        }
        if !scene.dir0 && !scene.dir1 {
            x0.to_mut().fill(renderer::NEUTRAL);
            y0.to_mut().fill(renderer::NEUTRAL);
            x1.to_mut().fill(renderer::NEUTRAL);
            y1.to_mut().fill(renderer::NEUTRAL);
            return true;
        }
        if scene.class >= 3 {
            return false;
        }
        if !scene.dir0 {
            x0.to_mut().fill(renderer::NEUTRAL);
            y0.to_mut().fill(renderer::NEUTRAL);
        }
        if !scene.dir1 {
            x1.to_mut().fill(renderer::NEUTRAL);
            y1.to_mut().fill(renderer::NEUTRAL);
        }
        false
    }

    fn scene_blend_invert_threshold(&self, scene: SceneBlend) -> bool {
        matches!(self.mode, Mode::SmoothFps | Mode::Nvof)
            && self.options.scene_blend()
            && scene.class < 3
            && scene.dir0
            && !scene.dir1
    }

    fn request_vector_neighbors(
        &self,
        request_frame: vs::RequestFrameFilter,
        source_frame: i32,
        frame_ctx: vs::Raw,
    ) {
        let level = self
            .options
            .vector_neighbor_level(self.mode.raw(), source_frame);
        if level == 0 {
            return;
        }
        request_node(
            request_frame,
            self.clips.vectors,
            source_frame.saturating_sub(1).max(0),
            frame_ctx,
        );
        request_node(
            request_frame,
            self.clips.vectors,
            source_frame.saturating_add(1),
            frame_ctx,
        );
        if level >= 2 {
            request_node(
                request_frame,
                self.clips.vectors,
                source_frame.saturating_sub(2).max(0),
                frame_ctx,
            );
        }
        if level >= 3 {
            request_node(
                request_frame,
                self.clips.vectors,
                source_frame.saturating_add(2),
                frame_ctx,
            );
        }
    }

    fn request_vectors(
        &self,
        request_frame: vs::RequestFrameFilter,
        frame: i32,
        source_frame: i32,
        frame_ctx: vs::Raw,
    ) {
        request_node(request_frame, self.clips.vectors, source_frame, frame_ctx);
        if self.options.request_hdr_vectors() {
            self.request_hdr_vectors(request_frame, source_frame, frame_ctx);
        }
        if (1..=255).contains(&self.phase_256(frame, source_frame)) {
            self.request_vector_neighbors(request_frame, source_frame, frame_ctx);
        }
    }

    fn request_hdr_vectors(
        &self,
        request_frame: vs::RequestFrameFilter,
        source_frame: i32,
        frame_ctx: vs::Raw,
    ) {
        request_node(request_frame, self.clips.vectors, source_frame, frame_ctx);
        if source_frame > 0 {
            request_node(
                request_frame,
                self.clips.vectors,
                source_frame.saturating_sub(1),
                frame_ctx,
            );
        }
        request_node(
            request_frame,
            self.clips.vectors,
            source_frame.saturating_add(1),
            frame_ctx,
        );
        if source_frame >= 2 {
            request_node(
                request_frame,
                self.clips.vectors,
                source_frame.saturating_sub(2),
                frame_ctx,
            );
        }
        request_node(
            request_frame,
            self.clips.vectors,
            source_frame.saturating_add(2),
            frame_ctx,
        );
    }

    fn request_source_8bit(
        &self,
        request_frame: vs::RequestFrameFilter,
        source_frame: i32,
        frame_ctx: vs::Raw,
    ) {
        request_node(request_frame, self.clips.vec_src, source_frame, frame_ctx);
        request_node(
            request_frame,
            self.clips.vec_src,
            source_frame.saturating_add(1),
            frame_ctx,
        );
        if self.options.request_source_plus_two(self.mode.raw()) {
            request_node(
                request_frame,
                self.clips.vec_src,
                source_frame.saturating_add(2),
                frame_ctx,
            );
        }
    }

    fn fetch_hdr_vectors(
        &self,
        get_frame: vs::GetFrameFilter,
        free_frame: Option<vs::FreeFrame>,
        source_frame: i32,
        frame_ctx: vs::Raw,
    ) {
        fetch_node(
            get_frame,
            free_frame,
            self.clips.vectors,
            source_frame,
            frame_ctx,
        );
        if source_frame > 0 {
            fetch_node(
                get_frame,
                free_frame,
                self.clips.vectors,
                source_frame.saturating_sub(1),
                frame_ctx,
            );
        }
        fetch_node(
            get_frame,
            free_frame,
            self.clips.vectors,
            source_frame.saturating_add(1),
            frame_ctx,
        );
        if source_frame >= 2 {
            fetch_node(
                get_frame,
                free_frame,
                self.clips.vectors,
                source_frame.saturating_sub(2),
                frame_ctx,
            );
        }
        fetch_node(
            get_frame,
            free_frame,
            self.clips.vectors,
            source_frame.saturating_add(2),
            frame_ctx,
        );
    }

    fn fetch_vector_neighbors(
        &self,
        get_frame: vs::GetFrameFilter,
        free_frame: Option<vs::FreeFrame>,
        source_frame: i32,
        frame_ctx: vs::Raw,
    ) {
        let level = self
            .options
            .vector_neighbor_level(self.mode.raw(), source_frame);
        if level == 0 {
            return;
        }
        fetch_node(
            get_frame,
            free_frame,
            self.clips.vectors,
            source_frame.saturating_sub(1).max(0),
            frame_ctx,
        );
        fetch_node(
            get_frame,
            free_frame,
            self.clips.vectors,
            source_frame.saturating_add(1),
            frame_ctx,
        );
        if level >= 2 {
            fetch_node(
                get_frame,
                free_frame,
                self.clips.vectors,
                source_frame.saturating_sub(2).max(0),
                frame_ctx,
            );
        }
        if level >= 3 {
            fetch_node(
                get_frame,
                free_frame,
                self.clips.vectors,
                source_frame.saturating_add(2),
                frame_ctx,
            );
        }
    }

    fn fetch_source_8bit(
        &self,
        get_frame: vs::GetFrameFilter,
        free_frame: Option<vs::FreeFrame>,
        source_frame: i32,
        frame_ctx: vs::Raw,
    ) {
        fetch_node(
            get_frame,
            free_frame,
            self.clips.vec_src,
            source_frame,
            frame_ctx,
        );
        fetch_node(
            get_frame,
            free_frame,
            self.clips.vec_src,
            source_frame.saturating_add(1),
            frame_ctx,
        );
        if self.options.request_source_plus_two(self.mode.raw()) {
            fetch_node(
                get_frame,
                free_frame,
                self.clips.vec_src,
                source_frame.saturating_add(2),
                frame_ctx,
            );
        }
    }

    unsafe fn neighbor_scene_classes(
        &self,
        api: &frame::PlaneApi,
        neighbors: [vs::ConstRaw; 4],
        vector_data: metadata::VectorData,
        vector_len: usize,
        source_frame: i32,
    ) -> [Option<i32>; 4] {
        if !self.options.scene_blend() {
            return [None; 4];
        }
        let keys = neighbor_keys(source_frame);
        let class = |index: usize| {
            unsafe {
                self.cached_decode(api, keys[index], neighbors[index], vector_data, vector_len)
            }
            .map(|entry| entry.scene_class)
        };
        [class(0), class(1), class(2), class(3)]
    }

    fn vector_neighbors(
        &self,
        get_frame: vs::GetFrameFilter,
        source_frame: i32,
        phase: i32,
        algo: i64,
        frame_ctx: vs::Raw,
    ) -> [vs::ConstRaw; 4] {
        if !(algo == 23 || self.options.scene_blend()) || !(1..=255).contains(&phase) {
            return [std::ptr::null(); 4];
        }
        let level = self
            .options
            .vector_neighbor_level(self.mode.raw(), source_frame);
        if level == 0 {
            return [std::ptr::null(); 4];
        }
        let prev1 = get_node(
            get_frame,
            self.clips.vectors,
            source_frame.saturating_sub(1).max(0),
            frame_ctx,
        );
        let next1 = get_node(
            get_frame,
            self.clips.vectors,
            source_frame.saturating_add(1),
            frame_ctx,
        );
        let prev2 = if level >= 2 {
            get_node(
                get_frame,
                self.clips.vectors,
                source_frame.saturating_sub(2).max(0),
                frame_ctx,
            )
        } else {
            std::ptr::null()
        };
        let next2 = if level >= 3 {
            get_node(
                get_frame,
                self.clips.vectors,
                source_frame.saturating_add(2),
                frame_ctx,
            )
        } else {
            std::ptr::null()
        };
        [prev2, prev1, next1, next2]
    }
}

fn neighbor_keys(source_frame: i32) -> [i64; 4] {
    [
        i64::from(source_frame.saturating_sub(2).max(0)),
        i64::from(source_frame.saturating_sub(1).max(0)),
        i64::from(source_frame.saturating_add(1)),
        i64::from(source_frame.saturating_add(2)),
    ]
}

fn request_node(
    request_frame: vs::RequestFrameFilter,
    node: vs::Raw,
    frame: i32,
    frame_ctx: vs::Raw,
) {
    if !node.is_null() {
        unsafe { request_frame(frame, node, frame_ctx) };
    }
}

fn get_node(
    get_frame: vs::GetFrameFilter,
    node: vs::Raw,
    frame: i32,
    frame_ctx: vs::Raw,
) -> vs::ConstRaw {
    if node.is_null() {
        std::ptr::null()
    } else {
        unsafe { get_frame(frame, node, frame_ctx) }
    }
}

fn fetch_node(
    get_frame: vs::GetFrameFilter,
    free_frame: Option<vs::FreeFrame>,
    node: vs::Raw,
    frame: i32,
    frame_ctx: vs::Raw,
) {
    drop_frame(get_node(get_frame, node, frame, frame_ctx), free_frame);
}

fn drop_frame(frame: vs::ConstRaw, free_frame: Option<vs::FreeFrame>) {
    if frame.is_null() {
        return;
    }
    if let Some(free_frame) = free_frame {
        unsafe { free_frame(frame) };
    }
}

fn drop_frames(frames: [vs::ConstRaw; 4], free_frame: Option<vs::FreeFrame>) {
    for frame in frames {
        drop_frame(frame, free_frame);
    }
}

fn blended_scene(raw: i32, neighbors: [Option<i32>; 4]) -> SceneBlend {
    let [prev2, prev1, next1, next2] = neighbors;
    let dir0 = raw < 3 && prev1.is_some_and(|prev1| prev1 < 3);
    let dir1 = raw < 3 && next1.is_some_and(|next1| next1 < 3);
    let (Some(prev1), Some(next1)) = (prev1, next1) else {
        return SceneBlend {
            class: raw,
            dir0,
            dir1,
        };
    };
    if prev1 >= 3 || next1 >= 3 {
        return SceneBlend {
            class: raw,
            dir0,
            dir1,
        };
    }
    let mut sum = raw + prev1 + next1;
    let mut count = 3;
    if let Some(prev2) = prev2
        && prev2 < 3
        && next2.is_none_or(|next2| next2 < 3)
    {
        sum += prev2;
        count += 1;
        if let Some(next2) = next2 {
            sum += next2;
            count += 1;
        }
    }
    SceneBlend {
        class: (sum * 2 + count) / (count * 2),
        dir0,
        dir1,
    }
}

fn bytes_per_sample(info: &vs::VideoInfo) -> usize {
    if video_format::source_depth(info) == 0 {
        1
    } else {
        2
    }
}

fn timing_bar(frame: i32, width: i32, height: i32, depth: i32, dst: renderer::FramePlanesMut<'_>) {
    let period = width.saturating_mul(2).saturating_sub(20);
    if period <= 0 {
        return;
    }
    let x = width
        .saturating_sub(frame.saturating_mul(4).rem_euclid(period))
        .saturating_sub(10)
        .abs()
        .saturating_add(2);
    fill_yuv420_rect(dst, x, 0, 6, height, depth, [80, 39, 198]);
}

fn qmode_overlay(scene_class: i32, width: i32, height: i32, mut dst: renderer::FramePlanesMut<'_>) {
    let [y, u, v] = qmode_color(scene_class);
    fill_byte_plane(&mut dst.y, 0, 0, width.min(50), height.min(50), y);
    fill_byte_plane(
        &mut dst.u,
        0,
        0,
        (width / 2).min(25),
        (height / 2).min(25),
        u,
    );
    fill_byte_plane(
        &mut dst.v,
        0,
        0,
        (width / 2).min(25),
        (height / 2).min(25),
        v,
    );
}

fn qmode_color(scene_class: i32) -> [u8; 3] {
    match scene_class {
        0 => [0x1D, 0xFF, 0x6B],
        1 => [0x95, 0x2B, 0x15],
        2 => [0xFF, 0x00, 0x94],
        3 => [0x4C, 0x54, 0xFF],
        _ => [0, 0, 0],
    }
}

fn debug_motion_planes(vectors: &[metadata::DecodedVector]) -> (Vec<u16>, Vec<u16>) {
    let mut x = Vec::with_capacity(vectors.len());
    let mut y = Vec::with_capacity(vectors.len());
    for vector in vectors {
        x.push(debug_motion_value(vector.dx));
        y.push(debug_motion_value(vector.dy));
    }
    (x, y)
}

fn debug_motion_value(value: i16) -> u16 {
    u16::try_from((i32::from(renderer::NEUTRAL) + i32::from(value)).clamp(0, i32::from(u16::MAX)))
        .unwrap_or(u16::MAX)
}

#[allow(clippy::similar_names, clippy::too_many_arguments)]
fn vector_overlay(
    motion: renderer::MotionPlanes<'_>,
    alpha: &[u8],
    block: metadata::VectorShape,
    origin: metadata::VectorShape,
    grid: metadata::VectorShape,
    output: &vs::VideoInfo,
    depth: i32,
    suppress_x_motion: bool,
    suppress_y_motion: bool,
    mut dst: renderer::FramePlanesMut<'_>,
) {
    let grid_w = usize::try_from(grid.width.max(0)).unwrap_or(0);
    let grid_h = usize::try_from(grid.height.max(0)).unwrap_or(0);
    let step_x = block.width.saturating_sub(origin.width);
    let step_y = block.height.saturating_sub(origin.height);
    for gy in 0..grid_h {
        for gx in 0..grid_w {
            let index = gy.saturating_mul(grid_w).saturating_add(gx);
            let alpha = alpha.get(index).copied().unwrap_or(0);
            let [cy, cu, cv] = vector_color(alpha);
            let gx = i32::try_from(gx).unwrap_or(i32::MAX);
            let gy = i32::try_from(gy).unwrap_or(i32::MAX);
            let x = block.width / 2 + step_x.saturating_mul(gx);
            let y = block.height / 2 + step_y.saturating_mul(gy);
            let line_x = x.saturating_sub(1);
            let line_y = y.saturating_sub(1);
            blend_vector_pixel(&mut dst, line_x, line_y, output, depth, [cy, cu, cv]);
            blend_vector_pixel(
                &mut dst,
                x.saturating_sub(2),
                line_y,
                output,
                depth,
                [cy, cu, cv],
            );
            blend_vector_pixel(&mut dst, x, line_y, output, depth, [cy, cu, cv]);
            blend_vector_pixel(
                &mut dst,
                line_x,
                y.saturating_sub(2),
                output,
                depth,
                [cy, cu, cv],
            );
            blend_vector_pixel(&mut dst, line_x, y, output, depth, [cy, cu, cv]);

            let dx = if suppress_x_motion {
                0
            } else {
                i32::from(motion.x.get(index).copied().unwrap_or(renderer::NEUTRAL))
                    - i32::from(renderer::NEUTRAL)
            };
            let dy = if suppress_y_motion {
                0
            } else {
                i32::from(motion.y.get(index).copied().unwrap_or(renderer::NEUTRAL))
                    - i32::from(renderer::NEUTRAL)
            };
            if dx.unsigned_abs() > 1 || dy.unsigned_abs() >= 2 {
                draw_vector_line(
                    &mut dst,
                    line_x,
                    line_y,
                    dx,
                    dy,
                    output,
                    depth,
                    [cy, cu, cv],
                );
            }
        }
    }
}

fn vector_color(alpha: u8) -> [u8; 3] {
    let inv = alpha ^ 0xFF;
    [
        ((u16::from(inv) * 255 + u16::from(alpha) * 76) >> 8) as u8,
        ((u16::from(inv) * 128 + u16::from(alpha) * 84) >> 8) as u8,
        ((u16::from(inv) * 128 + u16::from(alpha) * 255) >> 8) as u8,
    ]
}

#[allow(clippy::too_many_arguments)]
fn draw_vector_line(
    dst: &mut renderer::FramePlanesMut<'_>,
    mut x: i32,
    mut y: i32,
    dx: i32,
    dy: i32,
    output: &vs::VideoInfo,
    depth: i32,
    color: [u8; 3],
) {
    let end_x = x.saturating_add(dx);
    let end_y = y.saturating_add(dy);
    let step_x = if dx > 0 { 1 } else { -1 };
    let step_y = if dy > 0 { 1 } else { -1 };
    let ax = dx.saturating_abs();
    let ay = dy.saturating_abs();
    let mut err = ax - ay;
    loop {
        write_vector_pixel(dst, x, y, output, depth, color);
        if x == end_x && y == end_y {
            break;
        }
        let twice = 2 * err;
        if twice > -ay {
            err -= ay;
            x += step_x;
        }
        if twice < ax {
            err += ax;
            y += step_y;
        }
    }
}

fn blend_vector_pixel(
    dst: &mut renderer::FramePlanesMut<'_>,
    x: i32,
    y: i32,
    output: &vs::VideoInfo,
    depth: i32,
    color: [u8; 3],
) {
    write_vector_pixel_inner(dst, x, y, output, depth, color, true);
}

fn write_vector_pixel(
    dst: &mut renderer::FramePlanesMut<'_>,
    x: i32,
    y: i32,
    output: &vs::VideoInfo,
    depth: i32,
    color: [u8; 3],
) {
    write_vector_pixel_inner(dst, x, y, output, depth, color, false);
}

fn write_vector_pixel_inner(
    dst: &mut renderer::FramePlanesMut<'_>,
    x: i32,
    y: i32,
    output: &vs::VideoInfo,
    depth: i32,
    color: [u8; 3],
    blend: bool,
) {
    if y < 0 || y >= output.height || x < 0 || x >= output.width.saturating_sub(1) {
        return;
    }
    let y_offset = vector_offset(x, y, dst.y.stride, depth, false);
    let u_offset = vector_offset(x, y, dst.u.stride, depth, true);
    if let Some(byte) = dst.y.data.get_mut(y_offset) {
        *byte = if blend {
            blend_vector_byte(*byte, color[0])
        } else {
            color[0]
        };
    }
    if let Some(byte) = dst.u.data.get_mut(u_offset) {
        *byte = if blend {
            blend_vector_byte(*byte, color[1])
        } else {
            color[1]
        };
    }
    if let Some(byte) = dst.v.data.get_mut(u_offset) {
        *byte = if blend {
            blend_vector_byte(*byte, color[2])
        } else {
            color[2]
        };
    }
}

fn vector_offset(x: i32, y: i32, stride: usize, depth: i32, chroma: bool) -> usize {
    let row = if chroma { y >> 1 } else { y };
    let col = if chroma {
        if depth == 0 { x >> 1 } else { x }
    } else if depth == 0 {
        x
    } else {
        x << 1
    };
    usize::try_from(row)
        .unwrap_or(0)
        .saturating_mul(stride)
        .saturating_add(usize::try_from(col.saturating_add(1)).unwrap_or(0))
}

fn blend_vector_byte(dst: u8, value: u8) -> u8 {
    ((u16::from(value) * 140 + u16::from(dst) * 115) >> 8) as u8
}

fn qmap_classes(
    vectors: &[metadata::DecodedVector],
    luma: &[u8],
    grid: metadata::VectorShape,
    thresholds: metadata::SceneThresholds,
) -> Vec<u8> {
    let width = usize::try_from(grid.width.max(0)).unwrap_or(0);
    let height = usize::try_from(grid.height.max(0)).unwrap_or(0);
    let mut out = vec![3; width.saturating_mul(height)];
    for (index, class) in out.iter_mut().enumerate() {
        let Some(vector) = vectors.get(index) else {
            continue;
        };
        let luma = i32::from(luma.get(index).copied().unwrap_or(1).max(1));
        let score = i32::try_from(vector.score)
            .unwrap_or(i32::MAX)
            .saturating_mul(255)
            / luma;
        *class = if score < thresholds.zero {
            u8::MAX
        } else if score < thresholds.m1 {
            0
        } else if score < thresholds.m2 {
            1
        } else if score < thresholds.scene {
            2
        } else {
            3
        };
    }
    out
}

fn qmap_overlay(
    qmap: &[u8],
    block: metadata::VectorShape,
    grid: metadata::VectorShape,
    output: &vs::VideoInfo,
    depth: i32,
    mut dst: renderer::FramePlanesMut<'_>,
) {
    let grid_w = usize::try_from(grid.width.max(0)).unwrap_or(0);
    let grid_h = usize::try_from(grid.height.max(0)).unwrap_or(0);
    for y in 0..grid_h {
        for x in 0..grid_w {
            let class = qmap
                .get(y.saturating_mul(grid_w).saturating_add(x))
                .copied()
                .unwrap_or(u8::MAX);
            if class == u8::MAX {
                continue;
            }
            let [cy, cu, cv] = qmode_color(i32::from(class));
            let x = i32::try_from(x).unwrap_or(i32::MAX);
            let y = i32::try_from(y).unwrap_or(i32::MAX);
            let rx = x.saturating_mul(block.width);
            let ry = y.saturating_mul(block.height);
            blend_qmap_plane(
                &mut dst.y,
                rx,
                ry,
                block.width,
                block.height,
                output.width,
                output.height,
                depth,
                false,
                cy,
            );
            blend_qmap_plane(
                &mut dst.u,
                rx,
                ry,
                block.width,
                block.height,
                output.width,
                output.height,
                depth,
                true,
                cu,
            );
            blend_qmap_plane(
                &mut dst.v,
                rx,
                ry,
                block.width,
                block.height,
                output.width,
                output.height,
                depth,
                true,
                cv,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn blend_qmap_plane(
    plane: &mut renderer::PlaneMut<'_>,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    frame_w: i32,
    frame_h: i32,
    depth: i32,
    chroma: bool,
    value: u8,
) {
    let y0 = y.max(0);
    let y1 = y.saturating_add(height).min(frame_h.max(0));
    let x0 = x.max(0);
    let x1 = x
        .saturating_add(width)
        .min(frame_w.saturating_sub(1).max(0));
    for py in y0..y1 {
        let row = if chroma { py >> 1 } else { py };
        let Ok(row) = usize::try_from(row) else {
            continue;
        };
        for px in x0..x1 {
            let col = if chroma {
                if depth == 0 { px >> 1 } else { px }
            } else if depth == 0 {
                px
            } else {
                px << 1
            };
            let Ok(col) = usize::try_from(col.saturating_add(1)) else {
                continue;
            };
            let offset = row.saturating_mul(plane.stride).saturating_add(col);
            if let Some(dst) = plane.data.get_mut(offset) {
                *dst = blend_qmap_byte(*dst, value);
            }
        }
    }
}

fn blend_qmap_byte(dst: u8, value: u8) -> u8 {
    ((u16::from(value) * 20 + u16::from(dst) * 235) >> 8) as u8
}

fn fill_yuv420_rect(
    mut dst: renderer::FramePlanesMut<'_>,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    depth: i32,
    color: [u16; 3],
) {
    fill_plane(&mut dst.y, x, y, width, height, depth, color[0]);
    fill_plane(
        &mut dst.u,
        x / 2,
        y / 2,
        width / 2,
        height / 2,
        depth,
        color[1],
    );
    fill_plane(
        &mut dst.v,
        x / 2,
        y / 2,
        width / 2,
        height / 2,
        depth,
        color[2],
    );
}

fn fill_byte_plane(
    plane: &mut renderer::PlaneMut<'_>,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    value: u8,
) {
    let x = usize::try_from(x.max(0)).unwrap_or(0);
    let y = usize::try_from(y.max(0)).unwrap_or(0);
    let width = usize::try_from(width.max(0)).unwrap_or(0);
    let height = usize::try_from(height.max(0)).unwrap_or(0);
    for row in y..y.saturating_add(height) {
        let start = row.saturating_mul(plane.stride).saturating_add(x);
        let end = start.saturating_add(width).min(plane.data.len());
        if start < end {
            plane.data[start..end].fill(value);
        }
    }
}

fn fill_plane(
    plane: &mut renderer::PlaneMut<'_>,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    depth: i32,
    value: u16,
) {
    let bytes = if depth == 0 { 1 } else { 2 };
    let shift = if depth == 10 { 2 } else { 8 };
    let sample = if depth == 0 { value } else { value << shift };
    let x = usize::try_from(x.max(0)).unwrap_or(0);
    let y = usize::try_from(y.max(0)).unwrap_or(0);
    let width = usize::try_from(width.max(0)).unwrap_or(0);
    let height = usize::try_from(height.max(0)).unwrap_or(0);
    for row in y..y.saturating_add(height) {
        let base = row.saturating_mul(plane.stride);
        for col in x..x.saturating_add(width) {
            let offset = base.saturating_add(col.saturating_mul(bytes));
            if depth == 0 {
                if let Some(dst) = plane.data.get_mut(offset) {
                    *dst = u8::try_from(sample).unwrap_or(0);
                }
            } else if let Some(dst) = plane.data.get_mut(offset..offset.saturating_add(2)) {
                dst.copy_from_slice(&sample.to_le_bytes());
            }
        }
    }
}

fn expand_plane(plane: renderer::Plane<'_>, width: usize, height: usize) -> Option<(Vec<u8>, usize)> {
    let (stride, span, shift) = plane.layout();
    if shift == 0 || shift > 2 || width == 0 || height == 0 {
        return None;
    }
    let pel = 1usize << shift;
    let out_stride = width.checked_mul(pel)?;
    let mut out = vec![0u8; out_stride.checked_mul(height.checked_mul(pel)?)?];
    for sy in 0..pel {
        for row in 0..height {
            let dst = out
                .get_mut((row * pel + sy) * out_stride..(row * pel + sy) * out_stride + out_stride)?;
            let sub = |sx: usize| {
                let base = ((sy << shift) | sx).checked_mul(span)?.checked_add(row.checked_mul(stride)?)?;
                plane.data.get(base..base.checked_add(width)?)
            };
            if pel == 2 {
                let (a, b) = (sub(0)?, sub(1)?);
                for ((chunk, &left), &right) in dst.chunks_exact_mut(2).zip(a).zip(b) {
                    chunk[0] = left;
                    chunk[1] = right;
                }
            } else {
                let (a, b, c, d) = (sub(0)?, sub(1)?, sub(2)?, sub(3)?);
                for ((((chunk, &s0), &s1), &s2), &s3) in
                    dst.chunks_exact_mut(4).zip(a).zip(b).zip(c).zip(d)
                {
                    chunk[0] = s0;
                    chunk[1] = s1;
                    chunk[2] = s2;
                    chunk[3] = s3;
                }
            }
        }
    }
    Some((out, out_stride))
}

fn renderer_vectors(vectors: &[metadata::DecodedVector]) -> Vec<renderer::Vector> {
    vectors
        .iter()
        .map(|vector| renderer::Vector {
            dx: vector.dx,
            dy: vector.dy,
            magnitude: vector.score,
        })
        .collect()
}

unsafe fn decode_frame_vectors(
    api: &frame::PlaneApi,
    frame: vs::ConstRaw,
    vector_data: metadata::VectorData,
    vector_len: usize,
) -> Option<metadata::DecodedVectors> {
    let (vector_ptr, _, _) = unsafe { api.read_plane(frame, 0, 1) }?;
    let vector_payload = unsafe { slice::from_raw_parts(vector_ptr, vector_len) };
    metadata::decode_vectors(
        vector_payload,
        vector_data.flags,
        vector_data.grid,
        vector_data.flags,
    )
}

unsafe fn pack_nv12_frame(
    api: &frame::PlaneApi,
    frame: vs::ConstRaw,
    width: usize,
    height: usize,
) -> Option<Vec<u8>> {
    if frame.is_null() {
        return None;
    }
    let chroma_height = height / 2;
    let (y, y_stride, y_len) = unsafe { api.read_plane(frame, 0, height) }?;
    let (u, uv_stride, u_len) = unsafe { api.read_plane(frame, 1, chroma_height) }?;
    let (v, _, v_len) = unsafe { api.read_plane(frame, 2, chroma_height) }?;
    Some(metadata::pack_nv12(
        unsafe { slice::from_raw_parts(y, y_len) },
        unsafe { slice::from_raw_parts(u, u_len) },
        unsafe { slice::from_raw_parts(v, v_len) },
        width,
        height,
        y_stride,
        uv_stride,
    ))
}

unsafe fn decode_vectors_input(
    api: &frame::PlaneApi,
    input: VectorInput<'_>,
    vector_data: metadata::VectorData,
    vector_len: usize,
) -> Option<metadata::DecodedVectors> {
    match input {
        VectorInput::Frame(frame) => unsafe {
            decode_frame_vectors(api, frame, vector_data, vector_len)
        },
        VectorInput::Payload(payload) => metadata::decode_vectors(
            payload.get(..vector_len)?,
            vector_data.flags,
            vector_data.grid,
            vector_data.flags,
        ),
    }
}

fn usize_height(value: i32) -> usize {
    usize::try_from(value.max(0)).unwrap_or(0)
}

pub(crate) struct FramePrep {
    decoded: std::sync::Arc<DecodedEntry>,

    mvx2: Vec<u16>,
    mvy2: Vec<u16>,
    mvx3: Vec<u16>,
    mvy3: Vec<u16>,
    magnitude0: Vec<u8>,
    magnitude1: Vec<u8>,
    side_ready: bool,
    raw_scene_class: i32,
    neighbor_classes: [Option<i32>; 4],
}

#[allow(
    clippy::too_many_arguments,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
fn gpu_render_frame(
    gpu: &crate::gpu::GpuContext,
    cpu: &renderer::CpuRenderer,
    dst: &mut renderer::FramePlanesMut<'_>,
    sources0: renderer::FramePlanes<'_>,
    sources1: renderer::FramePlanes<'_>,
    base0: renderer::MotionPlanes<'_>,
    base1: renderer::MotionPlanes<'_>,
    next0: renderer::MotionPlanes<'_>,
    prev1: renderer::MotionPlanes<'_>,
    coverage: renderer::MaskPlanes<'_>,
    area: Option<renderer::MaskPlanes<'_>>,
    algorithm: i32,
    source_frame: i32,
) -> bool {
    fn uploads(
        sources: renderer::FramePlanes<'_>,
        plane_y: renderer::GpuPlaneParams,
        plane_uv: renderer::GpuPlaneParams,
    ) -> [crate::gpu::UploadPlane<'_>; 3] {
        [
            crate::gpu::UploadPlane {
                data: sources.y.data,
                stride: sources.y.stride,
                width: plane_y.width as usize,
                height: plane_y.height as usize,
            },
            crate::gpu::UploadPlane {
                data: sources.u.data,
                stride: sources.u.stride,
                width: plane_uv.width as usize,
                height: plane_uv.height as usize,
            },
            crate::gpu::UploadPlane {
                data: sources.v.data,
                stride: sources.v.stride,
                width: plane_uv.width as usize,
                height: plane_uv.height as usize,
            },
        ]
    }
    if std::env::var_os("SVP_KERNEL_OFF").is_some() {
        return false;
    }
    let (phase, _) = cpu.gpu_thresholds();
    let (grid_w, grid_h) = cpu.gpu_grid();
    let plane_y = cpu.gpu_plane_params(false);
    let plane_uv = cpu.gpu_plane_params(true);
    let Some(src0) = gpu.cache_frame(
        i64::from(source_frame),
        uploads(sources0, plane_y, plane_uv),
    ) else {
        return false;
    };
    let Some(src1) = gpu.cache_frame(
        i64::from(source_frame) + 1,
        uploads(sources1, plane_y, plane_uv),
    ) else {
        return false;
    };
    let motions = [
        (base0.x, base0.y),
        (base1.x, base1.y),
        (next0.x, next0.y),
        (prev1.x, prev1.y),
    ];
    let make = |chroma: bool| {
        let p = cpu.gpu_plane_params(chroma);
        crate::gpu::KernelParams {
            algorithm,
            width: p.width,
            height: p.height,
            x_ratio: p.x_div,
            y_ratio: p.y_div,
            pel: p.source_step,
            block_w: p.block_w,
            block_h: p.block_h,
            origin_x: p.origin_x,
            origin_y: p.origin_y,
            phase,
            has_sad: i32::from(area.is_some()),
            linear_luma: i32::from(!chroma),
        }
    };
    gpu.render_frame(
        src0.bufs(),
        src1.bufs(),
        dst.y.data,
        dst.y.stride as i32,
        make(false),
        dst.u.data,
        dst.u.stride as i32,
        make(true),
        dst.v.data,
        dst.v.stride as i32,
        make(true),
        i64::from(source_frame),
        grid_w,
        grid_h,
        motions,
        (coverage.a, coverage.b),
        area.map(|mask| (mask.a, mask.b)),
    )
    .is_some()
}

#[allow(
    clippy::too_many_arguments,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
unsafe fn super_frame_planes(
    api: &frame::PlaneApi,
    frame: vs::ConstRaw,
    info: &vs::VideoInfo,
    super_info: Option<&vs::VideoInfo>,
    sdata: i64,
) -> Option<renderer::FramePlanes<'static>> {
    let lane = usize::try_from(metadata::super_data(sdata).scale()).ok()?;
    if frame.is_null() || !matches!(lane, 2 | 4) {
        return None;
    }
    let shift = lane.trailing_zeros();
    let y_height = usize_height(info.height);
    let uv_height = y_height / 2;
    let read_height = usize_height(super_info.map_or(info.height, |info| info.height));
    let y = unsafe { api.read_plane(frame, 0, read_height) }?;
    let u = unsafe { api.read_plane(frame, 1, read_height / 2) }?;
    let v = unsafe { api.read_plane(frame, 2, read_height / 2) }?;
    Some(renderer::FramePlanes {
        y: unsafe { super_plane(y.0, y.1, y.2, y_height, shift) }?,
        u: unsafe { super_plane(u.0, u.1, u.2, uv_height, shift) }?,
        v: unsafe { super_plane(v.0, v.1, v.2, uv_height, shift) }?,
    })
}

unsafe fn plane(ptr: *const u8, stride: usize, len: usize) -> renderer::Plane<'static> {
    renderer::Plane::linear(unsafe { slice::from_raw_parts(ptr, len) }, stride)
}

unsafe fn super_plane(
    ptr: *const u8,
    stride: usize,
    len: usize,
    height: usize,
    shift: u32,
) -> Option<renderer::Plane<'static>> {
    renderer::Plane::super_plane(
        unsafe { slice::from_raw_parts(ptr, len) },
        stride,
        stride.checked_mul(height)?,
        shift,
    )
}

unsafe fn plane_mut(ptr: *mut u8, stride: usize, len: usize) -> renderer::PlaneMut<'static> {
    renderer::PlaneMut {
        data: unsafe { slice::from_raw_parts_mut(ptr, len) },
        stride,
    }
}

#[allow(clippy::needless_pass_by_value)]
unsafe fn offset_plane_mut(
    ptr: *mut u8,
    stride: usize,
    len: usize,
    x: i32,
    y: i32,
    bytes_per_sample: usize,
) -> renderer::PlaneMut<'static> {
    let x = usize::try_from(x.max(0)).unwrap_or(0);
    let y = usize::try_from(y.max(0)).unwrap_or(0);
    let offset = y
        .saturating_mul(stride)
        .saturating_add(x.saturating_mul(bytes_per_sample))
        .min(len);
    renderer::PlaneMut {
        data: unsafe { slice::from_raw_parts_mut(ptr.add(offset), len - offset) },
        stride,
    }
}

#[allow(clippy::cast_precision_loss)]
fn f64_i64(value: i64) -> f64 {
    value as f64
}
