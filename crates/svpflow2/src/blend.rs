use std::ffi::CString;

use crate::core::{FilterState, drop_frame, get_node};
use crate::gpu::{BlendJob, GpuContext, OutputPlane, UploadPlane};
use crate::reference::Held;
use crate::{filter, frame, strings, video_format, vs};

struct BlendState {
    inner: FilterState,
    plan: Plan,
}

struct Plan {
    weights: Vec<f32>,
    divisor: f32,
    gamma: f32,
    info: vs::VideoInfo,
    step: (i64, i64),
    frames: i32,
}

pub(crate) unsafe extern "system" fn create_smooth_fps_blend(
    input: vs::ConstRaw,
    output: vs::Raw,
    _: vs::Raw,
    core: vs::Raw,
    vsapi: vs::ConstRaw,
) -> isize {
    let Some((inner, api)) = (unsafe { filter::build_state(0, input, output, core, vsapi) }) else {
        return 0;
    };
    let state = match unsafe { plan(&inner, input, vsapi) } {
        Ok(plan) => BlendState { inner, plan },
        Err(message) => {
            let message = CString::new(format!("SVSmoothFpsBlend: {message}")).unwrap_or_default();
            unsafe {
                filter::set_error(output, vsapi, message.as_ptr());
                inner.free(vsapi);
            }
            return 0;
        }
    };
    unsafe {
        (api.create_filter)(
            input,
            output,
            strings::BLEND_FILTER_NAME.as_ptr().cast(),
            init_filter,
            get_frame,
            free_filter,
            vs::FM_PARALLEL,
            0,
            Box::into_raw(Box::new(state)).cast(),
            core,
        );
    }
    0
}

unsafe fn plan(
    inner: &FilterState,
    input: vs::ConstRaw,
    vsapi: vs::ConstRaw,
) -> Result<Plan, &'static str> {
    let options = &inner.options;
    let reason = if !video_format::is_cpu_source(&inner.video_info) {
        Some("clip must be YUV 4:2:0 or 4:4:4 8-bit")
    } else if inner.gpu.is_none() && !inner.cpu_fits() {
        Some("GPU rendering is required for vectors with odd overlap")
    } else if options.padding(&vs::Source::source(&inner.video_info)) != (0, 0)
        || options.debug_qmap()
        || options.debug_vectors()
        || options.debug_qmode()
        || options.debug_tt()
    {
        Some("'light' and 'debug' options are not supported")
    } else {
        None
    };
    if let Some(reason) = reason {
        return Err(reason);
    }
    let (Some(get_float), Some(get_int)) = (unsafe {
        (
            vs::table_fn::<vs::PropGetFloat>(vsapi, vs::PROP_GET_FLOAT),
            vs::table_fn::<vs::PropGetInt>(vsapi, vs::PROP_GET_INT),
        )
    }) else {
        return Err("invalid VSAPI table");
    };
    let mut weights = Vec::new();
    for index in 0.. {
        let mut error = 0;
        let weight = unsafe {
            get_float(
                input,
                strings::WEIGHTS.as_ptr().cast(),
                index,
                &raw mut error,
            )
        };
        if error != 0 {
            break;
        }
        weights.push(weight);
    }
    let divisor = weights.iter().sum::<f64>();
    if weights.len() % 2 == 0 || !divisor.is_finite() || divisor == 0.0 {
        return Err("'weights' must hold an odd number of values");
    }
    let int = |key: &[u8], default| {
        let mut error = 0;
        let value = unsafe { get_int(input, key.as_ptr().cast(), 0, &raw mut error) };
        if error == 0 { value } else { default }
    };
    let fps = (int(strings::FPS_NUM, 0), int(strings::FPS_DEN, 1));
    let mut error = 0;
    let gamma = unsafe { get_float(input, strings::GAMMA.as_ptr().cast(), 0, &raw mut error) };
    let gamma = if error == 0 { gamma } else { 1.0 };
    if !gamma.is_finite() || gamma <= 0.0 {
        return Err("'gamma' must be positive");
    }
    let mut info = inner.output_info();
    if fps.0 <= 0 || fps.1 <= 0 || info.fps_num <= 0 || info.fps_den <= 0 {
        return Err("'fpsnum' and 'fpsden' must be positive");
    }
    let step = (info.fps_num * fps.1, info.fps_den * fps.0);
    let frames = info.num_frames;
    info.num_frames = i32::try_from(i64::from(frames) * step.1 / step.0)
        .unwrap_or(i32::MAX)
        .max(1);
    info.fps_num = fps.0;
    info.fps_den = fps.1;
    Ok(Plan {
        weights: weights.into_iter().map(|weight| weight as f32).collect(),
        divisor: divisor as f32,
        gamma: gamma as f32,
        info,
        step,
        frames,
    })
}

impl Plan {
    fn center(&self, n: i32) -> i64 {
        i64::from(n) * self.step.0 / self.step.1
    }

    fn window(&self, n: i32) -> impl Iterator<Item = (i32, f32)> + '_ {
        let radius = i64::try_from(self.weights.len() / 2).unwrap_or(0);
        let center = self.center(n);
        let last = i64::from(self.frames - 1);
        (center - radius..=center + radius)
            .rev()
            .zip(self.weights.iter().rev())
            .map(move |(frame, &weight)| (frame.clamp(0, last) as i32, weight))
    }
}

impl BlendState {
    fn fast_blend_level(&self) -> u8 {
        if !self.inner.options.fast_blend() {
            0
        } else if self.inner.options.fast() >= 3 {
            2
        } else {
            1
        }
    }

    fn window_entries(&self, n: i32, single_per_run: bool) -> Vec<(i32, f32)> {
        if !self.inner.options.fast_blend()
            || self.plan.weights.len() <= 5
            || self.inner.options.dedup().is_some()
            || self.inner.options.mask_still().is_some()
        {
            return self.plan.window(n).collect();
        }
        let params = self.inner.options.reference();
        let rates = svpflow_core::smooth_engine::Rates::new(
            &params,
            &vs::Source::source(&self.inner.video_info),
        );
        let centroid = |slice: &[(i32, f32)], total: f32| -> i32 {
            let target = total * 0.5;
            let mut acc = 0.0f32;
            for &(frame, weight) in slice {
                acc += weight;
                if acc >= target {
                    return frame;
                }
            }
            slice[slice.len() / 2].0
        };
        let mut out = Vec::with_capacity(8);
        let mut run: Vec<(i32, f32)> = Vec::new();
        let mut run_key: Option<(i32, bool)> = None;
        let flush = |out: &mut Vec<(i32, f32)>, run: &mut Vec<(i32, f32)>| {
            if run.is_empty() {
                return;
            }
            let total: f32 = run.iter().map(|&(_, w)| w).sum();
            if total <= 0.0 || (!single_per_run && run.len() <= 2) {
                out.extend_from_slice(run);
                run.clear();
                return;
            }
            if single_per_run {
                out.push((centroid(run, total), total));
                run.clear();
                return;
            }
            let half_total = total * 0.5;
            let mut acc = 0.0f32;
            let mut split = 1;
            for (idx, &(_, weight)) in run.iter().enumerate() {
                acc += weight;
                if acc >= half_total {
                    split = (idx + 1).clamp(1, run.len() - 1);
                    break;
                }
            }
            let left = &run[..split];
            let right = &run[split..];
            let w0: f32 = left.iter().map(|&(_, w)| w).sum();
            let w1 = total - w0;
            let f0 = centroid(left, w0);
            let f1 = centroid(right, w1);
            if f0 == f1 {
                out.push((f0, total));
            } else {
                out.push((f0, w0));
                out.push((f1, w1));
            }
            run.clear();
        };
        for (frame, weight) in self.plan.window(n) {
            let step = rates.step(&params, frame);
            let key = (step.phase != 0 && step.phase != 256).then_some((step.n, step.phase > 127));
            if key.is_some() && key == run_key {
                run.push((frame, weight));
            } else {
                flush(&mut out, &mut run);
                if key.is_some() {
                    run_key = key;
                    run.push((frame, weight));
                } else {
                    run_key = None;
                    out.push((frame, weight));
                }
            }
        }
        flush(&mut out, &mut run);
        out
    }

    unsafe fn render(
        &self,
        n: i32,
        frame_ctx: vs::Raw,
        core: vs::Raw,
        vsapi: vs::ConstRaw,
        held: &Held,
    ) -> Option<vs::ConstRaw> {
        let api = unsafe { frame::PlaneApi::load(vsapi) }?;
        let get_frame = unsafe { vs::table_fn::<vs::GetFrameFilter>(vsapi, vs::GET_FRAME_FILTER) }?;
        let free_frame = unsafe { vs::table_fn::<vs::FreeFrame>(vsapi, vs::FREE_FRAME) };
        let sizes = self.sizes()?;
        let entries = self.window_entries(n, false);
        if let Some(sources) = unsafe { self.inner.pure_sources(&entries, frame_ctx, vsapi) }
            && let Some(output) = unsafe {
                self.blend_sources_cpu(
                    n, &sources, &api, get_frame, free_frame, frame_ctx, core, sizes,
                )
            }
        {
            return Some(output);
        }
        let gpu = self.inner.gpu.as_ref()?;
        let [(width, height), chroma, _] = sizes;
        let first_src = self
            .inner
            .source_frame(self.plan.window(n).last().map_or(0, |(f, _)| f));
        let last_src = self
            .inner
            .source_frame(self.plan.window(n).next().map_or(0, |(f, _)| f));
        let span = usize::try_from(last_src.saturating_sub(first_src).max(0) + 2).unwrap_or(2);
        gpu.keep_frames(span.saturating_mul(4).min(32));
        let job = gpu.blend_begin(
            width.max(chroma.0 * 2),
            height,
            chroma.1,
            self.plan.gamma,
            self.inner.options.adaptive_blend(),
            self.inner.options.fast() >= 1,
            self.fast_blend_level(),
        )?;
        for (frame, weight) in entries {
            job.start(weight);
            let rendered = unsafe {
                self.inner
                    .get_frame(frame, Some(&job), frame_ctx, core, vsapi, held)
            };
            let added = job.taken()
                || unsafe {
                    self.add_frame(
                        gpu, &job, &api, get_frame, free_frame, frame, rendered, frame_ctx, sizes,
                    )
                }
                .is_some();
            drop_frame(rendered, free_frame);
            if !added || job.failed() {
                gpu.blend_abort(job);
                return None;
            }
        }
        let output = unsafe { self.new_output(n, &api, get_frame, free_frame, frame_ctx, core) };
        let Some(output) = output else {
            gpu.blend_abort(job);
            return None;
        };
        let mut planes = Vec::with_capacity(3);
        for (plane, (width, height)) in (0..).zip(sizes) {
            let Some((data, stride, len)) = (unsafe { api.write_plane(output, plane, height) })
            else {
                break;
            };
            planes.push(OutputPlane {
                data: unsafe { std::slice::from_raw_parts_mut(data, len) },
                stride,
                width,
                height,
            });
        }
        let done = if let Ok(planes) = <[OutputPlane<'_>; 3]>::try_from(planes) {
            gpu.blend_finish(job, self.plan.divisor, planes)
        } else {
            gpu.blend_abort(job);
            None
        };
        if done.is_none() {
            unsafe { api.free(output.cast_const()) };
            return None;
        }
        let info = &self.plan.info;
        unsafe { api.set_duration(output, info.fps_den, info.fps_num) };
        Some(output.cast_const())
    }

    fn sizes(&self) -> Option<[(usize, usize); 3]> {
        let info = &self.plan.info;
        let width = usize::try_from(info.width).ok()?;
        let height = usize::try_from(info.height).ok()?;
        let (cx, cy) = video_format::chroma_divisors(info);
        let chroma = (width / cx as usize, height / cy as usize);
        Some([(width, height), chroma, chroma])
    }

    unsafe fn new_output(
        &self,
        n: i32,
        api: &frame::PlaneApi,
        get_frame: vs::GetFrameFilter,
        free_frame: Option<vs::FreeFrame>,
        frame_ctx: vs::Raw,
        core: vs::Raw,
    ) -> Option<vs::Raw> {
        let info = &self.plan.info;
        let center = self
            .plan
            .center(n)
            .clamp(0, i64::from(self.plan.frames - 1)) as i32;
        let source = get_node(
            get_frame,
            self.inner.clips.source,
            self.inner.source_frame(center),
            frame_ctx,
        );
        let output = unsafe { api.new_frame(info.format, info.width, info.height, source, core) };
        drop_frame(source, free_frame);
        output
    }

    unsafe fn render_cpu(
        &self,
        n: i32,
        frame_ctx: vs::Raw,
        core: vs::Raw,
        vsapi: vs::ConstRaw,
        held: &Held,
    ) -> Option<vs::ConstRaw> {
        let api = unsafe { frame::PlaneApi::load(vsapi) }?;
        let get_frame = unsafe { vs::table_fn::<vs::GetFrameFilter>(vsapi, vs::GET_FRAME_FILTER) }?;
        let free_frame = unsafe { vs::table_fn::<vs::FreeFrame>(vsapi, vs::FREE_FRAME) };
        let sizes = self.sizes()?;
        let Plan { gamma, divisor, .. } = self.plan;
        let curved = |plane| plane == 0 && gamma.to_bits() != 1.0_f32.to_bits();
        let mut sums = sizes.map(|(width, height)| vec![0.0f32; width * height]);
        let entries: Vec<(i32, f32)> = if self.inner.options.fast_blend()
            && self.plan.weights.len() > 5
        {
            self.window_entries(n, self.inner.options.fast() >= 3)
        } else if self.inner.options.adaptive_blend() && self.plan.weights.len() > 7 {
            let stride = if self.plan.weights.len() > 31 { 3 } else { 2 };
            let raw: Vec<(i32, f32)> = self.plan.window(n).collect();
            raw.chunks(stride)
                .map(|chunk| {
                    let mid = chunk[chunk.len() / 2].0;
                    let wsum: f32 = chunk.iter().map(|&(_, w)| w).sum();
                    (mid, wsum)
                })
                .collect()
        } else {
            self.plan.window(n).collect()
        };
        if let Some(sources) = unsafe { self.inner.pure_sources(&entries, frame_ctx, vsapi) }
            && let Some(output) = unsafe {
                self.blend_sources_cpu(
                    n, &sources, &api, get_frame, free_frame, frame_ctx, core, sizes,
                )
            }
        {
            return Some(output);
        }
        for (frame, weight) in entries {
            let rendered = unsafe {
                self.inner
                    .get_frame(frame, None, frame_ctx, core, vsapi, held)
            };
            let added = !rendered.is_null()
                && (0..)
                    .zip(&mut sums)
                    .zip(sizes)
                    .all(|((plane, sum), (width, height))| {
                        let Some((data, stride, len)) =
                            (unsafe { api.read_plane(rendered, plane, height) })
                        else {
                            return false;
                        };
                        let data = unsafe { std::slice::from_raw_parts(data, len) };
                        for (row, sum) in data.chunks(stride).zip(sum.chunks_mut(width)) {
                            for (sum, &value) in sum.iter_mut().zip(row) {
                                let value = f32::from(value);
                                let value = if curved(plane) {
                                    (value / 255.0).powf(gamma)
                                } else {
                                    value
                                };
                                *sum += weight * value;
                            }
                        }
                        true
                    });
            drop_frame(rendered, free_frame);
            if !added {
                return None;
            }
        }
        let output = unsafe { self.new_output(n, &api, get_frame, free_frame, frame_ctx, core) }?;
        for (plane, (sum, (width, height))) in (0..).zip(sums.iter().zip(sizes)) {
            let Some((data, stride, len)) = (unsafe { api.write_plane(output, plane, height) })
            else {
                unsafe { api.free(output.cast_const()) };
                return None;
            };
            let data = unsafe { std::slice::from_raw_parts_mut(data, len) };
            for (row, sum) in data.chunks_mut(stride).zip(sum.chunks(width)) {
                for (pixel, &sum) in row.iter_mut().zip(sum) {
                    let value = sum / divisor;
                    let value = if curved(plane) {
                        255.0 * value.max(0.0).powf(1.0 / gamma)
                    } else {
                        value
                    };
                    *pixel = value.round_ties_even().clamp(0.0, 255.0) as u8;
                }
            }
        }
        let info = &self.plan.info;
        unsafe { api.set_duration(output, info.fps_den, info.fps_num) };
        Some(output.cast_const())
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn blend_sources_cpu(
        &self,
        n: i32,
        sources: &[(i32, f32)],
        api: &frame::PlaneApi,
        get_frame: vs::GetFrameFilter,
        free_frame: Option<vs::FreeFrame>,
        frame_ctx: vs::Raw,
        core: vs::Raw,
        sizes: [(usize, usize); 3],
    ) -> Option<vs::ConstRaw> {
        if sources.is_empty() {
            return None;
        }
        let Plan { gamma, divisor, .. } = self.plan;
        let curved_luma = gamma.to_bits() != 1.0_f32.to_bits();
        let coalesce = self.inner.options.fast() >= 1 && !curved_luma;
        let mut combined: Vec<(i32, f32)> = Vec::with_capacity(sources.len());
        if coalesce {
            for &(idx, weight) in sources {
                if let Some(slot) = combined.iter_mut().find(|(k, _)| *k == idx) {
                    slot.1 += weight;
                } else {
                    combined.push((idx, weight));
                }
            }
        } else {
            combined.extend_from_slice(sources);
        }
        let mut frames: Vec<vs::ConstRaw> = Vec::with_capacity(combined.len());
        let mut weights: Vec<f32> = Vec::with_capacity(combined.len());
        for &(idx, weight) in &combined {
            let raw = get_node(get_frame, self.inner.clips.source, idx, frame_ctx);
            if raw.is_null() {
                for frame in frames {
                    drop_frame(frame, free_frame);
                }
                return None;
            }
            frames.push(raw);
            weights.push(weight);
        }
        let output = unsafe { self.new_output(n, api, get_frame, free_frame, frame_ctx, core) };
        let Some(output) = output else {
            for frame in frames {
                drop_frame(frame, free_frame);
            }
            return None;
        };
        let to_lin = curved_luma.then(|| {
            std::array::from_fn::<f32, 256, _>(|i| (f32::from(i as u8) / 255.0).powf(gamma))
        });
        let inv_gamma = 1.0 / gamma;
        let exact_copy = frames.len() == 1
            && !curved_luma
            && (weights[0] / divisor).to_bits() == 1.0_f32.to_bits();
        let mut ok = true;
        for (plane, (width, height)) in (0i32..).zip(sizes) {
            let Some((dst_ptr, dst_stride, dst_len)) =
                (unsafe { api.write_plane(output, plane, height) })
            else {
                ok = false;
                break;
            };
            let dst = unsafe { std::slice::from_raw_parts_mut(dst_ptr, dst_len) };
            let mut src_planes: Vec<(&[u8], usize)> = Vec::with_capacity(frames.len());
            for &raw in &frames {
                let Some((ptr, stride, len)) = (unsafe { api.read_plane(raw, plane, height) })
                else {
                    ok = false;
                    break;
                };
                src_planes.push((unsafe { std::slice::from_raw_parts(ptr, len) }, stride));
            }
            if !ok {
                break;
            }
            if exact_copy {
                let (src, src_stride) = src_planes[0];
                for y in 0..height {
                    dst[y * dst_stride..y * dst_stride + width]
                        .copy_from_slice(&src[y * src_stride..y * src_stride + width]);
                }
                continue;
            }
            let mut rows: Vec<&[u8]> = Vec::with_capacity(src_planes.len());
            if plane == 0
                && let Some(lut) = &to_lin
            {
                for y in 0..height {
                    rows.clear();
                    for &(data, stride) in &src_planes {
                        rows.push(&data[y * stride..y * stride + width]);
                    }
                    let out_row = &mut dst[y * dst_stride..y * dst_stride + width];
                    blend_row_gamma(out_row, &rows, &weights, divisor, lut, inv_gamma);
                }
            } else {
                for y in 0..height {
                    rows.clear();
                    for &(data, stride) in &src_planes {
                        rows.push(&data[y * stride..y * stride + width]);
                    }
                    let out_row = &mut dst[y * dst_stride..y * dst_stride + width];
                    blend_row_f32(out_row, &rows, &weights, divisor);
                }
            }
        }
        for frame in frames {
            drop_frame(frame, free_frame);
        }
        if !ok {
            unsafe { api.free(output.cast_const()) };
            return None;
        }
        let info = &self.plan.info;
        unsafe { api.set_duration(output, info.fps_den, info.fps_num) };
        Some(output.cast_const())
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn add_frame(
        &self,
        gpu: &GpuContext,
        job: &BlendJob,
        api: &frame::PlaneApi,
        get_frame: vs::GetFrameFilter,
        free_frame: Option<vs::FreeFrame>,
        frame: i32,
        rendered: vs::ConstRaw,
        frame_ctx: vs::Raw,
        sizes: [(usize, usize); 3],
    ) -> Option<()> {
        if rendered.is_null() {
            return None;
        }
        let pixels = unsafe { api.read_plane(rendered, 0, sizes[0].1) }?.0;
        let source_frame = self.inner.source_frame(frame);
        let reach = if self.inner.options.dedup().is_some() {
            4
        } else {
            1
        };
        let key = (source_frame..=source_frame.saturating_add(reach)).find(|&index| {
            let source = get_node(get_frame, self.inner.clips.source, index, frame_ctx);
            let same = !source.is_null()
                && unsafe { api.read_plane(source, 0, sizes[0].1) }
                    .is_some_and(|plane| plane.0 == pixels);
            drop_frame(source, free_frame);
            same
        });
        let plane = |plane: i32, (width, height): (usize, usize)| {
            let (data, stride, len) = unsafe { api.read_plane(rendered, plane, height) }?;
            Some(UploadPlane {
                data: unsafe { std::slice::from_raw_parts(data, len) },
                stride,
                width,
                height,
            })
        };
        let planes = [
            plane(0, sizes[0])?,
            plane(1, sizes[1])?,
            plane(2, sizes[2])?,
        ];
        gpu.blend_planes(job, key.map(i64::from), planes)
    }
}

unsafe extern "system" fn init_filter(
    _: vs::ConstRaw,
    _: vs::Raw,
    instance_data: *mut vs::Raw,
    node: vs::Raw,
    _: vs::Raw,
    vsapi: vs::ConstRaw,
) {
    let state = unsafe { &*(*instance_data).cast::<BlendState>() };
    if let Some(set_video_info) =
        unsafe { vs::table_fn::<vs::SetVideoInfo>(vsapi, vs::SET_VIDEO_INFO) }
    {
        unsafe { set_video_info(&raw const state.plan.info, 1, node) };
    }
}

unsafe extern "system" fn free_filter(instance_data: vs::Raw, _: vs::Raw, vsapi: vs::ConstRaw) {
    let state = unsafe { Box::from_raw(instance_data.cast::<BlendState>()) };
    unsafe { state.inner.free(vsapi) };
}

unsafe extern "system" fn get_frame(
    n: i32,
    activation_reason: i32,
    instance_data: *mut vs::Raw,
    frame_data: *mut vs::Raw,
    frame_ctx: vs::Raw,
    core: vs::Raw,
    vsapi: vs::ConstRaw,
) -> vs::ConstRaw {
    let state = unsafe { &*(*instance_data).cast::<BlendState>() };
    let held = unsafe { std::mem::replace(&mut *frame_data, std::ptr::null_mut()) };
    let mut held = if held.is_null() {
        Box::<Held>::default()
    } else {
        unsafe { Box::from_raw(held.cast::<Held>()) }
    };
    match activation_reason {
        vs::AR_INITIAL => {
            let mut requested = -1;
            for (frame, _) in state.plan.window(n) {
                if frame != requested {
                    unsafe {
                        state
                            .inner
                            .request_frame(frame, frame_ctx, vsapi, &mut held);
                    }
                    requested = frame;
                }
            }
            unsafe { *frame_data = Box::into_raw(held).cast() };
            std::ptr::null()
        }
        vs::AR_ALL_FRAMES_READY => {
            let window: Vec<i32> = state.plan.window(n).map(|(frame, _)| frame).collect();
            // called again once the dedup vectors are ready
            if !held.more
                && unsafe {
                    state
                        .inner
                        .reference_more(&window, frame_ctx, vsapi, &mut held)
                }
            {
                held.more = true;
                unsafe { *frame_data = Box::into_raw(held).cast() };
                return std::ptr::null();
            }
            let gpu = state.inner.gpu.as_ref();
            if let Some(gpu) = gpu {
                gpu.retry();
            }
            let fits = state.inner.cpu_fits();
            let output = if fits && gpu.is_none_or(GpuContext::cpu_render) {
                None
            } else {
                loop {
                    let mark = gpu.map(GpuContext::mark);
                    let output = unsafe { state.render(n, frame_ctx, core, vsapi, &held) };
                    if output.is_some()
                        || !gpu.zip(mark).is_some_and(|(gpu, mark)| gpu.recover(mark))
                    {
                        break output;
                    }
                }
            };
            let output = output.or_else(|| {
                fits.then(|| unsafe { state.render_cpu(n, frame_ctx, core, vsapi, &held) })
                    .flatten()
            });
            if output.is_none()
                && let Some(set_error) =
                    unsafe { vs::table_fn::<vs::SetFilterError>(vsapi, vs::SET_FILTER_ERROR) }
            {
                unsafe { set_error(c"SVSmoothFpsBlend: blend failed".as_ptr(), frame_ctx) };
            }
            output.unwrap_or(std::ptr::null())
        }
        _ => std::ptr::null(),
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn blend_row_f32_avx2(dst: &mut [u8], srcs: &[&[u8]], weights: &[f32], divisor: f32) {
    use std::arch::x86_64::{
        _MM_FROUND_NO_EXC, _MM_FROUND_TO_NEAREST_INT, _mm_loadu_si128, _mm_packus_epi16,
        _mm_srli_si128, _mm_storeu_si128, _mm256_add_ps, _mm256_castsi256_si128, _mm256_cvtepi32_ps,
        _mm256_cvtepu8_epi32, _mm256_cvtps_epi32, _mm256_div_ps, _mm256_extracti128_si256,
        _mm256_max_ps, _mm256_min_ps, _mm256_mul_ps, _mm256_packus_epi32,
        _mm256_permute4x64_epi64, _mm256_round_ps, _mm256_set1_ps, _mm256_setzero_ps,
    };
    let width = dst.len();
    let vdiv = _mm256_set1_ps(divisor);
    let vzero = _mm256_setzero_ps();
    let v255 = _mm256_set1_ps(255.0);
    let mut x = 0usize;
    while x + 16 <= width {
        let mut s0 = vzero;
        let mut s1 = vzero;
        for (src, &w) in srcs.iter().zip(weights) {
            let vw = _mm256_set1_ps(w);
            let raw = unsafe { _mm_loadu_si128(src.as_ptr().add(x).cast()) };
            let lo = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(raw));
            let hi = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128::<8>(raw)));
            s0 = _mm256_add_ps(_mm256_mul_ps(vw, lo), s0);
            s1 = _mm256_add_ps(_mm256_mul_ps(vw, hi), s1);
        }
        let q0 = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            _mm256_div_ps(s0, vdiv),
        );
        let q1 = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            _mm256_div_ps(s1, vdiv),
        );
        let i0 = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_max_ps(q0, vzero), v255));
        let i1 = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_max_ps(q1, vzero), v255));
        let p16 = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi32(i0, i1));
        let p8 = _mm_packus_epi16(_mm256_castsi256_si128(p16), _mm256_extracti128_si256::<1>(p16));
        unsafe { _mm_storeu_si128(dst.as_mut_ptr().add(x).cast(), p8) };
        x += 16;
    }
    for (idx, pixel) in dst[x..].iter_mut().enumerate() {
        let col = x + idx;
        let mut sum = 0.0f32;
        for (src, &w) in srcs.iter().zip(weights) {
            sum += w * f32::from(src[col]);
        }
        *pixel = (sum / divisor).round_ties_even().clamp(0.0, 255.0) as u8;
    }
}

#[cfg(target_arch = "aarch64")]
fn blend_row_f32_neon(dst: &mut [u8], srcs: &[&[u8]], weights: &[f32], divisor: f32) {
    use std::arch::aarch64::{
        vaddq_f32, vcombine_u16, vcvtq_f32_u32, vcvtq_u32_f32, vdivq_f32, vdupq_n_f32,
        vget_high_u16, vget_low_u16, vld1_u8, vmaxq_f32, vminq_f32, vmovl_u8, vmovl_u16,
        vmulq_f32, vqmovn_u16, vqmovn_u32, vrndnq_f32, vst1_u8,
    };
    let width = dst.len();
    let vdiv = unsafe { vdupq_n_f32(divisor) };
    let vzero = unsafe { vdupq_n_f32(0.0) };
    let v255 = unsafe { vdupq_n_f32(255.0) };
    let mut x = 0usize;
    while x + 8 <= width {
        let mut s0 = vzero;
        let mut s1 = vzero;
        for (src, &w) in srcs.iter().zip(weights) {
            unsafe {
                let vw = vdupq_n_f32(w);
                let u16s = vmovl_u8(vld1_u8(src.as_ptr().add(x)));
                let lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16s)));
                let hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16s)));
                s0 = vaddq_f32(vmulq_f32(vw, lo), s0);
                s1 = vaddq_f32(vmulq_f32(vw, hi), s1);
            }
        }
        unsafe {
            let q0 = vrndnq_f32(vdivq_f32(s0, vdiv));
            let q1 = vrndnq_f32(vdivq_f32(s1, vdiv));
            let u0 = vqmovn_u32(vcvtq_u32_f32(vminq_f32(vmaxq_f32(q0, vzero), v255)));
            let u1 = vqmovn_u32(vcvtq_u32_f32(vminq_f32(vmaxq_f32(q1, vzero), v255)));
            vst1_u8(dst.as_mut_ptr().add(x), vqmovn_u16(vcombine_u16(u0, u1)));
        }
        x += 8;
    }
    for (idx, pixel) in dst[x..].iter_mut().enumerate() {
        let col = x + idx;
        let mut sum = 0.0f32;
        for (src, &w) in srcs.iter().zip(weights) {
            sum += w * f32::from(src[col]);
        }
        *pixel = (sum / divisor).round_ties_even().clamp(0.0, 255.0) as u8;
    }
}

fn blend_row_f32(dst: &mut [u8], srcs: &[&[u8]], weights: &[f32], divisor: f32) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        unsafe { blend_row_f32_avx2(dst, srcs, weights, divisor) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    {
        blend_row_f32_neon(dst, srcs, weights, divisor);
        return;
    }
    #[allow(unreachable_code)]
    for (col, pixel) in dst.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        for (src, &w) in srcs.iter().zip(weights) {
            sum += w * f32::from(src[col]);
        }
        *pixel = (sum / divisor).round_ties_even().clamp(0.0, 255.0) as u8;
    }
}

fn blend_row_gamma(
    dst: &mut [u8],
    srcs: &[&[u8]],
    weights: &[f32],
    divisor: f32,
    lut: &[f32; 256],
    inv_gamma: f32,
) {
    for (col, pixel) in dst.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        for (src, &w) in srcs.iter().zip(weights) {
            sum += w * lut[usize::from(src[col])];
        }
        let value = 255.0 * (sum / divisor).max(0.0).powf(inv_gamma);
        *pixel = value.round_ties_even().clamp(0.0, 255.0) as u8;
    }
}
