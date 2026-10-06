use std::ffi::CString;

use crate::core::{FilterState, drop_frame, get_node};
use crate::gpu::{BlendJob, GpuContext, OutputPlane, UploadPlane};
use crate::{filter, frame, strings, video_format, vs};

struct BlendState {
    inner: FilterState,
    plan: Plan,
}

struct Plan {
    weights: Vec<f32>,
    divisor: f32,
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
    let reason = if inner.gpu.is_none() {
        Some("GPU rendering is required (build the super clip with gpu:1)")
    } else if !video_format::is_yuv420p8(&inner.video_info) {
        Some("clip must be YUV 4:2:0 8-bit")
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
    unsafe fn render(
        &self,
        n: i32,
        frame_ctx: vs::Raw,
        core: vs::Raw,
        vsapi: vs::ConstRaw,
    ) -> Option<vs::ConstRaw> {
        let gpu = self.inner.gpu.as_ref()?;
        let api = unsafe { frame::PlaneApi::load(vsapi) }?;
        let get_frame = unsafe { vs::table_fn::<vs::GetFrameFilter>(vsapi, vs::GET_FRAME_FILTER) }?;
        let free_frame = unsafe { vs::table_fn::<vs::FreeFrame>(vsapi, vs::FREE_FRAME) };
        let Plan { info, .. } = &self.plan;
        let width = usize::try_from(info.width).ok()?;
        let height = usize::try_from(info.height).ok()?;
        let sizes = [
            (width, height),
            (width / 2, height / 2),
            (width / 2, height / 2),
        ];
        let job = gpu.blend_begin(width, height, height / 2)?;
        for (frame, weight) in self.plan.window(n) {
            job.start(weight);
            let rendered = unsafe {
                self.inner
                    .get_frame(frame, Some(&job), frame_ctx, core, vsapi)
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
        let key = [source_frame, source_frame.saturating_add(1)]
            .into_iter()
            .find(|&index| {
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
    _: *mut vs::Raw,
    frame_ctx: vs::Raw,
    core: vs::Raw,
    vsapi: vs::ConstRaw,
) -> vs::ConstRaw {
    let state = unsafe { &*(*instance_data).cast::<BlendState>() };
    match activation_reason {
        vs::AR_INITIAL => {
            let mut requested = -1;
            for (frame, _) in state.plan.window(n) {
                if frame != requested {
                    unsafe { state.inner.request_frame(frame, frame_ctx, vsapi) };
                    requested = frame;
                }
            }
            std::ptr::null()
        }
        vs::AR_ALL_FRAMES_READY => {
            let output = unsafe { state.render(n, frame_ctx, core, vsapi) };
            if output.is_none()
                && let Some(set_error) =
                    unsafe { vs::table_fn::<vs::SetFilterError>(vsapi, vs::SET_FILTER_ERROR) }
            {
                unsafe { set_error(c"SVSmoothFpsBlend: GPU blend failed".as_ptr(), frame_ctx) };
            }
            output.unwrap_or(std::ptr::null())
        }
        _ => std::ptr::null(),
    }
}
