use std::sync::{Arc, Mutex};

use svpflow_core::still;

use crate::core::{drop_frame, get_node, request_node};
use crate::{filter, frame, video_format, vs};

struct StillState {
    clip: vs::Raw,
    source: vs::Raw,
    info: vs::VideoInfo,
    step: (i64, i64),
    limits: [f32; 3],
    masks: Mutex<Vec<(i32, Arc<Vec<u8>>)>>,
}

pub(crate) unsafe extern "system" fn create_still(
    input: vs::ConstRaw,
    output: vs::Raw,
    _: vs::Raw,
    core: vs::Raw,
    vsapi: vs::ConstRaw,
) -> isize {
    let (Some(get_node), Some(get_info), Some(get_float), Some(create)) = (unsafe {
        (
            vs::table_fn::<vs::PropGetNode>(vsapi, vs::PROP_GET_NODE),
            vs::table_fn::<vs::GetVideoInfo>(vsapi, vs::GET_VIDEO_INFO),
            vs::table_fn::<vs::PropGetFloat>(vsapi, vs::PROP_GET_FLOAT),
            vs::table_fn::<vs::CreateFilter>(vsapi, vs::CREATE_FILTER),
        )
    }) else {
        return 0;
    };
    let mut error = 0;
    let clip = unsafe { get_node(input, c"clip".as_ptr(), 0, &raw mut error) };
    let source = unsafe { get_node(input, c"source".as_ptr(), 0, &raw mut error) };
    if clip.is_null() || source.is_null() {
        unsafe { vs::free_nodes([clip, source], vsapi) };
        return 0;
    }
    let (info, original) = unsafe { (*get_info(clip), *get_info(source)) };
    let step = (
        original.fps_num * info.fps_den,
        original.fps_den * info.fps_num,
    );
    if !video_format::is_cpu_source(&info)
        || info.format != original.format
        || (info.width, info.height) != (original.width, original.height)
        || step.0 <= 0
        || step.1 <= 0
    {
        unsafe {
            vs::free_nodes([clip, source], vsapi);
            filter::set_error(
                output,
                vsapi,
                c"SVStill: both clips must be YUV420P8 or YUV444P8 of one size with frame rates"
                    .as_ptr(),
            );
        }
        return 0;
    }
    let number = |key: &std::ffi::CStr| {
        let mut error = 0;
        let value = unsafe { get_float(input, key.as_ptr(), 0, &raw mut error) };
        (error == 0).then_some(value)
    };
    let state = StillState {
        clip,
        source,
        info,
        step,
        limits: still::limits(number(c"limit"), number(c"edge"), number(c"tolerance")),
        masks: Mutex::new(Vec::new()),
    };
    unsafe {
        create(
            input,
            output,
            c"SVStill".as_ptr(),
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

impl StillState {
    #[allow(clippy::cast_precision_loss)]
    fn position(&self, n: i32) -> (i32, f32) {
        let at = i64::from(n) * self.step.0;
        (
            (at / self.step.1) as i32,
            (at % self.step.1) as f32 / self.step.1 as f32,
        )
    }

    unsafe fn render(
        &self,
        n: i32,
        frame_ctx: vs::Raw,
        core: vs::Raw,
        vsapi: vs::ConstRaw,
    ) -> Option<vs::ConstRaw> {
        let api = unsafe { frame::PlaneApi::load(vsapi) }?;
        let get_frame = unsafe { vs::table_fn::<vs::GetFrameFilter>(vsapi, vs::GET_FRAME_FILTER) }?;
        let free_frame = unsafe { vs::table_fn::<vs::FreeFrame>(vsapi, vs::FREE_FRAME) };
        let width = usize::try_from(self.info.width).ok()?;
        let height = usize::try_from(self.info.height).ok()?;
        let (index, time) = self.position(n);
        let made = get_node(get_frame, self.clip, n, frame_ctx);
        if time == 0.0 || made.is_null() {
            return (!made.is_null()).then_some(made);
        }
        let first = get_node(get_frame, self.source, index, frame_ctx);
        let second = get_node(get_frame, self.source, index.saturating_add(1), frame_ctx);
        let output = unsafe { api.copy(made, core) };
        let (x_div, y_div) = video_format::chroma_divisors(&self.info);
        let done = (|| {
            let output = output?;
            let plane = |frame, plane, rows| {
                let (data, pitch, len) = unsafe { api.read_plane(frame, plane, rows) }?;
                Some((unsafe { std::slice::from_raw_parts(data, len) }, pitch))
            };
            let (luma_a, pitch) = plane(first, 0, height)?;
            let (luma_b, _) = plane(second, 0, height)?;
            let cached = self.masks.lock().ok().and_then(|masks| {
                masks
                    .iter()
                    .find(|(key, _)| *key == index)
                    .map(|(_, mask)| Arc::clone(mask))
            });
            let mask = cached.unwrap_or_else(|| {
                let mask = Arc::new(still::mask(
                    luma_a,
                    luma_b,
                    pitch,
                    width,
                    height,
                    self.limits,
                ));
                if let Ok(mut masks) = self.masks.lock() {
                    if masks.len() >= 8 {
                        masks.remove(0);
                    }
                    masks.push((index, Arc::clone(&mask)));
                }
                mask
            });
            for index in 0..3 {
                let (width, height) = if index == 0 {
                    (width, height)
                } else {
                    (width / x_div as usize, height / y_div as usize)
                };
                let (a, pitch) = plane(first, index, height)?;
                let (b, _) = plane(second, index, height)?;
                let (data, _, len) = unsafe { api.write_plane(output, index, height) }?;
                let target = unsafe { std::slice::from_raw_parts_mut(data, len) };
                still::apply(
                    target,
                    a,
                    b,
                    pitch,
                    width,
                    height,
                    &mask,
                    self.info.width as usize,
                    time,
                );
            }
            Some(output.cast_const())
        })();
        for frame in [made, first, second] {
            drop_frame(frame, free_frame);
        }
        if done.is_none()
            && let Some(output) = output
        {
            unsafe { api.free(output.cast_const()) };
        }
        done
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
    let state = unsafe { &*(*instance_data).cast::<StillState>() };
    if let Some(set_video_info) =
        unsafe { vs::table_fn::<vs::SetVideoInfo>(vsapi, vs::SET_VIDEO_INFO) }
    {
        unsafe { set_video_info(&raw const state.info, 1, node) };
    }
}

unsafe extern "system" fn free_filter(instance_data: vs::Raw, _: vs::Raw, vsapi: vs::ConstRaw) {
    let state = unsafe { Box::from_raw(instance_data.cast::<StillState>()) };
    unsafe { vs::free_nodes([state.clip, state.source], vsapi) };
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
    let state = unsafe { &*(*instance_data).cast::<StillState>() };
    match activation_reason {
        vs::AR_INITIAL => {
            if let Some(request) =
                unsafe { vs::table_fn::<vs::RequestFrameFilter>(vsapi, vs::REQUEST_FRAME_FILTER) }
            {
                let (index, time) = state.position(n);
                request_node(request, state.clip, n, frame_ctx);
                if time != 0.0 {
                    request_node(request, state.source, index, frame_ctx);
                    request_node(request, state.source, index.saturating_add(1), frame_ctx);
                }
            }
            std::ptr::null()
        }
        vs::AR_ALL_FRAMES_READY => {
            unsafe { state.render(n, frame_ctx, core, vsapi) }.unwrap_or(std::ptr::null())
        }
        _ => std::ptr::null(),
    }
}
