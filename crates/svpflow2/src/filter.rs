use std::ffi::c_char;
use std::slice;

use crate::{core, light, metadata, nvof, options, params, strings, video_format, vs};

type FilterData = core::FilterState;

struct CreateApi {
    create_filter: vs::CreateFilter,
    get_node: vs::PropGetNode,
    get_video_info: vs::GetVideoInfo,
}

pub(crate) unsafe extern "system" fn create_smooth_fps(
    input: vs::ConstRaw,
    output: vs::Raw,
    _: vs::Raw,
    core: vs::Raw,
    vsapi: vs::ConstRaw,
) -> isize {
    unsafe { create_filter(0, input, output, core, vsapi) }
}

pub(crate) unsafe extern "system" fn create_smooth_fps_nvof(
    input: vs::ConstRaw,
    output: vs::Raw,
    _: vs::Raw,
    core: vs::Raw,
    vsapi: vs::ConstRaw,
) -> isize {
    unsafe { create_filter(1, input, output, core, vsapi) }
}

pub(crate) unsafe extern "system" fn create_smooth_fps_rife(
    input: vs::ConstRaw,
    output: vs::Raw,
    _: vs::Raw,
    core: vs::Raw,
    vsapi: vs::ConstRaw,
) -> isize {
    unsafe { create_filter(2, input, output, core, vsapi) }
}

unsafe fn create_filter(
    mode: i32,
    input: vs::ConstRaw,
    output: vs::Raw,
    core: vs::Raw,
    vsapi: vs::ConstRaw,
) -> isize {
    let Some(api) = (unsafe { load_create_api(output, vsapi) }) else {
        return 0;
    };
    let Some(mut options) = (unsafe { parse_params(mode, input, output, vsapi) }) else {
        return 0;
    };
    options.apply_core_threads(unsafe { core_threads(core, vsapi) });
    if mode != 0 && !unsafe { validate_options(&options, mode, output, vsapi) } {
        return 0;
    }

    let Some((source, video_info)) = (unsafe { source_info(input, output, vsapi, &api) }) else {
        return 0;
    };
    if mode != 0
        && !unsafe { validate_timing_options(&options, mode, source, &video_info, output, vsapi) }
    {
        return 0;
    }
    options.normalize_scene_mode(mode, &video_info);
    options.apply_source_depth(video_format::source_depth(&video_info));
    let mut data = unsafe { collect_state(mode, input, vsapi, &api, source, video_info, options) };
    if let metadata::VectorRecord::Ready(vectors) = data.vector_data() {
        data.options.scale_scene_limits(vectors.block);
    }
    let source_8bit_scale = data.mask_area_uses_source_8bit_scale();
    data.options.apply_mask_area_scale(source_8bit_scale);
    data.options.apply_debug_mask_scale();
    data.options.apply_mask_cover_algo(mode);
    if !unsafe { validate_nvof_runtime(mode, &data, output, vsapi) } {
        unsafe { data.free(vsapi) };
        return 0;
    }
    if !unsafe { validate_vec_src(mode, &data, output, vsapi, &api) } {
        unsafe { data.free(vsapi) };
        return 0;
    }
    if !unsafe { init_nvof_runtime(mode, &mut data, output, vsapi, &api) } {
        unsafe { data.free(vsapi) };
        return 0;
    }
    if !unsafe { validate_vectors(mode, &data, output, vsapi) } {
        unsafe { data.free(vsapi) };
        return 0;
    }
    if mode == 0 && !unsafe { validate_mode0_options(&data, output, vsapi) } {
        unsafe { data.free(vsapi) };
        return 0;
    }
    unsafe { apply_vector_defaults(mode, &mut data) };
    if !unsafe { validate_super(mode, &data, output, vsapi) } {
        unsafe { data.free(vsapi) };
        return 0;
    }
    unsafe { apply_request_flags(mode, &mut data) };
    if !unsafe { validate_cubic(&data, output, vsapi) } {
        unsafe { data.free(vsapi) };
        return 0;
    }
    if !unsafe { validate_overlap(&data, output, vsapi) } {
        unsafe { data.free(vsapi) };
        return 0;
    }
    if !unsafe { validate_source_format(&data, output, vsapi) } {
        unsafe { data.free(vsapi) };
        return 0;
    }
    if !unsafe { validate_gpu_runtime(&data, output, vsapi) } {
        unsafe { data.free(vsapi) };
        return 0;
    }
    let data = Box::into_raw(Box::new(data)).cast();

    unsafe {
        (api.create_filter)(
            input,
            output,
            strings::FILTER_NAME.as_ptr().cast(),
            init_filter,
            get_frame,
            free_filter,
            100,
            0,
            data,
            core,
        );
    }
    0
}

unsafe fn validate_options(
    options: &options::Options,
    mode: i32,
    output: vs::Raw,
    vsapi: vs::ConstRaw,
) -> bool {
    match options.validate(mode) {
        Ok(()) => true,
        Err(options::ValidateError::Algo) => {
            unsafe { set_error(output, vsapi, strings::ERR_ALGO.as_ptr().cast()) };
            false
        }
        Err(options::ValidateError::SceneMode) => {
            unsafe { set_error(output, vsapi, strings::ERR_SCENE_MODE.as_ptr().cast()) };
            false
        }
        Err(options::ValidateError::SceneModeRate) => {
            unsafe { set_error(output, vsapi, strings::ERR_SCENE_MODE_RATE.as_ptr().cast()) };
            false
        }
    }
}

unsafe fn validate_timing_options(
    options: &options::Options,
    mode: i32,
    source: vs::Raw,
    video_info: &vs::VideoInfo,
    output: vs::Raw,
    vsapi: vs::ConstRaw,
) -> bool {
    if let Err(options::ValidateError::SceneModeRate) = options.validate_timing(mode, video_info) {
        unsafe { set_error(output, vsapi, strings::ERR_SCENE_MODE_RATE.as_ptr().cast()) };
        unsafe { vs::free_nodes([source], vsapi) };
        return false;
    }
    true
}

unsafe fn validate_mode0_options(data: &FilterData, output: vs::Raw, vsapi: vs::ConstRaw) -> bool {
    if !unsafe { validate_options(&data.options, 0, output, vsapi) } {
        return false;
    }
    match data.options.validate_timing(0, &data.video_info) {
        Err(options::ValidateError::SceneModeRate) => {
            unsafe { set_error(output, vsapi, strings::ERR_SCENE_MODE_RATE.as_ptr().cast()) };
            false
        }
        Ok(()) | Err(options::ValidateError::Algo | options::ValidateError::SceneMode) => true,
    }
}

unsafe fn load_create_api(output: vs::Raw, vsapi: vs::ConstRaw) -> Option<CreateApi> {
    let create_filter = unsafe { vs::table_fn::<vs::CreateFilter>(vsapi, vs::CREATE_FILTER) };
    let get_node = unsafe { vs::table_fn::<vs::PropGetNode>(vsapi, vs::PROP_GET_NODE) };
    let get_video_info = unsafe { vs::table_fn::<vs::GetVideoInfo>(vsapi, vs::GET_VIDEO_INFO) };
    if let (Some(create_filter), Some(get_node), Some(get_video_info)) =
        (create_filter, get_node, get_video_info)
    {
        Some(CreateApi {
            create_filter,
            get_node,
            get_video_info,
        })
    } else {
        unsafe { set_error(output, vsapi, strings::ERR_VSAPI.as_ptr().cast()) };
        None
    }
}

unsafe fn core_threads(core: vs::Raw, vsapi: vs::ConstRaw) -> i32 {
    let Some(get_core_info) =
        (unsafe { vs::table_fn::<vs::GetCoreInfo>(vsapi, vs::GET_CORE_INFO) })
    else {
        return 0;
    };
    let info = unsafe { get_core_info(core) };
    if info.is_null() {
        0
    } else {
        unsafe { (*info).num_threads }
    }
}

unsafe fn source_info(
    input: vs::ConstRaw,
    output: vs::Raw,
    vsapi: vs::ConstRaw,
    api: &CreateApi,
) -> Option<(vs::Raw, vs::VideoInfo)> {
    let mut error = 0;
    let source = unsafe { (api.get_node)(input, strings::CLIP.as_ptr().cast(), 0, &raw mut error) };
    let mut video_info = if source.is_null() {
        vs::VideoInfo::empty()
    } else {
        let source_info = unsafe { (api.get_video_info)(source) };
        if source_info.is_null() {
            vs::VideoInfo::empty()
        } else {
            unsafe { *source_info }
        }
    };
    if video_info.fps_den == 0
        && !unsafe { fill_fps(input, output, vsapi, source, &mut video_info) }
    {
        return None;
    }
    Some((source, video_info))
}

unsafe fn fill_fps(
    input: vs::ConstRaw,
    output: vs::Raw,
    vsapi: vs::ConstRaw,
    source: vs::Raw,
    video_info: &mut vs::VideoInfo,
) -> bool {
    let Some(get_float) = (unsafe { vs::table_fn::<vs::PropGetFloat>(vsapi, vs::PROP_GET_FLOAT) })
    else {
        unsafe { set_error(output, vsapi, strings::ERR_VSAPI.as_ptr().cast()) };
        return false;
    };
    let fps = unsafe { get_float(input, strings::FPS.as_ptr().cast(), 0, std::ptr::null_mut()) };
    if fps < 0.1 {
        unsafe { set_error(output, vsapi, strings::ERR_FPS.as_ptr().cast()) };
        unsafe { vs::free_nodes([source], vsapi) };
        return false;
    }
    video_info.fps_num = fps_to_num(fps);
    video_info.fps_den = 1000;
    true
}

unsafe fn collect_state(
    mode: i32,
    input: vs::ConstRaw,
    vsapi: vs::ConstRaw,
    api: &CreateApi,
    source: vs::Raw,
    video_info: vs::VideoInfo,
    options: options::Options,
) -> FilterData {
    let get_int = unsafe { vs::table_fn::<vs::PropGetInt>(vsapi, vs::PROP_GET_INT) };
    let vdata = get_int_prop(get_int, input, strings::VDATA.as_ptr().cast());
    let mut clips = core::Clips::empty();
    clips.source = source;
    let sdata = unsafe { collect_mode_clips(mode, input, api, get_int, vdata, &mut clips) };
    let mut src_err = 0;
    clips.src = unsafe { (api.get_node)(input, strings::SRC.as_ptr().cast(), 0, &raw mut src_err) };
    if clips.src.is_null() {
        clips.src = unsafe {
            (api.get_node)(
                input,
                strings::CLIP.as_ptr().cast(),
                0,
                std::ptr::null_mut(),
            )
        };
    }
    let super_info = unsafe { clip_info(clips.super_clip, api) };
    let source_8bit_mode = source_8bit_mode(mode, &clips);
    let generated_vdata = generated_vector_data(
        mode,
        vdata,
        source_8bit_mode,
        &video_info,
        options.request_scene_mode(mode),
    );
    let render_mode = if options.disables_render_for_identity_rate(&video_info) {
        0
    } else {
        unsafe { render_mode(&options, vdata, generated_vdata) }
    };

    let gpu = if mode == 0 && render_mode == 2 && !options.cpu_render() {
        crate::gpu::GpuContext::new(
            i32::try_from(options.gpu_id()).unwrap_or(0),
            options.gpu_qn(),
        )
    } else {
        None
    };
    if let Some(g) = gpu.as_ref() {
        eprintln!("open-svpflow: GPU renderer active on {}", g.device_name());
    }
    FilterData {
        mode: core::Mode::from_raw(mode),
        clips,
        sdata,
        vdata,
        generated_vdata,
        source_8bit_mode,
        render_mode,
        request_super: false,
        options,
        light: light::LightState::new(),
        video_info,
        super_info,
        gpu,
        nvof: None,
        prep_cache: std::sync::Mutex::new(Vec::new()),
        decode_cache: std::sync::Mutex::new(Vec::new()),
        expand_cache: std::sync::Mutex::new(Vec::new()),
    }
}

fn generated_vector_data(
    mode: i32,
    vdata: i64,
    source_8bit_mode: bool,
    source: &vs::VideoInfo,
    scene_mode: i64,
) -> Option<metadata::VectorData> {
    if vdata != 0 {
        match unsafe { metadata::vector_data(vdata) } {
            metadata::VectorRecord::Ready(vectors) => Some(vectors),
            metadata::VectorRecord::Missing | metadata::VectorRecord::Invalid => None,
        }
    } else if mode == 1 || source_8bit_mode {
        Some(metadata::VectorData::generated_source_8bit(
            source.width,
            source.height,
            scene_mode,
        ))
    } else if mode == 2 {
        Some(metadata::VectorData::generated_rife(
            source.width,
            source.height,
        ))
    } else {
        None
    }
}

unsafe fn clip_info(node: vs::Raw, api: &CreateApi) -> Option<vs::VideoInfo> {
    if node.is_null() {
        return None;
    }
    let info = unsafe { (api.get_video_info)(node) };
    if info.is_null() {
        None
    } else {
        Some(unsafe { *info })
    }
}

unsafe fn collect_mode_clips(
    mode: i32,
    input: vs::ConstRaw,
    api: &CreateApi,
    get_int: Option<vs::PropGetInt>,
    vdata: i64,
    clips: &mut core::Clips,
) -> i64 {
    if mode == 0 {
        clips.super_clip = unsafe {
            (api.get_node)(
                input,
                strings::SUPER.as_ptr().cast(),
                0,
                std::ptr::null_mut(),
            )
        };
        clips.vectors = unsafe {
            (api.get_node)(
                input,
                strings::VECTORS.as_ptr().cast(),
                0,
                std::ptr::null_mut(),
            )
        };
        get_int_prop(get_int, input, strings::SDATA.as_ptr().cast())
    } else {
        let mut error = 0;
        let vec_src =
            unsafe { (api.get_node)(input, strings::VEC_SRC.as_ptr().cast(), 0, &raw mut error) };
        let vec_src = if mode == 1 && vec_src.is_null() {
            unsafe {
                (api.get_node)(
                    input,
                    strings::CLIP.as_ptr().cast(),
                    0,
                    std::ptr::null_mut(),
                )
            }
        } else {
            vec_src
        };
        if mode == 2 && vdata != 0 {
            clips.vectors = vec_src;
        } else {
            clips.vec_src = vec_src;
        }
        if mode == 2 {
            clips.rife_out = unsafe {
                (api.get_node)(
                    input,
                    strings::RIFE_OUT.as_ptr().cast(),
                    0,
                    std::ptr::null_mut(),
                )
            };
        }
        0
    }
}

unsafe fn validate_vec_src(
    mode: i32,
    data: &FilterData,
    output: vs::Raw,
    vsapi: vs::ConstRaw,
    api: &CreateApi,
) -> bool {
    if mode == 0 {
        return true;
    }
    if mode == 2 && !data.clips.vectors.is_null() {
        return true;
    }
    if data.clips.vec_src.is_null() {
        if mode == 2 {
            return true;
        }
        if video_format::needs_8bit_vec_src(&data.video_info) {
            unsafe { set_error(output, vsapi, strings::ERR_VEC_SRC_REQUIRED.as_ptr().cast()) };
            return false;
        }
        return true;
    }
    let info = unsafe { (api.get_video_info)(data.clips.vec_src) };
    if info.is_null() {
        unsafe { set_error(output, vsapi, strings::ERR_VEC_SRC_FORMAT.as_ptr().cast()) };
        return false;
    }
    let vec_info = unsafe { *info };
    if !video_format::is_yuv420p8(&vec_info) || vec_info.num_frames != data.video_info.num_frames {
        unsafe { set_error(output, vsapi, strings::ERR_VEC_SRC_FORMAT.as_ptr().cast()) };
        return false;
    }
    if !valid_vec_src_ratio(data.video_info.width, vec_info.width) {
        unsafe { set_error(output, vsapi, strings::ERR_VEC_SRC_RATIO.as_ptr().cast()) };
        return false;
    }
    if (vec_info.width.saturating_add(3) / 4) < 40 || vec_info.height <= 127 {
        unsafe { set_error(output, vsapi, strings::ERR_VEC_SRC_BLOCKS.as_ptr().cast()) };
        return false;
    }
    true
}

fn source_8bit_mode(mode: i32, clips: &core::Clips) -> bool {
    mode == 1 || mode == 2 && !clips.vec_src.is_null()
}

unsafe fn validate_nvof_runtime(
    mode: i32,
    data: &FilterData,
    output: vs::Raw,
    vsapi: vs::ConstRaw,
) -> bool {
    if mode != 1 || !data.source_8bit_mode {
        return true;
    }
    if nvof::NvofContext::cuda_available() {
        true
    } else {
        unsafe { set_error(output, vsapi, strings::ERR_CUDA_UNAVAILABLE.as_ptr().cast()) };
        false
    }
}

unsafe fn init_nvof_runtime(
    mode: i32,
    data: &mut FilterData,
    output: vs::Raw,
    vsapi: vs::ConstRaw,
    api: &CreateApi,
) -> bool {
    if mode != 1 {
        return true;
    }
    let Some(info) = (unsafe { clip_info(data.clips.vec_src, api) }) else {
        unsafe { set_error(output, vsapi, strings::ERR_VEC_SRC_FORMAT.as_ptr().cast()) };
        return false;
    };
    let scale = if info.width > 0 {
        data.video_info.width / info.width
    } else {
        1
    };
    match nvof::NvofContext::new(
        info.width,
        info.height,
        scale.max(1),
        data.options.nvof_quality(),
        data.options.nvof_gpu_id(),
    ) {
        Ok(context) => {
            data.nvof = Some(context);
            true
        }
        Err(nvof::InitError::CudaUnavailable) => {
            unsafe { set_error(output, vsapi, strings::ERR_CUDA_UNAVAILABLE.as_ptr().cast()) };
            false
        }
        Err(nvof::InitError::Failed(code)) => {
            unsafe { set_error_code(output, vsapi, strings::ERR_NVOF_INIT, code) };
            false
        }
    }
}

unsafe fn validate_gpu_runtime(data: &FilterData, output: vs::Raw, vsapi: vs::ConstRaw) -> bool {
    if data.render_mode != 2 {
        return true;
    }
    let code = if data.options.gpu_api() == 2 { -1 } else { 1 };
    if code != -1 {
        return true;
    }
    if data.options.fallback_enabled() && !data.source_8bit_mode {
        unsafe {
            set_error_code_suffix(
                output,
                vsapi,
                strings::ERR_GPU_INIT,
                code,
                strings::ERR_GPU_FALLBACK,
            );
        };
    } else {
        unsafe { set_error_code(output, vsapi, strings::ERR_GPU_INIT, code) };
    }
    false
}

unsafe fn validate_vectors(
    mode: i32,
    data: &FilterData,
    output: vs::Raw,
    vsapi: vs::ConstRaw,
) -> bool {
    if mode == 2 && data.vdata != 0 {
        return match data.vector_data() {
            metadata::VectorRecord::Invalid => {
                unsafe {
                    set_error(
                        output,
                        vsapi,
                        strings::ERR_VECTORS_INVALID_2.as_ptr().cast(),
                    );
                };
                false
            }
            metadata::VectorRecord::Missing | metadata::VectorRecord::Ready(_) => true,
        };
    }
    if mode != 0 {
        return true;
    }
    match data.vector_data() {
        metadata::VectorRecord::Missing => {
            unsafe {
                set_error(
                    output,
                    vsapi,
                    strings::ERR_VECTORS_INVALID_1.as_ptr().cast(),
                );
            };
            false
        }
        metadata::VectorRecord::Invalid => {
            unsafe {
                set_error(
                    output,
                    vsapi,
                    strings::ERR_VECTORS_INVALID_2.as_ptr().cast(),
                );
            };
            false
        }
        metadata::VectorRecord::Ready(vectors) => {
            if vectors.shape.width == data.video_info.width
                && vectors.shape.height == data.video_info.height
            {
                true
            } else {
                unsafe { set_error(output, vsapi, strings::ERR_VECTORS_SIZE.as_ptr().cast()) };
                false
            }
        }
    }
}

unsafe fn apply_vector_defaults(mode: i32, data: &mut FilterData) {
    match data.vector_data() {
        metadata::VectorRecord::Ready(vectors) => {
            data.options.apply_cubic_default(!vectors.rejects_cubic());
        }
        metadata::VectorRecord::Missing if mode != 0 => data.options.apply_cubic_default(true),
        metadata::VectorRecord::Missing | metadata::VectorRecord::Invalid => {}
    }
}

unsafe fn validate_super(
    mode: i32,
    data: &FilterData,
    output: vs::Raw,
    vsapi: vs::ConstRaw,
) -> bool {
    if mode != 0 {
        return true;
    }
    let metadata::VectorRecord::Ready(vectors) = data.vector_data() else {
        return true;
    };
    let super_data = metadata::super_data(data.sdata);
    if !super_data.matches_vectors(&vectors) {
        unsafe { set_error(output, vsapi, strings::ERR_SUPER_PARAMS.as_ptr().cast()) };
        return false;
    }
    if vectors.delta == 1 {
        true
    } else {
        unsafe { set_error(output, vsapi, strings::ERR_VECTORS_DELTA.as_ptr().cast()) };
        false
    }
}

unsafe fn apply_request_flags(mode: i32, data: &mut FilterData) {
    if mode != 0 {
        return;
    }
    let metadata::VectorRecord::Ready(vectors) = data.vector_data() else {
        return;
    };
    data.request_super = matches!(data.render_mode, 1 | 2) && !vectors.marker_is_one();
}

unsafe fn validate_cubic(data: &FilterData, output: vs::Raw, vsapi: vs::ConstRaw) -> bool {
    if !data.options.cubic_positive() {
        return true;
    }
    let metadata::VectorRecord::Ready(vectors) = data.vector_data() else {
        return true;
    };
    if vectors.rejects_cubic() {
        unsafe { set_error(output, vsapi, strings::ERR_CUBIC.as_ptr().cast()) };
        false
    } else {
        true
    }
}

unsafe fn validate_overlap(data: &FilterData, output: vs::Raw, vsapi: vs::ConstRaw) -> bool {
    if data.render_mode != 1 {
        return true;
    }
    let metadata::VectorRecord::Ready(vectors) = data.vector_data() else {
        return true;
    };
    if vectors.has_odd_overlap() {
        unsafe { set_error(output, vsapi, strings::ERR_OVERLAP_CPU.as_ptr().cast()) };
        false
    } else {
        true
    }
}

unsafe fn validate_source_format(data: &FilterData, output: vs::Raw, vsapi: vs::ConstRaw) -> bool {
    let requires_cpu = unsafe { requires_cpu_source(data) };
    if requires_cpu && !video_format::is_cpu_source(&data.video_info) {
        unsafe { set_error(output, vsapi, strings::ERR_SOURCE_CPU.as_ptr().cast()) };
        return false;
    }
    if !requires_cpu && !video_format::is_yuv420_source(&data.video_info) {
        unsafe { set_error(output, vsapi, strings::ERR_SOURCE_YUV.as_ptr().cast()) };
        return false;
    }
    true
}

unsafe fn requires_cpu_source(data: &FilterData) -> bool {
    data.requires_cpu_source()
}

unsafe fn render_mode(
    options: &options::Options,
    vdata: i64,
    generated_vdata: Option<metadata::VectorData>,
) -> i32 {
    if options.cpu_render() {
        return 0;
    }
    let record = if vdata != 0 {
        generated_vdata.map_or(
            metadata::VectorRecord::Invalid,
            metadata::VectorRecord::Ready,
        )
    } else {
        generated_vdata.map_or(
            metadata::VectorRecord::Missing,
            metadata::VectorRecord::Ready,
        )
    };
    let metadata::VectorRecord::Ready(vectors) = record else {
        return 2;
    };
    if vectors.render_mode_one() { 1 } else { 2 }
}

fn valid_vec_src_ratio(source_width: i32, vec_width: i32) -> bool {
    if source_width <= 0 || vec_width <= 0 {
        return false;
    }
    let ratio = f64::from(source_width) / f64::from(vec_width);
    let lower = ratio.floor();
    let rounded = if ratio - lower < 0.45 {
        lower
    } else {
        let upper = ratio.ceil();
        if upper - ratio < 0.45 {
            upper
        } else {
            return false;
        }
    };
    [1.0, 2.0, 4.0, 6.0, 8.0]
        .into_iter()
        .any(|value| (rounded - value).abs() < f64::EPSILON)
}

fn get_int_prop(get_int: Option<vs::PropGetInt>, input: vs::ConstRaw, key: *const c_char) -> i64 {
    let mut error = 0;
    get_int.map_or(0, |get_int| unsafe {
        get_int(input, key, 0, &raw mut error)
    })
}

unsafe extern "system" fn init_filter(
    _: vs::ConstRaw,
    _: vs::Raw,
    instance_data: *mut vs::Raw,
    node: vs::Raw,
    _: vs::Raw,
    vsapi: vs::ConstRaw,
) {
    if instance_data.is_null() {
        return;
    }
    let data = unsafe { *instance_data }.cast::<FilterData>();
    if data.is_null() {
        return;
    }
    if let Some(set_video_info) =
        unsafe { vs::table_fn::<vs::SetVideoInfo>(vsapi, vs::SET_VIDEO_INFO) }
    {
        let info = unsafe { (*data).output_info() };
        unsafe { set_video_info(&raw const info, 1, node) };
    }
}

unsafe extern "system" fn free_filter(instance_data: vs::Raw, _: vs::Raw, vsapi: vs::ConstRaw) {
    if !instance_data.is_null() {
        unsafe {
            let data = Box::from_raw(instance_data.cast::<FilterData>());
            data.free(vsapi);
        }
    }
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
    if instance_data.is_null() {
        return std::ptr::null();
    }
    let data = unsafe { *instance_data }.cast::<FilterData>();
    if data.is_null() {
        return std::ptr::null();
    }
    let state = unsafe { &*data };
    match activation_reason {
        0 => {
            unsafe { state.request_frame(n, frame_ctx, vsapi) };
            std::ptr::null()
        }
        2 => unsafe { state.get_frame(n, frame_ctx, core, vsapi) },
        _ => std::ptr::null(),
    }
}

unsafe fn set_error(output: vs::Raw, vsapi: vs::ConstRaw, message: *const c_char) {
    if let Some(set_error) = unsafe { vs::table_fn::<vs::SetError>(vsapi, vs::SET_ERROR) } {
        unsafe { set_error(output, message) };
    }
}

unsafe fn set_error_code(output: vs::Raw, vsapi: vs::ConstRaw, prefix: &[u8], code: i32) {
    unsafe { set_error_code_suffix(output, vsapi, prefix, code, b"\0") };
}

unsafe fn set_error_code_suffix(
    output: vs::Raw,
    vsapi: vs::ConstRaw,
    prefix: &[u8],
    code: i32,
    suffix: &[u8],
) {
    let mut message = Vec::with_capacity(prefix.len() + 12);
    message.extend_from_slice(&prefix[..prefix.len().saturating_sub(1)]);
    message.extend_from_slice(code.to_string().as_bytes());
    message.extend_from_slice(&suffix[..suffix.len().saturating_sub(1)]);
    message.push(0);
    unsafe { set_error(output, vsapi, message.as_ptr().cast()) };
}

unsafe fn parse_params(
    mode: i32,
    input: vs::ConstRaw,
    output: vs::Raw,
    vsapi: vs::ConstRaw,
) -> Option<options::Options> {
    let Some(get_data) = (unsafe { vs::table_fn::<vs::PropGetData>(vsapi, vs::PROP_GET_DATA) })
    else {
        return Some(options::Options::for_mode(mode));
    };
    let Some(get_data_size) =
        (unsafe { vs::table_fn::<vs::PropGetDataSize>(vsapi, vs::PROP_GET_DATA_SIZE) })
    else {
        return Some(options::Options::for_mode(mode));
    };
    let data = unsafe { get_data(input, strings::OPT.as_ptr().cast(), 0, std::ptr::null_mut()) };
    let size =
        unsafe { get_data_size(input, strings::OPT.as_ptr().cast(), 0, std::ptr::null_mut()) };
    if data.is_null() || size <= 0 {
        return Some(options::Options::for_mode(mode));
    }
    let Ok(size) = usize::try_from(size) else {
        return Some(options::Options::for_mode(mode));
    };
    let bytes = unsafe { slice::from_raw_parts(data.cast::<u8>(), size) };
    match params::parse(bytes) {
        Ok(value) => Some(options::Options::from_value(&value, mode)),
        Err(error) => {
            let mut message = Vec::with_capacity(strings::ERR_PARAMS.len() + error.len());
            message.extend_from_slice(&strings::ERR_PARAMS[..strings::ERR_PARAMS.len() - 1]);
            message.extend_from_slice(error.as_bytes());
            message.push(0);
            unsafe { set_error(output, vsapi, message.as_ptr().cast()) };
            None
        }
    }
}

#[allow(clippy::cast_possible_truncation)]
fn fps_to_num(fps: f64) -> i64 {
    (fps * 1000.0) as i64
}
