use crate::analyse::{self, AnalyseParams, RawPlane, SuperFrameView, SuperParams};
use crate::{params, video_format, vs};

struct AnalyseState {
    super_node: vs::Raw,
    src_node: vs::Raw,
    vi: vs::VideoInfo,
    params: AnalyseParams,
    super_height: usize,
    _header: Box<[i32]>,
    gray_format: vs::ConstRaw,
}

pub(crate) unsafe extern "system" fn create_analyse(
    input: vs::ConstRaw,
    output: vs::Raw,
    _user: vs::Raw,
    core: vs::Raw,
    vsapi: vs::ConstRaw,
) -> isize {
    if let Err(message) = unsafe { create_analyse_inner(input, output, core, vsapi) } {
        let message = std::ffi::CString::new(message).unwrap_or_default();
        unsafe { vs::set_error(output, vsapi, message.as_ptr()) };
    }
    0
}

unsafe fn create_analyse_inner(
    input: vs::ConstRaw,
    output: vs::Raw,
    core: vs::Raw,
    vsapi: vs::ConstRaw,
) -> Result<(), String> {
    const INVALID_API: &str = "SVAnalyse: invalid VSAPI";
    let get_node =
        unsafe { vs::table_fn::<vs::PropGetNode>(vsapi, vs::PROP_GET_NODE) }.ok_or(INVALID_API)?;
    let get_vi = unsafe { vs::table_fn::<vs::GetVideoInfo>(vsapi, vs::GET_VIDEO_INFO) }
        .ok_or(INVALID_API)?;
    let create_filter =
        unsafe { vs::table_fn::<vs::CreateFilter>(vsapi, vs::CREATE_FILTER) }.ok_or(INVALID_API)?;
    let prop_set_int =
        unsafe { vs::table_fn::<vs::PropSetInt>(vsapi, vs::PROP_SET_INT) }.ok_or(INVALID_API)?;
    let get_int =
        unsafe { vs::table_fn::<vs::PropGetInt>(vsapi, vs::PROP_GET_INT) }.ok_or(INVALID_API)?;

    let mut err = 0;
    let super_node = unsafe { get_node(input, c"clip".as_ptr(), 0, &raw mut err) };
    if super_node.is_null() {
        return Err("SVAnalyse: clip (super) required".into());
    }
    let src_node = unsafe { get_node(input, c"src".as_ptr(), 0, &raw mut err) };
    let free_nodes = || unsafe {
        vs::free_node(super_node, vsapi);
        if !src_node.is_null() {
            vs::free_node(src_node, vsapi);
        }
    };
    if src_node.is_null() {
        free_nodes();
        return Err("SVAnalyse: src required".into());
    }
    let result = (|| {
        let sdata = unsafe { get_int(input, c"sdata".as_ptr(), 0, &raw mut err) };
        if err != 0 {
            return Err("SVAnalyse: sdata required".to_string());
        }
        let src_vi = unsafe { *get_vi(src_node) };
        let super_vi = unsafe { *get_vi(super_node) };
        if !video_format::is_supported(&src_vi) {
            return Err("SVAnalyse: src must be YUV420P8 or YUV444P8".into());
        }
        let (x_div, y_div) = video_format::chroma_divisors(&src_vi);
        let chroma_shift = (x_div.trailing_zeros(), y_div.trailing_zeros());

        let options = unsafe { read_opt_bytes(input, vsapi) };
        let options = options
            .as_deref()
            .and_then(|bytes| params::parse(bytes).ok());
        let super_params = SuperParams::unpack(sdata)?;
        let params = AnalyseParams::new(options.as_ref(), super_params, chroma_shift)?;

        let full = super_params.has_finest_level();
        let expected_width = if full {
            super_params.width
        } else {
            analyse::plane_size(super_params.width, 1)
        };
        let super_height = crate::super_opts::super_plane_height(
            super_params.width,
            super_params.height,
            super_params.pel,
            super_params.levels,
            full,
        );
        if super_vi.width != expected_width
            || super_vi.height != super_height
            || super_vi.format != src_vi.format
        {
            return Err("SVAnalyse: invalid super clip layout".into());
        }
        let gray = unsafe { get_gray8_format(core, vsapi, super_vi.format) };
        if gray.is_null() {
            return Err("SVAnalyse: cannot get Gray8 format".into());
        }

        let header: Box<[i32]> = Box::new(analyse::analysis_header(&params));
        unsafe { prop_set_int(output, c"data".as_ptr(), header.as_ptr() as i64, 0) };

        let mut num_frames = src_vi.num_frames;
        if super_vi.num_frames > 0 && super_vi.num_frames < num_frames {
            num_frames = super_vi.num_frames;
        }
        let vi = vs::VideoInfo {
            format: gray,
            fps_num: src_vi.fps_num,
            fps_den: src_vi.fps_den,
            width: analyse::frame_len(&params) as i32,
            height: 1,
            num_frames,
            flags: 0,
        };
        Ok(AnalyseState {
            super_node,
            src_node,
            vi,
            params,
            super_height: super_height as usize,
            _header: header,
            gray_format: gray,
        })
    })();
    let state = match result {
        Ok(state) => state,
        Err(message) => {
            free_nodes();
            return Err(message);
        }
    };
    let state = Box::into_raw(Box::new(state));
    unsafe {
        create_filter(
            input,
            output,
            c"SVAnalyse".as_ptr(),
            init_analyse,
            get_frame_analyse,
            free_analyse,
            vs::FM_PARALLEL,
            0,
            state.cast(),
            core,
        );
    }
    Ok(())
}

unsafe fn read_opt_bytes(input: vs::ConstRaw, vsapi: vs::ConstRaw) -> Option<Vec<u8>> {
    let get_data = unsafe { vs::table_fn::<vs::PropGetData>(vsapi, vs::PROP_GET_DATA)? };
    let get_size = unsafe { vs::table_fn::<vs::PropGetDataSize>(vsapi, vs::PROP_GET_DATA_SIZE)? };
    let mut err = 0;
    let ptr = unsafe { get_data(input, c"opt".as_ptr(), 0, &raw mut err) };
    if ptr.is_null() || err != 0 {
        return None;
    }
    let size = unsafe { get_size(input, c"opt".as_ptr(), 0, &raw mut err) };
    if size <= 0 {
        return None;
    }
    Some(unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), size as usize) }.to_vec())
}

unsafe fn get_gray8_format(
    core: vs::Raw,
    vsapi: vs::ConstRaw,
    fallback: vs::ConstRaw,
) -> vs::ConstRaw {
    type GetFormatPreset = unsafe extern "system" fn(i32, vs::Raw) -> vs::ConstRaw;
    const GET_FORMAT_PRESET: usize = 176;
    const PF_GRAY8: i32 = 1_000_010;
    if let Some(preset) = unsafe { vs::table_fn::<GetFormatPreset>(vsapi, GET_FORMAT_PRESET) } {
        let format = unsafe { preset(PF_GRAY8, core) };
        if !format.is_null() {
            return format;
        }
    }
    fallback
}

unsafe extern "system" fn init_analyse(
    _in: vs::ConstRaw,
    _out: vs::Raw,
    instance_data: *mut vs::Raw,
    node: vs::Raw,
    _core: vs::Raw,
    vsapi: vs::ConstRaw,
) {
    let state = unsafe { &*(*instance_data).cast::<AnalyseState>() };
    if let Some(set_vi) = unsafe { vs::table_fn::<vs::SetVideoInfo>(vsapi, vs::SET_VIDEO_INFO) } {
        unsafe { set_vi(&raw const state.vi, 1, node) };
    }
}

unsafe extern "system" fn free_analyse(
    instance_data: vs::Raw,
    _core: vs::Raw,
    vsapi: vs::ConstRaw,
) {
    let state = unsafe { Box::from_raw(instance_data.cast::<AnalyseState>()) };
    unsafe { vs::free_node(state.super_node, vsapi) };
    unsafe { vs::free_node(state.src_node, vsapi) };
}

struct FrameApi {
    get_frame: vs::GetFrameFilter,
    get_stride: vs::GetStride,
    get_read: vs::GetReadPtr,
    free_frame: vs::FreeFrame,
}

impl FrameApi {
    unsafe fn load(vsapi: vs::ConstRaw) -> Option<Self> {
        Some(Self {
            get_frame: unsafe { vs::table_fn::<vs::GetFrameFilter>(vsapi, vs::GET_FRAME_FILTER)? },
            get_stride: unsafe { vs::table_fn::<vs::GetStride>(vsapi, vs::GET_STRIDE)? },
            get_read: unsafe { vs::table_fn::<vs::GetReadPtr>(vsapi, vs::GET_READ_PTR)? },
            free_frame: unsafe { vs::table_fn::<vs::FreeFrame>(vsapi, vs::FREE_FRAME)? },
        })
    }

    unsafe fn planes<'f>(
        &self,
        frame: vs::ConstRaw,
        rows: usize,
        y_shift: u32,
    ) -> [RawPlane<'f>; 3] {
        std::array::from_fn(|plane| {
            let pitch = unsafe { (self.get_stride)(frame, plane as i32) } as usize;
            let rows = if plane == 0 { rows } else { rows >> y_shift };
            let data = unsafe {
                std::slice::from_raw_parts((self.get_read)(frame, plane as i32), pitch * rows)
            };
            RawPlane { data, pitch }
        })
    }
}

unsafe extern "system" fn get_frame_analyse(
    n: i32,
    activation: i32,
    instance_data: *mut vs::Raw,
    _frame_data: *mut vs::Raw,
    frame_ctx: vs::Raw,
    core: vs::Raw,
    vsapi: vs::ConstRaw,
) -> vs::ConstRaw {
    let state = unsafe { &*(*instance_data).cast::<AnalyseState>() };
    let params = &state.params;
    let full = params.super_params.has_finest_level();
    let next = (n + params.delta).min(state.vi.num_frames - 1);

    if activation == vs::AR_INITIAL {
        if let Some(request) =
            unsafe { vs::table_fn::<vs::RequestFrameFilter>(vsapi, vs::REQUEST_FRAME_FILTER) }
        {
            for frame in [n, next] {
                unsafe { request(frame, state.super_node, frame_ctx) };
                if !full {
                    unsafe { request(frame, state.src_node, frame_ctx) };
                }
            }
        }
        return std::ptr::null();
    }
    if activation != vs::AR_ALL_FRAMES_READY {
        return std::ptr::null();
    }
    let Some(api) = (unsafe { FrameApi::load(vsapi) }) else {
        return std::ptr::null();
    };
    let Some(new_frame) =
        (unsafe { vs::table_fn::<vs::NewVideoFrame>(vsapi, vs::NEW_VIDEO_FRAME) })
    else {
        return std::ptr::null();
    };
    let Some(get_write) = (unsafe { vs::table_fn::<vs::GetWritePtr>(vsapi, vs::GET_WRITE_PTR) })
    else {
        return std::ptr::null();
    };

    let fetch = |node: vs::Raw, frame: i32| unsafe { (api.get_frame)(frame, node, frame_ctx) };
    let supers = [fetch(state.super_node, n), fetch(state.super_node, next)];
    let sources = if full {
        [std::ptr::null(); 2]
    } else {
        [fetch(state.src_node, n), fetch(state.src_node, next)]
    };
    let release = |frames: &[vs::ConstRaw]| {
        for &frame in frames.iter().filter(|f| !f.is_null()) {
            unsafe { (api.free_frame)(frame) };
        }
    };
    if supers.iter().any(|f| f.is_null()) || (!full && sources.iter().any(|f| f.is_null())) {
        release(&supers);
        release(&sources);
        return std::ptr::null();
    }

    let sp = params.super_params;
    let y_shift = params.chroma_shift.1;
    let view = |index: usize| SuperFrameView {
        planes: unsafe { api.planes(supers[index], state.super_height, y_shift) },
        source: (!full).then(|| unsafe { api.planes(sources[index], sp.height as usize, y_shift) }),
        width: sp.width,
        height: sp.height,
        pel: sp.pel,
        chroma_shift: params.chroma_shift,
    };
    let level0 = if full {
        state.super_node
    } else {
        state.src_node
    } as usize;
    let payload = analyse::analyse(
        params,
        &view(0),
        &view(1),
        Some([(level0, n), (level0, next)]),
    );
    release(&sources);

    let out = unsafe { new_frame(state.gray_format, payload.len() as i32, 1, supers[0], core) };
    if !out.is_null() {
        let dst = unsafe { get_write(out, 0) };
        if !dst.is_null() {
            unsafe { std::ptr::copy_nonoverlapping(payload.as_ptr(), dst, payload.len()) };
        }
    }
    release(&supers);
    out.cast_const()
}
