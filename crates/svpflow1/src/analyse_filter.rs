use crate::analyse_opts::AnalyseOpts;
use crate::analyse_search::{self, SuperPlanes};
use crate::{params, vs};

struct AnalyseState {
    super_node: vs::Raw,
    src_node: vs::Raw,
    vi: vs::VideoInfo,
    opts: AnalyseOpts,

    #[allow(dead_code)]
    vdata: Box<[i32]>,
    #[allow(dead_code)]
    vdata_handle: i64,
    gray_format: vs::ConstRaw,
    payload_len: i32,
}

pub(crate) unsafe extern "system" fn create_analyse(
    input: vs::ConstRaw,
    output: vs::Raw,
    _user: vs::Raw,
    core: vs::Raw,
    vsapi: vs::ConstRaw,
) -> isize {
    unsafe { create_analyse_inner(input, output, core, vsapi) }
}

unsafe fn create_analyse_inner(
    input: vs::ConstRaw,
    output: vs::Raw,
    core: vs::Raw,
    vsapi: vs::ConstRaw,
) -> isize {
    let Some(get_node) = (unsafe { vs::table_fn::<vs::PropGetNode>(vsapi, vs::PROP_GET_NODE) })
    else {
        unsafe { vs::set_error(output, vsapi, c"SVAnalyse: invalid VSAPI".as_ptr()) };
        return 0;
    };
    let Some(get_vi) = (unsafe { vs::table_fn::<vs::GetVideoInfo>(vsapi, vs::GET_VIDEO_INFO) })
    else {
        unsafe { vs::set_error(output, vsapi, c"SVAnalyse: invalid VSAPI".as_ptr()) };
        return 0;
    };
    let Some(create_filter) =
        (unsafe { vs::table_fn::<vs::CreateFilter>(vsapi, vs::CREATE_FILTER) })
    else {
        unsafe { vs::set_error(output, vsapi, c"SVAnalyse: invalid VSAPI".as_ptr()) };
        return 0;
    };
    let Some(prop_set_int) = (unsafe { vs::table_fn::<vs::PropSetInt>(vsapi, vs::PROP_SET_INT) })
    else {
        unsafe { vs::set_error(output, vsapi, c"SVAnalyse: invalid VSAPI".as_ptr()) };
        return 0;
    };
    let Some(get_int) = (unsafe { vs::table_fn::<vs::PropGetInt>(vsapi, vs::PROP_GET_INT) }) else {
        unsafe { vs::set_error(output, vsapi, c"SVAnalyse: invalid VSAPI".as_ptr()) };
        return 0;
    };

    let mut err = 0;
    let super_node = unsafe { get_node(input, c"clip".as_ptr(), 0, &raw mut err) };
    if super_node.is_null() {
        unsafe { vs::set_error(output, vsapi, c"SVAnalyse: clip (super) required".as_ptr()) };
        return 0;
    }
    let src_node = unsafe { get_node(input, c"src".as_ptr(), 0, &raw mut err) };
    if src_node.is_null() {
        unsafe { vs::free_node(super_node, vsapi) };
        unsafe { vs::set_error(output, vsapi, c"SVAnalyse: src required".as_ptr()) };
        return 0;
    }
    let sdata = unsafe { get_int(input, c"sdata".as_ptr(), 0, &raw mut err) };
    if err != 0 {
        unsafe { vs::free_node(super_node, vsapi) };
        unsafe { vs::free_node(src_node, vsapi) };
        unsafe { vs::set_error(output, vsapi, c"SVAnalyse: sdata required".as_ptr()) };
        return 0;
    }

    let src_vi = unsafe { *get_vi(src_node) };
    let super_vi = unsafe { *get_vi(super_node) };

    let opt_raw = unsafe { read_opt_bytes(input, vsapi) };
    let opt_val = opt_raw.as_ref().and_then(|b| params::parse(b).ok());
    let mut opts = match AnalyseOpts::from_opt(opt_val.as_ref(), sdata, None) {
        Ok(o) => o,
        Err(msg) => {
            unsafe { vs::free_node(super_node, vsapi) };
            unsafe { vs::free_node(src_node, vsapi) };
            let cmsg = std::ffi::CString::new(msg).unwrap_or_default();
            unsafe { vs::set_error(output, vsapi, cmsg.as_ptr()) };
            return 0;
        }
    };
    if src_vi.width > 0 {
        opts.width = src_vi.width;
    }
    if src_vi.height > 0 {
        opts.height = src_vi.height;
    }

    let gray = unsafe { get_gray8_format(core, vsapi, super_vi.format) };
    if gray.is_null() {
        unsafe { vs::free_node(super_node, vsapi) };
        unsafe { vs::free_node(src_node, vsapi) };
        unsafe {
            vs::set_error(
                output,
                vsapi,
                c"SVAnalyse: cannot get Gray8 format".as_ptr(),
            );
        };
        return 0;
    }

    let vdata_vec = analyse_search::pack_vdata_header(&opts);
    let vdata: Box<[i32]> = vdata_vec.into_boxed_slice();
    let vdata_handle = vdata.as_ptr() as usize as i64;
    unsafe { prop_set_int(output, c"data".as_ptr(), vdata_handle, 0) };

    let (bw, bh, ox, oy) = opts.output_block();
    let (gw, gh) = opts.grid(bw, bh, ox, oy);
    let count = (gw * gh) as usize;
    let n_reg = match opts.vectors {
        1 | 2 => 1,
        _ => 2,
    };
    let payload_len = (0x40 + n_reg * (4 + count * 8)) as i32;

    let mut out_vi = vs::VideoInfo {
        format: gray,
        fps_num: src_vi.fps_num,
        fps_den: src_vi.fps_den,
        width: payload_len,
        height: 1,
        num_frames: src_vi.num_frames,
        flags: 0,
    };

    if super_vi.num_frames > 0 && super_vi.num_frames < out_vi.num_frames {
        out_vi.num_frames = super_vi.num_frames;
    }

    let state = Box::into_raw(Box::new(AnalyseState {
        super_node,
        src_node,
        vi: out_vi,
        opts,
        vdata,
        vdata_handle,
        gray_format: gray,
        payload_len,
    }));

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
    0
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
    if let Some(gp) = unsafe { vs::table_fn::<GetFormatPreset>(vsapi, GET_FORMAT_PRESET) } {
        let fmt = unsafe { gp(PF_GRAY8, core) };
        if !fmt.is_null() {
            return fmt;
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

    if activation == vs::AR_INITIAL {
        if let Some(request) =
            unsafe { vs::table_fn::<vs::RequestFrameFilter>(vsapi, vs::REQUEST_FRAME_FILTER) }
        {
            let delta = state.opts.delta.max(1);
            unsafe { request(n, state.super_node, frame_ctx) };
            unsafe { request(n + delta, state.super_node, frame_ctx) };
        }
        return std::ptr::null();
    }
    if activation != vs::AR_ALL_FRAMES_READY {
        return std::ptr::null();
    }

    let Some(get_frame) =
        (unsafe { vs::table_fn::<vs::GetFrameFilter>(vsapi, vs::GET_FRAME_FILTER) })
    else {
        return std::ptr::null();
    };
    let Some(new_frame) =
        (unsafe { vs::table_fn::<vs::NewVideoFrame>(vsapi, vs::NEW_VIDEO_FRAME) })
    else {
        return std::ptr::null();
    };
    let Some(get_stride) = (unsafe { vs::table_fn::<vs::GetStride>(vsapi, vs::GET_STRIDE) }) else {
        return std::ptr::null();
    };
    let Some(get_read) = (unsafe { vs::table_fn::<vs::GetReadPtr>(vsapi, vs::GET_READ_PTR) })
    else {
        return std::ptr::null();
    };
    let Some(get_write) = (unsafe { vs::table_fn::<vs::GetWritePtr>(vsapi, vs::GET_WRITE_PTR) })
    else {
        return std::ptr::null();
    };
    let free_frame = unsafe { vs::table_fn::<vs::FreeFrame>(vsapi, vs::FREE_FRAME) };

    let delta = state.opts.delta.max(1);

    let n2 = (n + delta).min(state.vi.num_frames.saturating_sub(1));

    let f0 = unsafe { get_frame(n, state.super_node, frame_ctx) };
    let f1 = unsafe { get_frame(n2, state.super_node, frame_ctx) };
    if f0.is_null() || f1.is_null() {
        if let Some(ff) = free_frame {
            if !f0.is_null() {
                unsafe { ff(f0) };
            }
            if !f1.is_null() {
                unsafe { ff(f1) };
            }
        }
        return std::ptr::null();
    }

    let sh = unsafe { super_height(get_stride, f0) };
    let planes0 = unsafe { load_super_planes(get_stride, get_read, f0, &state.opts, sh) };
    let planes1 = unsafe { load_super_planes(get_stride, get_read, f1, &state.opts, sh) };

    let (bwd, fwd) = match (planes0.as_ref(), planes1.as_ref()) {
        (Some(p0), Some(p1)) => analyse_search::analyse_pair(p0, p1, &state.opts),
        _ => (None, None),
    };

    let payload = analyse_search::pack_vector_frame(&state.opts, bwd.as_deref(), fwd.as_deref());

    let out = unsafe { new_frame(state.gray_format, payload.len() as i32, 1, f0, core) };
    if out.is_null() {
        if let Some(ff) = free_frame {
            unsafe { ff(f0) };
            unsafe { ff(f1) };
        }
        return std::ptr::null();
    }

    let stride = unsafe { get_stride(out.cast_const(), 0) } as usize;
    let ptr = unsafe { get_write(out, 0) };
    if !ptr.is_null() && stride > 0 {
        let ncopy = payload.len().min(stride);
        unsafe {
            std::ptr::copy_nonoverlapping(payload.as_ptr(), ptr, ncopy);
            if stride > ncopy {
                std::ptr::write_bytes(ptr.add(ncopy), 0, stride - ncopy);
            }
        }
    }

    if let Some(ff) = free_frame {
        unsafe { ff(f0) };
        unsafe { ff(f1) };
    }
    let _ = state.vdata_handle;
    let _ = state.payload_len;
    out.cast_const()
}

unsafe fn super_height(get_stride: vs::GetStride, frame: vs::ConstRaw) -> usize {
    let _ = (get_stride, frame);
    0
}

unsafe fn load_super_planes<'a>(
    get_stride: vs::GetStride,
    get_read: vs::GetReadPtr,
    frame: vs::ConstRaw,
    opts: &AnalyseOpts,
    _sh_unused: usize,
) -> Option<SuperPlanes<'a>> {
    let y_stride = unsafe { get_stride(frame, 0) } as usize;
    let u_stride = unsafe { get_stride(frame, 1) } as usize;
    let v_stride = unsafe { get_stride(frame, 2) } as usize;
    let y_ptr = unsafe { get_read(frame, 0) };
    let u_ptr = unsafe { get_read(frame, 1) };
    let v_ptr = unsafe { get_read(frame, 2) };
    if y_ptr.is_null() || u_ptr.is_null() || v_ptr.is_null() {
        return None;
    }
    let y_h = crate::super_opts::super_plane_height(opts.height, opts.pel, opts.super_levels, true)
        as usize;
    let uv_h = y_h / 2;
    let y = unsafe { std::slice::from_raw_parts(y_ptr, y_stride * y_h) };
    let u = unsafe { std::slice::from_raw_parts(u_ptr, u_stride * uv_h) };
    let v = unsafe { std::slice::from_raw_parts(v_ptr, v_stride * uv_h) };
    Some(SuperPlanes {
        y,
        y_stride,
        u,
        u_stride,
        v,
        v_stride,
        luma_w: opts.width as usize,
        luma_h: opts.height as usize,
        pel: opts.pel,
        levels: opts.super_levels,
        full: true,
    })
}
