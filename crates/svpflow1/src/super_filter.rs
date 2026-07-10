use crate::{params, super_build, super_opts::SuperOpts, vs};

struct SuperState {
    node: vs::Raw,
    vi: vs::VideoInfo,
    opts: SuperOpts,
}

pub(crate) unsafe extern "system" fn create_super(
    input: vs::ConstRaw,
    output: vs::Raw,
    _user: vs::Raw,
    core: vs::Raw,
    vsapi: vs::ConstRaw,
) -> isize {
    unsafe { create_super_inner(input, output, core, vsapi) }
}

unsafe fn create_super_inner(
    input: vs::ConstRaw,
    output: vs::Raw,
    core: vs::Raw,
    vsapi: vs::ConstRaw,
) -> isize {
    let Some(get_node) = (unsafe { vs::table_fn::<vs::PropGetNode>(vsapi, vs::PROP_GET_NODE) })
    else {
        unsafe { vs::set_error(output, vsapi, c"SVSuper: invalid VSAPI table".as_ptr()) };
        return 0;
    };
    let Some(get_vi) = (unsafe { vs::table_fn::<vs::GetVideoInfo>(vsapi, vs::GET_VIDEO_INFO) })
    else {
        unsafe { vs::set_error(output, vsapi, c"SVSuper: invalid VSAPI table".as_ptr()) };
        return 0;
    };
    let Some(create_filter) =
        (unsafe { vs::table_fn::<vs::CreateFilter>(vsapi, vs::CREATE_FILTER) })
    else {
        unsafe { vs::set_error(output, vsapi, c"SVSuper: invalid VSAPI table".as_ptr()) };
        return 0;
    };
    let Some(prop_set_int) = (unsafe { vs::table_fn::<vs::PropSetInt>(vsapi, vs::PROP_SET_INT) })
    else {
        unsafe { vs::set_error(output, vsapi, c"SVSuper: invalid VSAPI table".as_ptr()) };
        return 0;
    };

    let mut err = 0;
    let node = unsafe { get_node(input, c"clip".as_ptr(), 0, &raw mut err) };
    if node.is_null() {
        unsafe { vs::set_error(output, vsapi, c"SVSuper: clip is required".as_ptr()) };
        return 0;
    }
    let vi_ptr = unsafe { get_vi(node) };
    if vi_ptr.is_null() {
        unsafe { vs::free_node(node, vsapi) };
        unsafe { vs::set_error(output, vsapi, c"SVSuper: invalid clip".as_ptr()) };
        return 0;
    }
    let vi = unsafe { *vi_ptr };

    if vi.width <= 0 || vi.height <= 0 || vi.format.is_null() {
        unsafe { vs::free_node(node, vsapi) };
        unsafe { vs::set_error(output, vsapi, c"SVSuper: Clip must be YV12".as_ptr()) };
        return 0;
    }

    let opt_val = unsafe { read_opt(input, vsapi) };
    let opts = match SuperOpts::from_opt(opt_val.as_ref(), vi.width, vi.height) {
        Ok(o) => o,
        Err(msg) => {
            unsafe { vs::free_node(node, vsapi) };
            let cmsg = std::ffi::CString::new(msg).unwrap_or_default();
            unsafe { vs::set_error(output, vsapi, cmsg.as_ptr()) };
            return 0;
        }
    };

    let sdata = opts.pack_data();
    unsafe { prop_set_int(output, c"data".as_ptr(), sdata, 0) };

    let mut out_vi = vi;
    out_vi.height = opts.super_height();

    let state = Box::into_raw(Box::new(SuperState {
        node,
        vi: out_vi,
        opts,
    }));

    unsafe {
        create_filter(
            input,
            output,
            c"SVSuper".as_ptr(),
            init_super,
            get_frame_super,
            free_super,
            vs::FM_PARALLEL,
            0,
            state.cast(),
            core,
        );
    }
    0
}

unsafe fn read_opt(input: vs::ConstRaw, vsapi: vs::ConstRaw) -> Option<params::Value> {
    let get_data = unsafe { vs::table_fn::<vs::PropGetData>(vsapi, vs::PROP_GET_DATA)? };
    let get_size = unsafe { vs::table_fn::<vs::PropGetDataSize>(vsapi, vs::PROP_GET_DATA_SIZE)? };
    let mut err = 0;
    let ptr = unsafe { get_data(input, c"opt".as_ptr(), 0, &raw mut err) };
    if ptr.is_null() || err != 0 {
        return None;
    }
    let size = unsafe { get_size(input, c"opt".as_ptr(), 0, &raw mut err) };
    if size <= 0 || err != 0 {
        return None;
    }
    let bytes = unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), size as usize) };
    params::parse(bytes).ok()
}

unsafe extern "system" fn init_super(
    _in: vs::ConstRaw,
    _out: vs::Raw,
    instance_data: *mut vs::Raw,
    node: vs::Raw,
    _core: vs::Raw,
    vsapi: vs::ConstRaw,
) {
    let state = unsafe { &*(*instance_data).cast::<SuperState>() };
    if let Some(set_vi) = unsafe { vs::table_fn::<vs::SetVideoInfo>(vsapi, vs::SET_VIDEO_INFO) } {
        unsafe { set_vi(&raw const state.vi, 1, node) };
    }
}

unsafe extern "system" fn free_super(instance_data: vs::Raw, core: vs::Raw, vsapi: vs::ConstRaw) {
    let _ = core;
    let state = unsafe { Box::from_raw(instance_data.cast::<SuperState>()) };
    unsafe { vs::free_node(state.node, vsapi) };
}

unsafe extern "system" fn get_frame_super(
    n: i32,
    activation: i32,
    instance_data: *mut vs::Raw,
    _frame_data: *mut vs::Raw,
    frame_ctx: vs::Raw,
    core: vs::Raw,
    vsapi: vs::ConstRaw,
) -> vs::ConstRaw {
    let state = unsafe { &*(*instance_data).cast::<SuperState>() };

    if activation == vs::AR_INITIAL {
        if let Some(request) =
            unsafe { vs::table_fn::<vs::RequestFrameFilter>(vsapi, vs::REQUEST_FRAME_FILTER) }
        {
            unsafe { request(n, state.node, frame_ctx) };
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

    let src_frame = unsafe { get_frame(n, state.node, frame_ctx) };
    if src_frame.is_null() {
        return std::ptr::null();
    }

    let out = unsafe {
        new_frame(
            state.vi.format,
            state.vi.width,
            state.vi.height,
            src_frame,
            core,
        )
    };
    if out.is_null() {
        if let Some(ff) = free_frame {
            unsafe { ff(src_frame) };
        }
        return std::ptr::null();
    }

    for plane in 0..3 {
        let stride = unsafe { get_stride(out.cast_const(), plane) } as usize;
        let ptr = unsafe { get_write(out, plane) };
        let h = if plane == 0 {
            state.vi.height as usize
        } else {
            (state.vi.height as usize) / 2
        };
        if !ptr.is_null() && stride > 0 {
            unsafe {
                std::ptr::write_bytes(ptr, 0, stride * h);
            }
        }
    }

    let src_w = state.opts.width as usize;
    let src_h = state.opts.height as usize;

    for plane in 0..3i32 {
        let (bw, bh) = if plane == 0 {
            (src_w, src_h)
        } else {
            (src_w / 2, src_h / 2)
        };
        let src_stride = unsafe { get_stride(src_frame, plane) } as usize;
        let src_ptr = unsafe { get_read(src_frame, plane) };
        let dst_stride = unsafe { get_stride(out.cast_const(), plane) } as usize;
        let dst_ptr = unsafe { get_write(out, plane) };
        if src_ptr.is_null() || dst_ptr.is_null() || src_stride == 0 || dst_stride == 0 {
            continue;
        }
        let src_h_bytes = bh;
        let dst_h = if plane == 0 {
            state.vi.height as usize
        } else {
            state.vi.height as usize / 2
        };
        let src_slice = unsafe { std::slice::from_raw_parts(src_ptr, src_stride * src_h_bytes) };
        let dst_slice = unsafe { std::slice::from_raw_parts_mut(dst_ptr, dst_stride * dst_h) };

        super_build::build_plane(
            dst_slice,
            dst_stride,
            src_slice,
            src_stride,
            bw,
            bh,
            src_w,
            src_h,
            plane != 0,
            &state.opts,
        );
    }

    if let Some(ff) = free_frame {
        unsafe { ff(src_frame) };
    }
    out.cast_const()
}
