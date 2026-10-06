use crate::{video_format, vs};

const HEADER: usize = 4;
const REGIONS: usize = HEADER + 4 * 15;

struct Api {
    get_frame: vs::GetFrameFilter,
    request: vs::RequestFrameFilter,
    new_frame: vs::NewVideoFrame,
    free_frame: vs::FreeFrame,
    get_stride: vs::GetStride,
    get_read: vs::GetReadPtr,
    get_write: vs::GetWritePtr,
}

impl Api {
    unsafe fn load(vsapi: vs::ConstRaw) -> Option<Self> {
        unsafe {
            Some(Self {
                get_frame: vs::table_fn(vsapi, vs::GET_FRAME_FILTER)?,
                request: vs::table_fn(vsapi, vs::REQUEST_FRAME_FILTER)?,
                new_frame: vs::table_fn(vsapi, vs::NEW_VIDEO_FRAME)?,
                free_frame: vs::table_fn(vsapi, vs::FREE_FRAME)?,
                get_stride: vs::table_fn(vsapi, vs::GET_STRIDE)?,
                get_read: vs::table_fn(vsapi, vs::GET_READ_PTR)?,
                get_write: vs::table_fn(vsapi, vs::GET_WRITE_PTR)?,
            })
        }
    }
}

struct State {
    node: vs::Raw,
    vi: vs::VideoInfo,
    header: Box<[i32; 15]>,
    multiply: i32,
}

type Process = unsafe fn(&State, &Api, vs::ConstRaw, vs::Raw);

unsafe fn fail(output: vs::Raw, vsapi: vs::ConstRaw, node: vs::Raw, message: &std::ffi::CStr) {
    unsafe {
        vs::free_node(node, vsapi);
        vs::set_error(output, vsapi, message.as_ptr());
    }
}

unsafe fn register(
    input: vs::ConstRaw,
    output: vs::Raw,
    core: vs::Raw,
    vsapi: vs::ConstRaw,
    name: &std::ffi::CStr,
    state: State,
    get_frame: vs::GetFrame,
) {
    let Some(create) = (unsafe { vs::table_fn::<vs::CreateFilter>(vsapi, vs::CREATE_FILTER) })
    else {
        return;
    };
    unsafe {
        create(
            input,
            output,
            name.as_ptr(),
            init,
            get_frame,
            free,
            vs::FM_PARALLEL,
            0,
            Box::into_raw(Box::new(state)).cast(),
            core,
        );
    }
}

pub(crate) unsafe extern "system" fn create_halve(
    input: vs::ConstRaw,
    output: vs::Raw,
    _: vs::Raw,
    core: vs::Raw,
    vsapi: vs::ConstRaw,
) -> isize {
    let (Some(get_node), Some(get_vi)) = (unsafe {
        (
            vs::table_fn::<vs::PropGetNode>(vsapi, vs::PROP_GET_NODE),
            vs::table_fn::<vs::GetVideoInfo>(vsapi, vs::GET_VIDEO_INFO),
        )
    }) else {
        return 0;
    };
    let mut err = 0;
    let node = unsafe { get_node(input, c"clip".as_ptr(), 0, &raw mut err) };
    if node.is_null() {
        return 0;
    }
    let mut vi = unsafe { *get_vi(node) };
    if !video_format::is_supported(&vi) || vi.width % 4 != 0 || vi.height % 4 != 0 {
        unsafe {
            fail(
                output,
                vsapi,
                node,
                c"SVHalve: clip must be YUV420P8 or YUV444P8 with a size that divides by 4",
            );
        }
        return 0;
    }
    vi.width /= 2;
    vi.height /= 2;
    let state = State {
        node,
        vi,
        header: Box::new([0; 15]),
        multiply: 0,
    };
    unsafe { register(input, output, core, vsapi, c"SVHalve", state, halve_frame) };
    0
}

pub(crate) unsafe extern "system" fn create_double_vectors(
    input: vs::ConstRaw,
    output: vs::Raw,
    _: vs::Raw,
    core: vs::Raw,
    vsapi: vs::ConstRaw,
) -> isize {
    let (Some(get_node), Some(get_vi), Some(get_int), Some(set_int)) = (unsafe {
        (
            vs::table_fn::<vs::PropGetNode>(vsapi, vs::PROP_GET_NODE),
            vs::table_fn::<vs::GetVideoInfo>(vsapi, vs::GET_VIDEO_INFO),
            vs::table_fn::<vs::PropGetInt>(vsapi, vs::PROP_GET_INT),
            vs::table_fn::<vs::PropSetInt>(vsapi, vs::PROP_SET_INT),
        )
    }) else {
        return 0;
    };
    let mut err = 0;
    let node = unsafe { get_node(input, c"clip".as_ptr(), 0, &raw mut err) };
    if node.is_null() {
        return 0;
    }
    let vdata = unsafe { get_int(input, c"vdata".as_ptr(), 0, &raw mut err) };
    let vi = unsafe { *get_vi(node) };
    let source = vdata as *const [i32; 15];
    let small = unsafe { source.as_ref() }.filter(|header| header[0] == 0xA0 && header[4] <= 2);
    let Some(small) = small.filter(|_| err == 0 && vi.width as usize > REGIONS) else {
        unsafe {
            fail(
                output,
                vsapi,
                node,
                c"SVDoubleVectors: vectors with pel 1 or 2 are required",
            );
        }
        return 0;
    };
    let mut header = Box::new(*small);
    for index in [2, 3, 7, 8, 9, 10] {
        header[index] *= 2;
    }
    let multiply = 2 / small[4];
    header[4] = 1;
    unsafe { set_int(output, c"data".as_ptr(), header.as_ptr() as i64, 0) };
    let state = State {
        node,
        vi,
        header,
        multiply,
    };
    unsafe {
        register(
            input,
            output,
            core,
            vsapi,
            c"SVDoubleVectors",
            state,
            double_frame,
        );
    }
    0
}

unsafe extern "system" fn init(
    _: vs::ConstRaw,
    _: vs::Raw,
    instance_data: *mut vs::Raw,
    node: vs::Raw,
    _: vs::Raw,
    vsapi: vs::ConstRaw,
) {
    let state = unsafe { &*(*instance_data).cast::<State>() };
    if let Some(set_vi) = unsafe { vs::table_fn::<vs::SetVideoInfo>(vsapi, vs::SET_VIDEO_INFO) } {
        unsafe { set_vi(&raw const state.vi, 1, node) };
    }
}

unsafe extern "system" fn free(instance_data: vs::Raw, _: vs::Raw, vsapi: vs::ConstRaw) {
    let state = unsafe { Box::from_raw(instance_data.cast::<State>()) };
    unsafe { vs::free_node(state.node, vsapi) };
}

unsafe fn run(
    n: i32,
    activation: i32,
    instance_data: *mut vs::Raw,
    frame_ctx: vs::Raw,
    core: vs::Raw,
    vsapi: vs::ConstRaw,
    process: Process,
) -> vs::ConstRaw {
    let state = unsafe { &*(*instance_data).cast::<State>() };
    let Some(api) = (unsafe { Api::load(vsapi) }) else {
        return std::ptr::null();
    };
    if activation == vs::AR_INITIAL {
        unsafe { (api.request)(n, state.node, frame_ctx) };
        return std::ptr::null();
    }
    if activation != vs::AR_ALL_FRAMES_READY {
        return std::ptr::null();
    }
    let source = unsafe { (api.get_frame)(n, state.node, frame_ctx) };
    if source.is_null() {
        return std::ptr::null();
    }
    let output = unsafe {
        (api.new_frame)(
            state.vi.format,
            state.vi.width,
            state.vi.height,
            source,
            core,
        )
    };
    if !output.is_null() {
        unsafe { process(state, &api, source, output) };
    }
    unsafe { (api.free_frame)(source) };
    output.cast_const()
}

unsafe extern "system" fn halve_frame(
    n: i32,
    activation: i32,
    instance_data: *mut vs::Raw,
    _: *mut vs::Raw,
    frame_ctx: vs::Raw,
    core: vs::Raw,
    vsapi: vs::ConstRaw,
) -> vs::ConstRaw {
    unsafe { run(n, activation, instance_data, frame_ctx, core, vsapi, halve) }
}

unsafe extern "system" fn double_frame(
    n: i32,
    activation: i32,
    instance_data: *mut vs::Raw,
    _: *mut vs::Raw,
    frame_ctx: vs::Raw,
    core: vs::Raw,
    vsapi: vs::ConstRaw,
) -> vs::ConstRaw {
    unsafe { run(n, activation, instance_data, frame_ctx, core, vsapi, double) }
}

unsafe fn halve(state: &State, api: &Api, source: vs::ConstRaw, output: vs::Raw) {
    let (x_div, y_div) = video_format::chroma_divisors(&state.vi);
    for plane in 0..3 {
        let (x_div, y_div) = if plane == 0 { (1, 1) } else { (x_div, y_div) };
        let width = state.vi.width as usize / x_div;
        let height = state.vi.height as usize / y_div;
        let (from_pitch, to_pitch) = unsafe {
            (
                (api.get_stride)(source, plane) as usize,
                (api.get_stride)(output.cast_const(), plane) as usize,
            )
        };
        let from = unsafe {
            std::slice::from_raw_parts((api.get_read)(source, plane), from_pitch * height * 2)
        };
        let to = unsafe {
            std::slice::from_raw_parts_mut((api.get_write)(output, plane), to_pitch * height)
        };
        for (to, from) in to.chunks_mut(to_pitch).zip(from.chunks(from_pitch * 2)) {
            let (top, bottom) = from.split_at(from_pitch);
            for ((to, top), bottom) in to[..width]
                .iter_mut()
                .zip(top.as_chunks::<2>().0)
                .zip(bottom.as_chunks::<2>().0)
            {
                let sum = u16::from(top[0])
                    + u16::from(top[1])
                    + u16::from(bottom[0])
                    + u16::from(bottom[1]);
                *to = ((sum + 2) >> 2) as u8;
            }
        }
    }
}

unsafe fn double(state: &State, api: &Api, source: vs::ConstRaw, output: vs::Raw) {
    let len = state.vi.width as usize;
    let from = unsafe { std::slice::from_raw_parts((api.get_read)(source, 0), len) };
    let to = unsafe { std::slice::from_raw_parts_mut((api.get_write)(output, 0), len) };
    to.copy_from_slice(from);
    for (to, value) in to[HEADER..REGIONS]
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(*state.header)
    {
        *to = value.to_le_bytes();
    }
    let count = (state.header[11] * state.header[12]) as usize;
    for region in to[REGIONS..].chunks_exact_mut(4 + 8 * count) {
        for block in region[4..].as_chunks_mut::<8>().0 {
            let (low, high) = block.split_at_mut(4);
            let read = |bytes: &[u8]| u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            let (vector, score) = (read(low).cast_signed(), read(high));
            let y = i32::from(vector as i16);
            let x = vector.wrapping_sub(y) >> 16;
            let limit = |value: i32| (value * state.multiply).clamp(-0x7FFF, 0x7FFF);
            let vector = (limit(x) << 16).wrapping_add(limit(y));
            let sad = (score & 0x00FF_FFFF).saturating_mul(4).min(0x00FF_FFFF);
            low.copy_from_slice(&vector.to_le_bytes());
            high.copy_from_slice(&((score & 0xFF00_0000) | sad).to_le_bytes());
        }
    }
}
