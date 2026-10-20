use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex, PoisonError};

use crate::analyse::{self, AnalyseParams, RawPlane, SuperFrameView, SuperParams};
use crate::{params, video_format, vs};

const MAX_AHEAD: i64 = 8;
const CACHE_SHARE: i64 = 16;

struct AnalyseState {
    super_node: vs::Raw,
    src_node: vs::Raw,
    vi: vs::VideoInfo,
    params: AnalyseParams,
    super_height: usize,
    _header: Box<[i32]>,
    gray_format: vs::ConstRaw,
    depth: i32,
    ahead: Mutex<HashMap<i32, Arc<Slot>>>,
}

struct Inputs {
    supers: [vs::ConstRaw; 2],
    sources: [vs::ConstRaw; 2],
}

unsafe impl Send for Inputs {}

enum Ahead {
    Queued(Inputs),
    Running,
    Done(Vec<u8>),
    Taken,
}

struct Slot {
    state: Mutex<Ahead>,
    done: Condvar,
}

impl Slot {
    fn new(state: Ahead) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(state),
            done: Condvar::new(),
        })
    }

    fn claim(&self) -> Option<Inputs> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        match std::mem::replace(&mut *state, Ahead::Running) {
            Ahead::Queued(inputs) => Some(inputs),
            other => {
                *state = other;
                None
            }
        }
    }

    fn running(&self) -> bool {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        matches!(*state, Ahead::Running)
    }

    fn set(&self, state: Ahead) {
        *self.state.lock().unwrap_or_else(PoisonError::into_inner) = state;
        self.done.notify_all();
    }

    fn take(&self) -> Option<Vec<u8>> {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let mut state = self
            .done
            .wait_while(state, |state| matches!(state, Ahead::Running))
            .unwrap_or_else(PoisonError::into_inner);
        match std::mem::replace(&mut *state, Ahead::Taken) {
            Ahead::Done(payload) => Some(payload),
            _ => None,
        }
    }
}

struct Job(*const AnalyseState);

unsafe impl Send for Job {}

impl AnalyseState {
    fn pair(&self, n: i32) -> [i32; 2] {
        [n, (n + self.params.delta).min(self.vi.num_frames - 1)]
    }

    fn full(&self) -> bool {
        self.params.super_params.has_finest_level()
    }

    unsafe fn fetch(&self, api: &FrameApi, n: i32, frame_ctx: vs::Raw) -> Option<Inputs> {
        let fetch = |node: vs::Raw| {
            self.pair(n)
                .map(|frame| unsafe { (api.get_frame)(frame, node, frame_ctx) })
        };
        let inputs = Inputs {
            supers: fetch(self.super_node),
            sources: if self.full() {
                [std::ptr::null(); 2]
            } else {
                fetch(self.src_node)
            },
        };
        let missing = |frames: &[vs::ConstRaw; 2]| frames.iter().any(|f| f.is_null());
        if missing(&inputs.supers) || (!self.full() && missing(&inputs.sources)) {
            unsafe { api.release(&inputs) };
            return None;
        }
        Some(inputs)
    }

    unsafe fn compute(&self, api: &FrameApi, n: i32, inputs: &Inputs) -> Vec<u8> {
        let params = &self.params;
        let sp = params.super_params;
        let y_shift = params.chroma_shift.1;
        let full = self.full();
        let view = |index: usize| SuperFrameView {
            planes: unsafe { api.planes(inputs.supers[index], self.super_height, y_shift) },
            source: (!full)
                .then(|| unsafe { api.planes(inputs.sources[index], sp.height as usize, y_shift) }),
            width: sp.width,
            height: sp.height,
            pel: sp.pel,
            chroma_shift: params.chroma_shift,
        };
        let level0 = if full { self.super_node } else { self.src_node } as usize;
        let keys = self.pair(n).map(|frame| (level0, frame));
        analyse::analyse(params, &view(0), &view(1), Some(keys))
    }

    unsafe fn look_ahead(&self, api: &FrameApi, n: i32, frame_ctx: vs::Raw) -> (Arc<Slot>, bool) {
        let mut ahead = self.ahead.lock().unwrap_or_else(PoisonError::into_inner);
        ahead.retain(|&frame, slot| {
            if (frame - n).abs() <= 2 * self.depth {
                return true;
            }
            match slot.claim() {
                Some(inputs) => {
                    unsafe { api.release(&inputs) };
                    slot.set(Ahead::Taken);
                    false
                }
                None => slot.running(),
            }
        });
        for frame in n + 1..=(n + self.depth).min(self.vi.num_frames - 1) {
            if ahead.contains_key(&frame) {
                continue;
            }
            let Some(inputs) = (unsafe { self.fetch(api, frame, frame_ctx) }) else {
                continue;
            };
            let slot = Slot::new(Ahead::Queued(inputs));
            ahead.insert(frame, slot.clone());
            let (job, api) = (Job(self), *api);
            svpflow_core::threads::spawn(move || {
                let job = job;
                if let Some(inputs) = slot.claim() {
                    let payload = unsafe { (*job.0).compute(&api, frame, &inputs) };
                    unsafe { api.release(&inputs) };
                    slot.set(Ahead::Done(payload));
                }
            });
        }
        if let Some(slot) = ahead.get(&n) {
            return (slot.clone(), false);
        }
        let slot = Slot::new(Ahead::Running);
        ahead.insert(n, slot.clone());
        (slot, true)
    }
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
    let get_core_info =
        unsafe { vs::table_fn::<vs::GetCoreInfo>(vsapi, vs::GET_CORE_INFO) }.ok_or(INVALID_API)?;

    svpflow_core::threads::set_default(unsafe { vs::core_threads(core, vsapi) });
    let mut err = 0;
    let super_node = unsafe { get_node(input, c"clip".as_ptr(), 0, &raw mut err) };
    if super_node.is_null() {
        return Err("SVAnalyse: clip (super) required".into());
    }
    let given_src = unsafe { get_node(input, c"src".as_ptr(), 0, &raw mut err) };
    let src_node = if given_src.is_null() {
        unsafe { get_node(input, c"clip".as_ptr(), 0, &raw mut err) }
    } else {
        given_src
    };
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
        if !full && given_src.is_null() {
            return Err("SVAnalyse: src required".into());
        }
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

        let frame_bytes = |vi: &vs::VideoInfo| {
            let luma = i64::from(vi.width) * i64::from(vi.height);
            luma + ((2 * luma) >> (chroma_shift.0 + chroma_shift.1))
        };
        let held = frame_bytes(&super_vi) + if full { 0 } else { frame_bytes(&src_vi) };
        let info = unsafe { get_core_info(core) };
        let cache = if info.is_null() {
            0
        } else {
            unsafe { (*info).max_framebuffer }
        };

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
            depth: (cache / CACHE_SHARE / held).clamp(0, MAX_AHEAD) as i32,
            ahead: Mutex::default(),
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
    if let Some(preset) =
        unsafe { vs::table_fn::<vs::GetFormatPreset>(vsapi, vs::GET_FORMAT_PRESET) }
    {
        let format = unsafe { preset(vs::PF_GRAY8, core) };
        if !format.is_null() {
            return format.cast();
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
    let ahead = std::mem::take(&mut *state.ahead.lock().unwrap_or_else(PoisonError::into_inner));
    for slot in ahead.into_values() {
        match (slot.claim(), unsafe { FrameApi::load(vsapi) }) {
            (Some(inputs), Some(api)) => unsafe { api.release(&inputs) },
            _ => drop(slot.take()),
        }
    }
    unsafe { vs::free_node(state.super_node, vsapi) };
    unsafe { vs::free_node(state.src_node, vsapi) };
}

#[derive(Clone, Copy)]
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

    unsafe fn release(&self, inputs: &Inputs) {
        for &frame in inputs.supers.iter().chain(&inputs.sources) {
            if !frame.is_null() {
                unsafe { (self.free_frame)(frame) };
            }
        }
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
    let last = state.vi.num_frames - 1;

    if activation == vs::AR_INITIAL {
        if let Some(request) =
            unsafe { vs::table_fn::<vs::RequestFrameFilter>(vsapi, vs::REQUEST_FRAME_FILTER) }
        {
            let mut frames: Vec<i32> = (n..=(n + state.depth).min(last).max(n))
                .flat_map(|frame| state.pair(frame))
                .collect();
            frames.sort_unstable();
            frames.dedup();
            for frame in frames {
                unsafe { request(frame, state.super_node, frame_ctx) };
                if !state.full() {
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

    let (slot, fresh) = unsafe { state.look_ahead(&api, n, frame_ctx) };
    let Some(inputs) = (unsafe { state.fetch(&api, n, frame_ctx) }) else {
        slot.set(Ahead::Taken);
        return std::ptr::null();
    };
    let ready = match slot.claim() {
        _ if fresh => None,
        Some(queued) => {
            unsafe { api.release(&queued) };
            None
        }
        None => slot.take(),
    };
    let payload = ready.unwrap_or_else(|| unsafe { state.compute(&api, n, &inputs) });
    slot.set(Ahead::Taken);
    let out = unsafe {
        new_frame(
            state.gray_format,
            payload.len() as i32,
            1,
            inputs.supers[0],
            core,
        )
    };
    if !out.is_null() {
        let dst = unsafe { get_write(out, 0) };
        if !dst.is_null() {
            unsafe { std::ptr::copy_nonoverlapping(payload.as_ptr(), dst, payload.len()) };
        }
    }
    unsafe { api.release(&inputs) };
    out.cast_const()
}
