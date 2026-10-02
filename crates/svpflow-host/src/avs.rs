use std::cell::Cell;
use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_void};
use std::sync::{Mutex, OnceLock};

use libloading::Library;

use crate::vs3::{self, ConstRaw, CoreInfo, Create, Format, Raw, VideoInfo};

pub struct Arg {
    pub key: &'static CStr,
    pub data: Option<&'static CStr>,
}

pub struct Function {
    pub name: &'static CStr,
    pub params: &'static CStr,
    pub args: &'static [Arg],
    pub create: Create,
}

#[must_use]
pub const fn arg(key: &'static CStr) -> Arg {
    Arg { key, data: None }
}

#[must_use]
pub const fn data_clip(key: &'static CStr, data: &'static CStr) -> Arg {
    Arg {
        key,
        data: Some(data),
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
union Payload {
    clip: Raw,
    boolean: c_char,
    integer: i32,
    floating: f32,
    string: *const c_char,
    array: *const AvsValue,
    #[cfg(target_pointer_width = "64")]
    wide: i64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AvsValue {
    kind: i16,
    array_size: i16,
    payload: Payload,
}

impl AvsValue {
    const VOID: Self = Self {
        kind: b'v' as i16,
        array_size: 0,
        payload: Payload {
            clip: std::ptr::null_mut(),
        },
    };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AvsVideoInfo {
    width: i32,
    height: i32,
    fps_num: u32,
    fps_den: u32,
    num_frames: i32,
    pixel_type: i32,
    audio_samples_per_second: i32,
    sample_type: i32,
    num_audio_samples: i64,
    nchannels: i32,
    image_type: i32,
}

#[repr(C)]
struct FilterInfo {
    child: Raw,
    vi: AvsVideoInfo,
    env: Raw,
    get_frame: Option<unsafe extern "system" fn(*mut FilterInfo, i32) -> Raw>,
    get_parity: Option<unsafe extern "system" fn(*mut FilterInfo, i32) -> i32>,
    get_audio: Option<unsafe extern "system" fn(*mut FilterInfo, Raw, i64, i64) -> i32>,
    set_cache_hints: Option<unsafe extern "system" fn(*mut FilterInfo, i32, i32) -> i32>,
    free_filter: Option<unsafe extern "system" fn(*mut FilterInfo)>,
    error: *const c_char,
    user_data: Raw,
}

type Apply = unsafe extern "system" fn(Raw, AvsValue, Raw) -> AvsValue;

macro_rules! api {
    ($($name:ident: fn($($arg:ty),*) $(-> $ret:ty)?;)*) => {
        struct Api {
            $($name: unsafe extern "system" fn($($arg),*) $(-> $ret)?,)*
        }

        impl Api {
            unsafe fn load(library: &Library) -> Option<Self> {
                Some(Self {
                    $($name: *unsafe {
                        library.get(concat!("avs_", stringify!($name), "\0").as_bytes())
                    }
                    .ok()?,)*
                })
            }
        }
    };
}

api! {
    add_function: fn(Raw, *const c_char, *const c_char, Apply, Raw) -> i32;
    new_c_filter: fn(Raw, *mut *mut FilterInfo, AvsValue, i32) -> Raw;
    take_clip: fn(AvsValue, Raw) -> Raw;
    set_to_clip: fn(*mut AvsValue, Raw);
    release_value: fn(AvsValue);
    copy_clip: fn(Raw) -> Raw;
    release_clip: fn(Raw);
    get_video_info: fn(Raw) -> *const AvsVideoInfo;
    get_frame: fn(Raw, i32) -> Raw;
    clip_get_error: fn(Raw) -> *const c_char;
    copy_video_frame: fn(Raw) -> Raw;
    release_video_frame: fn(Raw);
    make_writable: fn(Raw, *mut Raw) -> i32;
    new_video_frame_p: fn(Raw, *const AvsVideoInfo, ConstRaw) -> Raw;
    get_pitch_p: fn(ConstRaw, i32) -> i32;
    get_read_ptr_p: fn(ConstRaw, i32) -> *const u8;
    get_write_ptr_p: fn(ConstRaw, i32) -> *mut u8;
    get_frame_props_ro: fn(Raw, ConstRaw) -> ConstRaw;
    get_frame_props_rw: fn(Raw, Raw) -> Raw;
    prop_get_int: fn(Raw, ConstRaw, *const c_char, i32, *mut i32) -> i64;
    prop_get_float: fn(Raw, ConstRaw, *const c_char, i32, *mut i32) -> f64;
    prop_get_data: fn(Raw, ConstRaw, *const c_char, i32, *mut i32) -> *const c_char;
    prop_get_data_size: fn(Raw, ConstRaw, *const c_char, i32, *mut i32) -> i32;
    prop_set_int: fn(Raw, Raw, *const c_char, i64, i32) -> i32;
    save_string: fn(Raw, *const c_char, i32) -> *mut c_char;
    get_env_property: fn(Raw, i32) -> usize;
}

fn libraries() -> Vec<Library> {
    let mut libraries = Vec::new();
    #[cfg(unix)]
    libraries.push(libloading::os::unix::Library::this().into());
    #[cfg(windows)]
    if let Ok(library) = libloading::os::windows::Library::open_already_loaded("avisynth") {
        libraries.push(library.into());
    }
    if let Ok(library) = unsafe { Library::new(libloading::library_filename("avisynth")) } {
        libraries.push(library);
    }
    libraries
}

fn api() -> Option<&'static Api> {
    static API: OnceLock<Option<Api>> = OnceLock::new();
    API.get_or_init(|| {
        libraries().into_iter().find_map(|library| {
            let api = unsafe { Api::load(&library) };
            std::mem::forget(library);
            api
        })
    })
    .as_ref()
}

fn loaded() -> &'static Api {
    api().unwrap_or_else(|| std::process::abort())
}

thread_local! {
    static ENV: Cell<Raw> = const { Cell::new(std::ptr::null_mut()) };
}

fn env() -> Raw {
    ENV.get()
}

const PLANAR_FILTER: i32 = !0b1_1000;
const CS_YV12: i32 = 0xA000_0008_u32.cast_signed();
const CS_YV24: i32 = 0xA000_030B_u32.cast_signed();
const CS_YUV420P10: i32 = 0xA005_0008_u32.cast_signed();
const CS_YUV420P16: i32 = 0xA001_0008_u32.cast_signed();
const CS_BGR32: i32 = 0x5000_0002;
const PACKED_BYTES: i32 = 4;

const PLANES: [i32; 3] = [1, 2, 4];
const CACHE_GET_MTMODE: i32 = 509;
const MT_NICE_FILTER: i32 = 1;
const MT_SERIALIZED: i32 = 3;
const AEP_THREADPOOL_THREADS: i32 = 3;

const FORMATS: [(i32, i32); 5] = [
    (CS_YV12, vs3::PF_YUV420P8),
    (CS_YV24, vs3::PF_YUV444P8),
    (CS_YUV420P10, vs3::PF_YUV420P10),
    (CS_YUV420P16, vs3::PF_YUV420P16),
    (CS_BGR32, vs3::PF_GRAY8),
];

fn format(pixel_type: i32) -> *const Format {
    FORMATS
        .iter()
        .find(|(cs, _)| cs & PLANAR_FILTER == pixel_type & PLANAR_FILTER)
        .and_then(|&(_, id)| vs3::preset(id))
        .map_or(std::ptr::null(), vs3::intern)
}

unsafe fn pixel_type(format: ConstRaw) -> i32 {
    let id = unsafe { format.cast::<Format>().as_ref() }.map_or(0, |format| format.id);
    FORMATS
        .iter()
        .find(|&&(_, preset)| preset == id)
        .map_or(0, |&(cs, _)| cs)
}

fn frame_rate(num: i64, den: i64) -> (u32, u32) {
    let (mut num, mut den) = (num.unsigned_abs(), den.unsigned_abs());
    let (mut a, mut b) = (num, den);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    if a > 1 {
        (num, den) = (num / a, den / a);
    }
    while num > u64::from(u32::MAX) || den > u64::from(u32::MAX) {
        (num, den) = (num >> 1, (den >> 1).max(1));
    }
    (
        u32::try_from(num).unwrap_or(u32::MAX),
        u32::try_from(den).unwrap_or(u32::MAX),
    )
}

enum Value {
    Int(i64),
    Float(f64),
    Data(CString),
    Clip(Raw),
}

#[derive(Default)]
struct Map {
    values: Vec<(CString, Value)>,
    error: Option<CString>,
}

impl Map {
    fn get(&self, key: &CStr) -> Option<&Value> {
        self.values
            .iter()
            .find(|(name, _)| name.as_c_str() == key)
            .map(|(_, value)| value)
    }

    fn set(&mut self, key: &CStr, value: Value) {
        self.values.push((key.to_owned(), value));
    }

    fn clip(&self) -> Option<Raw> {
        self.values.iter().find_map(|(_, value)| match value {
            Value::Clip(clip) => Some(*clip),
            _ => None,
        })
    }
}

impl Drop for Map {
    fn drop(&mut self) {
        for (_, value) in &self.values {
            if let Value::Clip(clip) = value {
                unsafe { (loaded().release_clip)(*clip) };
            }
        }
    }
}

thread_local! {
    static ARGS: Cell<[ConstRaw; 2]> = const { Cell::new([std::ptr::null(); 2]) };
}

unsafe fn args<'a>(map: ConstRaw) -> Option<&'a mut Map> {
    ARGS.get()
        .contains(&map)
        .then(|| unsafe { &mut *map.cast_mut().cast::<Map>() })
}

unsafe fn find<T>(
    args: &Map,
    key: *const c_char,
    error: *mut i32,
    pick: impl FnOnce(&Value) -> Option<T>,
) -> Option<T> {
    let value = args.get(unsafe { CStr::from_ptr(key) }).and_then(pick);
    if !error.is_null() {
        unsafe { *error = i32::from(value.is_none()) };
    }
    value
}

unsafe extern "system" fn prop_get_int(
    map: ConstRaw,
    key: *const c_char,
    index: i32,
    error: *mut i32,
) -> i64 {
    let Some(args) = (unsafe { args(map) }) else {
        return unsafe { (loaded().prop_get_int)(env(), map, key, index, error) };
    };
    let pick = |value: &Value| match value {
        Value::Int(value) => Some(*value),
        _ => None,
    };
    unsafe { find(args, key, error, pick) }.unwrap_or(0)
}

unsafe extern "system" fn prop_get_float(
    map: ConstRaw,
    key: *const c_char,
    index: i32,
    error: *mut i32,
) -> f64 {
    let Some(args) = (unsafe { args(map) }) else {
        return unsafe { (loaded().prop_get_float)(env(), map, key, index, error) };
    };
    #[allow(clippy::cast_precision_loss)]
    let pick = |value: &Value| match value {
        Value::Float(value) => Some(*value),
        Value::Int(value) => Some(*value as f64),
        _ => None,
    };
    unsafe { find(args, key, error, pick) }.unwrap_or(0.0)
}

unsafe extern "system" fn prop_get_data(
    map: ConstRaw,
    key: *const c_char,
    index: i32,
    error: *mut i32,
) -> *const c_char {
    let Some(args) = (unsafe { args(map) }) else {
        return unsafe { (loaded().prop_get_data)(env(), map, key, index, error) };
    };
    let pick = |value: &Value| match value {
        Value::Data(value) => Some(value.as_ptr()),
        _ => None,
    };
    unsafe { find(args, key, error, pick) }.unwrap_or(std::ptr::null())
}

unsafe extern "system" fn prop_get_data_size(
    map: ConstRaw,
    key: *const c_char,
    index: i32,
    error: *mut i32,
) -> i32 {
    let Some(args) = (unsafe { args(map) }) else {
        return unsafe { (loaded().prop_get_data_size)(env(), map, key, index, error) };
    };
    let pick = |value: &Value| match value {
        Value::Data(value) => i32::try_from(value.as_bytes().len()).ok(),
        _ => None,
    };
    unsafe { find(args, key, error, pick) }.unwrap_or(0)
}

unsafe extern "system" fn prop_get_node(
    map: ConstRaw,
    key: *const c_char,
    _index: i32,
    error: *mut i32,
) -> Raw {
    let Some(args) = (unsafe { args(map) }) else {
        return std::ptr::null_mut();
    };
    let pick = |value: &Value| match value {
        Value::Clip(clip) => Some(unsafe { (loaded().copy_clip)(*clip) }),
        _ => None,
    };
    unsafe { find(args, key, error, pick) }.unwrap_or(std::ptr::null_mut())
}

unsafe extern "system" fn prop_set_int(
    map: Raw,
    key: *const c_char,
    value: i64,
    append: i32,
) -> i32 {
    match unsafe { args(map) } {
        Some(args) => {
            args.set(unsafe { CStr::from_ptr(key) }, Value::Int(value));
            0
        }
        None => unsafe { (loaded().prop_set_int)(env(), map, key, value, append) },
    }
}

unsafe extern "system" fn set_error(map: Raw, message: *const c_char) {
    if let Some(args) = unsafe { args(map) } {
        args.error = Some(unsafe { CStr::from_ptr(message) }.to_owned());
    }
}

struct Adapter {
    instance: Raw,
    get_frame: vs3::GetFrame,
    free: vs3::FreeFilter,
    mt_mode: i32,
    info: Option<VideoInfo>,
}

struct FrameContext {
    error: *const c_char,
}

unsafe extern "system" fn set_video_info(info: *const VideoInfo, _count: i32, node: Raw) {
    unsafe { (*node.cast::<Adapter>()).info = Some(*info) };
}

unsafe fn output_info(info: &VideoInfo, child: &AvsVideoInfo, data: Option<i64>) -> AvsVideoInfo {
    let pixel_type = unsafe { pixel_type(info.format) };
    let (fps_num, fps_den) = frame_rate(info.fps_num, info.fps_den);
    let mut output = AvsVideoInfo {
        width: if pixel_type == CS_BGR32 {
            info.width / PACKED_BYTES
        } else {
            info.width
        },
        height: info.height,
        fps_num,
        fps_den,
        num_frames: info.num_frames,
        pixel_type,
        ..*child
    };
    if let Some(data) = data {
        output.audio_samples_per_second = 0;
        output.sample_type = 0;
        output.num_audio_samples = data;
        output.nchannels = 0;
    }
    output
}

unsafe extern "system" fn create_filter(
    input: ConstRaw,
    output: Raw,
    _name: *const c_char,
    init: vs3::InitFilter,
    get_frame: vs3::GetFrame,
    free: vs3::FreeFilter,
    mode: i32,
    _flags: i32,
    instance: Raw,
    core: Raw,
) {
    let adapter = Box::into_raw(Box::new(Adapter {
        instance,
        get_frame,
        free,
        mt_mode: if mode == vs3::FM_PARALLEL {
            MT_NICE_FILTER
        } else {
            MT_SERIALIZED
        },
        info: None,
    }));
    unsafe {
        init(
            input,
            output,
            &raw mut (*adapter).instance,
            adapter.cast(),
            core,
            table(),
        );
    }
    let created = (|| {
        let child = unsafe { args(input) }?.clip()?;
        let output = unsafe { args(output) }.filter(|output| output.error.is_none())?;
        let info = unsafe { (*adapter).info }?;
        let api = loaded();
        let data = match output.get(c"data") {
            Some(Value::Int(data)) => Some(*data),
            _ => None,
        };
        let mut value = AvsValue::VOID;
        let mut filter = std::ptr::null_mut();
        let clip = unsafe {
            (api.set_to_clip)(&raw mut value, child);
            let clip = (api.new_c_filter)(core, &raw mut filter, value, 1);
            (api.release_value)(value);
            clip
        };
        let filter = unsafe { &mut *filter };
        filter.vi = unsafe { output_info(&info, &filter.vi, data) };
        filter.get_frame = Some(filter_get_frame);
        filter.set_cache_hints = Some(filter_cache_hints);
        filter.free_filter = Some(filter_free);
        filter.user_data = adapter.cast();
        output.set(c"clip", Value::Clip(clip));
        Some(())
    })();
    if created.is_none() {
        let adapter = unsafe { Box::from_raw(adapter) };
        unsafe { (adapter.free)(adapter.instance, core, table()) };
    }
}

unsafe extern "system" fn filter_get_frame(filter: *mut FilterInfo, n: i32) -> Raw {
    let filter = unsafe { &mut *filter };
    let adapter = filter.user_data.cast::<Adapter>();
    let mut context = FrameContext {
        error: std::ptr::null(),
    };
    let mut data = std::ptr::null_mut();
    let mut frame = std::ptr::null();
    let previous_env = ENV.replace(filter.env);
    for reason in [vs3::AR_INITIAL, vs3::AR_ALL_FRAMES_READY] {
        frame = unsafe {
            ((*adapter).get_frame)(
                n,
                reason,
                &raw mut (*adapter).instance,
                &raw mut data,
                (&raw mut context).cast(),
                filter.env,
                table(),
            )
        };
        if !frame.is_null() || !context.error.is_null() {
            break;
        }
    }
    ENV.set(previous_env);
    if frame.is_null() {
        filter.error = if context.error.is_null() {
            c"SVPFlow: cannot get the frame".as_ptr()
        } else {
            context.error
        };
    }
    frame.cast_mut()
}

unsafe extern "system" fn filter_cache_hints(filter: *mut FilterInfo, hint: i32, _: i32) -> i32 {
    if hint == CACHE_GET_MTMODE {
        unsafe { (*(*filter).user_data.cast::<Adapter>()).mt_mode }
    } else {
        0
    }
}

unsafe extern "system" fn filter_free(filter: *mut FilterInfo) {
    let filter = unsafe { &*filter };
    let adapter = unsafe { Box::from_raw(filter.user_data.cast::<Adapter>()) };
    unsafe { (adapter.free)(adapter.instance, filter.env, table()) };
}

unsafe extern "system" fn set_filter_error(message: *const c_char, context: Raw) {
    unsafe {
        (*context.cast::<FrameContext>()).error = (loaded().save_string)(env(), message, -1);
    }
}

unsafe extern "system" fn get_frame_filter(n: i32, node: Raw, context: Raw) -> ConstRaw {
    let api = loaded();
    let frame = unsafe { (api.get_frame)(node, n) };
    if frame.is_null() {
        unsafe { (*context.cast::<FrameContext>()).error = (api.clip_get_error)(node) };
    }
    frame
}

unsafe extern "system" fn request_frame_filter(_n: i32, _node: Raw, _context: Raw) {}

unsafe extern "system" fn get_video_info(node: Raw) -> *const VideoInfo {
    static INFOS: Mutex<Option<HashMap<usize, Box<VideoInfo>>>> = Mutex::new(None);
    let info = unsafe { &*(loaded().get_video_info)(node) };
    let packed = info.pixel_type == CS_BGR32;
    let converted = VideoInfo {
        format: format(info.pixel_type).cast(),
        fps_num: i64::from(info.fps_num),
        fps_den: i64::from(info.fps_den),
        width: if packed {
            info.width * PACKED_BYTES
        } else {
            info.width
        },
        height: info.height,
        num_frames: info.num_frames,
        flags: 0,
    };
    let mut infos = INFOS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let slot = infos
        .get_or_insert_with(HashMap::new)
        .entry(node as usize)
        .or_insert_with(|| Box::new(converted));
    **slot = converted;
    &raw const **slot
}

unsafe extern "system" fn get_format_preset(id: i32, _core: Raw) -> *const Format {
    vs3::preset(id).map_or(std::ptr::null(), vs3::intern)
}

unsafe extern "system" fn new_video_frame(
    format: *const Format,
    width: i32,
    height: i32,
    source: ConstRaw,
    core: Raw,
) -> Raw {
    let pixel_type = unsafe { pixel_type(format.cast()) };
    let info = AvsVideoInfo {
        width: if pixel_type == CS_BGR32 {
            width / PACKED_BYTES
        } else {
            width
        },
        height,
        fps_num: 0,
        fps_den: 0,
        num_frames: 0,
        pixel_type,
        audio_samples_per_second: 0,
        sample_type: 0,
        num_audio_samples: 0,
        nchannels: 0,
        image_type: 0,
    };
    unsafe { (loaded().new_video_frame_p)(core, &raw const info, source) }
}

unsafe extern "system" fn copy_frame(frame: ConstRaw, core: Raw) -> Raw {
    let api = loaded();
    unsafe {
        let mut copy = (api.copy_video_frame)(frame.cast_mut());
        (api.make_writable)(core, &raw mut copy);
        copy
    }
}

unsafe extern "system" fn get_core_info(core: Raw) -> *const CoreInfo {
    thread_local! {
        static INFO: Cell<CoreInfo> = const { Cell::new(CoreInfo {
            version: std::ptr::null(),
            core: 0,
            api: 0,
            num_threads: 0,
            max_framebuffer: 0,
            used_framebuffer: 0,
        }) };
    }
    let threads = unsafe { (loaded().get_env_property)(core, AEP_THREADPOOL_THREADS) };
    let mut info = INFO.get();
    info.num_threads = i32::try_from(threads).unwrap_or(0);
    INFO.set(info);
    INFO.with(Cell::as_ptr)
}

fn plane(index: i32) -> i32 {
    usize::try_from(index)
        .ok()
        .and_then(|index| PLANES.get(index).copied())
        .unwrap_or(0)
}

unsafe extern "system" fn free_frame(frame: ConstRaw) {
    unsafe { (loaded().release_video_frame)(frame.cast_mut()) };
}

unsafe extern "system" fn free_node(node: Raw) {
    unsafe { (loaded().release_clip)(node) };
}

unsafe extern "system" fn get_stride(frame: ConstRaw, index: i32) -> i32 {
    unsafe { (loaded().get_pitch_p)(frame, plane(index)) }
}

unsafe extern "system" fn get_read_ptr(frame: ConstRaw, index: i32) -> *const u8 {
    unsafe { (loaded().get_read_ptr_p)(frame, plane(index)) }
}

unsafe extern "system" fn get_write_ptr(frame: Raw, index: i32) -> *mut u8 {
    unsafe { (loaded().get_write_ptr_p)(frame, plane(index)) }
}

unsafe extern "system" fn get_frame_props_ro(frame: ConstRaw) -> ConstRaw {
    unsafe { (loaded().get_frame_props_ro)(env(), frame) }
}

unsafe extern "system" fn get_frame_props_rw(frame: Raw) -> Raw {
    unsafe { (loaded().get_frame_props_rw)(env(), frame) }
}

fn table() -> ConstRaw {
    static TABLE: OnceLock<Box<[usize]>> = OnceLock::new();
    TABLE
        .get_or_init(|| {
            vs3::table(&[
                (vs3::GET_CORE_INFO, get_core_info as *const () as usize),
                (vs3::FREE_FRAME, free_frame as *const () as usize),
                (vs3::FREE_NODE, free_node as *const () as usize),
                (vs3::NEW_VIDEO_FRAME, new_video_frame as *const () as usize),
                (vs3::COPY_FRAME, copy_frame as *const () as usize),
                (vs3::CREATE_FILTER, create_filter as *const () as usize),
                (vs3::SET_ERROR, set_error as *const () as usize),
                (
                    vs3::SET_FILTER_ERROR,
                    set_filter_error as *const () as usize,
                ),
                (
                    vs3::GET_FORMAT_PRESET,
                    get_format_preset as *const () as usize,
                ),
                (
                    vs3::GET_FRAME_FILTER,
                    get_frame_filter as *const () as usize,
                ),
                (
                    vs3::REQUEST_FRAME_FILTER,
                    request_frame_filter as *const () as usize,
                ),
                (vs3::GET_STRIDE, get_stride as *const () as usize),
                (vs3::GET_READ_PTR, get_read_ptr as *const () as usize),
                (vs3::GET_WRITE_PTR, get_write_ptr as *const () as usize),
                (vs3::GET_VIDEO_INFO, get_video_info as *const () as usize),
                (vs3::SET_VIDEO_INFO, set_video_info as *const () as usize),
                (
                    vs3::GET_FRAME_PROPS_RO,
                    get_frame_props_ro as *const () as usize,
                ),
                (
                    vs3::GET_FRAME_PROPS_RW,
                    get_frame_props_rw as *const () as usize,
                ),
                (vs3::PROP_GET_INT, prop_get_int as *const () as usize),
                (vs3::PROP_GET_FLOAT, prop_get_float as *const () as usize),
                (vs3::PROP_GET_DATA, prop_get_data as *const () as usize),
                (
                    vs3::PROP_GET_DATA_SIZE,
                    prop_get_data_size as *const () as usize,
                ),
                (vs3::PROP_GET_NODE, prop_get_node as *const () as usize),
                (vs3::PROP_SET_INT, prop_set_int as *const () as usize),
            ])
        })
        .as_ptr()
        .cast()
}

unsafe fn argument(api: &Api, env: Raw, input: &mut Map, arg: &Arg, value: AvsValue) {
    let value = match u8::try_from(value.kind).unwrap_or(0) {
        b'c' => {
            let clip = unsafe { (api.take_clip)(value, env) };
            let info = unsafe { &*(api.get_video_info)(clip) };
            if let Some(data) = arg.data
                && info.audio_samples_per_second == 0
                && info.nchannels == 0
            {
                input.set(data, Value::Int(info.num_audio_samples));
            }
            Value::Clip(clip)
        }
        b's' => Value::Data(unsafe { CStr::from_ptr(value.payload.string) }.to_owned()),
        b'i' => Value::Int(i64::from(unsafe { value.payload.integer })),
        b'f' => Value::Float(f64::from(unsafe { value.payload.floating })),
        b'b' => Value::Int(i64::from(unsafe { value.payload.boolean } != 0)),
        _ => return,
    };
    input.set(arg.key, value);
}

unsafe extern "system" fn apply(env: Raw, args: AvsValue, user_data: Raw) -> AvsValue {
    let api = loaded();
    let function = unsafe { &*user_data.cast::<Function>() };
    let mut input = Map::default();
    let mut output = Map::default();
    let values = if args.kind == i16::from(b'a') {
        let len = usize::try_from(args.array_size).unwrap_or(0);
        unsafe { std::slice::from_raw_parts(args.payload.array, len) }
    } else {
        &[]
    };
    for (arg, &value) in function.args.iter().zip(values) {
        unsafe { argument(api, env, &mut input, arg, value) };
    }
    let previous_env = ENV.replace(env);
    let previous = ARGS.replace([
        (&raw const input).cast(),
        (&raw const output).cast::<c_void>(),
    ]);
    unsafe {
        (function.create)(
            (&raw const input).cast(),
            (&raw mut output).cast(),
            std::ptr::null_mut(),
            env,
            table(),
        );
    }
    ARGS.set(previous);
    ENV.set(previous_env);
    let mut result = AvsValue::VOID;
    if let Some(error) = &output.error {
        result.kind = i16::from(b'e');
        result.payload.string = unsafe { (api.save_string)(env, error.as_ptr(), -1) };
    } else if let Some(Value::Clip(clip)) = output.get(c"clip") {
        unsafe { (api.set_to_clip)(&raw mut result, *clip) };
    }
    result
}

pub unsafe fn init(
    env: Raw,
    functions: &'static [Function],
    description: &'static CStr,
) -> *const c_char {
    if let Some(api) = api() {
        for function in functions {
            unsafe {
                (api.add_function)(
                    env,
                    function.name.as_ptr(),
                    function.params.as_ptr(),
                    apply,
                    std::ptr::from_ref(function).cast_mut().cast(),
                );
            }
        }
    }
    description.as_ptr()
}
