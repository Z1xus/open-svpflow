#![allow(unsafe_code, clippy::missing_safety_doc)]

use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_void};
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::{Mutex, OnceLock};

type Raw = *mut c_void;
type ConstRaw = *const c_void;

pub type Create = unsafe extern "system" fn(ConstRaw, Raw, Raw, Raw, ConstRaw) -> isize;
type Init = unsafe extern "system" fn(ConstRaw, Raw, *mut Raw, Raw, Raw, ConstRaw);
type GetFrame3 =
    unsafe extern "system" fn(i32, i32, *mut Raw, *mut Raw, Raw, Raw, ConstRaw) -> ConstRaw;
type Free3 = unsafe extern "system" fn(Raw, Raw, ConstRaw);

pub struct Function {
    pub name: &'static CStr,
    pub args: &'static CStr,
    pub returns: &'static CStr,
    pub create: Create,
}

pub struct Plugin {
    pub identifier: &'static CStr,
    pub namespace: &'static CStr,
    pub name: &'static CStr,
    pub version: i32,
}

#[repr(C)]
struct PluginApi {
    get_api_version: unsafe extern "system" fn() -> i32,
    config_plugin: unsafe extern "system" fn(
        *const c_char,
        *const c_char,
        *const c_char,
        i32,
        i32,
        i32,
        Raw,
    ) -> i32,
    register_function: unsafe extern "system" fn(
        *const c_char,
        *const c_char,
        *const c_char,
        Create,
        Raw,
        Raw,
    ) -> i32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Format4 {
    color_family: i32,
    sample_type: i32,
    bits_per_sample: i32,
    bytes_per_sample: i32,
    sub_sampling_w: i32,
    sub_sampling_h: i32,
    num_planes: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VideoInfo4 {
    format: Format4,
    fps_num: i64,
    fps_den: i64,
    width: i32,
    height: i32,
    num_frames: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Format3 {
    name: [u8; 32],
    id: i32,
    color_family: i32,
    sample_type: i32,
    bits_per_sample: i32,
    bytes_per_sample: i32,
    sub_sampling_w: i32,
    sub_sampling_h: i32,
    num_planes: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VideoInfo3 {
    format: *const Format3,
    fps_num: i64,
    fps_den: i64,
    width: i32,
    height: i32,
    num_frames: i32,
    flags: i32,
}

unsafe impl Send for VideoInfo3 {}

#[repr(C)]
#[derive(Clone, Copy)]
struct CoreInfo {
    version: *const c_char,
    core: i32,
    api: i32,
    threads: i32,
    max_framebuffer: i64,
    used_framebuffer: i64,
}

#[repr(C)]
struct Dependency {
    source: Raw,
    pattern: i32,
}

const CREATE_VIDEO_FILTER: usize = 0;
const FREE_NODE: usize = 56;
const GET_VIDEO_INFO: usize = 80;
const NEW_VIDEO_FRAME: usize = 96;
const FREE_FRAME: usize = 128;
const COPY_FRAME: usize = 144;
const GET_FRAME_PROPS_RO: usize = 152;
const GET_FRAME_PROPS_RW: usize = 160;
const GET_STRIDE: usize = 168;
const GET_READ_PTR: usize = 176;
const GET_WRITE_PTR: usize = 184;
const GET_FRAME_FILTER: usize = 304;
const REQUEST_FRAME_FILTER: usize = 312;
const SET_FILTER_ERROR: usize = 336;
const MAP_SET_ERROR: usize = 408;
const MAP_GET_ERROR: usize = 416;
const MAP_NUM_KEYS: usize = 424;
const MAP_GET_KEY: usize = 432;
const MAP_NUM_ELEMENTS: usize = 448;
const MAP_GET_TYPE: usize = 456;
const MAP_GET_INT: usize = 472;
const MAP_SET_INT: usize = 496;
const MAP_GET_FLOAT: usize = 512;
const MAP_GET_DATA: usize = 552;
const MAP_GET_DATA_SIZE: usize = 560;
const MAP_GET_NODE: usize = 584;
const GET_CORE_INFO: usize = 808;

const PT_VIDEO_NODE: i32 = 3;
const TABLE_LEN: usize = 110;

static API4: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());

fn api() -> ConstRaw {
    API4.load(Ordering::Acquire)
}

unsafe fn f<T: Copy>(offset: usize) -> T {
    unsafe { api().cast::<u8>().add(offset).cast::<T>().read_unaligned() }
}

extern "C" fn missing() {
    std::process::abort();
}

fn table() -> ConstRaw {
    static TABLE: OnceLock<Box<[usize; TABLE_LEN]>> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut t = Box::new([missing as *const () as usize; TABLE_LEN]);
        let entries: [(usize, usize); 24] = [
            (16, get_core_info as *const () as usize),
            (48, free_frame as *const () as usize),
            (56, free_node as *const () as usize),
            (72, new_video_frame as *const () as usize),
            (80, copy_frame as *const () as usize),
            (136, create_filter as *const () as usize),
            (144, set_error as *const () as usize),
            (160, set_filter_error as *const () as usize),
            (176, get_format_preset as *const () as usize),
            (208, get_frame_filter as *const () as usize),
            (216, request_frame_filter as *const () as usize),
            (240, get_stride as *const () as usize),
            (248, get_read_ptr as *const () as usize),
            (256, get_write_ptr as *const () as usize),
            (304, get_video_info as *const () as usize),
            (312, set_video_info as *const () as usize),
            (344, get_frame_props_ro as *const () as usize),
            (352, get_frame_props_rw as *const () as usize),
            (392, prop_get_int as *const () as usize),
            (400, prop_get_float as *const () as usize),
            (408, prop_get_data as *const () as usize),
            (416, prop_get_data_size as *const () as usize),
            (424, prop_get_node as *const () as usize),
            (456, prop_set_int as *const () as usize),
        ];
        for (offset, function) in entries {
            t[offset / 8] = function;
        }
        t
    });
    table.as_ptr().cast()
}

const CM_GRAY: i32 = 1_000_000;
const CM_RGB: i32 = 2_000_000;
const CM_YUV: i32 = 3_000_000;

const PRESETS: [(i32, i32, i32, i32, i32, i32); 20] = [
    (CM_GRAY + 10, 1, 0, 8, 0, 0),
    (CM_GRAY + 11, 1, 0, 16, 0, 0),
    (CM_GRAY + 12, 1, 1, 16, 0, 0),
    (CM_GRAY + 13, 1, 1, 32, 0, 0),
    (CM_YUV + 10, 3, 0, 8, 1, 1),
    (CM_YUV + 11, 3, 0, 8, 1, 0),
    (CM_YUV + 12, 3, 0, 8, 0, 0),
    (CM_YUV + 13, 3, 0, 8, 2, 2),
    (CM_YUV + 14, 3, 0, 8, 2, 0),
    (CM_YUV + 15, 3, 0, 8, 0, 1),
    (CM_YUV + 16, 3, 0, 9, 1, 1),
    (CM_YUV + 17, 3, 0, 9, 1, 0),
    (CM_YUV + 18, 3, 0, 9, 0, 0),
    (CM_YUV + 19, 3, 0, 10, 1, 1),
    (CM_YUV + 20, 3, 0, 10, 1, 0),
    (CM_YUV + 21, 3, 0, 10, 0, 0),
    (CM_YUV + 22, 3, 0, 16, 1, 1),
    (CM_YUV + 23, 3, 0, 16, 1, 0),
    (CM_YUV + 24, 3, 0, 16, 0, 0),
    (CM_RGB + 10, 2, 0, 8, 0, 0),
];

fn format3(format: Format4) -> Format3 {
    let id = PRESETS
        .iter()
        .find(|p| {
            (p.1, p.2, p.3, p.4, p.5)
                == (
                    format.color_family,
                    format.sample_type,
                    format.bits_per_sample,
                    format.sub_sampling_w,
                    format.sub_sampling_h,
                )
        })
        .map_or(0, |p| p.0);
    let family = match format.color_family {
        1 => CM_GRAY,
        2 => CM_RGB,
        3 => CM_YUV,
        _ => 0,
    };
    Format3 {
        name: [0; 32],
        id,
        color_family: family,
        sample_type: format.sample_type,
        bits_per_sample: format.bits_per_sample,
        bytes_per_sample: format.bytes_per_sample,
        sub_sampling_w: format.sub_sampling_w,
        sub_sampling_h: format.sub_sampling_h,
        num_planes: format.num_planes,
    }
}

fn format4(format: &Format3) -> Format4 {
    Format4 {
        color_family: format.color_family / 1_000_000,
        sample_type: format.sample_type,
        bits_per_sample: format.bits_per_sample,
        bytes_per_sample: format.bytes_per_sample,
        sub_sampling_w: format.sub_sampling_w,
        sub_sampling_h: format.sub_sampling_h,
        num_planes: format.num_planes,
    }
}

#[allow(clippy::vec_box)]
fn stable_format(format: Format3) -> *const Format3 {
    static FORMATS: Mutex<Vec<Box<Format3>>> = Mutex::new(Vec::new());
    let mut formats = FORMATS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let key = |f: &Format3| {
        (
            f.id,
            f.color_family,
            f.sample_type,
            f.bits_per_sample,
            f.sub_sampling_w,
            f.sub_sampling_h,
        )
    };
    if let Some(found) = formats.iter().find(|f| key(f) == key(&format)) {
        return &raw const **found;
    }
    formats.push(Box::new(format));
    &raw const **formats.last().expect("pushed")
}

struct Adapter {
    instance: Raw,
    get_frame: GetFrame3,
    free: Free3,
    info: Option<VideoInfo4>,
}

unsafe extern "system" fn get_frame4(
    n: i32,
    reason: i32,
    instance: Raw,
    frame_data: *mut Raw,
    ctx: Raw,
    core: Raw,
    _: ConstRaw,
) -> ConstRaw {
    let adapter = unsafe { &mut *instance.cast::<Adapter>() };
    let reason = if reason == 1 { 2 } else { reason };
    unsafe {
        (adapter.get_frame)(
            n,
            reason,
            &raw mut adapter.instance,
            frame_data,
            ctx,
            core,
            table(),
        )
    }
}

unsafe extern "system" fn free4(instance: Raw, core: Raw, _: ConstRaw) {
    let adapter = unsafe { Box::from_raw(instance.cast::<Adapter>()) };
    unsafe { (adapter.free)(adapter.instance, core, table()) };
}

unsafe extern "system" fn create_filter(
    input: ConstRaw,
    output: Raw,
    name: *const c_char,
    init: Init,
    get_frame: GetFrame3,
    free: Free3,
    mode: i32,
    _flags: i32,
    instance: Raw,
    core: Raw,
) {
    let adapter = Box::into_raw(Box::new(Adapter {
        instance,
        get_frame,
        free,
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
    let failed = !unsafe {
        f::<unsafe extern "system" fn(ConstRaw) -> *const c_char>(MAP_GET_ERROR)(output)
    }
    .is_null();
    let Some(info) = (unsafe { (*adapter).info }).filter(|_| !failed) else {
        let adapter = unsafe { Box::from_raw(adapter) };
        unsafe { (adapter.free)(adapter.instance, core, table()) };
        return;
    };
    let dependencies = unsafe { input_nodes(input) };
    let mode = match mode {
        200 => 1,
        300 => 2,
        400 => 3,
        _ => 0,
    };
    unsafe {
        f::<
            unsafe extern "system" fn(
                Raw,
                *const c_char,
                *const VideoInfo4,
                GetFrame4,
                Free4,
                i32,
                *const Dependency,
                i32,
                Raw,
                Raw,
            ),
        >(CREATE_VIDEO_FILTER)(
            output,
            name,
            &raw const info,
            get_frame4,
            free4,
            mode,
            dependencies.as_ptr(),
            i32::try_from(dependencies.len()).unwrap_or(i32::MAX),
            adapter.cast(),
            core,
        );
    }
    for dependency in dependencies {
        unsafe { free_node(dependency.source) };
    }
}

type GetFrame4 = unsafe extern "system" fn(i32, i32, Raw, *mut Raw, Raw, Raw, ConstRaw) -> ConstRaw;
type Free4 = unsafe extern "system" fn(Raw, Raw, ConstRaw);

unsafe fn input_nodes(input: ConstRaw) -> Vec<Dependency> {
    let num_keys: unsafe extern "system" fn(ConstRaw) -> i32 = unsafe { f(MAP_NUM_KEYS) };
    let get_key: unsafe extern "system" fn(ConstRaw, i32) -> *const c_char =
        unsafe { f(MAP_GET_KEY) };
    let get_type: unsafe extern "system" fn(ConstRaw, *const c_char) -> i32 =
        unsafe { f(MAP_GET_TYPE) };
    let count: unsafe extern "system" fn(ConstRaw, *const c_char) -> i32 =
        unsafe { f(MAP_NUM_ELEMENTS) };
    let get_node: unsafe extern "system" fn(ConstRaw, *const c_char, i32, *mut i32) -> Raw =
        unsafe { f(MAP_GET_NODE) };
    let mut nodes = Vec::new();
    for index in 0..unsafe { num_keys(input) } {
        let key = unsafe { get_key(input, index) };
        if key.is_null() || unsafe { get_type(input, key) } != PT_VIDEO_NODE {
            continue;
        }
        for element in 0..unsafe { count(input, key) } {
            let mut error = 0;
            let node = unsafe { get_node(input, key, element, &raw mut error) };
            if error == 0 && !node.is_null() {
                nodes.push(Dependency {
                    source: node,
                    pattern: 0,
                });
            }
        }
    }
    nodes
}

unsafe extern "system" fn set_video_info(info: *const VideoInfo3, _count: i32, node: Raw) {
    let adapter = unsafe { &mut *node.cast::<Adapter>() };
    let info = unsafe { &*info };
    let format = if info.format.is_null() {
        Format4::default()
    } else {
        format4(unsafe { &*info.format })
    };
    adapter.info = Some(VideoInfo4 {
        format,
        fps_num: info.fps_num,
        fps_den: info.fps_den,
        width: info.width,
        height: info.height,
        num_frames: info.num_frames,
    });
}

unsafe extern "system" fn get_video_info(node: Raw) -> *const VideoInfo3 {
    static INFOS: Mutex<Option<HashMap<usize, Box<VideoInfo3>>>> = Mutex::new(None);
    let info =
        unsafe { &*f::<unsafe extern "system" fn(Raw) -> *const VideoInfo4>(GET_VIDEO_INFO)(node) };
    let converted = VideoInfo3 {
        format: stable_format(format3(info.format)),
        fps_num: info.fps_num,
        fps_den: info.fps_den,
        width: info.width,
        height: info.height,
        num_frames: info.num_frames,
        flags: 0,
    };
    let mut infos = INFOS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let slot = infos
        .get_or_insert_with(HashMap::new)
        .entry(node as *const () as usize)
        .or_insert_with(|| Box::new(converted));
    **slot = converted;
    &raw const **slot
}

unsafe extern "system" fn get_format_preset(id: i32, _core: Raw) -> *const Format3 {
    let Some(&(id, family, sample, bits, sw, sh)) = PRESETS.iter().find(|p| p.0 == id) else {
        return std::ptr::null();
    };
    let format = Format4 {
        color_family: family,
        sample_type: sample,
        bits_per_sample: bits,
        bytes_per_sample: (bits + 7) / 8,
        sub_sampling_w: sw,
        sub_sampling_h: sh,
        num_planes: if family == 1 { 1 } else { 3 },
    };
    let mut converted = format3(format);
    converted.id = id;
    stable_format(converted)
}

unsafe extern "system" fn new_video_frame(
    format: *const Format3,
    width: i32,
    height: i32,
    source: ConstRaw,
    core: Raw,
) -> Raw {
    let format = format4(unsafe { &*format });
    unsafe {
        f::<unsafe extern "system" fn(*const Format4, i32, i32, ConstRaw, Raw) -> Raw>(
            NEW_VIDEO_FRAME,
        )(&raw const format, width, height, source, core)
    }
}

unsafe extern "system" fn get_core_info(core: Raw) -> *const CoreInfo {
    thread_local! {
        static INFO: std::cell::Cell<CoreInfo> = const { std::cell::Cell::new(CoreInfo {
            version: std::ptr::null(),
            core: 0,
            api: 0,
            threads: 0,
            max_framebuffer: 0,
            used_framebuffer: 0,
        }) };
    }
    let mut info = INFO.get();
    unsafe {
        f::<unsafe extern "system" fn(Raw, *mut CoreInfo)>(GET_CORE_INFO)(core, &raw mut info);
    }
    INFO.set(info);
    INFO.with(std::cell::Cell::as_ptr)
}

unsafe extern "system" fn get_stride(frame: ConstRaw, plane: i32) -> i32 {
    let stride =
        unsafe { f::<unsafe extern "system" fn(ConstRaw, i32) -> isize>(GET_STRIDE)(frame, plane) };
    i32::try_from(stride).unwrap_or(0)
}

macro_rules! forward {
    ($name:ident, $offset:expr, ($($arg:ident: $ty:ty),*) -> $ret:ty) => {
        unsafe extern "system" fn $name($($arg: $ty),*) -> $ret {
            unsafe { f::<unsafe extern "system" fn($($ty),*) -> $ret>($offset)($($arg),*) }
        }
    };
}

forward!(free_frame, FREE_FRAME, (frame: ConstRaw) -> ());
forward!(free_node, FREE_NODE, (node: Raw) -> ());
forward!(copy_frame, COPY_FRAME, (frame: ConstRaw, core: Raw) -> Raw);
forward!(set_error, MAP_SET_ERROR, (map: Raw, message: *const c_char) -> ());
forward!(set_filter_error, SET_FILTER_ERROR, (message: *const c_char, ctx: Raw) -> ());
forward!(get_frame_filter, GET_FRAME_FILTER, (n: i32, node: Raw, ctx: Raw) -> ConstRaw);
forward!(request_frame_filter, REQUEST_FRAME_FILTER, (n: i32, node: Raw, ctx: Raw) -> ());
forward!(get_read_ptr, GET_READ_PTR, (frame: ConstRaw, plane: i32) -> *const u8);
forward!(get_write_ptr, GET_WRITE_PTR, (frame: Raw, plane: i32) -> *mut u8);
forward!(get_frame_props_ro, GET_FRAME_PROPS_RO, (frame: ConstRaw) -> ConstRaw);
forward!(get_frame_props_rw, GET_FRAME_PROPS_RW, (frame: Raw) -> Raw);
forward!(prop_get_int, MAP_GET_INT, (map: ConstRaw, key: *const c_char, index: i32, error: *mut i32) -> i64);
forward!(prop_get_float, MAP_GET_FLOAT, (map: ConstRaw, key: *const c_char, index: i32, error: *mut i32) -> f64);
forward!(prop_get_data, MAP_GET_DATA, (map: ConstRaw, key: *const c_char, index: i32, error: *mut i32) -> *const c_char);
forward!(prop_get_data_size, MAP_GET_DATA_SIZE, (map: ConstRaw, key: *const c_char, index: i32, error: *mut i32) -> i32);
forward!(prop_get_node, MAP_GET_NODE, (map: ConstRaw, key: *const c_char, index: i32, error: *mut i32) -> Raw);
forward!(prop_set_int, MAP_SET_INT, (map: Raw, key: *const c_char, value: i64, append: i32) -> i32);

fn registry() -> &'static Mutex<Vec<(usize, Create)>> {
    static REGISTRY: Mutex<Vec<(usize, Create)>> = Mutex::new(Vec::new());
    &REGISTRY
}

unsafe extern "system" fn call(
    input: ConstRaw,
    output: Raw,
    data: Raw,
    core: Raw,
    vsapi: ConstRaw,
) -> isize {
    API4.store(vsapi.cast_mut(), Ordering::Release);
    let create = registry()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .find(|(key, _)| *key == data as *const () as usize)
        .map(|&(_, create)| create);
    if let Some(create) = create {
        unsafe { create(input, output, std::ptr::null_mut(), core, table()) };
    }
    0
}

fn args4(args: &CStr) -> CString {
    let text = args.to_string_lossy();
    let converted: Vec<String> = text
        .split(';')
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut fields: Vec<&str> = part.split(':').collect();
            if fields.get(1) == Some(&"clip") {
                fields[1] = "vnode";
            }
            fields.join(":")
        })
        .collect();
    CString::new(converted.join(";") + ";").unwrap_or_default()
}

pub unsafe fn init(plugin: Raw, api: ConstRaw, config: &Plugin, functions: &[Function]) {
    if api.is_null() {
        return;
    }
    let api = unsafe { &*api.cast::<PluginApi>() };
    unsafe {
        (api.config_plugin)(
            config.identifier.as_ptr(),
            config.namespace.as_ptr(),
            config.name.as_ptr(),
            config.version,
            4 << 16,
            0,
            plugin,
        );
    }
    let mut registry = registry()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for (index, function) in functions.iter().enumerate() {
        let key = index + 1;
        registry.push((key, function.create));
        let args = args4(function.args);
        unsafe {
            (api.register_function)(
                function.name.as_ptr(),
                args.as_ptr(),
                function.returns.as_ptr(),
                call,
                key as Raw,
                plugin,
            );
        }
    }
}
