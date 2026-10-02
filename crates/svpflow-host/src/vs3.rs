use std::ffi::{c_char, c_void};
use std::sync::Mutex;

pub type Raw = *mut c_void;
pub type ConstRaw = *const c_void;

pub type Create = unsafe extern "system" fn(ConstRaw, Raw, Raw, Raw, ConstRaw) -> isize;
pub type Config =
    unsafe extern "system" fn(*const c_char, *const c_char, *const c_char, i32, i32, Raw);
pub type Register = unsafe extern "system" fn(*const c_char, *const c_char, Create, Raw, Raw);
pub type GetCoreInfo = unsafe extern "system" fn(Raw) -> *const CoreInfo;
pub type CreateFilter = unsafe extern "system" fn(
    ConstRaw,
    Raw,
    *const c_char,
    InitFilter,
    GetFrame,
    FreeFilter,
    i32,
    i32,
    Raw,
    Raw,
);
pub type SetError = unsafe extern "system" fn(Raw, *const c_char);
pub type SetFilterError = unsafe extern "system" fn(*const c_char, Raw);
pub type GetFormatPreset = unsafe extern "system" fn(i32, Raw) -> *const Format;
pub type GetVideoInfo = unsafe extern "system" fn(Raw) -> *const VideoInfo;
pub type SetVideoInfo = unsafe extern "system" fn(*const VideoInfo, i32, Raw);
pub type PropGetInt = unsafe extern "system" fn(ConstRaw, *const c_char, i32, *mut i32) -> i64;
pub type PropGetFloat = unsafe extern "system" fn(ConstRaw, *const c_char, i32, *mut i32) -> f64;
pub type PropGetData =
    unsafe extern "system" fn(ConstRaw, *const c_char, i32, *mut i32) -> *const c_char;
pub type PropGetDataSize = unsafe extern "system" fn(ConstRaw, *const c_char, i32, *mut i32) -> i32;
pub type PropGetNode = unsafe extern "system" fn(ConstRaw, *const c_char, i32, *mut i32) -> Raw;
pub type PropSetInt = unsafe extern "system" fn(Raw, *const c_char, i64, i32) -> i32;
pub type GetFrame =
    unsafe extern "system" fn(i32, i32, *mut Raw, *mut Raw, Raw, Raw, ConstRaw) -> ConstRaw;
pub type FreeFilter = unsafe extern "system" fn(Raw, Raw, ConstRaw);
pub type InitFilter = unsafe extern "system" fn(ConstRaw, Raw, *mut Raw, Raw, Raw, ConstRaw);
pub type FreeFrame = unsafe extern "system" fn(ConstRaw);
pub type FreeNode = unsafe extern "system" fn(Raw);
pub type GetFrameFilter = unsafe extern "system" fn(i32, Raw, Raw) -> ConstRaw;
pub type RequestFrameFilter = unsafe extern "system" fn(i32, Raw, Raw);
pub type CopyFrame = unsafe extern "system" fn(ConstRaw, Raw) -> Raw;
pub type NewVideoFrame = unsafe extern "system" fn(ConstRaw, i32, i32, ConstRaw, Raw) -> Raw;
pub type GetFramePropsRo = unsafe extern "system" fn(ConstRaw) -> ConstRaw;
pub type GetFramePropsRw = unsafe extern "system" fn(Raw) -> Raw;
pub type GetStride = unsafe extern "system" fn(ConstRaw, i32) -> i32;
pub type GetReadPtr = unsafe extern "system" fn(ConstRaw, i32) -> *const u8;
pub type GetWritePtr = unsafe extern "system" fn(Raw, i32) -> *mut u8;

const fn slot(index: usize) -> usize {
    index * size_of::<usize>()
}

pub const GET_CORE_INFO: usize = slot(2);
pub const FREE_FRAME: usize = slot(6);
pub const FREE_NODE: usize = slot(7);
pub const NEW_VIDEO_FRAME: usize = slot(9);
pub const COPY_FRAME: usize = slot(10);
pub const CREATE_FILTER: usize = slot(17);
pub const SET_ERROR: usize = slot(18);
pub const SET_FILTER_ERROR: usize = slot(20);
pub const GET_FORMAT_PRESET: usize = slot(22);
pub const GET_FRAME_FILTER: usize = slot(26);
pub const REQUEST_FRAME_FILTER: usize = slot(27);
pub const GET_STRIDE: usize = slot(30);
pub const GET_READ_PTR: usize = slot(31);
pub const GET_WRITE_PTR: usize = slot(32);
pub const GET_VIDEO_INFO: usize = slot(38);
pub const SET_VIDEO_INFO: usize = slot(39);
pub const GET_FRAME_PROPS_RO: usize = slot(43);
pub const GET_FRAME_PROPS_RW: usize = slot(44);
pub const PROP_GET_INT: usize = slot(49);
pub const PROP_GET_FLOAT: usize = slot(50);
pub const PROP_GET_DATA: usize = slot(51);
pub const PROP_GET_DATA_SIZE: usize = slot(52);
pub const PROP_GET_NODE: usize = slot(53);
pub const PROP_SET_INT: usize = slot(57);

const SLOTS: usize = 110;

pub const AR_INITIAL: i32 = 0;
pub const AR_ALL_FRAMES_READY: i32 = 2;
pub const FM_PARALLEL: i32 = 100;

pub const CM_GRAY: i32 = 1_000_000;
pub const CM_RGB: i32 = 2_000_000;
pub const CM_YUV: i32 = 3_000_000;

pub const PF_GRAY8: i32 = CM_GRAY + 10;
pub const PF_YUV420P8: i32 = CM_YUV + 10;
pub const PF_YUV444P8: i32 = CM_YUV + 12;
pub const PF_YUV420P10: i32 = CM_YUV + 19;
pub const PF_YUV420P16: i32 = CM_YUV + 22;

pub const PRESETS: [(i32, i32, i32, i32, i32, i32); 20] = [
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

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Format {
    pub name: [u8; 32],
    pub id: i32,
    pub color_family: i32,
    pub sample_type: i32,
    pub bits_per_sample: i32,
    pub bytes_per_sample: i32,
    pub sub_sampling_w: i32,
    pub sub_sampling_h: i32,
    pub num_planes: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VideoInfo {
    pub format: ConstRaw,
    pub fps_num: i64,
    pub fps_den: i64,
    pub width: i32,
    pub height: i32,
    pub num_frames: i32,
    pub flags: i32,
}

unsafe impl Send for VideoInfo {}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CoreInfo {
    pub version: *const c_char,
    pub core: i32,
    pub api: i32,
    pub num_threads: i32,
    pub max_framebuffer: i64,
    pub used_framebuffer: i64,
}

impl VideoInfo {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            format: std::ptr::null(),
            fps_num: 0,
            fps_den: 0,
            width: 0,
            height: 0,
            num_frames: 0,
            flags: 0,
        }
    }
}

#[must_use]
pub fn preset(id: i32) -> Option<Format> {
    let &(id, family, sample_type, bits, sub_sampling_w, sub_sampling_h) =
        PRESETS.iter().find(|preset| preset.0 == id)?;
    Some(Format {
        name: [0; 32],
        id,
        color_family: family * CM_GRAY,
        sample_type,
        bits_per_sample: bits,
        bytes_per_sample: (bits + 7) / 8,
        sub_sampling_w,
        sub_sampling_h,
        num_planes: if family == 1 { 1 } else { 3 },
    })
}

#[allow(clippy::vec_box)]
pub fn intern(format: Format) -> *const Format {
    static FORMATS: Mutex<Vec<Box<Format>>> = Mutex::new(Vec::new());
    let mut formats = FORMATS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let key = |f: &Format| {
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

extern "C" fn missing() {
    std::process::abort();
}

#[must_use]
pub fn table(entries: &[(usize, usize)]) -> Box<[usize]> {
    let mut table = vec![missing as *const () as usize; SLOTS].into_boxed_slice();
    for &(offset, function) in entries {
        table[offset / size_of::<usize>()] = function;
    }
    table
}

pub unsafe fn table_fn<T: Copy>(table: ConstRaw, offset: usize) -> Option<T> {
    if table.is_null() {
        return None;
    }
    Some(unsafe { table.cast::<u8>().add(offset).cast::<T>().read_unaligned() })
}

pub unsafe fn free_node(node: Raw, vsapi: ConstRaw) {
    unsafe { free_nodes([node], vsapi) };
}

pub unsafe fn free_nodes<const N: usize>(nodes: [Raw; N], vsapi: ConstRaw) {
    let Some(free_node) = (unsafe { table_fn::<FreeNode>(vsapi, FREE_NODE) }) else {
        return;
    };
    for node in nodes {
        if !node.is_null() {
            unsafe { free_node(node) };
        }
    }
}

pub unsafe fn set_error(output: Raw, vsapi: ConstRaw, message: *const c_char) {
    if let Some(set_error) = unsafe { table_fn::<SetError>(vsapi, SET_ERROR) } {
        unsafe { set_error(output, message) };
    }
}
