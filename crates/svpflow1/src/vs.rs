use std::ffi::{c_char, c_void};

pub(crate) type Raw = *mut c_void;
pub(crate) type ConstRaw = *const c_void;

pub(crate) type Create = unsafe extern "system" fn(ConstRaw, Raw, Raw, Raw, ConstRaw) -> isize;
pub(crate) type Config =
    unsafe extern "system" fn(*const c_char, *const c_char, *const c_char, i32, i32, Raw);
pub(crate) type Register =
    unsafe extern "system" fn(*const c_char, *const c_char, Create, Raw, Raw);
pub(crate) type CreateFilter = unsafe extern "system" fn(
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
pub(crate) type SetError = unsafe extern "system" fn(Raw, *const c_char);
pub(crate) type GetVideoInfo = unsafe extern "system" fn(Raw) -> *const VideoInfo;
pub(crate) type SetVideoInfo = unsafe extern "system" fn(*const VideoInfo, i32, Raw);
pub(crate) type PropGetInt =
    unsafe extern "system" fn(ConstRaw, *const c_char, i32, *mut i32) -> i64;
pub(crate) type PropGetData =
    unsafe extern "system" fn(ConstRaw, *const c_char, i32, *mut i32) -> *const c_char;
pub(crate) type PropGetDataSize =
    unsafe extern "system" fn(ConstRaw, *const c_char, i32, *mut i32) -> i32;
pub(crate) type PropGetNode =
    unsafe extern "system" fn(ConstRaw, *const c_char, i32, *mut i32) -> Raw;
pub(crate) type PropSetInt = unsafe extern "system" fn(Raw, *const c_char, i64, i32) -> i32;
pub(crate) type GetFrame =
    unsafe extern "system" fn(i32, i32, *mut Raw, *mut Raw, Raw, Raw, ConstRaw) -> ConstRaw;
pub(crate) type FreeFilter = unsafe extern "system" fn(Raw, Raw, ConstRaw);
pub(crate) type InitFilter = unsafe extern "system" fn(ConstRaw, Raw, *mut Raw, Raw, Raw, ConstRaw);
pub(crate) type FreeFrame = unsafe extern "system" fn(ConstRaw);
pub(crate) type FreeNode = unsafe extern "system" fn(Raw);
pub(crate) type GetFrameFilter = unsafe extern "system" fn(i32, Raw, Raw) -> ConstRaw;
pub(crate) type NewVideoFrame = unsafe extern "system" fn(ConstRaw, i32, i32, ConstRaw, Raw) -> Raw;
pub(crate) type GetStride = unsafe extern "system" fn(ConstRaw, i32) -> i32;
pub(crate) type GetReadPtr = unsafe extern "system" fn(ConstRaw, i32) -> *const u8;
pub(crate) type GetWritePtr = unsafe extern "system" fn(Raw, i32) -> *mut u8;
pub(crate) type RequestFrameFilter = unsafe extern "system" fn(i32, Raw, Raw);

pub(crate) const FREE_FRAME: usize = 48;
pub(crate) const NEW_VIDEO_FRAME: usize = 72;
pub(crate) const FREE_NODE: usize = 56;
pub(crate) const CREATE_FILTER: usize = 136;
pub(crate) const SET_ERROR: usize = 144;
pub(crate) const GET_STRIDE: usize = 240;
pub(crate) const GET_READ_PTR: usize = 248;
pub(crate) const GET_WRITE_PTR: usize = 256;
pub(crate) const GET_FRAME_FILTER: usize = 208;
pub(crate) const REQUEST_FRAME_FILTER: usize = 216;
pub(crate) const GET_VIDEO_INFO: usize = 304;
pub(crate) const SET_VIDEO_INFO: usize = 312;
pub(crate) const PROP_GET_INT: usize = 392;

pub(crate) const PROP_GET_DATA: usize = 408;
pub(crate) const PROP_GET_DATA_SIZE: usize = 416;
pub(crate) const PROP_GET_NODE: usize = 424;
pub(crate) const PROP_SET_INT: usize = 456;

pub(crate) const AR_INITIAL: i32 = 0;
pub(crate) const AR_ALL_FRAMES_READY: i32 = 2;
pub(crate) const FM_PARALLEL: i32 = 100;

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct VideoInfo {
    pub(crate) format: ConstRaw,
    pub(crate) fps_num: i64,
    pub(crate) fps_den: i64,
    pub(crate) width: i32,
    pub(crate) height: i32,
    pub(crate) num_frames: i32,
    pub(crate) flags: i32,
}

pub(crate) unsafe fn table_fn<T: Copy>(table: ConstRaw, offset: usize) -> Option<T> {
    if table.is_null() {
        return None;
    }
    Some(unsafe { table.cast::<u8>().add(offset).cast::<T>().read_unaligned() })
}

pub(crate) unsafe fn free_node(node: Raw, vsapi: ConstRaw) {
    if node.is_null() {
        return;
    }
    if let Some(free_node) = unsafe { table_fn::<FreeNode>(vsapi, FREE_NODE) } {
        unsafe { free_node(node) };
    }
}

pub(crate) unsafe fn set_error(output: Raw, vsapi: ConstRaw, msg: *const c_char) {
    if let Some(set_error) = unsafe { table_fn::<SetError>(vsapi, SET_ERROR) } {
        unsafe { set_error(output, msg) };
    }
}
