use crate::vs;

const FORMAT_ID: usize = 32;
const YUV420P8_ALT: i32 = 1_000_010;
const YUV420P8: i32 = 3_000_010;
const YUV420P10: i32 = 3_000_019;
const YUV420P16: i32 = 3_000_022;

pub(crate) fn is_cpu_source(info: &vs::VideoInfo) -> bool {
    is_yuv420p8(info)
}

pub(crate) fn is_yuv420_source(info: &vs::VideoInfo) -> bool {
    matches!(
        unsafe { format_id(info) },
        YUV420P8_ALT | YUV420P8 | YUV420P10 | YUV420P16
    )
}

pub(crate) fn is_yuv420p8(info: &vs::VideoInfo) -> bool {
    unsafe { format_id(info) == YUV420P8 }
}

pub(crate) fn needs_8bit_vec_src(info: &vs::VideoInfo) -> bool {
    matches!(unsafe { format_id(info) }, YUV420P10 | YUV420P16)
}

pub(crate) fn source_unmodified_depth(info: &vs::VideoInfo) -> i32 {
    match unsafe { format_id(info) } {
        YUV420P10 => 10,
        YUV420P16 => 16,
        _ => 0,
    }
}

pub(crate) fn source_depth(info: &vs::VideoInfo) -> i32 {
    source_unmodified_depth(info)
}

unsafe fn format_id(info: &vs::VideoInfo) -> i32 {
    if info.format.is_null() {
        return 0;
    }
    unsafe {
        info.format
            .cast::<u8>()
            .add(FORMAT_ID)
            .cast::<i32>()
            .read_unaligned()
    }
}
