use crate::vs;

const FORMAT_ID: usize = 32;
const SUBSAMPLING_W: usize = 52;
const SUBSAMPLING_H: usize = 56;
const YUV420P8: i32 = 3_000_010;
const YUV444P8: i32 = 3_000_012;

pub(crate) fn is_supported(info: &vs::VideoInfo) -> bool {
    matches!(unsafe { field(info, FORMAT_ID) }, YUV420P8 | YUV444P8)
}

pub(crate) fn chroma_divisors(info: &vs::VideoInfo) -> (usize, usize) {
    let shift = |offset| unsafe { field(info, offset) }.clamp(0, 1) as u32;
    (
        1usize << shift(SUBSAMPLING_W),
        1usize << shift(SUBSAMPLING_H),
    )
}

unsafe fn field(info: &vs::VideoInfo, offset: usize) -> i32 {
    if info.format.is_null() {
        return 0;
    }
    unsafe {
        info.format
            .cast::<u8>()
            .add(offset)
            .cast::<i32>()
            .read_unaligned()
    }
}
