use std::ptr;

use crate::{strings, vs};

pub(crate) struct PlaneApi {
    new_video_frame: vs::NewVideoFrame,
    free_frame: Option<vs::FreeFrame>,
    get_frame_props_ro: Option<vs::GetFramePropsRo>,
    get_frame_props_rw: Option<vs::GetFramePropsRw>,
    prop_get_int: Option<vs::PropGetInt>,
    prop_get_data: Option<vs::PropGetData>,
    prop_get_data_size: Option<vs::PropGetDataSize>,
    prop_set_int: Option<vs::PropSetInt>,
    get_stride: vs::GetStride,
    get_read_ptr: vs::GetReadPtr,
    get_write_ptr: vs::GetWritePtr,
}

#[derive(Clone, Copy)]
pub(crate) struct TimingProps {
    duration_num: i64,
    duration_den: i64,
    pts: i64,
}

#[derive(Clone, Copy)]
struct PlaneCopy {
    source: vs::ConstRaw,
    output: vs::Raw,
    plane: i32,
    width: usize,
    height: usize,
    pad_x: i32,
    pad_y: i32,
    bytes_per_sample: usize,
}

impl PlaneApi {
    pub(crate) unsafe fn load(vsapi: vs::ConstRaw) -> Option<Self> {
        Some(Self {
            new_video_frame: unsafe { vs::table_fn(vsapi, vs::NEW_VIDEO_FRAME) }?,
            free_frame: unsafe { vs::table_fn(vsapi, vs::FREE_FRAME) },
            get_frame_props_ro: unsafe { vs::table_fn(vsapi, vs::GET_FRAME_PROPS_RO) },
            get_frame_props_rw: unsafe { vs::table_fn(vsapi, vs::GET_FRAME_PROPS_RW) },
            prop_get_int: unsafe { vs::table_fn(vsapi, vs::PROP_GET_INT) },
            prop_get_data: unsafe { vs::table_fn(vsapi, vs::PROP_GET_DATA) },
            prop_get_data_size: unsafe { vs::table_fn(vsapi, vs::PROP_GET_DATA_SIZE) },
            prop_set_int: unsafe { vs::table_fn(vsapi, vs::PROP_SET_INT) },
            get_stride: unsafe { vs::table_fn(vsapi, vs::GET_STRIDE) }?,
            get_read_ptr: unsafe { vs::table_fn(vsapi, vs::GET_READ_PTR) }?,
            get_write_ptr: unsafe { vs::table_fn(vsapi, vs::GET_WRITE_PTR) }?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn selected_copy(
        &self,
        source: vs::ConstRaw,
        timing_source: vs::ConstRaw,
        next_source: vs::ConstRaw,
        core: vs::Raw,
        input: &vs::VideoInfo,
        output: &vs::VideoInfo,
        padding: (i32, i32),
        bytes_per_sample: usize,
        raw_phase: f64,
        ratio: f64,
    ) -> vs::ConstRaw {
        if source.is_null() || input.format.is_null() {
            return source;
        }
        let frame = unsafe {
            (self.new_video_frame)(input.format, output.width, output.height, source, core)
        };
        if frame.is_null() {
            return source;
        }
        if !unsafe { self.copy_planes(source, frame, input, padding, bytes_per_sample) } {
            if let Some(free_frame) = self.free_frame {
                unsafe { free_frame(frame.cast_const()) };
            }
            return source;
        }
        unsafe {
            self.copy_interpolated_timing(timing_source, next_source, frame, raw_phase, ratio);
        };
        if let Some(free_frame) = self.free_frame {
            unsafe { free_frame(source) };
        }
        frame.cast_const()
    }

    pub(crate) unsafe fn new_frame(
        &self,
        format: vs::ConstRaw,
        width: i32,
        height: i32,
        prop_src: vs::ConstRaw,
        core: vs::Raw,
    ) -> Option<vs::Raw> {
        let frame = unsafe { (self.new_video_frame)(format, width, height, prop_src, core) };
        (!frame.is_null()).then_some(frame)
    }

    pub(crate) unsafe fn free(&self, frame: vs::ConstRaw) {
        if let Some(free_frame) = self.free_frame {
            unsafe { free_frame(frame) };
        }
    }

    pub(crate) unsafe fn copy_interpolated_timing(
        &self,
        source: vs::ConstRaw,
        next_source: vs::ConstRaw,
        output: vs::Raw,
        raw_phase: f64,
        ratio: f64,
    ) {
        let mut timing = unsafe { self.read_timing(source) };
        if ratio > 0.0 && ratio.is_finite() && raw_phase.is_finite() {
            let current = duration_seconds(timing.duration_num, timing.duration_den);
            let next = unsafe { self.read_timing(next_source) };
            let next = duration_seconds(next.duration_num, next.duration_den);
            let end = 256.0 / ratio + raw_phase;
            let duration = if end < 256.0 {
                current / ratio
            } else {
                ((256.0 - raw_phase) * current + (end - 256.0) * next) / 256.0
            };
            timing.duration_num = trunc_f64_i64(duration * 10_000_000.0);
            timing.duration_den = 10_000_000;
            if timing.pts != 0 {
                timing.pts =
                    trunc_f64_i64(f64_i64(timing.pts) + current * 1_000_000.0 * raw_phase / 256.0);
            }
        }
        unsafe { self.write_timing(output, timing) };
    }

    pub(crate) unsafe fn scene_change_next(&self, frame: vs::ConstRaw) -> bool {
        let (Some(get_props), Some(get_int)) = (self.get_frame_props_ro, self.prop_get_int) else {
            return false;
        };
        let props = unsafe { get_props(frame) };
        if props.is_null() {
            return false;
        }
        let mut err = 0;
        let value = unsafe {
            get_int(
                props,
                strings::SCENE_CHANGE_NEXT.as_ptr().cast(),
                0,
                &raw mut err,
            )
        };
        value != 0
    }

    pub(crate) unsafe fn dovi_changed(&self, left: vs::ConstRaw, right: vs::ConstRaw) -> bool {
        let left = unsafe { self.read_data(left, strings::DOVI) };
        let right = unsafe { self.read_data(right, strings::DOVI) };
        dovi_changed(&left, &right)
    }

    pub(crate) unsafe fn read_plane(
        &self,
        frame: vs::ConstRaw,
        plane: i32,
        height: usize,
    ) -> Option<(*const u8, usize, usize)> {
        let ptr = unsafe { (self.get_read_ptr)(frame, plane) };
        let stride = usize_positive(unsafe { (self.get_stride)(frame, plane) })?;
        (!ptr.is_null()).then_some((ptr, stride, stride.saturating_mul(height)))
    }

    pub(crate) unsafe fn write_plane(
        &self,
        frame: vs::Raw,
        plane: i32,
        height: usize,
    ) -> Option<(*mut u8, usize, usize)> {
        let ptr = unsafe { (self.get_write_ptr)(frame, plane) };
        let stride = usize_positive(unsafe { (self.get_stride)(frame.cast_const(), plane) })?;
        (!ptr.is_null()).then_some((ptr, stride, stride.saturating_mul(height)))
    }

    unsafe fn copy_planes(
        &self,
        source: vs::ConstRaw,
        output: vs::Raw,
        info: &vs::VideoInfo,
        padding: (i32, i32),
        bytes_per_sample: usize,
    ) -> bool {
        let (pad_x, pad_y) = padding;
        let Some(width) = usize_positive(info.width) else {
            return false;
        };
        let Some(height) = usize_positive(info.height) else {
            return false;
        };
        let y = unsafe {
            self.copy_plane(PlaneCopy {
                source,
                output,
                plane: 0,
                width,
                height,
                pad_x,
                pad_y,
                bytes_per_sample,
            })
        };
        let u = unsafe {
            self.copy_plane(PlaneCopy {
                source,
                output,
                plane: 1,
                width: width / 2,
                height: height / 2,
                pad_x: pad_x / 2,
                pad_y: pad_y / 2,
                bytes_per_sample,
            })
        };
        let v = unsafe {
            self.copy_plane(PlaneCopy {
                source,
                output,
                plane: 2,
                width: width / 2,
                height: height / 2,
                pad_x: pad_x / 2,
                pad_y: pad_y / 2,
                bytes_per_sample,
            })
        };
        y && u && v
    }

    unsafe fn copy_plane(&self, copy: PlaneCopy) -> bool {
        let src = unsafe { (self.get_read_ptr)(copy.source, copy.plane) };
        let dst = unsafe { (self.get_write_ptr)(copy.output, copy.plane) };
        if src.is_null() || dst.is_null() {
            return false;
        }
        let src_stride = unsafe { (self.get_stride)(copy.source, copy.plane) };
        let dst_stride = unsafe { (self.get_stride)(copy.output.cast_const(), copy.plane) };
        let (Some(src_stride), Some(dst_stride), Some(x), Some(y)) = (
            isize_positive(src_stride),
            isize_positive(dst_stride),
            isize_nonnegative(copy.pad_x),
            isize_nonnegative(copy.pad_y),
        ) else {
            return false;
        };
        let row_len = copy.width.saturating_mul(copy.bytes_per_sample);
        let dst_offset = y
            .saturating_mul(dst_stride)
            .saturating_add(x.saturating_mul(isize::try_from(copy.bytes_per_sample).unwrap_or(0)));
        for row in 0..copy.height {
            let row = isize::try_from(row).unwrap_or(isize::MAX);
            unsafe {
                ptr::copy_nonoverlapping(
                    src.offset(row.saturating_mul(src_stride)),
                    dst.offset(dst_offset.saturating_add(row.saturating_mul(dst_stride))),
                    row_len,
                );
            }
        }
        true
    }

    unsafe fn read_timing(&self, frame: vs::ConstRaw) -> TimingProps {
        let mut timing = TimingProps {
            duration_num: 0,
            duration_den: 1,
            pts: 0,
        };
        let (Some(get_props), Some(get_int)) = (self.get_frame_props_ro, self.prop_get_int) else {
            return timing;
        };
        let props = unsafe { get_props(frame) };
        if props.is_null() {
            return timing;
        }
        if let (Some(num), Some(den)) = (
            unsafe { read_int(get_int, props, strings::DURATION_NUM) },
            unsafe { read_int(get_int, props, strings::DURATION_DEN) },
        ) && num >= 0
            && den >= 0
        {
            timing.duration_num = num;
            timing.duration_den = den;
        }
        if let Some(pts) = unsafe { read_int(get_int, props, strings::PTS) } {
            timing.pts = pts;
        }
        timing
    }

    unsafe fn read_data(&self, frame: vs::ConstRaw, key: &'static [u8]) -> Vec<u8> {
        let (Some(get_props), Some(get_data), Some(get_size)) = (
            self.get_frame_props_ro,
            self.prop_get_data,
            self.prop_get_data_size,
        ) else {
            return Vec::new();
        };
        let props = unsafe { get_props(frame) };
        if props.is_null() {
            return Vec::new();
        }
        let mut err = 0;
        let size = unsafe { get_size(props, key.as_ptr().cast(), 0, &raw mut err) };
        let Ok(size) = usize::try_from(size) else {
            return Vec::new();
        };
        let data = unsafe { get_data(props, key.as_ptr().cast(), 0, &raw mut err) };
        if data.is_null() || size == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(data.cast::<u8>(), size) }.to_vec()
        }
    }

    unsafe fn write_timing(&self, frame: vs::Raw, timing: TimingProps) {
        let (Some(get_props), Some(set_int)) = (self.get_frame_props_rw, self.prop_set_int) else {
            return;
        };
        let props = unsafe { get_props(frame) };
        if props.is_null() {
            return;
        }
        if timing.duration_num >= 0 && timing.duration_den >= 0 {
            unsafe {
                set_int(
                    props,
                    strings::DURATION_NUM.as_ptr().cast(),
                    timing.duration_num,
                    0,
                );
                set_int(
                    props,
                    strings::DURATION_DEN.as_ptr().cast(),
                    timing.duration_den,
                    0,
                );
            }
        }
        unsafe {
            set_int(props, strings::PTS.as_ptr().cast(), timing.pts, 0);
        }
    }
}

fn dovi_changed(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return true;
    }
    if left.is_empty() {
        return false;
    }
    if left.len() != 0xA5C {
        return left != right;
    }
    dovi_section_changed(left, right, 0x54, 0x58, 0x7C)
        || dovi_section_changed(left, right, 0x3AC, 0x3B0, 0x3D4)
        || dovi_section_changed(left, right, 0x704, 0x708, 0x72C)
}

fn dovi_section_changed(
    left: &[u8],
    right: &[u8],
    count_offset: usize,
    values_offset: usize,
    flags_offset: usize,
) -> bool {
    let count = usize::from(left.get(count_offset).copied().unwrap_or(0));
    if count != usize::from(right.get(count_offset).copied().unwrap_or(0)) {
        return true;
    }
    let value_len = count.saturating_mul(4);
    range(left, values_offset, value_len) != range(right, values_offset, value_len)
        || range(left, flags_offset, count) != range(right, flags_offset, count)
}

fn range(data: &[u8], offset: usize, len: usize) -> &[u8] {
    data.get(offset..offset.saturating_add(len)).unwrap_or(&[])
}

fn duration_seconds(num: i64, den: i64) -> f64 {
    if den == 0 {
        0.0
    } else {
        f64_i64(num) / f64_i64(den)
    }
}

#[allow(clippy::cast_precision_loss)]
fn f64_i64(value: i64) -> f64 {
    value as f64
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn trunc_f64_i64(value: f64) -> i64 {
    if value.is_nan() {
        0
    } else if value >= i64::MAX as f64 {
        i64::MAX
    } else if value <= i64::MIN as f64 {
        i64::MIN
    } else {
        value.trunc() as i64
    }
}

unsafe fn read_int(
    get_int: vs::PropGetInt,
    props: vs::ConstRaw,
    key: &'static [u8],
) -> Option<i64> {
    let mut err = 0;
    let value = unsafe { get_int(props, key.as_ptr().cast(), 0, &raw mut err) };
    (err == 0).then_some(value)
}

fn usize_positive(value: i32) -> Option<usize> {
    usize::try_from(value).ok().filter(|value| *value > 0)
}

fn isize_positive(value: i32) -> Option<isize> {
    isize::try_from(value).ok().filter(|value| *value > 0)
}

fn isize_nonnegative(value: i32) -> Option<isize> {
    isize::try_from(value).ok().filter(|value| *value >= 0)
}
