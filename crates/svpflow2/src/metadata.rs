pub(crate) use svpflow_core::metadata::*;

pub(crate) unsafe fn vector_data(handle: i64) -> VectorRecord {
    if handle == 0 {
        return VectorRecord::Missing;
    }
    let Ok(address) = usize::try_from(handle) else {
        return VectorRecord::Invalid;
    };
    let data = unsafe { std::slice::from_raw_parts(address as *const u8, VECTOR_DATA_LEN) };
    svpflow_core::metadata::vector_data(data)
}
