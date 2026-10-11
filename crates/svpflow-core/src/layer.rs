pub const CHUNK: i32 = 12;

#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn limit(limit: Option<f64>) -> f32 {
    limit.unwrap_or(6.0) as f32
}

#[must_use]
pub fn pairs(chunk: i32, last: i32) -> std::ops::RangeInclusive<i32> {
    chunk * CHUNK..=(chunk * CHUNK + CHUNK - 1).min(last)
}
