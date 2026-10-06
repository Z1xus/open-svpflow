#![deny(unsafe_code)]
#![allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::similar_names
)]

pub mod frame_math;
pub mod metadata;
pub mod params;
pub mod renderer;
pub mod smooth;
pub mod smooth_engine;
pub mod smooth_options;
pub mod still;
