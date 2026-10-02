#![allow(
    unsafe_code,
    clippy::missing_panics_doc,
    clippy::missing_safety_doc,
    clippy::must_use_candidate
)]

#[cfg(not(target_arch = "wasm32"))]
pub mod avs;
pub mod vs3;
pub mod vs4;
