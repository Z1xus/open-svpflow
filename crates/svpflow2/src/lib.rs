#![allow(unsafe_code)]
#![allow(clippy::missing_safety_doc)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::needless_range_loop,
    clippy::question_mark,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]

#[global_allocator]
static ALLOC: svpflow_host::alloc::Cached = svpflow_host::alloc::Cached::new();

mod abi;
#[cfg(feature = "blend")]
mod blend;
mod core;
mod filter;
mod frame;
mod gpu;
mod hdr;
mod light;
mod metadata;
mod nvof;
mod peel;
mod reference;
#[cfg(feature = "still")]
mod still;
mod strings;
mod video_format;
mod vs;

pub(crate) use svpflow_core::{params, renderer, smooth_options as options};

pub use abi::{
    VapourSynthPluginInit, VapourSynthPluginInit2, avisynth_c_plugin_init, svpGetVersion,
};
