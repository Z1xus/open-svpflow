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

mod abi;
mod core;
mod filter;
mod frame;
mod gpu;
mod hdr;
mod light;
mod metadata;
mod nvof;
mod options;
mod strings;
mod video_format;
mod vs;

pub(crate) use svpflow_core::{params, renderer};

pub use abi::{VapourSynthPluginInit, svpGetVersion};
