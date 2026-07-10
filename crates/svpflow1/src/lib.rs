#![allow(unsafe_code)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::too_many_arguments)]
#![allow(
    clippy::bool_to_int_with_if,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_ptr_alignment,
    clippy::cast_sign_loss,
    clippy::collapsible_if,
    clippy::comparison_chain,
    clippy::explicit_iter_loop,
    clippy::field_reassign_with_default,
    clippy::if_same_then_else,
    clippy::manual_checked_ops,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unnecessary_cast
)]

#[cfg(not(target_arch = "wasm32"))]
mod abi;
#[cfg(not(target_arch = "wasm32"))]
mod analyse_filter;
mod analyse_opts;
mod analyse_search;
mod portable;
mod super_build;
#[cfg(not(target_arch = "wasm32"))]
mod super_filter;
mod super_opts;
#[cfg(not(target_arch = "wasm32"))]
mod vs;

pub(crate) use svpflow_core::params;

pub use portable::{Analyser, SuperBuilder, SuperFrame};

#[cfg(not(target_arch = "wasm32"))]
pub use abi::{VapourSynthPluginInit, svpGetVersion};
