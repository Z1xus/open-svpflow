use std::ffi::c_char;

use svpflow_host::avs;

use crate::{filter, strings, vs};

#[unsafe(no_mangle)]
pub extern "system" fn svpGetVersion() -> i64 {
    strings::VERSION
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn VapourSynthPluginInit(
    config: Option<vs::Config>,
    register: Option<vs::Register>,
    plugin: vs::Raw,
) {
    let Some(config) = config else {
        return;
    };
    unsafe {
        config(
            strings::PLUGIN_ID.as_ptr().cast(),
            strings::PLUGIN_NS.as_ptr().cast(),
            strings::PLUGIN_NAME.as_ptr().cast(),
            0x30002,
            1,
            plugin,
        );
    }
    let register = unsafe { register.unwrap_unchecked() };
    unsafe {
        register(
            strings::SMOOTH_FPS.as_ptr().cast(),
            strings::ARGS_SMOOTH_FPS.as_ptr().cast(),
            filter::create_smooth_fps,
            std::ptr::null_mut(),
            plugin,
        );
        register(
            strings::SMOOTH_FPS_NVOF.as_ptr().cast(),
            strings::ARGS_NVOF.as_ptr().cast(),
            filter::create_smooth_fps_nvof,
            std::ptr::null_mut(),
            plugin,
        );
        register(
            strings::SMOOTH_FPS_RIFE.as_ptr().cast(),
            strings::ARGS_RIFE.as_ptr().cast(),
            filter::create_smooth_fps_rife,
            std::ptr::null_mut(),
            plugin,
        );
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn VapourSynthPluginInit2(plugin: vs::Raw, api: vs::ConstRaw) {
    let text = |bytes: &'static [u8]| std::ffi::CStr::from_bytes_with_nul(bytes).unwrap_or(c"");
    let config = svpflow_host::vs4::Plugin {
        identifier: text(strings::PLUGIN_ID),
        namespace: text(strings::PLUGIN_NS),
        name: text(strings::PLUGIN_NAME),
        version: 0x30002,
    };
    let functions = [
        svpflow_host::vs4::Function {
            name: text(strings::SMOOTH_FPS),
            args: text(strings::ARGS_SMOOTH_FPS),
            returns: c"clip:vnode;",
            create: filter::create_smooth_fps,
        },
        svpflow_host::vs4::Function {
            name: text(strings::SMOOTH_FPS_NVOF),
            args: text(strings::ARGS_NVOF),
            returns: c"clip:vnode;",
            create: filter::create_smooth_fps_nvof,
        },
        svpflow_host::vs4::Function {
            name: text(strings::SMOOTH_FPS_RIFE),
            args: text(strings::ARGS_RIFE),
            returns: c"clip:vnode;",
            create: filter::create_smooth_fps_rife,
        },
    ];
    unsafe { svpflow_host::vs4::init(plugin, api, &config, &functions) };
}

static AVS_FUNCTIONS: [avs::Function; 3] = [
    avs::Function {
        name: c"SVSmoothFps",
        params: c"cccs[mt]i[sar]f[src]c",
        args: &[
            avs::arg(c"clip"),
            avs::data_clip(c"super", c"sdata"),
            avs::data_clip(c"vectors", c"vdata"),
            avs::arg(c"opt"),
            avs::arg(c"mt"),
            avs::arg(c"sar"),
            avs::arg(c"src"),
        ],
        create: filter::create_smooth_fps,
    },
    avs::Function {
        name: c"SVSmoothFps_NVOF",
        params: c"cs[nvof_src]c[mt]i[sar]f[src]c",
        args: &[
            avs::arg(c"clip"),
            avs::arg(c"opt"),
            avs::arg(c"nvof_src"),
            avs::arg(c"mt"),
            avs::arg(c"sar"),
            avs::arg(c"src"),
        ],
        create: filter::create_smooth_fps_nvof,
    },
    avs::Function {
        name: c"SVSmoothFps_RIFE",
        params: c"csc[vec_src]c[mt]i[sar]f[src]c",
        args: &[
            avs::arg(c"clip"),
            avs::arg(c"opt"),
            avs::arg(c"rife_out"),
            avs::data_clip(c"vec_src", c"vdata"),
            avs::arg(c"mt"),
            avs::arg(c"sar"),
            avs::arg(c"src"),
        ],
        create: filter::create_smooth_fps_rife,
    },
];

#[unsafe(no_mangle)]
pub unsafe extern "system" fn avisynth_c_plugin_init(env: vs::Raw) -> *const c_char {
    unsafe { avs::init(env, &AVS_FUNCTIONS, c"SVPFlow2") }
}
