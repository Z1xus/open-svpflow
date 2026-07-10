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
