use crate::{analyse_filter, super_filter, vs};

pub const VERSION: i64 = 1_174_405_393;

#[unsafe(no_mangle)]
pub extern "system" fn svpGetVersion() -> i64 {
    VERSION
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
            c"com.svp-team.flow1".as_ptr(),
            c"svp1".as_ptr(),
            c"SVPFlow1".as_ptr(),
            0x30002,
            1,
            plugin,
        );
    }
    let Some(register) = register else {
        return;
    };
    unsafe {
        register(
            c"Super".as_ptr(),
            c"clip:clip;opt:data".as_ptr(),
            super_filter::create_super,
            std::ptr::null_mut(),
            plugin,
        );
        register(
            c"Analyse".as_ptr(),
            c"clip:clip;sdata:int;src:clip;opt:data".as_ptr(),
            analyse_filter::create_analyse,
            std::ptr::null_mut(),
            plugin,
        );
    }
}
