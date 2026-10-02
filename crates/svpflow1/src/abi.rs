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

#[unsafe(no_mangle)]
pub unsafe extern "system" fn VapourSynthPluginInit2(plugin: vs::Raw, api: vs::ConstRaw) {
    let config = svpflow_host::vs4::Plugin {
        identifier: c"com.svp-team.flow1",
        namespace: c"svp1",
        name: c"SVPFlow1",
        version: 0x30002,
    };
    let functions = [
        svpflow_host::vs4::Function {
            name: c"Super",
            args: c"clip:clip;opt:data",
            returns: c"clip:vnode;data:int;",
            create: super_filter::create_super,
        },
        svpflow_host::vs4::Function {
            name: c"Analyse",
            args: c"clip:clip;sdata:int;src:clip;opt:data",
            returns: c"clip:vnode;data:int;",
            create: analyse_filter::create_analyse,
        },
    ];
    unsafe { svpflow_host::vs4::init(plugin, api, &config, &functions) };
}
