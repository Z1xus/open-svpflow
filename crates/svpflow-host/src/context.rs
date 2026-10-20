use std::ffi::c_void;
use std::sync::Mutex;

type Handle = *mut c_void;
#[cfg(any(target_os = "linux", target_os = "windows"))]
type Lookup = unsafe extern "C" fn(Handle) -> Handle;

#[cfg(target_os = "linux")]
const SIBLINGS: [&str; 2] = ["libsvpflow1_vs.so", "libsvpflow2_vs.so"];
#[cfg(target_os = "windows")]
const SIBLINGS: [&str; 2] = ["svpflow1_vs.dll", "svpflow2_vs.dll"];

// contexts are never released, so plugins can share them
static OWN: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

#[unsafe(no_mangle)]
pub extern "C" fn osvp_cl_context_v1(device: Handle) -> Handle {
    let own = OWN
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    own.iter()
        .find(|entry| entry.0 == device as usize)
        .map_or(std::ptr::null_mut(), |entry| entry.1 as Handle)
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn sibling(_: Handle) -> Handle {
    std::ptr::null_mut()
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn sibling(device: Handle) -> Handle {
    for name in SIBLINGS {
        #[cfg(target_os = "linux")]
        let library = unsafe { libloading::os::unix::Library::open(Some(name), 0x1 | 0x4) };
        #[cfg(target_os = "windows")]
        let library = unsafe { libloading::os::windows::Library::open_already_loaded(name) };
        let Ok(library) = library else {
            continue;
        };
        let Ok(lookup) = (unsafe { library.get::<Lookup>(b"osvp_cl_context_v1\0") }) else {
            continue;
        };
        let lookup: Lookup = *lookup;
        if lookup as *const () == osvp_cl_context_v1 as *const () {
            continue;
        }
        let context = unsafe { lookup(device) };
        if !context.is_null() {
            // the library stays loaded for the whole process
            std::mem::forget(library);
            return context;
        }
    }
    std::ptr::null_mut()
}

pub fn shared(device: Handle, create: impl FnOnce() -> Handle) -> Handle {
    let mut own = OWN
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(entry) = own.iter().find(|entry| entry.0 == device as usize) {
        return entry.1 as Handle;
    }
    let mut context = sibling(device);
    if context.is_null() {
        context = create();
    }
    if !context.is_null() {
        own.push((device as usize, context as usize));
    }
    context
}
