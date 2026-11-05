#![allow(
    unsafe_code,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::missing_safety_doc,
    clippy::must_use_candidate
)]

#[cfg(not(target_arch = "wasm32"))]
pub mod alloc;
#[cfg(not(target_arch = "wasm32"))]
pub mod avs;
#[cfg(not(target_arch = "wasm32"))]
pub mod context;
#[cfg(not(target_arch = "wasm32"))]
pub mod opencl;
pub mod vs3;
pub mod vs4;

#[cfg(windows)]
pub fn pin() {
    unsafe extern "system" {
        fn GetModuleHandleExW(
            flags: u32,
            name: *const u16,
            module: *mut *mut std::ffi::c_void,
        ) -> i32;
    }
    const PIN: u32 = 1;
    const FROM_ADDRESS: u32 = 4;
    let mut module = std::ptr::null_mut();
    unsafe {
        GetModuleHandleExW(
            PIN | FROM_ADDRESS,
            (pin as *const ()).cast::<u16>(),
            &raw mut module,
        )
    };
}

#[cfg(not(windows))]
pub fn pin() {}

static LIVE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub struct Live(extern "C" fn());

impl Live {
    pub fn new(idle: extern "C" fn()) -> Self {
        LIVE.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        Self(idle)
    }
}

impl Drop for Live {
    fn drop(&mut self) {
        if LIVE.fetch_sub(1, std::sync::atomic::Ordering::AcqRel) == 1 {
            (self.0)();
            #[cfg(all(target_os = "linux", target_env = "gnu"))]
            {
                unsafe extern "C" {
                    fn malloc_trim(pad: usize) -> i32;
                }
                unsafe { malloc_trim(0) };
            }
        }
    }
}
