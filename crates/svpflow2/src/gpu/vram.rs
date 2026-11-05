use core::ffi::{c_char, c_void};

use super::{CL_SUCCESS, ClDeviceId, ClUint, OpenCl};

const CL_DEVICE_VENDOR_ID: ClUint = 0x1001;
const NVIDIA: u32 = 0x10DE;
const CL_DEVICE_PCI_BUS_ID_NV: ClUint = 0x4008;
const CL_DEVICE_PCI_SLOT_ID_NV: ClUint = 0x4009;
const CL_DEVICE_PCI_DOMAIN_ID_NV: ClUint = 0x400A;
const CL_DEVICE_GLOBAL_FREE_MEMORY_AMD: ClUint = 0x4039;

type NvmlDevice = *mut c_void;
type FnInit = unsafe extern "C" fn() -> i32;
type FnByBus = unsafe extern "C" fn(*const c_char, *mut NvmlDevice) -> i32;
type FnCount = unsafe extern "C" fn(*mut u32) -> i32;
type FnByIndex = unsafe extern "C" fn(u32, *mut NvmlDevice) -> i32;
type FnMemory = unsafe extern "C" fn(NvmlDevice, *mut Memory) -> i32;

#[repr(C)]
#[derive(Default)]
struct Memory {
    total: u64,
    free: u64,
    used: u64,
}

struct Nvml {
    _library: libloading::Library,
    by_bus: FnByBus,
    count: FnCount,
    by_index: FnByIndex,
    memory: FnMemory,
}

static NVML: std::sync::OnceLock<Option<Nvml>> = std::sync::OnceLock::new();

pub(super) struct Vram(Source);

enum Source {
    Nvml(NvmlDevice),
    Amd,
    Unknown,
}

unsafe impl Send for Vram {}
unsafe impl Sync for Vram {}

impl Vram {
    pub(super) unsafe fn open(cl: &OpenCl, device: ClDeviceId) -> Self {
        if unsafe { amd_free(cl, device) }.is_some() {
            return Self(Source::Amd);
        }
        if unsafe { device_uint(cl, device, CL_DEVICE_VENDOR_ID) } != Some(NVIDIA) {
            return Self(Source::Unknown);
        }
        let handle = NVML
            .get_or_init(|| unsafe { Nvml::load() })
            .as_ref()
            .and_then(|nvml| unsafe { nvml.device(cl, device) });
        Self(handle.map_or(Source::Unknown, Source::Nvml))
    }

    pub(super) unsafe fn free(&self, cl: &OpenCl, device: ClDeviceId) -> Option<usize> {
        match &self.0 {
            Source::Nvml(handle) => NVML.get()?.as_ref()?.free(*handle),
            Source::Amd => unsafe { amd_free(cl, device) },
            Source::Unknown => None,
        }
    }
}

impl Nvml {
    unsafe fn load() -> Option<Self> {
        let name = if cfg!(target_os = "windows") {
            "nvml.dll"
        } else {
            "libnvidia-ml.so.1"
        };
        let library = unsafe { libloading::Library::new(name) }.ok()?;
        let init: FnInit = unsafe { symbol(&library, b"nvmlInit_v2\0") }?;
        let nvml = Self {
            by_bus: unsafe { symbol(&library, b"nvmlDeviceGetHandleByPciBusId_v2\0") }?,
            count: unsafe { symbol(&library, b"nvmlDeviceGetCount_v2\0") }?,
            by_index: unsafe { symbol(&library, b"nvmlDeviceGetHandleByIndex_v2\0") }?,
            memory: unsafe { symbol(&library, b"nvmlDeviceGetMemoryInfo\0") }?,
            _library: library,
        };
        (unsafe { init() } == 0).then_some(nvml)
    }

    unsafe fn device(&self, cl: &OpenCl, device: ClDeviceId) -> Option<NvmlDevice> {
        let mut handle: NvmlDevice = std::ptr::null_mut();
        let mut devices = 0;
        let status = match unsafe { pci_bus(cl, device) } {
            Some(bus) => unsafe { (self.by_bus)(bus.as_ptr(), &raw mut handle) },
            None if unsafe { (self.count)(&raw mut devices) } == 0 && devices == 1 => unsafe {
                (self.by_index)(0, &raw mut handle)
            },
            None => -1,
        };
        (status == 0 && self.free(handle).is_some()).then_some(handle)
    }

    fn free(&self, handle: NvmlDevice) -> Option<usize> {
        let mut memory = Memory::default();
        let status = unsafe { (self.memory)(handle, &raw mut memory) };
        (status == 0)
            .then(|| usize::try_from(memory.free).ok())
            .flatten()
    }
}

unsafe fn symbol<T: Copy>(library: &libloading::Library, name: &[u8]) -> Option<T> {
    unsafe { library.get::<T>(name) }.ok().map(|symbol| *symbol)
}

unsafe fn pci_bus(cl: &OpenCl, device: ClDeviceId) -> Option<std::ffi::CString> {
    let bus = unsafe { device_uint(cl, device, CL_DEVICE_PCI_BUS_ID_NV) }?;
    let slot = unsafe { device_uint(cl, device, CL_DEVICE_PCI_SLOT_ID_NV) }?;
    let domain = unsafe { device_uint(cl, device, CL_DEVICE_PCI_DOMAIN_ID_NV) }.unwrap_or(0);
    let text = format!("{domain:08x}:{bus:02x}:{:02x}.{:x}", slot >> 3, slot & 7);
    std::ffi::CString::new(text).ok()
}

unsafe fn device_uint(cl: &OpenCl, device: ClDeviceId, param: ClUint) -> Option<u32> {
    let mut value = 0u32;
    let status = unsafe {
        (cl.GetDeviceInfo)(
            device,
            param,
            size_of::<u32>(),
            (&raw mut value).cast(),
            std::ptr::null_mut(),
        )
    };
    (status == CL_SUCCESS).then_some(value)
}

unsafe fn amd_free(cl: &OpenCl, device: ClDeviceId) -> Option<usize> {
    let mut kilobytes = [0usize; 2];
    let status = unsafe {
        (cl.GetDeviceInfo)(
            device,
            CL_DEVICE_GLOBAL_FREE_MEMORY_AMD,
            size_of_val(&kilobytes),
            kilobytes.as_mut_ptr().cast(),
            std::ptr::null_mut(),
        )
    };
    (status == CL_SUCCESS).then(|| kilobytes[0].saturating_mul(1024))
}
