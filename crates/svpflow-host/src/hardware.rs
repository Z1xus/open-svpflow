use core::ffi::c_void;

use crate::opencl::{
    CL_DEVICE_EXTENSIONS, CL_DEVICE_GLOBAL_MEM_SIZE, CL_DEVICE_HOST_UNIFIED_MEMORY,
    CL_DEVICE_LOCAL_MEM_SIZE, CL_DEVICE_MAX_COMPUTE_UNITS, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
    CL_DEVICE_MAX_WORK_GROUP_SIZE, CL_DEVICE_NAME, CL_DEVICE_VENDOR_ID, CL_SUCCESS, ClUint, Handle,
    OpenCl,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Other,
}

#[derive(Clone, Copy, Debug)]
pub struct GpuProfile {
    pub vendor: GpuVendor,
    pub compute_units: usize,
    pub global_mem: usize,
    pub max_alloc: usize,
    pub local_mem: usize,
    pub max_work_group: usize,
    pub unified_memory: bool,
    pub has_fp16: bool,
    pub has_subgroups: bool,
    pub preferred_warp: usize,
}

impl Default for GpuProfile {
    fn default() -> Self {
        Self {
            vendor: GpuVendor::Other,
            compute_units: 8,
            global_mem: 2usize << 30,
            max_alloc: 512usize << 20,
            local_mem: 32usize << 10,
            max_work_group: 256,
            unified_memory: false,
            has_fp16: false,
            has_subgroups: false,
            preferred_warp: 32,
        }
    }
}

impl GpuProfile {
    pub unsafe fn probe(cl: &OpenCl, device: Handle) -> Self {
        let vendor_id = unsafe { device_u32(cl, device, CL_DEVICE_VENDOR_ID) }.unwrap_or(0);
        let name = unsafe { device_string(cl, device, CL_DEVICE_NAME) };
        let extensions = unsafe { device_string(cl, device, CL_DEVICE_EXTENSIONS) };
        let vendor = match vendor_id {
            0x10DE => GpuVendor::Nvidia,
            0x1002 => GpuVendor::Amd,
            0x8086 => GpuVendor::Intel,
            0x0102_7F00 => GpuVendor::Apple,
            _ if name.contains("Apple") => GpuVendor::Apple,
            _ if name.contains("NVIDIA") || name.contains("GeForce") || name.contains("Quadro") => {
                GpuVendor::Nvidia
            }
            _ if name.contains("AMD") || name.contains("Radeon") => GpuVendor::Amd,
            _ if name.contains("Intel") => GpuVendor::Intel,
            _ => GpuVendor::Other,
        };
        let compute_units = unsafe { device_u32(cl, device, CL_DEVICE_MAX_COMPUTE_UNITS) }
            .map_or(8, |v| (v as usize).max(1));
        let global_mem = unsafe { device_u64(cl, device, CL_DEVICE_GLOBAL_MEM_SIZE) }
            .and_then(|v| usize::try_from(v).ok())
            .unwrap_or(2usize << 30);
        let max_alloc = unsafe { device_u64(cl, device, CL_DEVICE_MAX_MEM_ALLOC_SIZE) }
            .and_then(|v| usize::try_from(v).ok())
            .unwrap_or(global_mem / 4);
        let local_mem = unsafe { device_u64(cl, device, CL_DEVICE_LOCAL_MEM_SIZE) }
            .and_then(|v| usize::try_from(v).ok())
            .unwrap_or(32usize << 10);
        let max_work_group = unsafe { device_usize(cl, device, CL_DEVICE_MAX_WORK_GROUP_SIZE) }
            .unwrap_or(256)
            .max(64);
        let unified_memory = vendor == GpuVendor::Apple
            || unsafe { device_u32(cl, device, CL_DEVICE_HOST_UNIFIED_MEMORY) }.unwrap_or(0) != 0;
        let has_fp16 = extensions.contains("cl_khr_fp16");
        let has_subgroups =
            extensions.contains("cl_khr_subgroups") || extensions.contains("cl_intel_subgroups");
        let preferred_warp = match vendor {
            GpuVendor::Amd => 64,
            GpuVendor::Intel => 16,
            GpuVendor::Nvidia | GpuVendor::Apple | GpuVendor::Other => 32,
        };
        Self {
            vendor,
            compute_units,
            global_mem,
            max_alloc,
            local_mem,
            max_work_group,
            unified_memory,
            has_fp16,
            has_subgroups,
            preferred_warp,
        }
    }

    pub fn render_units(&self, requested: usize, workers: usize) -> usize {
        if requested != 2 {
            return requested.clamp(1, 16);
        }
        if self.compute_units >= 40 && self.global_mem >= (12usize << 30) && workers >= 16 {
            return 4;
        }
        if self.compute_units >= 28 && self.global_mem >= (8usize << 30) && workers >= 12 {
            return 3;
        }
        2
    }

    pub fn upload_queues(&self, workers: usize) -> usize {
        if self.unified_memory {
            workers.clamp(2, 6)
        } else {
            workers.clamp(2, 8)
        }
    }

    pub fn analyse_units(&self, workers: usize) -> usize {
        let vram_gib = self.global_mem >> 30;
        let by_mem = match vram_gib {
            0..=2 => 2,
            3..=5 => 4,
            6..=11 => 6,
            _ => 8,
        };
        let by_cu = (self.compute_units / 4).clamp(2, 8);
        workers.clamp(2, 8).min(by_mem).min(by_cu).max(2)
    }

    pub fn pool_budget_bytes(&self) -> usize {
        (self.global_mem / 16).clamp(32usize << 20, 512usize << 20)
    }

    pub fn use_pinned_staging(&self) -> bool {
        !self.unified_memory
    }

    pub fn workgroup_2d(&self, width: usize, height: usize, preferred_multiple: usize) -> [usize; 2] {
        let warp = if preferred_multiple >= 8 && preferred_multiple.is_power_of_two() {
            preferred_multiple.min(64)
        } else {
            self.preferred_warp
        };
        let (lx, ly) = match (self.vendor, warp) {
            (GpuVendor::Intel, 8 | 16) => (16, 8),
            _ => (16, 16),
        };
        let mut x = lx.min(width.next_power_of_two().max(1)).max(1);
        let mut y = ly.min(height.next_power_of_two().max(1)).max(1);
        while x * y > self.max_work_group {
            if y > x && y > 1 {
                y /= 2;
            } else if x > 1 {
                x /= 2;
            } else {
                break;
            }
        }
        [x, y]
    }
}

pub fn logical_cores() -> usize {
    std::thread::available_parallelism().map_or(1, usize::from)
}

pub fn total_ram_bytes() -> usize {
    #[cfg(target_os = "linux")]
    {
        unsafe extern "C" {
            fn sysconf(name: i32) -> isize;
        }
        const SC_PHYS_PAGES: i32 = 85;
        const SC_PAGE_SIZE: i32 = 30;
        let pages = unsafe { sysconf(SC_PHYS_PAGES) };
        let page = unsafe { sysconf(SC_PAGE_SIZE) };
        if pages > 0 && page > 0 {
            return pages.cast_unsigned().saturating_mul(page.cast_unsigned());
        }
    }
    #[cfg(target_os = "macos")]
    {
        unsafe extern "C" {
            fn sysctlbyname(
                name: *const core::ffi::c_char,
                oldp: *mut c_void,
                oldlenp: *mut usize,
                newp: *const c_void,
                newlen: usize,
            ) -> i32;
        }
        let mut bytes = 0u64;
        let mut len = size_of::<u64>();
        let rc = unsafe {
            sysctlbyname(
                c"hw.memsize".as_ptr(),
                (&raw mut bytes).cast(),
                &raw mut len,
                core::ptr::null(),
                0,
            )
        };
        if rc == 0 && bytes > 0 {
            return usize::try_from(bytes).unwrap_or(usize::MAX);
        }
    }
    #[cfg(windows)]
    {
        #[repr(C)]
        struct MemoryStatusEx {
            length: u32,
            memory_load: u32,
            total_phys: u64,
            avail_phys: u64,
            total_page_file: u64,
            avail_page_file: u64,
            total_virtual: u64,
            avail_virtual: u64,
            avail_extended_virtual: u64,
        }
        unsafe extern "system" {
            fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
        }
        let mut status = MemoryStatusEx {
            length: size_of::<MemoryStatusEx>() as u32,
            memory_load: 0,
            total_phys: 0,
            avail_phys: 0,
            total_page_file: 0,
            avail_page_file: 0,
            total_virtual: 0,
            avail_virtual: 0,
            avail_extended_virtual: 0,
        };
        if unsafe { GlobalMemoryStatusEx(&raw mut status) } != 0 && status.total_phys > 0 {
            return usize::try_from(status.total_phys).unwrap_or(usize::MAX);
        }
    }
    8usize << 30
}

unsafe fn device_u32(cl: &OpenCl, device: Handle, param: ClUint) -> Option<u32> {
    let mut val = 0u32;
    let rc = unsafe {
        (cl.GetDeviceInfo)(
            device,
            param,
            size_of::<u32>(),
            (&raw mut val).cast(),
            core::ptr::null_mut(),
        )
    };
    (rc == CL_SUCCESS).then_some(val)
}

unsafe fn device_u64(cl: &OpenCl, device: Handle, param: ClUint) -> Option<u64> {
    let mut val = 0u64;
    let rc = unsafe {
        (cl.GetDeviceInfo)(
            device,
            param,
            size_of::<u64>(),
            (&raw mut val).cast(),
            core::ptr::null_mut(),
        )
    };
    (rc == CL_SUCCESS).then_some(val)
}

unsafe fn device_usize(cl: &OpenCl, device: Handle, param: ClUint) -> Option<usize> {
    let mut val = 0usize;
    let rc = unsafe {
        (cl.GetDeviceInfo)(
            device,
            param,
            size_of::<usize>(),
            (&raw mut val).cast(),
            core::ptr::null_mut(),
        )
    };
    (rc == CL_SUCCESS).then_some(val)
}

unsafe fn device_string(cl: &OpenCl, device: Handle, param: ClUint) -> String {
    let mut len = 0usize;
    if unsafe {
        (cl.GetDeviceInfo)(
            device,
            param,
            0,
            core::ptr::null_mut::<c_void>(),
            &raw mut len,
        )
    } != CL_SUCCESS
        || len == 0
    {
        return String::new();
    }
    let mut buf = vec![0u8; len];
    if unsafe {
        (cl.GetDeviceInfo)(
            device,
            param,
            len,
            buf.as_mut_ptr().cast(),
            core::ptr::null_mut(),
        )
    } != CL_SUCCESS
    {
        return String::new();
    }
    while buf.last() == Some(&0) {
        buf.pop();
    }
    String::from_utf8_lossy(&buf).into_owned()
}
