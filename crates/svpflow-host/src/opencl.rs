use core::ffi::{c_char, c_void};

pub type ClInt = i32;
pub type ClUint = u32;
pub type Handle = *mut c_void;

pub const CL_SUCCESS: ClInt = 0;
pub const CL_FALSE: ClUint = 0;
pub const CL_TRUE: ClUint = 1;
pub const CL_DEVICE_TYPE_GPU: u64 = 1 << 2;
pub const CL_MEM_READ_WRITE: u64 = 1;
pub const CL_MEM_ALLOC_HOST_PTR: u64 = 1 << 4;
pub const CL_MAP_READ: u64 = 1;
pub const CL_EVENT_COMMAND_EXECUTION_STATUS: ClUint = 0x11D3;

#[repr(C)]
pub struct ClImageFormat {
    pub channel_order: ClUint,
    pub channel_data_type: ClUint,
}

pub type FnGetPlatformIDs = unsafe extern "C" fn(ClUint, *mut Handle, *mut ClUint) -> ClInt;
pub type FnGetDeviceIDs =
    unsafe extern "C" fn(Handle, u64, ClUint, *mut Handle, *mut ClUint) -> ClInt;
pub type FnGetDeviceInfo =
    unsafe extern "C" fn(Handle, ClUint, usize, *mut c_void, *mut usize) -> ClInt;
pub type FnCreateContext = unsafe extern "C" fn(
    *const isize,
    ClUint,
    *const Handle,
    *const c_void,
    *mut c_void,
    *mut ClInt,
) -> Handle;
pub type FnCreateCommandQueue = unsafe extern "C" fn(Handle, Handle, u64, *mut ClInt) -> Handle;
pub type FnCreateProgramWithSource =
    unsafe extern "C" fn(Handle, ClUint, *const *const c_char, *const usize, *mut ClInt) -> Handle;
pub type FnBuildProgram = unsafe extern "C" fn(
    Handle,
    ClUint,
    *const Handle,
    *const c_char,
    *const c_void,
    *mut c_void,
) -> ClInt;
pub type FnGetProgramBuildInfo =
    unsafe extern "C" fn(Handle, Handle, ClUint, usize, *mut c_void, *mut usize) -> ClInt;
pub type FnCreateKernel = unsafe extern "C" fn(Handle, *const c_char, *mut ClInt) -> Handle;
pub type FnCreateBuffer =
    unsafe extern "C" fn(Handle, u64, usize, *mut c_void, *mut ClInt) -> Handle;
pub type FnCreateImage2D = unsafe extern "C" fn(
    Handle,
    u64,
    *const ClImageFormat,
    usize,
    usize,
    usize,
    *mut c_void,
    *mut ClInt,
) -> Handle;
pub type FnEnqueueMapBuffer = unsafe extern "C" fn(
    Handle,
    Handle,
    ClUint,
    u64,
    usize,
    usize,
    ClUint,
    *const c_void,
    *mut c_void,
    *mut ClInt,
) -> *mut c_void;
pub type FnEnqueueReadBuffer = unsafe extern "C" fn(
    Handle,
    Handle,
    ClUint,
    usize,
    usize,
    *mut c_void,
    ClUint,
    *const c_void,
    *mut c_void,
) -> ClInt;
pub type FnEnqueueReadBufferRect = unsafe extern "C" fn(
    Handle,
    Handle,
    ClUint,
    *const usize,
    *const usize,
    *const usize,
    usize,
    usize,
    usize,
    usize,
    *mut c_void,
    ClUint,
    *const c_void,
    *mut c_void,
) -> ClInt;
pub type FnEnqueueWriteImage = unsafe extern "C" fn(
    Handle,
    Handle,
    ClUint,
    *const usize,
    *const usize,
    usize,
    usize,
    *const c_void,
    ClUint,
    *const c_void,
    *mut c_void,
) -> ClInt;
pub type FnSetKernelArg = unsafe extern "C" fn(Handle, ClUint, usize, *const c_void) -> ClInt;
pub type FnEnqueueNDRangeKernel = unsafe extern "C" fn(
    Handle,
    Handle,
    ClUint,
    *const usize,
    *const usize,
    *const usize,
    ClUint,
    *const c_void,
    *mut c_void,
) -> ClInt;
pub type FnFinish = unsafe extern "C" fn(Handle) -> ClInt;
pub type FnWaitForEvents = unsafe extern "C" fn(ClUint, *const Handle) -> ClInt;
pub type FnGetEventInfo =
    unsafe extern "C" fn(Handle, ClUint, usize, *mut c_void, *mut usize) -> ClInt;
pub type FnKernelInfo =
    unsafe extern "C" fn(Handle, Handle, ClUint, usize, *mut c_void, *mut usize) -> ClInt;
pub type FnRelease = unsafe extern "C" fn(Handle) -> ClInt;

#[allow(non_snake_case)]
pub type FnCreateImage = unsafe extern "C" fn(
    Handle,
    u64,
    *const ClImageFormat,
    *const ImageDesc,
    *mut c_void,
    *mut ClInt,
) -> Handle;

#[repr(C)]
pub struct ImageDesc {
    pub image_type: u32,
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub array_size: usize,
    pub row_pitch: usize,
    pub slice_pitch: usize,
    pub mip_levels: u32,
    pub samples: u32,
    pub buffer: Handle,
}

#[allow(non_snake_case)]
pub struct OpenCl {
    library: libloading::Library,
    pub GetPlatformIDs: FnGetPlatformIDs,
    pub GetDeviceIDs: FnGetDeviceIDs,
    pub GetDeviceInfo: FnGetDeviceInfo,
    pub CreateContext: FnCreateContext,
    pub CreateCommandQueue: FnCreateCommandQueue,
    pub CreateProgramWithSource: FnCreateProgramWithSource,
    pub BuildProgram: FnBuildProgram,
    pub CreateKernel: FnCreateKernel,
    pub CreateBuffer: FnCreateBuffer,
    pub EnqueueReadBuffer: FnEnqueueReadBuffer,
    pub EnqueueMapBuffer: FnEnqueueMapBuffer,
    pub EnqueueWriteBuffer: FnEnqueueReadBuffer,
    pub SetKernelArg: FnSetKernelArg,
    pub EnqueueNDRangeKernel: FnEnqueueNDRangeKernel,
    pub Finish: FnFinish,
    pub Flush: FnFinish,
    pub WaitForEvents: FnWaitForEvents,
    pub GetEventInfo: FnGetEventInfo,
    pub ReleaseEvent: FnRelease,
    pub ReleaseMemObject: FnRelease,
}

impl OpenCl {
    pub unsafe fn symbol<T: Copy>(&self, name: &'static str) -> Result<T, String> {
        unsafe { self.library.get::<T>(name.as_bytes()) }
            .map(|symbol| *symbol)
            .map_err(|e| format!("missing {}: {e}", name.trim_end_matches('\0')))
    }

    pub unsafe fn load() -> Result<Self, String> {
        let name = if cfg!(target_os = "windows") {
            "OpenCL.dll"
        } else if cfg!(target_os = "macos") {
            "/System/Library/Frameworks/OpenCL.framework/OpenCL"
        } else {
            "libOpenCL.so.1"
        };
        let lib = unsafe { libloading::Library::new(name) }
            .map_err(|e| format!("{name} not loadable: {e}"))?;
        macro_rules! sym {
            ($name:literal) => {
                *unsafe {
                    lib.get(concat!($name, "\0").as_bytes())
                        .map_err(|e| format!("missing {}: {e}", $name))?
                }
            };
        }
        let me = {
            Self {
                GetPlatformIDs: sym!("clGetPlatformIDs"),
                GetDeviceIDs: sym!("clGetDeviceIDs"),
                GetDeviceInfo: sym!("clGetDeviceInfo"),
                CreateContext: sym!("clCreateContext"),
                CreateCommandQueue: sym!("clCreateCommandQueue"),
                CreateProgramWithSource: sym!("clCreateProgramWithSource"),
                BuildProgram: sym!("clBuildProgram"),
                CreateKernel: sym!("clCreateKernel"),
                CreateBuffer: sym!("clCreateBuffer"),
                EnqueueReadBuffer: sym!("clEnqueueReadBuffer"),
                EnqueueMapBuffer: sym!("clEnqueueMapBuffer"),
                EnqueueWriteBuffer: sym!("clEnqueueWriteBuffer"),
                SetKernelArg: sym!("clSetKernelArg"),
                EnqueueNDRangeKernel: sym!("clEnqueueNDRangeKernel"),
                Finish: sym!("clFinish"),
                Flush: sym!("clFlush"),
                WaitForEvents: sym!("clWaitForEvents"),
                GetEventInfo: sym!("clGetEventInfo"),
                ReleaseEvent: sym!("clReleaseEvent"),
                ReleaseMemObject: sym!("clReleaseMemObject"),
                library: lib,
            }
        };
        Ok(me)
    }
}
