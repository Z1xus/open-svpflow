#![allow(unsafe_code)]
#![allow(clippy::missing_safety_doc)]
use core::ffi::{c_char, c_void};
use std::ffi::CString;

mod blend;

pub use blend::BlendJob;
#[cfg(feature = "blend")]
pub use blend::OutputPlane;

type ClInt = i32;
type ClUint = u32;
type ClPlatformId = *mut c_void;
type ClDeviceId = *mut c_void;
type ClContext = *mut c_void;
type ClCommandQueue = *mut c_void;
type ClProgram = *mut c_void;
type ClKernel = *mut c_void;
type ClMem = *mut c_void;

#[repr(C)]
struct ClImageFormat {
    channel_order: ClUint,
    channel_data_type: ClUint,
}

const CL_SUCCESS: ClInt = 0;
const CL_EVENT_COMMAND_EXECUTION_STATUS: ClUint = 0x11D3;
const CL_DEVICE_TYPE_GPU: u64 = 1 << 2;
const CL_DEVICE_NAME: ClUint = 0x102B;
const CL_MEM_READ_WRITE: u64 = 1 << 0;
const CL_MEM_ALLOC_HOST_PTR: u64 = 1 << 4;
const CL_MAP_READ: u64 = 1;
const CL_FALSE: ClUint = 0;
const CL_TRUE: ClUint = 1;
const CL_PROGRAM_BUILD_LOG: ClUint = 0x1183;
const CL_R: ClUint = 0x10B0;
const CL_RGBA: ClUint = 0x10B5;
const CL_UNORM_INT8: ClUint = 0x10D2;
const CL_UNORM_INT16: ClUint = 0x10D3;
const CL_FLOAT: ClUint = 0x10DE;

type FnGetPlatformIDs = unsafe extern "C" fn(ClUint, *mut ClPlatformId, *mut ClUint) -> ClInt;
type FnGetDeviceIDs =
    unsafe extern "C" fn(ClPlatformId, u64, ClUint, *mut ClDeviceId, *mut ClUint) -> ClInt;
type FnGetDeviceInfo =
    unsafe extern "C" fn(ClDeviceId, ClUint, usize, *mut c_void, *mut usize) -> ClInt;
type FnCreateContext = unsafe extern "C" fn(
    *const isize,
    ClUint,
    *const ClDeviceId,
    *const c_void,
    *mut c_void,
    *mut ClInt,
) -> ClContext;
type FnCreateCommandQueue =
    unsafe extern "C" fn(ClContext, ClDeviceId, u64, *mut ClInt) -> ClCommandQueue;
type FnCreateProgramWithSource = unsafe extern "C" fn(
    ClContext,
    ClUint,
    *const *const c_char,
    *const usize,
    *mut ClInt,
) -> ClProgram;
type FnBuildProgram = unsafe extern "C" fn(
    ClProgram,
    ClUint,
    *const ClDeviceId,
    *const c_char,
    *const c_void,
    *mut c_void,
) -> ClInt;
type FnGetProgramBuildInfo =
    unsafe extern "C" fn(ClProgram, ClDeviceId, ClUint, usize, *mut c_void, *mut usize) -> ClInt;
type FnCreateKernel = unsafe extern "C" fn(ClProgram, *const c_char, *mut ClInt) -> ClKernel;
type FnCreateBuffer = unsafe extern "C" fn(ClContext, u64, usize, *mut c_void, *mut ClInt) -> ClMem;
type FnCreateImage2D = unsafe extern "C" fn(
    ClContext,
    u64,
    *const ClImageFormat,
    usize,
    usize,
    usize,
    *mut c_void,
    *mut ClInt,
) -> ClMem;
type FnEnqueueReadBuffer = unsafe extern "C" fn(
    ClCommandQueue,
    ClMem,
    ClUint,
    usize,
    usize,
    *mut c_void,
    ClUint,
    *const c_void,
    *mut c_void,
) -> ClInt;
type FnEnqueueWriteImage = unsafe extern "C" fn(
    ClCommandQueue,
    ClMem,
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
type FnEnqueueMapBuffer = unsafe extern "C" fn(
    ClCommandQueue,
    ClMem,
    ClUint,
    u64,
    usize,
    usize,
    ClUint,
    *const ClEvent,
    *mut ClEvent,
    *mut ClInt,
) -> *mut c_void;
type FnSetKernelArg = unsafe extern "C" fn(ClKernel, ClUint, usize, *const c_void) -> ClInt;
type FnEnqueueNDRangeKernel = unsafe extern "C" fn(
    ClCommandQueue,
    ClKernel,
    ClUint,
    *const usize,
    *const usize,
    *const usize,
    ClUint,
    *const c_void,
    *mut c_void,
) -> ClInt;
type FnFinish = unsafe extern "C" fn(ClCommandQueue) -> ClInt;
type ClEvent = *mut c_void;
type FnWaitForEvents = unsafe extern "C" fn(ClUint, *const ClEvent) -> ClInt;
type FnGetEventInfo =
    unsafe extern "C" fn(ClEvent, ClUint, usize, *mut c_void, *mut usize) -> ClInt;
type FnRelease = unsafe extern "C" fn(*mut c_void) -> ClInt;

#[allow(non_snake_case)]
struct OpenCl {
    _lib: libloading::Library,
    GetPlatformIDs: FnGetPlatformIDs,
    GetDeviceIDs: FnGetDeviceIDs,
    GetDeviceInfo: FnGetDeviceInfo,
    CreateContext: FnCreateContext,
    CreateCommandQueue: FnCreateCommandQueue,
    CreateProgramWithSource: FnCreateProgramWithSource,
    BuildProgram: FnBuildProgram,
    GetProgramBuildInfo: FnGetProgramBuildInfo,
    CreateKernel: FnCreateKernel,
    CreateBuffer: FnCreateBuffer,
    CreateImage2D: FnCreateImage2D,
    CreateImage: Option<blend::FnCreateImage>,
    EnqueueReadBuffer: FnEnqueueReadBuffer,
    EnqueueWriteImage: FnEnqueueWriteImage,
    EnqueueMapBuffer: FnEnqueueMapBuffer,
    SetKernelArg: FnSetKernelArg,
    EnqueueNDRangeKernel: FnEnqueueNDRangeKernel,
    Finish: FnFinish,
    Flush: FnFinish,
    WaitForEvents: FnWaitForEvents,
    GetEventInfo: FnGetEventInfo,
    ReleaseEvent: FnRelease,
    ReleaseMemObject: FnRelease,
    ReleaseKernel: FnRelease,
    ReleaseProgram: FnRelease,
    ReleaseCommandQueue: FnRelease,
    ReleaseContext: FnRelease,
}

impl OpenCl {
    unsafe fn load() -> Result<Self, String> {
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
                GetProgramBuildInfo: sym!("clGetProgramBuildInfo"),
                CreateKernel: sym!("clCreateKernel"),
                CreateBuffer: sym!("clCreateBuffer"),
                CreateImage2D: sym!("clCreateImage2D"),
                CreateImage: unsafe { lib.get(b"clCreateImage\0") }.ok().map(|f| *f),
                EnqueueReadBuffer: sym!("clEnqueueReadBuffer"),
                EnqueueWriteImage: sym!("clEnqueueWriteImage"),
                EnqueueMapBuffer: sym!("clEnqueueMapBuffer"),
                SetKernelArg: sym!("clSetKernelArg"),
                EnqueueNDRangeKernel: sym!("clEnqueueNDRangeKernel"),
                Finish: sym!("clFinish"),
                Flush: sym!("clFlush"),
                WaitForEvents: sym!("clWaitForEvents"),
                GetEventInfo: sym!("clGetEventInfo"),
                ReleaseEvent: sym!("clReleaseEvent"),
                ReleaseMemObject: sym!("clReleaseMemObject"),
                ReleaseKernel: sym!("clReleaseKernel"),
                ReleaseProgram: sym!("clReleaseProgram"),
                ReleaseCommandQueue: sym!("clReleaseCommandQueue"),
                ReleaseContext: sym!("clReleaseContext"),
                _lib: lib,
            }
        };
        Ok(me)
    }
}

pub struct GpuContext {
    cl: OpenCl,
    context: ClContext,
    device: ClDeviceId,
    program: ClProgram,
    variants: std::sync::Mutex<Vec<(u32, ClProgram)>>,
    device_name: String,
    hi10: bool,

    units: Vec<std::sync::Mutex<Unit>>,
    next_unit: std::sync::atomic::AtomicUsize,
    cache_resources: Vec<std::sync::Mutex<CacheResources>>,
    next_upload: std::sync::atomic::AtomicUsize,

    super_cache: std::sync::Mutex<Vec<(i64, SuperCell)>>,
    image_pool: std::sync::Arc<ImagePool>,
    staging: std::sync::Mutex<Vec<Staging>>,
    still: std::sync::Mutex<StillState>,
    blend_pool: std::sync::Mutex<Vec<blend::Buffers>>,
    blend_units: std::sync::Mutex<Vec<Unit>>,
}

struct Staging {
    mem: ClMem,
    len: usize,
    host: *mut u8,
}

unsafe impl Send for Staging {}

type SuperCell = std::sync::Arc<std::sync::OnceLock<Option<std::sync::Arc<SuperEntry>>>>;

struct CacheResources {
    kernel: ClKernel,
    queue: ClCommandQueue,
}

struct SuperEntry {
    still: std::sync::OnceLock<Option<(StillMask, std::sync::Arc<SuperEntry>)>>,
    mems: [ClMem; 4],
    ready: ClEvent,
    wait: FnWaitForEvents,
    release_event: FnRelease,
    release: FnRelease,
    pool: std::sync::Arc<ImagePool>,
    dims: (usize, usize),
}

struct PooledImages {
    dims: (usize, usize),
    linear: ClMem,
    packed: ClMem,
}

unsafe impl Send for PooledImages {}

type ImagePool = std::sync::Mutex<Vec<PooledImages>>;

const IMAGE_POOL_CAP: usize = 16;

unsafe impl Send for SuperEntry {}
unsafe impl Sync for SuperEntry {}
impl Drop for SuperEntry {
    fn drop(&mut self) {
        if !self.ready.is_null() {
            unsafe {
                (self.wait)(1, &raw const self.ready);
                (self.release_event)(self.ready);
            }
        }
        if let Some(Some((StillMask(still, done), _))) = self.still.get() {
            unsafe {
                (self.wait)(1, done);
                (self.release_event)(*done);
                (self.release)(*still);
            }
        }
        if let Ok(mut pool) = self.pool.lock()
            && pool.len() < IMAGE_POOL_CAP
        {
            pool.push(PooledImages {
                dims: self.dims,
                linear: self.mems[0],
                packed: self.mems[1],
            });
            return;
        }
        for (i, &m) in self.mems.iter().enumerate() {
            if !m.is_null() && !self.mems[..i].contains(&m) {
                unsafe { (self.release)(m) };
            }
        }
    }
}

pub struct SuperHandle {
    entry: std::sync::Arc<SuperEntry>,
}
impl SuperHandle {
    #[must_use]
    pub fn bufs(&self) -> [GpuBuf; 3] {
        [
            GpuBuf(self.entry.mems[0], self.entry.ready),
            GpuBuf(self.entry.mems[1], self.entry.ready),
            GpuBuf(self.entry.mems[2], self.entry.ready),
        ]
    }

    #[must_use]
    pub fn sources(&self, linear: bool) -> [GpuBuf; 3] {
        [
            GpuBuf(
                self.entry.mems[if linear { 0 } else { 3 }],
                self.entry.ready,
            ),
            GpuBuf(self.entry.mems[1], self.entry.ready),
            GpuBuf(self.entry.mems[2], self.entry.ready),
        ]
    }
}

struct Unit {
    kernel: ClKernel,
    queue: ClCommandQueue,
    dst: [(ClMem, usize); 3],
    motion: [ImageSlot; 2],
    motion_key: [i64; 2],
    mask: ImageSlot,
    packed_base: Vec<u16>,
    packed_ext: Vec<u16>,
    packed_mask: Vec<u8>,
    variant_kernels: Vec<(u32, ClKernel)>,
    blend: blend::UnitState,
}

#[derive(Clone, Copy)]
struct ImageSlot {
    mem: ClMem,
    width: usize,
    height: usize,
    data_type: ClUint,
}

impl ImageSlot {
    const EMPTY: Self = Self {
        mem: std::ptr::null_mut(),
        width: 0,
        height: 0,
        data_type: 0,
    };
}

unsafe impl Send for Unit {}

impl Unit {
    fn new(kernel: ClKernel, queue: ClCommandQueue) -> Self {
        let z = (std::ptr::null_mut(), 0usize);
        Self {
            kernel,
            queue,
            dst: [z; 3],
            motion: [ImageSlot::EMPTY; 2],
            motion_key: [i64::MIN; 2],
            mask: ImageSlot::EMPTY,
            packed_base: Vec::new(),
            packed_ext: Vec::new(),
            packed_mask: Vec::new(),
            variant_kernels: Vec::new(),
            blend: blend::UnitState::default(),
        }
    }
}

#[derive(Clone, Copy)]
pub struct UploadPlane<'a> {
    pub data: &'a [u8],
    pub stride: usize,
    pub width: usize,
    pub height: usize,
}

const DEFAULT_UNITS: usize = 2;
const MIN_UNITS: usize = 1;
const MAX_UNITS: usize = 8;

#[derive(Clone, Copy)]
pub struct GpuBuf(ClMem, ClEvent);

#[derive(Clone, Copy, PartialEq)]
pub struct StillMask(ClMem, ClEvent);

struct StillState {
    queue: ClCommandQueue,
    kernels: [ClKernel; 2],
    scratch: ClMem,
    dims: (usize, usize),
}

unsafe impl Send for StillState {}

const SUPER_CACHE_CAP: usize = 32;
const UPLOAD_QUEUES: usize = 4;

unsafe impl Send for GpuContext {}
unsafe impl Sync for GpuContext {}

impl Drop for GpuContext {
    fn drop(&mut self) {
        unsafe {
            if let Ok(mut cache) = self.super_cache.lock() {
                cache.clear();
            }
            for unit in &self.units {
                if let Ok(u) = unit.lock() {
                    self.release_unit(&u);
                }
            }
            if let Ok(mut pool) = self.blend_units.lock() {
                for unit in pool.drain(..) {
                    self.release_unit(&unit);
                }
            }
            for resources in &self.cache_resources {
                if let Ok(cache) = resources.lock() {
                    (self.cl.ReleaseKernel)(cache.kernel);
                    (self.cl.ReleaseCommandQueue)(cache.queue);
                }
            }
            if let Ok(mut cache) = self.super_cache.lock() {
                cache.clear();
            }
            if let Ok(mut pool) = self.staging.lock() {
                for staging in pool.drain(..) {
                    (self.cl.ReleaseMemObject)(staging.mem);
                }
            }
            if let Ok(still) = self.still.lock() {
                for kernel in still.kernels {
                    if !kernel.is_null() {
                        (self.cl.ReleaseKernel)(kernel);
                    }
                }
                if !still.scratch.is_null() {
                    (self.cl.ReleaseMemObject)(still.scratch);
                }
                if !still.queue.is_null() {
                    (self.cl.ReleaseCommandQueue)(still.queue);
                }
            }
            if let Ok(mut pool) = self.blend_pool.lock() {
                for buffers in pool.drain(..) {
                    (self.cl.ReleaseMemObject)(buffers.sums);
                    (self.cl.ReleaseMemObject)(buffers.bytes);
                }
            }
            if let Ok(mut pool) = self.image_pool.lock() {
                for images in pool.drain(..) {
                    (self.cl.ReleaseMemObject)(images.linear);
                    (self.cl.ReleaseMemObject)(images.packed);
                }
            }
            if let Ok(variants) = self.variants.lock() {
                for &(_, program) in variants.iter() {
                    (self.cl.ReleaseProgram)(program);
                }
            }
            (self.cl.ReleaseProgram)(self.program);
            (self.cl.ReleaseContext)(self.context);
        }
    }
}

impl GpuContext {
    unsafe fn release_unit(&self, u: &Unit) {
        unsafe {
            for b in &u.dst {
                if !b.0.is_null() {
                    (self.cl.ReleaseMemObject)(b.0);
                }
            }
            for image in u.motion.iter().chain(std::iter::once(&u.mask)) {
                if !image.mem.is_null() {
                    (self.cl.ReleaseMemObject)(image.mem);
                }
            }
            (self.cl.ReleaseKernel)(u.kernel);
            for &(_, kernel) in &u.variant_kernels {
                (self.cl.ReleaseKernel)(kernel);
            }
            for kernel in u.blend.kernels.into_iter().flatten() {
                (self.cl.ReleaseKernel)(kernel);
            }
            if !u.blend.masks.0.is_null() {
                (self.cl.ReleaseMemObject)(u.blend.masks.0);
            }
            (self.cl.ReleaseCommandQueue)(u.queue);
        }
    }

    pub fn new(gpuid: i32, qn: i64, hi10: bool) -> Option<Self> {
        let units = usize::try_from(qn)
            .unwrap_or(DEFAULT_UNITS)
            .clamp(MIN_UNITS, MAX_UNITS);
        unsafe { Self::try_new(gpuid, units, hi10) }.ok()
    }

    unsafe fn try_new(gpuid: i32, num_units: usize, hi10: bool) -> Result<Self, String> {
        let cl = unsafe { OpenCl::load()? };

        let mut num_plat = 0u32;
        if unsafe { (cl.GetPlatformIDs)(0, std::ptr::null_mut(), &raw mut num_plat) } != CL_SUCCESS
            || num_plat == 0
        {
            return Err("no OpenCL platforms".into());
        }
        let mut plats = vec![std::ptr::null_mut(); num_plat as usize];
        unsafe { (cl.GetPlatformIDs)(num_plat, plats.as_mut_ptr(), &raw mut num_plat) };
        let want = u32::try_from(gpuid.max(0)).unwrap_or(0);
        for plat in plats {
            let mut num_dev = 0u32;
            if unsafe {
                (cl.GetDeviceIDs)(
                    plat,
                    CL_DEVICE_TYPE_GPU,
                    0,
                    std::ptr::null_mut(),
                    &raw mut num_dev,
                )
            } != CL_SUCCESS
                || num_dev == 0
            {
                continue;
            }
            let mut devs = vec![std::ptr::null_mut(); num_dev as usize];
            unsafe {
                (cl.GetDeviceIDs)(
                    plat,
                    CL_DEVICE_TYPE_GPU,
                    num_dev,
                    devs.as_mut_ptr(),
                    &raw mut num_dev,
                );
            }
            let idx = if (want as usize) < devs.len() {
                want as usize
            } else {
                0
            };
            let device = devs[idx];
            let mut err = 0;
            let context = unsafe {
                (cl.CreateContext)(
                    std::ptr::null(),
                    1,
                    &raw const device,
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    &raw mut err,
                )
            };
            if err != CL_SUCCESS || context.is_null() {
                continue;
            }

            let base = format!("#define HI10P {}\n{KERNEL_SRC}", u8::from(hi10));
            let program = match unsafe { build_program(&cl, context, device, &base) } {
                Ok(p) => p,
                Err(e) => {
                    unsafe { (cl.ReleaseContext)(context) };
                    return Err(e);
                }
            };
            let mut units = Vec::with_capacity(num_units);
            for _ in 0..num_units {
                let name = CString::new("render_frame").map_err(|_| "nul")?;
                let mut kerr = 0;
                let kernel = unsafe { (cl.CreateKernel)(program, name.as_ptr(), &raw mut kerr) };
                let queue = unsafe { (cl.CreateCommandQueue)(context, device, 0, &raw mut err) };
                if kerr != CL_SUCCESS || kernel.is_null() || err != CL_SUCCESS || queue.is_null() {
                    continue;
                }
                units.push(std::sync::Mutex::new(Unit::new(kernel, queue)));
            }
            if units.is_empty() {
                unsafe {
                    (cl.ReleaseProgram)(program);
                    (cl.ReleaseContext)(context);
                }
                return Err("no render units".into());
            }
            let linear_name = CString::new("linear_luma").map_err(|_| "nul")?;
            let mut kerr = 0;
            let linear_kernel =
                unsafe { (cl.CreateKernel)(program, linear_name.as_ptr(), &raw mut kerr) };
            let cache_queue = unsafe { (cl.CreateCommandQueue)(context, device, 0, &raw mut err) };
            if kerr != CL_SUCCESS
                || linear_kernel.is_null()
                || err != CL_SUCCESS
                || cache_queue.is_null()
            {
                unsafe {
                    if !linear_kernel.is_null() {
                        (cl.ReleaseKernel)(linear_kernel);
                    }
                    if !cache_queue.is_null() {
                        (cl.ReleaseCommandQueue)(cache_queue);
                    }
                    for unit in &units {
                        if let Ok(unit) = unit.lock() {
                            (cl.ReleaseKernel)(unit.kernel);
                            (cl.ReleaseCommandQueue)(unit.queue);
                        }
                    }
                    (cl.ReleaseProgram)(program);
                    (cl.ReleaseContext)(context);
                }
                return Err("linear-luma render unit unavailable".into());
            }
            let mut cache_resources = vec![std::sync::Mutex::new(CacheResources {
                kernel: linear_kernel,
                queue: cache_queue,
            })];
            for _ in 1..UPLOAD_QUEUES {
                let mut kerr = 0;
                let kernel =
                    unsafe { (cl.CreateKernel)(program, linear_name.as_ptr(), &raw mut kerr) };
                let queue = unsafe { (cl.CreateCommandQueue)(context, device, 0, &raw mut err) };
                if kerr != CL_SUCCESS || kernel.is_null() || err != CL_SUCCESS || queue.is_null() {
                    unsafe {
                        if !kernel.is_null() {
                            (cl.ReleaseKernel)(kernel);
                        }
                        if !queue.is_null() {
                            (cl.ReleaseCommandQueue)(queue);
                        }
                    }
                    break;
                }
                cache_resources.push(std::sync::Mutex::new(CacheResources { kernel, queue }));
            }
            let device_name = unsafe { device_name(&cl, device) };
            return Ok(Self {
                cl,
                context,
                device,
                program,
                variants: std::sync::Mutex::new(Vec::new()),
                hi10,
                device_name,
                units,
                next_unit: std::sync::atomic::AtomicUsize::new(0),
                cache_resources,
                next_upload: std::sync::atomic::AtomicUsize::new(0),
                super_cache: std::sync::Mutex::new(Vec::new()),
                image_pool: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
                staging: std::sync::Mutex::new(Vec::new()),
                still: std::sync::Mutex::new(StillState {
                    queue: std::ptr::null_mut(),
                    kernels: [std::ptr::null_mut(); 2],
                    scratch: std::ptr::null_mut(),
                    dims: (0, 0),
                }),
                blend_pool: std::sync::Mutex::new(Vec::new()),
                blend_units: std::sync::Mutex::new(Vec::new()),
            });
        }
        Err("no usable GPU device".into())
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    fn acquire(&self) -> std::sync::MutexGuard<'_, Unit> {
        let n = self.units.len();
        let start = self
            .next_unit
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            % n;
        for i in 0..n {
            let idx = (start + i) % n;
            if let Ok(g) = self.units[idx].try_lock() {
                return g;
            }
        }
        self.units[start]
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

pub use svpflow_core::smooth_engine::GpuParams as KernelParams;

struct Buf<'a> {
    ctx: &'a GpuContext,
    mem: ClMem,
}
impl Drop for Buf<'_> {
    fn drop(&mut self) {
        unsafe { (self.ctx.cl.ReleaseMemObject)(self.mem) };
    }
}

impl GpuContext {
    pub fn cache_frame(&self, key: i64, planes: [UploadPlane<'_>; 3]) -> Option<SuperHandle> {
        let cell = {
            let mut cache = self.super_cache.lock().ok()?;
            if let Some(pos) = cache.iter().position(|(k, _)| *k == key) {
                let entry = cache.remove(pos);
                let cell = std::sync::Arc::clone(&entry.1);
                cache.push(entry);
                cell
            } else {
                let cell = std::sync::Arc::new(std::sync::OnceLock::new());
                if cache.len() >= SUPER_CACHE_CAP {
                    cache.remove(0);
                }
                cache.push((key, std::sync::Arc::clone(&cell)));
                cell
            }
        };
        let entry = cell
            .get_or_init(|| self.upload_frame(planes).map(std::sync::Arc::new))
            .clone();
        let Some(entry) = entry else {
            if let Ok(mut cache) = self.super_cache.lock() {
                cache.retain(|(_, cached)| !std::sync::Arc::ptr_eq(cached, &cell));
            }
            return None;
        };
        Some(SuperHandle { entry })
    }

    fn upload_frame(&self, planes: [UploadPlane<'_>; 3]) -> Option<SuperEntry> {
        let width = planes[0].width;
        let height = planes[0].height;
        let dims = (width, height + planes[1].height);
        let reused = self.image_pool.lock().ok().and_then(|mut pool| {
            let index = pool.iter().position(|images| images.dims == dims)?;
            Some(pool.swap_remove(index))
        });
        let (linear, packed) = if let Some(images) = reused {
            (images.linear, images.packed)
        } else {
            let depth = if self.hi10 {
                CL_UNORM_INT16
            } else {
                CL_UNORM_INT8
            };
            let packed = unsafe { self.create_image(CL_R, depth, dims.0, dims.1) }?;
            let linear = unsafe { self.create_image(CL_R, CL_FLOAT, width, height) }?;
            let mems = (linear.mem, packed.mem);
            std::mem::forget(linear);
            std::mem::forget(packed);
            mems
        };
        let mut entry = SuperEntry {
            still: std::sync::OnceLock::new(),
            mems: [linear, packed, packed, packed],
            ready: std::ptr::null_mut(),
            wait: self.cl.WaitForEvents,
            release_event: self.cl.ReleaseEvent,
            release: self.cl.ReleaseMemObject,
            pool: std::sync::Arc::clone(&self.image_pool),
            dims,
        };
        let resources = self.upload_resources()?;
        unsafe {
            self.enqueue_plane_at(resources.queue, packed, planes[0], 0, 0)?;
            self.enqueue_plane_at(resources.queue, packed, planes[1], 0, height)?;
            self.enqueue_plane_at(resources.queue, packed, planes[2], planes[1].width, height)?;
            if (self.cl.SetKernelArg)(
                resources.kernel,
                0,
                size_of::<ClMem>(),
                (&raw const entry.mems[3]).cast(),
            ) != CL_SUCCESS
                || (self.cl.SetKernelArg)(
                    resources.kernel,
                    1,
                    size_of::<ClMem>(),
                    (&raw const entry.mems[0]).cast(),
                ) != CL_SUCCESS
            {
                return None;
            }
            let global = [width, height];
            if (self.cl.EnqueueNDRangeKernel)(
                resources.queue,
                resources.kernel,
                2,
                std::ptr::null(),
                global.as_ptr(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                (&raw mut entry.ready).cast(),
            ) != CL_SUCCESS
                || (self.cl.Flush)(resources.queue) != CL_SUCCESS
            {
                return None;
            }
        }
        Some(entry)
    }

    unsafe fn enqueue_plane_at(
        &self,
        q: ClCommandQueue,
        mem: ClMem,
        plane: UploadPlane<'_>,
        x: usize,
        y: usize,
    ) -> Option<()> {
        let bytes = if self.hi10 { 2 } else { 1 };
        if plane.data.len()
            < plane
                .stride
                .saturating_mul(plane.height.saturating_sub(1))
                .saturating_add(plane.width * bytes)
        {
            return None;
        }
        let origin = [x, y, 0];
        let region = [plane.width, plane.height, 1];
        let rc = unsafe {
            (self.cl.EnqueueWriteImage)(
                q,
                mem,
                CL_TRUE,
                origin.as_ptr(),
                region.as_ptr(),
                plane.stride,
                0,
                plane.data.as_ptr().cast(),
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        (rc == CL_SUCCESS).then_some(())
    }

    unsafe fn create_image(
        &self,
        order: ClUint,
        data_type: ClUint,
        width: usize,
        height: usize,
    ) -> Option<Buf<'_>> {
        if width == 0 || height == 0 {
            return None;
        }
        let format = ClImageFormat {
            channel_order: order,
            channel_data_type: data_type,
        };
        let mut err = 0;
        let mem = unsafe {
            (self.cl.CreateImage2D)(
                self.context,
                CL_MEM_READ_WRITE,
                &raw const format,
                width,
                height,
                0,
                std::ptr::null_mut(),
                &raw mut err,
            )
        };
        (err == CL_SUCCESS && !mem.is_null()).then_some(Buf { ctx: self, mem })
    }

    unsafe fn enqueue_image_raw(
        &self,
        q: ClCommandQueue,
        mem: ClMem,
        width: usize,
        height: usize,
        row_pitch: usize,
        host: *const c_void,
    ) -> Option<()> {
        let origin = [0usize; 3];
        let region = [width, height, 1];
        let rc = unsafe {
            (self.cl.EnqueueWriteImage)(
                q,
                mem,
                CL_FALSE,
                origin.as_ptr(),
                region.as_ptr(),
                row_pitch,
                0,
                host,
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        (rc == CL_SUCCESS).then_some(())
    }

    unsafe fn ensure_image(
        &self,
        slot: &mut ImageSlot,
        width: usize,
        height: usize,
        data_type: ClUint,
    ) -> Option<ClMem> {
        if slot.mem.is_null()
            || slot.width != width
            || slot.height != height
            || slot.data_type != data_type
        {
            if !slot.mem.is_null() {
                unsafe { (self.cl.ReleaseMemObject)(slot.mem) };
            }
            let image = unsafe { self.create_image(CL_RGBA, data_type, width, height) }?;
            *slot = ImageSlot {
                mem: image.mem,
                width,
                height,
                data_type,
            };
            std::mem::forget(image);
        }
        Some(slot.mem)
    }

    unsafe fn ensure_buffer(&self, slot: &mut (ClMem, usize), bytes: usize) -> Option<ClMem> {
        let bytes = bytes.max(1);
        if slot.1 < bytes {
            if !slot.0.is_null() {
                unsafe { (self.cl.ReleaseMemObject)(slot.0) };
            }
            let mut err = 0;
            let mem = unsafe {
                (self.cl.CreateBuffer)(
                    self.context,
                    CL_MEM_READ_WRITE,
                    bytes,
                    std::ptr::null_mut(),
                    &raw mut err,
                )
            };
            if err != CL_SUCCESS || mem.is_null() {
                slot.1 = 0;
                slot.0 = std::ptr::null_mut();
                return None;
            }
            *slot = (mem, bytes);
        }
        Some(slot.0)
    }

    unsafe fn take_staging(&self, queue: ClCommandQueue, bytes: usize) -> Option<Staging> {
        let reused = self.staging.lock().ok().and_then(|mut pool| {
            let index = pool.iter().position(|staging| staging.len >= bytes)?;
            Some(pool.swap_remove(index))
        });
        if reused.is_some() {
            return reused;
        }
        let mut err = 0;
        let mem = unsafe {
            (self.cl.CreateBuffer)(
                self.context,
                CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                bytes,
                std::ptr::null_mut(),
                &raw mut err,
            )
        };
        if err != CL_SUCCESS || mem.is_null() {
            return None;
        }
        let host = unsafe {
            (self.cl.EnqueueMapBuffer)(
                queue,
                mem,
                CL_TRUE,
                CL_MAP_READ,
                0,
                bytes,
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
                &raw mut err,
            )
        };
        if err != CL_SUCCESS || host.is_null() {
            unsafe { (self.cl.ReleaseMemObject)(mem) };
            return None;
        }
        Some(Staging {
            mem,
            len: bytes,
            host: host.cast(),
        })
    }

    fn put_staging(&self, staging: Staging) {
        match self.staging.lock() {
            Ok(mut pool) => pool.push(staging),
            Err(_) => unsafe {
                (self.cl.ReleaseMemObject)(staging.mem);
            },
        }
    }

    fn upload_resources(&self) -> Option<std::sync::MutexGuard<'_, CacheResources>> {
        let count = self.cache_resources.len();
        let start = self
            .next_upload
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        for offset in 0..count {
            if let Ok(guard) = self.cache_resources[(start + offset) % count].try_lock() {
                return Some(guard);
            }
        }
        self.cache_resources[start % count].lock().ok()
    }

    fn variant_kernel(&self, kernels: &mut Vec<(u32, ClKernel)>, variant: u32) -> Option<ClKernel> {
        if let Some(&(_, kernel)) = kernels.iter().find(|(key, _)| *key == variant) {
            return Some(kernel);
        }
        let program = {
            let mut variants = self.variants.lock().ok()?;
            if let Some(&(_, program)) = variants.iter().find(|(key, _)| *key == variant) {
                program
            } else {
                let source = format!(
                    "#define ALGO {}\n#define CUBIC {}\n#define CUBIC_REF {}\n#define HAS_SAD {}\n#define LINEAR_LUMA {}\n#define DITHER {}\n#define BLEND {}\n#define STILL {}\n#define HI10P {}\n{}",
                    variant & 0xFF,
                    (variant >> 8) & 1,
                    (variant >> 9) & 1,
                    (variant >> 10) & 1,
                    (variant >> 11) & 1,
                    (variant >> 12) & 1,
                    (variant >> 13) & 1,
                    (variant >> 14) & 1,
                    u8::from(self.hi10),
                    KERNEL_SRC
                );
                let program =
                    unsafe { build_program(&self.cl, self.context, self.device, &source) }.ok()?;
                variants.push((variant, program));
                program
            }
        };
        let name = if variant & blend::BLEND_VARIANT == 0 {
            c"render_frame"
        } else {
            c"blend_frame"
        };
        let mut err = 0;
        let kernel = unsafe { (self.cl.CreateKernel)(program, name.as_ptr(), &raw mut err) };
        if err != CL_SUCCESS || kernel.is_null() {
            return None;
        }
        kernels.push((variant, kernel));
        Some(kernel)
    }

    unsafe fn wait_event(&self, queue: ClCommandQueue, event: ClEvent) -> bool {
        unsafe { (self.cl.Flush)(queue) };
        let done = loop {
            let mut status: ClInt = 1;
            let got = unsafe {
                (self.cl.GetEventInfo)(
                    event,
                    CL_EVENT_COMMAND_EXECUTION_STATUS,
                    size_of::<ClInt>(),
                    (&raw mut status).cast(),
                    std::ptr::null_mut(),
                )
            };
            if got != CL_SUCCESS || status < 0 {
                break false;
            }
            if status == 0 {
                break true;
            }
            std::thread::sleep(std::time::Duration::from_micros(50));
        };
        unsafe { (self.cl.ReleaseEvent)(event) };
        done
    }

    unsafe fn set_i32(&self, k: ClKernel, i: u32, v: i32) -> bool {
        unsafe { (self.cl.SetKernelArg)(k, i, 4, (&raw const v).cast()) == CL_SUCCESS }
    }

    fn variant(p: &KernelParams, hi10: bool) -> u32 {
        u32::try_from(p.algorithm & 0xFF).unwrap_or(23)
            | u32::from(p.cubic != 0) << 8
            | u32::from(p.cubic_ref != 0) << 9
            | u32::from(p.has_sad != 0) << 10
            | u32::from(p.linear_luma != 0) << 11
            | u32::from(p.dither != 0 && !hi10) << 12
    }

    unsafe fn upload_motion(
        &self,
        unit: &mut Unit,
        algorithm: i32,
        motion_key: i64,
        motion_width: usize,
        motion_height: usize,
        motions: [(&[u16], &[u16]); 4],
    ) -> Option<(ClMem, ClMem)> {
        let count = motion_width.checked_mul(motion_height)?;
        let motion_count = if algorithm == 23 { 4 } else { 2 };
        if count == 0
            || motions[..motion_count]
                .iter()
                .any(|(x, y)| x.len() < count || y.len() < count)
        {
            return None;
        }
        let Unit {
            queue,
            motion,
            motion_key: cached_motion_key,
            packed_base,
            packed_ext,
            ..
        } = unit;
        let q = *queue;
        packed_base.clear();
        packed_ext.clear();
        unsafe {
            let base_mem =
                self.ensure_image(&mut motion[0], motion_width, motion_height, CL_UNORM_INT16)?;
            if cached_motion_key[0] != motion_key {
                packed_base.reserve(count * 4);
                for i in 0..count {
                    packed_base.extend_from_slice(&[
                        motions[0].0[i],
                        motions[1].1[i],
                        motions[1].0[i],
                        motions[0].1[i],
                    ]);
                }
                self.enqueue_image_raw(
                    q,
                    base_mem,
                    motion_width,
                    motion_height,
                    motion_width * 8,
                    packed_base.as_ptr().cast(),
                )?;
                cached_motion_key[0] = motion_key;
            }
            let ext_mem = if algorithm == 23 {
                let mem =
                    self.ensure_image(&mut motion[1], motion_width, motion_height, CL_UNORM_INT16)?;
                if cached_motion_key[1] != motion_key {
                    packed_ext.reserve(count * 4);
                    for i in 0..count {
                        packed_ext.extend_from_slice(&[
                            motions[2].0[i],
                            motions[3].1[i],
                            motions[3].0[i],
                            motions[2].1[i],
                        ]);
                    }
                    self.enqueue_image_raw(
                        q,
                        mem,
                        motion_width,
                        motion_height,
                        motion_width * 8,
                        packed_ext.as_ptr().cast(),
                    )?;
                    cached_motion_key[1] = motion_key;
                }
                mem
            } else {
                base_mem
            };
            Some((base_mem, ext_mem))
        }
    }

    fn pack_masks(
        packed: &mut Vec<u8>,
        count: usize,
        needs_coverage: bool,
        coverage: (&[u8], &[u8]),
        area: Option<(&[u8], &[u8])>,
    ) -> Option<()> {
        if needs_coverage && (coverage.0.len() < count || coverage.1.len() < count) {
            return None;
        }
        if let Some((a, b)) = area
            && (a.len() < count || b.len() < count)
        {
            return None;
        }
        packed.reserve(count * 4);
        for i in 0..count {
            let sad_f = area.map_or(0, |(a, _)| a[i]);
            let sad_b = area.map_or(0, |(_, b)| b[i]);
            packed.extend_from_slice(&[
                sad_b,
                if needs_coverage { coverage.0[i] } else { 0 },
                if needs_coverage { coverage.1[i] } else { 0 },
                sad_f,
            ]);
        }
        Some(())
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn bind_fields(
        &self,
        unit: &mut Unit,
        variant: u32,
        py: &KernelParams,
        motion_key: i64,
        motion_width: usize,
        motion_height: usize,
        motions: [(&[u16], &[u16]); 4],
        coverage: (&[u8], &[u8]),
        area: Option<(&[u8], &[u8])>,
    ) -> Option<ClKernel> {
        let count = motion_width.checked_mul(motion_height)?;
        let k = self.variant_kernel(&mut unit.variant_kernels, variant)?;
        let q = unit.queue;
        unsafe {
            let (base_mem, ext_mem) = self.upload_motion(
                unit,
                py.algorithm,
                motion_key,
                motion_width,
                motion_height,
                motions,
            )?;
            let Unit {
                mask, packed_mask, ..
            } = unit;
            packed_mask.clear();

            let needs_coverage = py.algorithm >= 21;
            let needs_mask = needs_coverage || py.has_sad != 0;
            let mask_mem = if needs_mask {
                Self::pack_masks(packed_mask, count, needs_coverage, coverage, area)?;
                let mem = self.ensure_image(mask, motion_width, motion_height, CL_UNORM_INT8)?;
                self.enqueue_image_raw(
                    q,
                    mem,
                    motion_width,
                    motion_height,
                    motion_width * 4,
                    packed_mask.as_ptr().cast(),
                )?;
                mem
            } else {
                base_mem
            };
            for (index, mem) in [(4, base_mem), (5, ext_mem), (6, mask_mem)] {
                if (self.cl.SetKernelArg)(k, index, size_of::<ClMem>(), (&raw const mem).cast())
                    != CL_SUCCESS
                {
                    return None;
                }
            }
        }
        Some(k)
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn enqueue_plane(
        &self,
        q: ClCommandQueue,
        k: ClKernel,
        dmem: ClMem,
        stride: i32,
        p: KernelParams,
        s0: GpuBuf,
        s1: GpuBuf,
        waits: &[ClEvent],
    ) -> Option<()> {
        unsafe {
            (self.cl.SetKernelArg)(k, 0, size_of::<ClMem>(), (&raw const dmem).cast());
            self.set_i32(k, 1, stride).then_some(())?;
            (self.cl.SetKernelArg)(k, 2, size_of::<ClMem>(), (&raw const s0.0).cast());
            (self.cl.SetKernelArg)(k, 3, size_of::<ClMem>(), (&raw const s1.0).cast());
            if (self.cl.SetKernelArg)(k, 7, size_of::<KernelParams>(), (&raw const p).cast())
                != CL_SUCCESS
            {
                return None;
            }
            let global = [
                usize::try_from(p.width).ok()?,
                usize::try_from(p.height).ok()?,
            ];
            if (self.cl.EnqueueNDRangeKernel)(
                q,
                k,
                2,
                std::ptr::null(),
                global.as_ptr(),
                std::ptr::null(),
                ClUint::try_from(waits.len()).ok()?,
                if waits.is_empty() {
                    std::ptr::null()
                } else {
                    waits.as_ptr().cast()
                },
                std::ptr::null_mut(),
            ) != CL_SUCCESS
            {
                return None;
            }
        }
        Some(())
    }

    pub fn still_mask(
        &self,
        first: &SuperHandle,
        second: &SuperHandle,
        limits: [f32; 3],
    ) -> Option<StillMask> {
        if self.hi10 || first.entry.dims != second.entry.dims {
            return None;
        }
        let dims = (first.entry.dims.0, first.entry.dims.1 / 3 * 2);
        let compute = || {
            let image = unsafe { self.create_image(CL_R, CL_UNORM_INT8, dims.0, dims.1) }?;
            let mut state = self.still.lock().ok()?;
            for (kernel, name) in state
                .kernels
                .iter_mut()
                .zip([c"still_pixels", c"still_mask"])
            {
                if kernel.is_null() {
                    let mut err = 0;
                    *kernel = unsafe {
                        (self.cl.CreateKernel)(self.program, name.as_ptr(), &raw mut err)
                    };
                    if err != CL_SUCCESS || kernel.is_null() {
                        return None;
                    }
                }
            }
            if state.dims != dims {
                let scratch = unsafe { self.create_image(CL_R, CL_UNORM_INT8, dims.0, dims.1) }?;
                if !state.scratch.is_null() {
                    unsafe { (self.cl.ReleaseMemObject)(state.scratch) };
                }
                state.scratch = scratch.mem;
                state.dims = dims;
                std::mem::forget(scratch);
            }
            if state.queue.is_null() {
                let mut err = 0;
                state.queue = unsafe {
                    (self.cl.CreateCommandQueue)(self.context, self.device, 0, &raw mut err)
                };
                if err != CL_SUCCESS || state.queue.is_null() {
                    return None;
                }
            }
            let [pixels, mask] = state.kernels;
            let queue = state.queue;
            let waits = Self::source_waits(first.bufs()[1], second.bufs()[1]);
            let global = [dims.0, dims.1];
            let mut done: ClEvent = std::ptr::null_mut();
            unsafe {
                let set = |kernel, index, mem: ClMem| {
                    (self.cl.SetKernelArg)(
                        kernel,
                        index,
                        size_of::<ClMem>(),
                        (&raw const mem).cast(),
                    ) == CL_SUCCESS
                };
                let run = |kernel, waits: &[ClEvent], done: *mut ClEvent| {
                    (self.cl.EnqueueNDRangeKernel)(
                        queue,
                        kernel,
                        2,
                        std::ptr::null(),
                        global.as_ptr(),
                        std::ptr::null(),
                        waits.len() as ClUint,
                        if waits.is_empty() {
                            std::ptr::null()
                        } else {
                            waits.as_ptr().cast()
                        },
                        done.cast(),
                    ) == CL_SUCCESS
                };
                for (index, limit) in (3..).zip(limits) {
                    (self.cl.SetKernelArg)(
                        pixels,
                        index,
                        size_of::<f32>(),
                        (&raw const limit).cast(),
                    );
                }
                let ready = set(pixels, 0, first.entry.mems[1])
                    && set(pixels, 1, second.entry.mems[1])
                    && set(pixels, 2, state.scratch)
                    && set(mask, 0, state.scratch)
                    && set(mask, 1, image.mem)
                    && run(pixels, &waits, std::ptr::null_mut())
                    && run(mask, &[], &raw mut done);
                if !ready || (self.cl.Flush)(queue) != CL_SUCCESS {
                    return None;
                }
            }
            let mem = image.mem;
            std::mem::forget(image);
            Some((StillMask(mem, done), std::sync::Arc::clone(&second.entry)))
        };
        first
            .entry
            .still
            .get_or_init(compute)
            .as_ref()
            .map(|(still, _)| *still)
    }

    fn source_waits(src0: GpuBuf, src1: GpuBuf) -> Vec<ClEvent> {
        let mut waits = Vec::with_capacity(2);
        for event in [src0.1, src1.1] {
            if !event.is_null() && !waits.contains(&event) {
                waits.push(event);
            }
        }
        waits
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render_frame(
        &self,
        src0: [GpuBuf; 3],
        src1: [GpuBuf; 3],
        dst_y: &mut [u8],
        sy: i32,
        py: KernelParams,
        dst_u: &mut [u8],
        su: i32,
        pu: KernelParams,
        dst_v: &mut [u8],
        sv: i32,
        pv: KernelParams,
        motion_key: i64,
        motion_width: usize,
        motion_height: usize,
        motions: [(&[u16], &[u16]); 4],
        coverage: (&[u8], &[u8]),
        area: Option<(&[u8], &[u8])>,
        still: Option<StillMask>,
    ) -> Option<()> {
        let mut unit = self.acquire();
        unsafe {
            let k = self.bind_fields(
                &mut unit,
                Self::variant(&py, self.hi10) | u32::from(still.is_some()) << 14,
                &py,
                motion_key,
                motion_width,
                motion_height,
                motions,
                coverage,
                area,
            )?;
            let mut waits = Self::source_waits(src0[0], src1[0]);
            if let Some(StillMask(still, ready)) = still {
                if (self.cl.SetKernelArg)(k, 8, size_of::<ClMem>(), (&raw const still).cast())
                    != CL_SUCCESS
                {
                    return None;
                }
                waits.push(ready);
            }
            let q = unit.queue;
            let depth = if self.hi10 { 2 } else { 1 };
            let y_mem = self.ensure_buffer(&mut unit.dst[0], dst_y.len())?;
            self.enqueue_plane(q, k, y_mem, sy / depth, py, src0[0], src1[0], &waits)?;
            let u_mem = self.ensure_buffer(&mut unit.dst[1], dst_u.len())?;
            self.enqueue_plane(q, k, u_mem, su / depth, pu, src0[1], src1[1], &[])?;
            let v_mem = self.ensure_buffer(&mut unit.dst[2], dst_v.len())?;
            self.enqueue_plane(q, k, v_mem, sv / depth, pv, src0[2], src1[2], &[])?;
            let mut done: ClEvent = std::ptr::null_mut();
            let total = dst_y.len() + dst_u.len() + dst_v.len();
            let staging = self.take_staging(q, total)?;
            let host = staging.host;
            let mut offset = 0;
            for (index, (dmem, output)) in [(y_mem, &*dst_y), (u_mem, &*dst_u), (v_mem, &*dst_v)]
                .into_iter()
                .enumerate()
            {
                if (self.cl.EnqueueReadBuffer)(
                    q,
                    dmem,
                    CL_FALSE,
                    0,
                    output.len(),
                    host.add(offset).cast(),
                    0,
                    std::ptr::null(),
                    if index == 2 {
                        (&raw mut done).cast()
                    } else {
                        std::ptr::null_mut()
                    },
                ) != CL_SUCCESS
                {
                    if !done.is_null() {
                        (self.cl.ReleaseEvent)(done);
                    }
                    (self.cl.Finish)(q);
                    self.put_staging(staging);
                    return None;
                }
                offset += output.len();
            }
            let ok = self.wait_event(q, done);
            drop(unit);
            if !ok {
                (self.cl.ReleaseMemObject)(staging.mem);
                return None;
            }
            let mut offset = 0;
            for output in [dst_y, dst_u, dst_v] {
                output.copy_from_slice(std::slice::from_raw_parts(host.add(offset), output.len()));
                offset += output.len();
            }
            self.put_staging(staging);
        }
        Some(())
    }
}

unsafe fn build_program(
    cl: &OpenCl,
    context: ClContext,
    device: ClDeviceId,
    source: &str,
) -> Result<ClProgram, String> {
    let cstr = CString::new(source).map_err(|_| "nul in source")?;
    let ptr = cstr.as_ptr();
    let len = source.len();
    let mut err = 0;
    let program = unsafe {
        (cl.CreateProgramWithSource)(context, 1, &raw const ptr, &raw const len, &raw mut err)
    };
    if err != CL_SUCCESS || program.is_null() {
        return Err(format!("clCreateProgramWithSource failed ({err})"));
    }
    let build = unsafe {
        (cl.BuildProgram)(
            program,
            1,
            &raw const device,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
        )
    };
    if build != CL_SUCCESS {
        let mut log_size = 0usize;
        unsafe {
            (cl.GetProgramBuildInfo)(
                program,
                device,
                CL_PROGRAM_BUILD_LOG,
                0,
                std::ptr::null_mut(),
                &raw mut log_size,
            );
        }
        let mut log = vec![0u8; log_size];
        unsafe {
            (cl.GetProgramBuildInfo)(
                program,
                device,
                CL_PROGRAM_BUILD_LOG,
                log_size,
                log.as_mut_ptr().cast(),
                std::ptr::null_mut(),
            );
        }
        unsafe { (cl.ReleaseProgram)(program) };
        return Err(format!("build failed: {}", String::from_utf8_lossy(&log)));
    }
    Ok(program)
}

unsafe fn device_name(cl: &OpenCl, device: ClDeviceId) -> String {
    let mut size = 0usize;
    unsafe {
        (cl.GetDeviceInfo)(
            device,
            CL_DEVICE_NAME,
            0,
            std::ptr::null_mut(),
            &raw mut size,
        );
    }
    let mut buf = vec![0u8; size];
    unsafe {
        (cl.GetDeviceInfo)(
            device,
            CL_DEVICE_NAME,
            size,
            buf.as_mut_ptr().cast(),
            std::ptr::null_mut(),
        );
    }
    while buf.last() == Some(&0) {
        buf.pop();
    }
    String::from_utf8_lossy(&buf).into_owned()
}

pub(crate) const KERNEL_SRC: &str = r"
#ifndef ALGO
#define ALGO 23
#endif
#ifndef CUBIC
#define CUBIC 0
#endif
#ifndef CUBIC_REF
#define CUBIC_REF 0
#endif
#ifndef HAS_SAD
#define HAS_SAD 0
#endif
#ifndef LINEAR_LUMA
#define LINEAR_LUMA 1
#endif
#ifndef DITHER
#define DITHER 0
#endif
#ifndef HI10P
#define HI10P 0
#endif
#ifndef BLEND
#define BLEND 0
#endif
#ifndef STILL
#define STILL 0
#endif
#define STILL_AT(still, p, x, y) (read_imagef(still, linear_sampler, \
    ((float2)(x, y) + 0.5f) * (float2)(p.x_ratio, p.y_ratio)).x > 0.4f)
#if HI10P
#define REF_SCALE(p) ((p)->linear_luma ? 256.0f : 16384.0f)
#else
#define REF_SCALE(p) 255.0f
#endif
const sampler_t field_sampler = CLK_NORMALIZED_COORDS_TRUE |
    CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
const sampler_t linear_sampler = CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
const sampler_t nearest_sampler = CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

typedef struct {
    int algorithm, width, height, x_ratio, y_ratio, pel;
    int block_w, block_h, origin_x, origin_y, phase, has_sad, linear_luma, cubic;
    int cubic_ref, offset_x, offset_y;
    float sad_blend;
    int dither;
} Params;

#if DITHER
__constant int bayer[8][8] = {
    1,49,13,61,4,52,16,64, 33,17,45,29,36,20,48,32,
    9,57,5,53,12,60,8,56, 41,25,37,21,44,28,40,24,
    3,51,15,63,2,50,14,62, 35,19,47,31,34,18,46,30,
    11,59,7,55,10,58,6,54, 43,27,39,23,42,26,38,22};
#endif

inline float median3(float a, float b, float c) {
    float lo = fmin(a, b);
    float hi = a + b - lo;
    return fmax(lo, fmin(hi, c));
}

inline float4 cubic_sample(read_only image2d_t image, float2 position) {
    float2 grid = position - (float2)(0.5f, 0.5f);
    float2 index = floor(grid);
    float2 f = grid - index;
    float2 r = 1.0f - f;
    float2 r2 = r * r;
    float2 f2 = f * f;
    float2 w0 = 1.0f/6.0f * r2 * r;
    float2 w1 = 2.0f/3.0f - 0.5f * f2 * (2.0f - f);
    float2 w2 = 2.0f/3.0f - 0.5f * r2 * (2.0f - r);
    float2 w3 = 1.0f/6.0f * f2 * f;
    float2 g0 = w0 + w1;
    float2 g1 = w2 + w3;
    float2 h0 = (w1 / g0) - 0.5f + index;
    float2 h1 = (w3 / g1) + 1.5f + index;
    float4 a = read_imagef(image, linear_sampler, h0);
    float4 b = read_imagef(image, linear_sampler, (float2)(h1.x, h0.y));
    float4 c = read_imagef(image, linear_sampler, (float2)(h0.x, h1.y));
    float4 d = read_imagef(image, linear_sampler, h1);
    a = mix(c, a, g0.y);
    b = mix(d, b, g0.y);
    return mix(b, a, g0.x);
}

inline float4 field_sample(read_only image2d_t image, float2 position, int cubic) {
#if CUBIC
    return cubic_sample(image, position);
#else
    return read_imagef(image, field_sampler, position);
#endif
}

inline float2 moved(const Params *p, float vx, float vy, int time) {
    float2 step = native_divide(
        (float2)(vx*65535.0f-1024.0f, vy*65535.0f-1024.0f) * time,
        (float2)((float)(p->x_ratio*p->pel), (float)(p->y_ratio*p->pel)) * 256.0f);
    return clamp((float2)(get_global_id(0), get_global_id(1)) + step,
        (float2)(0.0f, 0.0f), (float2)(p->width-1, p->height-1));
}

inline float source_sample(
    read_only image2d_t source, const Params *p, float vx, float vy, int time)
{
    float2 position = ((float2)(p->offset_x, p->offset_y) + (float2)(0.5f, 0.5f))
        + moved(p, vx, vy, time);
#if CUBIC_REF
    return REF_SCALE(p) * cubic_sample(source, position).x;
#else
    return REF_SCALE(p) * read_imagef(source, linear_sampler, position).x;
#endif
}

inline float linear_source_sample(
    read_only image2d_t source, const Params *p, float vx, float vy, int time)
{
    float2 position = ((float2)(p->offset_x, p->offset_y) + (float2)(0.5f, 0.5f))
        + moved(p, vx, vy, time);
    return REF_SCALE(p) * read_imagef(source, linear_sampler, position).x;
}

inline float base_sample(read_only image2d_t source, const Params *p) {
    float2 position = ((float2)(p->offset_x, p->offset_y) + (float2)(0.5f, 0.5f))
        + (float2)(get_global_id(0), get_global_id(1));
#if CUBIC_REF
    return REF_SCALE(p) * cubic_sample(source, position).x;
#else
    return REF_SCALE(p) * read_imagef(source, linear_sampler, position).x;
#endif
}

kernel void linear_luma(read_only image2d_t source, write_only image2d_t destination) {
    int2 position = (int2)(get_global_id(0), get_global_id(1));
#if HI10P
    float value = 64.0f * read_imagef(source, nearest_sampler, position).x;
#else
    float value = read_imagef(source, nearest_sampler, position).x;
#endif
    if (value < 0.081f) value = native_divide(value, 4.5f);
    else value = native_powr(native_divide(value+0.099f, 1.099f), 1.0f/0.45f);
    write_imagef(destination, position, (float4)(value, 0.0f, 0.0f, 0.0f));
}

kernel void render_frame(
    global uchar *destination, int destination_stride,
    read_only image2d_t source_f, read_only image2d_t source_b,
    read_only image2d_t vectors, read_only image2d_t vectors_ext,
    read_only image2d_t masks, Params p
#if STILL
    , read_only image2d_t still
#endif
    )
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= p.width || y >= p.height) return;

    float2 field_position = native_divide(
        (float2)(x*p.x_ratio-p.origin_x, y*p.y_ratio-p.origin_y),
        (float2)(p.block_w, p.block_h));
    float4 vector = field_sample(vectors, field_position, CUBIC);
#if ALGO >= 21 || HAS_SAD
    float4 mask = field_sample(masks, field_position, CUBIC);
#endif
    float time = native_divide((float)p.phase, 256.0f);

    float ref_b = source_sample(source_b, &p, vector.z, vector.y, 256-p.phase);
    float ref_f = source_sample(source_f, &p, vector.x, vector.w, p.phase);
#if ALGO == 13 || ALGO == 22 || HAS_SAD
    float ref_b0 = base_sample(source_b, &p);
    float ref_f0 = base_sample(source_f, &p);
#endif
#if ALGO == 23
    float4 ext = field_sample(vectors_ext, field_position, CUBIC);
    float ref_bb = linear_source_sample(source_b, &p, ext.z, ext.y, 256-p.phase);
    float ref_ff = linear_source_sample(source_f, &p, ext.x, ext.w, p.phase);
#endif
    float result;

#if ALGO == 1
    result = ref_b + ref_f*0.00001f;
#elif ALGO == 2
    result = ref_f + ref_b*0.00001f;
#elif ALGO == 11
    result = mix(ref_f, ref_b, time);
#elif ALGO == 13
    result = median3(ref_f, ref_b, mix(ref_f0, ref_b0, p.sad_blend));
#elif ALGO == 21
    result = mix(mix(ref_f, ref_b, mask.y), mix(ref_b, ref_f, mask.z), time);
#elif ALGO == 22
    result = median3(
        mix(ref_f, ref_b, mask.y),
        mix(ref_b, ref_f, mask.z),
        mix(ref_f0, ref_b0, time));
#else
    result = mix(
        mix(ref_f, median3(ref_b, ref_bb, ref_f), mask.y),
        mix(ref_b, median3(ref_b, ref_ff, ref_f), mask.z),
        time);
#endif

#if HAS_SAD
# if ALGO == 1
    result = mix(result, ref_b0, mask.x);
# elif ALGO == 2
    result = mix(result, ref_f0, mask.w);
# else
    result = mix(result, mix(ref_f0, ref_b0, p.sad_blend), fmax(mask.w, mask.x));
# endif
#endif

#if STILL
    if (STILL_AT(still, p, x, y))
        result = mix(base_sample(source_f, &p), base_sample(source_b, &p), time);
#endif

#if LINEAR_LUMA
    if (p.linear_luma) {
#if HI10P
        result = native_divide(result, 256.0f);
#else
        result = native_divide(result, 255.0f);
#endif
        if (result < 0.018f) result *= 4.5f;
        else result = 1.099f*native_powr(result, 0.45f)-0.099f;
#if HI10P
        result *= 256.0f;
#else
        result *= 255.0f;
#endif
    }
#endif
#if HI10P
    ((global ushort *)destination)[y*destination_stride+x] =
        (ushort)clamp(round(4.0f*result), 0.0f, 1023.0f);
#else
# if DITHER
    result += native_divide((float)(bayer[x%8][y%8]-32), 65.0f);
# endif
    destination[y*destination_stride+x] = (uchar)clamp(round(result), 0.0f, 255.0f);
#endif
}

#if BLEND && !HI10P
#define MAX_STEPS 16
typedef struct {
    int count;
    int phase[MAX_STEPS];
    float sad_blend[MAX_STEPS];
    float weight[MAX_STEPS];
    float gamma;
} Steps;

inline float4 layer_sample(read_only image2d_array_t image, float2 position, int layer) {
#if CUBIC
    float2 grid = position - (float2)(0.5f, 0.5f);
    float2 index = floor(grid);
    float2 f = grid - index;
    float2 r = 1.0f - f;
    float2 r2 = r * r;
    float2 f2 = f * f;
    float2 w0 = 1.0f/6.0f * r2 * r;
    float2 w1 = 2.0f/3.0f - 0.5f * f2 * (2.0f - f);
    float2 w2 = 2.0f/3.0f - 0.5f * r2 * (2.0f - r);
    float2 w3 = 1.0f/6.0f * f2 * f;
    float2 g0 = w0 + w1;
    float2 g1 = w2 + w3;
    float2 h0 = (w1 / g0) - 0.5f + index;
    float2 h1 = (w3 / g1) + 1.5f + index;
    float4 a = read_imagef(image, linear_sampler, (float4)(h0, (float)layer, 0.0f));
    float4 b = read_imagef(image, linear_sampler, (float4)(h1.x, h0.y, (float)layer, 0.0f));
    float4 c = read_imagef(image, linear_sampler, (float4)(h0.x, h1.y, (float)layer, 0.0f));
    float4 d = read_imagef(image, linear_sampler, (float4)(h1, (float)layer, 0.0f));
    a = mix(c, a, g0.y);
    b = mix(d, b, g0.y);
    return mix(b, a, g0.x);
#else
    return read_imagef(image, field_sampler, (float4)(position, (float)layer, 0.0f));
#endif
}

kernel void blend_frame(
    global float *sums, int stride,
    read_only image2d_t source_f, read_only image2d_t source_b,
    read_only image2d_t vectors, read_only image2d_t vectors_ext,
    read_only image2d_array_t masks, Params p, Steps steps
#if STILL
    , read_only image2d_t still
#endif
    )
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= p.width || y >= p.height) return;

    float2 field_position = native_divide(
        (float2)(x*p.x_ratio-p.origin_x, y*p.y_ratio-p.origin_y),
        (float2)(p.block_w, p.block_h));
    float4 vector = field_sample(vectors, field_position, CUBIC);
#if ALGO == 13 || ALGO == 22 || HAS_SAD
    float ref_b0 = base_sample(source_b, &p);
    float ref_f0 = base_sample(source_f, &p);
#endif
#if ALGO == 23
    float4 ext = field_sample(vectors_ext, field_position, CUBIC);
#endif
    global float *sum = sums + (p.offset_y+y)*stride + p.offset_x + x;
    float total = *sum;
#if STILL
    int kept = STILL_AT(still, p, x, y);
    float kept_f = 0.0f, kept_b = 0.0f;
    if (kept) {
        kept_f = base_sample(source_f, &p);
        kept_b = base_sample(source_b, &p);
    }
#endif

    for (int step = 0; step < steps.count; step++) {
        p.phase = steps.phase[step];
        p.sad_blend = steps.sad_blend[step];
        float time = native_divide((float)p.phase, 256.0f);
        float result;
#if STILL
        if (kept) {
            result = mix(kept_f, kept_b, time);
        } else
#endif
        {
#if ALGO >= 21 || HAS_SAD
        float4 mask = layer_sample(masks, field_position, step);
#endif

        float ref_b = source_sample(source_b, &p, vector.z, vector.y, 256-p.phase);
        float ref_f = source_sample(source_f, &p, vector.x, vector.w, p.phase);
#if ALGO == 23
        float ref_bb = linear_source_sample(source_b, &p, ext.z, ext.y, 256-p.phase);
        float ref_ff = linear_source_sample(source_f, &p, ext.x, ext.w, p.phase);
#endif

#if ALGO == 1
        result = ref_b + ref_f*0.00001f;
#elif ALGO == 2
        result = ref_f + ref_b*0.00001f;
#elif ALGO == 11
        result = mix(ref_f, ref_b, time);
#elif ALGO == 13
        result = median3(ref_f, ref_b, mix(ref_f0, ref_b0, p.sad_blend));
#elif ALGO == 21
        result = mix(mix(ref_f, ref_b, mask.y), mix(ref_b, ref_f, mask.z), time);
#elif ALGO == 22
        result = median3(
            mix(ref_f, ref_b, mask.y),
            mix(ref_b, ref_f, mask.z),
            mix(ref_f0, ref_b0, time));
#else
        result = mix(
            mix(ref_f, median3(ref_b, ref_bb, ref_f), mask.y),
            mix(ref_b, median3(ref_b, ref_ff, ref_f), mask.z),
            time);
#endif

#if HAS_SAD
# if ALGO == 1
        result = mix(result, ref_b0, mask.x);
# elif ALGO == 2
        result = mix(result, ref_f0, mask.w);
# else
        result = mix(result, mix(ref_f0, ref_b0, p.sad_blend), fmax(mask.w, mask.x));
# endif
#endif
        }

#if LINEAR_LUMA
        if (p.linear_luma) {
            result = native_divide(result, 255.0f);
            if (result < 0.018f) result *= 4.5f;
            else result = 1.099f*native_powr(result, 0.45f)-0.099f;
            result *= 255.0f;
        }
#endif
#if DITHER
        result += native_divide((float)(bayer[x%8][y%8]-32), 65.0f);
#endif
        {
#pragma OPENCL FP_CONTRACT OFF
            float value = clamp(round(result), 0.0f, 255.0f);
            if (steps.gamma != 1.0f) value = powr(value / 255.0f, steps.gamma);
            float term = steps.weight[step] * value;
            total = term + total;
        }
    }
    *sum = total;
}
#endif

inline float detail(read_only image2d_t image, int2 at, float value) {
    return value - read_imagef(image, linear_sampler, (float2)(at.x + 1, at.y + 1)).x;
}

kernel void still_pixels(
    read_only image2d_t first, read_only image2d_t second, write_only image2d_t same,
    float limit, float strength, float tolerance)
{
    int2 at = (int2)(get_global_id(0), get_global_id(1));
    float a = read_imagef(first, nearest_sampler, at).x;
    float b = read_imagef(second, nearest_sampler, at).x;
    float edge_a = detail(first, at, a), edge_b = detail(second, at, b);
    float edge = fmax(fabs(edge_a), fabs(edge_b));
    int still = fabs(a - b) <= limit
        || (edge > strength && fabs(edge_a - edge_b) <= tolerance*edge);
    write_imagef(same, at, (float4)(still, 0.0f, 0.0f, 0.0f));
}

kernel void still_mask(read_only image2d_t same, write_only image2d_t still)
{
    int2 at = (int2)(get_global_id(0), get_global_id(1));
    float2 corner = (float2)(at.x, at.y);
    float full = fmax(
        fmax(read_imagef(same, linear_sampler, corner).x,
            read_imagef(same, linear_sampler, corner + (float2)(1.0f, 0.0f)).x),
        fmax(read_imagef(same, linear_sampler, corner + (float2)(0.0f, 1.0f)).x,
            read_imagef(same, linear_sampler, corner + (float2)(1.0f, 1.0f)).x));
    write_imagef(still, at, (float4)(full > 0.99f, 0.0f, 0.0f, 0.0f));
}

kernel void blend_source(
    global float *sums, int stride, read_only image2d_t source, float weight,
    float gamma, int luma)
{
#pragma OPENCL FP_CONTRACT OFF
    int2 position = (int2)(get_global_id(0), get_global_id(1));
    float value = round(255.0f * read_imagef(source, nearest_sampler, position).x);
    if (gamma != 1.0f && position.y < luma) value = powr(value / 255.0f, gamma);
    float term = weight * value;
    global float *sum = sums + position.y*stride + position.x;
    *sum = term + *sum;
}

kernel void blend_resolve(
    global float *sums, global uchar *destination, float divisor, float gamma, uint luma)
{
    size_t i = get_global_id(0);
    float value = sums[i] / divisor;
    if (gamma != 1.0f && i < luma) value = 255.0f * powr(value, 1.0f / gamma);
    destination[i] = (uchar)clamp(rint(value), 0.0f, 255.0f);
    sums[i] = 0.0f;
}
";
