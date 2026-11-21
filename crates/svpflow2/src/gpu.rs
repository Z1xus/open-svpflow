#![allow(unsafe_code)]
#![allow(clippy::missing_safety_doc)]
use core::ffi::c_void;
use std::ffi::CString;

use svpflow_host::context::{finish_source, has_shared_ptr, lookup_source, plane_fingerprint};
use svpflow_host::hardware::GpuProfile;
use svpflow_host::opencl::{
    CL_DEVICE_TYPE_GPU, CL_FALSE, CL_MAP_READ, CL_MEM_ALLOC_HOST_PTR, CL_MEM_COPY_HOST_PTR,
    CL_MEM_OBJECT_ALLOCATION_FAILURE, CL_MEM_READ_WRITE, CL_OUT_OF_HOST_MEMORY,
    CL_OUT_OF_RESOURCES, CL_SUCCESS, CL_TRUE, ClImageFormat, ClInt, ClUint, FnCreateImage,
    FnCreateImage2D, FnEnqueueReadBufferRect, FnEnqueueWriteImage, FnGetProgramBuildInfo,
    FnRelease, FnWaitForEvents, OpenCl as CoreCl,
};

#[allow(non_snake_case)]
struct OpenCl {
    core: CoreCl,
    GetProgramBuildInfo: FnGetProgramBuildInfo,
    CreateImage2D: FnCreateImage2D,
    CreateImage: Option<FnCreateImage>,
    EnqueueReadBufferRect: FnEnqueueReadBufferRect,
    EnqueueWriteImage: FnEnqueueWriteImage,
    EnqueueReadImage: FnEnqueueWriteImage,
    ReleaseKernel: FnRelease,
    ReleaseProgram: FnRelease,
    ReleaseCommandQueue: FnRelease,
}

impl std::ops::Deref for OpenCl {
    type Target = CoreCl;

    fn deref(&self) -> &CoreCl {
        &self.core
    }
}

impl OpenCl {
    unsafe fn load() -> Result<Self, String> {
        let core = unsafe { CoreCl::load()? };
        Ok(Self {
            GetProgramBuildInfo: unsafe { core.symbol("clGetProgramBuildInfo\0")? },
            CreateImage2D: unsafe { core.symbol("clCreateImage2D\0")? },
            CreateImage: unsafe { core.symbol("clCreateImage\0") }.ok(),
            EnqueueReadBufferRect: unsafe { core.symbol("clEnqueueReadBufferRect\0")? },
            EnqueueWriteImage: unsafe { core.symbol("clEnqueueWriteImage\0")? },
            EnqueueReadImage: unsafe { core.symbol("clEnqueueReadImage\0")? },
            ReleaseKernel: unsafe { core.symbol("clReleaseKernel\0")? },
            ReleaseProgram: unsafe { core.symbol("clReleaseProgram\0")? },
            ReleaseCommandQueue: unsafe { core.symbol("clReleaseCommandQueue\0")? },
            core,
        })
    }
}

mod blend;
mod cover;
mod layer;
mod vram;

pub use layer::{Failed, Layer, LayerHeld, LayerPair};

pub use blend::BlendJob;
#[cfg(feature = "blend")]
pub use blend::OutputPlane;
pub use cover::Cover;

type ClDeviceId = *mut c_void;
type ClContext = *mut c_void;
type ClCommandQueue = *mut c_void;
type ClProgram = *mut c_void;
type ClKernel = *mut c_void;
type ClMem = *mut c_void;
type ClEvent = *mut c_void;

const CL_DEVICE_NAME: ClUint = 0x102B;
const CL_DEVICE_GLOBAL_MEM_SIZE: ClUint = 0x101F;
const CL_DEVICE_MAX_PARAMETER_SIZE: ClUint = 0x1017;
const CL_MAP_WRITE: u64 = 1 << 1;
const CL_PROGRAM_BUILD_LOG: ClUint = 0x1183;
const CL_R: ClUint = 0x10B0;
const CL_RGBA: ClUint = 0x10B5;
const CL_UNORM_INT8: ClUint = 0x10D2;
const CL_UNORM_INT16: ClUint = 0x10D3;
const CL_HALF_FLOAT: ClUint = 0x10DD;
const CL_FLOAT: ClUint = 0x10DE;

#[allow(clippy::struct_excessive_bools)]
pub struct GpuContext {
    cl: OpenCl,
    context: ClContext,
    device: ClDeviceId,
    profile: GpuProfile,
    program: ClProgram,
    variants: &'static std::sync::Mutex<Vec<(u32, ClProgram)>>,
    device_name: String,
    depth: u32,
    linear: bool,

    units: Vec<std::sync::Mutex<Unit>>,
    next_unit: std::sync::atomic::AtomicUsize,
    cache_resources: Vec<std::sync::Mutex<CacheResources>>,
    next_upload: std::sync::atomic::AtomicUsize,

    super_cache: std::sync::Mutex<Vec<(i64, SuperCell)>>,
    super_held: std::sync::Mutex<Vec<(i64, std::sync::Weak<SuperEntry>)>>,
    super_evicted: std::sync::Mutex<std::collections::VecDeque<i64>>,
    super_cap: std::sync::atomic::AtomicUsize,
    frame_floor: std::sync::atomic::AtomicUsize,
    frame_bytes: std::sync::atomic::AtomicUsize,
    frame_budget: usize,
    fuse_fits: bool,
    fuse_ext_fits: bool,
    cover_layer: bool,
    image_pool: std::sync::Arc<ImagePool>,
    still: std::sync::Mutex<StillState>,
    hold: ClMem,
    layer: std::sync::Mutex<layer::LayerState>,
    blend_pool: std::sync::Mutex<Vec<blend::Buffers>>,
    blend_slots: std::sync::Arc<blend::Slots>,
    blending: std::sync::atomic::AtomicBool,
    readbacks: std::sync::Arc<blend::Slots>,
    blend_units: std::sync::Mutex<Vec<Unit>>,
    pins: std::sync::Mutex<Vec<Pin>>,
    dst_pool: std::sync::Mutex<Vec<[(ClMem, usize); 3]>>,
    pressure: Pressure,
    vram: vram::Vram,
    reserve: usize,
}

struct Pin {
    mem: ClMem,
    ptr: *mut u8,
    size: usize,
}

unsafe impl Send for Pin {}

type SuperCell = std::sync::Arc<std::sync::OnceLock<Option<std::sync::Arc<SuperEntry>>>>;

struct CacheResources {
    kernel: ClKernel,
    copy_kernel: ClKernel,
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
    pool_cap: usize,
    dims: (usize, usize),
    luma: (usize, usize),
}

struct PooledImages {
    dims: (usize, usize),
    luma: (usize, usize),
    linear: ClMem,
    packed: ClMem,
}

unsafe impl Send for PooledImages {}

type ImagePool = std::sync::Mutex<Vec<PooledImages>>;

const RECENT_FRAMES: usize = 2;
const IMAGE_POOL_CAP: usize = 3;
const FRAME_BUDGET_SHARE: usize = 4;
const EVICTED_KEYS: usize = 32;

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
            && pool.len() < self.pool_cap
        {
            pool.push(PooledImages {
                dims: self.dims,
                luma: self.luma,
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
    pub fn share(&self) -> Self {
        Self {
            entry: std::sync::Arc::clone(&self.entry),
        }
    }

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
    xfer: ClCommandQueue,
    motion: [ImageSlot; MOTION_SLOTS],
    motion_key: [i64; MOTION_SLOTS],
    mask: ImageSlot,
    packed_base: Vec<u16>,
    packed_ext: Vec<u16>,
    packed_mask: Vec<u8>,
    held: Vec<Box<dyn std::any::Any + Send>>,
    variant_kernels: Vec<(u32, ClKernel)>,
    blend: blend::UnitState,
    cover: cover::UnitCover,
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
    fn new(kernel: ClKernel, queue: ClCommandQueue, xfer: ClCommandQueue) -> Self {
        Self {
            kernel,
            queue,
            xfer,
            motion: [ImageSlot::EMPTY; MOTION_SLOTS],
            motion_key: [i64::MIN; MOTION_SLOTS],
            mask: ImageSlot::EMPTY,
            packed_base: Vec::new(),
            packed_ext: Vec::new(),
            packed_mask: Vec::new(),
            held: Vec::new(),
            variant_kernels: Vec::new(),
            blend: blend::UnitState::default(),
            cover: cover::UnitCover::default(),
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

#[derive(Clone, Copy)]
pub struct Mark {
    epoch: usize,
    events: usize,
}

struct Pressure {
    epoch: std::sync::atomic::AtomicUsize,
    events: std::sync::atomic::AtomicUsize,
    strained: std::sync::atomic::AtomicBool,
    cpu: std::sync::atomic::AtomicBool,
    fallback: std::sync::Mutex<Fallback>,
}

struct Fallback {
    since: std::time::Instant,
    wait: std::time::Duration,
    free: Option<usize>,
}

const RESERVE_SHARE: usize = 16;
const RETRY_FIRST: std::time::Duration = std::time::Duration::from_secs(1);
const RETRY_MOST: std::time::Duration = std::time::Duration::from_secs(32);

const UPLOAD_QUEUES: usize = 4;
const MOTION_SLOTS: usize = 8;

unsafe impl Send for GpuContext {}
unsafe impl Sync for GpuContext {}

impl Drop for GpuContext {
    fn drop(&mut self) {
        unsafe {
            if let Ok(mut cache) = self.super_cache.lock() {
                cache.clear();
            }
            if let (Ok(mut pins), Some(Ok(unit))) =
                (self.pins.lock(), self.units.first().map(|u| u.lock()))
            {
                for pin in pins.drain(..) {
                    self.release_pin(unit.queue, &pin);
                }
            }
            for unit in &self.units {
                if let Ok(u) = unit.lock() {
                    self.release_unit(&u);
                }
            }
            if let Ok(mut pool) = self.dst_pool.lock() {
                for set in pool.drain(..) {
                    for b in set {
                        if !b.0.is_null() {
                            (self.cl.ReleaseMemObject)(b.0);
                        }
                    }
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
                    if !cache.copy_kernel.is_null() {
                        (self.cl.ReleaseKernel)(cache.copy_kernel);
                    }
                    (self.cl.ReleaseCommandQueue)(cache.queue);
                }
            }
            if let Ok(mut cache) = self.super_cache.lock() {
                cache.clear();
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
            self.release_layer();
            if !self.hold.is_null() {
                (self.cl.ReleaseMemObject)(self.hold);
            }
            if let Ok(mut pool) = self.blend_pool.lock() {
                for buffers in pool.drain(..) {
                    (self.cl.ReleaseMemObject)(buffers.sums);
                    (self.cl.ReleaseMemObject)(buffers.bytes);
                }
            }
            if let Ok(mut pool) = self.image_pool.lock() {
                for images in pool.drain(..) {
                    if !images.linear.is_null() {
                        (self.cl.ReleaseMemObject)(images.linear);
                    }
                    (self.cl.ReleaseMemObject)(images.packed);
                }
            }
        }
    }
}

impl GpuContext {
    unsafe fn release_unit(&self, u: &Unit) {
        unsafe {
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
            self.release_cover(&u.cover);
            (self.cl.ReleaseCommandQueue)(u.queue);
            if !u.xfer.is_null() {
                (self.cl.ReleaseCommandQueue)(u.xfer);
            }
        }
    }

    pub fn new(gpuid: i32, qn: i64, depth: u32, linear: bool) -> Option<Self> {
        let units = usize::try_from(qn)
            .unwrap_or(DEFAULT_UNITS)
            .clamp(MIN_UNITS, MAX_UNITS);
        unsafe { Self::try_new(gpuid, units, depth, linear) }.ok()
    }

    unsafe fn try_new(
        gpuid: i32,
        num_units: usize,
        depth: u32,
        linear: bool,
    ) -> Result<Self, String> {
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
            let Some(&Shared {
                context,
                program,
                ref variants,
            }) = (unsafe { shared(&cl, device, depth) })
            else {
                continue;
            };
            let profile = unsafe { GpuProfile::probe(&cl.core, device) };
            let workers = svpflow_core::threads::workers();
            let target_units = profile.render_units(num_units, workers);
            let mut err = 0;
            let mut units = Vec::with_capacity(target_units);
            for _ in 0..target_units {
                let name = CString::new("render_frame").map_err(|_| "nul")?;
                let mut errs = [0; 3];
                let kernel = unsafe { (cl.CreateKernel)(program, name.as_ptr(), &raw mut errs[0]) };
                let queue =
                    unsafe { (cl.CreateCommandQueue)(context, device, 0, &raw mut errs[1]) };
                let xfer = unsafe { (cl.CreateCommandQueue)(context, device, 0, &raw mut errs[2]) };
                if errs == [CL_SUCCESS; 3]
                    && !kernel.is_null()
                    && !queue.is_null()
                    && !xfer.is_null()
                {
                    units.push(std::sync::Mutex::new(Unit::new(kernel, queue, xfer)));
                    continue;
                }
                unsafe {
                    if !kernel.is_null() {
                        (cl.ReleaseKernel)(kernel);
                    }
                    for queue in [queue, xfer].into_iter().filter(|queue| !queue.is_null()) {
                        (cl.ReleaseCommandQueue)(queue);
                    }
                }
            }
            if units.is_empty() {
                return Err("no render units".into());
            }
            let linear_name = CString::new("linear_luma").map_err(|_| "nul")?;
            let mut kerr = 0;
            let linear_kernel =
                unsafe { (cl.CreateKernel)(program, linear_name.as_ptr(), &raw mut kerr) };
            let mut cerr = 0;
            let copy_kernel =
                unsafe { (cl.CreateKernel)(program, c"copy_planes_to_packed".as_ptr(), &raw mut cerr) };
            let copy_kernel = if cerr == CL_SUCCESS {
                copy_kernel
            } else {
                std::ptr::null_mut()
            };
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
                    if !copy_kernel.is_null() {
                        (cl.ReleaseKernel)(copy_kernel);
                    }
                    if !cache_queue.is_null() {
                        (cl.ReleaseCommandQueue)(cache_queue);
                    }
                    for unit in &units {
                        if let Ok(unit) = unit.lock() {
                            (cl.ReleaseKernel)(unit.kernel);
                            (cl.ReleaseCommandQueue)(unit.queue);
                            (cl.ReleaseCommandQueue)(unit.xfer);
                        }
                    }
                }
                return Err("linear-luma render unit unavailable".into());
            }
            let mut cache_resources = vec![std::sync::Mutex::new(CacheResources {
                kernel: linear_kernel,
                copy_kernel,
                queue: cache_queue,
            })];
            let upload_queues = profile.upload_queues(workers).max(UPLOAD_QUEUES);
            for _ in 1..upload_queues {
                let mut kerr = 0;
                let kernel =
                    unsafe { (cl.CreateKernel)(program, linear_name.as_ptr(), &raw mut kerr) };
                let mut cerr = 0;
                let copy = unsafe {
                    (cl.CreateKernel)(program, c"copy_planes_to_packed".as_ptr(), &raw mut cerr)
                };
                let copy = if cerr == CL_SUCCESS {
                    copy
                } else {
                    std::ptr::null_mut()
                };
                let queue = unsafe { (cl.CreateCommandQueue)(context, device, 0, &raw mut err) };
                if kerr != CL_SUCCESS || kernel.is_null() || err != CL_SUCCESS || queue.is_null() {
                    unsafe {
                        if !kernel.is_null() {
                            (cl.ReleaseKernel)(kernel);
                        }
                        if !copy.is_null() {
                            (cl.ReleaseKernel)(copy);
                        }
                        if !queue.is_null() {
                            (cl.ReleaseCommandQueue)(queue);
                        }
                    }
                    break;
                }
                cache_resources.push(std::sync::Mutex::new(CacheResources {
                    kernel,
                    copy_kernel: copy,
                    queue,
                }));
            }
            let device_name = unsafe { device_name(&cl, device) };
            let total = unsafe { device_size(&cl, device, CL_DEVICE_GLOBAL_MEM_SIZE) };
            let frame_budget = total.unwrap_or(usize::MAX) / FRAME_BUDGET_SHARE;
            let vram = unsafe { vram::Vram::open(&cl, device) };
            let max_param = unsafe { device_size(&cl, device, CL_DEVICE_MAX_PARAMETER_SIZE) };
            let fuse_fits = max_param.is_some_and(|size| size >= blend::FUSED_ARG_BYTES);
            let fuse_ext_fits = max_param.is_some_and(|size| size >= blend::FUSED_EXT_ARG_BYTES);
            let cover_layer = cl.CreateImage.is_some() && {
                let mut cerr = 0;
                let kernel = unsafe {
                    (cl.CreateKernel)(program, c"cover_finish_layer".as_ptr(), &raw mut cerr)
                };
                if cerr == CL_SUCCESS && !kernel.is_null() {
                    unsafe { (cl.ReleaseKernel)(kernel) };
                    true
                } else {
                    false
                }
            };
            let blend_slots = std::sync::Arc::new(blend::Slots::new(units.len()));
            let readbacks = std::sync::Arc::new(blend::Slots::new(2 * units.len()));
            return Ok(Self {
                cl,
                context,
                device,
                profile,
                program,
                variants,
                depth,
                linear,
                device_name,
                units,
                next_unit: std::sync::atomic::AtomicUsize::new(0),
                cache_resources,
                next_upload: std::sync::atomic::AtomicUsize::new(0),
                super_cache: std::sync::Mutex::new(Vec::new()),
                super_held: std::sync::Mutex::new(Vec::new()),
                super_evicted: std::sync::Mutex::new(std::collections::VecDeque::new()),
                super_cap: std::sync::atomic::AtomicUsize::new(RECENT_FRAMES),
                frame_floor: std::sync::atomic::AtomicUsize::new(0),
                frame_bytes: std::sync::atomic::AtomicUsize::new(0),
                frame_budget,
                fuse_fits,
                fuse_ext_fits,
                cover_layer,
                image_pool: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
                still: std::sync::Mutex::new(StillState {
                    queue: std::ptr::null_mut(),
                    kernels: [std::ptr::null_mut(); 2],
                    scratch: std::ptr::null_mut(),
                    dims: (0, 0),
                }),
                hold: std::ptr::null_mut(),
                layer: std::sync::Mutex::new(layer::LayerState::new()),
                blend_pool: std::sync::Mutex::new(Vec::new()),
                readbacks,
                blend_slots,
                blending: std::sync::atomic::AtomicBool::new(false),
                blend_units: std::sync::Mutex::new(Vec::new()),
                pins: std::sync::Mutex::new(Vec::new()),
                dst_pool: std::sync::Mutex::new(Vec::new()),
                pressure: Pressure {
                    epoch: std::sync::atomic::AtomicUsize::new(0),
                    events: std::sync::atomic::AtomicUsize::new(0),
                    strained: std::sync::atomic::AtomicBool::new(false),
                    cpu: std::sync::atomic::AtomicBool::new(false),
                    fallback: std::sync::Mutex::new(Fallback {
                        since: std::time::Instant::now(),
                        wait: std::time::Duration::ZERO,
                        free: None,
                    }),
                },
                vram,
                reserve: total.unwrap_or(0) / RESERVE_SHARE,
            });
        }
        Err("no usable GPU device".into())
    }

    pub fn is_linear(&self) -> bool {
        self.linear
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    fn acquire(&self) -> std::sync::MutexGuard<'_, Unit> {
        let n = self.blend_slots.total().min(self.units.len());
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
    pub fn linear(&self) -> bool {
        self.linear
    }

    pub fn keep_frames(&self, count: usize) {
        self.frame_floor
            .fetch_max(count, std::sync::atomic::Ordering::Relaxed);
    }

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
                let mut held = self.super_held.lock().ok()?;
                let cap = self.super_cap.load(std::sync::atomic::Ordering::Relaxed);
                if cache.len() >= cap {
                    let (old, gone) = cache.remove(0);
                    held.retain(|(k, entry)| *k != old && entry.strong_count() > 0);
                    if let Some(Some(entry)) = gone.get() {
                        held.push((old, std::sync::Arc::downgrade(entry)));
                    }
                    if let Ok(mut evicted) = self.super_evicted.lock() {
                        evicted.push_back(old);
                        if evicted.len() > EVICTED_KEYS {
                            evicted.pop_front();
                        }
                    }
                }
                let found = held.iter().find(|(k, _)| *k == key);
                if let Some(entry) = found.and_then(|(_, entry)| entry.upgrade()) {
                    let _ = cell.set(Some(entry));
                } else if let Ok(mut evicted) = self.super_evicted.lock()
                    && let Some(at) = evicted.iter().position(|k| *k == key)
                {
                    evicted.remove(at);
                    let bytes = self.frame_bytes.load(std::sync::atomic::Ordering::Relaxed);
                    let floor = self.frame_floor.load(std::sync::atomic::Ordering::Relaxed);
                    let limit = if self.blending.load(std::sync::atomic::Ordering::Relaxed) {
                        self.blend_slots.total()
                    } else {
                        2 * self.units.len()
                    };
                    let most = (svpflow_core::threads::workers().min(limit).max(floor)
                        + RECENT_FRAMES)
                        .min((self.frame_budget / bytes.max(1)).max(RECENT_FRAMES));
                    let _ = self.super_cap.fetch_update(
                        std::sync::atomic::Ordering::Relaxed,
                        std::sync::atomic::Ordering::Relaxed,
                        |cap| {
                            (cap < most && !self.strained() && self.room(bytes)).then_some(cap + 1)
                        },
                    );
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
        let dims = (
            width.max(planes[1].width + planes[2].width),
            height + planes[1].height,
        );
        let depth = if self.depth > 8 { 2 } else { 1 };
        let bytes = dims.0 * dims.1 * depth + if self.linear { width * height * 4 } else { 0 };
        self.frame_bytes
            .store(bytes, std::sync::atomic::Ordering::Relaxed);
        let reused = self.image_pool.lock().ok().and_then(|mut pool| {
            let index = pool
                .iter()
                .position(|images| (images.dims, images.luma) == (dims, (width, height)))?;
            Some(pool.swap_remove(index))
        });
        let (linear, packed) = if let Some(images) = reused {
            (images.linear, images.packed)
        } else {
            let depth = if self.depth > 8 {
                CL_UNORM_INT16
            } else {
                CL_UNORM_INT8
            };
            let packed = unsafe { self.create_image(CL_R, depth, dims.0, dims.1) }?;
            let linear = if self.linear {
                let linear = unsafe { self.create_image(CL_R, CL_FLOAT, width, height) }?;
                let mem = linear.mem;
                std::mem::forget(linear);
                mem
            } else {
                std::ptr::null_mut()
            };
            let mems = (linear, packed.mem);
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
            pool_cap: self
                .super_cap
                .load(std::sync::atomic::Ordering::Relaxed)
                .min(IMAGE_POOL_CAP),
            dims,
            luma: (width, height),
        };
        let resources = self.upload_resources()?;
        let mut device_copied = false;
        if self.depth == 8
            && !resources.copy_kernel.is_null()
            && unsafe { has_shared_ptr(self.context, planes[0].data.as_ptr() as usize) }
        {
            let hash = plane_fingerprint([
                (planes[0].data, planes[0].stride, planes[0].width, planes[0].height),
                (planes[1].data, planes[1].stride, planes[1].width, planes[1].height),
                (planes[2].data, planes[2].stride, planes[2].width, planes[2].height),
            ]);
            if let Some(shared) = unsafe { lookup_source(self.context, hash) } {
                let mut copy_done: ClEvent = std::ptr::null_mut();
                let dims_match = shared.width == width
                    && shared.height == height
                    && shared.chroma_width == planes[1].width
                    && shared.chroma_height == planes[1].height
                    && shared.chroma_width == planes[2].width
                    && shared.chroma_height == planes[2].height
                    && shared.pitches[1] == shared.pitches[2];
                if dims_match {
                    let ck = resources.copy_kernel;
                    let w_i = i32::try_from(width).ok();
                    let h_i = i32::try_from(height).ok();
                    let cw_i = i32::try_from(planes[1].width).ok();
                    let ch_i = i32::try_from(planes[1].height).ok();
                    let yp_i = i32::try_from(shared.pitches[0]).ok();
                    let uvp_i = i32::try_from(shared.pitches[1]).ok();
                    if let (Some(w_i), Some(h_i), Some(cw_i), Some(ch_i), Some(yp_i), Some(uvp_i)) =
                        (w_i, h_i, cw_i, ch_i, yp_i, uvp_i)
                    {
                        let waits = u32::from(!shared.event.is_null());
                        let wait_ptr = if waits == 0 {
                            std::ptr::null()
                        } else {
                            (&raw const shared.event).cast()
                        };
                        let global = [width.max(planes[1].width), height.max(planes[1].height)];
                        let ok = unsafe {
                            (self.cl.SetKernelArg)(ck, 0, size_of::<ClMem>(), (&raw const shared.mems[0]).cast()) == CL_SUCCESS
                                && self.set_i32(ck, 1, yp_i)
                                && (self.cl.SetKernelArg)(ck, 2, size_of::<ClMem>(), (&raw const shared.mems[1]).cast()) == CL_SUCCESS
                                && (self.cl.SetKernelArg)(ck, 3, size_of::<ClMem>(), (&raw const shared.mems[2]).cast()) == CL_SUCCESS
                                && self.set_i32(ck, 4, uvp_i)
                                && (self.cl.SetKernelArg)(ck, 5, size_of::<ClMem>(), (&raw const packed).cast()) == CL_SUCCESS
                                && self.set_i32(ck, 6, w_i)
                                && self.set_i32(ck, 7, h_i)
                                && self.set_i32(ck, 8, cw_i)
                                && self.set_i32(ck, 9, ch_i)
                                && self.note((self.cl.EnqueueNDRangeKernel)(
                                    resources.queue,
                                    ck,
                                    2,
                                    std::ptr::null(),
                                    global.as_ptr(),
                                    std::ptr::null(),
                                    waits,
                                    wait_ptr,
                                    (&raw mut copy_done).cast(),
                                )) == CL_SUCCESS
                        };
                        if ok {
                            device_copied = true;
                            unsafe { (self.cl.Flush)(resources.queue) };
                        } else if !copy_done.is_null() {
                            unsafe { (self.cl.ReleaseEvent)(copy_done) };
                            copy_done = std::ptr::null_mut();
                        }
                    }
                }
                unsafe { finish_source(self.context, shared.mems[0], copy_done) };
                if device_copied {
                    if self.linear {
                        if !copy_done.is_null() {
                            unsafe { (self.cl.ReleaseEvent)(copy_done) };
                        }
                    } else {
                        entry.ready = copy_done;
                    }
                }
                unsafe {
                    for mem in shared.mems {
                        if !mem.is_null() {
                            (self.cl.ReleaseMemObject)(mem);
                        }
                    }
                    if !shared.event.is_null() {
                        (self.cl.ReleaseEvent)(shared.event);
                    }
                }
            }
        }
        let mut write_done: ClEvent = std::ptr::null_mut();
        if !device_copied {
            unsafe {
                self.enqueue_plane_at(resources.queue, packed, planes[0], 0, 0, std::ptr::null_mut())?;
                self.enqueue_plane_at(resources.queue, packed, planes[1], 0, height, std::ptr::null_mut())?;
                self.enqueue_plane_at(
                    resources.queue,
                    packed,
                    planes[2],
                    planes[1].width,
                    height,
                    &raw mut write_done,
                )?;
            }
        }
        unsafe {
            if self.linear {
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
                    if !write_done.is_null() {
                        self.wait_event(resources.queue, write_done);
                    }
                    return None;
                }
                let global = [width, height];
                if self.note((self.cl.EnqueueNDRangeKernel)(
                    resources.queue,
                    resources.kernel,
                    2,
                    std::ptr::null(),
                    global.as_ptr(),
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    (&raw mut entry.ready).cast(),
                )) != CL_SUCCESS
                    || (self.cl.Flush)(resources.queue) != CL_SUCCESS
                {
                    if !write_done.is_null() {
                        self.wait_event(resources.queue, write_done);
                    }
                    return None;
                }
            }
            if !write_done.is_null() && !self.wait_event(resources.queue, write_done) {
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
        event: *mut ClEvent,
    ) -> Option<()> {
        let bytes = if self.depth > 8 { 2 } else { 1 };
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
            self.note((self.cl.EnqueueWriteImage)(
                q,
                mem,
                CL_FALSE,
                origin.as_ptr(),
                region.as_ptr(),
                plane.stride,
                0,
                plane.data.as_ptr().cast(),
                0,
                std::ptr::null(),
                event.cast(),
            ))
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
        let make = || {
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
        };
        make().or_else(|| {
            self.relieve();
            make().or_else(|| self.exhausted())
        })
    }

    unsafe fn create_buffer(&self, bytes: usize) -> Option<ClMem> {
        let make = || {
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
            (err == CL_SUCCESS && !mem.is_null()).then_some(mem)
        };
        make().or_else(|| {
            self.relieve();
            make().or_else(|| self.exhausted())
        })
    }

    pub fn mark(&self) -> Mark {
        Mark {
            epoch: self
                .pressure
                .epoch
                .load(std::sync::atomic::Ordering::Acquire),
            events: self
                .pressure
                .events
                .load(std::sync::atomic::Ordering::Acquire),
        }
    }

    pub fn cpu_render(&self) -> bool {
        self.pressure.cpu.load(std::sync::atomic::Ordering::Acquire)
    }

    pub fn retry(&self) {
        if !self.cpu_render() {
            return;
        }
        let Ok(fallback) = self.pressure.fallback.try_lock() else {
            return;
        };
        let waited = fallback.since.elapsed();
        if waited < fallback.wait {
            return;
        }
        let freed = match (fallback.free, self.free()) {
            (Some(then), Some(now)) => now >= then.saturating_add(self.reserve),
            _ => true,
        };
        if (freed || waited >= RETRY_MOST) && self.cpu_render() {
            self.blend_slots.restore(self.units.len());
            self.pressure
                .strained
                .store(false, std::sync::atomic::Ordering::Release);
            self.pressure
                .cpu
                .store(false, std::sync::atomic::Ordering::Release);
            eprintln!("open-svpflow: GPU memory is free again, using the GPU render");
        }
    }

    pub fn strained(&self) -> bool {
        self.pressure
            .strained
            .load(std::sync::atomic::Ordering::Acquire)
    }

    fn free(&self) -> Option<usize> {
        unsafe { self.vram.free(&self.cl, self.device) }
    }

    fn room(&self, bytes: usize) -> bool {
        self.free()
            .is_none_or(|free| free >= bytes.saturating_add(self.reserve))
    }

    pub fn exhausted_since(&self, mark: Mark) -> bool {
        self.pressure
            .events
            .load(std::sync::atomic::Ordering::Acquire)
            != mark.events
    }

    fn exhausted<T>(&self) -> Option<T> {
        self.pressure
            .events
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        None
    }

    fn note(&self, code: ClInt) -> ClInt {
        if matches!(
            code,
            CL_MEM_OBJECT_ALLOCATION_FAILURE | CL_OUT_OF_RESOURCES | CL_OUT_OF_HOST_MEMORY
        ) {
            self.exhausted::<()>();
        }
        code
    }

    pub fn recover(&self, mark: Mark) -> bool {
        if !self.exhausted_since(mark) {
            return false;
        }
        let mut fallback = self
            .pressure
            .fallback
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if self.mark().epoch != mark.epoch {
            return true;
        }
        self.relieve();
        self.super_cap
            .store(RECENT_FRAMES, std::sync::atomic::Ordering::Relaxed);
        if !self.blend_slots.shrink() {
            if self.pressure.cpu.load(std::sync::atomic::Ordering::Acquire) {
                return false;
            }
            fallback.wait = (fallback.wait * 2).clamp(RETRY_FIRST, RETRY_MOST);
            fallback.since = std::time::Instant::now();
            fallback.free = self.free();
            self.pressure
                .cpu
                .store(true, std::sync::atomic::Ordering::Release);
            eprintln!("open-svpflow: not enough GPU memory, using the CPU render");
        }
        self.pressure
            .strained
            .store(true, std::sync::atomic::Ordering::Release);
        self.pressure
            .epoch
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        true
    }

    fn relieve(&self) {
        if let Ok(mut state) = self.layer.try_lock() {
            state.release_cached();
        }
        let frames = self
            .super_cache
            .lock()
            .map(|mut cache| std::mem::take(&mut *cache));
        drop(frames);
        if let Ok(mut held) = self.super_held.lock() {
            held.clear();
        }
        let images = self
            .image_pool
            .lock()
            .map(|mut pool| std::mem::take(&mut *pool));
        for images in images.into_iter().flatten() {
            unsafe {
                if !images.linear.is_null() {
                    (self.cl.ReleaseMemObject)(images.linear);
                }
                (self.cl.ReleaseMemObject)(images.packed);
            }
        }
        let buffers = self
            .blend_pool
            .lock()
            .map(|mut pool| std::mem::take(&mut *pool));
        for buffers in buffers.into_iter().flatten() {
            unsafe { self.release_buffers(&buffers) };
        }
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
            self.note((self.cl.EnqueueWriteImage)(
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
            ))
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
                *slot = ImageSlot::EMPTY;
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
            let Some(mem) = (unsafe { self.create_buffer(bytes) }) else {
                slot.1 = 0;
                slot.0 = std::ptr::null_mut();
                return None;
            };
            *slot = (mem, bytes);
        }
        Some(slot.0)
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
                let extra = if variant >> 19 != 0 {
                    format!(
                        "#define FAST_OCC {}\n#define FAST_GPU {}\n#define FAST_BLEND {}\n",
                        (variant >> 19) & 1,
                        (variant >> 20) & 1,
                        (variant >> 21) & 3,
                    )
                } else {
                    String::new()
                };
                let source = format!(
                    "#define ALGO {}\n#define CUBIC {}\n#define CUBIC_REF {}\n#define HAS_SAD {}\n#define LINEAR_LUMA {}\n#define DITHER {}\n#define BLEND {}\n#define STILL {}\n#define LAYER {}\n#define HOLD {}\n#define FUSE {}\n#define ADAPTIVE {}\n{extra}#define DEPTH {}\n{}",
                    variant & 0xFF,
                    (variant >> 8) & 1,
                    (variant >> 9) & 1,
                    (variant >> 10) & 1,
                    (variant >> 11) & 1,
                    (variant >> 12) & 1,
                    (variant >> 13) & 1,
                    (variant >> 14) & 1,
                    (variant >> 15) & 1,
                    (variant >> 16) & 1,
                    (variant >> 17) & 1,
                    (variant >> 18) & 1,
                    self.depth,
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
        let done = unsafe { self.cl.wait_event(queue, event) };
        if !done {
            unsafe { self.note((self.cl.Finish)(queue)) };
        }
        unsafe { (self.cl.ReleaseEvent)(event) };
        done
    }

    unsafe fn set_i32(&self, k: ClKernel, i: u32, v: i32) -> bool {
        unsafe { (self.cl.SetKernelArg)(k, i, 4, (&raw const v).cast()) == CL_SUCCESS }
    }

    fn variant(p: &KernelParams, high: bool) -> u32 {
        u32::try_from(p.algorithm & 0xFF).unwrap_or(23)
            | u32::from(p.cubic != 0) << 8
            | u32::from(p.cubic_ref != 0) << 9
            | u32::from(p.has_sad != 0) << 10
            | u32::from(p.linear_luma != 0) << 11
            | u32::from(p.dither != 0 && !high) << 12
    }

    unsafe fn upload_motion(
        &self,
        unit: &mut Unit,
        slot: usize,
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
            let base_mem = self.ensure_image(
                &mut motion[slot],
                motion_width,
                motion_height,
                CL_UNORM_INT16,
            )?;
            if cached_motion_key[slot] != motion_key {
                interleave4_u16(
                    packed_base,
                    count,
                    motions[0].0,
                    motions[1].1,
                    motions[1].0,
                    motions[0].1,
                );
                self.enqueue_image_raw(
                    q,
                    base_mem,
                    motion_width,
                    motion_height,
                    motion_width * 8,
                    packed_base.as_ptr().cast(),
                )?;
                cached_motion_key[slot] = motion_key;
            }
            let ext_mem = if algorithm == 23 {
                let mem = self.ensure_image(
                    &mut motion[slot + 1],
                    motion_width,
                    motion_height,
                    CL_UNORM_INT16,
                )?;
                if cached_motion_key[slot + 1] != motion_key {
                    interleave4_u16(
                        packed_ext,
                        count,
                        motions[2].0,
                        motions[3].1,
                        motions[3].0,
                        motions[2].1,
                    );
                    self.enqueue_image_raw(
                        q,
                        mem,
                        motion_width,
                        motion_height,
                        motion_width * 8,
                        packed_ext.as_ptr().cast(),
                    )?;
                    cached_motion_key[slot + 1] = motion_key;
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
        let start = packed.len();
        packed.resize(start + count * 4, 0);
        let dst = &mut packed[start..start + count * 4];
        match (needs_coverage, area) {
            (true, Some((af, ab))) => {
                interleave4_u8(dst, count, ab, coverage.0, coverage.1, af);
            }
            (true, None) => {
                interleave4_u8_zero_outer(dst, count, coverage.0, coverage.1);
            }
            (false, Some((af, ab))) => {
                for i in 0..count {
                    let o = i * 4;
                    dst[o] = ab[i];
                    dst[o + 1] = 0;
                    dst[o + 2] = 0;
                    dst[o + 3] = af[i];
                }
            }
            (false, None) => {}
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
        cover: Option<&Cover<'_>>,
    ) -> Option<ClKernel> {
        let count = motion_width.checked_mul(motion_height)?;
        let k = self.variant_kernel(&mut unit.variant_kernels, variant)?;
        let q = unit.queue;
        unsafe {
            let (base_mem, ext_mem) = self.upload_motion(
                unit,
                0,
                py.algorithm,
                motion_key,
                motion_width,
                motion_height,
                motions,
            )?;
            let mask_mem = if let Some(cover) = cover {
                self.bind_cover(unit, cover)?
            } else if py.algorithm >= 21 || py.has_sad != 0 {
                let Unit {
                    mask, packed_mask, ..
                } = unit;
                packed_mask.clear();
                let needs_coverage = py.algorithm >= 21;
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
        event: *mut ClEvent,
        quad: bool,
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
            let width = usize::try_from(p.width).ok()?;
            let height = usize::try_from(p.height).ok()?;
            let global = if quad {
                [width.div_ceil(4), height.div_ceil(2)]
            } else {
                [width, height]
            };
            if self.note((self.cl.EnqueueNDRangeKernel)(
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
                event.cast(),
            )) != CL_SUCCESS
            {
                return None;
            }
        }
        Some(())
    }

    pub fn set_hold(&mut self, pixels: &[u8], width: usize, height: usize) -> Option<()> {
        let format = ClImageFormat {
            channel_order: CL_R,
            channel_data_type: CL_UNORM_INT8,
        };
        let mut err = 0;
        let mem = unsafe {
            (self.cl.CreateImage2D)(
                self.context,
                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                &raw const format,
                width,
                height,
                width,
                pixels.as_ptr().cast_mut().cast(),
                &raw mut err,
            )
        };
        (err == CL_SUCCESS && !mem.is_null()).then(|| self.hold = mem)
    }

    pub fn hold(&self) -> Option<StillMask> {
        (!self.hold.is_null()).then_some(StillMask(self.hold, std::ptr::null_mut()))
    }

    fn mask_variant(&self, mask: Option<StillMask>) -> u32 {
        mask.map_or(
            0,
            |StillMask(mem, _)| if mem == self.hold { 1 << 16 } else { 1 << 14 },
        )
    }

    pub fn still_mask(
        &self,
        first: &SuperHandle,
        second: &SuperHandle,
        limits: [f32; 3],
    ) -> Option<StillMask> {
        if self.depth > 8 || first.entry.dims != second.entry.dims {
            return None;
        }
        let dims = first.entry.luma;
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
                    self.note((self.cl.EnqueueNDRangeKernel)(
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
                    )) == CL_SUCCESS
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

    unsafe fn take_pin(&self, q: ClCommandQueue, size: usize) -> Option<Pin> {
        if !self.profile.use_pinned_staging() {
            return None;
        }
        let pooled = self.pins.lock().ok()?.pop();
        if let Some(pin) = pooled {
            if pin.size >= size {
                return Some(pin);
            }
            unsafe { self.release_pin(q, &pin) };
        }
        let mut err = 0;
        unsafe {
            let mem = (self.cl.CreateBuffer)(
                self.context,
                CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                size,
                std::ptr::null_mut(),
                &raw mut err,
            );
            if err != CL_SUCCESS || mem.is_null() {
                return None;
            }
            let ptr = (self.cl.EnqueueMapBuffer)(
                q,
                mem,
                CL_TRUE,
                CL_MAP_READ | CL_MAP_WRITE,
                0,
                size,
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
                &raw mut err,
            );
            if err != CL_SUCCESS || ptr.is_null() {
                (self.cl.ReleaseMemObject)(mem);
                return None;
            }
            Some(Pin {
                mem,
                ptr: ptr.cast(),
                size,
            })
        }
    }

    unsafe fn release_pin(&self, q: ClCommandQueue, pin: &Pin) {
        unsafe {
            (self.cl.EnqueueUnmapMemObject)(
                q,
                pin.mem,
                pin.ptr.cast(),
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            );
            (self.cl.ReleaseMemObject)(pin.mem);
        }
    }

    fn put_pin(&self, pin: Pin) {
        if let Ok(mut pins) = self.pins.lock() {
            pins.push(pin);
        } else {
            unsafe { (self.cl.ReleaseMemObject)(pin.mem) };
        }
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
        layer: Option<&Layer>,
        cover: Option<&Cover<'_>>,
        fast: (u8, bool),
    ) -> Option<()> {
        let layer = layer.filter(|l| !l.is_empty());
        let _readback = self.readbacks.take()?;
        let mut unit = self.acquire();
        let use_quad = fast.1 && self.depth == 8 && still.is_none() && layer.is_none();
        let fast_bits = if fast.0 >= 1 { 1 << 19 } else { 0 }
            | if use_quad { 1 << 20 } else { 0 };
        unsafe {
            let k = self.bind_fields(
                &mut unit,
                Self::variant(&py, self.depth > 8)
                    | self.mask_variant(still)
                    | if layer.is_some() {
                        layer::LAYER_VARIANT
                    } else {
                        0
                    }
                    | fast_bits,
                &py,
                motion_key,
                motion_width,
                motion_height,
                motions,
                coverage,
                area,
                cover,
            )?;
            let mut waits = Self::source_waits(src0[0], src1[0]);
            if let Some(StillMask(still, ready)) = still {
                if (self.cl.SetKernelArg)(k, 8, size_of::<ClMem>(), (&raw const still).cast())
                    != CL_SUCCESS
                {
                    return None;
                }
                if !ready.is_null() {
                    waits.push(ready);
                }
            }
            if let Some(layer) = layer {
                let base = 8 + ClUint::from(still.is_some());
                let [fit, cover] = layer.images();
                for (index, mem) in (base..).zip([fit, cover, src0[1].0, src1[1].0]) {
                    if (self.cl.SetKernelArg)(k, index, size_of::<ClMem>(), (&raw const mem).cast())
                        != CL_SUCCESS
                    {
                        return None;
                    }
                }
            }
            let q = unit.queue;
            let xq = unit.xfer;
            let depth = if self.depth > 8 { 2 } else { 1 };
            let mut dst = self
                .dst_pool
                .lock()
                .ok()?
                .pop()
                .unwrap_or([(std::ptr::null_mut(), 0); 3]);
            let mems = [
                self.ensure_buffer(&mut dst[0], dst_y.len()),
                self.ensure_buffer(&mut dst[1], dst_u.len()),
                self.ensure_buffer(&mut dst[2], dst_v.len()),
            ];
            let [Some(y_mem), Some(u_mem), Some(v_mem)] = mems else {
                self.put_dst(dst);
                return None;
            };
            let mut kernel_done: ClEvent = std::ptr::null_mut();
            let queued = self
                .enqueue_plane(
                    q,
                    k,
                    y_mem,
                    sy / depth,
                    py,
                    src0[0],
                    src1[0],
                    &waits,
                    std::ptr::null_mut(),
                    use_quad,
                )
                .and_then(|()| {
                    self.enqueue_plane(
                        q,
                        k,
                        u_mem,
                        su / depth,
                        pu,
                        src0[1],
                        src1[1],
                        &[],
                        std::ptr::null_mut(),
                        use_quad,
                    )
                })
                .and_then(|()| {
                    self.enqueue_plane(
                        q,
                        k,
                        v_mem,
                        sv / depth,
                        pv,
                        src0[2],
                        src1[2],
                        &[],
                        &raw mut kernel_done,
                        use_quad,
                    )
                });
            if queued.is_none() {
                self.note((self.cl.Finish)(q));
                self.put_dst(dst);
                return None;
            }
            (self.cl.Flush)(q);
            let mut done: ClEvent = std::ptr::null_mut();
            let total = dst_y.len() + dst_u.len() + dst_v.len();
            let pin = self.take_pin(xq, total);
            let mut offset = 0;
            for (index, (dmem, output)) in [
                (y_mem, &mut *dst_y),
                (u_mem, &mut *dst_u),
                (v_mem, &mut *dst_v),
            ]
            .into_iter()
            .enumerate()
            {
                let target = match &pin {
                    Some(pin) => pin.ptr.add(offset),
                    None => output.as_mut_ptr(),
                };
                offset += output.len();
                let (count, list) = if index == 0 {
                    (1, (&raw const kernel_done).cast())
                } else {
                    (0, std::ptr::null())
                };
                if self.note((self.cl.EnqueueReadBuffer)(
                    xq,
                    dmem,
                    CL_FALSE,
                    0,
                    output.len(),
                    target.cast(),
                    count,
                    list,
                    if index == 2 {
                        (&raw mut done).cast()
                    } else {
                        std::ptr::null_mut()
                    },
                )) != CL_SUCCESS
                {
                    if !done.is_null() {
                        (self.cl.ReleaseEvent)(done);
                    }
                    self.note((self.cl.Finish)(xq));
                    self.note((self.cl.Finish)(q));
                    (self.cl.ReleaseEvent)(kernel_done);
                    self.put_dst(dst);
                    if let Some(pin) = pin {
                        self.put_pin(pin);
                    }
                    return None;
                }
            }
            (self.cl.ReleaseEvent)(kernel_done);
            let held = (
                std::mem::take(&mut unit.held),
                std::mem::take(&mut unit.packed_base),
                std::mem::take(&mut unit.packed_ext),
                std::mem::take(&mut unit.packed_mask),
            );
            drop(unit);
            let finished = self.wait_event(xq, done);
            drop(held);
            self.put_dst(dst);
            if let Some(pin) = pin {
                if finished {
                    let mut at = pin.ptr.cast_const();
                    for output in [dst_y, dst_u, dst_v] {
                        copy_stream(at, output.as_mut_ptr(), output.len());
                        at = at.add(output.len());
                    }
                }
                self.put_pin(pin);
            }
            finished.then_some(())?;
        }
        Some(())
    }

    fn put_dst(&self, dst: [(ClMem, usize); 3]) {
        if let Ok(mut pool) = self.dst_pool.lock() {
            pool.push(dst);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::cast_ptr_alignment)]
unsafe fn copy_avx2(src: *const u8, dst: *mut u8, len: usize) {
    use std::arch::x86_64::{__m256i, _mm_sfence, _mm256_loadu_si256, _mm256_stream_si256};
    let head = dst.align_offset(32).min(len);
    unsafe {
        std::ptr::copy_nonoverlapping(src, dst, head);
        let mut at = head;
        while at + 32 <= len {
            _mm256_stream_si256(
                dst.add(at).cast::<__m256i>(),
                _mm256_loadu_si256(src.add(at).cast()),
            );
            at += 32;
        }
        _mm_sfence();
        std::ptr::copy_nonoverlapping(src.add(at), dst.add(at), len - at);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(clippy::cast_ptr_alignment)]
unsafe fn copy_avx512(src: *const u8, dst: *mut u8, len: usize) {
    use std::arch::x86_64::{__m512i, _mm_sfence, _mm512_loadu_si512, _mm512_stream_si512};
    let head = dst.align_offset(64).min(len);
    unsafe {
        std::ptr::copy_nonoverlapping(src, dst, head);
        let mut at = head;
        while at + 64 <= len {
            _mm512_stream_si512(
                dst.add(at).cast::<__m512i>(),
                _mm512_loadu_si512(src.add(at).cast()),
            );
            at += 64;
        }
        _mm_sfence();
        std::ptr::copy_nonoverlapping(src.add(at), dst.add(at), len - at);
    }
}

unsafe fn copy_stream(src: *const u8, dst: *mut u8, len: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            return unsafe { copy_avx512(src, dst, len) };
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe { copy_avx2(src, dst, len) };
        }
    }
    unsafe { std::ptr::copy_nonoverlapping(src, dst, len) };
}

struct Shared {
    context: ClContext,
    program: ClProgram,
    variants: std::sync::Mutex<Vec<(u32, ClProgram)>>,
}

unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

// no program builds while a clip frees its context
unsafe fn shared(cl: &OpenCl, device: ClDeviceId, depth: u32) -> Option<&'static Shared> {
    static ALL: std::sync::Mutex<Vec<(usize, u32, &'static Shared)>> =
        std::sync::Mutex::new(Vec::new());
    let mut all = ALL.lock().ok()?;
    let key = (device as usize, depth);
    if let Some(&(.., found)) = all.iter().find(|entry| (entry.0, entry.1) == key) {
        return Some(found);
    }
    let context = svpflow_host::context::shared(device, || {
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
        if err == CL_SUCCESS {
            context
        } else {
            std::ptr::null_mut()
        }
    });
    if context.is_null() {
        return None;
    }
    let base = format!("#define DEPTH {depth}\n{KERNEL_SRC}");
    let Ok(program) = (unsafe { build_program(cl, context, device, &base) }) else {
        return None;
    };
    let made = Box::leak(Box::new(Shared {
        context,
        program,
        variants: std::sync::Mutex::new(Vec::new()),
    }));
    all.push((key.0, key.1, made));
    Some(made)
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
    let _build = svpflow_core::OPENCL_BUILD.lock();
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
#ifndef DEPTH
#define DEPTH 8
#endif
#ifndef BLEND
#define BLEND 0
#endif
#ifndef STILL
#define STILL 0
#endif
#ifndef LAYER
#define LAYER 0
#endif
#ifndef HOLD
#define HOLD 0
#endif
#ifndef FUSE
#define FUSE 0
#endif
#ifndef ADAPTIVE
#define ADAPTIVE 0
#endif
#ifndef FAST_OCC
#define FAST_OCC 0
#endif
#ifndef FAST_GPU
#define FAST_GPU 0
#endif
#ifndef FAST_BLEND
#define FAST_BLEND 0
#endif
#define LAYER_MIN 0.2f
#define LAYER_GROUP 3
#define LAYER_FILL 4.0f
#define LAYER_NEAR 0.0f
#define LAYER_SAME 2.5f
#define LAYER_APART 24.0f
#define MASK_AT(mask, p, x, y) read_imagef(mask, linear_sampler, \
    ((float2)(x, y) + 0.5f) * (float2)(p.x_ratio, p.y_ratio)).x
#define STILL_AT(still, p, x, y) (MASK_AT(still, p, x, y) > 0.4f)
#if DEPTH > 8
#define REF_SCALE(p) ((p)->linear_luma ? 256.0f : (float)(1 << (24 - DEPTH)))
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
#if CUBIC_REF && FAST_OCC
    if (p->cubic_ref) return REF_SCALE(p) * cubic_sample(source, position).x;
    return REF_SCALE(p) * read_imagef(source, linear_sampler, position).x;
#elif CUBIC_REF
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

#if FAST_GPU && DEPTH == 8 && !LAYER && !STILL && !HOLD
inline float2 moved_step(const Params *p, float vx, float vy, int time) {
    return native_divide(
        (float2)(vx*65535.0f-1024.0f, vy*65535.0f-1024.0f) * time,
        (float2)((float)(p->x_ratio*p->pel), (float)(p->y_ratio*p->pel)) * 256.0f);
}

inline float source_at(read_only image2d_t source, const Params *p, float2 pos) {
    float2 clamped = clamp(pos, (float2)(0.0f, 0.0f), (float2)(p->width - 1, p->height - 1));
    float2 position = (float2)(p->offset_x, p->offset_y) + (float2)(0.5f, 0.5f) + clamped;
#if CUBIC_REF && FAST_OCC
    if (p->cubic_ref) return REF_SCALE(p) * cubic_sample(source, position).x;
    return REF_SCALE(p) * read_imagef(source, linear_sampler, position).x;
#elif CUBIC_REF
    return REF_SCALE(p) * cubic_sample(source, position).x;
#else
    return REF_SCALE(p) * read_imagef(source, linear_sampler, position).x;
#endif
}

inline float linear_at(read_only image2d_t source, const Params *p, float2 pos) {
    float2 clamped = clamp(pos, (float2)(0.0f, 0.0f), (float2)(p->width - 1, p->height - 1));
    float2 position = (float2)(p->offset_x, p->offset_y) + (float2)(0.5f, 0.5f) + clamped;
    return REF_SCALE(p) * read_imagef(source, linear_sampler, position).x;
}
#endif

#if FAST_BLEND
inline float ign_noise(int x, int y) {
    float v = 0.06711056f * (float)x + 0.00583715f * (float)y;
    float f = 52.9829189f * (v - floor(v));
    return f - floor(f);
}

inline float linear_source_sample_f(
    read_only image2d_t source, const Params *p, float vx, float vy, float time_f)
{
    float2 step = native_divide(
        (float2)(vx*65535.0f-1024.0f, vy*65535.0f-1024.0f) * time_f,
        (float2)((float)(p->x_ratio*p->pel), (float)(p->y_ratio*p->pel)) * 256.0f);
    float2 moved_pos = clamp((float2)(get_global_id(0), get_global_id(1)) + step,
        (float2)(0.0f, 0.0f), (float2)(p->width-1, p->height-1));
    float2 position = ((float2)(p->offset_x, p->offset_y) + (float2)(0.5f, 0.5f)) + moved_pos;
    return REF_SCALE(p) * read_imagef(source, linear_sampler, position).x;
}
#endif

const sampler_t cell_sampler = CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

inline float cell_sample(read_only image2d_t source, const Params *p) {
    return REF_SCALE(p) * read_imagef(source, cell_sampler,
        (int2)(p->offset_x + get_global_id(0), p->offset_y + get_global_id(1))).x;
}

inline float base_sample(read_only image2d_t source, const Params *p) {
#if CUBIC_REF && FAST_OCC
    if (p->cubic_ref) {
        float2 position = ((float2)(p->offset_x, p->offset_y) + (float2)(0.5f, 0.5f))
            + (float2)(get_global_id(0), get_global_id(1));
        return REF_SCALE(p) * cubic_sample(source, position).x;
    }
    return cell_sample(source, p);
#elif CUBIC_REF
    float2 position = ((float2)(p->offset_x, p->offset_y) + (float2)(0.5f, 0.5f))
        + (float2)(get_global_id(0), get_global_id(1));
    return REF_SCALE(p) * cubic_sample(source, position).x;
#else
    return cell_sample(source, p);
#endif
}

#if FAST_GPU && DEPTH == 8 && !LAYER && !STILL && !HOLD
inline float base_at(read_only image2d_t source, const Params *p, int px, int py) {
#if CUBIC_REF && FAST_OCC
    if (p->cubic_ref) {
        float2 position = (float2)(p->offset_x + px, p->offset_y + py) + (float2)(0.5f, 0.5f);
        return REF_SCALE(p) * cubic_sample(source, position).x;
    }
    return REF_SCALE(p) * read_imagef(source, cell_sampler, (int2)(p->offset_x + px, p->offset_y + py)).x;
#elif CUBIC_REF
    float2 position = (float2)(p->offset_x + px, p->offset_y + py) + (float2)(0.5f, 0.5f);
    return REF_SCALE(p) * cubic_sample(source, position).x;
#else
    return REF_SCALE(p) * read_imagef(source, cell_sampler, (int2)(p->offset_x + px, p->offset_y + py)).x;
#endif
}
#endif

#if HOLD
inline float held_value(read_only image2d_t source, const Params *p) {
    float value = cell_sample(source, p);
#if LINEAR_LUMA
    if (p->linear_luma) {
        value = native_divide(value, 255.0f);
        if (value < 0.018f) value *= 4.5f;
        else value = 1.099f*native_powr(value, 0.45f)-0.099f;
        value *= 255.0f;
    }
#endif
    return value;
}
#endif

inline float2 full_step(const Params *p, float vx, float vy) {
    return native_divide(
        (float2)(vx*65535.0f-1024.0f, vy*65535.0f-1024.0f),
        (float2)((float)(p->x_ratio*p->pel), (float)(p->y_ratio*p->pel)));
}

inline float2 layer_luma(const Params *p, float2 at) {
    return (at+0.5f) * (float2)(p->x_ratio, p->y_ratio);
}

inline float layer_offset(const Params *p, float4 model) {
    return p->offset_y == 0 ? model.y : (p->offset_x == 0 ? model.z : model.w);
}

inline float layer_gap(read_only image2d_t cover, const Params *p, float2 at) {
    return 2040.0f*read_imagef(cover, cell_sampler, layer_luma(p, at)*0.125f).x - 8.0f;
}

inline float layer_reach(const Params *p, float2 f, float2 b) {
    float2 far = fmax(fabs(f), fabs(b)) * (float2)(p->x_ratio, p->y_ratio);
    return fmax(fmax(far.x, far.y) + 1.0f, 16.0f);
}

inline float layer_raw(read_only image2d_t raw, const Params *p, float2 at) {
    return 255.0f * read_imagef(raw, linear_sampler,
        (float2)(p->offset_x, p->offset_y) + at + 0.5f).x;
}

inline int layer_plain(const Params *p, float4 model) {
    return fabs(model.x-1.0f) < 1e-4f && fabs(layer_offset(p, model)) < 1e-2f;
}

inline int layer_set(float4 model) {
    return fabs(model.x-1.0f) > 0.02f || fabs(model.y) > 1.0f;
}

inline float2 layer_scene(
    read_only image2d_t raw, read_only image2d_t layer, const Params *p, float2 at, int moved)
{
    float4 model = read_imagef(layer, linear_sampler, layer_luma(p, at));
    float value = layer_raw(raw, p, at);
    if (fabs(model.x) >= LAYER_MIN) {
        value = native_divide(value - layer_offset(p, model), model.x);
        return (float2)(clamp(value, 0.0f, 255.0f), 1.0f);
    }
    if (!moved) return (float2)(value, 0.0f);
    float2 top = (float2)(p->width-1, p->height-1);
    float total = 0.0f, weight = 0.0f;
    for (int ring = 1; ring <= 3 && weight == 0.0f; ring++) {
        float radius = LAYER_FILL*ring / p->x_ratio;
        for (int turn = 0; turn < 8; turn++) {
            float2 side = (float2)(
                (turn == 0 || turn == 1 || turn == 7) - (turn >= 3 && turn <= 5),
                (turn >= 1 && turn <= 3) - (turn >= 5));
            float2 near = clamp(at + side*radius, (float2)(0.0f, 0.0f), top);
            float4 m = read_imagef(layer, linear_sampler, layer_luma(p, near));
            if (fabs(m.x) < LAYER_MIN) continue;
            float scene = native_divide(layer_raw(raw, p, near) - layer_offset(p, m), m.x);
            total += clamp(scene, 0.0f, 255.0f);
            weight += 1.0f;
        }
    }
    return weight > 0.0f ? (float2)(total/weight, 1.0f) : (float2)(value, 0.0f);
}

inline void layer_agree(
    read_only image2d_t raw_f, read_only image2d_t raw_b, read_only image2d_t layer,
    const Params *p, float2 at_f, float2 at_b, float2 *moved_f, float2 *moved_b)
{
    int on_f = !layer_plain(p, read_imagef(layer, linear_sampler, layer_luma(p, at_f)));
    int on_b = !layer_plain(p, read_imagef(layer, linear_sampler, layer_luma(p, at_b)));
    if (on_f == on_b || moved_f->y < 0.5f || moved_b->y < 0.5f) return;
    if (on_f) {
        float raw = layer_raw(raw_f, p, at_f);
        if (fabs(raw - moved_b->x) < fabs(moved_f->x - moved_b->x)) moved_f->x = raw;
    } else {
        float raw = layer_raw(raw_b, p, at_b);
        if (fabs(raw - moved_f->x) < fabs(moved_b->x - moved_f->x)) moved_b->x = raw;
    }
}

kernel void linear_luma(read_only image2d_t source, write_only image2d_t destination) {
    int2 position = (int2)(get_global_id(0), get_global_id(1));
#if DEPTH > 8
    float value = (float)(1 << (16 - DEPTH)) * read_imagef(source, nearest_sampler, position).x;
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
#if STILL || HOLD
    , read_only image2d_t still
#endif
#if LAYER
    , read_only image2d_t layer, read_only image2d_t cover,
    read_only image2d_t raw_f, read_only image2d_t raw_b
#endif
    )
{
#if FAST_GPU && DEPTH == 8 && !LAYER && !STILL && !HOLD
    int x0 = get_global_id(0) * 4, y0 = get_global_id(1) * 2;
    if (x0 >= p.width || y0 >= p.height) return;
    float time = native_divide((float)p.phase, 256.0f);
    uchar out0[4], out1[4];
    int has_y1 = y0 + 1 < p.height;
    for (int q = 0; q < 2; q++) {
        int qx = x0 + q * 2;
        if (qx >= p.width) {
            out0[q * 2] = 0; out0[q * 2 + 1] = 0;
            out1[q * 2] = 0; out1[q * 2 + 1] = 0;
            continue;
        }
        float2 field_position = native_divide(
            (float2)(((float)qx + 0.5f) * p.x_ratio - p.origin_x, ((float)y0 + 0.5f) * p.y_ratio - p.origin_y),
            (float2)(p.block_w, p.block_h));
        float4 vector = field_sample(vectors, field_position, CUBIC);
#if ALGO >= 21 || HAS_SAD
        float4 mask = field_sample(masks, field_position, CUBIC);
#endif
        float2 step_b = moved_step(&p, vector.z, vector.y, 256 - p.phase);
        float2 step_f = moved_step(&p, vector.x, vector.w, p.phase);
#if ALGO == 23
# if FAST_OCC
        int occ_y = mask.y > 0.01f;
        int occ_z = mask.z > 0.01f;
        float2 step_bb = (float2)(0.0f);
        float2 step_ff = (float2)(0.0f);
        if (occ_y || occ_z) {
            float4 ext = field_sample(vectors_ext, field_position, CUBIC);
            if (occ_y) step_bb = moved_step(&p, ext.z, ext.y, 256 - p.phase);
            if (occ_z) step_ff = moved_step(&p, ext.x, ext.w, p.phase);
        }
# else
        float4 ext = field_sample(vectors_ext, field_position, CUBIC);
        float2 step_bb = moved_step(&p, ext.z, ext.y, 256 - p.phase);
        float2 step_ff = moved_step(&p, ext.x, ext.w, p.phase);
# endif
#endif
        for (int dy = 0; dy < 2; dy++) {
            int py = y0 + dy;
            if (dy == 1 && !has_y1) {
                out1[q * 2] = 0; out1[q * 2 + 1] = 0;
                continue;
            }
            for (int dx = 0; dx < 2; dx++) {
                int px = qx + dx;
                if (px >= p.width) {
                    if (dy == 0) out0[q * 2 + dx] = 0;
                    else out1[q * 2 + dx] = 0;
                    continue;
                }
                float2 base_xy = (float2)(px, py);
                float ref_b = source_at(source_b, &p, base_xy + step_b);
                float ref_f = source_at(source_f, &p, base_xy + step_f);
#if ALGO == 13 || ALGO == 22 || HAS_SAD
                float ref_b0 = base_at(source_b, &p, px, py);
                float ref_f0 = base_at(source_f, &p, px, py);
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
# if FAST_OCC
                float r_lo = ref_f, r_hi = ref_b;
                if (occ_y) {
                    float ref_bb = linear_at(source_b, &p, base_xy + step_bb);
                    r_lo = mix(ref_f, median3(ref_b, ref_bb, ref_f), mask.y);
                }
                if (occ_z) {
                    float ref_ff = linear_at(source_f, &p, base_xy + step_ff);
                    r_hi = mix(ref_b, median3(ref_b, ref_ff, ref_f), mask.z);
                }
                result = mix(r_lo, r_hi, time);
# else
                float ref_bb = linear_at(source_b, &p, base_xy + step_bb);
                float ref_ff = linear_at(source_f, &p, base_xy + step_ff);
                result = mix(
                    mix(ref_f, median3(ref_b, ref_bb, ref_f), mask.y),
                    mix(ref_b, median3(ref_b, ref_ff, ref_f), mask.z),
                    time);
# endif
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
#if LINEAR_LUMA
                if (p.linear_luma) {
                    result = native_divide(result, 255.0f);
                    if (result < 0.018f) result *= 4.5f;
                    else result = 1.099f*native_powr(result, 0.45f)-0.099f;
                    result *= 255.0f;
                }
#endif
#if DITHER
                result += native_divide((float)(bayer[px%8][py%8]-32), 65.0f);
#endif
                uchar val = (uchar)clamp(round(result), 0.0f, 255.0f);
                if (dy == 0) out0[q * 2 + dx] = val;
                else out1[q * 2 + dx] = val;
            }
        }
    }
    if (x0 + 4 <= p.width && ((destination_stride & 3) == 0)) {
        *((global uchar4 *)(destination + y0 * destination_stride + x0)) =
            (uchar4)(out0[0], out0[1], out0[2], out0[3]);
        if (has_y1) {
            *((global uchar4 *)(destination + (y0 + 1) * destination_stride + x0)) =
                (uchar4)(out1[0], out1[1], out1[2], out1[3]);
        }
    } else {
        for (int dx = 0; dx < 4 && x0 + dx < p.width; dx++) {
            destination[y0 * destination_stride + x0 + dx] = out0[dx];
            if (has_y1) {
                destination[(y0 + 1) * destination_stride + x0 + dx] = out1[dx];
            }
        }
    }
    return;
#else
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
    float result;
    int layered = 0;
#if LAYER
    float2 at = (float2)(x, y), top = (float2)(p.width-1, p.height-1);
    float2 step_f = full_step(&p, vector.x, vector.w);
    float2 step_b = full_step(&p, vector.z, vector.y);
    float gap = layer_gap(cover, &p, at);
    float2 at_f = clamp(at + step_f*time, (float2)(0.0f, 0.0f), top);
    float2 at_b = clamp(at + step_b*(1.0f-time), (float2)(0.0f, 0.0f), top);
    layered = gap <= layer_reach(&p, step_f*time, step_b*(1.0f-time));
    if (layered) {
        float4 model = read_imagef(layer, linear_sampler, layer_luma(&p, at));
        float own_f = layer_raw(raw_f, &p, at), own_b = layer_raw(raw_b, &p, at);
        if (fabs(model.x) < LAYER_MIN || (gap <= LAYER_NEAR && !layer_plain(&p, model) && fabs(own_f-own_b) <= LAYER_SAME)) {
            result = mix(own_f, own_b, time);
        } else {
            float2 moved_f = layer_scene(raw_f, layer, &p, at_f, 1);
            float2 moved_b = layer_scene(raw_b, layer, &p, at_b, 1);
            layer_agree(raw_f, raw_b, layer, &p, at_f, at_b, &moved_f, &moved_b);
            float2 base_f = layer_scene(raw_f, layer, &p, at, 0);
            float2 base_b = layer_scene(raw_b, layer, &p, at, 0);
            float base = mix(base_f.x, base_b.x, time);
            float scene;
# if ALGO == 1
            scene = moved_b.y > 0.5f ? moved_b.x : (moved_f.y > 0.5f ? moved_f.x : base);
# elif ALGO == 2
            scene = moved_f.y > 0.5f ? moved_f.x : (moved_b.y > 0.5f ? moved_b.x : base);
# else
            if (moved_f.y > 0.5f && moved_b.y > 0.5f) scene = median3(moved_f.x, moved_b.x, base);
            else scene = moved_f.y > 0.5f ? moved_f.x : (moved_b.y > 0.5f ? moved_b.x : base);
# endif
            result = model.x*scene + layer_offset(&p, model);
        }
    } else
#endif
    {

    float ref_b = source_sample(source_b, &p, vector.z, vector.y, 256-p.phase);
    float ref_f = source_sample(source_f, &p, vector.x, vector.w, p.phase);
#if ALGO == 13 || ALGO == 22 || HAS_SAD
    float ref_b0 = base_sample(source_b, &p);
    float ref_f0 = base_sample(source_f, &p);
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
# if FAST_OCC
    float r_lo = ref_f, r_hi = ref_b;
    if (mask.y > 0.01f || mask.z > 0.01f) {
        float4 ext = field_sample(vectors_ext, field_position, CUBIC);
        if (mask.y > 0.01f) {
            float ref_bb = linear_source_sample(source_b, &p, ext.z, ext.y, 256-p.phase);
            r_lo = mix(ref_f, median3(ref_b, ref_bb, ref_f), mask.y);
        }
        if (mask.z > 0.01f) {
            float ref_ff = linear_source_sample(source_f, &p, ext.x, ext.w, p.phase);
            r_hi = mix(ref_b, median3(ref_b, ref_ff, ref_f), mask.z);
        }
    }
    result = mix(r_lo, r_hi, time);
# else
    float r_lo = ref_f, r_hi = ref_b;
    if (mask.y != 0.0f || mask.z != 0.0f) {
        float4 ext = field_sample(vectors_ext, field_position, CUBIC);
        if (mask.y != 0.0f) {
            float ref_bb = linear_source_sample(source_b, &p, ext.z, ext.y, 256-p.phase);
            r_lo = mix(ref_f, median3(ref_b, ref_bb, ref_f), mask.y);
        }
        if (mask.z != 0.0f) {
            float ref_ff = linear_source_sample(source_f, &p, ext.x, ext.w, p.phase);
            r_hi = mix(ref_b, median3(ref_b, ref_ff, ref_f), mask.z);
        }
    }
    result = mix(r_lo, r_hi, time);
# endif
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
        result = mix(cell_sample(source_f, &p), cell_sample(source_b, &p), time);
#endif
    }

#if LINEAR_LUMA
    if (p.linear_luma && !layered) {
#if DEPTH > 8
        result = native_divide(result, 256.0f);
#else
        result = native_divide(result, 255.0f);
#endif
        if (result < 0.018f) result *= 4.5f;
        else result = 1.099f*native_powr(result, 0.45f)-0.099f;
#if DEPTH > 8
        result *= 256.0f;
#else
        result *= 255.0f;
#endif
    }
#endif
#if HOLD
    float weight = MASK_AT(still, p, x, y);
    if (weight < 1.0f) result = mix(held_value(source_f, &p), result, weight);
#endif
#if DEPTH > 8
    ((global ushort *)destination)[y*destination_stride+x] =
        (ushort)clamp(round((float)(1 << (DEPTH - 8))*result), 0.0f, (float)((1 << DEPTH) - 1));
#else
# if DITHER
    result += native_divide((float)(bayer[x%8][y%8]-32), 65.0f);
# endif
    destination[y*destination_stride+x] = (uchar)clamp(round(result), 0.0f, 255.0f);
#endif
#endif
}

#if BLEND && DEPTH == 8
#define MAX_STEPS 16
typedef struct {
    int count;
    int phase[MAX_STEPS];
    float sad_blend[MAX_STEPS];
    float weight[MAX_STEPS];
    float gamma;
    float source;
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

inline float blend_value(
    read_only image2d_t source_f, read_only image2d_t source_b, Params *p,
    float4 vector, float4 mask, float4 ext, float ref_f0, float ref_b0, float time)
{
    float result;
    float ref_b = source_sample(source_b, p, vector.z, vector.y, 256-p->phase);
    float ref_f = source_sample(source_f, p, vector.x, vector.w, p->phase);

#if ALGO == 1
    result = ref_b + ref_f*0.00001f;
#elif ALGO == 2
    result = ref_f + ref_b*0.00001f;
#elif ALGO == 11
    result = mix(ref_f, ref_b, time);
#elif ALGO == 13
    result = median3(ref_f, ref_b, mix(ref_f0, ref_b0, p->sad_blend));
#elif ALGO == 21
    result = mix(mix(ref_f, ref_b, mask.y), mix(ref_b, ref_f, mask.z), time);
#elif ALGO == 22
    result = median3(
        mix(ref_f, ref_b, mask.y),
        mix(ref_b, ref_f, mask.z),
        mix(ref_f0, ref_b0, time));
#else
# if FAST_OCC
    float r_lo = ref_f, r_hi = ref_b;
    if (mask.y > 0.01f) {
        float ref_bb = linear_source_sample(source_b, p, ext.z, ext.y, 256-p->phase);
        r_lo = mix(ref_f, median3(ref_b, ref_bb, ref_f), mask.y);
    }
    if (mask.z > 0.01f) {
        float ref_ff = linear_source_sample(source_f, p, ext.x, ext.w, p->phase);
        r_hi = mix(ref_b, median3(ref_b, ref_ff, ref_f), mask.z);
    }
    result = mix(r_lo, r_hi, time);
# else
    float r_lo = ref_f, r_hi = ref_b;
    if (mask.y != 0.0f) {
        float ref_bb = linear_source_sample(source_b, p, ext.z, ext.y, 256-p->phase);
        r_lo = mix(ref_f, median3(ref_b, ref_bb, ref_f), mask.y);
    }
    if (mask.z != 0.0f) {
        float ref_ff = linear_source_sample(source_f, p, ext.x, ext.w, p->phase);
        r_hi = mix(ref_b, median3(ref_b, ref_ff, ref_f), mask.z);
    }
    result = mix(r_lo, r_hi, time);
# endif
#endif

#if HAS_SAD
# if ALGO == 1
    result = mix(result, ref_b0, mask.x);
# elif ALGO == 2
    result = mix(result, ref_f0, mask.w);
# else
    result = mix(result, mix(ref_f0, ref_b0, p->sad_blend), fmax(mask.w, mask.x));
# endif
#endif
    return result;
}

#if FAST_BLEND
inline float fast_blend_tap(
    read_only image2d_t source_f, read_only image2d_t source_b, const Params *p,
    float4 vector, float4 mask, float pf, int is_chroma)
{
    float time = pf * 0.00390625f;
#if FAST_BLEND >= 2
    if (is_chroma) {
        return pf <= 128.0f
            ? linear_source_sample_f(source_f, p, vector.x, vector.w, pf)
            : linear_source_sample_f(source_b, p, vector.z, vector.y, 256.0f - pf);
    }
#endif
    float ref_f = linear_source_sample_f(source_f, p, vector.x, vector.w, pf);
    float ref_b = linear_source_sample_f(source_b, p, vector.z, vector.y, 256.0f - pf);
#if ALGO == 1
    return ref_b;
#elif ALGO == 2
    return ref_f;
#elif ALGO >= 21
    return mix(mix(ref_f, ref_b, mask.y), mix(ref_b, ref_f, mask.z), time);
#else
    return mix(ref_f, ref_b, time);
#endif
}
#endif

inline float blend_round(float result, const Params *p, int layered, int x, int y)
{
#if LINEAR_LUMA
    if (p->linear_luma && !layered) {
        result = native_divide(result, 255.0f);
        if (result < 0.018f) result *= 4.5f;
        else result = 1.099f*native_powr(result, 0.45f)-0.099f;
        result *= 255.0f;
    }
#endif
#if DITHER
    result += native_divide((float)(bayer[x%8][y%8]-32), 65.0f);
#endif
    return clamp(round(result), 0.0f, 255.0f);
}

#if FUSE
inline float fuse_slot(
    read_only image2d_t source, read_only image2d_t source_f, read_only image2d_t source_b,
    read_only image2d_t vectors,
#if ADAPTIVE && ALGO == 23 && !FAST_BLEND
    read_only image2d_t vectors_ext,
#endif
    Params p, Steps steps, float total, int x, int y, int slot_id)
{
    if (steps.source != 0.0f) {
#pragma OPENCL FP_CONTRACT OFF
        float value = round(255.0f *
            read_imagef(source, nearest_sampler, (int2)(p.offset_x + x, p.offset_y + y)).x);
        float term = steps.source * value;
        total = term + total;
    }
    if (steps.count == 0) return total;
    float2 field_position = native_divide(
        (float2)(x*p.x_ratio-p.origin_x, y*p.y_ratio-p.origin_y),
        (float2)(p.block_w, p.block_h));
    float4 vector = field_sample(vectors, field_position, CUBIC);
    float2 sf = full_step(&p, vector.x, vector.w);
    float2 sb = full_step(&p, vector.z, vector.y);
#if FAST_BLEND
    float4 mask = (float4)(0.0f);
# if ALGO >= 21
    float2 dv = sf + sb;
    float occ = clamp(native_sqrt(dv.x * dv.x + dv.y * dv.y) * 0.25f, 0.0f, 1.0f);
    mask = (float4)(0.0f, occ, occ, 0.0f);
# endif
# if ALGO != 1 && ALGO != 2
    int zero_mv = sf.x == 0.0f && sf.y == 0.0f && sb.x == 0.0f && sb.y == 0.0f;
    if (zero_mv) {
        float rf = linear_source_sample_f(source_f, &p, vector.x, vector.w, 0.0f);
        float rb = linear_source_sample_f(source_b, &p, vector.z, vector.y, 0.0f);
        if (rf == rb) {
            float value = blend_round(rf, &p, 0, x, y);
            float wsum = 0.0f;
            for (int step = 0; step < steps.count; step++) wsum += steps.weight[step];
            return total + wsum * value;
        }
    }
# endif
    int is_chroma = p.x_ratio > 1 || p.y_ratio > 1;
    int cap = (FAST_BLEND >= 2 || is_chroma) ? 1 : 2;
    float travel = fmax(fmax(fabs(sf.x), fabs(sf.y)), fmax(fabs(sb.x), fabs(sb.y)));
    int want = travel < 0.5f ? 1 : cap;
    int buckets = min(want, steps.count);
    if (steps.count == 2) {
        float p0 = (float)steps.phase[0];
        float p1 = (float)steps.phase[1];
        float dp = (p1 - p0) * 0.5f;
        if (buckets == 1) {
            float lo = clamp(p0 - dp, 0.0f, 256.0f);
            float hi = clamp(p1 + dp, 0.0f, 256.0f);
            float j = ign_noise(x + slot_id * 17, y + slot_id * 31);
            float pf = mix(lo, hi, j);
            float result = fast_blend_tap(source_f, source_b, &p, vector, mask, pf, is_chroma);
            float value = blend_round(result, &p, 0, x, y);
            total += (steps.weight[0] + steps.weight[1]) * value;
        } else {
            float j0 = ign_noise(x + slot_id * 17, y + slot_id * 31);
            float pf0 = mix(clamp(p0 - dp, 0.0f, 256.0f), clamp(p0 + dp, 0.0f, 256.0f), j0);
            float r0 = fast_blend_tap(source_f, source_b, &p, vector, mask, pf0, is_chroma);
            total += steps.weight[0] * blend_round(r0, &p, 0, x, y);
            float j1 = ign_noise(x + slot_id * 17 + 7, y + slot_id * 31 + 13);
            float pf1 = mix(clamp(p1 - dp, 0.0f, 256.0f), clamp(p1 + dp, 0.0f, 256.0f), j1);
            float r1 = fast_blend_tap(source_f, source_b, &p, vector, mask, pf1, is_chroma);
            total += steps.weight[1] * blend_round(r1, &p, 0, x, y);
        }
        return total;
    }
    for (int b = 0; b < buckets; b++) {
        int s0 = (b * steps.count) / buckets;
        int s1 = ((b + 1) * steps.count) / buckets;
        float wsum = 0.0f;
        for (int s = s0; s < s1; s++) wsum += steps.weight[s];
        float j = ign_noise(x + slot_id * 17 + b * 7, y + slot_id * 31 + b * 13);
        float pf = s1 - s0 > 1
            ? mix((float)steps.phase[s0], (float)steps.phase[s1 - 1], j)
            : (float)steps.phase[s0];
        float result = fast_blend_tap(source_f, source_b, &p, vector, mask, pf, is_chroma);
        total += wsum * blend_round(result, &p, 0, x, y);
    }
    return total;
#else
#if ADAPTIVE && ALGO == 23
# if FAST_OCC
    float2 dv_occ = sf + sb;
    float occ_val = clamp(native_sqrt(dv_occ.x * dv_occ.x + dv_occ.y * dv_occ.y) * 0.25f, 0.0f, 1.0f);
    float4 ext = occ_val > 0.01f ? field_sample(vectors_ext, field_position, CUBIC) : (float4)(0.0f);
# else
    float4 ext = field_sample(vectors_ext, field_position, CUBIC);
# endif
#else
    float4 ext = (float4)(0.0f);
#endif
#if ALGO == 13 || ALGO == 22
    float ref_b0 = base_sample(source_b, &p);
    float ref_f0 = base_sample(source_f, &p);
#else
    float ref_b0 = 0.0f;
    float ref_f0 = 0.0f;
#endif
#if ALGO != 1 && ALGO != 2
    int zero_mv = sf.x == 0.0f && sf.y == 0.0f && sb.x == 0.0f && sb.y == 0.0f;
#if ADAPTIVE && ALGO == 23 && !FAST_OCC
    float2 ef = full_step(&p, ext.x, ext.w);
    float2 eb = full_step(&p, ext.z, ext.y);
    zero_mv = zero_mv && ef.x == 0.0f && ef.y == 0.0f && eb.x == 0.0f && eb.y == 0.0f;
#endif
    if (zero_mv) {
        float rf = source_sample(source_f, &p, vector.x, vector.w, 0);
        float rb = source_sample(source_b, &p, vector.z, vector.y, 0);
#if ADAPTIVE && ALGO == 23 && !FAST_OCC
        float rff = linear_source_sample(source_f, &p, ext.x, ext.w, 0);
        float rbb = linear_source_sample(source_b, &p, ext.z, ext.y, 0);
        int same = rf == rb && rff == rf && rbb == rb;
#elif ALGO == 13 || ALGO == 22
        int same = rf == rb && ref_f0 == rf && ref_b0 == rb;
#else
        int same = rf == rb;
#endif
        if (same) {
            float value = blend_round(rf, &p, 0, x, y);
            for (int step = 0; step < steps.count; step++) {
#pragma OPENCL FP_CONTRACT OFF
                float term = steps.weight[step] * value;
                total = term + total;
            }
            return total;
        }
    }
#endif
#if ADAPTIVE
    float4 mask = (float4)(0.0f);
#if ALGO >= 21
    float2 dv = sf + sb;
    float occ = clamp(native_sqrt(dv.x * dv.x + dv.y * dv.y) * 0.25f, 0.0f, 1.0f);
    mask = (float4)(0.0f, occ, occ, 0.0f);
#endif
    float travel = fmax(fmax(fabs(sf.x), fabs(sf.y)), fmax(fabs(sb.x), fabs(sb.y)));
    int want = travel < 0.25f ? 1 : (travel < 1.5f ? 2 : (travel < 4.0f ? 4 : (travel < 8.0f ? 8 : steps.count)));
    int buckets = min(want, steps.count);
    for (int b = 0; b < buckets; b++) {
        int s0 = (b * steps.count) / buckets;
        int s1 = ((b + 1) * steps.count) / buckets;
        int mid = (s0 + s1 - 1) >> 1;
        float wsum = 0.0f;
        for (int s = s0; s < s1; s++) wsum += steps.weight[s];
        p.phase = steps.phase[mid];
        p.sad_blend = steps.sad_blend[mid];
        float time = native_divide((float)p.phase, 256.0f);
        float result = blend_value(source_f, source_b, &p, vector, mask, ext, ref_f0, ref_b0, time);
        float value = blend_round(result, &p, 0, x, y);
        {
#pragma OPENCL FP_CONTRACT OFF
            float term = wsum * value;
            total = term + total;
        }
    }
    return total;
#else
    for (int step = 0; step < steps.count; step++) {
        p.phase = steps.phase[step];
        p.sad_blend = steps.sad_blend[step];
        float time = native_divide((float)p.phase, 256.0f);
        float result = blend_value(
            source_f, source_b, &p, vector, (float4)(0.0f), ext, ref_f0, ref_b0, time);
        float value = blend_round(result, &p, 0, x, y);
        {
#pragma OPENCL FP_CONTRACT OFF
            float term = steps.weight[step] * value;
            total = term + total;
        }
    }
    return total;
#endif
#endif
}

kernel void blend_frame(
    global float *sums, int stride,
    read_only image2d_t source0, read_only image2d_t forward0, read_only image2d_t backward0,
    read_only image2d_t vectors0,
    read_only image2d_t source1, read_only image2d_t forward1, read_only image2d_t backward1,
    read_only image2d_t vectors1,
    read_only image2d_t source2, read_only image2d_t forward2, read_only image2d_t backward2,
    read_only image2d_t vectors2,
    read_only image2d_t source3, read_only image2d_t forward3, read_only image2d_t backward3,
    read_only image2d_t vectors3,
    Params p, Steps steps0, Steps steps1, Steps steps2, Steps steps3
#if ADAPTIVE && ALGO == 23 && !FAST_BLEND
    , read_only image2d_t ext0, read_only image2d_t ext1,
    read_only image2d_t ext2, read_only image2d_t ext3
#endif
    )
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= p.width || y >= p.height) return;
    int overwrite = stride < 0;
    stride = abs(stride);
    global float *sum = sums + (p.offset_y+y)*stride + p.offset_x + x;
    float total = overwrite ? 0.0f : *sum;
#if ADAPTIVE && ALGO == 23 && !FAST_BLEND
    total = fuse_slot(source0, forward0, backward0, vectors0, ext0, p, steps0, total, x, y, 0);
    total = fuse_slot(source1, forward1, backward1, vectors1, ext1, p, steps1, total, x, y, 1);
    total = fuse_slot(source2, forward2, backward2, vectors2, ext2, p, steps2, total, x, y, 2);
    total = fuse_slot(source3, forward3, backward3, vectors3, ext3, p, steps3, total, x, y, 3);
#else
    total = fuse_slot(source0, forward0, backward0, vectors0, p, steps0, total, x, y, 0);
    total = fuse_slot(source1, forward1, backward1, vectors1, p, steps1, total, x, y, 1);
    total = fuse_slot(source2, forward2, backward2, vectors2, p, steps2, total, x, y, 2);
    total = fuse_slot(source3, forward3, backward3, vectors3, p, steps3, total, x, y, 3);
#endif
    *sum = total;
}
#else
kernel void blend_frame(
    global float *sums, int stride,
    read_only image2d_t source_f, read_only image2d_t source_b,
    read_only image2d_t vectors, read_only image2d_t vectors_ext,
    read_only image2d_array_t masks, Params p, Steps steps
#if STILL || HOLD
    , read_only image2d_t still
#endif
#if LAYER
    , read_only image2d_t layer, read_only image2d_t cover,
    read_only image2d_t raw_f, read_only image2d_t raw_b
#endif
    )
{
    local float curve[256];
    if (steps.gamma != 1.0f) {
        int lane = get_local_id(1)*get_local_size(0) + get_local_id(0);
        int lanes = get_local_size(0)*get_local_size(1);
        for (int i = lane; i < 256; i += lanes) curve[i] = powr((float)i / 255.0f, steps.gamma);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= p.width || y >= p.height) return;

    int overwrite = stride < 0;
    stride = abs(stride);
    float2 field_position = native_divide(
        (float2)(x*p.x_ratio-p.origin_x, y*p.y_ratio-p.origin_y),
        (float2)(p.block_w, p.block_h));
    float4 vector = field_sample(vectors, field_position, CUBIC);
#if ALGO == 13 || ALGO == 22 || HAS_SAD
    float ref_b0 = base_sample(source_b, &p);
    float ref_f0 = base_sample(source_f, &p);
#else
    float ref_b0 = 0.0f;
    float ref_f0 = 0.0f;
#endif
    float4 ext = (float4)(0.0f);
    global float *sum = sums + (p.offset_y+y)*stride + p.offset_x + x;
    float total = overwrite ? 0.0f : *sum;
#if STILL
    int kept = STILL_AT(still, p, x, y);
    float kept_f = 0.0f, kept_b = 0.0f;
    if (kept) {
        kept_f = cell_sample(source_f, &p);
        kept_b = cell_sample(source_b, &p);
    }
#endif
#if HOLD
    float weight = MASK_AT(still, p, x, y);
    float held = held_value(source_f, &p);
#endif
#if LAYER
    float2 at = (float2)(x, y), top = (float2)(p.width-1, p.height-1);
    float2 step_f = full_step(&p, vector.x, vector.w);
    float2 step_b = full_step(&p, vector.z, vector.y);
    float gap = layer_gap(cover, &p, at);
    float2 zsf = step_f, zsb = step_b;
    int can_layer = gap <= layer_reach(&p, step_f, step_b);
#else
    float2 zsf = full_step(&p, vector.x, vector.w);
    float2 zsb = full_step(&p, vector.z, vector.y);
    int can_layer = 0;
#endif
#if STILL
    int is_kept = kept;
    if (is_kept && !can_layer && kept_f == kept_b) {
        float value = blend_round(kept_f, &p, 0, x, y);
#if HOLD
        if (weight < 1.0f) value = round(mix(held, value, weight));
#endif
        if (steps.gamma != 1.0f) value = curve[(int)value];
        for (int step = 0; step < steps.count; step++) {
#pragma OPENCL FP_CONTRACT OFF
            float term = steps.weight[step] * value;
            total = term + total;
        }
        *sum = total;
        return;
    }
#else
    int is_kept = 0;
#endif

#if !LAYER
    if (!is_kept) {
        int zero_mv = zsf.x == 0.0f && zsf.y == 0.0f && zsb.x == 0.0f && zsb.y == 0.0f;
#if ALGO == 23 && !FAST_BLEND
        if (zero_mv) {
            ext = field_sample(vectors_ext, field_position, CUBIC);
            float2 zef = full_step(&p, ext.x, ext.w);
            float2 zeb = full_step(&p, ext.z, ext.y);
            zero_mv = zef.x == 0.0f && zef.y == 0.0f && zeb.x == 0.0f && zeb.y == 0.0f;
        }
#endif
#if (ALGO == 1 || ALGO == 2) && !HAS_SAD
        if (zero_mv) {
            float rf = source_sample(source_f, &p, vector.x, vector.w, 0);
            float rb = source_sample(source_b, &p, vector.z, vector.y, 0);
            float res = ALGO == 1 ? (rb + rf * 0.00001f) : (rf + rb * 0.00001f);
            float value = blend_round(res, &p, 0, x, y);
#if HOLD
            if (weight < 1.0f) value = round(mix(held, value, weight));
#endif
            if (steps.gamma != 1.0f) value = curve[(int)value];
            for (int step = 0; step < steps.count; step++) {
#pragma OPENCL FP_CONTRACT OFF
                float term = steps.weight[step] * value;
                total = term + total;
            }
            *sum = total;
            return;
        }
#elif ALGO != 1 && ALGO != 2
        if (zero_mv) {
            float rf = source_sample(source_f, &p, vector.x, vector.w, 0);
            float rb = source_sample(source_b, &p, vector.z, vector.y, 0);
#if ALGO == 23 && !FAST_BLEND
            float rff = linear_source_sample(source_f, &p, ext.x, ext.w, 0);
            float rbb = linear_source_sample(source_b, &p, ext.z, ext.y, 0);
#if HAS_SAD
            int same = rf == rb && rff == rf && rbb == rb && ref_f0 == rf && ref_b0 == rb;
#else
            int same = rf == rb && rff == rf && rbb == rb;
#endif
#elif ALGO == 13 || ALGO == 22 || HAS_SAD
            int same = rf == rb && ref_f0 == rf && ref_b0 == rb;
#else
            int same = rf == rb;
#endif
            if (same) {
                float value = blend_round(rf, &p, 0, x, y);
#if HOLD
                if (weight < 1.0f) value = round(mix(held, value, weight));
#endif
                if (steps.gamma != 1.0f) value = curve[(int)value];
                for (int step = 0; step < steps.count; step++) {
#pragma OPENCL FP_CONTRACT OFF
                    float term = steps.weight[step] * value;
                    total = term + total;
                }
                *sum = total;
                return;
            }
        }
#endif
    }
#endif

#if LAYER
    if (can_layer) {
        float4 l_model = read_imagef(layer, linear_sampler, layer_luma(&p, at));
        float l_own_f = layer_raw(raw_f, &p, at);
        float l_own_b = layer_raw(raw_b, &p, at);
        int l_opaque = fabs(l_model.x) < LAYER_MIN
            || (gap <= LAYER_NEAR && !layer_plain(&p, l_model) && fabs(l_own_f - l_own_b) <= LAYER_SAME);
        if (l_opaque && gap <= 16.0f && l_own_f == l_own_b) {
            float value = blend_round(l_own_f, &p, 1, x, y);
#if HOLD
            if (weight < 1.0f) value = round(mix(held, value, weight));
#endif
            if (steps.gamma != 1.0f) value = curve[(int)value];
            for (int step = 0; step < steps.count; step++) {
#pragma OPENCL FP_CONTRACT OFF
                float term = steps.weight[step] * value;
                total = term + total;
            }
            *sum = total;
            return;
        }
        float l_base_f = 0.0f, l_base_b = 0.0f;
        if (!l_opaque) {
            float loff = layer_offset(&p, l_model);
            l_base_f = clamp(native_divide(l_own_f - loff, l_model.x), 0.0f, 255.0f);
            l_base_b = clamp(native_divide(l_own_b - loff, l_model.x), 0.0f, 255.0f);
        }
#if FAST_BLEND
        if (!is_kept) {
            int fb_chroma = p.x_ratio > 1 || p.y_ratio > 1;
            int fb_cap = (FAST_BLEND >= 2 || fb_chroma) ? 1 : 2;
            float fb_travel = fmax(fmax(fabs(zsf.x), fabs(zsf.y)), fmax(fabs(zsb.x), fabs(zsb.y)));
            int fb_want = fb_travel < 0.5f ? 1 : fb_cap;
            int fb_buckets = min(fb_want, steps.count);
            float4 fb_mask = (float4)(0.0f);
#if ALGO >= 21
            float2 fb_dv = zsf + zsb;
            float fb_occ = clamp(native_sqrt(fb_dv.x * fb_dv.x + fb_dv.y * fb_dv.y) * 0.25f, 0.0f, 1.0f);
            fb_mask = (float4)(0.0f, fb_occ, fb_occ, 0.0f);
#endif
            for (int b = 0; b < fb_buckets; b++) {
                float wsum;
                float pf;
                int sad_idx;
                float j = ign_noise(x + b * 7, y + b * 13);
                if (steps.count == 2) {
                    float p0 = (float)steps.phase[0];
                    float p1 = (float)steps.phase[1];
                    float dp = (p1 - p0) * 0.5f;
                    float lo = fb_buckets == 1 ? clamp(p0 - dp, 0.0f, 256.0f)
                        : clamp((b == 0 ? p0 : p1) - dp, 0.0f, 256.0f);
                    float hi = fb_buckets == 1 ? clamp(p1 + dp, 0.0f, 256.0f)
                        : clamp((b == 0 ? p0 : p1) + dp, 0.0f, 256.0f);
                    wsum = fb_buckets == 1 ? (steps.weight[0] + steps.weight[1]) : steps.weight[b];
                    pf = mix(lo, hi, j);
                    sad_idx = b;
                } else {
                    int s0 = (b * steps.count) / fb_buckets;
                    int s1 = ((b + 1) * steps.count) / fb_buckets;
                    sad_idx = (s0 + s1 - 1) >> 1;
                    wsum = 0.0f;
                    for (int s = s0; s < s1; s++) wsum += steps.weight[s];
                    pf = s1 - s0 > 1
                        ? mix((float)steps.phase[s0], (float)steps.phase[s1 - 1], j)
                        : (float)steps.phase[s0];
                }
                float result;
                int layered = 0;
                float time = pf * 0.00390625f;
                if (gap <= layer_reach(&p, step_f*time, step_b*(1.0f-time))) {
                    layered = 1;
                    if (l_opaque) {
                        result = mix(l_own_f, l_own_b, time);
                    } else {
                        float2 at_f = clamp(at + step_f*time, (float2)(0.0f, 0.0f), top);
                        float2 at_b = clamp(at + step_b*(1.0f-time), (float2)(0.0f, 0.0f), top);
                        float2 moved_f = layer_scene(raw_f, layer, &p, at_f, 1);
                        float2 moved_b = layer_scene(raw_b, layer, &p, at_b, 1);
                        layer_agree(raw_f, raw_b, layer, &p, at_f, at_b, &moved_f, &moved_b);
                        float base = mix(l_base_f, l_base_b, time);
                        float scene;
# if ALGO == 1
                        scene = moved_b.y > 0.5f ? moved_b.x : (moved_f.y > 0.5f ? moved_f.x : base);
# elif ALGO == 2
                        scene = moved_f.y > 0.5f ? moved_f.x : (moved_b.y > 0.5f ? moved_b.x : base);
# else
                        if (moved_f.y > 0.5f && moved_b.y > 0.5f) scene = median3(moved_f.x, moved_b.x, base);
                        else scene = moved_f.y > 0.5f ? moved_f.x : (moved_b.y > 0.5f ? moved_b.x : base);
# endif
                        result = l_model.x*scene + layer_offset(&p, l_model);
                    }
                } else {
                    result = fast_blend_tap(source_f, source_b, &p, vector, fb_mask, pf, fb_chroma);
#if HAS_SAD
                    float4 m = layer_sample(masks, field_position, sad_idx);
                    result = mix(result, mix(ref_f0, ref_b0, steps.sad_blend[sad_idx]), fmax(m.w, m.x));
#endif
                }
                float value = blend_round(result, &p, layered, x, y);
#if HOLD
                if (weight < 1.0f) value = round(mix(held, value, weight));
#endif
                if (steps.gamma != 1.0f) value = curve[(int)value];
                total += wsum * value;
            }
            *sum = total;
            return;
        }
#endif
#if !(FAST_BLEND && !STILL)
#if ALGO == 23
        ext = field_sample(vectors_ext, field_position, CUBIC);
#endif
        for (int step = 0; step < steps.count; step++) {
            p.phase = steps.phase[step];
            p.sad_blend = steps.sad_blend[step];
            float time = native_divide((float)p.phase, 256.0f);
            float result;
            int layered = gap <= layer_reach(&p, step_f*time, step_b*(1.0f-time));
            if (layered) {
                if (l_opaque) {
                    result = mix(l_own_f, l_own_b, time);
                } else {
                    float2 at_f = clamp(at + step_f*time, (float2)(0.0f, 0.0f), top);
                    float2 at_b = clamp(at + step_b*(1.0f-time), (float2)(0.0f, 0.0f), top);
                    float2 moved_f = layer_scene(raw_f, layer, &p, at_f, 1);
                    float2 moved_b = layer_scene(raw_b, layer, &p, at_b, 1);
                    layer_agree(raw_f, raw_b, layer, &p, at_f, at_b, &moved_f, &moved_b);
                    float base = mix(l_base_f, l_base_b, time);
                    float scene;
# if ALGO == 1
                    scene = moved_b.y > 0.5f ? moved_b.x : (moved_f.y > 0.5f ? moved_f.x : base);
# elif ALGO == 2
                    scene = moved_f.y > 0.5f ? moved_f.x : (moved_b.y > 0.5f ? moved_b.x : base);
# else
                    if (moved_f.y > 0.5f && moved_b.y > 0.5f)
                        scene = median3(moved_f.x, moved_b.x, base);
                    else scene = moved_f.y > 0.5f ? moved_f.x : (moved_b.y > 0.5f ? moved_b.x : base);
# endif
                    result = l_model.x*scene + layer_offset(&p, l_model);
                }
            } else
#if STILL
            if (kept) {
                result = mix(kept_f, kept_b, time);
            } else
#endif
            {
#if ALGO >= 21 || HAS_SAD
                float4 mask = layer_sample(masks, field_position, step);
#else
                float4 mask = (float4)(0.0f);
#endif
                result = blend_value(
                    source_f, source_b, &p, vector, mask, ext, ref_f0, ref_b0, time);
            }
            float value = blend_round(result, &p, layered, x, y);
#if HOLD
            if (weight < 1.0f) value = round(mix(held, value, weight));
#endif
            {
#pragma OPENCL FP_CONTRACT OFF
                if (steps.gamma != 1.0f) value = curve[(int)value];
                float term = steps.weight[step] * value;
                total = term + total;
            }
        }
        *sum = total;
        return;
#endif
    }
#endif

#if FAST_BLEND
    if (!is_kept) {
    int fb_chroma = p.x_ratio > 1 || p.y_ratio > 1;
    int fb_cap = (FAST_BLEND >= 2 || fb_chroma) ? 1 : 2;
    float fb_travel = fmax(fmax(fabs(zsf.x), fabs(zsf.y)), fmax(fabs(zsb.x), fabs(zsb.y)));
    int fb_want = fb_travel < 0.5f ? 1 : fb_cap;
    int fb_buckets = min(fb_want, steps.count);
    float4 fb_mask = (float4)(0.0f);
#if ALGO >= 21
    float2 fb_dv = zsf + zsb;
    float fb_occ = clamp(native_sqrt(fb_dv.x * fb_dv.x + fb_dv.y * fb_dv.y) * 0.25f, 0.0f, 1.0f);
    fb_mask = (float4)(0.0f, fb_occ, fb_occ, 0.0f);
#endif
    for (int b = 0; b < fb_buckets; b++) {
        float wsum;
        float pf;
        int sad_idx;
        float j = ign_noise(x + b * 7, y + b * 13);
        if (steps.count == 2) {
            float p0 = (float)steps.phase[0];
            float p1 = (float)steps.phase[1];
            float dp = (p1 - p0) * 0.5f;
            float lo = fb_buckets == 1 ? clamp(p0 - dp, 0.0f, 256.0f)
                : clamp((b == 0 ? p0 : p1) - dp, 0.0f, 256.0f);
            float hi = fb_buckets == 1 ? clamp(p1 + dp, 0.0f, 256.0f)
                : clamp((b == 0 ? p0 : p1) + dp, 0.0f, 256.0f);
            wsum = fb_buckets == 1 ? (steps.weight[0] + steps.weight[1]) : steps.weight[b];
            pf = mix(lo, hi, j);
            sad_idx = b;
        } else {
            int s0 = (b * steps.count) / fb_buckets;
            int s1 = ((b + 1) * steps.count) / fb_buckets;
            sad_idx = (s0 + s1 - 1) >> 1;
            wsum = 0.0f;
            for (int s = s0; s < s1; s++) wsum += steps.weight[s];
            pf = s1 - s0 > 1
                ? mix((float)steps.phase[s0], (float)steps.phase[s1 - 1], j)
                : (float)steps.phase[s0];
        }
        float result = fast_blend_tap(source_f, source_b, &p, vector, fb_mask, pf, fb_chroma);
#if HAS_SAD
        float4 m = layer_sample(masks, field_position, sad_idx);
        result = mix(result, mix(ref_f0, ref_b0, steps.sad_blend[sad_idx]), fmax(m.w, m.x));
#endif
        float value = blend_round(result, &p, 0, x, y);
#if HOLD
        if (weight < 1.0f) value = round(mix(held, value, weight));
#endif
        if (steps.gamma != 1.0f) value = curve[(int)value];
        total += wsum * value;
    }
    *sum = total;
    return;
    }
#endif

#if !(FAST_BLEND && !STILL)
#if ALGO == 23
    ext = field_sample(vectors_ext, field_position, CUBIC);
#endif

#if ADAPTIVE && !FAST_BLEND
    if (!is_kept) {
    float atravel = fmax(fmax(fabs(zsf.x), fabs(zsf.y)), fmax(fabs(zsb.x), fabs(zsb.y)));
    int awant = atravel < 0.25f ? 1 : (atravel < 1.5f ? 2 : (atravel < 4.0f ? 4 : (atravel < 8.0f ? 8 : steps.count)));
    int abuckets = min(awant, steps.count);
    float4 amask = (float4)(0.0f);
#if ALGO >= 21
    float2 adv = zsf + zsb;
    float aocc = clamp(native_sqrt(adv.x * adv.x + adv.y * adv.y) * 0.25f, 0.0f, 1.0f);
    amask = (float4)(0.0f, aocc, aocc, 0.0f);
#endif
    for (int b = 0; b < abuckets; b++) {
        int s0 = (b * steps.count) / abuckets;
        int s1 = ((b + 1) * steps.count) / abuckets;
        int mid = (s0 + s1 - 1) >> 1;
        float wsum = 0.0f;
        for (int s = s0; s < s1; s++) wsum += steps.weight[s];
        p.phase = steps.phase[mid];
        p.sad_blend = steps.sad_blend[mid];
        float time = native_divide((float)p.phase, 256.0f);
#if HAS_SAD
        float4 m = layer_sample(masks, field_position, mid);
        amask.x = m.x;
        amask.w = m.w;
#endif
        float result = blend_value(source_f, source_b, &p, vector, amask, ext, ref_f0, ref_b0, time);
        float value = blend_round(result, &p, 0, x, y);
#if HOLD
        if (weight < 1.0f) value = round(mix(held, value, weight));
#endif
        {
#pragma OPENCL FP_CONTRACT OFF
            if (steps.gamma != 1.0f) value = curve[(int)value];
            float term = wsum * value;
            total = term + total;
        }
    }
    *sum = total;
    return;
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
#else
            float4 mask = (float4)(0.0f);
#endif
            result = blend_value(
                source_f, source_b, &p, vector, mask, ext, ref_f0, ref_b0, time);
        }

        float value = blend_round(result, &p, 0, x, y);
#if HOLD
        if (weight < 1.0f) value = round(mix(held, value, weight));
#endif
        {
#pragma OPENCL FP_CONTRACT OFF
            if (steps.gamma != 1.0f) value = curve[(int)value];
            float term = steps.weight[step] * value;
            total = term + total;
        }
    }
    *sum = total;
#endif
}
#endif
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
    int overwrite = stride < 0;
    stride = abs(stride);
    int2 position = (int2)(get_global_id(0), get_global_id(1));
    float value = round(255.0f * read_imagef(source, nearest_sampler, position).x);
    if (gamma != 1.0f && position.y < luma) value = powr(value / 255.0f, gamma);
    float term = weight * value;
    global float *sum = sums + position.y*stride + position.x;
    *sum = term + (overwrite ? 0.0f : *sum);
}

kernel void blend_resolve_source(
    global float *sums, global uchar *destination, float divisor, float gamma, uint luma,
    read_only image2d_t source, float weight, int width)
{
    size_t i = get_global_id(0);
    float total;
    {
#pragma OPENCL FP_CONTRACT OFF
        float value = round(255.0f *
            read_imagef(source, nearest_sampler, (int2)(i % width, i / width)).x);
        float term = weight * value;
        total = term + sums[i];
    }
    float value = total / divisor;
    if (gamma != 1.0f && i < luma) value = 255.0f * powr(fmax(value, 0.0f), 1.0f / gamma);
    destination[i] = (uchar)clamp(rint(value), 0.0f, 255.0f);
    sums[i] = 0.0f;
}

kernel void blend_resolve(
    global float *sums, global uchar *destination, float divisor, float gamma, uint luma)
{
    size_t i = get_global_id(0);
    float value = sums[i] / divisor;
    if (gamma != 1.0f && i < luma) value = 255.0f * powr(fmax(value, 0.0f), 1.0f / gamma);
    destination[i] = (uchar)clamp(rint(value), 0.0f, 255.0f);
    sums[i] = 0.0f;
}

inline float2 layer_model(const float *t, float n, float limit, int strict) {
    float mp = t[0]/n, mi = t[1]/n;
    float vp = t[2]/n - mp*mp, vi = t[4]/n - mi*mi, cov = t[3]/n - mp*mi;
    float a = cov/(vp+4.0f), b = mi - a*mp;
    float same = vi + vp - 2.0f*cov + (mi-mp)*(mi-mp);
    float left = fmax(vi - 2.0f*a*cov + a*a*vp, 0.0f);
    float g = clamp((1.0f - left/fmax(same, 1e-3f) - 0.5f)/0.2f, 0.0f, 1.0f)
        * clamp((same-limit)/limit, 0.0f, 1.0f);
    if (strict) g *= clamp((limit*0.5f - left)/(limit*0.25f), 0.0f, 1.0f);
    return (float2)(1.0f + g*(a-1.0f), g*b);
}

inline float layer_error(const float *t, float n, float2 m) {
    return (t[4] - 2.0f*m.x*t[3] - 2.0f*m.y*t[1] + m.x*m.x*t[2] + 2.0f*m.x*m.y*t[0])/n
        + m.y*m.y;
}

inline void layer_terms(
    float *v, read_only image2d_t first, read_only image2d_t second,
    read_only image2d_t vectors, const Params *p, int channels,
    float2 at, float2 top, float2 field_position, float ia, float ib)
{
    float4 vector = p->cubic ? cubic_sample(vectors, field_position)
        : read_imagef(vectors, field_sampler, field_position);
    float2 step_f = full_step(p, vector.x, vector.w);
    float2 step_b = full_step(p, vector.z, vector.y);
    float2 zero = (float2)(0.0f, 0.0f);
    float pb = layer_raw(first, p, clamp(at+step_f, zero, top));
    float pa = layer_raw(second, p, clamp(at+step_b, zero, top));
    v[0] = pb + pa;
    v[1] = ib + ia;
    if (channels == 6) {
        v[2] = pb*pb + pa*pa;
        v[3] = pb*ib + pa*ia;
        v[4] = ib*ib + ia*ia;
        v[5] = (fabs(pb-ib) > LAYER_APART) + (fabs(pa-ia) > LAYER_APART);
    }
}

kernel void layer_stats(
    global float *sums,
    read_only image2d_t f0, read_only image2d_t f1, read_only image2d_t f2,
    read_only image2d_t f3, read_only image2d_t f4, read_only image2d_t f5,
    read_only image2d_t f6,
    read_only image2d_t m0, read_only image2d_t m1, read_only image2d_t m2,
    read_only image2d_t m3, read_only image2d_t m4, read_only image2d_t m5,
    Params p, int base, int channels, int reset, int pairs)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= p.width || y >= p.height) return;
    int plane_n = p.width * p.height;
    global float *sum = sums + base + y * p.width + x;
    float2 at = (float2)(x, y), top = (float2)(p.width-1, p.height-1);
    float2 field_position = native_divide(
        (float2)(x*p.x_ratio-p.origin_x, y*p.y_ratio-p.origin_y),
        (float2)(p.block_w, p.block_h));
    float v[6], total[6];
    float ia = layer_raw(f0, &p, at);
    float ib = layer_raw(f1, &p, at);
    layer_terms(v, f0, f1, m0, &p, channels, at, top, field_position, ia, ib);
    for (int c = 0; c < channels; c++) total[c] = reset ? v[c] : sum[c * plane_n] + v[c];
    if (pairs > 1) {
        ia = ib;
        ib = layer_raw(f2, &p, at);
        layer_terms(v, f1, f2, m1, &p, channels, at, top, field_position, ia, ib);
        for (int c = 0; c < channels; c++) total[c] += v[c];
    }
    if (pairs > 2) {
        ia = ib;
        ib = layer_raw(f3, &p, at);
        layer_terms(v, f2, f3, m2, &p, channels, at, top, field_position, ia, ib);
        for (int c = 0; c < channels; c++) total[c] += v[c];
    }
    if (pairs > 3) {
        ia = ib;
        ib = layer_raw(f4, &p, at);
        layer_terms(v, f3, f4, m3, &p, channels, at, top, field_position, ia, ib);
        for (int c = 0; c < channels; c++) total[c] += v[c];
    }
    if (pairs > 4) {
        ia = ib;
        ib = layer_raw(f5, &p, at);
        layer_terms(v, f4, f5, m4, &p, channels, at, top, field_position, ia, ib);
        for (int c = 0; c < channels; c++) total[c] += v[c];
    }
    if (pairs > 5) {
        ia = ib;
        ib = layer_raw(f6, &p, at);
        layer_terms(v, f5, f6, m5, &p, channels, at, top, field_position, ia, ib);
        for (int c = 0; c < channels; c++) total[c] += v[c];
    }
    for (int c = 0; c < channels; c++) sum[c * plane_n] = total[c];
}

inline float2 layer_pick(
    global const float **parts, const int *counts, int idx, int plane_n, const float *own, int n_cur,
    float limit, int strict, float proof, int *from, int *to)
{
    float2 model = (float2)(1.0f, 0.0f);
    float error = 0.0f, seen = 0.0f;
    for (int window = 0; window < 3; window++) {
        int lo = window == 2 ? 1 : 0, hi = window == 1 ? 1 : 2;
        float t[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float n = 0.0f;
        for (int part = lo; part <= hi; part++) {
            if (!counts[part]) continue;
            for (int c = 0; c < 6; c++) t[c] += parts[part][c * plane_n + idx];
            n += counts[part];
        }
        float2 m = layer_model(t, n, limit, strict);
        float e = layer_error(own, n_cur, m);
        if (window == 0 || e*1.15f + 2.0f < error) {
            model = m;
            error = e;
            seen = t[5];
            *from = lo;
            *to = hi;
        }
    }
    float same = (own[4] - 2.0f*own[3] + own[2])/n_cur;
    return error < 1.7f*same + 4.0f && seen >= proof ? model : (float2)(1.0f, 0.0f);
}

kernel void layer_fit(
    global const float *prev, global const float *cur, global const float *next,
    int n_prev, int n_cur, int n_next, write_only image2d_t fit, int width, int height,
    float limit, int chroma_w, int chroma_h, int strict, float proof)
{
    int x = get_global_id(0), y = get_global_id(1);
    int luma_n = width * height;
    int idx = y * width + x;
    float seen_total = (n_prev ? prev[5 * luma_n + idx] : 0.0f)
        + cur[5 * luma_n + idx]
        + (n_next ? next[5 * luma_n + idx] : 0.0f);
    if (seen_total < proof) {
        write_imagef(fit, (int2)(x, y), (float4)(1.0f, 0.0f, 0.0f, 0.0f));
        return;
    }
    int chroma_n = chroma_w * chroma_h;
    int u_base = luma_n * 6;
    int v_base = u_base + chroma_n * 2;
    int c_idx = (y / (height / chroma_h)) * chroma_w + x / (width / chroma_w);
    global const float *parts[3] = {prev, cur, next};
    int counts[3] = {n_prev, n_cur, n_next};
    float own[6];
    for (int c = 0; c < 6; c++) own[c] = cur[c * luma_n + idx];
    int from = 1, to = 1;
    float2 model = layer_pick(parts, counts, idx, luma_n, own, n_cur, limit*limit, strict, proof, &from, &to);
    int near = 0;
    for (int part = 0; part < 3 && strict; part += 2) {
        if (!counts[part]) continue;
        float t[6];
        for (int c = 0; c < 6; c++) t[c] = parts[part][c * luma_n + idx];
        float2 m = layer_model(t, counts[part], limit*limit, 1);
        near |= layer_set((float4)(m.x, m.y, 0.0f, 0.0f));
    }
    if (model.x == 1.0f && model.y == 0.0f && near)
        model = layer_pick(parts, counts, idx, luma_n, own, n_cur, limit*limit, 0, proof, &from, &to);
    float2 offset = (float2)(0.0f, 0.0f);
    if (model.x != 1.0f || model.y != 0.0f) {
        float t[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float n = 0.0f;
        for (int part = from; part <= to; part++) {
            if (!counts[part]) continue;
            for (int c = 0; c < 2; c++) {
                t[c] += parts[part][u_base + c * chroma_n + c_idx];
                t[2+c] += parts[part][v_base + c * chroma_n + c_idx];
            }
            n += counts[part];
        }
        offset = (float2)(t[1] - model.x*t[0], t[3] - model.x*t[2]) / n;
    }
    write_imagef(fit, (int2)(x, y), (float4)(model.x, model.y, offset.x, offset.y));
}

kernel void layer_clean(read_only image2d_t fit, write_only image2d_t clean)
{
    int2 at = (int2)(get_global_id(0), get_global_id(1));
    float4 model = read_imagef(fit, cell_sampler, at);
    if (layer_set(model)) {
        int near = 0;
        for (int y = -1; y <= 1; y++)
            for (int x = -1; x <= 1; x++)
                near += layer_set(read_imagef(fit, cell_sampler, at + (int2)(x, y)));
        if (near < LAYER_GROUP) model = (float4)(1.0f, 0.0f, 0.0f, 0.0f);
    }
    write_imagef(clean, at, model);
}

kernel void layer_cover(read_only image2d_t fit, global uchar *cover, int width, int height)
{
    int cx = get_global_id(0), cy = get_global_id(1);
    int found = 0;
    for (int y = cy*8; y < min(cy*8+8, height) && !found; y++) {
        for (int x = cx*8; x < min(cx*8+8, width); x++) {
            float4 m = read_imagef(fit, nearest_sampler, (int2)(x, y));
            if (layer_set(m)) {
                found = 1;
                break;
            }
        }
    }
    cover[cy*((width+7)/8) + cx] = found;
}

kernel void copy_planes_to_packed(
    global const uchar *src_y, int y_pitch,
    global const uchar *src_u, global const uchar *src_v, int uv_pitch,
    write_only image2d_t packed,
    int w, int h, int cw, int ch)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x < w && y < h) {
        float v = (float)src_y[y * y_pitch + x] / 255.0f;
        write_imagef(packed, (int2)(x, y), (float4)(v, 0.0f, 0.0f, 0.0f));
    }
    if (x < cw && y < ch) {
        float vu = (float)src_u[y * uv_pitch + x] / 255.0f;
        float vv = (float)src_v[y * uv_pitch + x] / 255.0f;
        write_imagef(packed, (int2)(x, h + y), (float4)(vu, 0.0f, 0.0f, 0.0f));
        write_imagef(packed, (int2)(cw + x, h + y), (float4)(vv, 0.0f, 0.0f, 0.0f));
    }
}

kernel void cover_clear(global int *points, int count)
{
    int i = get_global_id(0);
    if (i < count) points[i] = 0;
}

kernel void cover_scatter(global const short2 *vectors, global int *points, int grid_w, int grid_h,
    int width, int height, int step_x, int step_y, int block_w, int block_h, int denom,
    int time_backward, int time_forward)
{
    int bx = get_global_id(0), by = get_global_id(1), set = get_global_id(2);
    short2 v = vectors[(set * grid_h + by) * grid_w + bx];
    int time = set ? time_forward : time_backward;
    int tx = time * v.x / denom, ty = time * v.y / denom;
    int qx = tx < 0 ? (tx - step_x + 1) / step_x : tx / step_x;
    int qy = ty < 0 ? (ty - step_y + 1) / step_y : ty / step_y;
    int cx = bx + qx, cy = by + qy;
    int rx = step_x * (qx + 1) - tx, ry = step_y * (qy + 1) - ty;
    int lx = block_w - rx, ly = block_h - ry;
    int stride = width + 2;
    global int *grid = points + set * stride * (height + 2);
    int2 at[4] = {(int2)(cx, cy), (int2)(cx + 1, cy), (int2)(cx + 1, cy + 1), (int2)(cx, cy + 1)};
    int value[4] = {ry * rx, ry * lx, ly * lx, rx * ly};
    for (int i = 0; i < 4; i++) {
        if ((uint)at[i].x < (uint)width && (uint)at[i].y < (uint)height)
            atomic_add(&grid[(at[i].y + 1) * stride + at[i].x + 1], value[i]);
    }
}

kernel void cover_finish(global const int *points, global const uchar *table, write_only image2d_t mask,
    int width, int height, int area, int top, float scale)
{
    int x = get_global_id(0), y = get_global_id(1);
    int stride = width + 2;
    float level[2];
    for (int set = 0; set < 2; set++) {
        global const int *grid = points + set * stride * (height + 2);
        int sum = 0;
        for (int row = 0; row < 3; row++)
            for (int column = 0; column < 3; column++)
                sum += grid[(y + row) * stride + x + column];
        int covered = min(sum >> 3, area);
        level[set] = covered < 0
            ? floor(fmin(255.0f, (float)(area - covered) * scale * 256.0f / (float)area))
            : (float)table[min(covered, top)];
    }
    write_imagef(mask, (int2)(x, y), (float4)(0.0f, level[1] / 255.0f, level[0] / 255.0f, 0.0f));
}

#if defined(__OPENCL_C_VERSION__) && __OPENCL_C_VERSION__ >= 120
kernel void cover_finish_layer(global const int *points, global const uchar *table,
    write_only image2d_array_t mask, int width, int height, int area, int top, float scale, int layer)
{
    int x = get_global_id(0), y = get_global_id(1);
    int stride = width + 2;
    float level[2];
    for (int set = 0; set < 2; set++) {
        global const int *grid = points + set * stride * (height + 2);
        int sum = 0;
        for (int row = 0; row < 3; row++)
            for (int column = 0; column < 3; column++)
                sum += grid[(y + row) * stride + x + column];
        int covered = min(sum >> 3, area);
        level[set] = covered < 0
            ? floor(fmin(255.0f, (float)(area - covered) * scale * 256.0f / (float)area))
            : (float)table[min(covered, top)];
    }
    write_imagef(mask, (int4)(x, y, layer, 0), (float4)(0.0f, level[1] / 255.0f, level[0] / 255.0f, 0.0f));
}
#endif
";

unsafe fn device_size(cl: &OpenCl, device: ClDeviceId, param: ClUint) -> Option<usize> {
    let mut size = 0u64;
    let status = unsafe {
        (cl.GetDeviceInfo)(
            device,
            param,
            size_of::<u64>(),
            (&raw mut size).cast(),
            std::ptr::null_mut(),
        )
    };
    usize::try_from(size)
        .ok()
        .filter(|&size| status == CL_SUCCESS && size > 0)
}

pub(crate) fn interleave4_u16(
    dst_vec: &mut Vec<u16>,
    count: usize,
    c0: &[u16],
    c1: &[u16],
    c2: &[u16],
    c3: &[u16],
) {
    dst_vec.resize(count * 4, 0);
    let dst = &mut dst_vec[..count * 4];
    #[cfg(target_arch = "x86_64")]
    let done = if std::arch::is_x86_feature_detected!("avx2") {
        unsafe { interleave4_u16_avx2(dst, count, c0, c1, c2, c3) }
    } else {
        0
    };
    #[cfg(target_arch = "aarch64")]
    let done = unsafe { interleave4_u16_neon(dst, count, c0, c1, c2, c3) };
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let done = 0usize;
    for idx in done..count {
        let off = idx * 4;
        dst[off] = c0[idx];
        dst[off + 1] = c1[idx];
        dst[off + 2] = c2[idx];
        dst[off + 3] = c3[idx];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn interleave4_u16_avx2(
    dst: &mut [u16],
    count: usize,
    c0: &[u16],
    c1: &[u16],
    c2: &[u16],
    c3: &[u16],
) -> usize {
    use std::arch::x86_64::{
        _mm256_loadu_si256, _mm256_permute2x128_si256, _mm256_storeu_si256, _mm256_unpackhi_epi16,
        _mm256_unpackhi_epi32, _mm256_unpacklo_epi16, _mm256_unpacklo_epi32,
    };
    let mut idx = 0;
    while idx + 16 <= count {
        unsafe {
            let va = _mm256_loadu_si256(c0.as_ptr().add(idx).cast());
            let vb = _mm256_loadu_si256(c1.as_ptr().add(idx).cast());
            let vc = _mm256_loadu_si256(c2.as_ptr().add(idx).cast());
            let vd = _mm256_loadu_si256(c3.as_ptr().add(idx).cast());
            let ab_lo = _mm256_unpacklo_epi16(va, vb);
            let ab_hi = _mm256_unpackhi_epi16(va, vb);
            let cd_lo = _mm256_unpacklo_epi16(vc, vd);
            let cd_hi = _mm256_unpackhi_epi16(vc, vd);
            let q0 = _mm256_unpacklo_epi32(ab_lo, cd_lo);
            let q1 = _mm256_unpackhi_epi32(ab_lo, cd_lo);
            let q2 = _mm256_unpacklo_epi32(ab_hi, cd_hi);
            let q3 = _mm256_unpackhi_epi32(ab_hi, cd_hi);
            let out = dst.as_mut_ptr().add(idx * 4);
            _mm256_storeu_si256(out.cast(), _mm256_permute2x128_si256::<0x20>(q0, q1));
            _mm256_storeu_si256(out.add(16).cast(), _mm256_permute2x128_si256::<0x20>(q2, q3));
            _mm256_storeu_si256(out.add(32).cast(), _mm256_permute2x128_si256::<0x31>(q0, q1));
            _mm256_storeu_si256(out.add(48).cast(), _mm256_permute2x128_si256::<0x31>(q2, q3));
        }
        idx += 16;
    }
    idx
}

#[cfg(target_arch = "aarch64")]
unsafe fn interleave4_u16_neon(
    dst: &mut [u16],
    count: usize,
    c0: &[u16],
    c1: &[u16],
    c2: &[u16],
    c3: &[u16],
) -> usize {
    use std::arch::aarch64::{uint16x8x4_t, vld1q_u16, vst4q_u16};
    let mut idx = 0;
    while idx + 8 <= count {
        unsafe {
            let quad = uint16x8x4_t(
                vld1q_u16(c0.as_ptr().add(idx)),
                vld1q_u16(c1.as_ptr().add(idx)),
                vld1q_u16(c2.as_ptr().add(idx)),
                vld1q_u16(c3.as_ptr().add(idx)),
            );
            vst4q_u16(dst.as_mut_ptr().add(idx * 4), quad);
        }
        idx += 8;
    }
    idx
}

fn interleave4_u8(dst: &mut [u8], count: usize, c0: &[u8], c1: &[u8], c2: &[u8], c3: &[u8]) {
    #[cfg(target_arch = "x86_64")]
    let done = if std::arch::is_x86_feature_detected!("avx2") {
        unsafe { interleave4_u8_avx2(dst, count, c0, c1, c2, c3) }
    } else {
        0
    };
    #[cfg(target_arch = "aarch64")]
    let done = unsafe { interleave4_u8_neon(dst, count, c0, c1, c2, c3) };
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let done = 0usize;
    for idx in done..count {
        let off = idx * 4;
        dst[off] = c0[idx];
        dst[off + 1] = c1[idx];
        dst[off + 2] = c2[idx];
        dst[off + 3] = c3[idx];
    }
}

fn interleave4_u8_zero_outer(dst: &mut [u8], count: usize, c1: &[u8], c2: &[u8]) {
    #[cfg(target_arch = "x86_64")]
    let done = if std::arch::is_x86_feature_detected!("avx2") {
        unsafe { interleave4_u8_zero_outer_avx2(dst, count, c1, c2) }
    } else {
        0
    };
    #[cfg(target_arch = "aarch64")]
    let done = unsafe { interleave4_u8_zero_outer_neon(dst, count, c1, c2) };
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let done = 0usize;
    for idx in done..count {
        let off = idx * 4;
        dst[off] = 0;
        dst[off + 1] = c1[idx];
        dst[off + 2] = c2[idx];
        dst[off + 3] = 0;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn interleave4_u8_avx2(
    dst: &mut [u8],
    count: usize,
    c0: &[u8],
    c1: &[u8],
    c2: &[u8],
    c3: &[u8],
) -> usize {
    use std::arch::x86_64::{
        _mm256_loadu_si256, _mm256_permute2x128_si256, _mm256_storeu_si256, _mm256_unpackhi_epi8,
        _mm256_unpackhi_epi16, _mm256_unpacklo_epi8, _mm256_unpacklo_epi16,
    };
    let mut idx = 0;
    while idx + 32 <= count {
        unsafe {
            let va = _mm256_loadu_si256(c0.as_ptr().add(idx).cast());
            let vb = _mm256_loadu_si256(c1.as_ptr().add(idx).cast());
            let vc = _mm256_loadu_si256(c2.as_ptr().add(idx).cast());
            let vd = _mm256_loadu_si256(c3.as_ptr().add(idx).cast());
            let ab_lo = _mm256_unpacklo_epi8(va, vb);
            let ab_hi = _mm256_unpackhi_epi8(va, vb);
            let cd_lo = _mm256_unpacklo_epi8(vc, vd);
            let cd_hi = _mm256_unpackhi_epi8(vc, vd);
            let q0 = _mm256_unpacklo_epi16(ab_lo, cd_lo);
            let q1 = _mm256_unpackhi_epi16(ab_lo, cd_lo);
            let q2 = _mm256_unpacklo_epi16(ab_hi, cd_hi);
            let q3 = _mm256_unpackhi_epi16(ab_hi, cd_hi);
            let out = dst.as_mut_ptr().add(idx * 4);
            _mm256_storeu_si256(out.cast(), _mm256_permute2x128_si256::<0x20>(q0, q1));
            _mm256_storeu_si256(out.add(32).cast(), _mm256_permute2x128_si256::<0x20>(q2, q3));
            _mm256_storeu_si256(out.add(64).cast(), _mm256_permute2x128_si256::<0x31>(q0, q1));
            _mm256_storeu_si256(out.add(96).cast(), _mm256_permute2x128_si256::<0x31>(q2, q3));
        }
        idx += 32;
    }
    idx
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn interleave4_u8_zero_outer_avx2(
    dst: &mut [u8],
    count: usize,
    c1: &[u8],
    c2: &[u8],
) -> usize {
    use std::arch::x86_64::{
        _mm256_loadu_si256, _mm256_permute2x128_si256, _mm256_setzero_si256, _mm256_storeu_si256,
        _mm256_unpackhi_epi8, _mm256_unpackhi_epi16, _mm256_unpacklo_epi8, _mm256_unpacklo_epi16,
    };
    let z = _mm256_setzero_si256();
    let mut idx = 0;
    while idx + 32 <= count {
        unsafe {
            let vb = _mm256_loadu_si256(c1.as_ptr().add(idx).cast());
            let vc = _mm256_loadu_si256(c2.as_ptr().add(idx).cast());
            let ab_lo = _mm256_unpacklo_epi8(z, vb);
            let ab_hi = _mm256_unpackhi_epi8(z, vb);
            let cd_lo = _mm256_unpacklo_epi8(vc, z);
            let cd_hi = _mm256_unpackhi_epi8(vc, z);
            let q0 = _mm256_unpacklo_epi16(ab_lo, cd_lo);
            let q1 = _mm256_unpackhi_epi16(ab_lo, cd_lo);
            let q2 = _mm256_unpacklo_epi16(ab_hi, cd_hi);
            let q3 = _mm256_unpackhi_epi16(ab_hi, cd_hi);
            let out = dst.as_mut_ptr().add(idx * 4);
            _mm256_storeu_si256(out.cast(), _mm256_permute2x128_si256::<0x20>(q0, q1));
            _mm256_storeu_si256(out.add(32).cast(), _mm256_permute2x128_si256::<0x20>(q2, q3));
            _mm256_storeu_si256(out.add(64).cast(), _mm256_permute2x128_si256::<0x31>(q0, q1));
            _mm256_storeu_si256(out.add(96).cast(), _mm256_permute2x128_si256::<0x31>(q2, q3));
        }
        idx += 32;
    }
    idx
}

#[cfg(target_arch = "aarch64")]
unsafe fn interleave4_u8_neon(
    dst: &mut [u8],
    count: usize,
    c0: &[u8],
    c1: &[u8],
    c2: &[u8],
    c3: &[u8],
) -> usize {
    use std::arch::aarch64::{uint8x16x4_t, vld1q_u8, vst4q_u8};
    let mut idx = 0;
    while idx + 16 <= count {
        unsafe {
            let quad = uint8x16x4_t(
                vld1q_u8(c0.as_ptr().add(idx)),
                vld1q_u8(c1.as_ptr().add(idx)),
                vld1q_u8(c2.as_ptr().add(idx)),
                vld1q_u8(c3.as_ptr().add(idx)),
            );
            vst4q_u8(dst.as_mut_ptr().add(idx * 4), quad);
        }
        idx += 16;
    }
    idx
}

#[cfg(target_arch = "aarch64")]
unsafe fn interleave4_u8_zero_outer_neon(
    dst: &mut [u8],
    count: usize,
    c1: &[u8],
    c2: &[u8],
) -> usize {
    use std::arch::aarch64::{uint8x16x4_t, vdupq_n_u8, vld1q_u8, vst4q_u8};
    let z = unsafe { vdupq_n_u8(0) };
    let mut idx = 0;
    while idx + 16 <= count {
        unsafe {
            let quad = uint8x16x4_t(
                z,
                vld1q_u8(c1.as_ptr().add(idx)),
                vld1q_u8(c2.as_ptr().add(idx)),
                z,
            );
            vst4q_u8(dst.as_mut_ptr().add(idx * 4), quad);
        }
        idx += 16;
    }
    idx
}
