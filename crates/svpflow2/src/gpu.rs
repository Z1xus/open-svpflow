#![allow(unsafe_code)]
#![allow(clippy::missing_safety_doc)]
use core::ffi::{c_char, c_void};
use std::ffi::CString;

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
const CL_DEVICE_TYPE_GPU: u64 = 1 << 2;
const CL_DEVICE_NAME: ClUint = 0x102B;
const CL_MEM_READ_WRITE: u64 = 1 << 0;
const CL_FALSE: ClUint = 0;
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
    EnqueueReadBuffer: FnEnqueueReadBuffer,
    EnqueueWriteImage: FnEnqueueWriteImage,
    SetKernelArg: FnSetKernelArg,
    EnqueueNDRangeKernel: FnEnqueueNDRangeKernel,
    Finish: FnFinish,
    ReleaseMemObject: FnRelease,
    ReleaseKernel: FnRelease,
    ReleaseProgram: FnRelease,
    ReleaseCommandQueue: FnRelease,
    ReleaseContext: FnRelease,
}

impl OpenCl {
    unsafe fn load() -> Result<Self, String> {
        let lib = unsafe { libloading::Library::new("OpenCL.dll") }
            .or_else(|_| unsafe { libloading::Library::new("OpenCL") })
            .map_err(|e| format!("OpenCL.dll not loadable: {e}"))?;
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
                EnqueueReadBuffer: sym!("clEnqueueReadBuffer"),
                EnqueueWriteImage: sym!("clEnqueueWriteImage"),
                SetKernelArg: sym!("clSetKernelArg"),
                EnqueueNDRangeKernel: sym!("clEnqueueNDRangeKernel"),
                Finish: sym!("clFinish"),
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
    program: ClProgram,
    device_name: String,

    units: Vec<std::sync::Mutex<Unit>>,
    next_unit: std::sync::atomic::AtomicUsize,
    cache_resources: std::sync::Mutex<CacheResources>,

    super_cache: std::sync::Mutex<Vec<(i64, SuperCell)>>,
}

type SuperCell = std::sync::Arc<std::sync::OnceLock<Option<std::sync::Arc<SuperEntry>>>>;

struct CacheResources {
    kernel: ClKernel,
    queue: ClCommandQueue,
}

struct SuperEntry {
    mems: [ClMem; 4],
    release: FnRelease,
}

unsafe impl Send for SuperEntry {}
unsafe impl Sync for SuperEntry {}
impl Drop for SuperEntry {
    fn drop(&mut self) {
        for &m in &self.mems {
            if !m.is_null() {
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
            GpuBuf(self.entry.mems[0]),
            GpuBuf(self.entry.mems[1]),
            GpuBuf(self.entry.mems[2]),
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
pub struct GpuBuf(ClMem);

const SUPER_CACHE_CAP: usize = 8;

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
                    (self.cl.ReleaseCommandQueue)(u.queue);
                }
            }
            if let Ok(cache) = self.cache_resources.lock() {
                (self.cl.ReleaseKernel)(cache.kernel);
                (self.cl.ReleaseCommandQueue)(cache.queue);
            }
            (self.cl.ReleaseProgram)(self.program);
            (self.cl.ReleaseContext)(self.context);
        }
    }
}

impl GpuContext {
    pub fn new(gpuid: i32, qn: i64) -> Option<Self> {
        let units = usize::try_from(qn)
            .unwrap_or(DEFAULT_UNITS)
            .clamp(MIN_UNITS, MAX_UNITS);
        unsafe { Self::try_new(gpuid, units) }.ok()
    }

    unsafe fn try_new(gpuid: i32, num_units: usize) -> Result<Self, String> {
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

            let program = match unsafe { build_program(&cl, context, device, KERNEL_SRC) } {
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
            let device_name = unsafe { device_name(&cl, device) };
            return Ok(Self {
                cl,
                context,
                program,
                device_name,
                units,
                next_unit: std::sync::atomic::AtomicUsize::new(0),
                cache_resources: std::sync::Mutex::new(CacheResources {
                    kernel: linear_kernel,
                    queue: cache_queue,
                }),
                super_cache: std::sync::Mutex::new(Vec::new()),
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

#[repr(C)]
#[derive(Clone, Copy)]
pub struct KernelParams {
    pub algorithm: i32,
    pub width: i32,
    pub height: i32,
    pub x_ratio: i32,
    pub y_ratio: i32,
    pub pel: i32,
    pub block_w: i32,
    pub block_h: i32,
    pub origin_x: i32,
    pub origin_y: i32,
    pub phase: i32,
    pub has_sad: i32,
    pub linear_luma: i32,
}

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
        let y_source =
            unsafe { self.create_image(CL_R, CL_UNORM_INT8, planes[0].width, planes[0].height) }?;
        let y_linear =
            unsafe { self.create_image(CL_R, CL_FLOAT, planes[0].width, planes[0].height) }?;
        let u =
            unsafe { self.create_image(CL_R, CL_UNORM_INT8, planes[1].width, planes[1].height) }?;
        let v =
            unsafe { self.create_image(CL_R, CL_UNORM_INT8, planes[2].width, planes[2].height) }?;
        let entry = SuperEntry {
            mems: [y_linear.mem, u.mem, v.mem, y_source.mem],
            release: self.cl.ReleaseMemObject,
        };
        std::mem::forget(y_source);
        std::mem::forget(y_linear);
        std::mem::forget(u);
        std::mem::forget(v);
        let resources = self.cache_resources.lock().ok()?;
        unsafe {
            self.enqueue_image(resources.queue, entry.mems[3], planes[0])?;
            self.enqueue_image(resources.queue, entry.mems[1], planes[1])?;
            self.enqueue_image(resources.queue, entry.mems[2], planes[2])?;
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
            let global = [planes[0].width, planes[0].height];
            if (self.cl.EnqueueNDRangeKernel)(
                resources.queue,
                resources.kernel,
                2,
                std::ptr::null(),
                global.as_ptr(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            ) != CL_SUCCESS
                || (self.cl.Finish)(resources.queue) != CL_SUCCESS
            {
                return None;
            }
        }
        Some(entry)
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

    unsafe fn enqueue_image(
        &self,
        q: ClCommandQueue,
        mem: ClMem,
        plane: UploadPlane<'_>,
    ) -> Option<()> {
        if plane.data.len()
            < plane
                .stride
                .saturating_mul(plane.height.saturating_sub(1))
                .saturating_add(plane.width)
        {
            return None;
        }
        unsafe {
            self.enqueue_image_raw(
                q,
                mem,
                plane.width,
                plane.height,
                plane.stride,
                plane.data.as_ptr().cast(),
            )
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

    unsafe fn set_i32(&self, k: ClKernel, i: u32, v: i32) -> bool {
        unsafe { (self.cl.SetKernelArg)(k, i, 4, (&raw const v).cast()) == CL_SUCCESS }
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
    ) -> Option<()> {
        let count = motion_width.checked_mul(motion_height)?;
        let motion_count = if py.algorithm == 23 { 4 } else { 2 };
        if count == 0
            || motions[..motion_count]
                .iter()
                .any(|(x, y)| x.len() < count || y.len() < count)
        {
            return None;
        }
        let mut unit = self.acquire();
        let Unit {
            kernel,
            queue,
            dst,
            motion,
            motion_key: cached_motion_key,
            mask,
            packed_base,
            packed_ext,
            packed_mask,
        } = &mut *unit;
        let k = *kernel;
        let q = *queue;
        packed_base.clear();
        packed_ext.clear();
        packed_mask.clear();
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
            let ext_mem = if py.algorithm == 23 {
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

            let needs_coverage = py.algorithm >= 21;
            let needs_mask = needs_coverage || py.has_sad != 0;
            let mask_mem = if needs_mask {
                if needs_coverage && (coverage.0.len() < count || coverage.1.len() < count) {
                    return None;
                }
                if let Some((a, b)) = area
                    && (a.len() < count || b.len() < count)
                {
                    return None;
                }
                packed_mask.reserve(count * 4);
                for i in 0..count {
                    let sad_f = area.map_or(0, |(a, _)| a[i]);
                    let sad_b = area.map_or(0, |(_, b)| b[i]);
                    packed_mask.extend_from_slice(&[
                        sad_b,
                        if needs_coverage { coverage.0[i] } else { 0 },
                        if needs_coverage { coverage.1[i] } else { 0 },
                        sad_f,
                    ]);
                }
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

            let run = |dslot: &mut (ClMem, usize),
                       dst: &mut [u8],
                       stride: i32,
                       p: KernelParams,
                       s0: GpuBuf,
                       s1: GpuBuf|
             -> Option<()> {
                let dmem = self.ensure_buffer(dslot, dst.len())?;
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
                    0,
                    std::ptr::null(),
                    std::ptr::null_mut(),
                ) != CL_SUCCESS
                {
                    return None;
                }

                if (self.cl.EnqueueReadBuffer)(
                    q,
                    dmem,
                    CL_FALSE,
                    0,
                    dst.len(),
                    dst.as_mut_ptr().cast(),
                    0,
                    std::ptr::null(),
                    std::ptr::null_mut(),
                ) != CL_SUCCESS
                {
                    return None;
                }
                Some(())
            };
            run(&mut dst[0], dst_y, sy, py, src0[0], src1[0])?;
            run(&mut dst[1], dst_u, su, pu, src0[1], src1[1])?;
            run(&mut dst[2], dst_v, sv, pv, src0[2], src1[2])?;
            if (self.cl.Finish)(q) != CL_SUCCESS {
                return None;
            }
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
const sampler_t linear_sampler = CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
const sampler_t nearest_sampler = CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

typedef struct {
    int algorithm, width, height, x_ratio, y_ratio, pel;
    int block_w, block_h, origin_x, origin_y, phase, has_sad, linear_luma;
} Params;

inline float median3(float a, float b, float c) {
    float lo = fmin(a, b);
    return fmax(lo, fmin(a + b - lo, c));
}

inline float4 cubic_sample(read_only image2d_t image, float2 position) {
    float2 grid = position - (float2)(0.5f, 0.5f);
    float2 index = floor(grid);
    float2 f = grid - index;
    float2 r = 1.0f - f;
    float2 w0 = r*r*r / 6.0f;
    float2 w1 = 2.0f/3.0f - 0.5f*f*f*(2.0f-f);
    float2 w2 = 2.0f/3.0f - 0.5f*r*r*(2.0f-r);
    float2 w3 = f*f*f / 6.0f;
    float2 g0 = w0 + w1;
    float2 g1 = w2 + w3;
    float2 h0 = w1/g0 - 0.5f + index;
    float2 h1 = w3/g1 + 1.5f + index;
    float4 a = read_imagef(image, linear_sampler, h0);
    float4 b = read_imagef(image, linear_sampler, (float2)(h1.x, h0.y));
    float4 c = read_imagef(image, linear_sampler, (float2)(h0.x, h1.y));
    float4 d = read_imagef(image, linear_sampler, h1);
    return mix(mix(d, b, g0.y), mix(c, a, g0.y), g0.x);
}

inline float source_sample(
    read_only image2d_t source, const Params *p, float vx, float vy, int time)
{
    float2 displacement = (float2)(vx*65535.0f-1024.0f, vy*65535.0f-1024.0f);
    displacement = native_divide(
        displacement * (float)time,
        (float2)(p->x_ratio*p->pel, p->y_ratio*p->pel) * 256.0f);
    float2 max_pos = (float2)(p->width-1, p->height-1);
    float2 position = clamp(
        (float2)(get_global_id(0), get_global_id(1)) + displacement,
        (float2)(0.0f, 0.0f), max_pos);
    return 255.0f * read_imagef(source, linear_sampler, position + 0.5f).x;
}

inline float base_sample(read_only image2d_t source) {
    float2 position = (float2)(get_global_id(0), get_global_id(1)) + 0.5f;
    return 255.0f * read_imagef(source, linear_sampler, position).x;
}

kernel void linear_luma(read_only image2d_t source, write_only image2d_t destination) {
    int2 position = (int2)(get_global_id(0), get_global_id(1));
    float value = read_imagef(source, nearest_sampler, position).x;
    value = value < 0.081f
        ? native_divide(value, 4.5f)
        : native_powr(native_divide(value+0.099f, 1.099f), 1.0f/0.45f);
    write_imagef(destination, position, (float4)(value, 0.0f, 0.0f, 0.0f));
}

kernel void render_frame(
    global uchar *destination, int destination_stride,
    read_only image2d_t source_f, read_only image2d_t source_b,
    read_only image2d_t vectors, read_only image2d_t vectors_ext,
    read_only image2d_t masks, Params p)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= p.width || y >= p.height) return;

    float2 vector_position = (float2)(
        (float)(x*p.x_ratio-p.origin_x)/(float)p.block_w,
        (float)(y*p.y_ratio-p.origin_y)/(float)p.block_h);
    float4 vector = cubic_sample(vectors, vector_position);
    float ref_f = source_sample(source_f, &p, vector.x, vector.w, p.phase);
    float ref_b = source_sample(source_b, &p, vector.z, vector.y, 256-p.phase);
    float ref_f0 = base_sample(source_f);
    float ref_b0 = base_sample(source_b);
    float time = native_divide((float)p.phase, 256.0f);
    float result;
    float4 mask = (float4)(0.0f);

    if (p.algorithm >= 21 || p.has_sad)
        mask = cubic_sample(masks, vector_position);

    if (p.algorithm == 1) {
        result = ref_b;
    } else if (p.algorithm == 2) {
        result = ref_f;
    } else if (p.algorithm == 11) {
        result = mix(ref_f, ref_b, time);
    } else if (p.algorithm == 13) {
        result = median3(ref_f, ref_b, mix(ref_f0, ref_b0, time));
    } else if (p.algorithm == 21) {
        result = mix(mix(ref_f, ref_b, mask.y), mix(ref_b, ref_f, mask.z), time);
    } else if (p.algorithm == 22) {
        result = median3(
            mix(ref_f, ref_b, mask.y),
            mix(ref_b, ref_f, mask.z),
            mix(ref_f0, ref_b0, time));
    } else {
        float4 ext = cubic_sample(vectors_ext, vector_position);
        float ref_ff = source_sample(source_f, &p, ext.x, ext.w, p.phase);
        float ref_bb = source_sample(source_b, &p, ext.z, ext.y, 256-p.phase);
        result = mix(
            mix(ref_f, median3(ref_b, ref_bb, ref_f), mask.y),
            mix(ref_b, median3(ref_b, ref_ff, ref_f), mask.z),
            time);
    }

    if (p.has_sad) {
        if (p.algorithm == 1)
            result = mix(result, ref_b0, mask.x);
        else if (p.algorithm == 2)
            result = mix(result, ref_f0, mask.w);
        else
            result = mix(result, mix(ref_f0, ref_b0, time), mix(mask.x, mask.w, time));
    }

    if (p.linear_luma) {
        float value = native_divide(result, 255.0f);
        value = value < 0.018f ? value*4.5f : 1.099f*native_powr(value, 0.45f)-0.099f;
        result = value*255.0f;
    }
    destination[y*destination_stride+x] = (uchar)clamp(round(result), 0.0f, 255.0f);
}
";
