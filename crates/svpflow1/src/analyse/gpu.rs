#![allow(unsafe_code)]
use core::ffi::{CStr, c_char, c_void};
use std::ffi::CString;
use std::sync::{Mutex, OnceLock};

use super::chain::Pyramid;
use super::field::recalc::{CompactPlan, RecalcPlan};
use super::field::{GpuSearchParams, Mv};

#[path = "gpu_chain.rs"]
mod chain;

type ClInt = i32;
type ClUint = u32;
type Handle = *mut c_void;

const CL_SUCCESS: ClInt = 0;
const CL_DEVICE_TYPE_GPU: u64 = 1 << 2;
const CL_MEM_READ_WRITE: u64 = 1;
const CL_MEM_ALLOC_HOST_PTR: u64 = 1 << 4;
const CL_MAP_READ: u64 = 1;
const CL_TRUE: ClUint = 1;
const CL_KERNEL_COMPILE_WORK_GROUP_SIZE: ClUint = 0x11B1;
const CHUNK: usize = 16384;
const RECORD: usize = 12;
const RECORD_BYTES: usize = RECORD * size_of::<i32>();
const CL_FALSE: ClUint = 0;
const CL_EVENT_COMMAND_EXECUTION_STATUS: ClUint = 0x11D3;
const UNITS: usize = 4;

type FnGetPlatformIDs = unsafe extern "C" fn(ClUint, *mut Handle, *mut ClUint) -> ClInt;
type FnGetDeviceIDs = unsafe extern "C" fn(Handle, u64, ClUint, *mut Handle, *mut ClUint) -> ClInt;
type FnCreateContext = unsafe extern "C" fn(
    *const isize,
    ClUint,
    *const Handle,
    *const c_void,
    *mut c_void,
    *mut ClInt,
) -> Handle;
type FnCreateCommandQueue = unsafe extern "C" fn(Handle, Handle, u64, *mut ClInt) -> Handle;
type FnCreateProgramWithSource =
    unsafe extern "C" fn(Handle, ClUint, *const *const c_char, *const usize, *mut ClInt) -> Handle;
type FnBuildProgram = unsafe extern "C" fn(
    Handle,
    ClUint,
    *const Handle,
    *const c_char,
    *const c_void,
    *mut c_void,
) -> ClInt;
type FnCreateKernel = unsafe extern "C" fn(Handle, *const c_char, *mut ClInt) -> Handle;
type FnCreateBuffer = unsafe extern "C" fn(Handle, u64, usize, *mut c_void, *mut ClInt) -> Handle;
type FnEnqueueBuffer = unsafe extern "C" fn(
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
type FnSetKernelArg = unsafe extern "C" fn(Handle, ClUint, usize, *const c_void) -> ClInt;
type FnEnqueueNDRangeKernel = unsafe extern "C" fn(
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
type FnEnqueueMapBuffer = unsafe extern "C" fn(
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
type FnRelease = unsafe extern "C" fn(Handle) -> ClInt;
type FnFinish = unsafe extern "C" fn(Handle) -> ClInt;
type FnWaitForEvents = unsafe extern "C" fn(ClUint, *const Handle) -> ClInt;
type FnGetEventInfo = unsafe extern "C" fn(Handle, ClUint, usize, *mut c_void, *mut usize) -> ClInt;
type FnKernelInfo =
    unsafe extern "C" fn(Handle, Handle, ClUint, usize, *mut c_void, *mut usize) -> ClInt;

#[allow(non_snake_case)]
struct OpenCl {
    _lib: libloading::Library,
    GetPlatformIDs: FnGetPlatformIDs,
    GetDeviceIDs: FnGetDeviceIDs,
    CreateContext: FnCreateContext,
    CreateCommandQueue: FnCreateCommandQueue,
    CreateProgramWithSource: FnCreateProgramWithSource,
    BuildProgram: FnBuildProgram,
    CreateKernel: FnCreateKernel,
    CreateBuffer: FnCreateBuffer,
    EnqueueReadBuffer: FnEnqueueBuffer,
    EnqueueWriteBuffer: FnEnqueueBuffer,
    EnqueueMapBuffer: FnEnqueueMapBuffer,
    SetKernelArg: FnSetKernelArg,
    EnqueueNDRangeKernel: FnEnqueueNDRangeKernel,
    ReleaseMemObject: FnRelease,
    ReleaseKernel: FnRelease,
    Finish: FnFinish,
    ReleaseEvent: FnRelease,
    WaitForEvents: FnWaitForEvents,
    GetEventInfo: FnGetEventInfo,
    GetKernelWorkGroupInfo: FnKernelInfo,
    Flush: FnFinish,
}

impl OpenCl {
    unsafe fn load() -> Option<Self> {
        let name = if cfg!(target_os = "windows") {
            "OpenCL.dll"
        } else if cfg!(target_os = "macos") {
            "/System/Library/Frameworks/OpenCL.framework/OpenCL"
        } else {
            "libOpenCL.so.1"
        };
        let lib = unsafe { libloading::Library::new(name) }.ok()?;
        macro_rules! sym {
            ($name:literal) => {
                *unsafe { lib.get(concat!($name, "\0").as_bytes()).ok()? }
            };
        }
        Some(Self {
            GetPlatformIDs: sym!("clGetPlatformIDs"),
            GetDeviceIDs: sym!("clGetDeviceIDs"),
            CreateContext: sym!("clCreateContext"),
            CreateCommandQueue: sym!("clCreateCommandQueue"),
            CreateProgramWithSource: sym!("clCreateProgramWithSource"),
            BuildProgram: sym!("clBuildProgram"),
            CreateKernel: sym!("clCreateKernel"),
            CreateBuffer: sym!("clCreateBuffer"),
            EnqueueReadBuffer: sym!("clEnqueueReadBuffer"),
            EnqueueWriteBuffer: sym!("clEnqueueWriteBuffer"),
            EnqueueMapBuffer: sym!("clEnqueueMapBuffer"),
            SetKernelArg: sym!("clSetKernelArg"),
            EnqueueNDRangeKernel: sym!("clEnqueueNDRangeKernel"),
            ReleaseMemObject: sym!("clReleaseMemObject"),
            ReleaseKernel: sym!("clReleaseKernel"),
            Finish: sym!("clFinish"),
            ReleaseEvent: sym!("clReleaseEvent"),
            WaitForEvents: sym!("clWaitForEvents"),
            GetEventInfo: sym!("clGetEventInfo"),
            GetKernelWorkGroupInfo: sym!("clGetKernelWorkGroupInfo"),
            Flush: sym!("clFlush"),
            _lib: lib,
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct Variant {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) satd: bool,
    pub(crate) shift: u32,
    pub(crate) dev: bool,
}

impl Variant {
    fn new(width: usize, height: usize, satd: bool) -> Self {
        Self {
            width,
            height,
            satd,
            shift: 1,
            dev: false,
        }
    }
}

#[derive(Clone, Copy)]
enum Arg {
    Mem(Handle),
    Int(i32),
    Long(i64),
    V4([i32; 4]),
    V8([i32; 8]),
}

#[derive(Default)]
struct Buffer {
    mem: usize,
    capacity: usize,
    host: usize,
    tag: u64,
}

type SearchKey = (Variant, usize, [i32; 5]);

struct SearchMems {
    src: [Handle; 3],
    reference: [Handle; 3],
    q: Handle,
    q2: Handle,
    pred: Handle,
    rev: Handle,
    res: [Handle; 2],
    aux: Handle,
    cen: Handle,
    out: Handle,
    dirty: [Handle; 2],
    dirs: usize,
    strides: [i32; 8],
}

#[derive(Default)]
struct Unit {
    queue: usize,
    kernels: Vec<(usize, &'static CStr, usize, usize)>,
    buffers: Vec<Buffer>,
    chain_stale: bool,
}

pub(crate) struct RefineGpu {
    cl: OpenCl,
    context: usize,
    device: usize,
    programs: Mutex<Vec<(Variant, usize)>>,
    search_programs: Mutex<Vec<(SearchKey, usize)>>,
    units: Vec<Mutex<Unit>>,
    cache: Mutex<Cache>,
    expand: OnceLock<Option<usize>>,
    chain: OnceLock<Option<usize>>,
    pool: Mutex<Vec<(usize, usize)>>,
    admission: Mutex<Admission>,
    admitted: std::sync::Condvar,
}

struct Admission {
    free: usize,
    waiting: Vec<(i32, u64)>,
    ticket: u64,
}

struct Permit<'g>(&'g RefineGpu);

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        if let Ok(mut admission) = self.0.admission.lock() {
            admission.free += 1;
        }
        self.0.admitted.notify_all();
    }
}

const POOLED: usize = 3 * UNITS;

struct Uploaded {
    mems: [usize; 3],
    sizes: [usize; 3],
    frame: Frame,
    event: usize,
    pyramid: Option<Pyramid>,
    coarse: [usize; 3],
    host: Option<std::sync::Arc<[Vec<u8>; 3]>>,
}

#[derive(Default)]
struct Cache {
    recent: std::collections::VecDeque<std::sync::Arc<CachedFrame>>,
    live: Vec<std::sync::Weak<CachedFrame>>,
}

impl Cache {
    fn find(&self, key: (usize, i32)) -> Option<std::sync::Arc<CachedFrame>> {
        self.recent
            .iter()
            .find(|e| e.key == key)
            .cloned()
            .or_else(|| {
                self.live
                    .iter()
                    .filter_map(std::sync::Weak::upgrade)
                    .find(|e| e.key == key)
            })
    }

    fn remove(&mut self, key: (usize, i32)) {
        self.recent.retain(|e| e.key != key);
        self.live
            .retain(|e| e.upgrade().is_some_and(|e| e.key != key));
    }
}

struct CachedFrame {
    key: (usize, i32),
    uses: std::sync::atomic::AtomicU32,
    gpu: *const RefineGpu,
    state: Mutex<Option<Uploaded>>,
}

unsafe impl Send for CachedFrame {}
unsafe impl Sync for CachedFrame {}

impl Drop for CachedFrame {
    fn drop(&mut self) {
        let gpu = unsafe { &*self.gpu };
        if let Ok(Some(uploaded)) = self.state.get_mut().map(Option::take) {
            unsafe {
                (gpu.cl.WaitForEvents)(1, (&raw const uploaded.event).cast::<Handle>());
                (gpu.cl.ReleaseEvent)(uploaded.event as Handle);
            }
            let mut pool = gpu.pool.lock().ok();
            for (mem, size) in uploaded.mems.into_iter().zip(uploaded.sizes) {
                match pool.as_mut() {
                    Some(pool) if pool.len() < POOLED => pool.push((mem, size)),
                    _ => unsafe {
                        (gpu.cl.ReleaseMemObject)(mem as Handle);
                    },
                }
            }
        }
    }
}

pub(crate) struct Planes<'a> {
    pub(crate) data: [&'a [u8]; 3],
    pub(crate) pitch: [usize; 3],
    pub(crate) base: [usize; 3],
    pub(crate) height: [i32; 3],
    pub(crate) pel: i32,
    pub(crate) expand: bool,
    pub(crate) tail: [usize; 3],
    pub(crate) pyramid: Option<Pyramid>,
    pub(crate) uses: u32,
}

pub(crate) fn refine_gpu() -> Option<&'static RefineGpu> {
    static GPU: OnceLock<Option<RefineGpu>> = OnceLock::new();
    GPU.get_or_init(|| unsafe { RefineGpu::new() }).as_ref()
}

impl RefineGpu {
    unsafe fn new() -> Option<Self> {
        let cl = unsafe { OpenCl::load()? };
        let mut count = 0;
        if unsafe { (cl.GetPlatformIDs)(0, std::ptr::null_mut(), &raw mut count) } != CL_SUCCESS {
            return None;
        }
        let mut platforms = vec![std::ptr::null_mut(); count as usize];
        unsafe { (cl.GetPlatformIDs)(count, platforms.as_mut_ptr(), &raw mut count) };
        for platform in platforms {
            let mut device = std::ptr::null_mut();
            let mut found = 0;
            if unsafe {
                (cl.GetDeviceIDs)(
                    platform,
                    CL_DEVICE_TYPE_GPU,
                    1,
                    &raw mut device,
                    &raw mut found,
                )
            } != CL_SUCCESS
                || found == 0
            {
                continue;
            }
            let mut err = 0;
            let context = svpflow_host::context::shared(device, || {
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
                continue;
            }
            let mut units = Vec::with_capacity(UNITS);
            for _ in 0..UNITS {
                let queue = unsafe { (cl.CreateCommandQueue)(context, device, 0, &raw mut err) };
                if err == CL_SUCCESS && !queue.is_null() {
                    units.push(Mutex::new(Unit {
                        queue: queue as usize,
                        ..Unit::default()
                    }));
                }
            }
            if units.is_empty() {
                continue;
            }
            return Some(Self {
                cl,
                context: context as usize,
                device: device as usize,
                programs: Mutex::new(Vec::new()),
                search_programs: Mutex::new(Vec::new()),
                cache: Mutex::new(Cache::default()),
                expand: OnceLock::new(),
                chain: OnceLock::new(),
                pool: Mutex::new(Vec::new()),
                admission: Mutex::new(Admission {
                    free: units.len(),
                    waiting: Vec::new(),
                    ticket: 0,
                }),
                admitted: std::sync::Condvar::new(),
                units,
            });
        }
        None
    }

    fn expand_program(&self) -> Option<usize> {
        *self.expand.get_or_init(|| self.build(EXPAND))
    }

    fn program(&self, variant: Variant) -> Option<usize> {
        let mut programs = self.programs.lock().ok()?;
        if let Some(&(_, program)) = programs.iter().find(|(v, _)| *v == variant) {
            return Some(program);
        }
        let kernel = if variant.width == 4 { COMPACT } else { KERNEL };
        let program = self.build(&format!(
            "#define W {}\n#define H {}\n#define SATD {}\n#define S {}\n#define RECORD {RECORD}\n{}{kernel}",
            variant.width,
            variant.height,
            u8::from(variant.satd),
            variant.shift,
            if variant.dev { "#define DEV 1\n" } else { "" }
        ))?;
        programs.push((variant, program));
        Some(program)
    }

    fn search_program(&self, variant: Variant, rmax: usize, spec: [i32; 5]) -> Option<usize> {
        let mut programs = self.search_programs.lock().ok()?;
        if let Some(&(_, program)) = programs.iter().find(|(v, _)| *v == (variant, rmax, spec)) {
            return Some(program);
        }
        let build = |portable: &str| {
            self.build_with(
                &format!(
                    "#define W {}\n#define H {}\n#define RMAX {}\n{}#define PELV {}\n#define LOGAV {}\n#define DONTV {}\n#define BLURV {}\n#define BORDERV {}\n{portable}{SEARCH}",
                    variant.width,
                    variant.height,
                    rmax,
                    if variant.satd { "#define SATD 1\n" } else { "" },
                    spec[0],
                    spec[1],
                    spec[2],
                    spec[3],
                    spec[4]
                ),
                c"-cl-fp32-correctly-rounded-divide-sqrt",
            )
        };
        let program = build("")
            .or_else(|| build("#define NO_REDUX 1\n"))
            .or_else(|| build("#define NO_NV_ASM 1\n"))?;
        programs.push(((variant, rmax, spec), program));
        Some(program)
    }

    fn search_pair(
        &self,
        variant: Variant,
        rmax: usize,
        q: &[i32; 40],
        border: usize,
    ) -> Option<(usize, usize)> {
        let spec = [q[6], q[31], q[32], q[34], 0];
        let program = self.search_program(variant, rmax, spec)?;
        let border_program = if border > 0 {
            self.search_program(variant, rmax, [spec[0], spec[1], spec[2], spec[3], 1])?
        } else {
            program
        };
        Some((program, border_program))
    }

    fn build(&self, text: &str) -> Option<usize> {
        self.build_with(text, c"")
    }

    fn build_with(&self, text: &str, options: &core::ffi::CStr) -> Option<usize> {
        let source = CString::new(text).ok()?;
        let cl = &self.cl;
        let mut err = 0;
        let pointer = source.as_ptr();
        let program = unsafe {
            (cl.CreateProgramWithSource)(
                self.context as Handle,
                1,
                &raw const pointer,
                std::ptr::null(),
                &raw mut err,
            )
        };
        if err != CL_SUCCESS || program.is_null() {
            return None;
        }
        let device = self.device as Handle;
        let _build = svpflow_core::OPENCL_BUILD.lock();
        let built = unsafe {
            (cl.BuildProgram)(
                program,
                1,
                &raw const device,
                options.as_ptr(),
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        if built != CL_SUCCESS {
            return None;
        }
        Some(program as usize)
    }

    pub(crate) fn session<'p>(&self, order: i32) -> Session<'_, 'p> {
        Session {
            gpu: self,
            order,
            unit: Lease(None),
            keep: Vec::new(),
            deferred: [None, None],
            frames: [None, None],
            tickets: Vec::new(),
            pending: Vec::new(),
        }
    }

    fn admit(&self, order: i32) -> (std::sync::MutexGuard<'_, Unit>, Permit<'_>) {
        let mut admission = self
            .admission
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let ticket = (order, admission.ticket);
        admission.ticket += 1;
        admission.waiting.push(ticket);
        let mut admission = self
            .admitted
            .wait_while(admission, |a| {
                a.free == 0 || a.waiting.iter().min() != Some(&ticket)
            })
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        admission.waiting.retain(|t| *t != ticket);
        admission.free -= 1;
        drop(admission);
        self.admitted.notify_all();
        let permit = Permit(self);
        let unit = self
            .units
            .iter()
            .find_map(|unit| unit.try_lock().ok())
            .unwrap_or_else(|| {
                self.units[0]
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
            });
        (unit, permit)
    }

    unsafe fn buffer(&self, unit: &mut Unit, index: usize, size: usize) -> Option<Handle> {
        if unit.buffers.len() <= index {
            unit.buffers.resize_with(index + 1, Buffer::default);
        }
        let buffer = &mut unit.buffers[index];
        if buffer.capacity < size {
            if buffer.mem != 0 {
                unsafe { (self.cl.ReleaseMemObject)(buffer.mem as Handle) };
                buffer.mem = 0;
            }
            let capacity = size.next_multiple_of(4096);
            let mut err = 0;
            let mem = unsafe {
                (self.cl.CreateBuffer)(
                    self.context as Handle,
                    CL_MEM_READ_WRITE,
                    capacity,
                    std::ptr::null_mut(),
                    &raw mut err,
                )
            };
            if err != CL_SUCCESS || mem.is_null() {
                buffer.capacity = 0;
                return None;
            }
            *buffer = Buffer {
                mem: mem as usize,
                capacity,
                host: 0,
                tag: 0,
            };
        }
        Some(buffer.mem as Handle)
    }

    // pageable reads queue behind the other plugin's copies in a shared context
    unsafe fn pinned(&self, unit: &mut Unit, index: usize, size: usize) -> Option<*mut u8> {
        if unit.buffers.len() <= index {
            unit.buffers.resize_with(index + 1, Buffer::default);
        }
        if unit.buffers[index].capacity < size {
            let old = std::mem::take(&mut unit.buffers[index]);
            if old.mem != 0 {
                unsafe { (self.cl.ReleaseMemObject)(old.mem as Handle) };
            }
            let capacity = size.next_multiple_of(4096);
            let mut err = 0;
            let mem = unsafe {
                (self.cl.CreateBuffer)(
                    self.context as Handle,
                    CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                    capacity,
                    std::ptr::null_mut(),
                    &raw mut err,
                )
            };
            if err != CL_SUCCESS || mem.is_null() {
                return None;
            }
            let host = unsafe {
                (self.cl.EnqueueMapBuffer)(
                    unit.queue as Handle,
                    mem,
                    CL_TRUE,
                    CL_MAP_READ,
                    0,
                    capacity,
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
            unit.buffers[index] = Buffer {
                mem: mem as usize,
                capacity,
                host: host as usize,
                tag: 0,
            };
        }
        Some(unit.buffers[index].host as *mut u8)
    }
}

#[derive(Clone, Copy)]
struct Frame {
    lengths: [i64; 3],
    pitches: [i64; 3],
    bases: [i64; 3],
    heights: [i64; 3],
    pel: i64,
}

type SessionFrame = (std::sync::Arc<CachedFrame>, [usize; 3], Frame, usize);

impl<'p> Session<'_, 'p> {
    pub(crate) fn defer(&mut self, slot: usize, key: (usize, i32), planes: Planes<'p>) {
        self.deferred[slot] = Some((key, planes));
    }
}

struct Lease<'g>(Option<(std::sync::MutexGuard<'g, Unit>, Permit<'g>)>);

impl std::ops::Deref for Lease<'_> {
    type Target = Unit;

    fn deref(&self) -> &Unit {
        &self.0.as_ref().expect("unit held").0
    }
}

impl std::ops::DerefMut for Lease<'_> {
    fn deref_mut(&mut self) -> &mut Unit {
        &mut self.0.as_mut().expect("unit held").0
    }
}

pub(crate) struct Session<'g, 'p> {
    gpu: &'g RefineGpu,
    order: i32,
    unit: Lease<'g>,
    frames: [Option<SessionFrame>; 2],
    tickets: Vec<(usize, usize)>,
    pending: Vec<usize>,
    keep: Vec<Box<[u8]>>,
    deferred: [Option<((usize, i32), Planes<'p>)>; 2],
}

impl Session<'_, '_> {
    unsafe fn transfer(&mut self, mem: Handle, data: &[u8], event: *mut Handle) -> bool {
        unsafe { self.transfer_at(mem, 0, data, event) }
    }

    unsafe fn transfer_at(
        &mut self,
        mem: Handle,
        offset: usize,
        data: &[u8],
        event: *mut Handle,
    ) -> bool {
        let status = unsafe {
            (self.gpu.cl.EnqueueWriteBuffer)(
                self.unit.queue as Handle,
                mem,
                CL_FALSE,
                offset,
                data.len(),
                data.as_ptr().cast_mut().cast(),
                0,
                std::ptr::null(),
                event.cast(),
            )
        };
        status == CL_SUCCESS
    }

    unsafe fn ckernel(&mut self, program: usize, name: &'static CStr) -> Option<(Handle, usize)> {
        if let Some(&(.., kernel, local)) = self
            .unit
            .kernels
            .iter()
            .find(|k| k.0 == program && k.1 == name)
        {
            return Some((kernel as Handle, local));
        }
        let cl = &self.gpu.cl;
        let mut err = 0;
        let kernel = unsafe { (cl.CreateKernel)(program as Handle, name.as_ptr(), &raw mut err) };
        if err != CL_SUCCESS || kernel.is_null() {
            return None;
        }
        let mut size = [0usize; 3];
        if unsafe {
            (cl.GetKernelWorkGroupInfo)(
                kernel,
                self.gpu.device as Handle,
                CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
                size_of_val(&size),
                size.as_mut_ptr().cast(),
                std::ptr::null_mut(),
            )
        } != CL_SUCCESS
        {
            unsafe { (cl.ReleaseKernel)(kernel) };
            return None;
        }
        self.unit
            .kernels
            .push((program, name, kernel as usize, size[0]));
        Some((kernel, size[0]))
    }

    unsafe fn launch(
        &mut self,
        (program, name): (usize, &'static CStr),
        args: &[Arg],
        global: usize,
        local: Option<usize>,
        waits: &[Handle],
        event: *mut Handle,
    ) -> Option<()> {
        let (kernel, _) = unsafe { self.ckernel(program, name)? };
        let cl = &self.gpu.cl;
        let queue = self.unit.queue as Handle;
        let set = args.iter().enumerate().all(|(index, arg)| {
            let (size, value): (usize, *const c_void) = match arg {
                Arg::Mem(v) => (size_of::<Handle>(), (&raw const *v).cast()),
                Arg::Int(v) => (4, (&raw const *v).cast()),
                Arg::Long(v) => (8, (&raw const *v).cast()),
                Arg::V4(v) => (16, v.as_ptr().cast()),
                Arg::V8(v) => (32, v.as_ptr().cast()),
            };
            unsafe { (cl.SetKernelArg)(kernel, index as ClUint, size, value) == CL_SUCCESS }
        });
        let global = match local {
            Some(local) => global.next_multiple_of(local),
            None => global.max(1),
        };
        if !set
            || unsafe {
                (cl.EnqueueNDRangeKernel)(
                    queue,
                    kernel,
                    1,
                    std::ptr::null(),
                    &raw const global,
                    local.as_ref().map_or(std::ptr::null(), std::ptr::from_ref),
                    waits.len() as ClUint,
                    if waits.is_empty() {
                        std::ptr::null()
                    } else {
                        waits.as_ptr().cast()
                    },
                    event.cast(),
                )
            } != CL_SUCCESS
        {
            unsafe { (cl.Finish)(queue) };
            return None;
        }
        Some(())
    }

    // writes do not block, so copies live until the session ends
    fn hold<'h>(&mut self, data: &[u8]) -> &'h [u8] {
        self.hold_boxed(data.into())
    }

    fn hold_boxed<'h>(&mut self, copy: Box<[u8]>) -> &'h [u8] {
        let held = unsafe { std::slice::from_raw_parts(copy.as_ptr(), copy.len()) };
        self.keep.push(copy);
        held
    }

    unsafe fn write(&mut self, index: usize, data: &[u8]) -> Option<Handle> {
        let mem = unsafe { self.gpu.buffer(&mut self.unit, index, data.len().max(1))? };
        if !data.is_empty() && !unsafe { self.transfer(mem, data, std::ptr::null_mut()) } {
            return None;
        }
        Some(mem)
    }

    fn flush(&mut self) {
        if self.unit.0.is_none() {
            self.unit = Lease(Some(self.gpu.admit(self.order)));
        }
        for slot in 0..2 {
            if let Some((key, planes)) = self.deferred[slot].take() {
                self.upload(slot, key, &planes);
            }
        }
    }

    fn upload(&mut self, slot: usize, key: (usize, i32), planes: &Planes<'_>) -> bool {
        self.frames[slot] = None;
        let gpu = self.gpu;
        let (entry, guard) = {
            let Ok(mut cache) = gpu.cache.lock() else {
                return false;
            };
            if let Some(entry) = cache.find(key) {
                drop(cache);
                let Ok(state) = entry.state.lock() else {
                    return false;
                };
                let Some(uploaded) = state.as_ref() else {
                    return false;
                };
                let found = (uploaded.mems, uploaded.frame, uploaded.event);
                drop(state);
                self.frames[slot] = Some((entry, found.0, found.1, found.2));
                return true;
            }
            let entry = std::sync::Arc::new(CachedFrame {
                key,
                uses: std::sync::atomic::AtomicU32::new(planes.uses),
                gpu,
                state: Mutex::new(None),
            });
            let Ok(guard) = entry.state.lock() else {
                return false;
            };
            let guard: std::sync::MutexGuard<'static, Option<Uploaded>> =
                unsafe { std::mem::transmute(guard) };
            cache.live.retain(|e| e.strong_count() > 0);
            cache.live.push(std::sync::Arc::downgrade(&entry));
            cache.recent.push_back(entry.clone());
            if cache.recent.len() > 2 * UNITS {
                cache.recent.pop_front();
            }
            (entry, guard)
        };
        let mut guard = guard;
        let uploaded = unsafe { self.upload_planes(planes) };
        let uploaded = uploaded.and_then(|mut u| match planes.pyramid {
            Some(pyramid) if !unsafe { self.build_levels(&mut u, pyramid) } => {
                for m in u.mems {
                    unsafe { (self.gpu.cl.ReleaseMemObject)(m as Handle) };
                }
                if u.event != 0 {
                    unsafe { (self.gpu.cl.ReleaseEvent)(u.event as Handle) };
                }
                None
            }
            _ => Some(u),
        });
        let result = uploaded.as_ref().map(|u| (u.mems, u.frame, u.event));
        *guard = uploaded;
        drop(guard);
        if let Some((mems, frame, event)) = result {
            self.frames[slot] = Some((entry, mems, frame, event));
            return true;
        }
        if let Ok(mut cache) = gpu.cache.lock() {
            cache.remove(key);
        }
        false
    }

    unsafe fn upload_planes(&mut self, planes: &Planes<'_>) -> Option<Uploaded> {
        let cl = &self.gpu.cl;
        let expand = if planes.expand && planes.pel == 2 {
            Some(self.gpu.expand_program()?)
        } else {
            None
        };
        let pel = planes.pel.max(1) as usize;
        let regions: [usize; 3] =
            std::array::from_fn(|plane| planes.pitch[plane] * planes.height[plane].max(0) as usize);
        let lengths: [usize; 3] = std::array::from_fn(|plane| {
            (planes.base[plane] + pel * pel * regions[plane]).min(planes.data[plane].len())
        });
        let totals: [usize; 3] = std::array::from_fn(|plane| {
            planes.tail[plane]
                .min(planes.data[plane].len())
                .max(lengths[plane])
        });
        let uploads: [usize; 3] = std::array::from_fn(|plane| {
            if expand.is_some() {
                regions[plane].min(lengths[plane])
            } else {
                lengths[plane]
            }
        });
        let coarse: [usize; 3] =
            std::array::from_fn(|plane| (totals[plane].max(1) + 64).next_multiple_of(256));
        let caps: [usize; 3] = std::array::from_fn(|plane| match planes.pyramid {
            Some(pyramid) => {
                let (luma, chroma) = pyramid.totals();
                coarse[plane] + if plane == 0 { luma } else { chroma } + 64
            }
            None => totals[plane].max(1) + 64,
        });
        let mut mems = [0usize; 3];
        let mut event: Handle = std::ptr::null_mut();
        for (plane, data) in planes.data.iter().enumerate() {
            let size = caps[plane];
            let pooled = self.gpu.pool.lock().ok().and_then(|mut pool| {
                let at = pool.iter().position(|&(_, s)| s == size)?;
                Some(pool.swap_remove(at).0)
            });
            let mem = if let Some(mem) = pooled {
                mem as Handle
            } else {
                let mut err = 0;
                let mem = unsafe {
                    (cl.CreateBuffer)(
                        self.gpu.context as Handle,
                        CL_MEM_READ_WRITE,
                        size,
                        std::ptr::null_mut(),
                        &raw mut err,
                    )
                };
                if err != CL_SUCCESS || mem.is_null() {
                    for &m in mems.iter().filter(|&&m| m != 0) {
                        unsafe { (cl.ReleaseMemObject)(m as Handle) };
                    }
                    return None;
                }
                mem
            };
            mems[plane] = mem as usize;
            let last = plane == 2 && expand.is_none();
            let region = regions[plane];
            let upload = uploads[plane];
            let target = if last {
                &raw mut event
            } else {
                std::ptr::null_mut()
            };
            let mut failed = false;
            if totals[plane] > lengths[plane] {
                failed = !unsafe {
                    self.transfer_at(
                        mem,
                        lengths[plane],
                        &data[lengths[plane]..totals[plane]],
                        std::ptr::null_mut(),
                    )
                };
            }
            failed = failed || !unsafe { self.transfer(mem, &data[..upload], target) };
            if let Some(program) = expand
                && !failed
            {
                failed = region * 4 > lengths[plane]
                    || unsafe {
                        self.launch(
                            (program, c"expand"),
                            &[
                                Arg::Mem(mem),
                                Arg::Long(planes.pitch[plane] as i64),
                                Arg::Long(i64::from(planes.height[plane])),
                            ],
                            3 * region,
                            None,
                            &[],
                            if plane == 2 {
                                &raw mut event
                            } else {
                                std::ptr::null_mut()
                            },
                        )
                    }
                    .is_none();
            }
            if failed {
                unsafe { (cl.Finish)(self.unit.queue as Handle) };
                for &m in mems.iter().filter(|&&m| m != 0) {
                    unsafe { (cl.ReleaseMemObject)(m as Handle) };
                }
                return None;
            }
        }
        Some(Uploaded {
            mems,
            sizes: caps,
            pyramid: planes.pyramid,
            coarse,
            host: None,
            frame: Frame {
                lengths: lengths.map(|length| length as i64),
                pitches: planes.pitch.map(|p| p as i64),
                bases: planes.base.map(|b| b as i64),
                heights: planes.height.map(i64::from),
                pel: i64::from(planes.pel),
            },
            event: event as usize,
        })
    }

    pub(crate) fn submit(
        &mut self,
        plan: &RecalcPlan,
        src: usize,
        reference: usize,
    ) -> Option<usize> {
        self.flush();
        let (source, target) = (
            self.frames[src].as_ref()?.2,
            self.frames[reference].as_ref()?.2,
        );
        let program = self
            .gpu
            .program(Variant::new(plan.width, plan.height, plan.satd))?;
        unsafe { self.enqueue(plan, src, reference, program, source, target) }
    }

    pub(crate) fn submit_compact(
        &mut self,
        plan: &CompactPlan,
        src: usize,
        reference: usize,
    ) -> Option<usize> {
        self.flush();
        let (source, target) = (
            self.frames[src].as_ref()?.2,
            self.frames[reference].as_ref()?.2,
        );
        let program = self.gpu.program(Variant {
            shift: plan.shift,
            ..Variant::new(4, 4, plan.satd)
        })?;
        unsafe { self.enqueue_compact(plan, src, reference, program, source, target) }
    }

    unsafe fn enqueue_compact(
        &mut self,
        plan: &CompactPlan,
        src: usize,
        reference: usize,
        program: usize,
        source: Frame,
        target: Frame,
    ) -> Option<usize> {
        let cl = &self.gpu.cl;
        let blocks = plan.blocks();
        let slot = self.tickets.len() % 2;
        let results = unsafe { self.gpu.pinned(&mut self.unit, 4 + slot, blocks * 8)? };
        let (source_mems, target_mems) = (
            self.frames[src].as_ref()?.1,
            self.frames[reference].as_ref()?.1,
        );
        let events: Vec<Handle> = self
            .frames
            .iter()
            .flatten()
            .map(|f| f.3 as Handle)
            .collect();
        let params = self.hold(bytes(&compact_params(
            &source,
            &target,
            blocks,
            plan.geometry,
        )));
        let overrides = self.hold(bytes(&plan.overrides));
        let seed = self.hold(bytes(&plan.seed));
        let params_mem = unsafe { self.write(slot, params)? };
        let seed_mem = unsafe { self.gpu.buffer(&mut self.unit, 2, seed.len())? };
        let results_mem = unsafe { self.gpu.buffer(&mut self.unit, 3, blocks * 8)? };
        let overrides_mem = unsafe { self.write(6, overrides)? };
        let args = recalc_args(
            source_mems.map(|m| m as Handle),
            target_mems.map(|m| m as Handle),
            &[
                Arg::Mem(params_mem),
                Arg::Mem(seed_mem),
                Arg::Mem(results_mem),
                Arg::Mem(overrides_mem),
            ],
        );
        let queue = self.unit.queue as Handle;
        let mut done: Handle = std::ptr::null_mut();
        if !unsafe { self.transfer(seed_mem, seed, std::ptr::null_mut()) }
            || unsafe {
                self.launch(
                    (program, c"recalc"),
                    &args,
                    blocks,
                    Some(64),
                    &events,
                    std::ptr::null_mut(),
                )
            }
            .is_none()
            || unsafe {
                (cl.EnqueueReadBuffer)(
                    queue,
                    results_mem,
                    CL_FALSE,
                    0,
                    blocks * 8,
                    results.cast(),
                    0,
                    std::ptr::null(),
                    (&raw mut done).cast(),
                )
            } != CL_SUCCESS
        {
            unsafe { (cl.Finish)(queue) };
            return None;
        }
        self.pending.push(done as usize);
        self.tickets.push((4 + slot, blocks));
        Some(self.tickets.len() - 1)
    }

    pub(crate) fn take_into(&mut self, ticket: usize, out: &mut [u32]) {
        let (index, blocks) = self.tickets[ticket];
        let host = self.unit.buffers[index].host as *const u32;
        out[..2 * blocks].copy_from_slice(unsafe { std::slice::from_raw_parts(host, 2 * blocks) });
    }

    unsafe fn enqueue(
        &mut self,
        plan: &RecalcPlan,
        src: usize,
        reference: usize,
        program: usize,
        source: Frame,
        target: Frame,
    ) -> Option<usize> {
        let cl = &self.gpu.cl;
        let blocks = plan.blocks();

        let slot = self.tickets.len() % 2;
        if blocks > 0 {
            let bests = unsafe { self.gpu.pinned(&mut self.unit, 4 + slot, blocks * 16)? };
            let mut params = [0i64; 24];
            params[..16].copy_from_slice(&frame_params(&source, &target));
            params[16..21].copy_from_slice(&[
                target.pel,
                plan.slots as i64,
                blocks as i64,
                i64::from(plan.pnew),
                i64::from(plan.pel),
            ]);
            let (source_mems, target_mems) = (
                self.frames[src].as_ref()?.1,
                self.frames[reference].as_ref()?.1,
            );
            let events: Vec<Handle> = self
                .frames
                .iter()
                .flatten()
                .map(|f| f.3 as Handle)
                .collect();
            let chunk = CHUNK.min(blocks);
            let params = self.hold(bytes(&params));
            let records = bytes(&plan.records);
            let params_mem = unsafe { self.write(slot, params)? };
            let records_mem = unsafe { self.gpu.buffer(&mut self.unit, 2, chunk * RECORD_BYTES)? };
            let bests_mem = unsafe { self.gpu.buffer(&mut self.unit, 3, chunk * 16)? };
            let args = recalc_args(
                source_mems.map(|m| m as Handle),
                target_mems.map(|m| m as Handle),
                &[
                    Arg::Mem(params_mem),
                    Arg::Mem(records_mem),
                    Arg::Mem(bests_mem),
                ],
            );
            let queue = self.unit.queue as Handle;
            let mut done: Handle = std::ptr::null_mut();
            let mut start = 0;
            while start < blocks {
                let count = CHUNK.min(blocks - start);
                let last = start + count == blocks;
                let waits: &[Handle] = if start == 0 { &events } else { &[] };
                if !unsafe {
                    self.transfer(
                        records_mem,
                        &records[start * RECORD_BYTES..(start + count) * RECORD_BYTES],
                        std::ptr::null_mut(),
                    )
                } || unsafe {
                    self.launch(
                        (program, c"recalc"),
                        &args,
                        count * 64,
                        Some(64),
                        waits,
                        std::ptr::null_mut(),
                    )
                }
                .is_none()
                    || unsafe {
                        (cl.EnqueueReadBuffer)(
                            queue,
                            bests_mem,
                            CL_FALSE,
                            0,
                            count * 16,
                            bests.add(start * 16).cast(),
                            0,
                            std::ptr::null(),
                            if last {
                                (&raw mut done).cast()
                            } else {
                                std::ptr::null_mut()
                            },
                        )
                    } != CL_SUCCESS
                {
                    unsafe { (cl.Finish)(queue) };
                    return None;
                }
                start += count;
            }
            self.pending.push(done as usize);
        }
        self.tickets.push((4 + slot, blocks));
        Some(self.tickets.len() - 1)
    }

    pub(crate) fn search(
        &mut self,
        params: &GpuSearchParams,
        pred: &[Mv],
        rev: &[Mv],
        src: usize,
        reference: usize,
    ) -> Option<usize> {
        self.flush();
        self.frames[src].as_ref()?;
        self.frames[reference].as_ref()?;
        let programs = self.gpu.search_pair(
            Variant::new(params.width, params.height, params.satd),
            params.rmax,
            &params.q,
            params.border,
        )?;
        unsafe { self.enqueue_search(params, pred, rev, src, reference, programs) }
    }

    unsafe fn launch_search(
        &mut self,
        m: &SearchMems,
        (blocks, border): (usize, usize),
        passes: usize,
        (program, border_program): (usize, usize),
        eight: bool,
        events: &[Handle],
    ) -> Option<()> {
        let per = if eight { 4 } else { 1 };
        let mut launches = Vec::new();
        for pass in 0..=passes {
            let name = if eight && pass > 0 {
                c"refine"
            } else {
                c"search"
            };
            launches.push((
                blocks,
                (m.res[(pass + 1) % 2], m.res[pass % 2]),
                (m.dirty[(pass + 1) % 2], m.dirty[pass % 2]),
                pass as i32,
                m.q,
                (program, name),
            ));
        }
        if border > 0 {
            launches.push((
                border,
                (m.res[passes % 2], m.res[(passes + 1) % 2]),
                (m.dirty[0], m.dirty[1]),
                0,
                m.q2,
                (border_program, c"search"),
            ));
        }
        for (index, (count, (rin, rout), (din, dout), pass, q, kernel)) in
            launches.into_iter().enumerate()
        {
            let (_, local) = unsafe { self.ckernel(kernel.0, kernel.1)? };
            if local < 32 {
                return None;
            }
            let groups = count.div_ceil(per * (local / 32));
            let mut strides = m.strides;
            strides[0] = if m.dirs > 1 { groups as i32 } else { i32::MAX };
            let args = [
                Arg::Mem(m.src[0]),
                Arg::Mem(m.src[1]),
                Arg::Mem(m.src[2]),
                Arg::Mem(m.reference[0]),
                Arg::Mem(m.reference[1]),
                Arg::Mem(m.reference[2]),
                Arg::Mem(q),
                Arg::Mem(m.pred),
                Arg::Mem(m.rev),
                Arg::Mem(rin),
                Arg::Mem(rout),
                Arg::Mem(m.cen),
                Arg::Mem(m.aux),
                Arg::Mem(m.out),
                Arg::Mem(din),
                Arg::Mem(dout),
                Arg::Int(pass),
                Arg::V8(strides),
            ];
            let waits: &[Handle] = if index == 0 { events } else { &[] };
            unsafe {
                self.launch(
                    kernel,
                    &args,
                    groups * m.dirs * local,
                    Some(local),
                    waits,
                    std::ptr::null_mut(),
                )?;
            }
        }
        Some(())
    }

    unsafe fn enqueue_search(
        &mut self,
        params: &GpuSearchParams,
        pred: &[Mv],
        rev: &[Mv],
        src: usize,
        reference: usize,
        programs: (usize, usize),
    ) -> Option<usize> {
        let cl = &self.gpu.cl;
        let blocks = pred.len();
        let total = blocks + params.border;
        let base = 7 + 12 * src;
        unsafe { self.gpu.pinned(&mut self.unit, base + 6, total * 16)? };
        let (source_mems, target_mems) = (
            self.frames[src].as_ref()?.1,
            self.frames[reference].as_ref()?.1,
        );
        let events: Vec<Handle> = self
            .frames
            .iter()
            .flatten()
            .map(|f| f.3 as Handle)
            .collect();
        let mut border_q = params.q;
        border_q[33] = 1;
        let q = self.hold(bytes(&params.q));
        let q2 = self.hold(bytes(&border_q));
        let q_mem = unsafe { self.write(base, q)? };
        let q2_mem = unsafe { self.write(base + 11, q2)? };
        let pred_mem = unsafe { self.write(base + 1, bytes(pred))? };
        let mut rev_rank = Vec::with_capacity(size_of_val(rev) + size_of_val(&params.rank[..]));
        rev_rank.extend_from_slice(bytes(rev));
        rev_rank.extend_from_slice(bytes(&params.rank));
        let rev_rank = self.hold_boxed(rev_rank.into());
        let rev_mem = unsafe { self.write(base + 2, rev_rank)? };
        let res = [
            unsafe { self.gpu.buffer(&mut self.unit, base + 3, total * 16)? },
            unsafe { self.gpu.buffer(&mut self.unit, base + 7, total * 16)? },
        ];
        let aux = unsafe { self.gpu.buffer(&mut self.unit, base + 4, total * 16)? };
        let cen = unsafe { self.gpu.buffer(&mut self.unit, base + 10, total * 16)? };
        let out = unsafe { self.gpu.buffer(&mut self.unit, base + 5, total * 16)? };
        let dirty = [
            unsafe { self.gpu.buffer(&mut self.unit, base + 8, total)? },
            unsafe { self.gpu.buffer(&mut self.unit, base + 9, total)? },
        ];
        let mems = SearchMems {
            src: source_mems.map(|m| m as Handle),
            reference: target_mems.map(|m| m as Handle),
            q: q_mem,
            q2: q2_mem,
            pred: pred_mem,
            rev: rev_mem,
            res,
            aux,
            cen,
            out,
            dirty,
            dirs: 1,
            strides: [0; 8],
        };
        let queue = self.unit.queue as Handle;
        let mut done: Handle = std::ptr::null_mut();
        unsafe {
            self.launch_search(
                &mems,
                (blocks, params.border),
                if params.q[32] == 0 { SWEEPS } else { 0 },
                programs,
                params.width == 8 && params.height == 8,
                &events,
            )?;
        }
        let host = self.unit.buffers[base + 6].host;
        if unsafe {
            (cl.EnqueueReadBuffer)(
                queue,
                out,
                CL_FALSE,
                0,
                total * 16,
                host as *mut c_void,
                0,
                std::ptr::null(),
                (&raw mut done).cast(),
            )
        } != CL_SUCCESS
        {
            unsafe { (cl.Finish)(queue) };
            return None;
        }
        self.pending.push(done as usize);
        self.tickets.push((base + 6, total));
        Some(self.tickets.len() - 1)
    }

    pub(crate) fn release(&mut self) {
        if self.unit.0.is_some() {
            self.wait();
            unsafe { (self.gpu.cl.Finish)(self.unit.queue as Handle) };
            self.unit = Lease(None);
        }
    }

    pub(crate) fn wait(&mut self) -> bool {
        if self.unit.0.is_none() {
            return true;
        }
        let cl = &self.gpu.cl;
        unsafe { (cl.Flush)(self.unit.queue as Handle) };
        let mut ok = true;
        for event in self.pending.drain(..) {
            let event = event as Handle;
            loop {
                let mut status: ClInt = 1;
                let got = unsafe {
                    (cl.GetEventInfo)(
                        event,
                        CL_EVENT_COMMAND_EXECUTION_STATUS,
                        size_of::<ClInt>(),
                        (&raw mut status).cast(),
                        std::ptr::null_mut(),
                    )
                };
                if got != CL_SUCCESS || status < 0 {
                    ok = false;
                    break;
                }
                if status == 0 {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_micros(50));
            }
            unsafe { (cl.ReleaseEvent)(event) };
        }
        if !ok {
            unsafe { (cl.Finish)(self.unit.queue as Handle) };
            self.unit.chain_stale = true;
        }
        ok
    }

    pub(crate) fn take(&mut self, ticket: usize) -> Vec<[i32; 4]> {
        self.view(ticket).to_vec()
    }

    pub(crate) fn view(&self, ticket: usize) -> &[[i32; 4]] {
        let (slot, blocks) = self.tickets[ticket];
        if blocks == 0 {
            return &[];
        }
        let host = self.unit.buffers[slot].host as *const [i32; 4];
        unsafe { std::slice::from_raw_parts(host, blocks) }
    }
}

impl Drop for Session<'_, '_> {
    fn drop(&mut self) {
        if !self.pending.is_empty() {
            self.wait();
        }
        if self.unit.0.is_some() {
            unsafe { (self.gpu.cl.Finish)(self.unit.queue as Handle) };
        }
        if let Ok(mut cache) = self.gpu.cache.lock() {
            for (entry, ..) in self.frames.iter().flatten() {
                let last = entry.uses.fetch_update(
                    std::sync::atomic::Ordering::AcqRel,
                    std::sync::atomic::Ordering::Acquire,
                    |uses| uses.checked_sub(1),
                );
                if last == Ok(1) {
                    cache.recent.retain(|e| !std::sync::Arc::ptr_eq(e, entry));
                }
            }
        }
    }
}

unsafe impl Send for RefineGpu {}
unsafe impl Sync for RefineGpu {}

fn recalc_args(src: [Handle; 3], reference: [Handle; 3], rest: &[Arg]) -> Vec<Arg> {
    src.into_iter()
        .chain(reference)
        .map(Arg::Mem)
        .chain(rest.iter().copied())
        .collect()
}

fn frame_params(source: &Frame, target: &Frame) -> [i64; 16] {
    let mut head = [0; 16];
    head[0..3].copy_from_slice(&source.lengths);
    head[3..6].copy_from_slice(&source.pitches);
    head[6..9].copy_from_slice(&target.lengths);
    head[9..12].copy_from_slice(&target.pitches);
    head[12..14].copy_from_slice(&target.bases[..2]);
    head[14..16].copy_from_slice(&target.heights[..2]);
    head
}

fn compact_params(source: &Frame, target: &Frame, blocks: usize, geometry: [i32; 20]) -> [i64; 41] {
    let g = geometry.map(i64::from);
    let mut params = [0; 41];
    params[..16].copy_from_slice(&frame_params(source, target));
    params[17] = blocks as i64;
    params[18..30].copy_from_slice(&g[..12]);
    params[30..33].copy_from_slice(&source.bases);
    params[33..].copy_from_slice(&g[12..]);
    params
}

fn bytes<T: Copy>(values: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), size_of_val(values)) }
}

const SEARCH: &str = include_str!("search.cl");

const SWEEPS: usize = 4;

const EXPAND: &str = r"
__kernel void expand(__global uchar *plane, long pitch, long height) {
    long size = pitch * height;
    long item = get_global_id(0);
    if (item >= 3 * size) return;
    int k = (int)(item / size) + 1;
    long i = item % size;
    long y = i / pitch, x = i % pitch;
    __global const uchar *p0 = plane;
    #define P0(j) ((j) < size ? (int)p0[j] : ((int)p0[(j) - size] + (int)p0[(j) - size + 1] + 1) >> 1)
    int value;
    if (k == 1) {
        value = (P0(i) + P0(i + 1) + 1) >> 1;
    } else if (k == 2) {
        value = y == 0 ? (int)p0[x] : (P0(i) + P0(i + pitch) + 1) >> 1;
    } else {
        value = y == 0 ? (int)p0[x + 1] : (P0(i) + P0(i + pitch + 1) + 1) >> 1;
    }
    plane[k * size + i] = (uchar)value;
}
";

const COMPACT: &str = r"
#define LO(v) (((v) << 16) >> 16)
#define HI(v) ((v) >> 16)
#define CW (4 >> S)
#ifdef DEV
#define SEEDT int4
#define OVT int
#define SEED(i) ((int2)(seed[i].x, seed[i].y))
#else
#define SEEDT int2
#define OVT int2
#define SEED(i) seed[i]
#endif

inline int fetch(__global const uchar *data, long length, long index) {
    return (index >= 0 && index < length) ? (int)data[index] : 0;
}

inline long subpel(long base, long pitch, long height, int pel, int x, int y) {
    if (pel == 1) return base + (long)y * pitch + x;
    int bits = pel == 2 ? 1 : 2;
    int mask = pel == 2 ? 1 : 3;
    long index = (long)((x & mask) | ((y & mask) << bits));
    return base + (long)(y >> bits) * pitch + (x >> bits) + index * pitch * height;
}

inline uint tile_cost(const int *src, __global const uchar *data, long length, long offset, long pitch) {
    int d[4][4];
    for (int y = 0; y < 4; y++) {
        long row = offset + (long)y * pitch;
        if (row >= 0 && row + 4 <= length) {
            int4 r = convert_int4(vload4(0, data + row));
            d[y][0] = r.x; d[y][1] = r.y; d[y][2] = r.z; d[y][3] = r.w;
        } else {
            for (int x = 0; x < 4; x++) d[y][x] = fetch(data, length, row + x);
        }
        for (int x = 0; x < 4; x++) d[y][x] = src[y * 4 + x] - d[y][x];
    }
#if SATD
    for (int y = 0; y < 4; y++) {
        int s0 = d[y][0] + d[y][1], s1 = d[y][2] + d[y][3];
        int d0 = d[y][0] - d[y][1], d1 = d[y][2] - d[y][3];
        d[y][0] = s0 + s1; d[y][1] = s0 - s1; d[y][2] = d0 + d1; d[y][3] = d0 - d1;
    }
    uint total = 0;
    for (int x = 0; x < 4; x++) {
        int s0 = d[0][x] + d[1][x], s1 = d[2][x] + d[3][x];
        int d0 = d[0][x] - d[1][x], d1 = d[2][x] - d[3][x];
        total += abs(s0 + s1) + abs(s0 - s1) + abs(d0 + d1) + abs(d0 - d1);
    }
    return total / 2;
#else
    uint total = 0;
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++) total += abs(d[y][x]);
    return total;
#endif
}

#if S
inline uint plane_cost(const int *src, __global const uchar *data, long length, long offset, long pitch) {
    uint total = 0;
    for (int x = 0; x < 2; x++) {
        total += abs(src[x] - fetch(data, length, offset + pitch + x));
        total += abs(src[2 + x] - fetch(data, length, offset + x));
    }
    return total;
}
#else
#define plane_cost tile_cost
#endif

inline int chroma_value(const int *su, const int *sv, __global const uchar *ru,
                        __global const uchar *rv, long lru, long lrv, long pru, long prv, long off) {
    uint u = plane_cost(su, ru, lru, off, pru);
    uint v = plane_cost(sv, rv, lrv, off, prv);
    return (int)((u + v) << (2 * S));
}

inline int outside(int v, int low, int high) {
    return v < low ? v - low : (v >= high ? v - high + 1 : 0);
}

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void recalc(__global const uchar *sy, __global const uchar *su, __global const uchar *sv,
            __global const uchar *ry, __global const uchar *ru, __global const uchar *rv,
            __global const long *p, __global const SEEDT *seed, __global uint2 *out,
            __global const OVT *overrides
#ifdef DEV
            , int seed_off, int out_off
#endif
            ) {
#ifdef DEV
    seed += seed_off;
    out += out_off;
#endif
    long block = get_global_id(0);
    if (block >= p[17]) return;
    int fw = (int)p[18], fh = (int)p[19], blk_x = (int)p[20], step_x = (int)p[21], step_y = (int)p[22];
    int radius = (int)p[23], bad_sad = (int)p[24], lambda = (int)p[25], diagonal = (int)p[26];
    int pnew = (int)p[27], pel = (int)p[28], count = (int)p[29];
    long lsy = p[0], lsu = p[1], lsv = p[2], psy = p[3], psu = p[4], psv = p[5];
    long lry = p[6], lru = p[7], lrv = p[8], pry = p[9], pru = p[10], prv = p[11];
    int x0 = (int)(block % blk_x) * step_x, y0 = (int)(block / blk_x) * step_y;
    int cx = (int)p[33], cy = (int)p[34], norm = (int)p[35];
    int odd_x = (int)p[36], even_x = (int)p[37], odd_y = (int)p[38], even_y = (int)p[39], flat = (int)p[40];
    int ki = min((int)(block % blk_x), 2 * cx - 1), kj = min((int)(block / blk_x), 2 * cy - 1);
    int off_x = -1 + 2 * (ki & 1), off_y = -1 + 2 * (kj & 1);
    int edge_x = ki == 0 || ki >= 2 * cx - 1, edge_y = kj == 0 || kj >= 2 * cy - 1;
    int near_x = ki / 2 + off_x, near_y = kj / 2 + off_y;
    int2 v1 = SEED(ki / 2 + (kj / 2) * cx), v2 = v1, v3 = v1, v4 = v1;
    if (edge_x && !edge_y) {
        v3 = v4 = SEED(ki / 2 + near_y * cx);
    } else if (!edge_x && edge_y) {
        v3 = v4 = SEED(near_x + (kj / 2) * cx);
    } else if (!edge_x && !edge_y) {
        v2 = SEED(near_x + (kj / 2) * cx);
        v3 = SEED(ki / 2 + near_y * cx);
        v4 = SEED(near_x + near_y * cx);
    }
    int tx, ty, vsad;
    if (flat) {
        tx = 9 * LO(v1.x) + 3 * LO(v2.x) + 3 * LO(v3.x) + LO(v4.x);
        ty = 9 * HI(v1.x) + 3 * HI(v2.x) + 3 * HI(v3.x) + HI(v4.x);
        vsad = 9 * v1.y + 3 * v2.y + 3 * v3.y + v4.y + 8;
    } else {
        int ax1 = off_x > 0 ? odd_x : even_x, ax2 = step_x * 4 - ax1;
        int ay1 = off_y > 0 ? odd_y : even_y, ay2 = step_y * 4 - ay1;
        int a11 = ax1 * ay1, a12 = ax1 * ay2, a21 = ax2 * ay1, a22 = ax2 * ay2;
        int norm_area = step_x * step_y;
        tx = (a11 * LO(v1.x) + a21 * LO(v2.x) + a12 * LO(v3.x) + a22 * LO(v4.x)) / norm_area;
        ty = (a11 * HI(v1.x) + a21 * HI(v2.x) + a12 * HI(v3.x) + a22 * HI(v4.x)) / norm_area;
        vsad = (a11 * v1.y + a21 * v2.y + a12 * v3.y + a22 * v4.y) / norm_area;
    }
    int px = ((int)(short)tx) >> norm, py = ((int)(short)ty) >> norm;
    px = (int)(short)px;
    py = (int)(short)py;
    vsad >>= 6;
    int sum = 0;
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++) sum += fetch(sy, lsy, p[30] + (long)(y0 + y) * psy + x0 + x);
    int average = sum >> 4;
    int best_x = px, best_y = py, bsad = vsad;
    if (bsad > bad_sad) {
        int min_x0 = -pel * x0, min_y0 = -pel * y0;
        int max_x0 = pel * (fw - x0 - 4), max_y0 = pel * (fh - y0 - 4);
        int dx = outside(px < 0 ? px - 2 : px + 2, min_x0, max_x0);
        int dy = outside(py < 0 ? py - 2 : py + 2, min_y0, max_y0);
        int pos_x = x0, pos_y = y0, cpos_x = x0 >> S, cpos_y = y0 >> S;
        int src_x = x0, src_y = y0, csrc_x = cpos_x, csrc_y = cpos_y;
        int low_x = min_x0, high_x = max_x0, low_y = min_y0, high_y = max_y0;
        if (dx != 0 || dy != 0) {
            pos_x = min(max(x0 - dx, 0), fw - 4);
            pos_y = min(max(y0 - dy, 0), fh - 4);
            dx = x0 - pos_x;
            dy = y0 - pos_y;
            cpos_x = (x0 >> S) - (dx >> S);
            cpos_y = (y0 >> S) - (dy >> S);
            src_x = pos_x; src_y = pos_y; csrc_x = cpos_x; csrc_y = cpos_y;
            low_x = min_x0 - min(dx, 0); high_x = max_x0 - max(dx, 0);
            low_y = min_y0 - min(dy, 0); high_y = max_y0 - max(dy, 0);
        } else if (count > 0) {
#ifdef DEV
            int first = overrides[block];
            if (first >= 0) {
#else
            int lo = 0, hi = count;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if (overrides[mid].x < block) lo = mid + 1; else hi = mid;
            }
            if (lo < count && overrides[lo].x == block) {
                int first = overrides[lo].y;
#endif
                src_x = (first % blk_x) * step_x;
                src_y = (first / blk_x) * step_y;
                csrc_x = src_x >> S;
                csrc_y = src_y >> S;
            }
        }
        int src_luma[16], src_u[CW * CW], src_v[CW * CW];
        long luma_at = p[30] + (long)src_y * psy + src_x;
        for (int b = 0; b < 16; b++) src_luma[b] = fetch(sy, lsy, luma_at + (long)(b >> 2) * psy + (b & 3));
        long u_at = p[31] + (long)csrc_y * psu + csrc_x, v_at = p[32] + (long)csrc_y * psv + csrc_x;
        for (int b = 0; b < CW * CW; b++) {
            src_u[b] = fetch(su, lsu, u_at + (long)(b / CW) * psu + b % CW);
            src_v[b] = fetch(sv, lsv, v_at + (long)(b / CW) * psv + b % CW);
        }
        int ref_x = pos_x * pel, ref_y = pos_y * pel, cref_x = cpos_x * pel, cref_y = cpos_y * pel;
#define LUMA(vx, vy) ((int)tile_cost(src_luma, ry, lry, subpel(p[12], pry, p[14], pel, ref_x + (vx), ref_y + (vy)), pry))
#define CHROMA(vx, vy) chroma_value(src_u, src_v, ru, rv, lru, lrv, pru, prv, subpel(p[13], pru, p[15], pel, cref_x + ((vx) >> S), cref_y + ((vy) >> S)))
        bsad = LUMA(px, py) + CHROMA(px, py);
        int min_cost = bsad;
        int min_x = max(px - radius, low_x), max_x = min(px + radius, high_x - 1);
        int min_y = max(py - radius, low_y), max_y = min(py + radius, high_y - 1);
        if (min_x <= max_x && min_y <= max_y) {
            for (int vy = min_y; vy <= max_y; vy++) {
                int dy = py - vy;
                uint dy2 = (uint)dy * (uint)dy;
                int luma_x = min_x;
                for (int vx = min_x; vx <= max_x; vx++) {
                    int dx = px - vx;
                    int cost = (int)((uint)lambda * ((uint)dx * (uint)dx + dy2)) >> 8;
                    if (cost >= min_cost) continue;
                    int at = pel == 1 ? luma_x : vx;
                    if (pel == 1) luma_x++;
                    int sad = LUMA(at, vy);
                    cost = (int)((uint)cost + (uint)sad + (uint)(((int)((uint)pnew * (uint)sad)) >> 8));
                    if (cost >= min_cost) continue;
                    int uv = CHROMA(vx, vy);
                    cost = (int)((uint)cost + (uint)uv + (uint)(((int)((uint)pnew * (uint)uv)) >> 8));
                    if (cost >= min_cost) continue;
                    best_x = vx;
                    best_y = vy;
                    bsad = sad + uv;
                    min_cost = cost;
                }
            }
        }
    }
    int mx = (int)(short)best_x, my = (int)(short)best_y;
    int relative = (abs(mx) + abs(my)) * 100 / diagonal;
    int score = bsad;
    if (relative > 50) {
        score = (int)((uint)bsad * 20u);
    } else if (relative > 15) {
        score = (int)((uint)bsad + (uint)((int)((uint)bsad * (uint)(relative - 15)) / 35 * 9));
    }
    out[block] = (uint2)(((uint)mx << 16) + (uint)my, (uint)((score & 0xFFFFFF) + (average << 24)));
}
";

const KERNEL: &str = r"
#define NONE_V (-2147483647 - 1)
#define LO(v) (((v) << 16) >> 16)
#define HI(v) ((v) >> 16)
#define MAX_SLOTS 49

inline int fetch(__global const uchar *data, long length, long index) {
    return (index >= 0 && index < length) ? (int)data[index] : 0;
}

inline long subpel(long base, long pitch, long height, int pel, int x, int y) {
    if (pel == 1) return base + (long)y * pitch + x;
    int bits = pel == 2 ? 1 : 2;
    int mask = pel == 2 ? 1 : 3;
    long index = (long)((x & mask) | ((y & mask) << bits));
    return base + (long)(y >> bits) * pitch + (x >> bits) + index * pitch * height;
}

inline uint tile_cost(__local const uchar *src, int spitch, __global const uchar *data, long length,
                      long offset, long pitch) {
    int d[4][4];
    for (int y = 0; y < 4; y++) {
        long row = offset + (long)y * pitch;
        int4 r;
        if (row >= 0 && row + 4 <= length) {
            r = convert_int4(vload4(0, data + row));
        } else {
            r = (int4)(fetch(data, length, row), fetch(data, length, row + 1),
                       fetch(data, length, row + 2), fetch(data, length, row + 3));
        }
        __local const uchar *s = src + y * spitch;
        d[y][0] = (int)s[0] - r.x;
        d[y][1] = (int)s[1] - r.y;
        d[y][2] = (int)s[2] - r.z;
        d[y][3] = (int)s[3] - r.w;
    }
#if SATD
    for (int y = 0; y < 4; y++) {
        int s0 = d[y][0] + d[y][1], s1 = d[y][2] + d[y][3];
        int d0 = d[y][0] - d[y][1], d1 = d[y][2] - d[y][3];
        d[y][0] = s0 + s1; d[y][1] = s0 - s1; d[y][2] = d0 + d1; d[y][3] = d0 - d1;
    }
    uint total = 0;
    for (int x = 0; x < 4; x++) {
        int s0 = d[0][x] + d[1][x], s1 = d[2][x] + d[3][x];
        int d0 = d[0][x] - d[1][x], d1 = d[2][x] - d[3][x];
        total += abs(s0 + s1) + abs(s0 - s1) + abs(d0 + d1) + abs(d0 - d1);
    }
    return total;
#else
    uint total = 0;
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++) total += abs(d[y][x]);
    return total;
#endif
}

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void recalc(__global const uchar *sy, __global const uchar *su, __global const uchar *sv,
            __global const uchar *ry, __global const uchar *ru, __global const uchar *rv,
            __global const long *p, __global const int *records, __global int4 *bests) {
    __local uchar src_luma[W * H];
    __local uchar src_u[(W / 2) * (H / 2)];
    __local uchar src_v[(W / 2) * (H / 2)];
    __local uint acc_luma[MAX_SLOTS], acc_u[MAX_SLOTS], acc_v[MAX_SLOTS];
    __local int sum_average;
    long block = get_group_id(0);
    int lid = get_local_id(0);
    long slots = p[17];
    int pnew = (int)p[19], pel = (int)p[20];
    long lsy = p[0], lsu = p[1], lsv = p[2], psy = p[3], psu = p[4], psv = p[5];
    long lry = p[6], lru = p[7], lrv = p[8], pry = p[9], pru = p[10], prv = p[11];
    __global const int *r = records + block * RECORD;
    int flags = r[11] & 3;
    int negative = (flags & 2) != 0, bad = (flags & 1) != 0;
    int bw = negative ? W / 2 : W, bh = negative ? H / 2 : H;
    int cw = W / 2, ch = H / 2;
    if (lid == 0) sum_average = 0;
    for (int i = lid; i < MAX_SLOTS; i += 64) {
        acc_luma[i] = 0;
        acc_u[i] = 0;
        acc_v[i] = 0;
    }
    int full = r[0], overlay = r[1], us = r[2], vs = r[3];
    for (int b = lid; b < bw * bh; b += 64) {
        int value = 0;
        if (overlay != NONE_V && b < (W / 2) * (H / 2)) {
            value = fetch(sy, lsy, overlay + (long)(b / (W / 2)) * psy + b % (W / 2));
        } else if (full != NONE_V) {
            value = fetch(sy, lsy, full + (long)(b / W) * psy + b % W);
        }
        src_luma[b] = (uchar)value;
    }
    if (!negative) {
        for (int b = lid; b < cw * ch; b += 64) {
            int y = b / cw, x = b % cw;
            src_u[b] = us == NONE_V ? 0 : (uchar)fetch(su, lsu, us + (long)y * psu + x);
            src_v[b] = vs == NONE_V ? 0 : (uchar)fetch(sv, lsv, vs + (long)y * psv + x);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (r[4] != NONE_V) {
        int part = 0;
        for (int b = lid; b < bw * bh; b += 64)
            part += fetch(sy, lsy, r[4] + (long)(b / bw) * psy + b % bw);
        atomic_add(&sum_average, part);
    }
    int min_x = LO(r[8]), max_x = HI(r[8]), min_y = LO(r[9]), max_y = HI(r[9]);
    int ww = max_x - min_x + 1, wh = max_y - min_y + 1;
    int px = LO(r[7]), py = HI(r[7]);
    int luma_tiles = (bw / 4) * (bh / 4);
    int chroma_tiles = negative ? 0 : (cw / 4) * (ch / 4);
    int per_slot = luma_tiles + 2 * chroma_tiles;
    if (bad) {
        for (int task = lid; task < slots * per_slot; task += 64) {
            int slot = task / per_slot, t = task % per_slot;
            int vx, vy;
            if (slot == 0) {
                vx = px;
                vy = py;
            } else {
                int c = slot - 1;
                if (ww <= 0 || wh <= 0 || c >= ww * wh) continue;
                vx = min_x + c % ww;
                vy = min_y + c / ww;
            }
            if (t < luma_tiles) {
                int tx = (t % (bw / 4)) * 4, ty = (t / (bw / 4)) * 4;
                long off = subpel(p[12], pry, p[14], pel, LO(r[5]) + vx, HI(r[5]) + vy);
                uint cost = tile_cost(src_luma + ty * bw + tx, bw, ry, lry,
                                      off + (long)ty * pry + tx, pry);
                atomic_add(&acc_luma[slot], cost);
            } else {
                int k = t - luma_tiles;
                int plane = k / chroma_tiles;
                k %= chroma_tiles;
                int tx = (k % (cw / 4)) * 4, ty = (k / (cw / 4)) * 4;
                long off = subpel(p[13], pru, p[15], pel, LO(r[6]) + (vx >> 1), HI(r[6]) + (vy >> 1));
                if (plane == 0) {
                    atomic_add(&acc_u[slot], tile_cost(src_u + ty * cw + tx, cw, ru, lru,
                                                       off + (long)ty * pru + tx, pru));
                } else {
                    atomic_add(&acc_v[slot], tile_cost(src_v + ty * cw + tx, cw, rv, lrv,
                                                       off + (long)ty * prv + tx, prv));
                }
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid != 0) return;
    int average = 0;
    if (r[4] != NONE_V && bh >= 4) average = sum_average >> (31 - clz(bw * bh));
    int bx = px, by = py, bsad = r[10];
    if (bad) {
#if SATD
#define LUMA(s) ((int)(acc_luma[s] / 2))
#define CHROMA(s) ((int)((acc_u[s] / 2 + acc_v[s] / 2) << 2))
#else
#define LUMA(s) ((int)acc_luma[s])
#define CHROMA(s) ((int)((acc_u[s] + acc_v[s]) << 2))
#endif
        bsad = (int)((uint)LUMA(0) + (uint)CHROMA(0));
        int min_cost = bsad;
        uint lambda = (uint)(r[11] >> 2);
        if (ww > 0 && wh > 0) {
            for (int vy = min_y; vy <= max_y; vy++) {
                int row = 1 + (vy - min_y) * ww;
                int dy = py - vy;
                uint dy2 = (uint)dy * (uint)dy;
                int luma_x = min_x;
                for (int vx = min_x; vx <= max_x; vx++) {
                    int dx = px - vx;
                    int cost = (int)(lambda * ((uint)dx * (uint)dx + dy2)) >> 8;
                    if (cost >= min_cost) continue;
                    int at = pel == 1 ? luma_x : vx;
                    if (pel == 1) luma_x++;
                    int sad = LUMA(row + (at - min_x));
                    cost = (int)((uint)cost + (uint)sad + (uint)(((int)((uint)pnew * (uint)sad)) >> 8));
                    if (cost >= min_cost) continue;
                    int uv = CHROMA(row + (vx - min_x));
                    cost = (int)((uint)cost + (uint)uv + (uint)(((int)((uint)pnew * (uint)uv)) >> 8));
                    if (cost >= min_cost) continue;
                    bx = vx;
                    by = vy;
                    bsad = (int)((uint)sad + (uint)uv);
                    min_cost = cost;
                }
            }
        }
    }
    if (negative) bsad = (int)((uint)bsad * 3u);
    bests[block] = (int4)(bx, by, bsad, average);
}
";
