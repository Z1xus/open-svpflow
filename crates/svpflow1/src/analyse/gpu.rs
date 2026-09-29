#![allow(unsafe_code)]
use core::ffi::{c_char, c_void};
use std::ffi::CString;
use std::sync::{Mutex, OnceLock};

use super::field::recalc::RecalcPlan;

type ClInt = i32;
type ClUint = u32;
type Handle = *mut c_void;

const CL_SUCCESS: ClInt = 0;
const CL_DEVICE_TYPE_GPU: u64 = 1 << 2;
const CL_MEM_READ_WRITE: u64 = 1;
const CL_MEM_ALLOC_HOST_PTR: u64 = 1 << 4;
const CL_MAP_WRITE: u64 = 1 << 1;
const STAGING: usize = 4 << 20;
const CL_FALSE: ClUint = 0;
const CL_TRUE: ClUint = 1;
const CL_EVENT_COMMAND_EXECUTION_STATUS: ClUint = 0x11D3;
const UNITS: usize = 32;

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
type FnRelease = unsafe extern "C" fn(Handle) -> ClInt;
type FnFinish = unsafe extern "C" fn(Handle) -> ClInt;
type FnWaitForEvents = unsafe extern "C" fn(ClUint, *const Handle) -> ClInt;
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
type FnEnqueueUnmap =
    unsafe extern "C" fn(Handle, Handle, *mut c_void, ClUint, *const c_void, *mut c_void) -> ClInt;
type FnGetEventInfo = unsafe extern "C" fn(Handle, ClUint, usize, *mut c_void, *mut usize) -> ClInt;

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
    SetKernelArg: FnSetKernelArg,
    EnqueueNDRangeKernel: FnEnqueueNDRangeKernel,
    ReleaseMemObject: FnRelease,
    Finish: FnFinish,
    ReleaseEvent: FnRelease,
    WaitForEvents: FnWaitForEvents,
    GetEventInfo: FnGetEventInfo,
    Flush: FnFinish,
    EnqueueMapBuffer: FnEnqueueMapBuffer,
    EnqueueUnmap: FnEnqueueUnmap,
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
            SetKernelArg: sym!("clSetKernelArg"),
            EnqueueNDRangeKernel: sym!("clEnqueueNDRangeKernel"),
            ReleaseMemObject: sym!("clReleaseMemObject"),
            Finish: sym!("clFinish"),
            ReleaseEvent: sym!("clReleaseEvent"),
            WaitForEvents: sym!("clWaitForEvents"),
            GetEventInfo: sym!("clGetEventInfo"),
            Flush: sym!("clFlush"),
            EnqueueMapBuffer: sym!("clEnqueueMapBuffer"),
            EnqueueUnmap: sym!("clEnqueueUnmapMemObject"),
            _lib: lib,
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct Variant {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) satd: bool,
}

#[derive(Default)]
struct Buffer {
    mem: usize,
    capacity: usize,
}

#[derive(Default)]
struct Unit {
    queue: usize,
    expand: usize,
    staging: usize,
    mapped: usize,
    capacity: usize,
    used: usize,
    retired: Vec<(usize, usize)>,
    kernels: Vec<(usize, usize)>,
    buffers: [Buffer; 18],
}

pub(crate) struct RefineGpu {
    cl: OpenCl,
    context: usize,
    device: usize,
    programs: Mutex<Vec<(Variant, usize)>>,
    units: Vec<Mutex<Unit>>,
    cache: Mutex<std::collections::VecDeque<std::sync::Arc<CachedFrame>>>,
    expand: OnceLock<Option<usize>>,
    pool: Mutex<Vec<(usize, usize)>>,
}

const CACHED_FRAMES: usize = 16;

struct Uploaded {
    mems: [usize; 3],
    sizes: [usize; 3],
    frame: Frame,
    event: usize,
}

struct CachedFrame {
    key: (usize, i32),
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
                    Some(pool) if pool.len() < 3 * CACHED_FRAMES => pool.push((mem, size)),
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
                units,
                cache: Mutex::new(std::collections::VecDeque::new()),
                expand: OnceLock::new(),
                pool: Mutex::new(Vec::new()),
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
        let program = self.build(&format!(
            "#define W {}\n#define H {}\n#define SATD {}\n{KERNEL}",
            variant.width,
            variant.height,
            u8::from(variant.satd)
        ))?;
        programs.push((variant, program));
        Some(program)
    }

    fn build(&self, text: &str) -> Option<usize> {
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
        let built = unsafe {
            (cl.BuildProgram)(
                program,
                1,
                &raw const device,
                c"".as_ptr(),
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        if built != CL_SUCCESS {
            return None;
        }
        Some(program as usize)
    }

    pub(crate) fn session(&self) -> Option<Session<'_>> {
        let n = self.units.len();
        let unit = self
            .units
            .iter()
            .find_map(|unit| unit.try_lock().ok())
            .or_else(|| self.units[0].lock().ok())?;
        let mut unit = unit;
        unit.used = 0;
        let _ = n;
        Some(Session {
            gpu: self,
            unit,
            frames: [None, None],
            tickets: Vec::new(),
            pending: Vec::new(),
        })
    }

    unsafe fn buffer(&self, unit: &mut Unit, index: usize, size: usize) -> Option<Handle> {
        let buffer = &mut unit.buffers[index];
        if buffer.capacity < size {
            if buffer.mem != 0 {
                unsafe { (self.cl.ReleaseMemObject)(buffer.mem as Handle) };
                buffer.mem = 0;
            }
            let capacity = size.next_power_of_two();
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
            };
        }
        Some(buffer.mem as Handle)
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

pub(crate) struct Session<'g> {
    gpu: &'g RefineGpu,
    unit: std::sync::MutexGuard<'g, Unit>,
    frames: [Option<SessionFrame>; 2],
    tickets: Vec<Vec<[i32; 4]>>,
    pending: Vec<usize>,
}

impl Session<'_> {
    unsafe fn staging(&mut self, needed: usize) -> Option<*mut u8> {
        let cl = &self.gpu.cl;
        if self.unit.used.next_multiple_of(256) + needed > self.unit.capacity {
            if self.unit.staging != 0 {
                let old = (self.unit.staging, self.unit.mapped);
                self.unit.retired.push(old);
                self.unit.staging = 0;
            }
            let capacity = (2 * self.unit.capacity)
                .max(STAGING)
                .max(needed.next_power_of_two());
            let mut err = 0;
            let mem = unsafe {
                (cl.CreateBuffer)(
                    self.gpu.context as Handle,
                    CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                    capacity,
                    std::ptr::null_mut(),
                    &raw mut err,
                )
            };
            if err != CL_SUCCESS || mem.is_null() {
                self.unit.capacity = 0;
                return None;
            }
            let pointer = unsafe {
                (cl.EnqueueMapBuffer)(
                    self.unit.queue as Handle,
                    mem,
                    CL_TRUE,
                    CL_MAP_WRITE,
                    0,
                    capacity,
                    0,
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    &raw mut err,
                )
            };
            if err != CL_SUCCESS || pointer.is_null() {
                unsafe { (cl.ReleaseMemObject)(mem) };
                self.unit.capacity = 0;
                return None;
            }
            self.unit.staging = mem as usize;
            self.unit.mapped = pointer as usize;
            self.unit.capacity = capacity;
            self.unit.used = 0;
        }
        Some(self.unit.mapped as *mut u8)
    }

    unsafe fn transfer(&mut self, mem: Handle, data: &[u8], event: *mut Handle) -> bool {
        let cl = &self.gpu.cl;
        let queue = self.unit.queue as Handle;
        let mut source = data.as_ptr();
        if let Some(pointer) = unsafe { self.staging(data.len()) } {
            let start = self.unit.used.next_multiple_of(256);
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), pointer.add(start), data.len());
                source = pointer.add(start);
            }
            self.unit.used = start + data.len();
        }
        let status = unsafe {
            (cl.EnqueueWriteBuffer)(
                queue,
                mem,
                CL_FALSE,
                0,
                data.len(),
                source.cast_mut().cast(),
                0,
                std::ptr::null(),
                event.cast(),
            )
        };
        status == CL_SUCCESS
    }

    unsafe fn write(&mut self, index: usize, data: &[u8]) -> Option<Handle> {
        let mem = unsafe { self.gpu.buffer(&mut self.unit, index, data.len().max(1))? };
        if !data.is_empty() && !unsafe { self.transfer(mem, data, std::ptr::null_mut()) } {
            return None;
        }
        Some(mem)
    }

    pub(crate) fn upload(&mut self, slot: usize, key: (usize, i32), planes: &Planes<'_>) -> bool {
        self.frames[slot] = None;
        let gpu = self.gpu;
        let (entry, guard) = {
            let Ok(mut cache) = gpu.cache.lock() else {
                return false;
            };
            if let Some(entry) = cache.iter().find(|e| e.key == key).cloned() {
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
                gpu,
                state: Mutex::new(None),
            });
            let Ok(guard) = entry.state.lock() else {
                return false;
            };
            let guard: std::sync::MutexGuard<'static, Option<Uploaded>> =
                unsafe { std::mem::transmute(guard) };
            cache.push_back(entry.clone());
            while cache.len() > CACHED_FRAMES {
                cache.pop_front();
            }
            (entry, guard)
        };
        let mut guard = guard;
        let uploaded = unsafe { self.upload_planes(planes) };
        let result = uploaded.as_ref().map(|u| (u.mems, u.frame, u.event));
        *guard = uploaded;
        drop(guard);
        if let Some((mems, frame, event)) = result {
            self.frames[slot] = Some((entry, mems, frame, event));
            return true;
        }
        if let Ok(mut cache) = gpu.cache.lock() {
            cache.retain(|e| e.key != key);
        }
        false
    }

    unsafe fn upload_planes(&mut self, planes: &Planes<'_>) -> Option<Uploaded> {
        let cl = &self.gpu.cl;
        let expand = if planes.expand && planes.pel == 2 {
            let program = self.gpu.expand_program()?;
            let kernel = match self.unit.expand {
                0 => {
                    let mut err = 0;
                    let kernel = unsafe {
                        (cl.CreateKernel)(program as Handle, c"expand".as_ptr(), &raw mut err)
                    };
                    if err != CL_SUCCESS || kernel.is_null() {
                        return None;
                    }
                    self.unit.expand = kernel as usize;
                    kernel
                }
                kernel => kernel as Handle,
            };
            Some(kernel)
        } else {
            None
        };
        let mut mems = [0usize; 3];
        let mut event: Handle = std::ptr::null_mut();
        for (plane, data) in planes.data.iter().enumerate() {
            let size = data.len().max(1);
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
            let region = planes.pitch[plane] * planes.height[plane].max(0) as usize;
            let upload = if expand.is_some() {
                region.min(data.len())
            } else {
                data.len()
            };
            let target = if last {
                &raw mut event
            } else {
                std::ptr::null_mut()
            };
            let mut failed = !unsafe { self.transfer(mem, &data[..upload], target) };
            if let Some(kernel) = expand
                && !failed
            {
                let pitch = planes.pitch[plane] as i64;
                let height = i64::from(planes.height[plane]);
                let args: [(usize, *const c_void); 3] = [
                    (size_of::<Handle>(), (&raw const mem).cast()),
                    (8, (&raw const pitch).cast()),
                    (8, (&raw const height).cast()),
                ];
                let global = 3 * region;
                failed = region * 4 > data.len()
                    || args
                        .iter()
                        .enumerate()
                        .any(|(index, &(size, value))| unsafe {
                            (cl.SetKernelArg)(kernel, index as ClUint, size, value) != CL_SUCCESS
                        })
                    || unsafe {
                        (cl.EnqueueNDRangeKernel)(
                            self.unit.queue as Handle,
                            kernel,
                            1,
                            std::ptr::null(),
                            &raw const global,
                            std::ptr::null(),
                            0,
                            std::ptr::null(),
                            if plane == 2 {
                                (&raw mut event).cast()
                            } else {
                                std::ptr::null_mut()
                            },
                        )
                    } != CL_SUCCESS;
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
            sizes: planes.data.map(|d| d.len().max(1)),
            frame: Frame {
                lengths: planes.data.map(|d| d.len() as i64),
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
        let (source, target) = (
            self.frames[src].as_ref()?.2,
            self.frames[reference].as_ref()?.2,
        );
        let variant = Variant {
            width: plan.width,
            height: plan.height,
            satd: plan.satd,
        };
        let program = self.gpu.program(variant)?;
        unsafe { self.enqueue(plan, src, reference, program, source, target) }
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

        let mut bests = vec![[0i32; 4]; blocks];
        if blocks > 0 {
            let kernel =
                if let Some(&(_, kernel)) = self.unit.kernels.iter().find(|(p, _)| *p == program) {
                    kernel as Handle
                } else {
                    let mut err = 0;
                    let kernel = unsafe {
                        (cl.CreateKernel)(program as Handle, c"recalc".as_ptr(), &raw mut err)
                    };
                    if err != CL_SUCCESS || kernel.is_null() {
                        return None;
                    }
                    self.unit.kernels.push((program, kernel as usize));
                    kernel
                };
            let field = self.tickets.len() % 2;
            let base = 6 + field * 6;
            let params: [i64; 24] = [
                source.lengths[0],
                source.lengths[1],
                source.lengths[2],
                source.pitches[0],
                source.pitches[1],
                source.pitches[2],
                target.lengths[0],
                target.lengths[1],
                target.lengths[2],
                target.pitches[0],
                target.pitches[1],
                target.pitches[2],
                target.bases[0],
                target.bases[1],
                target.heights[0],
                target.heights[1],
                target.pel,
                plan.slots as i64,
                blocks as i64,
                i64::from(plan.pnew),
                i64::from(plan.pel),
                0,
                0,
                0,
            ];
            let mut mems = [std::ptr::null_mut(); 9];
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
            for plane in 0..3 {
                mems[plane] = source_mems[plane] as Handle;
                mems[3 + plane] = target_mems[plane] as Handle;
            }
            mems[6] = unsafe { self.write(base, bytes(&params))? };
            mems[7] = unsafe { self.write(base + 1, bytes(&plan.records))? };
            mems[8] = unsafe { self.gpu.buffer(&mut self.unit, base + 5, blocks * 16)? };
            let set = mems.iter().enumerate().all(|(index, mem)| unsafe {
                (cl.SetKernelArg)(
                    kernel,
                    index as ClUint,
                    size_of::<Handle>(),
                    (&raw const *mem).cast(),
                ) == CL_SUCCESS
            });
            if !set {
                return None;
            }
            let queue = self.unit.queue as Handle;
            let (local, global) = (64usize, blocks * 64);
            if unsafe {
                (cl.EnqueueNDRangeKernel)(
                    queue,
                    kernel,
                    1,
                    std::ptr::null(),
                    &raw const global,
                    &raw const local,
                    events.len() as ClUint,
                    events.as_ptr().cast(),
                    std::ptr::null_mut(),
                )
            } != CL_SUCCESS
            {
                return None;
            }
            let mut done: Handle = std::ptr::null_mut();
            if unsafe {
                (cl.EnqueueReadBuffer)(
                    queue,
                    mems[8],
                    CL_FALSE,
                    0,
                    blocks * 16,
                    bests.as_mut_ptr().cast(),
                    0,
                    std::ptr::null(),
                    (&raw mut done).cast(),
                )
            } != CL_SUCCESS
            {
                return None;
            }
            self.pending.push(done as usize);
        }
        self.tickets.push(bests);
        Some(self.tickets.len() - 1)
    }

    pub(crate) fn wait(&mut self) -> bool {
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
        ok
    }

    pub(crate) fn take(&mut self, ticket: usize) -> Vec<[i32; 4]> {
        std::mem::take(&mut self.tickets[ticket])
    }
}

impl Drop for Session<'_> {
    fn drop(&mut self) {
        if !self.pending.is_empty() {
            self.wait();
        }
        let cl = &self.gpu.cl;
        let queue = self.unit.queue as Handle;
        for &(mem, mapped) in &self.unit.retired {
            unsafe {
                (cl.EnqueueUnmap)(
                    queue,
                    mem as Handle,
                    mapped as *mut c_void,
                    0,
                    std::ptr::null(),
                    std::ptr::null_mut(),
                );
            }
        }
        unsafe { (cl.Finish)(queue) };
        for (mem, _) in self.unit.retired.drain(..) {
            unsafe { (cl.ReleaseMemObject)(mem as Handle) };
        }
    }
}

unsafe impl Send for RefineGpu {}
unsafe impl Sync for RefineGpu {}

fn bytes<T: Copy>(values: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), size_of_val(values)) }
}

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

const KERNEL: &str = r"
#define NONE_V (-2147483647 - 1)
#define RECORD 12
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
