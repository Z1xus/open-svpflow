#![cfg_attr(not(feature = "blend"), allow(dead_code))]

use core::ffi::c_void;
use std::cell::{Cell, RefCell};
use std::sync::Arc;

use super::{
    CL_FALSE, CL_MEM_READ_WRITE, CL_RGBA, CL_SUCCESS, CL_UNORM_INT8, ClContext, ClEvent,
    ClImageFormat, ClInt, ClKernel, ClMem, GpuBuf, GpuContext, KernelParams, StillMask, SuperEntry,
    SuperHandle, Unit, UploadPlane,
};

pub(super) const BLEND_VARIANT: u32 = 1 << 13;
const CL_MEM_OBJECT_IMAGE2D_ARRAY: u32 = 0x10F3;
const MAX_STEPS: usize = 16;

pub(super) type FnCreateImage = unsafe extern "C" fn(
    ClContext,
    u64,
    *const ClImageFormat,
    *const ImageDesc,
    *mut c_void,
    *mut ClInt,
) -> ClMem;

#[repr(C)]
pub(super) struct ImageDesc {
    image_type: u32,
    width: usize,
    height: usize,
    depth: usize,
    array_size: usize,
    row_pitch: usize,
    slice_pitch: usize,
    mip_levels: u32,
    samples: u32,
    buffer: ClMem,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Steps {
    count: i32,
    phase: [i32; MAX_STEPS],
    sad_blend: [f32; MAX_STEPS],
    weight: [f32; MAX_STEPS],
    gamma: f32,
}

#[derive(Default)]
pub(super) struct UnitState {
    pub(super) kernels: Option<[ClKernel; 2]>,
    pub(super) masks: (ClMem, usize, usize),
}

impl Default for Steps {
    fn default() -> Self {
        Self {
            count: 0,
            phase: [0; MAX_STEPS],
            sad_blend: [0.0; MAX_STEPS],
            weight: [0.0; MAX_STEPS],
            gamma: 1.0,
        }
    }
}

pub(super) struct Buffers {
    pub(super) sums: ClMem,
    pub(super) bytes: ClMem,
    len: usize,
}

unsafe impl Send for Buffers {}

struct Group {
    kernel: ClKernel,
    frames: [Arc<SuperEntry>; 2],
    sources: [[GpuBuf; 3]; 2],
    fields: (ClMem, ClMem),
    still: Option<StillMask>,
    key: (i64, bool, usize, usize),
    params: [KernelParams; 3],
    steps: Steps,
    masks: Vec<u8>,
}

#[derive(Default)]
struct Pending {
    group: Option<Group>,
    frames: Vec<Arc<SuperEntry>>,
    motion: Vec<Vec<u16>>,
    masks: Vec<Vec<u8>>,
}

pub struct BlendJob {
    buffers: Buffers,
    unit: RefCell<Unit>,
    width: usize,
    rows: usize,
    luma: usize,
    gamma: f32,
    weight: Cell<f32>,
    taken: Cell<bool>,
    failed: Cell<bool>,
    pending: RefCell<Pending>,
}

impl BlendJob {
    pub fn start(&self, weight: f32) {
        self.weight.set(weight);
        self.taken.set(false);
    }

    pub fn taken(&self) -> bool {
        self.taken.get()
    }

    pub fn failed(&self) -> bool {
        self.failed.get()
    }
}

pub struct OutputPlane<'a> {
    pub data: &'a mut [u8],
    pub stride: usize,
    pub width: usize,
    pub height: usize,
}

fn shared(mut params: [KernelParams; 3]) -> [KernelParams; 3] {
    for plane in &mut params {
        plane.phase = 0;
        plane.sad_blend = 0.0;
    }
    params
}

impl GpuContext {
    fn blend_unit(&self) -> Option<Unit> {
        if let Some(unit) = self.blend_units.lock().ok().and_then(|mut pool| pool.pop()) {
            return Some(unit);
        }
        let (mut kernel_err, mut queue_err) = (0, 0);
        unsafe {
            let kernel =
                (self.cl.CreateKernel)(self.program, c"render_frame".as_ptr(), &raw mut kernel_err);
            let queue =
                (self.cl.CreateCommandQueue)(self.context, self.device, 0, &raw mut queue_err);
            if kernel_err != CL_SUCCESS || queue_err != CL_SUCCESS {
                if !kernel.is_null() {
                    (self.cl.ReleaseKernel)(kernel);
                }
                if !queue.is_null() {
                    (self.cl.ReleaseCommandQueue)(queue);
                }
                return None;
            }
            Some(Unit::new(kernel, queue))
        }
    }

    fn blend_kernels(&self, unit: &mut Unit) -> Option<[ClKernel; 2]> {
        if unit.blend.kernels.is_none() {
            let mut kernels = [std::ptr::null_mut(); 2];
            for (kernel, name) in kernels.iter_mut().zip([c"blend_source", c"blend_resolve"]) {
                let mut err = 0;
                *kernel =
                    unsafe { (self.cl.CreateKernel)(self.program, name.as_ptr(), &raw mut err) };
                if err != CL_SUCCESS || kernel.is_null() {
                    return None;
                }
            }
            unit.blend.kernels = Some(kernels);
        }
        unit.blend.kernels
    }

    unsafe fn blend_buffer(&self, bytes: usize) -> Option<ClMem> {
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
    }

    unsafe fn mask_layers(&self, unit: &mut Unit, width: usize, height: usize) -> Option<ClMem> {
        let (mem, w, h) = unit.blend.masks;
        if !mem.is_null() && (w, h) == (width, height) {
            return Some(mem);
        }
        let format = ClImageFormat {
            channel_order: CL_RGBA,
            channel_data_type: CL_UNORM_INT8,
        };
        let desc = ImageDesc {
            image_type: CL_MEM_OBJECT_IMAGE2D_ARRAY,
            width,
            height,
            depth: 0,
            array_size: MAX_STEPS,
            row_pitch: 0,
            slice_pitch: 0,
            mip_levels: 0,
            samples: 0,
            buffer: std::ptr::null_mut(),
        };
        let mut err = 0;
        let created = unsafe {
            (self.cl.CreateImage?)(
                self.context,
                CL_MEM_READ_WRITE,
                &raw const format,
                &raw const desc,
                std::ptr::null_mut(),
                &raw mut err,
            )
        };
        if err != CL_SUCCESS || created.is_null() {
            return None;
        }
        if !mem.is_null() {
            unsafe { (self.cl.ReleaseMemObject)(mem) };
        }
        unit.blend.masks = (created, width, height);
        Some(created)
    }

    unsafe fn enqueue_resolve(
        &self,
        unit: &mut Unit,
        buffers: &Buffers,
        divisor: f32,
        gamma: f32,
        luma: usize,
    ) -> Option<()> {
        let luma = u32::try_from(luma).ok()?;
        let [_, resolve] = self.blend_kernels(unit)?;
        unsafe {
            let set = self.cl.SetKernelArg;
            if set(
                resolve,
                0,
                size_of::<ClMem>(),
                (&raw const buffers.sums).cast(),
            ) != CL_SUCCESS
                || set(
                    resolve,
                    1,
                    size_of::<ClMem>(),
                    (&raw const buffers.bytes).cast(),
                ) != CL_SUCCESS
                || set(resolve, 2, size_of::<f32>(), (&raw const divisor).cast()) != CL_SUCCESS
                || set(resolve, 3, size_of::<f32>(), (&raw const gamma).cast()) != CL_SUCCESS
                || set(resolve, 4, size_of::<u32>(), (&raw const luma).cast()) != CL_SUCCESS
            {
                return None;
            }
            ((self.cl.EnqueueNDRangeKernel)(
                unit.queue,
                resolve,
                1,
                std::ptr::null(),
                &raw const buffers.len,
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            ) == CL_SUCCESS)
                .then_some(())
        }
    }

    unsafe fn release_buffers(&self, buffers: &Buffers) {
        unsafe {
            (self.cl.ReleaseMemObject)(buffers.sums);
            (self.cl.ReleaseMemObject)(buffers.bytes);
        }
    }

    fn release_job(&self, job: BlendJob, keep: bool) {
        let unit = job.unit.into_inner();
        if !keep {
            unsafe { (self.cl.Finish)(unit.queue) };
        }
        drop(job.pending);
        match self.blend_pool.lock() {
            Ok(mut pool) if keep && pool.len() < super::IMAGE_POOL_CAP => pool.push(job.buffers),
            _ => unsafe { self.release_buffers(&job.buffers) },
        }
        match self.blend_units.lock() {
            Ok(mut pool) if pool.len() < super::IMAGE_POOL_CAP => pool.push(unit),
            _ => unsafe { self.release_unit(&unit) },
        }
    }

    pub fn blend_begin(
        &self,
        width: usize,
        height: usize,
        chroma_height: usize,
        gamma: f32,
    ) -> Option<BlendJob> {
        if self.hi10 || self.cl.CreateImage.is_none() {
            return None;
        }
        let rows = height + chroma_height;
        let len = width.checked_mul(rows).filter(|&len| len > 0)?;
        let mut unit = self.blend_unit()?;
        let reused = self.blend_pool.lock().ok().and_then(|mut pool| {
            let index = pool.iter().position(|buffers| buffers.len == len)?;
            Some(pool.swap_remove(index))
        });
        let created = || {
            let sums = unsafe { self.blend_buffer(len * size_of::<f32>()) }?;
            let Some(bytes) = (unsafe { self.blend_buffer(len) }) else {
                unsafe { (self.cl.ReleaseMemObject)(sums) };
                return None;
            };
            Some(Buffers { sums, bytes, len })
        };
        let fresh = reused.is_none();
        let Some(buffers) = reused.or_else(created) else {
            unsafe { self.release_unit(&unit) };
            return None;
        };
        if fresh && unsafe { self.enqueue_resolve(&mut unit, &buffers, 1.0, 1.0, 0) }.is_none() {
            unsafe {
                self.release_buffers(&buffers);
                self.release_unit(&unit);
            }
            return None;
        }
        Some(BlendJob {
            buffers,
            unit: RefCell::new(unit),
            width,
            rows,
            luma: height,
            gamma,
            weight: Cell::new(0.0),
            taken: Cell::new(false),
            failed: Cell::new(false),
            pending: RefCell::default(),
        })
    }

    fn flush(&self, job: &BlendJob) -> Option<()> {
        let mut pending = job.pending.borrow_mut();
        let Some(group) = pending.group.take() else {
            return Some(());
        };
        job.failed.set(true);
        let mut unit = job.unit.borrow_mut();
        let (width, height) = (group.key.2, group.key.3);
        let layers = unsafe { self.mask_layers(&mut unit, width, height) }?;
        let stride = i32::try_from(job.width).ok()?;
        let mut waits = Self::source_waits(group.sources[0][0], group.sources[1][0]);
        waits.extend(group.still.map(|StillMask(_, ready)| ready));
        let masks = group.masks.as_ptr();
        let filled = !group.masks.is_empty();
        pending.masks.push(group.masks);
        pending.frames.extend(group.frames);
        unsafe {
            if filled {
                let origin = [0usize; 3];
                let region = [width, height, group.steps.count as usize];
                if (self.cl.EnqueueWriteImage)(
                    unit.queue,
                    layers,
                    CL_FALSE,
                    origin.as_ptr(),
                    region.as_ptr(),
                    width * 4,
                    0,
                    masks.cast(),
                    0,
                    std::ptr::null(),
                    std::ptr::null_mut(),
                ) != CL_SUCCESS
                {
                    return None;
                }
            }
            let set = self.cl.SetKernelArg;
            let kernel = group.kernel;
            for (index, mem) in [(4, group.fields.0), (5, group.fields.1), (6, layers)] {
                if set(kernel, index, size_of::<ClMem>(), (&raw const mem).cast()) != CL_SUCCESS {
                    return None;
                }
            }
            if let Some(StillMask(still, _)) = group.still
                && set(kernel, 9, size_of::<ClMem>(), (&raw const still).cast()) != CL_SUCCESS
            {
                return None;
            }
            for (index, plane) in group.params.into_iter().enumerate() {
                let steps = Steps {
                    gamma: if index == 0 { job.gamma } else { 1.0 },
                    ..group.steps
                };
                if set(kernel, 8, size_of::<Steps>(), (&raw const steps).cast()) != CL_SUCCESS {
                    return None;
                }
                self.enqueue_plane(
                    unit.queue,
                    kernel,
                    job.buffers.sums,
                    stride,
                    plane,
                    group.sources[0][index],
                    group.sources[1][index],
                    if index == 0 { &waits } else { &[] },
                )?;
            }
            (self.cl.Flush)(unit.queue);
        }
        job.failed.set(false);
        Some(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn blend_render(
        &self,
        job: &BlendJob,
        src0: &SuperHandle,
        src1: &SuperHandle,
        linear: bool,
        params: [KernelParams; 3],
        motion_key: i64,
        motion_width: usize,
        motion_height: usize,
        motions: [(&[u16], &[u16]); 4],
        coverage: (&[u8], &[u8]),
        area: Option<(&[u8], &[u8])>,
        still: Option<StillMask>,
    ) -> Option<()> {
        let key = (motion_key, linear, motion_width, motion_height);
        let joins = job.pending.borrow().group.as_ref().is_some_and(|group| {
            group.key == key
                && group.still == still
                && (group.steps.count as usize) < MAX_STEPS
                && shared(group.params) == shared(params)
                && Arc::ptr_eq(&group.frames[0], &src0.entry)
                && Arc::ptr_eq(&group.frames[1], &src1.entry)
        });
        if !joins {
            self.flush(job)?;
            let mut unit = job.unit.borrow_mut();
            let unit = &mut *unit;
            let variant =
                Self::variant(&params[0], false) | BLEND_VARIANT | u32::from(still.is_some()) << 14;
            let kernel = self.variant_kernel(&mut unit.variant_kernels, variant);
            let fields = unsafe {
                self.upload_motion(
                    unit,
                    params[0].algorithm,
                    motion_key,
                    motion_width,
                    motion_height,
                    motions,
                )
            };
            let mut pending = job.pending.borrow_mut();
            for packed in [&mut unit.packed_base, &mut unit.packed_ext] {
                if !packed.is_empty() {
                    pending.motion.push(std::mem::take(packed));
                }
            }
            pending.group = Some(Group {
                kernel: kernel?,
                frames: [Arc::clone(&src0.entry), Arc::clone(&src1.entry)],
                sources: [src0.sources(linear), src1.sources(linear)],
                fields: fields?,
                still,
                key,
                params: shared(params),
                steps: Steps::default(),
                masks: Vec::new(),
            });
        }
        let mut pending = job.pending.borrow_mut();
        let group = pending.group.as_mut()?;
        let needs_coverage = params[0].algorithm >= 21;
        if needs_coverage || params[0].has_sad != 0 {
            Self::pack_masks(
                &mut group.masks,
                motion_width * motion_height,
                needs_coverage,
                coverage,
                area,
            )?;
        }
        let step = group.steps.count as usize;
        group.steps.phase[step] = params[0].phase;
        group.steps.sad_blend[step] = params[0].sad_blend;
        group.steps.weight[step] = job.weight.get();
        group.steps.count += 1;
        job.taken.set(true);
        Some(())
    }

    pub fn blend_planes(
        &self,
        job: &BlendJob,
        key: Option<i64>,
        planes: [UploadPlane<'_>; 3],
    ) -> Option<()> {
        self.flush(job)?;
        let entry = match key {
            Some(key) => self.cache_frame(key, planes)?.entry,
            None => Arc::new(self.upload_frame(planes)?),
        };
        if entry.dims != (job.width, job.rows) {
            return None;
        }
        let mut unit = job.unit.borrow_mut();
        let [source, _] = self.blend_kernels(&mut unit)?;
        let weight = job.weight.get();
        let stride = i32::try_from(job.width).ok()?;
        let luma = i32::try_from(job.luma).ok()?;
        let image = entry.mems[1];
        let ready: ClEvent = entry.ready;
        let global = [job.width, job.rows];
        job.pending.borrow_mut().frames.push(Arc::clone(&entry));
        unsafe {
            let set = self.cl.SetKernelArg;
            if set(
                source,
                0,
                size_of::<ClMem>(),
                (&raw const job.buffers.sums).cast(),
            ) != CL_SUCCESS
                || set(source, 1, size_of::<i32>(), (&raw const stride).cast()) != CL_SUCCESS
                || set(source, 2, size_of::<ClMem>(), (&raw const image).cast()) != CL_SUCCESS
                || set(source, 3, size_of::<f32>(), (&raw const weight).cast()) != CL_SUCCESS
                || set(source, 4, size_of::<f32>(), (&raw const job.gamma).cast()) != CL_SUCCESS
                || set(source, 5, size_of::<i32>(), (&raw const luma).cast()) != CL_SUCCESS
            {
                return None;
            }
            if (self.cl.EnqueueNDRangeKernel)(
                unit.queue,
                source,
                2,
                std::ptr::null(),
                global.as_ptr(),
                std::ptr::null(),
                u32::from(!ready.is_null()),
                if ready.is_null() {
                    std::ptr::null()
                } else {
                    (&raw const ready).cast()
                },
                std::ptr::null_mut(),
            ) != CL_SUCCESS
            {
                job.failed.set(true);
                return None;
            }
            (self.cl.Flush)(unit.queue);
        }
        job.taken.set(true);
        Some(())
    }

    pub fn blend_abort(&self, job: BlendJob) {
        self.release_job(job, false);
    }

    pub fn blend_finish(
        &self,
        job: BlendJob,
        divisor: f32,
        planes: [OutputPlane<'_>; 3],
    ) -> Option<()> {
        let fits = planes[0].width == job.width
            && planes[0].height + planes[1].height == job.rows
            && planes[1].width + planes[2].width <= job.width
            && planes.iter().all(|plane| {
                plane.width <= plane.stride
                    && plane.height > 0
                    && plane.data.len() >= plane.stride * (plane.height - 1) + plane.width
            });
        if !fits || self.flush(&job).is_none() || job.failed() {
            self.blend_abort(job);
            return None;
        }
        let mut unit = job.unit.borrow_mut();
        let queue = unit.queue;
        let read = unsafe {
            self.enqueue_resolve(
                &mut unit,
                &job.buffers,
                divisor,
                job.gamma,
                job.width * job.luma,
            )
            .and_then(|()| self.take_staging(queue, job.buffers.len))
            .and_then(|staging| {
                let mut done: ClEvent = std::ptr::null_mut();
                if (self.cl.EnqueueReadBuffer)(
                    queue,
                    job.buffers.bytes,
                    CL_FALSE,
                    0,
                    job.buffers.len,
                    staging.host.cast(),
                    0,
                    std::ptr::null(),
                    (&raw mut done).cast(),
                ) == CL_SUCCESS
                {
                    Some((staging, done))
                } else {
                    self.put_staging(staging);
                    None
                }
            })
        };
        drop(unit);
        let Some((staging, done)) = read else {
            self.blend_abort(job);
            return None;
        };
        if !unsafe { self.wait_event(queue, done) } {
            unsafe { (self.cl.ReleaseMemObject)(staging.mem) };
            self.blend_abort(job);
            return None;
        }
        let host = unsafe { std::slice::from_raw_parts(staging.host, job.buffers.len) };
        let luma_rows = planes[0].height;
        let chroma_width = planes[1].width;
        for (index, plane) in planes.into_iter().enumerate() {
            let (x, y) = match index {
                0 => (0, 0),
                1 => (0, luma_rows),
                _ => (chroma_width, luma_rows),
            };
            for row in 0..plane.height {
                let from = (y + row) * job.width + x;
                plane.data[row * plane.stride..][..plane.width]
                    .copy_from_slice(&host[from..from + plane.width]);
            }
        }
        self.put_staging(staging);
        self.release_job(job, true);
        Some(())
    }
}
