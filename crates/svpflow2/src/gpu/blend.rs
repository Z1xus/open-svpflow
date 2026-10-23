#![cfg_attr(not(feature = "blend"), allow(dead_code))]

use std::cell::{Cell, RefCell};
use std::sync::{Arc, Condvar, Mutex};
use svpflow_host::opencl::ImageDesc;

use super::{
    CL_FALSE, CL_MEM_READ_WRITE, CL_RGBA, CL_SUCCESS, CL_UNORM_INT8, ClEvent, ClImageFormat,
    ClKernel, ClMem, ClUint, GpuBuf, GpuContext, KernelParams, Layer, StillMask, SuperEntry,
    SuperHandle, Unit, UploadPlane, layer::LAYER_VARIANT, note,
};

pub(super) const BLEND_VARIANT: u32 = 1 << 13;
const FUSE_VARIANT: u32 = 1 << 17;
const MAX_SLOTS: usize = 4;
pub(super) const FUSED_ARG_BYTES: usize = (size_of::<ClMem>() + size_of::<i32>())
    .next_multiple_of(align_of::<ClMem>())
    + MAX_SLOTS * (4 * size_of::<ClMem>() + size_of::<Steps>())
    + size_of::<KernelParams>();
const CL_MEM_OBJECT_IMAGE2D_ARRAY: u32 = 0x10F3;
const MAX_STEPS: usize = 16;

#[repr(C)]
#[derive(Clone, Copy)]
struct Steps {
    count: i32,
    phase: [i32; MAX_STEPS],
    sad_blend: [f32; MAX_STEPS],
    weight: [f32; MAX_STEPS],
    gamma: f32,
    source: f32,
}

#[derive(Default)]
pub(super) struct UnitState {
    pub(super) kernels: Option<[ClKernel; 3]>,
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
            source: 0.0,
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
    variant: u32,
    fused: bool,
    frames: [Arc<SuperEntry>; 2],
    sources: [[GpuBuf; 3]; 2],
    fields: (ClMem, ClMem),
    still: Option<StillMask>,
    layer: Option<Arc<Layer>>,
    key: (i64, bool, usize, usize),
    params: [KernelParams; 3],
    steps: Steps,
    masks: Vec<u8>,
}

struct FuseSlot {
    source: Option<(Arc<SuperEntry>, f32)>,
    group: Option<Group>,
}

#[derive(Default)]
struct Pending {
    group: Option<Group>,
    slots: Vec<FuseSlot>,
    frames: Vec<Arc<SuperEntry>>,
    layers: Vec<Arc<Layer>>,
    motion: Vec<Vec<u16>>,
    masks: Vec<Vec<u8>>,
}

struct Counts {
    free: usize,
    total: usize,
    debt: usize,
}

pub(super) struct Slots(Mutex<Counts>, Condvar);

impl Slots {
    pub(super) fn new(count: usize) -> Self {
        Self(
            Mutex::new(Counts {
                free: count,
                total: count,
                debt: 0,
            }),
            Condvar::new(),
        )
    }

    fn counts(&self) -> std::sync::MutexGuard<'_, Counts> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub(super) fn total(&self) -> usize {
        self.counts().total
    }

    pub(super) fn shrink(&self) -> bool {
        let mut counts = self.counts();
        if counts.total == 1 {
            return false;
        }
        counts.total -= 1;
        if counts.free > 0 {
            counts.free -= 1;
        } else {
            counts.debt += 1;
        }
        true
    }

    fn take(self: &Arc<Self>) -> Option<Slot> {
        let mut counts = self.counts();
        while counts.free == 0 {
            counts = self.1.wait(counts).ok()?;
        }
        counts.free -= 1;
        Some(Slot(Arc::clone(self)))
    }
}

pub struct Slot(Arc<Slots>);

impl Drop for Slot {
    fn drop(&mut self) {
        let mut counts = self.0.counts();
        if counts.debt > 0 {
            counts.debt -= 1;
        } else {
            counts.free += 1;
            self.0.1.notify_one();
        }
    }
}

pub struct BlendJob {
    _slot: Slot,
    buffers: Buffers,
    unit: RefCell<Unit>,
    width: usize,
    rows: usize,
    luma: usize,
    gamma: f32,
    fuse: bool,
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
            Some(Unit::new(kernel, queue, std::ptr::null_mut()))
        }
    }

    fn blend_kernels(&self, unit: &mut Unit) -> Option<[ClKernel; 3]> {
        if unit.blend.kernels.is_none() {
            let mut kernels = [std::ptr::null_mut(); 3];
            for (kernel, name) in
                kernels
                    .iter_mut()
                    .zip([c"blend_source", c"blend_resolve", c"blend_resolve_source"])
            {
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
            return super::exhausted();
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
        tail: Option<(&SuperEntry, f32)>,
    ) -> Option<()> {
        let luma = u32::try_from(luma).ok()?;
        let [_, plain, with_source] = self.blend_kernels(unit)?;
        let resolve = if tail.is_some() { with_source } else { plain };
        unsafe {
            let set = self.cl.SetKernelArg;
            let mut ready: ClEvent = std::ptr::null_mut();
            if let Some((entry, weight)) = tail {
                let image = entry.mems[1];
                let width = i32::try_from(buffers.len / entry.dims.1).ok()?;
                ready = entry.ready;
                if set(resolve, 5, size_of::<ClMem>(), (&raw const image).cast()) != CL_SUCCESS
                    || set(resolve, 6, size_of::<f32>(), (&raw const weight).cast()) != CL_SUCCESS
                    || set(resolve, 7, size_of::<i32>(), (&raw const width).cast()) != CL_SUCCESS
                {
                    return None;
                }
            }
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
            (note((self.cl.EnqueueNDRangeKernel)(
                unit.queue,
                resolve,
                1,
                std::ptr::null(),
                &raw const buffers.len,
                std::ptr::null(),
                u32::from(!ready.is_null()),
                if ready.is_null() {
                    std::ptr::null()
                } else {
                    (&raw const ready).cast()
                },
                std::ptr::null_mut(),
            )) == CL_SUCCESS)
                .then_some(())
        }
    }

    pub(super) unsafe fn release_buffers(&self, buffers: &Buffers) {
        unsafe {
            (self.cl.ReleaseMemObject)(buffers.sums);
            (self.cl.ReleaseMemObject)(buffers.bytes);
        }
    }

    fn release_job(&self, job: BlendJob, keep: bool) {
        let unit = job.unit.into_inner();
        if !keep {
            unsafe { note((self.cl.Finish)(unit.queue)) };
        }
        drop(job.pending);
        match self.blend_pool.lock() {
            Ok(mut pool) if keep && pool.len() < self.blend_slots.total() => pool.push(job.buffers),
            _ => unsafe { self.release_buffers(&job.buffers) },
        }
        match self.blend_units.lock() {
            Ok(mut pool) if pool.len() < self.blend_slots.total() => pool.push(unit),
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
        let slot = self.blend_slots.take()?;
        let unit = self.blend_unit()?;
        let reused = self.blend_pool.lock().ok().and_then(|mut pool| {
            let index = pool.iter().position(|buffers| buffers.len == len)?;
            Some(pool.swap_remove(index))
        });
        let fresh = reused.is_none();
        let created = || {
            let sums = unsafe { self.create_buffer(len * size_of::<f32>()) }?;
            let Some(bytes) = (unsafe { self.create_buffer(len) }) else {
                unsafe { (self.cl.ReleaseMemObject)(sums) };
                return None;
            };
            Some(Buffers { sums, bytes, len })
        };
        let Some(buffers) = reused.or_else(created) else {
            unsafe { self.release_unit(&unit) };
            return None;
        };
        let mut unit = unit;
        if fresh
            && unsafe { self.enqueue_resolve(&mut unit, &buffers, 1.0, 1.0, 0, None) }.is_none()
        {
            unsafe {
                self.release_buffers(&buffers);
                self.release_unit(&unit);
            }
            return None;
        }
        Some(BlendJob {
            _slot: slot,
            buffers,
            unit: RefCell::new(unit),
            width,
            rows,
            luma: height,
            gamma,
            fuse: self.fuse_fits && gamma.to_bits() == 1.0_f32.to_bits(),
            weight: Cell::new(0.0),
            taken: Cell::new(false),
            failed: Cell::new(false),
            pending: RefCell::default(),
        })
    }

    fn flush(&self, job: &BlendJob) -> Option<()> {
        self.retire(job)?;
        self.launch_slots(job)
    }

    fn retire(&self, job: &BlendJob) -> Option<()> {
        let fused = job.pending.borrow().group.as_ref().map(|group| group.fused);
        match fused {
            None => Some(()),
            Some(true) => {
                let full = {
                    let mut pending = job.pending.borrow_mut();
                    let group = pending.group.take()?;
                    match pending.slots.last_mut() {
                        Some(slot) if slot.group.is_none() => slot.group = Some(group),
                        _ => pending.slots.push(FuseSlot {
                            source: None,
                            group: Some(group),
                        }),
                    }
                    pending.slots.len() == MAX_SLOTS
                };
                if full {
                    self.launch_slots(job)
                } else {
                    Some(())
                }
            }
            Some(false) => {
                self.launch_slots(job)?;
                self.launch_current(job)
            }
        }
    }

    fn launch_slots(&self, job: &BlendJob) -> Option<()> {
        let mut pending = job.pending.borrow_mut();
        let slots = std::mem::take(&mut pending.slots);
        let Some(first) = slots.iter().find_map(|slot| slot.group.as_ref()) else {
            drop(pending);
            for (entry, weight) in slots.iter().filter_map(|slot| slot.source.as_ref()) {
                job.failed.set(true);
                let mut unit = job.unit.borrow_mut();
                unsafe { self.enqueue_source(job, &mut unit, entry, *weight) }?;
                job.failed.set(false);
            }
            return Some(());
        };
        job.failed.set(true);
        let mut unit = job.unit.borrow_mut();
        let kernel =
            self.variant_kernel(&mut unit.variant_kernels, first.variant | FUSE_VARIANT)?;
        let stride = i32::try_from(job.width).ok()?;
        let mut waits: Vec<ClEvent> = Vec::new();
        let mut args = Vec::with_capacity(MAX_SLOTS);
        for slot in 0..MAX_SLOTS {
            let entry = slots.get(slot);
            let group = entry.and_then(|slot| slot.group.as_ref());
            let shape = group.unwrap_or(first);
            let (source, weight) = match entry.and_then(|slot| slot.source.as_ref()) {
                Some((frame, weight)) => (frame.mems[1], *weight),
                None => (first.frames[0].mems[1], 0.0),
            };
            let mut steps = group.map_or_else(Steps::default, |group| group.steps);
            steps.source = weight;
            if let Some((frame, _)) = entry.and_then(|slot| slot.source.as_ref()) {
                waits.push(frame.ready);
            }
            if group.is_some() {
                waits.extend([shape.sources[0][0].1, shape.sources[1][0].1]);
            }
            args.push((source, shape.sources, shape.fields.0, steps));
        }
        waits.retain(|event| !event.is_null());
        waits.dedup();
        let queue = unit.queue;
        unsafe {
            let set = self.cl.SetKernelArg;
            for plane in 0..3 {
                let params = first.params[plane];
                let set_mem = |index: ClUint, mem: ClMem| {
                    set(kernel, index, size_of::<ClMem>(), (&raw const mem).cast()) == CL_SUCCESS
                };
                if !set_mem(0, job.buffers.sums)
                    || set(kernel, 1, size_of::<i32>(), (&raw const stride).cast()) != CL_SUCCESS
                {
                    return None;
                }
                for (at, (source, sources, vectors, steps)) in args.iter().enumerate() {
                    let base = 2 + 4 * at as ClUint;
                    for (offset, image) in
                        [*source, sources[0][plane].0, sources[1][plane].0, *vectors]
                            .into_iter()
                            .enumerate()
                    {
                        if !set_mem(base + offset as ClUint, image) {
                            return None;
                        }
                    }
                    if set(
                        kernel,
                        18 + 1 + at as ClUint,
                        size_of::<Steps>(),
                        (&raw const *steps).cast(),
                    ) != CL_SUCCESS
                    {
                        return None;
                    }
                }
                if set(
                    kernel,
                    18,
                    size_of::<KernelParams>(),
                    (&raw const params).cast(),
                ) != CL_SUCCESS
                {
                    return None;
                }
                let global = [
                    usize::try_from(params.width).ok()?,
                    usize::try_from(params.height).ok()?,
                ];
                let wait = if plane == 0 { &waits[..] } else { &[] };
                if note((self.cl.EnqueueNDRangeKernel)(
                    queue,
                    kernel,
                    2,
                    std::ptr::null(),
                    global.as_ptr(),
                    std::ptr::null(),
                    ClUint::try_from(wait.len()).ok()?,
                    if wait.is_empty() {
                        std::ptr::null()
                    } else {
                        wait.as_ptr().cast()
                    },
                    std::ptr::null_mut(),
                )) != CL_SUCCESS
                {
                    return None;
                }
            }
            (self.cl.Flush)(queue);
        }
        for slot in slots {
            if let Some(group) = slot.group {
                pending.frames.extend(group.frames);
            }
        }
        job.failed.set(false);
        Some(())
    }

    fn launch_current(&self, job: &BlendJob) -> Option<()> {
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
                if note((self.cl.EnqueueWriteImage)(
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
                )) != CL_SUCCESS
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
            if let Some(layer) = &group.layer {
                let base = 9 + ClUint::from(group.still.is_some());
                let [fit, cover] = layer.images();
                let raw = [group.sources[0][1].0, group.sources[1][1].0];
                for (index, mem) in (base..).zip([fit, cover, raw[0], raw[1]]) {
                    if set(kernel, index, size_of::<ClMem>(), (&raw const mem).cast()) != CL_SUCCESS
                    {
                        return None;
                    }
                }
            }
            pending.layers.extend(group.layer);
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
                    std::ptr::null_mut(),
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
        layer: Option<&Arc<Layer>>,
    ) -> Option<()> {
        let key = (motion_key, linear, motion_width, motion_height);
        let joins = job.pending.borrow().group.as_ref().is_some_and(|group| {
            group.key == key
                && group.still == still
                && group.layer.is_some() == layer.is_some()
                && (group.steps.count as usize) < MAX_STEPS
                && shared(group.params) == shared(params)
                && Arc::ptr_eq(&group.frames[0], &src0.entry)
                && Arc::ptr_eq(&group.frames[1], &src1.entry)
        });
        if !joins {
            let variant = Self::variant(&params[0], false)
                | BLEND_VARIANT
                | u32::from(still.is_some()) << 14
                | if layer.is_some() { LAYER_VARIANT } else { 0 };
            let fused = job.fuse
                && matches!(params[0].algorithm, 1 | 2 | 11 | 13)
                && params[0].has_sad == 0
                && still.is_none()
                && layer.is_none();
            self.retire(job)?;
            let compatible = fused
                && job
                    .pending
                    .borrow()
                    .slots
                    .iter()
                    .find_map(|slot| slot.group.as_ref())
                    .is_none_or(|group| {
                        group.variant == variant
                            && group.key.1 == linear
                            && group.key.2 == motion_width
                            && group.key.3 == motion_height
                            && shared(group.params) == shared(params)
                    });
            if !compatible {
                self.launch_slots(job)?;
            }
            let slot = if fused {
                let pending = job.pending.borrow();
                match pending.slots.last() {
                    Some(last) if last.group.is_none() => pending.slots.len() - 1,
                    _ => pending.slots.len(),
                }
            } else {
                0
            };
            let mut unit = job.unit.borrow_mut();
            let unit = &mut *unit;
            let kernel = if fused {
                Some(std::ptr::null_mut())
            } else {
                self.variant_kernel(&mut unit.variant_kernels, variant)
            };
            let fields = unsafe {
                self.upload_motion(
                    unit,
                    2 * slot,
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
                variant,
                fused,
                frames: [Arc::clone(&src0.entry), Arc::clone(&src1.entry)],
                sources: [src0.sources(linear), src1.sources(linear)],
                fields: fields?,
                still,
                layer: layer.cloned(),
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
        if job.fuse {
            self.retire(job)?;
        } else {
            self.flush(job)?;
        }
        let entry = match key {
            Some(key) => self.cache_frame(key, planes)?.entry,
            None => Arc::new(self.upload_frame(planes)?),
        };
        if entry.dims != (job.width, job.rows) {
            return None;
        }
        let weight = job.weight.get();
        job.pending.borrow_mut().frames.push(Arc::clone(&entry));
        if job.fuse {
            if job.pending.borrow().slots.len() == MAX_SLOTS {
                self.launch_slots(job)?;
            }
            job.pending.borrow_mut().slots.push(FuseSlot {
                source: Some((entry, weight)),
                group: None,
            });
        } else {
            let mut unit = job.unit.borrow_mut();
            unsafe { self.enqueue_source(job, &mut unit, &entry, weight) }?;
        }
        if key.is_none() {
            self.flush(job)?;
            let queue = job.unit.borrow().queue;
            if unsafe { note((self.cl.Finish)(queue)) } != CL_SUCCESS {
                return None;
            }
            let mut pending = job.pending.borrow_mut();
            pending.frames.clear();
            pending.motion.clear();
            pending.masks.clear();
        }
        job.taken.set(true);
        Some(())
    }

    unsafe fn enqueue_source(
        &self,
        job: &BlendJob,
        unit: &mut Unit,
        entry: &SuperEntry,
        weight: f32,
    ) -> Option<()> {
        let [source, _, _] = self.blend_kernels(unit)?;
        let stride = i32::try_from(job.width).ok()?;
        let luma = i32::try_from(job.luma).ok()?;
        let image = entry.mems[1];
        let ready: ClEvent = entry.ready;
        let global = [job.width, job.rows];
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
            if note((self.cl.EnqueueNDRangeKernel)(
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
            )) != CL_SUCCESS
            {
                job.failed.set(true);
                return None;
            }
            (self.cl.Flush)(unit.queue);
        }
        Some(())
    }

    pub fn admit(&self) -> Option<Slot> {
        (self.mark().epoch > 0)
            .then(|| self.blend_slots.take())
            .flatten()
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
        let fits = planes[0].width <= job.width
            && planes[0].height + planes[1].height == job.rows
            && planes[1].width + planes[2].width <= job.width
            && planes.iter().all(|plane| {
                plane.width <= plane.stride
                    && plane.height > 0
                    && plane.data.len() >= plane.stride * (plane.height - 1) + plane.width
            });
        let mut tail = None;
        if !fits || self.retire(&job).is_none() {
            self.blend_abort(job);
            return None;
        }
        {
            let mut pending = job.pending.borrow_mut();
            match pending.slots.pop_if(|slot| slot.group.is_none()) {
                Some(FuseSlot {
                    source: Some(last), ..
                }) => tail = Some(last),
                Some(slot) => pending.slots.push(slot),
                None => {}
            }
        }
        if self.launch_slots(&job).is_none() || job.failed() {
            self.blend_abort(job);
            return None;
        }
        let mut unit = job.unit.borrow_mut();
        let queue = unit.queue;
        let luma_rows = planes[0].height;
        let chroma_width = planes[1].width;
        let mut done: ClEvent = std::ptr::null_mut();
        let read = unsafe {
            self.enqueue_resolve(
                &mut unit,
                &job.buffers,
                divisor,
                job.gamma,
                job.width * job.luma,
                tail.as_ref().map(|(entry, weight)| (&**entry, *weight)),
            )
            .and_then(|()| {
                for (index, plane) in planes.into_iter().enumerate() {
                    let origin = match index {
                        0 => [0, 0, 0],
                        1 => [0, luma_rows, 0],
                        _ => [chroma_width, luma_rows, 0],
                    };
                    let region = [plane.width, plane.height, 1];
                    if note((self.cl.EnqueueReadBufferRect)(
                        queue,
                        job.buffers.bytes,
                        CL_FALSE,
                        origin.as_ptr(),
                        [0; 3].as_ptr(),
                        region.as_ptr(),
                        job.width,
                        0,
                        plane.stride,
                        0,
                        plane.data.as_mut_ptr().cast(),
                        0,
                        std::ptr::null(),
                        if index == 2 {
                            (&raw mut done).cast()
                        } else {
                            std::ptr::null_mut()
                        },
                    )) != CL_SUCCESS
                    {
                        note((self.cl.Finish)(queue));
                        return None;
                    }
                }
                Some(())
            })
        };
        drop(unit);
        if read.is_none() {
            self.blend_abort(job);
            return None;
        }
        if !unsafe { self.wait_event(queue, done) } {
            self.blend_abort(job);
            return None;
        }
        self.release_job(job, true);
        Some(())
    }
}
