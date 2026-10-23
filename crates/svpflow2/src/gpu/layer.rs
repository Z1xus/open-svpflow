use std::sync::Arc;

use super::{
    CL_FALSE, CL_HALF_FLOAT, CL_R, CL_RGBA, CL_SUCCESS, CL_TRUE, CL_UNORM_INT8, CL_UNORM_INT16,
    ClCommandQueue, ClKernel, ClMem, ClUint, FnRelease, GpuContext, ImageSlot, KernelParams,
    SuperHandle, note,
};

pub(super) const LAYER_VARIANT: u32 = 1 << 15;
const STATS_CAP: usize = 4;
const FIT_CAP: usize = 6;

struct Owned(ClMem, FnRelease);

unsafe impl Send for Owned {}
unsafe impl Sync for Owned {}
impl Drop for Owned {
    fn drop(&mut self) {
        unsafe { (self.1)(self.0) };
    }
}

pub struct Layer {
    fit: Owned,
    cover: Owned,
}

pub struct Failed;

pub struct LayerPair {
    pub first: SuperHandle,
    pub second: SuperHandle,
    pub packed: Vec<u16>,
}

#[derive(Default)]
pub struct LayerHeld {
    stats: Vec<(i64, Arc<Owned>, i32)>,
    fits: Vec<(i64, Arc<Layer>)>,
}

pub(super) struct LayerState {
    queue: ClCommandQueue,
    kernels: [ClKernel; 4],
    rough: ImageSlot,
    vectors: [ImageSlot; 4],
    found: (ClMem, usize),
    stats: Vec<(i64, Arc<Owned>, i32)>,
    fits: Vec<(i64, Arc<Layer>)>,
    held: Vec<(i64, std::sync::Weak<Layer>)>,
}

unsafe impl Send for LayerState {}

impl LayerState {
    pub(super) fn release_cached(&mut self) {
        self.stats.clear();
        self.fits.clear();
        self.held.clear();
    }

    fn keep(&mut self, chunk: i64, fit: &Arc<Layer>, cap: usize) {
        if self.fits.len() >= cap {
            let (old, gone) = self.fits.remove(0);
            self.held
                .retain(|(key, fit)| *key != old && *key != chunk && fit.strong_count() > 0);
            self.held.push((old, Arc::downgrade(&gone)));
        }
        self.fits.push((chunk, Arc::clone(fit)));
    }

    pub(super) const fn new() -> Self {
        Self {
            queue: std::ptr::null_mut(),
            kernels: [std::ptr::null_mut(); 4],
            rough: ImageSlot::EMPTY,
            vectors: [ImageSlot::EMPTY; 4],
            found: (std::ptr::null_mut(), 0),
            stats: Vec::new(),
            fits: Vec::new(),
            held: Vec::new(),
        }
    }
}

fn distances(cells: &mut [u8], width: usize) {
    for cell in cells.iter_mut() {
        *cell = if *cell == 0 { 255 } else { 0 };
    }
    let near = |cells: &[u8], at: usize, others: [Option<usize>; 4]| {
        others.into_iter().flatten().fold(cells[at], |best, other| {
            best.min(cells[other].saturating_add(1))
        })
    };
    let height = cells.len() / width;
    for y in 0..height {
        for x in 0..width {
            let at = y * width + x;
            let (left, right, up) = (x > 0, x + 1 < width, y > 0);
            cells[at] = near(
                cells,
                at,
                [
                    left.then(|| at - 1),
                    up.then(|| at - width),
                    (up && left).then(|| at - width - 1),
                    (up && right).then(|| at - width + 1),
                ],
            );
        }
    }
    for y in (0..height).rev() {
        for x in (0..width).rev() {
            let at = y * width + x;
            let (left, right, down) = (x > 0, x + 1 < width, y + 1 < height);
            cells[at] = near(
                cells,
                at,
                [
                    right.then(|| at + 1),
                    down.then(|| at + width),
                    (down && right).then(|| at + width + 1),
                    (down && left).then(|| at + width - 1),
                ],
            );
        }
    }
}

impl Layer {
    pub(super) fn images(&self) -> [ClMem; 2] {
        [self.fit.0, self.cover.0]
    }
}

impl GpuContext {
    pub(super) unsafe fn release_layer(&self) {
        if let Ok(mut state) = self.layer.lock() {
            state.stats.clear();
            state.fits.clear();
            state.held.clear();
            unsafe {
                for kernel in state.kernels {
                    if !kernel.is_null() {
                        (self.cl.ReleaseKernel)(kernel);
                    }
                }
                for mem in state
                    .vectors
                    .iter()
                    .map(|slot| slot.mem)
                    .chain([state.rough.mem, state.found.0])
                {
                    if !mem.is_null() {
                        (self.cl.ReleaseMemObject)(mem);
                    }
                }
                if !state.queue.is_null() {
                    (self.cl.ReleaseCommandQueue)(state.queue);
                }
            }
        }
    }

    pub fn layer_hold(&self, chunk: i32, last: i32, held: &mut LayerHeld) -> Vec<i32> {
        let near = (chunk - 1).max(0)..=(chunk + 1).min(last);
        let Ok(state) = self.layer.lock() else {
            return near.collect();
        };
        let key = i64::from(chunk);
        let fit = state.fits.iter().find(|(other, _)| *other == key);
        let fit = fit.map(|(_, fit)| Arc::clone(fit)).or_else(|| {
            let gone = state.held.iter().find(|(other, _)| *other == key);
            gone.and_then(|(_, fit)| fit.upgrade())
        });
        if let Some(fit) = fit {
            if !held.fits.iter().any(|(other, _)| *other == key) {
                held.fits.push((key, fit));
            }
            return Vec::new();
        }
        near.filter(|&chunk| {
            let key = i64::from(chunk);
            match state.stats.iter().find(|(other, _, _)| *other == key) {
                Some((_, sums, count)) => {
                    if !held.stats.iter().any(|(other, _, _)| *other == key) {
                        held.stats.push((key, Arc::clone(sums), *count));
                    }
                    false
                }
                None => true,
            }
        })
        .collect()
    }

    unsafe fn set<T>(&self, kernel: ClKernel, index: ClUint, value: &T) -> bool {
        unsafe {
            (self.cl.SetKernelArg)(
                kernel,
                index,
                size_of::<T>(),
                std::ptr::from_ref(value).cast(),
            ) == CL_SUCCESS
        }
    }

    unsafe fn run(&self, queue: ClCommandQueue, kernel: ClKernel, size: [usize; 2]) -> bool {
        unsafe {
            note((self.cl.EnqueueNDRangeKernel)(
                queue,
                kernel,
                2,
                std::ptr::null(),
                size.as_ptr(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            )) == CL_SUCCESS
        }
    }

    unsafe fn layer_setup(&self, state: &mut LayerState) -> Option<()> {
        if state.queue.is_null() {
            let mut err = 0;
            state.queue =
                unsafe { (self.cl.CreateCommandQueue)(self.context, self.device, 0, &raw mut err) };
            if err != CL_SUCCESS || state.queue.is_null() {
                state.queue = std::ptr::null_mut();
                return None;
            }
        }
        for (kernel, name) in state.kernels.iter_mut().zip([
            c"layer_stats",
            c"layer_fit",
            c"layer_clean",
            c"layer_cover",
        ]) {
            if kernel.is_null() {
                let mut err = 0;
                *kernel =
                    unsafe { (self.cl.CreateKernel)(self.program, name.as_ptr(), &raw mut err) };
                if err != CL_SUCCESS || kernel.is_null() {
                    *kernel = std::ptr::null_mut();
                    return None;
                }
            }
        }
        Some(())
    }

    unsafe fn layer_stats(
        &self,
        state: &mut LayerState,
        sums: ClMem,
        group: &[LayerPair],
        params: &[KernelParams; 3],
        grid: (usize, usize),
        reset: bool,
    ) -> Option<()> {
        let count = grid.0 * grid.1;
        if group.iter().any(|pair| pair.packed.len() < count * 4) {
            return None;
        }
        let (queue, kernel) = (state.queue, state.kernels[0]);
        let mut frames: Vec<ClMem> = group.iter().map(|pair| pair.first.bufs()[1].0).collect();
        frames.push(group[group.len() - 1].second.bufs()[1].0);
        frames.resize(5, frames[0]);
        unsafe {
            let mut vectors = [std::ptr::null_mut(); 4];
            for (index, pair) in group.iter().enumerate() {
                let image =
                    self.ensure_image(&mut state.vectors[index], grid.0, grid.1, CL_UNORM_INT16)?;
                let origin = [0usize; 3];
                let region = [grid.0, grid.1, 1];
                if note((self.cl.EnqueueWriteImage)(
                    queue,
                    image,
                    CL_FALSE,
                    origin.as_ptr(),
                    region.as_ptr(),
                    grid.0 * 8,
                    0,
                    pair.packed.as_ptr().cast(),
                    0,
                    std::ptr::null(),
                    std::ptr::null_mut(),
                )) != CL_SUCCESS
                {
                    return None;
                }
                vectors[index] = image;
            }
            let spare = vectors[0];
            for image in &mut vectors {
                if image.is_null() {
                    *image = spare;
                }
            }
            let mut waits = Vec::with_capacity(5);
            for pair in group {
                for handle in [&pair.first, &pair.second] {
                    let event = handle.bufs()[1].1;
                    if !event.is_null() && !waits.contains(&event) {
                        waits.push(event);
                    }
                }
            }
            if !waits.is_empty()
                && (self.cl.WaitForEvents)(waits.len() as ClUint, waits.as_ptr().cast())
                    != CL_SUCCESS
            {
                return None;
            }
            let luma = usize::try_from(params[0].width * params[0].height).ok()?;
            let chroma = usize::try_from(params[1].width * params[1].height).ok()?;
            let mut ready = self.set(kernel, 0, &sums);
            for (index, image) in frames.iter().chain(&vectors).enumerate() {
                ready &= self.set(kernel, 1 + index as ClUint, image);
            }
            if !ready {
                return None;
            }
            let (pairs, reset) = (i32::try_from(group.len()).ok()?, i32::from(reset));
            for (plane, base, channels) in [
                (0, 0, 5i32),
                (1, luma * 5, 2),
                (2, luma * 5 + chroma * 2, 2),
            ] {
                let p = params[plane];
                let size = [
                    usize::try_from(p.width).ok()?,
                    usize::try_from(p.height).ok()?,
                ];
                let ok = self.set(kernel, 10, &p)
                    && self.set(kernel, 11, &i32::try_from(base).ok()?)
                    && self.set(kernel, 12, &channels)
                    && self.set(kernel, 13, &reset)
                    && self.set(kernel, 14, &pairs)
                    && self.run(queue, kernel, size);
                if !ok {
                    return None;
                }
            }
        }
        Some(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn layer_fit(
        &self,
        chunk: i64,
        last: i64,
        params: &[KernelParams; 3],
        grid: (usize, usize),
        limit: f32,
        held: &LayerHeld,
        pairs: impl FnMut(i64) -> Option<Vec<LayerPair>>,
    ) -> Result<Option<Arc<Layer>>, Failed> {
        if self.hi10 {
            return Ok(None);
        }
        self.fit_chunk(chunk, last, params, grid, limit, held, pairs)
            .map(Some)
            .ok_or(Failed)
    }

    fn fit_chunk(
        &self,
        chunk: i64,
        last: i64,
        params: &[KernelParams; 3],
        grid: (usize, usize),
        limit: f32,
        held: &LayerHeld,
        mut pairs: impl FnMut(i64) -> Option<Vec<LayerPair>>,
    ) -> Option<Arc<Layer>> {
        if let Some((_, fit)) = held.fits.iter().find(|(key, _)| *key == chunk) {
            return Some(Arc::clone(fit));
        }
        let mut state = self.layer.lock().ok()?;
        let (stats_cap, fit_cap) = if self.mark().epoch == 0 {
            (STATS_CAP, FIT_CAP)
        } else {
            (3, 2)
        };
        if let Some(pos) = state.fits.iter().position(|(key, _)| *key == chunk) {
            let entry = state.fits.remove(pos);
            let fit = Arc::clone(&entry.1);
            state.fits.push(entry);
            return Some(fit);
        }
        let gone = state.held.iter().find(|(key, _)| *key == chunk);
        if let Some(fit) = gone.and_then(|(_, fit)| fit.upgrade()) {
            state.keep(chunk, &fit, fit_cap);
            return Some(fit);
        }
        unsafe { self.layer_setup(&mut state) }?;
        let width = usize::try_from(params[0].width).ok()?;
        let height = usize::try_from(params[0].height).ok()?;
        let chroma = usize::try_from(params[1].width * params[1].height).ok()?;
        let bytes = (width * height * 5 + chroma * 4) * size_of::<f32>();
        let mut parts = [None, None, None];
        for (slot, key) in parts.iter_mut().zip([chunk - 1, chunk, chunk + 1]) {
            if key < 0 || key > last {
                continue;
            }
            let found = held
                .stats
                .iter()
                .chain(&state.stats)
                .find(|(k, _, _)| *k == key);
            if let Some(found) = found {
                *slot = Some((Arc::clone(&found.1), found.2));
                continue;
            }
            let pairs = pairs(key)?;
            let mem = unsafe { self.create_buffer(bytes) }.or_else(|| {
                state.release_cached();
                unsafe { self.create_buffer(bytes) }
            })?;
            let sums = Arc::new(Owned(mem, self.cl.ReleaseMemObject));
            let stats = pairs
                .chunks(4)
                .enumerate()
                .try_for_each(|(index, group)| unsafe {
                    self.layer_stats(&mut state, mem, group, params, grid, index == 0)
                });
            if unsafe { note((self.cl.Finish)(state.queue)) } != CL_SUCCESS || stats.is_none() {
                return None;
            }
            let count = i32::try_from(pairs.len() * 2).ok()?;
            if state.stats.len() >= stats_cap {
                state.stats.remove(0);
            }
            state.stats.push((key, Arc::clone(&sums), count));
            *slot = Some((sums, count));
        }
        let own = parts[1].clone()?;
        let cells = (width.div_ceil(8), height.div_ceil(8));
        let fit = unsafe { self.create_image(CL_RGBA, CL_HALF_FLOAT, width, height) }?;
        let cover = unsafe { self.create_image(CL_R, CL_UNORM_INT8, cells.0, cells.1) }?;
        let made = Layer {
            fit: Owned(fit.mem, self.cl.ReleaseMemObject),
            cover: Owned(cover.mem, self.cl.ReleaseMemObject),
        };
        std::mem::forget(fit);
        std::mem::forget(cover);
        let [fit_kernel, clean_kernel, cover_kernel] =
            [state.kernels[1], state.kernels[2], state.kernels[3]];
        let rough = unsafe { self.ensure_image(&mut state.rough, width, height, CL_HALF_FLOAT) }?;
        let mut gaps = vec![0u8; cells.0 * cells.1];
        let found = unsafe { self.ensure_buffer(&mut state.found, gaps.len()) }?;
        let (w, h) = (params[0].width, params[0].height);
        unsafe {
            let mut ready = true;
            for (index, part) in (0..).zip(&parts) {
                let (mem, count) = part.as_ref().map_or((own.0.0, 0), |(sums, n)| (sums.0, *n));
                ready &=
                    self.set(fit_kernel, index, &mem) && self.set(fit_kernel, index + 3, &count);
            }
            ready = ready
                && self.set(fit_kernel, 6, &rough)
                && self.set(fit_kernel, 7, &w)
                && self.set(fit_kernel, 8, &h)
                && self.set(fit_kernel, 9, &limit)
                && self.set(fit_kernel, 10, &params[1].width)
                && self.set(fit_kernel, 11, &params[1].height)
                && self.run(state.queue, fit_kernel, [width, height])
                && self.set(clean_kernel, 0, &rough)
                && self.set(clean_kernel, 1, &made.fit.0)
                && self.run(state.queue, clean_kernel, [width, height])
                && self.set(cover_kernel, 0, &made.fit.0)
                && self.set(cover_kernel, 1, &found)
                && self.set(cover_kernel, 2, &w)
                && self.set(cover_kernel, 3, &h)
                && self.run(state.queue, cover_kernel, [cells.0, cells.1])
                && note((self.cl.EnqueueReadBuffer)(
                    state.queue,
                    found,
                    CL_TRUE,
                    0,
                    gaps.len(),
                    gaps.as_mut_ptr().cast(),
                    0,
                    std::ptr::null(),
                    std::ptr::null_mut(),
                )) == CL_SUCCESS;
            if !ready {
                return None;
            }
            distances(&mut gaps, cells.0);
            let origin = [0usize; 3];
            let region = [cells.0, cells.1, 1];
            if note((self.cl.EnqueueWriteImage)(
                state.queue,
                made.cover.0,
                CL_TRUE,
                origin.as_ptr(),
                region.as_ptr(),
                cells.0,
                0,
                gaps.as_ptr().cast(),
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            )) != CL_SUCCESS
            {
                return None;
            }
        }
        let made = Arc::new(made);
        state.keep(chunk, &made, fit_cap);
        Some(made)
    }
}
