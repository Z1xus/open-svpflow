use core::ffi::c_void;

use super::{
    CL_FALSE, CL_SUCCESS, CL_UNORM_INT8, ClCommandQueue, ClKernel, ClMem, ClUint, GpuContext, Unit,
};
use svpflow_core::smooth::CoverLayout;

pub struct Cover<'a> {
    pub key: i64,
    pub vectors: &'a dyn Fn(&mut Vec<i16>),
    pub layout: &'a CoverLayout,
    pub times: [i32; 2],
}

pub(super) struct UnitCover {
    kernels: Option<[ClKernel; 3]>,
    vectors: (ClMem, usize),
    points: (ClMem, usize),
    table: (ClMem, usize),
    key: Option<i64>,
    host: Vec<i16>,
    table_host: Vec<u8>,
}

impl Default for UnitCover {
    fn default() -> Self {
        let none = (std::ptr::null_mut(), 0);
        Self {
            kernels: None,
            vectors: none,
            points: none,
            table: none,
            key: None,
            host: Vec::new(),
            table_host: Vec::new(),
        }
    }
}

impl GpuContext {
    pub(super) unsafe fn release_cover(&self, cover: &UnitCover) {
        unsafe {
            for kernel in cover.kernels.into_iter().flatten() {
                (self.cl.ReleaseKernel)(kernel);
            }
            for (mem, _) in [cover.vectors, cover.points, cover.table] {
                if !mem.is_null() {
                    (self.cl.ReleaseMemObject)(mem);
                }
            }
        }
    }

    unsafe fn cover_kernels(&self, cover: &mut UnitCover) -> Option<[ClKernel; 3]> {
        if let Some(kernels) = cover.kernels {
            return Some(kernels);
        }
        let mut kernels = [std::ptr::null_mut(); 3];
        for (kernel, name) in
            kernels
                .iter_mut()
                .zip([c"cover_clear", c"cover_scatter", c"cover_finish"])
        {
            let mut err = 0;
            *kernel = unsafe { (self.cl.CreateKernel)(self.program, name.as_ptr(), &raw mut err) };
            if err != CL_SUCCESS || kernel.is_null() {
                return None;
            }
        }
        cover.kernels = Some(kernels);
        Some(kernels)
    }

    unsafe fn write_buffer(
        &self,
        queue: ClCommandQueue,
        mem: ClMem,
        data: *const c_void,
        bytes: usize,
    ) -> Option<()> {
        let status = unsafe {
            (self.cl.EnqueueWriteBuffer)(
                queue,
                mem,
                CL_FALSE,
                0,
                bytes,
                data.cast_mut(),
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        (status == CL_SUCCESS).then_some(())
    }

    unsafe fn launch(
        &self,
        queue: ClCommandQueue,
        kernel: ClKernel,
        args: &[(usize, *const c_void)],
        global: &[usize],
    ) -> Option<()> {
        for (index, &(size, value)) in args.iter().enumerate() {
            let status = unsafe { (self.cl.SetKernelArg)(kernel, index as ClUint, size, value) };
            if status != CL_SUCCESS {
                return None;
            }
        }
        let status = unsafe {
            (self.cl.EnqueueNDRangeKernel)(
                queue,
                kernel,
                global.len() as ClUint,
                std::ptr::null(),
                global.as_ptr(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        (status == CL_SUCCESS).then_some(())
    }

    pub(super) unsafe fn bind_cover(&self, unit: &mut Unit, cover: &Cover<'_>) -> Option<ClMem> {
        let layout = cover.layout;
        let (width, height) = (layout.size.0 as usize, layout.size.1 as usize);
        let (grid_w, grid_h) = (layout.grid.0 as usize, layout.grid.1 as usize);
        let queue = unit.queue;
        let kernels = unsafe { self.cover_kernels(&mut unit.cover) }?;
        let mask = unsafe { self.ensure_image(&mut unit.mask, width, height, CL_UNORM_INT8) }?;
        let state = &mut unit.cover;
        let vectors_bytes = 2 * grid_w * grid_h * 2 * size_of::<i16>();
        let vectors_old = state.vectors.0;
        let vectors = unsafe { self.ensure_buffer(&mut state.vectors, vectors_bytes) }?;
        if state.key != Some(cover.key) || vectors != vectors_old {
            (cover.vectors)(&mut state.host);
            if state.host.len() * size_of::<i16>() != vectors_bytes {
                return None;
            }
            unsafe {
                self.write_buffer(queue, vectors, state.host.as_ptr().cast(), vectors_bytes)
            }?;
            state.key = Some(cover.key);
        }
        let table_old = state.table.0;
        let table = unsafe { self.ensure_buffer(&mut state.table, layout.table.len()) }?;
        if state.table_host != layout.table || table != table_old {
            state.table_host.clone_from(&layout.table);
            unsafe {
                self.write_buffer(
                    queue,
                    table,
                    state.table_host.as_ptr().cast(),
                    state.table_host.len(),
                )
            }?;
        }
        let stride = width + 2;
        let plane = stride * (height + 2);
        let points =
            unsafe { self.ensure_buffer(&mut state.points, 2 * plane * size_of::<i32>()) }?;
        let ptr = |value: &ClMem| (size_of::<ClMem>(), std::ptr::from_ref(value).cast());
        let int = |value: &i32| (size_of::<i32>(), std::ptr::from_ref(value).cast());
        let count = i32::try_from(2 * plane).ok()?;
        unsafe {
            self.launch(
                queue,
                kernels[0],
                &[ptr(&points), int(&count)],
                &[2 * plane],
            )?;
            let [step_x, step_y] = [layout.step.0, layout.step.1];
            let [block_w, block_h] = [layout.block.0, layout.block.1];
            let [w, h, gw, gh] = [layout.size.0, layout.size.1, layout.grid.0, layout.grid.1];
            self.launch(
                queue,
                kernels[1],
                &[
                    ptr(&vectors),
                    ptr(&points),
                    int(&gw),
                    int(&gh),
                    int(&w),
                    int(&h),
                    int(&step_x),
                    int(&step_y),
                    int(&block_w),
                    int(&block_h),
                    int(&layout.denom),
                    int(&cover.times[0]),
                    int(&cover.times[1]),
                ],
                &[grid_w, grid_h, 2],
            )?;
            let top = i32::try_from(layout.table.len()).ok()? - 1;
            self.launch(
                queue,
                kernels[2],
                &[
                    ptr(&points),
                    ptr(&table),
                    ptr(&mask),
                    int(&w),
                    int(&h),
                    int(&layout.area),
                    int(&top),
                    (size_of::<f32>(), (&raw const layout.scale).cast()),
                ],
                &[width, height],
            )?;
        }
        Some(mask)
    }
}
