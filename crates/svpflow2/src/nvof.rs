#![allow(unsafe_code)]
#![allow(non_snake_case)]

use core::ffi::c_void;
use std::sync::{Arc, Mutex};

use crate::metadata;

type CuResult = i32;
type CuDevice = i32;
type CuContext = *mut c_void;
type CuArray = *mut c_void;
type CuStream = *mut c_void;

type NvStatus = i32;
type NvHandle = *mut c_void;
type NvBuffer = *mut c_void;

const CUDA_SUCCESS: CuResult = 0;
const NV_SUCCESS: NvStatus = 0;
const NV_API_VERSION_1: u32 = 0x10;
const NV_BUFFER_TYPE_CUARRAY: i32 = 1;
const NV_USAGE_INPUT: i32 = 1;
const NV_USAGE_OUTPUT: i32 = 2;
const NV_USAGE_COST: i32 = 4;
const NV_FORMAT_NV12: i32 = 2;
const NV_FORMAT_SHORT2: i32 = 5;
const NV_FORMAT_UINT: i32 = 6;
const NV_GRID_4: i32 = 4;
const NV_MODE_OPTICAL_FLOW: i32 = 1;
const NV_PERF_SLOW: i32 = 5;
const NV_PERF_MEDIUM: i32 = 10;
const NV_PERF_FAST: i32 = 20;
const CU_MEMORYTYPE_HOST: i32 = 1;
const CU_MEMORYTYPE_ARRAY: i32 = 3;
const CACHE_CAPACITY: usize = 16;

#[derive(Debug, Clone, Copy)]
pub(crate) enum InitError {
    CudaUnavailable,
    Failed(i32),
}

#[repr(C)]
#[derive(Default)]
struct CudaMemcpy2D {
    src_x_in_bytes: usize,
    src_y: usize,
    src_memory_type: i32,
    src_host: *const c_void,
    src_device: u64,
    src_array: CuArray,
    src_pitch: usize,
    dst_x_in_bytes: usize,
    dst_y: usize,
    dst_memory_type: i32,
    dst_host: *mut c_void,
    dst_device: u64,
    dst_array: CuArray,
    dst_pitch: usize,
    width_in_bytes: usize,
    height: usize,
}

type CuInit = unsafe extern "system" fn(u32) -> CuResult;
type CuDeviceGetCount = unsafe extern "system" fn(*mut i32) -> CuResult;
type CuDeviceGet = unsafe extern "system" fn(*mut CuDevice, i32) -> CuResult;
type CuCtxCreate = unsafe extern "system" fn(*mut CuContext, u32, CuDevice) -> CuResult;
type CuCtxDestroy = unsafe extern "system" fn(CuContext) -> CuResult;
type CuCtxPush = unsafe extern "system" fn(CuContext) -> CuResult;
type CuCtxPop = unsafe extern "system" fn(*mut CuContext) -> CuResult;
type CuMemcpy2D = unsafe extern "system" fn(*const CudaMemcpy2D) -> CuResult;

struct CudaApi {
    _lib: libloading::Library,
    Init: CuInit,
    DeviceGetCount: CuDeviceGetCount,
    DeviceGet: CuDeviceGet,
    CtxCreate: CuCtxCreate,
    CtxDestroy: CuCtxDestroy,
    CtxPush: CuCtxPush,
    CtxPop: CuCtxPop,
    Memcpy2D: CuMemcpy2D,
}

impl CudaApi {
    unsafe fn load() -> Result<Self, InitError> {
        let lib = unsafe { libloading::Library::new("nvcuda.dll") }
            .map_err(|_| InitError::CudaUnavailable)?;
        macro_rules! sym {
            ($name:literal) => {
                *unsafe {
                    lib.get(concat!($name, "\0").as_bytes())
                        .map_err(|_| InitError::CudaUnavailable)?
                }
            };
        }
        Ok(Self {
            Init: sym!("cuInit"),
            DeviceGetCount: sym!("cuDeviceGetCount"),
            DeviceGet: sym!("cuDeviceGet"),
            CtxCreate: sym!("cuCtxCreate_v2"),
            CtxDestroy: sym!("cuCtxDestroy_v2"),
            CtxPush: sym!("cuCtxPushCurrent_v2"),
            CtxPop: sym!("cuCtxPopCurrent_v2"),
            Memcpy2D: sym!("cuMemcpy2D_v2"),
            _lib: lib,
        })
    }
}

#[repr(C)]
#[derive(Default)]
struct NvInitParamsV1 {
    width: u32,
    height: u32,
    out_grid_size: i32,
    hint_grid_size: i32,
    mode: i32,
    perf_level: i32,
    enable_external_hints: i32,
    enable_output_cost: i32,
    private_data: *mut c_void,
}

#[repr(C)]
#[derive(Default)]
struct NvBufferDescriptor {
    width: u32,
    height: u32,
    usage: i32,
    format: i32,
}

#[repr(C)]
#[derive(Default)]
struct NvExecuteInputV1 {
    input: NvBuffer,
    reference: NvBuffer,
    external_hints: NvBuffer,
    disable_temporal_hints: i32,
    padding: u32,
    private_data: *mut c_void,
}

#[repr(C)]
#[derive(Default)]
struct NvExecuteOutput {
    output: NvBuffer,
    cost: NvBuffer,
    private_data: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NvBufferStride {
    x_bytes: u32,
    y_bytes: u32,
}

#[repr(C)]
#[derive(Default)]
struct NvStrideInfo {
    planes: [NvBufferStride; 3],
    num_planes: u32,
}

type NvCreateFlow = unsafe extern "system" fn(CuContext, *mut NvHandle) -> NvStatus;
type NvInit = unsafe extern "system" fn(NvHandle, *const NvInitParamsV1) -> NvStatus;
type NvCreateBuffer =
    unsafe extern "system" fn(NvHandle, *const NvBufferDescriptor, i32, *mut NvBuffer) -> NvStatus;
type NvGetArray = unsafe extern "system" fn(NvBuffer) -> CuArray;
type NvGetDevicePtr = unsafe extern "system" fn(NvBuffer) -> u64;
type NvGetStride = unsafe extern "system" fn(NvBuffer, *mut NvStrideInfo) -> NvStatus;
type NvSetStreams = unsafe extern "system" fn(NvHandle, CuStream, CuStream) -> NvStatus;
type NvExecute =
    unsafe extern "system" fn(NvHandle, *const NvExecuteInputV1, *mut NvExecuteOutput) -> NvStatus;
type NvDestroyBuffer = unsafe extern "system" fn(NvBuffer) -> NvStatus;
type NvDestroy = unsafe extern "system" fn(NvHandle) -> NvStatus;
type NvGetLastError = unsafe extern "system" fn(NvHandle, *mut u8, *mut u32) -> NvStatus;
type NvGetCaps = unsafe extern "system" fn(NvHandle, i32, *mut u32, *mut u32) -> NvStatus;

#[repr(C)]
#[derive(Default)]
struct NvApiFunctions {
    CreateFlow: Option<NvCreateFlow>,
    Init: Option<NvInit>,
    CreateBuffer: Option<NvCreateBuffer>,
    GetArray: Option<NvGetArray>,
    GetDevicePtr: Option<NvGetDevicePtr>,
    GetStride: Option<NvGetStride>,
    SetStreams: Option<NvSetStreams>,
    Execute: Option<NvExecute>,
    DestroyBuffer: Option<NvDestroyBuffer>,
    Destroy: Option<NvDestroy>,
    GetLastError: Option<NvGetLastError>,
    GetCaps: Option<NvGetCaps>,
}

type NvCreateInstance = unsafe extern "system" fn(u32, *mut NvApiFunctions) -> NvStatus;

struct NvApi {
    _lib: libloading::Library,
    functions: NvApiFunctions,
}

impl NvApi {
    unsafe fn load() -> Result<Self, InitError> {
        let lib = unsafe { libloading::Library::new("nvofapi64.dll") }
            .map_err(|_| InitError::Failed(0xB_0000))?;
        let create: NvCreateInstance = *unsafe {
            lib.get(b"NvOFAPICreateInstanceCuda\0")
                .map_err(|_| InitError::Failed(0xB_0001))?
        };
        let mut functions = NvApiFunctions::default();
        let status = unsafe { create(NV_API_VERSION_1, &raw mut functions) };
        if status != NV_SUCCESS {
            return Err(InitError::Failed(0xC_0000 + status));
        }
        if functions.CreateFlow.is_none()
            || functions.Init.is_none()
            || functions.CreateBuffer.is_none()
            || functions.GetArray.is_none()
            || functions.GetStride.is_none()
            || functions.Execute.is_none()
            || functions.DestroyBuffer.is_none()
            || functions.Destroy.is_none()
        {
            return Err(InitError::Failed(0xC_0004));
        }
        Ok(Self {
            _lib: lib,
            functions,
        })
    }
}

struct Buffer {
    handle: NvBuffer,
    array: CuArray,
    stride: NvStrideInfo,
}

struct Session {
    cuda: CudaApi,
    nv: NvApi,
    context: CuContext,
    handle: NvHandle,
    inputs: [Buffer; 2],
    output: Buffer,
    cost: Buffer,
    width: usize,
    height: usize,
    grid_width: usize,
    grid_height: usize,
    scale: i32,
}

unsafe impl Send for Session {}

pub(crate) struct NvofContext {
    session: Mutex<Session>,
    cache: Mutex<Vec<(i32, Arc<Vec<u8>>)>>,
    width: usize,
    height: usize,
}

unsafe impl Send for NvofContext {}
unsafe impl Sync for NvofContext {}

impl NvofContext {
    pub(crate) fn cuda_available() -> bool {
        unsafe {
            let Ok(cuda) = CudaApi::load() else {
                return false;
            };
            (cuda.Init)(0) == CUDA_SUCCESS
        }
    }

    pub(crate) fn new(
        width: i32,
        height: i32,
        scale: i32,
        quality: i64,
        gpu_id: i64,
    ) -> Result<Self, InitError> {
        let width = usize::try_from(width).map_err(|_| InitError::Failed(5))?;
        let height = usize::try_from(height).map_err(|_| InitError::Failed(5))?;
        if width < 4 || height < 4 {
            return Err(InitError::Failed(5));
        }
        let session = unsafe { Session::new(width, height, scale, quality, gpu_id)? };
        Ok(Self {
            session: Mutex::new(session),
            cache: Mutex::new(Vec::new()),
            width,
            height,
        })
    }

    pub(crate) const fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    pub(crate) fn cached(&self, frame: i32) -> Option<Arc<Vec<u8>>> {
        let mut cache = self.cache.lock().ok()?;
        let pos = cache.iter().position(|(key, _)| *key == frame)?;
        let entry = cache.remove(pos);
        let payload = Arc::clone(&entry.1);
        cache.push(entry);
        Some(payload)
    }

    pub(crate) fn generate(
        &self,
        frame: i32,
        input: &[u8],
        reference: &[u8],
        vector_data: metadata::VectorData,
    ) -> Result<Arc<Vec<u8>>, i32> {
        if let Some(payload) = self.cached(frame) {
            return Ok(payload);
        }
        let payload = {
            let mut session = self.session.lock().map_err(|_| -1)?;
            Arc::new(session.execute(input, reference, vector_data)?)
        };
        let mut cache = self.cache.lock().map_err(|_| -1)?;
        if let Some(pos) = cache.iter().position(|(key, _)| *key == frame) {
            let entry = cache.remove(pos);
            let existing = Arc::clone(&entry.1);
            cache.push(entry);
            return Ok(existing);
        }
        if cache.len() >= CACHE_CAPACITY {
            cache.remove(0);
        }
        cache.push((frame, Arc::clone(&payload)));
        Ok(payload)
    }
}

impl Session {
    unsafe fn new(
        width: usize,
        height: usize,
        scale: i32,
        quality: i64,
        gpu_id: i64,
    ) -> Result<Self, InitError> {
        let cuda = unsafe { CudaApi::load()? };
        let status = unsafe { (cuda.Init)(0) };
        if status != CUDA_SUCCESS {
            return Err(InitError::CudaUnavailable);
        }
        let mut count = 0;
        let status = unsafe { (cuda.DeviceGetCount)(&raw mut count) };
        if status != CUDA_SUCCESS || count <= 0 {
            return Err(InitError::CudaUnavailable);
        }
        let ordinal = i32::try_from(gpu_id).unwrap_or(0);
        if ordinal < 0 || ordinal >= count {
            return Err(InitError::Failed(0x2_0000 + count));
        }
        let mut device = 0;
        let status = unsafe { (cuda.DeviceGet)(&raw mut device, ordinal) };
        if status != CUDA_SUCCESS {
            return Err(InitError::Failed(0x3_0000 + status));
        }
        let mut context = std::ptr::null_mut();
        let status = unsafe { (cuda.CtxCreate)(&raw mut context, 0, device) };
        if status != CUDA_SUCCESS || context.is_null() {
            return Err(InitError::Failed(0x4_0000 + status));
        }

        unsafe { Self::new_in_context(cuda, context, width, height, scale, quality) }
    }

    unsafe fn new_in_context(
        cuda: CudaApi,
        context: CuContext,
        width: usize,
        height: usize,
        scale: i32,
        quality: i64,
    ) -> Result<Self, InitError> {
        let Ok(width_u32) = u32::try_from(width) else {
            unsafe { (cuda.CtxDestroy)(context) };
            return Err(InitError::Failed(5));
        };
        let Ok(height_u32) = u32::try_from(height) else {
            unsafe { (cuda.CtxDestroy)(context) };
            return Err(InitError::Failed(5));
        };
        let nv = match unsafe { NvApi::load() } {
            Ok(nv) => nv,
            Err(error) => {
                unsafe { (cuda.CtxDestroy)(context) };
                return Err(error);
            }
        };
        let mut handle = std::ptr::null_mut();
        let status =
            unsafe { (nv.functions.CreateFlow.unwrap_unchecked())(context, &raw mut handle) };
        if status != NV_SUCCESS || handle.is_null() {
            unsafe { (cuda.CtxDestroy)(context) };
            return Err(InitError::Failed(0xD_0000 + status));
        }
        let init = NvInitParamsV1 {
            width: width_u32,
            height: height_u32,
            out_grid_size: NV_GRID_4,
            mode: NV_MODE_OPTICAL_FLOW,
            perf_level: match quality {
                0 => NV_PERF_FAST,
                1 => NV_PERF_MEDIUM,
                _ => NV_PERF_SLOW,
            },
            enable_output_cost: 1,
            ..NvInitParamsV1::default()
        };
        let status = unsafe { (nv.functions.Init.unwrap_unchecked())(handle, &raw const init) };
        if status != NV_SUCCESS {
            unsafe {
                (nv.functions.Destroy.unwrap_unchecked())(handle);
                (cuda.CtxDestroy)(context);
            }
            return Err(InitError::Failed(0xE_0000 + status));
        }

        let grid_width = width / 4;
        let grid_height = height / 4;
        let input_desc = NvBufferDescriptor {
            width: width_u32,
            height: height_u32,
            usage: NV_USAGE_INPUT,
            format: NV_FORMAT_NV12,
        };
        let output_desc = NvBufferDescriptor {
            width: u32::try_from(grid_width).unwrap_or(u32::MAX),
            height: u32::try_from(grid_height).unwrap_or(u32::MAX),
            usage: NV_USAGE_OUTPUT,
            format: NV_FORMAT_SHORT2,
        };
        let cost_desc = NvBufferDescriptor {
            width: u32::try_from(grid_width).unwrap_or(u32::MAX),
            height: u32::try_from(grid_height).unwrap_or(u32::MAX),
            usage: NV_USAGE_COST,
            format: NV_FORMAT_UINT,
        };
        let mut created = Vec::with_capacity(4);
        for (descriptor, error_base) in [
            (&input_desc, 0x10_0000),
            (&input_desc, 0x10_0000),
            (&output_desc, 0x10_0000),
            (&cost_desc, 0x11_0000),
        ] {
            match unsafe { create_buffer(&nv, handle, descriptor, error_base) } {
                Ok(buffer) => created.push(buffer),
                Err(error) => {
                    for buffer in created {
                        unsafe {
                            (nv.functions.DestroyBuffer.unwrap_unchecked())(buffer.handle);
                        }
                    }
                    unsafe {
                        (nv.functions.Destroy.unwrap_unchecked())(handle);
                        (cuda.CtxDestroy)(context);
                    }
                    return Err(error);
                }
            }
        }
        let mut created = created.into_iter();
        let input0 = created.next().unwrap();
        let input1 = created.next().unwrap();
        let output = created.next().unwrap();
        let cost = created.next().unwrap();

        let mut popped = std::ptr::null_mut();
        unsafe { (cuda.CtxPop)(&raw mut popped) };
        Ok(Self {
            cuda,
            nv,
            context,
            handle,
            inputs: [input0, input1],
            output,
            cost,
            width,
            height,
            grid_width,
            grid_height,
            scale: scale.max(1),
        })
    }

    fn execute(
        &mut self,
        input: &[u8],
        reference: &[u8],
        vector_data: metadata::VectorData,
    ) -> Result<Vec<u8>, i32> {
        let required = self.width.saturating_mul(self.height).saturating_mul(3) / 2;
        if input.len() < required || reference.len() < required {
            return Err(5);
        }
        let status = unsafe { (self.cuda.CtxPush)(self.context) };
        if status != CUDA_SUCCESS {
            return Err(0x15_0000 + status);
        }
        let result = unsafe { self.execute_current(input, reference, vector_data) };
        let mut popped = std::ptr::null_mut();
        unsafe { (self.cuda.CtxPop)(&raw mut popped) };
        result
    }

    unsafe fn execute_current(
        &mut self,
        input: &[u8],
        reference: &[u8],
        vector_data: metadata::VectorData,
    ) -> Result<Vec<u8>, i32> {
        unsafe {
            self.upload(&self.inputs[0], input)?;
            self.upload(&self.inputs[1], reference)?;
        }
        let current = unsafe { self.run_direction(0, 1, input)? };
        let previous = if vector_data.flags & 1 != 0 {
            Some(unsafe { self.run_direction(1, 0, reference)? })
        } else {
            None
        };
        Ok(metadata::encode_vector_payload(
            vector_data,
            previous.as_deref(),
            &current,
        ))
    }

    unsafe fn upload(&self, buffer: &Buffer, nv12: &[u8]) -> Result<(), i32> {
        let y_size = self.width.saturating_mul(self.height);
        let y = CudaMemcpy2D {
            src_memory_type: CU_MEMORYTYPE_HOST,
            src_host: nv12.as_ptr().cast(),
            src_pitch: self.width,
            dst_memory_type: CU_MEMORYTYPE_ARRAY,
            dst_array: buffer.array,
            width_in_bytes: self.width,
            height: self.height,
            ..CudaMemcpy2D::default()
        };
        let status = unsafe { (self.cuda.Memcpy2D)(&raw const y) };
        if status != CUDA_SUCCESS {
            return Err(0x15_0000 + status);
        }
        let uv = CudaMemcpy2D {
            src_memory_type: CU_MEMORYTYPE_HOST,
            src_host: unsafe { nv12.as_ptr().add(y_size) }.cast(),
            src_pitch: self.width,
            dst_y: buffer.stride.planes[0].y_bytes as usize,
            dst_memory_type: CU_MEMORYTYPE_ARRAY,
            dst_array: buffer.array,
            width_in_bytes: self.width,
            height: self.height.div_ceil(2),
            ..CudaMemcpy2D::default()
        };
        let status = unsafe { (self.cuda.Memcpy2D)(&raw const uv) };
        if status != CUDA_SUCCESS {
            return Err(0x15_0000 + status);
        }
        Ok(())
    }

    unsafe fn run_direction(
        &self,
        input_index: usize,
        reference_index: usize,
        luma: &[u8],
    ) -> Result<Vec<metadata::DecodedVector>, i32> {
        let execute_in = NvExecuteInputV1 {
            input: self.inputs[input_index].handle,
            reference: self.inputs[reference_index].handle,
            disable_temporal_hints: 1,
            ..NvExecuteInputV1::default()
        };
        let mut execute_out = NvExecuteOutput {
            output: self.output.handle,
            cost: self.cost.handle,
            ..NvExecuteOutput::default()
        };
        let status = unsafe {
            (self.nv.functions.Execute.unwrap_unchecked())(
                self.handle,
                &raw const execute_in,
                &raw mut execute_out,
            )
        };
        if status != NV_SUCCESS {
            return Err(0x16_0000 + status);
        }

        let count = self.grid_width.saturating_mul(self.grid_height);
        let mut flows = vec![0u32; count];
        let mut costs = vec![0u32; count];
        unsafe {
            self.download(&self.output, flows.as_mut_ptr().cast(), 4, self.grid_width)?;
            self.download(&self.cost, costs.as_mut_ptr().cast(), 4, self.grid_width)?;
        }
        Ok(pack_vectors(
            &flows,
            &costs,
            luma,
            self.width,
            self.height,
            self.scale,
        ))
    }

    unsafe fn download(
        &self,
        buffer: &Buffer,
        destination: *mut c_void,
        bytes_per_cell: usize,
        cells_per_row: usize,
    ) -> Result<(), i32> {
        let row_bytes = cells_per_row.saturating_mul(bytes_per_cell);
        let copy = CudaMemcpy2D {
            src_memory_type: CU_MEMORYTYPE_ARRAY,
            src_array: buffer.array,
            dst_memory_type: CU_MEMORYTYPE_HOST,
            dst_host: destination,
            dst_pitch: row_bytes,
            width_in_bytes: row_bytes,
            height: self.grid_height,
            ..CudaMemcpy2D::default()
        };
        let status = unsafe { (self.cuda.Memcpy2D)(&raw const copy) };
        if status == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(0x17_0000 + status)
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe {
            let pushed = (self.cuda.CtxPush)(self.context) == CUDA_SUCCESS;
            if pushed {
                for buffer in self.inputs.iter().chain([&self.output, &self.cost]) {
                    (self.nv.functions.DestroyBuffer.unwrap_unchecked())(buffer.handle);
                }
                (self.nv.functions.Destroy.unwrap_unchecked())(self.handle);
                let mut popped = std::ptr::null_mut();
                (self.cuda.CtxPop)(&raw mut popped);
            }
            (self.cuda.CtxDestroy)(self.context);
        }
    }
}

unsafe fn create_buffer(
    nv: &NvApi,
    handle: NvHandle,
    descriptor: &NvBufferDescriptor,
    error_base: i32,
) -> Result<Buffer, InitError> {
    let mut buffer = std::ptr::null_mut();
    let status = unsafe {
        (nv.functions.CreateBuffer.unwrap_unchecked())(
            handle,
            descriptor,
            NV_BUFFER_TYPE_CUARRAY,
            &raw mut buffer,
        )
    };
    if status != NV_SUCCESS || buffer.is_null() {
        return Err(InitError::Failed(error_base + status));
    }
    let array = unsafe { (nv.functions.GetArray.unwrap_unchecked())(buffer) };
    let mut stride = NvStrideInfo::default();
    let status = unsafe { (nv.functions.GetStride.unwrap_unchecked())(buffer, &raw mut stride) };
    if status != NV_SUCCESS || array.is_null() {
        unsafe { (nv.functions.DestroyBuffer.unwrap_unchecked())(buffer) };
        return Err(InitError::Failed(error_base + status));
    }
    Ok(Buffer {
        handle: buffer,
        array,
        stride,
    })
}

fn pack_vectors(
    flows: &[u32],
    costs: &[u32],
    luma: &[u8],
    width: usize,
    height: usize,
    scale: i32,
) -> Vec<metadata::DecodedVector> {
    let grid_width = width / 4;
    let grid_height = height / 4;
    let divisor = if scale <= 1 {
        8.0f32
    } else if scale <= 7 {
        4.0f32
    } else {
        2.0f32
    };
    let cost_shift = ((f64::from(scale.max(1)).log2() * 2.0) as u32).min(31);
    let mut output = Vec::with_capacity(grid_width.saturating_mul(grid_height));
    for gy in 0..grid_height {
        for gx in 0..grid_width {
            let index = gy.saturating_mul(grid_width).saturating_add(gx);
            let flow = flows.get(index).copied().unwrap_or(0);
            let raw = flow.to_le_bytes();
            let x = i16::from_le_bytes([raw[0], raw[1]]);
            let y = i16::from_le_bytes([raw[2], raw[3]]);
            let dx = (f32::from(x) / divisor) as i16;
            let dy = (f32::from(y) / divisor) as i16;
            let mut sum = 0u16;
            for row in gy.saturating_mul(4)..(gy + 1).saturating_mul(4).min(height) {
                for col in gx.saturating_mul(4)..(gx + 1).saturating_mul(4).min(width) {
                    sum = sum.wrapping_add(u16::from(
                        luma.get(row.saturating_mul(width) + col)
                            .copied()
                            .unwrap_or(0),
                    ));
                }
            }
            let raw1 = costs
                .get(index)
                .copied()
                .unwrap_or(0)
                .wrapping_shl(cost_shift)
                .wrapping_add(u32::from(sum & 0xFFF0).wrapping_shl(20));
            output.push(metadata::DecodedVector {
                dx,
                dy,
                score: raw1 & 0x00FF_FFFF,
                luma: (raw1 >> 24) as u8,
            });
        }
    }
    output
}
