use std::sync::{Condvar, Mutex, OnceLock, PoisonError};

use crate::opencl::{CL_SUCCESS, FnRelease, Handle, OpenCl};

#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
type Lookup = unsafe extern "C" fn(Handle) -> Handle;
#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
type LookupSource = unsafe extern "C" fn(Handle, u64, *mut SharedSource) -> i32;
#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
type FinishSource = unsafe extern "C" fn(Handle, Handle, Handle) -> i32;
#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
type HasPtr = unsafe extern "C" fn(Handle, usize) -> i32;
#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
type LookupAny = unsafe extern "C" fn() -> Handle;
#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
type CpuActive = unsafe extern "C" fn() -> usize;

#[cfg(target_os = "linux")]
const SIBLINGS: [&str; 2] = ["libsvpflow1_vs.so", "libsvpflow2_vs.so"];
#[cfg(target_os = "macos")]
const SIBLINGS: [&str; 2] = ["libsvpflow1_vs.dylib", "libsvpflow2_vs.dylib"];
#[cfg(target_os = "windows")]
const SIBLINGS: [&str; 2] = ["svpflow1_vs.dll", "svpflow2_vs.dll"];

static OWN: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

const HANDOFF_CAP: usize = 24;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SharedSource {
    pub y_ptr: usize,
    pub mems: [Handle; 3],
    pub pitches: [usize; 3],
    pub width: usize,
    pub height: usize,
    pub chroma_width: usize,
    pub chroma_height: usize,
    pub event: Handle,
}

impl Default for SharedSource {
    fn default() -> Self {
        Self {
            y_ptr: 0,
            mems: [std::ptr::null_mut(); 3],
            pitches: [0; 3],
            width: 0,
            height: 0,
            chroma_width: 0,
            chroma_height: 0,
            event: std::ptr::null_mut(),
        }
    }
}

struct StoredSource {
    context: usize,
    y_ptr: usize,
    hash: u64,
    mems: [usize; 3],
    pitches: [usize; 3],
    width: usize,
    height: usize,
    chroma_width: usize,
    chroma_height: usize,
    event: usize,
    readers: usize,
    consumers: Vec<usize>,
    retain_mem: FnRelease,
    release_mem: FnRelease,
    retain_event: FnRelease,
    release_event: FnRelease,
}

impl Drop for StoredSource {
    fn drop(&mut self) {
        for mem in self.mems {
            if mem != 0 {
                unsafe { (self.release_mem)(mem as Handle) };
            }
        }
        if self.event != 0 {
            unsafe { (self.release_event)(self.event as Handle) };
        }
        for ev in self.consumers.drain(..) {
            if ev != 0 {
                unsafe { (self.release_event)(ev as Handle) };
            }
        }
    }
}

struct SourceTable {
    active: Vec<StoredSource>,
    retired: Vec<StoredSource>,
}

static SOURCES: Mutex<SourceTable> = Mutex::new(SourceTable {
    active: Vec::new(),
    retired: Vec::new(),
});
static SOURCES_CV: Condvar = Condvar::new();
static SOURCE_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static CPU_ACTIVE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[unsafe(no_mangle)]
pub extern "C" fn osvp_cpu_active_v1() -> usize {
    CPU_ACTIVE.load(std::sync::atomic::Ordering::Relaxed)
}

#[unsafe(no_mangle)]
pub extern "C" fn osvp_cl_has_sources_v1() -> i32 {
    i32::from(SOURCE_COUNT.load(std::sync::atomic::Ordering::Relaxed) > 0)
}

#[unsafe(no_mangle)]
pub extern "C" fn osvp_cl_has_ptr_v1(context: Handle, y_ptr: usize) -> i32 {
    if SOURCE_COUNT.load(std::sync::atomic::Ordering::Relaxed) == 0 || y_ptr == 0 {
        return 0;
    }
    let table = SOURCES.lock().unwrap_or_else(PoisonError::into_inner);
    i32::from(
        table
            .active
            .iter()
            .any(|s| s.context == context as usize && s.y_ptr == y_ptr),
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn osvp_cl_context_v1(device: Handle) -> Handle {
    let own = OWN.lock().unwrap_or_else(PoisonError::into_inner);
    own.iter()
        .find(|entry| entry.0 == device as usize)
        .map_or(std::ptr::null_mut(), |entry| entry.1 as Handle)
}

#[unsafe(no_mangle)]
pub extern "C" fn osvp_cl_context_any_v1() -> Handle {
    let own = OWN.lock().unwrap_or_else(PoisonError::into_inner);
    own.first()
        .map_or(std::ptr::null_mut(), |entry| entry.1 as Handle)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn osvp_cl_finish_source_v1(
    context: Handle,
    mem0: Handle,
    event: Handle,
) -> i32 {
    if context.is_null() || mem0.is_null() {
        return 0;
    }
    let mut table = SOURCES.lock().unwrap_or_else(PoisonError::into_inner);
    let SourceTable { active, retired } = &mut *table;
    let entry = active
        .iter_mut()
        .rev()
        .chain(retired.iter_mut().rev())
        .find(|entry| entry.context == context as usize && entry.mems[0] == mem0 as usize);
    let Some(entry) = entry else {
        return 0;
    };
    if !event.is_null() && unsafe { (entry.retain_event)(event) } == CL_SUCCESS {
        entry.consumers.push(event as usize);
    }
    entry.readers = entry.readers.saturating_sub(1);
    if entry.readers == 0 {
        SOURCES_CV.notify_all();
    }
    1
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn osvp_cl_lookup_source_v1(
    context: Handle,
    hash: u64,
    out: *mut SharedSource,
) -> i32 {
    if out.is_null() {
        return 0;
    }
    let mut table = SOURCES.lock().unwrap_or_else(PoisonError::into_inner);
    let Some(entry) = table
        .active
        .iter_mut()
        .rev()
        .find(|entry| entry.context == context as usize && entry.hash == hash)
    else {
        return 0;
    };
    for (idx, &mem) in entry.mems.iter().enumerate() {
        if mem == 0 || unsafe { (entry.retain_mem)(mem as Handle) } != CL_SUCCESS {
            for &prev in &entry.mems[..idx] {
                unsafe { (entry.release_mem)(prev as Handle) };
            }
            return 0;
        }
    }
    if entry.event != 0 && unsafe { (entry.retain_event)(entry.event as Handle) } != CL_SUCCESS {
        for &mem in &entry.mems {
            unsafe { (entry.release_mem)(mem as Handle) };
        }
        return 0;
    }
    entry.readers += 1;
    unsafe {
        *out = SharedSource {
            y_ptr: entry.y_ptr,
            mems: entry.mems.map(|m| m as Handle),
            pitches: entry.pitches,
            width: entry.width,
            height: entry.height,
            chroma_width: entry.chroma_width,
            chroma_height: entry.chroma_height,
            event: entry.event as Handle,
        };
    }
    1
}

#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
fn open_sibling(name: &str) -> Option<libloading::Library> {
    #[cfg(target_os = "linux")]
    let library = unsafe { libloading::os::unix::Library::open(Some(name), 0x1 | 0x4) };
    #[cfg(target_os = "macos")]
    let library = unsafe { libloading::os::unix::Library::open(Some(name), 0x1 | 0x10) };
    #[cfg(target_os = "windows")]
    let library = libloading::os::windows::Library::open_already_loaded(name);
    library.ok().map(Into::into)
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn sibling(_: Handle) -> Handle {
    std::ptr::null_mut()
}

#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
fn sibling(device: Handle) -> Handle {
    for name in SIBLINGS {
        let Some(library) = open_sibling(name) else {
            continue;
        };
        let Ok(lookup) = (unsafe { library.get::<Lookup>(b"osvp_cl_context_v1\0") }) else {
            continue;
        };
        let lookup: Lookup = *lookup;
        if lookup as *const () == osvp_cl_context_v1 as *const () {
            continue;
        }
        let context = unsafe { lookup(device) };
        if !context.is_null() {
            std::mem::forget(library);
            return context;
        }
    }
    std::ptr::null_mut()
}

pub fn any_shared() -> Handle {
    let own = osvp_cl_context_any_v1();
    if !own.is_null() {
        return own;
    }
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    for name in SIBLINGS {
        let Some(library) = open_sibling(name) else {
            continue;
        };
        let Ok(lookup) = (unsafe { library.get::<LookupAny>(b"osvp_cl_context_any_v1\0") }) else {
            continue;
        };
        let lookup: LookupAny = *lookup;
        if lookup as *const () == osvp_cl_context_any_v1 as *const () {
            continue;
        }
        let context = unsafe { lookup() };
        if !context.is_null() {
            std::mem::forget(library);
            return context;
        }
    }
    std::ptr::null_mut()
}

pub fn shared(device: Handle, create: impl FnOnce() -> Handle) -> Handle {
    let mut context = osvp_cl_context_v1(device);
    if !context.is_null() {
        return context;
    }
    context = sibling(device);
    let mut own = OWN.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(entry) = own.iter().find(|entry| entry.0 == device as usize) {
        return entry.1 as Handle;
    }
    if context.is_null() {
        context = create();
    }
    if !context.is_null() {
        own.push((device as usize, context as usize));
    }
    context
}

pub fn plane_fingerprint(planes: [(&[u8], usize, usize, usize); 3]) -> u64 {
    const P1: u64 = 0x9E37_79B1_85EB_CA87;
    const P2: u64 = 0xC2B2_AE3D_27D4_EB4F;
    const P3: u64 = 0x1656_67B1_9E37_79F9;
    const P4: u64 = 0x85EB_CA77_C2B2_AE63;
    let mut s0 = P1;
    let mut s1 = P2;
    let mut s2 = P3;
    let mut s3 = P4;
    for (data, stride, width, height) in planes {
        if width == 0
            || height == 0
            || data.len()
                < stride
                    .saturating_mul(height.saturating_sub(1))
                    .saturating_add(width)
        {
            return 0;
        }
        s0 = s0.wrapping_add(width as u64).wrapping_mul(P2);
        s1 = s1.wrapping_add(height as u64).wrapping_mul(P3);
        for y in 0..height {
            let row = &data[y * stride..y * stride + width];
            let (chunks, tail) = row.as_chunks::<32>();
            for chunk in chunks {
                let v0 = u64::from_le_bytes(chunk[0..8].try_into().unwrap());
                let v1 = u64::from_le_bytes(chunk[8..16].try_into().unwrap());
                let v2 = u64::from_le_bytes(chunk[16..24].try_into().unwrap());
                let v3 = u64::from_le_bytes(chunk[24..32].try_into().unwrap());
                s0 = (s0 ^ v0).rotate_left(27).wrapping_mul(P1);
                s1 = (s1 ^ v1).rotate_left(31).wrapping_mul(P2);
                s2 = (s2 ^ v2).rotate_left(29).wrapping_mul(P3);
                s3 = (s3 ^ v3).rotate_left(23).wrapping_mul(P4);
            }
            let (words, bytes) = tail.as_chunks::<8>();
            for word in words {
                let val = u64::from_le_bytes(*word);
                s0 = (s0 ^ val).rotate_left(27).wrapping_mul(P1);
            }
            for &byte in bytes {
                s1 = (s1 ^ u64::from(byte)).wrapping_mul(P2);
            }
        }
    }
    let mut hash = s0 ^ s1.rotate_left(17) ^ s2.rotate_left(33) ^ s3.rotate_left(49);
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    hash ^= hash >> 33;
    if hash == 0 { 1 } else { hash }
}

pub unsafe fn publish_source(
    cl: &OpenCl,
    context: Handle,
    hash: u64,
    source: SharedSource,
) {
    if hash == 0 || context.is_null() {
        return;
    }
    let (Some(retain_mem), Some(retain_event)) = (cl.RetainMemObject, cl.RetainEvent) else {
        return;
    };
    for &mem in &source.mems {
        if mem.is_null() {
            return;
        }
    }
    for (i, &mem) in source.mems.iter().enumerate() {
        if unsafe { retain_mem(mem) } != CL_SUCCESS {
            for &prev in &source.mems[..i] {
                unsafe { (cl.ReleaseMemObject)(prev) };
            }
            return;
        }
    }
    if !source.event.is_null() && unsafe { retain_event(source.event) } != CL_SUCCESS {
        for &mem in &source.mems {
            unsafe { (cl.ReleaseMemObject)(mem) };
        }
        return;
    }
    let entry = StoredSource {
        context: context as usize,
        y_ptr: source.y_ptr,
        hash,
        mems: source.mems.map(|m| m as usize),
        pitches: source.pitches,
        width: source.width,
        height: source.height,
        chroma_width: source.chroma_width,
        chroma_height: source.chroma_height,
        event: source.event as usize,
        readers: 0,
        consumers: Vec::new(),
        retain_mem,
        release_mem: cl.ReleaseMemObject,
        retain_event,
        release_event: cl.ReleaseEvent,
    };
    let mut table = SOURCES.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(pos) = table
        .active
        .iter()
        .position(|s| s.context == entry.context && s.hash == entry.hash)
    {
        let old = table.active.remove(pos);
        if old.readers > 0 || !old.consumers.is_empty() {
            table.retired.push(old);
        }
    }
    if table.active.len() >= HANDOFF_CAP {
        let old = table.active.remove(0);
        if old.readers > 0 || !old.consumers.is_empty() {
            table.retired.push(old);
        }
    }
    table.active.push(entry);
    SOURCE_COUNT.store(table.active.len(), std::sync::atomic::Ordering::Relaxed);
}

pub unsafe fn unpublish_mems(cl: &OpenCl, mems: [usize; 3]) -> bool {
    let mut removed = Vec::new();
    {
        let mut table = SOURCES.lock().unwrap_or_else(PoisonError::into_inner);
        let mut i = 0;
        while i < table.active.len() {
            if table.active[i].mems == mems {
                let entry = table.active.remove(i);
                table.retired.push(entry);
            } else {
                i += 1;
            }
        }
        SOURCE_COUNT.store(table.active.len(), std::sync::atomic::Ordering::Relaxed);
        table = SOURCES_CV
            .wait_while(table, |t| {
                t.retired.iter().any(|e| e.mems == mems && e.readers > 0)
            })
            .unwrap_or_else(PoisonError::into_inner);
        let mut j = 0;
        while j < table.retired.len() {
            if table.retired[j].mems == mems {
                removed.push(table.retired.remove(j));
            } else {
                j += 1;
            }
        }
    }
    let mut reusable = true;
    for mut entry in removed {
        for ev in entry.consumers.drain(..) {
            if ev != 0 {
                let h = ev as Handle;
                if unsafe { (cl.WaitForEvents)(1, &raw const h) } != CL_SUCCESS {
                    reusable = false;
                }
                unsafe { (cl.ReleaseEvent)(h) };
            }
        }
    }
    reusable
}

#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
fn sibling_finish_source() -> Option<FinishSource> {
    static FN: OnceLock<Option<FinishSource>> = OnceLock::new();
    *FN.get_or_init(|| {
        for name in SIBLINGS {
            let Some(library) = open_sibling(name) else {
                continue;
            };
            let Ok(lookup) = (unsafe { library.get::<FinishSource>(b"osvp_cl_finish_source_v1\0") })
            else {
                continue;
            };
            let lookup: FinishSource = *lookup;
            if lookup as *const () == osvp_cl_finish_source_v1 as *const () {
                continue;
            }
            std::mem::forget(library);
            return Some(lookup);
        }
        None
    })
}

pub unsafe fn finish_source(context: Handle, mem0: Handle, event: Handle) {
    if context.is_null() || mem0.is_null() {
        return;
    }
    if unsafe { osvp_cl_finish_source_v1(context, mem0, event) } != 0 {
        return;
    }
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    if let Some(finish) = sibling_finish_source() {
        unsafe { finish(context, mem0, event) };
    }
}

pub fn purge_sources() {
    let mut table = SOURCES.lock().unwrap_or_else(PoisonError::into_inner);
    table.active.clear();
    table.retired.retain(|e| e.readers > 0);
    SOURCE_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
type HasSources = unsafe extern "C" fn() -> i32;

#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
fn sibling_has_sources() -> Option<HasSources> {
    static FN: OnceLock<Option<HasSources>> = OnceLock::new();
    *FN.get_or_init(|| {
        for name in SIBLINGS {
            let Some(library) = open_sibling(name) else {
                continue;
            };
            let Ok(lookup) = (unsafe { library.get::<HasSources>(b"osvp_cl_has_sources_v1\0") })
            else {
                continue;
            };
            let lookup: HasSources = *lookup;
            if lookup as *const () == osvp_cl_has_sources_v1 as *const () {
                continue;
            }
            std::mem::forget(library);
            return Some(lookup);
        }
        None
    })
}

#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
fn sibling_has_ptr() -> Option<HasPtr> {
    static FN: OnceLock<Option<HasPtr>> = OnceLock::new();
    *FN.get_or_init(|| {
        for name in SIBLINGS {
            let Some(library) = open_sibling(name) else {
                continue;
            };
            let Ok(lookup) = (unsafe { library.get::<HasPtr>(b"osvp_cl_has_ptr_v1\0") }) else {
                continue;
            };
            let lookup: HasPtr = *lookup;
            if lookup as *const () == osvp_cl_has_ptr_v1 as *const () {
                continue;
            }
            std::mem::forget(library);
            return Some(lookup);
        }
        None
    })
}

pub unsafe fn has_shared_ptr(context: Handle, y_ptr: usize) -> bool {
    if context.is_null() || y_ptr == 0 {
        return false;
    }
    if osvp_cl_has_ptr_v1(context, y_ptr) != 0 {
        return true;
    }
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    if let Some(check) = sibling_has_ptr()
        && unsafe { check(context, y_ptr) } != 0
    {
        return true;
    }
    false
}

pub fn has_shared_sources() -> bool {
    if SOURCE_COUNT.load(std::sync::atomic::Ordering::Relaxed) > 0 {
        return true;
    }
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    if let Some(check) = sibling_has_sources()
        && unsafe { check() } != 0
    {
        return true;
    }
    false
}

#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
fn sibling_lookup_source() -> Option<LookupSource> {
    static FN: OnceLock<Option<LookupSource>> = OnceLock::new();
    *FN.get_or_init(|| {
        for name in SIBLINGS {
            let Some(library) = open_sibling(name) else {
                continue;
            };
            let Ok(lookup) =
                (unsafe { library.get::<LookupSource>(b"osvp_cl_lookup_source_v1\0") })
            else {
                continue;
            };
            let lookup: LookupSource = *lookup;
            if lookup as *const () == osvp_cl_lookup_source_v1 as *const () {
                continue;
            }
            std::mem::forget(library);
            return Some(lookup);
        }
        None
    })
}

pub unsafe fn lookup_source(context: Handle, hash: u64) -> Option<SharedSource> {
    if hash == 0 || context.is_null() {
        return None;
    }
    let mut out = SharedSource::default();
    if unsafe { osvp_cl_lookup_source_v1(context, hash, &raw mut out) } != 0 {
        return Some(out);
    }
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    if let Some(lookup) = sibling_lookup_source()
        && unsafe { lookup(context, hash, &raw mut out) } != 0
    {
        return Some(out);
    }
    None
}

#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
fn sibling_cpu_active() -> Option<CpuActive> {
    static FN: OnceLock<Option<CpuActive>> = OnceLock::new();
    *FN.get_or_init(|| {
        for name in SIBLINGS {
            let Some(library) = open_sibling(name) else {
                continue;
            };
            let Ok(lookup) = (unsafe { library.get::<CpuActive>(b"osvp_cpu_active_v1\0") }) else {
                continue;
            };
            let lookup: CpuActive = *lookup;
            if lookup as *const () == osvp_cpu_active_v1 as *const () {
                continue;
            }
            std::mem::forget(library);
            return Some(lookup);
        }
        None
    })
}

pub fn cpu_busy(delta: isize) -> usize {
    if delta < 0 {
        let sub = delta.unsigned_abs();
        return CPU_ACTIVE
            .fetch_sub(sub, std::sync::atomic::Ordering::Relaxed)
            .saturating_sub(sub);
    }
    let local = if delta > 0 {
        let add = delta.cast_unsigned();
        CPU_ACTIVE.fetch_add(add, std::sync::atomic::Ordering::Relaxed) + add
    } else {
        CPU_ACTIVE.load(std::sync::atomic::Ordering::Relaxed)
    };
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    let remote = sibling_cpu_active().map_or(0, |f| unsafe { f() });
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    let remote = 0;
    local.saturating_add(remote)
}
