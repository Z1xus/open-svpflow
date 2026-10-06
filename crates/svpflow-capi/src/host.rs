use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::cell::UnsafeCell;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::ffi::{CStr, CString, c_char, c_void};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, PoisonError};

use svpflow_host::vs3::{self, ConstRaw, CoreInfo, Format, Raw, VideoInfo};

const ALIGN: usize = 64;
const CACHE_BYTES: usize = 1 << 30;
const POOL_BYTES: usize = 256 << 20;
const READ_AHEAD: i32 = 64;

pub type Read = unsafe extern "C" fn(*mut c_void, i32, *const *mut u8, *const isize) -> i32;
pub type Release = unsafe extern "C" fn(*mut c_void);

struct Buffer {
    data: NonNull<u8>,
    len: usize,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

static POOL: Mutex<(Vec<Buffer>, usize)> = Mutex::new((Vec::new(), 0));

impl Buffer {
    fn new(len: usize) -> Self {
        let len = len.max(ALIGN);
        let mut pool = POOL.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(index) = pool.0.iter().position(|buffer| buffer.len == len) {
            pool.1 -= len;
            return pool.0.swap_remove(index);
        }
        drop(pool);
        let layout = Layout::from_size_align(len, ALIGN).expect("frame layout");
        let data = NonNull::new(unsafe { alloc_zeroed(layout) })
            .unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        Self { data, len }
    }

    fn copy(&self) -> Self {
        let copy = Self::new(self.len);
        unsafe { std::ptr::copy_nonoverlapping(self.data.as_ptr(), copy.data.as_ptr(), self.len) };
        copy
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        let mut pool = POOL.lock().unwrap_or_else(PoisonError::into_inner);
        if pool.1 + self.len <= POOL_BYTES {
            pool.1 += self.len;
            pool.0.push(Self {
                data: self.data,
                len: self.len,
            });
            return;
        }
        drop(pool);
        let layout = Layout::from_size_align(self.len, ALIGN).expect("frame layout");
        unsafe { dealloc(self.data.as_ptr(), layout) };
    }
}

#[derive(Clone)]
pub enum Value {
    Int(i64),
    Float(f64),
    Data(CString),
    Node(Arc<Node>),
}

#[derive(Clone, Default)]
pub struct Map {
    values: Vec<(CString, Value)>,
    pub error: Option<CString>,
}

impl Map {
    pub fn set(&mut self, key: &CStr, value: Value) {
        self.values.push((key.to_owned(), value));
    }

    fn get(&self, key: &CStr, index: i32) -> Option<&Value> {
        self.values
            .iter()
            .filter(|(name, _)| name.as_c_str() == key)
            .nth(usize::try_from(index).ok()?)
            .map(|(_, value)| value)
    }

    pub fn node(&self, key: &CStr) -> Option<Arc<Node>> {
        match self.get(key, 0)? {
            Value::Node(node) => Some(Arc::clone(node)),
            _ => None,
        }
    }

    pub fn int(&self, key: &CStr) -> Option<i64> {
        match self.get(key, 0)? {
            Value::Int(value) => Some(*value),
            _ => None,
        }
    }
}

#[derive(Clone)]
struct Plane {
    data: Arc<Buffer>,
    stride: usize,
}

pub struct Frame {
    planes: UnsafeCell<Vec<Plane>>,
    props: UnsafeCell<Map>,
}

unsafe impl Send for Frame {}
unsafe impl Sync for Frame {}

impl Frame {
    fn new(format: &Format, width: i32, height: i32, props: Map) -> Self {
        let planes = (0..format.num_planes)
            .map(|plane| {
                let (shift_w, shift_h) = if plane == 0 {
                    (0, 0)
                } else {
                    (format.sub_sampling_w, format.sub_sampling_h)
                };
                let width = usize::try_from(width >> shift_w).unwrap_or(0);
                let height = usize::try_from(height >> shift_h).unwrap_or(0);
                let bytes = usize::try_from(format.bytes_per_sample).unwrap_or(1);
                let stride = (width * bytes).next_multiple_of(ALIGN);
                Plane {
                    data: Arc::new(Buffer::new(stride * height)),
                    stride,
                }
            })
            .collect();
        Self {
            planes: UnsafeCell::new(planes),
            props: UnsafeCell::new(props),
        }
    }

    fn planes(&self) -> &[Plane] {
        unsafe { &*self.planes.get() }
    }

    fn bytes(&self) -> usize {
        self.planes().iter().map(|plane| plane.data.len).sum()
    }

    pub fn stride(&self, plane: i32) -> usize {
        usize::try_from(plane)
            .ok()
            .and_then(|plane| self.planes().get(plane))
            .map_or(0, |plane| plane.stride)
    }

    pub fn read(&self, plane: i32) -> *const u8 {
        usize::try_from(plane)
            .ok()
            .and_then(|plane| self.planes().get(plane))
            .map_or(std::ptr::null(), |plane| plane.data.data.as_ptr())
    }

    unsafe fn write(&self, plane: i32) -> *mut u8 {
        let planes = unsafe { &mut *self.planes.get() };
        let Some(plane) = usize::try_from(plane)
            .ok()
            .and_then(|plane| planes.get_mut(plane))
        else {
            return std::ptr::null_mut();
        };
        if Arc::get_mut(&mut plane.data).is_none() {
            plane.data = Arc::new(plane.data.copy());
        }
        plane.data.data.as_ptr()
    }
}

#[derive(Default)]
struct Reader {
    next: i32,
    ahead: Vec<(i32, Arc<Frame>)>,
}

struct Source {
    read: Read,
    release: Option<Release>,
    user: *mut c_void,
    state: Mutex<Reader>,
}

enum Kind {
    Source(Source),
    Filter {
        instance: UnsafeCell<Raw>,
        get_frame: vs3::GetFrame,
        free: vs3::FreeFilter,
    },
}

pub struct Node {
    host: Arc<Host>,
    info: UnsafeCell<VideoInfo>,
    kind: Kind,
}

unsafe impl Send for Node {}
unsafe impl Sync for Node {}

type Produced = Result<Arc<Frame>, Arc<CStr>>;

#[derive(Default)]
struct Slot {
    state: Mutex<(Option<Produced>, Vec<Arc<Task>>)>,
    ready: Condvar,
}

type Cell = Arc<Slot>;

pub struct Pending(#[allow(dead_code)] Cell);

struct Entry {
    cell: Cell,
    bytes: usize,
    tick: u64,
}

#[derive(Default)]
struct Cache {
    entries: HashMap<(usize, i32), Entry>,
    bytes: usize,
    tick: u64,
}

struct Task {
    node: Arc<Node>,
    n: i32,
    priority: i32,
    cell: Cell,
    waiting: AtomicUsize,
    context: UnsafeCell<FrameContext>,
}

unsafe impl Send for Task {}
unsafe impl Sync for Task {}

struct FrameContext {
    error: Option<Arc<CStr>>,
    data: Raw,
    requests: Vec<(Arc<Node>, i32)>,
    frames: Vec<((usize, i32), Cell)>,
}

struct Job {
    order: Reverse<(i32, u64)>,
    run: Box<dyn FnOnce() + Send>,
}

impl PartialEq for Job {
    fn eq(&self, other: &Self) -> bool {
        self.order == other.order
    }
}

impl Eq for Job {}

impl PartialOrd for Job {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Job {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.order.cmp(&other.order)
    }
}

#[derive(Default)]
struct Queue {
    jobs: Mutex<(BinaryHeap<Job>, u64, bool)>,
    ready: Condvar,
}

pub struct Host {
    info: CoreInfo,
    functions: Vec<(CString, vs3::Create)>,
    cache: Mutex<Cache>,
    queue: Arc<Queue>,
}

unsafe impl Send for Host {}
unsafe impl Sync for Host {}

impl Node {
    pub fn info(&self) -> VideoInfo {
        unsafe { *self.info.get() }
    }

    pub fn host(&self) -> &Arc<Host> {
        &self.host
    }

    fn id(&self) -> usize {
        std::ptr::from_ref(self) as usize
    }

    fn frame(&self, n: i32) -> i32 {
        n.clamp(0, (self.info().num_frames - 1).max(0))
    }

    pub fn prepare(self: &Arc<Self>, n: i32) -> Pending {
        Pending(self.start(n, n))
    }

    fn start(self: &Arc<Self>, n: i32, priority: i32) -> Cell {
        let n = self.frame(n);
        let mut cache = self
            .host
            .cache
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        cache.tick += 1;
        let tick = cache.tick;
        if let Some(entry) = cache.entries.get_mut(&(self.id(), n)) {
            entry.tick = tick;
            return Arc::clone(&entry.cell);
        }
        let cell = Cell::default();
        cache.entries.insert(
            (self.id(), n),
            Entry {
                cell: Arc::clone(&cell),
                bytes: 0,
                tick,
            },
        );
        drop(cache);
        let task = Arc::new(Task {
            node: Arc::clone(self),
            n,
            priority,
            cell: Arc::clone(&cell),
            waiting: AtomicUsize::new(0),
            context: UnsafeCell::new(FrameContext {
                error: None,
                data: std::ptr::null_mut(),
                requests: Vec::new(),
                frames: Vec::new(),
            }),
        });
        if matches!(self.kind, Kind::Source(_)) {
            self.host.spawn(priority, move || task.begin());
        } else {
            task.begin();
        }
        cell
    }

    pub fn get(self: &Arc<Self>, n: i32) -> Produced {
        let cell = self.start(n, n);
        let mut state = cell.state.lock().unwrap_or_else(PoisonError::into_inner);
        loop {
            if let Some(produced) = &state.0 {
                return produced.clone();
            }
            state = cell
                .ready
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    fn read(&self, source: &Source, n: i32) -> Produced {
        let info = self.info();
        let format = unsafe { &*info.format.cast::<Format>() };
        let mut state = source.state.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(index) = state.ahead.iter().position(|(frame, _)| *frame == n) {
            return Ok(state.ahead.swap_remove(index).1);
        }
        let first = if n < state.next || n - state.next > READ_AHEAD {
            n
        } else {
            state.next
        };
        let mut wanted = None;
        for index in first..=n {
            let mut props = Map::default();
            props.set(c"_DurationNum", Value::Int(info.fps_den));
            props.set(c"_DurationDen", Value::Int(info.fps_num));
            let frame = Frame::new(format, info.width, info.height, props);
            let mut planes = [std::ptr::null_mut(); 3];
            let mut strides = [0isize; 3];
            for (plane, (data, stride)) in
                (0..format.num_planes).zip(planes.iter_mut().zip(&mut strides))
            {
                *data = unsafe { frame.write(plane) };
                *stride = frame.stride(plane).cast_signed();
            }
            if unsafe { (source.read)(source.user, index, planes.as_ptr(), strides.as_ptr()) } != 0
            {
                return Err(Arc::from(c"the source callback failed"));
            }
            let frame = Arc::new(frame);
            if index == n {
                wanted = Some(frame);
            } else {
                state.ahead.retain(|(frame, _)| *frame > index - READ_AHEAD);
                state.ahead.push((index, frame));
            }
        }
        state.next = state.next.max(n + 1);
        wanted.ok_or_else(|| Arc::from(c"the source callback failed"))
    }
}

impl Task {
    fn call(self: &Arc<Self>, reason: i32) -> ConstRaw {
        let Kind::Filter {
            instance,
            get_frame,
            ..
        } = &self.node.kind
        else {
            return std::ptr::null();
        };
        unsafe {
            get_frame(
                self.n,
                reason,
                instance.get(),
                &raw mut (*self.context.get()).data,
                Arc::as_ptr(self).cast_mut().cast(),
                Arc::as_ptr(&self.node.host).cast_mut().cast(),
                table(),
            )
        }
    }

    fn begin(self: Arc<Self>) {
        if let Kind::Source(source) = &self.node.kind {
            let produced = self.node.read(source, self.n);
            return self.complete(produced);
        }
        let frame = self.call(vs3::AR_INITIAL);
        let context = unsafe { &mut *self.context.get() };
        if !frame.is_null() || context.error.is_some() {
            return self.finish(frame);
        }
        self.waiting.store(1, Ordering::Relaxed);
        for (node, n) in std::mem::take(&mut context.requests) {
            let cell = node.start(n, self.priority);
            let mut state = cell.state.lock().unwrap_or_else(PoisonError::into_inner);
            if state.0.is_none() {
                self.waiting.fetch_add(1, Ordering::Relaxed);
                state.1.push(Arc::clone(&self));
            }
            drop(state);
            context.frames.push(((node.id(), n), cell));
        }
        if self.waiting.fetch_sub(1, Ordering::AcqRel) == 1 {
            let host = Arc::clone(&self.node.host);
            host.spawn(self.priority, move || self.run());
        }
    }

    fn run(self: Arc<Self>) {
        let frame = self.call(vs3::AR_ALL_FRAMES_READY);
        self.finish(frame);
    }

    fn finish(self: Arc<Self>, frame: ConstRaw) {
        let context = unsafe { &mut *self.context.get() };
        context.frames.clear();
        let produced = if frame.is_null() {
            Err(context
                .error
                .take()
                .unwrap_or_else(|| Arc::from(c"the filter returned no frame")))
        } else {
            Ok(unsafe { Arc::from_raw(frame.cast::<Frame>()) })
        };
        self.complete(produced);
    }

    fn complete(self: Arc<Self>, produced: Produced) {
        let host = &self.node.host;
        let key = (self.node.id(), self.n);
        {
            let mut cache = host.cache.lock().unwrap_or_else(PoisonError::into_inner);
            let cached = cache
                .entries
                .get(&key)
                .is_some_and(|entry| Arc::ptr_eq(&entry.cell, &self.cell));
            match &produced {
                Ok(frame) if cached => {
                    let bytes = frame.bytes();
                    if let Some(entry) = cache.entries.get_mut(&key) {
                        entry.bytes = bytes;
                    }
                    cache.bytes += bytes;
                    while cache.bytes > CACHE_BYTES {
                        let oldest = cache
                            .entries
                            .iter()
                            .filter(|(_, entry)| {
                                entry.bytes > 0 && Arc::strong_count(&entry.cell) == 1
                            })
                            .min_by_key(|(_, entry)| entry.tick)
                            .map(|(&key, _)| key);
                        let Some(entry) = oldest.and_then(|key| cache.entries.remove(&key)) else {
                            break;
                        };
                        cache.bytes -= entry.bytes;
                    }
                }
                Err(_) if cached => {
                    cache.entries.remove(&key);
                }
                _ => {}
            }
        }
        let waiters = {
            let mut state = self
                .cell
                .state
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            state.0 = Some(produced);
            std::mem::take(&mut state.1)
        };
        self.cell.ready.notify_all();
        for waiter in waiters {
            if waiter.waiting.fetch_sub(1, Ordering::AcqRel) == 1 {
                host.spawn(waiter.priority, move || waiter.run());
            }
        }
    }
}

impl Drop for Node {
    fn drop(&mut self) {
        let id = self.id();
        match &self.kind {
            Kind::Filter { instance, free, .. } => {
                let core = Arc::as_ptr(&self.host).cast_mut().cast();
                unsafe { free(*instance.get(), core, table()) };
            }
            Kind::Source(source) => {
                if let Some(release) = source.release {
                    unsafe { release(source.user) };
                }
            }
        }
        let mut cache = self
            .host
            .cache
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let mut freed = 0;
        cache.entries.retain(|key, entry| {
            if key.0 == id {
                freed += entry.bytes;
            }
            key.0 != id
        });
        cache.bytes -= freed;
    }
}

unsafe extern "system" fn config(
    _: *const c_char,
    _: *const c_char,
    _: *const c_char,
    _: i32,
    _: i32,
    _: Raw,
) {
}

unsafe extern "system" fn register(
    name: *const c_char,
    _: *const c_char,
    create: vs3::Create,
    _: Raw,
    plugin: Raw,
) {
    let functions = unsafe { &mut *plugin.cast::<Vec<(CString, vs3::Create)>>() };
    functions.push((unsafe { CStr::from_ptr(name) }.to_owned(), create));
}

type PluginInit = unsafe extern "system" fn(vs3::Config, vs3::Register, Raw);

impl Host {
    pub fn new(directory: Option<&std::path::Path>, threads: usize) -> Result<Arc<Self>, String> {
        let mut functions: Vec<(CString, vs3::Create)> = Vec::new();
        for name in ["svpflow1_vs", "svpflow2_vs"] {
            let file = libloading::library_filename(name);
            let path =
                directory.map_or_else(|| file.clone().into(), |directory| directory.join(&file));
            let library = unsafe { libloading::Library::new(&path) }
                .map_err(|error| format!("cannot load {}: {error}", path.display()))?;
            let init = *unsafe { library.get::<PluginInit>(b"VapourSynthPluginInit\0") }
                .map_err(|error| format!("{}: {error}", path.display()))?;
            unsafe { init(config, register, (&raw mut functions).cast()) };
            std::mem::forget(library);
        }
        let threads = if threads == 0 {
            std::thread::available_parallelism().map_or(1, usize::from)
        } else {
            threads
        };
        let queue = Arc::new(Queue::default());
        for _ in 0..threads {
            let queue = Arc::clone(&queue);
            std::thread::spawn(move || work(&queue));
        }
        Ok(Arc::new(Self {
            info: CoreInfo {
                version: c"open-svpflow".as_ptr(),
                core: 0,
                api: 3 << 16,
                num_threads: i32::try_from(threads).unwrap_or(1),
                max_framebuffer: 0,
                used_framebuffer: 0,
            },
            functions,
            cache: Mutex::default(),
            queue,
        }))
    }

    pub fn threads(&self) -> i32 {
        self.info.num_threads
    }

    pub fn busy(&self) -> bool {
        let cache = self.cache.lock().unwrap_or_else(PoisonError::into_inner);
        let used: usize = cache
            .entries
            .values()
            .filter(|entry| Arc::strong_count(&entry.cell) > 1)
            .map(|entry| entry.bytes)
            .sum();
        used > CACHE_BYTES / 2
    }

    fn spawn(&self, priority: i32, job: impl FnOnce() + Send + 'static) {
        let mut jobs = self
            .queue
            .jobs
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        jobs.1 += 1;
        let order = Reverse((priority, jobs.1));
        jobs.0.push(Job {
            order,
            run: Box::new(job),
        });
        drop(jobs);
        self.queue.ready.notify_one();
    }

    pub fn source(
        self: &Arc<Self>,
        info: VideoInfo,
        read: Read,
        release: Option<Release>,
        user: *mut c_void,
    ) -> Arc<Node> {
        Arc::new(Node {
            host: Arc::clone(self),
            info: UnsafeCell::new(info),
            kind: Kind::Source(Source {
                read,
                release,
                user,
                state: Mutex::default(),
            }),
        })
    }

    pub fn invoke(self: &Arc<Self>, name: &CStr, input: &Map) -> Result<Map, CString> {
        let create = self
            .functions
            .iter()
            .find(|(function, _)| function.as_c_str() == name)
            .ok_or_else(|| c"the function is not in the loaded plugins".to_owned())?
            .1;
        let mut output = Map::default();
        unsafe {
            create(
                std::ptr::from_ref(input).cast(),
                (&raw mut output).cast(),
                std::ptr::null_mut(),
                Arc::as_ptr(self).cast_mut().cast(),
                table(),
            );
        }
        match output.error.take() {
            Some(error) => Err(error),
            None => Ok(output),
        }
    }
}

impl Drop for Host {
    fn drop(&mut self) {
        self.queue
            .jobs
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .2 = true;
        self.queue.ready.notify_all();
    }
}

fn work(queue: &Queue) {
    let mut jobs = queue.jobs.lock().unwrap_or_else(PoisonError::into_inner);
    loop {
        if let Some(job) = jobs.0.pop() {
            drop(jobs);
            (job.run)();
            jobs = queue.jobs.lock().unwrap_or_else(PoisonError::into_inner);
        } else if jobs.2 {
            return;
        } else {
            jobs = queue
                .ready
                .wait(jobs)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }
}

unsafe fn host(core: Raw) -> Arc<Host> {
    unsafe {
        Arc::increment_strong_count(core.cast::<Host>());
        Arc::from_raw(core.cast::<Host>())
    }
}

unsafe extern "system" fn create_filter(
    input: ConstRaw,
    output: Raw,
    _: *const c_char,
    init: vs3::InitFilter,
    get_frame: vs3::GetFrame,
    free: vs3::FreeFilter,
    _: i32,
    _: i32,
    instance: Raw,
    core: Raw,
) {
    let node = Arc::new(Node {
        host: unsafe { host(core) },
        info: UnsafeCell::new(VideoInfo::empty()),
        kind: Kind::Filter {
            instance: UnsafeCell::new(instance),
            get_frame,
            free,
        },
    });
    let Kind::Filter { instance, .. } = &node.kind else {
        return;
    };
    unsafe {
        init(
            input,
            output,
            instance.get(),
            Arc::as_ptr(&node).cast_mut().cast(),
            core,
            table(),
        );
    }
    let output = unsafe { &mut *output.cast::<Map>() };
    if output.error.is_none() {
        output.set(c"clip", Value::Node(node));
    }
}

unsafe extern "system" fn set_video_info(info: *const VideoInfo, _: i32, node: Raw) {
    unsafe { *(*node.cast::<Node>()).info.get() = *info };
}

unsafe extern "system" fn get_video_info(node: Raw) -> *const VideoInfo {
    unsafe { (*node.cast::<Node>()).info.get() }
}

unsafe extern "system" fn get_core_info(core: Raw) -> *const CoreInfo {
    unsafe { &raw const (*core.cast::<Host>()).info }
}

unsafe extern "system" fn get_format_preset(id: i32, _: Raw) -> *const Format {
    vs3::preset(id).map_or(std::ptr::null(), vs3::intern)
}

unsafe extern "system" fn free_node(node: Raw) {
    drop(unsafe { Arc::from_raw(node.cast::<Node>()) });
}

unsafe extern "system" fn free_frame(frame: ConstRaw) {
    drop(unsafe { Arc::from_raw(frame.cast::<Frame>()) });
}

unsafe extern "system" fn new_video_frame(
    format: *const Format,
    width: i32,
    height: i32,
    source: ConstRaw,
    _: Raw,
) -> Raw {
    let props = unsafe { source.cast::<Frame>().as_ref() }
        .map(|source| unsafe { (*source.props.get()).clone() })
        .unwrap_or_default();
    let frame = Frame::new(unsafe { &*format }, width, height, props);
    Arc::into_raw(Arc::new(frame)).cast_mut().cast()
}

unsafe extern "system" fn copy_frame(frame: ConstRaw, _: Raw) -> Raw {
    let frame = unsafe { &*frame.cast::<Frame>() };
    let copy = Frame {
        planes: UnsafeCell::new(frame.planes().to_vec()),
        props: UnsafeCell::new(unsafe { (*frame.props.get()).clone() }),
    };
    Arc::into_raw(Arc::new(copy)).cast_mut().cast()
}

unsafe extern "system" fn get_stride(frame: ConstRaw, plane: i32) -> i32 {
    i32::try_from(unsafe { &*frame.cast::<Frame>() }.stride(plane)).unwrap_or(0)
}

unsafe extern "system" fn get_read_ptr(frame: ConstRaw, plane: i32) -> *const u8 {
    unsafe { &*frame.cast::<Frame>() }.read(plane)
}

unsafe extern "system" fn get_write_ptr(frame: Raw, plane: i32) -> *mut u8 {
    unsafe { (*frame.cast::<Frame>()).write(plane) }
}

unsafe extern "system" fn get_frame_props_ro(frame: ConstRaw) -> ConstRaw {
    unsafe { (*frame.cast::<Frame>()).props.get().cast_const().cast() }
}

unsafe extern "system" fn get_frame_props_rw(frame: Raw) -> Raw {
    unsafe { (*frame.cast::<Frame>()).props.get().cast() }
}

unsafe fn find<T>(
    map: ConstRaw,
    key: *const c_char,
    index: i32,
    error: *mut i32,
    pick: impl FnOnce(&Value) -> Option<T>,
) -> Option<T> {
    let map = unsafe { &*map.cast::<Map>() };
    let value = map
        .get(unsafe { CStr::from_ptr(key) }, index)
        .and_then(pick);
    if !error.is_null() {
        unsafe { *error = i32::from(value.is_none()) };
    }
    value
}

unsafe extern "system" fn prop_get_int(
    map: ConstRaw,
    key: *const c_char,
    index: i32,
    error: *mut i32,
) -> i64 {
    let pick = |value: &Value| match value {
        Value::Int(value) => Some(*value),
        _ => None,
    };
    unsafe { find(map, key, index, error, pick) }.unwrap_or(0)
}

unsafe extern "system" fn prop_get_float(
    map: ConstRaw,
    key: *const c_char,
    index: i32,
    error: *mut i32,
) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let pick = |value: &Value| match value {
        Value::Float(value) => Some(*value),
        Value::Int(value) => Some(*value as f64),
        _ => None,
    };
    unsafe { find(map, key, index, error, pick) }.unwrap_or(0.0)
}

unsafe extern "system" fn prop_get_data(
    map: ConstRaw,
    key: *const c_char,
    index: i32,
    error: *mut i32,
) -> *const c_char {
    let pick = |value: &Value| match value {
        Value::Data(value) => Some(value.as_ptr()),
        _ => None,
    };
    unsafe { find(map, key, index, error, pick) }.unwrap_or(std::ptr::null())
}

unsafe extern "system" fn prop_get_data_size(
    map: ConstRaw,
    key: *const c_char,
    index: i32,
    error: *mut i32,
) -> i32 {
    let pick = |value: &Value| match value {
        Value::Data(value) => i32::try_from(value.as_bytes().len()).ok(),
        _ => None,
    };
    unsafe { find(map, key, index, error, pick) }.unwrap_or(0)
}

unsafe extern "system" fn prop_get_node(
    map: ConstRaw,
    key: *const c_char,
    index: i32,
    error: *mut i32,
) -> Raw {
    let pick = |value: &Value| match value {
        Value::Node(node) => Some(Arc::into_raw(Arc::clone(node)).cast_mut().cast()),
        _ => None,
    };
    unsafe { find(map, key, index, error, pick) }.unwrap_or(std::ptr::null_mut())
}

unsafe extern "system" fn prop_set_int(
    map: Raw,
    key: *const c_char,
    value: i64,
    append: i32,
) -> i32 {
    let map = unsafe { &mut *map.cast::<Map>() };
    let key = unsafe { CStr::from_ptr(key) };
    if append == 0 {
        map.values.retain(|(name, _)| name.as_c_str() != key);
    }
    map.set(key, Value::Int(value));
    0
}

unsafe extern "system" fn set_error(map: Raw, message: *const c_char) {
    unsafe { (*map.cast::<Map>()).error = Some(CStr::from_ptr(message).to_owned()) };
}

unsafe fn context<'a>(task: Raw) -> &'a mut FrameContext {
    unsafe { &mut *(*task.cast::<Task>()).context.get() }
}

unsafe fn node(node: Raw) -> Arc<Node> {
    unsafe {
        Arc::increment_strong_count(node.cast::<Node>());
        Arc::from_raw(node.cast::<Node>())
    }
}

unsafe extern "system" fn set_filter_error(message: *const c_char, task: Raw) {
    unsafe { context(task).error = Some(Arc::from(CStr::from_ptr(message))) };
}

unsafe extern "system" fn get_frame_filter(n: i32, source: Raw, task: Raw) -> ConstRaw {
    let context = unsafe { context(task) };
    let key = unsafe { (source as usize, (*source.cast::<Node>()).frame(n)) };
    let produced = context
        .frames
        .iter()
        .find(|(frame, _)| *frame == key)
        .and_then(|(_, cell)| {
            let state = cell.state.lock().unwrap_or_else(PoisonError::into_inner);
            state.0.clone()
        });
    match produced {
        Some(Ok(frame)) => Arc::into_raw(frame).cast(),
        Some(Err(error)) => {
            context.error = Some(error);
            std::ptr::null()
        }
        None => {
            context.error = Some(Arc::from(c"a filter read a frame that it did not request"));
            std::ptr::null()
        }
    }
}

unsafe extern "system" fn request_frame_filter(n: i32, source: Raw, task: Raw) {
    let context = unsafe { context(task) };
    let source = unsafe { node(source) };
    let n = source.frame(n);
    if !context
        .requests
        .iter()
        .any(|(node, frame)| Arc::ptr_eq(node, &source) && *frame == n)
    {
        context.requests.push((source, n));
    }
}

fn table() -> ConstRaw {
    static TABLE: OnceLock<Box<[usize]>> = OnceLock::new();
    TABLE
        .get_or_init(|| {
            vs3::table(&[
                (vs3::GET_CORE_INFO, get_core_info as *const () as usize),
                (vs3::FREE_FRAME, free_frame as *const () as usize),
                (vs3::FREE_NODE, free_node as *const () as usize),
                (vs3::NEW_VIDEO_FRAME, new_video_frame as *const () as usize),
                (vs3::COPY_FRAME, copy_frame as *const () as usize),
                (vs3::CREATE_FILTER, create_filter as *const () as usize),
                (vs3::SET_ERROR, set_error as *const () as usize),
                (
                    vs3::SET_FILTER_ERROR,
                    set_filter_error as *const () as usize,
                ),
                (
                    vs3::GET_FORMAT_PRESET,
                    get_format_preset as *const () as usize,
                ),
                (
                    vs3::GET_FRAME_FILTER,
                    get_frame_filter as *const () as usize,
                ),
                (
                    vs3::REQUEST_FRAME_FILTER,
                    request_frame_filter as *const () as usize,
                ),
                (vs3::GET_STRIDE, get_stride as *const () as usize),
                (vs3::GET_READ_PTR, get_read_ptr as *const () as usize),
                (vs3::GET_WRITE_PTR, get_write_ptr as *const () as usize),
                (vs3::GET_VIDEO_INFO, get_video_info as *const () as usize),
                (vs3::SET_VIDEO_INFO, set_video_info as *const () as usize),
                (
                    vs3::GET_FRAME_PROPS_RO,
                    get_frame_props_ro as *const () as usize,
                ),
                (
                    vs3::GET_FRAME_PROPS_RW,
                    get_frame_props_rw as *const () as usize,
                ),
                (vs3::PROP_GET_INT, prop_get_int as *const () as usize),
                (vs3::PROP_GET_FLOAT, prop_get_float as *const () as usize),
                (vs3::PROP_GET_DATA, prop_get_data as *const () as usize),
                (
                    vs3::PROP_GET_DATA_SIZE,
                    prop_get_data_size as *const () as usize,
                ),
                (vs3::PROP_GET_NODE, prop_get_node as *const () as usize),
                (vs3::PROP_SET_INT, prop_set_int as *const () as usize),
            ])
        })
        .as_ptr()
        .cast()
}
