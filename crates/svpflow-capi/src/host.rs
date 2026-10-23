use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::cell::UnsafeCell;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::ffi::{CStr, CString, c_char, c_void};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex, MutexGuard, OnceLock, PoisonError};
use std::thread::JoinHandle;

use svpflow_host::vs3::{self, ConstRaw, CoreInfo, Format, Raw, VideoInfo};

const ALIGN: usize = 64;
pub const READ_AHEAD: i32 = 64;

#[cfg(target_os = "linux")]
fn memory() -> Option<usize> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = text.lines().find(|line| line.starts_with("MemTotal:"))?;
    Some(line.split_whitespace().nth(1)?.parse::<usize>().ok()? * 1024)
}

#[cfg(windows)]
fn memory() -> Option<usize> {
    #[repr(C)]
    struct Status {
        length: u32,
        load: u32,
        total: u64,
        rest: [u64; 6],
    }
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(status: *mut Status) -> i32;
    }
    let mut status = Status {
        length: u32::try_from(size_of::<Status>()).ok()?,
        load: 0,
        total: 0,
        rest: [0; 6],
    };
    let done = unsafe { GlobalMemoryStatusEx(&raw mut status) } != 0;
    done.then(|| usize::try_from(status.total).ok()).flatten()
}

#[cfg(target_os = "macos")]
fn memory() -> Option<usize> {
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            value: *mut c_void,
            size: *mut usize,
            new: *mut c_void,
            new_size: usize,
        ) -> i32;
    }
    let (mut total, mut size) = (0u64, size_of::<u64>());
    let none = std::ptr::null_mut();
    let value = (&raw mut total).cast();
    let code = unsafe { sysctlbyname(c"hw.memsize".as_ptr(), value, &raw mut size, none, 0) };
    (code == 0).then(|| usize::try_from(total).ok()).flatten()
}

#[cfg(not(any(target_os = "linux", windows, target_os = "macos")))]
fn memory() -> Option<usize> {
    None
}

fn budget() -> usize {
    static BUDGET: OnceLock<usize> = OnceLock::new();
    *BUDGET
        .get_or_init(|| memory().map_or(1 << 30, |memory| (memory / 16).clamp(256 << 20, 1 << 30)))
}

pub type Read = unsafe extern "C" fn(*mut c_void, i32, *const *mut u8, *const isize) -> i32;
pub type Release = unsafe extern "C" fn(*mut c_void);
pub type Need = unsafe extern "C" fn(*mut c_void, i32, *mut i32) -> i32;
pub type Render = unsafe extern "C" fn(
    *mut c_void,
    i32,
    *const *const Frame,
    i32,
    *const *mut u8,
    *const isize,
) -> i32;

pub const SOURCE_FAILED: &CStr = c"the source callback failed";
pub const RENDER_FAILED: &CStr = c"the render callback failed";

const NEEDED: usize = 8;

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
        if pool.1 + self.len <= budget() / 4 {
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

struct Mapped {
    input: Arc<Node>,
    need: Need,
    render: Render,
    release: Option<Release>,
    user: *mut c_void,
}

enum Kind {
    Source(Source),
    Mapped(Mapped),
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

struct Queue {
    jobs: BinaryHeap<Job>,
    count: u64,
    target: usize,
    live: usize,
}

struct Pool {
    queue: Mutex<Queue>,
    ready: Condvar,
    exited: Condvar,
    handles: Mutex<Vec<JoinHandle<()>>>,
}

static WANT: AtomicUsize = AtomicUsize::new(0);

static WORKERS: Pool = Pool {
    queue: Mutex::new(Queue {
        jobs: BinaryHeap::new(),
        count: 0,
        target: 0,
        live: 0,
    }),
    ready: Condvar::new(),
    exited: Condvar::new(),
    handles: Mutex::new(Vec::new()),
};

thread_local! {
    static WORKER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn threads() -> usize {
    match WANT.load(Ordering::Relaxed) {
        0 => std::thread::available_parallelism().map_or(1, usize::from),
        count => count,
    }
}

fn pool() -> &'static Pool {
    &WORKERS
}

pub fn set_threads(count: usize) {
    WANT.store(count, Ordering::Relaxed);
    let running = WORKERS
        .queue
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .target
        > 0;
    if running {
        WORKERS.resize(threads());
    }
    for plugin in PLUGINS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .iter()
    {
        if let Some(set) = plugin.set {
            unsafe { set(count) };
        }
    }
}

pub fn stop() {
    WORKERS.resize(0);
    for plugin in PLUGINS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .iter()
    {
        if let Some(stop) = plugin.stop {
            unsafe { stop() };
        }
    }
}

struct Plugin {
    set: Option<unsafe extern "C" fn(usize)>,
    stop: Option<unsafe extern "C" fn()>,
}

static PLUGINS: Mutex<Vec<Plugin>> = Mutex::new(Vec::new());

impl Pool {
    fn spawn(&'static self, priority: i32, job: impl FnOnce() + Send + 'static) {
        let mut queue = self.queue.lock().unwrap_or_else(PoisonError::into_inner);
        queue.count += 1;
        let order = Reverse((priority, queue.count));
        queue.jobs.push(Job {
            order,
            run: Box::new(job),
        });
        if queue.target == 0 && queue.live == 0 {
            queue.target = threads();
        }
        self.grow(&mut queue);
        drop(queue);
        self.ready.notify_one();
    }

    fn grow(&'static self, queue: &mut Queue) {
        let mut handles = self.handles.lock().unwrap_or_else(PoisonError::into_inner);
        handles.retain(|handle| !handle.is_finished());
        while queue.live < queue.target {
            queue.live += 1;
            handles.push(std::thread::spawn(move || self.work()));
        }
    }

    // count 0 stops the workers after the queue is empty
    fn resize(&'static self, count: usize) {
        let mut queue = self.queue.lock().unwrap_or_else(PoisonError::into_inner);
        queue.target = count;
        self.grow(&mut queue);
        self.ready.notify_all();
        // a worker cannot wait for itself
        if count > 0 || WORKER.get() {
            return;
        }
        let queue = self
            .exited
            .wait_while(queue, |queue| queue.live > 0 && queue.target == 0)
            .unwrap_or_else(PoisonError::into_inner);
        if queue.target > 0 {
            return;
        }
        drop(queue);
        let handles =
            std::mem::take(&mut *self.handles.lock().unwrap_or_else(PoisonError::into_inner));
        for handle in handles {
            let _ = handle.join();
        }
    }

    fn work(&self) {
        WORKER.set(true);
        let mut queue = self.queue.lock().unwrap_or_else(PoisonError::into_inner);
        loop {
            if let Some(job) = queue.jobs.pop() {
                drop(queue);
                (job.run)();
                queue = self.queue.lock().unwrap_or_else(PoisonError::into_inner);
            } else if queue.live > queue.target {
                queue.live -= 1;
                self.exited.notify_all();
                return;
            } else {
                queue = self
                    .ready
                    .wait(queue)
                    .unwrap_or_else(PoisonError::into_inner);
            }
        }
    }
}

fn cache() -> MutexGuard<'static, Cache> {
    static CACHE: LazyLock<Mutex<Cache>> = LazyLock::new(Mutex::default);
    CACHE.lock().unwrap_or_else(PoisonError::into_inner)
}

pub fn busy() -> bool {
    let cache = cache();
    let used: usize = cache
        .entries
        .values()
        .filter(|entry| Arc::strong_count(&entry.cell) > 1)
        .map(|entry| entry.bytes)
        .sum();
    used > budget() / 2
}

pub struct Host {
    info: CoreInfo,
    functions: &'static [(CString, vs3::Create)],
}

unsafe impl Send for Host {}
unsafe impl Sync for Host {}

impl Node {
    pub fn info(&self) -> VideoInfo {
        unsafe { *self.info.get() }
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
        let mut cache = cache();
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
            pool().spawn(priority, move || task.begin());
        } else {
            task.begin();
        }
        cell
    }

    pub fn get(self: &Arc<Self>, n: i32) -> Produced {
        let cell = self.start(n, n);
        let state = cell.state.lock().unwrap_or_else(PoisonError::into_inner);
        let state = cell.ready.wait_while(state, |state| state.0.is_none());
        let state = state.unwrap_or_else(PoisonError::into_inner);
        state
            .0
            .clone()
            .unwrap_or_else(|| Err(Arc::from(c"no frame")))
    }

    pub fn window(&self) -> i32 {
        let frames = budget() / 4 / self.bytes().max(1);
        let frames = frames.min(8 * threads()).clamp(8, READ_AHEAD as usize);
        i32::try_from(frames).unwrap_or(READ_AHEAD)
    }

    fn bytes(&self) -> usize {
        let info = self.info();
        let format = unsafe { &*info.format.cast::<Format>() };
        let luma = usize::try_from(info.width * info.height * format.bytes_per_sample).unwrap_or(0);
        let chroma = luma >> (format.sub_sampling_w + format.sub_sampling_h);
        luma + chroma * usize::try_from(format.num_planes - 1).unwrap_or(0)
    }

    fn blank(&self) -> (Frame, [*mut u8; 3], [isize; 3]) {
        let info = self.info();
        let format = unsafe { &*info.format.cast::<Format>() };
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
        (frame, planes, strides)
    }

    fn fill(&self, source: &Source, n: i32) -> Produced {
        let (frame, planes, strides) = self.blank();
        if unsafe { (source.read)(source.user, n, planes.as_ptr(), strides.as_ptr()) } != 0 {
            return Err(Arc::from(SOURCE_FAILED));
        }
        Ok(Arc::new(frame))
    }

    fn render(&self, mapped: &Mapped, n: i32, inputs: &[((usize, i32), Cell)]) -> Produced {
        let mut frames = Vec::with_capacity(inputs.len());
        for (_, cell) in inputs {
            let state = cell.state.lock().unwrap_or_else(PoisonError::into_inner);
            match &state.0 {
                Some(Ok(frame)) => frames.push(Arc::clone(frame)),
                Some(Err(error)) => return Err(Arc::clone(error)),
                None => return Err(Arc::from(c"an input frame is missing")),
            }
        }
        let pointers: Vec<*const Frame> = frames.iter().map(Arc::as_ptr).collect();
        let count = i32::try_from(pointers.len()).unwrap_or(0);
        let (frame, planes, strides) = self.blank();
        let (planes, strides) = (planes.as_ptr(), strides.as_ptr());
        let user = mapped.user;
        if unsafe { (mapped.render)(user, n, pointers.as_ptr(), count, planes, strides) } != 0 {
            return Err(Arc::from(RENDER_FAILED));
        }
        Ok(Arc::new(frame))
    }

    fn read(&self, source: &Source, n: i32) -> Produced {
        let mut state = source.state.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(index) = state.ahead.iter().position(|(frame, _)| *frame == n) {
            return Ok(state.ahead.swap_remove(index).1);
        }
        let reach = self.window() + 16;
        let first = if n < state.next || n - state.next > reach {
            n
        } else {
            state.next
        };
        let mut wanted = None;
        for index in first..=n {
            let frame = self.fill(source, index)?;
            if index == n {
                wanted = Some(frame);
            } else {
                state.ahead.retain(|(frame, _)| *frame > index - reach);
                state.ahead.push((index, frame));
            }
        }
        state.next = state.next.max(n + 1);
        wanted.ok_or_else(|| Arc::from(SOURCE_FAILED))
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

    fn dropped(&self) -> bool {
        let key = (self.node.id(), self.n);
        let mut cache = cache();
        let cached = cache
            .entries
            .get(&key)
            .is_some_and(|entry| Arc::ptr_eq(&entry.cell, &self.cell));
        if Arc::strong_count(&self.cell) > 1 + usize::from(cached) {
            return false;
        }
        if cached {
            cache.entries.remove(&key);
        }
        true
    }

    fn begin(self: Arc<Self>) {
        match &self.node.kind {
            Kind::Source(source) => {
                if self.dropped() {
                    return;
                }
                let produced = self.node.read(source, self.n);
                return self.complete(produced);
            }
            Kind::Mapped(mapped) => {
                let mut frames = [0; NEEDED];
                let count = unsafe { (mapped.need)(mapped.user, self.n, frames.as_mut_ptr()) };
                let count = usize::try_from(count).unwrap_or(0).min(NEEDED);
                let input = &mapped.input;
                let requests = frames[..count]
                    .iter()
                    .map(|&n| (Arc::clone(input), input.frame(n)));
                unsafe { &mut *self.context.get() }.requests = requests.collect();
                return self.wait();
            }
            Kind::Filter { .. } => {}
        }
        let frame = self.call(vs3::AR_INITIAL);
        let context = unsafe { &*self.context.get() };
        if !frame.is_null() || context.error.is_some() {
            return self.finish(frame);
        }
        self.wait();
    }

    fn wait(self: Arc<Self>) {
        let context = unsafe { &mut *self.context.get() };
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
            pool().spawn(self.priority, move || self.run());
        }
    }

    fn run(self: Arc<Self>) {
        if self.dropped() {
            self.call(vs3::AR_ERROR);
            unsafe { &mut *self.context.get() }.frames.clear();
            return;
        }
        if let Kind::Mapped(mapped) = &self.node.kind {
            let frames = std::mem::take(&mut unsafe { &mut *self.context.get() }.frames);
            let produced = self.node.render(mapped, self.n, &frames);
            return self.complete(produced);
        }
        let frame = self.call(vs3::AR_ALL_FRAMES_READY);
        let context = unsafe { &*self.context.get() };
        // a filter may request more frames in its second call
        if frame.is_null() && context.error.is_none() && !context.requests.is_empty() {
            return self.wait();
        }
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
        let key = (self.node.id(), self.n);
        {
            let mut cache = cache();
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
                    while cache.bytes > budget() {
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
                pool().spawn(waiter.priority, move || waiter.run());
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
            Kind::Source(Source { release, user, .. })
            | Kind::Mapped(Mapped { release, user, .. }) => {
                if let Some(release) = release {
                    unsafe { release(*user) };
                }
            }
        }
        let mut cache = cache();
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
    pub fn new(directory: Option<&std::path::Path>) -> Result<Arc<Self>, String> {
        static FUNCTIONS: OnceLock<Result<Vec<(CString, vs3::Create)>, String>> = OnceLock::new();
        let functions = FUNCTIONS.get_or_init(|| {
            let mut functions = Vec::new();
            for name in ["svpflow1_vs", "svpflow2_vs"] {
                let file = libloading::library_filename(name);
                let path = directory
                    .map_or_else(|| file.clone().into(), |directory| directory.join(&file));
                let library = unsafe { libloading::Library::new(&path) }
                    .map_err(|error| format!("cannot load {}: {error}", path.display()))?;
                let init = *unsafe { library.get::<PluginInit>(b"VapourSynthPluginInit\0") }
                    .map_err(|error| format!("{}: {error}", path.display()))?;
                unsafe { init(config, register, (&raw mut functions).cast()) };
                let set =
                    unsafe { library.get::<unsafe extern "C" fn(usize)>(b"osvp_plugin_threads\0") }
                        .ok()
                        .map(|symbol| *symbol);
                let stop = unsafe { library.get::<unsafe extern "C" fn()>(b"osvp_plugin_stop\0") }
                    .ok()
                    .map(|symbol| *symbol);
                if let Some(set) = set {
                    unsafe { set(WANT.load(Ordering::Relaxed)) };
                }
                PLUGINS
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push(Plugin { set, stop });
                std::mem::forget(library);
            }
            Ok(functions)
        });
        Ok(Arc::new(Self {
            info: CoreInfo {
                version: vs3::LATER_REQUESTS.as_ptr(),
                core: 0,
                api: 3 << 16,
                num_threads: i32::try_from(threads()).unwrap_or(1),
                max_framebuffer: 0,
                used_framebuffer: 0,
            },
            functions: functions.as_ref().map_err(Clone::clone)?,
        }))
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

    pub fn map(
        self: &Arc<Self>,
        info: VideoInfo,
        input: Arc<Node>,
        need: Need,
        render: Render,
        release: Option<Release>,
        user: *mut c_void,
    ) -> Arc<Node> {
        Arc::new(Node {
            host: Arc::clone(self),
            info: UnsafeCell::new(info),
            kind: Kind::Mapped(Mapped {
                input,
                need,
                render,
                release,
                user,
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

// vapoursynth allows freeing null
unsafe extern "system" fn free_frame(frame: ConstRaw) {
    if !frame.is_null() {
        drop(unsafe { Arc::from_raw(frame.cast::<Frame>()) });
    }
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
