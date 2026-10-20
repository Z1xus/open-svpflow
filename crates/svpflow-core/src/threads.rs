use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::JoinHandle;

use rayon::{Scope, ThreadPool, ThreadPoolBuilder};

static WANT: AtomicUsize = AtomicUsize::new(0);
static POOL: Mutex<Option<Arc<Pool>>> = Mutex::new(None);

struct Pool {
    pool: Option<ThreadPool>,
    handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
    size: usize,
    users: Mutex<usize>,
    idle: Condvar,
}

impl Pool {
    fn new() -> Self {
        let handles = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&handles);
        let size = pool_size();
        let pool = ThreadPoolBuilder::new()
            .num_threads(size)
            .spawn_handler(move |thread| {
                let mut builder = std::thread::Builder::new();
                if let Some(name) = thread.name() {
                    builder = builder.name(name.to_owned());
                }
                if let Some(size) = thread.stack_size() {
                    builder = builder.stack_size(size);
                }
                let handle = builder.spawn(|| thread.run())?;
                sink.lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push(handle);
                Ok(())
            })
            .build()
            .expect("thread pool");
        Self {
            pool: Some(pool),
            handles,
            size,
            users: Mutex::new(0),
            idle: Condvar::new(),
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.pool = None;
        let handles =
            std::mem::take(&mut *self.handles.lock().unwrap_or_else(PoisonError::into_inner));
        let me = std::thread::current().id();
        for handle in handles {
            if handle.thread().id() != me {
                let _ = handle.join();
            }
        }
    }
}

struct Use(Arc<Pool>);

impl Drop for Use {
    fn drop(&mut self) {
        *self.0.users.lock().unwrap_or_else(PoisonError::into_inner) -= 1;
        self.0.idle.notify_all();
    }
}

pub fn workers() -> usize {
    match WANT.load(Ordering::Relaxed) {
        0 => std::thread::available_parallelism().map_or(1, usize::from),
        n => n,
    }
}

// host frame threads also run filters, so the pool gets three quarters
fn pool_size() -> usize {
    (workers() * 3).div_ceil(4).max(2)
}

pub fn set(count: usize) {
    WANT.store(count, Ordering::Relaxed);
}

pub fn set_default(count: usize) {
    let _ = WANT.compare_exchange(0, count, Ordering::Relaxed, Ordering::Relaxed);
}

pub fn shutdown() {
    let taken = POOL.lock().unwrap_or_else(PoisonError::into_inner).take();
    let Some(pool) = taken else {
        return;
    };
    // a pool thread cannot wait for itself
    if rayon::current_thread_index().is_none() {
        let users = pool.users.lock().unwrap_or_else(PoisonError::into_inner);
        drop(
            pool.idle
                .wait_while(users, |users| *users > 0)
                .unwrap_or_else(PoisonError::into_inner),
        );
    }
}

fn with<R>(f: impl FnOnce(&ThreadPool) -> R) -> R {
    let (used, stale) = {
        let mut slot = POOL.lock().unwrap_or_else(PoisonError::into_inner);
        let stale = if slot.as_ref().is_none_or(|pool| pool.size != pool_size()) {
            slot.replace(Arc::new(Pool::new()))
        } else {
            None
        };
        let pool = Arc::clone(slot.as_ref().expect("pool"));
        *pool.users.lock().unwrap_or_else(PoisonError::into_inner) += 1;
        (Use(pool), stale)
    };
    drop(stale);
    f(used.0.pool.as_ref().expect("pool"))
}

pub fn run<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    if rayon::current_thread_index().is_some() {
        f()
    } else {
        with(|pool| pool.install(f))
    }
}

pub fn join<A: Send, B: Send>(
    a: impl FnOnce() -> A + Send,
    b: impl FnOnce() -> B + Send,
) -> (A, B) {
    run(|| rayon::join(a, b))
}

pub fn scope<'s, R: Send>(f: impl FnOnce(&Scope<'s>) -> R + Send) -> R {
    run(|| rayon::scope(f))
}

pub fn spawn(f: impl FnOnce() + Send + 'static) {
    with(|pool| pool.spawn(f));
}
