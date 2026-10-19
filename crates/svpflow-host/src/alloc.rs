use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;

const MIN: usize = 64 << 10;
const MAX: usize = 64 << 20;
const SLOTS: usize = 32;
const BUDGET: usize = 128 << 20;

#[derive(Clone, Copy)]
struct Slot {
    size: usize,
    ptr: usize,
    age: u64,
}

const EMPTY: Slot = Slot {
    size: 0,
    ptr: 0,
    age: 0,
};

struct Held {
    slots: [Slot; SLOTS],
    bytes: usize,
    clock: u64,
}

pub struct Cached(Mutex<Held>);

impl Cached {
    pub const fn new() -> Self {
        Self(Mutex::new(Held {
            slots: [EMPTY; SLOTS],
            bytes: 0,
            clock: 0,
        }))
    }

    pub fn trim(&self) {
        let slots = {
            let Ok(mut held) = self.0.lock() else {
                return;
            };
            held.bytes = 0;
            std::mem::replace(&mut held.slots, [EMPTY; SLOTS])
        };
        for slot in slots.iter().filter(|slot| slot.size > 0) {
            unsafe { System.dealloc(slot.ptr as *mut u8, class_layout(slot.size, 16)) };
        }
    }
}

impl Default for Cached {
    fn default() -> Self {
        Self::new()
    }
}

fn class_layout(size: usize, align: usize) -> Layout {
    unsafe { Layout::from_size_align_unchecked(size, align) }
}

// sizes round up to an eighth of their power of two, so blocks of a class are interchangeable
fn class(layout: Layout) -> Option<Layout> {
    if layout.align() > 16 || !(MIN..=MAX).contains(&layout.size()) {
        return None;
    }
    let step = layout.size().next_power_of_two() / 8;
    Some(class_layout(
        layout.size().next_multiple_of(step),
        layout.align(),
    ))
}

fn effective(layout: Layout) -> Layout {
    class(layout).unwrap_or(layout)
}

impl Cached {
    fn take(&self, size: usize) -> Option<*mut u8> {
        let mut held = self.0.lock().ok()?;
        let slot = held.slots.iter_mut().find(|slot| slot.size == size)?;
        let ptr = slot.ptr as *mut u8;
        *slot = EMPTY;
        held.bytes -= size;
        Some(ptr)
    }
}

unsafe impl GlobalAlloc for Cached {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let Some(layout) = class(layout) else {
            return unsafe { System.alloc(layout) };
        };
        self.take(layout.size())
            .unwrap_or_else(|| unsafe { System.alloc(layout) })
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let Some(class) = class(layout) else {
            return unsafe { System.alloc_zeroed(layout) };
        };
        match self.take(class.size()) {
            Some(ptr) => {
                unsafe { ptr.write_bytes(0, layout.size()) };
                ptr
            }
            None => unsafe { System.alloc_zeroed(class) },
        }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let old = effective(layout);
        let new = effective(class_layout(new_size, layout.align()));
        if old.size() == new.size() {
            return ptr;
        }
        unsafe { System.realloc(ptr, old, new.size()) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let Some(layout) = class(layout) else {
            return unsafe { System.dealloc(ptr, layout) };
        };
        let mut evicted = [EMPTY; SLOTS];
        let mut count = 0;
        let mut stored = false;
        if let Ok(mut held) = self.0.lock() {
            while held.bytes + layout.size() > BUDGET || held.slots.iter().all(|slot| slot.size > 0)
            {
                let Some(oldest) = held
                    .slots
                    .iter_mut()
                    .filter(|slot| slot.size > 0)
                    .min_by_key(|slot| slot.age)
                else {
                    break;
                };
                evicted[count] = std::mem::replace(oldest, EMPTY);
                held.bytes -= evicted[count].size;
                count += 1;
            }
            held.clock += 1;
            let age = held.clock;
            if let Some(slot) = held.slots.iter_mut().find(|slot| slot.size == 0) {
                *slot = Slot {
                    size: layout.size(),
                    ptr: ptr as usize,
                    age,
                };
                held.bytes += layout.size();
                stored = true;
            }
        }
        for slot in &evicted[..count] {
            unsafe { System.dealloc(slot.ptr as *mut u8, class_layout(slot.size, 16)) };
        }
        if !stored {
            unsafe { System.dealloc(ptr, layout) };
        }
    }
}
