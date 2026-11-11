use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex};

pub(crate) struct Mask {
    pub(crate) width: i32,
    pub(crate) height: i32,
    pub(crate) pixels: Vec<u8>,
}

static MASKS: Mutex<Vec<(i64, Arc<Mask>)>> = Mutex::new(Vec::new());
static NEXT: AtomicI64 = AtomicI64::new(1);

pub(crate) fn get(id: i64, width: i32, height: i32) -> Option<Arc<Mask>> {
    let masks = MASKS.lock().ok()?;
    let (_, mask) = masks.iter().find(|(key, _)| *key == id)?;
    ((mask.width, mask.height) == (width, height)).then(|| Arc::clone(mask))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn osvp_plugin_mask(pixels: *const u8, width: i32, height: i32) -> i64 {
    let (Ok(w), Ok(h)) = (usize::try_from(width), usize::try_from(height)) else {
        return 0;
    };
    if pixels.is_null() || w == 0 || h == 0 {
        return 0;
    }
    let pixels = unsafe { std::slice::from_raw_parts(pixels, w * h) }.to_vec();
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    let mask = Arc::new(Mask {
        width,
        height,
        pixels,
    });
    MASKS.lock().map_or(0, |mut masks| {
        masks.push((id, mask));
        id
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn osvp_plugin_unmask(id: i64) {
    if let Ok(mut masks) = MASKS.lock() {
        masks.retain(|(key, _)| *key != id);
    }
}
