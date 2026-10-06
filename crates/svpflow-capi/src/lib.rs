#![allow(clippy::missing_safety_doc)]

mod host;

use std::cell::RefCell;
use std::collections::VecDeque;
use std::ffi::{CStr, CString, c_char, c_void};
use std::sync::{Arc, Mutex, PoisonError};

use host::{Frame, Host, Map, Node, Pending, Read, Value};
use svpflow_host::vs3;

const FORMATS: [i32; 4] = [
    vs3::PF_YUV420P8,
    vs3::PF_YUV444P8,
    vs3::PF_YUV420P10,
    vs3::PF_YUV420P16,
];

#[repr(C)]
pub struct VideoInfo {
    format: i32,
    width: i32,
    height: i32,
    fps_num: i64,
    fps_den: i64,
    num_frames: i32,
}

pub struct Context(Arc<Host>);

pub struct Clip {
    node: Arc<Node>,
    ahead: Mutex<(i32, VecDeque<(i32, Pending)>)>,
}

thread_local! {
    static ERROR: RefCell<CString> = RefCell::default();
}

fn fail<T>(message: impl Into<Vec<u8>>) -> *mut T {
    let message = CString::new(message).unwrap_or_default();
    ERROR.with(|error| *error.borrow_mut() = message);
    std::ptr::null_mut()
}

fn clip(node: Arc<Node>) -> *mut Clip {
    Box::into_raw(Box::new(Clip {
        node,
        ahead: Mutex::default(),
    }))
}

unsafe fn text(value: *const c_char) -> CString {
    if value.is_null() {
        c"{}".to_owned()
    } else {
        unsafe { CStr::from_ptr(value) }.to_owned()
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn osvp_error() -> *const c_char {
    ERROR.with(|error| error.borrow().as_ptr())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn osvp_create(plugin_dir: *const c_char, threads: i32) -> *mut Context {
    let directory = unsafe { plugin_dir.as_ref() }.map(|directory| {
        unsafe { CStr::from_ptr(directory) }
            .to_string_lossy()
            .into_owned()
    });
    let directory = directory.as_deref().map(std::path::Path::new);
    match Host::new(directory, usize::try_from(threads).unwrap_or(0)) {
        Ok(host) => Box::into_raw(Box::new(Context(host))),
        Err(error) => fail(error),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn osvp_destroy(context: *mut Context) {
    if !context.is_null() {
        drop(unsafe { Box::from_raw(context) });
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn osvp_source(
    context: *const Context,
    info: *const VideoInfo,
    read: Read,
    user: *mut c_void,
) -> *mut Clip {
    let (context, info) = unsafe { (&*context, &*info) };
    let format = usize::try_from(info.format)
        .ok()
        .and_then(|format| FORMATS.get(format))
        .and_then(|&id| vs3::preset(id));
    let Some(format) = format else {
        return fail("unknown format");
    };
    if info.width <= 0
        || info.height <= 0
        || info.num_frames <= 0
        || info.fps_num <= 0
        || info.fps_den <= 0
    {
        return fail("the size, the frame rate and the frame count must be positive");
    }
    let info = vs3::VideoInfo {
        format: vs3::intern(format).cast(),
        fps_num: info.fps_num,
        fps_den: info.fps_den,
        width: info.width,
        height: info.height,
        num_frames: info.num_frames,
        flags: 0,
    };
    clip(context.0.source(info, read, user))
}

unsafe fn vectors(
    host: &Arc<Host>,
    source: &Arc<Node>,
    super_opt: *const c_char,
    analyse_opt: *const c_char,
    smooth_opt: *const c_char,
    half: bool,
) -> Result<Map, CString> {
    let missing = || c"the plugin returned no clip".to_owned();
    let call = |name: &CStr, values: Vec<(&CStr, Value)>| {
        let mut input = Map::default();
        for (key, value) in values {
            input.set(key, value);
        }
        host.invoke(name, &input)
    };
    let clip = |node: &Arc<Node>| Value::Node(Arc::clone(node));
    let analysed = if half {
        call(c"Halve", vec![(c"clip", clip(source))])?
            .node(c"clip")
            .ok_or_else(missing)?
    } else {
        Arc::clone(source)
    };
    let options = Value::Data(unsafe { text(super_opt) });
    let mut supers = call(
        c"Super",
        vec![(c"clip", clip(&analysed)), (c"opt", options)],
    )?;
    let sdata = supers.int(c"data").ok_or_else(missing)?;
    let mut found = call(
        c"Analyse",
        vec![
            (c"clip", clip(&supers.node(c"clip").ok_or_else(missing)?)),
            (c"sdata", Value::Int(sdata)),
            (c"src", clip(&analysed)),
            (c"opt", Value::Data(unsafe { text(analyse_opt) })),
        ],
    )?;
    if half {
        found = call(
            c"DoubleVectors",
            vec![
                (c"clip", clip(&found.node(c"clip").ok_or_else(missing)?)),
                (
                    c"vdata",
                    Value::Int(found.int(c"data").ok_or_else(missing)?),
                ),
            ],
        )?;
        let gpu = (sdata >> 48) & 0xFF;
        let options = CString::new(format!("{{pel:1,gpu:{gpu}}}")).unwrap_or_default();
        supers = call(
            c"Super",
            vec![(c"clip", clip(source)), (c"opt", Value::Data(options))],
        )?;
    }
    let mut input = Map::default();
    input.set(c"clip", clip(source));
    input.set(c"super", clip(&supers.node(c"clip").ok_or_else(missing)?));
    input.set(
        c"sdata",
        Value::Int(supers.int(c"data").ok_or_else(missing)?),
    );
    input.set(c"vectors", clip(&found.node(c"clip").ok_or_else(missing)?));
    input.set(
        c"vdata",
        Value::Int(found.int(c"data").ok_or_else(missing)?),
    );
    input.set(c"opt", Value::Data(unsafe { text(smooth_opt) }));
    Ok(input)
}

fn finish(host: &Arc<Host>, name: &CStr, input: Result<Map, CString>) -> *mut Clip {
    let output = input.and_then(|input| host.invoke(name, &input));
    match output.as_ref().map(|output| output.node(c"clip")) {
        Ok(Some(node)) => clip(node),
        Ok(None) => fail("the plugin returned no clip"),
        Err(error) => fail(error.as_bytes()),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn osvp_smooth_fps(
    context: *const Context,
    source: *const Clip,
    super_opt: *const c_char,
    analyse_opt: *const c_char,
    smooth_opt: *const c_char,
    half_analysis: i32,
) -> *mut Clip {
    let (host, source) = unsafe { (&(*context).0, &(*source).node) };
    let half = half_analysis != 0;
    let input = unsafe { vectors(host, source, super_opt, analyse_opt, smooth_opt, half) };
    finish(host, c"SmoothFps", input)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn osvp_smooth_fps_blend(
    context: *const Context,
    source: *const Clip,
    super_opt: *const c_char,
    analyse_opt: *const c_char,
    smooth_opt: *const c_char,
    half_analysis: i32,
    weights: *const f64,
    weight_count: i32,
    fps_num: i64,
    fps_den: i64,
) -> *mut Clip {
    let (host, source) = unsafe { (&(*context).0, &(*source).node) };
    let weights =
        unsafe { std::slice::from_raw_parts(weights, usize::try_from(weight_count).unwrap_or(0)) };
    let half = half_analysis != 0;
    let input = unsafe { vectors(host, source, super_opt, analyse_opt, smooth_opt, half) }.map(
        |mut input| {
            for &weight in weights {
                input.set(c"weights", Value::Float(weight));
            }
            input.set(c"fpsnum", Value::Int(fps_num));
            input.set(c"fpsden", Value::Int(fps_den));
            input
        },
    );
    finish(host, c"SmoothFpsBlend", input)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn osvp_clip_info(clip: *const Clip, info: *mut VideoInfo) {
    let source = unsafe { (*clip).node.info() };
    let id = unsafe { source.format.cast::<vs3::Format>().as_ref() }.map_or(0, |format| format.id);
    let format = FORMATS.iter().position(|&format| format == id);
    unsafe {
        *info = VideoInfo {
            format: format
                .and_then(|format| i32::try_from(format).ok())
                .unwrap_or(-1),
            width: source.width,
            height: source.height,
            fps_num: source.fps_num,
            fps_den: source.fps_den,
            num_frames: source.num_frames,
        };
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn osvp_clip_free(clip: *mut Clip) {
    if !clip.is_null() {
        drop(unsafe { Box::from_raw(clip) });
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn osvp_get_frame(clip: *const Clip, n: i32) -> *const Frame {
    let clip = unsafe { &*clip };
    let host = clip.node.host();
    let last = clip.node.info().num_frames - 1;
    {
        let mut ahead = clip.ahead.lock().unwrap_or_else(PoisonError::into_inner);
        let (next, pending) = &mut *ahead;
        let limit = n + 8 * host.threads();
        if !(n + 1..=limit + 1).contains(next) {
            *next = n + 1;
            pending.clear();
        }
        while pending.front().is_some_and(|(frame, _)| *frame < n) {
            pending.pop_front();
        }
        let end = limit.min(*next + 1).min(last);
        while *next <= end && !host.busy() {
            pending.push_back((*next, clip.node.prepare(*next)));
            *next += 1;
        }
    }
    match clip.node.get(n) {
        Ok(frame) => Arc::into_raw(frame),
        Err(error) => fail(error.to_bytes()),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn osvp_frame_data(frame: *const Frame, plane: i32) -> *const u8 {
    unsafe { &*frame }.read(plane)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn osvp_frame_stride(frame: *const Frame, plane: i32) -> isize {
    unsafe { &*frame }.stride(plane).cast_signed()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn osvp_frame_free(frame: *const Frame) {
    if !frame.is_null() {
        drop(unsafe { Arc::from_raw(frame) });
    }
}
