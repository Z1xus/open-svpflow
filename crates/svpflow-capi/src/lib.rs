#![allow(clippy::missing_safety_doc)]

mod host;

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char, c_void};
use std::sync::{Arc, Mutex, PoisonError};

use host::{Frame, Host, Map, Node, Read, Value};
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
    scheduled: Mutex<i32>,
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
        scheduled: Mutex::new(0),
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
) -> Result<Map, CString> {
    let mut input = Map::default();
    input.set(c"clip", Value::Node(Arc::clone(source)));
    input.set(c"opt", Value::Data(unsafe { text(super_opt) }));
    let output = host.invoke(c"Super", &input)?;
    let missing = || c"the plugin returned no clip".to_owned();
    let (super_clip, sdata) = (
        output.node(c"clip").ok_or_else(missing)?,
        output.int(c"data").ok_or_else(missing)?,
    );
    let mut input = Map::default();
    input.set(c"clip", Value::Node(Arc::clone(&super_clip)));
    input.set(c"sdata", Value::Int(sdata));
    input.set(c"src", Value::Node(Arc::clone(source)));
    input.set(c"opt", Value::Data(unsafe { text(analyse_opt) }));
    let output = host.invoke(c"Analyse", &input)?;
    let mut input = Map::default();
    input.set(c"clip", Value::Node(Arc::clone(source)));
    input.set(c"super", Value::Node(super_clip));
    input.set(c"sdata", Value::Int(sdata));
    input.set(
        c"vectors",
        Value::Node(output.node(c"clip").ok_or_else(missing)?),
    );
    input.set(
        c"vdata",
        Value::Int(output.int(c"data").ok_or_else(missing)?),
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
) -> *mut Clip {
    let (host, source) = unsafe { (&(*context).0, &(*source).node) };
    let input = unsafe { vectors(host, source, super_opt, analyse_opt, smooth_opt) };
    finish(host, c"SmoothFps", input)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn osvp_smooth_fps_blend(
    context: *const Context,
    source: *const Clip,
    super_opt: *const c_char,
    analyse_opt: *const c_char,
    smooth_opt: *const c_char,
    weights: *const f64,
    weight_count: i32,
    fps_num: i64,
    fps_den: i64,
) -> *mut Clip {
    let (host, source) = unsafe { (&(*context).0, &(*source).node) };
    let weights =
        unsafe { std::slice::from_raw_parts(weights, usize::try_from(weight_count).unwrap_or(0)) };
    let input =
        unsafe { vectors(host, source, super_opt, analyse_opt, smooth_opt) }.map(|mut input| {
            for &weight in weights {
                input.set(c"weights", Value::Float(weight));
            }
            input.set(c"fpsnum", Value::Int(fps_num));
            input.set(c"fpsden", Value::Int(fps_den));
            input
        });
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
    let last = clip.node.info().num_frames - 1;
    let lookahead = clip.node.host().lookahead();
    {
        let mut scheduled = clip
            .scheduled
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if !(n + 1..=n + 1 + lookahead).contains(&*scheduled) {
            *scheduled = n + 1;
        }
        let ahead = (n + lookahead).min(*scheduled + 1).min(last);
        while *scheduled <= ahead {
            clip.node.prepare(*scheduled);
            *scheduled += 1;
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
