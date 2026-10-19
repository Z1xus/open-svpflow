use std::ffi::CStr;
use std::sync::{Arc, Mutex};

use svpflow_core::layer::{self, Fit, Stats};
use svpflow_core::smooth::{FieldShape, VectorField};
use svpflow_core::smooth_engine as engine;

use crate::core::{drop_frame, get_node, request_node};
use crate::{filter, frame, metadata, video_format, vs};

const STATS_CAP: usize = 4;
const FIT_CAP: usize = 6;

type Kept<T> = Vec<(i32, Arc<T>)>;
type Vectors<'a> = (&'a [u16], &'a [u16]);

fn find<T>(kept: &Kept<T>, key: i32) -> Option<Arc<T>> {
    let found = kept.iter().find(|(other, _)| *other == key)?;
    Some(Arc::clone(&found.1))
}

fn keep<T>(kept: &mut Kept<T>, key: i32, value: &Arc<T>, cap: usize) {
    kept.retain(|(other, _)| *other != key);
    if kept.len() >= cap {
        kept.remove(0);
    }
    kept.push((key, Arc::clone(value)));
}

pub(crate) unsafe fn planes(
    api: &frame::PlaneApi,
    frame: vs::ConstRaw,
    (height, chroma): (usize, usize),
) -> Option<[layer::Plane<'static>; 3]> {
    if frame.is_null() {
        return None;
    }
    let plane = |index: i32, rows: usize| {
        let (data, pitch, len) = unsafe { api.read_plane(frame, index, rows) }?;
        let data = unsafe { std::slice::from_raw_parts(data, len) };
        Some(layer::Plane { data, pitch })
    };
    Some([plane(0, height)?, plane(1, chroma)?, plane(2, chroma)?])
}

pub(crate) fn motion<'a>(
    forward: Vectors<'a>,
    backward: Vectors<'a>,
    data: &metadata::VectorData,
) -> layer::Motion<'a> {
    let shape = FieldShape::from_data(data);
    let block = data.effective_block();
    layer::Motion {
        forward,
        backward,
        grid: (
            shape.packed_width() as usize,
            shape.packed_height() as usize,
        ),
        block: (block.width, block.height),
        origin: (shape.overlap_x / 2, shape.overlap_y / 2),
        pel: shape.pel,
    }
}

pub(crate) fn frames(
    chunk: i32,
    last: i32,
) -> (std::ops::RangeInclusive<i32>, std::ops::RangeInclusive<i32>) {
    let pairs = layer::pairs(chunk, last);
    (*pairs.start()..=pairs.end() + 1, pairs)
}

pub(crate) struct Chunks<'a> {
    pub(crate) api: &'a frame::PlaneApi,
    pub(crate) get: &'a dyn Fn(vs::Raw, i32) -> vs::ConstRaw,
    pub(crate) source: vs::Raw,
    pub(crate) vectors: vs::Raw,
    pub(crate) data: &'a metadata::VectorData,
    pub(crate) size: (usize, usize),
    pub(crate) last: i32,
    pub(crate) ratio: usize,
}

impl Chunks<'_> {
    fn stats(&self, chunk: i32) -> Option<Stats> {
        let (sources, vectors) = frames(chunk, self.last);
        let sources: Vec<_> = sources.map(|k| (self.get)(self.source, k)).collect();
        let vectors: Vec<_> = vectors.map(|k| (self.get)(self.vectors, k)).collect();
        let made = (|| {
            let planes: Vec<_> = sources
                .iter()
                .map(|&frame| unsafe {
                    planes(self.api, frame, (self.size.1, self.size.1 / self.ratio))
                })
                .collect::<Option<_>>()?;
            let motions: Vec<_> = vectors
                .iter()
                .map(|&frame| {
                    let (data, stride, len) = (!frame.is_null())
                        .then(|| unsafe { self.api.read_plane(frame, 0, 1) })
                        .flatten()?;
                    let mut field = VectorField::new(FieldShape::from_data(self.data));
                    field.update(unsafe { std::slice::from_raw_parts(data, len.max(stride)) });
                    let none = |_| None::<Arc<VectorField>>;
                    let zero = (false, false);
                    Some(engine::motions(
                        &field, true, true, true, false, zero, none, 0,
                    ))
                })
                .collect::<Option<_>>()?;
            let pairs: Vec<_> = motions
                .iter()
                .zip(planes.windows(2))
                .map(|(fields, planes)| layer::Pair {
                    first: planes[0],
                    second: planes[1],
                    motion: motion(
                        (&fields[0], &fields[1]),
                        (&fields[2], &fields[3]),
                        self.data,
                    ),
                })
                .collect();
            Some(layer::stats(&pairs, self.size, self.ratio))
        })();
        for frame in sources.into_iter().chain(vectors) {
            unsafe { self.api.free(frame) };
        }
        made
    }
}

#[derive(Default)]
pub(crate) struct Layers {
    stats: Kept<Stats>,
    fits: Kept<Fit>,
}

#[derive(Default)]
pub(crate) struct Held {
    stats: Kept<Stats>,
    fits: Kept<Fit>,
}

fn pin<T>(kept: &mut Kept<T>, key: i32, value: Arc<T>) {
    if find(kept, key).is_none() {
        kept.push((key, value));
    }
}

impl Layers {
    pub(crate) fn ready(&self, chunk: i32) -> Option<Arc<Fit>> {
        find(&self.fits, chunk)
    }

    pub(crate) fn hold(&self, chunk: i32, last: i32, held: &mut Held) -> Vec<i32> {
        if let Some(fit) = find(&self.fits, chunk) {
            pin(&mut held.fits, chunk, fit);
            return Vec::new();
        }
        let near = (chunk - 1).max(0)..=(chunk + 1).min(last);
        near.filter(|&key| match find(&self.stats, key) {
            Some(stats) => {
                pin(&mut held.stats, key, stats);
                false
            }
            None => true,
        })
        .collect()
    }

    pub(crate) fn fit(
        &mut self,
        chunks: &Chunks,
        chunk: i32,
        limit: f32,
        held: &Held,
    ) -> Option<Arc<Fit>> {
        if let Some(fit) = find(&held.fits, chunk).or_else(|| find(&self.fits, chunk)) {
            return Some(fit);
        }
        let parts = [chunk - 1, chunk, chunk + 1].map(|key| {
            if key < 0 || key > chunks.last / layer::CHUNK {
                return None;
            }
            let stats = find(&held.stats, key)
                .or_else(|| find(&self.stats, key))
                .or_else(|| chunks.stats(key).map(Arc::new))?;
            keep(&mut self.stats, key, &stats, STATS_CAP);
            Some(stats)
        });
        let parts = parts.each_ref().map(Option::as_deref);
        let fit = Arc::new(layer::fit(parts, chunks.size, limit, chunks.ratio)?);
        keep(&mut self.fits, chunk, &fit, FIT_CAP);
        Some(fit)
    }
}

enum Kind {
    Layer {
        data: metadata::VectorData,
        limit: f32,
        last: i32,
        layers: Mutex<Layers>,
    },
    Peel,
    Unpeel {
        step: (i64, i64),
        last: i32,
    },
}

struct State {
    kind: Kind,
    clips: [vs::Raw; 3],
    info: vs::VideoInfo,
    size: (usize, usize),
    ratio: usize,
}

fn ratio(info: &vs::VideoInfo) -> usize {
    video_format::chroma_divisors(info).0 as usize
}

unsafe fn clips<const N: usize>(
    input: vs::ConstRaw,
    vsapi: vs::ConstRaw,
    keys: [&CStr; N],
) -> Option<([vs::Raw; N], [vs::VideoInfo; N])> {
    let get_node = unsafe { vs::table_fn::<vs::PropGetNode>(vsapi, vs::PROP_GET_NODE) }?;
    let get_info = unsafe { vs::table_fn::<vs::GetVideoInfo>(vsapi, vs::GET_VIDEO_INFO) }?;
    let mut error = 0;
    let nodes = keys.map(|key| unsafe { get_node(input, key.as_ptr(), 0, &raw mut error) });
    if nodes.iter().any(|node| node.is_null()) {
        unsafe { vs::free_nodes(nodes, vsapi) };
        return None;
    }
    Some((nodes, nodes.map(|node| unsafe { *get_info(node) })))
}

unsafe fn create(
    state: Result<State, &CStr>,
    name: &CStr,
    input: vs::ConstRaw,
    output: vs::Raw,
    core: vs::Raw,
    vsapi: vs::ConstRaw,
) {
    let create = unsafe { vs::table_fn::<vs::CreateFilter>(vsapi, vs::CREATE_FILTER) };
    match (state, create) {
        (Ok(state), Some(create)) => unsafe {
            create(
                input,
                output,
                name.as_ptr(),
                init_filter,
                get_frame,
                free_filter,
                vs::FM_PARALLEL,
                0,
                Box::into_raw(Box::new(state)).cast(),
                core,
            );
        },
        (Err(message), _) => unsafe { filter::set_error(output, vsapi, message.as_ptr()) },
        _ => {}
    }
}

fn size(info: &vs::VideoInfo) -> (usize, usize) {
    (info.width as usize, info.height as usize)
}

pub(crate) unsafe extern "system" fn create_layer(
    input: vs::ConstRaw,
    output: vs::Raw,
    _: vs::Raw,
    core: vs::Raw,
    vsapi: vs::ConstRaw,
) -> isize {
    let Some((nodes, [info, _])) = (unsafe { clips(input, vsapi, [c"clip", c"vectors"]) }) else {
        return 0;
    };
    let state = (|| {
        let get_int = unsafe { vs::table_fn::<vs::PropGetInt>(vsapi, vs::PROP_GET_INT) }?;
        let get_float = unsafe { vs::table_fn::<vs::PropGetFloat>(vsapi, vs::PROP_GET_FLOAT) }?;
        let preset = unsafe { vs::table_fn::<vs::GetFormatPreset>(vsapi, vs::GET_FORMAT_PRESET) }?;
        let mut error = 0;
        let vdata = unsafe { get_int(input, c"vdata".as_ptr(), 0, &raw mut error) };
        let metadata::VectorRecord::Ready(data) = (unsafe { metadata::vector_data(vdata) }) else {
            return None;
        };
        let limit = unsafe { get_float(input, c"limit".as_ptr(), 0, &raw mut error) };
        let id = if ratio(&info) == 1 {
            vs::PF_YUV444P16
        } else {
            vs::PF_YUV420P16
        };
        let format = unsafe { preset(id, core) };
        let even = info.width % 2 == 0 && info.height % 2 == 0;
        let fits = video_format::is_cpu_source(&info) && even && info.num_frames > 1;
        (fits && !format.is_null()).then(|| State {
            kind: Kind::Layer {
                data,
                limit: layer::limit((error == 0).then_some(limit)),
                last: info.num_frames - 2,
                layers: Mutex::default(),
            },
            clips: [nodes[0], nodes[1], std::ptr::null_mut()],
            info: vs::VideoInfo {
                format: format.cast(),
                height: info.height * 2,
                ..info
            },
            size: size(&info),
            ratio: ratio(&info),
        })
    })();
    if state.is_none() {
        unsafe { vs::free_nodes(nodes, vsapi) };
    }
    let state = state
        .ok_or(c"SVLayer: the clip must be YUV420P8 or YUV444P8 with its vectors from SVAnalyse");
    unsafe { create(state, c"SVLayer", input, output, core, vsapi) };
    0
}

pub(crate) unsafe extern "system" fn create_peel(
    input: vs::ConstRaw,
    output: vs::Raw,
    _: vs::Raw,
    core: vs::Raw,
    vsapi: vs::ConstRaw,
) -> isize {
    let Some((nodes, [info, fit])) = (unsafe { clips(input, vsapi, [c"clip", c"layer"]) }) else {
        return 0;
    };
    let fits = video_format::is_cpu_source(&info)
        && (fit.width, fit.height) == (info.width, info.height * 2);
    if !fits {
        unsafe { vs::free_nodes(nodes, vsapi) };
    }
    let state = fits.then(|| State {
        kind: Kind::Peel,
        clips: [nodes[0], nodes[1], std::ptr::null_mut()],
        info,
        size: size(&info),
        ratio: ratio(&info),
    });
    let state =
        state.ok_or(c"SVPeel: the clip must be YUV420P8 or YUV444P8 with its layer from SVLayer");
    unsafe { create(state, c"SVPeel", input, output, core, vsapi) };
    0
}

pub(crate) unsafe extern "system" fn create_unpeel(
    input: vs::ConstRaw,
    output: vs::Raw,
    _: vs::Raw,
    core: vs::Raw,
    vsapi: vs::ConstRaw,
) -> isize {
    let keys = [c"clip", c"source", c"layer"];
    let Some((nodes, [info, source, fit])) = (unsafe { clips(input, vsapi, keys) }) else {
        return 0;
    };
    let step = (source.fps_num * info.fps_den, source.fps_den * info.fps_num);
    let fits = video_format::is_cpu_source(&info)
        && info.format == source.format
        && (info.width, info.height) == (source.width, source.height)
        && (fit.width, fit.height) == (info.width, info.height * 2)
        && step.0 > 0
        && step.1 > 0;
    if !fits {
        unsafe { vs::free_nodes(nodes, vsapi) };
    }
    let state = fits.then(|| State {
        kind: Kind::Unpeel {
            step,
            last: source.num_frames - 1,
        },
        clips: nodes,
        info,
        size: size(&info),
        ratio: ratio(&info),
    });
    let state = state.ok_or(
        c"SVUnpeel: both clips must be YUV420P8 or YUV444P8 of one size with frame rates and the layer of the source",
    );
    unsafe { create(state, c"SVUnpeel", input, output, core, vsapi) };
    0
}

#[allow(clippy::cast_ptr_alignment)]
unsafe fn fit<'a>(
    api: &frame::PlaneApi,
    frame: vs::ConstRaw,
    size: (usize, usize),
    ratio: usize,
) -> Option<layer::Layer<'a>> {
    if frame.is_null() {
        return None;
    }
    let plane = |index: i32, rows: usize| {
        let (data, pitch, len) = unsafe { api.read_plane(frame, index, rows) }?;
        let data = data.cast::<i16>();
        data.is_aligned().then(|| {
            (
                unsafe { std::slice::from_raw_parts(data, len / 2) },
                pitch / 2,
            )
        })
    };
    let (luma, pitch) = plane(0, size.1 * 2)?;
    let (first, chroma) = plane(1, size.1 * 2 / ratio)?;
    let (second, _) = plane(2, size.1 * 2 / ratio)?;
    Some(layer::Layer {
        luma,
        chroma: [first, second],
        pitch: (pitch, chroma),
        size,
        ratio,
    })
}

impl State {
    unsafe fn layer(
        &self,
        api: &frame::PlaneApi,
        fit: &Fit,
        core: vs::Raw,
    ) -> Option<vs::ConstRaw> {
        let vs::VideoInfo {
            format,
            width,
            height,
            ..
        } = self.info;
        let output = unsafe { api.new_frame(format, width, height, std::ptr::null(), core) }?;
        let half = self.size.0 / self.ratio;
        let planes = [
            (&fit.luma, self.size.0),
            (&fit.chroma[0], half),
            (&fit.chroma[1], half),
        ];
        for (index, (values, width)) in (0..).zip(planes) {
            let rows = values.len() / width;
            let Some((data, pitch, len)) = (unsafe { api.write_plane(output, index, rows) }) else {
                unsafe { api.free(output) };
                return None;
            };
            let target = unsafe { std::slice::from_raw_parts_mut(data, len) };
            for (row, values) in values.chunks_exact(width).enumerate() {
                let bytes = values.iter().flat_map(|value| value.to_ne_bytes());
                for (target, byte) in target[row * pitch..].iter_mut().zip(bytes) {
                    *target = byte;
                }
            }
        }
        Some(output.cast_const())
    }

    unsafe fn changed(
        &self,
        api: &frame::PlaneApi,
        frame: vs::ConstRaw,
        core: vs::Raw,
        mut change: impl FnMut(usize, &mut [u8], usize),
    ) -> Option<vs::ConstRaw> {
        let output = unsafe { api.copy(frame, core) }?;
        for plane in 0..3 {
            let rows = if plane == 0 {
                self.size.1
            } else {
                self.size.1 / self.ratio
            };
            let Some((data, pitch, len)) = (unsafe { api.write_plane(output, plane, rows) }) else {
                unsafe { api.free(output) };
                return None;
            };
            let target = unsafe { std::slice::from_raw_parts_mut(data, len) };
            change(plane as usize, target, pitch);
        }
        Some(output.cast_const())
    }

    #[allow(clippy::cast_precision_loss)]
    fn position(step: (i64, i64), n: i32) -> (i32, f32) {
        let at = i64::from(n) * step.0;
        ((at / step.1) as i32, (at % step.1) as f32 / step.1 as f32)
    }
}

unsafe extern "system" fn init_filter(
    _: vs::ConstRaw,
    _: vs::Raw,
    instance_data: *mut vs::Raw,
    node: vs::Raw,
    _: vs::Raw,
    vsapi: vs::ConstRaw,
) {
    let state = unsafe { &*(*instance_data).cast::<State>() };
    if let Some(set_video_info) =
        unsafe { vs::table_fn::<vs::SetVideoInfo>(vsapi, vs::SET_VIDEO_INFO) }
    {
        unsafe { set_video_info(&raw const state.info, 1, node) };
    }
}

unsafe extern "system" fn free_filter(instance_data: vs::Raw, _: vs::Raw, vsapi: vs::ConstRaw) {
    let state = unsafe { Box::from_raw(instance_data.cast::<State>()) };
    unsafe { vs::free_nodes(state.clips, vsapi) };
}

unsafe extern "system" fn get_frame(
    n: i32,
    activation_reason: i32,
    instance_data: *mut vs::Raw,
    frame_data: *mut vs::Raw,
    frame_ctx: vs::Raw,
    core: vs::Raw,
    vsapi: vs::ConstRaw,
) -> vs::ConstRaw {
    let state = unsafe { &*(*instance_data).cast::<State>() };
    let held = unsafe { std::mem::replace(&mut *frame_data, std::ptr::null_mut()) };
    let held = (!held.is_null()).then(|| unsafe { Box::from_raw(held.cast::<Held>()) });
    let ready = match activation_reason {
        vs::AR_INITIAL => false,
        vs::AR_ALL_FRAMES_READY => true,
        _ => return std::ptr::null(),
    };
    let made = (|| {
        let api = unsafe { frame::PlaneApi::load(vsapi) }?;
        let request =
            unsafe { vs::table_fn::<vs::RequestFrameFilter>(vsapi, vs::REQUEST_FRAME_FILTER) }?;
        let get = unsafe { vs::table_fn::<vs::GetFrameFilter>(vsapi, vs::GET_FRAME_FILTER) }?;
        let free = unsafe { vs::table_fn::<vs::FreeFrame>(vsapi, vs::FREE_FRAME) };
        let [first, second, third] = state.clips;
        match &state.kind {
            Kind::Layer {
                data,
                limit,
                last,
                layers,
            } => {
                let chunk = n.min(*last) / layer::CHUNK;
                let mut layers = layers.lock().ok()?;
                if !ready {
                    if let Some(fit) = layers.ready(chunk) {
                        return unsafe { state.layer(&api, &fit, core) };
                    }
                    let mut held = Box::<Held>::default();
                    for chunk in layers.hold(chunk, last / layer::CHUNK, &mut held) {
                        let (sources, vectors) = frames(chunk, *last);
                        sources.for_each(|k| request_node(request, first, k, frame_ctx));
                        vectors.for_each(|k| request_node(request, second, k, frame_ctx));
                    }
                    unsafe { *frame_data = Box::into_raw(held).cast() };
                    return None;
                }
                let chunks = Chunks {
                    api: &api,
                    get: &|node, k| get_node(get, node, k, frame_ctx),
                    source: first,
                    vectors: second,
                    data,
                    size: state.size,
                    last: *last,
                    ratio: state.ratio,
                };
                let fit = layers.fit(&chunks, chunk, *limit, held.as_deref()?)?;
                drop(layers);
                unsafe { state.layer(&api, &fit, core) }
            }
            Kind::Peel => {
                if !ready {
                    request_node(request, first, n, frame_ctx);
                    request_node(request, second, n, frame_ctx);
                    return None;
                }
                let frames = [first, second].map(|node| get_node(get, node, n, frame_ctx));
                let made = (|| {
                    let raw = unsafe {
                        planes(&api, frames[0], (state.size.1, state.size.1 / state.ratio))
                    }?;
                    let fit = unsafe { fit(&api, frames[1], state.size, state.ratio) }?;
                    unsafe {
                        state.changed(&api, frames[0], core, |plane, target, pitch| {
                            fit.peel(plane, target, pitch, raw[plane]);
                        })
                    }
                })();
                for frame in frames {
                    drop_frame(frame, free);
                }
                made
            }
            Kind::Unpeel { step, last } => {
                let (index, time) = State::position(*step, n);
                let (index, next) = (index.min(*last), index.saturating_add(1).min(*last));
                let wanted = [(first, n), (second, index), (second, next), (third, index)];
                if !ready {
                    for (node, k) in wanted {
                        request_node(request, node, k, frame_ctx);
                    }
                    return None;
                }
                let frames = wanted.map(|(node, k)| get_node(get, node, k, frame_ctx));
                let made = (|| {
                    let sources = layer::Frames {
                        first: unsafe {
                            planes(&api, frames[1], (state.size.1, state.size.1 / state.ratio))
                        }?,
                        second: unsafe {
                            planes(&api, frames[2], (state.size.1, state.size.1 / state.ratio))
                        }?,
                        time,
                    };
                    let fit = unsafe { fit(&api, frames[3], state.size, state.ratio) }?;
                    if frames[0].is_null() {
                        return None;
                    }
                    unsafe {
                        state.changed(&api, frames[0], core, |plane, target, pitch| {
                            fit.unpeel(plane, target, pitch, &sources);
                        })
                    }
                })();
                for frame in frames {
                    drop_frame(frame, free);
                }
                made
            }
        }
    })();
    made.unwrap_or(std::ptr::null())
}
