#![allow(clippy::cast_precision_loss, clippy::many_single_char_names)]

use std::slice;

use svpflow_core::layer;
use svpflow_core::smooth::{
    BlockVector, FieldShape, Frame, FrameMut, PackedVectors, Plane, PlaneMut, Renderer,
    SceneLimits, VectorField,
};
use svpflow_core::smooth_engine::{self as engine, Rates};

use crate::core::{FilterState, Mode, drop_frame, get_node, request_node, super_frame_planes};
use crate::vs::Source as _;
use crate::{frame, metadata, options::ReferenceParams, vs};

const WORKERS: usize = 6;

fn cache_cap(bytes: usize) -> usize {
    ((96 << 20) / bytes.max(1)).clamp(4, 48)
}

// longer runs are still pictures
const RUN: i32 = 3;

// the layers that a frame uses, from its request until it is made
#[derive(Default)]
pub(crate) struct Held {
    cpu: crate::peel::Held,
    gpu: crate::gpu::LayerHeld,
    // the dedup vectors are requested
    pub(crate) more: bool,
}

struct Fetch {
    get_frame: vs::GetFrameFilter,
    free_frame: Option<vs::FreeFrame>,
    frame_ctx: vs::Raw,
}

impl Fetch {
    fn get(&self, node: vs::Raw, n: i32) -> vs::ConstRaw {
        get_node(self.get_frame, node, n, self.frame_ctx)
    }

    fn drop(&self, frame: vs::ConstRaw) {
        drop_frame(frame, self.free_frame);
    }
}

struct Merged {
    _keep: Option<std::sync::Arc<crate::core::SuperExpand>>,
    y: &'static [u8],
    u: &'static [u8],
    v: &'static [u8],
    y_pitch: usize,
    uv_pitch: usize,
    slack: usize,
}

impl Merged {
    fn frame(&self) -> Frame<'_> {
        Frame {
            y: Plane {
                data: self.y,
                pitch: self.y_pitch,
                slack: self.slack,
            },
            u: Plane {
                data: self.u,
                pitch: self.uv_pitch,
                slack: self.slack,
            },
            v: Plane {
                data: self.v,
                pitch: self.uv_pitch,
                slack: self.slack,
            },
        }
    }
}

impl FilterState {
    fn chroma(&self) -> (i32, i32) {
        crate::video_format::chroma_divisors(&self.video_info)
    }

    pub(crate) fn reference_enabled(&self) -> bool {
        (matches!(self.mode, Mode::SmoothFps) || self.nvof.is_some())
            && (self.render_mode == 1 || (self.render_mode == 2 && self.gpu.is_some()))
            && (!self.source_8bit_mode || self.nvof.is_some())
            && !(self.render_mode == 2 && self.options.block_enabled())
            && self.options.reference_supported()
            && matches!(self.vector_data(), metadata::VectorRecord::Ready(_))
    }

    fn cached_field(
        &self,
        fetch: &Fetch,
        api: &frame::PlaneApi,
        (k, n): (i32, i32),
        shape: FieldShape,
    ) -> Option<std::sync::Arc<VectorField>> {
        let decode = || {
            let mut field = VectorField::new(shape);
            self.with_vectors(fetch, api, (k, n), |payload| field.update(payload))
                .map(|_| std::sync::Arc::new(field))
        };
        if self.nvof.is_some() {
            return decode();
        }
        let cap = self.field_cap(shape);
        crate::core::cache_cell(&self.field_cache, i64::from(k), cap)?
            .get_or_init(decode)
            .clone()
    }

    fn uploads(
        &self,
        api: &frame::PlaneApi,
        raw: vs::ConstRaw,
    ) -> Option<[crate::gpu::UploadPlane<'static>; 3]> {
        let w = usize::try_from(self.video_info.width).ok()?;
        let h = usize::try_from(self.video_info.height).ok()?;
        let (cx, cy) = (self.chroma().0 as usize, self.chroma().1 as usize);
        let plane = |p: i32, pw: usize, ph: usize| -> Option<crate::gpu::UploadPlane<'static>> {
            let (ptr, stride, len) = unsafe { api.read_plane(raw, p, ph) }?;
            Some(crate::gpu::UploadPlane {
                data: unsafe { slice::from_raw_parts(ptr, len) },
                stride,
                width: pw,
                height: ph,
            })
        };
        Some([
            plane(0, w, h)?,
            plane(1, w / cx, h / cy)?,
            plane(2, w / cx, h / cy)?,
        ])
    }

    fn field_cap(&self, shape: FieldShape) -> usize {
        let cap = cache_cap(shape.blocks() * std::mem::size_of::<BlockVector>() * 2);
        if self.options.mask_peel().is_some() {
            cap.max(3 * layer::CHUNK as usize + 16)
        } else {
            cap
        }
    }

    fn layer_plan(&self) -> Option<(i32, f32)> {
        let limit = self.options.mask_peel()?;
        let last = self.video_info.num_frames - 2;
        let chroma = self.chroma();
        let even = chroma == (2, 2) || chroma == (1, 1);
        (self.nvof.is_none() && last >= 0 && even).then_some((last, limit))
    }

    fn layer_wanted(&self, wanted: &mut Vec<(vs::Raw, i32)>, n: i32, held: &mut Held) {
        let Some((last, _)) = self.layer_plan() else {
            return;
        };
        let (chunk, chunks) = (n / layer::CHUNK, last / layer::CHUNK);
        let missing = if let Some(gpu) = self.gpu.as_ref() {
            gpu.layer_hold(chunk, chunks, &mut held.gpu)
        } else {
            let Ok(layers) = self.layers.lock() else {
                return;
            };
            layers.hold(chunk, chunks, &mut held.cpu)
        };
        for c in missing {
            let (sources, vectors) = crate::peel::frames(c, last);
            wanted.extend(sources.map(|k| (self.clips.source, k)));
            wanted.extend(vectors.map(|k| (self.clips.vectors, k)));
        }
    }

    fn layer_cpu(
        &self,
        api: &frame::PlaneApi,
        fetch: &Fetch,
        n: i32,
        data: &metadata::VectorData,
        held: &Held,
    ) -> Option<std::sync::Arc<layer::Fit>> {
        let (last, limit) = self.layer_plan()?;
        if !crate::video_format::is_cpu_source(&self.video_info) {
            return None;
        }
        let chunks = crate::peel::Chunks {
            api,
            get: &|node, k| fetch.get(node, k),
            source: self.clips.source,
            vectors: self.clips.vectors,
            data,
            size: (
                self.video_info.width as usize,
                self.video_info.height as usize,
            ),
            last,
            ratio: self.chroma().0 as usize,
        };
        let mut layers = self.layers.lock().ok()?;
        layers.fit(&chunks, n / layer::CHUNK, limit, &held.cpu)
    }

    fn repeat(&self, fetch: &Fetch, api: &frame::PlaneApi, k: i32, threshold: f64) -> bool {
        if k < 1 || k >= self.video_info.num_frames {
            return false;
        }
        let cached =
            |cache: &Vec<(i32, bool)>| cache.iter().find(|(key, _)| *key == k).map(|v| v.1);
        if let Some(known) = self
            .repeat_cache
            .lock()
            .ok()
            .and_then(|cache| cached(&cache))
        {
            return known;
        }
        let frames = [k - 1, k].map(|k| fetch.get(self.clips.source, k));
        let same = (|| {
            let height = usize::try_from(self.video_info.height).ok()?;
            let width = usize::try_from(self.video_info.width).ok()?;
            let plane = |frame: vs::ConstRaw| {
                let (ptr, stride, len) = (!frame.is_null())
                    .then(|| unsafe { api.read_plane(frame, 0, height) })
                    .flatten()?;
                Some((unsafe { slice::from_raw_parts(ptr, len) }, stride))
            };
            let ((first, stride), (second, _)) = (plane(frames[0])?, plane(frames[1])?);
            let mut total = 0u64;
            let mut count = 0u64;
            for y in (0..height).step_by(4) {
                let row = y * stride..y * stride + width;
                let pairs = first.get(row.clone())?.iter().zip(second.get(row)?);
                for (a, b) in pairs.step_by(4) {
                    total += u64::from(a.abs_diff(*b));
                    count += 1;
                }
            }
            Some((total as f64) < threshold * count as f64)
        })();
        for frame in frames {
            fetch.drop(frame);
        }
        let Some(same) = same else {
            return false;
        };
        if let Ok(mut cache) = self.repeat_cache.lock() {
            if cache.len() >= 512 {
                cache.remove(0);
            }
            cache.push((k, same));
        }
        same
    }

    fn dedup(&self, fetch: &Fetch, api: &frame::PlaneApi, n: i32, raw: f64) -> (i32, f64) {
        let Some(threshold) = self.dedup_limit() else {
            return (n, raw);
        };
        let repeat = |k: i32| self.repeat(fetch, api, k, threshold);
        let start = (n - RUN..=n).rev().find(|&k| !repeat(k));
        let end = (n + 1..=n + 1 + RUN).find(|&k| !repeat(k));
        let end = end.filter(|&end| end < self.video_info.num_frames);
        match (start, end) {
            (Some(start), Some(end)) if end - start > 1 => {
                let position = f64::from(n - start) * 256.0 + raw;
                (end - 1, position / f64::from(end - start))
            }
            _ => (n, raw),
        }
    }

    fn dedup_limit(&self) -> Option<f64> {
        let plain = self.nvof.is_none() && crate::video_format::source_depth(&self.video_info) <= 8;
        self.options.dedup().filter(|_| plain)
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn layer(
        &self,
        api: &frame::PlaneApi,
        fetch: &Fetch,
        gpu: &crate::gpu::GpuContext,
        n: i32,
        params: &ReferenceParams,
        vector_data: &metadata::VectorData,
        field: &VectorField,
        held: &Held,
    ) -> Option<std::sync::Arc<crate::gpu::Layer>> {
        let (last, limit) = self.layer_plan()?;
        let shape = *field.shape();
        let (width, height) = (self.video_info.width, self.video_info.height);
        let cells = (
            usize::try_from(shape.packed_width()).ok()?,
            usize::try_from(shape.packed_height()).ok()?,
        );
        let motion = |k: i32, field: &VectorField| {
            let none = |_| None::<std::sync::Arc<VectorField>>;
            engine::motions(field, true, true, true, false, (false, false), none, k)
        };
        let kernel = engine::gpu_params(
            params,
            vector_data,
            (width, height),
            self.chroma(),
            13,
            256,
            false,
            false,
        );
        let cap = self.field_cap(shape);
        let pairs = |chunk: i64| {
            let pairs: Vec<i32> = layer::pairs(i32::try_from(chunk).ok()?, last).collect();
            let needed: Vec<i32> = (pairs[0]..=pairs[pairs.len() - 1] + 1).collect();
            let frames: Vec<_> = needed
                .iter()
                .map(|&k| fetch.get(self.clips.source, k))
                .collect();
            let vectors: Vec<_> = pairs
                .iter()
                .map(|&k| fetch.get(self.clips.vectors, k))
                .collect();
            let planes: Option<Vec<_>> = frames
                .iter()
                .map(|&frame| {
                    (!frame.is_null())
                        .then(|| self.uploads(api, frame))
                        .flatten()
                })
                .collect();
            let payloads: Option<Vec<_>> = vectors
                .iter()
                .map(|&frame| {
                    let (ptr, stride, len) =
                        (!frame.is_null()).then(|| unsafe { api.read_plane(frame, 0, 1) })??;
                    Some(unsafe { slice::from_raw_parts(ptr, len.max(stride)) })
                })
                .collect();
            let made = planes.zip(payloads).map(|(planes, payloads)| {
                let handles: Vec<_> = planes.iter().map(|_| std::sync::Mutex::new(None)).collect();
                let moved: Vec<_> = payloads
                    .iter()
                    .map(|_| std::sync::Mutex::new(None))
                    .collect();
                let fields = &self.field_cache;
                std::thread::scope(|scope| {
                    for worker in 0..WORKERS {
                        let (planes, payloads, handles, moved) =
                            (&planes, &payloads, &handles, &moved);
                        let (needed, pairs, motion) = (&needed, &pairs, &motion);
                        scope.spawn(move || {
                            for index in (worker..planes.len()).step_by(WORKERS) {
                                let handle =
                                    gpu.cache_frame(i64::from(needed[index]), planes[index]);
                                if let Ok(mut slot) = handles[index].lock() {
                                    *slot = handle;
                                }
                            }
                            for index in (worker..payloads.len()).step_by(WORKERS) {
                                let k = pairs[index];
                                let field = crate::core::cache_cell(fields, i64::from(k), cap)
                                    .and_then(|cell| {
                                        cell.get_or_init(|| {
                                            let mut field = VectorField::new(shape);
                                            field.update(payloads[index]);
                                            Some(std::sync::Arc::new(field))
                                        })
                                        .clone()
                                    });
                                if let (Some(field), Ok(mut slot)) = (field, moved[index].lock()) {
                                    *slot = Some(motion(k, &field));
                                }
                            }
                        });
                    }
                });
                (handles, moved)
            });
            for frame in frames.into_iter().chain(vectors) {
                fetch.drop(frame);
            }
            let (handles, moved) = made?;
            let handles: Vec<crate::gpu::SuperHandle> = handles
                .into_iter()
                .map(|slot| slot.into_inner().ok().flatten())
                .collect::<Option<_>>()?;
            let frame = |k: i32| Some(handles[needed.binary_search(&k).ok()?].share());
            pairs
                .iter()
                .zip(moved)
                .map(|(&k, slot)| {
                    let motions = slot.into_inner().ok().flatten()?;
                    Some(crate::gpu::LayerPair {
                        first: frame(k)?,
                        second: frame(k + 1)?,
                        motions,
                    })
                })
                .collect()
        };
        gpu.layer_fit(
            i64::from(n / layer::CHUNK),
            i64::from(last / layer::CHUNK),
            &kernel,
            cells,
            limit,
            &held.gpu,
            pairs,
        )
    }

    fn cached_quality(&self, frame: i32, luma: &mut [u8]) -> Option<i32> {
        let cache = self.quality_cache.lock().ok()?;
        let (_, class, cached) = cache.iter().find(|(k, _, _)| *k == frame)?;
        luma.copy_from_slice(cached);
        Some(*class)
    }

    fn store_quality(&self, frame: i32, class: i32, luma: &[u8]) {
        if let Ok(mut cache) = self.quality_cache.lock() {
            if cache.iter().any(|(k, _, _)| *k == frame) {
                return;
            }
            if cache.len() >= 64 {
                cache.remove(0);
            }
            cache.push((frame, class, luma.to_vec()));
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn reference_gpu(
        &self,
        api: &frame::PlaneApi,
        gpu: &crate::gpu::GpuContext,
        source: vs::ConstRaw,
        next_source: vs::ConstRaw,
        n: i32,
        frame: i32,
        params: [crate::gpu::KernelParams; 3],
        linear: bool,
        width: i32,
        height: i32,
        motions: [(&[u16], &[u16]); 4],
        coverage: (&[u8], &[u8]),
        area: Option<(&[u8], &[u8])>,
        motion_key: i64,
        blend: Option<&crate::gpu::BlendJob>,
        layer: Option<&std::sync::Arc<crate::gpu::Layer>>,
        core: vs::Raw,
    ) -> vs::ConstRaw {
        let w = usize::try_from(self.video_info.width).unwrap_or(0);
        let h = usize::try_from(self.video_info.height).unwrap_or(0);
        let (Some(current), Some(next)) =
            (self.uploads(api, source), self.uploads(api, next_source))
        else {
            return std::ptr::null();
        };
        let (Some(current), Some(next)) = (
            gpu.cache_frame(i64::from(n), current),
            gpu.cache_frame(i64::from(n) + 1, next),
        ) else {
            return std::ptr::null();
        };
        let still = self
            .options
            .mask_still()
            .and_then(|limits| gpu.still_mask(&current, &next, limits));
        if let Some(job) = blend {
            gpu.blend_render(
                job,
                &current,
                &next,
                linear,
                params,
                motion_key,
                usize::try_from(width).unwrap_or(0),
                usize::try_from(height).unwrap_or(0),
                motions,
                coverage,
                area,
                still,
                layer,
            );
            return std::ptr::null();
        }
        let output_info = self.output_info();
        let Some(output) = (unsafe {
            api.new_frame(
                self.video_info.format,
                output_info.width,
                output_info.height,
                source,
                core,
            )
        }) else {
            return std::ptr::null();
        };
        let (pad_x, pad_y) = self.options.padding(&self.video_info.source());
        let bytes = if crate::video_format::source_depth(&self.video_info) > 8 {
            2
        } else {
            1
        };
        let (cx, cy) = (self.chroma().0 as usize, self.chroma().1 as usize);
        let plane = |p: i32, rows: usize| -> Option<(&'static mut [u8], i32)> {
            let (dx, dy) = if p == 0 { (1, 1) } else { (cx, cy) };
            let (px, py) = (
                usize::try_from(pad_x).ok()? / dx,
                usize::try_from(pad_y).ok()? / dy,
            );
            let (ptr, pitch, len) = unsafe { api.write_plane(output, p, rows + 2 * py) }?;
            let start = py * pitch + px * bytes;
            let span = (rows.checked_sub(1)?) * pitch + (w / dx) * bytes;
            if start + span > len {
                return None;
            }
            Some((
                unsafe { slice::from_raw_parts_mut(ptr.add(start), span) },
                i32::try_from(pitch).ok()?,
            ))
        };
        let (Some((y, sy)), Some((u, su)), Some((v, sv))) =
            (plane(0, h), plane(1, h / cy), plane(2, h / cy))
        else {
            unsafe { api.free(output.cast_const()) };
            return std::ptr::null();
        };
        let key = motion_key;
        let done = gpu.render_frame(
            current.sources(linear),
            next.sources(linear),
            y,
            sy,
            params[0],
            u,
            su,
            params[1],
            v,
            sv,
            params[2],
            key,
            usize::try_from(width).unwrap_or(0),
            usize::try_from(height).unwrap_or(0),
            motions,
            coverage,
            area,
            still,
            layer.map(|layer| &**layer),
        );
        if done.is_none() {
            unsafe { api.free(output.cast_const()) };
            return std::ptr::null();
        }
        let timing = self.options.timing(&self.video_info.source());
        let ratio = crate::core::f64_i64(timing.frame_num) / crate::core::f64_i64(timing.frame_den);
        let raw_phase = timing.raw_phase_256(frame, n);
        unsafe { api.copy_interpolated_timing(source, next_source, output, raw_phase, ratio) };
        output.cast_const()
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn reference_overlays(
        &self,
        api: &frame::PlaneApi,
        output: vs::Raw,
        frame: i32,
        class: i32,
        field: &VectorField,
        (luma, limits): (&[u8], &SceneLimits),
    ) {
        let phase = self
            .options
            .timing(&self.video_info.source())
            .output_phase_256(frame);
        let backward = phase > 127;
        let shape = *field.shape();
        let (pw, ph) = (shape.packed_width(), shape.packed_height());
        let count = usize::try_from(pw * ph).unwrap_or(0);
        let w = self.video_info.width;
        let h = self.video_info.height;
        let (cx, cy) = self.chroma();
        let shift = (cx.trailing_zeros(), cy.trailing_zeros());
        let planes = || -> Option<[(&'static mut [u8], usize); 3]> {
            let hu = usize::try_from(h).ok()?;
            let plane = |p: i32, rows: usize| -> Option<(&'static mut [u8], usize)> {
                let (ptr, pitch, len) = unsafe { api.write_plane(output, p, rows) }?;
                Some((unsafe { slice::from_raw_parts_mut(ptr, len) }, pitch))
            };
            let ch = hu / cy as usize;
            Some([plane(0, hu)?, plane(1, ch)?, plane(2, ch)?])
        };
        let step_x = shape.block_w - shape.overlap_x;
        let step_y = shape.block_h - shape.overlap_y;
        if self.options.debug_qmap()
            && let Some([(y, yp), (u, up), (v, _)]) = planes()
        {
            let mut map = vec![3u8; shape.blocks()];
            field.quality_map(backward, limits, luma, &mut map);
            draw_quality_map(
                &map,
                &shape,
                (step_x, step_y),
                (w, h),
                (y, yp),
                (u, v, up, shift),
            );
        }
        if self.options.debug_vectors()
            && let Some([(y, yp), (u, up), (v, _)]) = planes()
        {
            let (mut xs, mut ys, mut sad) =
                (vec![0u16; count], vec![0u16; count], vec![0u8; count]);
            field.pack(backward, &mut xs, &mut ys, pw, ph);
            field.sad_mask(
                backward,
                &mut sad,
                self.options.mask_area_scale(),
                self.options.mask_area_sharp(),
                pw,
                ph,
            );
            let zero = (self.options.debug_zerox(), self.options.debug_zeroy());
            draw_vectors(
                (&xs, &ys, &sad),
                &shape,
                (w, h),
                zero,
                (y, yp),
                (u, v, up, shift),
            );
        }
        if self.options.debug_qmode() {
            unsafe { self.apply_qmode_overlay(api, output, class) };
        }
    }

    fn rates(&self, params: &ReferenceParams) -> Rates {
        Rates::new(params, &self.video_info.source())
    }

    pub(crate) fn reference_request(
        &self,
        request_frame: vs::RequestFrameFilter,
        frame: i32,
        frame_ctx: vs::Raw,
        held: &mut Held,
    ) {
        let params = self.options.reference();
        let rates = self.rates(&params);
        let engine::Step { n, phase, .. } = rates.step(&params, frame);
        let dedup = self.dedup_limit().is_some();
        let mut wanted = Vec::new();
        let sources = if dedup {
            (n - RUN - 1).max(0)..=n + RUN + 1
        } else {
            n..=n + 1
        };
        wanted.extend(sources.map(|k| (self.clips.source, k)));
        if engine::needs_vectors(phase) || dedup && !self.later {
            let last = if dedup { n + RUN } else { n };
            self.vectors_wanted(&mut wanted, n..=last, engine::vector_reach(&params), held);
        }
        request_all(request_frame, wanted, frame_ctx);
    }

    fn vectors_wanted(
        &self,
        wanted: &mut Vec<(vs::Raw, i32)>,
        pairs: std::ops::RangeInclusive<i32>,
        reach: i32,
        held: &mut Held,
    ) {
        let (n, last) = (*pairs.start(), *pairs.end());
        if self.request_super {
            wanted.extend((n..=last + 1).map(|k| (self.clips.super_clip, k)));
        }
        if self.nvof.is_some() {
            let extra = nvof_extra(&self.options.reference());
            wanted.extend((n..=n + 1 + extra).map(|k| (self.clips.vec_src, k)));
        } else {
            wanted.extend(((n - reach).max(0)..=last + reach).map(|k| (self.clips.vectors, k)));
            for base in n..=last {
                self.layer_wanted(wanted, base, held);
            }
        }
    }

    pub(crate) unsafe fn reference_more(
        &self,
        frames: &[i32],
        frame_ctx: vs::Raw,
        vsapi: vs::ConstRaw,
        held: &mut Held,
    ) -> bool {
        if !self.later || self.dedup_limit().is_none() || !self.reference_enabled() {
            return false;
        }
        let (Some(get_frame), Some(request_frame), Some(api)) = (unsafe {
            (
                vs::table_fn::<vs::GetFrameFilter>(vsapi, vs::GET_FRAME_FILTER),
                vs::table_fn::<vs::RequestFrameFilter>(vsapi, vs::REQUEST_FRAME_FILTER),
                frame::PlaneApi::load(vsapi),
            )
        }) else {
            return false;
        };
        let fetch = Fetch {
            get_frame,
            free_frame: unsafe { vs::table_fn::<vs::FreeFrame>(vsapi, vs::FREE_FRAME) },
            frame_ctx,
        };
        let params = self.options.reference();
        let rates = self.rates(&params);
        let mut wanted = Vec::new();
        for &frame in frames {
            let engine::Step { n, raw, phase } = rates.step(&params, frame);
            let moved = self.dedup(&fetch, &api, n, raw);
            if !engine::needs_vectors(phase) && moved != (n, raw) {
                self.vectors_wanted(&mut wanted, moved.0..=moved.0, 0, held);
            }
        }
        if wanted.is_empty() {
            return false;
        }
        request_all(request_frame, wanted, frame_ctx);
        true
    }

    fn with_vectors<R>(
        &self,
        fetch: &Fetch,
        api: &frame::PlaneApi,
        (k, n): (i32, i32),
        f: impl FnOnce(&[u8]) -> R,
    ) -> Option<R> {
        if let Some(nvof) = self.nvof.as_ref() {
            let payload = match nvof.cached(k) {
                Some(payload) => payload,
                None if k < n || k > n + nvof_extra(&self.options.reference()) => return None,
                None => {
                    let metadata::VectorRecord::Ready(data) = self.vector_data() else {
                        return None;
                    };
                    let (w, h) = nvof.dimensions();
                    let current = fetch.get(self.clips.vec_src, k);
                    let next = fetch.get(self.clips.vec_src, k + 1);
                    let packed = unsafe {
                        (
                            crate::core::pack_nv12_frame(api, current, w, h),
                            crate::core::pack_nv12_frame(api, next, w, h),
                        )
                    };
                    fetch.drop(current);
                    fetch.drop(next);
                    let (Some(current), Some(next)) = packed else {
                        return None;
                    };
                    nvof.generate(k, &current, &next, data).ok()?
                }
            };
            return Some(f(&payload));
        }
        let vectors = fetch.get(self.clips.vectors, k);
        if vectors.is_null() {
            return None;
        }
        let result = unsafe { api.read_plane(vectors, 0, 1) }
            .map(|(ptr, stride, len)| f(unsafe { slice::from_raw_parts(ptr, len.max(stride)) }));
        fetch.drop(vectors);
        result
    }

    #[allow(clippy::too_many_lines)]
    pub(crate) unsafe fn reference_get(
        &self,
        frame: i32,
        blend: Option<&crate::gpu::BlendJob>,
        frame_ctx: vs::Raw,
        core: vs::Raw,
        vsapi: vs::ConstRaw,
        held: &Held,
    ) -> vs::ConstRaw {
        let Some(get_frame) =
            (unsafe { vs::table_fn::<vs::GetFrameFilter>(vsapi, vs::GET_FRAME_FILTER) })
        else {
            return std::ptr::null();
        };
        let fetch = Fetch {
            get_frame,
            free_frame: unsafe { vs::table_fn::<vs::FreeFrame>(vsapi, vs::FREE_FRAME) },
            frame_ctx,
        };
        let Some(api) = (unsafe { frame::PlaneApi::load(vsapi) }) else {
            return std::ptr::null();
        };
        let metadata::VectorRecord::Ready(vector_data) = self.vector_data() else {
            return std::ptr::null();
        };
        let params = self.options.reference();
        let rates = self.rates(&params);
        let engine::Step {
            mut n,
            mut raw,
            mut phase,
        } = rates.step(&params, frame);
        let moved = self.dedup(&fetch, &api, n, raw);
        // its neighbors were not requested
        let narrow = self.later && moved != (n, raw) && !engine::needs_vectors(phase);
        if moved != (n, raw) {
            (n, raw) = moved;
            phase = rates.phase(&params, raw);
        }
        let shape = FieldShape::from_data(&vector_data);
        let mut luma = vec![0u8; shape.blocks()];
        let mut luma_table = None;
        let limits = engine::limits(&params);
        let load = |k: i32| self.cached_field(&fetch, &api, (k, n), shape);
        let mut quality = |k: i32| -> i32 {
            if narrow && k != n {
                return 3;
            }
            if self.nvof.is_none()
                && let Some(cell) = crate::core::cache_cell(&self.quality_cells, i64::from(k), 64)
            {
                let mut fresh = false;
                let value = cell.get_or_init(|| {
                    let field = load(k)?;
                    fresh = true;
                    let (class, cached) =
                        engine::quality(&field, &params, &limits, &mut luma_table, &mut luma);
                    Some(std::sync::Arc::new((class, cached)))
                });
                return value.as_ref().map_or(3, |value| {
                    if !fresh {
                        luma.copy_from_slice(&value.1);
                    }
                    value.0
                });
            }
            if let Some(class) = self.cached_quality(k, &mut luma) {
                return class;
            }
            let Some(field) = load(k) else {
                return 3;
            };
            let (class, cached) =
                engine::quality(&field, &params, &limits, &mut luma_table, &mut luma);
            self.store_quality(k, class, &cached);
            class
        };
        let (class, neighbors_ok) = engine::classify(&params, n, phase, &mut quality);
        let field = || load(n).unwrap_or_else(|| std::sync::Arc::new(VectorField::new(shape)));
        phase = engine::adapt_phase(&params, &rates, raw, class, phase);
        let source = fetch.get(self.clips.source, n);
        let next_source = fetch.get(self.clips.source, n + 1);
        if source.is_null() || next_source.is_null() {
            fetch.drop(source);
            fetch.drop(next_source);
            return std::ptr::null();
        }
        let copy = engine::copy_choice(&params, phase, class, self.options.debug_vectors());
        if let Some(use_next) = copy {
            let (selected, other) = if use_next {
                (next_source, source)
            } else {
                (source, next_source)
            };
            let output = unsafe {
                self.padded_output(
                    selected,
                    source,
                    next_source,
                    frame,
                    n,
                    std::ptr::null(),
                    core,
                    vsapi,
                )
            };
            fetch.drop(other);
            if !output.is_null() && engine::needs_vectors(phase) {
                let field = field();
                unsafe {
                    self.reference_overlays(
                        &api,
                        output.cast_mut(),
                        frame,
                        class,
                        &field,
                        (&luma, &limits),
                    );
                };
                if self.options.debug_tt() {
                    unsafe { self.apply_timing_bar(&api, output.cast_mut(), frame) };
                }
            }
            return output;
        }
        let (use_fwd, use_bwd, force13) = engine::directions(&params, phase, class);
        let field = field();
        let output = unsafe {
            self.reference_calculate(
                &api,
                &fetch,
                &field,
                source,
                next_source,
                n,
                frame,
                phase,
                use_fwd,
                use_bwd,
                neighbors_ok,
                force13,
                class,
                (&luma, &limits),
                &params,
                &vector_data,
                blend,
                held,
                core,
            )
        };
        fetch.drop(source);
        fetch.drop(next_source);
        output
    }

    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::fn_params_excessive_bools
    )]
    unsafe fn reference_calculate(
        &self,
        api: &frame::PlaneApi,
        fetch: &Fetch,
        field: &VectorField,
        source: vs::ConstRaw,
        next_source: vs::ConstRaw,
        n: i32,
        frame: i32,
        phase: i32,
        use_fwd: bool,
        use_bwd: bool,
        neighbors_ok: bool,
        force13: bool,
        class: i32,
        (luma, limits): (&[u8], &SceneLimits),
        params: &ReferenceParams,
        vector_data: &metadata::VectorData,
        blend: Option<&crate::gpu::BlendJob>,
        held: &Held,
        core: vs::Raw,
    ) -> vs::ConstRaw {
        let shape = *field.shape();
        let width = shape.packed_width();
        let height = shape.packed_height();
        let time = engine::render_time(phase, use_fwd, use_bwd);
        let neither = !use_fwd && !use_bwd;
        let algo = params.algo;
        let engine::Masks {
            cover_bwd,
            cover_fwd,
            sad: sad_masks,
        } = engine::masks(field, params, time, neither);
        let want_extended = neighbors_ok && algo == 23 && !neither;
        let (zerox, zeroy) = (self.options.debug_zerox(), self.options.debug_zeroy());
        let motion_key = engine::motion_key(n, use_fwd, use_bwd, want_extended, (zerox, zeroy));
        let compute = || {
            std::sync::Arc::new(engine::motions(
                field,
                self.render_mode == 2,
                use_fwd,
                use_bwd,
                want_extended,
                (zerox, zeroy),
                |k| self.cached_field(fetch, api, (k, n), shape),
                n,
            ))
        };
        let cap = cache_cap(usize::try_from(width * height).unwrap_or(0) * 16);
        let motions = if self.options.debug_qmap() || self.options.debug_vectors() {
            compute()
        } else if let Some(cell) = crate::core::cache_cell(&self.motion_cache, motion_key, cap) {
            cell.get_or_init(|| Some(compute()))
                .clone()
                .unwrap_or_else(compute)
        } else {
            compute()
        };
        let extended = want_extended;
        let overlay = (extended && self.options.debug_qmap())
            .then(|| {
                let load = |k| self.cached_field(fetch, api, (k, n), shape);
                load(n + 1).or_else(|| load(n - 1))
            })
            .flatten();
        let field = overlay.as_deref().unwrap_or(field);
        let [fwd_x, fwd_y, bwd_x, bwd_y, next_x, next_y, prev_x, prev_y] =
            motions.each_ref().map(Vec::as_slice);
        let selected = engine::select(algo, use_fwd, use_bwd, extended, force13);
        let interp = !neither && !params.block;
        let sad = engine::merged_sad(selected, sad_masks.as_ref());
        let kind = engine::kind(selected, sad.is_some());
        let chroma = self.chroma();
        let mut renderer = Renderer::new(engine::render_shape(
            vector_data,
            self.video_info.width,
            self.video_info.height,
            params.area_blend,
            chroma,
        ));
        renderer.set_time(time);
        if self.render_mode == 2
            && let Some(gpu) = self.gpu.as_ref()
        {
            let linear = engine::gpu_linear(params, self.video_info.width);
            let kernel_params = engine::gpu_params(
                params,
                vector_data,
                (self.video_info.width, self.video_info.height),
                chroma,
                selected,
                time,
                sad_masks.is_some(),
                self.options.dither(),
            );
            let layer = unsafe { self.layer(api, fetch, gpu, n, params, vector_data, field, held) };
            let output = unsafe {
                self.reference_gpu(
                    api,
                    gpu,
                    source,
                    next_source,
                    n,
                    frame,
                    kernel_params,
                    linear,
                    width,
                    height,
                    [
                        (bwd_x, bwd_y),
                        (fwd_x, fwd_y),
                        (prev_x, prev_y),
                        (next_x, next_y),
                    ],
                    (&cover_fwd, &cover_bwd),
                    sad_masks
                        .as_ref()
                        .map(|(first, second)| (&second[..], &first[..])),
                    motion_key,
                    blend,
                    layer.as_ref(),
                    core,
                )
            };
            if !output.is_null() {
                unsafe {
                    self.reference_overlays(
                        api,
                        output.cast_mut(),
                        frame,
                        class,
                        field,
                        (luma, limits),
                    );
                };
                unsafe { self.apply_light_border(api, output.cast_mut(), frame) };
                if self.options.debug_tt() {
                    unsafe { self.apply_timing_bar(api, output.cast_mut(), frame) };
                }
            }
            return output;
        }
        let merged = |k: i32, raw: vs::ConstRaw| -> Option<Merged> {
            if shape.pel <= 1 || !self.request_super {
                let h = usize::try_from(self.video_info.height).ok()?;
                let (y, yp, yl) = unsafe { api.read_plane(raw, 0, h) }?;
                let (u, up, ul) = unsafe { api.read_plane(raw, 1, h / chroma.1 as usize) }?;
                let (v, _, vl) = unsafe { api.read_plane(raw, 2, h / chroma.1 as usize) }?;
                return Some(Merged {
                    _keep: None,
                    y: unsafe { slice::from_raw_parts(y, yl) },
                    u: unsafe { slice::from_raw_parts(u, ul) },
                    v: unsafe { slice::from_raw_parts(v, vl) },
                    y_pitch: yp,
                    uv_pitch: up,
                    slack: 0,
                });
            }
            let sup = fetch.get(self.clips.super_clip, k);
            let planes = unsafe {
                super_frame_planes(
                    api,
                    sup,
                    &self.video_info,
                    self.super_info.as_ref(),
                    self.sdata,
                )
            };
            let expand = planes.and_then(|planes| self.cached_expand(i64::from(k), &planes));
            fetch.drop(sup);
            let expand = expand?;
            let (y, u, v, y_pitch, uv_pitch) = expand.parts();
            let (y, u, v) = unsafe {
                (
                    slice::from_raw_parts(y.as_ptr(), y.len()),
                    slice::from_raw_parts(u.as_ptr(), u.len()),
                    slice::from_raw_parts(v.as_ptr(), v.len()),
                )
            };
            Some(Merged {
                _keep: Some(expand),
                y,
                u,
                v,
                y_pitch,
                uv_pitch,
                slack: crate::core::EXPAND_SLACK,
            })
        };
        let (Some(cur), Some(nxt)) = (merged(n, source), merged(n + 1, next_source)) else {
            return std::ptr::null();
        };
        let output_info = self.output_info();
        let Some(output) = (unsafe {
            api.new_frame(
                self.video_info.format,
                output_info.width,
                output_info.height,
                source,
                core,
            )
        }) else {
            return std::ptr::null();
        };
        let h = usize::try_from(self.video_info.height).unwrap_or(0);
        let w = usize::try_from(self.video_info.width).unwrap_or(0);
        let (pad_x, pad_y) = self.options.padding(&self.video_info.source());
        let (cx, cy) = (chroma.0 as usize, chroma.1 as usize);
        let plane_mut = |p: i32, rows: usize| -> Option<PlaneMut<'static>> {
            let (dx, dy) = if p == 0 { (1, 1) } else { (cx, cy) };
            let (px, py) = (
                usize::try_from(pad_x).ok()? / dx,
                usize::try_from(pad_y).ok()? / dy,
            );
            let (ptr, pitch, len) = unsafe { api.write_plane(output, p, rows + 2 * py) }?;
            let start = py * pitch + px;
            let span = (rows.checked_sub(1)?) * pitch + w / dx;
            if start + span > len {
                return None;
            }
            Some(PlaneMut {
                data: unsafe { slice::from_raw_parts_mut(ptr.add(start), span) },
                pitch,
            })
        };
        let (Some(y), Some(u), Some(v)) =
            (plane_mut(0, h), plane_mut(1, h / cy), plane_mut(2, h / cy))
        else {
            unsafe { api.free(output.cast_const()) };
            return std::ptr::null();
        };
        let mut dst = FrameMut { y, u, v };
        let vectors = PackedVectors {
            fwd_x,
            fwd_y,
            bwd_x,
            bwd_y,
            next_fwd_x: next_x,
            next_fwd_y: next_y,
            prev_bwd_x: prev_x,
            prev_bwd_y: prev_y,
            cover_bwd: &cover_bwd,
            cover_fwd: &cover_fwd,
            sad: sad.as_deref().unwrap_or(&[]),
        };
        renderer.render(kind, interp, &mut dst, nxt.frame(), cur.frame(), &vectors);
        let raw = unsafe {
            (
                crate::peel::planes(api, source, (h, h / cy)),
                crate::peel::planes(api, next_source, (h, h / cy)),
            )
        };
        let phase = if selected == 1 { 256 - time } else { time };
        let phase = phase as f32 / 256.0;
        if let (Some(limits), (Some(first), Some(second))) = (self.options.mask_still(), raw) {
            let mask = crate::core::cache_cell(&self.still_cache, n.into(), 8).and_then(|cell| {
                let (a, b) = (first[0], second[0]);
                let mask = || svpflow_core::still::mask(a.data, b.data, a.pitch, w, h, limits);
                cell.get_or_init(|| Some(std::sync::Arc::new(mask())))
                    .clone()
            });
            if let Some(mask) = mask {
                for (plane, target) in [&mut dst.y, &mut dst.u, &mut dst.v].into_iter().enumerate()
                {
                    let (a, b) = (first[plane], second[plane]);
                    let (w, h) = if plane == 0 { (w, h) } else { (w / cx, h / cy) };
                    let (data, pitch) = (&mut *target.data, target.pitch);
                    let size = self.video_info.width as usize;
                    svpflow_core::still::apply(
                        data, pitch, a.data, b.data, a.pitch, w, h, &mask, size, phase,
                    );
                }
            }
        }
        if let (Some(fit), (Some(first), Some(second))) =
            (self.layer_cpu(api, fetch, n, vector_data, held), raw)
        {
            let fit = fit.layer((w, h), cx);
            let frames = layer::Frames {
                first,
                second,
                time: phase,
            };
            let motion = crate::peel::motion((fwd_x, fwd_y), (bwd_x, bwd_y), vector_data);
            if !fit.empty() {
                for (plane, target) in [&mut dst.y, &mut dst.u, &mut dst.v].into_iter().enumerate()
                {
                    fit.render(plane, target.data, target.pitch, &frames, &motion, selected);
                }
            }
        }
        unsafe { self.reference_overlays(api, output, frame, class, field, (luma, limits)) };
        unsafe { self.apply_light_border(api, output, frame) };
        if self.options.debug_tt() {
            unsafe { self.apply_timing_bar(api, output, frame) };
        }
        let timing = self.options.timing(&self.video_info.source());
        let ratio = crate::core::f64_i64(timing.frame_num) / crate::core::f64_i64(timing.frame_den);
        let raw_phase = timing.raw_phase_256(frame, n);
        unsafe { api.copy_interpolated_timing(source, next_source, output, raw_phase, ratio) };
        output.cast_const()
    }
}

fn draw_quality_map(
    map: &[u8],
    shape: &FieldShape,
    (step_x, step_y): (i32, i32),
    (w, h): (i32, i32),
    (y, yp): (&mut [u8], usize),
    (u, v, up, (sx, sy)): (&mut [u8], &mut [u8], usize, (u32, u32)),
) {
    let mut color = [0u8; 3];
    for by in 0..shape.grid_h {
        for bx in 0..shape.grid_w {
            color = match map[(bx + shape.grid_w * by) as usize] {
                0 => [29, 255, 107],
                1 => [149, 43, 21],
                2 => [255, 0, 148],
                3 => [76, 84, 255],
                0xFF => continue,
                _ => color,
            };
            for py in by * step_y..(by + 1) * step_y {
                if py < 0 || py >= h {
                    continue;
                }
                for px in bx * step_x..(bx + 1) * step_x {
                    if px < 0 || px >= w {
                        continue;
                    }
                    let (py, px) = (py as usize, px as usize);
                    let blend = |dst: &mut u8, c: u8| {
                        *dst = ((20 * u16::from(c) + 235 * u16::from(*dst)) >> 8) as u8;
                    };
                    blend(&mut y[py * yp + px], color[0]);
                    let c = (py >> sy) * up + (px >> sx);
                    blend(&mut u[c], color[1]);
                    blend(&mut v[c], color[2]);
                }
            }
        }
    }
}

#[allow(clippy::too_many_lines)]
fn draw_vectors(
    (xs, ys, sad): (&[u16], &[u16], &[u8]),
    shape: &FieldShape,
    (w, h): (i32, i32),
    (zero_x, zero_y): (bool, bool),
    (y, yp): (&mut [u8], usize),
    (u, v, up, (sx, sy)): (&mut [u8], &mut [u8], usize, (u32, u32)),
) {
    let step_x = shape.block_w - shape.overlap_x;
    let step_y = shape.block_h - shape.overlap_y;
    let packed = shape.packed_width();
    let mut put = |px: i32, py: i32, c: [i32; 3], blend: bool| {
        let (x, yy) = (px as usize, py as usize);
        let mix = |dst: &mut u8, c: i32| {
            *dst = if blend {
                ((140 * u16::from(c as u8) + 115 * u16::from(*dst)) >> 8) as u8
            } else {
                c as u8
            };
        };
        mix(&mut y[yy * yp + x], c[0]);
        let ci = (yy >> sy) * up + (x >> sx);
        mix(&mut u[ci], c[1]);
        mix(&mut v[ci], c[2]);
    };
    for by in 0..shape.grid_h {
        for bx in 0..shape.grid_w {
            let cx = bx * step_x + shape.block_w / 2;
            let cy = by * step_y + shape.block_h / 2;
            let index = (bx + packed * by) as usize;
            let m = i32::from(sad[index]);
            let color = [
                (76 * m + 255 * (255 - m)) >> 8,
                (((255 - m) << 7) + 84 * m) >> 8,
                (255 * m + ((255 - m) << 7)) >> 8,
            ];
            let (mut x0, mut y0) = (cx - 1, cy - 1);
            let inside = |x: i32, yv: i32| x >= 0 && x < w && yv >= 0 && yv < h;
            if y0 < h && y0 >= 0 {
                for px in [x0, cx - 2, cx] {
                    if px < w && px >= 0 {
                        put(px, y0, color, true);
                    }
                }
            }
            if inside(x0, cy - 2) {
                put(x0, cy - 2, color, true);
            }
            if inside(x0, cy) {
                put(x0, cy, color, true);
            }
            let (vy, ay) = if zero_y {
                if zero_x {
                    continue;
                }
                (0, 0)
            } else {
                let vy = i32::from(ys[index]) - 1024;
                (vy, vy.abs())
            };
            let (x1, ax, sx) = if zero_x {
                if ay <= 1 {
                    continue;
                }
                (x0, 0, -1)
            } else {
                let vx = i32::from(xs[index]) - 1024;
                let (x1, ax) = (x0 + vx, vx.abs());
                if ay <= 1 && ax <= 1 {
                    continue;
                }
                (x1, ax, if x0 < x1 { 1 } else { -1 })
            };
            let y1 = y0 + vy;
            let sy = if y0 < y1 { 1 } else { -1 };
            let mut err = ax - ay;
            let mut state = 0;
            loop {
                match state {
                    0 => {
                        if y0 >= 0 && inside(x0, y0) {
                            put(x0, y0, color, false);
                        }
                        state = 1;
                    }
                    1 => state = if y0 == y1 { 2 } else { 3 },
                    2 => {
                        if x0 == x1 {
                            break;
                        }
                        state = 3;
                    }
                    _ => {
                        let e2 = 2 * err;
                        if e2 > -ay {
                            x0 += sx;
                            err -= ay;
                        }
                        if ax <= e2 {
                            state = 0;
                            continue;
                        }
                        y0 += sy;
                        err += ax;
                        if y0 >= 0 {
                            if inside(x0, y0) {
                                put(x0, y0, color, false);
                            }
                            state = 1;
                        } else {
                            state = if y0 == y1 { 2 } else { 3 };
                        }
                    }
                }
            }
        }
    }
}

fn nvof_extra(params: &ReferenceParams) -> i32 {
    i32::from(params.level != 0 || params.algo == 23 || params.algo > 89)
}

fn request_all(
    request_frame: vs::RequestFrameFilter,
    mut wanted: Vec<(vs::Raw, i32)>,
    frame_ctx: vs::Raw,
) {
    wanted.sort_unstable_by_key(|&(node, k)| (node as usize, k));
    wanted.dedup();
    for (node, k) in wanted {
        request_node(request_frame, node, k, frame_ctx);
    }
}
