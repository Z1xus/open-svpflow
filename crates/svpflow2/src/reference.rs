#![allow(clippy::cast_precision_loss, clippy::many_single_char_names)]

use std::slice;

use svpflow_core::smooth::{
    Algo, FieldShape, Frame, FrameMut, PackedVectors, Plane, PlaneMut, RenderShape, Renderer,
    SceneLimits, VectorField,
};

use crate::core::{FilterState, Mode, drop_frame, get_node, request_node, super_frame_planes};
use crate::{frame, metadata, options::ReferenceParams, vs};

struct Rates {
    num: u64,
    den: u64,
}

impl Rates {
    fn source_frame(&self, frame: i32) -> i32 {
        let frame = u64::try_from(frame.max(0)).unwrap_or(0);
        i32::try_from(frame * self.den / self.num).unwrap_or(i32::MAX)
    }

    fn raw_phase(&self, frame: i32, source: i32) -> f64 {
        256.0
            * (f64::from(frame) * f64::from(self.den as i32) / self.num as f64 - f64::from(source))
    }

    fn ratio(&self) -> f64 {
        self.num as f64 / self.den as f64
    }

    fn time_to_fixed(&self, raw: f64, mode: i32) -> i32 {
        let step = f64::from(self.den as i32) * 256.0 / self.num as f64;
        let mut left = (raw / step).floor() as i32;
        let rem = raw - f64::from(left) * step;
        let mut right = ((256.0 - raw - 0.001) / step).floor() as i32;
        let whole = if mode <= 1 {
            left += i32::from(rem as i32 >= (rem - step).abs() as i32);
            let tail = 256.0 - (f64::from(right) * step + raw);
            if tail as i32 > (tail - step).abs() as i32 {
                right += 1;
            }
            left << 8
        } else {
            let mut shifted = left << 8;
            if rem as i32 <= 0 {
                if left <= 0 {
                    shifted = 0;
                    left = 0;
                } else {
                    left -= 1;
                    shifted = left << 8;
                }
            }
            if (256.0 - (f64::from(right) * step + raw) - step).abs() < 0.1 {
                right += 1;
            }
            shifted
        };
        let total = left + right;
        if total == 0 {
            return 0;
        }
        let result = whole / total;
        if mode & !2 != 0 {
            if right >= left {
                result / 2
            } else {
                256 - (right << 7) / total
            }
        } else {
            result
        }
    }
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
    pub(crate) fn reference_enabled(&self) -> bool {
        (matches!(self.mode, Mode::SmoothFps) || self.nvof.is_some())
            && (self.render_mode == 1 || (self.render_mode == 2 && self.gpu.is_some()))
            && (!self.source_8bit_mode || self.nvof.is_some())
            && !(self.render_mode == 2 && self.options.block_enabled())
            && self.options.reference_supported()
            && matches!(self.vector_data(), metadata::VectorRecord::Ready(_))
    }

    fn cached_quality(&self, frame: i32, luma: &mut [u8]) -> Option<i32> {
        let cache = self.quality_cache.lock().ok()?;
        let (_, class, cached) = cache.iter().find(|(k, _, _)| *k == frame)?;
        luma.copy_from_slice(cached);
        Some(*class)
    }

    fn store_quality(&self, frame: i32, class: i32, luma: &[u8]) {
        if let Ok(mut cache) = self.quality_cache.lock() {
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
        core: vs::Raw,
    ) -> vs::ConstRaw {
        let w = usize::try_from(self.video_info.width).unwrap_or(0);
        let h = usize::try_from(self.video_info.height).unwrap_or(0);
        let uploads = |raw: vs::ConstRaw| -> Option<[crate::gpu::UploadPlane<'static>; 3]> {
            let plane =
                |p: i32, pw: usize, ph: usize| -> Option<crate::gpu::UploadPlane<'static>> {
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
                plane(1, w / 2, h / 2)?,
                plane(2, w / 2, h / 2)?,
            ])
        };
        let (Some(current), Some(next)) = (uploads(source), uploads(next_source)) else {
            return std::ptr::null();
        };
        let (Some(current), Some(next)) = (
            gpu.cache_frame(i64::from(n), current),
            gpu.cache_frame(i64::from(n) + 1, next),
        ) else {
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
        let (pad_x, pad_y) = self.options.padding(&self.video_info);
        let bytes = if crate::video_format::source_depth(&self.video_info) > 8 {
            2
        } else {
            1
        };
        let plane = |p: i32, rows: usize| -> Option<(&'static mut [u8], i32)> {
            let div = if p == 0 { 1 } else { 2 };
            let (px, py) = (
                usize::try_from(pad_x / div).ok()?,
                usize::try_from(pad_y / div).ok()?,
            );
            let (ptr, pitch, len) = unsafe { api.write_plane(output, p, rows + 2 * py) }?;
            let start = py * pitch + px * bytes;
            let span = (rows.checked_sub(1)?) * pitch + (w / div as usize) * bytes;
            if start + span > len {
                return None;
            }
            Some((
                unsafe { slice::from_raw_parts_mut(ptr.add(start), span) },
                i32::try_from(pitch).ok()?,
            ))
        };
        let (Some((y, sy)), Some((u, su)), Some((v, sv))) =
            (plane(0, h), plane(1, h / 2), plane(2, h / 2))
        else {
            unsafe { api.free(output.cast_const()) };
            return std::ptr::null();
        };
        let key = i64::from(frame);
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
        );
        if done.is_none() {
            unsafe { api.free(output.cast_const()) };
            return std::ptr::null();
        }
        let timing = self.options.timing(&self.video_info);
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
            .timing(&self.video_info)
            .output_phase_256(frame);
        let backward = phase > 127;
        let shape = *field.shape();
        let (pw, ph) = (shape.packed_width(), shape.packed_height());
        let count = usize::try_from(pw * ph).unwrap_or(0);
        let w = self.video_info.width;
        let h = self.video_info.height;
        let planes = || -> Option<[(&'static mut [u8], usize); 3]> {
            let hu = usize::try_from(h).ok()?;
            let plane = |p: i32, rows: usize| -> Option<(&'static mut [u8], usize)> {
                let (ptr, pitch, len) = unsafe { api.write_plane(output, p, rows) }?;
                Some((unsafe { slice::from_raw_parts_mut(ptr, len) }, pitch))
            };
            Some([plane(0, hu)?, plane(1, hu / 2)?, plane(2, hu / 2)?])
        };
        let step_x = shape.block_w - shape.overlap_x;
        let step_y = shape.block_h - shape.overlap_y;
        if self.options.debug_qmap()
            && let Some([(y, yp), (u, up), (v, _)]) = planes()
        {
            let mut map = vec![3u8; shape.blocks()];
            field.quality_map(backward, limits, luma, &mut map);
            draw_quality_map(&map, &shape, (step_x, step_y), (w, h), (y, yp), (u, v, up));
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
            draw_vectors((&xs, &ys, &sad), &shape, (w, h), zero, (y, yp), (u, v, up));
        }
        if self.options.debug_qmode() {
            unsafe { self.apply_qmode_overlay(api, output, class) };
        }
    }

    fn rates(&self, params: &ReferenceParams) -> Rates {
        let src_num = u64::try_from(self.video_info.fps_num.max(1)).unwrap_or(1);
        let src_den = u64::try_from(self.video_info.fps_den.max(1)).unwrap_or(1);
        let (out_num, out_den) = if params.absolute {
            (params.rate_num, params.rate_den)
        } else {
            (src_num * params.rate_num, src_den * params.rate_den)
        };
        Rates {
            num: out_num * src_den,
            den: out_den * src_num,
        }
    }

    pub(crate) fn reference_request(
        &self,
        request_frame: vs::RequestFrameFilter,
        frame: i32,
        frame_ctx: vs::Raw,
    ) {
        let params = self.options.reference();
        let n = self.rates(&params).source_frame(frame);
        for k in [n, n + 1] {
            request_node(request_frame, self.clips.source, k, frame_ctx);
            if self.request_super {
                request_node(request_frame, self.clips.super_clip, k, frame_ctx);
            }
        }
        let reach = if params.level >= 2 { 2 } else { 1 };
        if self.nvof.is_some() {
            for k in n..=n + 1 + nvof_extra(&params) {
                request_node(request_frame, self.clips.vec_src, k, frame_ctx);
            }
            return;
        }
        for k in (n - reach).max(0)..=n + reach {
            request_node(request_frame, self.clips.vectors, k, frame_ctx);
        }
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
        frame_ctx: vs::Raw,
        core: vs::Raw,
        vsapi: vs::ConstRaw,
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
        let n = rates.source_frame(frame);
        let raw = rates.raw_phase(frame, n);
        let mut phase = (raw + 0.5) as i32;
        if matches!(params.scene_mode, 1 | 2) {
            phase = rates.time_to_fixed(raw, if params.scene_mode == 1 { 0 } else { 2 });
        }
        let shape = FieldShape::from_data(&vector_data);
        let mut field = VectorField::new(shape);
        let mut luma = vec![0u8; shape.blocks()];
        let mut luma_table = None;
        let limits = SceneLimits {
            blocks: params.blocks,
            zero: params.zero,
            m1: params.m1,
            m2: params.m2,
            scene: params.scene,
        };
        let load = |field: &mut VectorField, k: i32| {
            self.with_vectors(&fetch, &api, (k, n), |payload| field.update(payload))
                .is_some()
        };
        let mut quality = |field: &mut VectorField, k: i32| -> i32 {
            if let Some(class) = self.cached_quality(k, &mut luma) {
                return class;
            }
            if !load(field, k) {
                return 3;
            }
            let table = luma_table.get_or_insert_with(|| field.luma_table(params.luma));
            field.avg_luma(&mut luma, table);
            let class = field.quality(false, &limits, &luma);
            let mut cached = luma.clone();
            if let Some(tail) = cached.len().checked_sub(4) {
                let mean = (field.luma_geomean() * 1_000_000.0) as i32;
                cached[tail..].copy_from_slice(&mean.to_le_bytes());
            }
            self.store_quality(k, class, &cached);
            class
        };
        let algo = params.algo;
        let mut class = 3;
        let mut neighbors_ok = false;
        if phase & !0x100 != 0 {
            let level = params.level;
            let (prev, next, prev2, next2) = if (level == 0 && algo != 23 && algo <= 89) || n <= 1 {
                (3, 3, -1, -1)
            } else {
                let prev = quality(&mut field, n - 1);
                let next = quality(&mut field, n + 1);
                if level <= 1 || n == 2 {
                    (prev, next, -1, -1)
                } else {
                    let prev2 = quality(&mut field, n - 2);
                    let next2 = if level > 2 {
                        quality(&mut field, n + 2)
                    } else {
                        -1
                    };
                    if prev2 == 3 || next2 == 3 {
                        (prev, next, -1, -1)
                    } else {
                        (prev, next, prev2, next2)
                    }
                }
            };
            let current = quality(&mut field, n);
            class = current;
            if level != 0 && prev != 3 && next != 3 && current != 3 {
                let sum = next + prev + current;
                let average = if prev2 < 0 {
                    f64::from(sum as f32) / 3.0
                } else if next2 < 0 {
                    f64::from((sum + prev2) as f32) * 0.25
                } else {
                    f64::from((sum + prev2 + next2) as f32) / 5.0
                };
                class = (f64::from(average as f32) + 0.5).floor() as i32;
            }
            neighbors_ok = prev <= 2 && next <= 2;
        }
        if params.scene_mode == 3 && (0..=2).contains(&class) {
            let adaptive = params.adaptive[class as usize];
            if adaptive >= 0 {
                phase = if rates.ratio() >= 2.0 {
                    rates.time_to_fixed(raw, adaptive)
                } else {
                    0
                };
            }
        }
        let source = fetch.get(self.clips.source, n);
        let next_source = fetch.get(self.clips.source, n + 1);
        if source.is_null() || next_source.is_null() {
            fetch.drop(source);
            fetch.drop(next_source);
            return std::ptr::null();
        }
        let copy = if phase == 0 {
            Some(false)
        } else if phase == 256 {
            Some(true)
        } else if (class == 3 && !params.blend) || self.options.debug_vectors() {
            Some(phase > 127)
        } else {
            None
        };
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
            if !output.is_null() && phase & !0x100 != 0 {
                load(&mut field, n);
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
        let (mut use_fwd, mut use_bwd) = if algo == 2 {
            (phase > 127, phase <= 127)
        } else {
            (true, algo > 10)
        };
        if class == 3 {
            use_fwd = false;
            use_bwd = false;
        }
        let force13 = class > 0 && params.force13;
        load(&mut field, n);
        let output = unsafe {
            self.reference_calculate(
                &api,
                &fetch,
                &mut field,
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
        field: &mut VectorField,
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
        core: vs::Raw,
    ) -> vs::ConstRaw {
        let shape = *field.shape();
        let width = shape.packed_width();
        let height = shape.packed_height();
        let count = usize::try_from(width * height).unwrap_or(0);
        let time = if use_fwd && !use_bwd {
            256 - phase
        } else {
            phase
        };
        let neither = !use_fwd && !use_bwd;
        let algo = params.algo;
        let unset = u16::from(self.render_mode != 2) * 1024;
        let neutral = || vec![unset; count];
        let (mut fwd_x, mut fwd_y, mut bwd_x, mut bwd_y) =
            (neutral(), neutral(), neutral(), neutral());
        let (mut next_x, mut next_y, mut prev_x, mut prev_y) =
            (neutral(), neutral(), neutral(), neutral());
        let mut cover_bwd = vec![0u8; count];
        let mut cover_fwd = vec![0u8; count];
        if use_fwd {
            field.pack(false, &mut fwd_x, &mut fwd_y, width, height);
        }
        if algo > 20 && !neither {
            field.cover_mask(
                false,
                &mut cover_bwd,
                params.cover,
                256 - time,
                width,
                height,
            );
        }
        if use_bwd {
            field.pack(true, &mut bwd_x, &mut bwd_y, width, height);
        }
        if algo > 20 && !neither {
            field.cover_mask(true, &mut cover_fwd, params.cover, time, width, height);
        }
        if neither {
            for buf in [&mut fwd_x, &mut fwd_y, &mut bwd_x, &mut bwd_y] {
                buf.fill(1024);
            }
        }
        let sad_masks = match params.sad {
            Some((scale, sharp)) if !neither => {
                let mut first = vec![0u8; count];
                let mut second = vec![0u8; count];
                field.sad_mask(false, &mut first, scale, sharp, width, height);
                field.sad_mask(true, &mut second, scale, sharp, width, height);
                Some((first, second))
            }
            _ => None,
        };
        let mut extended = false;
        if neighbors_ok && algo == 23 && !neither {
            let load = |field: &mut VectorField, k: i32| {
                self.with_vectors(fetch, api, (k, n), |payload| field.update(payload));
            };
            load(field, n - 1);
            field.pack(true, &mut prev_x, &mut prev_y, width, height);
            load(field, n + 1);
            field.pack(false, &mut next_x, &mut next_y, width, height);
            extended = true;
        }
        if self.options.debug_zerox() {
            for buf in [&mut fwd_x, &mut bwd_x, &mut next_x, &mut prev_x] {
                buf.fill(1024);
            }
        }
        if self.options.debug_zeroy() {
            for buf in [&mut fwd_y, &mut bwd_y, &mut next_y, &mut prev_y] {
                buf.fill(1024);
            }
        }
        let mut selected = algo;
        if selected == 2 {
            if use_fwd && !use_bwd {
                selected = 1;
            }
        } else if selected == 23 || selected > 10 {
            if selected == 23 && !extended {
                selected = 21;
            }
            if force13 {
                selected = 13;
            }
        }
        let interp = !neither && !params.block;
        if neither {
            selected = 11;
        }
        let sad = sad_masks.as_ref().map(|(first, second)| match selected {
            1 => first.clone(),
            2 => second.clone(),
            _ => first.iter().zip(second).map(|(a, b)| *a.max(b)).collect(),
        });
        let sad_on = sad.is_some();
        let kind = match selected {
            1 if sad_on => Algo::FastSad { next: true },
            2 if sad_on => Algo::FastSad { next: false },
            11 if sad_on => Algo::NoMaskSad { median: false },
            13 if sad_on => Algo::NoMaskSad { median: true },
            21 if sad_on => Algo::NormalSad { simple: true },
            22 if sad_on => Algo::NormalSad { simple: false },
            1 => Algo::Fast { next: true },
            2 => Algo::Fast { next: false },
            11 => Algo::NoMask { median: false },
            13 => Algo::NoMask { median: true },
            21 => Algo::Normal { simple: true },
            22 => Algo::Normal { simple: false },
            23 => Algo::Extended,
            _ => Algo::Fill,
        };
        let block = vector_data.effective_block();
        let mut renderer = Renderer::new(RenderShape {
            width: self.video_info.width,
            height: self.video_info.height,
            step_x: block.width,
            step_y: block.height,
            grid_w: width,
            grid_h: height,
            origin_x: shape.block_w / 2,
            origin_y: shape.block_h / 2,
            pel: shape.pel,
            blend: params.area_blend,
        });
        renderer.set_time(time);
        if self.render_mode == 2
            && let Some(gpu) = self.gpu.as_ref()
        {
            let area_blend = params.area_blend;
            let gpu_time = if selected == 1 { 256 - time } else { time };
            let fraction = (f64::from(gpu_time) * 0.003_906_25) as f32;
            let blend = area_blend as f32;
            let sad_blend = if gpu_time > 126 {
                (1.0 - f64::from(blend * (1.0 - f64::from(fraction)) as f32)) as f32
            } else {
                fraction * blend
            };
            let cubic = params.cubic & 1 != 0;
            let cubic_ref = params.cubic & 2 != 0;
            let linear = params.linear && self.video_info.width < 3001;
            let block = vector_data.effective_block();
            let make = |chroma: bool, offset_x: i32, offset_y: i32| crate::gpu::KernelParams {
                algorithm: selected,
                width: if chroma {
                    self.video_info.width / 2
                } else {
                    self.video_info.width
                },
                height: if chroma {
                    self.video_info.height / 2
                } else {
                    self.video_info.height
                },
                x_ratio: if chroma { 2 } else { 1 },
                y_ratio: if chroma { 2 } else { 1 },
                pel: shape.pel,
                block_w: if cubic {
                    block.width
                } else {
                    block.width * width
                },
                block_h: if cubic {
                    block.height
                } else {
                    block.height * height
                },
                origin_x: shape.overlap_x / 2,
                origin_y: shape.overlap_y / 2,
                phase: gpu_time,
                has_sad: i32::from(sad_masks.is_some()),
                linear_luma: i32::from(!chroma && linear),
                cubic: i32::from(cubic),
                cubic_ref: i32::from(cubic_ref),
                offset_x,
                offset_y,
                sad_blend,
                dither: i32::from(self.options.dither()),
            };
            let output = unsafe {
                self.reference_gpu(
                    api,
                    gpu,
                    source,
                    next_source,
                    n,
                    frame,
                    [
                        make(false, 0, 0),
                        make(true, 0, self.video_info.height),
                        make(true, self.video_info.width / 2, self.video_info.height),
                    ],
                    linear,
                    width,
                    height,
                    [
                        (&bwd_x, &bwd_y),
                        (&fwd_x, &fwd_y),
                        (&prev_x, &prev_y),
                        (&next_x, &next_y),
                    ],
                    (&cover_fwd, &cover_bwd),
                    sad_masks
                        .as_ref()
                        .map(|(first, second)| (&second[..], &first[..])),
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
                let (u, up, ul) = unsafe { api.read_plane(raw, 1, h / 2) }?;
                let (v, _, vl) = unsafe { api.read_plane(raw, 2, h / 2) }?;
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
        let (pad_x, pad_y) = self.options.padding(&self.video_info);
        let plane_mut = |p: i32, rows: usize| -> Option<PlaneMut<'static>> {
            let div = if p == 0 { 1 } else { 2 };
            let (px, py) = (
                usize::try_from(pad_x / div).ok()?,
                usize::try_from(pad_y / div).ok()?,
            );
            let (ptr, pitch, len) = unsafe { api.write_plane(output, p, rows + 2 * py) }?;
            let start = py * pitch + px;
            let span = (rows.checked_sub(1)?) * pitch + w / div as usize;
            if start + span > len {
                return None;
            }
            Some(PlaneMut {
                data: unsafe { slice::from_raw_parts_mut(ptr.add(start), span) },
                pitch,
            })
        };
        let (Some(y), Some(u), Some(v)) =
            (plane_mut(0, h), plane_mut(1, h / 2), plane_mut(2, h / 2))
        else {
            unsafe { api.free(output.cast_const()) };
            return std::ptr::null();
        };
        let mut dst = FrameMut { y, u, v };
        let vectors = PackedVectors {
            fwd_x: &fwd_x,
            fwd_y: &fwd_y,
            bwd_x: &bwd_x,
            bwd_y: &bwd_y,
            next_fwd_x: &next_x,
            next_fwd_y: &next_y,
            prev_bwd_x: &prev_x,
            prev_bwd_y: &prev_y,
            cover_bwd: &cover_bwd,
            cover_fwd: &cover_fwd,
            sad: sad.as_deref().unwrap_or(&[]),
        };
        renderer.render(kind, interp, &mut dst, nxt.frame(), cur.frame(), &vectors);
        unsafe { self.reference_overlays(api, output, frame, class, field, (luma, limits)) };
        unsafe { self.apply_light_border(api, output, frame) };
        if self.options.debug_tt() {
            unsafe { self.apply_timing_bar(api, output, frame) };
        }
        let timing = self.options.timing(&self.video_info);
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
    (u, v, up): (&mut [u8], &mut [u8], usize),
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
                    let c = (py >> 1) * up + (px >> 1);
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
    (u, v, up): (&mut [u8], &mut [u8], usize),
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
        let ci = (yy >> 1) * up + (x >> 1);
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
