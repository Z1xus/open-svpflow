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
        matches!(self.mode, Mode::SmoothFps)
            && self.render_mode == 1
            && !self.source_8bit_mode
            && self.options.reference_supported(&self.video_info)
            && matches!(self.vector_data(), metadata::VectorRecord::Ready(_))
    }

    fn cached_quality(&self, frame: i32) -> Option<i32> {
        let cache = self.quality_cache.lock().ok()?;
        cache
            .iter()
            .find(|(k, _)| *k == frame)
            .map(|&(_, class)| class)
    }

    fn store_quality(&self, frame: i32, class: i32) {
        if let Ok(mut cache) = self.quality_cache.lock() {
            if cache.len() >= 64 {
                cache.remove(0);
            }
            cache.push((frame, class));
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
        for k in (n - reach).max(0)..=n + reach {
            request_node(request_frame, self.clips.vectors, k, frame_ctx);
        }
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
        let limits = SceneLimits {
            blocks: params.blocks,
            zero: params.zero,
            m1: params.m1,
            m2: params.m2,
            scene: params.scene,
        };
        let load = |field: &mut VectorField, k: i32| {
            let vectors = fetch.get(self.clips.vectors, k);
            if vectors.is_null() {
                return false;
            }
            if let Some((ptr, stride, len)) = unsafe { api.read_plane(vectors, 0, 1) } {
                let payload = unsafe { slice::from_raw_parts(ptr, len.max(stride)) };
                field.update(payload);
            }
            fetch.drop(vectors);
            true
        };
        let mut quality = |field: &mut VectorField, k: i32| -> i32 {
            if let Some(class) = self.cached_quality(k) {
                return class;
            }
            if !load(field, k) {
                return 3;
            }
            field.avg_luma(&mut luma, params.luma);
            let class = field.quality(false, &limits, &luma);
            self.store_quality(k, class);
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
        } else if class == 3 && !params.blend {
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
        let neutral = || vec![1024u16; count];
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
        let mut extended = false;
        if neighbors_ok && algo == 23 && !neither {
            let load = |field: &mut VectorField, k: i32| {
                let vectors = fetch.get(self.clips.vectors, k);
                if let Some((ptr, stride, len)) = unsafe { api.read_plane(vectors, 0, 1) } {
                    let payload = unsafe { slice::from_raw_parts(ptr, len.max(stride)) };
                    field.update(payload);
                }
                fetch.drop(vectors);
            };
            load(field, n - 1);
            field.pack(true, &mut prev_x, &mut prev_y, width, height);
            load(field, n + 1);
            field.pack(false, &mut next_x, &mut next_y, width, height);
            extended = true;
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
        let interp = !neither;
        if neither {
            selected = 11;
        }
        let kind = match selected {
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
        let plane_mut = |p: i32, rows: usize| -> Option<PlaneMut<'static>> {
            let (ptr, pitch, len) = unsafe { api.write_plane(output, p, rows) }?;
            Some(PlaneMut {
                data: unsafe { slice::from_raw_parts_mut(ptr, len) },
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
        };
        renderer.render(kind, interp, &mut dst, nxt.frame(), cur.frame(), &vectors);
        let timing = self.options.timing(&self.video_info);
        let ratio = crate::core::f64_i64(timing.frame_num) / crate::core::f64_i64(timing.frame_den);
        let raw_phase = timing.raw_phase_256(frame, n);
        unsafe { api.copy_interpolated_timing(source, next_source, output, raw_phase, ratio) };
        output.cast_const()
    }
}
