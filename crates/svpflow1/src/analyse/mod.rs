#[cfg(not(target_arch = "wasm32"))]
mod chain;
mod field;
#[cfg(not(target_arch = "wasm32"))]
mod gpu;
#[cfg(target_arch = "wasm32")]
#[path = "gpu_stub.rs"]
mod gpu;
mod metric;
mod params;
mod planes;
mod pyramid;

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use gpu::{purge, trim};
pub(crate) use params::{AnalyseParams, SuperParams};
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use planes::plane_size;
pub(crate) use planes::{RawPlane, SuperFrameView};

use crate::{super_build, super_opts::SuperOpts};
use pyramid::Pyramid;

const MAGIC: i32 = 0xA0;

pub(crate) fn analysis_header(params: &AnalyseParams) -> [i32; 15] {
    let stage = params.output();
    let l = stage.layout;
    let sp = params.super_params;
    [
        MAGIC,
        params.vectors,
        l.width,
        l.height,
        sp.pel,
        stage.levels,
        0,
        sp.width,
        sp.height,
        l.overlap_x,
        l.overlap_y,
        stage.blk_x,
        stage.blk_y,
        params.gpu_mode,
        params.delta,
    ]
}

pub(crate) fn frame_len(params: &AnalyseParams) -> usize {
    4 * (1 + 15) + 2 * region_len(params)
}

fn region_len(params: &AnalyseParams) -> usize {
    4 + 8 * params.output_count()
}

pub(crate) fn analyse(
    params: &AnalyseParams,
    current: &SuperFrameView<'_>,
    next: &SuperFrameView<'_>,
    keys: Option<[(usize, i32, u32); 2]>,
) -> Vec<u32> {
    let count = params.output_count();
    let mut words = vec![0u32; frame_len(params) / 4];
    let backward_only = params.vectors == 2;
    let (first, second, keys) = if backward_only {
        (current, next, keys)
    } else {
        (next, current, keys.map(|[a, b]| [b, a]))
    };
    {
        let (head, regions) = words.split_at_mut(16);
        head[0] = 16;
        for (word, value) in head[1..].iter_mut().zip(analysis_header(params)) {
            *word = value as u32;
        }
        let marker = (2 * params.written_count() + 1) as u32;
        let (region0, region1) = regions.split_at_mut(1 + 2 * count);
        region0[0] = if backward_only { 0 } else { marker };
        region1[0] = marker;
        let (region0, region1) = (&mut region0[1..], &mut region1[1..]);
        let outputs = if backward_only {
            [Some(region1), None]
        } else {
            [Some(region0), (params.vectors == 3).then_some(region1)]
        };
        if params.fast.level >= 1 && frames_static(first, second) {
            let [out0, out1] = outputs;
            if let Some(out) = out0 {
                fill_static_output(params, first, out);
            }
            if let Some(out) = out1 {
                fill_static_output(params, second, out);
            }
            return words;
        }
        #[cfg(not(target_arch = "wasm32"))]
        let prefer_cpu_overlap = !params.super_params.lean
            && params.super_params.pel == 1
            && params.stages.len() == 1
            && params.stages[0].blk_x * params.stages[0].blk_y <= 14_000
            && svpflow_core::threads::workers() >= 6;
        #[cfg(not(target_arch = "wasm32"))]
        let plan = keys
            .filter(|_| {
                !prefer_cpu_overlap
                    && params.gpu_mode == 1
                    && gpu::refine_gpu().is_some_and(gpu::RefineGpu::chain_ready)
            })
            .and_then(|_| chain::ChainPlan::new(params, first, second));
        #[cfg(target_arch = "wasm32")]
        let plan: Option<()> = None;
        #[cfg(not(target_arch = "wasm32"))]
        if let (Some(plan), Some(keys)) = (&plan, keys) {
            if let Some(failed) = run_chain(params, plan, [first, second], keys, outputs) {
                cpu_search_stages(params, first, second, failed);
            }
            return words;
        }
        let _ = plan;
        let rebuilt = params
            .super_params
            .lean
            .then(|| [Rebuilt::new(params, first), Rebuilt::new(params, second)]);
        let views = rebuilt
            .as_ref()
            .map(|r| [r[0].view(params), r[1].view(params)]);
        let [first, second] = views.as_ref().map_or([first, second], |v| [&v[0], &v[1]]);
        #[cfg(not(target_arch = "wasm32"))]
        let allow_gpu_refine = !prefer_cpu_overlap;
        #[cfg(target_arch = "wasm32")]
        let allow_gpu_refine = true;
        let exact_dontv_gpu = allow_gpu_refine
            && params.gpu_mode == 1
            && params.stages[0].settings(params, 0, false).search_param == 0
            && params.stages[0].settings(params, 0, false).search_type
                == field::SearchType::Exhaustive
            && matches!(params.stages[0].layout.width, 8 | 16 | 32)
            && matches!(params.stages[0].layout.height, 8 | 16 | 32);
        let gpu = (allow_gpu_refine
            && keys.is_some()
            && (params.stages[0].gpu_searchable(params)
                || exact_dontv_gpu
                || params.stages.len() > 1
                    && params.stages[1..]
                        .iter()
                        .any(|stage| stage.gpu_refinable(params))))
        .then(gpu::refine_gpu)
        .flatten();
        let order = keys.map_or(0, |k| k[0].1.min(k[1].1));
        let mut session = gpu.map(|gpu| gpu.session(order));
        if let Some(session) = session.as_mut() {
            let keys = keys.unwrap_or_default();
            let main = &params.stages[0];
            let tail_levels = main.gpu_searchable(params) && main.gpu_level(1);
            for (slot, (frame, key)) in [first, second].into_iter().zip(keys).enumerate() {
                let level = frame.level(0);
                let tail = if tail_levels {
                    let top = frame.level(params.stages[0].levels - 1);
                    [top.y, top.u, top.v].map(|p| p.base() + p.pitch * p.height as usize)
                } else {
                    [0; 3]
                };
                let planes = gpu::Planes {
                    data: [level.y.data(), level.u.data(), level.v.data()],
                    pitch: [level.y.pitch, level.u.pitch, level.v.pitch],
                    base: [level.y.base(), level.u.base(), level.v.base()],
                    width: [
                        level.y.width.max(0) as usize,
                        level.u.width.max(0) as usize,
                        level.v.width.max(0) as usize,
                    ],
                    height: [level.y.height, level.u.height, level.v.height],
                    pel: level.y.pel(),
                    expand: params.super_params.gpu == 1 && params.super_params.flags & 2 == 0,
                    tail,
                    pyramid: None,
                    lean: false,
                    uses: key.2,
                };
                session.defer(slot, (key.0, key.1), planes);
            }
        }
        search_stages(params, first, second, outputs, session.as_mut());
    }
    words
}

struct Rebuilt {
    planes: [Vec<u8>; 3],
    pitches: [usize; 3],
}

impl Rebuilt {
    fn new(params: &AnalyseParams, frame: &SuperFrameView<'_>) -> Self {
        let sp = params.super_params;
        let opts = SuperOpts::from_params(&sp);
        let source = frame.source.expect("lean frames carry the source planes");
        let rows = usize::try_from(opts.super_height()).unwrap_or(0);
        let (sx, sy) = params.chroma_shift;
        let pitches = source.map(|plane| plane.pitch);
        let planes = std::array::from_fn(|index| {
            let shift = if index == 0 { (0, 0) } else { (sx, sy) };
            let mut plane = vec![0u8; pitches[index] * (rows >> shift.1)];
            super_build::build_plane(
                &mut plane,
                pitches[index],
                source[index].data,
                source[index].pitch,
                sp.width as usize,
                sp.height as usize,
                shift,
                &opts,
            );
            plane
        });
        Self { planes, pitches }
    }

    fn view(&self, params: &AnalyseParams) -> SuperFrameView<'_> {
        let sp = params.super_params;
        SuperFrameView {
            planes: std::array::from_fn(|index| RawPlane {
                data: &self.planes[index],
                pitch: self.pitches[index],
            }),
            source: None,
            width: sp.width,
            height: sp.height,
            pel: sp.pel,
            chroma_shift: params.chroma_shift,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn cpu_search_stages(
    params: &AnalyseParams,
    first: &SuperFrameView<'_>,
    second: &SuperFrameView<'_>,
    outputs: [Option<&mut [u32]>; 2],
) {
    if params.super_params.lean {
        let (first, second) = (Rebuilt::new(params, first), Rebuilt::new(params, second));
        search_stages(
            params,
            &first.view(params),
            &second.view(params),
            outputs,
            None,
        );
    } else {
        search_stages(params, first, second, outputs, None);
    }
}

fn frames_static(first: &SuperFrameView<'_>, second: &SuperFrameView<'_>) -> bool {
    let left = first.level(0).y;
    let right = second.level(0).y;
    let (width, height) = (left.width.min(right.width), left.height.min(right.height));
    if width <= 0 || height <= 0 {
        return false;
    }
    let (ad, bd) = (left.data(), right.data());
    let (ap, bp) = (left.pitch, right.pitch);
    let (ab, bb) = (left.base(), right.base());
    let (wu, hu) = (width as usize, height as usize);
    if ab + (hu - 1) * ap + wu > ad.len() || bb + (hu - 1) * bp + wu > bd.len() {
        return false;
    }
    for phase in [0usize, 8usize] {
        let mut row = phase;
        while row < hu {
            let ra = &ad[ab + row * ap..];
            let rb = &bd[bb + row * bp..];
            let mut col = phase;
            while col < wu {
                if ra[col] != rb[col] {
                    return false;
                }
                col += 16;
            }
            row += 16;
        }
    }
    true
}

fn fill_static_output(params: &AnalyseParams, frame: &SuperFrameView<'_>, out: &mut [u32]) {
    let stage = params.output();
    let layout = stage.layout;
    let (bw, bh) = (layout.width, layout.height);
    let (sx, sy) = (bw - layout.overlap_x, bh - layout.overlap_y);
    let (blk_x, blk_y) = (stage.blk_x, stage.blk_y);
    let plane = frame.level(0).y;
    let data = plane.data();
    let base = plane.base();
    let pitch = plane.pitch;
    let (max_x, max_y) = (plane.width.max(1) - 1, plane.height.max(1) - 1);
    let qx0 = bw >> 2;
    let qx1 = (bw * 3) >> 2;
    let qy0 = bh >> 2;
    let qy1 = (bh * 3) >> 2;
    let sample = |px: i32, py: i32| -> u32 {
        let idx = base + py.clamp(0, max_y) as usize * pitch + px.clamp(0, max_x) as usize;
        u32::from(data.get(idx).copied().unwrap_or(128))
    };
    for by in 0..blk_y {
        let y0 = by * sy;
        for bx in 0..blk_x {
            let x0 = bx * sx;
            let avg = (sample(x0 + qx0, y0 + qy0)
                + sample(x0 + qx1, y0 + qy0)
                + sample(x0 + qx0, y0 + qy1)
                + sample(x0 + qx1, y0 + qy1)
                + 2)
                >> 2;
            let idx = (by * blk_x + bx) as usize;
            out[2 * idx] = 0;
            out[2 * idx + 1] = avg << 24;
        }
    }
}

fn search_stages(
    params: &AnalyseParams,
    first: &SuperFrameView<'_>,
    second: &SuperFrameView<'_>,
    outputs: [Option<&mut [u32]>; 2],
    mut session: Option<&mut gpu::Session<'_, '_>>,
) {
    let mirror = params.fast.mirror() && params.vectors == 3;
    let start_level = i32::from(params.fast.start_level).max(params.min_level());
    let main = &params.stages[0];
    if params.fast.start_level >= 1 && start_level < main.levels {
        let [out_forward, out_backward] = outputs;
        let mut lambda = 0;
        let mut previous = Pyramid::new(params, main, false, start_level);
        previous.search(
            params,
            main,
            first,
            second,
            &mut lambda,
            [None, None],
            |_, _, _, _| {},
            session.as_deref_mut(),
            start_level,
        );
        let out_stage = params.output();
        if let Some(fwd) = previous.field_ref(false, start_level as usize) {
            if let Some(out) = out_forward {
                fwd.upsample_bilateral_output(
                    out_stage,
                    params.super_params.pel,
                    &first.level(0).y,
                    &first.level(start_level).y,
                    out,
                );
            }
            if let Some(out) = out_backward {
                if mirror {
                    let mut bwd = main.field(
                        start_level,
                        params.super_params.pel,
                        params.chroma_shift,
                        false,
                        params.fast,
                    );
                    bwd.mirror_from_forward(
                        fwd,
                        &second.level(start_level),
                        &first.level(start_level),
                        params.fast,
                    );
                    bwd.upsample_bilateral_output(
                        out_stage,
                        params.super_params.pel,
                        &second.level(0).y,
                        &second.level(start_level).y,
                        out,
                    );
                } else if let Some(bwd) = previous.field_ref(true, start_level as usize) {
                    bwd.upsample_bilateral_output(
                        out_stage,
                        params.super_params.pel,
                        &second.level(0).y,
                        &second.level(start_level).y,
                        out,
                    );
                }
            }
        }
        return;
    }
    let [mut out_forward, mut out_backward] = outputs;
    let last = params.stages.len() - 1;
    let mut lambda = 0;
    let mut previous = Pyramid::new(params, main, false, params.min_level());
    if let Some(session) = session.as_deref_mut()
        && last == 1
        && params.min_level() < main.levels - 1
    {
        let stage = &params.stages[1];
        let mut refined = Pyramid::new(params, stage, true, params.min_level());
        let mut settings = stage.settings(params, 0, true);
        let mut bwd_slot = if mirror { None } else { out_backward.take() };
        let mut starts = [None, None];
        let mut refine_lambda = 0;
        previous.search(
            params,
            main,
            first,
            second,
            &mut lambda,
            [None, None],
            |dir, field, now, sess| {
                if dir == 0 {
                    settings.lambda = now;
                }
                let output = if dir == 0 {
                    out_forward.as_deref_mut()
                } else {
                    bwd_slot.as_deref_mut()
                };
                let lambda_out = (dir == 0).then_some(&mut refine_lambda);
                starts[dir] = refined.refine_start(
                    dir,
                    field,
                    &settings,
                    [first, second],
                    sess,
                    output,
                    lambda_out,
                );
            },
            Some(&mut *session),
            params.min_level(),
        );
        refined.refine_finish(
            starts,
            [first, second],
            session,
            [out_forward.as_deref_mut(), bwd_slot],
            &previous,
            params.min_level() as usize,
        );
        if mirror
            && let (Some(fwd), Some(bwd)) = (out_forward.as_deref(), out_backward.as_deref_mut())
        {
            field::Field::mirror_packed_output(
                params.output(),
                params.super_params.pel,
                params.fast,
                fwd,
                bwd,
                &second.level(0),
                &first.level(0),
            );
        }
        return;
    }
    let main_outputs = if last == 0 {
        [
            out_forward.as_deref_mut(),
            if mirror { None } else { out_backward.as_deref_mut() },
        ]
    } else {
        [None, None]
    };
    previous.search(
        params,
        main,
        first,
        second,
        &mut lambda,
        main_outputs,
        |_, _, _, _| {},
        session.as_deref_mut(),
        params.min_level(),
    );
    for (index, stage) in params.stages.iter().enumerate().skip(1) {
        let mut refined = Pyramid::new(params, stage, false, params.min_level());
        let stage_outputs = if index == last {
            [
                out_forward.as_deref_mut(),
                if mirror { None } else { out_backward.as_deref_mut() },
            ]
        } else {
            [None, None]
        };
        refined.recalculate(
            params,
            stage,
            &previous,
            first,
            second,
            &mut lambda,
            stage_outputs,
            session.as_deref_mut(),
        );
        previous = refined;
    }
    if mirror
        && let (Some(fwd), Some(bwd)) = (out_forward.as_deref(), out_backward)
    {
        field::Field::mirror_packed_output(
            params.output(),
            params.super_params.pel,
            params.fast,
            fwd,
            bwd,
            &second.level(0),
            &first.level(0),
        );
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn run_chain<'o>(
    params: &AnalyseParams,
    plan: &chain::ChainPlan,
    frames: [&SuperFrameView<'_>; 2],
    keys: [(usize, i32, u32); 2],
    mut outputs: [Option<&'o mut [u32]>; 2],
) -> Option<[Option<&'o mut [u32]>; 2]> {
    let done = (|| {
        let early = if plan.lean {
            None
        } else {
            Some(cpu_search(params, plan, frames)?)
        };
        let staged = frames.map(zero_pads);
        let mut session = gpu::refine_gpu()?.session(keys[0].1.min(keys[1].1));
        for (slot, (frame, key)) in frames.into_iter().zip(keys).enumerate() {
            let level = frame.level(0);
            let planes = gpu::Planes {
                data: match &staged[slot] {
                    Some([y, u, v]) => [y, u, v],
                    None => [level.y.data(), level.u.data(), level.v.data()],
                },
                pitch: [level.y.pitch, level.u.pitch, level.v.pitch],
                base: [level.y.base(), level.u.base(), level.v.base()],
                width: [
                    level.y.width.max(0) as usize,
                    level.u.width.max(0) as usize,
                    level.v.width.max(0) as usize,
                ],
                height: [level.y.height, level.u.height, level.v.height],
                pel: level.y.pel(),
                expand: params.super_params.gpu == 1 && params.super_params.flags & 2 == 0,
                tail: [0; 3],
                pyramid: Some(plan.pyramid),
                lean: plan.lean,
                uses: key.2,
            };
            session.defer(slot, (key.0, key.1), planes);
        }
        let top = if let Some(top) = early {
            top
        } else {
            let hosts = [session.host_levels(0, plan)?, session.host_levels(1, plan)?];
            session.release();
            let frame = |index: usize| chain::HostFrame {
                pyramid: plan.pyramid,
                first: plan.host_first(),
                planes: [
                    &hosts[index][0][..],
                    &hosts[index][1][..],
                    &hosts[index][2][..],
                ],
            };
            cpu_search(params, plan, [&frame(0), &frame(1)])?
        };
        let tickets = session.chain(plan, &top)?;
        if !session.wait() {
            return None;
        }
        for (ticket, out) in tickets.into_iter().zip(outputs.iter_mut()) {
            session.take_into(ticket, out.as_deref_mut()?);
        }
        Some(())
    })();
    if done.is_some() { None } else { Some(outputs) }
}

#[cfg(not(target_arch = "wasm32"))]
fn zero_pads(frame: &SuperFrameView<'_>) -> Option<[Vec<u8>; 3]> {
    let source = frame.source?;
    let widths = [
        frame.width,
        frame.width >> frame.chroma_shift.0,
        frame.width >> frame.chroma_shift.0,
    ];
    if source
        .iter()
        .zip(widths)
        .all(|(plane, width)| plane.pitch == width as usize)
    {
        return None;
    }
    Some(std::array::from_fn(|index| {
        let (plane, width) = (source[index], widths[index] as usize);
        let mut data = plane.data.to_vec();
        for row in data.chunks_exact_mut(plane.pitch) {
            row[width..].fill(0);
        }
        data
    }))
}

#[cfg(not(target_arch = "wasm32"))]
fn cpu_search<F: planes::LevelSource>(
    params: &AnalyseParams,
    plan: &chain::ChainPlan,
    frames: [&F; 2],
) -> Option<Vec<Vec<[i32; 4]>>> {
    let main = &params.stages[0];
    let start = plan.top() as i32;
    let mut cpu = Pyramid::new(params, main, false, start);
    let mut lambda = 0;
    cpu.search(
        params,
        main,
        frames[0],
        frames[1],
        &mut lambda,
        [None, None],
        |_, _, _, _| {},
        None,
        start,
    );
    let dirs = if plan.mirror {
        1
    } else {
        1 + usize::from(plan.both)
    };
    (0..dirs)
        .map(|d| {
            cpu.fields(d == 1)[start as usize]
                .as_ref()
                .map(field::Field::gpu_out)
        })
        .collect()
}
