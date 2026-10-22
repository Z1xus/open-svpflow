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

pub(crate) use params::{AnalyseParams, SuperParams};
pub(crate) use planes::{RawPlane, SuperFrameView, plane_size};

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
    keys: Option<[(usize, i32); 2]>,
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
        #[cfg(not(target_arch = "wasm32"))]
        let plan = keys
            .filter(|_| gpu::refine_gpu().is_some_and(gpu::RefineGpu::chain_ready))
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
        if params.super_params.lean {
            cpu_search_stages(params, first, second, outputs);
            return words;
        }
        let gpu = (keys.is_some()
            && (params.stages[0].gpu_searchable(params)
                || params.stages.len() > 1
                    && params.stages[1..]
                        .iter()
                        .any(|stage| stage.gpu_refinable(params))))
        .then(gpu::refine_gpu)
        .flatten();
        let mut session = gpu.and_then(gpu::RefineGpu::session);
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
                    height: [level.y.height, level.u.height, level.v.height],
                    pel: level.y.pel(),
                    expand: params.super_params.gpu == 1 && params.super_params.flags & 2 == 0,
                    tail,
                    pyramid: None,
                };
                session.defer(slot, key, planes);
            }
        }
        search_stages(params, first, second, outputs, session.as_mut());
    }
    words
}

const STRIDE_ALIGN: usize = 32;

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
        let pitches = std::array::from_fn(|index| {
            let width = if index == 0 { sp.width } else { sp.width >> sx };
            (width as usize).next_multiple_of(STRIDE_ALIGN)
        });
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

fn search_stages(
    params: &AnalyseParams,
    first: &SuperFrameView<'_>,
    second: &SuperFrameView<'_>,
    outputs: [Option<&mut [u32]>; 2],
    mut session: Option<&mut gpu::Session<'_, '_>>,
) {
    let mut outputs = Some(outputs);
    let last = params.stages.len() - 1;
    let mut take = |index: usize| {
        if index == last {
            outputs.take().unwrap_or_default()
        } else {
            [None, None]
        }
    };
    let mut lambda = 0;
    let main = &params.stages[0];
    let mut previous = Pyramid::new(params, main, false, params.min_level());
    if let Some(session) = session.as_deref_mut()
        && last == 1
        && params.min_level() < main.levels - 1
    {
        let stage = &params.stages[1];
        let mut refined = Pyramid::new(params, stage, true, params.min_level());
        let mut settings = stage.settings(params, 0, true);
        let [mut out_forward, mut out_backward] = take(1);
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
                    out_backward.as_deref_mut()
                };
                let lambda_out = (dir == 0).then_some(&mut refine_lambda);
                starts[dir] = refined.refine_start(
                    dir,
                    field,
                    &settings,
                    [first, second],
                    sess.expect("session"),
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
            [out_forward, out_backward],
            &previous,
            params.min_level() as usize,
        );
        return;
    }
    previous.search(
        params,
        main,
        first,
        second,
        &mut lambda,
        take(0),
        |_, _, _, _| {},
        session.as_deref_mut(),
        params.min_level(),
    );
    for (index, stage) in params.stages.iter().enumerate().skip(1) {
        let mut refined = Pyramid::new(params, stage, false, params.min_level());
        refined.recalculate(
            params,
            stage,
            &previous,
            first,
            second,
            &mut lambda,
            take(index),
            session.as_deref_mut(),
        );
        previous = refined;
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn run_chain<'o>(
    params: &AnalyseParams,
    plan: &chain::ChainPlan,
    frames: [&SuperFrameView<'_>; 2],
    keys: [(usize, i32); 2],
    mut outputs: [Option<&'o mut [u32]>; 2],
) -> Option<[Option<&'o mut [u32]>; 2]> {
    let done = (|| {
        let mut session = gpu::refine_gpu()?.session()?;
        for (slot, (frame, key)) in frames.into_iter().zip(keys).enumerate() {
            let level = frame.level(0);
            let planes = gpu::Planes {
                data: [level.y.data(), level.u.data(), level.v.data()],
                pitch: [level.y.pitch, level.u.pitch, level.v.pitch],
                base: [level.y.base(), level.u.base(), level.v.base()],
                height: [level.y.height, level.u.height, level.v.height],
                pel: level.y.pel(),
                expand: params.super_params.gpu == 1 && params.super_params.flags & 2 == 0,
                tail: [0; 3],
                pyramid: Some(plan.pyramid),
            };
            session.defer(slot, key, planes);
        }
        let top = if plan.lean {
            let hosts = [session.host_levels(0, plan)?, session.host_levels(1, plan)?];
            let frame = |index: usize| chain::HostFrame {
                pyramid: plan.pyramid,
                start: plan.top() as i32,
                planes: [
                    &hosts[index][0][..],
                    &hosts[index][1][..],
                    &hosts[index][2][..],
                ],
            };
            cpu_search(params, plan, [&frame(0), &frame(1)])
        } else {
            cpu_search(params, plan, frames)
        };
        let tickets = session.chain(plan, &top)?;
        if !session.wait() {
            return None;
        }
        for (ticket, out) in tickets.into_iter().zip(outputs.iter_mut()) {
            session.take_words(ticket, out.as_deref_mut()?);
        }
        Some(())
    })();
    if done.is_some() { None } else { Some(outputs) }
}

#[cfg(not(target_arch = "wasm32"))]
fn cpu_search<F: planes::LevelSource>(
    params: &AnalyseParams,
    plan: &chain::ChainPlan,
    frames: [&F; 2],
) -> Vec<Vec<[i32; 4]>> {
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
    (0..=usize::from(plan.both))
        .map(|d| {
            cpu.fields(d == 1)[start as usize]
                .as_ref()
                .expect("start level")
                .gpu_out()
        })
        .collect()
}
