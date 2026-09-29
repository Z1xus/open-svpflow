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
) -> Vec<u8> {
    let count = params.output_count();
    let mut regions = [vec![0u32; 2 * count], vec![0u32; 2 * count]];
    let backward_only = params.vectors == 2;
    let (first, second, keys) = if backward_only {
        (current, next, keys)
    } else {
        (next, current, keys.map(|[a, b]| [b, a]))
    };
    {
        let [region0, region1] = &mut regions;
        let outputs = if backward_only {
            [Some(&mut region1[..]), None]
        } else {
            [
                Some(&mut region0[..]),
                (params.vectors == 3).then_some(&mut region1[..]),
            ]
        };
        let gpu = (keys.is_some()
            && params.stages.len() > 1
            && params.stages[1..]
                .iter()
                .any(|stage| stage.gpu_refinable(params)))
        .then(gpu::refine_gpu)
        .flatten();
        let mut session = gpu.and_then(gpu::RefineGpu::session);
        if let Some(session) = session.as_mut() {
            let keys = keys.unwrap_or_default();
            for (slot, (frame, key)) in [first, second].into_iter().zip(keys).enumerate() {
                let level = frame.level(0);
                let planes = gpu::Planes {
                    data: [level.y.data(), level.u.data(), level.v.data()],
                    pitch: [level.y.pitch, level.u.pitch, level.v.pitch],
                    base: [level.y.base(), level.u.base(), level.v.base()],
                    height: [level.y.height, level.u.height, level.v.height],
                    pel: level.y.pel(),
                    expand: params.super_params.gpu == 1 && params.super_params.flags & 2 == 0,
                };
                session.upload(slot, key, &planes);
            }
        }
        search_stages(params, first, second, outputs, session.as_mut());
    }
    let written = [!backward_only, true];
    pack_frame(params, &regions, written)
}

fn search_stages(
    params: &AnalyseParams,
    first: &SuperFrameView<'_>,
    second: &SuperFrameView<'_>,
    outputs: [Option<&mut [u32]>; 2],
    mut session: Option<&mut gpu::Session<'_>>,
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
    let mut previous = Pyramid::new(params, main);
    if let Some(session) = session.as_deref_mut()
        && last == 1
        && params.min_level() < main.levels - 1
    {
        let stage = &params.stages[1];
        let mut refined = Pyramid::new(params, stage);
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
            |dir, field, now| {
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
                    session,
                    output,
                    lambda_out,
                );
            },
        );
        refined.refine_finish(
            starts,
            [first, second],
            session,
            [out_forward, out_backward],
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
        |_, _, _| {},
    );
    for (index, stage) in params.stages.iter().enumerate().skip(1) {
        let mut refined = Pyramid::new(params, stage);
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

fn pack_frame(params: &AnalyseParams, regions: &[Vec<u32>; 2], written: [bool; 2]) -> Vec<u8> {
    let mut out = Vec::with_capacity(frame_len(params));
    out.extend_from_slice(&16i32.to_le_bytes());
    for value in analysis_header(params) {
        out.extend_from_slice(&value.to_le_bytes());
    }
    let marker = (2 * params.written_count() + 1) as i32;
    for (region, written) in regions.iter().zip(written) {
        out.extend_from_slice(&if written { marker } else { 0 }.to_le_bytes());
        for value in region {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
    out
}
