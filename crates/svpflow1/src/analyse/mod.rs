mod field;
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
) -> Vec<u8> {
    let count = params.output_count();
    let mut regions = [vec![0u32; 2 * count], vec![0u32; 2 * count]];
    let backward_only = params.vectors == 2;
    let (first, second) = if backward_only {
        (current, next)
    } else {
        (next, current)
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
        search_stages(params, first, second, outputs);
    }
    let written = [!backward_only, true];
    pack_frame(params, &regions, written)
}

fn search_stages(
    params: &AnalyseParams,
    first: &SuperFrameView<'_>,
    second: &SuperFrameView<'_>,
    outputs: [Option<&mut [u32]>; 2],
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
    previous.search(params, main, first, second, &mut lambda, take(0));
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
