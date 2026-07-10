use crate::analyse_opts::AnalyseOpts;
use crate::super_opts::reduce_dim;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use core::arch::wasm32::{
    u8x16_sub_sat, u16x8_extadd_pairwise_u8x16, u32x4, u32x4_add, u32x4_extadd_pairwise_u16x8,
    u32x4_extract_lane, v128, v128_load, v128_load32_zero, v128_load64_zero, v128_or,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, _mm_add_epi16, _mm_add_epi32, _mm_add_epi64, _mm_cvtsi32_si128, _mm_cvtsi128_si32,
    _mm_loadl_epi64, _mm_loadu_si128, _mm_madd_epi16, _mm_max_epi16, _mm_sad_epu8, _mm_set1_epi16,
    _mm_setzero_si128, _mm_srli_si128, _mm_sub_epi16, _mm_unpackhi_epi32, _mm_unpacklo_epi8,
    _mm_unpacklo_epi16, _mm_unpacklo_epi32,
};
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Vec8 {
    pub(crate) dx: i16,
    pub(crate) dy: i16,
    pub(crate) score: u32,
    pub(crate) luma: u8,
}

#[allow(dead_code)]
pub(crate) struct SuperPlanes<'a> {
    pub(crate) y: &'a [u8],
    pub(crate) y_stride: usize,
    pub(crate) u: &'a [u8],
    pub(crate) u_stride: usize,
    pub(crate) v: &'a [u8],
    pub(crate) v_stride: usize,
    pub(crate) luma_w: usize,
    pub(crate) luma_h: usize,
    pub(crate) pel: i32,
    pub(crate) levels: i32,
    pub(crate) full: bool,
}

impl SuperPlanes<'_> {
    fn level_y_offset(&self, level: i32) -> usize {
        let mut y = 0usize;
        let pel = self.pel as usize;
        for lv in 0..level {
            let h = reduce_dim(self.luma_h as i32, lv) as usize;
            let sub = if lv == 0 && self.full { pel * pel } else { 1 };
            y += h * sub;
        }
        y
    }

    fn level_size(&self, level: i32) -> (usize, usize) {
        (
            reduce_dim(self.luma_w as i32, level) as usize,
            reduce_dim(self.luma_h as i32, level) as usize,
        )
    }
}

pub(crate) fn analyse_pair(
    cur: &SuperPlanes<'_>,
    ref_super: &SuperPlanes<'_>,
    opts: &AnalyseOpts,
) -> (Option<Vec<Vec8>>, Option<Vec<Vec8>>) {
    let do_bwd = opts.vectors & 1 != 0;
    let do_fwd = opts.vectors & 2 != 0;

    if do_bwd && do_fwd {
        let (bwd, fwd) = exact_search_hierarchy_bidir(cur, ref_super, opts);
        return (Some(bwd), Some(fwd));
    }

    let fwd = if do_fwd {
        Some(exact_search_hierarchy(cur, ref_super, opts))
    } else {
        None
    };
    let bwd = if do_bwd {
        Some(exact_search_hierarchy(ref_super, cur, opts))
    } else {
        None
    };
    (bwd, fwd)
}

fn exact_search_hierarchy_bidir(
    cur: &SuperPlanes<'_>,
    refp: &SuperPlanes<'_>,
    opts: &AnalyseOpts,
) -> (Vec<Vec8>, Vec<Vec8>) {
    let mut fwd = Vec::new();
    let mut bwd = Vec::new();
    let mut field_gw = 0;
    let mut field_gh = 0;
    let mut global_fwd = (0, 0);
    let mut global_bwd = (0, 0);
    let mut order = Vec::new();

    for level in (0..opts.levels.max(1)).rev() {
        let (gw, gh) = level_grid(
            opts,
            level,
            opts.block_w,
            opts.block_h,
            opts.overlap_x,
            opts.overlap_y,
        );
        let pel = if level == 0 { opts.pel.max(1) } else { 1 };
        order = if order.is_empty() {
            initial_search_order(gw, gh)
        } else {
            refine_search_order(&order, gw, gh)
        };
        let mut next_fwd = if fwd.is_empty() {
            vec![unsearched_vector(); (gw * gh) as usize]
        } else {
            exact_interpolate_level(
                &fwd,
                field_gw,
                field_gh,
                gw,
                gh,
                opts.block_w,
                opts.block_h,
                opts.overlap_x,
                opts.overlap_y,
                pel,
            )
        };
        let mut next_bwd = if bwd.is_empty() {
            vec![unsearched_vector(); (gw * gh) as usize]
        } else {
            exact_interpolate_level(
                &bwd,
                field_gw,
                field_gh,
                gw,
                gh,
                opts.block_w,
                opts.block_h,
                opts.overlap_x,
                opts.overlap_y,
                pel,
            )
        };
        rayon::join(
            || {
                exact_search_level(
                    cur,
                    refp,
                    opts,
                    level,
                    global_fwd,
                    &mut next_fwd,
                    gw,
                    gh,
                    &order,
                );
            },
            || {
                exact_search_level(
                    refp,
                    cur,
                    opts,
                    level,
                    global_bwd,
                    &mut next_bwd,
                    gw,
                    gh,
                    &order,
                );
            },
        );
        if level > 0 {
            (global_fwd, global_bwd) = reconcile_bidir_gmv(
                exact_global_doubled(&next_fwd),
                exact_global_doubled(&next_bwd),
            );
        }
        fwd = next_fwd;
        bwd = next_bwd;
        field_gw = gw;
        field_gh = gh;
    }

    let fwd = exact_recalculate(cur, refp, opts, fwd, field_gw, field_gh);
    let bwd = exact_recalculate(refp, cur, opts, bwd, field_gw, field_gh);
    let (bw, _, _, _) = opts.output_block();
    let mut fwd = fwd;
    let mut bwd = bwd;
    rescale_scores(&mut fwd, opts.width, opts.height, bw);
    rescale_scores(&mut bwd, opts.width, opts.height, bw);
    (bwd, fwd)
}

#[derive(Clone, Copy)]
struct SearchBest {
    mv: (i32, i32),
    sad: u32,
    cost: i64,
}

fn exact_search_hierarchy(
    src: &SuperPlanes<'_>,
    refp: &SuperPlanes<'_>,
    opts: &AnalyseOpts,
) -> Vec<Vec8> {
    let mut field = Vec::new();
    let mut field_gw = 0;
    let mut field_gh = 0;
    let mut global = (0, 0);
    let mut order = Vec::new();

    for level in (0..opts.levels.max(1)).rev() {
        let (gw, gh) = level_grid(
            opts,
            level,
            opts.block_w,
            opts.block_h,
            opts.overlap_x,
            opts.overlap_y,
        );
        let pel = if level == 0 { opts.pel.max(1) } else { 1 };
        order = if order.is_empty() {
            initial_search_order(gw, gh)
        } else {
            refine_search_order(&order, gw, gh)
        };
        let mut next = if field.is_empty() {
            vec![unsearched_vector(); (gw * gh) as usize]
        } else {
            exact_interpolate_level(
                &field,
                field_gw,
                field_gh,
                gw,
                gh,
                opts.block_w,
                opts.block_h,
                opts.overlap_x,
                opts.overlap_y,
                pel,
            )
        };
        exact_search_level(src, refp, opts, level, global, &mut next, gw, gh, &order);
        if level > 0 {
            global = exact_global_doubled(&next);
        }
        field = next;
        field_gw = gw;
        field_gh = gh;
    }

    let mut field = exact_recalculate(src, refp, opts, field, field_gw, field_gh);
    let (bw, _, _, _) = opts.output_block();
    rescale_scores(&mut field, opts.width, opts.height, bw);
    field
}

fn unsearched_vector() -> Vec8 {
    Vec8 {
        score: u32::MAX,
        ..Vec8::default()
    }
}

fn initial_search_order(gw: i32, gh: i32) -> Vec<(i32, i32)> {
    let mut order = Vec::with_capacity((gw * gh).max(0) as usize);
    for y in 1..gh - 1 {
        if y & 1 == 0 {
            order.extend((1..gw - 1).map(|x| (x, y)));
        } else {
            order.extend((1..gw - 1).rev().map(|x| (x, y)));
        }
    }
    for x in 0..gw {
        order.push((x, 0));
        if gh > 1 {
            order.push((x, gh - 1));
        }
    }
    for y in 1..gh - 1 {
        order.push((0, y));
        if gw > 1 {
            order.push((gw - 1, y));
        }
    }
    order
}

fn refine_search_order(coarse: &[(i32, i32)], gw: i32, gh: i32) -> Vec<(i32, i32)> {
    let mut interior = Vec::with_capacity((gw * gh).max(0) as usize);
    let mut boundary = Vec::new();
    let mut seen = vec![false; (gw * gh).max(0) as usize];
    for &(x, y) in coarse {
        for child in [
            (2 * x, 2 * y),
            (2 * x + 1, 2 * y),
            (2 * x + 1, 2 * y + 1),
            (2 * x, 2 * y + 1),
        ] {
            let (cx, cy) = child;
            if cx < 0 || cy < 0 || cx >= gw || cy >= gh {
                continue;
            }
            let idx = (cy * gw + cx) as usize;
            if seen[idx] {
                continue;
            }
            seen[idx] = true;
            let on_main_path = match (cx & 1, cy & 1) {
                (0, 0) => x > 0 && y > 0,
                (1, 0) => cx < gw - 1 && y > 0,
                (1, 1) => cx < gw - 1 && cy < gh - 1,
                _ => x > 0 && cy < gh - 1,
            };
            if on_main_path {
                interior.push(child);
            } else {
                boundary.push(child);
            }
        }
    }
    interior.extend(boundary);
    for child in initial_search_order(gw, gh) {
        let idx = (child.1 * gw + child.0) as usize;
        if !seen[idx] {
            seen[idx] = true;
            interior.push(child);
        }
    }
    interior
}

fn exact_interpolate_level(
    coarse: &[Vec8],
    cgw: i32,
    cgh: i32,
    gw: i32,
    gh: i32,
    bw: i32,
    bh: i32,
    ox: i32,
    oy: i32,
    pel: i32,
) -> Vec<Vec8> {
    let log_pel = pel.trailing_zeros() as i32;
    let norm = (3 - log_pel).max(0);
    let mul = (log_pel - 3).max(0);
    let normov = ((bw - ox) * (bh - oy)).max(1);
    let odd_x = 3 * bw - 2 * ox;
    let even_x = 3 * bw - 4 * ox;
    let odd_y = 3 * bh - 2 * oy;
    let even_y = 3 * bh - 4 * oy;
    let mut out = vec![Vec8::default(); (gw * gh) as usize];

    let get = |x: i32, y: i32| coarse[(y * cgw + x) as usize];
    for y in 0..gh {
        for x in 0..gw {
            let ix = x.min(2 * cgw - 1);
            let iy = y.min(2 * cgh - 1);
            let sx = 2 * (ix & 1) - 1;
            let sy = 2 * (iy & 1) - 1;
            let cx = ix / 2;
            let cy = iy / 2;
            let edge_x = ix == 0 || ix >= 2 * cgw - 1;
            let edge_y = iy == 0 || iy >= 2 * cgh - 1;
            let p = get(cx, cy);
            let mut qx = if edge_x { p } else { get(cx + sx, cy) };
            let mut qy = if edge_y { p } else { get(cx, cy + sy) };
            if edge_y && !edge_x {
                qy = qx;
                qx = p;
            }
            let qxy = if edge_x && edge_y {
                p
            } else if edge_x {
                qy
            } else if edge_y {
                qy
            } else {
                get(cx + sx, cy + sy)
            };

            let (mut vx, mut vy, sad) = if ox == 0 && oy == 0 {
                (
                    9 * i32::from(p.dx)
                        + 3 * i32::from(qx.dx)
                        + 3 * i32::from(qy.dx)
                        + i32::from(qxy.dx),
                    9 * i32::from(p.dy)
                        + 3 * i32::from(qx.dy)
                        + 3 * i32::from(qy.dy)
                        + i32::from(qxy.dy),
                    (9 * u64::from(p.score)
                        + 3 * u64::from(qx.score)
                        + 3 * u64::from(qy.score)
                        + u64::from(qxy.score)
                        + 8)
                        >> 4,
                )
            } else {
                let ax1 = if sx > 0 { odd_x } else { even_x };
                let ax2 = 4 * (bw - ox) - ax1;
                let ay1 = if sy > 0 { odd_y } else { even_y };
                let ay2 = 4 * (bh - oy) - ay1;
                let w11 = i64::from(ax1 * ay1);
                let w21 = i64::from(ax2 * ay1);
                let w12 = i64::from(ax1 * ay2);
                let w22 = i64::from(ax2 * ay2);
                let dx = (w11 * i64::from(p.dx)
                    + w21 * i64::from(qx.dx)
                    + w12 * i64::from(qy.dx)
                    + w22 * i64::from(qxy.dx))
                    / i64::from(normov);
                let dy = (w11 * i64::from(p.dy)
                    + w21 * i64::from(qx.dy)
                    + w12 * i64::from(qy.dy)
                    + w22 * i64::from(qxy.dy))
                    / i64::from(normov);
                let score = (w11 * i64::from(p.score)
                    + w21 * i64::from(qx.score)
                    + w12 * i64::from(qy.score)
                    + w22 * i64::from(qxy.score))
                    / i64::from(normov);
                (dx as i32, dy as i32, (score.max(0) as u64) >> 4)
            };
            vx = (vx >> norm) << mul;
            vy = (vy >> norm) << mul;
            out[(y * gw + x) as usize] = Vec8 {
                dx: vx.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                dy: vy.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                score: sad.min(u64::from(u32::MAX)) as u32,
                luma: p.luma,
            };
        }
    }
    out
}

fn exact_search_level(
    src: &SuperPlanes<'_>,
    refp: &SuperPlanes<'_>,
    opts: &AnalyseOpts,
    level: i32,
    global: (i32, i32),
    field: &mut [Vec8],
    gw: i32,
    gh: i32,
    order: &[(i32, i32)],
) {
    let bw = opts.block_w;
    let bh = opts.block_h;
    let step_x = (bw - opts.overlap_x).max(1);
    let step_y = (bh - opts.overlap_y).max(1);
    let pel = if level == 0 { opts.pel.max(1) } else { 1 };
    let radius = search_radius(opts, level == 0, level, opts.levels.max(1));
    let search_type = if level == 0 {
        opts.search_type
    } else {
        opts.coarse_search_type
    };
    let satd = if level == 0 {
        opts.fine_satd
    } else {
        opts.coarse_satd
    };
    let try_many = level > 0 && opts.coarse_trymany;
    let mut bad_count = 0i32;
    let mut processed = vec![false; ((gw + 1) * (gh + 1)).max(0) as usize];

    for &(bx, by) in order {
        let idx = (by * gw + bx) as usize;
        let px = bx * step_x;
        let py = by * step_y;
        let bounds = mv_bounds(
            px,
            py,
            bw,
            bh,
            src.level_size(level).0 as i32,
            src.level_size(level).1 as i32,
            pel,
        );
        let zero = exact_clip((0, 0), bounds);
        let interpolated = field[idx];
        let predictor = interpolated;
        let pred_mv = exact_clip((predictor.dx.into(), predictor.dy.into()), bounds);
        let lambda_base = exact_lambda(opts, level, pel);
        let lambda = adapt_lambda(lambda_base, opts.lsad, predictor.score);
        let mut seeds = [(0, 0); 8];
        let mut seed_count = 0;
        for ny in (by - 1).max(0)..=(by + 1).min(gh - 1) {
            for nx in (bx - 1).max(0)..=(bx + 1).min(gw - 1) {
                if processed[(ny * (gw + 1) + nx) as usize] {
                    let neighbour = field[(ny * gw + nx) as usize];
                    seeds[seed_count] = (neighbour.dx.into(), neighbour.dy.into());
                    seed_count += 1;
                }
            }
        }
        let mut best = exact_search_block(
            src,
            refp,
            level,
            px,
            py,
            bw,
            bh,
            pel,
            satd,
            pred_mv,
            predictor.score != u32::MAX,
            zero,
            exact_clip(global, bounds),
            &seeds[..seed_count],
            bounds,
            lambda,
            opts,
            search_type,
            radius,
            try_many,
        );
        let found = best.sad;
        let bad_limit = opts
            .coarse_bad_sad
            .saturating_add(opts.coarse_bad_sad.saturating_mul(bad_count) / 16);
        if level > 0 && idx > 1 && opts.coarse_bad_sad > 0 && best.sad as i32 > bad_limit {
            bad_count += 1;
            if opts.coarse_bad_range < 0 {
                let max_r = -opts.coarse_bad_range * pel;
                let mut r = pel;
                while r < max_r {
                    exact_expanding(
                        src,
                        refp,
                        level,
                        px,
                        py,
                        bw,
                        bh,
                        pel,
                        satd,
                        pred_mv,
                        bounds,
                        lambda,
                        opts.pnew,
                        (0, 0),
                        r,
                        pel,
                        &mut best,
                    );
                    if best.sad < found / 4 {
                        break;
                    }
                    r += pel;
                }
            } else if opts.coarse_bad_range > 0 {
                exact_umh(
                    src,
                    refp,
                    level,
                    px,
                    py,
                    bw,
                    bh,
                    pel,
                    satd,
                    pred_mv,
                    bounds,
                    lambda,
                    opts.pnew,
                    (0, 0),
                    opts.coarse_bad_range * pel,
                    &mut best,
                );
            }
            let center = best.mv;
            exact_expanding(
                src, refp, level, px, py, bw, bh, pel, satd, pred_mv, bounds, lambda, opts.pnew,
                center, 1, 1, &mut best,
            );
        }
        field[idx] = Vec8 {
            dx: best.mv.0 as i16,
            dy: best.mv.1 as i16,
            score: best.sad,
            luma: block_luma_dc(src, level, px, py, bw, bh),
        };
        processed[(by * (gw + 1) + bx) as usize] = true;
    }
}

#[allow(clippy::too_many_arguments)]
fn exact_search_block(
    src: &SuperPlanes<'_>,
    refp: &SuperPlanes<'_>,
    level: i32,
    px: i32,
    py: i32,
    bw: i32,
    bh: i32,
    pel: i32,
    satd: bool,
    predictor: (i32, i32),
    field_valid: bool,
    zero: (i32, i32),
    global: (i32, i32),
    neighbours: &[(i32, i32)],
    bounds: (i32, i32, i32, i32),
    lambda: i32,
    opts: &AnalyseOpts,
    search_type: i32,
    radius: i32,
    try_many: bool,
) -> SearchBest {
    let special = |mv: (i32, i32), penalty: i32| {
        let mv = exact_clip(mv, bounds);
        let sad = exact_sad(src, refp, level, px, py, bw, bh, mv, satd, pel);
        SearchBest {
            mv,
            sad,
            cost: i64::from(sad) + ((i64::from(penalty) * i64::from(sad)) >> 8),
        }
    };
    let field = field_valid.then(|| special(predictor, 0));
    let zero_best = special(zero, opts.pzero);
    let global_best = special(global, opts.pglobal);

    if try_many {
        let mut results = [zero_best; 11];
        let mut result_count = 0;
        for (mut start, penalty, enabled) in [
            (zero_best, opts.pzero, true),
            (global_best, opts.pglobal, true),
            (field.unwrap_or(zero_best), 0, field.is_some()),
        ] {
            if !enabled {
                continue;
            }
            exact_refine_search(
                src,
                refp,
                level,
                px,
                py,
                bw,
                bh,
                pel,
                satd,
                predictor,
                bounds,
                lambda,
                opts.pnew,
                search_type,
                radius,
                &mut start,
            );
            start.cost = i64::from(start.sad) + ((i64::from(penalty) * i64::from(start.sad)) >> 8);
            results[result_count] = start;
            result_count += 1;
        }
        for &mv in neighbours {
            let mv = exact_clip(mv, bounds);
            let sad = exact_sad(src, refp, level, px, py, bw, bh, mv, satd, pel);
            let mut start = SearchBest {
                mv,
                sad,
                cost: i64::from(sad) + ((i64::from(opts.pnbour) * i64::from(sad)) >> 8),
            };
            exact_refine_search(
                src,
                refp,
                level,
                px,
                py,
                bw,
                bh,
                pel,
                satd,
                predictor,
                bounds,
                lambda,
                opts.pnew,
                search_type,
                radius,
                &mut start,
            );
            start.cost =
                i64::from(start.sad) + ((i64::from(opts.pnbour) * i64::from(start.sad)) >> 8);
            results[result_count] = start;
            result_count += 1;
        }
        return results[..result_count]
            .iter()
            .copied()
            .min_by_key(|value| value.cost)
            .unwrap_or(zero_best);
    }

    let mut best = zero_best;
    if global_best.cost < best.cost {
        best = global_best;
    }
    if let Some(field) = field
        && field.cost < best.cost
    {
        best = field;
    }
    for &mv in neighbours {
        let mv = exact_clip(mv, bounds);
        let sad = exact_sad(src, refp, level, px, py, bw, bh, mv, satd, pel);
        let cost = i64::from(sad) + ((i64::from(opts.pnbour) * i64::from(sad)) >> 8);
        if cost < best.cost {
            best = SearchBest { mv, sad, cost };
        }
    }
    exact_refine_search(
        src,
        refp,
        level,
        px,
        py,
        bw,
        bh,
        pel,
        satd,
        predictor,
        bounds,
        lambda,
        opts.pnew,
        search_type,
        radius,
        &mut best,
    );
    best
}

#[allow(clippy::too_many_arguments)]
fn exact_refine_search(
    src: &SuperPlanes<'_>,
    refp: &SuperPlanes<'_>,
    level: i32,
    px: i32,
    py: i32,
    bw: i32,
    bh: i32,
    pel: i32,
    satd: bool,
    predictor: (i32, i32),
    bounds: (i32, i32, i32, i32),
    lambda: i32,
    pnew: i32,
    search_type: i32,
    radius: i32,
    best: &mut SearchBest,
) {
    match search_type {
        2 => exact_hex2(
            src, refp, level, px, py, bw, bh, pel, satd, predictor, bounds, lambda, pnew, radius,
            best,
        ),
        3 => exact_umh(
            src, refp, level, px, py, bw, bh, pel, satd, predictor, bounds, lambda, pnew, best.mv,
            radius, best,
        ),
        4 => {
            let center = best.mv;
            for r in 1..=radius {
                exact_expanding(
                    src, refp, level, px, py, bw, bh, pel, satd, predictor, bounds, lambda, pnew,
                    center, r, 1, best,
                );
            }
        }
        _ => {}
    }
}

#[allow(clippy::too_many_arguments)]
fn exact_expanding(
    src: &SuperPlanes<'_>,
    refp: &SuperPlanes<'_>,
    level: i32,
    px: i32,
    py: i32,
    bw: i32,
    bh: i32,
    pel: i32,
    satd: bool,
    predictor: (i32, i32),
    bounds: (i32, i32, i32, i32),
    lambda: i32,
    pnew: i32,
    center: (i32, i32),
    radius: i32,
    step: i32,
    best: &mut SearchBest,
) {
    let mut i = -radius + step;
    while i < radius {
        for mv in [
            (center.0 + i, center.1 - radius),
            (center.0 + i, center.1 + radius),
        ] {
            exact_check(
                src, refp, level, px, py, bw, bh, pel, satd, predictor, bounds, lambda, pnew, mv,
                best,
            );
        }
        i += step;
    }
    let mut j = -radius + step;
    while j < radius {
        for mv in [
            (center.0 - radius, center.1 + j),
            (center.0 + radius, center.1 + j),
        ] {
            exact_check(
                src, refp, level, px, py, bw, bh, pel, satd, predictor, bounds, lambda, pnew, mv,
                best,
            );
        }
        j += step;
    }
    for mv in [
        (center.0 - radius, center.1 - radius),
        (center.0 - radius, center.1 + radius),
        (center.0 + radius, center.1 - radius),
        (center.0 + radius, center.1 + radius),
    ] {
        exact_check(
            src, refp, level, px, py, bw, bh, pel, satd, predictor, bounds, lambda, pnew, mv, best,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn exact_hex2(
    src: &SuperPlanes<'_>,
    refp: &SuperPlanes<'_>,
    level: i32,
    px: i32,
    py: i32,
    bw: i32,
    bh: i32,
    pel: i32,
    satd: bool,
    predictor: (i32, i32),
    bounds: (i32, i32, i32, i32),
    lambda: i32,
    pnew: i32,
    radius: i32,
    best: &mut SearchBest,
) {
    let mut center = best.mv;
    if radius > 1 {
        let hex = [(-2, 0), (-1, 2), (1, 2), (2, 0), (1, -2), (-1, -2)];
        for _ in 0..(radius / 2).max(1) {
            let before = best.mv;
            for delta in hex {
                exact_check(
                    src,
                    refp,
                    level,
                    px,
                    py,
                    bw,
                    bh,
                    pel,
                    satd,
                    predictor,
                    bounds,
                    lambda,
                    pnew,
                    (center.0 + delta.0, center.1 + delta.1),
                    best,
                );
            }
            if best.mv == before {
                break;
            }
            center = best.mv;
        }
    }
    exact_expanding(
        src, refp, level, px, py, bw, bh, pel, satd, predictor, bounds, lambda, pnew, center, 1, 1,
        best,
    );
}

#[allow(clippy::too_many_arguments)]
fn exact_umh(
    src: &SuperPlanes<'_>,
    refp: &SuperPlanes<'_>,
    level: i32,
    px: i32,
    py: i32,
    bw: i32,
    bh: i32,
    pel: i32,
    satd: bool,
    predictor: (i32, i32),
    bounds: (i32, i32, i32, i32),
    lambda: i32,
    pnew: i32,
    center: (i32, i32),
    radius: i32,
    best: &mut SearchBest,
) {
    let mut d = 1;
    while d < radius {
        for mv in [
            (center.0 - d, center.1),
            (center.0 + d, center.1),
            (center.0, center.1 - d),
            (center.0, center.1 + d),
        ] {
            exact_check(
                src, refp, level, px, py, bw, bh, pel, satd, predictor, bounds, lambda, pnew, mv,
                best,
            );
        }
        d += 2;
    }
    let hex4 = [
        (-4, 2),
        (-4, 1),
        (-4, 0),
        (-4, -1),
        (-4, -2),
        (4, -2),
        (4, -1),
        (4, 0),
        (4, 1),
        (4, 2),
        (2, 3),
        (0, 4),
        (-2, 3),
        (-2, -3),
        (0, -4),
        (2, -3),
    ];
    for scale in 1..=radius / 4 {
        for delta in hex4 {
            exact_check(
                src,
                refp,
                level,
                px,
                py,
                bw,
                bh,
                pel,
                satd,
                predictor,
                bounds,
                lambda,
                pnew,
                (center.0 + delta.0 * scale, center.1 + delta.1 * scale),
                best,
            );
        }
    }
    exact_hex2(
        src, refp, level, px, py, bw, bh, pel, satd, predictor, bounds, lambda, pnew, radius, best,
    );
}

#[allow(clippy::too_many_arguments)]
fn exact_check(
    src: &SuperPlanes<'_>,
    refp: &SuperPlanes<'_>,
    level: i32,
    px: i32,
    py: i32,
    bw: i32,
    bh: i32,
    pel: i32,
    satd: bool,
    predictor: (i32, i32),
    bounds: (i32, i32, i32, i32),
    lambda: i32,
    pnew: i32,
    mv: (i32, i32),
    best: &mut SearchBest,
) {
    if !exact_ok(mv, bounds) {
        return;
    }
    let motion = exact_motion_penalty(lambda, predictor, mv);
    if motion >= best.cost {
        return;
    }
    let Some(sad) = exact_sad_if_better(
        src, refp, level, px, py, bw, bh, mv, satd, pel, motion, pnew, best.cost,
    ) else {
        return;
    };
    let cost = motion + i64::from(sad) + ((i64::from(pnew) * i64::from(sad)) >> 8);
    if cost < best.cost {
        *best = SearchBest { mv, sad, cost };
    }
}

#[allow(clippy::too_many_arguments)]
fn exact_sad_if_better(
    src: &SuperPlanes<'_>,
    refp: &SuperPlanes<'_>,
    level: i32,
    px: i32,
    py: i32,
    bw: i32,
    bh: i32,
    mv: (i32, i32),
    satd: bool,
    pel: i32,
    motion: i64,
    pnew: i32,
    best_cost: i64,
) -> Option<u32> {
    let pel = pel.max(1);
    let (lw, lh) = src.level_size(level);
    let shifted_edge = level == 0
        && pel >= 2
        && (mv.0 < -pel * px || mv.1 < -pel * py)
        && edge_shift_origins(px, py, mv.0, mv.1, bw, bh, lw as i32, lh as i32, pel).is_some();
    if shifted_edge || pnew < 0 {
        return Some(exact_sad(src, refp, level, px, py, bw, bh, mv, satd, pel));
    }

    let bw_u = bw as usize;
    let bh_u = bh as usize;
    let (luma, _) = block_cost_luma_interior(
        src, refp, level, px, py, bw_u, bh_u, mv.0, mv.1, satd, pel, false,
    );
    let lower_bound = motion + i64::from(luma) + ((i64::from(pnew) * i64::from(luma)) >> 8);
    if lower_bound >= best_cost {
        return None;
    }

    let chroma = chroma_sad_x4(
        src,
        refp,
        level,
        pel,
        mv.0,
        mv.1,
        px as usize,
        py as usize,
        bw_u,
        bh_u,
        lw,
        lh,
    );
    Some(luma.saturating_add(chroma))
}

fn exact_sad(
    src: &SuperPlanes<'_>,
    refp: &SuperPlanes<'_>,
    level: i32,
    px: i32,
    py: i32,
    bw: i32,
    bh: i32,
    mv: (i32, i32),
    satd: bool,
    pel: i32,
) -> u32 {
    let (luma, chroma) = block_cost_lc(src, refp, level, px, py, bw, bh, mv.0, mv.1, satd, pel);
    luma.saturating_add(chroma)
}

fn exact_motion_penalty(lambda: i32, predictor: (i32, i32), mv: (i32, i32)) -> i64 {
    let dx = i64::from(predictor.0 - mv.0);
    let dy = i64::from(predictor.1 - mv.1);
    (i64::from(lambda) * (dx * dx + dy * dy)) >> 8
}

fn exact_clip(mv: (i32, i32), bounds: (i32, i32, i32, i32)) -> (i32, i32) {
    let max_x = (bounds.2 - 1).max(bounds.0);
    let max_y = (bounds.3 - 1).max(bounds.1);
    (mv.0.clamp(bounds.0, max_x), mv.1.clamp(bounds.1, max_y))
}

fn exact_ok(mv: (i32, i32), bounds: (i32, i32, i32, i32)) -> bool {
    mv.0 >= bounds.0 && mv.1 >= bounds.1 && mv.0 < bounds.2 && mv.1 < bounds.3
}

fn exact_lambda(opts: &AnalyseOpts, level: i32, pel: i32) -> i32 {
    let mut value = opts.lambda / (pel * pel).max(1);
    if opts.plevel == 1 {
        value = value.saturating_mul(1i32 << level);
    } else if opts.plevel == 2 {
        value = value.saturating_mul(1i32 << (2 * level));
    }
    value
}

fn exact_global_doubled(field: &[Vec8]) -> (i32, i32) {
    let mode = |select: fn(&Vec8) -> i16| {
        let mut values: Vec<i32> = field.iter().map(|v| i32::from(select(v))).collect();
        values.sort_unstable();
        let mut best = values.first().copied().unwrap_or(0);
        let mut best_count = 0usize;
        let mut i = 0usize;
        while i < values.len() {
            let value = values[i];
            let mut j = i + 1;
            while j < values.len() && values[j] == value {
                j += 1;
            }
            if j - i > best_count {
                best = value;
                best_count = j - i;
            }
            i = j;
        }
        best
    };
    let mx = mode(|v| v.dx);
    let my = mode(|v| v.dy);
    let mut sx = 0i64;
    let mut sy = 0i64;
    let mut count = 0i64;
    for value in field {
        if (i32::from(value.dx) - mx).abs() < 6 && (i32::from(value.dy) - my).abs() < 6 {
            sx += i64::from(value.dx);
            sy += i64::from(value.dy);
            count += 1;
        }
    }
    if count == 0 {
        (2 * mx, 2 * my)
    } else {
        ((2 * sx / count) as i32, (2 * sy / count) as i32)
    }
}

fn exact_recalculate(
    src: &SuperPlanes<'_>,
    refp: &SuperPlanes<'_>,
    opts: &AnalyseOpts,
    field: Vec<Vec8>,
    old_gw: i32,
    old_gh: i32,
) -> Vec<Vec8> {
    let Some(thsad) = opts.refine_thsad else {
        return field;
    };
    let old_bw = opts.block_w;
    let old_bh = opts.block_h;
    let old_ox = opts.overlap_x;
    let old_oy = opts.overlap_y;
    let (bw, bh, ox, oy) = opts.output_block();
    let (gw, gh) = opts.grid(bw, bh, ox, oy);
    let old_step_x = (old_bw - old_ox).max(1);
    let old_step_y = (old_bh - old_oy).max(1);
    let step_x = (bw - ox).max(1);
    let step_y = (bh - oy).max(1);
    let pel = opts.pel.max(1);
    let (lw, lh) = src.level_size(0);
    let mut out = vec![Vec8::default(); (gw * gh) as usize];

    let get =
        |x: i32, y: i32| field[(y.clamp(0, old_gh - 1) * old_gw + x.clamp(0, old_gw - 1)) as usize];
    for by in 0..gh {
        let dir = if opts.sort && by & 1 != 0 { -1 } else { 1 };
        for pos in 0..gw {
            let bx = if dir > 0 { pos } else { gw - 1 - pos };
            let center_x = bw / 2 + step_x * bx;
            let center_y = bh / 2 + step_y * by;
            let old_x = (center_x - old_bw / 2) / old_step_x;
            let old_y = (center_y - old_bh / 2) / old_step_y;
            let delta_x = (center_x - (old_bw / 2 + old_step_x * old_x)).max(0);
            let delta_y = (center_y - (old_bh / 2 + old_step_y * old_y)).max(0);
            let p00 = get(old_x, old_y);
            let p10 = get(old_x + 1, old_y);
            let p01 = get(old_x, old_y + 1);
            let p11 = get(old_x + 1, old_y + 1);
            let interp = |a: i64, b: i64, c: i64, d: i64| {
                let top = a * i64::from(old_step_x) + i64::from(delta_x) * (b - a);
                let bottom = c * i64::from(old_step_x) + i64::from(delta_x) * (d - c);
                (top + i64::from(delta_y) * (bottom - top) / i64::from(old_step_y))
                    / i64::from(old_step_x)
            };
            let mv = (
                interp(p00.dx.into(), p10.dx.into(), p01.dx.into(), p11.dx.into()) as i32,
                interp(p00.dy.into(), p10.dy.into(), p01.dy.into(), p11.dy.into()) as i32,
            );
            let parent_sad = interp(
                p00.score.into(),
                p10.score.into(),
                p01.score.into(),
                p11.score.into(),
            ) * i64::from(bw * bh)
                / i64::from(old_bw * old_bh);
            let px = bx * step_x;
            let py = by * step_y;
            let border = bx == 0 || by == 0 || bx == gw - 1 || by == gh - 1;
            let factor = if border && bw > 4 { 3 } else { 1 };
            let eval_bw = if factor == 3 { bw / 2 } else { bw };
            let eval_bh = if factor == 3 { bh / 2 } else { bh };
            let eval_px = px + if factor == 3 { bw / 4 } else { 0 };
            let eval_py = py + if factor == 3 { bh / 4 } else { 0 };
            let bounds = mv_bounds(
                eval_px, eval_py, eval_bw, eval_bh, lw as i32, lh as i32, pel,
            );
            let predictor = exact_clip(mv, bounds);
            let mut best = SearchBest {
                mv: predictor,
                sad: parent_sad.max(0).min(i64::from(u32::MAX)) as u32,
                cost: parent_sad.max(0),
            };
            let threshold = thsad / factor;
            if parent_sad > i64::from(threshold) {
                best.sad = exact_sad(
                    src,
                    refp,
                    0,
                    eval_px,
                    eval_py,
                    eval_bw,
                    eval_bh,
                    predictor,
                    opts.refine_satd,
                    pel,
                );
                best.cost = i64::from(best.sad);
                let lambda = if by == 0 {
                    0
                } else {
                    (opts.lambda >> 2) / factor
                };
                exact_refine_search(
                    src,
                    refp,
                    0,
                    eval_px,
                    eval_py,
                    eval_bw,
                    eval_bh,
                    pel,
                    opts.refine_satd,
                    predictor,
                    bounds,
                    lambda,
                    opts.pnew / factor,
                    opts.refine_search_type,
                    opts.refine_distance.abs(),
                    &mut best,
                );
            }
            out[(by * gw + bx) as usize] = Vec8 {
                dx: best.mv.0 as i16,
                dy: best.mv.1 as i16,
                score: best.sad.saturating_mul(factor as u32),
                luma: block_luma_dc(src, 0, px, py, bw, bh),
            };
        }
    }
    out
}

fn reconcile_bidir_gmv(fwd: (i32, i32), bwd: (i32, i32)) -> ((i32, i32), (i32, i32)) {
    let ax = fwd.0.wrapping_add(bwd.0) / 2;
    let ay = fwd.1.wrapping_add(bwd.1) / 2;
    ((fwd.0 - ax, fwd.1 - ay), (bwd.0 - ax, bwd.1 - ay))
}

fn level_grid(opts: &AnalyseOpts, level: i32, bw: i32, bh: i32, ox: i32, oy: i32) -> (i32, i32) {
    let step_x = (bw - ox).max(1);
    let step_y = (bh - oy).max(1);
    let gw0 = ((opts.width - ox) / step_x).max(0);
    let gh0 = ((opts.height - oy) / step_y).max(0);
    let end_x = ox + step_x * gw0;
    let end_y = oy + step_y * gh0;
    let gw = (((end_x >> level) - ox) / step_x).max(1);
    let gh = (((end_y >> level) - oy) / step_y).max(1);
    (gw, gh)
}

fn search_radius(opts: &AnalyseOpts, fine: bool, level: i32, nlevels: i32) -> i32 {
    let d = if fine {
        opts.distance
    } else {
        opts.coarse_distance
    };
    if d == 0 {
        if level == 0 && nlevels > 1 { 0 } else { 10 }
    } else {
        d.abs()
    }
}

fn adapt_lambda(nlambda: i32, lsad: i32, pred_score: u32) -> i32 {
    let lsad = lsad.max(1) as f64;
    let half_score = if pred_score == u32::MAX {
        -1.0
    } else {
        (pred_score >> 1) as f64
    };
    let t = lsad / (lsad + half_score);
    ((nlambda as f64) * t * t).min(21_474_836.47) as i32
}

fn block_luma_dc(src: &SuperPlanes<'_>, level: i32, px: i32, py: i32, bw: i32, bh: i32) -> u8 {
    let (lw, lh) = src.level_size(level);
    let y_off = src.level_y_offset(level);
    let mut sum = 0u32;
    let mut count = 0u32;
    for row in 0..bh as usize {
        let y = py as usize + row;
        if y >= lh {
            break;
        }
        for col in 0..bw as usize {
            let x = px as usize + col;
            if x >= lw {
                break;
            }
            sum += u32::from(
                src.y
                    .get((y_off + y) * src.y_stride + x)
                    .copied()
                    .unwrap_or(0),
            );
            count += 1;
        }
    }
    if count == 0 {
        0
    } else {
        (sum / count).min(255) as u8
    }
}

#[inline]
fn mv_bounds(
    px: i32,
    py: i32,
    bw: i32,
    bh: i32,
    lw: i32,
    lh: i32,
    pel: i32,
) -> (i32, i32, i32, i32) {
    let pel = pel.max(1);
    (
        -pel * px,
        -pel * py,
        pel * (lw - px - bw).max(0),
        pel * (lh - py - bh).max(0),
    )
}

fn edge_shift_origins(
    px: i32,
    py: i32,
    mvx: i32,
    mvy: i32,
    bw: i32,
    bh: i32,
    lw: i32,
    lh: i32,
    pel: i32,
) -> Option<(i32, i32, i32, i32)> {
    let pel = pel.max(1);
    let (min_x, min_y, max_x, max_y) = mv_bounds(px, py, bw, bh, lw, lh, pel);
    let ox = if mvx < min_x {
        mvx - min_x
    } else if max_x > min_x && mvx >= max_x {
        mvx + 1 - max_x
    } else {
        0
    };
    let oy = if mvy < min_y {
        mvy - min_y
    } else if max_y > min_y && mvy >= max_y {
        mvy + 1 - max_y
    } else {
        0
    };
    if ox == 0 && oy == 0 {
        return None;
    }
    let cx = (px - ox).clamp(0, (lw - bw).max(0));
    let cy = (py - oy).clamp(0, (lh - bh).max(0));
    let (rx, ry) = match pel {
        1 => (cx + mvx, cy + mvy),
        4 => ((4 * cx + mvx) >> 2, (4 * cy + mvy) >> 2),
        _ => ((2 * cx + mvx) >> 1, (2 * cy + mvy) >> 1),
    };
    Some((cx, cy, rx, ry))
}

#[inline]
fn chroma_origins_after_luma_shift(px: i32, py: i32, cx: i32, cy: i32) -> (i32, i32) {
    ((px >> 1) - ((px - cx) >> 1), (py >> 1) - ((py - cy) >> 1))
}

fn block_cost_lc(
    src: &SuperPlanes<'_>,
    refp: &SuperPlanes<'_>,
    level: i32,
    px: i32,
    py: i32,
    bw: i32,
    bh: i32,
    mvx: i32,
    mvy: i32,
    use_satd: bool,
    pel: i32,
) -> (u32, u32) {
    let pel = pel.max(1);

    let (lw, lh) = src.level_size(level);
    if level == 0 && pel >= 2 {
        let min_x = -pel * px;
        let min_y = -pel * py;
        if mvx < min_x || mvy < min_y {
            if let Some((l, c, _)) =
                block_cost_edge_lc(src, refp, level, px, py, bw, bh, mvx, mvy, use_satd, pel)
            {
                return (l, c);
            }
        }
    }

    let bw_u = bw as usize;
    let bh_u = bh as usize;
    let px_u = px as usize;
    let py_u = py as usize;
    let (sad_l, _) = block_cost_luma_interior(
        src, refp, level, px, py, bw_u, bh_u, mvx, mvy, use_satd, pel, false,
    );

    let sad_c = chroma_sad_x4(
        src, refp, level, pel, mvx, mvy, px_u, py_u, bw_u, bh_u, lw, lh,
    );
    (sad_l, sad_c)
}

fn block_cost_luma_interior(
    src: &SuperPlanes<'_>,
    refp: &SuperPlanes<'_>,
    level: i32,
    px: i32,
    py: i32,
    bw: usize,
    bh: usize,
    mvx: i32,
    mvy: i32,
    use_satd: bool,
    pel: i32,
    compute_luma: bool,
) -> (u32, u8) {
    let (lw, lh) = src.level_size(level);
    let px_u = px as usize;
    let py_u = py as usize;
    let y_off = src.level_y_offset(level);

    let (sub_x, sub_y, mv_x_px, mv_y_px) = if level == 0 && pel >= 2 {
        (mvx & 1, mvy & 1, mvx >> 1, mvy >> 1)
    } else {
        (0, 0, mvx, mvy)
    };
    let sub_idx = ((sub_y * pel) + sub_x) as usize;
    let cur_row0 = y_off;
    let ref_row0 = y_off + sub_idx * lh;

    let mut sum = 0u32;
    let sad_l =
        if use_satd && matches!(bw, 4 | 8 | 16 | 32) && matches!(bh, 4 | 8 | 16 | 32) && bw == bh {
            if compute_luma {
                for row in 0..bh {
                    let sy = py_u + row;
                    if sy >= lh {
                        break;
                    }
                    for col in 0..bw {
                        let sx = px_u + col;
                        if sx >= lw {
                            break;
                        }
                        sum += u32::from(
                            src.y
                                .get((cur_row0 + sy) * src.y_stride + sx)
                                .copied()
                                .unwrap_or(0),
                        );
                    }
                }
            }
            satd_luma_clamp(
                src, refp, cur_row0, ref_row0, px_u, py_u, mv_x_px, mv_y_px, lw, lh, bw,
            )
        } else {
            #[cfg(any(
                target_arch = "x86_64",
                all(target_arch = "wasm32", target_feature = "simd128")
            ))]
            {
                let rx0 = px + mv_x_px;
                let ry0 = py + mv_y_px;
                let interior = px >= 0
                    && py >= 0
                    && rx0 >= 0
                    && ry0 >= 0
                    && px as usize + bw <= lw
                    && py as usize + bh <= lh
                    && rx0 as usize + bw <= lw
                    && ry0 as usize + bh <= lh;
                if interior && matches!(bw, 4 | 8 | 16 | 32) {
                    let src_idx = (cur_row0 + py_u) * src.y_stride + px_u;
                    let ref_idx = (ref_row0 + ry0 as usize) * refp.y_stride + rx0 as usize;

                    let sad = unsafe {
                        sad_u8_simd(
                            src.y.as_ptr().add(src_idx),
                            src.y_stride,
                            refp.y.as_ptr().add(ref_idx),
                            refp.y_stride,
                            bw,
                            bh,
                        )
                    };
                    if compute_luma {
                        for row in 0..bh {
                            let start = src_idx + row * src.y_stride;
                            sum += src.y[start..start + bw]
                                .iter()
                                .map(|&v| u32::from(v))
                                .sum::<u32>();
                        }
                    }
                    let luma = (sum / (bw * bh).max(1) as u32).min(255) as u8;
                    return (sad, luma);
                }
            }

            let mut sad = 0u32;
            for row in 0..bh {
                let sy = py_u + row;
                let ry = (py + row as i32 + mv_y_px).clamp(0, lh as i32 - 1) as usize;
                if sy >= lh {
                    break;
                }
                for col in 0..bw {
                    let sx = px_u + col;
                    let rx = (px + col as i32 + mv_x_px).clamp(0, lw as i32 - 1) as usize;
                    if sx >= lw {
                        break;
                    }
                    let a = src
                        .y
                        .get((cur_row0 + sy) * src.y_stride + sx)
                        .copied()
                        .unwrap_or(0);
                    let b = refp
                        .y
                        .get((ref_row0 + ry) * refp.y_stride + rx)
                        .copied()
                        .unwrap_or(0);
                    sad = sad.saturating_add(u32::from(a.abs_diff(b)));
                    if compute_luma {
                        sum = sum.saturating_add(u32::from(a));
                    }
                }
            }
            sad
        };
    let luma = (sum / (bw * bh).max(1) as u32).min(255) as u8;
    (sad_l, luma)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn sad_u8_simd(
    src: *const u8,
    src_stride: usize,
    refp: *const u8,
    ref_stride: usize,
    width: usize,
    height: usize,
) -> u32 {
    let mut acc = _mm_setzero_si128();
    for row in 0..height {
        unsafe {
            let a = src.add(row * src_stride);
            let b = refp.add(row * ref_stride);
            match width {
                4 => {
                    let av = _mm_cvtsi32_si128(std::ptr::read_unaligned(a.cast::<i32>()));
                    let bv = _mm_cvtsi32_si128(std::ptr::read_unaligned(b.cast::<i32>()));
                    acc = _mm_add_epi64(acc, _mm_sad_epu8(av, bv));
                }
                8 => {
                    let av = _mm_loadl_epi64(a.cast::<__m128i>());
                    let bv = _mm_loadl_epi64(b.cast::<__m128i>());
                    acc = _mm_add_epi64(acc, _mm_sad_epu8(av, bv));
                }
                16 | 32 => {
                    for x in (0..width).step_by(16) {
                        let av = _mm_loadu_si128(a.add(x).cast::<__m128i>());
                        let bv = _mm_loadu_si128(b.add(x).cast::<__m128i>());
                        acc = _mm_add_epi64(acc, _mm_sad_epu8(av, bv));
                    }
                }
                _ => std::hint::unreachable_unchecked(),
            }
        }
    }
    let lo = _mm_cvtsi128_si32(acc) as u32;
    let hi = _mm_cvtsi128_si32(_mm_srli_si128::<8>(acc)) as u32;
    lo + hi
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn sad_u8_simd(
    src: *const u8,
    src_stride: usize,
    refp: *const u8,
    ref_stride: usize,
    width: usize,
    height: usize,
) -> u32 {
    let mut acc = u32x4(0, 0, 0, 0);
    for row in 0..height {
        unsafe {
            let a = src.add(row * src_stride);
            let b = refp.add(row * ref_stride);
            for x in (0..width).step_by(16) {
                let remaining = width - x;
                let av = if remaining >= 16 {
                    v128_load(a.add(x).cast::<v128>())
                } else if remaining >= 8 {
                    v128_load64_zero(a.add(x).cast::<u64>())
                } else {
                    v128_load32_zero(a.add(x).cast::<u32>())
                };
                let bv = if remaining >= 16 {
                    v128_load(b.add(x).cast::<v128>())
                } else if remaining >= 8 {
                    v128_load64_zero(b.add(x).cast::<u64>())
                } else {
                    v128_load32_zero(b.add(x).cast::<u32>())
                };
                let difference = v128_or(u8x16_sub_sat(av, bv), u8x16_sub_sat(bv, av));
                let pairs = u16x8_extadd_pairwise_u8x16(difference);
                acc = u32x4_add(acc, u32x4_extadd_pairwise_u16x8(pairs));
            }
        }
    }
    u32x4_extract_lane::<0>(acc)
        + u32x4_extract_lane::<1>(acc)
        + u32x4_extract_lane::<2>(acc)
        + u32x4_extract_lane::<3>(acc)
}

fn block_cost_edge_lc(
    src: &SuperPlanes<'_>,
    refp: &SuperPlanes<'_>,
    level: i32,
    px: i32,
    py: i32,
    bw: i32,
    bh: i32,
    mvx: i32,
    mvy: i32,
    use_satd: bool,
    pel: i32,
) -> Option<(u32, u32, u8)> {
    let (lw, lh) = src.level_size(level);
    let (cx, cy, rx0, ry0) =
        edge_shift_origins(px, py, mvx, mvy, bw, bh, lw as i32, lh as i32, pel)?;
    let y_off = src.level_y_offset(level);

    let sub_x = mvx & 1;
    let sub_y = mvy & 1;
    let sub_idx = ((sub_y * pel) + sub_x) as usize;
    let cur_row0 = y_off;
    let ref_row0 = y_off + sub_idx * lh;
    let bw_u = bw as usize;
    let bh_u = bh as usize;
    let lw_i = lw as i32;
    let lh_i = lh as i32;

    let mut sum = 0u32;
    let sad_l = if use_satd
        && matches!(bw_u, 4 | 8 | 16 | 32)
        && matches!(bh_u, 4 | 8 | 16 | 32)
        && bw_u == bh_u
    {
        for row in 0..bh_u {
            let sy = (cy + row as i32).clamp(0, (lh_i - 1).max(0)) as usize;
            for col in 0..bw_u {
                let sx = (cx + col as i32).clamp(0, (lw_i - 1).max(0)) as usize;
                sum += u32::from(
                    src.y
                        .get((cur_row0 + sy) * src.y_stride + sx)
                        .copied()
                        .unwrap_or(0),
                );
            }
        }
        satd_luma_edge(
            src, refp, cur_row0, ref_row0, cx, cy, rx0, ry0, lw, lh, bw_u,
        )
    } else {
        let mut sad = 0u32;
        for row in 0..bh_u {
            let sy = (cy + row as i32).clamp(0, (lh_i - 1).max(0)) as usize;
            let ry = (ry0 + row as i32).clamp(0, (lh_i - 1).max(0)) as usize;
            for col in 0..bw_u {
                let sx = (cx + col as i32).clamp(0, (lw_i - 1).max(0)) as usize;
                let rx = (rx0 + col as i32).clamp(0, (lw_i - 1).max(0)) as usize;
                let a = src
                    .y
                    .get((cur_row0 + sy) * src.y_stride + sx)
                    .copied()
                    .unwrap_or(0);
                let b = refp
                    .y
                    .get((ref_row0 + ry) * refp.y_stride + rx)
                    .copied()
                    .unwrap_or(0);
                sad = sad.saturating_add(u32::from(a.abs_diff(b)));
                sum = sum.saturating_add(u32::from(a));
            }
        }
        sad
    };

    let cw = bw_u / 2;
    let ch = bh_u / 2;
    let (clw, clh) = (lw / 2, lh / 2);
    let (ccx, ccy) = chroma_origins_after_luma_shift(px, py, cx, cy);
    let crx0 = ((mvx >> 1) + 2 * ccx) >> 1;
    let cry0 = ((mvy >> 1) + 2 * ccy) >> 1;
    let c_off = chroma_level_offset(src, level);

    let c_sub = (((mvy & 1) * pel) + (mvx & 1)) as usize;
    let c_ref = c_off + c_sub * clh;
    let clw_i = clw as i32;
    let clh_i = clh as i32;
    let mut sad_uv = 0u32;
    for row in 0..ch {
        let sy = (ccy + row as i32).clamp(0, (clh_i - 1).max(0)) as usize;
        let ry = (cry0 + row as i32).clamp(0, (clh_i - 1).max(0)) as usize;
        for col in 0..cw {
            let sx = (ccx + col as i32).clamp(0, (clw_i - 1).max(0)) as usize;
            let rx = (crx0 + col as i32).clamp(0, (clw_i - 1).max(0)) as usize;
            let au = src
                .u
                .get((c_off + sy) * src.u_stride + sx)
                .copied()
                .unwrap_or(128);
            let av = src
                .v
                .get((c_off + sy) * src.v_stride + sx)
                .copied()
                .unwrap_or(128);
            let bu = refp
                .u
                .get((c_ref + ry) * refp.u_stride + rx)
                .copied()
                .unwrap_or(128);
            let bv = refp
                .v
                .get((c_ref + ry) * refp.v_stride + rx)
                .copied()
                .unwrap_or(128);
            sad_uv = sad_uv
                .saturating_add(u32::from(au.abs_diff(bu)))
                .saturating_add(u32::from(av.abs_diff(bv)));
        }
    }
    let sad_c = sad_uv.saturating_mul(4);
    let luma = (sum / (bw_u * bh_u).max(1) as u32).min(255) as u8;
    Some((sad_l, sad_c, luma))
}

fn chroma_sad_x4(
    src: &SuperPlanes<'_>,
    refp: &SuperPlanes<'_>,
    level: i32,
    pel: i32,
    mvx: i32,
    mvy: i32,
    px: usize,
    py: usize,
    bw: usize,
    bh: usize,
    lw: usize,
    lh: usize,
) -> u32 {
    let cw = bw / 2;
    let ch = bh / 2;
    let cpx = px / 2;
    let cpy = py / 2;
    let (clw, clh) = (lw / 2, lh / 2);
    let c_off = chroma_level_offset(src, level);

    let (c_sub_idx, mv_cx, mv_cy) = if level == 0 && pel >= 2 {
        let sx = mvx & 1;
        let sy = mvy & 1;
        let idx = ((sy * pel) + sx) as usize;
        (idx, mvx >> 2, mvy >> 2)
    } else {
        (0usize, mvx >> 1, mvy >> 1)
    };

    let c_cur = c_off;
    let c_ref = c_off
        + if level == 0 && pel > 1 {
            c_sub_idx * clh
        } else {
            0
        };

    #[cfg(any(
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))]
    if matches!(cw, 4 | 8 | 16) {
        let rx0 = cpx as i32 + mv_cx;
        let ry0 = cpy as i32 + mv_cy;
        if cpx + cw <= clw
            && cpy + ch <= clh
            && rx0 >= 0
            && ry0 >= 0
            && rx0 as usize + cw <= clw
            && ry0 as usize + ch <= clh
        {
            let src_u = (c_cur + cpy) * src.u_stride + cpx;
            let src_v = (c_cur + cpy) * src.v_stride + cpx;
            let ref_u = (c_ref + ry0 as usize) * refp.u_stride + rx0 as usize;
            let ref_v = (c_ref + ry0 as usize) * refp.v_stride + rx0 as usize;
            let src_u_end = src_u + (ch - 1) * src.u_stride + cw;
            let src_v_end = src_v + (ch - 1) * src.v_stride + cw;
            let ref_u_end = ref_u + (ch - 1) * refp.u_stride + cw;
            let ref_v_end = ref_v + (ch - 1) * refp.v_stride + cw;
            if src_u_end <= src.u.len()
                && src_v_end <= src.v.len()
                && ref_u_end <= refp.u.len()
                && ref_v_end <= refp.v.len()
            {
                return unsafe {
                    sad_u8_simd(
                        src.u.as_ptr().add(src_u),
                        src.u_stride,
                        refp.u.as_ptr().add(ref_u),
                        refp.u_stride,
                        cw,
                        ch,
                    )
                    .saturating_add(sad_u8_simd(
                        src.v.as_ptr().add(src_v),
                        src.v_stride,
                        refp.v.as_ptr().add(ref_v),
                        refp.v_stride,
                        cw,
                        ch,
                    ))
                    .saturating_mul(4)
                };
            }
        }
    }

    let mut sad_uv = 0u32;
    for row in 0..ch {
        let sy = cpy + row;
        let ry = (cpy as i32 + row as i32 + mv_cy).clamp(0, clh as i32 - 1) as usize;
        if sy >= clh {
            break;
        }
        for col in 0..cw {
            let sx = cpx + col;
            let rx = (cpx as i32 + col as i32 + mv_cx).clamp(0, clw as i32 - 1) as usize;
            if sx >= clw {
                break;
            }
            let au = src
                .u
                .get((c_cur + sy) * src.u_stride + sx)
                .copied()
                .unwrap_or(128);
            let av = src
                .v
                .get((c_cur + sy) * src.v_stride + sx)
                .copied()
                .unwrap_or(128);
            let bu = refp
                .u
                .get((c_ref + ry) * refp.u_stride + rx)
                .copied()
                .unwrap_or(128);
            let bv = refp
                .v
                .get((c_ref + ry) * refp.v_stride + rx)
                .copied()
                .unwrap_or(128);
            sad_uv = sad_uv
                .saturating_add(u32::from(au.abs_diff(bu)))
                .saturating_add(u32::from(av.abs_diff(bv)));
        }
    }
    sad_uv.saturating_mul(4)
}

fn satd_luma_edge(
    src: &SuperPlanes<'_>,
    refp: &SuperPlanes<'_>,
    cur_row0: usize,
    ref_row0: usize,
    cx: i32,
    cy: i32,
    rx0: i32,
    ry0: i32,
    lw: usize,
    lh: usize,
    n: usize,
) -> u32 {
    let lw_i = lw as i32;
    let lh_i = lh as i32;
    let mut total = 0u32;
    let tiles = n / 4;
    for ty in 0..tiles {
        for tx in 0..tiles {
            let mut diff = [[0i32; 4]; 4];
            for r in 0..4 {
                let sy = (cy + (ty * 4 + r) as i32).clamp(0, (lh_i - 1).max(0)) as usize;
                let ry = (ry0 + (ty * 4 + r) as i32).clamp(0, (lh_i - 1).max(0)) as usize;
                for c in 0..4 {
                    let sx = (cx + (tx * 4 + c) as i32).clamp(0, (lw_i - 1).max(0)) as usize;
                    let rx = (rx0 + (tx * 4 + c) as i32).clamp(0, (lw_i - 1).max(0)) as usize;
                    let a = i32::from(
                        src.y
                            .get((cur_row0 + sy) * src.y_stride + sx)
                            .copied()
                            .unwrap_or(0),
                    );
                    let b = i32::from(
                        refp.y
                            .get((ref_row0 + ry) * refp.y_stride + rx)
                            .copied()
                            .unwrap_or(0),
                    );
                    diff[r][c] = a - b;
                }
            }
            total = total.saturating_add(hadamard4_satd(diff));
        }
    }
    total
}

fn satd_luma_clamp(
    src: &SuperPlanes<'_>,
    refp: &SuperPlanes<'_>,
    cur_row0: usize,
    ref_row0: usize,
    px: usize,
    py: usize,
    mv_x: i32,
    mv_y: i32,
    lw: usize,
    lh: usize,
    n: usize,
) -> u32 {
    let rx0 = px as i32 + mv_x;
    let ry0 = py as i32 + mv_y;
    let src_end = (cur_row0 + py + n - 1)
        .saturating_mul(src.y_stride)
        .saturating_add(px + n);
    let ref_end = if ry0 >= 0 {
        (ref_row0 + ry0 as usize + n - 1)
            .saturating_mul(refp.y_stride)
            .saturating_add(rx0.max(0) as usize + n)
    } else {
        usize::MAX
    };
    let interior = px + n <= lw
        && py + n <= lh
        && rx0 >= 0
        && ry0 >= 0
        && rx0 as usize + n <= lw
        && ry0 as usize + n <= lh
        && src_end <= src.y.len()
        && ref_end <= refp.y.len();
    if interior {
        let src_idx = (cur_row0 + py) * src.y_stride + px;
        let ref_idx = (ref_row0 + ry0 as usize) * refp.y_stride + rx0 as usize;

        return unsafe {
            satd_luma_interior(
                src.y.as_ptr().add(src_idx),
                src.y_stride,
                refp.y.as_ptr().add(ref_idx),
                refp.y_stride,
                n,
            )
        };
    }

    let mut total = 0u32;
    let tiles = n / 4;
    for ty in 0..tiles {
        for tx in 0..tiles {
            let mut diff = [[0i32; 4]; 4];
            for r in 0..4 {
                let sy = py + ty * 4 + r;
                let ry = (py as i32 + (ty * 4 + r) as i32 + mv_y).clamp(0, lh as i32 - 1) as usize;
                for c in 0..4 {
                    let sx = px + tx * 4 + c;
                    let rx =
                        (px as i32 + (tx * 4 + c) as i32 + mv_x).clamp(0, lw as i32 - 1) as usize;
                    let a = if sy < lh && sx < lw {
                        i32::from(
                            src.y
                                .get((cur_row0 + sy) * src.y_stride + sx)
                                .copied()
                                .unwrap_or(0),
                        )
                    } else {
                        0
                    };
                    let b = i32::from(
                        refp.y
                            .get((ref_row0 + ry) * refp.y_stride + rx)
                            .copied()
                            .unwrap_or(0),
                    );
                    diff[r][c] = a - b;
                }
            }
            total = total.saturating_add(hadamard4_satd(diff));
        }
    }
    total
}

#[cfg_attr(target_arch = "x86_64", target_feature(enable = "sse2"))]
unsafe fn satd_luma_interior(
    src: *const u8,
    src_stride: usize,
    refp: *const u8,
    ref_stride: usize,
    n: usize,
) -> u32 {
    let mut total = 0u32;
    let tiles = n / 4;
    for ty in 0..tiles {
        for tx in 0..tiles {
            #[cfg(target_arch = "x86_64")]
            {
                let offset_a = ty * 4 * src_stride + tx * 4;
                let offset_b = ty * 4 * ref_stride + tx * 4;

                total = total.saturating_add(unsafe {
                    hadamard4_satd_sse2(
                        src.add(offset_a),
                        src_stride,
                        refp.add(offset_b),
                        ref_stride,
                    )
                });
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                let mut diff = [[0i32; 4]; 4];
                for (r, diff_row) in diff.iter_mut().enumerate() {
                    let offset_a = (ty * 4 + r) * src_stride + tx * 4;
                    let offset_b = (ty * 4 + r) * ref_stride + tx * 4;
                    for (c, d) in diff_row.iter_mut().enumerate() {
                        unsafe {
                            *d = i32::from(*src.add(offset_a + c))
                                - i32::from(*refp.add(offset_b + c));
                        }
                    }
                }
                total = total.saturating_add(hadamard4_satd(diff));
            }
        }
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn hadamard4_satd_sse2(
    src: *const u8,
    src_stride: usize,
    refp: *const u8,
    ref_stride: usize,
) -> u32 {
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn h4(x0: __m128i, x1: __m128i, x2: __m128i, x3: __m128i) -> [__m128i; 4] {
        let s0 = _mm_add_epi16(x0, x1);
        let s1 = _mm_sub_epi16(x0, x1);
        let s2 = _mm_add_epi16(x2, x3);
        let s3 = _mm_sub_epi16(x2, x3);
        [
            _mm_add_epi16(s0, s2),
            _mm_add_epi16(s1, s3),
            _mm_sub_epi16(s0, s2),
            _mm_sub_epi16(s1, s3),
        ]
    }

    let zero = _mm_setzero_si128();
    let mut rows = [zero; 4];
    for (row, out) in rows.iter_mut().enumerate() {
        unsafe {
            let a = _mm_cvtsi32_si128(std::ptr::read_unaligned(
                src.add(row * src_stride).cast::<i32>(),
            ));
            let b = _mm_cvtsi32_si128(std::ptr::read_unaligned(
                refp.add(row * ref_stride).cast::<i32>(),
            ));
            *out = _mm_sub_epi16(_mm_unpacklo_epi8(a, zero), _mm_unpacklo_epi8(b, zero));
        }
    }

    let vertical = unsafe { h4(rows[0], rows[1], rows[2], rows[3]) };
    let t0 = _mm_unpacklo_epi16(vertical[0], vertical[1]);
    let t1 = _mm_unpacklo_epi16(vertical[2], vertical[3]);
    let cols01 = _mm_unpacklo_epi32(t0, t1);
    let cols23 = _mm_unpackhi_epi32(t0, t1);

    let coeff = unsafe {
        h4(
            cols01,
            _mm_srli_si128::<8>(cols01),
            cols23,
            _mm_srli_si128::<8>(cols23),
        )
    };

    let mut sums = zero;
    for values in coeff {
        let abs = _mm_max_epi16(values, _mm_sub_epi16(zero, values));
        sums = _mm_add_epi16(sums, abs);
    }
    let pairs = _mm_madd_epi16(sums, _mm_set1_epi16(1));
    (_mm_cvtsi128_si32(_mm_add_epi32(pairs, _mm_srli_si128::<4>(pairs))) as u32) >> 1
}

fn hadamard4_satd(diff: [[i32; 4]; 4]) -> u32 {
    let mut t = [[0i32; 4]; 4];
    for i in 0..4 {
        let s0 = diff[i][0] + diff[i][1];
        let s1 = diff[i][0] - diff[i][1];
        let s2 = diff[i][2] + diff[i][3];
        let s3 = diff[i][2] - diff[i][3];
        t[i][0] = s0 + s2;
        t[i][1] = s1 + s3;
        t[i][2] = s0 - s2;
        t[i][3] = s1 - s3;
    }
    let mut s = 0i32;
    for j in 0..4 {
        let s0 = t[0][j] + t[1][j];
        let s1 = t[0][j] - t[1][j];
        let s2 = t[2][j] + t[3][j];
        let s3 = t[2][j] - t[3][j];
        s += (s0 + s2).abs() + (s1 + s3).abs() + (s0 - s2).abs() + (s1 - s3).abs();
    }
    (s.max(0) as u32) >> 1
}

fn chroma_level_offset(src: &SuperPlanes<'_>, level: i32) -> usize {
    let mut y = 0usize;
    let pel = src.pel as usize;
    for lv in 0..level {
        let h = reduce_dim(src.luma_h as i32, lv) as usize / 2;
        let sub = if lv == 0 && src.full { pel * pel } else { 1 };
        y += h.max(1) * sub;
    }
    y
}

#[inline]
fn rescale_scores(field: &mut [Vec8], width: i32, height: i32, blk: i32) {
    let diag = {
        let f = (f64::from(width).powi(2) + f64::from(height).powi(2)).sqrt();
        (f as i32).max(0)
    };
    let denom = (diag.saturating_mul(blk.max(1))).max(1);
    for v in field.iter_mut() {
        let mag = i32::from(v.dx).abs() + i32::from(v.dy).abs();
        let ratio = (100 * mag) / denom;
        let mut score = v.score as i32;
        if ratio >= 51 {
            score = score.saturating_mul(20);
        } else if ratio >= 16 {
            let v71 = ((3_926_827_243u64)
                .wrapping_mul(score as u32 as u64)
                .wrapping_mul((ratio - 15) as u32 as u64))
                >> 32;
            let v71 = v71 as i32;
            score = score.saturating_add(9 * ((v71 >> 31) + (v71 >> 5)));
        }
        v.score = (score as u32) & 0x00FF_FFFF;
    }
}

pub(crate) fn pack_vector_frame(
    opts: &AnalyseOpts,
    previous: Option<&[Vec8]>,
    current: Option<&[Vec8]>,
) -> Vec<u8> {
    let (bw, bh, ox, oy) = opts.output_block();
    let (gw, gh) = opts.grid(bw, bh, ox, oy);
    let count = (gw * gh) as usize;
    let flags = (if previous.is_some() { 1 } else { 0 }) | (if current.is_some() { 2 } else { 0 });

    let region_size = 4 + count * 8;
    let mut n_regions = 0;
    if previous.is_some() {
        n_regions += 1;
    }
    if current.is_some() {
        n_regions += 1;
    }
    let total = 0x40 + n_regions * region_size;
    let mut out = vec![0u8; total];

    write_i32(&mut out, 0, 16);
    write_i32(&mut out, 4, 0xA0);
    write_i32(&mut out, 8, flags);
    write_i32(&mut out, 0x0C, bw);
    write_i32(&mut out, 0x10, bh);
    write_i32(&mut out, 0x14, opts.pel);
    write_i32(
        &mut out,
        0x18,
        if opts.refine_thsad.is_some() {
            1
        } else {
            opts.levels
        },
    );
    write_i32(&mut out, 0x1C, 0);
    write_i32(&mut out, 0x20, opts.width);
    write_i32(&mut out, 0x24, opts.height);
    write_i32(&mut out, 0x28, ox);
    write_i32(&mut out, 0x2C, oy);
    write_i32(&mut out, 0x30, gw);
    write_i32(&mut out, 0x34, gh);
    write_i32(&mut out, 0x38, 1);
    write_i32(&mut out, 0x3C, opts.delta);

    let mut cursor = 0x40;
    let marker = (2 * count + 1) as i32;
    if let Some(vecs) = previous {
        write_i32(&mut out, cursor, marker);
        cursor += 4;
        cursor = write_vecs(&mut out, cursor, vecs, count);
    }
    if let Some(vecs) = current {
        write_i32(&mut out, cursor, marker);
        cursor += 4;
        let _ = write_vecs(&mut out, cursor, vecs, count);
    }
    let _ = cursor;
    out
}

fn write_vecs(out: &mut [u8], mut cursor: usize, vecs: &[Vec8], count: usize) -> usize {
    for i in 0..count {
        let v = vecs.get(i).copied().unwrap_or_default();
        let d0 = ((v.dx as u32) << 16) | (v.dy as u16 as u32);
        let d1 = (v.score & 0x00FF_FFFF) | ((v.luma as u32) << 24);
        write_u32(out, cursor, d0);
        write_u32(out, cursor + 4, d1);
        cursor += 8;
    }
    cursor
}

pub(crate) fn pack_vdata_header(opts: &AnalyseOpts) -> Vec<i32> {
    let (bw, bh, ox, oy) = opts.output_block();
    let (gw, gh) = opts.grid(bw, bh, ox, oy);
    let flags = match opts.vectors {
        1 => 1,
        2 => 2,
        _ => 3,
    };
    let mut h = vec![0i32; 16];
    h[0] = 0xA0;
    h[1] = flags;
    h[2] = bw;
    h[3] = bh;
    h[4] = opts.pel;
    h[5] = if opts.refine_thsad.is_some() {
        1
    } else {
        opts.levels
    };
    h[7] = opts.width;
    h[8] = opts.height;
    h[9] = ox;
    h[10] = oy;
    h[11] = gw;
    h[12] = gh;
    h[13] = 1;
    h[14] = opts.delta;
    h
}

fn write_i32(buf: &mut [u8], off: usize, v: i32) {
    if let Some(dst) = buf.get_mut(off..off + 4) {
        dst.copy_from_slice(&v.to_le_bytes());
    }
}

fn write_u32(buf: &mut [u8], off: usize, v: u32) {
    if let Some(dst) = buf.get_mut(off..off + 4) {
        dst.copy_from_slice(&v.to_le_bytes());
    }
}
