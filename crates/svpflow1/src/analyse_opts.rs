use crate::params::Value;
use crate::super_opts::SuperOpts;

#[derive(Clone, Debug)]
#[allow(dead_code, clippy::struct_excessive_bools)]
pub(crate) struct AnalyseOpts {
    pub(crate) vectors: i32,
    pub(crate) block_w: i32,
    pub(crate) block_h: i32,
    pub(crate) overlap_mode: i32,
    pub(crate) overlap_x: i32,
    pub(crate) overlap_y: i32,
    pub(crate) delta: i32,
    pub(crate) pel: i32,
    pub(crate) selector: i32,
    pub(crate) levels: i32,
    pub(crate) super_levels: i32,
    pub(crate) width: i32,
    pub(crate) height: i32,
    pub(crate) lambda: i32,
    pub(crate) lsad: i32,
    pub(crate) plevel: f64,
    pub(crate) pnew: i32,
    pub(crate) pglobal: i32,
    pub(crate) pzero: i32,
    pub(crate) pnbour: i32,
    pub(crate) prev: i32,
    pub(crate) search_type: i32,
    pub(crate) coarse_search_type: i32,
    pub(crate) distance: i32,
    pub(crate) coarse_distance: i32,
    pub(crate) coarse_trymany: bool,
    pub(crate) coarse_bad_sad: i32,
    pub(crate) coarse_bad_range: i32,
    pub(crate) fine_satd: bool,
    pub(crate) coarse_satd: bool,
    pub(crate) refine_thsad: Option<i32>,
    pub(crate) refine_search_type: i32,
    pub(crate) refine_distance: i32,
    pub(crate) refine_satd: bool,
    pub(crate) sort: bool,
}

impl AnalyseOpts {
    pub(crate) fn from_opt(
        opt: Option<&Value>,
        sdata: i64,
        super_opts_hint: Option<&SuperOpts>,
    ) -> Result<Self, String> {
        let (width, height, pel, super_levels, selector) = unpack_sdata(sdata, super_opts_hint)?;

        let vectors = opt.and_then(|v| v.int_at(&["vectors"])).unwrap_or(3) as i32;
        if !(1..=3).contains(&vectors) {
            return Err("SVAnalyse: vectors must be 1, 2 or 3".into());
        }

        let block_w = opt.and_then(|v| v.int_at(&["block", "w"])).unwrap_or(16) as i32;
        let block_h = opt
            .and_then(|v| v.int_at(&["block", "h"]))
            .unwrap_or(block_w as i64) as i32;
        if !matches!(block_w, 4 | 8 | 16 | 32) {
            return Err("SVAnalyse: block.w must be 4/8/16/32".into());
        }

        let overlap_mode = opt
            .and_then(|v| v.int_at(&["block", "overlap"]))
            .unwrap_or(2) as i32;
        let (overlap_x, overlap_y) = overlap_from_mode(block_w, block_h, overlap_mode)?;

        let delta = opt
            .and_then(|v| v.int_at(&["special", "delta"]))
            .or_else(|| opt.and_then(|v| v.int_at(&["delta"])))
            .unwrap_or(1) as i32;

        let lambda = opt
            .and_then(|v| v.float_at(&["main", "penalty", "lambda"]))
            .or_else(|| opt.and_then(|v| v.float_at(&["penalty", "lambda"])))
            .unwrap_or(10.0);
        let lsad = opt
            .and_then(|v| v.int_at(&["main", "penalty", "lsad"]))
            .or_else(|| opt.and_then(|v| v.int_at(&["penalty", "lsad"])))
            .unwrap_or(8000) as i32;
        let plevel = opt
            .and_then(|v| v.float_at(&["main", "penalty", "plevel"]))
            .or_else(|| opt.and_then(|v| v.float_at(&["penalty", "plevel"])))
            .unwrap_or(1.5)
            .max(0.01);
        let pnew = opt
            .and_then(|v| v.int_at(&["main", "penalty", "pnew"]))
            .or_else(|| opt.and_then(|v| v.int_at(&["penalty", "pnew"])))
            .unwrap_or(50) as i32;
        let pglobal = opt
            .and_then(|v| v.int_at(&["main", "penalty", "pglobal"]))
            .or_else(|| opt.and_then(|v| v.int_at(&["penalty", "pglobal"])))
            .unwrap_or(50) as i32;
        let pzero = opt
            .and_then(|v| v.int_at(&["main", "penalty", "pzero"]))
            .or_else(|| opt.and_then(|v| v.int_at(&["penalty", "pzero"])))
            .unwrap_or(100) as i32;
        let pnbour = opt
            .and_then(|v| v.int_at(&["main", "penalty", "pnbour"]))
            .or_else(|| opt.and_then(|v| v.int_at(&["penalty", "pnbour"])))
            .unwrap_or(50) as i32;
        let prev = opt
            .and_then(|v| v.int_at(&["main", "penalty", "prev"]))
            .or_else(|| opt.and_then(|v| v.int_at(&["penalty", "prev"])))
            .unwrap_or(0) as i32;

        let search_type = opt
            .and_then(|v| v.int_at(&["main", "search", "type"]))
            .or_else(|| opt.and_then(|v| v.int_at(&["search", "type"])))
            .unwrap_or(4) as i32;
        let distance = opt
            .and_then(|v| v.int_at(&["main", "search", "distance"]))
            .or_else(|| opt.and_then(|v| v.int_at(&["search", "distance"])))
            .unwrap_or(2) as i32;
        let coarse_search_type = opt
            .and_then(|v| v.int_at(&["main", "search", "coarse", "type"]))
            .or_else(|| opt.and_then(|v| v.int_at(&["search", "coarse", "type"])))
            .unwrap_or(search_type as i64) as i32;
        let coarse_distance = opt
            .and_then(|v| v.int_at(&["main", "search", "coarse", "distance"]))
            .or_else(|| opt.and_then(|v| v.int_at(&["search", "coarse", "distance"])))
            .unwrap_or(0) as i32;
        let coarse_trymany = opt
            .and_then(|v| v.bool_at(&["main", "search", "coarse", "trymany"]))
            .or_else(|| opt.and_then(|v| v.bool_at(&["search", "coarse", "trymany"])))
            .unwrap_or(false);
        let coarse_bad_sad = opt
            .and_then(|v| v.int_at(&["main", "search", "coarse", "bad", "sad"]))
            .or_else(|| opt.and_then(|v| v.int_at(&["search", "coarse", "bad", "sad"])))
            .unwrap_or(1000) as i32;
        let coarse_bad_range = opt
            .and_then(|v| v.int_at(&["main", "search", "coarse", "bad", "range"]))
            .unwrap_or(-24) as i32;

        let fine_satd = opt
            .and_then(|v| v.bool_at(&["main", "search", "satd"]))
            .or_else(|| opt.and_then(|v| v.bool_at(&["search", "satd"])))
            .unwrap_or(false);
        let coarse_satd = opt
            .and_then(|v| v.bool_at(&["main", "search", "coarse", "satd"]))
            .unwrap_or(true);

        let sort = opt
            .and_then(|v| v.bool_at(&["main", "search", "sort"]))
            .or_else(|| opt.and_then(|v| v.bool_at(&["sort"])))
            .unwrap_or(true);

        let refine = opt
            .and_then(|v| v.array_at(&["refine"]))
            .and_then(|values| values.first());
        let refine_thsad = refine
            .and_then(|v| v.int_at(&["thsad"]))
            .map(|x| x as i32 * ((block_w / 2) * (block_h / 2)).max(1) / 32);
        let refine_search_type = refine
            .and_then(|v| v.int_at(&["search", "type"]))
            .unwrap_or(search_type as i64) as i32;
        let refine_distance = refine
            .and_then(|v| v.int_at(&["search", "distance"]))
            .unwrap_or(distance as i64) as i32;
        let refine_satd = refine
            .and_then(|v| v.bool_at(&["search", "satd"]))
            .unwrap_or(fine_satd);

        let mut levels = auto_search_levels(width, height, block_w, block_h, overlap_x, overlap_y)
            .min(super_levels)
            .max(1);

        if let Some(ml) = opt.and_then(|v| v.int_at(&["main", "levels"])) {
            if ml > 0 {
                levels = (ml as i32).min(super_levels).max(1);
            }
        }

        let area = (block_w * block_h).max(1) as f64;
        let lambda = (2.0 * area * 1000.0 * 0.015_625 * lambda) as i32;
        let area_scale = (block_w * block_h).max(1) / 32;
        let lsad = lsad.saturating_mul(area_scale);
        let coarse_bad_sad = coarse_bad_sad.saturating_mul(area_scale);

        Ok(Self {
            vectors,
            block_w,
            block_h,
            overlap_mode,
            overlap_x,
            overlap_y,
            delta,
            pel,
            selector,
            levels,
            super_levels,
            width,
            height,
            lambda,
            lsad,
            plevel,
            pnew,
            pglobal,
            pzero,
            pnbour,
            prev,
            search_type,
            coarse_search_type,
            distance,
            coarse_distance,
            coarse_trymany,
            coarse_bad_sad,
            coarse_bad_range,
            fine_satd,
            coarse_satd,
            refine_thsad,
            refine_search_type,
            refine_distance,
            refine_satd,
            sort,
        })
    }

    pub(crate) fn output_block(&self) -> (i32, i32, i32, i32) {
        if self.refine_thsad.is_some() {
            (
                (self.block_w / 2).max(4),
                (self.block_h / 2).max(4),
                (self.overlap_x / 2).max(0),
                (self.overlap_y / 2).max(0),
            )
        } else {
            (self.block_w, self.block_h, self.overlap_x, self.overlap_y)
        }
    }

    pub(crate) fn grid(&self, bw: i32, bh: i32, ox: i32, oy: i32) -> (i32, i32) {
        let sx = (bw - ox).max(1);
        let sy = (bh - oy).max(1);
        let gw = ((self.width - ox) / sx).max(1);
        let gh = ((self.height - oy) / sy).max(1);
        (gw, gh)
    }
}

fn unpack_sdata(sdata: i64, hint: Option<&SuperOpts>) -> Result<(i32, i32, i32, i32, i32), String> {
    let raw = u64::from_ne_bytes(sdata.to_ne_bytes());
    let height = (raw & 0xFFFF) as i32;
    let width = ((raw >> 16) & 0xFFFF) as i32;
    let pel = ((raw >> 32) & 0xFF) as i32;
    let levels = ((raw >> 40) & 0xFF) as i32;
    let selector = ((raw >> 48) & 0xFF) as i32;
    if width <= 0 || height <= 0 || !(1..=4).contains(&pel) || levels <= 0 {
        if let Some(h) = hint {
            return Ok((h.width, h.height, h.pel, h.levels, h.gpu));
        }
        return Err("SVAnalyse: invalid super clip params".into());
    }
    Ok((width, height, pel, levels, selector))
}

pub(crate) fn overlap_from_mode(bw: i32, bh: i32, mode: i32) -> Result<(i32, i32), String> {
    match mode {
        0 => Ok((0, 0)),
        1 => Ok((bw / 8, bh / 8)),
        2 => Ok((bw / 4, bh / 4)),
        3 => Ok((bw / 2, bh / 2)),
        _ => Err(
            "SVAnalyse: overlap must be 0, 1 (=1/8*block), 2 (=1/4*block) or 3 (=1/2*block)".into(),
        ),
    }
}

pub(crate) fn auto_search_levels(w: i32, h: i32, bw: i32, bh: i32, ox: i32, oy: i32) -> i32 {
    let sx = (bw - ox).max(1);
    let sy = (bh - oy).max(1);
    let gw = ((w - ox) / sx).max(0);
    let gh = ((h - oy) / sy).max(0);
    if gw <= 0 || gh <= 0 {
        return 1;
    }

    let end_x = ox + sx * gw;
    let end_y = oy + sy * gh;
    let mut levels = 0i32;
    loop {
        let gw_l = ((end_x >> levels) - ox) / sx;
        let gh_l = ((end_y >> levels) - oy) / sy;
        if gw_l <= 0 || gh_l <= 0 {
            break;
        }
        levels += 1;
        if levels > 12 {
            break;
        }
    }
    levels.max(1)
}
