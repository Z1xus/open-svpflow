use super::field::{BlockLayout, Field, SearchSettings, SearchType};
use crate::params::Value;

const SUPER_NO_FINEST_LEVEL: u8 = 2;
const BLOCK_DIV: i32 = 64;

const fn block_count(size: i32, overlap: i32, block: i32) -> i32 {
    (size - overlap) / (block - overlap)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SuperParams {
    pub(crate) width: i32,
    pub(crate) height: i32,
    pub(crate) pel: i32,
    pub(crate) levels: i32,
    pub(crate) gpu: i32,
    pub(crate) flags: u8,
}

impl SuperParams {
    pub(crate) fn unpack(data: i64) -> Result<Self, String> {
        let raw = data.to_le_bytes();
        let params = Self {
            height: i32::from(u16::from_le_bytes([raw[0], raw[1]])),
            width: i32::from(u16::from_le_bytes([raw[2], raw[3]])),
            pel: i32::from(raw[4]),
            levels: i32::from(raw[5]),
            gpu: i32::from(raw[6]),
            flags: raw[7],
        };
        if params.height <= 0 || !(1..=4).contains(&params.pel) || params.levels < 1 {
            return Err("SVAnalyse: wrong super clip (pseudoaudio) parameters".into());
        }
        Ok(params)
    }

    pub(crate) const fn has_finest_level(&self) -> bool {
        self.flags & SUPER_NO_FINEST_LEVEL == 0
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Stage {
    pub(crate) layout: BlockLayoutParams,
    pub(crate) blk_x: i32,
    pub(crate) blk_y: i32,
    pub(crate) levels: i32,
    levels_skipped: i32,
    search_type: i64,
    search_param: i32,
    lambda: i32,
    lsad: i32,
    pnew: i32,
    satd: u8,
    bad_sad: i32,
    pub(crate) full_math_level: i32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct BlockLayoutParams {
    pub(crate) width: i32,
    pub(crate) height: i32,
    pub(crate) overlap_x: i32,
    pub(crate) overlap_y: i32,
}

impl BlockLayoutParams {
    const fn layout(self) -> BlockLayout {
        BlockLayout {
            width: self.width,
            height: self.height,
            overlap_x: self.overlap_x,
            overlap_y: self.overlap_y,
        }
    }
}

impl Stage {
    fn covered_width(&self) -> i32 {
        (self.layout.width - self.layout.overlap_x) * self.blk_x + self.layout.overlap_x
    }

    fn covered_height(&self) -> i32 {
        (self.layout.height - self.layout.overlap_y) * self.blk_y + self.layout.overlap_y
    }

    pub(crate) fn field(&self, level: i32, pel: i32, chroma_shift: (u32, u32)) -> Field {
        let l = self.layout;
        let blk_x = block_count(self.covered_width() >> level, l.overlap_x, l.width);
        let blk_y = block_count(self.covered_height() >> level, l.overlap_y, l.height);
        let fine = level < self.full_math_level;
        let satd = (fine && self.satd & 1 != 0) || (!fine && self.satd & 2 != 0);
        Field::new(
            blk_x,
            blk_y,
            l.layout(),
            if level == 0 { pel } else { 1 },
            level,
            self.levels_skipped + self.levels,
            level == self.levels - 1,
            satd,
            chroma_shift,
        )
    }

    pub(crate) fn gpu_refinable(&self, params: &AnalyseParams) -> bool {
        let settings = self.settings(params, 0, true);
        let l = self.layout;
        params.gpu_mode == 1
            && params.chroma_shift == (1, 1)
            && settings.search_type == SearchType::Exhaustive
            && settings.search_param.abs() <= 3
            && matches!((l.width, l.height), (8, 8) | (16, 8 | 16))
    }

    pub(crate) fn settings(&self, params: &AnalyseParams, level: i32, top: bool) -> SearchSettings {
        let fine = !top && level < self.full_math_level;
        let (search_type, search_param) = if fine {
            (params.search_type, params.search_param)
        } else {
            (self.search_type, self.search_param)
        };
        SearchSettings {
            search_type: SearchType::from_option(search_type),
            search_param,
            lambda: self.lambda,
            lsad: self.lsad,
            pnew: self.pnew,
            plevel: params.plevel,
            pzero: params.pzero,
            pglobal: params.pglobal,
            pnbour: params.pnbour,
            preverse: params.preverse,
            bad_sad: self.bad_sad,
            bad_range: params.bad_range != 0,
            try_many: params.try_many && (top || level > self.full_math_level),
            flags: params.flags,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct AnalyseParams {
    pub(crate) super_params: SuperParams,
    pub(crate) gpu_mode: i32,
    pub(crate) vectors: i32,
    pub(crate) delta: i32,
    pub(crate) flags: i32,
    pub(crate) chroma_shift: (u32, u32),
    search_type: i64,
    search_param: i32,
    plevel: f64,
    pzero: i32,
    pglobal: i32,
    pnbour: i32,
    preverse: i32,
    bad_range: i32,
    try_many: bool,
    pub(crate) sort: bool,
    pub(crate) stages: Vec<Stage>,
}

impl AnalyseParams {
    #[allow(clippy::too_many_lines)]
    pub(crate) fn new(
        options: Option<&Value>,
        super_params: SuperParams,
        chroma_shift: (u32, u32),
    ) -> Result<Self, String> {
        let int =
            |path: &[&str], default: i64| options.and_then(|o| o.int_at(path)).unwrap_or(default);
        let float =
            |path: &[&str], default: f64| options.and_then(|o| o.float_at(path)).unwrap_or(default);
        let flag =
            |path: &[&str], default: bool| options.and_then(|o| o.bool_at(path)).unwrap_or(default);
        let sp = super_params;

        let gpu = int(&["gpu"], -1);
        let from_mvtools = sp.flags > 3 || sp.gpu > 2 || gpu >= 0;
        let gpu_mode = if from_mvtools {
            gpu.max(0) as i32
        } else {
            sp.gpu
        };

        let vectors = int(&["vectors"], 3) as i32;
        if !(1..=3).contains(&vectors) {
            return Err("SVAnalyse: vectors type must be between 1 and 3".into());
        }
        let width = int(&["block", "w"], 16) as i32;
        let height = int(&["block", "h"], i64::from(width)) as i32;
        let divisor = match int(&["block", "overlap"], 2) {
            0 => 0,
            1 => 8,
            2 => 4,
            3 => 2,
            _ => {
                return Err(
                    "SVAnalyse: overlap must be 0, 1 (=1/8*block), 2 (=1/4*block) or 3 (=1/2*block)".into(),
                );
            }
        };
        if !matches!(
            (width, height),
            (4 | 8, 4) | (8 | 16, 8) | (16 | 32, 16) | (32, 32)
        ) {
            return Err("SVAnalyse: Block's size must be 8x8, 16x8, 16x16, 32x16, 32x32".into());
        }
        let overlap = |size: i32| {
            if divisor == 0 || width == 4 {
                0
            } else {
                size / divisor
            }
        };
        let mut layout = BlockLayoutParams {
            width,
            height,
            overlap_x: overlap(width),
            overlap_y: overlap(height),
        };
        let blk_x = block_count(sp.width, layout.overlap_x, layout.width);
        let blk_y = block_count(sp.height, layout.overlap_y, layout.height);
        let covered_w = layout.step_x() * blk_x + layout.overlap_x;
        let covered_h = layout.step_y() * blk_y + layout.overlap_y;
        let mut levels_max = 0;
        while block_count(covered_w >> levels_max, layout.overlap_x, layout.width) > 0
            && block_count(covered_h >> levels_max, layout.overlap_y, layout.height) > 0
        {
            levels_max += 1;
        }
        let requested = int(&["main", "levels"], 0) as i32;
        let levels = if requested > 0 {
            requested
        } else {
            levels_max + requested
        };
        if levels > sp.levels {
            return Err("SVAnalyse: it is not enough levels  in super clip".into());
        }
        if levels < 2 || levels > levels_max {
            return Err("SVAnalyse: non-valid number of levels".into());
        }

        let mut block_mul = width * height * 2;
        let lambda = (float(&["main", "penalty", "lambda"], 10.0) * 1000.0 * f64::from(block_mul)
            / f64::from(BLOCK_DIV)) as i32;
        let mut lsad = int(&["main", "penalty", "lsad"], 8000) as i32 * block_mul / BLOCK_DIV;
        let plevel = match float(&["main", "penalty", "plevel"], 1.5) {
            p if p <= 0.01 => 1.0,
            p => p,
        };
        let pnew = int(&["main", "penalty", "pnew"], 50) as i32;
        let bad_sad =
            int(&["main", "search", "coarse", "bad", "sad"], 1000) as i32 * block_mul / BLOCK_DIV;
        let search_type = int(&["main", "search", "type"], 4);
        let search_param = int(&["main", "search", "distance"], i64::from(-2 * sp.pel)) as i32;
        let fine_satd = flag(&["main", "search", "satd"], false);
        let coarse_satd = flag(&["main", "search", "coarse", "satd"], true);
        let coarse_width = int(&["main", "search", "coarse", "width"], 1050) as i32;
        let flags = int(&["flags"], 0) as i32;

        let min_level = i32::from(gpu_mode == 2);
        let full_math_level = (min_level..levels)
            .filter(|&i| coarse_width > 0 && covered_w >> i >= coarse_width)
            .map(|i| i + 1)
            .max()
            .unwrap_or(1);
        let mut stages = vec![Stage {
            layout,
            blk_x,
            blk_y,
            levels,
            levels_skipped: levels_max - levels,
            search_type: int(&["main", "search", "coarse", "type"], 4),
            search_param: int(&["main", "search", "coarse", "distance"], 0) as i32,
            lambda,
            lsad,
            pnew,
            satd: u8::from(fine_satd) | (u8::from(coarse_satd) << 1),
            bad_sad,
            full_math_level,
        }];

        let refines = options.and_then(|o| o.array_at(&["refine"])).unwrap_or(&[]);
        let refines = if gpu_mode == 2 || width == 4 {
            &[][..]
        } else {
            refines
        };
        for refine in refines {
            let at = |path: &[&str], default: i64| refine.int_at(path).unwrap_or(default);
            if layout.overlap_x == 1 || layout.overlap_y == 1 {
                return Err("SVAnalyse: Overlap size must be >= 1 after recalcualtion".into());
            }
            if layout.width < 8 || layout.height < 8 {
                return Err("SVAnalyse: Block's size must be >= 4x4 after recalcualtion".into());
            }
            layout = BlockLayoutParams {
                width: layout.width / 2,
                height: layout.height / 2,
                overlap_x: layout.overlap_x / 2,
                overlap_y: layout.overlap_y / 2,
            };
            block_mul /= 4;
            lsad /= 4;
            let threshold = at(&["thsad"], 200) as i32 * block_mul / BLOCK_DIV;
            stages.push(Stage {
                layout,
                blk_x: block_count(sp.width, layout.overlap_x, layout.width),
                blk_y: block_count(sp.height, layout.overlap_y, layout.height),
                levels: 1,
                levels_skipped: 0,
                search_type: at(&["search", "type"], search_type),
                search_param: (at(&["search", "distance"], i64::from(sp.pel)) as i32).max(1),
                lambda: 0,
                lsad,
                pnew: refine
                    .int_at(&["penalty", "pnew"])
                    .map_or(pnew, |v| v as i32),
                satd: u8::from(refine.bool_at(&["search", "satd"]).unwrap_or(fine_satd)),
                bad_sad: if threshold < 0 { i32::MAX } else { threshold },
                full_math_level: 1,
            });
        }

        Ok(Self {
            super_params: sp,
            gpu_mode,
            vectors,
            delta: (int(&["special", "delta"], 1) as i32).max(1),
            flags,
            chroma_shift,
            search_type,
            search_param,
            plevel,
            pzero: int(&["main", "penalty", "pzero"], 100) as i32,
            pglobal: int(&["main", "penalty", "pglobal"], 50) as i32,
            pnbour: int(&["main", "penalty", "pnbour"], 50) as i32,
            preverse: int(&["main", "penalty", "prev"], 0) as i32,
            bad_range: int(&["main", "search", "coarse", "bad", "range"], -24) as i32,
            try_many: flag(&["main", "search", "coarse", "trymany"], false),
            sort: flag(&["main", "search", "sort"], true),
            stages,
        })
    }

    pub(crate) fn min_level(&self) -> i32 {
        i32::from(self.gpu_mode == 2)
    }

    pub(crate) fn output(&self) -> &Stage {
        self.stages.last().expect("main stage")
    }

    pub(crate) fn output_count(&self) -> usize {
        let stage = self.output();
        (stage.blk_x * stage.blk_y) as usize
    }

    pub(crate) fn written_count(&self) -> usize {
        let stage = self.output();
        let l = stage.layout;
        let level = self.min_level();
        let blk_x = block_count(stage.covered_width() >> level, l.overlap_x, l.width);
        let blk_y = block_count(stage.covered_height() >> level, l.overlap_y, l.height);
        (blk_x * blk_y) as usize
    }
}

impl BlockLayoutParams {
    const fn step_x(self) -> i32 {
        self.width - self.overlap_x
    }

    const fn step_y(self) -> i32 {
        self.height - self.overlap_y
    }
}
