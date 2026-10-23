use super::field::{Mv, SearchGeom};
use super::params::AnalyseParams;
use super::planes::{LevelFrame, LevelSource, PlaneView, SuperFrameView, plane_size};

pub(crate) const RANK_BLOCKS: i32 = 8192;
const LEVEL_ALIGN: usize = 256;
const LEVEL_SLACK: usize = 128;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct Pyramid {
    pub(crate) width: i32,
    pub(crate) height: i32,
    pub(crate) levels: i32,
    pub(crate) pitch: usize,
    pub(crate) chroma_pitch: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct LevelGeom {
    pub(crate) width: i32,
    pub(crate) height: i32,
    pub(crate) pitch: usize,
    pub(crate) chroma_width: i32,
    pub(crate) chroma_height: i32,
    pub(crate) chroma_pitch: usize,
}

impl Pyramid {
    pub(crate) fn geom(&self, level: i32) -> LevelGeom {
        let width = plane_size(self.width, level);
        let height = plane_size(self.height, level);
        LevelGeom {
            width,
            height,
            pitch: self.pitch,
            chroma_width: width >> 1,
            chroma_height: height >> 1,
            chroma_pitch: self.chroma_pitch,
        }
    }

    fn sizes(&self, level: i32) -> (usize, usize) {
        let g = self.geom(level);
        (
            (g.pitch * g.height.max(1) as usize + LEVEL_SLACK).next_multiple_of(LEVEL_ALIGN),
            (g.chroma_pitch * g.chroma_height.max(1) as usize + LEVEL_SLACK)
                .next_multiple_of(LEVEL_ALIGN),
        )
    }

    pub(crate) fn slack(&self, level: i32, luma: bool) -> usize {
        let g = self.geom(level);
        let (y, c) = self.sizes(level);
        if luma {
            y - g.pitch * g.height.max(1) as usize
        } else {
            c - g.chroma_pitch * g.chroma_height.max(1) as usize
        }
    }

    pub(crate) fn offsets(&self, level: i32) -> (usize, usize) {
        (1..level).fold((0, 0), |(y, c), l| {
            let (sy, sc) = self.sizes(l);
            (y + sy, c + sc)
        })
    }

    pub(crate) fn totals(&self) -> (usize, usize) {
        self.offsets(self.levels)
    }
}

pub(crate) struct LevelPlan {
    pub(crate) level: i32,
    pub(crate) blk_x: i32,
    pub(crate) blk_y: i32,
    pub(crate) q: [i32; 40],
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) rmax: usize,
    pub(crate) satd: bool,
    pub(crate) border: usize,
    pub(crate) pel: i32,
    pub(crate) step_x: i32,
    pub(crate) step_y: i32,
    pub(crate) flat: bool,
    pub(crate) sorted: bool,
}

pub(crate) struct RefinePlan {
    pub(crate) geometry: [i32; 20],
    pub(crate) satd: bool,
    pub(crate) shift: u32,
    pub(crate) blk_x: i32,
    pub(crate) blk_y: i32,
    pub(crate) step_x: i32,
    pub(crate) step_y: i32,
    pub(crate) bad_sad: i32,
    pub(crate) frame: (i32, i32),
    pub(crate) odd_even: [i32; 4],
    pub(crate) norm: i32,
    pub(crate) norm_area: i32,
    pub(crate) flat: bool,
    pub(crate) pel: i32,
}

pub(crate) struct ChainPlan {
    pub(crate) levels: Vec<LevelPlan>,
    pub(crate) pyramid: Pyramid,
    pub(crate) both: bool,
    pub(crate) lean: bool,
    pub(crate) refine: Option<RefinePlan>,
    pub(crate) sort_shift: i32,
    pub(crate) sort_buckets: i32,
}

impl ChainPlan {
    pub(crate) fn top(&self) -> usize {
        self.levels.len() - 1
    }

    pub(crate) fn host_first(&self) -> i32 {
        (self.top() as i32 - 1).max(1)
    }

    pub(crate) fn output_blocks(&self) -> usize {
        match &self.refine {
            Some(r) => (r.blk_x * r.blk_y) as usize,
            None => (self.levels[0].blk_x * self.levels[0].blk_y) as usize,
        }
    }

    pub(crate) fn new(
        params: &AnalyseParams,
        first: &SuperFrameView<'_>,
        second: &SuperFrameView<'_>,
    ) -> Option<Self> {
        let sp = params.super_params;
        let main = &params.stages[0];
        if params.min_level() != 0
            || params.stages.len() > 2
            || sp.custom_scale
            || (!sp.lean && (first.source.is_some() || second.source.is_some()))
            || !main.gpu_searchable(params)
            || main.levels < 2
            || sp.levels < main.levels
        {
            return None;
        }
        let (l0, l1) = (first.level(0), second.level(0));
        if l0.y.pitch != l1.y.pitch
            || l0.u.pitch != l1.u.pitch
            || l0.u.pitch != l0.v.pitch
            || l1.u.pitch != l1.v.pitch
            || l0.y.base() != 0
            || l0.u.base() != 0
            || l1.y.base() != 0
            || l1.u.base() != 0
        {
            return None;
        }
        let top = params.exact_floor(main.levels - 1);
        let pyramid = Pyramid {
            width: sp.width,
            height: sp.height,
            levels: if sp.lean { main.levels } else { top },
            pitch: l0.y.pitch,
            chroma_pitch: l0.u.pitch,
        };
        let mut levels = Vec::new();
        let mut lambda0 = 0;
        for level in 0..=top {
            let field = main.field(level, sp.pel, params.chroma_shift, true);
            let mut settings = main.settings(params, level, level == top);
            if level == top {
                settings.pnew = 0;
                settings.lambda = 0;
            }
            let geom = if level == 0 {
                SearchGeom {
                    width: l0.y.width,
                    height: l0.y.height,
                    src_pitch: [l0.y.pitch as i32, l0.u.pitch as i32],
                    src_base: [0, 0],
                    ref_pitch: [l1.y.pitch as i32, l1.u.pitch as i32],
                    ref_base: [0, 0],
                    ref_height: [l1.y.height, l1.u.height],
                }
            } else {
                let g = pyramid.geom(level);
                SearchGeom {
                    width: g.width,
                    height: g.height,
                    src_pitch: [g.pitch as i32, g.chroma_pitch as i32],
                    src_base: [0, 0],
                    ref_pitch: [g.pitch as i32, g.chroma_pitch as i32],
                    ref_base: [0, 0],
                    ref_height: [g.height, g.chroma_height],
                }
            };
            let mut lambda = 0;
            let gp =
                field.gpu_search_q(&geom, &settings, Mv::default(), Some(&mut lambda), true)?;
            if level == 0 {
                lambda0 = lambda;
            }
            let layout = main.layout;
            let sorted = level < top && params.sort && level >= main.full_math_level;
            levels.push(LevelPlan {
                level,
                blk_x: field.blk_x,
                blk_y: field.blk_y,
                q: gp.q,
                width: gp.width,
                height: gp.height,
                rmax: gp.rmax,
                satd: gp.satd,
                border: gp.border,
                pel: if level == 0 { sp.pel } else { 1 },
                step_x: layout.width - layout.overlap_x,
                step_y: layout.height - layout.overlap_y,
                flat: layout.overlap_x == 0 && layout.overlap_y == 0,
                sorted,
            });
        }
        for pair in levels.windows(2) {
            let (fine, coarse) = (&pair[0], &pair[1]);
            if fine.sorted {
                let layout = main.layout;
                let add_x = fine.blk_x - coarse.blk_x * 2;
                let add_y = fine.blk_y - coarse.blk_y * 2;
                if coarse.blk_x * coarse.blk_y > RANK_BLOCKS
                    || add_x > 2
                    || add_y > 2
                    || layout.width * layout.height < 2
                {
                    return None;
                }
            }
        }
        let area = main.layout.width * main.layout.height;
        let log = (area * 256 / 2048).ilog2() as i32;
        let refine = if params.stages.len() == 2 {
            let stage = &params.stages[1];
            let refined = stage.field(0, sp.pel, params.chroma_shift, true);
            let coarse = main.field(0, sp.pel, params.chroma_shift, true);
            let mut settings = stage.settings(params, 0, true);
            settings.lambda = lambda0;
            if !refined.compact_eligible(&settings) || sp.pel > 2 {
                return None;
            }
            let (w, h) = (l0.y.width, l0.y.height);
            let interp = refined.interpolation(&coarse);
            Some(RefinePlan {
                geometry: refined.compact_geometry(&coarse, w, h, &settings),
                satd: refined.uses_satd(),
                shift: params.chroma_shift.0,
                blk_x: refined.blk_x,
                blk_y: refined.blk_y,
                step_x: interp.step_x,
                step_y: interp.step_y,
                bad_sad: settings.bad_sad,
                frame: (w, h),
                odd_even: [interp.odd_x, interp.even_x, interp.odd_y, interp.even_y],
                norm: interp.norm,
                norm_area: interp.norm_area,
                flat: interp.flat,
                pel: sp.pel,
            })
        } else {
            None
        };
        Some(Self {
            levels,
            pyramid,
            both: params.vectors == 3,
            lean: sp.lean,
            refine,
            sort_shift: log + 1,
            sort_buckets: (area * 255) >> log,
        })
    }
}

pub(crate) struct HostFrame<'a> {
    pub(crate) pyramid: Pyramid,
    pub(crate) first: i32,
    pub(crate) planes: [&'a [u8]; 3],
}

impl LevelSource for HostFrame<'_> {
    fn level(&self, level: i32) -> LevelFrame<'_> {
        let g = self.pyramid.geom(level);
        let rows = |shift: u32| -> usize {
            (self.first..level)
                .map(|l| (plane_size(self.pyramid.height, l) >> shift) as usize)
                .sum()
        };
        LevelFrame {
            y: PlaneView::new(
                self.planes[0],
                rows(0) * g.pitch,
                g.pitch,
                g.width,
                g.height,
                1,
            ),
            u: PlaneView::new(
                self.planes[1],
                rows(1) * g.chroma_pitch,
                g.chroma_pitch,
                g.chroma_width,
                g.chroma_height,
                1,
            ),
            v: PlaneView::new(
                self.planes[2],
                rows(1) * g.chroma_pitch,
                g.chroma_pitch,
                g.chroma_width,
                g.chroma_height,
                1,
            ),
        }
    }
}
