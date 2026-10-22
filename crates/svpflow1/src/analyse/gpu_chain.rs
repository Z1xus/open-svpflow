use super::{
    Arg, CL_FALSE, CL_SUCCESS, CL_TRUE, Frame, Handle, RefineGpu, SWEEPS, SearchMems, Session,
    Uploaded, Variant, bytes, compact_params, recalc_args,
};
use crate::analyse::chain::{ChainPlan, LevelPlan, Pyramid, RANK_BLOCKS, RefinePlan};
use std::ptr::null_mut;

pub(super) const CHAIN: &str = include_str!("chain.cl");

const FREQ: usize = 16384;
const SORT_SCRATCH: usize = 2 * RANK_BLOCKS as usize;
const BASE: usize = 64;
const ROLES: usize = 16;
const SCRATCH: usize = 512;
const SHARED: usize = 40;
const WORDS: usize = 48;
const PINNED: usize = 32;

const Q: usize = 0;
const Q2: usize = 1;
const PRED: usize = 2;
const REV: usize = 3;
const RES0: usize = 4;
const RES1: usize = 5;
const AUX: usize = 6;
const CEN: usize = 7;
const OUT: usize = 13;
const D0: usize = 9;
const D1: usize = 10;
const MINSAD: usize = 11;
const MINIDX: usize = 12;

fn slab(bytes: usize) -> usize {
    bytes.next_multiple_of(256)
}

#[derive(Clone, Copy)]
struct Slabs {
    out: usize,
    pred: usize,
    min: usize,
    rev: usize,
    q: usize,
    dirty: usize,
}

impl Slabs {
    fn new(lp: &LevelPlan) -> Self {
        let nb = (lp.blk_x * lp.blk_y) as usize;
        let total = nb + lp.border;
        Self {
            out: slab(total * 16),
            pred: slab(nb * 8),
            min: slab(nb * 4),
            rev: slab(nb * 12),
            q: slab(160),
            dirty: slab(total),
        }
    }

    fn stride(&self) -> [i32; 8] {
        [
            (self.out / 16) as i32,
            (self.pred / 8) as i32,
            (self.min / 4) as i32,
            (self.rev / 8) as i32,
            0,
            0,
            0,
            0,
        ]
    }
}

struct Shared {
    pred: usize,
    res: usize,
    dirty: usize,
    min: usize,
    out: [usize; 2],
}

#[derive(Clone, Copy)]
struct Lb {
    q: Handle,
    q2: Handle,
    pred: Handle,
    rev: Handle,
    res: [Handle; 2],
    aux: Handle,
    cen: Handle,
    out: Handle,
    dirty: [Handle; 2],
    minsad: Handle,
    minidx: Handle,
    slabs: Slabs,
}

impl RefineGpu {
    pub(crate) fn chain_ready(&self) -> bool {
        self.chain_program().is_some()
    }

    fn chain_program(&self) -> Option<usize> {
        *self.chain.get_or_init(|| {
            self.build(&format!(
                "#define FREQ {FREQ}\n#define SR_MAX {RANK_BLOCKS}\n{CHAIN}"
            ))
        })
    }
}

impl Session<'_, '_> {
    unsafe fn cbuf(&mut self, index: usize, size: usize) -> Option<(Handle, bool)> {
        let before = self.unit.buffers.get(index).map_or(0, |b| b.capacity);
        let mem = unsafe { self.gpu.buffer(&mut self.unit, index, size.max(16))? };
        Some((mem, before < size.max(16)))
    }

    pub(super) unsafe fn build_levels(&mut self, up: &mut Uploaded, pyramid: Pyramid) -> bool {
        let Frame { pitches, .. } = up.frame;
        if pitches[1] != pitches[2] || up.frame.bases != [0; 3] {
            return false;
        }
        let Some(chain) = self.gpu.chain_program() else {
            return false;
        };
        let mems = up.mems.map(|m| m as Handle);
        let mut last: Handle = null_mut();
        for level in 1..pyramid.levels {
            let (g, p) = (pyramid.geom(level), pyramid.geom(level - 1));
            let (so_y, so_c, sp_y, sp_c) = if level == 1 {
                (0, 0, pitches[0] as i32, pitches[1] as i32)
            } else {
                let (y, c) = pyramid.offsets(level - 1);
                (
                    (up.coarse[0] + y) as i32,
                    (up.coarse[1] + c) as i32,
                    p.pitch as i32,
                    p.chroma_pitch as i32,
                )
            };
            let (d_y, d_c) = pyramid.offsets(level);
            let n = (g.width * g.height + 2 * g.chroma_width * g.chroma_height) as usize;
            let args = [
                Arg::Mem(mems[0]),
                Arg::Mem(mems[1]),
                Arg::Mem(mems[2]),
                Arg::Mem(mems[0]),
                Arg::Mem(mems[1]),
                Arg::Mem(mems[2]),
                Arg::V4([sp_y, sp_c, g.pitch as i32, g.chroma_pitch as i32]),
                Arg::V4([g.width, g.height, g.chroma_width, g.chroma_height]),
                Arg::V4([
                    so_y,
                    so_c,
                    (up.coarse[0] + d_y) as i32,
                    (up.coarse[1] + d_c) as i32,
                ]),
            ];
            let launched = unsafe {
                self.launch(
                    (chain, c"reduce"),
                    &args,
                    n,
                    None,
                    &[],
                    if level + 1 == pyramid.levels {
                        &raw mut last
                    } else {
                        null_mut()
                    },
                )
            };
            if launched.is_none() {
                return false;
            }
        }
        if !last.is_null() {
            unsafe {
                (self.gpu.cl.ReleaseEvent)(up.event as Handle);
                (self.gpu.cl.Flush)(self.unit.queue as Handle);
            }
            up.event = last as usize;
        }
        true
    }

    unsafe fn level_bufs(
        &mut self,
        lp: &LevelPlan,
        slabs: Slabs,
        shared: &Shared,
        dirs: usize,
    ) -> Option<Lb> {
        let at = |role: usize| BASE + lp.level as usize * ROLES + role;
        let get = |this: &mut Self, index: usize, size: usize| unsafe {
            this.cbuf(index, size).map(|b| b.0)
        };
        let scratch = |role: usize| SCRATCH + role;
        Some(Lb {
            q: get(self, at(Q), slabs.q * dirs)?,
            q2: get(self, at(Q2), slabs.q * dirs)?,
            pred: get(self, scratch(PRED), shared.pred)?,
            rev: get(self, at(REV), slabs.rev * dirs)?,
            res: [
                get(self, scratch(RES0), shared.res)?,
                get(self, scratch(RES1), shared.res)?,
            ],
            aux: get(self, scratch(AUX), shared.res)?,
            cen: get(self, scratch(CEN), shared.res)?,
            out: get(
                self,
                scratch(OUT + lp.level as usize % 2),
                shared.out[lp.level as usize % 2],
            )?,
            dirty: [
                get(self, scratch(D0), shared.dirty)?,
                get(self, scratch(D1), shared.dirty)?,
            ],
            minsad: get(self, scratch(MINSAD), shared.min)?,
            minidx: get(self, scratch(MINIDX), shared.min)?,
            slabs,
        })
    }

    pub(crate) fn chain(
        &mut self,
        plan: &ChainPlan,
        top_out: &[Vec<[i32; 4]>],
    ) -> Option<Vec<usize>> {
        self.flush();
        let chain = self.gpu.chain_program()?;
        let dirs = 1 + usize::from(plan.both);
        let top = plan.top();
        let mut infos = Vec::new();
        for slot in 0..2 {
            let frame = self.frames[slot].as_ref()?;
            let state = frame.0.state.lock().ok()?;
            let up = state.as_ref()?;
            if up.pyramid != Some(plan.pyramid) {
                return None;
            }
            infos.push((frame.1.map(|m| m as Handle), up.coarse));
        }
        if infos[0].1 != infos[1].1 || infos[0].1[1] != infos[0].1[2] {
            return None;
        }
        let events: Vec<Handle> = self.upload_events();
        let queue = self.unit.queue as Handle;
        let stale = std::mem::replace(&mut self.unit.chain_stale, true);
        let all: Vec<Slabs> = plan.levels.iter().map(Slabs::new).collect();
        let most = |size: fn(&Slabs) -> usize| all.iter().map(size).max().unwrap_or(0) * dirs;
        let parity = |p: usize| {
            all.iter()
                .skip(p)
                .step_by(2)
                .map(|s| s.out)
                .max()
                .unwrap_or(0)
                * dirs
        };
        let shared = Shared {
            pred: most(|s| s.pred),
            res: most(|s| s.out),
            dirty: most(|s| s.dirty),
            min: most(|s| s.min),
            out: [parity(0), parity(1)],
        };
        let mut bufs: Vec<Lb> = Vec::new();
        for (lp, slabs) in plan.levels.iter().zip(&all) {
            bufs.push(unsafe { self.level_bufs(lp, *slabs, &shared, dirs)? });
        }
        let ready = &mut self.unit.buffers[SCRATCH + MINSAD].tag;
        let need = shared.min / 4;
        if stale || (*ready as usize) < need {
            *ready = need as u64;
            unsafe {
                self.launch(
                    (chain, c"pinit"),
                    &[
                        Arg::Mem(bufs[0].minsad),
                        Arg::Mem(bufs[0].minidx),
                        Arg::Int(need as i32),
                        Arg::Int(need as i32),
                        Arg::Int(1),
                    ],
                    need,
                    Some(64),
                    &[],
                    null_mut(),
                )?;
            }
        }
        for (level, lp) in plan.levels.iter().enumerate() {
            let b = bufs[level];
            let nb = (lp.blk_x * lp.blk_y) as usize;
            for d in 0..dirs {
                let mut q = lp.q;
                if level > 0 {
                    let (y, c) = plan.pyramid.offsets(level as i32);
                    let coarse = infos[0].1;
                    q[10] = (coarse[0] + y) as i32;
                    q[11] = (coarse[1] + c) as i32;
                    q[15] = q[10];
                    q[16] = q[11];
                }
                let mut q2 = q;
                q2[33] = 1;
                let (q, q2) = (self.hold(bytes(&q)), self.hold(bytes(&q2)));
                let at = d * b.slabs.q;
                for (mem, data) in [(b.q, q), (b.q2, q2)] {
                    if !unsafe { self.transfer_at(mem, at, data, std::ptr::null_mut()) } {
                        return None;
                    }
                }
                if level == top {
                    let out = self.hold(bytes(top_out.get(d)?));
                    if out.len() != (nb + lp.border) * 16
                        || !unsafe {
                            self.transfer_at(b.out, d * b.slabs.out, out, std::ptr::null_mut())
                        }
                    {
                        return None;
                    }
                }
            }
        }
        let mut waits: &[Handle] = &events;
        let both = i32::from(plan.both);
        for level in (0..top).rev() {
            let lp = &plan.levels[level];
            let (b, c) = (bufs[level], bufs[level + 1]);
            let nb = (lp.blk_x * lp.blk_y) as i32;
            let coarse = &plan.levels[level + 1];
            let cnb = (coarse.blk_x * coarse.blk_y) as i32;
            let (hist, _) = unsafe { self.cbuf(SHARED, 2 * FREQ * 4)? };
            let (res, _) = unsafe { self.cbuf(SHARED + 1, 16)? };
            unsafe {
                self.launch(
                    (chain, c"gest"),
                    &[
                        Arg::Mem(c.out),
                        Arg::Int((c.slabs.out / 16) as i32),
                        Arg::Mem(hist),
                        Arg::Mem(res),
                        Arg::Int(cnb),
                    ],
                    256 * dirs,
                    Some(256),
                    waits,
                    null_mut(),
                )?;
                waits = &[];
                self.launch(
                    (chain, c"gfin"),
                    &[
                        Arg::Mem(res),
                        Arg::Mem(b.q),
                        Arg::Mem(b.q2),
                        Arg::Int((b.slabs.q / 4) as i32),
                        Arg::Int(lp.pel),
                        Arg::Int(both),
                    ],
                    1,
                    None,
                    &[],
                    null_mut(),
                )?;
            }
            let norm = 3 - lp.pel.ilog2() as i32;
            let geo = [
                lp.blk_x,
                lp.blk_y,
                coarse.blk_x,
                coarse.blk_y,
                lp.step_x,
                lp.step_y,
                lp.width as i32,
                lp.height as i32,
            ];
            let blend = [i32::from(lp.flat), norm, lp.step_x * lp.step_y, 0];
            let mut st = b.slabs.stride();
            st[4] = both;
            let mut st_coarse = st;
            st_coarse[0] = (c.slabs.out / 16) as i32;
            let global = nb as usize * dirs;
            unsafe {
                self.launch(
                    (chain, c"pred1"),
                    &[
                        Arg::Mem(c.out),
                        Arg::Mem(b.pred),
                        Arg::Mem(b.minsad),
                        Arg::V8(geo),
                        Arg::V4(blend),
                        Arg::V8(st_coarse),
                    ],
                    global,
                    Some(64),
                    &[],
                    null_mut(),
                )?;
                self.launch(
                    (chain, c"pred2"),
                    &[
                        Arg::Mem(b.pred),
                        Arg::Mem(b.minsad),
                        Arg::Mem(b.minidx),
                        Arg::V8(geo),
                        Arg::V8(st),
                    ],
                    global,
                    Some(64),
                    &[],
                    null_mut(),
                )?;
                self.launch(
                    (chain, c"pred3"),
                    &[
                        Arg::Mem(b.pred),
                        Arg::Mem(b.minsad),
                        Arg::Mem(b.minidx),
                        Arg::Mem(b.rev),
                        Arg::Int(nb),
                        Arg::V8(st),
                    ],
                    global,
                    Some(64),
                    &[],
                    null_mut(),
                )?;
            }
            if lp.rmax > 0 {
                if lp.sorted {
                    let (scratch, _) = unsafe { self.cbuf(SHARED + 2, SORT_SCRATCH * 4 * dirs)? };
                    let mut st_sort = st;
                    st_sort[0] = (c.slabs.out / 16) as i32;
                    unsafe {
                        self.launch(
                            (chain, c"srank"),
                            &[
                                Arg::Mem(c.out),
                                Arg::Mem(b.rev),
                                Arg::Mem(scratch),
                                Arg::V8([
                                    lp.blk_x,
                                    lp.blk_y,
                                    coarse.blk_x,
                                    coarse.blk_y,
                                    0,
                                    0,
                                    0,
                                    0,
                                ]),
                                Arg::V4([plan.sort_shift, plan.sort_buckets, 0, 0]),
                                Arg::V8(st_sort),
                            ],
                            1024 * dirs,
                            Some(1024),
                            &[],
                            null_mut(),
                        )?;
                    }
                    self.unit.buffers[BASE + level * ROLES + REV].tag = 0;
                } else {
                    let at = BASE + level * ROLES + REV;
                    let tag = (u64::from(lp.blk_x as u32) << 32)
                        | u64::from(lp.blk_y as u32)
                        | (dirs as u64) << 62;
                    if self.unit.buffers[at].tag != tag {
                        unsafe {
                            self.launch(
                                (chain, c"rank_serp"),
                                &[
                                    Arg::Mem(b.rev),
                                    Arg::Int(nb),
                                    Arg::Int(lp.blk_x),
                                    Arg::Int(lp.blk_y),
                                    Arg::V8(st),
                                ],
                                global,
                                Some(64),
                                &[],
                                null_mut(),
                            )?;
                        }
                        self.unit.buffers[at].tag = tag;
                    }
                }
            }
            let mems = SearchMems {
                src: infos[0].0,
                reference: infos[1].0,
                q: b.q,
                q2: b.q2,
                pred: b.pred,
                rev: b.rev,
                res: b.res,
                aux: b.aux,
                cen: b.cen,
                out: b.out,
                dirty: b.dirty,
                dirs,
                strides: [
                    0,
                    (b.slabs.q / 4) as i32,
                    (b.slabs.pred / 8) as i32,
                    (b.slabs.rev / 8) as i32,
                    (b.slabs.out / 16) as i32,
                    b.slabs.dirty as i32,
                    0,
                    0,
                ],
            };
            unsafe { self.chain_search(lp, &mems)? };
        }
        let blocks = plan.output_blocks();
        let words_slab = slab(blocks * 8);
        let (words, _) = unsafe { self.cbuf(WORDS, words_slab * dirs)? };
        let out0 = bufs[0];
        let mut st = out0.slabs.stride();
        st[4] = both;
        st[5] = (words_slab / 8) as i32;
        match &plan.refine {
            None => unsafe {
                self.launch(
                    (chain, c"packout"),
                    &[
                        Arg::Mem(out0.out),
                        Arg::Mem(words),
                        Arg::Int(blocks as i32),
                        Arg::V8(st),
                    ],
                    blocks * dirs,
                    Some(64),
                    &[],
                    null_mut(),
                )?;
            },
            Some(refine) => {
                for d in 0..dirs {
                    let (src_slot, ref_slot) = if d == 0 { (0, 1) } else { (1, 0) };
                    let offsets = (
                        (d * out0.slabs.out / 16) as i32,
                        (d * words_slab / 8) as i32,
                    );
                    unsafe {
                        self.chain_refine(
                            refine,
                            (out0.out, words),
                            offsets,
                            (infos[src_slot].0, infos[ref_slot].0),
                            (src_slot, ref_slot),
                            stale,
                        )?;
                    }
                }
            }
        }
        let mut tickets = Vec::new();
        for d in 0..dirs {
            let host = unsafe { self.gpu.pinned(&mut self.unit, PINNED + d, blocks * 8)? };
            let mut done: Handle = std::ptr::null_mut();
            if unsafe {
                (self.gpu.cl.EnqueueReadBuffer)(
                    queue,
                    words,
                    CL_FALSE,
                    d * words_slab,
                    blocks * 8,
                    host.cast(),
                    0,
                    std::ptr::null(),
                    (&raw mut done).cast(),
                )
            } != CL_SUCCESS
            {
                return None;
            }
            if d + 1 == dirs {
                self.pending.push(done as usize);
            } else {
                unsafe { (self.gpu.cl.ReleaseEvent)(done) };
            }
            self.tickets.push((PINNED + d, blocks));
            tickets.push(self.tickets.len() - 1);
        }
        self.unit.chain_stale = false;
        Some(tickets)
    }

    unsafe fn chain_search(&mut self, lp: &LevelPlan, mems: &SearchMems) -> Option<()> {
        let programs = self.gpu.search_pair(
            Variant::new(lp.width, lp.height, lp.satd),
            lp.rmax,
            &lp.q,
            lp.border,
        )?;
        let nb = (lp.blk_x * lp.blk_y) as usize;
        unsafe {
            self.launch_search(
                mems,
                (nb, lp.border),
                if lp.q[32] == 0 { SWEEPS } else { 0 },
                programs,
                lp.width == 8 && lp.height == 8,
                &[],
            )
        }
    }

    unsafe fn chain_refine(
        &mut self,
        refine: &RefinePlan,
        (seed, words): (Handle, Handle),
        (seed_off, words_off): (i32, i32),
        (src, reference): ([Handle; 3], [Handle; 3]),
        (src_slot, ref_slot): (usize, usize),
        stale: bool,
    ) -> Option<()> {
        let chain = self.gpu.chain_program()?;
        let blocks = (refine.blk_x * refine.blk_y) as usize;
        let (flags, _) = unsafe { self.cbuf(SHARED + 3, blocks)? };
        let (ovr, _) = unsafe { self.cbuf(SHARED + 4, blocks * 4)? };
        let (counter, fresh) = unsafe { self.cbuf(SHARED + 5, 4)? };
        if fresh || stale {
            let zero = self.hold(&[0u8; 4]);
            if !unsafe { self.transfer(counter, zero, null_mut()) } {
                return None;
            }
        }
        let r = refine;
        let (coarse_x, coarse_y) = (r.geometry[12], r.geometry[13]);
        unsafe {
            self.launch(
                (chain, c"ovf"),
                &[
                    Arg::Mem(seed),
                    Arg::Mem(flags),
                    Arg::Mem(ovr),
                    Arg::Mem(counter),
                    Arg::V8([
                        r.blk_x,
                        r.blk_y,
                        r.step_x,
                        r.step_y,
                        coarse_x,
                        coarse_y,
                        i32::from(r.flat),
                        r.norm,
                    ]),
                    Arg::V8([
                        r.odd_even[0],
                        r.odd_even[1],
                        r.odd_even[2],
                        r.odd_even[3],
                        r.bad_sad,
                        r.frame.0,
                        r.frame.1,
                        r.pel,
                    ]),
                    Arg::V4([r.norm_area, seed_off, 0, 0]),
                ],
                blocks,
                Some(64),
                &[],
                null_mut(),
            )?;
            self.launch(
                (chain, c"ovs"),
                &[
                    Arg::Mem(flags),
                    Arg::Mem(ovr),
                    Arg::Mem(counter),
                    Arg::Int(r.blk_x),
                    Arg::Int(r.blk_y),
                ],
                256,
                Some(256),
                &[],
                null_mut(),
            )?;
        }
        let (source, target) = (
            self.frames[src_slot].as_ref()?.2,
            self.frames[ref_slot].as_ref()?.2,
        );
        let program = self.gpu.program(Variant {
            shift: r.shift,
            dev: true,
            ..Variant::new(4, 4, r.satd)
        })?;
        let mut geometry = r.geometry;
        geometry[11] = 1;
        let params = self.hold(bytes(&compact_params(&source, &target, blocks, geometry)));
        let (params_mem, _) = unsafe { self.cbuf(SHARED + 6, params.len())? };
        if !unsafe { self.transfer(params_mem, params, null_mut()) } {
            return None;
        }
        let args = recalc_args(
            src,
            reference,
            &[
                Arg::Mem(params_mem),
                Arg::Mem(seed),
                Arg::Mem(words),
                Arg::Mem(ovr),
                Arg::Int(seed_off),
                Arg::Int(words_off),
            ],
        );
        unsafe {
            self.launch(
                (program, c"recalc"),
                &args,
                blocks,
                Some(64),
                &[],
                null_mut(),
            )
        }
    }

    pub(crate) fn host_levels(
        &mut self,
        slot: usize,
        plan: &ChainPlan,
    ) -> Option<std::sync::Arc<[Vec<u8>; 3]>> {
        self.flush();
        let frame = self.frames[slot].as_ref()?;
        let mut state = frame.0.state.lock().ok()?;
        let up = state.as_mut()?;
        if up.pyramid != Some(plan.pyramid) {
            return None;
        }
        if let Some(host) = &up.host {
            return Some(host.clone());
        }
        let start = plan.top() as i32;
        let (first_y, first_c) = plan.pyramid.offsets(start);
        let (end_y, end_c) = plan.pyramid.totals();
        let queue = self.unit.queue as Handle;
        let wait = up.event as Handle;
        let mut host: [Vec<u8>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for (plane, data) in host.iter_mut().enumerate() {
            let (from, to) = if plane == 0 {
                (first_y, end_y)
            } else {
                (first_c, end_c)
            };
            *data = vec![0u8; to - from];
            let status = unsafe {
                (self.gpu.cl.EnqueueReadBuffer)(
                    queue,
                    up.mems[plane] as Handle,
                    CL_TRUE,
                    up.coarse[plane] + from,
                    to - from,
                    data.as_mut_ptr().cast(),
                    1,
                    (&raw const wait).cast(),
                    std::ptr::null_mut(),
                )
            };
            if status != CL_SUCCESS {
                return None;
            }
        }
        let host = std::sync::Arc::new(host);
        up.host = Some(host.clone());
        Some(host)
    }
}
