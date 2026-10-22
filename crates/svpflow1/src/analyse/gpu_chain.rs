use super::{
    CL_FALSE, CL_SUCCESS, ClUint, Frame, Handle, OpenCl, RefineGpu, SWEEPS, SearchMems, Session,
    Uploaded, Variant, bytes,
};
use crate::analyse::chain::{ChainPlan, LevelPlan, Pyramid, RefinePlan};
use core::ffi::c_void;
use std::ffi::CString;

pub(super) const CHAIN: &str = include_str!("chain.cl");

const FREQ: usize = 16384;
const SORT_SCRATCH: usize = 2 * crate::analyse::chain::RANK_BLOCKS as usize;
const BASE: usize = 64;
const ROLES: usize = 16;
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
const OUT: usize = 8;
const D0: usize = 9;
const D1: usize = 10;
const MINSAD: usize = 11;
const MINIDX: usize = 12;

enum Arg {
    Mem(Handle),
    Int(i32),
    V4([i32; 4]),
    V8([i32; 8]),
}

unsafe fn set_args(cl: &OpenCl, kernel: Handle, args: &[Arg]) -> bool {
    args.iter().enumerate().all(|(index, arg)| {
        let (size, value): (usize, *const c_void) = match arg {
            Arg::Mem(mem) => (size_of::<Handle>(), (&raw const *mem).cast()),
            Arg::Int(v) => (4, (&raw const *v).cast()),
            Arg::V4(v) => (16, v.as_ptr().cast()),
            Arg::V8(v) => (32, v.as_ptr().cast()),
        };
        unsafe { (cl.SetKernelArg)(kernel, index as ClUint, size, value) == CL_SUCCESS }
    })
}

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
        *self.chain.get_or_init(|| self.build(CHAIN))
    }
}

impl Session<'_, '_> {
    unsafe fn ckernel(&mut self, name: &'static str) -> Option<Handle> {
        let program = self.gpu.chain_program()?;
        if let Some(&(_, _, kernel)) = self
            .unit
            .ckernels
            .iter()
            .find(|(p, n, _)| *p == program && *n == name)
        {
            return Some(kernel as Handle);
        }
        let mut err = 0;
        let cname = CString::new(name).ok()?;
        let kernel =
            unsafe { (self.gpu.cl.CreateKernel)(program as Handle, cname.as_ptr(), &raw mut err) };
        if err != CL_SUCCESS || kernel.is_null() {
            return None;
        }
        self.unit.ckernels.push((program, name, kernel as usize));
        Some(kernel)
    }

    unsafe fn launch(
        &mut self,
        name: &'static str,
        args: &[Arg],
        global: usize,
        local: Option<usize>,
        waits: &[Handle],
    ) -> Option<()> {
        let kernel = unsafe { self.ckernel(name)? };
        if !unsafe { set_args(&self.gpu.cl, kernel, args) } {
            return None;
        }
        let global = match local {
            Some(local) => global.next_multiple_of(local),
            None => global.max(1),
        };
        let status = unsafe {
            (self.gpu.cl.EnqueueNDRangeKernel)(
                self.unit.queue as Handle,
                kernel,
                1,
                std::ptr::null(),
                &raw const global,
                local.as_ref().map_or(std::ptr::null(), std::ptr::from_ref),
                waits.len() as ClUint,
                if waits.is_empty() {
                    std::ptr::null()
                } else {
                    waits.as_ptr().cast()
                },
                std::ptr::null_mut(),
            )
        };
        (status == CL_SUCCESS).then_some(())
    }

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
        let mems = up.mems.map(|m| m as Handle);
        let mut last: Handle = std::ptr::null_mut();
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
            let Some(kernel) = (unsafe { self.ckernel("reduce") }) else {
                return false;
            };
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
            let cl = &self.gpu.cl;
            if !unsafe { set_args(cl, kernel, &args) } {
                return false;
            }
            let status = unsafe {
                (cl.EnqueueNDRangeKernel)(
                    self.unit.queue as Handle,
                    kernel,
                    1,
                    std::ptr::null(),
                    &raw const n,
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    if level + 1 == pyramid.levels {
                        (&raw mut last).cast()
                    } else {
                        std::ptr::null_mut()
                    },
                )
            };
            if status != CL_SUCCESS {
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

    unsafe fn level_bufs(&mut self, lp: &LevelPlan, dirs: usize) -> Option<(Lb, bool)> {
        let slabs = Slabs::new(lp);
        let at = |role: usize| BASE + lp.level as usize * ROLES + role;
        let get = |this: &mut Self, role: usize, per: usize| unsafe {
            this.cbuf(at(role), per * dirs).map(|b| b.0)
        };
        let fresh = unsafe { self.cbuf(at(MINSAD), slabs.min * dirs)?.1 };
        let lb = Lb {
            q: get(self, Q, slabs.q)?,
            q2: get(self, Q2, slabs.q)?,
            pred: get(self, PRED, slabs.pred)?,
            rev: get(self, REV, slabs.rev)?,
            res: [get(self, RES0, slabs.out)?, get(self, RES1, slabs.out)?],
            aux: get(self, AUX, slabs.out)?,
            cen: get(self, CEN, slabs.out)?,
            out: get(self, OUT, slabs.out)?,
            dirty: [get(self, D0, slabs.dirty)?, get(self, D1, slabs.dirty)?],
            minsad: get(self, MINSAD, slabs.min)?,
            minidx: get(self, MINIDX, slabs.min)?,
            slabs,
        };
        Some((lb, fresh))
    }

    pub(crate) fn chain(
        &mut self,
        plan: &ChainPlan,
        top_out: &[Vec<[i32; 4]>],
    ) -> Option<Vec<usize>> {
        unsafe { self.chain_inner(plan, top_out) }
    }

    unsafe fn chain_inner(
        &mut self,
        plan: &ChainPlan,
        top_out: &[Vec<[i32; 4]>],
    ) -> Option<Vec<usize>> {
        self.flush();
        self.gpu.chain_program()?;
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
        let events: Vec<Handle> = self
            .frames
            .iter()
            .flatten()
            .map(|f| f.3 as Handle)
            .collect();
        let queue = self.unit.queue as Handle;
        let stale = std::mem::replace(&mut self.unit.chain_stale, true);
        let mut bufs: Vec<Lb> = Vec::new();
        for lp in &plan.levels {
            let (b, fresh) = unsafe { self.level_bufs(lp, dirs)? };
            if fresh || stale {
                let nb = lp.blk_x * lp.blk_y;
                unsafe {
                    self.launch(
                        "pinit",
                        &[
                            Arg::Mem(b.minsad),
                            Arg::Mem(b.minidx),
                            Arg::Int(nb),
                            Arg::Int((b.slabs.min / 4) as i32),
                            Arg::Int(dirs as i32),
                        ],
                        nb as usize * dirs,
                        Some(64),
                        &[],
                    )?;
                }
            }
            bufs.push(b);
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
                    "gest",
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
                )?;
                waits = &[];
                self.launch(
                    "gfin",
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
                    "pred1",
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
                )?;
                self.launch(
                    "pred2",
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
                )?;
                self.launch(
                    "pred3",
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
                )?;
            }
            if lp.rmax > 0 {
                if lp.sorted {
                    let (scratch, _) = unsafe { self.cbuf(SHARED + 2, SORT_SCRATCH * 4 * dirs)? };
                    let mut st_sort = st;
                    st_sort[0] = (c.slabs.out / 16) as i32;
                    unsafe {
                        self.launch(
                            "srank",
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
                        )?;
                    }
                } else {
                    let at = BASE + level * ROLES + REV;
                    let tag = (u64::from(lp.blk_x as u32) << 32) | u64::from(lp.blk_y as u32);
                    if self.unit.buffers[at].tag != tag {
                        unsafe {
                            self.launch(
                                "rank_serp",
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
                    "packout",
                    &[
                        Arg::Mem(out0.out),
                        Arg::Mem(words),
                        Arg::Int(blocks as i32),
                        Arg::V8(st),
                    ],
                    blocks * dirs,
                    Some(64),
                    &[],
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
        let variant = Variant {
            width: lp.width,
            height: lp.height,
            satd: lp.satd,
            shift: 1,
            dev: false,
        };
        let spec = [lp.q[6], lp.q[31], lp.q[32], lp.q[34], 0];
        let program = self.gpu.search_program(variant, lp.rmax, spec)?;
        let border_program = if lp.border > 0 {
            let mut spec = spec;
            spec[4] = 1;
            self.gpu.search_program(variant, lp.rmax, spec)?
        } else {
            program
        };
        let eight = lp.width == 8 && lp.height == 8;
        let main_kernel = unsafe { self.search_kernel(program, false)? };
        let border_kernel = unsafe { self.search_kernel(border_program, false)? };
        let refine_kernel = if eight {
            unsafe { self.search_kernel(program, true) }
        } else {
            None
        };
        let nb = (lp.blk_x * lp.blk_y) as usize;
        unsafe {
            self.launch_search(
                mems,
                (nb, lp.border),
                if lp.q[32] == 0 { SWEEPS } else { 0 },
                (main_kernel, border_kernel, refine_kernel),
                eight,
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
    ) -> Option<()> {
        let blocks = (refine.blk_x * refine.blk_y) as usize;
        let (flags, _) = unsafe { self.cbuf(SHARED + 3, blocks)? };
        let (ovr, _) = unsafe { self.cbuf(SHARED + 4, blocks * 4)? };
        let (counter, fresh) = unsafe { self.cbuf(SHARED + 5, 4)? };
        if fresh {
            let zero = self.hold(&[0u8; 4]);
            if !unsafe { self.transfer(counter, zero, std::ptr::null_mut()) } {
                return None;
            }
        }
        let r = refine;
        let (coarse_x, coarse_y) = (r.geometry[12], r.geometry[13]);
        unsafe {
            self.launch(
                "ovf",
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
            )?;
            self.launch(
                "ovs",
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
            )?;
        }
        let (source, target) = (
            self.frames[src_slot].as_ref()?.2,
            self.frames[ref_slot].as_ref()?.2,
        );
        let variant = Variant {
            width: 4,
            height: 4,
            satd: r.satd,
            shift: r.shift,
            dev: true,
        };
        let program = self.gpu.program(variant)?;
        let kernel =
            if let Some(&(_, kernel)) = self.unit.kernels.iter().find(|(p, _)| *p == program) {
                kernel as Handle
            } else {
                let mut err = 0;
                let kernel = unsafe {
                    (self.gpu.cl.CreateKernel)(program as Handle, c"recalc".as_ptr(), &raw mut err)
                };
                if err != CL_SUCCESS || kernel.is_null() {
                    return None;
                }
                self.unit.kernels.push((program, kernel as usize));
                kernel
            };
        let mut g = r.geometry.map(i64::from);
        g[11] = 1;
        let params: [i64; 41] = [
            source.lengths[0],
            source.lengths[1],
            source.lengths[2],
            source.pitches[0],
            source.pitches[1],
            source.pitches[2],
            target.lengths[0],
            target.lengths[1],
            target.lengths[2],
            target.pitches[0],
            target.pitches[1],
            target.pitches[2],
            target.bases[0],
            target.bases[1],
            target.heights[0],
            target.heights[1],
            0,
            blocks as i64,
            g[0],
            g[1],
            g[2],
            g[3],
            g[4],
            g[5],
            g[6],
            g[7],
            g[8],
            g[9],
            g[10],
            g[11],
            source.bases[0],
            source.bases[1],
            source.bases[2],
            g[12],
            g[13],
            g[14],
            g[15],
            g[16],
            g[17],
            g[18],
            g[19],
        ];
        let params = self.hold(bytes(&params));
        let (params_mem, _) = unsafe { self.cbuf(SHARED + 6, params.len())? };
        if !unsafe { self.transfer(params_mem, params, std::ptr::null_mut()) } {
            return None;
        }
        let mems = [
            src[0],
            src[1],
            src[2],
            reference[0],
            reference[1],
            reference[2],
            params_mem,
            seed,
            words,
            ovr,
        ];
        let mut args: Vec<Arg> = mems.into_iter().map(Arg::Mem).collect();
        args.push(Arg::Int(seed_off));
        args.push(Arg::Int(words_off));
        if !unsafe { set_args(&self.gpu.cl, kernel, &args) } {
            return None;
        }
        let (local, global) = (64usize, blocks.next_multiple_of(64));
        let status = unsafe {
            (self.gpu.cl.EnqueueNDRangeKernel)(
                self.unit.queue as Handle,
                kernel,
                1,
                std::ptr::null(),
                &raw const global,
                &raw const local,
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        (status == CL_SUCCESS).then_some(())
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
                    1,
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

    pub(crate) fn take_words(&self, ticket: usize, out: &mut [u32]) {
        let (index, blocks) = self.tickets[ticket];
        let host = self.unit.buffers[index].host as *const u32;
        out[..2 * blocks].copy_from_slice(unsafe { std::slice::from_raw_parts(host, 2 * blocks) });
    }
}
