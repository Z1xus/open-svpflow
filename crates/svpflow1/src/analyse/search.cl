#define CW (W / 2)
#define CH (H / 2)
#define AREA (W * H)
#define CAREA (CW * CH)
#define LO(v) (((v) << 16) >> 16)
#define HI(v) ((v) >> 16)
#define PK(x, y) (((x) & 0xFFFF) | ((y) << 16))

#define P_BLKX 0
#define P_BLKY 1
#define P_STEPX 2
#define P_STEPY 3
#define P_W 4
#define P_H 5
#define P_PEL 6
#define P_SPY 7
#define P_SPU 8
#define P_SBY 10
#define P_SBU 11
#define P_RPY 13
#define P_RPU 14
#define P_RBY 15
#define P_RBU 16
#define P_RHY 17
#define P_RHU 18
#define P_LAMBDA 19
#define P_LSAD 20
#define P_PNEW 21
#define P_PZERO 22
#define P_PGLOBAL 23
#define P_PNBOUR 24
#define P_PREVERSE 25
#define P_PARAM 26
#define P_ADAPT 27
#define P_GX 28
#define P_GY 29
#define P_DIAG 30
#define P_LOGA 31
#define P_DONT 32
#define P_MODE 33
#define P_BLUR 34

#if defined(__NV_CL_C_VERSION) && !defined(NO_NV_ASM)
#define NVK 1
#define BPW 2
inline uint sad4(uint a, uint b, uint c) {
    uint r;
    asm("vabsdiff4.u32.u32.u32.add %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}
inline uint funnel(uint lo, uint hi, uint sft) {
    uint r;
    asm("shf.r.wrap.b32 %0,%1,%2,%3;" : "=r"(r) : "r"(lo), "r"(hi), "r"(sft));
    return r;
}
inline void wsync(void) { asm volatile("bar.warp.sync 0xffffffff;" ::: "memory"); }
inline uint shfl_x(uint v, uint m, __local uint *s) {
    uint r;
    asm volatile("shfl.sync.bfly.b32 %0,%1,%2,0x1f,0xffffffff;" : "=r"(r) : "r"(v), "r"(m));
    return r;
}
inline uint shfl_i(uint v, uint i, __local uint *s) {
    uint r;
    asm volatile("shfl.sync.idx.b32 %0,%1,%2,0x1f,0xffffffff;" : "=r"(r) : "r"(v), "r"(i));
    return r;
}
inline uint ballot(bool p, __local uint *s) {
    uint r;
    asm volatile("{ .reg .pred q; setp.ne.u32 q,%1,0; vote.sync.ballot.b32 %0,q,0xffffffff; }" : "=r"(r) : "r"((uint)p));
    return r;
}
inline uint match_any(uint v, __local uint *s) {
    uint r;
    asm volatile("match.any.sync.b32 %0,%1,0xffffffff;" : "=r"(r) : "r"(v));
    return r;
}
#ifdef NO_REDUX
inline uint radd(uint v, __local uint *s) {
    for (uint m = 16; m; m >>= 1) v += shfl_x(v, m, s);
    return v;
}
inline int wmin(int v, __local uint *s) {
    for (uint m = 16; m; m >>= 1) v = min(v, (int)shfl_x((uint)v, m, s));
    return v;
}
#else
inline uint radd(uint v, __local uint *s) {
    uint r;
    asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(v));
    return r;
}
inline int wmin(int v, __local uint *s) {
    int r;
    asm volatile("redux.sync.min.s32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(v));
    return r;
}
#endif
#else
#define BPW 1
inline uint sad4(uint a, uint b, uint c) {
    uchar4 d = abs_diff(as_uchar4(a), as_uchar4(b));
    return c + d.x + d.y + d.z + d.w;
}
inline uint funnel(uint lo, uint hi, uint sft) {
    return sft ? (lo >> sft) | (hi << (32 - sft)) : lo;
}
inline void wsync(void) { barrier(CLK_LOCAL_MEM_FENCE); }
inline uint shfl_i(uint v, uint i, __local uint *s) {
    uint l = get_local_id(0) & 31;
    wsync();
    s[l] = v;
    wsync();
    uint r = s[i];
    wsync();
    return r;
}
inline uint shfl_x(uint v, uint m, __local uint *s) { return shfl_i(v, (get_local_id(0) & 31) ^ m, s); }
inline uint ballot(bool p, __local uint *s) {
    uint l = get_local_id(0) & 31;
    wsync();
    s[l] = p;
    wsync();
    uint r = 0;
    for (int i = 0; i < 32; i++) r |= (s[i] ? 1u : 0u) << i;
    wsync();
    return r;
}
inline uint match_any(uint v, __local uint *s) {
    uint l = get_local_id(0) & 31;
    wsync();
    s[l] = v;
    wsync();
    uint r = 0;
    for (int i = 0; i < 32; i++) r |= (s[i] == v ? 1u : 0u) << i;
    wsync();
    return r;
}
inline uint radd(uint v, __local uint *s) {
    for (uint m = 16; m; m >>= 1) v += shfl_x(v, m, s);
    return v;
}
inline int wmin(int v, __local uint *s) {
    for (uint m = 16; m; m >>= 1) v = min(v, (int)shfl_x((uint)v, m, s));
    return v;
}
#endif
#define WGS (32 * BPW)
inline int wsum(int v, __local uint *s) { return (int)radd((uint)v, s); }

#define LW (W / 4)
#define LTR (LW / 2)
#define LT (H * LTR)
#define CWW (CW / 4)
#define CTW (CWW < 2 ? CWW : 2)
#define CTR (CWW / CTW)
#define CT (H * CTR)
#define CB (2 * LT)
#define SSW (2 * LT + CT * CTW)
#define NCHL ((W + 2 * RMAX + 30) / 16)
#define NCHC ((CW + RMAX + 30) / 16)
#define RSL (NCHL * 4)
#define RSC (NCHC * 4)
#ifdef LANEG
#define ROWSL 1
#define ROWSC 1
#else
#define ROWSL (H + 2 * RMAX)
#define ROWSC (CH + RMAX)
#endif
#define NSMAX ((2 * RMAX + 1) * (2 * RMAX + 1))
#define IMAX 2147483647
#ifdef SATD
#define SLOWFN satd_at
#else
#define SLOWFN sad_slow
#endif
#ifndef PELV
#define PELV q[P_PEL]
#endif
#ifndef LOGAV
#define LOGAV q[P_LOGA]
#endif
#ifndef BLURV
#define BLURV q[P_BLUR]
#endif
#ifndef BORDERV
#define BORDERV q[P_MODE]
#endif
#ifndef ADAPTV
#define ADAPTV q[P_ADAPT]
#endif

#if W == 16 && H == 16 && !defined(SATD)
#define FASTREG 1
#elif W == 8 && H == 8
#define LANEG 1
#endif

#ifndef NSLOT
#define NSLOT 2
#endif
#define VV (2 * NSLOT)

typedef struct {
    int px, py, cpx, cpy, min_x, max_x, min_y, max_y;
} Sh;

inline long subpel(long base, long pitch, long height, int pel, int x, int y) {
    if (pel == 1) return base + (long)y * pitch + x;
    int bits = pel == 2 ? 1 : 2;
    int mask = pel == 2 ? 1 : 3;
    long index = (long)((x & mask) | ((y & mask) << bits));
    return base + (long)(y >> bits) * pitch + (x >> bits) + index * pitch * height;
}

inline Sh make_sh(int x0, int y0, int pel, int pw, int ph, int vx, int vy, int margin) {
    Sh s;
    int omin_x = -pel * x0, omin_y = -pel * y0;
    int omax_x = pel * (pw - x0 - W), omax_y = pel * (ph - y0 - H);
    if (margin > 0) {
        vx = vx < 0 ? vx - margin : vx + margin;
        vy = vy < 0 ? vy - margin : vy + margin;
    }
    int dx = vx < omin_x ? vx - omin_x : vx >= omax_x ? vx - omax_x + 1 : 0;
    int dy = vy < omin_y ? vy - omin_y : vy >= omax_y ? vy - omax_y + 1 : 0;
    s.px = x0; s.py = y0; s.cpx = x0 >> 1; s.cpy = y0 >> 1;
    s.min_x = omin_x; s.max_x = omax_x; s.min_y = omin_y; s.max_y = omax_y;
    if (dx == 0 && dy == 0) return s;
    int x = clamp(x0 - dx, 0, pw - W);
    int y = clamp(y0 - dy, 0, ph - H);
    dx = x0 - x;
    dy = y0 - y;
    s.px = x; s.py = y;
    s.cpx = (x0 >> 1) - (dx >> 1);
    s.cpy = (y0 >> 1) - (dy >> 1);
    s.max_x = omax_x - max(dx, 0); s.min_x = omin_x - min(dx, 0);
    s.max_y = omax_y - max(dy, 0); s.min_y = omin_y - min(dy, 0);
    return s;
}

inline bool sh_inside(const Sh *s, int vx, int vy) {
    return vx >= s->min_x && vx < s->max_x && vy >= s->min_y && vy < s->max_y;
}

inline bool sh_fits(const Sh *s, int pel, int pw, int ph, int vx, int vy) {
    int x = s->px * pel + vx, y = s->py * pel + vy;
    return x >= 0 && x <= pel * (pw - W) && y >= 0 && y <= pel * (ph - H);
}

inline void sad_slow(__global const uchar *gy, __global const uchar *gu, __global const uchar *gv,
                     __global const uchar *ry, __global const uchar *ru, __global const uchar *rv,
                     __constant int *q, const Sh *s, int pel, int vx, int vy, int *luma, int *chroma) {
    uint rp = q[P_RPY], cp = q[P_RPU];
    uint off = (uint)subpel(q[P_RBY], rp, q[P_RHY], pel, s->px * pel + vx, s->py * pel + vy);
    uint sb = (uint)q[P_SBY] + (uint)s->py * (uint)q[P_SPY] + (uint)s->px;
    uint acc = 0;
    #pragma unroll 1
    for (int j = 0; j < H; j++) {
        uint so = sb + (uint)j * (uint)q[P_SPY], ro = off + (uint)j * rp;
        __global const uint *sw = (__global const uint *)(gy + (so & ~3u));
        __global const uint *rw = (__global const uint *)(ry + (ro & ~3u));
        uint ss_ = (so & 3u) * 8u, rs_ = (ro & 3u) * 8u;
        uint sp_ = sw[0], rpv = rw[0];
        #pragma unroll
        for (int i = 0; i < W / 4; i++) {
            uint sn = sw[i + 1], rn = rw[i + 1];
            acc = sad4(funnel(sp_, sn, ss_), funnel(rpv, rn, rs_), acc);
            sp_ = sn;
            rpv = rn;
        }
    }
    *luma = (int)acc;
    uint coff = (uint)subpel(q[P_RBU], cp, q[P_RHU], pel, s->cpx * pel + (vx >> 1), s->cpy * pel + (vy >> 1));
    uint cb = (uint)q[P_SBU] + (uint)s->cpy * (uint)q[P_SPU] + (uint)s->cpx;
    uint cacc = 0;
    #pragma unroll 1
    for (int j = 0; j < CH; j++) {
        uint so = cb + (uint)j * (uint)q[P_SPU], ro = coff + (uint)j * cp;
        uint sa = so & ~3u, ra = ro & ~3u;
        uint ss_ = (so & 3u) * 8u, rs_ = (ro & 3u) * 8u;
        __global const uint *uw = (__global const uint *)(gu + sa);
        __global const uint *vw = (__global const uint *)(gv + sa);
        __global const uint *ruw = (__global const uint *)(ru + ra);
        __global const uint *rvw = (__global const uint *)(rv + ra);
        uint up = uw[0], vp = vw[0], rup = ruw[0], rvp = rvw[0];
        #pragma unroll
        for (int i = 0; i < CW / 4; i++) {
            uint un = uw[i + 1], vn = vw[i + 1], run = ruw[i + 1], rvn = rvw[i + 1];
            cacc = sad4(funnel(up, un, ss_), funnel(rup, run, rs_), cacc);
            cacc = sad4(funnel(vp, vn, ss_), funnel(rvp, rvn, rs_), cacc);
            up = un; vp = vn; rup = run; rvp = rvn;
        }
    }
    *chroma = (int)cacc << 2;
}

#ifdef SATD
#if W == 32
#define TROW(t, r) ((t) >> 1)
#define TCOL(t, r) (2 * (r) + ((t) & 1))
#define TPIX(t, y, x) (((t) >> 1) * W + 8 * (y) + 4 * ((t) & 1) + (x))
#else
#define TROW(t, r) (((t) / (W / 4)) * 4 + (r))
#define TCOL(t, r) ((t) % (W / 4))
#define TPIX(t, y, x) ((((t) / (W / 4)) * 4 + (y)) * W + ((t) % (W / 4)) * 4 + (x))
#endif
inline int had4x4(int *d) {
    int t[16];
    #pragma unroll
    for (int y = 0; y < 4; y++) {
        int p0 = d[y * 4], p1 = d[y * 4 + 1], p2 = d[y * 4 + 2], p3 = d[y * 4 + 3];
        int s0 = p0 + p1, s1 = p2 + p3, d0 = p0 - p1, d1 = p2 - p3;
        t[y * 4] = s0 + s1; t[y * 4 + 1] = s0 - s1; t[y * 4 + 2] = d0 + d1; t[y * 4 + 3] = d0 - d1;
    }
    int sum = 0;
    #pragma unroll
    for (int x = 0; x < 4; x++) {
        int s0 = t[x] + t[4 + x], s1 = t[8 + x] + t[12 + x], d0 = t[x] - t[4 + x], d1 = t[8 + x] - t[12 + x];
        sum += abs(s0 + s1) + abs(s0 - s1) + abs(d0 + d1) + abs(d0 - d1);
    }
    return sum;
}

inline void satd_at(__global const uchar *gy, __global const uchar *gu, __global const uchar *gv,
                    __global const uchar *ry, __global const uchar *ru, __global const uchar *rv,
                    __constant int *q, const Sh *s, int pel, int vx, int vy, int *luma, int *chroma) {
    long rp = q[P_RPY], cp = q[P_RPU];
    long off = subpel(q[P_RBY], rp, q[P_RHY], pel, s->px * pel + vx, s->py * pel + vy);
    long sb = q[P_SBY] + (long)s->py * q[P_SPY] + s->px;
    long sp = q[P_SPY];
    int total = 0;
    for (int t = 0; t < (H / 4) * (W / 4); t++) {
        int d[16];
        #pragma unroll
        for (int y = 0; y < 4; y++)
            #pragma unroll
            for (int x = 0; x < 4; x++) {
                int pix = TPIX(t, y, x);
                int py = pix / W, px = pix % W;
                d[y * 4 + x] = (int)gy[sb + py * sp + px] - (int)ry[off + py * rp + px];
            }
        total += had4x4(d) / 2;
    }
    *luma = total;
    long coff = subpel(q[P_RBU], cp, q[P_RHU], pel, s->cpx * pel + (vx >> 1), s->cpy * pel + (vy >> 1));
    long cs = q[P_SBU] + (long)s->cpy * q[P_SPU] + s->cpx;
    long csp = q[P_SPU];
    int tu = 0, tv = 0;
    for (int ty = 0; ty < CH; ty += 4)
        for (int tx = 0; tx < CW; tx += 4) {
            int du[16], dv[16];
            #pragma unroll
            for (int y = 0; y < 4; y++)
                #pragma unroll
                for (int x = 0; x < 4; x++) {
                    long so = cs + (ty + y) * csp + tx + x;
                    long ro = coff + (ty + y) * cp + tx + x;
                    du[y * 4 + x] = (int)gu[so] - (int)ru[ro];
                    dv[y * 4 + x] = (int)gv[so] - (int)rv[ro];
                }
            tu += had4x4(du) / 2;
            tv += had4x4(dv) / 2;
        }
    *chroma = (tu + tv) << 2;
}
#endif

inline int pen(int sad, int pnew) { return (int)((uint)sad + (uint)(((int)((uint)pnew * (uint)sad)) >> 8)); }

inline int motion(int lambda, int cx, int cy, int vx, int vy) {
    int dx = cx - vx, dy = cy - vy;
    return (int)(((uint)lambda * ((uint)(dx * dx) + (uint)(dy * dy)))) >> 8;
}

inline int scaled_lambda(int lambda, int lsad, int psad) {
    float ratio = (float)lsad / (float)(lsad + (psad >> 1));
    uint bits = as_uint(ratio);
    ulong mant = (bits & 0x7FFFFFu) | 0x800000u;
    int shift = 2 * (150 - (int)((bits >> 23) & 0xFFu));
    if (shift < 0) return 21474836;
    ulong p1 = (ulong)(uint)lambda * mant;
    ulong lo = p1 * mant, hi = mul_hi(p1, mant);
    ulong r;
    if (shift >= 128) r = 0;
    else if (shift >= 64) r = hi >> (shift - 64);
    else if (shift == 0) r = lo;
    else r = (lo >> shift) | (hi << (64 - shift));
    return (int)min(r, 21474836UL);
}

inline void emit(__global int4 *out, int b, int x, int y, int sad, int avg, int diag) {
    int relative = (abs(x) + abs(y)) * 100 / diag;
    int s = sad;
    if (relative > 50) s = (int)((uint)sad * 20u);
    else if (relative > 15) s = (int)((uint)sad + (uint)(((int)((uint)sad * (uint)(relative - 15))) / 35 * 9));
    uint score = ((uint)s & 0x00FFFFFFu) + ((uint)avg << 24);
    out[b] = (int4)(PK(x, y), sad, (int)(((uint)x << 16) + (uint)y), (int)score);
}

inline int blurred(__local const uchar *sy, int i) {
    const int shift = W + 1;
    int other = i < AREA - shift ? i + shift : i - (AREA - shift);
    return ((int)sy[i] + (int)sy[other]) >> 1;
}

inline int nbx(int i) { return i < 3 ? i - 1 : i < 5 ? (i == 3 ? -1 : 1) : i - 6; }
inline int nby(int i) { return i < 3 ? -1 : i < 5 ? 0 : 1; }

inline void stage(__local uint4 *dst, int stride4, __global const uchar *buf, uint off0, uint pitch, int rows, int nch, int l) {
    int total = rows * nch;
    uint m = 0xFFFFFFFFu / (uint)nch + 1u;
    for (int j = l; j < total; j += 32) {
        int row = (int)mul_hi((uint)j, m);
        int ch = j - row * nch;
        uint a = ((off0 + (uint)row * pitch) & ~15u) + 16u * (uint)ch;
        dst[row * stride4 + ch] = *(__global const uint4 *)(buf + a);
    }
}

inline void stage2(__local uint4 *dst, int stride4, int plane4, __global const uchar *b0, __global const uchar *b1,
                   uint off0, uint pitch, int rows, int nch, int l) {
    int per = rows * nch, total = 2 * per;
    uint m = 0xFFFFFFFFu / (uint)nch + 1u;
    for (int j = l; j < total; j += 32) {
        int p = j >= per;
        int jj = j - p * per;
        int row = (int)mul_hi((uint)jj, m);
        int ch = jj - row * nch;
        uint a = ((off0 + (uint)row * pitch) & ~15u) + 16u * (uint)ch;
        __global const uchar *buf = p ? b1 : b0;
        dst[p * plane4 + row * stride4 + ch] = *(__global const uint4 *)(buf + a);
    }
}

#ifdef SATD
#define TL ((H / 4) * (W / 4))
#define TC ((CH / 4) * (CW / 4))
inline __attribute__((always_inline)) void tasks(__local const uint *ss, __local const uint *stl, __local const uint *stc,
                  uint g, const uint G, int lrow, uint lcol, int crow, uint ccol, uint *accL, uint *accC) {
    uint sl = (lcol & 3) * 8, wc = lcol >> 2;
    uint a = 0;
    #pragma unroll 1
    for (uint i = 0; i < (TL + G - 1) / G; i++) {
        uint t = g + i * G;
        if (t < TL) {
            int d[16];
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                __local const uint *w = stl + (TROW(t, r) + lrow) * RSL + wc + TCOL(t, r);
                uint rw = funnel(w[0], w[1], sl);
                uint sw = ss[TROW(t, r) * (W / 4) + TCOL(t, r)];
                int4 dd = convert_int4(as_uchar4(sw)) - convert_int4(as_uchar4(rw));
                d[r * 4] = dd.x; d[r * 4 + 1] = dd.y; d[r * 4 + 2] = dd.z; d[r * 4 + 3] = dd.w;
            }
            a += (uint)(had4x4(d) / 2);
        }
    }
    uint sc = (ccol & 3) * 8, cc = ccol >> 2;
    uint b = 0;
    #pragma unroll 1
    for (uint i = 0; i < (2 * TC + G - 1) / G; i++) {
        uint t = g + i * G;
        if (t < 2 * TC) {
            uint p = t / TC, idx = t % TC;
            uint ty = idx / (CW / 4), tx = idx % (CW / 4);
            int d[16];
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                __local const uint *w = stc + p * (ROWSC * RSC) + (ty * 4 + r + crow) * RSC + cc + tx;
                uint rw = funnel(w[0], w[1], sc);
                uint sw = ss[CB + p * (CAREA / 4) + (ty * 4 + r) * (CW / 4) + tx];
                int4 dd = convert_int4(as_uchar4(sw)) - convert_int4(as_uchar4(rw));
                d[r * 4] = dd.x; d[r * 4 + 1] = dd.y; d[r * 4 + 2] = dd.z; d[r * 4 + 3] = dd.w;
            }
            b += (uint)(had4x4(d) / 2);
        }
    }
    *accL = a;
    *accC = b;
}

#else
inline __attribute__((always_inline)) void tasks(__local const uint *ss, __local const uint *stl, __local const uint *stc,
                  uint g, const uint G, int lrow, uint lcol, int crow, uint ccol, uint *accL, uint *accC) {
    uint sl = (lcol & 3) * 8, wc = lcol >> 2;
    uint a = 0;
    #pragma unroll 4
    for (uint i = 0; i < (LT + G - 1) / G; i++) {
        uint t = g + i * G;
        if (t < LT) {
            uint row = t / LTR, grp = t % LTR;
            __local const uint *w = stl + (row + lrow) * RSL + wc + grp * 2;
            uint a0 = w[0], a1 = w[1], a2 = w[2];
            a = sad4(ss[2 * t], funnel(a0, a1, sl), a);
            a = sad4(ss[2 * t + 1], funnel(a1, a2, sl), a);
        }
    }
    uint sc = (ccol & 3) * 8, cc = ccol >> 2;
    uint b = 0;
    #pragma unroll 4
    for (uint i = 0; i < (CT + G - 1) / G; i++) {
        uint t = g + i * G;
        if (t < CT) {
            uint pr = t / CTR, grp = t % CTR;
            uint p = pr / CH, cr = pr % CH;
            __local const uint *w = stc + p * (ROWSC * RSC) + (cr + crow) * RSC + cc + grp * CTW;
            uint a0 = w[0], a1 = w[1];
            b = sad4(ss[CB + t * CTW], funnel(a0, a1, sc), b);
            if (CTW == 2) b = sad4(ss[CB + t * CTW + 1], funnel(a1, w[2], sc), b);
        }
    }
    *accL = a;
    *accC = b;
}

#endif

inline void cand_sad(__global const uchar *ry, __global const uchar *ru, __global const uchar *rv, __constant int *q,
                     int x0, int y0, int pel,
                     __local const uint *ss, __local uint4 *stl, __local uint4 *stc, __local uint *scr,
                     int vx, int vy, int *sl, int *sc, int l) {
    uint off0 = (uint)subpel(q[P_RBY], q[P_RPY], q[P_RHY], pel, x0 * pel + vx, y0 * pel + vy);
    uint coff0 = (uint)subpel(q[P_RBU], q[P_RPU], q[P_RHU], pel, (x0 >> 1) * pel + (vx >> 1), (y0 >> 1) * pel + (vy >> 1));
    wsync();
    stage(stl, NCHL, ry, off0, (uint)q[P_RPY], H, (W + 30) / 16, l);
    stage2(stc, NCHC, ROWSC * NCHC, ru, rv, coff0, (uint)q[P_RPU], CH, (CW + 30) / 16, l);
    wsync();
    uint al, ac;
    tasks(ss, (__local const uint *)stl, (__local const uint *)stc, l, 32, 0, off0 & 15u, 0, coff0 & 15u, &al, &ac);
    *sl = (int)radd(al, scr);
    *sc = (int)radd(ac, scr) << 2;
}

#ifdef LANEG
inline uint2 grp_part(__global const uchar *ry, __global const uchar *ru, __global const uchar *rv, __constant int *q,
                      int x0, int y0, int pel, const uint *sr, uint r, int vx, int vy) {
    uint rp = q[P_RPY], cp = q[P_RPU];
    uint off0 = (uint)subpel(q[P_RBY], q[P_RPY], q[P_RHY], pel, x0 * pel + vx, y0 * pel + vy);
    uint coff0 = (uint)subpel(q[P_RBU], q[P_RPU], q[P_RHU], pel, (x0 >> 1) * pel + (vx >> 1), (y0 >> 1) * pel + (vy >> 1));
#ifdef SATD
    bool lum = r < 4;
    uint o0 = lum ? off0 + (r >> 1) * 4u * rp + (r & 1u) * 4u : coff0;
    uint pitch = lum ? rp : cp;
    __global const uchar *rb = lum ? ry : (r & 1u) ? rv : ru;
    int d[16];
    #pragma unroll
    for (int rr = 0; rr < 4; rr++) {
        uint o = o0 + (uint)rr * pitch;
        __global const uint *pw = (__global const uint *)(rb + (o & ~3u));
        uint rw = funnel(pw[0], pw[1], (o & 3u) * 8u);
        int4 dd = convert_int4(as_uchar4(sr[rr])) - convert_int4(as_uchar4(rw));
        d[rr * 4] = dd.x; d[rr * 4 + 1] = dd.y; d[rr * 4 + 2] = dd.z; d[rr * 4 + 3] = dd.w;
    }
    uint th = (uint)(had4x4(d) / 2);
    return (uint2)(lum ? th : 0u, (r == 4u || r == 5u) ? th : 0u);
#else
    uint o = off0 + r * rp;
    __global const uint *p = (__global const uint *)(ry + (o & ~3u));
    uint sh = (o & 3u) * 8u, c0 = p[0], c1 = p[1], c2 = p[2];
    uint al = sad4(sr[1], funnel(c1, c2, sh), sad4(sr[0], funnel(c0, c1, sh), 0u));
    uint co = coff0 + (r & 3u) * cp;
    __global const uint *pc = (__global const uint *)(((r & 4u) ? rv : ru) + (co & ~3u));
    uint ac = sad4(sr[2], funnel(pc[0], pc[1], (co & 3u) * 8u), 0u);
    return (uint2)(al | (ac << 16), 0u);
#endif
}

inline uint2 grp_sum(uint2 v, __local uint *scr) {
    v.x += shfl_x(v.x, 4, scr);
#ifdef SATD
    v.y += shfl_x(v.y, 4, scr);
#endif
    v.x += shfl_x(v.x, 2, scr);
#ifdef SATD
    v.y += shfl_x(v.y, 2, scr);
#endif
    v.x += shfl_x(v.x, 1, scr);
#ifdef SATD
    v.y += shfl_x(v.y, 1, scr);
#endif
    return v;
}

inline void grp_split(uint2 v, int *luma, int *chroma) {
#ifdef SATD
    *luma = (int)v.x;
    *chroma = (int)v.y << 2;
#else
    *luma = (int)(v.x & 0xFFFFu);
    *chroma = (int)(v.x >> 16) << 2;
#endif
}

inline void grp_eval(__global const uchar *ry, __global const uchar *ru, __global const uchar *rv, __constant int *q,
                     int x0, int y0, int pel, const uint *sr, uint r, int vx, int vy, __local uint *scr, int *luma, int *chroma) {
    grp_split(grp_sum(grp_part(ry, ru, rv, q, x0, y0, pel, sr, r, vx, vy), scr), luma, chroma);
}

inline uint grp_src(__global const uchar *sy, __global const uchar *su, __global const uchar *sv, __constant int *q,
                    int x0, int y0, uint r, uint *sr) {
    uint spy = q[P_SPY], spu = q[P_SPU];
#ifdef SATD
    bool lum = r < 4;
    uint o0 = lum ? (uint)q[P_SBY] + (uint)(y0 + (int)(r >> 1) * 4) * spy + (uint)x0 + (r & 1u) * 4u
                  : (uint)q[P_SBU] + (uint)(y0 >> 1) * spu + (uint)(x0 >> 1);
    uint pitch = lum ? spy : spu;
    __global const uchar *sb = lum ? sy : (r & 1u) ? sv : su;
    #pragma unroll
    for (int rr = 0; rr < 4; rr++) {
        uint o = o0 + (uint)rr * pitch;
        __global const uint *pw = (__global const uint *)(sb + (o & ~3u));
        sr[rr] = funnel(pw[0], pw[1], (o & 3u) * 8u);
    }
    return lum ? sad4(sr[3], 0u, sad4(sr[2], 0u, sad4(sr[1], 0u, sad4(sr[0], 0u, 0u)))) : 0u;
#else
    uint so = (uint)q[P_SBY] + (uint)(y0 + (int)r) * spy + (uint)x0;
    __global const uint *p = (__global const uint *)(sy + (so & ~3u));
    uint sh = (so & 3u) * 8u, c0 = p[0], c1 = p[1], c2 = p[2];
    sr[0] = funnel(c0, c1, sh);
    sr[1] = funnel(c1, c2, sh);
    uint co = (uint)q[P_SBU] + (uint)((y0 >> 1) + (int)(r & 3u)) * spu + (uint)(x0 >> 1);
    __global const uint *pc = (__global const uint *)(((r & 4u) ? sv : su) + (co & ~3u));
    sr[2] = funnel(pc[0], pc[1], (co & 3u) * 8u);
    sr[3] = 0;
    return sad4(sr[1], 0u, sad4(sr[0], 0u, 0u));
#endif
}
#endif

#ifdef FASTREG
inline uint2 glu(uint4 v, uint off, uint h, __local uint *scr) {
    uint s = off & 15u, q = s >> 2;
    uint r0 = shfl_x(h ? v.x : v.z, 1, scr), r1 = shfl_x(h ? v.y : v.w, 1, scr);
    uint u0 = h ? r0 : v.x, u1 = h ? r1 : v.y, u2 = h ? v.x : v.z, u3 = h ? v.y : v.w, u4 = h ? v.z : r0, u5 = h ? v.w : r1;
    uint t0 = (q & 1) ? u1 : u0, t1 = (q & 1) ? u2 : u1, t2 = (q & 1) ? u3 : u2, t3 = (q & 1) ? u4 : u3, t4 = (q & 1) ? u5 : u4;
    uint a0 = (q & 2) ? t2 : t0, a1 = (q & 2) ? t3 : t1, a2 = (q & 2) ? t4 : t2;
    uint sft = (s & 3) * 8;
    return (uint2)(funnel(a0, a1, sft), funnel(a1, a2, sft));
}

inline uint gcu(uint2 v, uint off, uint h, __local uint *scr) {
    uint s = off & 7u, q = s >> 2;
    uint e = shfl_x(h ? v.x : v.y, 1, scr);
    uint lo = h ? (q ? v.x : e) : (q ? v.y : v.x);
    uint hi = h ? (q ? v.y : v.x) : (q ? e : v.y);
    return funnel(lo, hi, (s & 3) * 8);
}

typedef struct {
    uint4 vl[NSLOT];
    uint2 vc[NSLOT];
    uint offl[NSLOT], offc[NSLOT];
    int vx[NSLOT], vy[NSLOT];
    bool act[NSLOT];
} Batch;

inline void issue_batch(Batch *B, __global const uchar *ry, __global const uchar *ru, __global const uchar *rv,
                        uint bl, uint bc, uint pyi, uint pui, uint rh, uint cplane, int cnt, const bool *slow) {
    #pragma unroll
    for (int s = 0; s < NSLOT; s++) {
        B->act[s] = s < cnt;
        B->offl[s] = bl + (uint)(B->vy[s] * (int)pyi + B->vx[s]);
        B->offc[s] = bc + (uint)((B->vy[s] >> 1) * (int)pui + (B->vx[s] >> 1));
        B->vl[s] = (uint4)(0);
        B->vc[s] = (uint2)(0);
        if (s < cnt && !slow[s]) {
            B->vl[s] = *(__global const uint4 *)(ry + (B->offl[s] & ~15u) + 16u * rh);
            B->vc[s] = *(__global const uint2 *)((cplane ? rv : ru) + (B->offc[s] & ~7u) + 8u * rh);
        }
    }
}

inline void finish_batch(const Batch *B, uint s0, uint s1, uint scw, uint rh, __local uint *scr, int *rl, int *rc) {
    uint v[VV];
    #pragma unroll
    for (int s = 0; s < NSLOT; s++) {
        uint pl = 0, pc = 0;
        if (B->act[s]) {
            uint2 g = glu(B->vl[s], B->offl[s], rh, scr);
            uint gc = gcu(B->vc[s], B->offc[s], rh, scr);
            pl = sad4(s1, g.y, sad4(s0, g.x, 0));
            pc = sad4(scw, gc, 0);
        }
        v[2 * s] = pl;
        v[2 * s + 1] = pc;
    }
    #pragma unroll
    for (int s = 0; s < NSLOT; s++) {
        rl[s] = (int)radd(v[2 * s], scr);
        rc[s] = (int)radd(v[2 * s + 1], scr) << 2;
    }
}
#endif

#if DONTV == 1 && defined(LANEG)
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void search(__global const uchar *sY, __global const uchar *sU, __global const uchar *sV,
            __global const uchar *rY, __global const uchar *rU, __global const uchar *rV,
            __constant int *q, __global const int2 *pred, __global const int2 *rev,
            __global const int4 *rin, __global int4 *rout,
            __global int4 *cen, __global int4 *aux, __global int4 *out,
            __global const uchar *din, __global uchar *dout, int pass) {
    const int lid = get_local_id(0), l = lid & 31, sub = lid >> 5;
    __local uint scr_[BPW][32];
    __local uint *scr = scr_[sub];
    const uint gr = l & 7u, gg = l >> 3;
    const int bxn = q[P_BLKX], byn = q[P_BLKY], nb = bxn * byn;
    const int border = BORDERV;
    const int nbt = border ? bxn + 1 + byn : nb;
    const int bq = (int)(get_group_id(0) * BPW + sub) * 4 + (int)gg;
    const bool act = bq < nbt;
    const int b = act ? bq : 0;
    int bi, bj, x0, y0, pidx;
    if (!border) {
        bi = b % bxn;
        bj = b / bxn;
        x0 = bi * q[P_STEPX];
        y0 = bj * q[P_STEPY];
        pidx = b;
    } else {
        if (b <= bxn) { bi = b; bj = byn; } else { bi = bxn; bj = b - bxn - 1; }
        x0 = bi < bxn ? bi * q[P_STEPX] : q[P_W] - W;
        y0 = bj < byn ? bj * q[P_STEPY] : q[P_H] - H;
        pidx = min(bi, bxn - 1) + min(bj, byn - 1) * bxn;
    }
    const int ob = border ? nb + b : b;
    const int pel = PELV;
    const int PW = q[P_W], PH = q[P_H];
    const int2 pv = pred[pidx];
    const int vx = LO(pv.x), vy = HI(pv.x);
    const int omin_x = -pel * x0, omin_y = -pel * y0;
    const int omax_x = pel * (PW - x0 - W), omax_y = pel * (PH - y0 - H);
    bool valid = act, shifted = false;
    Sh t;
    if (!(vx >= omin_x && vx < omax_x && vy >= omin_y && vy < omax_y)) {
        t = make_sh(x0, y0, pel, PW, PH, vx, vy, 0);
        if (!sh_fits(&t, pel, PW, PH, vx, vy)) valid = false;
        else shifted = t.px != x0 || t.py != y0;
    }
    uint sr[4];
    uint pls = grp_src(sY, sU, sV, q, x0, y0, gr, sr);
    uint2 part = (uint2)(0u, 0u);
    if (valid && !shifted) part = grp_part(rY, rU, rV, q, x0, y0, pel, sr, gr, vx, vy);
    part = grp_sum(part, scr);
    pls += shfl_x(pls, 4, scr);
    pls += shfl_x(pls, 2, scr);
    pls += shfl_x(pls, 1, scr);
    if (gr == 0 && act) {
        int sl, sc;
        grp_split(part, &sl, &sc);
        if (valid && shifted) SLOWFN(sY, sU, sV, rY, rU, rV, q, &t, pel, vx, vy, &sl, &sc);
        int rx = 0, ry = 0, rsad = 0;
        if (valid) { rx = vx; ry = vy; rsad = sl + sc; }
        emit(out, ob, rx, ry, rsad, (int)pls >> LOGAV, q[P_DIAG]);
    }
}
#else
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void search(__global const uchar *sY, __global const uchar *sU, __global const uchar *sV,
            __global const uchar *rY, __global const uchar *rU, __global const uchar *rV,
            __constant int *q, __global const int2 *pred, __global const int2 *rev,
            __global const int4 *rin, __global int4 *rout,
            __global int4 *cen, __global int4 *aux, __global int4 *out,
            __global const uchar *din, __global uchar *dout, int pass) {
    const int lid = get_local_id(0), l = lid & 31, sub = lid >> 5;
    __local uint scr_[BPW][32];
    __local uint ss_[BPW][SSW] __attribute__((aligned(16)));
    __local uint4 stl_[BPW][ROWSL * NCHL];
    __local uint4 stc_[BPW][2 * ROWSC * NCHC];
    __local int svk_[BPW][NSMAX];
    __local int svm_[BPW][NSMAX];
    __local uint *scr = scr_[sub];
    __local uint *ss = ss_[sub];
    __local uint4 *stl = stl_[sub];
    __local uint4 *stc = stc_[sub];
    __local int *svk = svk_[sub];
    __local int *svm = svm_[sub];

    const int bxn = q[P_BLKX], byn = q[P_BLKY], nb = bxn * byn;
    __global const int *rk = (__global const int *)(rev + nb);
    const int border = BORDERV;
    const int nbt = border ? bxn + 1 + byn : nb;
    const int b = get_group_id(0) * BPW + sub;
    if (b >= nbt) return;
    int bi, bj, x0, y0, pidx;
    if (!border) {
        bi = b % bxn;
        bj = b / bxn;
        x0 = bi * q[P_STEPX];
        y0 = bj * q[P_STEPY];
        pidx = b;
    } else {
        if (b <= bxn) { bi = b; bj = byn; } else { bi = bxn; bj = b - bxn - 1; }
        x0 = bi < bxn ? bi * q[P_STEPX] : q[P_W] - W;
        y0 = bj < byn ? bj * q[P_STEPY] : q[P_H] - H;
        pidx = min(bi, bxn - 1) + min(bj, byn - 1) * bxn;
    }
    const int ob = border ? nb + b : b;
    const int pel = PELV;
    const int PW = q[P_W], PH = q[P_H];

    int nd = 0, nvec = 0;
    int4 cn = (int4)(0), ax = (int4)(0);
    int2 pv = (int2)(0), pa = (int2)(0), ra = (int2)(0);
    bool inb = false;
    if (pass != 0) {
        int nx = bi + nbx(l), ny = bj + nby(l);
        inb = l < 8 && nx >= 0 && nx < bxn && ny >= 0 && ny < byn;
        int ni = inb ? ny * bxn + nx : b;
        nd = din[ni];
        nvec = rin[ni].x;
        cn = cen[b];
        ax = aux[b];
        if (!ballot(inb && nd != 0, scr)) {
            if (l == 0) {
                rout[ob] = rin[ob];
                dout[ob] = 0;
            }
            return;
        }
        nd = inb && nd != 0;
    } else {
        pv = pred[pidx];
        if (!border) {
            int ii = l - 4;
            int nx = bi + nbx(ii), ny = bj + nby(ii);
            bool in_ = (uint)ii < 8u && nx >= 0 && nx < bxn && ny >= 0 && ny < byn;
            int ni = in_ ? ny * bxn + nx : pidx;
            pa = pred[ni];
            int rn = DONTV ? 0 : rk[ni];
            inb = in_ && (DONTV || rn < rk[b]);
            ra = rev[b];
        }
    }

#ifdef FASTREG
    const uint rrow = l >> 1, rh = l & 1, cplane = l >> 4, crow = (l >> 1) & 7;
    const uint soffL = (uint)q[P_SBY] + (uint)(y0 + (int)rrow) * (uint)q[P_SPY] + (uint)x0;
    const uint soffC = (uint)q[P_SBU] + (uint)((y0 >> 1) + (int)crow) * (uint)q[P_SPU] + (uint)(x0 >> 1);
    uint4 svl = *(__global const uint4 *)(sY + (soffL & ~15u) + 16u * rh);
    uint2 svc = *(__global const uint2 *)((cplane ? sV : sU) + (soffC & ~7u) + 8u * rh);
    uint s0 = 0, s1 = 0, scw = 0;
#endif
    const int omin_x = -pel * x0, omin_y = -pel * y0;
    const int omax_x = pel * (PW - x0 - W), omax_y = pel * (PH - y0 - H);
    const int blur = BLURV;

    int vx = 0, vy = 0, valid = 0;
    if (pass == 0) {
        if (!border) {
            int packed = l == 1 ? ra.x : l == 0 ? pv.x : pa.x;
            vx = LO(packed); vy = HI(packed);
            if (l == 2) { vx = q[P_GX]; vy = q[P_GY]; }
            if (l == 3) { vx = 0; vy = 0; }
            valid = l == 0 || l == 2 || l == 3 || (l == 1 && ra.y > 0) || inb;
            if (DONTV) valid = l == 0;
        } else {
            if (DONTV && l > 0) { valid = 0; }
            else if (l == 0) { vx = LO(pv.x); vy = HI(pv.x); valid = 1; }
            else if (l == 2) { vx = q[P_GX]; vy = q[P_GY]; valid = 1; }
            else if (l == 3) { valid = 1; }
            else if (l >= 4 && l < 7) {
                int k = l - 4;
                if (bj == byn) {
                    int x1 = max(bi - 1, 0);
                    int x2 = bi < bxn ? min(bxn - 1, bi + 1) : bi - 1;
                    int nx = x1 + k;
                    if (nx <= x2) { int nv = rin[(byn - 1) * bxn + nx].x; vx = LO(nv); vy = HI(nv); valid = 1; }
                } else {
                    int y1 = max(bj - 1, 0);
                    int y2 = min(byn - 1, bj + 1);
                    int ny = y1 + k;
                    if (ny <= y2) { int nv = rin[ny * bxn + bxn - 1].x; vx = LO(nv); vy = HI(nv); valid = 1; }
                }
            }
        }
    } else {
        vx = LO(nvec); vy = HI(nvec);
        valid = nd && nvec != cn.x;
    }
    if (valid && !(vx >= omin_x && vx < omax_x && vy >= omin_y && vy < omax_y)) {
        Sh t = make_sh(x0, y0, pel, PW, PH, vx, vy, 0);
        if (!sh_fits(&t, pel, PW, PH, vx, vy)) valid = 0;
    }

    int dupi = -1;
    {
        const uint vmask = ballot(valid != 0, scr);
        uint mm = match_any((uint)PK(vx, vy), scr) & vmask;
        if (valid) {
            int first = 31 - (int)clz(mm & (0u - mm));
            if (first != l) dupi = first;
        }
    }
    uint om = ballot(valid && dupi < 0, scr);
    int mls = 0, mlc = 0;
    uint pl = 0, pb = 0, pu = 0, pw = 0;

#define SUMS_REDUCE() \
    pl = radd(pl, scr); \
    pu = radd(pu, scr); \
    pw = radd(pw, scr); \
    pb = radd(pb, scr);

#ifdef FASTREG
    uint bl, bc;
    if (pel == 1) {
        bl = (uint)q[P_RBY] + (uint)(y0 + (int)rrow) * (uint)q[P_RPY] + (uint)x0;
        bc = (uint)q[P_RBU] + (uint)((y0 >> 1) + (int)crow) * (uint)q[P_RPU] + (uint)(x0 >> 1);
    }
    const uint pyi = (uint)q[P_RPY], pui = (uint)q[P_RPU];
    bool prepped = false;
#define SRCPREP() do { \
        uint2 g_ = glu(svl, soffL, rh, scr); \
        s0 = g_.x; s1 = g_.y; \
        scw = gcu(svc, soffC, rh, scr); \
        pl = sad4(s1, 0u, sad4(s0, 0u, 0u)); \
        if (pass == 0) { uint t_ = sad4(scw, 0u, 0u); pu = cplane ? 0 : t_; pw = cplane ? t_ : 0; } \
        if (blur && pass == 0) { \
            ss[2 * l] = s0; ss[2 * l + 1] = s1; ss[CB + l] = scw; \
            wsync(); \
            for (int i = l; i < AREA; i += 32) pb += blurred((__local const uchar *)ss, i); \
        } \
        SUMS_REDUCE() \
        prepped = true; \
    } while (0)
    while (om) {
        Batch B;
        int cnt = 0, cj[NSLOT];
        bool slowf[NSLOT];
        #pragma unroll
        for (int s = 0; s < NSLOT; s++) {
            cj[s] = 0;
            slowf[s] = false;
            B.vx[s] = 0;
            B.vy[s] = 0;
            if (om) {
                int j = 31 - (int)clz(om & (0u - om));
                om &= om - 1;
                cj[s] = j;
                B.vx[s] = (int)shfl_i((uint)vx, j, scr);
                B.vy[s] = (int)shfl_i((uint)vy, j, scr);
                slowf[s] = !(B.vx[s] >= omin_x && B.vx[s] < omax_x && B.vy[s] >= omin_y && B.vy[s] < omax_y);
                cnt++;
            }
        }
        if (pel == 1) {
            issue_batch(&B, rY, rU, rV, bl, bc, pyi, pui, rh, cplane, cnt, slowf);
        } else {
            #pragma unroll
            for (int s = 0; s < NSLOT; s++) {
                B.act[s] = s < cnt;
                B.vl[s] = (uint4)(0); B.vc[s] = (uint2)(0);
                B.offl[s] = 0; B.offc[s] = 0;
                if (s < cnt && !slowf[s]) {
                    B.offl[s] = (uint)subpel(q[P_RBY], q[P_RPY], q[P_RHY], pel, x0 * pel + B.vx[s], y0 * pel + B.vy[s]) + rrow * pyi;
                    B.offc[s] = (uint)subpel(q[P_RBU], q[P_RPU], q[P_RHU], pel, (x0 >> 1) * pel + (B.vx[s] >> 1), (y0 >> 1) * pel + (B.vy[s] >> 1)) + crow * pui;
                    B.vl[s] = *(__global const uint4 *)(rY + (B.offl[s] & ~15u) + 16u * rh);
                    B.vc[s] = *(__global const uint2 *)((cplane ? rV : rU) + (B.offc[s] & ~7u) + 8u * rh);
                }
            }
        }
        if (!prepped) SRCPREP();
        int rl[NSLOT], rc[NSLOT];
        finish_batch(&B, s0, s1, scw, rh, scr, rl, rc);
        #pragma unroll
        for (int s = 0; s < NSLOT; s++)
            if (s < cnt && l == cj[s]) { mls = rl[s]; mlc = rc[s]; }
    }
    if (!prepped) SRCPREP();
    if (valid && dupi < 0 && !(vx >= omin_x && vx < omax_x && vy >= omin_y && vy < omax_y)) {
        Sh t = make_sh(x0, y0, pel, PW, PH, vx, vy, 0);
        if (t.px != x0 || t.py != y0) sad_slow(sY, sU, sV, rY, rU, rV, q, &t, pel, vx, vy, &mls, &mlc);
    }
#else
    {
        wsync();
        uint off0 = (uint)q[P_SBY] + (uint)y0 * (uint)q[P_SPY] + (uint)x0;
        uint coff0 = (uint)q[P_SBU] + (uint)(y0 >> 1) * (uint)q[P_SPU] + (uint)(x0 >> 1);
#if defined(SATD) || defined(LANEG)
        for (int k = l; k < AREA / 4; k += 32) {
            uint o = off0 + (k / (W / 4)) * (uint)q[P_SPY] + 4 * (k % (W / 4));
            __global const uint *p = (__global const uint *)(sY + (o & ~3u));
            ss[k] = funnel(p[0], p[1], (o & 3) * 8);
        }
        for (int k = l; k < CAREA / 4; k += 32) {
            uint o = coff0 + (k / (CW / 4)) * (uint)q[P_SPU] + 4 * (k % (CW / 4));
            __global const uint *pu_ = (__global const uint *)(sU + (o & ~3u));
            __global const uint *pv_ = (__global const uint *)(sV + (o & ~3u));
            uint sft = (o & 3) * 8;
            ss[CB + k] = funnel(pu_[0], pu_[1], sft);
            ss[CB + CAREA / 4 + k] = funnel(pv_[0], pv_[1], sft);
        }
#else
        stage(stl, NCHL, sY, off0, (uint)q[P_SPY], H, (W + 30) / 16, l);
        stage2(stc, NCHC, ROWSC * NCHC, sU, sV, coff0, (uint)q[P_SPU], CH, (CW + 30) / 16, l);
        wsync();
        {
            __local const uint *stlw = (__local const uint *)stl;
            __local const uint *stcw = (__local const uint *)stc;
            uint sl = (off0 & 3u) * 8, wc = (off0 & 15u) >> 2;
            for (uint t = l; t < LT; t += 32) {
                uint row = t / LTR, grp = t % LTR;
                __local const uint *w = stlw + row * RSL + wc + grp * 2;
                uint a0 = w[0], a1 = w[1], a2 = w[2];
                ss[2 * t] = funnel(a0, a1, sl);
                ss[2 * t + 1] = funnel(a1, a2, sl);
            }
            uint sc = (coff0 & 3u) * 8, cc = (coff0 & 15u) >> 2;
            for (uint t = l; t < CT; t += 32) {
                uint pr = t / CTR, grp = t % CTR;
                uint p = pr / CH, cr = pr % CH;
                __local const uint *w = stcw + p * (ROWSC * RSC) + cr * RSC + cc + grp * CTW;
                uint a0 = w[0], a1 = w[1];
                ss[CB + t * CTW] = funnel(a0, a1, sc);
                if (CTW == 2) ss[CB + t * CTW + 1] = funnel(a1, w[2], sc);
            }
        }
#endif
        wsync();
        for (int k = l; k < 2 * LT; k += 32) pl = sad4(ss[k], 0u, pl);
        if (blur && pass == 0) for (int i = l; i < AREA; i += 32) pb += blurred((__local const uchar *)ss, i);
        if (pass == 0) for (int k = l; k < CAREA / 4; k += 32) {
            pu = sad4(ss[CB + k], 0u, pu);
            pw = sad4(ss[CB + CAREA / 4 + k], 0u, pw);
        }
        SUMS_REDUCE()
    }
#ifdef LANEG
    const uint gr = l & 7u, gg = l >> 3;
    uint sr[4];
#ifdef SATD
    #pragma unroll
    for (int rr = 0; rr < 4; rr++) sr[rr] = gr < 4 ? ss[((gr >> 1) * 4 + rr) * 2 + (gr & 1)] : ss[16 + (gr & 1) * 4 + rr];
#else
    sr[0] = ss[2 * gr]; sr[1] = ss[2 * gr + 1]; sr[2] = ss[16 + gr]; sr[3] = 0;
#endif
    {
        const uint omall = om;
        const uint slot = popcount(omall & ((1u << l) - 1u));
        uint rd = 0;
        while (om) {
            uint js[4];
            int cnt = 0;
            #pragma unroll
            for (int s = 0; s < 4; s++) {
                js[s] = 0;
                if (om) {
                    js[s] = 31u - clz(om & (0u - om));
                    om &= om - 1;
                    cnt++;
                }
            }
            uint jj = gg == 0 ? js[0] : gg == 1 ? js[1] : gg == 2 ? js[2] : js[3];
            int jx = (int)shfl_i((uint)vx, jj, scr), jy = (int)shfl_i((uint)vy, jj, scr);
            int sl = 0, sc = 0;
            bool inside = jx >= omin_x && jx < omax_x && jy >= omin_y && jy < omax_y;
            bool ok = (int)gg < cnt;
            if (ok && !inside) {
                Sh t = make_sh(x0, y0, pel, PW, PH, jx, jy, 0);
                ok = t.px == x0 && t.py == y0;
            }
            grp_eval(rY, rU, rV, q, x0, y0, pel, sr, gr, ok ? jx : 0, ok ? jy : 0, scr, &sl, &sc);
            int ml = (int)shfl_i((uint)sl, (slot & 3u) * 8u, scr), mc = (int)shfl_i((uint)sc, (slot & 3u) * 8u, scr);
            if ((slot >> 2) == rd && ((omall >> l) & 1u)) { mls = ml; mlc = mc; }
            rd++;
        }
    }
    if (valid && dupi < 0 && !(vx >= omin_x && vx < omax_x && vy >= omin_y && vy < omax_y)) {
        Sh t = make_sh(x0, y0, pel, PW, PH, vx, vy, 0);
        if (t.px != x0 || t.py != y0) SLOWFN(sY, sU, sV, rY, rU, rV, q, &t, pel, vx, vy, &mls, &mlc);
    }
#else
    while (om) {
        int j = 31 - (int)clz(om & (0u - om));
        om &= om - 1;
        int jx = (int)shfl_i((uint)vx, j, scr), jy = (int)shfl_i((uint)vy, j, scr);
        int rl, rc;
        Sh t = make_sh(x0, y0, pel, PW, PH, jx, jy, 0);
        if (t.px != x0 || t.py != y0) { rl = 0; rc = 0; }
        else cand_sad(rY, rU, rV, q, x0, y0, pel, ss, stl, stc, scr, jx, jy, &rl, &rc, l);
        if (l == j) { mls = rl; mlc = rc; }
    }
    if (valid && dupi < 0 && !(vx >= omin_x && vx < omax_x && vy >= omin_y && vy < omax_y)) {
        Sh t = make_sh(x0, y0, pel, PW, PH, vx, vy, 0);
        if (t.px != x0 || t.py != y0) SLOWFN(sY, sU, sV, rY, rU, rV, q, &t, pel, vx, vy, &mls, &mlc);
    }
#endif
#endif
    const int avg_l = (int)pl >> LOGAV;
    {
        uint from = dupi >= 0 ? (uint)dupi : (uint)l;
        mls = (int)shfl_i((uint)mls, from, scr);
        mlc = (int)shfl_i((uint)mlc, from, scr);
    }

    int p = IMAX;
    if (valid) {
        int sad = mls + mlc;
        if (pass == 0) {
            int penalty = l == 0 ? 0 : l == 1 ? q[P_PREVERSE] : l == 2 ? q[P_PGLOBAL] : l == 3 ? q[P_PZERO] : q[P_PNBOUR];
            p = (int)((uint)sad + (l == 0 ? (uint)penalty : (uint)(((int)((uint)penalty * (uint)sad)) >> 8)));
        } else {
            p = pen(sad, q[P_PNBOUR]);
        }
    }
    const int pmin = wmin(p, scr);
    const uint wmask = ballot(valid && p == pmin, scr);
    const int widx = wmask ? 31 - (int)clz(wmask & (0u - wmask)) : IMAX;
    bool found = widx != IMAX && (pass == 0 || pmin < cn.z);

    int lambda, radius, cxp, cyp;
    if (pass == 0) {
        int psad = pv.y;
        lambda = scaled_lambda(q[P_LAMBDA], q[P_LSAD], psad);
        radius = q[P_PARAM];
        if (ADAPTV) {
            int avg_c = blur ? (int)pb >> LOGAV : avg_l;
            int au = (int)pu >> (LOGAV - 2);
            int av = (int)pw >> (LOGAV - 2);
            int dl = 0, du = 0, dv = 0;
#ifdef SATD
            __local const uchar *sy = (__local const uchar *)ss;
            __local const uchar *su = (__local const uchar *)(ss + CB);
            __local const uchar *sv = su + CAREA;
            for (int t = l; t < AREA / 16; t += 32) {
                int d[16];
                for (int y = 0; y < 4; y++)
                    for (int x = 0; x < 4; x++) {
                        int i = TPIX(t, y, x);
                        d[y * 4 + x] = (blur ? blurred(sy, i) : (int)sy[i]) - avg_c;
                    }
                dl += had4x4(d) / 2;
            }
            for (int t = l; t < CAREA / 16; t += 32) {
                int tx = (t % (CW / 4)) * 4, ty = (t / (CW / 4)) * 4;
                int d[16], e[16];
                for (int y = 0; y < 4; y++)
                    for (int x = 0; x < 4; x++) {
                        int i = (ty + y) * CW + tx + x;
                        d[y * 4 + x] = (int)su[i] - au;
                        e[y * 4 + x] = (int)sv[i] - av;
                    }
                du += had4x4(d) / 2;
                dv += had4x4(e) / 2;
            }
#else
            if (blur) {
                for (int i = l; i < AREA; i += 32) dl += abs(blurred((__local const uchar *)ss, i) - avg_c);
            } else {
#ifdef FASTREG
                uint a4 = (uint)avg_c * 0x01010101u;
                dl = (int)sad4(s1, a4, sad4(s0, a4, 0u));
#else
                uint a4 = (uint)avg_c * 0x01010101u;
                uint t = 0;
                for (int k = l; k < 2 * LT; k += 32) t = sad4(ss[k], a4, t);
                dl = (int)t;
#endif
            }
            {
#ifdef FASTREG
                uint t_ = sad4(scw, cplane ? (uint)av * 0x01010101u : (uint)au * 0x01010101u, 0u);
                du = cplane ? 0 : (int)t_;
                dv = cplane ? (int)t_ : 0;
#else
                uint au4 = (uint)au * 0x01010101u, av4 = (uint)av * 0x01010101u;
                uint tu = 0, tv = 0;
                for (int k = l; k < CAREA / 4; k += 32) {
                    tu = sad4(ss[CB + k], au4, tu);
                    tv = sad4(ss[CB + CAREA / 4 + k], av4, tv);
                }
                du = (int)tu; dv = (int)tv;
#endif
            }
#endif
            dl = wsum(dl, scr);
            du = wsum(du, scr);
            dv = wsum(dv, scr);
            int contrast = min(dl / AREA + du / CAREA + dv / CAREA, 255);
            radius = contrast < 2 ? 0 : 1 + ((radius * min(contrast * 255 / 150, 255)) >> 8);
        }
        cxp = LO(pv.x); cyp = HI(pv.x);
        {
            int pp0 = (int)shfl_i((uint)p, 0, scr), pp1 = (int)shfl_i((uint)p, 1, scr);
            if (pp1 != IMAX && pp1 < pp0) {
                cxp = (int)shfl_i((uint)vx, 1, scr);
                cyp = (int)shfl_i((uint)vy, 1, scr);
            }
        }
        found = widx != IMAX;
    } else {
        cxp = LO(ax.x); cyp = HI(ax.x);
        lambda = ax.y; radius = ax.z;
    }
    int cvx, cvy, craw, cmin, cvl = 0, cvc = 0;
    if (found) {
        cvx = (int)shfl_i((uint)vx, widx, scr);
        cvy = (int)shfl_i((uint)vy, widx, scr);
        cvl = (int)shfl_i((uint)mls, widx, scr);
        cvc = (int)shfl_i((uint)mlc, widx, scr);
        craw = cvl + cvc;
        cmin = (int)shfl_i((uint)p, widx, scr);
    } else {
        cvx = LO(cn.x); cvy = HI(cn.x); craw = cn.y; cmin = cn.z;
    }
    const bool do_ex = found && !DONTV;
    const int bx = cvx, by = cvy;

    int rst = 0;
    if (do_ex) {
        rst = min(radius, RMAX);
        if (lambda > 0) {
            ulong x = (ulong)((uint)(min(craw, 8000000) * 256 + 255));
            while (rst > 0 && x < (ulong)(rst * rst) * (ulong)lambda) rst--;
        }
    }
    const int n = 2 * rst + 1;

    int bcost = IMAX, bk = IMAX, bsl = 0, bsc = 0;
    if (do_ex) {
        const int pnew = q[P_PNEW];
        const int nk = n * n;
        const Sh cs = make_sh(x0, y0, pel, PW, PH, bx, by, 2);
        const bool shifted = cs.px != x0 || cs.py != y0;
        const int kc = shifted ? -1 : rst * n + rst;
        int ns = 0;
        if (rst > 0 || shifted) {
            for (int kb = 0; kb < nk; kb += 32) {
                int k = kb + l;
                bool ok = false;
                int mot = 0;
                if (k < nk) {
                    int ex_vx = bx + k % n - rst, ex_vy = by + k / n - rst;
                    if (sh_inside(&cs, ex_vx, ex_vy)) {
                        mot = motion(lambda, cxp, cyp, ex_vx, ex_vy);
                        ok = mot < craw;
                    }
                }
                uint m = ballot(ok && k != kc, scr);
                if (ok && k != kc) {
                    int pos = ns + (int)popcount(m & ((1u << l) - 1u));
                    svk[pos] = k;
                    svm[pos] = mot;
                }
                ns += (int)popcount(m);
            }
            wsync();
        }
        if (kc >= 0 && sh_inside(&cs, bx, by)) {
            int mot = motion(lambda, cxp, cyp, bx, by);
            if (mot < craw) {
                int t = (int)((uint)mot + (uint)pen(cvl, pnew));
                t = (int)((uint)t + (uint)pen(cvc, pnew));
                if (t < craw) { bcost = t; bk = kc; bsl = cvl; bsc = cvc; }
            }
        }
        int cbest = bcost, ckbest = bk, csl = bsl, csc = bsc;
        bcost = IMAX; bk = IMAX; bsl = 0; bsc = 0;
#ifdef LANEG
        if (shifted) {
            for (int i0 = 0; i0 < ns; i0 += 32) {
                int i = i0 + l;
                int cst = IMAX, sl = 0, sc = 0, kk = 0;
                if (i < ns) {
                    kk = svk[i];
                    SLOWFN(sY, sU, sV, rY, rU, rV, q, &cs, pel, bx + kk % n - rst, by + kk / n - rst, &sl, &sc);
                    int t = (int)((uint)svm[i] + (uint)pen(sl, pnew));
                    t = (int)((uint)t + (uint)pen(sc, pnew));
                    if (t < craw) cst = t;
                }
                int mn = wmin(cst, scr);
                if (mn != IMAX && mn < bcost) {
                    uint msk = ballot(cst == mn, scr);
                    int w = 31 - (int)clz(msk & (0u - msk));
                    bcost = mn;
                    bk = (int)shfl_i((uint)kk, w, scr);
                    bsl = (int)shfl_i((uint)sl, w, scr);
                    bsc = (int)shfl_i((uint)sc, w, scr);
                }
            }
        } else {
            for (int base = 0; base < ns; base += 4) {
                int i = base + (int)gg;
                bool vld = i < ns;
                int k = svk[vld ? i : 0];
                int sl, sc;
                grp_eval(rY, rU, rV, q, x0, y0, pel, sr, gr, bx + k % n - rst, by + k / n - rst, scr, &sl, &sc);
                int cst = IMAX;
                if (vld && gr == 0) {
                    int t = (int)((uint)svm[i] + (uint)pen(sl, pnew));
                    t = (int)((uint)t + (uint)pen(sc, pnew));
                    if (t < craw) cst = t;
                }
                int mn = wmin(cst, scr);
                if (mn != IMAX && mn < bcost) {
                    uint msk = ballot(cst == mn, scr);
                    int w = 31 - (int)clz(msk & (0u - msk));
                    bcost = mn;
                    bk = (int)shfl_i((uint)k, w, scr);
                    bsl = (int)shfl_i((uint)sl, w, scr);
                    bsc = (int)shfl_i((uint)sc, w, scr);
                }
            }
        }
#else
#ifdef SATD
        if (shifted) {
        for (int i0 = 0; i0 < ns; i0 += 32) {
            int i = i0 + l;
            int cst = IMAX, sl = 0, sc = 0, kk = 0;
            if (i < ns) {
                kk = svk[i];
                satd_at(sY, sU, sV, rY, rU, rV, q, &cs, pel, bx + kk % n - rst, by + kk / n - rst, &sl, &sc);
                int t = (int)((uint)svm[i] + (uint)pen(sl, pnew));
                t = (int)((uint)t + (uint)pen(sc, pnew));
                if (t < craw) cst = t;
            }
            int mn = wmin(cst, scr);
            if (mn != IMAX && mn < bcost) {
                uint msk = ballot(cst == mn, scr);
                int w = 31 - (int)clz(msk & (0u - msk));
                bcost = mn;
                bk = (int)shfl_i((uint)kk, w, scr);
                bsl = (int)shfl_i((uint)sl, w, scr);
                bsc = (int)shfl_i((uint)sc, w, scr);
            }
        }
        } else
#endif
        if (ns > 0) {
            const int wx = x0 + bx - rst, wy = y0 + by - rst;
            const int ccx = (x0 >> 1) + ((bx - rst) >> 1), ccy = (y0 >> 1) + ((by - rst) >> 1);
            const bool staged = !shifted && ns >= 2 && pel == 1
                && wx >= 0 && wx + W + 2 * rst <= PW && wy >= 0 && wy + H + 2 * rst <= PH
                && ccx >= 0 && ccx + CW + rst <= PW / 2 && ccy >= 0 && ccy + CH + rst <= PH / 2;
            if (staged) {
                uint loff = (uint)(q[P_RBY] + wy * q[P_RPY] + wx);
                uint coff = (uint)(q[P_RBU] + ccy * q[P_RPU] + ccx);
#ifdef FASTREG
                ss[2 * l] = s0; ss[2 * l + 1] = s1; ss[CB + l] = scw;
#endif
                wsync();
                stage(stl, NCHL, rY, loff, (uint)q[P_RPY], H + 2 * rst, (W + 2 * rst + 30) / 16, l);
                stage2(stc, NCHC, ROWSC * NCHC, rU, rV, coff, (uint)q[P_RPU], CH + rst, (CW + rst + 30) / 16, l);
                wsync();
                const bool g4 = ns <= 8;
                const int G = g4 ? 4 : 1;
                const int per = 32 / G;
                const uint gl = (uint)(l & (G - 1));
                const int ci = l / G;
                const uint s_l = loff & 15u, s_c = coff & 15u;
                for (int base = 0; base < ns; base += per) {
                    int i = base + ci;
                    bool vld = i < ns;
                    int ii = vld ? i : 0;
                    int k = svk[ii];
                    int dx = k % n - rst, dy = k / n - rst;
                    int cdx = ((bx + dx) >> 1) - ((bx - rst) >> 1), cdy = ((by + dy) >> 1) - ((by - rst) >> 1);
                    uint al, ac;
                    if (g4) {
                        tasks(ss, (__local const uint *)stl, (__local const uint *)stc, gl, 4, dy + rst, s_l + (uint)(dx + rst), cdy, s_c + (uint)cdx, &al, &ac);
                        al += shfl_x(al, 2, scr);
                        ac += shfl_x(ac, 2, scr);
                        al += shfl_x(al, 1, scr);
                        ac += shfl_x(ac, 1, scr);
                    } else {
                        tasks(ss, (__local const uint *)stl, (__local const uint *)stc, gl, 1, dy + rst, s_l + (uint)(dx + rst), cdy, s_c + (uint)cdx, &al, &ac);
                    }
                    int sl = (int)al, sc = (int)ac << 2;
                    int cst = IMAX;
                    if (vld && gl == 0) {
                        int t = (int)((uint)svm[ii] + (uint)pen(sl, pnew));
                        t = (int)((uint)t + (uint)pen(sc, pnew));
                        if (t < craw) cst = t;
                    }
                    int mn = wmin(cst, scr);
                    if (mn != IMAX && mn < bcost) {
                        uint msk = ballot(cst == mn, scr);
                        int w = 31 - (int)clz(msk & (0u - msk));
                        bcost = mn;
                        bk = (int)shfl_i((uint)k, w, scr);
                        bsl = (int)shfl_i((uint)sl, w, scr);
                        bsc = (int)shfl_i((uint)sc, w, scr);
                    }
                }
            } else {
#ifdef FASTREG
                if (shifted || pel != 1) {
                    for (int i0 = 0; i0 < ns; i0 += 32) {
                        int i = i0 + l;
                        int cst = IMAX, sl = 0, sc = 0, kk = 0;
                        if (i < ns) {
                            kk = svk[i];
                            sad_slow(sY, sU, sV, rY, rU, rV, q, &cs, pel, bx + kk % n - rst, by + kk / n - rst, &sl, &sc);
                            int t = (int)((uint)svm[i] + (uint)pen(sl, pnew));
                            t = (int)((uint)t + (uint)pen(sc, pnew));
                            if (t < craw) cst = t;
                        }
                        int mn = wmin(cst, scr);
                        if (mn != IMAX && mn < bcost) {
                            uint msk = ballot(cst == mn, scr);
                            int w = 31 - (int)clz(msk & (0u - msk));
                            bcost = mn;
                            bk = (int)shfl_i((uint)kk, w, scr);
                            bsl = (int)shfl_i((uint)sl, w, scr);
                            bsc = (int)shfl_i((uint)sc, w, scr);
                        }
                    }
                } else
                for (int i0 = 0; i0 < ns; i0 += NSLOT) {
                    Batch B;
                    int cnt = min(ns - i0, NSLOT), kk[NSLOT];
                    #pragma unroll
                    for (int s = 0; s < NSLOT; s++) {
                        kk[s] = 0;
                        B.vx[s] = 0;
                        B.vy[s] = 0;
                        if (s < cnt) {
                            kk[s] = svk[i0 + s];
                            B.vx[s] = bx + kk[s] % n - rst;
                            B.vy[s] = by + kk[s] / n - rst;
                        }
                    }
                    int rl[NSLOT], rc[NSLOT];
                    if (!shifted && pel == 1) {
                        const bool noslow[NSLOT] = {false};
                        issue_batch(&B, rY, rU, rV, bl, bc, pyi, pui, rh, cplane, cnt, noslow);
                        finish_batch(&B, s0, s1, scw, rh, scr, rl, rc);
                    } else {
                        #pragma unroll
                        for (int s = 0; s < NSLOT; s++) {
                            rl[s] = 0; rc[s] = 0;
                            if (s < cnt) sad_slow(sY, sU, sV, rY, rU, rV, q, &cs, pel, B.vx[s], B.vy[s], &rl[s], &rc[s]);
                        }
                    }
                    #pragma unroll
                    for (int s = 0; s < NSLOT; s++)
                        if (s < cnt) {
                            int t = (int)((uint)svm[i0 + s] + (uint)pen(rl[s], pnew));
                            t = (int)((uint)t + (uint)pen(rc[s], pnew));
                            if (t < craw && t < bcost) { bcost = t; bk = kk[s]; bsl = rl[s]; bsc = rc[s]; }
                        }
                }
#else
                for (int i = 0; i < ns; i++) {
                    int k = svk[i];
                    int ex_vx = bx + k % n - rst, ex_vy = by + k / n - rst;
                    int sl, sc;
                    if (shifted) sad_slow(sY, sU, sV, rY, rU, rV, q, &cs, pel, ex_vx, ex_vy, &sl, &sc);
                    else cand_sad(rY, rU, rV, q, x0, y0, pel, ss, stl, stc, scr, ex_vx, ex_vy, &sl, &sc, l);
                    int t = (int)((uint)svm[i] + (uint)pen(sl, pnew));
                    t = (int)((uint)t + (uint)pen(sc, pnew));
                    if (t < craw && t < bcost) { bcost = t; bk = k; bsl = sl; bsc = sc; }
                }
#endif
            }
        }
#endif
        if (cbest < bcost || (cbest == bcost && cbest != IMAX && ckbest < bk)) {
            bcost = cbest; bk = ckbest; bsl = csl; bsc = csc;
        }
    }

    if (l == 0) {
        int rx, ry, rsad, rmin;
        bool wrote = false;
        if (do_ex) {
            if (bk != IMAX) {
                rx = bx + bk % n - rst;
                ry = by + bk / n - rst;
                rsad = bsl + bsc;
                rmin = bcost;
            } else {
                rx = bx; ry = by; rsad = craw; rmin = craw;
            }
            wrote = true;
        } else if (DONTV) {
            rx = cvx; ry = cvy; rsad = craw; rmin = craw;
            wrote = true;
        } else {
            int4 old = rin[b];
            rx = LO(old.x); ry = HI(old.x); rsad = old.y; rmin = old.z;
        }
        int prevxy = pass == 0 ? PK(LO(pv.x), HI(pv.x)) : rin[b].x;
        rout[ob] = (int4)(PK(rx, ry), rsad, rmin, 0);
        dout[ob] = PK(rx, ry) != prevxy;
        if (pass == 0) {
            cen[ob] = (int4)(PK(cvx, cvy), craw, cmin, 0);
            aux[ob] = (int4)(PK(cxp, cyp), lambda, radius, 0);
        } else if (do_ex) {
            cen[ob] = (int4)(PK(cvx, cvy), craw, cmin, 0);
        }
        if (wrote) emit(out, ob, rx, ry, rsad, avg_l, q[P_DIAG]);
    }
}
#endif
