#define LO(v) (((v) << 16) >> 16)
#define HI(v) ((v) >> 16)
#define PK(x, y) (((x) & 0xFFFF) | ((y) << 16))
#define IMAX 2147483647

inline int tapv(__global const uchar *s, int sp, int r, int c) { return (int)s[(long)r * sp + c]; }

inline int vert(__global const uchar *s, int sp, int tw, int th, int y, int c) {
    int r = 2 * y;
    if (y == 0 || y + 1 >= max(th, 2)) return (tapv(s, sp, r, c) + tapv(s, sp, r + 1, c) + 1) >> 1;
    int w2 = 2 * tw;
    if (w2 % 8 != 0 && c >= (w2 >= 8 ? w2 / 8 * 8 : 0)) {
        long at = (long)r * sp + c;
        return ((int)s[at - sp / 2] + 5 * tapv(s, sp, r - 1, c) + 10 * (tapv(s, sp, r, c) + tapv(s, sp, r + 1, c)) +
                5 * (int)s[at + sp / 2] + (int)s[at + 3 * (sp / 2)] + 16) >> 5;
    }
    return (tapv(s, sp, r - 2, c) + 5 * (tapv(s, sp, r - 1, c) + tapv(s, sp, r + 2, c)) +
            10 * (tapv(s, sp, r, c) + tapv(s, sp, r + 1, c)) + tapv(s, sp, r + 3, c) + 16) >> 5;
}

__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void reduce(__global const uchar *s0, __global const uchar *s1, __global const uchar *s2,
            __global uchar *d0, __global uchar *d1, __global uchar *d2, int4 geo, int4 dims, int4 offs) {
    __local uchar st[36][40];
    __local uchar vt[16][40];
    int tx0 = (dims.x + 15) >> 4, ty0 = (dims.y + 15) >> 4, nt0 = tx0 * ty0;
    int tx1 = (dims.z + 15) >> 4, ty1 = (dims.w + 15) >> 4, nt1 = tx1 * ty1;
    int wg = (int)get_group_id(0);
    __global const uchar *s;
    __global uchar *d;
    int sp, dp, tw, th, tx;
    if (wg < nt0) {
        s = s0 + offs.x; d = d0 + offs.z; sp = geo.x; dp = geo.z; tw = dims.x; th = dims.y; tx = tx0;
    } else {
        wg -= nt0;
        if (wg >= 2 * nt1) return;
        if (wg < nt1) { s = s1 + offs.y; d = d1 + offs.w; } else { s = s2 + offs.y; d = d2 + offs.w; wg -= nt1; }
        sp = geo.y; dp = geo.w; tw = dims.z; th = dims.w; tx = tx1;
    }
    int bx = wg % tx, by = wg / tx;
    int ox0 = bx * 16, oy0 = by * 16;
    int c0 = 2 * ox0 - 2, r0 = 2 * oy0 - 2;
    int w2 = 2 * tw, h2 = 2 * th;
    int lid = (int)get_local_id(0);
    for (int idx = lid; idx < 36 * 36; idx += 256) {
        int rv = idx / 36, cu = idx % 36;
        int r = r0 + rv, c = c0 + cu;
        st[rv][cu] = (r >= 0 && r < h2 && c >= 0 && c < w2) ? s[(long)r * sp + c] : (uchar)0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int odd_tail = (w2 & 7) != 0;
    int tail_c = w2 >= 8 ? (w2 & ~7) : 0;
    int max_th = max(th, 2);
    for (int idx = lid; idx < 16 * 36; idx += 256) {
        int ly = idx / 36, cu = idx % 36;
        int y = oy0 + ly, c = c0 + cu;
        if (y < th && c >= 0 && c < w2) {
            int val;
            if (y == 0 || y + 1 >= max_th) {
                val = ((int)st[2 * ly + 2][cu] + (int)st[2 * ly + 3][cu] + 1) >> 1;
            } else if (odd_tail && c >= tail_c) {
                val = vert(s, sp, tw, th, y, c);
            } else {
                val = ((int)st[2 * ly][cu] + 5 * ((int)st[2 * ly + 1][cu] + (int)st[2 * ly + 4][cu]) +
                       10 * ((int)st[2 * ly + 2][cu] + (int)st[2 * ly + 3][cu]) + (int)st[2 * ly + 5][cu] + 16) >> 5;
            }
            vt[ly][cu] = (uchar)val;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int lx = lid & 15, ly = lid >> 4;
    int x = ox0 + lx, y = oy0 + ly;
    if (x < tw && y < th) {
        int u = 2 * lx + 2;
        uchar out;
        if (x == 0) {
            out = (uchar)(((int)vt[ly][2] + (int)vt[ly][3] + 1) >> 1);
        } else if (x == tw - 1) {
            out = (uchar)(((int)vt[ly][u] + (int)vt[ly][u + 1] + 1) >> 1);
        } else {
            int a = x == 1 ? (((int)vt[ly][2] + (int)vt[ly][3] + 1) >> 1) : (int)vt[ly][u - 2];
            int sum = a + 5 * ((int)vt[ly][u - 1] + (int)vt[ly][u + 2]) +
                      10 * ((int)vt[ly][u] + (int)vt[ly][u + 1]) + (int)vt[ly][u + 3];
            out = (uchar)((sum + 16) >> 5);
        }
        d[(long)y * dp + x] = out;
        if (x == tw - 1) for (int k = tw; k < dp; k++) d[(long)y * dp + k] = 0;
    }
}

inline int comp(int4 v, int c) { return c ? HI(v.x) : LO(v.x); }

__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void gest(__global const int4 *vs, int vstride, __global int *hist, __global int *res, int n) {
    const int lid = (int)get_local_id(0), dir = (int)get_group_id(0);
    __global const int4 *v = vs + dir * vstride;
    __global int *h = hist + dir * FREQ;
    __local int low, high, sumx, sumy, cnt, med[2];
    __local int bc[256], bi[256];
    for (int c = 0; c < 2; c++) {
        for (int i = lid; i < FREQ; i += 256) h[i] = 0;
        if (lid == 0) { low = FREQ - 1; high = 0; }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        int lmin = FREQ - 1, lmax = 0;
        for (int i = lid; i < n; i += 256) {
            int idx = (FREQ >> 1) + comp(v[i], c);
            if (idx >= 0 && idx < FREQ) {
                atomic_inc(&h[idx]);
                lmin = min(lmin, idx);
                lmax = max(lmax, idx);
            }
        }
        atomic_min(&low, lmin);
        atomic_max(&high, lmax);
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        int best = IMAX, bcnt = -1;
        for (int i = low + lid; i <= high; i += 256) {
            int f = h[i];
            if (f > bcnt) { bcnt = f; best = i; }
        }
        bc[lid] = bcnt;
        bi[lid] = best;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = 128; s > 0; s >>= 1) {
            if (lid < s && (bc[lid + s] > bc[lid] || (bc[lid + s] == bc[lid] && bi[lid + s] < bi[lid]))) {
                bc[lid] = bc[lid + s];
                bi[lid] = bi[lid + s];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) med[c] = (low > high ? FREQ - 1 : bi[0]) - (FREQ >> 1);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) { sumx = 0; sumy = 0; cnt = 0; }
    barrier(CLK_LOCAL_MEM_FENCE);
    int px = 0, py = 0, pc = 0;
    for (int i = lid; i < n; i += 256) {
        int x = LO(v[i].x), y = HI(v[i].x);
        if (abs(x - med[0]) < 6 && abs(y - med[1]) < 6) { px += x; py += y; pc++; }
    }
    atomic_add(&sumx, px);
    atomic_add(&sumy, py);
    atomic_add(&cnt, pc);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) {
        int gx = cnt > 0 ? 2 * sumx / cnt : 2 * med[0];
        int gy = cnt > 0 ? 2 * sumy / cnt : 2 * med[1];
        res[2 * dir] = (int)(short)gx;
        res[2 * dir + 1] = (int)(short)gy;
    }
}

__kernel void gfin(__global const int *res, __global int *qf, __global int *qf2, int qstride, int pel,
                   int both) {
    __global int *qb = qf + qstride;
    __global int *qb2 = qf2 + qstride;
    int fx = res[0], fy = res[1];
    if (both) {
        int bx = res[2], by = res[3];
        int vx = (int)(((long)fx + (long)bx) / 2), vy = (int)(((long)fy + (long)by) / 2);
        fx = (int)(short)(fx - vx);
        fy = (int)(short)(fy - vy);
        bx = (int)(short)(bx - vx);
        fy = (int)(short)(fy - vy);
        qb[28] = qb2[28] = pel * bx;
        qb[29] = qb2[29] = pel * by;
    }
    qf[28] = qf2[28] = pel * fx;
    qf[29] = qf2[29] = pel * fy;
}

inline int floordiv(int a, int b) {
    int q = a / b;
    return (a % b < 0) ? q - 1 : q;
}

#define PG_NX g.s0
#define PG_NY g.s1
#define PG_CX g.s2
#define PG_CY g.s3
#define PG_SX g.s4
#define PG_SY g.s5
#define PG_BW g.s6
#define PG_BH g.s7

inline int conv_sad(int sad, int flat, int norm_area, uint limit) {
    uint mag = sad < 0 ? 0u - (uint)sad : (uint)sad;
    if (mag <= limit) return sad;
    if (flat) return ((int)((uint)sad * 16u + 8u)) >> 4;
    return ((int)((uint)sad * (16u * (uint)norm_area)) / norm_area) >> 4;
}

__kernel void pred1(__global const int4 *cout, __global int2 *pred, __global int *minsad, int8 g, int4 h,
                    int8 st) {
    int gid = (int)get_global_id(0);
    int nx = PG_NX, ny = PG_NY, cx = PG_CX, cy = PG_CY;
    int d = gid / (nx * ny), idx = gid % (nx * ny);
    if (d > st.s4) return;
    cout += d * st.s0;
    pred += d * st.s1;
    minsad += d * st.s2;
    int i = idx % nx, j = idx / nx;
    int norm = h.y;
    if (i >= 2 * cx || j >= 2 * cy) {
        int ix = min(i / 2, cx), jy = min(j / 2, cy);
        int bid = jy == cy ? ix : cx + 1 + jy;
        int4 b = cout[cx * cy + bid];
        int x = (int)(short)(LO(b.x) << (4 - norm));
        int y = (int)(short)(HI(b.x) << (4 - norm));
        pred[idx] = (int2)(PK(x, y), b.y);
        return;
    }
    int4 c = cout[(j / 2) * cx + i / 2];
    int tx = ((int)(short)(LO(c.x) << 4)) >> norm;
    int ty = ((int)(short)(HI(c.x) << 4)) >> norm;
    uint limit = (uint)(IMAX / (16 * h.z));
    int sad = conv_sad(c.y, h.x, h.z, limit);
    pred[idx] = (int2)(PK(tx, ty), sad);
    int qx = floordiv((PG_BW >> 1) + tx, PG_SX);
    int qy = floordiv((PG_BH >> 1) + ty, PG_SY);
    int rbx = clamp(i + qx, 0, nx - 1), rby = clamp(j + qy, 0, ny - 1);
    if (sad >= 0) atomic_min(&minsad[rby * nx + rbx], sad);
}

__kernel void pred2(__global const int2 *pred, __global const int *minsad, __global int *minidx, int8 g,
                    int8 st) {
    int gid = (int)get_global_id(0);
    int nx = PG_NX, ny = PG_NY, cx = PG_CX, cy = PG_CY;
    int d = gid / (nx * ny), idx = gid % (nx * ny);
    if (d > st.s4) return;
    pred += d * st.s1;
    minsad += d * st.s2;
    minidx += d * st.s2;
    int i = idx % nx, j = idx / nx;
    if (i >= 2 * cx || j >= 2 * cy) return;
    int2 p = pred[idx];
    int sad = p.y;
    if (sad < 0) return;
    int qx = floordiv((PG_BW >> 1) + LO(p.x), PG_SX);
    int qy = floordiv((PG_BH >> 1) + HI(p.x), PG_SY);
    int rbx = clamp(i + qx, 0, nx - 1), rby = clamp(j + qy, 0, ny - 1);
    int t = rby * nx + rbx;
    if (sad == minsad[t]) atomic_min(&minidx[t], idx);
}

__kernel void pred3(__global const int2 *pred, __global int *minsad, __global int *minidx, __global int2 *rev,
                    int nb, int8 st) {
    int gid = (int)get_global_id(0);
    int d = gid / nb, i = gid % nb;
    if (d > st.s4) return;
    pred += d * st.s1;
    minsad += d * st.s2;
    minidx += d * st.s2;
    rev += (st.s4 ? 1 - d : d) * st.s3;
    int m = minidx[i];
    if (m == IMAX) {
        rev[i] = (int2)(0, -1);
    } else {
        int2 p = pred[m];
        rev[i] = (int2)(PK(-LO(p.x), -HI(p.x)), p.y);
    }
    minsad[i] = IMAX;
    minidx[i] = IMAX;
}

__kernel void pinit(__global int *minsad, __global int *minidx, int nb, int stride, int dirs) {
    int gid = (int)get_global_id(0);
    int d = gid / nb, i = gid % nb;
    if (d >= dirs) return;
    minsad += d * stride;
    minidx += d * stride;
    minsad[i] = IMAX;
    minidx[i] = IMAX;
}

__kernel void rank_serp(__global int2 *rev, int nb, int nx, int ny, int8 st) {
    int gid = (int)get_global_id(0);
    int d = gid / nb, i = gid % nb;
    if (d > st.s4) return;
    rev += d * st.s3;
    __global int *rank = (__global int *)(rev + nb);
    int bx = i % nx, by = i / nx;
    if (nx < 2 || ny < 2) {
        rank[i] = i;
        return;
    }
    int interior = (ny - 2) * (nx - 2);
    int r;
    if (by == 0) r = interior + 2 * bx;
    else if (by == ny - 1) r = interior + 2 * bx + 1;
    else if (bx == 0) r = interior + 2 * nx + 2 * (by - 1);
    else if (bx == nx - 1) r = interior + 2 * nx + 2 * (by - 1) + 1;
    else r = (by - 1) * (nx - 2) + ((by & 1) == 0 ? bx - 1 : nx - 2 - bx);
    rank[i] = r;
}

#define MAXE 32

inline int child_entries(int cxi, int cyi, int cx, int cy, int nx, int ny, int2 *pos, int *intr) {
    int add_x = nx - cx * 2, add_y = ny - cy * 2;
    int x = cxi * 2, y = cyi * 2;
    int n = 0;
    bool on_edge = x == 0 || y == 0 || cxi == cx - 1 || cyi == cy - 1;
#define PUSH(c, X, Y) if (n < MAXE) { pos[n] = (int2)((X), (Y)); intr[n] = (c) ? 1 : 0; n++; }
    if (!on_edge) {
        PUSH(1, x, y) PUSH(1, x + 1, y) PUSH(1, x + 1, y + 1) PUSH(1, x, y + 1)
        return n;
    }
    PUSH(x > 0 && y > 0, x, y)
    PUSH(x + 1 < nx - 1 && y > 0, x + 1, y)
    PUSH(x + 1 < nx - 1 && y + 1 < ny - 1, x + 1, y + 1)
    PUSH(x > 0 && y + 1 < ny - 1, x, y + 1)
    bool last_x = add_x != 0 && cxi == cx - 1, last_y = add_y != 0 && cyi == cy - 1;
    if (last_x) {
        for (int i = 0; i < add_x; i++) {
            PUSH(i < add_x - 1 && y > 0, x + 2 + i, y)
            PUSH(i < add_x - 1 && y + 1 < ny - 1, x + 2 + i, y + 1)
            if (last_y) {
                for (int k = 0; k < add_y; k++) PUSH(i < add_x - 1 && k < add_y - 1, x + 2 + i, y + 2 + k)
            }
        }
    }
    if (last_y) {
        for (int k = 0; k < add_y; k++) {
            PUSH(x > 0 && k < add_y - 1, x, y + 2 + k)
            PUSH(x + 1 < nx - 1 && k < add_y - 1, x + 1, y + 2 + k)
        }
    }
    return n;
}

__kernel __attribute__((reqd_work_group_size(SR_WG, 1, 1)))
void srank(__global const int4 *cout, __global int2 *rev, __global int *scratch, int8 g, int4 h, int8 st) {
    const int dir = (int)get_group_id(0);
    cout += dir * st.s0;
    rev += dir * st.s3;
    scratch += dir * 2 * SR_MAX;
    __global int *rank = (__global int *)(rev + PG_NX * PG_NY);
    __local ushort bk[SR_MAX];
    __local ushort cnt[SR_WG * 16];
    __local int scan[SR_WG];
    const int lid = (int)get_local_id(0);
    int nx = PG_NX, ny = PG_NY, cx = PG_CX, cy = PG_CY;
    int n = cx * cy;
    int shift = h.x, buckets = h.y;
    __global int *perm = scratch;
    __global int *alt = scratch + SR_MAX;
    for (int i = lid; i < nx * ny; i += SR_WG) rank[i] = 0;
    for (int i = lid; i < n; i += SR_WG) {
        int sad = cout[i].y;
        bk[i] = (ushort)(sad < 0 ? buckets - 1 : min(sad >> shift, buckets - 1));
        perm[i] = i;
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    int per = (n + SR_WG - 1) / SR_WG;
    int lo = min(lid * per, n), hi = min(lo + per, n);
    int max_b = max(buckets - 1, 0);
    int passes = 0;
    for (int m = max_b; m > 0; m >>= 4) passes++;
    if (passes < 1) passes = 1;
    __global int *src_p = perm;
    __global int *dst_p = alt;
    for (int p = 0; p < passes; p++) {
        int sh = p * 4;
        ushort local_c[16];
        for (int d = 0; d < 16; d++) local_c[d] = 0;
        for (int i = lo; i < hi; i++) {
            int idx = src_p[i];
            int dig = ((int)bk[idx] >> sh) & 15;
            local_c[dig]++;
        }
        for (int d = 0; d < 16; d++) cnt[d * SR_WG + lid] = local_c[d];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < 16) {
            int sum = 0;
            for (int t = 0; t < SR_WG; t++) {
                int v = (int)cnt[lid * SR_WG + t];
                cnt[lid * SR_WG + t] = (ushort)sum;
                sum += v;
            }
            scan[lid] = sum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid == 0) {
            int sum = 0;
            for (int d = 0; d < 16; d++) {
                int v = scan[d];
                scan[d] = sum;
                sum += v;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        int off[16];
        for (int d = 0; d < 16; d++) off[d] = scan[d] + (int)cnt[d * SR_WG + lid];
        for (int i = lo; i < hi; i++) {
            int idx = src_p[i];
            int dig = ((int)bk[idx] >> sh) & 15;
            dst_p[off[dig]++] = idx;
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        __global int *tmp = src_p;
        src_p = dst_p;
        dst_p = tmp;
    }
    if (src_p != perm) {
        for (int i = lid; i < n; i += SR_WG) perm[i] = src_p[i];
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
    int ci = 0, cb = 0;
    for (int s = lo; s < hi; s++) {
        int c = perm[s];
        int2 pos[MAXE];
        int intr[MAXE];
        int m = child_entries(c % cx, c / cx, cx, cy, nx, ny, pos, intr);
        for (int e = 0; e < m; e++) {
            if (intr[e]) ci++; else cb++;
        }
    }
    scan[lid] = ci;
    barrier(CLK_LOCAL_MEM_FENCE);
    int off_i = 0;
    for (int t = 0; t < lid; t++) off_i += scan[t];
    int total_i = 0;
    for (int t = 0; t < SR_WG; t++) total_i += scan[t];
    barrier(CLK_LOCAL_MEM_FENCE);
    scan[lid] = cb;
    barrier(CLK_LOCAL_MEM_FENCE);
    int off_b = 0;
    for (int t = 0; t < lid; t++) off_b += scan[t];
    int ri = off_i, rb = total_i + off_b;
    for (int s = lo; s < hi; s++) {
        int c = perm[s];
        int2 pos[MAXE];
        int intr[MAXE];
        int m = child_entries(c % cx, c / cx, cx, cy, nx, ny, pos, intr);
        for (int e = 0; e < m; e++) {
            int r = intr[e] ? ri++ : rb++;
            if (pos[e].x >= 0 && pos[e].x < nx && pos[e].y >= 0 && pos[e].y < ny) rank[pos[e].y * nx + pos[e].x] = r;
        }
    }
}

__kernel void packout(__global const int4 *out, __global int2 *words, int nb, int8 st) {
    int gid = (int)get_global_id(0);
    int d = gid / nb, i = gid % nb;
    if (d > st.s4) return;
    out += d * st.s0;
    words += d * st.s5;
    int4 o = out[i];
    words[i] = (int2)(o.z, o.w);
}

#define OVR_TARGET 1
#define OVR_OUT 2

inline int serp_pos(int nx, int ny, int bx, int by) {
    int interior = (ny - 2) * (nx - 2);
    if (by == 0) return interior + 2 * bx;
    if (by == ny - 1) return interior + 2 * bx + 1;
    if (bx == 0) return interior + 2 * nx + 2 * (by - 1);
    if (bx == nx - 1) return interior + 2 * nx + 2 * (by - 1) + 1;
    return (by - 1) * (nx - 2) + ((by & 1) == 0 ? bx - 1 : nx - 2 - bx);
}

inline void serp_block(int nx, int ny, int p, int *bx, int *by) {
    int interior = (ny - 2) * (nx - 2);
    if (p < interior) {
        int y = 1 + p / (nx - 2), i = p % (nx - 2);
        *bx = (y & 1) == 0 ? 1 + i : nx - 2 - i;
        *by = y;
    } else if (p < interior + 2 * nx) {
        int t = p - interior;
        *bx = t / 2;
        *by = (t & 1) == 0 ? 0 : ny - 1;
    } else {
        int t = p - interior - 2 * nx;
        *bx = (t & 1) == 0 ? 0 : nx - 1;
        *by = 1 + t / 2;
    }
}

inline void ovr_neighbours(__global const int4 *c, int cx, int cy, int k_x, int l_y, int2 *v, int *off_x, int *off_y) {
    int i = min(k_x, 2 * cx - 1), j = min(l_y, 2 * cy - 1);
    *off_x = -1 + 2 * (i & 1);
    *off_y = -1 + 2 * (j & 1);
    bool edge_x = i == 0 || i >= 2 * cx - 1, edge_y = j == 0 || j >= 2 * cy - 1;
#define AT(a, b) ((int2)(c[(a) + (b) * cx].x, c[(a) + (b) * cx].y))
    int2 v1 = AT(i / 2, j / 2);
    int2 v2 = v1, v3 = v1, v4 = v1;
    if (edge_x && !edge_y) {
        v3 = v4 = AT(i / 2, j / 2 + *off_y);
    } else if (!edge_x && edge_y) {
        v3 = v4 = AT(i / 2 + *off_x, j / 2);
    } else if (!edge_x && !edge_y) {
        v2 = AT(i / 2 + *off_x, j / 2);
        v3 = AT(i / 2, j / 2 + *off_y);
        v4 = AT(i / 2 + *off_x, j / 2 + *off_y);
    }
    v[0] = v1; v[1] = v2; v[2] = v3; v[3] = v4;
}

__kernel void ovf(__global const int4 *c, __global uchar *flags, __global int *ovr, __global int *counter,
                  int8 a, int8 b, int4 d) {
    c += d.y;
    int blk = (int)get_global_id(0);
    int nx = a.s0, ny = a.s1, step_x = a.s2, step_y = a.s3, cx = a.s4, cy = a.s5, flat = a.s6, norm = a.s7;
    int odd_x = b.s0, even_x = b.s1, odd_y = b.s2, even_y = b.s3, bad_sad = b.s4, fw = b.s5, fh = b.s6, pel = b.s7;
    int norm_area = d.x;
    if (blk >= nx * ny) return;
    int bx = blk % nx, by = blk / nx;
    ovr[blk] = -1;
    int2 v[4];
    int off_x, off_y;
    ovr_neighbours(c, cx, cy, bx, by, v, &off_x, &off_y);
    int ax1 = off_x > 0 ? odd_x : even_x, ax2 = step_x * 4 - ax1;
    int ay1 = off_y > 0 ? odd_y : even_y, ay2 = step_y * 4 - ay1;
    int a11 = ax1 * ay1, a21 = ax2 * ay1, a12 = ax1 * ay2, a22 = ax2 * ay2;
    int sad, tx, ty;
    if (flat) {
        sad = (9 * v[0].y + 3 * v[1].y + 3 * v[2].y + v[3].y + 8) >> 6;
        tx = 9 * LO(v[0].x) + 3 * LO(v[1].x) + 3 * LO(v[2].x) + LO(v[3].x);
        ty = 9 * HI(v[0].x) + 3 * HI(v[1].x) + 3 * HI(v[2].x) + HI(v[3].x);
    } else {
        sad = ((a11 * v[0].y + a21 * v[1].y + a12 * v[2].y + a22 * v[3].y) / norm_area) >> 6;
        tx = (a11 * LO(v[0].x) + a21 * LO(v[1].x) + a12 * LO(v[2].x) + a22 * LO(v[3].x)) / norm_area;
        ty = (a11 * HI(v[0].x) + a21 * HI(v[1].x) + a12 * HI(v[2].x) + a22 * HI(v[3].x)) / norm_area;
    }
    uchar f = 0;
    if (sad > bad_sad) {
        f = OVR_TARGET;
        int mx = ((int)(short)tx) >> norm, my = ((int)(short)ty) >> norm;
        mx = (int)(short)mx;
        my = (int)(short)my;
        int x0 = bx * step_x, y0 = by * step_y;
        int px = mx < 0 ? mx - 2 : mx + 2, py = my < 0 ? my - 2 : my + 2;
        bool out = px < -pel * x0 || px >= pel * (fw - x0 - 4) || py < -pel * y0 || py >= pel * (fh - y0 - 4);
        if (out) {
            f = OVR_OUT;
            atomic_inc(counter);
        }
    }
    flags[serp_pos(nx, ny, bx, by)] = f;
}

#define OS_WG 256

__kernel __attribute__((reqd_work_group_size(OS_WG, 1, 1)))
void ovs(__global const uchar *flags, __global int *ovr, __global int *counter, int nx, int ny) {
    __local int first_a[OS_WG], has_t[OS_WG], carry[OS_WG], first_b[OS_WG];
    if (*counter == 0) return;
    const int lid = (int)get_local_id(0);
    int count = nx * ny;
    int per = (count + OS_WG - 1) / OS_WG;
    int lo = min(lid * per, count), hi = min(lo + per, count);
    int fa = -1, fb = -1, ht = 0;
    for (int p = lo; p < hi; p++) {
        uchar f = flags[p];
        if (f == OVR_OUT) {
            if (fb < 0) fb = p;
            if (!ht && fa < 0) fa = p;
        } else if (f == OVR_TARGET) {
            ht = 1;
            fb = -1;
        }
    }
    first_a[lid] = fa;
    has_t[lid] = ht;
    first_b[lid] = fb;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) {
        int pending = -1;
        for (int t = 0; t < OS_WG; t++) {
            carry[t] = pending;
            if (has_t[t]) pending = first_b[t];
            else if (pending < 0) pending = first_a[t];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int pending = carry[lid];
    for (int p = lo; p < hi; p++) {
        uchar f = flags[p];
        if (f == OVR_OUT) {
            if (pending < 0) pending = p;
        } else if (f == OVR_TARGET) {
            if (pending >= 0) {
                int bx, by, ox, oy;
                serp_block(nx, ny, p, &bx, &by);
                serp_block(nx, ny, pending, &ox, &oy);
                ovr[by * nx + bx] = oy * nx + ox;
            }
            pending = -1;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (lid == 0) *counter = 0;
}

inline long ch_subpel(long base, long pitch, long height, int pel, int x, int y) {
    if (pel == 1) return base + (long)y * pitch + x;
    int bits = pel == 2 ? 1 : 2;
    int mask = pel == 2 ? 1 : 3;
    long index = (long)((x & mask) | ((y & mask) << bits));
    return base + (long)(y >> bits) * pitch + (x >> bits) + index * pitch * height;
}

inline uint ch_pack_score(int vx, int vy, int sad, int avg, int diag) {
    int relative = (abs(vx) + abs(vy)) * 100 / max(diag, 1);
    int s = sad;
    if (relative > 50) s = (int)((uint)sad * 20u);
    else if (relative > 15) s = (int)((uint)sad + (uint)(((int)((uint)sad * (uint)(relative - 15))) / 35 * 9));
    return ((uint)s & 0x00FFFFFFu) + ((uint)avg << 24);
}

inline int ch_eval_luma(__global const uchar *sy, __global const uchar *ry,
                        long sbase, long rbase, int sp, int rp, int rh, int pel,
                        int fw, int fh, int x0, int y0, int bw, int bh, int step_row,
                        int vx, int vy) {
    if (vx < -pel * x0 || vx >= pel * (fw - x0 - bw) || vy < -pel * y0 || vy >= pel * (fh - y0 - bh))
        return IMAX;
    long so = sbase + (long)y0 * sp + x0;
    long ro = ch_subpel(rbase, rp, rh, pel, x0 * pel + vx, y0 * pel + vy);
    uint acc = 0;
    for (int y = 0; y < bh; y += step_row) {
        long sr = so + (long)y * sp;
        long rr = ro + (long)y * rp;
        for (int x = 0; x < bw; x++) {
            acc += (uint)abs((int)sy[sr + x] - (int)ry[rr + x]);
        }
    }
    return step_row == 2 ? (int)(acc << 1) : (int)acc;
}

inline int ch_avg_luma(__global const uchar *sy, long sbase, int sp, int x0, int y0, int bw, int bh) {
    long so = sbase + (long)y0 * sp + x0;
    uint sum = 0;
    for (int y = 0; y < bh; y++) {
        long sr = so + (long)y * sp;
        for (int x = 0; x < bw; x++) sum += (uint)sy[sr + x];
    }
    int shift = 31 - (int)clz((uint)(bw * bh));
    return (int)(sum >> shift);
}

inline void ch_unpack_word(uint2 w, int *vx, int *vy, int *sad) {
    int y = (int)(short)w.x;
    int x = (int)(short)((w.x - (uint)y) >> 16);
    *vx = x;
    *vy = y;
    *sad = (int)(w.y & 0x00FFFFFFu);
}

__kernel void mirror_splat1(__global const uint2 *words, __global int *minsad, int8 g) {
    int i = (int)get_global_id(0);
    int nx = g.s0, ny = g.s1, sx = max(g.s2 * g.s4, 1), sy = max(g.s3 * g.s4, 1);
    if (i >= nx * ny) return;
    int vx, vy, sad;
    ch_unpack_word(words[i], &vx, &vy, &sad);
    int bx = i % nx, by = i / nx;
    int tx = clamp(bx + floordiv(vx + (sx >> 1), sx), 0, nx - 1);
    int ty = clamp(by + floordiv(vy + (sy >> 1), sy), 0, ny - 1);
    atomic_min(&minsad[ty * nx + tx], sad);
}

__kernel void mirror_splat2(__global const uint2 *words, __global const int *minsad, __global int *minidx, int8 g) {
    int i = (int)get_global_id(0);
    int nx = g.s0, ny = g.s1, sx = max(g.s2 * g.s4, 1), sy = max(g.s3 * g.s4, 1);
    if (i >= nx * ny) return;
    int vx, vy, sad;
    ch_unpack_word(words[i], &vx, &vy, &sad);
    int bx = i % nx, by = i / nx;
    int tx = clamp(bx + floordiv(vx + (sx >> 1), sx), 0, nx - 1);
    int ty = clamp(by + floordiv(vy + (sy >> 1), sy), 0, ny - 1);
    int t = ty * nx + tx;
    if (sad == minsad[t]) atomic_min(&minidx[t], i);
}

__kernel void mirror_verify(__global uint2 *words, __global int *minsad, __global int *minidx,
                            __global const uchar *sy, __global const uchar *ry,
                            int8 g0, int8 g1) {
    int i = (int)get_global_id(0);
    int nx = g0.s0, ny = g0.s1, step_x = g0.s2, step_y = g0.s3;
    int bw = g0.s4, bh = g0.s5, pel = g0.s6, words_stride = g0.s7;
    int fw = g1.s0, fh = g1.s1, sp = g1.s2, rp = g1.s3;
    int diag = g1.s4, step_row = g1.s5;
    if (i >= nx * ny) return;
    int bx = i % nx, by = i / nx;
    int m = minidx[i];
    if (m == IMAX) {
        int best_ms = IMAX;
        int ox[4] = { -1, 1, 0, 0 };
        int oy[4] = { 0, 0, -1, 1 };
        for (int k = 0; k < 4; k++) {
            int nnx = bx + ox[k], nny = by + oy[k];
            if (nnx >= 0 && nnx < nx && nny >= 0 && nny < ny) {
                int ni = nny * nx + nnx;
                int nm = minidx[ni];
                int ms = minsad[ni];
                if (nm != IMAX && ms < best_ms) {
                    best_ms = ms;
                    m = nm;
                }
            }
        }
    }
    int fvx, fvy, fsad;
    ch_unpack_word(words[i], &fvx, &fvy, &fsad);
    int cx[3], cy[3];
    if (m != IMAX) {
        int svx, svy, ssad;
        ch_unpack_word(words[m], &svx, &svy, &ssad);
        cx[0] = -svx; cy[0] = -svy;
    } else {
        cx[0] = -fvx; cy[0] = -fvy;
    }
    cx[1] = -fvx; cy[1] = -fvy;
    cx[2] = 0; cy[2] = 0;
    int x0 = bx * step_x, y0 = by * step_y;
    int best_x = 0, best_y = 0, best_sad = IMAX;
    for (int k = 0; k < 3; k++) {
        bool dup = false;
        for (int p = 0; p < k; p++) {
            if (cx[p] == cx[k] && cy[p] == cy[k]) dup = true;
        }
        if (dup) continue;
        int s = ch_eval_luma(sy, ry, 0, 0, sp, rp, fh, pel, fw, fh, x0, y0, bw, bh, step_row, cx[k], cy[k]);
        if (s < best_sad) {
            best_sad = s;
            best_x = cx[k];
            best_y = cy[k];
        }
    }
    if (best_sad == IMAX) { best_x = 0; best_y = 0; best_sad = 0; }
    int avg = ch_avg_luma(sy, 0, sp, x0, y0, bw, bh);
    words[words_stride + i] = (uint2)(((uint)best_x << 16) + (uint)best_y, ch_pack_score(best_x, best_y, best_sad, avg, diag));
    minsad[i] = IMAX;
    minidx[i] = IMAX;
}

__kernel void mirror_l1_splat1(__global const int4 *out, __global int *minsad, int8 g) {
    int i = (int)get_global_id(0);
    int nx = g.s0, ny = g.s1, sx = max(g.s2, 1), sy = max(g.s3, 1);
    if (i >= nx * ny) return;
    int4 o = out[i];
    int vx = LO(o.x), vy = HI(o.x), sad = max(o.y, 0);
    int bx = i % nx, by = i / nx;
    int tx = clamp(bx + floordiv(vx + (sx >> 1), sx), 0, nx - 1);
    int ty = clamp(by + floordiv(vy + (sy >> 1), sy), 0, ny - 1);
    atomic_min(&minsad[ty * nx + tx], sad);
}

__kernel void mirror_l1_splat2(__global const int4 *out, __global const int *minsad, __global int *minidx, int8 g) {
    int i = (int)get_global_id(0);
    int nx = g.s0, ny = g.s1, sx = max(g.s2, 1), sy = max(g.s3, 1);
    if (i >= nx * ny) return;
    int4 o = out[i];
    int vx = LO(o.x), vy = HI(o.x), sad = max(o.y, 0);
    int bx = i % nx, by = i / nx;
    int tx = clamp(bx + floordiv(vx + (sx >> 1), sx), 0, nx - 1);
    int ty = clamp(by + floordiv(vy + (sy >> 1), sy), 0, ny - 1);
    int t = ty * nx + tx;
    if (sad == minsad[t]) atomic_min(&minidx[t], i);
}

__kernel void mirror_l1_verify(__global int4 *out, __global int *minsad, __global int *minidx,
                               __global const uchar *sy, __global const uchar *ry,
                               int8 g0, int8 g1) {
    int i = (int)get_global_id(0);
    int nx = g0.s0, ny = g0.s1, step_x = g0.s2, step_y = g0.s3;
    int bw = g0.s4, bh = g0.s5, out_stride = g0.s6, step_row = g0.s7;
    int fw = g1.s0, fh = g1.s1, sp = g1.s2, rp = g1.s3;
    int diag = g1.s4, base_off = g1.s5;
    if (i >= nx * ny) return;
    int bx = i % nx, by = i / nx;
    int m = minidx[i];
    if (m == IMAX) {
        int best_ms = IMAX;
        int ox[4] = { -1, 1, 0, 0 };
        int oy[4] = { 0, 0, -1, 1 };
        for (int k = 0; k < 4; k++) {
            int nnx = bx + ox[k], nny = by + oy[k];
            if (nnx >= 0 && nnx < nx && nny >= 0 && nny < ny) {
                int ni = nny * nx + nnx;
                int nm = minidx[ni];
                int ms = minsad[ni];
                if (nm != IMAX && ms < best_ms) {
                    best_ms = ms;
                    m = nm;
                }
            }
        }
    }
    int4 fo = out[i];
    int fvx = LO(fo.x), fvy = HI(fo.x);
    int cx[3], cy[3];
    if (m != IMAX) {
        int4 so = out[m];
        cx[0] = -LO(so.x); cy[0] = -HI(so.x);
    } else {
        cx[0] = -fvx; cy[0] = -fvy;
    }
    cx[1] = -fvx; cy[1] = -fvy;
    cx[2] = 0; cy[2] = 0;
    int x0 = bx * step_x, y0 = by * step_y;
    int best_x = 0, best_y = 0, best_sad = IMAX;
    for (int k = 0; k < 3; k++) {
        bool dup = false;
        for (int p = 0; p < k; p++) {
            if (cx[p] == cx[k] && cy[p] == cy[k]) dup = true;
        }
        if (dup) continue;
        int s = ch_eval_luma(sy, ry, base_off, base_off, sp, rp, fh, 1, fw, fh, x0, y0, bw, bh, step_row, cx[k], cy[k]);
        if (s < best_sad) {
            best_sad = s;
            best_x = cx[k];
            best_y = cy[k];
        }
    }
    if (best_sad == IMAX) { best_x = 0; best_y = 0; best_sad = 0; }
    int avg = ch_avg_luma(sy, base_off, sp, x0, y0, bw, bh);
    out[out_stride + i] = (int4)(PK(best_x, best_y), best_sad, (int)(((uint)best_x << 16) + (uint)best_y), (int)ch_pack_score(best_x, best_y, best_sad, avg, diag));
    minsad[i] = IMAX;
    minidx[i] = IMAX;
}

__kernel void upsample_bilateral(__global const int4 *l1_out, __global const uchar *y0, __global const uchar *y1,
                                 __global uint2 *words, int8 g0, int8 g1, int8 g2) {
    int gid = (int)get_global_id(0);
    int obx = g0.s0, oby = g0.s1, o_bw = g0.s2, o_bh = g0.s3;
    int o_sx = g0.s4, o_sy = g0.s5, fw = g0.s6, fh = g0.s7;
    int cbx = g1.s0, cby = g1.s1, c_bw = g1.s2, c_bh = g1.s3;
    int c_sx = g1.s4, c_sy = g1.s5, py0 = g1.s6, py1 = g1.s7;
    int out_pel = g2.s0, diag = g2.s1, l1_stride = g2.s2, words_stride = g2.s3, dirs = g2.s4;
    int nb = obx * oby;
    int d = gid / nb, i = gid % nb;
    if (d >= dirs) return;
    __global const int4 *co = l1_out + d * l1_stride;
    __global const uchar *sy = d == 0 ? y0 : y1;
    int sp = d == 0 ? py0 : py1;
    __global uint2 *dst = words + d * words_stride;
    int bx = i % obx, by = i / obx;
    int x0 = bx * o_sx, y0_px = by * o_sy;
    int y_fine = ch_avg_luma(sy, 0, sp, x0, y0_px, o_bw, o_bh);
    float fx = ((float)(x0 * 2 + o_bw) * 0.25f - (float)c_bw * 0.5f) / (float)max(c_sx, 1);
    float fy = ((float)(y0_px * 2 + o_bh) * 0.25f - (float)c_bh * 0.5f) / (float)max(c_sy, 1);
    int ix0 = clamp((int)floor(fx), 0, cbx - 1);
    int iy0 = clamp((int)floor(fy), 0, cby - 1);
    int ix1 = min(ix0 + 1, cbx - 1);
    int iy1 = min(iy0 + 1, cby - 1);
    int wx = (int)(clamp(fx - (float)ix0, 0.0f, 1.0f) * 64.0f);
    int wy = (int)(clamp(fy - (float)iy0, 0.0f, 1.0f) * 64.0f);
    int sw[4] = { (64 - wx) * (64 - wy), wx * (64 - wy), (64 - wx) * wy, wx * wy };
    int ixs[4] = { ix0, ix1, ix0, ix1 };
    int iys[4] = { iy0, iy0, iy1, iy1 };
    long sum_w = 0, sum_x = 0, sum_y = 0, sum_sad = 0, max_w = -1;
    int best_k = 0;
    int min_vx = IMAX, max_vx = -IMAX, min_vy = IMAX, max_vy = -IMAX;
    int4 cv[4];
    for (int k = 0; k < 4; k++) {
        int4 o = co[iys[k] * cbx + ixs[k]];
        cv[k] = o;
        int vx = LO(o.x), vy = HI(o.x);
        min_vx = min(min_vx, vx); max_vx = max(max_vx, vx);
        min_vy = min(min_vy, vy); max_vy = max(max_vy, vy);
        int yc;
        if (o.w != 0) {
            yc = (int)((uint)o.w >> 24);
        } else {
            int sx_c = clamp((ixs[k] * c_sx + (c_bw >> 1)) * 2, 0, fw - 1);
            int sy_c = clamp((iys[k] * c_sy + (c_bh >> 1)) * 2, 0, fh - 1);
            yc = (int)sy[(long)sy_c * sp + sx_c];
        }
        int dy = abs(y_fine - yc);
        int rw = 256 / (1 + ((dy * dy) >> 6));
        long w = (long)max(sw[k], 1) * (long)rw;
        if (w > max_w) { max_w = w; best_k = k; }
        sum_w += w;
        sum_x += w * (long)vx;
        sum_y += w * (long)vy;
        sum_sad += w * (long)max(o.y, 0);
    }
    int mv_scale = 2 * out_pel;
    int sad_num = max(o_bw * o_bh, 1), sad_den = max(c_bw * c_bh, 1);
    int vx, vy, sad;
    if ((max_vx - min_vx) + (max_vy - min_vy) > 4) {
        int4 o = cv[best_k];
        vx = LO(o.x) * mv_scale;
        vy = HI(o.x) * mv_scale;
        sad = (int)((long)max(o.y, 0) * (long)sad_num / (long)sad_den);
    } else {
        long div = max(sum_w, 1L);
        long half_div = div >> 1;
        long nx = sum_x * (long)mv_scale;
        long ny = sum_y * (long)mv_scale;
        vx = (int)((nx >= 0 ? nx + half_div : nx - half_div) / div);
        vy = (int)((ny >= 0 ? ny + half_div : ny - half_div) / div);
        sad = (int)((sum_sad / div) * (long)sad_num / (long)sad_den);
    }
    vx = clamp(vx, -out_pel * x0, out_pel * max(fw - x0 - o_bw, 0));
    vy = clamp(vy, -out_pel * y0_px, out_pel * max(fh - y0_px - o_bh, 0));
    dst[i] = (uint2)(((uint)vx << 16) + (uint)vy, ch_pack_score(vx, vy, sad, y_fine, diag));
}
