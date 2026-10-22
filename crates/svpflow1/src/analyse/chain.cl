#define LO(v) (((v) << 16) >> 16)
#define HI(v) ((v) >> 16)
#define PK(x, y) (((x) & 0xFFFF) | ((y) << 16))
#define IMAX 2147483647

inline int tapv(__global const uchar *s, int sp, int r, int c) { return (int)s[(long)r * sp + c]; }

inline int vert(__global const uchar *s, int sp, int th, int y, int c) {
    int r = 2 * y;
    if (y == 0 || y + 1 >= max(th, 2)) return (tapv(s, sp, r, c) + tapv(s, sp, r + 1, c) + 1) >> 1;
    return (tapv(s, sp, r - 2, c) + 5 * (tapv(s, sp, r - 1, c) + tapv(s, sp, r + 2, c)) +
            10 * (tapv(s, sp, r, c) + tapv(s, sp, r + 1, c)) + tapv(s, sp, r + 3, c) + 16) >> 5;
}

inline uchar reduce_px(__global const uchar *s, int sp, int tw, int th, int x, int y) {
    if (x == 0) return (uchar)((vert(s, sp, th, y, 0) + vert(s, sp, th, y, 1) + 1) >> 1);
    if (x == tw - 1) return (uchar)((vert(s, sp, th, y, 2 * x) + vert(s, sp, th, y, 2 * x + 1) + 1) >> 1);
    int a = x == 1 ? (vert(s, sp, th, y, 0) + vert(s, sp, th, y, 1) + 1) >> 1 : vert(s, sp, th, y, 2 * x - 2);
    int sum = a + 5 * (vert(s, sp, th, y, 2 * x - 1) + vert(s, sp, th, y, 2 * x + 2)) +
              10 * (vert(s, sp, th, y, 2 * x) + vert(s, sp, th, y, 2 * x + 1)) + vert(s, sp, th, y, 2 * x + 3);
    return (uchar)((sum + 16) >> 5);
}

__kernel void reduce(__global const uchar *s0, __global const uchar *s1, __global const uchar *s2,
                     __global uchar *d0, __global uchar *d1, __global uchar *d2, int4 geo, int4 dims, int4 offs) {
    int i = (int)get_global_id(0);
    int n0 = dims.x * dims.y, n1 = dims.z * dims.w;
    __global const uchar *s;
    __global uchar *d;
    int sp, dp, tw, th;
    if (i < n0) {
        s = s0 + offs.x; d = d0 + offs.z; sp = geo.x; dp = geo.z; tw = dims.x; th = dims.y;
    } else {
        i -= n0;
        if (i >= 2 * n1) return;
        if (i < n1) { s = s1 + offs.y; d = d1 + offs.w; } else { s = s2 + offs.y; d = d2 + offs.w; i -= n1; }
        sp = geo.y; dp = geo.w; tw = dims.z; th = dims.w;
    }
    int x = i % tw, y = i / tw;
    d[(long)y * dp + x] = reduce_px(s, sp, tw, th, x, y);
    if (x == tw - 1) for (int k = tw; k < dp; k++) d[(long)y * dp + k] = 0;
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
    __local int bk[SR_MAX];
    __local int scan[SR_WG];
    __local int tot[2];
    const int lid = (int)get_local_id(0);
    int nx = PG_NX, ny = PG_NY, cx = PG_CX, cy = PG_CY;
    int n = cx * cy;
    int shift = h.x, buckets = h.y;
    __global int *perm = scratch;
    __global int *base = scratch + SR_MAX;
    for (int i = lid; i < nx * ny; i += SR_WG) rank[i] = 0;
    for (int i = lid; i < n; i += SR_WG) {
        int sad = cout[i].y;
        bk[i] = sad < 0 ? buckets - 1 : min(sad >> shift, buckets - 1);
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    for (int i = lid; i < n; i += SR_WG) {
        int b = bk[i], pos = 0;
        for (int j = 0; j < n; j++) {
            int o = bk[j];
            pos += (o < b || (o == b && j < i)) ? 1 : 0;
        }
        perm[pos] = i;
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    int per = (n + SR_WG - 1) / SR_WG;
    int lo = min(lid * per, n), hi = min(lo + per, n);
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
