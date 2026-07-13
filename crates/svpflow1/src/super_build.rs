use crate::super_opts::{SuperOpts, reduce_dim};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, _mm_add_epi16, _mm_loadl_epi64, _mm_mullo_epi16, _mm_packus_epi16, _mm_set1_epi16,
    _mm_setzero_si128, _mm_srai_epi16, _mm_storel_epi64, _mm_sub_epi16, _mm_unpacklo_epi8,
};

pub(crate) fn build_plane(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,

    src_w: usize,
    src_h: usize,

    luma_w: usize,
    luma_h: usize,
    chroma: bool,
    opts: &SuperOpts,
) {
    let levels = opts.levels as usize;
    let pel = opts.pel as usize;
    if levels == 0 || src_w == 0 || src_h == 0 {
        return;
    }

    let level_size = |lv: usize| -> (usize, usize) {
        let yw = reduce_dim(luma_w as i32, lv as i32) as usize;
        let yh = reduce_dim(luma_h as i32, lv as i32) as usize;
        if chroma { (yw / 2, yh / 2) } else { (yw, yh) }
    };

    let mut level_y = Vec::with_capacity(levels);
    let mut y = 0usize;
    for lv in 0..levels {
        level_y.push(y);
        let (_, h) = level_size(lv);
        let sub = if lv == 0 && opts.full { pel * pel } else { 1 };
        y += h * sub;
    }

    let (base_w, base_h) = level_size(0);

    {
        let row0 = level_y[0];
        for row in 0..base_h.min(src_h) {
            let dst_off = (row0 + row) * dst_stride;
            let src_off = row * src_stride;
            let n = base_w
                .min(src_w)
                .min(dst.len().saturating_sub(dst_off))
                .min(src.len().saturating_sub(src_off));
            if n > 0 {
                dst[dst_off..dst_off + n].copy_from_slice(&src[src_off..src_off + n]);
            }

            if base_w > 0 && dst_stride > base_w {
                let edge = dst[dst_off + base_w - 1];
                for p in base_w..dst_stride {
                    if dst_off + p < dst.len() {
                        dst[dst_off + p] = edge;
                    }
                }
            }
        }
    }

    if opts.scale_down == 4 {
        for lv in 0..levels.saturating_sub(1) {
            let (src_w_lv, src_h_lv) = level_size(lv);
            let (dst_w_lv, dst_h_lv) = level_size(lv + 1);
            reduce_6tap(
                dst,
                dst_stride,
                level_y[lv + 1],
                dst_w_lv,
                dst_h_lv,
                level_y[lv],
                src_w_lv,
                src_h_lv,
            );
        }
    }

    if pel >= 2 && opts.gpu <= 1 && opts.full {
        if opts.scale_up == 0 {
            fill_bilinear_pel2(dst, dst_stride, level_y[0], base_w, base_h);
        } else if opts.scale_up == 2 {
            fill_bicubic_pel2(dst, dst_stride, level_y[0], base_w, base_h);
        }
    }
}

fn fill_bicubic_pel2(dst: &mut [u8], stride: usize, row0: usize, w: usize, h: usize) {
    let planes = [row0, row0 + h, row0 + 2 * h, row0 + 3 * h];
    bicubic_horizontal(dst, stride, planes[0], planes[1], w, h);
    bicubic_vertical(dst, stride, planes[0], planes[2], w, h);
    bicubic_horizontal(dst, stride, planes[2], planes[3], w, h);
}

fn bicubic_horizontal(buf: &mut [u8], stride: usize, src: usize, dst: usize, w: usize, h: usize) {
    for y in 0..h {
        let src = (src + y) * stride;
        let dst = (dst + y) * stride;
        let mut x = 0;
        while x < w {
            #[cfg(target_arch = "x86_64")]
            if x >= 2 && x + 8 <= w.saturating_sub(4) {
                unsafe { cubic8(buf, dst + x, src + x, 1) };
                x += 8;
                continue;
            }
            buf[dst + x] = half(buf, src + x, 1, x, w);
            x += 1;
        }
    }
}

fn bicubic_vertical(buf: &mut [u8], stride: usize, src: usize, dst: usize, w: usize, h: usize) {
    for y in 0..h {
        let mut x = 0;
        while x < w {
            #[cfg(target_arch = "x86_64")]
            if y >= 2 && y + 4 < h && x + 8 <= w {
                unsafe { cubic8(buf, (dst + y) * stride + x, (src + y) * stride + x, stride) };
                x += 8;
                continue;
            }
            buf[(dst + y) * stride + x] = half(buf, (src + y) * stride + x, stride, y, h);
            x += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn cubic8(buf: &mut [u8], dst: usize, src: usize, step: usize) {
    unsafe {
        let zero = _mm_setzero_si128();
        let load = |i| _mm_unpacklo_epi8(_mm_loadl_epi64(buf.as_ptr().add(i).cast()), zero);
        let outer = _mm_add_epi16(load(src - 2 * step), load(src + 3 * step));
        let inner = _mm_mullo_epi16(
            _mm_add_epi16(load(src), load(src + step)),
            _mm_set1_epi16(20),
        );
        let near = _mm_mullo_epi16(
            _mm_add_epi16(load(src - step), load(src + 2 * step)),
            _mm_set1_epi16(5),
        );
        let value = _mm_srai_epi16::<5>(_mm_add_epi16(
            _mm_sub_epi16(_mm_add_epi16(outer, inner), near),
            _mm_set1_epi16(16),
        ));
        _mm_storel_epi64(
            buf.as_mut_ptr().add(dst).cast::<__m128i>(),
            _mm_packus_epi16(value, zero),
        );
    }
}

#[inline]
fn half(buf: &[u8], i: usize, step: usize, p: usize, n: usize) -> u8 {
    if p < 2 || p + 4 >= n {
        return if p + 1 == n {
            buf[i]
        } else {
            avg_epu8(buf[i], buf[i + step])
        };
    }
    let value = i32::from(buf[i - 2 * step])
        + i32::from(buf[i + 3 * step])
        + 5 * (4 * i32::from(buf[i]) - i32::from(buf[i - step]) + 4 * i32::from(buf[i + step])
            - i32::from(buf[i + 2 * step]))
        + 16;
    (value >> 5).clamp(0, 255) as u8
}

fn fill_bilinear_pel2(dst: &mut [u8], stride: usize, row0: usize, w: usize, h: usize) {
    let full = row0;
    let h_plane = row0 + h;
    let v_plane = row0 + 2 * h;
    let hv_plane = row0 + 3 * h;

    for y in 0..h {
        let src_base = (full + y) * stride;
        let dst_base = (h_plane + y) * stride;
        for x in 0..w {
            let a = dst.get(src_base + x).copied().unwrap_or(0);
            let b = dst.get(src_base + x + 1).copied().unwrap_or(a);
            if dst_base + x < dst.len() {
                dst[dst_base + x] = avg_epu8(a, b);
            }
        }
    }

    for y in 0..h {
        let src_base = (full + y) * stride;
        let next_base = (full + y + 1) * stride;
        let dst_base = (v_plane + y) * stride;
        for x in 0..w {
            let a = dst.get(src_base + x).copied().unwrap_or(0);
            let b = dst.get(next_base + x).copied().unwrap_or(a);
            if dst_base + x < dst.len() {
                dst[dst_base + x] = avg_epu8(a, b);
            }
        }
    }
    copy_row(dst, v_plane * stride, full * stride, w);

    for y in 0..h {
        let src_base = (full + y) * stride;
        let diag_base = (full + y + 1) * stride + 1;
        let dst_base = (hv_plane + y) * stride;
        for x in 0..w {
            let a = dst.get(src_base + x).copied().unwrap_or(0);
            let b = dst.get(diag_base + x).copied().unwrap_or(a);
            if dst_base + x < dst.len() {
                dst[dst_base + x] = avg_epu8(a, b);
            }
        }
    }
    copy_row(dst, hv_plane * stride, full * stride + 1, w);
}

fn copy_row(buf: &mut [u8], dst_off: usize, src_off: usize, n: usize) {
    let n = n
        .min(buf.len().saturating_sub(dst_off))
        .min(buf.len().saturating_sub(src_off));
    if n == 0 {
        return;
    }
    if dst_off == src_off {
        return;
    }

    if dst_off < src_off {
        for i in 0..n {
            buf[dst_off + i] = buf[src_off + i];
        }
    } else if dst_off > src_off + n || src_off > dst_off + n {
        buf.copy_within(src_off..src_off + n, dst_off);
    } else {
        let tmp = buf[src_off..src_off + n].to_vec();
        buf[dst_off..dst_off + n].copy_from_slice(&tmp);
    }
}

#[inline]
fn avg_epu8(a: u8, b: u8) -> u8 {
    ((u16::from(a) + u16::from(b) + 1) >> 1) as u8
}

fn reduce_6tap(
    buf: &mut [u8],
    stride: usize,
    dst_row0: usize,
    dst_w: usize,
    dst_h: usize,
    src_row0: usize,
    src_w: usize,
    src_h: usize,
) {
    if dst_w == 0 || dst_h == 0 || src_w == 0 || src_h == 0 {
        return;
    }

    let inter_w = src_w;
    let mut inter = vec![0u8; inter_w.saturating_mul(dst_h)];

    for dy in 0..dst_h {
        let sy = dy * 2;
        let win0 = sy as isize - 2;
        let use_edge = dy == 0 || dy + 1 == dst_h || win0 < 0 || (sy + 3) >= src_h;
        if use_edge {
            let y0 = sy.min(src_h.saturating_sub(1));
            let y1 = (sy + 1).min(src_h.saturating_sub(1));
            for x in 0..inter_w {
                let a = sample(buf, stride, src_row0, y0, x);
                let b = sample(buf, stride, src_row0, y1, x);
                inter[dy * inter_w + x] = avg_epu8(a, b);
            }
        } else {
            let start = win0 as usize;
            for x in 0..inter_w {
                let p0 = u32::from(sample(buf, stride, src_row0, start, x));
                let p1 = u32::from(sample(buf, stride, src_row0, start + 1, x));
                let p2 = u32::from(sample(buf, stride, src_row0, start + 2, x));
                let p3 = u32::from(sample(buf, stride, src_row0, start + 3, x));
                let p4 = u32::from(sample(buf, stride, src_row0, start + 4, x));
                let p5 = u32::from(sample(buf, stride, src_row0, start + 5, x));
                let v = (p0 + p5 + 5 * (p1 + p4) + 10 * (p2 + p3) + 16) >> 5;
                inter[dy * inter_w + x] = v.min(255) as u8;
            }
        }
    }

    let mut row_scratch = vec![0; inter_w];
    for dy in 0..dst_h {
        let row = &mut inter[dy * inter_w..dy * inter_w + inter_w];
        horizontal_6tap_inplace(row, dst_w, &mut row_scratch);
        let dst_base = (dst_row0 + dy) * stride;
        let n = inter_w.min(buf.len().saturating_sub(dst_base));
        if n > 0 {
            buf[dst_base..dst_base + n].copy_from_slice(&row[..n]);
        }
    }
}

fn horizontal_6tap_inplace(row: &mut [u8], dst_w: usize, scratch: &mut [u8]) {
    if dst_w == 0 || row.is_empty() {
        return;
    }
    if dst_w <= 2 {
        for i in 0..dst_w {
            let a = row.get(2 * i).copied().unwrap_or(0);
            let b = row.get(2 * i + 1).copied().unwrap_or(a);
            row[i] = avg_epu8(a, b);
        }
        return;
    }

    let Some(src) = scratch.get_mut(..row.len()) else {
        return;
    };
    src.copy_from_slice(row);
    let src_w = src.len();

    row[0] = avg_epu8(src[0], src.get(1).copied().unwrap_or(src[0]));

    let mut left = row[0];
    let interior = dst_w.saturating_sub(2);
    for i in 0..interior {
        let k = 5 + 2 * i;
        let p_k = src.get(k).copied().unwrap_or_else(|| src[src_w - 1]);
        let p_km3 = src.get(k - 3).copied().unwrap_or(0);
        let p_km2 = src.get(k - 2).copied().unwrap_or(0);
        let p_km4 = src.get(k - 4).copied().unwrap_or(0);
        let p_even = src.get((k - 1) & !1).copied().unwrap_or(0);
        let v = (u32::from(left)
            + u32::from(p_k)
            + 10 * (u32::from(p_km3) + u32::from(p_km2))
            + 5 * (u32::from(p_km4) + u32::from(p_even))
            + 16)
            >> 5;
        row[i + 1] = v.min(255) as u8;
        left = p_km3;
    }

    let mut i = dst_w.saturating_sub(1).max(1);
    while i < dst_w {
        let a = src.get(2 * i).copied().unwrap_or(0);
        let b = src.get(2 * i + 1).copied().unwrap_or(a);
        row[i] = avg_epu8(a, b);
        i += 1;
    }
}

#[inline]
fn sample(buf: &[u8], stride: usize, row0: usize, y: usize, x: usize) -> u8 {
    buf.get((row0 + y) * stride + x).copied().unwrap_or(0)
}
