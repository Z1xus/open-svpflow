#[cfg(all(
    target_arch = "x86_64",
    any(target_feature = "avx2", target_feature = "ssse3")
))]
use core::arch::x86_64::{__m128i, _mm_unpacklo_epi64};
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use core::arch::x86_64::{
    __m256i, _mm256_abs_epi16, _mm256_add_epi16, _mm256_add_epi32, _mm256_castsi256_si128,
    _mm256_cvtepu8_epi16, _mm256_extracti128_si256, _mm256_hadd_epi16, _mm256_hsub_epi16,
    _mm256_madd_epi16, _mm256_max_epi16, _mm256_set1_epi16, _mm256_setzero_si256, _mm256_sub_epi16,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    _mm_add_epi32, _mm_add_epi64, _mm_cvtsi128_si32, _mm_loadl_epi64, _mm_loadu_si128,
    _mm_sad_epu8, _mm_setzero_si128, _mm_shuffle_epi32,
};

pub(crate) type Kernel = fn(&[u8], &[u8], usize) -> u32;
pub(crate) type PairKernel = fn([&[u8]; 2], [&[u8]; 2], usize) -> u32;
#[derive(Clone, Copy)]
pub(crate) struct Shape {
    pub(crate) width: usize,
    pub(crate) height: usize,
}

impl Shape {
    pub(crate) const fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }

    pub(crate) const fn area(self) -> usize {
        self.width * self.height
    }

    pub(crate) const fn half(self) -> Self {
        Self::new(self.width / 2, self.height / 2)
    }

    pub(crate) const fn span(self, pitch: usize) -> usize {
        (self.height - 1) * pitch + self.width
    }
}

pub(crate) fn luma_kernel(shape: Shape, satd: bool) -> Kernel {
    match (shape.width, shape.height, satd) {
        (32, 32, true) => satd_rows32::<32>,
        (32, 16, true) => satd_rows32::<16>,
        _ => generic(shape, satd),
    }
}

pub(crate) fn chroma420_kernel(luma: Shape, satd: bool) -> Kernel {
    match (luma.width, luma.height) {
        (4, 4) => sad_2x2_swapped,
        (8, 4) => sad_4x2_swapped,
        _ => generic(luma.half(), satd),
    }
}

pub(crate) fn chroma420_pair(luma: Shape, satd: bool) -> Option<PairKernel> {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    if satd && (luma.width, luma.height) == (8, 8) {
        return Some(|src, reference, pitch| {
            assert!(src.iter().all(|s| s.len() >= 16));
            assert!(reference.iter().all(|r| r.len() >= 3 * pitch + 4));
            unsafe { satd_4x4_pair_avx2(src, reference, pitch) }
        });
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    if satd && (luma.width, luma.height) == (16, 8) {
        return Some(|src, reference, pitch| {
            assert!(src.iter().all(|s| s.len() >= 32));
            assert!(reference.iter().all(|r| r.len() >= 3 * pitch + 8));
            unsafe { satd_8x4_pair_avx2(src, reference, pitch) }
        });
    }
    let _ = (luma, satd);
    None
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn stacked_pair(luma: Shape, satd: bool) -> bool {
    satd && (luma.width, luma.height) == (16, 8)
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn stacked_pair(_: Shape, _: bool) -> bool {
    false
}

pub(crate) fn null_kernel(_: &[u8], _: &[u8], _: usize) -> u32 {
    0
}

fn generic(shape: Shape, use_satd: bool) -> Kernel {
    match (shape.width, shape.height, use_satd) {
        (32, 32, false) => sad::<32, 32>,
        (32, 16, false) => sad::<32, 16>,
        (16, 16, false) => sad::<16, 16>,
        (16, 8, false) => sad::<16, 8>,
        (8, 8, false) => sad::<8, 8>,
        (8, 4, false) => sad::<8, 4>,
        (4, 4, false) => sad::<4, 4>,
        (16, 16, true) => wide_satd::<16, 16>().unwrap_or(satd::<16, 16>),
        (16, 8, true) => wide_satd::<16, 8>().unwrap_or(satd::<16, 8>),
        (8, 8, true) => wide_satd::<8, 8>().unwrap_or(satd::<8, 8>),
        (8, 4, true) => satd::<8, 4>,
        (4, 4, true) => satd::<4, 4>,
        (32, 32, true) => satd::<32, 32>,
        (32, 16, true) => satd::<32, 16>,
        _ => null_kernel,
    }
}

pub(crate) fn block_average(block: &[u8], shape: Shape) -> i32 {
    if shape.height < 4 {
        return 0;
    }
    let sum: u32 = block[..shape.area()].iter().map(|&p| u32::from(p)).sum();
    (sum >> shape.area().trailing_zeros()) as i32
}

fn sad<const W: usize, const H: usize>(src: &[u8], reference: &[u8], pitch: usize) -> u32 {
    #[cfg(target_arch = "x86_64")]
    if W >= 8 {
        return unsafe { sad_sse2::<W, H>(src, reference, pitch) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if W >= 4 {
        return simd128::sad(src, W, reference, pitch, W, H);
    }
    sad_scalar(src, reference, pitch, W, H)
}

fn sad_scalar(src: &[u8], reference: &[u8], pitch: usize, width: usize, height: usize) -> u32 {
    let mut sum = 0;
    for row in 0..height {
        let s = &src[row * width..][..width];
        let r = &reference[row * pitch..][..width];
        sum += s
            .iter()
            .zip(r)
            .map(|(&a, &b)| u32::from(a.abs_diff(b)))
            .sum::<u32>();
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn sad_sse2<const W: usize, const H: usize>(
    src: &[u8],
    reference: &[u8],
    pitch: usize,
) -> u32 {
    assert!(src.len() >= W * H && reference.len() >= (H - 1) * pitch + W);
    let mut acc = _mm_setzero_si128();
    for row in 0..H {
        let s = unsafe { src.as_ptr().add(row * W) };
        let r = unsafe { reference.as_ptr().add(row * pitch) };
        if W == 8 {
            let a = unsafe { _mm_loadl_epi64(s.cast()) };
            let b = unsafe { _mm_loadl_epi64(r.cast()) };
            acc = _mm_add_epi64(acc, _mm_sad_epu8(a, b));
        } else {
            for chunk in (0..W).step_by(16) {
                let a = unsafe { _mm_loadu_si128(s.add(chunk).cast()) };
                let b = unsafe { _mm_loadu_si128(r.add(chunk).cast()) };
                acc = _mm_add_epi64(acc, _mm_sad_epu8(a, b));
            }
        }
    }
    let high = _mm_shuffle_epi32::<0b1110>(acc);
    _mm_cvtsi128_si32(_mm_add_epi32(acc, high)) as u32
}

fn sad_2x2_swapped(src: &[u8], reference: &[u8], pitch: usize) -> u32 {
    sad_rows_swapped(src, reference, pitch, 2)
}

fn sad_4x2_swapped(src: &[u8], reference: &[u8], pitch: usize) -> u32 {
    sad_rows_swapped(src, reference, pitch, 4)
}

fn sad_rows_swapped(src: &[u8], reference: &[u8], pitch: usize, width: usize) -> u32 {
    let (first, second) = src[..2 * width].split_at(width);
    let lower = &reference[pitch..pitch + width];
    let upper = &reference[..width];
    let diff = |a: &[u8], b: &[u8]| {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| u32::from(x.abs_diff(y)))
            .sum::<u32>()
    };
    diff(first, lower) + diff(second, upper)
}

fn satd<const W: usize, const H: usize>(src: &[u8], reference: &[u8], pitch: usize) -> u32 {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if W.is_multiple_of(4) && H.is_multiple_of(4) {
        return simd128::satd(src, W, reference, pitch, W, H);
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    if W >= 16 || (W == 8 && H.is_multiple_of(8)) {
        return unsafe { satd_avx2::<W, H>(src, reference, pitch) };
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    if W == 8 && H.is_multiple_of(4) {
        return unsafe { satd_8x4_avx2(src, reference, pitch, H) };
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "ssse3"))]
    if W == 4 && H == 4 {
        return unsafe { satd_4x4_ssse3(src, reference, pitch) };
    }
    satd_scalar(src, W, reference, pitch, W, H)
}

fn satd_scalar(
    src: &[u8],
    src_pitch: usize,
    reference: &[u8],
    pitch: usize,
    width: usize,
    height: usize,
) -> u32 {
    let mut total = 0;
    for ty in (0..height).step_by(4) {
        for tx in (0..width).step_by(4) {
            let mut d = [[0i32; 4]; 4];
            for (y, row) in d.iter_mut().enumerate() {
                for (x, value) in row.iter_mut().enumerate() {
                    let s = src[(ty + y) * src_pitch + tx + x];
                    let r = reference[(ty + y) * pitch + tx + x];
                    *value = i32::from(s) - i32::from(r);
                }
            }
            total += hadamard4x4_abs_sum(d);
        }
    }
    total / 2
}

fn hadamard4x4_abs_sum(mut d: [[i32; 4]; 4]) -> u32 {
    for row in &mut d {
        let [p0, p1, p2, p3] = *row;
        let (s0, s1, d0, d1) = (p0 + p1, p2 + p3, p0 - p1, p2 - p3);
        *row = [s0 + s1, s0 - s1, d0 + d1, d0 - d1];
    }
    let mut sum = 0;
    for x in 0..4 {
        let (s0, s1) = (d[0][x] + d[1][x], d[2][x] + d[3][x]);
        let (d0, d1) = (d[0][x] - d[1][x], d[2][x] - d[3][x]);
        sum += (s0 + s1).unsigned_abs() + (s0 - s1).unsigned_abs();
        sum += (d0 + d1).unsigned_abs() + (d0 - d1).unsigned_abs();
    }
    sum
}

fn satd_rows32<const H: usize>(src: &[u8], reference: &[u8], pitch: usize) -> u32 {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    return (0..H)
        .map(|row| {
            simd128::satd(
                &src[row * 32..][..32],
                8,
                &reference[row * pitch..][..32],
                8,
                8,
                4,
            )
        })
        .sum();
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    return unsafe { satd_rows32_avx2::<H>(src, reference, pitch) };
    #[allow(unreachable_code)]
    (0..H)
        .map(|row| {
            satd_scalar(
                &src[row * 32..][..32],
                8,
                &reference[row * pitch..][..32],
                8,
                8,
                4,
            )
        })
        .sum()
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
const fn rows_per_tile(width: usize) -> usize {
    if width == 8 { 8 } else { 4 }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn load_tile<const W: usize>(
    base: *const u8,
    stride: usize,
    ty: usize,
    tx: usize,
) -> [__m256i; 4] {
    std::array::from_fn(|y| {
        if W == 8 {
            let low = unsafe { _mm_loadl_epi64(base.add((ty + y) * stride + tx).cast()) };
            let high = unsafe { _mm_loadl_epi64(base.add((ty + y + 4) * stride + tx).cast()) };
            _mm256_cvtepu8_epi16(_mm_unpacklo_epi64(low, high))
        } else {
            _mm256_cvtepu8_epi16(unsafe {
                _mm_loadu_si128(base.add((ty + y) * stride + tx).cast())
            })
        }
    })
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
#[inline]
fn tile_maxima([r0, r1, r2, r3]: [__m256i; 4]) -> __m256i {
    let (s0, d0) = (_mm256_add_epi16(r0, r1), _mm256_sub_epi16(r0, r1));
    let (s1, d1) = (_mm256_add_epi16(r2, r3), _mm256_sub_epi16(r2, r3));
    let pair = |a: __m256i, b: __m256i| {
        let sums = _mm256_hadd_epi16(a, b);
        let diffs = _mm256_hsub_epi16(a, b);
        (
            _mm256_abs_epi16(_mm256_hadd_epi16(sums, diffs)),
            _mm256_abs_epi16(_mm256_hsub_epi16(sums, diffs)),
        )
    };
    let (x0, y0) = pair(s0, d0);
    let (x1, y1) = pair(s1, d1);
    _mm256_add_epi16(_mm256_max_epi16(x0, x1), _mm256_max_epi16(y0, y1))
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
#[inline]
fn horizontal_sum(acc: __m256i) -> u32 {
    let folded = _mm_add_epi32(
        _mm256_castsi256_si128(acc),
        _mm256_extracti128_si256::<1>(acc),
    );
    let folded = _mm_add_epi32(folded, _mm_shuffle_epi32::<0b1110>(folded));
    let folded = _mm_add_epi32(folded, _mm_shuffle_epi32::<0b0001>(folded));
    _mm_cvtsi128_si32(folded) as u32
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn satd_avx2<const W: usize, const H: usize>(
    src: &[u8],
    reference: &[u8],
    pitch: usize,
) -> u32 {
    assert!(src.len() >= W * H && reference.len() >= (H - 1) * pitch + W);
    let mut acc = _mm256_setzero_si256();
    for ty in (0..H).step_by(rows_per_tile(W)) {
        for tx in (0..W).step_by(16) {
            let s = unsafe { load_tile::<W>(src.as_ptr(), W, ty, tx) };
            let r = unsafe { load_tile::<W>(reference.as_ptr(), pitch, ty, tx) };
            let diff = std::array::from_fn(|k| _mm256_sub_epi16(s[k], r[k]));
            acc = _mm256_add_epi32(
                acc,
                _mm256_madd_epi16(tile_maxima(diff), _mm256_set1_epi16(1)),
            );
        }
    }
    horizontal_sum(acc)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn satd_8x4_tiles(
    src: [Option<*const u8>; 2],
    src_pitch: usize,
    reference: [Option<*const u8>; 2],
    pitch: usize,
) -> __m256i {
    let load = |base: [Option<*const u8>; 2], stride: usize, y: usize| {
        let half = |ptr: Option<*const u8>| {
            ptr.map_or(_mm_setzero_si128(), |ptr| unsafe {
                _mm_loadl_epi64(ptr.add(y * stride).cast())
            })
        };
        _mm256_cvtepu8_epi16(_mm_unpacklo_epi64(half(base[0]), half(base[1])))
    };
    let diff = std::array::from_fn(|y| {
        _mm256_sub_epi16(load(src, src_pitch, y), load(reference, pitch, y))
    });
    _mm256_madd_epi16(tile_maxima(diff), _mm256_set1_epi16(1))
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn satd_8x4_pair_avx2(src: [&[u8]; 2], reference: [&[u8]; 2], pitch: usize) -> u32 {
    horizontal_sum(unsafe {
        satd_8x4_tiles(
            src.map(|s| Some(s.as_ptr())),
            8,
            reference.map(|r| Some(r.as_ptr())),
            pitch,
        )
    })
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn satd_8x4_avx2(src: &[u8], reference: &[u8], pitch: usize, height: usize) -> u32 {
    assert!(src.len() >= 8 * height && reference.len() >= (height - 1) * pitch + 8);
    let (s, r) = (src.as_ptr(), reference.as_ptr());
    let mut acc = _mm256_setzero_si256();
    for ty in (0..height).step_by(8) {
        let second = ty + 4 < height;
        let pick = |base: *const u8, stride: usize| {
            [
                Some(unsafe { base.add(ty * stride) }),
                second.then(|| unsafe { base.add((ty + 4) * stride) }),
            ]
        };
        acc = _mm256_add_epi32(acc, unsafe {
            satd_8x4_tiles(pick(s, 8), 8, pick(r, pitch), pitch)
        });
    }
    horizontal_sum(acc)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn satd_rows32_avx2<const H: usize>(src: &[u8], reference: &[u8], pitch: usize) -> u32 {
    assert!(src.len() >= 32 * H && reference.len() >= (H - 1) * pitch + 32);
    let (s, r) = (src.as_ptr(), reference.as_ptr());
    let mut acc = _mm256_setzero_si256();
    for row in (0..H).step_by(2) {
        let pick = |base: *const u8, stride: usize| {
            [
                Some(unsafe { base.add(row * stride) }),
                (row + 1 < H).then(|| unsafe { base.add((row + 1) * stride) }),
            ]
        };
        acc = _mm256_add_epi32(acc, unsafe {
            satd_8x4_tiles(pick(s, 32), 8, pick(r, pitch), 8)
        });
    }
    horizontal_sum(acc)
}

#[cfg(all(target_arch = "x86_64", target_feature = "ssse3"))]
#[target_feature(enable = "ssse3")]
unsafe fn rows_4x4(base: *const u8, stride: usize) -> [__m128i; 2] {
    use core::arch::x86_64::{_mm_cvtsi32_si128, _mm_unpacklo_epi8, _mm_unpacklo_epi32};
    let load = |row: usize| {
        _mm_cvtsi32_si128(unsafe { base.add(row * stride).cast::<i32>().read_unaligned() })
    };
    let zero = _mm_setzero_si128();
    [(0, 2), (1, 3)].map(|(a, b)| _mm_unpacklo_epi8(_mm_unpacklo_epi32(load(a), load(b)), zero))
}

#[cfg(all(target_arch = "x86_64", target_feature = "ssse3"))]
#[target_feature(enable = "ssse3")]
fn coefficients_4x4([even, odd]: [__m128i; 2]) -> [__m128i; 2] {
    use core::arch::x86_64::{
        _mm_add_epi16, _mm_setr_epi16, _mm_shufflehi_epi16, _mm_shufflelo_epi16, _mm_sign_epi16,
        _mm_sub_epi16, _mm_unpackhi_epi64,
    };
    let columns = |v: __m128i| {
        let swapped = _mm_unpackhi_epi64(v, v);
        _mm_unpacklo_epi64(_mm_add_epi16(v, swapped), _mm_sub_epi16(v, swapped))
    };
    let odd_lanes = _mm_setr_epi16(1, -1, 1, -1, 1, -1, 1, -1);
    let upper_lanes = _mm_setr_epi16(1, 1, -1, -1, 1, 1, -1, -1);
    [
        columns(_mm_add_epi16(even, odd)),
        columns(_mm_sub_epi16(even, odd)),
    ]
    .map(|v| {
        let pairs = _mm_shufflehi_epi16::<0b1011_0001>(_mm_shufflelo_epi16::<0b1011_0001>(v));
        let stage1 = _mm_add_epi16(_mm_sign_epi16(v, odd_lanes), pairs);
        let quads = _mm_shufflehi_epi16::<0b0100_1110>(_mm_shufflelo_epi16::<0b0100_1110>(stage1));
        _mm_add_epi16(_mm_sign_epi16(stage1, upper_lanes), quads)
    })
}

#[cfg(all(target_arch = "x86_64", target_feature = "ssse3"))]
#[target_feature(enable = "ssse3")]
fn abs_sum_halved(values: [__m128i; 2]) -> u32 {
    use core::arch::x86_64::{_mm_abs_epi16, _mm_add_epi16, _mm_madd_epi16, _mm_set1_epi16};
    let total = _mm_add_epi16(_mm_abs_epi16(values[0]), _mm_abs_epi16(values[1]));
    let wide = _mm_madd_epi16(total, _mm_set1_epi16(1));
    let wide = _mm_add_epi32(wide, _mm_shuffle_epi32::<0b1110>(wide));
    let wide = _mm_add_epi32(wide, _mm_shuffle_epi32::<0b0001>(wide));
    (_mm_cvtsi128_si32(wide) as u32) / 2
}

#[cfg(all(target_arch = "x86_64", target_feature = "ssse3"))]
#[target_feature(enable = "ssse3")]
unsafe fn satd_4x4_ssse3(src: &[u8], reference: &[u8], pitch: usize) -> u32 {
    use core::arch::x86_64::_mm_sub_epi16;
    assert!(src.len() >= 16 && reference.len() >= 3 * pitch + 4);
    let s = unsafe { rows_4x4(src.as_ptr(), 4) };
    let r = unsafe { rows_4x4(reference.as_ptr(), pitch) };
    abs_sum_halved(coefficients_4x4([
        _mm_sub_epi16(s[0], r[0]),
        _mm_sub_epi16(s[1], r[1]),
    ]))
}

#[cfg(target_arch = "x86_64")]
fn wide_satd<const W: usize, const H: usize>() -> Option<Kernel> {
    (std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl"))
    .then_some(|src: &[u8], reference: &[u8], pitch: usize| {
        assert!(src.len() >= W * H && reference.len() >= (H - 1) * pitch + W);
        unsafe { satd_avx512::<W, H>(src.as_ptr(), reference.as_ptr(), pitch) }
    })
}

#[cfg(not(target_arch = "x86_64"))]
fn wide_satd<const W: usize, const H: usize>() -> Option<Kernel> {
    None
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
unsafe fn widen_8x8_avx512(base: *const u8, stride: usize) -> [core::arch::x86_64::__m512i; 2] {
    unsafe {
        widen_rows_avx512([
            base,
            base.add(stride),
            base.add(2 * stride),
            base.add(3 * stride),
            base.add(4 * stride),
            base.add(5 * stride),
            base.add(6 * stride),
            base.add(7 * stride),
        ])
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
unsafe fn widen_rows_avx512(rows: [*const u8; 8]) -> [core::arch::x86_64::__m512i; 2] {
    use core::arch::x86_64::{
        _mm256_mask_set1_epi64, _mm256_set1_epi64x, _mm256_setr_epi8, _mm512_broadcast_i64x4,
        _mm512_maddubs_epi16, _mm512_mask_set1_epi64, _mm512_zextsi256_si512,
    };
    macro_rules! row {
        ($y:expr) => {
            unsafe { rows[$y].cast::<i64>().read_unaligned() }
        };
    }
    macro_rules! gather {
        ($a:expr, $b:expr, $c:expr, $d:expr) => {{
            let low = _mm256_mask_set1_epi64(_mm256_set1_epi64x(row!($a)), 0b0101, row!($b));
            let wide = _mm512_mask_set1_epi64(_mm512_zextsi256_si512(low), 0b1010_0000, row!($c));
            _mm512_mask_set1_epi64(wide, 0b0101_0000, row!($d))
        }};
    }
    let hmul = _mm512_broadcast_i64x4(_mm256_setr_epi8(
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1,
        1, -1, 1, -1,
    ));
    [
        _mm512_maddubs_epi16(gather!(0, 2, 4, 6), hmul),
        _mm512_maddubs_epi16(gather!(1, 3, 5, 7), hmul),
    ]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
fn satd_8x8_from_avx512(
    src: [core::arch::x86_64::__m512i; 2],
    reference: [core::arch::x86_64::__m512i; 2],
) -> core::arch::x86_64::__m512i {
    use core::arch::x86_64::{
        _mm512_abs_epi16, _mm512_add_epi16, _mm512_bsrli_epi128, _mm512_max_epi16,
        _mm512_srli_epi32, _mm512_sub_epi16, _mm512_unpackhi_epi64, _mm512_unpacklo_epi64,
    };
    let even = _mm512_sub_epi16(src[0], reference[0]);
    let odd = _mm512_sub_epi16(src[1], reference[1]);
    let sums = _mm512_add_epi16(even, odd);
    let diffs = _mm512_sub_epi16(odd, even);
    let high = _mm512_unpackhi_epi64(sums, diffs);
    let low = _mm512_unpacklo_epi64(sums, diffs);
    let first = _mm512_abs_epi16(_mm512_add_epi16(low, high));
    let second = _mm512_abs_epi16(_mm512_sub_epi16(high, low));
    let first = _mm512_max_epi16(first, _mm512_bsrli_epi128::<2>(first));
    let second = _mm512_max_epi16(second, _mm512_srli_epi32::<16>(second));
    _mm512_add_epi16(first, second)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn satd_8x8_avx512(
    src: *const u8,
    src_pitch: usize,
    reference: *const u8,
    pitch: usize,
) -> core::arch::x86_64::__m512i {
    unsafe {
        satd_8x8_from_avx512(
            widen_8x8_avx512(src, src_pitch),
            widen_8x8_avx512(reference, pitch),
        )
    }
}

pub(crate) trait LumaCost {
    fn cost(&self, reference: &[u8], pitch: usize) -> u32;

    #[inline]
    fn cost_until(&self, reference: &[u8], pitch: usize, _stop: impl Fn(u32) -> bool) -> u32 {
        self.cost(reference, pitch)
    }
}

pub(crate) trait ChromaCost {
    fn cost(&self, u: &[u8], v: &[u8], pitch: usize) -> u32;
}

pub(crate) struct IndirectLuma<'a> {
    pub(crate) kernel: Kernel,
    pub(crate) src: &'a [u8],
}

impl LumaCost for IndirectLuma<'_> {
    #[inline]
    fn cost(&self, reference: &[u8], pitch: usize) -> u32 {
        (self.kernel)(self.src, reference, pitch)
    }
}

pub(crate) struct IndirectChroma<'a> {
    pub(crate) kernel: Kernel,
    pub(crate) pair: Option<PairKernel>,
    pub(crate) src: [&'a [u8]; 2],
}

impl ChromaCost for IndirectChroma<'_> {
    #[inline]
    fn cost(&self, u: &[u8], v: &[u8], pitch: usize) -> u32 {
        match self.pair {
            Some(pair) => pair(self.src, [u, v], pitch),
            None => (self.kernel)(self.src[0], u, pitch) + (self.kernel)(self.src[1], v, pitch),
        }
    }
}

#[derive(Clone, Copy)]
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) enum Wide {
    S8x8,
    S16x8,
    S16x16,
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn wide_shape(shape: Shape, satd: bool) -> Option<Wide> {
    if !satd
        || !std::arch::is_x86_feature_detected!("avx512bw")
        || !std::arch::is_x86_feature_detected!("avx512vl")
    {
        return None;
    }
    match (shape.width, shape.height) {
        (8, 8) => Some(Wide::S8x8),
        (16, 8) => Some(Wide::S16x8),
        (16, 16) => Some(Wide::S16x16),
        _ => None,
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn wide_shape(_: Shape, _: bool) -> Option<Wide> {
    None
}

#[cfg(target_arch = "x86_64")]
pub(crate) struct WideLuma<const W: usize, const H: usize> {
    rows: [[core::arch::x86_64::__m512i; 2]; 4],
}

#[cfg(target_arch = "x86_64")]
impl<const W: usize, const H: usize> WideLuma<W, H> {
    pub(crate) unsafe fn from_source(source: &WideSource) -> Self {
        Self { rows: source.rows }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct WideSource {
    #[cfg(target_arch = "x86_64")]
    rows: [[core::arch::x86_64::__m512i; 2]; 4],
}

impl Default for WideSource {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

impl WideSource {
    pub(crate) fn prepare(&mut self, wide: Wide, src: &[u8]) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            match wide {
                Wide::S8x8 => fill_wide::<8, 8>(&mut self.rows, src),
                Wide::S16x8 => fill_wide::<16, 8>(&mut self.rows, src),
                Wide::S16x16 => fill_wide::<16, 16>(&mut self.rows, src),
            }
        };
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = (self, wide, src);
            unreachable!()
        }
    }

    #[cfg_attr(not(target_arch = "x86_64"), allow(clippy::trivially_copy_pass_by_ref))]
    pub(crate) fn cost(&self, wide: Wide, reference: &[u8], pitch: usize) -> u32 {
        #[cfg(target_arch = "x86_64")]
        match wide {
            Wide::S8x8 => wide_cost::<8, 8>(&self.rows, reference, pitch),
            Wide::S16x8 => wide_cost::<16, 8>(&self.rows, reference, pitch),
            Wide::S16x16 => wide_cost::<16, 16>(&self.rows, reference, pitch),
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = (self, wide, reference, pitch);
            unreachable!()
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn fill_wide<const W: usize, const H: usize>(
    rows: &mut [[core::arch::x86_64::__m512i; 2]; 4],
    src: &[u8],
) {
    assert!(src.len() >= W * H);
    for (index, (ty, tx)) in tiles::<W, H>().enumerate() {
        rows[index] = unsafe { widen_8x8_avx512(src.as_ptr().add(ty * W + tx), W) };
    }
}

#[cfg(target_arch = "x86_64")]
fn wide_cost<const W: usize, const H: usize>(
    rows: &[[core::arch::x86_64::__m512i; 2]; 4],
    reference: &[u8],
    pitch: usize,
) -> u32 {
    assert!(reference.len() >= (H - 1) * pitch + W);
    unsafe { satd_wide::<W, H>(rows, reference.as_ptr(), pitch) }
}

#[cfg(target_arch = "x86_64")]
impl<const W: usize, const H: usize> LumaCost for WideLuma<W, H> {
    #[inline]
    fn cost(&self, reference: &[u8], pitch: usize) -> u32 {
        assert!(reference.len() >= (H - 1) * pitch + W);
        unsafe { satd_wide::<W, H>(&self.rows, reference.as_ptr(), pitch) }
    }

    #[inline]
    fn cost_until(&self, reference: &[u8], pitch: usize, stop: impl Fn(u32) -> bool) -> u32 {
        assert!(reference.len() >= (H - 1) * pitch + W);
        unsafe { satd_wide_until::<W, H>(&self.rows, reference.as_ptr(), pitch, stop) }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub(crate) struct Pair4x4<'a> {
    pub(crate) src: [&'a [u8]; 2],
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
impl ChromaCost for Pair4x4<'_> {
    #[inline]
    fn cost(&self, u: &[u8], v: &[u8], pitch: usize) -> u32 {
        assert!(self.src.iter().all(|s| s.len() >= 16));
        assert!(u.len() >= 3 * pitch + 4 && v.len() >= 3 * pitch + 4);
        unsafe { satd_4x4_pair_avx2(self.src, [u, v], pitch) }
    }
}

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
pub(crate) struct WidePair8x4 {
    src: [core::arch::x86_64::__m512i; 2],
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
unsafe fn stacked_rows(u: *const u8, v: *const u8, pitch: usize) -> [*const u8; 8] {
    unsafe {
        [
            u,
            u.add(pitch),
            u.add(2 * pitch),
            u.add(3 * pitch),
            v,
            v.add(pitch),
            v.add(2 * pitch),
            v.add(3 * pitch),
        ]
    }
}

#[cfg(target_arch = "x86_64")]
impl WidePair8x4 {
    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    pub(crate) unsafe fn new(src: [&[u8]; 2]) -> Self {
        assert!(src.iter().all(|s| s.len() >= 32));
        Self {
            src: unsafe { widen_rows_avx512(stacked_rows(src[0].as_ptr(), src[1].as_ptr(), 8)) },
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
unsafe fn satd_stacked_8x4(
    src: [core::arch::x86_64::__m512i; 2],
    u: *const u8,
    v: *const u8,
    pitch: usize,
) -> u32 {
    use core::arch::x86_64::{_mm512_madd_epi16, _mm512_reduce_add_epi32, _mm512_set1_epi32};
    let reference = unsafe { widen_rows_avx512(stacked_rows(u, v, pitch)) };
    _mm512_reduce_add_epi32(_mm512_madd_epi16(
        satd_8x8_from_avx512(src, reference),
        _mm512_set1_epi32(1),
    )) as u32
}

#[cfg(target_arch = "x86_64")]
impl ChromaCost for WidePair8x4 {
    #[inline]
    fn cost(&self, u: &[u8], v: &[u8], pitch: usize) -> u32 {
        assert!(u.len() >= 3 * pitch + 8 && v.len() >= 3 * pitch + 8);
        unsafe { satd_stacked_8x4(self.src, u.as_ptr(), v.as_ptr(), pitch) }
    }
}

#[cfg(target_arch = "x86_64")]
fn tiles<const W: usize, const H: usize>() -> impl Iterator<Item = (usize, usize)> {
    (0..H)
        .step_by(8)
        .flat_map(|ty| (0..W).step_by(8).map(move |tx| (ty, tx)))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
unsafe fn satd_wide<const W: usize, const H: usize>(
    src: &[[core::arch::x86_64::__m512i; 2]; 4],
    reference: *const u8,
    pitch: usize,
) -> u32 {
    use core::arch::x86_64::{
        _mm512_add_epi32, _mm512_madd_epi16, _mm512_reduce_add_epi32, _mm512_set1_epi32,
        _mm512_setzero_si512,
    };
    let mut acc = _mm512_setzero_si512();
    for (index, (ty, tx)) in tiles::<W, H>().enumerate() {
        let widened = unsafe { widen_8x8_avx512(reference.add(ty * pitch + tx), pitch) };
        let block = satd_8x8_from_avx512(src[index], widened);
        acc = _mm512_add_epi32(acc, _mm512_madd_epi16(block, _mm512_set1_epi32(1)));
    }
    _mm512_reduce_add_epi32(acc) as u32
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
unsafe fn satd_wide_until<const W: usize, const H: usize>(
    src: &[[core::arch::x86_64::__m512i; 2]; 4],
    reference: *const u8,
    pitch: usize,
    stop: impl Fn(u32) -> bool,
) -> u32 {
    use core::arch::x86_64::{
        _mm512_add_epi32, _mm512_madd_epi16, _mm512_reduce_add_epi32, _mm512_set1_epi32,
        _mm512_setzero_si512,
    };
    let count = (W / 8) * (H / 8);
    let mut acc = _mm512_setzero_si512();
    for (index, (ty, tx)) in tiles::<W, H>().enumerate() {
        let widened = unsafe { widen_8x8_avx512(reference.add(ty * pitch + tx), pitch) };
        let block = satd_8x8_from_avx512(src[index], widened);
        acc = _mm512_add_epi32(acc, _mm512_madd_epi16(block, _mm512_set1_epi32(1)));
        if index + 1 < count {
            let partial = _mm512_reduce_add_epi32(acc) as u32;
            if stop(partial) {
                return partial;
            }
        }
    }
    _mm512_reduce_add_epi32(acc) as u32
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn satd_avx512<const W: usize, const H: usize>(
    src: *const u8,
    reference: *const u8,
    pitch: usize,
) -> u32 {
    use core::arch::x86_64::{
        _mm512_add_epi32, _mm512_madd_epi16, _mm512_reduce_add_epi32, _mm512_set1_epi32,
        _mm512_setzero_si512,
    };
    let mut acc = _mm512_setzero_si512();
    for ty in (0..H).step_by(8) {
        for tx in (0..W).step_by(8) {
            let block = unsafe {
                satd_8x8_avx512(
                    src.add(ty * W + tx),
                    W,
                    reference.add(ty * pitch + tx),
                    pitch,
                )
            };
            acc = _mm512_add_epi32(acc, _mm512_madd_epi16(block, _mm512_set1_epi32(1)));
        }
    }
    _mm512_reduce_add_epi32(acc) as u32
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn satd_4x4_pair_avx2(src: [&[u8]; 2], reference: [&[u8]; 2], pitch: usize) -> u32 {
    use core::arch::x86_64::{
        _mm256_set_m128i, _mm256_set1_epi32, _mm256_srli_epi32, _mm256_unpackhi_epi64,
        _mm256_unpacklo_epi64,
    };
    let s: [[__m128i; 2]; 2] = src.map(|p| unsafe { rows_4x4(p.as_ptr(), 4) });
    let r: [[__m128i; 2]; 2] = reference.map(|p| unsafe { rows_4x4(p.as_ptr(), pitch) });
    let even = _mm256_sub_epi16(
        _mm256_set_m128i(s[1][0], s[0][0]),
        _mm256_set_m128i(r[1][0], r[0][0]),
    );
    let odd = _mm256_sub_epi16(
        _mm256_set_m128i(s[1][1], s[0][1]),
        _mm256_set_m128i(r[1][1], r[0][1]),
    );
    let columns = |v: __m256i| {
        let swapped = _mm256_unpackhi_epi64(v, v);
        _mm256_unpacklo_epi64(_mm256_add_epi16(v, swapped), _mm256_sub_epi16(v, swapped))
    };
    let a = columns(_mm256_add_epi16(even, odd));
    let b = columns(_mm256_sub_epi16(even, odd));
    let sums = _mm256_abs_epi16(_mm256_hadd_epi16(a, b));
    let diffs = _mm256_abs_epi16(_mm256_hsub_epi16(a, b));
    let maxima = _mm256_add_epi16(
        _mm256_max_epi16(sums, _mm256_srli_epi32::<16>(sums)),
        _mm256_max_epi16(diffs, _mm256_srli_epi32::<16>(diffs)),
    );
    horizontal_sum(_mm256_madd_epi16(maxima, _mm256_set1_epi32(1)))
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code, clippy::many_single_char_names)]
mod simd128 {
    use core::arch::wasm32::{
        i16x8_abs, i16x8_add, i16x8_shuffle, i16x8_sub, i32x4_add, i32x4_extadd_pairwise_i16x8,
        i32x4_extract_lane, i32x4_shuffle, u8x16_sub_sat, u16x8_extadd_pairwise_u8x16,
        u16x8_load_extend_u8x8, u32x4_extadd_pairwise_u16x8, v128, v128_load, v128_load32_zero,
        v128_load64_zero, v128_or,
    };

    fn span(width: usize, height: usize, pitch: usize) -> usize {
        (height - 1) * pitch + width
    }

    fn horizontal(v: v128) -> v128 {
        let swapped = i16x8_shuffle::<1, 0, 3, 2, 5, 4, 7, 6>(v, v);
        let sums = i16x8_add(v, swapped);
        let diffs = i16x8_sub(v, swapped);
        let u = i16x8_shuffle::<0, 8, 2, 10, 4, 12, 6, 14>(sums, diffs);
        let w = i16x8_shuffle::<2, 3, 0, 1, 6, 7, 4, 5>(u, u);
        i16x8_add(i16x8_abs(i16x8_add(u, w)), i16x8_abs(i16x8_sub(u, w)))
    }

    fn total(acc: v128) -> u32 {
        let folded = i32x4_add(acc, i32x4_shuffle::<2, 3, 0, 1>(acc, acc));
        let folded = i32x4_add(folded, i32x4_shuffle::<1, 0, 3, 2>(folded, folded));
        i32x4_extract_lane::<0>(folded) as u32
    }

    unsafe fn row8(src: *const u8, reference: *const u8) -> v128 {
        unsafe {
            i16x8_sub(
                u16x8_load_extend_u8x8(src),
                u16x8_load_extend_u8x8(reference),
            )
        }
    }

    unsafe fn row4x2(src: *const u8, sp: usize, reference: *const u8, rp: usize) -> v128 {
        let mut a = [0u8; 8];
        let mut b = [0u8; 8];
        unsafe {
            std::ptr::copy_nonoverlapping(src, a.as_mut_ptr(), 4);
            std::ptr::copy_nonoverlapping(src.add(sp), a.as_mut_ptr().add(4), 4);
            std::ptr::copy_nonoverlapping(reference, b.as_mut_ptr(), 4);
            std::ptr::copy_nonoverlapping(reference.add(rp), b.as_mut_ptr().add(4), 4);
            row8(a.as_ptr(), b.as_ptr())
        }
    }

    pub(super) fn satd(
        src: &[u8],
        sp: usize,
        reference: &[u8],
        rp: usize,
        width: usize,
        height: usize,
    ) -> u32 {
        assert!(src.len() >= span(width, height, sp) && reference.len() >= span(width, height, rp));
        let (s, r) = (src.as_ptr(), reference.as_ptr());
        let mut acc = v128_zero();
        if width == 4 {
            for ty in (0..height).step_by(4) {
                let (top, bottom) = unsafe {
                    (
                        row4x2(s.add(ty * sp), sp, r.add(ty * rp), rp),
                        row4x2(s.add((ty + 2) * sp), sp, r.add((ty + 2) * rp), rp),
                    )
                };
                let sum = i16x8_add(top, bottom);
                let diff = i16x8_sub(top, bottom);
                let sum_swap = i16x8_shuffle::<4, 5, 6, 7, 0, 1, 2, 3>(sum, sum);
                let diff_swap = i16x8_shuffle::<4, 5, 6, 7, 0, 1, 2, 3>(diff, diff);
                let part = i16x8_add(
                    i16x8_add(
                        horizontal(i16x8_add(sum, sum_swap)),
                        horizontal(i16x8_sub(sum, sum_swap)),
                    ),
                    i16x8_add(
                        horizontal(i16x8_add(diff, diff_swap)),
                        horizontal(i16x8_sub(diff, diff_swap)),
                    ),
                );
                acc = i32x4_add(acc, i32x4_extadd_pairwise_i16x8(part));
            }
            return total(acc) / 8;
        }
        for ty in (0..height).step_by(4) {
            for tx in (0..width).step_by(8) {
                let d: [v128; 4] = std::array::from_fn(|k| unsafe {
                    row8(s.add((ty + k) * sp + tx), r.add((ty + k) * rp + tx))
                });
                let (a, b) = (i16x8_add(d[0], d[1]), i16x8_add(d[2], d[3]));
                let (c, e) = (i16x8_sub(d[0], d[1]), i16x8_sub(d[2], d[3]));
                let part = i16x8_add(
                    i16x8_add(horizontal(i16x8_add(a, b)), horizontal(i16x8_sub(a, b))),
                    i16x8_add(horizontal(i16x8_add(c, e)), horizontal(i16x8_sub(c, e))),
                );
                acc = i32x4_add(acc, i32x4_extadd_pairwise_i16x8(part));
            }
        }
        total(acc) / 4
    }

    fn v128_zero() -> v128 {
        core::arch::wasm32::i32x4_splat(0)
    }

    pub(super) fn sad(
        src: &[u8],
        sp: usize,
        reference: &[u8],
        rp: usize,
        width: usize,
        height: usize,
    ) -> u32 {
        assert!(src.len() >= span(width, height, sp) && reference.len() >= span(width, height, rp));
        let (s, r) = (src.as_ptr(), reference.as_ptr());
        let mut acc = v128_zero();
        let mut wide = v128_zero();
        for y in 0..height {
            let mut x = 0;
            while x < width {
                let (a, b, step) = unsafe {
                    let (a, b) = (s.add(y * sp + x), r.add(y * rp + x));
                    if width - x >= 16 {
                        (v128_load(a.cast()), v128_load(b.cast()), 16)
                    } else if width - x >= 8 {
                        (v128_load64_zero(a.cast()), v128_load64_zero(b.cast()), 8)
                    } else {
                        (v128_load32_zero(a.cast()), v128_load32_zero(b.cast()), 4)
                    }
                };
                let d = v128_or(u8x16_sub_sat(a, b), u8x16_sub_sat(b, a));
                acc = i16x8_add(acc, u16x8_extadd_pairwise_u8x16(d));
                x += step;
            }
            if y % 32 == 31 {
                wide = i32x4_add(wide, u32x4_extadd_pairwise_u16x8(acc));
                acc = v128_zero();
            }
        }
        total(i32x4_add(wide, u32x4_extadd_pairwise_u16x8(acc)))
    }
}
