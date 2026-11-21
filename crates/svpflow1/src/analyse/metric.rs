#[cfg(all(
    target_arch = "x86_64",
    any(target_feature = "avx2", target_feature = "ssse3")
))]
use core::arch::x86_64::{__m128i, _mm_unpacklo_epi64};
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use core::arch::x86_64::{
    __m256i, _mm256_abs_epi16, _mm256_add_epi16, _mm256_add_epi32, _mm256_add_epi64,
    _mm256_castsi128_si256, _mm256_castsi256_si128, _mm256_extracti128_si256,
    _mm256_inserti128_si256, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_maddubs_epi16,
    _mm256_max_epi16, _mm256_sad_epu8, _mm256_set_m128i, _mm256_set1_epi32, _mm256_setr_epi8,
    _mm256_setzero_si256, _mm256_srli_epi32, _mm256_sub_epi16,
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

pub(crate) fn luma_strided_kernel(shape: Shape) -> Kernel {
    match (shape.width, shape.height) {
        (32, 32) => sad_strided::<32, 32>,
        (32, 16) => sad_strided::<32, 16>,
        (16, 16) => sad_strided::<16, 16>,
        (16, 8) => sad_strided::<16, 8>,
        (8, 8) => sad_strided::<8, 8>,
        (8, 4) => sad_strided::<8, 4>,
        (4, 4) => sad_strided::<4, 4>,
        _ => generic(shape, false),
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

pub(crate) fn block_average(block: &[u8], pitch: usize, shape: Shape) -> i32 {
    if shape.height < 4 {
        return 0;
    }
    #[cfg(target_arch = "x86_64")]
    if pitch == shape.width && shape.area().is_multiple_of(16) {
        let sum = unsafe { contiguous_sum(&block[..shape.area()]) };
        return (sum >> shape.area().trailing_zeros()) as i32;
    }
    #[cfg(target_arch = "aarch64")]
    if pitch == shape.width && shape.area().is_multiple_of(16) {
        let sum = neon::contiguous_sum(&block[..shape.area()]);
        return (sum >> shape.area().trailing_zeros()) as i32;
    }
    let sum = match shape.width {
        2 => rows_sum::<2>(block, pitch, shape.height),
        4 => rows_sum::<4>(block, pitch, shape.height),
        8 => rows_sum::<8>(block, pitch, shape.height),
        16 => rows_sum::<16>(block, pitch, shape.height),
        _ => rows_sum::<32>(block, pitch, shape.height),
    };
    (sum >> shape.area().trailing_zeros()) as i32
}

#[cfg(target_arch = "x86_64")]
unsafe fn contiguous_sum(block: &[u8]) -> u32 {
    if block.len() >= 64 && std::arch::is_x86_feature_detected!("avx512bw") {
        return unsafe { contiguous_sum_avx512(block) };
    }
    if block.len() >= 32 && std::arch::is_x86_feature_detected!("avx2") {
        return unsafe { contiguous_sum_avx2(block) };
    }
    unsafe { contiguous_sum_sse2(block) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn contiguous_sum_avx512(block: &[u8]) -> u32 {
    use core::arch::x86_64::{
        _mm512_add_epi64, _mm512_loadu_si512, _mm512_reduce_add_epi64, _mm512_sad_epu8,
        _mm512_setzero_si512,
    };
    let zero = _mm512_setzero_si512();
    let mut acc = zero;
    let (chunks, rem) = block.as_chunks::<64>();
    for chunk in chunks {
        let value = unsafe { _mm512_loadu_si512(chunk.as_ptr().cast()) };
        acc = _mm512_add_epi64(acc, _mm512_sad_epu8(value, zero));
    }
    let total = _mm512_reduce_add_epi64(acc) as u32;
    if rem.is_empty() {
        total
    } else {
        total + unsafe { contiguous_sum_sse2(rem) }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn contiguous_sum_avx2(block: &[u8]) -> u32 {
    use core::arch::x86_64::{
        _mm256_add_epi64, _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_loadu_si256,
        _mm256_sad_epu8, _mm256_setzero_si256,
    };
    let zero = _mm256_setzero_si256();
    let mut acc = zero;
    let (chunks, rem) = block.as_chunks::<32>();
    for chunk in chunks {
        let value = unsafe { _mm256_loadu_si256(chunk.as_ptr().cast()) };
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(value, zero));
    }
    let sum = _mm_add_epi64(
        _mm256_castsi256_si128(acc),
        _mm256_extracti128_si256::<1>(acc),
    );
    let total = _mm_cvtsi128_si32(_mm_add_epi32(sum, _mm_shuffle_epi32::<0b1110>(sum))) as u32;
    if rem.is_empty() {
        total
    } else {
        total + unsafe { contiguous_sum_sse2(rem) }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn contiguous_sum_sse2(block: &[u8]) -> u32 {
    let mut acc = _mm_setzero_si128();
    for chunk in block.as_chunks::<16>().0 {
        let value = unsafe { _mm_loadu_si128(chunk.as_ptr().cast()) };
        acc = _mm_add_epi64(acc, _mm_sad_epu8(value, _mm_setzero_si128()));
    }
    let high = _mm_shuffle_epi32::<0b1110>(acc);
    _mm_cvtsi128_si32(_mm_add_epi32(acc, high)) as u32
}

fn rows_sum<const W: usize>(block: &[u8], pitch: usize, height: usize) -> u32 {
    (0..height)
        .map(|row| {
            let row: &[u8; W] = block[row * pitch..][..W].try_into().unwrap();
            row.iter().map(|&p| u32::from(p)).sum::<u32>()
        })
        .sum()
}

fn sad<const W: usize, const H: usize>(src: &[u8], reference: &[u8], pitch: usize) -> u32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    if (W == 8 || W == 16 || W == 32) && H.is_multiple_of(2) {
        assert!(src.len() >= W * H && reference.len() >= (H - 1) * pitch + W);
        if W == 32 && std::arch::is_x86_feature_detected!("avx512bw") {
            return unsafe {
                sad_rows_pitched_avx512::<32>(src.as_ptr(), W, reference.as_ptr(), pitch, H)
            };
        }
        return unsafe { sad_until::<W, H>(src.as_ptr(), reference.as_ptr(), pitch, |_| false) };
    }
    #[cfg(target_arch = "x86_64")]
    if W >= 8 {
        assert!(src.len() >= W * H && reference.len() >= (H - 1) * pitch + W);
        return unsafe { sad_rows_pitched::<W>(src.as_ptr(), W, reference.as_ptr(), pitch, H) };
    }
    #[cfg(target_arch = "aarch64")]
    if W >= 4 {
        return neon::sad(src, W, reference, pitch, W, H);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if W >= 4 {
        return simd128::sad(src, W, reference, pitch, W, H);
    }
    sad_scalar(src, reference, pitch, W, H)
}

fn sad_strided<const W: usize, const H: usize>(src: &[u8], reference: &[u8], pitch: usize) -> u32 {
    assert!(src.len() >= W * H && reference.len() >= (H - 1) * pitch + W);
    #[cfg(target_arch = "x86_64")]
    if W >= 8 {
        return unsafe {
            sad_rows_pitched::<W>(src.as_ptr(), W * 2, reference.as_ptr(), pitch * 2, H / 2)
        } << 1;
    }
    #[cfg(target_arch = "aarch64")]
    if W >= 4 {
        return neon::sad(src, W * 2, reference, pitch * 2, W, H / 2) << 1;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if W >= 4 {
        return simd128::sad(src, W * 2, reference, pitch * 2, W, H / 2) << 1;
    }
    let mut sum = 0u32;
    for row in (0..H).step_by(2) {
        let s = &src[row * W..][..W];
        let r = &reference[row * pitch..][..W];
        sum += s
            .iter()
            .zip(r)
            .map(|(&a, &b)| u32::from(a.abs_diff(b)))
            .sum::<u32>();
    }
    sum << 1
}

pub(crate) fn sad_pitched_strided(
    shape: Shape,
    a: &[u8],
    a_pitch: usize,
    b: &[u8],
    b_pitch: usize,
) -> u32 {
    if shape.height < 4 || !shape.height.is_multiple_of(2) {
        return sad_pitched(shape, a, a_pitch, b, b_pitch);
    }
    let half = Shape::new(shape.width, shape.height / 2);
    sad_pitched(half, a, a_pitch * 2, b, b_pitch * 2) << 1
}

pub(crate) fn sad_pitched(shape: Shape, a: &[u8], a_pitch: usize, b: &[u8], b_pitch: usize) -> u32 {
    assert!(a.len() >= shape.span(a_pitch) && b.len() >= shape.span(b_pitch));
    #[cfg(target_arch = "aarch64")]
    if shape.width >= 4 && shape.width.is_multiple_of(4) {
        return neon::sad(a, a_pitch, b, b_pitch, shape.width, shape.height);
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let (a, b) = (a.as_ptr(), b.as_ptr());
        match (shape.width, shape.height) {
            (4, 4) => return sad_4x4_pitched(a, a_pitch, b, b_pitch),
            (8, h) if h >= 2 => return sad_rows_pitched::<8>(a, a_pitch, b, b_pitch, h),
            (16, h) if h >= 2 => return sad_rows_pitched::<16>(a, a_pitch, b, b_pitch, h),
            (32, h) if h >= 2 => return sad_rows_pitched::<32>(a, a_pitch, b, b_pitch, h),
            _ => {}
        }
    }
    (0..shape.height)
        .map(|row| {
            a[row * a_pitch..][..shape.width]
                .iter()
                .zip(&b[row * b_pitch..][..shape.width])
                .map(|(&x, &y)| u32::from(x.abs_diff(y)))
                .sum::<u32>()
        })
        .sum()
}

#[cfg(target_arch = "x86_64")]
unsafe fn sad_rows_pitched<const W: usize>(
    src: *const u8,
    src_pitch: usize,
    reference: *const u8,
    reference_pitch: usize,
    height: usize,
) -> u32 {
    if W >= 16 && height >= 2 && std::arch::is_x86_feature_detected!("avx512bw") {
        return unsafe {
            sad_rows_pitched_avx512::<W>(src, src_pitch, reference, reference_pitch, height)
        };
    }
    if W >= 16 && std::arch::is_x86_feature_detected!("avx2") {
        return unsafe {
            sad_rows_pitched_avx2::<W>(src, src_pitch, reference, reference_pitch, height)
        };
    }
    unsafe { sad_rows_pitched_sse2::<W>(src, src_pitch, reference, reference_pitch, height) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn sad_rows_pitched_avx512<const W: usize>(
    src: *const u8,
    src_pitch: usize,
    reference: *const u8,
    reference_pitch: usize,
    height: usize,
) -> u32 {
    use core::arch::x86_64::{
        _mm256_castsi128_si256, _mm256_inserti128_si256, _mm256_loadu_si256, _mm512_add_epi64,
        _mm512_castsi256_si512, _mm512_inserti64x4, _mm512_reduce_add_epi64, _mm512_sad_epu8,
        _mm512_setzero_si512,
    };
    let mut acc = _mm512_setzero_si512();
    let mut row = 0;
    if W == 32 {
        while row + 2 <= height {
            let (a, b) = unsafe {
                let s0 = _mm256_loadu_si256(src.add(row * src_pitch).cast());
                let s1 = _mm256_loadu_si256(src.add((row + 1) * src_pitch).cast());
                let r0 = _mm256_loadu_si256(reference.add(row * reference_pitch).cast());
                let r1 = _mm256_loadu_si256(reference.add((row + 1) * reference_pitch).cast());
                (
                    _mm512_inserti64x4::<1>(_mm512_castsi256_si512(s0), s1),
                    _mm512_inserti64x4::<1>(_mm512_castsi256_si512(r0), r1),
                )
            };
            acc = _mm512_add_epi64(acc, _mm512_sad_epu8(a, b));
            row += 2;
        }
    } else if W == 16 {
        while row + 4 <= height {
            let (a, b) = unsafe {
                let s01 = _mm256_inserti128_si256::<1>(
                    _mm256_castsi128_si256(_mm_loadu_si128(src.add(row * src_pitch).cast())),
                    _mm_loadu_si128(src.add((row + 1) * src_pitch).cast()),
                );
                let s23 = _mm256_inserti128_si256::<1>(
                    _mm256_castsi128_si256(_mm_loadu_si128(src.add((row + 2) * src_pitch).cast())),
                    _mm_loadu_si128(src.add((row + 3) * src_pitch).cast()),
                );
                let r01 = _mm256_inserti128_si256::<1>(
                    _mm256_castsi128_si256(_mm_loadu_si128(
                        reference.add(row * reference_pitch).cast(),
                    )),
                    _mm_loadu_si128(reference.add((row + 1) * reference_pitch).cast()),
                );
                let r23 = _mm256_inserti128_si256::<1>(
                    _mm256_castsi128_si256(_mm_loadu_si128(
                        reference.add((row + 2) * reference_pitch).cast(),
                    )),
                    _mm_loadu_si128(reference.add((row + 3) * reference_pitch).cast()),
                );
                (
                    _mm512_inserti64x4::<1>(_mm512_castsi256_si512(s01), s23),
                    _mm512_inserti64x4::<1>(_mm512_castsi256_si512(r01), r23),
                )
            };
            acc = _mm512_add_epi64(acc, _mm512_sad_epu8(a, b));
            row += 4;
        }
    }
    let total = _mm512_reduce_add_epi64(acc) as u32;
    if row < height {
        total
            + unsafe {
                sad_rows_pitched_avx2::<W>(
                    src.add(row * src_pitch),
                    src_pitch,
                    reference.add(row * reference_pitch),
                    reference_pitch,
                    height - row,
                )
            }
    } else {
        total
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sad_rows_pitched_avx2<const W: usize>(
    src: *const u8,
    src_pitch: usize,
    reference: *const u8,
    reference_pitch: usize,
    height: usize,
) -> u32 {
    use core::arch::x86_64::{
        _mm256_add_epi64, _mm256_castsi128_si256, _mm256_castsi256_si128, _mm256_extracti128_si256,
        _mm256_inserti128_si256, _mm256_loadu_si256, _mm256_sad_epu8, _mm256_setzero_si256,
    };
    let mut acc = _mm256_setzero_si256();
    let mut row = 0;
    if W == 32 {
        while row < height {
            let (a, b) = unsafe {
                (
                    _mm256_loadu_si256(src.add(row * src_pitch).cast()),
                    _mm256_loadu_si256(reference.add(row * reference_pitch).cast()),
                )
            };
            acc = _mm256_add_epi64(acc, _mm256_sad_epu8(a, b));
            row += 1;
        }
    } else if W == 16 {
        while row + 2 <= height {
            let (a, b) = unsafe {
                let s = _mm256_inserti128_si256::<1>(
                    _mm256_castsi128_si256(_mm_loadu_si128(src.add(row * src_pitch).cast())),
                    _mm_loadu_si128(src.add((row + 1) * src_pitch).cast()),
                );
                let r = _mm256_inserti128_si256::<1>(
                    _mm256_castsi128_si256(_mm_loadu_si128(
                        reference.add(row * reference_pitch).cast(),
                    )),
                    _mm_loadu_si128(reference.add((row + 1) * reference_pitch).cast()),
                );
                (s, r)
            };
            acc = _mm256_add_epi64(acc, _mm256_sad_epu8(a, b));
            row += 2;
        }
    }
    let sum = _mm_add_epi64(
        _mm256_castsi256_si128(acc),
        _mm256_extracti128_si256::<1>(acc),
    );
    let total = _mm_cvtsi128_si32(_mm_add_epi32(sum, _mm_shuffle_epi32::<0b1110>(sum))) as u32;
    if row < height {
        total
            + unsafe {
                sad_rows_pitched_sse2::<W>(
                    src.add(row * src_pitch),
                    src_pitch,
                    reference.add(row * reference_pitch),
                    reference_pitch,
                    height - row,
                )
            }
    } else {
        total
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn sad_rows_pitched_sse2<const W: usize>(
    src: *const u8,
    src_pitch: usize,
    reference: *const u8,
    reference_pitch: usize,
    height: usize,
) -> u32 {
    let mut acc = _mm_setzero_si128();
    for row in 0..height {
        let (source, target) = unsafe {
            (
                src.add(row * src_pitch),
                reference.add(row * reference_pitch),
            )
        };
        if W == 8 {
            let a = unsafe { _mm_loadl_epi64(source.cast()) };
            let b = unsafe { _mm_loadl_epi64(target.cast()) };
            acc = _mm_add_epi64(acc, _mm_sad_epu8(a, b));
        } else {
            for chunk in (0..W).step_by(16) {
                let a = unsafe { _mm_loadu_si128(source.add(chunk).cast()) };
                let b = unsafe { _mm_loadu_si128(target.add(chunk).cast()) };
                acc = _mm_add_epi64(acc, _mm_sad_epu8(a, b));
            }
        }
    }
    let high = _mm_shuffle_epi32::<0b1110>(acc);
    _mm_cvtsi128_si32(_mm_add_epi32(acc, high)) as u32
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn sad_4x4_pitched(a: *const u8, a_pitch: usize, b: *const u8, b_pitch: usize) -> u32 {
    use core::arch::x86_64::{_mm_cvtsi32_si128, _mm_unpacklo_epi32};
    let gather = |p: *const u8, pitch: usize| unsafe {
        let load =
            |row: usize| _mm_cvtsi32_si128(p.add(row * pitch).cast::<i32>().read_unaligned());
        _mm_unpacklo_epi64(
            _mm_unpacklo_epi32(load(0), load(1)),
            _mm_unpacklo_epi32(load(2), load(3)),
        )
    };
    let sums = _mm_sad_epu8(gather(a, a_pitch), gather(b, b_pitch));
    let high = _mm_shuffle_epi32::<0b1110>(sums);
    _mm_cvtsi128_si32(_mm_add_epi32(sums, high)) as u32
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
    #[cfg(target_arch = "aarch64")]
    if W.is_multiple_of(4) && H.is_multiple_of(4) {
        return neon::satd(src, W, reference, pitch, W, H);
    }
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
    #[cfg(target_arch = "aarch64")]
    return (0..H)
        .map(|row| {
            neon::satd(
                &src[row * 32..][..32],
                8,
                &reference[row * pitch..][..32],
                8,
                8,
                4,
            )
        })
        .sum();
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
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
    {
        return unsafe { satd_rows32_avx512::<H>(src, reference, pitch) };
    }
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn satd_rows32_avx512<const H: usize>(
    src: &[u8],
    reference: &[u8],
    pitch: usize,
) -> u32 {
    use core::arch::x86_64::{
        _mm512_add_epi32, _mm512_madd_epi16, _mm512_reduce_add_epi32, _mm512_set1_epi32,
        _mm512_setzero_si512,
    };
    assert!(src.len() >= 32 * H && reference.len() >= (H - 1) * pitch + 32);
    let (s, r) = (src.as_ptr(), reference.as_ptr());
    let mut acc = _mm512_setzero_si512();
    let one = _mm512_set1_epi32(1);
    for row in (0..H).step_by(2) {
        let s0 = unsafe { s.add(row * 32) };
        let r0 = unsafe { r.add(row * pitch) };
        let (s1, r1) = if row + 1 < H {
            unsafe { (s.add((row + 1) * 32), r.add((row + 1) * pitch)) }
        } else {
            (s0, r0)
        };
        let block = unsafe {
            let sw = widen_rows_avx512([
                s0,
                s0.add(8),
                s0.add(16),
                s0.add(24),
                s1,
                s1.add(8),
                s1.add(16),
                s1.add(24),
            ]);
            let rw = widen_rows_avx512([
                r0,
                r0.add(8),
                r0.add(16),
                r0.add(24),
                r1,
                r1.add(8),
                r1.add(16),
                r1.add(24),
            ]);
            satd_8x8_from_avx512(sw, rw)
        };
        let term = _mm512_madd_epi16(block, one);
        if row + 1 < H {
            acc = _mm512_add_epi32(acc, term);
        } else {
            use core::arch::x86_64::_mm512_mask_add_epi32;
            acc = _mm512_mask_add_epi32(acc, 0x00ff, acc, term);
        }
    }
    _mm512_reduce_add_epi32(acc) as u32
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
const fn rows_per_tile(width: usize) -> usize {
    if width == 8 { 8 } else { 4 }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
#[inline]
fn hmul_avx2() -> __m256i {
    _mm256_setr_epi8(
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1,
        1, -1, 1, -1,
    )
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
#[inline]
fn widen_16_avx2(row: __m128i, hmul: __m256i) -> __m256i {
    _mm256_maddubs_epi16(_mm256_set_m128i(row, row), hmul)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn load_tile<const W: usize>(
    base: *const u8,
    stride: usize,
    ty: usize,
    tx: usize,
    hmul: __m256i,
) -> [__m256i; 4] {
    std::array::from_fn(|y| {
        let row = if W == 8 {
            let low = unsafe { _mm_loadl_epi64(base.add((ty + y) * stride + tx).cast()) };
            let high = unsafe { _mm_loadl_epi64(base.add((ty + y + 4) * stride + tx).cast()) };
            _mm_unpacklo_epi64(low, high)
        } else {
            unsafe { _mm_loadu_si128(base.add((ty + y) * stride + tx).cast()) }
        };
        widen_16_avx2(row, hmul)
    })
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
#[inline]
fn tile_maxima([r0, r1, r2, r3]: [__m256i; 4]) -> __m256i {
    let (s0, d0) = (_mm256_add_epi16(r0, r1), _mm256_sub_epi16(r0, r1));
    let (s1, d1) = (_mm256_add_epi16(r2, r3), _mm256_sub_epi16(r2, r3));
    let fold = |v: __m256i| {
        let a = _mm256_abs_epi16(v);
        _mm256_max_epi16(a, _mm256_srli_epi32::<16>(a))
    };
    let m0 = fold(_mm256_add_epi16(s0, s1));
    let m1 = fold(_mm256_sub_epi16(s0, s1));
    let m2 = fold(_mm256_add_epi16(d0, d1));
    let m3 = fold(_mm256_sub_epi16(d0, d1));
    _mm256_add_epi16(_mm256_add_epi16(m0, m1), _mm256_add_epi16(m2, m3))
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
    let hmul = hmul_avx2();
    let one = _mm256_set1_epi32(1);
    let mut acc = _mm256_setzero_si256();
    for ty in (0..H).step_by(rows_per_tile(W)) {
        for tx in (0..W).step_by(16) {
            let s = unsafe { load_tile::<W>(src.as_ptr(), W, ty, tx, hmul) };
            let r = unsafe { load_tile::<W>(reference.as_ptr(), pitch, ty, tx, hmul) };
            let diff = std::array::from_fn(|k| _mm256_sub_epi16(s[k], r[k]));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(tile_maxima(diff), one));
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
    let hmul = hmul_avx2();
    let load = |base: [Option<*const u8>; 2], stride: usize, y: usize| {
        let half = |ptr: Option<*const u8>| {
            ptr.map_or(_mm_setzero_si128(), |ptr| unsafe {
                _mm_loadl_epi64(ptr.add(y * stride).cast())
            })
        };
        widen_16_avx2(_mm_unpacklo_epi64(half(base[0]), half(base[1])), hmul)
    };
    let diff = std::array::from_fn(|y| {
        _mm256_sub_epi16(load(src, src_pitch, y), load(reference, pitch, y))
    });
    _mm256_madd_epi16(tile_maxima(diff), _mm256_set1_epi32(1))
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
        _mm_unpacklo_epi64, _mm256_set_m128i, _mm256_setr_epi8, _mm512_broadcast_i64x4,
        _mm512_castsi256_si512, _mm512_inserti64x4, _mm512_maddubs_epi16,
    };
    macro_rules! gather {
        ($a:expr, $b:expr, $c:expr, $d:expr) => {{
            let ba = unsafe {
                _mm_unpacklo_epi64(
                    _mm_loadl_epi64(rows[$b].cast()),
                    _mm_loadl_epi64(rows[$a].cast()),
                )
            };
            let dc = unsafe {
                _mm_unpacklo_epi64(
                    _mm_loadl_epi64(rows[$d].cast()),
                    _mm_loadl_epi64(rows[$c].cast()),
                )
            };
            let lo = _mm256_set_m128i(ba, ba);
            let hi = _mm256_set_m128i(dc, dc);
            _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi)
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

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub(crate) struct SadLuma<'a, const W: usize, const H: usize> {
    pub(crate) src: &'a [u8],
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
impl<const W: usize, const H: usize> LumaCost for SadLuma<'_, W, H> {
    #[inline]
    fn cost(&self, reference: &[u8], pitch: usize) -> u32 {
        self.cost_until(reference, pitch, |_| false)
    }

    #[inline]
    fn cost_until(&self, reference: &[u8], pitch: usize, stop: impl Fn(u32) -> bool) -> u32 {
        assert!(self.src.len() >= W * H && reference.len() >= (H - 1) * pitch + W);
        unsafe { sad_until::<W, H>(self.src.as_ptr(), reference.as_ptr(), pitch, stop) }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn sad_until<const W: usize, const H: usize>(
    src: *const u8,
    reference: *const u8,
    pitch: usize,
    stop: impl Fn(u32) -> bool,
) -> u32 {
    let mut acc = _mm256_setzero_si256();
    if W == 32 {
        for row in 0..H {
            let (a, b) = unsafe {
                (
                    _mm256_loadu_si256(src.add(row * W).cast()),
                    _mm256_loadu_si256(reference.add(row * pitch).cast()),
                )
            };
            acc = _mm256_add_epi64(acc, _mm256_sad_epu8(a, b));
            if row + 1 == H / 2 {
                let total = sad_total(acc);
                if stop(total) {
                    return total;
                }
            }
        }
        return sad_total(acc);
    }
    for row in (0..H).step_by(2) {
        let (a, b) = unsafe {
            let r0 = reference.add(row * pitch);
            let r1 = reference.add((row + 1) * pitch);
            if W == 16 {
                let b = _mm256_inserti128_si256::<1>(
                    _mm256_castsi128_si256(_mm_loadu_si128(r0.cast())),
                    _mm_loadu_si128(r1.cast()),
                );
                (_mm256_loadu_si256(src.add(row * W).cast()), b)
            } else {
                let b = _mm_unpacklo_epi64(_mm_loadl_epi64(r0.cast()), _mm_loadl_epi64(r1.cast()));
                (
                    _mm256_castsi128_si256(_mm_loadu_si128(src.add(row * W).cast())),
                    _mm256_castsi128_si256(b),
                )
            }
        };
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(a, b));
        if row + 2 == H / 2 {
            let total = sad_total(acc);
            if stop(total) {
                return total;
            }
        }
    }
    sad_total(acc)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
#[target_feature(enable = "avx2")]
fn sad_total(acc: __m256i) -> u32 {
    let sum = _mm_add_epi64(
        _mm256_castsi256_si128(acc),
        _mm256_extracti128_si256::<1>(acc),
    );
    _mm_cvtsi128_si32(_mm_add_epi32(sum, _mm_shuffle_epi32::<0b1110>(sum))) as u32
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
    use core::arch::x86_64::{_mm_cvtsi32_si128, _mm_unpacklo_epi32};
    let hmul = hmul_avx2();
    let load = |pair: [&[u8]; 2], stride: usize, y: usize| {
        let a = _mm_cvtsi32_si128(unsafe {
            pair[0].as_ptr().add(y * stride).cast::<i32>().read_unaligned()
        });
        let b = _mm_cvtsi32_si128(unsafe {
            pair[1].as_ptr().add(y * stride).cast::<i32>().read_unaligned()
        });
        widen_16_avx2(_mm_unpacklo_epi32(a, b), hmul)
    };
    let diff = std::array::from_fn(|y| _mm256_sub_epi16(load(src, 4, y), load(reference, pitch, y)));
    horizontal_sum(_mm256_madd_epi16(tile_maxima(diff), _mm256_set1_epi32(1)))
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

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, clippy::many_single_char_names)]
mod neon {
    use core::arch::aarch64::{
        int16x8_t, uint16x4_t, vabd_u8, vabdq_u8, vabsq_s16, vadd_u16, vaddlv_u16, vaddlvq_u8,
        vaddq_s16, vaddvq_u32, vcombine_s16, vdup_n_u16, vdup_n_u32, vdupq_n_u16, vdupq_n_u32,
        vget_low_s16, vld1_lane_u32, vld1_u8, vld1q_lane_u32, vld1q_u8, vmaxq_s16, vpadal_u8,
        vpadalq_u8, vpadalq_u16, vreinterpret_u8_u32, vreinterpret_u16_s16, vreinterpretq_s16_u16,
        vreinterpretq_u8_u32, vsubl_u8, vsubq_s16, vuzp1q_s16, vuzp2q_s16,
    };

    fn span(width: usize, height: usize, pitch: usize) -> usize {
        (height - 1) * pitch + width
    }

    pub(super) fn contiguous_sum(block: &[u8]) -> u32 {
        unsafe {
            let mut acc32 = vdupq_n_u32(0);
            let mut acc16 = vdupq_n_u16(0);
            for (index, chunk) in block.as_chunks::<16>().0.iter().enumerate() {
                acc16 = vpadalq_u8(acc16, vld1q_u8(chunk.as_ptr()));
                if index & 31 == 31 {
                    acc32 = vpadalq_u16(acc32, acc16);
                    acc16 = vdupq_n_u16(0);
                }
            }
            vaddvq_u32(vpadalq_u16(acc32, acc16))
        }
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
        unsafe {
            if width == 4 && height == 4 {
                let mut a = vdupq_n_u32(0);
                let mut b = vdupq_n_u32(0);
                a = vld1q_lane_u32::<0>(s.cast(), a);
                a = vld1q_lane_u32::<1>(s.add(sp).cast(), a);
                a = vld1q_lane_u32::<2>(s.add(2 * sp).cast(), a);
                a = vld1q_lane_u32::<3>(s.add(3 * sp).cast(), a);
                b = vld1q_lane_u32::<0>(r.cast(), b);
                b = vld1q_lane_u32::<1>(r.add(rp).cast(), b);
                b = vld1q_lane_u32::<2>(r.add(2 * rp).cast(), b);
                b = vld1q_lane_u32::<3>(r.add(3 * rp).cast(), b);
                return u32::from(vaddlvq_u8(vabdq_u8(
                    vreinterpretq_u8_u32(a),
                    vreinterpretq_u8_u32(b),
                )));
            }
            if width == 8 {
                let mut acc = vdup_n_u16(0);
                for y in 0..height {
                    let a = vld1_u8(s.add(y * sp));
                    let b = vld1_u8(r.add(y * rp));
                    acc = vpadal_u8(acc, vabd_u8(a, b));
                }
                return vaddlv_u16(acc);
            }
            if width >= 16 {
                let mut acc = vdupq_n_u16(0);
                let mut wide = vdupq_n_u32(0);
                for y in 0..height {
                    let mut x = 0;
                    while x + 16 <= width {
                        let a = vld1q_u8(s.add(y * sp + x));
                        let b = vld1q_u8(r.add(y * rp + x));
                        acc = vpadalq_u8(acc, vabdq_u8(a, b));
                        x += 16;
                    }
                    if y & 31 == 31 {
                        wide = vpadalq_u16(wide, acc);
                        acc = vdupq_n_u16(0);
                    }
                }
                return vaddvq_u32(vpadalq_u16(wide, acc));
            }
            let mut total = 0u32;
            for y in 0..height {
                for x in 0..width {
                    total += u32::from((*s.add(y * sp + x)).abs_diff(*r.add(y * rp + x)));
                }
            }
            total
        }
    }

    #[inline]
    unsafe fn row8(src: *const u8, reference: *const u8) -> int16x8_t {
        unsafe { vreinterpretq_s16_u16(vsubl_u8(vld1_u8(src), vld1_u8(reference))) }
    }

    #[inline]
    unsafe fn horizontal_max(v: int16x8_t) -> uint16x4_t {
        unsafe {
            let even = vuzp1q_s16(v, v);
            let odd = vuzp2q_s16(v, v);
            let sums = vaddq_s16(even, odd);
            let diffs = vsubq_s16(even, odd);
            let au = vabsq_s16(vcombine_s16(vget_low_s16(sums), vget_low_s16(diffs)));
            let e2 = vuzp1q_s16(au, au);
            let o2 = vuzp2q_s16(au, au);
            vreinterpret_u16_s16(vget_low_s16(vmaxq_s16(e2, o2)))
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
        unsafe {
            let mut total = 0u32;
            if width == 4 {
                let zero = vdup_n_u32(0);
                for ty in (0..height).step_by(4) {
                    let d: [int16x8_t; 4] = std::array::from_fn(|k| {
                        let a = vreinterpret_u8_u32(vld1_lane_u32::<0>(
                            s.add((ty + k) * sp).cast(),
                            zero,
                        ));
                        let b = vreinterpret_u8_u32(vld1_lane_u32::<0>(
                            r.add((ty + k) * rp).cast(),
                            zero,
                        ));
                        vreinterpretq_s16_u16(vsubl_u8(a, b))
                    });
                    let (a, b) = (vaddq_s16(d[0], d[1]), vaddq_s16(d[2], d[3]));
                    let (c, e) = (vsubq_s16(d[0], d[1]), vsubq_s16(d[2], d[3]));
                    let sum0 = vadd_u16(
                        horizontal_max(vaddq_s16(a, b)),
                        horizontal_max(vsubq_s16(a, b)),
                    );
                    let sum1 = vadd_u16(
                        horizontal_max(vaddq_s16(c, e)),
                        horizontal_max(vsubq_s16(c, e)),
                    );
                    total += vaddlv_u16(vadd_u16(sum0, sum1));
                }
                return total;
            }
            for ty in (0..height).step_by(4) {
                for tx in (0..width).step_by(8) {
                    let d: [int16x8_t; 4] = std::array::from_fn(|k| {
                        row8(s.add((ty + k) * sp + tx), r.add((ty + k) * rp + tx))
                    });
                    let (a, b) = (vaddq_s16(d[0], d[1]), vaddq_s16(d[2], d[3]));
                    let (c, e) = (vsubq_s16(d[0], d[1]), vsubq_s16(d[2], d[3]));
                    let sum0 = vadd_u16(
                        horizontal_max(vaddq_s16(a, b)),
                        horizontal_max(vsubq_s16(a, b)),
                    );
                    let sum1 = vadd_u16(
                        horizontal_max(vaddq_s16(c, e)),
                        horizontal_max(vsubq_s16(c, e)),
                    );
                    total += vaddlv_u16(vadd_u16(sum0, sum1));
                }
            }
            total
        }
    }
}
