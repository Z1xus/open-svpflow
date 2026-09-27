#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi32, _mm_add_epi64, _mm_cvtsi128_si32, _mm_loadl_epi64,
    _mm_loadu_si128, _mm_sad_epu8, _mm_setzero_si128, _mm_shuffle_epi32, _mm_unpacklo_epi64,
    _mm256_abs_epi16, _mm256_add_epi16, _mm256_add_epi32, _mm256_castsi256_si128,
    _mm256_cvtepu8_epi16, _mm256_extracti128_si256, _mm256_madd_epi16, _mm256_set1_epi16,
    _mm256_setr_epi16, _mm256_setzero_si256, _mm256_shufflehi_epi16, _mm256_shufflelo_epi16,
    _mm256_sign_epi16, _mm256_sub_epi16,
};

pub(crate) type Kernel = fn(&[u8], &[u8], usize) -> u32;

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
        (16, 16, true) => satd::<16, 16>,
        (16, 8, true) => satd::<16, 8>,
        (8, 8, true) => satd::<8, 8>,
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
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    if W >= 16 || (W == 8 && H.is_multiple_of(8)) {
        return unsafe { satd_avx2::<W, H>(src, reference, pitch) };
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
#[target_feature(enable = "avx2")]
unsafe fn satd_avx2<const W: usize, const H: usize>(
    src: &[u8],
    reference: &[u8],
    pitch: usize,
) -> u32 {
    assert!(src.len() >= W * H && reference.len() >= (H - 1) * pitch + W);
    let row = |base: *const u8, stride: usize, y: usize, x: usize| -> __m256i {
        if W == 8 {
            let low = unsafe { _mm_loadl_epi64(base.add(y * stride + x).cast()) };
            let high = unsafe { _mm_loadl_epi64(base.add((y + 4) * stride + x).cast()) };
            _mm256_cvtepu8_epi16(_mm_unpacklo_epi64(low, high))
        } else {
            _mm256_cvtepu8_epi16(unsafe {
                _mm_loadu_si128(base.add(y * stride + x).cast::<__m128i>())
            })
        }
    };
    let rows_per_tile = if W == 8 { 8 } else { 4 };
    let mut acc = _mm256_setzero_si256();
    for ty in (0..H).step_by(rows_per_tile) {
        for tx in (0..W).step_by(16) {
            let diff = |y: usize| {
                _mm256_sub_epi16(
                    row(src.as_ptr(), W, ty + y, tx),
                    row(reference.as_ptr(), pitch, ty + y, tx),
                )
            };
            let sums = hadamard_tile_abs(diff(0), diff(1), diff(2), diff(3));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(sums, _mm256_set1_epi16(1)));
        }
    }
    let folded = _mm_add_epi32(
        _mm256_castsi256_si128(acc),
        _mm256_extracti128_si256::<1>(acc),
    );
    let folded = _mm_add_epi32(folded, _mm_shuffle_epi32::<0b1110>(folded));
    let folded = _mm_add_epi32(folded, _mm_shuffle_epi32::<0b0001>(folded));
    (_mm_cvtsi128_si32(folded) as u32) / 2
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
fn hadamard_tile_abs(r0: __m256i, r1: __m256i, r2: __m256i, r3: __m256i) -> __m256i {
    let (s0, d0) = (_mm256_add_epi16(r0, r1), _mm256_sub_epi16(r0, r1));
    let (s1, d1) = (_mm256_add_epi16(r2, r3), _mm256_sub_epi16(r2, r3));
    let columns = [
        _mm256_add_epi16(s0, s1),
        _mm256_sub_epi16(s0, s1),
        _mm256_add_epi16(d0, d1),
        _mm256_sub_epi16(d0, d1),
    ];
    let odd = _mm256_setr_epi16(1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1);
    let upper = _mm256_setr_epi16(1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1);
    let mut total = _mm256_setzero_si256();
    for v in columns {
        let pairs = _mm256_shufflehi_epi16::<0b1011_0001>(_mm256_shufflelo_epi16::<0b1011_0001>(v));
        let stage1 = _mm256_add_epi16(_mm256_sign_epi16(v, odd), pairs);
        let quads =
            _mm256_shufflehi_epi16::<0b0100_1110>(_mm256_shufflelo_epi16::<0b0100_1110>(stage1));
        let stage2 = _mm256_add_epi16(_mm256_sign_epi16(stage1, upper), quads);
        total = _mm256_add_epi16(total, _mm256_abs_epi16(stage2));
    }
    total
}
