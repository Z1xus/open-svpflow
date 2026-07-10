use crate::renderer::{FramePlanes, FramePlanesMut};

const SIG_PEAK: f32 = 0.5;
const SIG_MUL: f32 = 0.5;
const PARAM: f32 = 1.0;

pub(crate) fn transform_420(
    width: i32,
    height: i32,
    src: FramePlanes<'_>,
    mut dst: FramePlanesMut<'_>,
) {
    let width = usize::try_from(width.max(0)).unwrap_or(0);
    let height = usize::try_from(height.max(0)).unwrap_or(0);
    for y in 0..height / 2 {
        for x in 0..width / 2 {
            transform_block(&src, &mut dst, x, y);
        }
    }
}

#[allow(clippy::many_single_char_names)]
fn transform_block(src: &FramePlanes<'_>, dst: &mut FramePlanesMut<'_>, x: usize, y: usize) {
    let x0 = x.saturating_mul(2);
    let x1 = x0.saturating_add(1);
    let y0 = y.saturating_mul(2);
    let y1 = y0.saturating_add(1);
    let u = f32::from(sample(src.u.data, src.u.stride, x, y)) / 255.0 - 0.5;
    let v = f32::from(sample(src.v.data, src.v.stride, x, y)) / 255.0 - 0.5;
    let samples = [
        sample(src.y.data, src.y.stride, x1, y0),
        sample(src.y.data, src.y.stride, x0, y0),
        sample(src.y.data, src.y.stride, x1, y1),
        sample(src.y.data, src.y.stride, x0, y1),
    ];
    let mut pixels = [(0.0, 0.0, 0.0); 4];
    for (pixel, y) in pixels.iter_mut().zip(samples) {
        *pixel = rgb_signal(f32::from(y) / 255.0, u, v);
    }
    let out = [
        finish_pixel(pixels[0]),
        finish_pixel(pixels[1]),
        finish_pixel(pixels[2]),
        finish_pixel(pixels[3]),
    ];
    write(dst.y.data, dst.y.stride, x1, y0, out[0].0);
    write(dst.y.data, dst.y.stride, x0, y0, out[1].0);
    write(dst.y.data, dst.y.stride, x1, y1, out[2].0);
    write(dst.y.data, dst.y.stride, x0, y1, out[3].0);
    write(dst.u.data, dst.u.stride, x, y, out[0].1);
    write(dst.v.data, dst.v.stride, x, y, out[0].2);
}

fn rgb_signal(y: f32, u: f32, v: f32) -> (f32, f32, f32) {
    let rgb = (
        (y + 1.5748 * v).clamp(0.0, 1.0),
        (y - 0.187_324 * u - 0.468_124 * v).clamp(0.0, 1.0),
        (y + 1.8556 * u).clamp(0.0, 1.0),
    );
    (pq(rgb.0), pq(rgb.1), pq(rgb.2))
}

#[allow(clippy::excessive_precision)]
fn pq(value: f32) -> f32 {
    let c = value.powf(0.012_683_3);
    (f32::max(c - 0.835_938, 0.0) / (18.851_562 - 18.6875 * c)).powf(6.277_385_2) * 100.0
}

#[allow(clippy::many_single_char_names)]
fn finish_pixel(rgb: (f32, f32, f32)) -> (u8, u8, u8) {
    let sig_orig = rgb.0.max(rgb.1).max(rgb.2);
    let sig = (
        tone(rgb.0 * SIG_MUL),
        tone(rgb.1 * SIG_MUL),
        tone(rgb.2 * SIG_MUL),
    );
    let sig_new = sig.0.max(sig.1).max(sig.2);
    let k = if sig_orig == 0.0 {
        0.0
    } else {
        sig_new / sig_orig
    };
    let lin = (rgb.0 * k, rgb.1 * k, rgb.2 * k);
    let coeff = 0.75 * (f32::max(sig_new - 0.18, 1.0e-6) / f32::max(sig_new, 1.0)).powf(1.5);
    let r = mix(lin.0, sig.0, coeff).powf(0.45);
    let g = mix(lin.1, sig.1, coeff).powf(0.45);
    let b = mix(lin.2, sig.2, coeff).powf(0.45);
    let y = (0.299 * r + 0.587 * g + 0.114 * b).clamp(0.0, 1.0);
    let u = (0.436 * (b - y) / (1.0 - 0.114) + 0.5).clamp(0.0, 1.0);
    let v = (0.615 * (r - y) / (1.0 - 0.299) + 0.5).clamp(0.0, 1.0);
    (byte(y), byte(u), byte(v))
}

fn tone(value: f32) -> f32 {
    f32::min(
        value / (value + PARAM) * ((SIG_PEAK + PARAM) / SIG_PEAK),
        1.0,
    )
}

fn mix(a: f32, b: f32, coeff: f32) -> f32 {
    a * (1.0 - coeff) + b * coeff
}

#[allow(clippy::cast_possible_truncation)]
fn byte(value: f32) -> u8 {
    u8::try_from((value * 255.0).round().clamp(0.0, 255.0) as i32).unwrap_or(0)
}

fn sample(data: &[u8], stride: usize, x: usize, y: usize) -> u8 {
    data.get(y.saturating_mul(stride).saturating_add(x))
        .copied()
        .unwrap_or(0)
}

fn write(data: &mut [u8], stride: usize, x: usize, y: usize, value: u8) {
    if let Some(out) = data.get_mut(y.saturating_mul(stride).saturating_add(x)) {
        *out = value;
    }
}
