#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn limits(limit: Option<f64>, edge: Option<f64>, tolerance: Option<f64>) -> [f32; 3] {
    [
        (limit.unwrap_or(4.5) / 255.0) as f32,
        (edge.unwrap_or(5.0) / 255.0) as f32,
        (tolerance.unwrap_or(30.0) / 100.0) as f32,
    ]
}

#[must_use]
pub fn mask(
    first: &[u8],
    second: &[u8],
    pitch: usize,
    width: usize,
    height: usize,
    [limit, edge, tolerance]: [f32; 3],
) -> Vec<u8> {
    let detail = |plane: &[u8], x: usize, y: usize| {
        let (right, below) = ((x + 1).min(width - 1), (y + 1).min(height - 1));
        let sum = u16::from(plane[y * pitch + x])
            + u16::from(plane[y * pitch + right])
            + u16::from(plane[below * pitch + x])
            + u16::from(plane[below * pitch + right]);
        (f32::from(plane[y * pitch + x]) - f32::from(sum) / 4.0) / 255.0
    };
    let mut same = vec![false; width * height];
    for (y, row) in same.chunks_exact_mut(width).enumerate() {
        for (x, same) in row.iter_mut().enumerate() {
            let change = f32::from(first[y * pitch + x].abs_diff(second[y * pitch + x])) / 255.0;
            let (a, b) = (detail(first, x, y), detail(second, x, y));
            let strength = a.abs().max(b.abs());
            *same = change <= limit || (strength > edge && (a - b).abs() <= tolerance * strength);
        }
    }
    let full = |x: usize, y: usize| {
        let (left, above) = (x.saturating_sub(1), y.saturating_sub(1));
        let (x, y) = (x.min(width - 1), y.min(height - 1));
        same[above * width + left]
            && same[above * width + x]
            && same[y * width + left]
            && same[y * width + x]
    };
    let mut still = vec![0; width * height];
    for (y, row) in still.chunks_exact_mut(width).enumerate() {
        for (x, still) in row.iter_mut().enumerate() {
            *still = u8::from(full(x, y) || full(x + 1, y) || full(x, y + 1) || full(x + 1, y + 1));
        }
    }
    still
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_arguments
)]
pub fn apply(
    output: &mut [u8],
    first: &[u8],
    second: &[u8],
    pitch: usize,
    width: usize,
    height: usize,
    still: &[u8],
    still_width: usize,
    time: f32,
) {
    let (step_x, step_y) = (still_width / width, still.len() / still_width / height);
    let cells = (step_x * step_y) as u32;
    for y in 0..height {
        for x in 0..width {
            let kept: u32 = (0..step_y)
                .flat_map(|dy| (0..step_x).map(move |dx| (dx, dy)))
                .map(|(dx, dy)| u32::from(still[(y * step_y + dy) * still_width + x * step_x + dx]))
                .sum();
            if kept * 10 > cells * 4 {
                let at = y * pitch + x;
                let (a, b) = (f32::from(first[at]), f32::from(second[at]));
                output[at] = (a + (b - a) * time).round() as u8;
            }
        }
    }
}
