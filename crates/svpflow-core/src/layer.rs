pub const CHUNK: i32 = 12;

#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn limit(limit: Option<f64>) -> f32 {
    limit.unwrap_or(6.0) as f32
}

#[must_use]
pub fn pairs(chunk: i32, last: i32) -> std::ops::RangeInclusive<i32> {
    chunk * CHUNK..=(chunk * CHUNK + CHUNK - 1).min(last)
}

const IDENTITY: [f32; 8] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];

pub struct Grid {
    pub width: usize,
    pub height: usize,
    pub cell: (f64, f64),
    pub origin: (f64, f64),
    pub pel: f64,
    pub frame: (f64, f64),
}

fn solve(mut m: [[f64; 9]; 8]) -> Option<[f64; 8]> {
    for col in 0..8 {
        let pivot = (col..8).max_by(|&a, &b| m[a][col].abs().total_cmp(&m[b][col].abs()))?;
        if m[pivot][col].abs() < 1e-12 {
            return None;
        }
        m.swap(col, pivot);
        for row in 0..8 {
            if row != col {
                let pivot = m[col];
                let factor = m[row][col] / pivot[col];
                for (value, pivot) in m[row].iter_mut().zip(pivot).skip(col) {
                    *value -= factor * pivot;
                }
            }
        }
    }
    Some(std::array::from_fn(|i| m[i][8] / m[i][i]))
}

#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub fn camera(xs: &[u16], ys: &[u16], grid: &Grid) -> [f32; 8] {
    let scale = grid.frame.0 / 2.0;
    let step = (grid.width * grid.height / 2000).max(1);
    let points: Vec<[f64; 4]> = (0..grid.width * grid.height)
        .step_by(step)
        .map(|i| {
            let x = grid.origin.0 + ((i % grid.width) as f64 + 0.5) * grid.cell.0;
            let y = grid.origin.1 + ((i / grid.width) as f64 + 0.5) * grid.cell.1;
            let (x, y) = (
                (x - grid.frame.0 / 2.0) / scale,
                (y - grid.frame.1 / 2.0) / scale,
            );
            let dx = (f64::from(xs[i]) - 1024.0) / grid.pel / scale;
            let dy = (f64::from(ys[i]) - 1024.0) / grid.pel / scale;
            [x, y, x + dx, y + dy]
        })
        .collect();
    let mut h = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    let mut limit = f64::INFINITY;
    for _ in 0..4 {
        let mut m = [[0.0; 9]; 8];
        let mut used = 0;
        for &[x, y, tx, ty] in &points {
            let d = h[6] * x + h[7] * y + 1.0;
            let ex = (h[0] * x + h[1] * y + h[2]) / d - tx;
            let ey = (h[3] * x + h[4] * y + h[5]) / d - ty;
            if (ex * ex + ey * ey).sqrt() * scale > limit {
                continue;
            }
            used += 1;
            for (row, t) in [
                ([x, y, 1.0, 0.0, 0.0, 0.0, -tx * x, -tx * y], tx),
                ([0.0, 0.0, 0.0, x, y, 1.0, -ty * x, -ty * y], ty),
            ] {
                for i in 0..8 {
                    for j in 0..8 {
                        m[i][j] += row[i] * row[j];
                    }
                    m[i][8] += row[i] * t;
                }
            }
        }
        if used < points.len() / 4 {
            return IDENTITY;
        }
        let Some(next) = solve(m) else {
            return IDENTITY;
        };
        h = next;
        limit = 2.0;
    }
    h.map(|v| v as f32)
}
