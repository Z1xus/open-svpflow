#![allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::unused_self
)]

use crate::{options, renderer, video_format, vs};

const TABLE_LEN: usize = 128;

const STRIP_TABLES: usize = 6;

pub(crate) struct LightState;

struct Rng {
    state: u32,
}

impl Rng {
    fn seeded_srand(seed: u32) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> i32 {
        self.state = self.state.wrapping_mul(214_013).wrapping_add(2_531_011);
        ((self.state >> 16) & 0x7fff) as i32
    }
}

struct PlaneCfg {
    base: i32,
    border: i32,
    table_h: usize,
    table_v: usize,
    table_corner: usize,
}

#[derive(Clone, Copy)]
enum Orient {
    Top,
    Bottom,
    Left,
    Right,
}

impl LightState {
    pub(crate) fn new() -> Self {
        Self
    }

    pub(crate) fn apply(
        &self,
        info: &vs::VideoInfo,
        padding: (i32, i32),
        params: options::LightParams,
        frame: i32,
        dst: &mut renderer::FramePlanesMut<'_>,
    ) {
        if padding == (0, 0) || !video_format::is_yuv420p8(info) {
            return;
        }
        let (pad_x, pad_y) = padding;

        let frame = frame.max(0);
        let seed = u32::try_from((frame + 1) / 2).unwrap_or(0);
        let mut rng = Rng::seeded_srand(seed);
        if frame % 2 == 0 {
            for _ in 0..15 {
                rng.next();
            }
        }
        let mut noise = [[0i32; TABLE_LEN]; STRIP_TABLES];
        for table in &mut noise {
            for value in table.iter_mut() {
                *value = rng.next() % 3 - 1;
            }
        }
        let luma = PlaneCfg {
            base: 0,
            border: params.border,
            table_h: 0,
            table_v: 2,
            table_corner: 4,
        };
        let chroma = PlaneCfg {
            base: 127,
            border: params.border / 2,
            table_h: 1,
            table_v: 3,
            table_corner: 5,
        };
        self.plane(
            &mut dst.y,
            info.width,
            info.height,
            pad_x,
            pad_y,
            &luma,
            params,
            &noise,
            &mut rng,
        );
        self.plane(
            &mut dst.u,
            info.width / 2,
            info.height / 2,
            pad_x / 2,
            pad_y / 2,
            &chroma,
            params,
            &noise,
            &mut rng,
        );
        self.plane(
            &mut dst.v,
            info.width / 2,
            info.height / 2,
            pad_x / 2,
            pad_y / 2,
            &chroma,
            params,
            &noise,
            &mut rng,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn plane(
        &self,
        plane: &mut renderer::PlaneMut<'_>,
        width: i32,
        height: i32,
        pad_x: i32,
        pad_y: i32,
        cfg: &PlaneCfg,
        params: options::LightParams,
        noise: &[[i32; TABLE_LEN]; STRIP_TABLES],
        rng: &mut Rng,
    ) {
        if width <= 0 || height <= 0 {
            return;
        }
        let geo = Geo {
            width,
            height,
            pad_x,
            pad_y,
        };

        let lights_h = params.lights;
        let lights_v = if width > 0 {
            (params.lights * height / width).max(1)
        } else {
            params.lights
        };
        let mut profiles: Vec<(Orient, Vec<f32>)> = Vec::new();
        if pad_y > 0 {
            profiles.push((
                Orient::Top,
                build_profile(
                    |r, col| read(plane, pad_x + col, pad_y + r),
                    width,
                    cfg.border,
                    lights_h,
                    params,
                    cfg.base,
                ),
            ));
            profiles.push((
                Orient::Bottom,
                build_profile(
                    |r, col| read(plane, pad_x + col, pad_y + height - 1 - r),
                    width,
                    cfg.border,
                    lights_h,
                    params,
                    cfg.base,
                ),
            ));
        }
        if pad_x > 0 {
            profiles.push((
                Orient::Left,
                build_profile(
                    |r, row| read(plane, pad_x + r, pad_y + row),
                    height,
                    cfg.border,
                    lights_v,
                    params,
                    cfg.base,
                ),
            ));
            profiles.push((
                Orient::Right,
                build_profile(
                    |r, row| read(plane, pad_x + width - 1 - r, pad_y + row),
                    height,
                    cfg.border,
                    lights_v,
                    params,
                    cfg.base,
                ),
            ));
        }
        let ends = Endpoints::from_profiles(&profiles, cfg.base, &geo, params);
        for (orient, profile) in &profiles {
            let table = match orient {
                Orient::Top | Orient::Bottom => cfg.table_h,
                Orient::Left | Orient::Right => cfg.table_v,
            };
            self.fill_strip(
                plane,
                *orient,
                profile,
                &noise[table],
                cfg.base,
                &geo,
                params,
                rng,
            );
        }
        self.fill_corners(
            plane,
            &geo,
            cfg,
            &ends,
            params,
            &noise[cfg.table_corner],
            rng,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn fill_strip(
        &self,
        plane: &mut renderer::PlaneMut<'_>,
        orient: Orient,
        profile: &[f32],
        noise: &[i32; TABLE_LEN],
        base: i32,
        geo: &Geo,
        params: options::LightParams,
        rng: &mut Rng,
    ) {
        let dim = i32::try_from(profile.len()).unwrap_or(0);
        let pad = match orient {
            Orient::Top | Orient::Bottom => geo.pad_y,
            Orient::Left | Orient::Right => geo.pad_x,
        };
        let faded = i64::from(pad) * i64::from(params.length) >= 200;
        for d in 0..pad {
            let factor = if faded && pad > 1 {
                match orient {
                    Orient::Top | Orient::Left => f64::from(d) / f64::from(pad - 1),
                    Orient::Bottom | Orient::Right => f64::from(pad - d) / f64::from(pad - 1),
                }
            } else {
                1.0
            };
            let base_rand = rng.next();
            let noise_base = base_rand - (base_rand & 0x80);
            let factor = factor as f32;
            let base_f = base as f32;
            for i in 0..dim {
                let value = trunc_f32((profile[i as usize] - base_f) * factor + base_f);
                let value = if (1..=254).contains(&(value & 0xFF)) {
                    value + noise[noise_index(noise_base, i)]
                } else {
                    value
                };
                let (x, y) = orient.coord(geo, d, i);
                write(plane, x, y, low_u8(value));
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn fill_corners(
        &self,
        plane: &mut renderer::PlaneMut<'_>,
        geo: &Geo,
        cfg: &PlaneCfg,
        ends: &Endpoints,
        params: options::LightParams,
        noise: &[i32; TABLE_LEN],
        rng: &mut Rng,
    ) {
        if geo.pad_x <= 0 || geo.pad_y <= 0 {
            return;
        }
        let grid = corner_geometry(geo.pad_x, geo.pad_y, params.length);

        let corners = [
            (0, 0, ends.left_top, ends.top_left, true, true),
            (
                geo.pad_x + geo.width,
                0,
                ends.right_top,
                ends.top_right,
                false,
                true,
            ),
            (
                0,
                geo.pad_y + geo.height,
                ends.left_bottom,
                ends.bottom_left,
                true,
                false,
            ),
            (
                geo.pad_x + geo.width,
                geo.pad_y + geo.height,
                ends.right_bottom,
                ends.bottom_right,
                false,
                false,
            ),
        ];
        for (ox, oy, a5, a6, mirror_x, mirror_y) in corners {
            for row in 0..geo.pad_y {
                let base_rand = rng.next();
                let gi = if mirror_y { geo.pad_y - 1 - row } else { row };
                for col in 0..geo.pad_x {
                    let gj = if mirror_x { geo.pad_x - 1 - col } else { col };
                    let cell = grid[(gi * geo.pad_x + gj) as usize];
                    let blend = f64::from(a6) * cell.mix + f64::from(a5) * (1.0 - cell.mix);
                    let value = cfg.base + trunc((blend - f64::from(cfg.base)) * cell.weight);
                    let value = if (1..=254).contains(&(value & 0xFFFF)) {
                        value + noise[noise_index(base_rand + col, 0)]
                    } else {
                        value
                    };
                    write(plane, ox + col, oy + row, low_u8(value));
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Cell {
    mix: f64,
    weight: f64,
}

fn corner_geometry(width: i32, height: i32, length: i32) -> Vec<Cell> {
    let mut grid = Vec::with_capacity((width.max(0) * height.max(0)) as usize);
    for i in 0..height {
        for j in 0..width {
            let x = f64::from(j + 1) / f64::from(width);
            let y = f64::from(i + 1) / f64::from(height);
            let distance = (x * x + y * y).sqrt() * 100.0;
            let mix = if x > y {
                y / x * 0.5
            } else if y >= 0.0001 {
                1.0 - x / y * 0.5
            } else {
                1.0
            };
            let weight = if length > 0 {
                (1.0 - distance / f64::from(length)).max(0.0)
            } else {
                0.0
            };
            grid.push(Cell { mix, weight });
        }
    }
    grid
}

struct Endpoints {
    top_left: i32,
    top_right: i32,
    bottom_left: i32,
    bottom_right: i32,
    left_top: i32,
    left_bottom: i32,
    right_top: i32,
    right_bottom: i32,
}

impl Endpoints {
    fn from_profiles(
        profiles: &[(Orient, Vec<f32>)],
        base: i32,
        geo: &Geo,
        params: options::LightParams,
    ) -> Self {
        let saved = |orient: fn(&Orient) -> bool, pad: i32, far: bool, first: bool| {
            let profile = profiles.iter().find(|(o, _)| orient(o)).map(|(_, p)| p);
            let Some(profile) = profile else {
                return base;
            };
            let edge = if first {
                profile.first()
            } else {
                profile.last()
            };
            let Some(&edge) = edge else {
                return base;
            };
            let faded = i64::from(pad) * i64::from(params.length) >= 200;
            let factor = if far && faded && pad > 1 {
                pad as f32 / (pad - 1) as f32
            } else {
                1.0
            };
            let base_f = base as f32;
            trunc_f32((edge - base_f) * factor + base_f)
        };
        let top = |o: &Orient| matches!(o, Orient::Top);
        let bottom = |o: &Orient| matches!(o, Orient::Bottom);
        let left = |o: &Orient| matches!(o, Orient::Left);
        let right = |o: &Orient| matches!(o, Orient::Right);
        Self {
            top_left: saved(top, geo.pad_y, false, true),
            top_right: saved(top, geo.pad_y, false, false),
            bottom_left: saved(bottom, geo.pad_y, true, true),
            bottom_right: saved(bottom, geo.pad_y, true, false),
            left_top: saved(left, geo.pad_x, false, true),
            left_bottom: saved(left, geo.pad_x, false, false),
            right_top: saved(right, geo.pad_x, true, true),
            right_bottom: saved(right, geo.pad_x, true, false),
        }
    }
}

struct Geo {
    width: i32,
    height: i32,
    pad_x: i32,
    pad_y: i32,
}

impl Orient {
    fn coord(self, geo: &Geo, d: i32, i: i32) -> (i32, i32) {
        match self {
            Self::Top => (geo.pad_x + i, d),
            Self::Bottom => (geo.pad_x + i, geo.pad_y + geo.height + d),
            Self::Left => (d, geo.pad_y + i),
            Self::Right => (geo.pad_x + geo.width + d, geo.pad_y + i),
        }
    }
}

fn build_profile(
    line: impl Fn(i32, i32) -> Option<u8>,
    dim: i32,
    border: i32,
    lights: i32,
    params: options::LightParams,
    base: i32,
) -> Vec<f32> {
    let dim_usize = usize::try_from(dim.max(0)).unwrap_or(0);
    if dim_usize == 0 {
        return Vec::new();
    }
    let border = border.max(1);
    let mut accum = vec![0i64; dim_usize];
    for col in 0..dim {
        let mut sum = 0i64;
        for r in 0..border {
            sum += i64::from(line(r, col).unwrap_or(0));
        }
        accum[col as usize] = sum;
    }
    let lights = lights.max(1);
    let half = trunc(f64::from(dim) * params.cell / f64::from(2 * lights));
    let window = 2 * half + 3;
    let control = control_points(&accum, dim, lights, border, half, window, base);
    cubic_upsample(&control, dim, lights)
}

#[allow(clippy::too_many_arguments)]
fn control_points(
    accum: &[i64],
    dim: i32,
    lights: i32,
    border: i32,
    half: i32,
    window: i32,
    base: i32,
) -> Vec<i32> {
    let mut control = vec![base; usize::try_from(lights.max(0)).unwrap_or(0)];
    for k in 0..lights {
        let center = (f64::from(k) + 0.5) * f64::from(dim) / f64::from(lights);
        let mut lo = trunc(center - f64::from(half + 1));
        if lo < 0 {
            lo = 0;
        }
        let mut hi = window + lo;
        if hi >= dim {
            hi = dim;
        }
        let count = hi - lo;
        if count > 0 && border > 0 {
            let mut sum = 0i64;
            for i in lo..hi {
                sum += accum[i as usize];
            }
            let divisor = i64::from(border) * i64::from(count);
            control[k as usize] = i32::try_from(sum / divisor).unwrap_or(base);
        }
    }
    control
}

#[allow(clippy::cast_precision_loss)]
fn cubic_upsample(control: &[i32], dim: i32, lights: i32) -> Vec<f32> {
    let mut out = vec![0.0f32; usize::try_from(dim.max(0)).unwrap_or(0)];
    if control.is_empty() {
        return out;
    }
    let width_f = (dim - 1) as f32;
    let lights_f = lights as f32;
    let last = lights - 1;
    let mut idx = -1i32;
    let mut p1 = control[0] as f32;
    let mut pr = control[0] as f32;
    let mut m0 = 0.0f32;
    let mut m1 = 0.0f32;
    for o in 0..dim {
        let nx = o as f32 / width_f;
        let ny = (idx as f32 + 0.5) / lights_f;
        if nx > ny {
            let adv = idx + 1;
            let right = adv.clamp(0, last);
            let left = idx.max(0);
            let lidx = idx.max(1) - 1;
            let rr = (idx + 2).clamp(0, last);
            p1 = control[left as usize] as f32;
            pr = control[right as usize] as f32;
            let pl = control[lidx as usize] as f32;
            let prr = control[rr as usize] as f32;
            let slope = (pr - p1) * 0.5;
            m0 = (p1 - pl) * 0.5 + slope;
            m1 = (prr - pr) * 0.5 + slope;
            idx = adv;
        }
        let t = nx * lights_f - idx as f32 + 0.5;

        let tt = t * t;
        let omt = 1.0 - t;
        let omt2 = omt * omt;
        let t2 = t + t;
        let term1 = ((t - 1.0) * tt) * m1;
        let term2 = ((3.0 - t2) * tt) * pr;
        let term3 = (omt2 * t) * m0;
        let term4 = ((t2 + 1.0) * omt2) * p1;
        let value = term1 + (term2 + (term3 + term4));
        out[o as usize] = value.clamp(0.0, 255.0);
    }
    out
}

fn read(plane: &renderer::PlaneMut<'_>, x: i32, y: i32) -> Option<u8> {
    let x = usize::try_from(x).ok()?;
    let y = usize::try_from(y).ok()?;
    plane
        .data
        .get(y.checked_mul(plane.stride)?.checked_add(x)?)
        .copied()
}

fn write(plane: &mut renderer::PlaneMut<'_>, x: i32, y: i32, value: u8) {
    let Ok(x) = usize::try_from(x) else {
        return;
    };
    let Ok(y) = usize::try_from(y) else {
        return;
    };
    if let Some(slot) = plane
        .data
        .get_mut(y.saturating_mul(plane.stride).saturating_add(x))
    {
        *slot = value;
    }
}

fn noise_index(base: i32, offset: i32) -> usize {
    usize::try_from(base.wrapping_add(offset) & 0x7f).unwrap_or(0)
}

#[allow(clippy::cast_possible_truncation)]
fn trunc(value: f64) -> i32 {
    value as i32
}

#[allow(clippy::cast_possible_truncation)]
fn trunc_f32(value: f32) -> i32 {
    value as i32
}

#[allow(clippy::cast_possible_truncation)]
fn low_u8(value: i32) -> u8 {
    value as u8
}
