#[derive(Clone, Copy)]
pub struct Timing {
    pub fps_num: i64,
    pub fps_den: i64,
    pub frame_num: i64,
    pub frame_den: i64,
}

impl Timing {
    pub fn new(
        source_fps_num: i64,
        source_fps_den: i64,
        absolute: bool,
        rate_num: i64,
        rate_den: i64,
    ) -> Self {
        let rate_num = non_zero(rate_num);
        let rate_den = non_zero(rate_den);
        let (fps_num, fps_den) = if absolute {
            normalize(rate_num, rate_den)
        } else {
            normalize(
                source_fps_num.saturating_mul(rate_num),
                source_fps_den.saturating_mul(rate_den),
            )
        };
        let (frame_num, frame_den) = normalize(
            source_fps_den.saturating_mul(fps_num),
            source_fps_num.saturating_mul(fps_den),
        );
        Self {
            fps_num,
            fps_den,
            frame_num,
            frame_den,
        }
    }

    pub fn source_frame(self, frame: i32, round_nearest: bool) -> i32 {
        if self.frame_num == 0 {
            return frame;
        }
        let scaled = i64::from(frame).saturating_mul(self.frame_den);
        if round_nearest {
            clamp_i32(scaled.saturating_add(self.frame_num / 2) / self.frame_num)
        } else {
            clamp_i32(scaled / self.frame_num)
        }
    }

    pub fn phase_256(self, frame: i32, source_frame: i32) -> i32 {
        if self.frame_num <= 0 {
            return 0;
        }
        let scaled = i128::from(frame).saturating_mul(i128::from(self.frame_den));
        let whole = i128::from(source_frame).saturating_mul(i128::from(self.frame_num));
        let remainder = scaled.saturating_sub(whole);
        if remainder <= 0 {
            return 0;
        }
        clamp_i32_128(
            remainder
                .saturating_mul(512)
                .saturating_add(i128::from(self.frame_num))
                / i128::from(self.frame_num).saturating_mul(2),
        )
    }

    pub fn raw_phase_256(self, frame: i32, source_frame: i32) -> f64 {
        (f64::from(frame) * as_f64(self.frame_den) / as_f64(self.frame_num)
            - f64::from(source_frame))
            * 256.0
    }

    pub fn output_phase_256(self, frame: i32) -> i32 {
        if self.frame_num <= 0 {
            return 0;
        }
        let scaled = i128::from(frame).saturating_mul(i128::from(self.frame_den));
        let remainder = scaled.rem_euclid(i128::from(self.frame_num));
        clamp_i32_128(
            remainder
                .saturating_mul(512)
                .saturating_add(i128::from(self.frame_num))
                / i128::from(self.frame_num).saturating_mul(2),
        )
    }

    pub fn scene_phase_256(self, raw: f64, mode: i64) -> i32 {
        let step = as_f64(self.frame_den) * 256.0 / as_f64(self.frame_num);
        if !raw.is_finite() || !step.is_finite() || step <= 0.0 {
            return 0;
        }
        let left_floor = (raw / step).floor();
        let left_rem = raw - left_floor * step;
        let left_floor = trunc_i32(left_floor);
        let left = if mode > 1 {
            if trunc_i32(left_rem) > 0 {
                left_floor
            } else {
                left_floor.saturating_sub(1).max(0)
            }
        } else {
            left_floor + i32::from(trunc_i32(left_rem) >= trunc_i32((left_rem - step).abs()))
        };
        let right_floor = ((256.0 - raw - 0.001) / step).floor();
        let right_rem = 256.0 - (right_floor * step + raw);
        let right = trunc_i32(right_floor)
            + i32::from(if mode > 1 {
                (right_rem - step).abs() < 0.1
            } else {
                trunc_i32(right_rem) > trunc_i32((right_rem - step).abs())
            });
        let total = left.saturating_add(right);
        if total == 0 {
            return 0;
        }
        let phase = left.saturating_mul(256) / total;
        if (mode & !2) == 0 {
            phase
        } else if left > right {
            256 - (right.saturating_mul(256) / total) / 2
        } else {
            phase / 2
        }
    }

    pub fn scale_frame_count(self, frames: i32) -> i32 {
        if frames <= 0 || self.frame_den == 0 {
            return frames;
        }
        let scaled =
            i64::from(frames.saturating_sub(1)).saturating_mul(self.frame_num) / self.frame_den + 1;
        clamp_i32(scaled)
    }
}

fn non_zero(value: i64) -> i64 {
    if value == 0 { 1 } else { value }
}

fn normalize(num: i64, den: i64) -> (i64, i64) {
    if num == 0 || den == 0 {
        return (0, 1);
    }
    let gcd = gcd(num.unsigned_abs(), den.unsigned_abs());
    (
        num / i64::try_from(gcd).unwrap_or(1),
        den / i64::try_from(gcd).unwrap_or(1),
    )
}

fn gcd(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let next = left % right;
        left = right;
        right = next;
    }
    left
}

#[allow(clippy::cast_precision_loss)]
fn as_f64(value: i64) -> f64 {
    value as f64
}

#[allow(clippy::cast_possible_truncation)]
fn trunc_i32(value: f64) -> i32 {
    if value.is_nan() {
        0
    } else if value >= f64::from(i32::MAX) {
        i32::MAX
    } else if value <= f64::from(i32::MIN) {
        i32::MIN
    } else {
        value.trunc() as i32
    }
}

fn clamp_i32(value: i64) -> i32 {
    i32::try_from(value).unwrap_or(if value.is_negative() {
        i32::MIN
    } else {
        i32::MAX
    })
}

fn clamp_i32_128(value: i128) -> i32 {
    i32::try_from(value).unwrap_or(if value.is_negative() {
        i32::MIN
    } else {
        i32::MAX
    })
}
