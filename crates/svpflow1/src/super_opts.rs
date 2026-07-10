use crate::params::Value;

#[derive(Clone, Copy, Debug)]
pub(crate) struct SuperOpts {
    pub(crate) pel: i32,
    pub(crate) gpu: i32,
    pub(crate) scale_down: i32,
    pub(crate) scale_up: i32,
    pub(crate) full: bool,
    pub(crate) rc: bool,
    pub(crate) levels: i32,
    pub(crate) width: i32,
    pub(crate) height: i32,
}

impl SuperOpts {
    pub(crate) fn from_opt(opt: Option<&Value>, width: i32, height: i32) -> Result<Self, String> {
        let pel = opt.and_then(|v| v.int_at(&["pel"])).unwrap_or(2) as i32;
        if !matches!(pel, 1 | 2 | 4) {
            return Err("SVSuper: pel has to be 1 or 2 or 4".into());
        }
        let gpu = opt.and_then(|v| v.int_at(&["gpu"])).unwrap_or(1) as i32;
        if !(0..=2).contains(&gpu) {
            return Err("SVSuper: gpu has to be 0 or 1 or 2".into());
        }
        let scale_down = opt.and_then(|v| v.int_at(&["scale", "down"])).unwrap_or(4) as i32;
        let mut scale_up = opt.and_then(|v| v.int_at(&["scale", "up"])).unwrap_or(2) as i32;

        if gpu != 0 {
            scale_up = 0;
        }
        let full = opt.and_then(|v| v.bool_at(&["full"])).unwrap_or(true);
        if pel >= 2 && !full {
            return Err("SVSuper: can't ignore finest level with high pel value".into());
        }
        let rc = opt.and_then(|v| v.bool_at(&["rc"])).unwrap_or(false);
        let levels = auto_levels(width, height);
        Ok(Self {
            pel,
            gpu,
            scale_down,
            scale_up,
            full,
            rc,
            levels,
            width,
            height,
        })
    }

    pub(crate) fn pack_data(&self) -> i64 {
        let mut flags = u8::from(self.rc);
        if !self.full {
            flags |= 2;
        }
        let raw = u64::from(self.height as u16)
            | (u64::from(self.width as u16) << 16)
            | (u64::from(self.pel as u8) << 32)
            | (u64::from(self.levels as u8) << 40)
            | (u64::from(self.gpu as u8) << 48)
            | (u64::from(flags) << 56);
        i64::from_ne_bytes(raw.to_ne_bytes())
    }

    pub(crate) fn super_height(&self) -> i32 {
        super_plane_height(self.height, self.pel, self.levels, self.full)
    }
}

pub(crate) fn reduce_dim(mut dim: i32, n: i32) -> i32 {
    for _ in 0..n {
        dim = 2 * (dim / 4);
    }
    dim
}

pub(crate) fn auto_levels(width: i32, height: i32) -> i32 {
    let mut i = 0;
    loop {
        let h = reduce_dim(height, i);
        let w = reduce_dim(width, i);
        if h < 4 || w < 4 {
            return i;
        }
        i += 1;
    }
}

pub(crate) fn super_plane_height(base_h: i32, pel: i32, levels: i32, full: bool) -> i32 {
    let mut total = if full { pel * pel * base_h } else { 0 };
    for lv in 1..levels {
        total += reduce_dim(base_h, lv);
    }
    total
}
