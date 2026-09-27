use super::metric::Shape;

pub(crate) const MAX_BLOCK_AREA: usize = 32 * 32;

#[derive(Clone, Copy)]
pub(crate) struct PlaneView<'a> {
    data: &'a [u8],
    base: usize,
    pub(crate) pitch: usize,
    pub(crate) width: i32,
    pub(crate) height: i32,
    pel: i32,
}

impl<'a> PlaneView<'a> {
    pub(crate) fn pel1(&self, x: i32, y: i32) -> isize {
        self.base as isize + y as isize * self.pitch as isize + x as isize
    }

    pub(crate) fn subpel(&self, x: i32, y: i32) -> isize {
        let (bits, mask) = match self.pel {
            1 => return self.pel1(x, y),
            2 => (1, 1),
            _ => (2, 3),
        };
        let index = ((x & mask) | ((y & mask) << bits)) as isize;
        let subplane = index * self.pitch as isize * self.height as isize;
        self.pel1(x >> bits, y >> bits) + subplane
    }

    pub(crate) fn copy_block(&self, offset: isize, shape: Shape, out: &mut [u8]) {
        for row in 0..shape.height {
            let start = offset + (row * self.pitch) as isize;
            let dst = &mut out[row * shape.width..][..shape.width];
            match self.slice(start, shape.width) {
                Some(src) => dst.copy_from_slice(src),
                None => self.copy_clamped(start, dst),
            }
        }
    }

    pub(crate) fn block<'s>(
        &'s self,
        offset: isize,
        shape: Shape,
        scratch: &'s mut [u8; MAX_BLOCK_AREA],
    ) -> (&'s [u8], usize) {
        if let Some(block) = self.slice(offset, shape.span(self.pitch)) {
            return (block, self.pitch);
        }
        for row in 0..shape.height {
            let start = offset + (row * self.pitch) as isize;
            self.copy_clamped(start, &mut scratch[row * shape.width..][..shape.width]);
        }
        (&scratch[..], shape.width)
    }

    fn slice(&self, offset: isize, len: usize) -> Option<&'a [u8]> {
        let start = usize::try_from(offset).ok()?;
        self.data.get(start..start.checked_add(len)?)
    }

    fn copy_clamped(&self, start: isize, dst: &mut [u8]) {
        for (i, value) in dst.iter_mut().enumerate() {
            let index = usize::try_from(start + i as isize).ok();
            *value = index.and_then(|i| self.data.get(i)).copied().unwrap_or(0);
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct LevelFrame<'a> {
    pub(crate) y: PlaneView<'a>,
    pub(crate) u: PlaneView<'a>,
    pub(crate) v: PlaneView<'a>,
}

#[derive(Clone, Copy)]
pub(crate) struct RawPlane<'a> {
    pub(crate) data: &'a [u8],
    pub(crate) pitch: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct SuperFrameView<'a> {
    pub(crate) planes: [RawPlane<'a>; 3],
    pub(crate) source: Option<[RawPlane<'a>; 3]>,
    pub(crate) width: i32,
    pub(crate) height: i32,
    pub(crate) pel: i32,
    pub(crate) chroma_shift: (u32, u32),
}

impl<'a> SuperFrameView<'a> {
    pub(crate) fn level(&self, level: i32) -> LevelFrame<'a> {
        let pel = if level == 0 { self.pel } else { 1 };
        let width = plane_size(self.width, level);
        let height = plane_size(self.height, level);
        let (sx, sy) = self.chroma_shift;
        let dims = [
            (width, height),
            (width >> sx, height >> sy),
            (width >> sx, height >> sy),
        ];
        let view = |index: usize| {
            let (w, h) = dims[index];
            let (raw, base) = if let (0, Some(source)) = (level, self.source) {
                (source[index], 0)
            } else {
                let raw = self.planes[index];
                let shift = if index == 0 { 0 } else { sy };
                (raw, self.level_offset(level, raw.pitch, shift))
            };
            PlaneView {
                data: raw.data,
                base,
                pitch: raw.pitch,
                width: w,
                height: h,
                pel,
            }
        };
        LevelFrame {
            y: view(0),
            u: view(1),
            v: view(2),
        }
    }

    fn level_offset(&self, level: i32, pitch: usize, y_shift: u32) -> usize {
        if level == 0 {
            return 0;
        }
        let finest_rows = if self.source.is_none() {
            (self.pel * self.pel) as usize * (self.height >> y_shift) as usize
        } else {
            0
        };
        let reduced_rows: usize = (1..level)
            .map(|lv| (plane_size(self.height, lv) >> y_shift) as usize)
            .sum();
        (finest_rows + reduced_rows) * pitch
    }
}

pub(crate) fn plane_size(size: i32, level: i32) -> i32 {
    (0..level).fold(size, |size, _| 2 * (size / 4))
}
