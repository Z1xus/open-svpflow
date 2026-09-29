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

    pub(crate) fn row_offset<const BITS: u32>(&self, y: i32) -> isize {
        let mask = (1 << BITS) - 1;
        let subplane = ((y & mask) << BITS) as isize * self.pitch as isize * self.height as isize;
        self.base as isize + (y >> BITS) as isize * self.pitch as isize + subplane
    }

    pub(crate) fn column_offset<const BITS: u32>(&self, x: i32) -> isize {
        let mask = (1 << BITS) - 1;
        let subplane = (x & mask) as isize * self.pitch as isize * self.height as isize;
        (x >> BITS) as isize + subplane
    }

    pub(crate) fn data(&self) -> &'a [u8] {
        self.data
    }

    pub(crate) fn subpel_extent(&self, x: (i32, i32), y: (i32, i32), span: usize) -> bool {
        let bits = match self.pel {
            1 => 0,
            2 => 1,
            _ => 2,
        };
        let pel = 1isize << bits;
        let low = self.pel1(x.0 >> bits, y.0 >> bits);
        let high = self.pel1(x.1 >> bits, y.1 >> bits)
            + (pel * pel - 1) * self.pitch as isize * self.height as isize;
        low >= 0 && high >= low && (high as usize).saturating_add(span) <= self.data.len()
    }

    pub(crate) fn copy_block(&self, offset: isize, shape: Shape, out: &mut [u8]) {
        let Some(block) = self.slice(offset, shape.span(self.pitch)) else {
            self.copy_block_clamped(offset, shape, out);
            return;
        };
        let p = self.pitch;
        match (shape.width, shape.height) {
            (4, 4) => copy_fixed::<4, 4>(block, p, out),
            (8, 8) => copy_fixed::<8, 8>(block, p, out),
            (8, 4) => copy_fixed::<8, 4>(block, p, out),
            (4, 2) => copy_fixed::<4, 2>(block, p, out),
            (2, 2) => copy_fixed::<2, 2>(block, p, out),
            (16, 16) => copy_fixed::<16, 16>(block, p, out),
            (16, 8) => copy_fixed::<16, 8>(block, p, out),
            (2, _) => copy_rows::<2>(block, p, out, shape.height),
            (4, _) => copy_rows::<4>(block, p, out, shape.height),
            (8, _) => copy_rows::<8>(block, p, out, shape.height),
            (16, _) => copy_rows::<16>(block, p, out, shape.height),
            _ => copy_rows::<32>(block, p, out, shape.height),
        }
    }

    #[cold]
    #[inline(never)]
    fn copy_block_clamped(&self, offset: isize, shape: Shape, out: &mut [u8]) {
        for row in 0..shape.height {
            let start = offset + (row * self.pitch) as isize;
            self.copy_clamped(start, &mut out[row * shape.width..][..shape.width]);
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

    pub(crate) fn slice(&self, offset: isize, len: usize) -> Option<&'a [u8]> {
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

#[inline]
fn copy_fixed<const W: usize, const H: usize>(block: &[u8], pitch: usize, out: &mut [u8]) {
    assert!(out.len() >= H * W && block.len() >= (H - 1) * pitch + W);
    let (src, dst) = (block.as_ptr(), out.as_mut_ptr());
    for row in 0..H {
        unsafe {
            let value = src.add(row * pitch).cast::<[u8; W]>().read_unaligned();
            dst.add(row * W).cast::<[u8; W]>().write_unaligned(value);
        }
    }
}

fn copy_rows<const W: usize>(block: &[u8], pitch: usize, out: &mut [u8], rows: usize) {
    assert!(out.len() >= rows * W && block.len() >= (rows - 1) * pitch + W);
    let (src, dst) = (block.as_ptr(), out.as_mut_ptr());
    for row in 0..rows {
        unsafe { std::ptr::copy_nonoverlapping(src.add(row * pitch), dst.add(row * W), W) };
    }
}
