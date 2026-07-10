use crate::{
    analyse_opts::{AnalyseOpts, auto_search_levels, overlap_from_mode},
    analyse_search::{self, SuperPlanes},
    super_build,
    super_opts::SuperOpts,
};

#[derive(Clone, Copy)]
pub struct SuperBuilder {
    opts: SuperOpts,
    source_y_len: usize,
    source_chroma_len: usize,
    output_y_len: usize,
    output_chroma_len: usize,
}

pub struct SuperFrame {
    opts: SuperOpts,
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
}

pub struct Analyser {
    opts: AnalyseOpts,
    super_opts: SuperOpts,
}

impl SuperBuilder {
    pub fn new(width: i32, height: i32, pel: i32) -> Result<Self, String> {
        if !matches!(pel, 1 | 2 | 4) {
            return Err("pel must be 1, 2 or 4".into());
        }
        if width <= 0 || height <= 0 || width % 2 != 0 || height % 2 != 0 {
            return Err("dimensions must be positive and even".into());
        }
        let mut opts = SuperOpts::from_opt(None, width, height)?;
        opts.pel = pel;
        let width = usize::try_from(width).map_err(|_| "invalid width")?;
        let height = usize::try_from(height).map_err(|_| "invalid height")?;
        let chroma_width = width / 2;
        let chroma_height = height / 2;
        let super_height =
            usize::try_from(opts.super_height()).map_err(|_| "invalid super size")?;
        let source_y_len = width.checked_mul(height).ok_or("source size overflow")?;
        let source_chroma_len = chroma_width
            .checked_mul(chroma_height)
            .ok_or("source size overflow")?;
        let output_y_len = width
            .checked_mul(super_height)
            .ok_or("super size overflow")?;
        let output_chroma_len = chroma_width
            .checked_mul(super_height / 2)
            .ok_or("super size overflow")?;
        Ok(Self {
            opts,
            source_y_len,
            source_chroma_len,
            output_y_len,
            output_chroma_len,
        })
    }

    #[must_use]
    pub const fn width(&self) -> i32 {
        self.opts.width
    }

    #[must_use]
    pub const fn height(&self) -> i32 {
        self.opts.height
    }

    #[must_use]
    pub const fn pel(&self) -> i32 {
        self.opts.pel
    }

    #[must_use]
    pub const fn levels(&self) -> i32 {
        self.opts.levels
    }

    #[must_use]
    pub const fn source_len(&self) -> usize {
        self.source_y_len + self.source_chroma_len * 2
    }

    #[must_use]
    pub const fn output_len(&self) -> usize {
        self.output_y_len + self.output_chroma_len * 2
    }

    pub fn build(&self, source: &[u8]) -> Result<SuperFrame, String> {
        if source.len() != self.source_len() {
            return Err("invalid source buffer length".into());
        }
        let width = usize::try_from(self.opts.width).map_err(|_| "invalid width")?;
        let height = usize::try_from(self.opts.height).map_err(|_| "invalid height")?;
        let chroma_width = width / 2;
        let chroma_height = height / 2;
        let (source_y, chroma) = source.split_at(self.source_y_len);
        let (source_u, source_v) = chroma.split_at(self.source_chroma_len);
        let mut y = vec![0; self.output_y_len];
        let mut u = vec![0; self.output_chroma_len];
        let mut v = vec![0; self.output_chroma_len];
        let opts = self.opts;
        rayon::join(
            || {
                super_build::build_plane(
                    &mut y, width, source_y, width, width, height, width, height, false, &opts,
                );
            },
            || {
                rayon::join(
                    || {
                        super_build::build_plane(
                            &mut u,
                            chroma_width,
                            source_u,
                            chroma_width,
                            chroma_width,
                            chroma_height,
                            width,
                            height,
                            true,
                            &opts,
                        );
                    },
                    || {
                        super_build::build_plane(
                            &mut v,
                            chroma_width,
                            source_v,
                            chroma_width,
                            chroma_width,
                            chroma_height,
                            width,
                            height,
                            true,
                            &opts,
                        );
                    },
                );
            },
        );
        Ok(SuperFrame { opts, y, u, v })
    }
}

impl SuperFrame {
    #[must_use]
    pub fn len(&self) -> usize {
        self.y.len() + self.u.len() + self.v.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.y.is_empty() && self.u.is_empty() && self.v.is_empty()
    }

    #[must_use]
    pub fn bytes(&self) -> Vec<u8> {
        let mut output = Vec::with_capacity(self.len());
        output.extend_from_slice(&self.y);
        output.extend_from_slice(&self.u);
        output.extend_from_slice(&self.v);
        output
    }

    fn planes(&self) -> SuperPlanes<'_> {
        SuperPlanes {
            y: &self.y,
            y_stride: usize::try_from(self.opts.width).unwrap_or(0),
            u: &self.u,
            u_stride: usize::try_from(self.opts.width / 2).unwrap_or(0),
            v: &self.v,
            v_stride: usize::try_from(self.opts.width / 2).unwrap_or(0),
            luma_w: usize::try_from(self.opts.width).unwrap_or(0),
            luma_h: usize::try_from(self.opts.height).unwrap_or(0),
            pel: self.opts.pel,
            levels: self.opts.levels,
            full: self.opts.full,
        }
    }
}

impl Analyser {
    pub fn new(
        super_builder: &SuperBuilder,
        block_width: i32,
        block_height: i32,
        overlap_mode: i32,
        vectors: i32,
    ) -> Result<Self, String> {
        if !matches!(block_width, 4 | 8 | 16 | 32) || !matches!(block_height, 4 | 8 | 16 | 32) {
            return Err("block dimensions must be 4, 8, 16 or 32".into());
        }
        if !(1..=3).contains(&vectors) {
            return Err("vectors must be 1, 2 or 3".into());
        }
        let (overlap_x, overlap_y) = overlap_from_mode(block_width, block_height, overlap_mode)?;
        let mut opts = AnalyseOpts::from_opt(
            None,
            super_builder.opts.pack_data(),
            Some(&super_builder.opts),
        )?;
        opts.vectors = vectors;
        opts.block_w = block_width;
        opts.block_h = block_height;
        opts.overlap_mode = overlap_mode;
        opts.overlap_x = overlap_x;
        opts.overlap_y = overlap_y;
        opts.levels = auto_search_levels(
            opts.width,
            opts.height,
            block_width,
            block_height,
            overlap_x,
            overlap_y,
        )
        .min(opts.super_levels)
        .max(1);
        opts.lambda = (f64::from(block_width * block_height) * 31.25) as i32;
        Ok(Self {
            opts,
            super_opts: super_builder.opts,
        })
    }

    #[must_use]
    pub fn vector_header(&self) -> Vec<i32> {
        analyse_search::pack_vdata_header(&self.opts)
    }

    pub fn analyse(&self, current: &SuperFrame, reference: &SuperFrame) -> Result<Vec<u8>, String> {
        if current.opts.pack_data() != self.super_opts.pack_data()
            || reference.opts.pack_data() != self.super_opts.pack_data()
        {
            return Err("super frame configuration mismatch".into());
        }
        let current = current.planes();
        let reference = reference.planes();
        let (previous, next) = analyse_search::analyse_pair(&current, &reference, &self.opts);
        Ok(analyse_search::pack_vector_frame(
            &self.opts,
            previous.as_deref(),
            next.as_deref(),
        ))
    }
}
