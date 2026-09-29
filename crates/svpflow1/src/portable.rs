use crate::{
    analyse::{self, AnalyseParams, RawPlane, SuperFrameView, SuperParams},
    params, super_build,
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
    params: AnalyseParams,
    super_opts: SuperOpts,
}

impl SuperBuilder {
    pub fn new(width: i32, height: i32, pel: i32) -> Result<Self, String> {
        if !matches!(pel, 1 | 2 | 4) {
            return Err("pel must be 1, 2 or 4".into());
        }
        Self::with_options(width, height, &format!("{{pel:{pel}}}"))
    }

    pub fn with_options(width: i32, height: i32, options: &str) -> Result<Self, String> {
        if width <= 0 || height <= 0 || width % 2 != 0 || height % 2 != 0 {
            return Err("dimensions must be positive and even".into());
        }
        let options = params::parse(options.as_bytes())?;
        let opts = SuperOpts::from_opt(Some(&options), width, height)?;
        if !opts.full {
            return Err("full:false is not supported here".into());
        }
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
        let (source_y, chroma) = source.split_at(self.source_y_len);
        let (source_u, source_v) = chroma.split_at(self.source_chroma_len);
        let mut y = vec![0; self.output_y_len];
        let mut u = vec![0; self.output_chroma_len];
        let mut v = vec![0; self.output_chroma_len];
        let opts = self.opts;
        rayon::join(
            || {
                super_build::build_plane(
                    &mut y,
                    width,
                    source_y,
                    width,
                    width,
                    height,
                    (0, 0),
                    &opts,
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
                            width,
                            height,
                            (1, 1),
                            &opts,
                        );
                    },
                    || {
                        super_build::build_plane(
                            &mut v,
                            chroma_width,
                            source_v,
                            chroma_width,
                            width,
                            height,
                            (1, 1),
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
    pub fn planes(&self) -> [&[u8]; 3] {
        [&self.y, &self.u, &self.v]
    }

    #[must_use]
    pub fn bytes(&self) -> Vec<u8> {
        let mut output = Vec::with_capacity(self.len());
        output.extend_from_slice(&self.y);
        output.extend_from_slice(&self.u);
        output.extend_from_slice(&self.v);
        output
    }

    fn view(&self) -> SuperFrameView<'_> {
        let width = usize::try_from(self.opts.width).unwrap_or(0);
        SuperFrameView {
            planes: [
                RawPlane {
                    data: &self.y,
                    pitch: width,
                },
                RawPlane {
                    data: &self.u,
                    pitch: width / 2,
                },
                RawPlane {
                    data: &self.v,
                    pitch: width / 2,
                },
            ],
            source: None,
            width: self.opts.width,
            height: self.opts.height,
            pel: self.opts.pel,
            chroma_shift: (1, 1),
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
        Self::with_options(
            super_builder,
            &format!(
                "{{block:{{w:{block_width},h:{block_height},overlap:{overlap_mode}}},vectors:{vectors}}}"
            ),
        )
    }

    pub fn with_options(super_builder: &SuperBuilder, options: &str) -> Result<Self, String> {
        let options = params::parse(options.as_bytes())?;
        let super_params = SuperParams::unpack(super_builder.opts.pack_data())?;
        Ok(Self {
            params: AnalyseParams::new(Some(&options), super_params, (1, 1))?,
            super_opts: super_builder.opts,
        })
    }

    #[must_use]
    pub fn vector_header(&self) -> Vec<i32> {
        let mut header = analyse::analysis_header(&self.params).to_vec();
        header.push(0);
        header
    }

    pub fn analyse(&self, current: &SuperFrame, reference: &SuperFrame) -> Result<Vec<u8>, String> {
        if current.opts.pack_data() != self.super_opts.pack_data()
            || reference.opts.pack_data() != self.super_opts.pack_data()
        {
            return Err("super frame configuration mismatch".into());
        }
        Ok(analyse::analyse(
            &self.params,
            &current.view(),
            &reference.view(),
            None,
        ))
    }
}
