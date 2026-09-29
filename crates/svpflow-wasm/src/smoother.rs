use std::collections::BTreeMap;
use std::ops::Range;

use rayon::prelude::*;
use svpflow_core::smooth::{Frame, FrameMut, Plane, PlaneMut};
use svpflow_core::smooth_engine::{Engine, Job, Output, interleave_pel};
use svpflow_core::smooth_options::SourceInfo;
use svpflow1_vs::SuperFrame;

struct Source {
    planes: [Vec<u8>; 3],
    expanded: Option<[Vec<u8>; 3]>,
}

pub struct Smoother {
    engine: Engine,
    width: usize,
    height: usize,
    last: i32,
    vectors: BTreeMap<i32, Vec<u8>>,
    frames: BTreeMap<i32, Source>,
}

impl Smoother {
    pub fn new(
        width: i32,
        height: i32,
        fps_num: i32,
        fps_den: i32,
        source_frames: i32,
        options: &str,
        vector_header: &[i32],
    ) -> Result<Self, String> {
        if fps_num <= 0 || fps_den <= 0 || source_frames <= 0 {
            return Err("invalid source timing".into());
        }
        let header: Vec<u8> = vector_header.iter().flat_map(|v| v.to_le_bytes()).collect();
        let source = SourceInfo {
            fps_num: i64::from(fps_num),
            fps_den: i64::from(fps_den),
            width,
            height,
        };
        Ok(Self {
            engine: Engine::new(options, &source, &header)?,
            width: usize::try_from(width).map_err(|_| "invalid width")?,
            height: usize::try_from(height).map_err(|_| "invalid height")?,
            last: source_frames - 1,
            vectors: BTreeMap::new(),
            frames: BTreeMap::new(),
        })
    }

    #[must_use]
    pub fn output_frames(&self) -> i32 {
        self.engine.output_frames(self.last + 1)
    }

    #[must_use]
    pub fn uses_super(&self) -> bool {
        self.engine.uses_super()
    }

    #[must_use]
    pub const fn frame_len(&self) -> usize {
        self.width * self.height * 3 / 2
    }

    #[must_use]
    pub fn plan(&self, frame: i32) -> [i32; 4] {
        let (n, vectors) = self.engine.plan(frame);
        let clamp = |k: i32| k.clamp(0, self.last);
        [
            clamp(n),
            clamp(n + 1),
            clamp(vectors.start),
            if vectors.is_empty() {
                clamp(vectors.start)
            } else {
                clamp(vectors.end - 1) + 1
            },
        ]
    }

    pub fn set_vectors(&mut self, k: i32, payload: &[u8]) {
        self.vectors.insert(k, payload.to_vec());
    }

    pub fn set_source(&mut self, k: i32, frame: &[u8]) -> Result<(), String> {
        if self.uses_super() {
            return Err("this configuration needs super frames".into());
        }
        let planes = self.split(frame)?;
        self.frames.insert(
            k,
            Source {
                planes,
                expanded: None,
            },
        );
        Ok(())
    }

    pub fn set_super(&mut self, k: i32, frame: &SuperFrame) -> Result<(), String> {
        let [y, u, v] = frame.planes();
        let (width, height) = (self.width, self.height);
        let take = |plane: &[u8], len: usize| {
            plane
                .get(..len)
                .map(<[u8]>::to_vec)
                .ok_or_else(|| String::from("super frame does not match the source size"))
        };
        let planes = [
            take(y, width * height)?,
            take(u, width * height / 4)?,
            take(v, width * height / 4)?,
        ];
        let expanded = if self.uses_super() {
            let pel = usize::try_from(self.engine.pel()).map_err(|_| "invalid pel")?;
            let expand = |plane: &[u8], pw: usize, ph: usize| {
                interleave_pel(plane, pw, ph, pel)
                    .ok_or_else(|| String::from("super frame does not match the vectors"))
            };
            Some([
                expand(y, width, height)?,
                expand(u, width / 2, height / 2)?,
                expand(v, width / 2, height / 2)?,
            ])
        } else {
            None
        };
        self.frames.insert(k, Source { planes, expanded });
        Ok(())
    }

    pub fn forget_before(&mut self, k: i32) {
        self.vectors = self.vectors.split_off(&k);
        self.frames = self.frames.split_off(&k);
    }

    pub fn render_into(&mut self, frame: i32, out: &mut [u8]) -> Result<(), String> {
        if out.len() != self.frame_len() {
            return Err("invalid output buffer length".into());
        }
        let last = self.last;
        let vectors = &self.vectors;
        let output = self
            .engine
            .prepare(frame, |k| vectors.get(&k.clamp(0, last)).map(Vec::as_slice));
        let source = |k: i32| {
            self.frames
                .get(&k.clamp(0, last))
                .ok_or_else(|| format!("source frame {} is missing", k.clamp(0, last)))
        };
        match output {
            Output::Copy(k) => {
                let [y, u, v] = &source(k)?.planes;
                let (oy, rest) = out.split_at_mut(y.len());
                let (ou, ov) = rest.split_at_mut(u.len());
                oy.copy_from_slice(y);
                ou.copy_from_slice(u);
                ov.copy_from_slice(v);
                Ok(())
            }
            Output::Render(job) => {
                let current = self.view(source(job.n)?);
                let next = self.view(source(job.n + 1)?);
                self.render_job(&job, current, next, out);
                Ok(())
            }
            Output::Gpu(_) => Err("unexpected GPU job".into()),
        }
    }

    pub fn prepare_gpu(&mut self, frame: i32) -> Output {
        let last = self.last;
        let vectors = &self.vectors;
        match self
            .engine
            .prepare_gpu(frame, |k| vectors.get(&k.clamp(0, last)).map(Vec::as_slice))
        {
            Output::Copy(k) => Output::Copy(k.clamp(0, last)),
            output => output,
        }
    }

    fn split(&self, frame: &[u8]) -> Result<[Vec<u8>; 3], String> {
        if frame.len() != self.frame_len() {
            return Err("invalid source buffer length".into());
        }
        let luma = self.width * self.height;
        let (y, chroma) = frame.split_at(luma);
        let (u, v) = chroma.split_at(luma / 4);
        Ok([y.to_vec(), u.to_vec(), v.to_vec()])
    }

    fn view<'a>(&self, source: &'a Source) -> Frame<'a> {
        let (planes, scale) = match &source.expanded {
            Some(expanded) => (expanded, usize::try_from(self.engine.pel()).unwrap_or(1)),
            None => (&source.planes, 1),
        };
        let plane = |data: &'a [u8], width: usize| Plane {
            data,
            pitch: width * scale,
            slack: 0,
        };
        Frame {
            y: plane(&planes[0], self.width),
            u: plane(&planes[1], self.width / 2),
            v: plane(&planes[2], self.width / 2),
        }
    }

    fn render_job(&self, job: &Job, current: Frame<'_>, next: Frame<'_>, out: &mut [u8]) {
        let blocks = job.blocks();
        let total = usize::try_from(blocks.end - blocks.start)
            .unwrap_or(1)
            .max(1);
        let parts = (rayon::current_num_threads() * 2).clamp(1, total);
        let bands: Vec<Range<i32>> = (0..parts)
            .map(|i| {
                let at = |i: usize| blocks.start + i32::try_from(i * total / parts).unwrap_or(0);
                at(i)..at(i + 1)
            })
            .filter(|band| !band.is_empty())
            .collect();
        let luma = self.width * self.height;
        let (y, chroma) = out.split_at_mut(luma);
        let (u, v) = chroma.split_at_mut(luma / 4);
        let mut rest = [y, u, v];
        let mut offsets = [0usize; 3];
        let pitches = [self.width, self.width / 2, self.width / 2];
        let mut work = Vec::with_capacity(bands.len());
        for band in bands {
            let [luma_rows, chroma_rows] = job.band_rows(band.clone());
            let rows = [&luma_rows, &chroma_rows, &chroma_rows];
            let slices: [&mut [u8]; 3] = std::array::from_fn(|p| {
                let start = rows[p].start * pitches[p];
                let end = (rows[p].end * pitches[p]).max(start);
                let plane = std::mem::take(&mut rest[p]);
                let (_, tail) =
                    plane.split_at_mut(start.saturating_sub(offsets[p]).min(plane.len()));
                let (band, after) = tail.split_at_mut((end - start).min(tail.len()));
                rest[p] = after;
                offsets[p] = end.max(offsets[p]);
                band
            });
            work.push((band, slices));
        }
        work.into_par_iter().for_each(|(band, [y, u, v])| {
            let mut dst = FrameMut {
                y: PlaneMut {
                    data: y,
                    pitch: pitches[0],
                },
                u: PlaneMut {
                    data: u,
                    pitch: pitches[1],
                },
                v: PlaneMut {
                    data: v,
                    pitch: pitches[2],
                },
            };
            job.render_band(&mut dst, current, next, band);
        });
    }
}
