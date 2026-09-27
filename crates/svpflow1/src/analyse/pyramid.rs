use super::field::{Field, Mv, VectorOutput};
use super::params::{AnalyseParams, Stage};
use super::planes::SuperFrameView;

pub(crate) struct Pyramid {
    forward: Vec<Option<Field>>,
    backward: Vec<Option<Field>>,
    both: bool,
}

impl Pyramid {
    pub(crate) fn new(params: &AnalyseParams, stage: &Stage) -> Self {
        let min_level = params.min_level();
        let build = || {
            (0..stage.levels)
                .map(|level| {
                    (level >= min_level)
                        .then(|| stage.field(level, params.super_params.pel, params.chroma_shift))
                })
                .collect()
        };
        Self {
            forward: build(),
            backward: build(),
            both: params.vectors == 3,
        }
    }

    fn pair(&mut self, level: usize) -> (&mut Field, Option<&mut Field>) {
        let forward = self.forward[level].as_mut().expect("searched level");
        let backward = if self.both {
            self.backward[level].as_mut()
        } else {
            None
        };
        (forward, backward)
    }

    pub(crate) fn search(
        &mut self,
        params: &AnalyseParams,
        stage: &Stage,
        first: &SuperFrameView<'_>,
        second: &SuperFrameView<'_>,
        lambda_out: &mut i32,
        outputs: [Option<VectorOutput<'_>>; 2],
    ) {
        let [mut out_forward, mut out_backward] = outputs;
        let top = stage.levels - 1;
        let min_level = params.min_level();
        let (mut global_forward, mut global_backward) = (Mv::UNSET, Mv::UNSET);
        let top_settings = stage.settings(params, top, true);
        {
            let (f, b) = self.pair(top as usize);
            f.search(
                &first.level(top),
                &second.level(top),
                &top_settings,
                global_forward,
                None,
                None,
            );
            if let Some(b) = b {
                b.search(
                    &second.level(top),
                    &first.level(top),
                    &top_settings,
                    global_backward,
                    None,
                    None,
                );
            }
        }
        for level in (min_level..top).rev() {
            let settings = stage.settings(params, level, false);
            let sort = params.sort && level >= stage.full_math_level;
            let coarse = level as usize + 1;
            order_level(&mut self.forward, level as usize, sort);
            if self.both {
                order_level(&mut self.backward, level as usize, sort);
            }
            self.forward[coarse]
                .as_ref()
                .expect("coarser level")
                .estimate_global_doubled(&mut global_forward);
            if self.both {
                self.backward[coarse]
                    .as_ref()
                    .expect("coarser level")
                    .estimate_global_doubled(&mut global_backward);
                let vx = i32::midpoint(i32::from(global_forward.x), i32::from(global_backward.x));
                let vy = i32::midpoint(i32::from(global_forward.y), i32::from(global_backward.y));
                global_forward.x = (i32::from(global_forward.x) - vx) as i16;
                global_forward.y = (i32::from(global_forward.y) - vy) as i16;
                global_backward.x = (i32::from(global_backward.x) - vx) as i16;
                global_forward.y = (i32::from(global_forward.y) - vy) as i16;
            }
            predict_level(&mut self.forward, level as usize);
            if self.both {
                predict_level(&mut self.backward, level as usize);
            }
            let finest = level == min_level;
            let (f, b) = self.pair(level as usize);
            let mut b = b;
            if let Some(b) = b.as_deref_mut() {
                std::mem::swap(&mut f.reverse, &mut b.reverse);
            }
            let (src, reference) = (first.level(level), second.level(level));
            let lambda = Some(&mut *lambda_out);
            f.search(
                &src,
                &reference,
                &settings,
                global_forward,
                lambda,
                if finest { out_forward.take() } else { None },
            );
            if let Some(b) = b {
                let out = if finest { out_backward.take() } else { None };
                b.search(&reference, &src, &settings, global_backward, None, out);
            }
        }
    }

    pub(crate) fn recalculate(
        &mut self,
        params: &AnalyseParams,
        stage: &Stage,
        previous: &Self,
        first: &SuperFrameView<'_>,
        second: &SuperFrameView<'_>,
        lambda_out: &mut i32,
        outputs: [Option<VectorOutput<'_>>; 2],
    ) {
        let [out_forward, out_backward] = outputs;
        let mut settings = stage.settings(params, 0, true);
        settings.lambda = *lambda_out;
        let seeds = [&previous.forward[0], &previous.backward[0]];
        for (fields, seed) in [&mut self.forward, &mut self.backward]
            .into_iter()
            .zip(seeds)
        {
            if let (Some(field), Some(seed)) = (fields[0].as_mut(), seed.as_ref()) {
                field.interpolate_prediction(seed, true);
                field.set_order(None);
            }
        }
        let (src, reference) = (first.level(0), second.level(0));
        let (f, b) = self.pair(0);
        f.recalculate(&src, &reference, &settings, Some(lambda_out), out_forward);
        if let Some(b) = b {
            b.recalculate(&reference, &src, &settings, None, out_backward);
        }
    }
}

fn order_level(fields: &mut [Option<Field>], level: usize, sort: bool) {
    let (fine, coarse) = fields.split_at_mut(level + 1);
    let coarse = coarse[0].as_mut().expect("coarser level");
    if sort {
        coarse.sort_blocks();
    }
    fine[level]
        .as_mut()
        .expect("searched level")
        .set_order(sort.then_some(&*coarse));
}

fn predict_level(fields: &mut [Option<Field>], level: usize) {
    let (fine, coarse) = fields.split_at_mut(level + 1);
    let coarse = coarse[0].as_ref().expect("coarser level");
    fine[level]
        .as_mut()
        .expect("searched level")
        .interpolate_prediction(coarse, false);
}
