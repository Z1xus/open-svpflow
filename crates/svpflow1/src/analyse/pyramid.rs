use super::field::recalc::RecalcPlan;
use super::field::{Field, Mv, VectorOutput};
use super::gpu::Session;
use super::params::{AnalyseParams, Stage};
use super::planes::LevelFrame;
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
        mut on_finest: impl FnMut(usize, &Field, i32),
    ) {
        let [mut out_forward, mut out_backward] = outputs;
        let top = stage.levels - 1;
        let min_level = params.min_level();
        let (mut global_forward, mut global_backward) = (Mv::UNSET, Mv::UNSET);
        let top_settings = stage.settings(params, top, true);
        {
            let (f, b) = self.pair(top as usize);
            let (first_top, second_top) = (first.level(top), second.level(top));
            both(
                || {
                    f.search(
                        &first_top,
                        &second_top,
                        &top_settings,
                        global_forward,
                        None,
                        None,
                    );
                },
                || {
                    if let Some(b) = b {
                        b.search(
                            &second_top,
                            &first_top,
                            &top_settings,
                            global_backward,
                            None,
                            None,
                        );
                    }
                },
            );
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
            let out_f = if finest { out_forward.take() } else { None };
            let out_b = if finest { out_backward.take() } else { None };
            if cfg!(target_arch = "wasm32") {
                let settings = &settings;
                let (src_ref, reference_ref) = (&src, &reference);
                let b_ref = b.as_deref_mut();
                both(
                    || {
                        f.search(
                            src_ref,
                            reference_ref,
                            settings,
                            global_forward,
                            lambda,
                            out_f,
                        );
                    },
                    || {
                        if let Some(b) = b_ref {
                            b.search(
                                reference_ref,
                                src_ref,
                                settings,
                                global_backward,
                                None,
                                out_b,
                            );
                        }
                    },
                );
                if finest {
                    on_finest(0, f, *lambda_out);
                    if let Some(b) = b {
                        on_finest(1, b, *lambda_out);
                    }
                }
                continue;
            }
            f.search(&src, &reference, &settings, global_forward, lambda, out_f);
            if finest {
                on_finest(0, f, *lambda_out);
            }
            if let Some(b) = b {
                let out = out_b;
                b.search(&reference, &src, &settings, global_backward, None, out);
                if finest {
                    on_finest(1, b, *lambda_out);
                }
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
        session: Option<&mut Session<'_>>,
    ) {
        let [mut out_forward, mut out_backward] = outputs;
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
        if session.is_none() {
            let (src, reference, settings) = (&src, &reference, &settings);
            both(
                || f.recalculate(src, reference, settings, Some(lambda_out), out_forward),
                || {
                    if let Some(b) = b {
                        b.recalculate(reference, src, settings, None, out_backward);
                    }
                },
            );
            return;
        }
        let mut plans = [RecalcPlan::default(), RecalcPlan::default()];
        let planned_f = f.plan_recalculate(&src, &settings, out_forward.is_some(), &mut plans[0]);
        if planned_f {
            *lambda_out = settings.lambda >> 2;
        } else {
            f.recalculate(
                &src,
                &reference,
                &settings,
                Some(lambda_out),
                out_forward.as_deref_mut(),
            );
        }
        let mut b = b;
        let planned_b = b.as_deref_mut().is_some_and(|b| {
            let planned =
                b.plan_recalculate(&reference, &settings, out_backward.is_some(), &mut plans[1]);
            if !planned {
                b.recalculate(
                    &reference,
                    &src,
                    &settings,
                    None,
                    out_backward.as_deref_mut(),
                );
            }
            planned
        });
        let mut session = session;
        let mut tickets = [None, None];
        if let Some(session) = session.as_deref_mut() {
            if planned_f {
                tickets[0] = session.submit(&plans[0], 0, 1);
            }
            if planned_b {
                tickets[1] = session.submit(&plans[1], 1, 0);
            }
            if tickets.iter().any(Option::is_some) && !session.wait() {
                tickets = [None, None];
            }
        }
        let mut results = |index: usize, src: &LevelFrame<'_>, reference: &LevelFrame<'_>| {
            tickets[index]
                .and_then(|ticket| session.as_deref_mut().map(|s| s.take(ticket)))
                .filter(|r| r.len() == plans[index].blocks())
                .unwrap_or_else(|| Field::evaluate_recalc_cpu(&plans[index], src, reference))
        };
        if planned_f {
            let r = results(0, &src, &reference);
            f.finish_recalculate(&plans[0], &r, out_forward);
        }
        if planned_b && let Some(b) = b {
            let r = results(1, &reference, &src);
            b.finish_recalculate(&plans[1], &r, out_backward);
        }
    }
}

pub(crate) struct RefineStart {
    plan: RecalcPlan,
    ticket: Option<usize>,
}

impl Pyramid {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn refine_start(
        &mut self,
        direction: usize,
        seed: &Field,
        settings: &super::field::SearchSettings,
        frames: [&SuperFrameView<'_>; 2],
        session: &mut Session<'_>,
        output: Option<VectorOutput<'_>>,
        lambda_out: Option<&mut i32>,
    ) -> Option<RefineStart> {
        let field = if direction == 0 {
            self.forward[0].as_mut()?
        } else if self.both {
            self.backward[0].as_mut()?
        } else {
            return None;
        };
        field.interpolate_prediction(seed, true);
        field.set_order(None);
        let (src, reference) = if direction == 0 {
            (frames[0].level(0), frames[1].level(0))
        } else {
            (frames[1].level(0), frames[0].level(0))
        };
        let mut plan = RecalcPlan::default();
        if !field.plan_recalculate(&src, settings, output.is_some(), &mut plan) {
            field.recalculate(&src, &reference, settings, lambda_out, output);
            return None;
        }
        if let Some(lambda) = lambda_out {
            *lambda = settings.lambda >> 2;
        }
        let ticket = session.submit(&plan, direction, 1 - direction);
        Some(RefineStart { plan, ticket })
    }

    pub(crate) fn refine_finish(
        &mut self,
        starts: [Option<RefineStart>; 2],
        frames: [&SuperFrameView<'_>; 2],
        session: &mut Session<'_>,
        outputs: [Option<VectorOutput<'_>>; 2],
    ) {
        let waited = starts.iter().flatten().any(|s| s.ticket.is_some()) && session.wait();
        for (direction, (start, output)) in starts.into_iter().zip(outputs).enumerate() {
            let Some(start) = start else {
                continue;
            };
            let field = if direction == 0 {
                self.forward[0].as_mut()
            } else {
                self.backward[0].as_mut()
            };
            let Some(field) = field else {
                continue;
            };
            let (src, reference) = if direction == 0 {
                (frames[0].level(0), frames[1].level(0))
            } else {
                (frames[1].level(0), frames[0].level(0))
            };
            let bests = start
                .ticket
                .filter(|_| waited)
                .map(|ticket| session.take(ticket))
                .filter(|r| r.len() == start.plan.blocks())
                .unwrap_or_else(|| Field::evaluate_recalc_cpu(&start.plan, &src, &reference));
            field.finish_recalculate(&start.plan, &bests, output);
        }
    }
}

fn both<A: Send, B: Send>(a: impl FnOnce() -> A + Send, b: impl FnOnce() -> B + Send) -> (A, B) {
    rayon::join(a, b)
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
