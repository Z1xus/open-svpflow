#![allow(dead_code, clippy::unused_self)]

use super::field::recalc::RecalcPlan;

pub(crate) struct RefineGpu;

pub(crate) struct Session<'g, 'p>(std::marker::PhantomData<(&'g (), &'p ())>);

pub(crate) struct Planes<'a> {
    pub(crate) data: [&'a [u8]; 3],
    pub(crate) pitch: [usize; 3],
    pub(crate) base: [usize; 3],
    pub(crate) height: [i32; 3],
    pub(crate) pel: i32,
    pub(crate) expand: bool,
}

pub(crate) fn refine_gpu() -> Option<&'static RefineGpu> {
    None
}

impl RefineGpu {
    pub(crate) fn session<'p>(&self) -> Option<Session<'_, 'p>> {
        None
    }
}

impl<'p> Session<'_, 'p> {
    pub(crate) fn defer(&mut self, _: usize, _: (usize, i32), _: Planes<'p>) {}

    pub(crate) fn submit(&mut self, _: &RecalcPlan, _: usize, _: usize) -> Option<usize> {
        None
    }

    pub(crate) fn submit_compact(
        &mut self,
        _: &super::field::recalc::CompactPlan,
        _: usize,
        _: usize,
    ) -> Option<usize> {
        None
    }

    pub(crate) fn take_into(&mut self, _: usize, _: &mut [u32]) {}

    pub(crate) fn wait(&mut self) -> bool {
        false
    }

    pub(crate) fn take(&mut self, _: usize) -> Vec<[i32; 4]> {
        Vec::new()
    }
}
