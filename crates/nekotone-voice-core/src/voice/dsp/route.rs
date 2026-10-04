//! Parallel layers inside one flat preset chain.
//!
//! Sound designers build creature voices from parallel layers (a main voice
//! plus a darker sub layer with its own saturation, say) that are mixed
//! back together. A preset stays a flat list of blocks; three routing
//! blocks give it layers through up to three layer buses:
//!
//! ```text
//! split(bus 1) ─ main blocks… ─ layer(bus 1) ─ layer blocks… ─ merge(bus 1) ─ shared blocks…
//!      │ input copied to the bus    │ swap: the bus (input) is now processed,
//!      │                            │ the processed main waits on the bus
//!      └────────────────────────────┴──────────▶ merge: layer·layer_db + main·main_db
//! ```
//!
//! [`crate::voice::chain::Chain`] runs these blocks (they need the chain's
//! bus buffers) and aligns the two sides of a merge when their latencies
//! differ. On their own (outside a chain) they pass the signal through.

use super::{db_to_lin, Block, Ctx, DelayLine, Smoothed};
use crate::voice::params::params;

/// Number of layer buses.
pub const BUSES: usize = 3;

params! {
    /// Start a parallel layer: copy the signal to a layer bus.
    pub struct SplitParams {
        /// Layer bus (1-3).
        bus: [1.0, 3.0, 1.0, ""],
    }
}

params! {
    /// Switch to the layer: from here the bus's copy is processed while the main signal waits on the bus.
    pub struct LayerParams {
        /// Layer bus (1-3).
        bus: [1.0, 3.0, 1.0, ""],
    }
}

params! {
    /// Mix the layer being processed with the main signal waiting on the bus.
    pub struct MergeParams {
        /// Layer bus (1-3).
        bus: [1.0, 3.0, 1.0, ""],
        /// Level of the layer (the signal processed since `layer`); -60 = off.
        layer_db: [-60.0, 12.0, -6.0, "dB"],
        /// Level of the main signal waiting on the bus; -60 = off.
        main_db: [-60.0, 12.0, 0.0, "dB"],
    }
}

/// Bus index (0-based) from a 1-based parameter.
pub fn bus_index(v: f32) -> usize {
    (v.round() as usize).clamp(1, BUSES) - 1
}

/// dB to gain where -60 dB and below is silence.
pub fn level(db: f32) -> f32 {
    if db <= -59.5 {
        0.0
    } else {
        db_to_lin(db)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RouteKind {
    Split,
    Layer,
    Merge,
}

/// A routing block. `route` does the work inside a chain.
pub struct Route {
    kind: RouteKind,
    bus: usize,
    merge: MergeParams,
    gl: Smoothed,
    gm: Smoothed,
    /// Alignment delays for a merge (layer side, main side).
    dl: Option<(DelayLine, usize)>,
    dm: Option<(DelayLine, usize)>,
}

impl Route {
    pub fn split(p: SplitParams) -> Self {
        Self::new(RouteKind::Split, bus_index(p.bus), MergeParams::default(), 48000.0)
    }
    pub fn layer(p: LayerParams) -> Self {
        Self::new(RouteKind::Layer, bus_index(p.bus), MergeParams::default(), 48000.0)
    }
    pub fn merge(p: MergeParams, rate: f32) -> Self {
        Self::new(RouteKind::Merge, bus_index(p.bus), p, rate)
    }
    fn new(kind: RouteKind, bus: usize, merge: MergeParams, rate: f32) -> Self {
        Route {
            kind,
            bus,
            merge,
            gl: Smoothed::new(level(merge.layer_db), 0.02, rate),
            gm: Smoothed::new(level(merge.main_db), 0.02, rate),
            dl: None,
            dm: None,
        }
    }
    pub fn kind(&self) -> RouteKind {
        self.kind
    }
    /// Delay the layer side by `layer` and the main side by `main` samples
    /// before mixing (set by the chain so both sides line up).
    pub fn align(&mut self, layer: usize, main: usize) {
        self.dl = (layer > 0).then(|| (DelayLine::new(layer + 1), layer));
        self.dm = (main > 0).then(|| (DelayLine::new(main + 1), main));
    }
}

impl Block for Route {
    fn process(&mut self, _buf: &mut [f32], _ctx: &mut Ctx) {}

    fn set_param(&mut self, i: usize, v: f32) {
        if self.kind == RouteKind::Merge && self.merge.set(i, v) {
            self.gl.set(level(self.merge.layer_db));
            self.gm.set(level(self.merge.main_db));
        }
    }

    fn bus(&self) -> Option<usize> {
        Some(self.bus)
    }

    fn route(&mut self, work: &mut [f32], bus: &mut [f32]) {
        match self.kind {
            RouteKind::Split => bus.copy_from_slice(work),
            RouteKind::Layer => work.swap_with_slice(bus),
            RouteKind::Merge => {
                for (w, b) in work.iter_mut().zip(bus.iter()) {
                    let mut l = *w;
                    let mut m = *b;
                    if let Some((d, n)) = self.dl.as_mut() {
                        d.push(l);
                        l = d.tap(*n);
                    }
                    if let Some((d, n)) = self.dm.as_mut() {
                        d.push(m);
                        m = d.tap(*n);
                    }
                    *w = l * self.gl.next() + m * self.gm.next();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_layer_merge_mixes_two_paths() {
        let mut bus = vec![0.0f32; 4];
        let mut work = vec![1.0f32, 2.0, 3.0, 4.0];
        Route::split(SplitParams { bus: 1.0 }).route(&mut work, &mut bus);
        assert_eq!(bus, work);
        for w in work.iter_mut() {
            *w *= 10.0; // "main processing"
        }
        Route::layer(LayerParams { bus: 1.0 }).route(&mut work, &mut bus);
        assert_eq!(work, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(bus, vec![10.0, 20.0, 30.0, 40.0]);
        let mut m = Route::merge(MergeParams { bus: 1.0, layer_db: 0.0, main_db: 0.0 }, 48000.0);
        m.route(&mut work, &mut bus);
        assert_eq!(work, vec![11.0, 22.0, 33.0, 44.0]);
        // alignment delays the earlier side
        let mut m = Route::merge(MergeParams { bus: 1.0, layer_db: -60.0, main_db: 0.0 }, 48000.0);
        m.align(0, 2);
        let mut work = vec![0.0f32; 4];
        let mut bus = vec![1.0f32, 2.0, 3.0, 4.0];
        m.route(&mut work, &mut bus);
        assert_eq!(work, vec![0.0, 0.0, 1.0, 2.0]);
    }
}
