//! The processing graph shared by real time and offline rendering:
//!
//! ```text
//! mono in ─ HPF ─ gate ─ de-esser ─ preset chain (blocks…, latency pad) ─ output gain ─ brick-wall limiter ─ mute ─ out
//!                                   └ old chain (while cross-fading) ┘
//! ```
//!
//! There is no dry path: the only way from input to output is through the
//! chain and the limiter. Every chain is padded to the same latency so
//! preset changes cross-fade without a doubled image.

use super::dsp::dynamics::{DeEsser, DeEsserParams, Gate, GateParams, Limiter, LimiterParams};
use super::blocks::BlockSpec;
use super::dsp::filters::{Biquad, Kind};
use super::dsp::route::{bus_index, level, Route, BUSES};
use super::dsp::profile::{Profile, Profiler};
use super::dsp::shifter::Shifter;
use super::dsp::{db_to_lin, Block, Ctx, FixedDelay, Smoothed};
use super::presets::BlockSlot;
use serde::{Deserialize, Serialize};

/// Microphone clean-up applied before every preset (set once for your mic).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FrontEnd {
    /// Rumble filter corner (12 dB/octave).
    pub hpf_hz: f32,
    pub gate: GateParams,
    pub deesser_on: bool,
    pub deesser: DeEsserParams,
    /// Realism guard: cut steady metallic ringing in the processed voice
    /// (see `dsp::tamer`); on by default.
    pub tamer_on: bool,
}

impl Default for FrontEnd {
    fn default() -> Self {
        FrontEnd { hpf_hz: 70.0, gate: GateParams::default(), deesser_on: true, deesser: DeEsserParams::default(), tamer_on: true }
    }
}

/// Latency every chain is padded to (the voice stage's latency).
pub fn chain_latency(rate: f32) -> usize {
    Shifter::latency_for(rate)
}

/// Largest buffer the chain processes in one go (the layer buses' size;
/// the processor's blocks are never larger).
const BUS_LEN: usize = 4096;

/// A preset instantiated at a sample rate.
pub struct Chain {
    blocks: Vec<Box<dyn Block>>,
    buses: Vec<Vec<f32>>,
    pad: FixedDelay,
    latency: usize,
}

impl Chain {
    pub fn new(slots: &[BlockSlot], rate: f32) -> Chain {
        // latency of the working signal and of each layer bus, so a merge
        // can line both sides up
        let mut lat = 0usize;
        let mut bus_lat = [0usize; BUSES];
        let mut blocks: Vec<Box<dyn Block>> = Vec::with_capacity(slots.len());
        for s in slots {
            match s.block {
                BlockSpec::Split(p) => bus_lat[bus_index(p.bus)] = lat,
                BlockSpec::Layer(p) => std::mem::swap(&mut lat, &mut bus_lat[bus_index(p.bus)]),
                _ => {}
            }
            if let BlockSpec::Merge(p) = s.block {
                let b = bus_index(p.bus);
                let t = lat.max(bus_lat[b]);
                let mut r = Route::merge(p, rate);
                r.align(t - lat, t - bus_lat[b]);
                lat = t;
                blocks.push(Box::new(r));
                continue;
            }
            let b = s.block.build(rate);
            lat += b.latency();
            blocks.push(b);
        }
        let target = chain_latency(rate);
        let pad = target.saturating_sub(lat);
        let buses = if slots.iter().any(|s| matches!(s.block, BlockSpec::Split(_))) { vec![vec![0.0; BUS_LEN]; BUSES] } else { Vec::new() };
        Chain { blocks, buses, pad: FixedDelay::new(pad), latency: lat + pad }
    }
    pub fn latency(&self) -> usize {
        self.latency
    }
    pub fn process(&mut self, buf: &mut [f32], ctx: &mut Ctx) {
        for chunk in buf.chunks_mut(BUS_LEN) {
            let n = chunk.len();
            for b in self.blocks.iter_mut() {
                match b.bus() {
                    Some(k) => {
                        if let Some(bus) = self.buses.get_mut(k) {
                            b.route(chunk, &mut bus[..n]);
                        }
                    }
                    None => b.process(chunk, ctx),
                }
            }
            self.pad.process(chunk, ctx);
        }
    }
    pub fn set_param(&mut self, block: usize, index: usize, value: f32) {
        if let Some(b) = self.blocks.get_mut(block) {
            b.set_param(index, value);
        }
    }
}

/// True when every path from the input to the output of `slots` passes
/// through a re-synthesising block (so a disguising preset cannot leak the
/// dry voice through a layer).
pub fn every_path_resynthesises(slots: &[BlockSlot]) -> bool {
    let mut work = false;
    let mut bus = [false; BUSES];
    for s in slots {
        match s.block {
            BlockSpec::Split(p) => bus[bus_index(p.bus)] = work,
            BlockSpec::Layer(p) => std::mem::swap(&mut work, &mut bus[bus_index(p.bus)]),
            BlockSpec::Merge(p) => {
                let b = bus[bus_index(p.bus)];
                let (on_l, on_m) = (level(p.layer_db) > 0.0, level(p.main_db) > 0.0);
                work = match (on_l, on_m) {
                    (true, true) => work && b,
                    (true, false) => work,
                    (false, true) => b,
                    (false, false) => true,
                };
            }
            b if b.resynthesises() => work = true,
            _ => {}
        }
    }
    work
}

/// Messages from the control side to the audio thread. Everything that
/// allocates (building a chain) happens before sending.
pub enum DspMsg {
    SwapChain(Box<Chain>),
    Param { block: usize, index: usize, value: f32 },
    Mute(bool),
    OutputGainDb(f32),
    FrontEnd(FrontEnd),
    CeilingDb(f32),
    /// Start the speaker profile again from this prior (a new calibration, or `None`).
    Profile(Option<Profile>),
    /// Play a sound pad (clip at the engine rate; the sender keeps a reference, see `pads`).
    PlayPad { id: u32, clip: std::sync::Arc<[f32]>, gain_db: f32 },
    /// Stop one pad, or all (`None`), with a short fade.
    StopPads(Option<u32>),
    /// How far the voice ducks while a pad plays (dB).
    PadDuck(f32),
}

/// Meter values updated by `process`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Meters {
    /// Peak of the input since the last reset (linear).
    pub in_peak: f32,
    /// Peak of the output since the last reset (linear).
    pub out_peak: f32,
    pub gate_open: bool,
    /// Bit `id % 32` set for each sound pad playing.
    pub pads_playing: u32,
    pub pitch_hz: f32,
    pub out_pitch_hz: f32,
    pub limiter_gain: f32,
    /// Who is speaking, as measured so far.
    pub profile: Profile,
}

pub struct Processor {
    rate: f32,
    max_block: usize,
    hpf: [Biquad; 1],
    gate: Gate,
    deesser: DeEsser,
    front: FrontEnd,
    /// Measures the speaker (after the gate, before the chain) for voices with targets.
    profiler: Profiler,
    chain: Box<Chain>,
    old: Option<Box<Chain>>,
    trash: Option<Box<Chain>>,
    /// Negative while the new chain is still filling its latency (silent).
    xfade_pos: i64,
    xfade_len: usize,
    scratch: Vec<f32>,
    out_gain: Smoothed,
    limiter: Limiter,
    muted: bool,
    mute_gain: f32,
    mute_step: f32,
    /// True once audio has been processed; a mute that arrives before that
    /// applies at once instead of fading (so "start muted" leaks nothing).
    started: bool,
    ctx: Ctx,
    /// Sound pads, mixed after the chain (see `pads`).
    pads: super::pads::PadMixer,
    /// Resonance tamer on the processed voice (before the pads).
    tamer: super::dsp::tamer::Tamer,
    pub meters: Meters,
}

impl Processor {
    pub fn new(rate: f32, max_block: usize, front: FrontEnd, slots: &[BlockSlot], output_gain_db: f32, ceiling_db: f32) -> Self {
        let max_block = max_block.clamp(16, 4096);
        Processor {
            rate,
            max_block,
            hpf: [Biquad::new(Kind::HighPass, front.hpf_hz, 0.707, 0.0, rate)],
            gate: Gate::new(front.gate, rate),
            deesser: DeEsser::new(front.deesser, rate),
            front,
            profiler: Profiler::new(rate, None),
            chain: Box::new(Chain::new(slots, rate)),
            old: None,
            trash: None,
            xfade_pos: 0,
            xfade_len: (rate * 0.04) as usize,
            scratch: vec![0.0; max_block],
            out_gain: Smoothed::new(db_to_lin(output_gain_db), 0.02, rate),
            limiter: Limiter::new(LimiterParams { ceiling_db, ..Default::default() }, rate),
            muted: false,
            mute_gain: 1.0,
            mute_step: 1.0 / (rate * 0.005),
            started: false,
            ctx: Ctx { rate, ..Default::default() },
            pads: super::pads::PadMixer::new(rate),
            tamer: super::dsp::tamer::Tamer::new(rate),
            meters: Meters { limiter_gain: 1.0, ..Default::default() },
        }
    }

    /// Start the speaker profile from a saved calibration (`None`: neutral).
    /// Allocates: call before processing or from the control side.
    pub fn with_profile(mut self, prior: Option<Profile>) -> Self {
        self.profiler = Profiler::new(self.rate, prior);
        self.ctx.profile = self.profiler.profile();
        self.meters.profile = self.ctx.profile;
        self
    }

    /// Forget the measured speaker and start again from `prior` (allocation-free).
    pub fn restart_profile(&mut self, prior: Option<Profile>) {
        self.profiler.restart(prior);
        self.ctx.profile = self.profiler.profile();
        self.meters.profile = self.ctx.profile;
    }

    /// The speaker profile measured so far.
    pub fn profile(&self) -> Profile {
        self.profiler.profile()
    }

    pub fn rate(&self) -> f32 {
        self.rate
    }

    /// Total algorithmic latency in samples (chain + limiter lookahead).
    pub fn latency(&self) -> usize {
        self.chain.latency() + self.limiter.latency()
    }

    pub fn ceiling(&self) -> f32 {
        self.limiter.ceiling()
    }

    pub fn gate_threshold_db(&self) -> f32 {
        self.gate.threshold_db()
    }

    pub fn noise_floor_db(&self) -> f32 {
        self.gate.noise_floor_db()
    }

    /// Apply a control message (audio thread; allocation-free).
    pub fn handle(&mut self, msg: DspMsg) {
        match msg {
            DspMsg::SwapChain(c) => self.swap_chain(c),
            DspMsg::Param { block, index, value } => self.chain.set_param(block, index, value),
            DspMsg::Mute(m) => {
                self.muted = m;
                if !self.started {
                    self.mute_gain = if m { 0.0 } else { 1.0 };
                }
            }
            DspMsg::OutputGainDb(db) => self.out_gain.set(db_to_lin(db)),
            DspMsg::FrontEnd(f) => self.set_front_end(f),
            DspMsg::CeilingDb(db) => self.limiter.set_ceiling_db(db),
            DspMsg::Profile(p) => self.restart_profile(p),
            DspMsg::PlayPad { id, clip, gain_db } => self.pads.play(id, clip, gain_db),
            DspMsg::StopPads(id) => self.pads.stop(id),
            DspMsg::PadDuck(db) => self.pads.set_duck_db(db),
        }
    }

    pub fn set_front_end(&mut self, f: FrontEnd) {
        self.front = f;
        self.hpf[0].set(Kind::HighPass, f.hpf_hz, 0.707, 0.0, self.rate);
        self.gate.set_params(f.gate);
        self.deesser.set_params(f.deesser);
    }

    /// Start a cross-fade to `c`. If a fade is already running, the fading-out
    /// chain is parked for collection and the current one fades out instead.
    pub fn swap_chain(&mut self, c: Box<Chain>) {
        let prev = std::mem::replace(&mut self.chain, c);
        if let Some(o) = self.old.take() {
            if self.trash.is_none() {
                self.trash = Some(o);
            } else {
                // both slots full: keep the older one fading (rare; the
                // control thread collects trash every few ms)
                self.old = Some(o);
                self.trash = Some(prev);
                self.xfade_pos = -(self.chain.latency() as i64);
                return;
            }
        }
        self.old = Some(prev);
        // a fresh chain is silent for its latency: start fading once it speaks
        self.xfade_pos = -(self.chain.latency() as i64);
    }

    /// A chain that finished fading out, to be dropped off the audio thread.
    pub fn take_trash(&mut self) -> Option<Box<Chain>> {
        self.trash.take()
    }

    pub fn is_muted(&self) -> bool {
        self.muted
    }

    /// Process mono samples in place.
    pub fn process(&mut self, buf: &mut [f32]) {
        let mb = self.max_block;
        for chunk in buf.chunks_mut(mb) {
            self.process_block(chunk);
        }
    }

    fn process_block(&mut self, buf: &mut [f32]) {
        let n = buf.len();
        let mut ip = self.meters.in_peak;
        for x in buf.iter_mut() {
            ip = ip.max(x.abs());
            *x = self.hpf[0].tick(*x);
        }
        self.meters.in_peak = ip;
        self.gate.process(buf, &mut self.ctx);
        if self.front.deesser_on {
            self.deesser.process(buf, &mut self.ctx);
        }
        self.meters.gate_open = self.gate.is_open();
        self.profiler.process(buf);
        self.ctx.profile = self.profiler.profile();
        self.meters.profile = self.ctx.profile;
        if let Some(old) = self.old.as_mut() {
            let s = &mut self.scratch[..n];
            s.copy_from_slice(buf);
            old.process(s, &mut self.ctx);
            self.chain.process(buf, &mut self.ctx);
            let len = self.xfade_len as f32;
            for (x, o) in buf.iter_mut().zip(s.iter()) {
                let t = (self.xfade_pos.max(0) as f32 / len).min(1.0);
                // equal-power fade
                let a = (t * std::f32::consts::FRAC_PI_2).sin();
                let b = (t * std::f32::consts::FRAC_PI_2).cos();
                *x = *x * a + *o * b;
                self.xfade_pos += 1;
            }
            if self.xfade_pos >= self.xfade_len as i64 && self.trash.is_none() {
                self.trash = self.old.take();
            }
        } else {
            self.chain.process(buf, &mut self.ctx);
        }
        self.meters.pitch_hz = self.ctx.pitch_hz;
        self.meters.out_pitch_hz = self.ctx.out_pitch_hz;
        // the realism guard's resonance tamer, on the voice only (not the pads)
        if self.front.tamer_on {
            self.tamer.process(buf, self.ctx.out_pitch_hz);
        }
        // pads: the voice ducks and the clips mix in, before the output gain,
        // the limiter and the mute (so the limiter holds both, mute silences both)
        self.pads.process(buf);
        self.meters.pads_playing = self.pads.playing_mask;
        for x in buf.iter_mut() {
            *x *= self.out_gain.next();
            if !x.is_finite() {
                *x = 0.0;
            }
        }
        self.limiter.min_gain = 1.0;
        self.limiter.process(buf, &mut self.ctx);
        self.meters.limiter_gain = self.limiter.min_gain;
        self.started = true;
        let mut op = self.meters.out_peak;
        for x in buf.iter_mut() {
            if self.muted {
                self.mute_gain = (self.mute_gain - self.mute_step).max(0.0);
            } else {
                self.mute_gain = (self.mute_gain + self.mute_step).min(1.0);
            }
            *x *= self.mute_gain;
            op = op.max(x.abs());
        }
        self.meters.out_peak = op;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voice::blocks::BlockSpec;
    use crate::voice::dsp::character::GainParams;

    fn slot(b: BlockSpec) -> BlockSlot {
        BlockSlot { id: "x".into(), block: b }
    }

    #[test]
    fn chains_are_padded_to_one_latency() {
        let rate = 48000.0;
        let a = Chain::new(&[slot(BlockSpec::Gain(GainParams::default()))], rate);
        let b = Chain::new(&[slot(BlockSpec::Voice(Default::default()))], rate);
        assert_eq!(a.latency(), b.latency());
        assert_eq!(a.latency(), chain_latency(rate));
    }

    #[test]
    fn pads_reach_the_output_under_the_limiter_and_mute_silences_them() {
        use crate::voice::pads::prepare_clip;
        let rate = 48000.0;
        let front = FrontEnd { gate: GateParams { threshold_db: -90.0, auto: 0.0, ..Default::default() }, ..Default::default() };
        let mut p = Processor::new(rate, 128, front, &[slot(BlockSpec::Gain(GainParams { db: 0.0 }))], 0.0, -1.0);
        // a loud clip (+12 dB gain on a 0.9 tone): the limiter must hold it
        let clip = prepare_clip(&(0..48000).map(|i| 0.9 * (i as f32 * 0.03).sin()).collect::<Vec<_>>(), rate);
        p.handle(DspMsg::PlayPad { id: 2, clip, gain_db: 12.0 });
        let mut out = Vec::new();
        for _ in 0..200 {
            let mut b = vec![0.0f32; 128];
            p.process(&mut b);
            out.extend_from_slice(&b);
        }
        let peak = out.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let ceiling = crate::voice::dsp::db_to_lin(-1.0);
        assert!(peak > 0.5 && peak <= ceiling * 1.001, "pad heard, limited: peak {peak}");
        assert_eq!(p.meters.pads_playing, 1 << 2);
        p.handle(DspMsg::Mute(true));
        let mut b = vec![0.0f32; 9600];
        p.process(&mut b);
        assert!(b[4800..].iter().all(|v| *v == 0.0), "mute silences the pads too");
        p.handle(DspMsg::StopPads(None));
        let mut b = vec![0.0f32; 1024];
        p.process(&mut b);
        assert_eq!(p.meters.pads_playing, 0);
    }

    #[test]
    fn mute_outputs_silence_and_crossfade_is_smooth() {
        let rate = 48000.0;
        let front = FrontEnd { gate: GateParams { threshold_db: -90.0, auto: 0.0, ..Default::default() }, ..Default::default() };
        let mut p = Processor::new(rate, 128, front, &[slot(BlockSpec::Gain(GainParams { db: 0.0 }))], 0.0, -1.0);
        let tone = |i: usize| 0.3 * (i as f32 * 2.0 * std::f32::consts::PI * 220.0 / rate).sin();
        let mut out = Vec::new();
        for blk in 0..400 {
            if blk == 150 {
                p.handle(DspMsg::SwapChain(Box::new(Chain::new(&[slot(BlockSpec::Gain(GainParams { db: -6.0 }))], rate))));
            }
            let mut b: Vec<f32> = (0..128).map(|k| tone(blk * 128 + k)).collect();
            p.process(&mut b);
            out.extend_from_slice(&b);
            if let Some(t) = p.take_trash() {
                drop(t);
            }
        }
        // no clicks: sample-to-sample change stays within a sine's slope
        // (skip the gate opening in the first 100 ms)
        let max_step = out[4800..].windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0, f32::max);

        assert!(max_step < 0.3 * 2.0 * std::f32::consts::PI * 220.0 / rate * 1.3, "step {max_step}");
        p.handle(DspMsg::Mute(true));
        let mut b = vec![0.5f32; 4800];
        p.process(&mut b);
        assert!(b[1000..].iter().all(|v| *v == 0.0));
    }
}
