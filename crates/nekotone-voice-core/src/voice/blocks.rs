//! The catalogue of effect blocks: a serialisable spec per block type and
//! the factory that turns a spec into a running [`Block`].

use super::dsp::character::*;
use super::dsp::delay::*;
use super::dsp::dynamics::*;
use super::dsp::filters::{self, *};
use super::dsp::modulation::*;
use super::dsp::reverb::*;
use super::dsp::route::*;
use super::dsp::shifter::*;
use super::dsp::timbre::*;
use super::dsp::Block;
use super::params::{index_of, ParamDef};
use serde::{Deserialize, Serialize};

/// One block's type and settings. JSON: `{"type": "reverb", "decay_s": 6, ...}`;
/// missing fields take their defaults.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BlockSpec {
    /// Pitch/formant shifter, harmoniser, robot, whisper (LP-PSOLA).
    Voice(VoiceParams),
    Whisper(WhisperParams),
    Gate(GateParams),
    DeEsser(DeEsserParams),
    HighPass(PassParams),
    LowPass(PassParams),
    BandPass(BandParams),
    Eq(EqParams),
    Compressor(CompressorParams),
    Limiter(LimiterParams),
    Gain(GainParams),
    Saturation(SaturationParams),
    Bitcrush(BitcrushParams),
    RingMod(RingModParams),
    Chorus(ChorusParams),
    Vibrato(VibratoParams),
    Tremolo(TremoloParams),
    Wobble(WobbleParams),
    Octaver(OctaverParams),
    Growl(GrowlParams),
    Reverb(ReverbParams),
    Echo(EchoParams),
    Radio(RadioParams),
    Timbre(TimbreParams),
    Split(SplitParams),
    Layer(LayerParams),
    Merge(MergeParams),
}

/// For the UI's "add block" menu.
#[derive(Debug, Clone, Serialize)]
pub struct BlockType {
    /// JSON `type` value.
    pub kind: &'static str,
    pub label: &'static str,
    pub description: &'static str,
    pub params: &'static [ParamDef],
    /// Default settings.
    pub default: BlockSpec,
}

macro_rules! each_block {
    ($m:ident) => {
        $m! {
            Voice, VoiceParams, "voice", "Voice (pitch & formant)", "Independent pitch and formant shift, harmony voices, vibrato, robot monotone, whisper";
            Whisper, WhisperParams, "whisper", "Whisper", "Re-synthesise the voice from noise through your vocal tract";
            Gate, GateParams, "gate", "Noise gate", "Silence below a threshold, with hysteresis and hold";
            DeEsser, DeEsserParams, "de_esser", "De-esser", "Tame harsh s and sh sounds";
            HighPass, PassParams, "high_pass", "High-pass", "Remove lows below a frequency";
            LowPass, PassParams, "low_pass", "Low-pass", "Remove highs above a frequency";
            BandPass, BandParams, "band_pass", "Band-pass", "Keep a band of frequencies";
            Eq, EqParams, "eq", "Equaliser", "Low shelf, two peaks and a high shelf";
            Compressor, CompressorParams, "compressor", "Compressor", "Even out loud and quiet words";
            Limiter, LimiterParams, "limiter", "Limiter", "Brick-wall peak limiter";
            Gain, GainParams, "gain", "Gain", "Volume";
            Saturation, SaturationParams, "saturation", "Saturation", "Warmth to heavy distortion";
            Bitcrush, BitcrushParams, "bitcrush", "Bitcrusher", "Low bit depth and sample rate";
            RingMod, RingModParams, "ring_mod", "Ring modulator", "Metallic, alien sidebands";
            Chorus, ChorusParams, "chorus", "Chorus / flanger", "Modulated delayed copies";
            Vibrato, VibratoParams, "vibrato", "Vibrato", "Pitch wobble";
            Tremolo, TremoloParams, "tremolo", "Tremolo", "Volume wobble";
            Wobble, WobbleParams, "wobble", "Wobble filter", "LFO-swept resonant filter";
            Octaver, OctaverParams, "octaver", "Octaver", "Adds an octave below (and above)";
            Growl, GrowlParams, "growl", "Growl", "Sub-harmonic rumble and rasp for monsters";
            Reverb, ReverbParams, "reverb", "Reverb", "Rooms, halls and cathedrals";
            Echo, EchoParams, "echo", "Echo", "Repeating delay";
            Radio, RadioParams, "radio", "Radio / telephone", "Band-limited speaker with drive and hiss";
            Timbre, TimbreParams, "timbre", "Timbre match", "Keeps level, body and brightness on target whoever speaks, loud or quiet";
            Split, SplitParams, "split", "Layer: split", "Start a parallel layer: copy the signal to a layer bus";
            Layer, LayerParams, "layer", "Layer: process layer", "From here, process the layer's copy (the main signal waits on the bus)";
            Merge, MergeParams, "merge", "Layer: merge", "Mix the layer with the main signal, each at its own level";
        }
    };
}

macro_rules! impl_spec {
    ($( $v:ident, $p:ident, $kind:literal, $label:literal, $desc:literal; )*) => {
        impl BlockSpec {
            /// Parameter table of this block type.
            pub fn params(&self) -> &'static [ParamDef] {
                match self { $( BlockSpec::$v(_) => $p::PARAMS, )* }
            }
            /// JSON `type` name.
            pub fn kind(&self) -> &'static str {
                match self { $( BlockSpec::$v(_) => $kind, )* }
            }
            /// Current value of a parameter by name.
            pub fn get(&self, name: &str) -> Option<f32> {
                let i = index_of(self.params(), name)?;
                match self { $( BlockSpec::$v(p) => p.get(i), )* }
            }
            /// Set a parameter by index (clamped). False when out of range.
            pub fn set_index(&mut self, i: usize, v: f32) -> bool {
                match self { $( BlockSpec::$v(p) => p.set(i, v), )* }
            }
            /// Clamp every field into range.
            pub fn sanitize(&mut self) {
                match self { $( BlockSpec::$v(p) => p.sanitize(), )* }
            }
        }

        /// Every block type with its parameters.
        pub fn block_types() -> Vec<BlockType> {
            vec![ $( BlockType {
                kind: $kind, label: $label, description: $desc,
                params: $p::PARAMS, default: BlockSpec::$v($p::default()),
            }, )* ]
        }
    };
}
each_block!(impl_spec);

impl BlockSpec {
    /// Index of a parameter by name.
    pub fn param_index(&self, name: &str) -> Option<usize> {
        index_of(self.params(), name)
    }

    /// True when the block rebuilds the voice from an analysis (so the
    /// original pitch cannot pass through it).
    pub fn resynthesises(&self) -> bool {
        matches!(self, BlockSpec::Voice(_) | BlockSpec::Whisper(_))
    }

    /// Create the running block at `rate`.
    pub fn build(&self, rate: f32) -> Box<dyn Block> {
        match *self {
            BlockSpec::Voice(p) => Box::new(Shifter::new(p, rate)),
            BlockSpec::Whisper(p) => Box::new(Shifter::new(p.to_voice(), rate)),
            BlockSpec::Gate(p) => Box::new(Gate::new(p, rate)),
            BlockSpec::DeEsser(p) => Box::new(DeEsser::new(p, rate)),
            BlockSpec::HighPass(p) => Box::new(Pass::new(filters::Kind::HighPass, p, rate)),
            BlockSpec::LowPass(p) => Box::new(Pass::new(filters::Kind::LowPass, p, rate)),
            BlockSpec::BandPass(p) => Box::new(Band::new(p, rate)),
            BlockSpec::Eq(p) => Box::new(Eq::new(p, rate)),
            BlockSpec::Compressor(p) => Box::new(Compressor::new(p, rate)),
            BlockSpec::Limiter(p) => Box::new(Limiter::new(p, rate)),
            BlockSpec::Gain(p) => Box::new(Gain::new(p, rate)),
            BlockSpec::Saturation(p) => Box::new(Saturation::new(p, rate)),
            BlockSpec::Bitcrush(p) => Box::new(Bitcrush::new(p, rate)),
            BlockSpec::RingMod(p) => Box::new(RingMod::new(p, rate)),
            BlockSpec::Chorus(p) => Box::new(Chorus::new(p, rate)),
            BlockSpec::Vibrato(p) => Box::new(Vibrato::new(p, rate)),
            BlockSpec::Tremolo(p) => Box::new(Tremolo::new(p, rate)),
            BlockSpec::Wobble(p) => Box::new(Wobble::new(p, rate)),
            BlockSpec::Octaver(p) => Box::new(Octaver::new(p, rate)),
            BlockSpec::Growl(p) => Box::new(Growl::new(p, rate)),
            BlockSpec::Reverb(p) => Box::new(Reverb::new(p, rate)),
            BlockSpec::Echo(p) => Box::new(Echo::new(p, rate)),
            BlockSpec::Radio(p) => Box::new(Radio::new(p, rate)),
            BlockSpec::Timbre(p) => Box::new(Timbre::new(p, rate)),
            BlockSpec::Split(p) => Box::new(Route::split(p)),
            BlockSpec::Layer(p) => Box::new(Route::layer(p)),
            BlockSpec::Merge(p) => Box::new(Route::merge(p, rate)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalogue_round_trips_through_json() {
        let types = block_types();
        assert_eq!(types.len(), 27);
        for t in &types {
            let j = serde_json::to_string(&t.default).unwrap();
            assert!(j.contains(&format!("\"type\":\"{}\"", t.kind)), "{j}");
            let back: BlockSpec = serde_json::from_str(&j).unwrap();
            assert_eq!(back, t.default);
            assert_eq!(back.kind(), t.kind);
            // every block builds and processes at two rates
            for rate in [44100.0, 48000.0] {
                let mut b = back.build(rate);
                let mut buf = vec![0.1f32; 256];
                b.process(&mut buf, &mut crate::voice::dsp::Ctx { rate, ..Default::default() });
                assert!(buf.iter().all(|v| v.is_finite()));
            }
        }
        let s: BlockSpec = serde_json::from_str(r#"{"type":"voice","pitch_st":4.5}"#).unwrap();
        assert_eq!(s.get("pitch_st"), Some(4.5));
        assert_eq!(s.get("formant"), Some(1.0));
    }
}
