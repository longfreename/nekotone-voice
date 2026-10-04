//! Sound pads (the stream deck): short clips mixed into the voice output,
//! after the voice chain and before the limiter and mute, with the voice
//! ducked while a pad plays.
//!
//! ```text
//!  voice chain ─▶ × duck ─┬─▶ output gain ─▶ limiter ─▶ mute ─▶ out
//!  pads (up to 4 at once) ─┘
//! ```
//!
//! Real-time safety: clips arrive already decoded and resampled to the
//! engine's rate (`Arc<[f32]>`, built off the audio thread). The engine keeps
//! every clip it has sent in a cache for the whole session, so when the
//! audio thread drops its handle the reference count never reaches zero
//! there: nothing is freed on the audio thread. Mixing allocates nothing.
//!
//! Privacy: a pad is a file, never the microphone; the path from the mic to
//! the output is unchanged (it still passes only through the voice chain).

use std::sync::Arc;

/// Pads that can sound at the same time (a new one replaces the oldest).
pub const MAX_VOICES: usize = 4;
/// Longest clip a pad plays (seconds); longer files are cut.
pub const MAX_CLIP_S: f32 = 20.0;
/// Most pads a session may load (their clips stay in memory while live).
pub const MAX_PADS: usize = 12;

#[derive(Clone)]
struct Voice {
    clip: Arc<[f32]>,
    pos: usize,
    gain: f32,
    id: u32,
    /// Fade-out samples left when stopping (0 = playing).
    fade: u32,
    stopping: bool,
}

/// The pad mixer on the audio thread.
pub struct PadMixer {
    voices: [Option<Voice>; MAX_VOICES],
    /// Voice gain while a pad plays (linear).
    duck: f32,
    duck_now: f32,
    attack: f32,
    release: f32,
    fade_len: u32,
    /// Ids of pads that are sounding (for the page), refreshed per block.
    pub playing_mask: u32,
}

impl PadMixer {
    pub fn new(rate: f32) -> Self {
        PadMixer {
            voices: Default::default(),
            duck: super::dsp::db_to_lin(-10.0),
            duck_now: 1.0,
            attack: 1.0 - (-1.0 / (0.020 * rate)).exp(),
            release: 1.0 - (-1.0 / (0.300 * rate)).exp(),
            fade_len: (0.010 * rate) as u32,
            playing_mask: 0,
        }
    }

    /// How far the voice goes down while a pad plays (dB, 0..40).
    pub fn set_duck_db(&mut self, db: f32) {
        self.duck = super::dsp::db_to_lin(-db.clamp(0.0, 40.0));
    }

    /// Start `clip` (already at the engine's rate) as pad `id` at `gain_db`.
    /// Pressing a pad that is playing starts it again from the top.
    pub fn play(&mut self, id: u32, clip: Arc<[f32]>, gain_db: f32) {
        let v = Voice { clip, pos: 0, gain: super::dsp::db_to_lin(gain_db.clamp(-40.0, 12.0)), id, fade: 0, stopping: false };
        // the same pad again, or a free slot, or the one that has played longest
        let slot = self
            .voices
            .iter()
            .position(|s| s.as_ref().is_some_and(|s| s.id == id))
            .or_else(|| self.voices.iter().position(|s| s.is_none()))
            .unwrap_or_else(|| {
                let mut best = 0;
                for (i, s) in self.voices.iter().enumerate() {
                    if let (Some(a), Some(b)) = (s, &self.voices[best]) {
                        if a.pos > b.pos {
                            best = i;
                        }
                    }
                }
                best
            });
        // the replaced handle is only a reference-count decrement (the engine keeps the clip)
        self.voices[slot] = Some(v);
    }

    /// Fade out pad `id` (or every pad when `None`) over 10 ms.
    pub fn stop(&mut self, id: Option<u32>) {
        for v in self.voices.iter_mut().flatten() {
            if id.is_none_or(|i| i == v.id) && !v.stopping {
                v.stopping = true;
                v.fade = self.fade_len.max(1);
            }
        }
    }

    pub fn is_playing(&self) -> bool {
        self.voices.iter().any(|v| v.is_some())
    }

    /// Duck the voice in `buf` and add the pads.
    pub fn process(&mut self, buf: &mut [f32]) {
        let active = self.is_playing();
        if !active && self.duck_now >= 0.9999 {
            self.playing_mask = 0;
            return;
        }
        let target = if active { self.duck } else { 1.0 };
        for (i, x) in buf.iter_mut().enumerate() {
            let a = if target < self.duck_now { self.attack } else { self.release };
            self.duck_now += (target - self.duck_now) * a;
            let mut mix = 0.0;
            for v in self.voices.iter_mut().flatten() {
                if let Some(s) = v.clip.get(v.pos + i) {
                    let fade = if v.stopping {
                        let f = v.fade.saturating_sub(i as u32) as f32 / self.fade_len.max(1) as f32;
                        f.clamp(0.0, 1.0)
                    } else {
                        1.0
                    };
                    mix += s * v.gain * fade;
                }
            }
            *x = *x * self.duck_now + mix;
        }
        let n = buf.len();
        let mut mask = 0u32;
        for slot in self.voices.iter_mut() {
            let done = match slot {
                Some(v) => {
                    v.pos += n;
                    if v.stopping {
                        v.fade = v.fade.saturating_sub(n as u32);
                    }
                    v.pos >= v.clip.len() || (v.stopping && v.fade == 0)
                }
                None => false,
            };
            if done {
                *slot = None;
            } else if let Some(v) = slot {
                mask |= 1 << (v.id % 32);
            }
        }
        self.playing_mask = mask;
    }
}

/// Mono samples of `clip` at `rate`, cut to [`MAX_CLIP_S`] (off the audio thread).
pub fn prepare_clip(samples: &[f32], rate: f32) -> Arc<[f32]> {
    let n = samples.len().min((MAX_CLIP_S * rate) as usize);
    let mut v: Vec<f32> = samples[..n].iter().map(|s| if s.is_finite() { *s } else { 0.0 }).collect();
    // 5 ms fades at both ends: no click at the start or at a cut
    let f = ((0.005 * rate) as usize).min(v.len() / 2);
    for i in 0..f {
        let g = i as f32 / f as f32;
        v[i] *= g;
        let j = v.len() - 1 - i;
        v[j] *= g;
    }
    v.into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(n: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|i| amp * (i as f32 * 0.05).sin()).collect()
    }

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
    }

    #[test]
    fn a_pad_mixes_in_and_the_voice_ducks_then_recovers() {
        let rate = 48000.0;
        let mut m = PadMixer::new(rate);
        m.set_duck_db(12.0);
        let clip = prepare_clip(&tone(24000, 0.3), rate); // 0.5 s
        m.play(1, clip, 0.0);
        let mut voice = vec![0.5f32; 96000];
        for c in voice.chunks_mut(256) {
            m.process(c);
        }
        // 100-400 ms: the pad sounds and the voice is 12 dB down
        let mid: Vec<f32> = voice[4800..19200].to_vec();
        let voice_part = mid.iter().sum::<f32>() / mid.len() as f32; // the tone averages ~0
        assert!((20.0 * (voice_part / 0.5).log10() + 12.0).abs() < 0.5, "ducked by {:.1} dB", 20.0 * (voice_part / 0.5).log10());
        assert!(rms(&mid) > 0.2, "the pad is heard");
        // 1.5 s after the pad ended (the release is 300 ms) the voice is back
        let end = voice[95000..].iter().sum::<f32>() / 1000.0;
        assert!((end - 0.5).abs() < 0.01, "recovered to {end}");
        assert!(!m.is_playing() && m.playing_mask == 0);
    }

    #[test]
    fn stop_fades_out_and_the_same_pad_restarts() {
        let rate = 48000.0;
        let mut m = PadMixer::new(rate);
        let clip = prepare_clip(&tone(96000, 0.3), rate);
        m.play(3, clip.clone(), 0.0);
        let mut b = vec![0.0f32; 4800];
        m.process(&mut b);
        assert_eq!(m.playing_mask, 1 << 3);
        m.play(3, clip.clone(), 0.0);
        assert_eq!(m.voices.iter().flatten().count(), 1, "pressing again restarts, it does not stack");
        m.stop(None);
        let mut b = vec![0.0f32; 2400];
        m.process(&mut b);
        assert!(b[480..].iter().all(|v| *v == 0.0), "silent after the 10 ms fade");
        let steps = b.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0, f32::max);
        assert!(steps < 0.05, "no click while fading: {steps}");
        assert!(!m.is_playing());
    }

    #[test]
    fn at_most_four_at_once_the_oldest_gives_way() {
        let rate = 48000.0;
        let mut m = PadMixer::new(rate);
        let clip = prepare_clip(&tone(96000, 0.1), rate);
        for id in 0..6 {
            m.play(id, clip.clone(), 0.0);
            let mut b = vec![0.0f32; 480];
            m.process(&mut b);
        }
        assert_eq!(m.voices.iter().flatten().count(), MAX_VOICES);
        assert_eq!(m.playing_mask.count_ones(), MAX_VOICES as u32);
        // the audio thread held only references: the caller's clip is still whole
        assert_eq!(clip.len(), 96000);
    }

    #[test]
    fn clips_are_cut_cleaned_and_faded() {
        let rate = 1000.0;
        let mut x = vec![1.0f32; 30_000];
        x[100] = f32::NAN;
        let c = prepare_clip(&x, rate);
        assert_eq!(c.len(), (MAX_CLIP_S * rate) as usize);
        assert_eq!(c[0], 0.0);
        assert!(c[100].is_finite());
        assert!(c[c.len() - 1].abs() < 1e-6);
    }
}
