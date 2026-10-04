//! Echo / delay with damped, filtered feedback.

use super::filters::{Biquad, Kind};
use super::{coef, Block, Ctx, DelayLine, Lfo};
use crate::voice::params::params;

params! {
    /// Echo with feedback that darkens on every repeat.
    pub struct EchoParams {
        /// Delay time.
        time_ms: [10.0, 2000.0, 320.0, "ms"],
        /// Feedback (repeats).
        feedback: [0.0, 0.95, 0.4, ""],
        /// Repeats lose treble above this.
        damping_hz: [500.0, 16000.0, 4000.0, "Hz"],
        /// Repeats lose bass below this.
        lowcut_hz: [20.0, 1000.0, 150.0, "Hz"],
        /// Slight tape wow on the repeats (0..1).
        wow: [0.0, 1.0, 0.1, ""],
        /// Wet level.
        mix: [0.0, 1.0, 0.3, ""],
    }
}

pub struct Echo {
    p: EchoParams,
    rate: f32,
    line: DelayLine,
    lp: Biquad,
    hp: Biquad,
    time: f32,
    time_a: f32,
    lfo: Lfo,
}

impl Echo {
    pub fn new(p: EchoParams, rate: f32) -> Self {
        Echo {
            p,
            rate,
            line: DelayLine::new((rate * 2.1) as usize),
            lp: Biquad::new(Kind::LowPass, p.damping_hz, 0.707, 0.0, rate),
            hp: Biquad::new(Kind::HighPass, p.lowcut_hz, 0.707, 0.0, rate),
            time: p.time_ms * 0.001 * rate,
            time_a: coef(0.05, rate),
            lfo: Lfo::new(0.0),
        }
    }
}

impl Block for Echo {
    fn process(&mut self, buf: &mut [f32], _ctx: &mut Ctx) {
        let target = self.p.time_ms * 0.001 * self.rate;
        let wow = self.p.wow * 0.0015 * self.rate;
        for x in buf.iter_mut() {
            self.time += (target - self.time) * self.time_a;
            let d = self.time + wow * self.lfo.tick(0.7, self.rate);
            let wet = self.line.read_cubic(d.max(2.0));
            let fb = self.hp.tick(self.lp.tick(wet)) * self.p.feedback;
            self.line.push(*x + fb);
            *x += wet * self.p.mix;
        }
    }
    fn set_param(&mut self, i: usize, v: f32) {
        if self.p.set(i, v) {
            self.lp.set(Kind::LowPass, self.p.damping_hz, 0.707, 0.0, self.rate);
            self.hp.set(Kind::HighPass, self.p.lowcut_hz, 0.707, 0.0, self.rate);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn echo_repeats_at_the_delay_time() {
        let rate = 48000.0;
        let mut e = Echo::new(EchoParams { time_ms: 100.0, feedback: 0.5, mix: 1.0, wow: 0.0, damping_hz: 16000.0, lowcut_hz: 20.0 }, rate);
        let mut b = vec![0.0f32; 24000];
        b[0] = 1.0;
        e.process(&mut b, &mut Ctx::default());
        let peak = (1000..24000).max_by(|a, c| b[*a].abs().total_cmp(&b[*c].abs())).unwrap();
        assert!((peak as i64 - 4800).abs() <= 3, "{peak}");
        assert!(b[9600 - 5..9600 + 5].iter().any(|v| v.abs() > 0.2));
    }
}
