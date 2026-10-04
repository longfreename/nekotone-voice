//! Shared audio helpers for the voice-only crate.

fn k_filters(rate: u32) -> ([f64; 3], [f64; 3], [f64; 3], [f64; 3]) {
    let fs = rate as f64;
    let (f0, g, q) = (1681.974450955533, 3.999843853973347, 0.7071752369554196);
    let k = (std::f64::consts::PI * f0 / fs).tan();
    let vh = 10f64.powf(g / 20.0);
    let vb = vh.powf(0.4996667741545416);
    let a0 = 1.0 + k / q + k * k;
    let pb = [(vh + vb * k / q + k * k) / a0, 2.0 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0];
    let pa = [1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0];
    let (f0, q) = (38.13547087602444, 0.5003270373238773);
    let k = (std::f64::consts::PI * f0 / fs).tan();
    let d = 1.0 + k / q + k * k;
    let rb = [1.0, -2.0, 1.0];
    let ra = [1.0, 2.0 * (k * k - 1.0) / d, (1.0 - k / q + k * k) / d];
    (pb, pa, rb, ra)
}

/// Zero-delay windowed-sinc resampling of one channel. Output length is `round(len · to / from)`.
pub(crate) fn resample_sinc(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if x.is_empty() || from == to || from == 0 || to == 0 {
        return x.to_vec();
    }
    const ZERO_CROSSINGS: usize = 32;
    const PHASES: usize = 512;
    const BETA: f64 = 9.0;
    let ratio = to as f64 / from as f64;
    let fc = 0.95 * 0.5f64.min(0.5 * ratio);
    let half = (ZERO_CROSSINGS as f64 / (2.0 * fc)).ceil() as usize;
    let bessel_i0 = |v: f64| {
        let (mut sum, mut term, mut k) = (1.0f64, 1.0f64, 1.0f64);
        while term > 1e-12 * sum {
            term *= (v / (2.0 * k)).powi(2);
            sum += term;
            k += 1.0;
        }
        sum
    };
    let i0b = bessel_i0(BETA);
    let len = 2 * half * PHASES + 2;
    let table: Vec<f32> = (0..len)
        .map(|i| {
            let d = i as f64 / PHASES as f64 - half as f64;
            let r = d / half as f64;
            if r.abs() >= 1.0 {
                return 0.0;
            }
            let arg = 2.0 * fc * d;
            let sinc = if arg.abs() < 1e-12 { 1.0 } else { (std::f64::consts::PI * arg).sin() / (std::f64::consts::PI * arg) };
            (2.0 * fc * sinc * bessel_i0(BETA * (1.0 - r * r).sqrt()) / i0b) as f32
        })
        .collect();
    let n = x.len() as i64;
    let out_len = (x.len() as f64 * ratio).round() as usize;
    let step = from as f64 / to as f64;
    (0..out_len)
        .map(|j| {
            let t = j as f64 * step;
            let k0 = t.floor() as i64;
            let frac = t - k0 as f64;
            let lo = (k0 - half as i64 + 1).max(0);
            let hi = (k0 + half as i64).min(n - 1);
            let mut acc = 0.0f32;
            let mut k = lo;
            while k <= hi {
                let pos = ((k0 - k) as f64 + frac + half as f64) * PHASES as f64;
                let i = pos.floor() as usize;
                let a = (pos - i as f64) as f32;
                let w = table[i] + (table[i + 1] - table[i]) * a;
                acc += x[k as usize] * w;
                k += 1;
            }
            acc
        })
        .collect()
}

/// Integrated loudness (LUFS) of interleaved audio, `None` when shorter than one 400 ms block or fully below gate.
pub(crate) fn integrated_lufs(x: &[f32], ch: u16, rate: u32) -> Option<f32> {
    let chn = ch.max(1) as usize;
    let n = x.len() / chn;
    if rate == 0 || n < rate as usize * 4 / 10 {
        return None;
    }
    let (pb, pa, rb, ra) = k_filters(rate);
    let mut sq = vec![0.0f64; n];
    for c in 0..chn {
        let (mut x1, mut x2, mut y1, mut y2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let (mut u1, mut u2, mut z1, mut z2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (i, acc) in sq.iter_mut().enumerate() {
            let s = x[i * chn + c] as f64;
            let y = pb[0] * s + pb[1] * x1 + pb[2] * x2 - pa[1] * y1 - pa[2] * y2;
            x2 = x1;
            x1 = s;
            y2 = y1;
            y1 = y;
            let u = (y as f32) as f64;
            let z = rb[0] * u + rb[1] * u1 + rb[2] * u2 - ra[1] * z1 - ra[2] * z2;
            u2 = u1;
            u1 = u;
            z2 = z1;
            z1 = z;
            *acc += z * z;
        }
    }
    if chn == 1 {
        for v in &mut sq {
            *v *= 2.0;
        }
    }
    let mut prefix = vec![0.0f64; n + 1];
    for i in 0..n {
        prefix[i + 1] = prefix[i] + sq[i];
    }
    let mean = |a: usize, b: usize| (prefix[b] - prefix[a]) / (b - a).max(1) as f64;
    let lufs = |ms: f64| if ms > 0.0 { -0.691 + 10.0 * ms.log10() } else { f64::NEG_INFINITY };
    let block = rate as usize * 4 / 10;
    let step = rate as usize / 10;
    let mut blocks = vec![];
    let mut a = 0;
    while a + block <= n {
        blocks.push(mean(a, a + block));
        a += step;
    }
    let abs_gated: Vec<f64> = blocks.into_iter().filter(|&m| lufs(m) > -70.0).collect();
    if abs_gated.is_empty() {
        return None;
    }
    let rel = lufs(abs_gated.iter().sum::<f64>() / abs_gated.len() as f64) - 10.0;
    let gated: Vec<f64> = abs_gated.into_iter().filter(|&m| lufs(m) > rel).collect();
    if gated.is_empty() {
        return None;
    }
    Some(lufs(gated.iter().sum::<f64>() / gated.len() as f64) as f32)
}
