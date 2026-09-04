//! Distributions on top of [`PhiloxStream`]: uniform floats/ints, normal
//! (Box–Muller with fixed pairing), truncated log-normal for Think-Core depth
//! sampling (SPEC 3.5), and shuffle/sample utilities. All deterministic in
//! (seed, stream, step, index).

use crate::philox::PhiloxStream;

/// Distribution samplers bound to one keyed stream.
#[derive(Debug, Clone, Copy)]
pub struct Rng {
    stream: PhiloxStream,
    /// Monotonic draw counter; every sampler consumes whole words.
    cursor: u64,
}

impl Rng {
    /// New RNG over (seed, stream_id).
    pub fn new(seed: u64, stream_id: u64) -> Self {
        Rng {
            stream: PhiloxStream::new(seed, stream_id),
            cursor: 0,
        }
    }

    /// Fork into a different stream id while keeping the cursor position
    /// class (used to keep subsystems independent).
    pub fn fork(&self, stream_id: u64) -> Rng {
        Rng {
            stream: PhiloxStream::new(self.stream.seed, stream_id),
            cursor: self.cursor,
        }
    }

    /// Explicitly address a draw without advancing the cursor.
    pub fn at(&self, step: u64, index: u64) -> [u32; 4] {
        self.stream.words(step, index, 0)
    }

    fn next_words(&mut self) -> [u32; 4] {
        let c = self.cursor;
        self.cursor = self.cursor.wrapping_add(1);
        self.stream.words(0, c, 0)
    }

    /// Uniform f64 in [0, 1).
    pub fn uniform_f64(&mut self) -> f64 {
        let w = self.next_words();
        ((w[0] >> 5) as f64 * 67108864.0 + (w[1] >> 6) as f64) / 9007199254740992.0
    }

    /// Uniform f32 in [0, 1).
    pub fn uniform_f32(&mut self) -> f32 {
        let w = self.next_words();
        (w[0] >> 8) as f32 / 16777216.0
    }

    /// Uniform f64 in [lo, hi).
    pub fn uniform_range_f64(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform_f64()
    }

    /// Uniform integer in [0, n) via Lemire rejection on the 32-bit word
    /// (unbiased, deterministic: rejected draws consume further Philox words).
    pub fn uniform_int(&mut self, n: u32) -> u32 {
        debug_assert!(n > 0);
        if n == 1 {
            return 0;
        }
        let zone = n.wrapping_neg() % n;
        loop {
            let w = self.next_words()[0] as u64;
            let m = w * n as u64;
            if (m as u32) >= zone {
                return (m >> 32) as u32;
            }
        }
    }

    /// Standard normal via Box–Muller; pairs (u1, u2) consumed in fixed order.
    pub fn normal_f64(&mut self) -> f64 {
        let w = self.next_words();
        let u1 = (((w[0] >> 5) as f64 * 67108864.0 + (w[1] >> 6) as f64) / 9007199254740992.0)
            .max(f64::MIN_POSITIVE);
        let u2 = ((w[2] >> 5) as f64 * 67108864.0 + (w[3] >> 6) as f64) / 9007199254740992.0;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Standard normal, f32 flavor.
    pub fn normal_f32(&mut self) -> f32 {
        self.normal_f64() as f32
    }

    // INVARIANT: coefficients are the full-precision AS241 constants
    // (Wichura 1988); truncation would degrade the 1e-15 guarantee.
    #[allow(clippy::excessive_precision)]
    /// Inverse standard normal CDF (Wichura AS241, as in R's `qnorm`),
    /// |error| ≈ 1e-15.
    pub fn normal_inv_cdf(p: f64) -> f64 {
        let q = p - 0.5;
        if q.abs() <= 0.425 {
            let r = 0.180625 - q * q;
            let num = poly(
                r,
                &[
                    2509.0809287301226727,
                    33430.575583588128105,
                    67265.770927008700853,
                    45921.953931549871457,
                    13731.693765509461125,
                    1971.5909503065514427,
                    133.14166789178437745,
                    3.387132872796366608,
                ],
            ) * q;
            let den = poly(
                r,
                &[
                    5226.495278852854561,
                    28729.085735721942674,
                    39307.89580009271061,
                    21213.794301586595867,
                    5394.1960214247511077,
                    687.1870074920579083,
                    42.313330701600911252,
                    1.0,
                ],
            );
            return num / den;
        }
        let tail = if q < 0.0 { p } else { 1.0 - p };
        let r = (-(tail.ln())).min(1e300).sqrt();
        let (num, den) = if r <= 5.0 {
            let r = r - 1.6;
            (
                poly(
                    r,
                    &[
                        7.7454501427834140764e-4,
                        0.0227238449892691845833,
                        0.24178072517745061177,
                        1.27045825245236838258,
                        3.64784832476320460504,
                        5.7694972214606914055,
                        4.6303378461565452959,
                        1.42343711074968357734,
                    ],
                ),
                poly(
                    r,
                    &[
                        1.05075007164441684324e-9,
                        5.475938084995344946e-4,
                        0.0151986665636164571966,
                        0.14810397642748007459,
                        0.68976733498510000455,
                        1.6763848301838038494,
                        2.05319162663775882187,
                        1.0,
                    ],
                ),
            )
        } else if r <= 27.0 {
            let r = r - 5.0;
            (
                poly(
                    r,
                    &[
                        2.01033439929228813265e-7,
                        2.71155556874348757815e-5,
                        0.0012426609473880784386,
                        0.026532189526576123093,
                        0.29656057182850489123,
                        1.7848265399172913358,
                        5.4637849111641143699,
                        6.6579046435011037772,
                    ],
                ),
                poly(
                    r,
                    &[
                        2.04426310338993978564e-15,
                        1.4215117583164458887e-7,
                        1.8463183175100546818e-5,
                        7.868691311456132591e-4,
                        0.0148753612908506148525,
                        0.13692988092273580531,
                        0.59983220655588793769,
                        1.0,
                    ],
                ),
            )
        } else {
            // Extreme tail: 0-th order asymptotic qn = r * sqrt(2)
            // (R's Maechler asymptotic; our samplers never reach r > 27 in
            // f64 practice without p being denormal).
            return if q < 0.0 {
                -r * std::f64::consts::SQRT_2
            } else {
                r * std::f64::consts::SQRT_2
            };
        };
        let val = num / den;
        if q < 0.0 {
            -val
        } else {
            val
        }
    }

    /// Truncated normal on [lo, hi] via inverse CDF (deterministic, exact
    /// bounds handling).
    pub fn truncated_normal(&mut self, lo: f64, hi: f64) -> f64 {
        let phi_lo = Phi::of(lo);
        let phi_hi = Phi::of(hi);
        let u = phi_lo + (phi_hi - phi_lo) * self.uniform_f64();
        Self::normal_inv_cdf(u.clamp(1e-300, 1.0 - 1e-16))
    }

    /// Sample L = round(clamp(TruncLogNormal(mu, sigma), 1, l_max)) with the
    /// lognormal truncated to [1, l_max] (SPEC 3.5 training schedule). Because
    /// the output is integral, truncation-then-rounding equals clamping.
    pub fn sample_depth(&mut self, mu: f64, sigma: f64, l_max: u32) -> u32 {
        let z = self.normal_f64();
        let v = (mu + sigma * z).exp();
        let l = v.round() as i64;
        l.clamp(1, l_max as i64) as u32
    }

    /// Bernoulli(p).
    pub fn bernoulli(&mut self, p: f64) -> bool {
        self.uniform_f64() < p
    }

    /// In-place Fisher–Yates shuffle with stream-indexed draws.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.uniform_int((i + 1) as u32) as usize;
            items.swap(i, j);
        }
    }

    /// Sample `k` distinct indices from [0, n) (partial Fisher–Yates over an
    /// index scratch space).
    pub fn sample_indices(&mut self, n: usize, k: usize, scratch: &mut Vec<usize>) -> Vec<usize> {
        scratch.clear();
        scratch.extend(0..n);
        let k = k.min(n);
        let mut out = Vec::with_capacity(k);
        for i in 0..k {
            let j = i + self.uniform_int((n - i) as u32) as usize;
            scratch.swap(i, j);
            out.push(scratch[i]);
        }
        out
    }
}

/// Standard normal CDF via erf (Abramowitz–Stegun 7.1.26 is not accurate
/// enough for tight bounds; use the Zelen & Severo rational form, |err|<7.5e-8).
pub struct Phi;

impl Phi {
    /// Φ(x).
    pub fn of(x: f64) -> f64 {
        0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
    }
}

/// erf via the Zelen & Severo rational approximation (|ε| < 7.5e-8), plus the
/// exact erf identity for the tails; adequate for CDF bounds in truncated
/// sampling (the quantile function is the accurate direction we rely on).
pub fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.5 * x);
    let tau = t
        * (-x * x - 1.26551223
            + t * (1.00002368
                + t * (0.37409196
                    + t * (0.09678418
                        + t * (-0.18628806
                            + t * (0.27886807
                                + t * (-1.13520398
                                    + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
            .exp();
    sign * (1.0 - tau)
}

/// Horner evaluation of a polynomial with coefficients in descending degree.
fn poly(x: f64, coeffs: &[f64]) -> f64 {
    let mut acc = 0.0f64;
    for c in coeffs {
        acc = acc * x + c;
    }
    acc
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_inv_cdf_known_quantiles() {
        // Known values from statistical tables.
        let cases = [
            (0.5f64, 0.0f64),
            (0.841344746068543, 1.0),
            (0.977249868051821, 2.0),
            (0.158655253931457, -1.0),
            (0.00003167124, -4.0), // approximation; |err| tolerance below
        ];
        for (p, x) in cases {
            let q = Rng::normal_inv_cdf(p);
            assert!((q - x).abs() < 2e-4, "p={p} q={q} x={x}");
        }
        for (p, x) in [
            (0.02275013194817921, -2.0),
            (0.9772498680518208, 2.0),
            (0.0013498980316300945, -3.0),
            (1e-12, -7.034483825301131),
        ] {
            let q = Rng::normal_inv_cdf(p);
            assert!((q - x).abs() < 2e-9, "p={p} q={q} x={x}");
        }
        // High-accuracy center checks (AS241 guarantees ~1e-15).
        for (p, x) in [(0.6, 0.2533471031358), (0.4, -0.2533471031358)] {
            assert!((Rng::normal_inv_cdf(p) - x).abs() < 1e-12);
        }
    }

    #[test]
    fn phi_erf_consistency() {
        assert!((Phi::of(0.0) - 0.5).abs() < 1e-7);
        assert!((Phi::of(1.96) - 0.9750021048517795).abs() < 1e-6);
        assert!((erf(0.0)).abs() < 1e-6, "erf(0)={}", erf(0.0));
        assert!((erf(1.0) - 0.8427007929497149).abs() < 1e-6);
    }

    #[test]
    fn depth_sampling_bounds_and_determinism() {
        let mut a = Rng::new(7, 3);
        let mut b = Rng::new(7, 3);
        for _ in 0..1000 {
            let (la, lb) = (
                a.sample_depth(2.0f64.ln(), 0.5, 8),
                b.sample_depth(2.0f64.ln(), 0.5, 8),
            );
            assert_eq!(la, lb);
            assert!((1..=8).contains(&la), "la={la}");
        }
        // Median depth should be 2 (mu = ln 2 in log space): with sigma 0.5,
        // most mass falls in {1,2,3}.
        let mut r = Rng::new(11, 0);
        let mut count_le3 = 0;
        for _ in 0..1000 {
            if r.sample_depth(2.0f64.ln(), 0.5, 8) <= 3 {
                count_le3 += 1;
            }
        }
        assert!(count_le3 > 800, "count_le3={count_le3}");
    }

    #[test]
    fn uniform_int_unbiased_and_bounded() {
        let mut r = Rng::new(5, 9);
        let n = 5u32;
        let mut counts = [0usize; 5];
        for _ in 0..50_000 {
            let v = r.uniform_int(n);
            assert!(v < n);
            counts[v as usize] += 1;
        }
        for c in counts {
            assert!(c > 9_000 && c < 11_000, "counts={counts:?}");
        }
    }

    #[test]
    fn shuffle_deterministic() {
        let mut a: Vec<u32> = (0..100).collect();
        let mut b: Vec<u32> = (0..100).collect();
        let mut r1 = Rng::new(3, 1);
        let mut r2 = Rng::new(3, 1);
        r1.shuffle(&mut a);
        r2.shuffle(&mut b);
        assert_eq!(a, b);
        // A permutation actually happened.
        let same = a
            .iter()
            .zip(0..100)
            .filter(|(x, i)| **x as usize == *i)
            .count();
        assert!(same < 10, "identity-like shuffle: {same} fixed");
    }

    #[test]
    fn normal_moments() {
        let mut r = Rng::new(21, 4);
        let n = 100_000;
        let mut s = 0.0;
        let mut s2 = 0.0;
        for _ in 0..n {
            let x = r.normal_f64();
            s += x;
            s2 += x * x;
        }
        let mean = s / n as f64;
        let var = s2 / n as f64 - mean * mean;
        assert!(mean.abs() < 0.02, "mean {mean}");
        assert!((var - 1.0).abs() < 0.05, "var {var}");
    }
}
