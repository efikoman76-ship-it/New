//! Philox 4x32-10 counter-based RNG (Random123 algorithm).
//!
//! Determinism contract (R11): every draw is a pure function of
//! (seed, stream_id, step, index), so results are independent of thread count,
//! rank layout, and evaluation order.

/// Philox4x32 multiplier constants (Random123).
const M0: u32 = 0xD251_1F53;
const M1: u32 = 0xCD9E_8D57;
/// Philox4x32 key-bump constants (golden gamma).
const W0: u32 = 0x9E37_79B9;
const W1: u32 = 0xBB67_AE85;

#[inline]
fn mulhilo(a: u32, b: u32) -> (u32, u32) {
    let p = a as u64 * b as u64;
    ((p >> 32) as u32, p as u32)
}

/// One Philox4x32 round (not the key bump).
#[inline]
fn round(c: [u32; 4], k: [u32; 2]) -> [u32; 4] {
    let (hi0, lo0) = mulhilo(M0, c[0]);
    let (hi1, lo1) = mulhilo(M1, c[2]);
    [hi1 ^ c[1] ^ k[0], lo1, hi0 ^ c[3] ^ k[1], lo0]
}

/// Philox4x32 with 10 rounds and the standard key-bump schedule.
pub fn philox4x32_10(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let mut c = counter;
    let mut k = key;
    for _ in 0..10 {
        c = round(c, k);
        k[0] = k[0].wrapping_add(W0);
        k[1] = k[1].wrapping_add(W1);
    }
    c
}

/// A keyed Philox stream: draws are addressed by (step, index) counters.
///
/// `step` selects the outer counter words (used e.g. per training step or per
/// token position) and `index` the inner word (element index within the step),
/// leaving one spare counter word for sub-indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhiloxStream {
    /// Stream identity, mixed into the key.
    pub stream_id: u64,
    /// Master seed, mixed into the key.
    pub seed: u64,
}

impl PhiloxStream {
    /// Create a stream.
    pub fn new(seed: u64, stream_id: u64) -> Self {
        PhiloxStream { stream_id, seed }
    }

    /// Key words derived from (seed, stream_id) via a splitmix64 finalizer so
    /// nearby seeds decorrelate.
    pub fn key(&self) -> [u32; 2] {
        let z = self
            .seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(self.stream_id);
        let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let z = z ^ (z >> 31);
        [(z >> 32) as u32, z as u32]
    }

    /// Draw 4 words of randomness at (step, index, sub).
    pub fn words(&self, step: u64, index: u64, sub: u32) -> [u32; 4] {
        let counter = [
            (step & 0xFFFF_FFFF) as u32,
            (step >> 32) as u32,
            (index & 0xFFFF_FFFF) as u32,
            ((index >> 32) as u32) ^ sub,
        ];
        philox4x32_10(counter, self.key())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_stream_separated() {
        let a = PhiloxStream::new(42, 1);
        let b = PhiloxStream::new(42, 1);
        let c = PhiloxStream::new(42, 2);
        assert_eq!(a.words(7, 9, 0), b.words(7, 9, 0));
        assert_ne!(a.words(7, 9, 0), c.words(7, 9, 0));
        assert_ne!(a.words(7, 9, 0), a.words(7, 10, 0));
        assert_ne!(a.words(7, 9, 0), a.words(8, 9, 0));
        assert_ne!(a.words(7, 9, 0), a.words(7, 9, 1));
    }

    #[test]
    fn uniform_distribution_sanity() {
        // Mean of uniforms over many draws must be near 0.5 with overwhelming
        // probability (fixed seed => deterministic assertion).
        let s = PhiloxStream::new(1234, 0);
        let n = 200_000u64;
        let mut sum = 0.0f64;
        for i in 0..n {
            let w = s.words(i >> 2, i & 3, 0);
            let u = (w[0] as f64 + 1.0) / 4294967297.0;
            sum += u;
        }
        let mean = sum / n as f64;
        assert!((mean - 0.5).abs() < 0.01, "mean {mean}");
    }

    #[test]
    fn counter_wraparound_distinct() {
        let s = PhiloxStream::new(9, 0);
        // Crossing the 32-bit boundary in `index` must not repeat the low word
        // pattern (the high word changes).
        let lo = s.words(0, 0xFFFF_FFF0, 0);
        let hi = s.words(0, 0x1_0000_0000, 0);
        assert_ne!(lo, hi);
    }
}
