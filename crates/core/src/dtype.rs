//! Dtypes and bit-exact low-precision formats.
//!
//! Implements bf16 (round-to-nearest-even), FP8 e4m3 (OCP "fn" flavor: finite,
//! no infinities), and the MX block formats (MXFP4 e2m1 codes and MXINT4 codes
//! with e8m0 shared scales over 32-element blocks). All conversions are
//! bit-exact and independent of build flags or hardware so the CPU path can
//! emulate the accelerator numerics exactly (SPEC 3.9).

use crate::RhizomeError;

/// Scalar element types used by tensors and kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    /// Double precision (reference path).
    F64,
    /// Single precision (CPU optimized path).
    F32,
    /// Brain float 16, stored as raw `u16` bit patterns.
    Bf16,
    /// OCP FP8 e4m3 (finite), stored as raw `u8` bit patterns.
    Fp8E4M3,
    /// Unsigned 8-bit (PKM slot ids, byte data).
    U8,
    /// Unsigned 32-bit (token ids, segment ids, plan node ids).
    U32,
    /// Signed 64-bit (counters, sample indices).
    I64,
    /// Boolean, one byte each (0 or 1).
    Bool,
}

impl Dtype {
    /// Size in bytes of one element on the wire.
    pub fn size_bytes(self) -> usize {
        match self {
            Dtype::F64 => 8,
            Dtype::F32 | Dtype::U32 => 4,
            Dtype::Bf16 => 2,
            Dtype::Fp8E4M3 | Dtype::U8 | Dtype::Bool => 1,
            Dtype::I64 => 8,
        }
    }

    /// Short lowercase name used in plans and reports.
    pub fn name(self) -> &'static str {
        match self {
            Dtype::F64 => "f64",
            Dtype::F32 => "f32",
            Dtype::Bf16 => "bf16",
            Dtype::Fp8E4M3 => "fp8e4m3",
            Dtype::U8 => "u8",
            Dtype::U32 => "u32",
            Dtype::I64 => "i64",
            Dtype::Bool => "bool",
        }
    }

    /// True when the dtype holds floating-point values.
    pub fn is_float(self) -> bool {
        matches!(self, Dtype::F64 | Dtype::F32 | Dtype::Bf16 | Dtype::Fp8E4M3)
    }
}

impl std::fmt::Display for Dtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ---------------------------------------------------------------------------
// bf16
// ---------------------------------------------------------------------------

/// Convert an `f32` to a bf16 bit pattern with round-to-nearest-even.
pub fn f32_to_bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        // Keep NaN: truncate payload, force the quiet bit.
        return ((bits >> 16) as u16) | 0x0040;
    }
    // Round to nearest, ties to even, on the truncated 16 high bits.
    let upper = bits >> 16;
    let lower = bits & 0xffff;
    let rounded = upper
        + match (lower, upper & 1) {
            (0x8000, 1) => 1, // exact tie, odd -> round up to even
            (0x8000, 0) => 0, // exact tie, even -> stay
            (l, _) if l > 0x8000 => 1,
            _ => 0,
        };
    rounded as u16
}

/// Convert a bf16 bit pattern to `f32` (exact).
pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Convenience: `f32` → bf16 value widened back to `f32` (quantize round-trip).
pub fn round_bf16(x: f32) -> f32 {
    bf16_bits_to_f32(f32_to_bf16_bits(x))
}

// ---------------------------------------------------------------------------
// FP8 e4m3 (OCP "fn": no inf, NaN = 0x7f/0xff, bias 7, max finite 448)
// ---------------------------------------------------------------------------

/// FP8 e4m3 exponent bias.
pub const FP8_BIAS: i32 = 7;
/// Smallest FP8 e4m3 subnormal step (2^(1-bias-3)).
pub const FP8_SUBNORMAL_STEP: f32 = 0.001953125; // 2^-9
/// Largest finite FP8 e4m3 magnitude.
pub const FP8_MAX: f32 = 448.0;

/// Convert `f32` to an FP8 e4m3 bit pattern, round-to-nearest-even on the
/// exponent-local representable grid, saturating above 464 to ±448
/// (documented in docs/kernels.md).
pub fn f32_to_fp8_e4m3_bits(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7f;
    }
    let sign = if x.is_sign_negative() { 1u8 << 7 } else { 0u8 };
    let a = x.abs();
    if a == 0.0 {
        return sign;
    }
    if a > 464.0 {
        return sign | 0x7e; // exp field 15, mantissa 7 => 448
    }
    // Subnormal region: values below 2^(1-BIAS) = 2^-6 live on a uniform grid
    // with step 2^-9.
    if a < 2.0f32.powi(1 - FP8_BIAS) {
        let units = (a / FP8_SUBNORMAL_STEP).round_ties_even();
        let units = units.clamp(0.0, 7.0) as u8;
        return sign | units;
    }
    // Normal region: value = (1 + m/8) * 2^e with m in [0,7]; round the
    // fraction on its 1/8 grid with ties to even m, carrying into e.
    let mut e = a.log2().floor() as i32; // a >= 2^-6 => e >= -6
    if e < 1 - FP8_BIAS {
        e = 1 - FP8_BIAS;
    }
    let m = a * 2.0f32.powi(-e); // in [1, 2)
    let mut q = ((m - 1.0) * 8.0).round_ties_even() as i32;
    if q >= 8 {
        // Carried past 2.0: promote the exponent.
        q = 0;
        e += 1;
    }
    let field = e + FP8_BIAS;
    debug_assert!((1..=15).contains(&field), "field={field} a={a}");
    sign | ((field as u8) << 3) | (q as u8 & 0x07)
}

/// Convert an FP8 e4m3 bit pattern to `f32` (exact).
pub fn fp8_e4m3_bits_to_f32(bits: u8) -> f32 {
    let sign = if bits & 0x80 != 0 { -1.0f32 } else { 1.0f32 };
    let exp = ((bits >> 3) & 0x0f) as i32;
    let mant = (bits & 0x07) as i32;
    if exp == 0x0f && mant == 0x07 {
        return f32::NAN;
    }
    if exp == 0 {
        return sign * (mant as f32) * FP8_SUBNORMAL_STEP;
    }
    sign * (1.0 + mant as f32 / 8.0) * 2.0f32.powi(exp - FP8_BIAS)
}

/// Convenience round-trip: quantize `x` to fp8 and widen back to `f32`.
pub fn round_fp8(x: f32) -> f32 {
    fp8_e4m3_bits_to_f32(f32_to_fp8_e4m3_bits(x))
}

// ---------------------------------------------------------------------------
// MX block formats: e8m0 scales + 4-bit codes, 32-element blocks
// ---------------------------------------------------------------------------

/// MX code kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MxKind {
    /// MXFP4: e2m1 codes {0,.5,1,1.5,2,3,4,6} × sign.
    Fp4,
    /// MXINT4: two's-complement −8..7.
    Int4,
}

impl MxKind {
    /// Name used in serialized plans.
    pub fn name(self) -> &'static str {
        match self {
            MxKind::Fp4 => "mxfp4",
            MxKind::Int4 => "mxint4",
        }
    }
}

/// MX block size (fixed by the OCP MX spec and by SPEC.md).
pub const MX_BLOCK: usize = 32;

/// Decode an e8m0 scale byte to its power-of-two value.
pub fn e8m0_to_f32(se: u8) -> f32 {
    // exponent bias 127 on the e8m0 byte; se==0xFF is reserved (NaN) and never
    // produced by the encoder.
    if se == 0xff {
        return f32::NAN;
    }
    2.0f32.powi((se as i32) - 127)
}

/// Encode a positive power-of-two scale to e8m0. Errors on non-finite or
/// out-of-range scales.
pub fn f32_to_e8m0(scale: f32) -> Result<u8, RhizomeError> {
    if !(scale.is_finite() && scale > 0.0) {
        return Err(RhizomeError::Numerical(format!(
            "e8m0 scale must be finite positive, got {scale}"
        )));
    }
    let e = scale.log2().round();
    if !(e as i32 >= -127 && e as i32 <= 126) {
        return Err(RhizomeError::Numerical(format!(
            "e8m0 scale {scale} out of range"
        )));
    }
    Ok((e as i32 + 127) as u8)
}

/// MXFP4 code grid: (code, value) for the positive half.
pub const FP4_GRID: [(u8, f32); 8] = [
    (0, 0.0),
    (1, 0.5),
    (2, 1.0),
    (3, 1.5),
    (4, 2.0),
    (5, 3.0),
    (6, 4.0),
    (7, 6.0),
];

/// Decode one MXFP4 code nibble (0..15) to f32.
pub fn fp4_code_to_f32(code: u8) -> f32 {
    let sign = if code & 8 != 0 { -1.0f32 } else { 1.0f32 };
    let mant = FP4_GRID[(code & 7) as usize].1;
    sign * mant
}

/// Encode one f32 to an MXFP4 code nibble. Ties (exact midpoints between two
/// grid values) round to the code whose low bit is even, matching the
/// round-to-nearest-even convention used for bf16/fp8.
pub fn f32_to_fp4_code(x: f32) -> u8 {
    let sign = if x.is_sign_negative() { 8u8 } else { 0u8 };
    if x.is_nan() {
        return sign; // NaN maps to ±0; callers guarantee finite inputs.
    }
    let a = x.abs();
    let mut best = 0u8;
    let mut best_key = (f32::INFINITY, 1u8);
    for (code, val) in FP4_GRID.iter() {
        let key = ((a - val).abs(), code & 1);
        if key < best_key {
            best_key = key;
            best = *code;
        }
    }
    sign | best
}

/// Decode one MXINT4 nibble (0..15, two's complement) to f32.
pub fn int4_code_to_f32(code: u8) -> f32 {
    let v = ((code as i8) << 4) >> 4; // sign-extend
    v as f32
}

/// Encode one f32 to an MXINT4 nibble (round-to-nearest-even, saturating to
/// the [-8, 7] code range).
pub fn f32_to_int4_code(x: f32) -> u8 {
    let r = x.round_ties_even().clamp(-8.0, 7.0);
    ((r as i8) & 0x0f) as u8
}

/// Quantize a block of 32 f32 values to (codes packed 2/byte, e8m0 scale).
/// The scale is the smallest power of two with `max|x| / scale <= group_max`
/// per the OCP MX flow.
pub fn quantize_mx_block(kind: MxKind, block: &[f32]) -> Result<(Vec<u8>, u8), RhizomeError> {
    if block.len() != MX_BLOCK {
        return Err(RhizomeError::Shape(format!(
            "MX block must be {MX_BLOCK} elements, got {}",
            block.len()
        )));
    }
    let mut amax = 0.0f32;
    for v in block {
        if !v.is_finite() {
            return Err(RhizomeError::Numerical(
                "non-finite value in MX block".into(),
            ));
        }
        amax = amax.max(v.abs());
    }
    let group_max = match kind {
        MxKind::Fp4 => 6.0,
        MxKind::Int4 => 7.0,
    };
    let scale = if amax == 0.0 {
        1.0
    } else {
        // Smallest power of two so that amax/scale <= group_max.
        let e = (amax / group_max).log2().ceil();
        2.0f32.powi(e as i32)
    };
    let se = f32_to_e8m0(scale)?;
    let inv = 1.0 / scale;
    let mut codes = Vec::with_capacity(MX_BLOCK / 2);
    for i in (0..MX_BLOCK).step_by(2) {
        let (lo, hi) = (block[i], block[i + 1]);
        let (clo, chi) = match kind {
            MxKind::Fp4 => (f32_to_fp4_code(lo * inv), f32_to_fp4_code(hi * inv)),
            MxKind::Int4 => (f32_to_int4_code(lo * inv), f32_to_int4_code(hi * inv)),
        };
        codes.push(clo | (chi << 4));
    }
    Ok((codes, se))
}

/// Dequantize one packed MX block to f32 values.
pub fn dequantize_mx_block(
    kind: MxKind,
    codes: &[u8],
    scale_byte: u8,
    out: &mut [f32],
) -> Result<(), RhizomeError> {
    if out.len() != MX_BLOCK {
        return Err(RhizomeError::Shape(format!(
            "MX block output must be {MX_BLOCK} elements"
        )));
    }
    let scale = e8m0_to_f32(scale_byte);
    for (i, v) in out.iter_mut().enumerate() {
        let byte = codes.get(i / 2).copied().unwrap_or(0);
        let nib = if i % 2 == 0 { byte & 0x0f } else { byte >> 4 };
        *v = match kind {
            MxKind::Fp4 => fp4_code_to_f32(nib),
            MxKind::Int4 => int4_code_to_f32(nib),
        } * scale;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_roundtrip_and_rne() {
        for x in [
            0.0f32,
            1.0,
            -2.0,
            0.5,
            3.140625,
            65536.0,
            2.0f32.powi(-100),
            2.0f32.powi(-127),
        ] {
            let b = f32_to_bf16_bits(x);
            assert_eq!(bf16_bits_to_f32(b), x, "bf16 roundtrip {x}");
        }
        // bf16 steps at [1,2) are 2^-7. Halfway between 1.0 (mantissa 0,
        // even) and 1+2^-7 is 1+2^-8: the tie rounds to 1.0.
        let tie = 1.0f32 + 2.0f32.powi(-8);
        assert_eq!(round_bf16(tie), 1.0);
        // Just past halfway rounds up.
        let past = 1.0f32 + 2.0f32.powi(-8) + 2.0f32.powi(-16);
        assert_eq!(round_bf16(past), 1.0 + 2.0f32.powi(-7));
        // Halfway between 1+2^-7 (mantissa 1, odd) and 1+2^-6 (mantissa 2)
        // ties to the even side: 1+2^-6.
        let tie_odd = 1.0f32 + 2.0f32.powi(-7) + 2.0f32.powi(-8);
        assert_eq!(round_bf16(tie_odd), 1.0 + 2.0f32.powi(-6));
        assert!(bf16_bits_to_f32(f32_to_bf16_bits(f32::NAN)).is_nan());
        // Negative zero keeps its sign.
        assert_eq!(f32_to_bf16_bits(-0.0), 0x8000);
    }

    #[test]
    fn fp8_known_values() {
        assert_eq!(round_fp8(0.0), 0.0);
        assert_eq!(f32_to_fp8_e4m3_bits(-0.0), 0x80);
        assert_eq!(round_fp8(1.0), 1.0);
        assert_eq!(round_fp8(448.0), 448.0);
        assert_eq!(round_fp8(-1.5), -1.5);
        assert_eq!(round_fp8(2.0f32.powi(-6)), 2.0f32.powi(-6));
        assert_eq!(round_fp8(2.0f32.powi(-9)), 2.0f32.powi(-9));
        assert_eq!(round_fp8(3.0 * 2.0f32.powi(-9)), 3.0 * 2.0f32.powi(-9));
        assert_eq!(round_fp8(1000.0), 448.0);
        assert_eq!(round_fp8(-1000.0), -448.0);
        assert!(round_fp8(f32::NAN).is_nan());
        // RNE halfway: 1.0625 between 1.0 (mant 0, even) and 1.125 -> 1.0.
        assert_eq!(round_fp8(1.0625), 1.0);
        // Halfway 1.1875 between 1.125 (mant 1) and 1.25 (mant 2, even) -> 1.25.
        assert_eq!(round_fp8(1.1875), 1.25);
        // Every finite bit pattern re-encodes exactly; NaNs decode NaN.
        for b in 0u8..=255 {
            let v = fp8_e4m3_bits_to_f32(b);
            if b & 0x7f == 0x7f {
                assert!(v.is_nan(), "b={b}");
            } else {
                assert!(v.is_finite(), "b={b} -> {v}");
                assert_eq!(f32_to_fp8_e4m3_bits(v), b, "b={b} v={v}");
            }
        }
        // Monotone non-decreasing over a sweep of positive values.
        let mut prev = 0.0f32;
        let mut x = 0.0f32;
        while x < 500.0 {
            let q = round_fp8(x);
            assert!(q >= prev - 1e-12, "not monotone at {x}");
            prev = q;
            x += 0.03125;
        }
    }

    #[test]
    fn mx_fp4_grid_and_blocks() {
        for (code, val) in FP4_GRID.iter() {
            assert_eq!(f32_to_fp4_code(*val), *code);
            assert_eq!(fp4_code_to_f32(*code), *val);
            assert_eq!(f32_to_fp4_code(-*val), *code | 8);
        }
        // Tie behavior: 2.5 between 2 (code 4, even) and 3 (code 5) -> 2.
        assert_eq!(fp4_code_to_f32(f32_to_fp4_code(2.5)), 2.0);
        // 3.5 between 3 (code 5) and 4 (code 6, even) -> 4.
        assert_eq!(fp4_code_to_f32(f32_to_fp4_code(3.5)), 4.0);
        // 5.0 between 4 (code 6, even) and 6 (code 7) -> 4.
        assert_eq!(fp4_code_to_f32(f32_to_fp4_code(5.0)), 4.0);
        // Saturation beyond 6.
        assert_eq!(fp4_code_to_f32(f32_to_fp4_code(100.0)), 6.0);
        let block: Vec<f32> = (0..32).map(|i| ((i as f32) - 16.0) * 0.03125).collect();
        let (codes, se) = quantize_mx_block(MxKind::Fp4, &block).unwrap();
        let mut out = vec![0.0f32; 32];
        dequantize_mx_block(MxKind::Fp4, &codes, se, &mut out).unwrap();
        let scale = e8m0_to_f32(se);
        for (a, b) in block.iter().zip(out.iter()) {
            // Max grid gap at scale s is s (between 4s and 6s): half-gap error.
            assert!((a - b).abs() <= scale + 1e-9, "a={a} b={b} s={scale}");
        }
    }

    #[test]
    fn mx_int4_roundtrip() {
        for v in -8i8..=7 {
            assert_eq!(int4_code_to_f32(f32_to_int4_code(v as f32)), v as f32);
        }
        let block: Vec<f32> = (0..32).map(|i| ((i as f32) - 16.0) * 0.125).collect();
        let (codes, se) = quantize_mx_block(MxKind::Int4, &block).unwrap();
        let mut out = vec![0.0f32; 32];
        dequantize_mx_block(MxKind::Int4, &codes, se, &mut out).unwrap();
        for (a, b) in block.iter().zip(out.iter()) {
            assert!((a - b).abs() <= 0.5 * e8m0_to_f32(se) + 1e-9, "a={a} b={b}");
        }
    }

    #[test]
    fn e8m0_range() {
        assert_eq!(f32_to_e8m0(1.0).unwrap(), 127);
        assert_eq!(f32_to_e8m0(2.0f32.powi(40)).unwrap(), 167);
        assert_eq!(e8m0_to_f32(127), 1.0);
        assert!(f32_to_e8m0(-1.0).is_err());
        assert!(f32_to_e8m0(0.0).is_err());
        assert!(f32_to_e8m0(f32::NAN).is_err());
    }
}
