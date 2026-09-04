//! Tensor storage: typed buffers plus MX-format tensors and conversions.
//!
//! Ops in `rhizome-ops` operate on raw slices via `TensorView`-style
//! helpers; this module defines ownership, dtype conversion, and MX
//! (de)quantization over whole tensors.

use crate::dtype::{
    dequantize_mx_block, e8m0_to_f32, f32_to_e8m0, quantize_mx_block, Dtype, MxKind, MX_BLOCK,
};
use crate::error::RhizomeError;
use crate::shape::Shape;

/// Typed, dense tensor storage in row-major order.
#[derive(Debug, Clone, PartialEq)]
pub enum Data {
    /// f64 elements (reference path).
    F64(Vec<f64>),
    /// f32 elements (optimized path).
    F32(Vec<f32>),
    /// bf16 elements as raw bit patterns.
    Bf16(Vec<u16>),
    /// FP8 e4m3 elements as raw bit patterns.
    Fp8(Vec<u8>),
    /// u8 elements.
    U8(Vec<u8>),
    /// u32 elements.
    U32(Vec<u32>),
    /// i64 elements.
    I64(Vec<i64>),
    /// bool elements (one byte each, 0/1).
    Bool(Vec<u8>),
}

impl Data {
    /// Allocate `n` elements of `dtype`, zero-filled.
    pub fn zeros(dtype: Dtype, n: usize) -> Data {
        match dtype {
            Dtype::F64 => Data::F64(vec![0.0; n]),
            Dtype::F32 => Data::F32(vec![0.0; n]),
            Dtype::Bf16 => Data::Bf16(vec![0; n]),
            Dtype::Fp8E4M3 => Data::Fp8(vec![0; n]),
            Dtype::U8 => Data::U8(vec![0; n]),
            Dtype::U32 => Data::U32(vec![0; n]),
            Dtype::I64 => Data::I64(vec![0; n]),
            Dtype::Bool => Data::Bool(vec![0; n]),
        }
    }

    /// The dtype of this storage.
    pub fn dtype(&self) -> Dtype {
        match self {
            Data::F64(_) => Dtype::F64,
            Data::F32(_) => Dtype::F32,
            Data::Bf16(_) => Dtype::Bf16,
            Data::Fp8(_) => Dtype::Fp8E4M3,
            Data::U8(_) => Dtype::U8,
            Data::U32(_) => Dtype::U32,
            Data::I64(_) => Dtype::I64,
            Data::Bool(_) => Dtype::Bool,
        }
    }

    /// Element count.
    pub fn len(&self) -> usize {
        match self {
            Data::F64(v) => v.len(),
            Data::F32(v) => v.len(),
            Data::Bf16(v) => v.len(),
            Data::Fp8(v) => v.len(),
            Data::U8(v) => v.len(),
            Data::U32(v) => v.len(),
            Data::I64(v) => v.len(),
            Data::Bool(v) => v.len(),
        }
    }

    /// True when empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Element `i` widened to f64 (bools → 0/1). Errors on out-of-range.
    pub fn get_f64(&self, i: usize) -> Result<f64, RhizomeError> {
        let oob = || RhizomeError::Index(format!("index {i} out of {}", self.len()));
        Ok(match self {
            Data::F64(v) => *v.get(i).ok_or_else(oob)?,
            Data::F32(v) => *v.get(i).ok_or_else(oob)? as f64,
            Data::Bf16(v) => crate::dtype::bf16_bits_to_f32(*v.get(i).ok_or_else(oob)?) as f64,
            Data::Fp8(v) => crate::dtype::fp8_e4m3_bits_to_f32(*v.get(i).ok_or_else(oob)?) as f64,
            Data::U8(v) => *v.get(i).ok_or_else(oob)? as f64,
            Data::U32(v) => *v.get(i).ok_or_else(oob)? as f64,
            Data::I64(v) => *v.get(i).ok_or_else(oob)? as f64,
            Data::Bool(v) => (*v.get(i).ok_or_else(oob)? != 0) as i32 as f64,
        })
    }

    /// Set element `i` from f64 (with the destination dtype's rounding).
    pub fn set_f64(&mut self, i: usize, x: f64) -> Result<(), RhizomeError> {
        if i >= self.len() {
            return Err(RhizomeError::Index(format!(
                "index {i} out of {}",
                self.len()
            )));
        }
        match self {
            Data::F64(v) => v[i] = x,
            Data::F32(v) => v[i] = x as f32,
            Data::Bf16(v) => v[i] = crate::dtype::f32_to_bf16_bits(x as f32),
            Data::Fp8(v) => v[i] = crate::dtype::f32_to_fp8_e4m3_bits(x as f32),
            Data::U8(v) => v[i] = x as u8,
            Data::U32(v) => v[i] = x as u32,
            Data::I64(v) => v[i] = x as i64,
            Data::Bool(v) => v[i] = (x != 0.0) as u8,
        }
        Ok(())
    }

    /// Convert to f64 storage.
    pub fn to_f64(&self) -> Result<Vec<f64>, RhizomeError> {
        let mut out = Vec::with_capacity(self.len());
        for i in 0..self.len() {
            out.push(self.get_f64(i)?);
        }
        Ok(out)
    }

    /// Convert f64 data to the target dtype storage (rounding per dtype).
    pub fn from_f64(dtype: Dtype, vals: &[f64]) -> Result<Data, RhizomeError> {
        if matches!(dtype, Dtype::U32 | Dtype::U8 | Dtype::I64 | Dtype::Bool)
            && vals.iter().any(|v| v.fract() != 0.0)
        {
            return Err(RhizomeError::Dtype(format!(
                "fractional value cannot be stored in {dtype}"
            )));
        }
        let mut d = Data::zeros(dtype, vals.len());
        for (i, v) in vals.iter().enumerate() {
            d.set_f64(i, *v)?;
        }
        Ok(d)
    }

    /// Borrow as f32 slice; error on dtype mismatch.
    pub fn as_f32(&self) -> Result<&[f32], RhizomeError> {
        match self {
            Data::F32(v) => Ok(v),
            other => Err(RhizomeError::Dtype(format!(
                "want f32, got {}",
                other.dtype()
            ))),
        }
    }

    /// Borrow as mutable f32 slice.
    pub fn as_f32_mut(&mut self) -> Result<&mut [f32], RhizomeError> {
        match self {
            Data::F32(v) => Ok(v),
            other => Err(RhizomeError::Dtype(format!(
                "want f32, got {}",
                other.dtype()
            ))),
        }
    }

    /// Borrow as f64 slice.
    pub fn as_f64(&self) -> Result<&[f64], RhizomeError> {
        match self {
            Data::F64(v) => Ok(v),
            other => Err(RhizomeError::Dtype(format!(
                "want f64, got {}",
                other.dtype()
            ))),
        }
    }

    /// Borrow as u32 slice.
    pub fn as_u32(&self) -> Result<&[u32], RhizomeError> {
        match self {
            Data::U32(v) => Ok(v),
            other => Err(RhizomeError::Dtype(format!(
                "want u32, got {}",
                other.dtype()
            ))),
        }
    }

    /// Borrow as u8 slice.
    pub fn as_u8(&self) -> Result<&[u8], RhizomeError> {
        match self {
            Data::U8(v) => Ok(v),
            other => Err(RhizomeError::Dtype(format!(
                "want u8, got {}",
                other.dtype()
            ))),
        }
    }

    /// Borrow as i64 slice.
    pub fn as_i64(&self) -> Result<&[i64], RhizomeError> {
        match self {
            Data::I64(v) => Ok(v),
            other => Err(RhizomeError::Dtype(format!(
                "want i64, got {}",
                other.dtype()
            ))),
        }
    }

    /// Borrow as bool slice.
    pub fn as_bool(&self) -> Result<&[u8], RhizomeError> {
        match self {
            Data::Bool(v) => Ok(v),
            other => Err(RhizomeError::Dtype(format!(
                "want bool, got {}",
                other.dtype()
            ))),
        }
    }
}

/// A dense tensor: shape + typed data.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    /// Shape.
    pub shape: Shape,
    /// Storage.
    pub data: Data,
}

impl Tensor {
    /// Create from data and shape (validates element count).
    pub fn new(shape: Shape, data: Data) -> Result<Self, RhizomeError> {
        if shape.numel() != data.len() {
            return Err(RhizomeError::Shape(format!(
                "shape {shape} has {} elements but data has {}",
                shape.numel(),
                data.len()
            )));
        }
        Ok(Tensor { shape, data })
    }

    /// Zeros of the given shape and dtype.
    pub fn zeros(dtype: Dtype, shape: Shape) -> Result<Self, RhizomeError> {
        let n = shape.numel();
        Ok(Tensor {
            shape,
            data: Data::zeros(dtype, n),
        })
    }

    /// The dtype.
    pub fn dtype(&self) -> Dtype {
        self.data.dtype()
    }

    /// Total elements.
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Convert all elements to an f64 vector (row-major).
    pub fn to_f64_vec(&self) -> Result<Vec<f64>, RhizomeError> {
        self.data.to_f64()
    }
}

/// An MX-quantized tensor: packed 4-bit codes plus one e8m0 scale per
/// 32-element block, with a row-major element layout (blocks never straddle
/// the last dimension's tail: the last dim is padded implicitly by zero codes
/// when it is not a multiple of 32).
#[derive(Debug, Clone, PartialEq)]
pub struct MxTensor {
    /// Kind (fp4 / int4).
    pub kind: MxKind,
    /// Logical shape.
    pub shape: Shape,
    /// Packed codes: 2 per byte, `ceil(last_dim/32)*32` codes per row run.
    pub codes: Vec<u8>,
    /// One e8m0 byte per 32-element block.
    pub scales: Vec<u8>,
    /// Padded length of the fastest dimension (multiple of 32).
    pub padded_last: usize,
}

impl MxTensor {
    /// Quantize a row-major f32 tensor with 32-element blocks along the last
    /// dimension.
    pub fn quantize(kind: MxKind, t: &Tensor) -> Result<Self, RhizomeError> {
        if t.dtype() != Dtype::F32 {
            return Err(RhizomeError::Dtype(format!(
                "MX quantize needs f32, got {}",
                t.dtype()
            )));
        }
        let dims = t.shape.dims();
        if dims.is_empty() {
            return Err(RhizomeError::Shape("MX quantize needs rank >= 1".into()));
        }
        let last = *dims
            .last()
            .ok_or_else(|| RhizomeError::Shape("no dims".into()))?;
        let rows: usize = dims[..dims.len() - 1].iter().product::<usize>().max(1);
        let padded_last = last.div_ceil(MX_BLOCK) * MX_BLOCK;
        let src = t.data.as_f32()?;
        let mut codes = Vec::with_capacity(rows * padded_last / 2);
        let mut scales = Vec::with_capacity(rows * padded_last / MX_BLOCK);
        let mut block = vec![0.0f32; MX_BLOCK];
        for r in 0..rows {
            let row = &src[r * last..(r + 1) * last];
            for b in 0..padded_last / MX_BLOCK {
                let start = b * MX_BLOCK;
                for (j, v) in block.iter_mut().enumerate() {
                    let idx = start + j;
                    *v = if idx < last { row[idx] } else { 0.0 };
                }
                let (c, s) = quantize_mx_block(kind, &block)?;
                codes.extend_from_slice(&c);
                scales.push(s);
            }
        }
        Ok(MxTensor {
            kind,
            shape: t.shape.clone(),
            codes,
            scales,
            padded_last,
        })
    }

    /// Dequantize to an f32 dense tensor.
    pub fn dequantize(&self) -> Result<Tensor, RhizomeError> {
        let dims = self.shape.dims();
        let last = *dims
            .last()
            .ok_or_else(|| RhizomeError::Shape("no dims".into()))?;
        let rows: usize = dims[..dims.len() - 1].iter().product::<usize>().max(1);
        let mut out = vec![0.0f32; rows * last];
        let mut block = [0.0f32; MX_BLOCK];
        for r in 0..rows {
            for b in 0..self.padded_last / MX_BLOCK {
                let bi = r * (self.padded_last / MX_BLOCK) + b;
                let cb = &self.codes[bi * MX_BLOCK / 2..(bi + 1) * MX_BLOCK / 2];
                dequantize_mx_block(self.kind, cb, self.scales[bi], &mut block)?;
                for (j, v) in block.iter().enumerate() {
                    let idx = b * MX_BLOCK + j;
                    if idx < last {
                        out[r * last + idx] = *v;
                    }
                }
            }
        }
        Tensor::new(self.shape.clone(), Data::F32(out))
    }

    /// Dequantize one block's scale (diagnostics).
    pub fn block_scale(&self, block_index: usize) -> Result<f32, RhizomeError> {
        self.scales
            .get(block_index)
            .map(|&s| e8m0_to_f32(s))
            .ok_or_else(|| RhizomeError::Index(format!("no block {block_index}")))
    }

    /// Bytes of payload (codes + scales).
    pub fn payload_bytes(&self) -> usize {
        self.codes.len() + self.scales.len()
    }
}

/// Encode a scale-tensor pair from f32 scales (helper for FP8 block formats:
/// e4m3 codes with 1×128-row / 128×128-block e8m0 scales are handled by the
/// quant crate; here we expose the shared e8m0 conversion).
pub fn scale_to_e8m0(scale: f32) -> Result<u8, RhizomeError> {
    f32_to_e8m0(scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::Shape;

    #[test]
    fn data_conversions() {
        let mut d = Data::zeros(Dtype::Bf16, 3);
        d.set_f64(0, 1.5).unwrap();
        assert_eq!(d.get_f64(0).unwrap(), 1.5);
        let d2 = Data::from_f64(Dtype::F32, &[0.25, -0.5]).unwrap();
        assert_eq!(d2.as_f32().unwrap(), &[0.25, -0.5]);
        assert!(Data::from_f64(Dtype::U32, &[0.5]).is_err());
        let f8 = Data::from_f64(Dtype::Fp8E4M3, &[0.0, 448.0]).unwrap();
        assert_eq!(f8.get_f64(1).unwrap(), 448.0);
    }

    #[test]
    fn tensor_roundtrip_and_errors() {
        let t = Tensor::zeros(Dtype::F64, Shape::from_dims([2, 3]).unwrap()).unwrap();
        assert_eq!(t.numel(), 6);
        assert!(Tensor::new(
            Shape::from_dims([2, 3]).unwrap(),
            Data::zeros(Dtype::F32, 5)
        )
        .is_err());
    }

    #[test]
    fn mx_tensor_quantize_dequantize() {
        let shape = Shape::from_dims([4, 40]).unwrap(); // last dim not a multiple of 32
        let vals: Vec<f32> = (0..160).map(|i| (i as f32) * 0.125 - 10.0).collect();
        let t = Tensor::new(shape.clone(), Data::F32(vals.clone())).unwrap();
        let scale_of_block = 2.0f32; // amax 10 -> 2^ceil(log2(10/6)) = 2
        for kind in [MxKind::Fp4, MxKind::Int4] {
            let q = MxTensor::quantize(kind, &t).unwrap();
            assert_eq!(q.padded_last, 64);
            let dq = q.dequantize().unwrap();
            let out = dq.data.as_f32().unwrap();
            // Worst-case quantization error: 1x block scale for fp4 (the
            // 4..6 grid gap), 0.5x for int4.
            let bound = if kind == MxKind::Fp4 { 1.0 } else { 0.5 };
            for (a, b) in vals.iter().zip(out.iter()) {
                assert!(
                    (a - b).abs() <= bound * scale_of_block + 1e-6,
                    "kind={kind:?} a={a} b={b}"
                );
            }
        }
    }

    #[test]
    fn mx_zero_block_scale_is_one() {
        let t = Tensor::new(Shape::from_dims([1, 32]).unwrap(), Data::F32(vec![0.0; 32])).unwrap();
        let q = MxTensor::quantize(MxKind::Fp4, &t).unwrap();
        assert_eq!(q.scales[0], 127);
    }
}
