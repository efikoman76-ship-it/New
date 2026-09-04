//! Shape math: dimensions, row-major strides, broadcast-free indexing.

use crate::RhizomeError;

/// A tensor shape: dimension sizes in row-major (C) order.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Construct from dimensions; errors if any dimension is zero (empty
    /// tensors are not part of the op vocabulary's domain).
    pub fn from_dims(dims: impl IntoIterator<Item = usize>) -> Result<Self, RhizomeError> {
        let dims: Vec<usize> = dims.into_iter().collect();
        if dims.contains(&0) {
            return Err(RhizomeError::Shape(format!(
                "zero-sized dimension in {dims:?}"
            )));
        }
        Ok(Shape { dims })
    }

    /// Construct without validation. Internal helper for shapes already known
    /// to be non-zero by construction (e.g. sliced from an existing shape).
    pub fn from_dims_unchecked(dims: Vec<usize>) -> Self {
        Shape { dims }
    }

    /// The dimension sizes.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Rank (number of dimensions).
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Total number of elements (1 for scalars).
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    /// Row-major strides in elements (last stride is 1).
    pub fn strides(&self) -> Vec<usize> {
        let mut s = vec![1usize; self.dims.len()];
        for i in (0..self.dims.len().saturating_sub(1)).rev() {
            s[i] = s[i + 1] * self.dims[i + 1];
        }
        s
    }

    /// Flatten a multi-index to a row-major offset; errors if out of range.
    pub fn offset(&self, idx: &[usize]) -> Result<usize, RhizomeError> {
        if idx.len() != self.dims.len() {
            return Err(RhizomeError::Shape(format!(
                "index rank {} != shape rank {}",
                idx.len(),
                self.dims.len()
            )));
        }
        let strides = self.strides();
        let mut off = 0usize;
        for ((i, d), s) in idx.iter().zip(self.dims.iter()).zip(strides.iter()) {
            if *i >= *d {
                return Err(RhizomeError::Index(format!("index {i} out of dim {d}")));
            }
            off += i * s;
        }
        Ok(off)
    }

    /// True when both shapes are equal.
    pub fn same(&self, other: &Shape) -> bool {
        self.dims == other.dims
    }

    /// Prepend `n` size-1 dimensions.
    pub fn unsqueeze_front(&self, n: usize) -> Shape {
        let mut dims = vec![1usize; n];
        dims.extend_from_slice(&self.dims);
        Shape { dims }
    }

    /// Remove leading size-1 dimensions until rank `r` is reached.
    pub fn squeeze_to(&self, r: usize) -> Result<Shape, RhizomeError> {
        if self.rank() < r {
            return Err(RhizomeError::Shape(format!(
                "rank {} < target {r}",
                self.rank()
            )));
        }
        let drop = self.rank() - r;
        if self.dims[..drop].iter().any(|d| *d != 1) {
            return Err(RhizomeError::Shape(
                "cannot squeeze non-unit leading dims".into(),
            ));
        }
        Ok(Shape {
            dims: self.dims[drop..].to_vec(),
        })
    }

    /// Matrix shape helpers: (rows, cols) for rank-2 shapes.
    pub fn mat2(&self) -> Result<(usize, usize), RhizomeError> {
        if self.rank() != 2 {
            return Err(RhizomeError::Shape(format!(
                "expected rank 2, got {}",
                self.rank()
            )));
        }
        Ok((self.dims[0], self.dims[1]))
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "]")
    }
}

/// Compute the contraction `C[m,n] += sum_k A[m,k] * B[k,n]` shape, validating
/// inner dimensions.
pub fn matmul_shape(a: &Shape, b: &Shape) -> Result<Shape, RhizomeError> {
    let (am, ak) = a.mat2()?;
    let (bk, bn) = b.mat2()?;
    if ak != bk {
        return Err(RhizomeError::Shape(format!(
            "matmul inner mismatch: {ak} vs {bk}"
        )));
    }
    Shape::from_dims([am, bn])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strides_and_offsets() {
        let s = Shape::from_dims([2, 3, 4]).unwrap();
        assert_eq!(s.numel(), 24);
        assert_eq!(s.strides(), vec![12, 4, 1]);
        assert_eq!(s.offset(&[1, 2, 3]).unwrap(), 12 + 8 + 3);
        assert!(s.offset(&[2, 0, 0]).is_err());
        assert!(s.offset(&[0, 0]).is_err());
    }

    #[test]
    fn zero_dim_rejected() {
        assert!(Shape::from_dims([2, 0, 3]).is_err());
        assert!(Shape::from_dims(Vec::<usize>::new()).is_ok());
    }

    #[test]
    fn matmul() {
        let a = Shape::from_dims([3, 4]).unwrap();
        let b = Shape::from_dims([4, 5]).unwrap();
        assert_eq!(matmul_shape(&a, &b).unwrap().dims(), [3, 5]);
        let c = Shape::from_dims([5, 5]).unwrap();
        assert!(matmul_shape(&a, &c).is_err());
    }
}
