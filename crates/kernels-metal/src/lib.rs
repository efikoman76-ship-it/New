//! Metal backend (feature `metal`).
//!
//! M0 scope: MSL kernel sources under `src/kernels/*.metal` are
//! compile-checked in CI (`xcrun -sdk macosx metal -c`); runtime dispatch
//! lands with M12.

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

/// Kinds of MSL kernels shipped in this crate.
pub mod kernel_names {
    /// RMSNorm over the last dimension (fp32, eps 1e-6).
    pub const RMSNORM_F32: &str = "rmsnorm_f32";
    /// SiLU elementwise.
    pub const SILU_F32: &str = "silu_f32";
    /// Decode GEMV (fp32).
    pub const GEMV_F32: &str = "gemv_f32";
}

/// True when the backend can be used in this build.
pub fn available() -> bool {
    cfg!(feature = "metal")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_stable() {
        assert_eq!(kernel_names::RMSNORM_F32, "rmsnorm_f32");
    }
}
