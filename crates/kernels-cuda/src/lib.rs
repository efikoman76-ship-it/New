//! CUDA backend (feature `cuda`).
//!
//! M0 scope: the kernel sources under `src/kernels/*.cu` are compiled by
//! `build.rs` (nvcc → PTX + cubin for sm_80/sm_90) whenever the `cuda`
//! feature is enabled, and their content hashes are exposed for plan-time
//! backend selection. Runtime launch FFI lands with M11 (ADR-0001 keeps the
//! FFI surface narrow and feature-gated).

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use rhizome_core::hash::blake3;
use std::path::PathBuf;

/// Description of one compiled kernel artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelArtifact {
    /// Source name (e.g. `rmsnorm.cu`).
    pub source: String,
    /// Architecture (e.g. `sm_80`).
    pub arch: String,
    /// Kind: `ptx` or `cubin`.
    pub kind: String,
    /// Path to the artifact produced by build.rs.
    pub path: PathBuf,
    /// BLAKE3 of the artifact bytes.
    pub hash: [u8; 32],
}

/// Compiled artifacts injected by build.rs (`RHIZOME_CUDA_KERNELS` points at
/// a manifest file listing one `kind arch name path` per line).
pub fn artifacts() -> Result<Vec<KernelArtifact>, rhizome_core::RhizomeError> {
    let manifest = option_env!("RHIZOME_CUDA_KERNELS").ok_or_else(|| {
        rhizome_core::RhizomeError::Unsupported("cuda feature not enabled".into())
    })?;
    let mut out = Vec::new();
    for line in manifest.lines() {
        let mut parts = line.split(' ');
        let (Some(kind), Some(arch), Some(source), Some(path)) =
            (parts.next(), parts.next(), parts.next(), parts.next())
        else {
            continue;
        };
        let bytes = std::fs::read(path)
            .map_err(|e| rhizome_core::RhizomeError::Io(format!("read kernel artifact: {e}")))?;
        out.push(KernelArtifact {
            source: source.to_string(),
            arch: arch.to_string(),
            kind: kind.to_string(),
            path: PathBuf::from(path),
            hash: blake3(&bytes),
        });
    }
    Ok(out)
}

/// True when the backend is available in this build.
pub fn available() -> bool {
    option_env!("RHIZOME_CUDA_KERNELS").is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unavailable_without_feature() {
        // In CI's default (feature-less) build the backend reports itself
        // unavailable instead of pretending; the cuda-compile job builds
        // with the feature and runs the same test with it present.
        if available() {
            let arts = artifacts();
            assert!(arts.is_ok(), "{arts:?}");
            for a in arts.unwrap() {
                assert!(a.path.exists(), "missing artifact {a:?}");
            }
        } else {
            assert!(artifacts().is_err());
        }
    }
}
