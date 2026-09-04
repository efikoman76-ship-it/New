//! Typed error enum for all library crates (R10).
//!
//! Binaries may use `anyhow`-style context aggregation; libraries return
//! [`RhizomeError`] variants. Every variant is data-carrying so tests and CI
//! can match on the failure class without string matching.

use core::fmt;

/// The single error type crossing crate boundaries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RhizomeError {
    /// A dimension or rank mismatch, with a human-readable context string.
    Shape(String),
    /// A dtype or format mismatch (e.g. bf16 tensor where f64 was required).
    Dtype(String),
    /// An unsupported (op, dtype, device) combination. Hard error at plan time
    /// per R4; never a silent fallback.
    Unsupported(String),
    /// Invalid configuration: field name and reason.
    Config(String),
    /// Serialization/deserialization failure (TOML, JSON, binary).
    Serde(String),
    /// File/system IO failure (message plus OS error string).
    Io(String),
    /// Invalid index, offset, or out-of-bounds access with context.
    Index(String),
    /// Numerical domain failure (NaN/Inf where finite required, log of <=0, …).
    Numerical(String),
    /// Collective/communication failure.
    Collective(String),
    /// Checkpoint/weight corruption (hash mismatch, bad magic, …).
    Integrity(String),
    /// Signature verification failure.
    Signature(String),
    /// Protocol/API misuse (bad request, invalid grammar, …).
    Protocol(String),
    /// Any otherwise-unclassified error with context.
    Other(String),
}

impl fmt::Display for RhizomeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RhizomeError::Shape(m) => write!(f, "shape error: {m}"),
            RhizomeError::Dtype(m) => write!(f, "dtype error: {m}"),
            RhizomeError::Unsupported(m) => write!(f, "unsupported operation: {m}"),
            RhizomeError::Config(m) => write!(f, "configuration error: {m}"),
            RhizomeError::Serde(m) => write!(f, "serialization error: {m}"),
            RhizomeError::Io(m) => write!(f, "io error: {m}"),
            RhizomeError::Index(m) => write!(f, "index error: {m}"),
            RhizomeError::Numerical(m) => write!(f, "numerical error: {m}"),
            RhizomeError::Collective(m) => write!(f, "collective error: {m}"),
            RhizomeError::Integrity(m) => write!(f, "integrity error: {m}"),
            RhizomeError::Signature(m) => write!(f, "signature error: {m}"),
            RhizomeError::Protocol(m) => write!(f, "protocol error: {m}"),
            RhizomeError::Other(m) => write!(f, "error: {m}"),
        }
    }
}

impl std::error::Error for RhizomeError {}

impl RhizomeError {
    /// Build an [`RhizomeError::Io`] from any `std::io::Error`.
    pub fn io(e: std::io::Error) -> Self {
        RhizomeError::Io(e.to_string())
    }
}

impl From<std::io::Error> for RhizomeError {
    fn from(e: std::io::Error) -> Self {
        RhizomeError::io(e)
    }
}

/// Convenience macro for building a [`RhizomeError`] with formatting,
/// mirroring `anyhow::anyhow!` for library code.
#[macro_export]
macro_rules! rerr {
    ($variant:ident, $($arg:tt)*) => {
        $crate::RhizomeError::$variant(format!($($arg)*))
    };
}
