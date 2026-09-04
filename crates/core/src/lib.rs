//! Foundation crate shared by every RHIZOME component.
//!
//! Contains: typed errors, dtypes and block-quantization formats, shape math,
//! tensor descriptors, the counter-based Philox RNG (R11), TOML/JSON subsets
//! used for configuration and serialization, hashing (BLAKE3 / SHA-2), and the
//! versioned model/training/serving configuration schema with parameter
//! accounting (`crate::params`).
//!
//! This crate is `#![forbid(unsafe_code)]` and builds without any external
//! dependency (ADR-0001).

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod config;
pub mod dtype;
pub mod error;
pub mod hash;
pub mod json;
pub mod params;
pub mod philox;
pub mod rng;
pub mod shape;
pub mod tensor;
pub mod toml;
pub mod version;

/// Re-exported convenience: the crate error type.
pub use error::RhizomeError;

/// Crate-wide Result alias.
pub type RhizomeResult<T> = Result<T, RhizomeError>;
