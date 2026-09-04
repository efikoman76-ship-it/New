//! Format/version constants for every serialized artifact in the repo.
//!
//! Bumping a version is a breaking change and requires an ADR.

/// Shard container magic ("RZSH").
pub const SHARD_MAGIC: [u8; 4] = *b"RZSH";
/// Weight container magic ("RZWT").
pub const WEIGHTS_MAGIC: [u8; 4] = *b"RZWT";
/// Plan container magic ("RZPL").
pub const PLAN_MAGIC: [u8; 4] = *b"RZPL";
/// Checkpoint container magic ("RZCK").
pub const CKPT_MAGIC: [u8; 4] = *b"RZCK";
/// Current shard format version.
pub const SHARD_VERSION: u32 = 1;
/// Current weights format version.
pub const WEIGHTS_VERSION: u32 = 1;
/// Current plan format version.
pub const PLAN_VERSION: u32 = 1;
/// Current checkpoint format version.
pub const CKPT_VERSION: u32 = 1;
/// Model schema version (config compatibility).
pub const SCHEMA_VERSION: u32 = 1;

/// Compute the schema hash used to bind shards, weights and plans to one
/// model configuration: BLAKE3 over a canonical rendering of the schema
/// document.
pub fn schema_hash(canonical: &[u8]) -> [u8; 32] {
    crate::hash::blake3(canonical)
}
