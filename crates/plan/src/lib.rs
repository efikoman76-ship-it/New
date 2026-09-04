//! Graph IR: forward graph from schema, backward derivation, liveness/arena planning, checkpointing, fusion, serialization.
//!
//! Milestone M0: module scaffold. Implementation lands with the milestone
//! schedule in docs/milestones.md; every public item is documented.

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
