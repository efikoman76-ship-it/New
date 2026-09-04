//! Muon/AdamW/sparse-AdamW optimizers, WSD schedule, μP, EMA, ZeRO-3, expert/pipeline/sequence parallelism, checkpoints, halting scheduler.
//!
//! Milestone M0: module scaffold. Implementation lands with the milestone
//! schedule in docs/milestones.md; every public item is documented.

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
