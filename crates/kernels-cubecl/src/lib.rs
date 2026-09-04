//! Portable long-tail ops backend (dependency-free implementation of the CubeCL-portable role; ADR-0002).
//!
//! Milestone M0: module scaffold. Implementation lands with the milestone
//! schedule in docs/milestones.md; every public item is documented.

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
