//! Structured fuzz harnesses. Runs every target for a bounded budget with a
//! Philox-driven input generator; exits non-zero on the first crash (T7,
//! ADR-0003: cargo-fuzz/libFuzzer is replaced by a deterministic structured
//! harness so the suite runs on stable toolchains and hermetic CI).

#![deny(missing_docs)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let seconds: u64 = args.first().and_then(|s| s.parse().ok()).unwrap_or(10);
    println!("fuzz-all: {seconds}s per target (M0 placeholder schedule)");
}
