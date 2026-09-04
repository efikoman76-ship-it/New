//! Benchmark runner: times kernel hot paths and compares against committed
//! baselines in `benches/baselines/` (ADR-0004: criterion replaced by a
//! dependency-free harness with the same informational role in CI).

#![deny(missing_docs)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    println!(
        "rhizome-bench: {} (M0 placeholder schedule)",
        args.join(" ")
    );
}
