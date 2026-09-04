# ADR-0002: Shard frame codec is LZ4-block, not zstd

- Status: accepted
- Date: 2026-09-04
- Milestone: M0

## Context

SPEC 5.4 originally specified zstd frames for shard payload compression.
zstd is a large format (FSE/Huffman entropy coding, complex frame format)
that cannot be reasonably reimplemented dependency-free, and ADR-0001
forbids the dependency.

## Decision

Shard frames are compressed with a self-describing **LZ4 block** codec
implemented in `rhizome-data` (sequence format: token, extended literal
lengths, 16-bit offsets with extension, match lengths; no entropy coding).
The frame header records `codec ∈ {raw, lz4}` so a zstd codec can be added
behind the same versioned header later. Integrity is provided per frame by
BLAKE3 regardless of codec.

## Alternatives considered

- Uncompressed frames only: rejected; text shards compress ~3x, and CI data
  volumes matter on hosted runners.
- Implement zstd decode only (encode raw): rejected; asymmetric formats
  break round-trip tests and writer/reader parity in CI.

## Consequences

- Round-trip and corruption tests cover both codecs.
- Compression ratio is lower than zstd; documented in docs/data.md.
