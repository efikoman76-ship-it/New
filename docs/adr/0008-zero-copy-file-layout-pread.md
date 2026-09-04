# ADR-0008: mmap-able layout served by pread on the CPU backend

- Status: accepted
- Date: 2026-09-04
- Milestone: M0

## Context

SPEC 7 requires shard and weight containers to be "memory-mappable" with the
CPU backend running "directly from the mmap". True `mmap(2)` requires libc
FFI (`unsafe`), and R6 confines `unsafe` to kernel/collective/allocator
crates. `rhizome-data` and the weight loader are otherwise `forbid(unsafe)`.

## Decision

Container layout is designed for mmap semantics: page-aligned sections,
absolute offsets in headers, no relocations required to interpret bytes
(big/little-endian fixed per format), zero-copy tensor descriptors. The
default reader uses `File::read_at` (pread) with an LRU page cache in
`rhizome-runtime`'s arena; a true mmap accessor is provided by
`rhizome-runtime` (the crate R6 permits for allocators) behind the same
trait when measured page-cache behavior demands it. Bit-exactness of
interpretation is identical either way, which the round-trip tests assert.

## Alternatives considered

- libc mmap everywhere: rejected (R6 scoping; portability to Windows CI).
- Reading whole files into memory: rejected for Edge-scale weights (2.5B
  params) on small hosts.

## Consequences

- All format tests run against both the pread reader and (when the mmap
  accessor lands with M9) the mmap reader.
