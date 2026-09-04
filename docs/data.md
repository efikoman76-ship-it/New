# Data (SPEC 5.4)

## Shard format (v1)

```
[header]
  magic "RZSH" | u32 version | u32 codec {raw=0,lz4=1}
  u64 frame_count | schema_hash blake3[32] | tokenizer/patcher_hash blake3[32]
  index: frame i -> (u64 byte_offset, u32 compressed_len, u32 raw_len, blake3[32])
[frames: lz4-block or raw payload]
```

- BLAKE3 per frame; corruption is a hard error.
- Deterministic resumable sampling: draw order is
  `hash(seed, epoch, rank, shard)` via Philox; resume replays the cursor.
- Packing with segment ids and per-segment positions (state resets at
  segment starts).
- Optional patch-boundary index for byte-path shards; header records the
  patcher hash; mismatch → error.

## Tools

BPE trainer, MinHash LSH dedup (Jaccard 0.8 banding), quality classifier,
PII/secret scrubbing, language id, patch precompute — all in
`rhizome-data`, exercised by CI on synthetic corpora.

## Synthetic generators (CI + T4)

copy, reverse, MQAR, modular arithmetic (depth 1–4), PCFG with computable
entropy — seeded Philox, byte-stable golden outputs for regression.
