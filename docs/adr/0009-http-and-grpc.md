# ADR-0009: HTTP/1.1+SSE server implemented in-repo; gRPC feature contract

- Status: accepted
- Date: 2026-09-04
- Milestone: M0 (server lands M7)

## Context

SPEC 6.7 requires HTTP/SSE (axum) and gRPC (tonic) APIs. ADR-0001 rules out
both crates. Hand-rolling HTTP/1.1 with keep-alive, content-length bodies,
and SSE streaming is well-bounded (~500 lines) and fully testable with an
in-process client. Full gRPC requires HTTP/2 framing plus HPACK (with
Huffman decoding) and protobuf codegen, a disproportionate surface to
hand-roll.

## Decision

- `rhizome-serve` implements an HTTP/1.1 server (threads, keep-alive,
  chunked responses, SSE) plus the OpenAI-compatible and native endpoints.
  Feature `serve-http` (default) gates it.
- Feature `serve-grpc` exists and defines the service surface (proto file
  under `crates/serve/proto/rhizome.proto`) and a typed error if enabled
  without a native transport build: enabling it requires a build with the
  native HTTP/2 stack, which is compile-checked but not shipped
  dependency-free. The HTTP/SSE API is wire-complete and is what CI tests
  (T6).

## Alternatives considered

- Implement HPACK + HTTP/2 by hand: rejected for now; recorded as the
  upgrade path if a consumer requires native gRPC before a dependency
  policy change.

## Consequences

- T6 runs against HTTP/SSE.
- The proto file documents the gRPC contract so the surface stays specified.
