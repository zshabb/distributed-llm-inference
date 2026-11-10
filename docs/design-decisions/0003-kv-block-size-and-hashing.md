# ADR-0003: KV block size 16, SHA-256 hash chains, deferred registration

- Status: Accepted
- Date: 2026-11-10

## Context
M2 introduces paged KV memory with prefix sharing. Three choices define its behavior: how big a block is, how a block's
content is identified, and when a block becomes shareable.

## Decisions

1. **Default block size 16 tokens** (configurable per manager).
   Internal fragmentation is at most 15 slots per sequence, under 1% of a 2K-token context. Prefix sharing works at
   16-token granularity, which fits typical system prompts. 16 is also a common tile size for paged-attention kernels
   on NVIDIA GPUs. The M6 kernel will be benchmarked at 8/16/32 before this default is final for GPU serving.
2. **Block identity = SHA-256 chain over (length-prefixed namespace, parent hash, int64 tokens).**
   A collision would silently serve one request another request's KV, which is both incorrect output and a data leak.
   Python's 64-bit `hash()` is not collision-resistant against crafted inputs. SHA-256 costs ~µs per block, which is
   negligible next to a forward pass.
3. **Register blocks only after their KV is computed** (`mark_computed`), never at allocation.
   Otherwise a request admitted in the same step could hit a block whose KV has not been written yet.
4. **Namespace is mandatory for multi-model or multi-tenant use.** It must include model/weights/adapter identity. Including a
   tenant id disables cross-tenant sharing, which closes the prefix-hit timing side channel at the cost of lower hit rates.

## Consequences
- Prefix sharing cannot help prefixes shorter than one block.
- Identical prompts admitted before either is computed are stored twice (counted in `duplicate_registrations`).
- Changing the hash function or layout invalidates no persistent state (the cache is in-memory only).
