# Provenance

- Run at revision `d6ffccc` (clean), `benchmarks/plans/m5-cpu.json`, 2026-11-11, laptop (i7-1355U, WSL2).
- **Redaction (2026-11-11, before first commit):** `campaign_log.jsonl` recorded each server's full command line,
  including the local home directory in the virtual-environment interpreter and the Hugging Face cache paths. The
  10 occurrences of that home-directory prefix were replaced with `~`. No other byte of any file was changed; no
  measurement is affected. The harness now writes `~` itself (`ce86c4f`), with a test.
- Saturation rule: these sweeps used the first rule, including the tail-biased achieved-rate test later replaced
  (`3685485`); see the GPU campaign's `ANALYSIS.md`.
