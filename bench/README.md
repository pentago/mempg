# Consolidate benchmarks

Measurement scripts for `memory_consolidate`'s meaning pass: the
`CONSOLIDATE_EMBED_THRESHOLD` cosine threshold (0.83) and the
`extractDetails`/`detailConflicts` cross-check. Re-run both when the
embedding model or the detail check changes. They call the real
`__internals` functions from `mempg.ts`, so they measure the shipped code.

## `bench:embeddinggemma-calibration`

```bash
bun run bench:embeddinggemma-calibration
```

- Embeds 26 hand-labeled pairs (8 duplicate, 8 update, 10 distinct) and
  prints per-category cosine ranges, a threshold sweep, and the lowest
  threshold with zero distinct-pair false merges.
- Needs only a reachable Ollama with `MEMPG_EMBED_MODEL` pulled. No database.
- Last result (embeddinggemma:300m, 2026-09-22): duplicate 0.789-0.925,
  update 0.540-0.813, distinct 0.332-0.749. 0.83 sits clear of every
  distinct pair.

## `bench:mempg-shaped-consolidate`

```bash
bun run bench:mempg-shaped-consolidate [--dims 768] [--ollama http://localhost:11434]
```

- Builds 808 labeled pairs in-process from the infra vocabulary in
  `bench/config.ts`, with injected numbers, paths and proper nouns. Categories:
  - `duplicate`: same fact, same injected value.
  - `update-*`: same shape, changed value, i.e. a genuinely different fact.
  - `distinct-related`: same topic, different aspect.
  - `fn-stress`: formatting-only differences.
- Prints a threshold sweep, precision/recall/FPR before and after the detail
  check, and a per-category breakdown.
- Needs a reachable Ollama and a `CREATEDB`-capable `MEMPG_*` user.
- **Safety**: it DROPs and recreates `mempg-shaped-consolidate-bench` on
  whatever server `MEMPG_*` points to. Set `MEMPG_*` explicitly.
- `--dims` must match the model's output dimension (`embeddinggemma:300m` = 768).
- Last result (embeddinggemma:300m, threshold 0.83, 2026-09-22):
  - FPR of auto-merged pairs is 0.002 after the detail check.
  - The detail check catches 100% of `update-*` pairs that reach the threshold.
  - 1/281 true duplicates are routed to `[meaning-uncertain]`.
