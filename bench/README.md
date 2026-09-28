# Benchmarks

Measurement scripts. They call the real `__internals` functions from
`mempg.ts`, so they measure the shipped code.

The two consolidate benchmarks cover `memory_consolidate`'s meaning pass: the
`CONSOLIDATE_EMBED_THRESHOLD` cosine threshold (0.83) and the
`extractDetails`/`detailConflicts` cross-check. Re-run both when the
embedding model or the detail check changes.

## `bench:backend-compare`

```bash
MEMPG_DB=mempg_bench bun run bench:backend-compare pg.json
MEMPG_BACKEND=sqlite MEMPG_SQLITE_PATH=/tmp/mempg-bench.db bun run bench:backend-compare sqlite.json
```

- Times every tool on whichever backend `MEMPG_BACKEND` selects, and scores
  recall on 30 fictional known-answer memories (a keyword query and a
  paraphrase each), keyword-only and hybrid. Rows already in the database act
  as distractors, so run it on a copy of a real corpus (`port:sqlite` makes
  the SQLite copy).
- **Safety**: it writes, deletes and runs a real consolidate, so it refuses
  a target whose `MEMPG_DB`/`MEMPG_SQLITE_PATH` lacks "bench".
- Needs a reachable Ollama. Fails on any tool error or logged write failure.
- Last result (2026-09-28, 467-row copy of a real corpus, embeddinggemma:300m
  on a 4-CPU Ollama container, two rounds each). Quality is a share of the 30
  items; latencies are p50 / p95 in ms:

  | | pgvector | SQLite |
  |---|---|---|
  | hybrid recall, paraphrase hit@1 / MRR | 0.93 / 0.94 | 0.93 / 0.94 |
  | keyword-only recall, paraphrase hit@1 / MRR | 0.07 / 0.14 | 0.27 / 0.34 |
  | keyword query hit@1 (both modes) | 1.00 | 1.00 |
  | injection, paraphrase in top 5 | 0.93 | 0.93 |
  | consolidate findings | 2 uncertain pairs | the same 2 pairs |
  | remember | 0.37-0.43 / 1.3 | 0.06-0.07 / 0.3-0.4 |
  | update | 0.23-0.27 / 0.3-0.4 | 0.03 / 0.15 |
  | forget | 0.27 / 0.4-0.5 | 0.30 / 0.6 |
  | recall, keyword-only | 0.73-0.86 / 2.0-2.1 | 0.33-0.37 / 0.6 |
  | recall, hybrid | 11-12 / 89-90 | 12-13 / 88-89 |
  | injection, hybrid | 83-85 / 89-91 | 13 / 89-90 |
  | injection, recency | 0.30-0.33 / 0.8-0.9 | 0.36-0.37 / 1.1-1.2 |
  | tags | 0.16-0.21 / 1.0-1.2 | 0.19 / 0.5 |
  | retag | 1.00 / 1.3-1.6 | 0.22-0.23 / 0.5 |
  | consolidate (dry or real) | 800-830 / 850 | 98-102 / 107 |

  Hybrid timings are the Ollama embed, which answers in ~12 or ~86 ms in both
  runs; earlier rounds showed the injection p50 at ~86 ms on SQLite too.
  SQLite's keyword-only edge most likely comes from bm25 weighting rare terms,
  which ts_rank does not; with embeddings on, the two backends tie.

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

- Builds 808 labeled pairs in-process from the infra vocabulary at the top of
  `mempg-shaped-consolidate.ts`, with injected numbers, paths and proper nouns. Categories:
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
