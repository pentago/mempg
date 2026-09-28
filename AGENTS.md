# Repository Guidelines

## Project Overview

mempg is a **single-file** oh-my-pi (omp) extension (`mempg.ts`). It gives agents persistent memory backed by Postgres (`pg_trgm` + `pgvector`), with optional Ollama embeddings.
- It is published as `@dzhi/mempg`, and only `mempg.ts` ships.
- `README.md` is the user-facing feature doc. This file covers workflow, patterns and traps.

## Architecture & Data Flow

- **Lifecycle**: the module is evaluated once per process. The factory `mempg(pi)` runs once per session, for the main session and for every subagent.
  - Module-level state is shared across sessions: the `sql` pool, `injectionCache`, `logSink`, and the refcount.
  - Per-session state lives in the factory closure (`firstSighting`).
- **Handlers**:
  - `before_agent_start`:
    - captures "remember that…" prompts deterministically (main session only, no LLM);
    - always appends the memory block to `event.systemPrompt`, even with zero rows.
  - `session_stop` requests one hidden continuation to store learnings, only if all of these hold:
    - the turn made ≥8 tool calls;
    - none of them was `memory_remember` or `memory_update`;
    - the prompt was not captured;
    - `stop_hook_active` is not set.
- **Tools**:
  - The 7 `memory_*` tools are defined in `TOOL_SPECS`, a host-neutral JSON-schema table and the source of truth.
  - They are converted to zod via `toZod(pi.zod, …)` and registered with `loadMode: "essential"`.
- **Visibility**:
  - `visibleRows()` is the single project boundary: `stack_fact`/`episodic` are global, `project_fact` only from its origin project.
  - Reads (injection, recall, tags) also apply `notSuperseded()`. Only recall and tags accept `global: true` to widen reach; injection never does.
  - Writes (forget, update, retag, the supersede link) are always bounded by `visibleRows()` and have no `global` opt-in. They skip `notSuperseded()`, so superseded rows stay fixable.
- **Retrieval**:
  - Keyword FTS is merged with pgvector cosine search via RRF (`hybridMerge`: k=60, top 2 vector rows reserved).
  - Any embed failure degrades silently to keyword-only.
  - Writes embed fire-and-forget. A NULL embedding means keyword-only until `bun run backfill`.
- **Consolidate** runs two passes:
  1. **Wording**: trigram similarity, computed in TS, with the project boundary enforced by `mutuallyVisibleRows()`.
  2. **Meaning**: embedding cosine, computed in SQL, with the boundary enforced by `mutuallyVisible()`.
  - The meaning pass excludes the wording pass's removals in both real and `dryRun` mode, so a preview matches a real run on the same snapshot.
  - `detailConflicts` routes a pair with a conflicting number, path or proper noun to `[meaning-uncertain]` instead of deleting it.
- **Cache invalidation is asymmetric**: remember, forget and update clear only the caller's directory. `retag` and a real `consolidate` clear the whole `injectionCache`.

## Key Directories

| Path | Purpose |
|---|---|
| `mempg.ts` | Entire extension, split into `// --- X ---` sections |
| `tests/mempg.test.ts` | Only test file; integration tests against a live DB |
| `deploy/` | `compose.yaml` (pgvector + optional ollama), `init/01-init.sh` (**authoritative DDL**), `backfill.ts`, upgrade SQL in `README.md` |
| `bench/` | Consolidate benchmarks: `embeddinggemma-calibration.ts` (26 pairs, no DB) and `mempg-shaped-consolidate.ts` (808 pairs, scratch DB). Re-run when changing the embed model or `detailConflicts` |
| `scripts/check-tag-version.sh` | pre-push guard: tag must equal `package.json` version |
| `.github/workflows/` | `check.yml` (PRs: lint, typecheck, test); `publish.yml` (`v*` tag → npm) |

## Development Commands

```bash
bun install
bun run check        # biome lint
bun run typecheck    # tsc --noEmit
bun test             # needs live Postgres
bun run backfill     # embed NULL-embedding rows
pre-commit install && pre-commit install --hook-type pre-push
```

- **Formatter**: Biome's formatter is **disabled** and the source is hand-wrapped. Never run `bun run format` alongside logic changes.
- **Pre-commit hooks** typecheck the whole project. Staging `mempg.ts` without matching `tests/` changes can fail.
- **Bench scripts** (`bun run bench:*`): `mempg-shaped-consolidate` DROPs and recreates its scratch DB on whatever server `MEMPG_*` points to. Always set `MEMPG_*` explicitly.

## Code Conventions & Common Patterns

- **Naming**: `SCREAMING_CASE` for constants, camelCase for everything else. Comments explain *why*.
- **Exports**:
  - Runtime `default` only.
  - Test and bench hooks go through `Object.assign(mempg, { __internals })`. Extend that object; never add named exports.
- **Query builders** are pure `(client: SQL, …)` functions. Keep that signature.
- **Errors**:
  - Tools never throw. They return `toolError(kind, action, e)`, which gives the model a generic `ERROR: …`.
  - The real driver message goes to `logError(kind, …)` only, rate-limited per kind.
- **Async**:
  - Side effects are fire-and-forget: `void p.catch(e => logError(kind, …))`.
  - Degradation paths return `null` instead of throwing.
- **Write caps**:
  - Content is 10-4000 chars; at most 10 tags, each up to 64 chars. These are abuse guards: reject with the actual size, never truncate.
  - A write over 700 chars gets a non-blocking nudge.
- **Supersede**: `memory_remember`'s `supersedes` inserts and links in one `sql.begin` transaction. A bad link rolls back the whole write.
- **Injection**:
  - Always extend `event.systemPrompt`, never `ctx.getSystemPrompt()`.
  - Read `ctx.cwd` per event, because `/move` changes it.

## Important Files

- `mempg.ts`: the factory `mempg(pi)`, `TOOL_SPECS`, `handleTransform` (injection), `visibleRows`, and `__internals` at the end of the file.
- `deploy/init/01-init.sh`: the schema. CI bootstraps from this exact file, so drift fails the build.
- `package.json`: `omp.extensions: ["./mempg.ts"]` is the plugin manifest.
- `README.md`: the tool list and env table. It can lag `TOOL_SPECS`. Update it (and `deploy/README.md` for schema changes) in the same commit.

## Runtime/Tooling Preferences

- **Bun only**: runtime, package manager and test runner. TypeScript + Biome.
- **Runtime imports** must be Bun built-ins only.
  - omp packages are `import type` devDependencies.
  - A new runtime import needs a real `dependencies` entry.
- **Bun SQL traps**:
  - Arrays need an element type: `sql.array(values, "text")`.
  - Never use `new SQL("postgres://…")` (it emits DEP0169). Use `makeSql()`.
- **No `console.*` on the extension path**; it corrupts omp's TUI. `logSink` goes to `pi.logger.error`. CLI scripts may use `console`.
- **`dispose()` is refcounted**, one retain/release per session. A plain `sql.close()` would kill the pool when a subagent ends.
- **`before_agent_start` can re-run for one submission.**
  - `firstSighting` is the only exactly-once guard for capture inserts. Keep it.
  - `CAPTURE_NOTE` is appended every time to stop the model storing the request twice.
- **Config**:
  - Config is env-only and read once at load: `MEMPG_HOST`, `MEMPG_PORT`, `MEMPG_USER`, `MEMPG_DB`, `MEMPG_PASSWORD`, `MEMPG_SSL`, `MEMPG_INJECTION`, `MEMPG_OLLAMA_HOST`, `MEMPG_OLLAMA_PORT`, `MEMPG_EMBED_MODEL`.
  - The embed model defaults to `embeddinggemma:300m` (768-dim).

## Testing & QA

- **Framework and CI**: `bun:test`, with no coverage tooling. CI runs on PRs against a pgvector container bootstrapped by `deploy/init/01-init.sh`, with no Ollama, so it runs keyword-only.
- **Hybrid tests** skip unless Ollama is reachable and the `embedding` column exists. A reachable Ollama without the model makes them time out. Pull the model, or set `MEMPG_OLLAMA_PORT=1`.
- **The last test closes the shared pool.** Never add DB tests after it. `afterAll` cleans `/tmp/mempg-test%` projects with its own client.
- **Where tests go**: nested describes under `"DB access layer"`. Factory tests go in `"omp extension"`, which uses a fake `ExtensionAPI` and never fires `session_shutdown`.
- **Isolation rules for new tests**:
  - Use a `/tmp/mempg-test-…` project, deleted in `finally` along with `invalidateInjection(dir)`.
  - The DB may hold a real corpus. Use unique markers like `` `zzz<name>${Date.now()}` ``.
  - Restore module globals in `finally`: `setInjectionMode`, `setOllamaBase`, `setLogSink`.
  - Keyword-only assertions should force `setOllamaBase("http://127.0.0.1:9")`.
  - Before asserting on async results, await `settleCaptures()` for captured rows and `untilEmbedded(id)` for embeddings.
- **Manual QA**:
  - Never use the live DB. Create a throwaway database, run `01-init.sh` against it, and export `MEMPG_*` before importing.

## Git / Release

- **Branching**: never commit to `main`. The user merges and tags.
- **Versioning**: bump the `package.json` version on the same branch as the release change, and tag the merge commit on `main`.
