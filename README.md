# mempg

Long-term memory for [oh-my-pi](https://github.com/can1357/oh-my-pi) (`omp`) agents, stored in your own Postgres or in a local SQLite file.

## What it does

An omp agent normally forgets everything when a session ends. mempg gives it a memory that lasts.

- **Saves what matters.** Say "remember that we use jose, not jsonwebtoken" and it is stored word for word. The agent also saves things it learns on its own.
- **Brings it back when relevant.** Before each reply, the memories that match your prompt are handed to the agent. Search understands meaning, so "API rate limit" finds a note about "requests per minute".
- **Keeps projects apart.** Facts about one project stay in that project. Facts about your tools in general are shared everywhere.
- **Handles corrections.** A new memory can replace an old one, so outdated facts stop showing up.
- **Never gets in the way.** If the database or the search model is down, the agent keeps working without memory.
- **Runs on infrastructure you control.** Your Postgres (or a SQLite file) and your Ollama model, local or remote. No third-party service or API key.

## Quick start

1. **Get a database.** Postgres with the `vector` extension. [`deploy/`](./deploy) has a ready Docker Compose setup, with an optional Ollama service for search by meaning. See [`deploy/README.md`](./deploy/README.md).

   **Or skip the server:** set `MEMPG_BACKEND=sqlite` and mempg keeps everything in one file, `~/.omp/agent/mempg.db`, created on first use. Moving existing memories over: `bun run port:sqlite` (see [`deploy/README.md`](./deploy/README.md#sqlite-instead-of-postgres)).
2. **Install the plugin:**

   ```bash
   omp plugin install @dzhi/mempg
   ```

   Or try it from a checkout for one run: `omp -e ./mempg.ts`.
3. **Point it at the database** with environment variables (Postgres only):

   ```bash
   export MEMPG_HOST=localhost MEMPG_USER=mempguser MEMPG_DB=mempg
   export MEMPG_PASSWORD=your-password
   ```

Keep omp's built-in `memory.backend` set to `off`, otherwise you get two memory blocks.

### Settings

All settings are environment variables. There are no plugin options.

| Variable            | Default               | Notes                                                                 |
| ------------------- | --------------------- | --------------------------------------------------------------------- |
| `MEMPG_BACKEND`     | `pgvector`            | `sqlite` stores memories in a local file instead of Postgres          |
| `MEMPG_SQLITE_PATH` | `~/.omp/agent/mempg.db` | SQLite only; the directory must exist                               |
| `MEMPG_HOST`        | `localhost`           |                                                                       |
| `MEMPG_PORT`        | `5432`                |                                                                       |
| `MEMPG_USER`        | `mempguser`           |                                                                       |
| `MEMPG_PASSWORD`    | empty                 |                                                                       |
| `MEMPG_DB`          | `mempg`               |                                                                       |
| `MEMPG_SSL`         | `disable`             | `disable`, `prefer`, `require`, `verify-ca`, `verify-full`            |
| `MEMPG_INJECTION`   | `relevance`           | `recency` injects the newest memories instead of the best matches     |
| `MEMPG_OLLAMA_HOST` | `localhost`           |                                                                       |
| `MEMPG_OLLAMA_PORT` | `11434`               |                                                                       |
| `MEMPG_EMBED_MODEL` | `embeddinggemma:300m` | A different model needs a matching column size (see `deploy/README.md`) |

**If the database is not on your machine, set `MEMPG_SSL=require` or stricter.** Otherwise the login crosses the network unencrypted.

Errors and fallbacks go to the omp log in `~/.omp/logs/`, never to your terminal.

## Using it well

- **Say "remember that…" when it matters.** That always saves. Whether the agent saves something on its own is up to its judgment.
- **Check the type.** A general tooling fact saved as a project fact is invisible in every other project. This is the most common way a useful memory goes missing.
- **Keep memories self-contained.** "Use jose, not jsonwebtoken, for Edge compatibility" makes sense on its own. "We picked the second option" does not.
- **Replace, don't pile up.** When a fact changes, save the new one with `supersedes` pointing at the old one.
- **Clean up now and then.** Run `memory_consolidate` with `dryRun: true` to preview, then run it for real.

---

## How it works

Every memory is a row in one table, `memories`, in Postgres or in the SQLite file.

### Memory types

The type decides where a memory is visible:

| Type                     | Visible             | Use for                                                                  |
| ------------------------ | ------------------- | ------------------------------------------------------------------------ |
| `project_fact` (default) | Its own project only | Facts about this project or customer ("customer A's staging DNS is flaky") |
| `stack_fact`             | Everywhere          | Facts about your tools ("our Terraform RDS module needs `ignore_changes`") |

A rule of thumb: would this help in another customer's repo that uses the same tools? Yes means `stack_fact`.

The same boundary applies to changes. A project can't edit, delete or retag another project's `project_fact`; the call fails and names the owning project. `memory_recall` and `memory_tags` take `global: true` to also read other projects' `project_fact` memories, and `memory_forget`/`memory_update` take it to override the write boundary (e.g. to finish a `memory_consolidate` cleanup from anywhere).

### How memories get saved

- **The agent** calls `memory_remember`.
- **You** write "remember that…", "don't forget…" or "keep in mind…". A fixed pattern, not a model, stores the text after the phrase, up to the first blank line. Questions ("remember when X broke?"), phrases inside code, and long pasted texts are skipped.
- **A checkpoint** runs after a busy turn in which nothing was saved: the agent gets one hidden follow-up asking it to store anything worth keeping.

Each new memory is embedded in the background with Ollama. If Ollama is down, the memory is still saved, and only keyword search finds it until `bun run backfill` embeds it.

Limits: 10-4000 characters and up to 10 tags. Too-large writes are rejected, not truncated. Long memories get a warning, because only the start of each one is shown to the agent.

### How memories come back

Before each agent run, mempg searches with your prompt by keyword (Postgres full-text, or SQLite FTS5) and by meaning (embeddings: pgvector on Postgres, compared in the plugin on SQLite), and merges the results. Near-duplicates are dropped, and the top 5 go into the system prompt, each shortened and labelled with its project. If nothing matches, the newest memories are used instead.

When something fails, it degrades instead of blocking: with Ollama down it searches by keyword only, and with the database down or slow (1-second deadline) the agent simply gets no memories.

### Tools

| Tool                 | What it does                                                          |
| -------------------- | --------------------------------------------------------------------- |
| `memory_remember`    | Store a memory; optional `type`, `tags`, `supersedes`                 |
| `memory_recall`      | Search by keyword and meaning; empty query lists the newest           |
| `memory_update`      | Rewrite a memory, keeping its original date                           |
| `memory_forget`      | Delete a memory by id                                                 |
| `memory_consolidate` | Remove duplicates, on demand only                                     |
| `memory_tags`        | List tags in use, with counts                                         |
| `memory_retag`       | Rename a tag everywhere you're allowed to                             |

### Corrections (`supersedes`)

`memory_remember` with `supersedes: <id>` saves the new memory and marks the old one as replaced. If the old one can't be marked, for example because it belongs to another project, nothing is saved.

A replaced memory is hidden from recall, injection and cleanup, but not deleted. `memory_recall` with `includeSuperseded: true` shows it and what replaced it. Deleting the newer memory makes the old one current again.

### Duplicate cleanup

Writes are never rejected as duplicates. `memory_consolidate` cleans up when you run it. It finds duplicates by similar wording and by similar meaning, keeps the newest memory of each group, deletes the rest, and returns the deleted texts so you can merge in anything unique.

A pair that matches by meaning but differs in a number, path or name (a rate limit of 100 vs 500) is flagged `[meaning-uncertain]` instead. Nothing is deleted; review it yourself.

`dryRun: true` previews a run without deleting. Cleanup respects the same project boundary as edits: a `project_fact` is only ever grouped with memories from its own project.

## Development

```bash
bun install
bun run check      # lint (Biome)
bun run typecheck  # tsc --noEmit
bun test           # integration tests, need Postgres with pgvector
bun run test:sqlite  # the SQLite backend, against a throwaway /tmp file
pre-commit install && pre-commit install --hook-type pre-push
```

The hooks run lint and typecheck on commit, and check that a pushed tag matches the `package.json` version. `bun test` writes to the database it's pointed at, so use a throwaway one. CI runs it against a fresh Postgres container.

Upgrading an existing database: see [`deploy/README.md`](./deploy/README.md#upgrading-an-existing-install). Benchmarks, including a Postgres-vs-SQLite comparison, live in [`bench/`](./bench/README.md).
