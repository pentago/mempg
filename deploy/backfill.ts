// One-time backfill: fills `embedding` for rows that predate hybrid retrieval
// (or were written while Ollama was down). Idempotent - only NULL rows are
// touched, so re-running is safe. Uses the plugin's own config resolution and
// embed client (via __internals), so MEMPG_* / MEMPG_OLLAMA_* / MEMPG_EMBED_MODEL
// behave exactly as for the plugin:
//
//   bun run backfill
//
// Requires the pgvector schema first (deploy/README.md, "Upgrading an existing
// install") and a reachable Ollama with the model pulled.
import mempg from "../mempg.ts";

const { __internals } = mempg;
const sql = __internals.sql;

function fail(message: string): never {
  console.error(`backfill: ${message}`);
  process.exit(1);
}

// Sanity before touching data: Ollama answers, the column exists, and the
// column dimension matches the model's output (vector(N) stores N in typmod).
const probe = await __internals.embed(["dimension probe"], 15_000);
if (!probe) fail(`Ollama unreachable or model missing at ${__internals.ollamaBase} - embeddings left untouched`);
const dims = probe[0].length;
const att = (await sql`SELECT atttypmod FROM pg_attribute WHERE attrelid = 'memories'::regclass AND attname = 'embedding'`) as {
  atttypmod: number;
}[];
if (att.length === 0) fail("memories.embedding column missing - apply the migration in deploy/README.md first");
if (Number(att[0].atttypmod) !== dims) {
  fail(`dimension mismatch: column is vector(${att[0].atttypmod}), model returns ${dims} - re-create the column for this model`);
}

let done = 0;
const t0 = performance.now();
while (true) {
  const batch = (await sql`SELECT id, content FROM memories WHERE embedding IS NULL ORDER BY id LIMIT 256`) as {
    id: number;
    content: string;
  }[];
  if (batch.length === 0) break;

  const vecs: number[][] = [];
  for (let i = 0; i < batch.length; i += 32) {
    // A null here (Ollama died mid-run) must abort, not skip: skipped rows
    // would stay NULL and the loop would re-pick them forever.
    const part = await __internals.embed(batch.slice(i, i + 32).map((r) => r.content), 60_000);
    if (!part) fail(`Ollama failed mid-run after ${done} rows - re-run to continue (idempotent)`);
    vecs.push(...part);
  }
  const ids = batch.map((r) => r.id);
  const lits = vecs.map((v) => JSON.stringify(v));
  await sql`
    UPDATE memories m SET embedding = v.e::vector
    FROM (SELECT unnest(${sql.array(ids, "integer")}) AS id, unnest(${sql.array(lits, "text")}) AS e) v
    WHERE m.id = v.id
  `;
  done += batch.length;
  console.log(`backfill: ${done} rows embedded (${((performance.now() - t0) / 1000).toFixed(0)}s)`);
}

console.log(done === 0 ? "backfill: nothing to do - every row already has an embedding" : `backfill: done, ${done} rows`);
await sql.close({ timeout: 0 });
