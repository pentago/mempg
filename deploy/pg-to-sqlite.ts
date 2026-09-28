// Copies every memory, and the cross-session recall records, from the
// Postgres database MEMPG_* points at into the SQLite file MEMPG_SQLITE_PATH.
// Ids, timestamps, tags, supersede links and embeddings carry over, and the
// id counter continues past Postgres' sequence, so "#id" references in memory
// text stay valid. Postgres is only read; the SQLite file must have no
// memories yet (it is created with the schema if missing):
//
//   MEMPG_BACKEND=sqlite MEMPG_SQLITE_PATH=$HOME/.omp/agent/mempg.sqlite bun run port:sqlite
import { SQL } from "bun";
import mempg from "../mempg.ts";

function fail(message: string): never {
  console.error(`port: ${message}`);
  process.exit(1);
}

if (process.env.MEMPG_BACKEND !== "sqlite") fail("set MEMPG_BACKEND=sqlite (and MEMPG_SQLITE_PATH) so mempg opens the target file");

const { __internals } = mempg;
const lite = __internals.sql;
const pg = new SQL({
  hostname: process.env.MEMPG_HOST || "localhost",
  port: Number(process.env.MEMPG_PORT) || 5432,
  username: process.env.MEMPG_USER || "mempguser",
  password: process.env.MEMPG_PASSWORD || "",
  database: process.env.MEMPG_DB || "mempg",
  ssl: __internals.resolveSslMode(process.env.MEMPG_SSL),
  max: 1,
});

await __internals.ready();
const [{ n: existing }] = (await lite`SELECT count(*) AS n FROM memories`) as { n: number }[];
if (existing > 0) fail(`target already has ${existing} memories; port into an empty file`);

type PgRow = {
  id: number;
  content: string;
  tags: string[] | null;
  session_id: string | null;
  project: string | null;
  created_at: string | null;
  memory_type: string;
  updated_at: string | null;
  embedding: string | null;
  superseded_by: number | null;
};
type Recall = { memory_id: number; session_id: string; recalled_at: string };
// Timestamps travel as exact UTC text with Postgres' microseconds (a JS Date
// would cut them to milliseconds). They sort with SQLite's own
// millisecond timestamps, which are always later than a ported row's.
const utc = (pg: SQL, col: string) => pg`to_char(${pg(col)} AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"') AS ${pg(col)}`;
// One read-only snapshot, so rows, recall records and the id sequence agree
// even while agents keep writing to Postgres.
const { rows, recalls, seq } = await pg.begin("ISOLATION LEVEL REPEATABLE READ READ ONLY", async (tx) => {
  const rows = (await tx`
    SELECT id, content, tags, session_id, project, ${utc(tx, "created_at")}, memory_type,
           ${utc(tx, "updated_at")}, embedding::text AS embedding, superseded_by
    FROM memories ORDER BY id
  `) as PgRow[];
  const recalls = (await tx`SELECT memory_id, session_id, ${utc(tx, "recalled_at")} FROM memory_recalls`) as Recall[];
  const [{ seq }] = (await tx`
    SELECT coalesce(pg_sequence_last_value(pg_get_serial_sequence('memories', 'id')::regclass), 0) AS seq
  `) as { seq: number | string }[];
  return { rows, recalls, seq: Number(seq) };
});
await pg.close({ timeout: 0 });

// pgvector's text form is a JSON array; SQLite stores the raw float32 bytes.
const blob = (text: string | null) => (text ? new Uint8Array(new Float32Array(JSON.parse(text) as number[]).buffer) : null);

// Target-shaped rows: what SQLite must hold afterwards, embedding included.
const want = rows.map((r) => ({
  id: r.id,
  content: r.content,
  tags: JSON.stringify(r.tags ?? []),
  session_id: r.session_id,
  project: r.project,
  created_at: r.created_at,
  memory_type: r.memory_type,
  updated_at: r.updated_at,
  superseded_by: r.superseded_by,
  embedding: blob(r.embedding),
}));

await lite.begin(async (tx) => {
  for (const w of want) {
    await tx`
      INSERT INTO memories (id, content, tags, session_id, project, created_at, memory_type, updated_at, embedding)
      VALUES (${w.id}, ${w.content}, ${w.tags}, ${w.session_id}, ${w.project}, ${w.created_at},
              ${w.memory_type}, ${w.updated_at}, ${w.embedding})
    `;
  }
  // Links go in after every row exists: foreign keys are enforced.
  for (const w of want) {
    if (w.superseded_by !== null) await tx`UPDATE memories SET superseded_by = ${w.superseded_by} WHERE id = ${w.id}`;
  }
  for (const r of recalls) {
    await tx`INSERT INTO memory_recalls (memory_id, session_id, recalled_at) VALUES (${r.memory_id}, ${r.session_id}, ${r.recalled_at})`;
  }
  await tx`UPDATE sqlite_sequence SET seq = max(seq, ${seq}) WHERE name = 'memories'`;
});

// Read everything back and compare field by field, embeddings by bytes.
const key = (row: Record<string, unknown>) =>
  JSON.stringify(row, (_, v) => (v instanceof Uint8Array ? Bun.hash(v).toString(16) : v));
const got = (await lite`
  SELECT id, content, tags, session_id, project, created_at, memory_type, updated_at, superseded_by, embedding
  FROM memories ORDER BY id
`) as Record<string, unknown>[];
const bad = want.findIndex((w, i) => got[i] === undefined || key(w) !== key(got[i]));
if (got.length !== want.length || bad !== -1) {
  fail(`memories differ after copy (${got.length} vs ${want.length} rows; first mismatch at id ${want[bad]?.id ?? "?"})`);
}
// Sorted in TS: Postgres and SQLite collate session ids differently.
const recallKeys = (list: Recall[]) => list.map((r) => `${r.memory_id}|${r.session_id}|${r.recalled_at}`).sort().join("\n");
const gotRecalls = (await lite`SELECT memory_id, session_id, recalled_at FROM memory_recalls`) as Recall[];
if (recallKeys(gotRecalls) !== recallKeys(recalls)) {
  fail("memory_recalls differ after copy");
}
const [{ seq: nextBase }] = (await lite`SELECT seq FROM sqlite_sequence WHERE name = 'memories'`) as { seq: number }[];
if (nextBase < seq) fail(`id counter ${nextBase} is behind Postgres' ${seq}`);

const embedded = want.filter((w) => w.embedding !== null).length;
console.log(`port: copied and verified ${got.length} memories (${embedded} embedded), ${gotRecalls.length} recall records; next id > ${nextBase}`);
await __internals.dispose();
