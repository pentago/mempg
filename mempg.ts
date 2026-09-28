// mempg - persistent memory extension for oh-my-pi (omp), on Postgres + pgvector
// (default) or a local SQLite file (MEMPG_BACKEND=sqlite).
import { SQL } from "bun";
import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";
import type { ZodLikeSchema } from "@oh-my-pi/omptype/zod";

// --- DB config: env > defaults. Plugin options are not read; password is deliberately env-only (never in config).
const SSL_MODES = ["disable", "prefer", "require", "verify-ca", "verify-full"] as const;
type SslMode = (typeof SSL_MODES)[number];

type DbConfig = {
  host: string;
  port: number;
  user: string;
  database: string;
  ssl: SslMode;
};
type RecallArgs = { query?: string; limit?: number; tags?: string[]; global?: boolean; includeSuperseded?: boolean };
type RememberArgs = {
  content: string;
  tags?: string[];
  type?: MemoryType;
  supersedes?: number;
};
type ForgetArgs = { id: number };
type UpdateArgs = { id: number; content: string; tags?: string[]; type?: MemoryType };
type TagsArgs = { limit?: number; global?: boolean };
type ConsolidateArgs = { dryRun?: boolean };
type RetagArgs = { old: string; new: string };

// Defaults to "disable" so the common localhost setup is unchanged; set MEMPG_SSL
// when the database is remote, otherwise the SCRAM handshake crosses the network
// in plaintext.
function resolveSslMode(raw: string | undefined): SslMode {
  if (!raw) return "disable";
  const mode = raw.toLowerCase() as SslMode;
  return SSL_MODES.includes(mode) ? mode : "disable";
}

// Defaults resolved ONCE at module init - env-only, no process spawning.
const defaultConfig: DbConfig = {
  host: process.env.MEMPG_HOST || "localhost",
  port: Number(process.env.MEMPG_PORT) || 5432,
  user: process.env.MEMPG_USER || "mempguser",
  database: process.env.MEMPG_DB || "mempg",
  ssl: resolveSslMode(process.env.MEMPG_SSL),
};
const password = process.env.MEMPG_PASSWORD || "";

// Options-object constructor, not a URL string: Bun's SQL parses string URLs via
// url.parse(), which emits the DEP0169 DeprecationWarning at load.
function makeSql(cfg: DbConfig): SQL {
  return new SQL({
    hostname: cfg.host,
    port: cfg.port,
    username: cfg.user,
    password,
    database: cfg.database,
    ssl: cfg.ssl,
    max: 2,
    // A dead database must fail fast: this pool is queried from the injection
    // hook, which sits in front of every prompt.
    connectionTimeout: 3,
    idleTimeout: 30,
  });
}

// Backend: anything but "sqlite" means Postgres + pgvector. Resolved once at
// load like the rest of the config. SQLite keeps the same tables in one local
// file (FTS5 for keywords, float32 blobs for embeddings), so no server is needed.
const SQLITE = process.env.MEMPG_BACKEND === "sqlite";
const SQLITE_PATH = process.env.MEMPG_SQLITE_PATH || `${process.env.HOME}/.omp/agent/mempg.sqlite`;

const sql = SQLITE ? new SQL({ adapter: "sqlite", filename: SQLITE_PATH }) : makeSql(defaultConfig);

// SQLite has no bootstrap step like deploy/init/01-init.sh, so the extension
// creates its schema on first use. The columns mirror 01-init.sh; tags are a
// JSON array, timestamps ISO-8601 UTC text, embeddings raw float32 bytes.
// memories_fts is an external-content FTS5 index kept in sync by triggers,
// standing in for Postgres' generated search_vector column.
const SQLITE_SCHEMA = [
  // One connection per process, so these per-connection settings stick.
  // WAL + busy_timeout let several omp processes share the file.
  "PRAGMA journal_mode = WAL",
  "PRAGMA busy_timeout = 5000",
  // Off by default in SQLite; supersede links and memory_recalls rely on it.
  "PRAGMA foreign_keys = ON",
  `CREATE TABLE IF NOT EXISTS memories (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    content          TEXT    NOT NULL,
    tags             TEXT    NOT NULL DEFAULT '[]' CHECK (json_valid(tags)),
    session_id       TEXT,
    project          TEXT,
    created_at       TEXT    DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    memory_type      TEXT    NOT NULL DEFAULT 'project_fact',
    updated_at       TEXT,
    embedding        BLOB,
    superseded_by    INTEGER REFERENCES memories(id) ON DELETE SET NULL,
    CONSTRAINT memories_type_check CHECK (memory_type IN ('stack_fact', 'project_fact'))
  )`,
  "CREATE INDEX IF NOT EXISTS idx_memories_project_created ON memories (project, created_at DESC)",
  `CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
    USING fts5(content, content='memories', content_rowid='id', tokenize='porter unicode61')`,
  `CREATE TRIGGER IF NOT EXISTS memories_fts_insert AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts (rowid, content) VALUES (new.id, new.content);
  END`,
  `CREATE TRIGGER IF NOT EXISTS memories_fts_delete AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts (memories_fts, rowid, content) VALUES ('delete', old.id, old.content);
  END`,
  `CREATE TRIGGER IF NOT EXISTS memories_fts_update AFTER UPDATE OF content ON memories BEGIN
    INSERT INTO memories_fts (memories_fts, rowid, content) VALUES ('delete', old.id, old.content);
    INSERT INTO memories_fts (rowid, content) VALUES (new.id, new.content);
  END`,
  `CREATE TABLE IF NOT EXISTS memory_recalls (
    memory_id   INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    session_id  TEXT    NOT NULL,
    recalled_at TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (memory_id, session_id)
  )`,
];

// Every DB entrypoint awaits this first. Bun opens (and creates) the SQLite
// file when the client is constructed, but a bad path only fails on the first
// query, so loading never throws. Statements run one by one: Bun's SQLite
// adapter neither runs an un-awaited query nor orders it before later ones.
// A failed init is retried by the next call instead of staying cached.
let schemaReady: Promise<void> | undefined;
function ready(): Promise<void> {
  if (!SQLITE) return Promise.resolve();
  schemaReady ??= (async () => {
    for (const statement of SQLITE_SCHEMA) await sql.unsafe(statement);
  })().catch((e: unknown) => {
    schemaReady = undefined;
    throw e;
  });
  return schemaReady;
}

interface MemoryRow {
  id: number;
  content: string;
  tags: string[] | string | null; // JSON text on SQLite; read through tagList()
  project: string;
  date: string;
}

// Injected lines render each memory's origin project (memories are global),
// so the injection queries must select project; id is never rendered.
type InjectionRow = Pick<MemoryRow, "content" | "tags" | "date" | "project">;

// --- Dialect: the few SQL spots where Postgres and SQLite differ. Pure
// builders of the backend's client, like the query builders below. ---

// The schema CHECKs tags are valid JSON; anything but an array of strings
// (a hand-edited row) reads as no tags rather than breaking a render.
function tagList(tags: string[] | string | null): string[] {
  if (typeof tags !== "string") return tags ?? [];
  const parsed: unknown = JSON.parse(tags);
  return Array.isArray(parsed) ? parsed.filter((t): t is string => typeof t === "string") : [];
}

function dateCol(client: SQL) {
  return SQLITE ? client`substr(created_at, 1, 10)` : client`to_char(created_at, 'YYYY-MM-DD')`;
}

function nowSql(client: SQL) {
  return SQLITE ? client`strftime('%Y-%m-%dT%H:%M:%fZ', 'now')` : client`now()`;
}

// A bound tag list. On Postgres, sql.array(tags) alone encodes text[] with
// quoted elements under bun 1.4.2; the element type hint is required.
function tagsParam(client: SQL, tags: string[]) {
  return SQLITE ? JSON.stringify(tags) : client.array(tags, "text");
}

// `id ${idIn(client, ids)}`: membership in a bound id list.
function idIn(client: SQL, ids: number[]) {
  return SQLITE
    ? client`IN (SELECT value FROM json_each(${JSON.stringify(ids)}))`
    : client`= ANY(${client.array(ids, "int8")})`;
}

// A bound embedding: a pgvector literal "[0.1,-0.2,...]" cast to vector, or
// the raw float32 bytes SQLite stores.
function vectorParam(client: SQL, vec: number[]) {
  return SQLITE ? new Uint8Array(new Float32Array(vec).buffer) : client`${JSON.stringify(vec)}::vector`;
}

// Callers only pass blobs whose byte length matches the vector they compare
// against; the floor just keeps a malformed blob from throwing.
function asFloat32(blob: Uint8Array): Float32Array {
  const length = Math.floor(blob.byteLength / 4);
  return blob.byteOffset % 4 === 0
    ? new Float32Array(blob.buffer, blob.byteOffset, length)
    : new Float32Array(blob.slice(0, length * 4).buffer);
}

function cosine(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / Math.sqrt(na * nb);
}

// Nearest neighbours by cosine over rows matching `conds` (a fragment of
// `AND ...` clauses). Postgres orders by pgvector's <=> operator.
// ponytail: SQLite scans every embedded candidate and ranks in TS, O(n) per
// query; move to the sqlite-vec extension once that scan shows up in recall
// latency (bench:backend-compare).
async function nearest<T>(client: SQL, cols: SQL.Query<unknown>, conds: SQL.Query<unknown>, vec: number[], limit: number): Promise<T[]> {
  if (!SQLITE) {
    return (await client`
      SELECT ${cols} FROM memories
      WHERE embedding IS NOT NULL ${conds}
      ORDER BY embedding <=> ${JSON.stringify(vec)}::vector
      LIMIT ${limit}
    `) as T[];
  }
  // Same-size vectors only, as pgvector's typed column guarantees: a row
  // embedded by a different model is skipped until the backfill re-embeds it.
  const rows = (await client`
    SELECT ${cols}, embedding FROM memories
    WHERE embedding IS NOT NULL AND length(embedding) = ${vec.length * 4} ${conds}
  `) as (T & { embedding?: Uint8Array })[];
  const query = Float32Array.from(vec);
  return rows
    .map((row) => ({ row, score: cosine(query, asFloat32(row.embedding as Uint8Array)) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map(({ row }) => {
      delete row.embedding;
      return row;
    });
}

// Keyword match. Postgres matches the generated search_vector; SQLite joins
// the FTS5 index, the only place bm25() can be evaluated (negated so higher
// is better, like ts_rank; keywordScore reads it). Callers add ftsJoin to
// FROM and ftsMatch to WHERE. An empty query matches nothing, as an empty
// tsquery does on Postgres (FTS5 rejects a bare '' but not '""').
function ftsJoin(client: SQL, q: string) {
  return SQLITE
    ? client`JOIN (
        SELECT rowid AS fts_id, -bm25(memories_fts) AS fts_rank FROM memories_fts WHERE memories_fts MATCH ${q || '""'}
      ) fts ON fts.fts_id = memories.id`
    : client``;
}

function ftsMatch(client: SQL, q: string) {
  return SQLITE ? client`1=1` : client`search_vector @@ to_tsquery('english', ${q})`;
}

// --- Rate-limited error logging ---

const lastLogTime = new Map<string, number>();

function rateLimitOk(kind: string): boolean {
  const now = Date.now();
  const last = lastLogTime.get(kind) ?? 0;
  if (now - last < 60_000) return false;
  lastLogTime.set(kind, now);
  return true;
}

// Test hook: the 60s rate-limit window is module state; tests clear it for determinism.
function resetRateLimit(): void {
  lastLogTime.clear();
}

// Diagnostics go through a sink: CLI scripts (bench/, deploy/backfill.ts)
// keep console.error; the omp factory routes them to pi.logger, since
// console output corrupts omp's TUI. Rate limiting is per kind.
const defaultLogSink = (line: string): void => console.error(line);
let logSink: (line: string) => void = defaultLogSink;
// undefined restores the default (tests undo the factory's sink with it).
function setLogSink(sink?: (line: string) => void): void {
  logSink = sink ?? defaultLogSink;
}
function logError(kind: string, message: string): void {
  if (!rateLimitOk(kind)) return;
  logSink(`[mempg] ${message}`);
}

// --- Query deadline ---

// Guards the injection query, which runs in front of every model request: a
// hung database must degrade to "no memories" rather than stall the turn.
//
// Races instead of cancelling: on bun 1.4.2 query.cancel() is a no-op for an
// in-flight Postgres query, so the connection stays busy until the server finishes.
async function withDeadline<T>(query: PromiseLike<T>, ms: number): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const deadline = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(Object.assign(new Error(`query exceeded ${ms}ms deadline`), { name: "DeadlineError" })), ms);
  });
  // An abandoned query that later rejects would otherwise surface as an
  // unhandled rejection and take the process down.
  Promise.resolve(query).catch(() => {});
  try {
    return await Promise.race([query, deadline]);
  } finally {
    clearTimeout(timer);
  }
}

// --- Embeddings (hybrid retrieval: keyword + vector) ---

// embeddinggemma:300m via Ollama over HTTP. Any failure degrades to keyword-only;
// embeddings can never break search. A model with different dimensions needs the
// column re-created and a re-embed (deploy/README.md).
const OLLAMA_HOST = process.env.MEMPG_OLLAMA_HOST || "localhost";
const OLLAMA_PORT = Number(process.env.MEMPG_OLLAMA_PORT) || 11434;
const EMBED_MODEL = process.env.MEMPG_EMBED_MODEL || "embeddinggemma:300m";
let ollamaBase = `http://${OLLAMA_HOST}:${OLLAMA_PORT}`;

// A cold model load (~2-3s) exceeds the injection deadline, so session_start warms
// the model and every request passes keep_alive. The write path tolerates cold loads.
const EMBED_QUERY_TIMEOUT_MS = 750;
const EMBED_WRITE_TIMEOUT_MS = 15_000;
// Candidate slice per hybrid half, mirroring the keyword LIMIT 20.
const EMBED_CANDIDATES = 20;

// Never rejects: any failure (daemon down, model missing, timeout) logs
// rate-limited and returns null, and the caller runs keyword-only.
async function embed(texts: string[], timeoutMs: number): Promise<number[][] | null> {
  try {
    const res = await fetch(`${ollamaBase}/api/embed`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      // keep_alive pins the model: the default 5m unload would make the first
      // query after any idle gap pay the ~2-3s cold load and lose to the timeout.
      body: JSON.stringify({ model: EMBED_MODEL, input: texts, keep_alive: "30m" }),
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!res.ok) throw new Error(`ollama /api/embed ${EMBED_MODEL}: HTTP ${res.status}`);
    return ((await res.json()) as { embeddings: number[][] }).embeddings;
  } catch (e: unknown) {
    logError("embed", `mempg embedding unavailable, keyword-only: ${e instanceof Error ? e.message : String(e)}`);
    return null;
  }
}

// Reciprocal rank fusion (k=60, the standard constant): a row scores
// 1/(60 + 1-based rank) in each list it appears in, so a hit in both lists
// outranks either list's tail. Keyed on content - identical text is the same
// memory.
function rrfMerge<T extends { content: string }>(lists: T[][], k = 60): T[] {
  const scores = new Map<string, { score: number; row: T }>();
  for (const list of lists) {
    list.forEach((row, i) => {
      const entry = scores.get(row.content) ?? { score: 0, row };
      entry.score += 1 / (k + i + 1);
      scores.set(row.content, entry);
    });
  }
  // Map preserves insertion order and Array.sort is stable, so fused-score
  // ties keep the keyword list's order (it is always merged first).
  return [...scores.values()].sort((a, b) => b.score - a.score).map((e) => e.row);
}

// The vector list's top `reserved` rows are unconditional, the rest is the RRF
// merge minus those picks: plain RRF dilutes vector hits when the keyword half is
// noisy, and reweighting doesn't help. reserved=0 or no vector rows = plain merge.
function hybridMerge<T extends { content: string }>(keywordRows: T[], vectorRows: T[], reserved = 2): T[] {
  const picked = vectorRows.slice(0, reserved);
  const rest = rrfMerge([keywordRows, vectorRows]).filter((r) => !picked.some((p) => p.content === r.content));
  return [...picked, ...rest];
}

// Off the write path: embed one memory and store its vector. Never throws -
// callers fire-and-forget it, so every failure (Ollama down, missing column)
// lands here. A row whose embedding stays NULL is keyword-only until the
// backfill (deploy/backfill.ts) or a later memory_update fills it. The write
// only lands while the row still holds `content`: a slow embed of the old text
// finishing after a memory_update must not overwrite the cleared vector.
async function embedAndStore(id: number, content: string): Promise<void> {
  const vecs = await embed([content], EMBED_WRITE_TIMEOUT_MS);
  if (!vecs) return;
  try {
    await sql`UPDATE memories SET embedding = ${vectorParam(sql, vecs[0])} WHERE id = ${id} AND content = ${content}`;
  } catch (e: unknown) {
    logError("embed-write", `mempg embedding store failed for #${id}: ${e instanceof Error ? e.message : String(e)}`);
  }
}

// In-flight write-path embeds, so tests and benches can wait for them before
// asserting or closing the pool. Session shutdown does not wait: a lost embed
// only leaves a NULL vector for the backfill, while waiting could hold omp's
// exit for up to EMBED_WRITE_TIMEOUT_MS.
const inflightEmbeds = new Set<Promise<void>>();

function track(inflight: Set<Promise<void>>, p: Promise<void>): void {
  inflight.add(p);
  void p.finally(() => inflight.delete(p));
}

function embedInBackground(id: number, content: string): void {
  track(inflightEmbeds, embedAndStore(id, content));
}

async function settleEmbeddings(): Promise<void> {
  await Promise.all(inflightEmbeds);
}

// --- Injection pipeline ---

// "relevance" (default) ranks visible memories against the latest prompt, falling
// back to recency on no match; MEMPG_INJECTION=recency injects the latest 5.
let injectionMode: "relevance" | "recency" =
  process.env.MEMPG_INJECTION === "recency" ? "recency" : "relevance";

function truncateMemory(content: string): string {
  if (content.length <= 600) return content;
  return `${content.slice(0, 600)}…[truncated]`;
}

// Memory content is interpolated verbatim into the system prompt; a stored
// memory containing the closing tag would otherwise end the block early and
// have its remainder read as top-level instructions.
function sanitizeMemory(content: string): string {
  return content.replaceAll("</persistent-project-memory>", "");
}

// Character-trigram Jaccard, approximating pg_trgm similarity.
function trigrams(text: string): Set<string> {
  const s = text.toLowerCase().replace(/\s+/g, " ");
  const out = new Set<string>();
  for (let i = 0; i < s.length - 2; i++) out.add(s.slice(i, i + 3));
  return out;
}

function nearDupeSets(A: Set<string>, B: Set<string>): boolean {
  if (A.size === 0 || B.size === 0) return false;
  let inter = 0;
  for (const t of A) if (B.has(t)) inter++;
  return inter / (A.size + B.size - inter) >= DEDUP_SIMILARITY;
}

// Greedily drops rows that near-dupe a kept row: writes never reject duplicates,
// so restatements of one fact would otherwise fill all 5 slots.
function collapseDupes(rows: InjectionRow[]): InjectionRow[] {
  const kept: InjectionRow[] = [];
  for (const row of rows) {
    if (kept.length >= 5) break;
    if (!kept.some((k) => nearDupeSets(trigrams(k.content), trigrams(row.content)))) kept.push(row);
  }
  return kept;
}

// Shared by the injected block and the session_stop checkpoint note, so the
// two can never drift into different write policies.
const WRITE_TRIGGERS: readonly string[] = [
  "the user corrects you, states a preference, or tells you how they want something done",
  "you discover why something is the way it is (a constraint, a gotcha, a non-obvious reason behind a decision)",
  "you solve something that took more than one attempt",
  "you learn a fact about this environment that isn't visible in the code",
];

// Emitted even for zero rows: the write/recall guidance matters most exactly
// when nothing is stored yet and the first memory has to be written.
function formatBlock(rows: InjectionRow[], projectDir: string): string {
  const lines: string[] = [
    "<persistent-project-memory>",
    // Framing precedes the rows so it isn't skimmed as a footnote.
    "This is your memory of this project across sessions. You have no other",
    "access to it - anything not recorded here or retrievable via memory_recall",
    "did not survive. Treat these as established facts you already know, not",
    "suggestions.",
    "",
    `Project: ${projectDir}`,
    rows.length === 0 ? "Memories: none visible here yet." : "Memories:",
  ];
  for (const row of rows) {
    const tags = tagList(row.tags);
    const tagStr = tags.length ? ` [${tags.join(", ")}]` : "";
    // Memories are shared, so show the origin project (basename keeps it short).
    const origin = row.project.split('/').pop() ?? row.project;
    lines.push(`- [${row.date}] (${origin})${tagStr} ${sanitizeMemory(truncateMemory(row.content))}`);
  }
  lines.push("");
  // Each bullet names an observable event: a threshold the model judges itself
  // always resolves toward skipping.
  lines.push(
    "Write to memory when any of these happen - do not defer, the session ends without warning and unwritten context is lost permanently:",
  );
  for (const t of WRITE_TRIGGERS) lines.push(`- ${t}`);
  lines.push("");
  // Framed as self-interested efficiency, not rule compliance: finding a past
  // answer is cheaper than rediscovering it.
  lines.push(
    "Search memory (memory_recall) before you debug an error or failing command, change config/infra/deploy/CI, make a design or tool choice, or start work in an area you have not touched this session - past attempts, decisions, and fixes are there, and finding them is cheaper than rediscovering them.",
  );
  lines.push("</persistent-project-memory>");
  return lines.join("\n");
}

// Keyed by directory + prompt hash so retries hit the cache. Zero-row results are
// cached too, so no-match prompts stop re-querying.
const injectionCache = new Map<string, string>();

// Retrieval query cap: FTS gains nothing from thousands of characters.
function capPromptQuery(text: string): string {
  return text.trim().slice(0, 512);
}

// stack_fact is visible everywhere; project_fact only from its origin project,
// unless recall passes global: true (injection never does).
function visibleRows(client: SQL, directory: string) {
  return client`(memory_type != 'project_fact' OR project = ${directory})`;
}

// Superseded rows are history, hidden from injection and normal recall. Kept
// separate from visibleRows because forget/update must still reach them.
function notSuperseded(client: SQL) {
  return client`superseded_by IS NULL`;
}

// Latest visible rows. "Newest" orderings end in `id DESC` everywhere: SQLite
// timestamps are millisecond text, so back-to-back writes can tie.
function buildRecencyQuery(client: SQL, directory: string) {
  // LIMIT 20: a candidate slice for collapseDupes, not the final block.
  return client`
    SELECT content, coalesce(tags, '{}') AS tags,
           ${dateCol(client)} AS date,
           project
    FROM memories
    WHERE ${visibleRows(client, directory)} AND ${notSuperseded(client)}
    ORDER BY created_at DESC, id DESC
    LIMIT 20
  `;
}

// Cross-session tiebreak: distinct sessions that recalled the memory, capped and
// weighted below the same-project boost. Repeat recalls in one session count once.
function crossSessionJoin(client: SQL) {
  return client`
    LEFT JOIN (
      SELECT memory_id, count(DISTINCT session_id) AS xsess
      FROM memory_recalls
      GROUP BY memory_id
    ) recalls ON recalls.memory_id = memories.id
  `;
}

// The keyword ranking: text rank plus the same-project boost (injection only,
// `directory` null skips it) and the cross-session tiebreak. On Postgres both
// are small constants beside ts_rank, whose scale is fixed (~0.06 per matched
// term). bm25's scale depends on corpus size (~1e-6 on a handful of rows), so
// on SQLite constants would swamp it; the same boosts become multipliers
// instead, sized to the share of a one-term ts_rank they amount to on
// Postgres (0.01 ~ 15%, 0.002 ~ 3%).
function keywordScore(client: SQL, tsQuery: string, directory: string | null) {
  if (SQLITE) {
    const project = directory === null ? client`0` : client`(CASE WHEN project = ${directory} THEN 0.15 ELSE 0 END)`;
    return client`fts.fts_rank * (1 + ${project} + min(coalesce(recalls.xsess, 0), 5) * 0.03)`;
  }
  const project = directory === null ? client`0` : client`(CASE WHEN project = ${directory} THEN 0.01 ELSE 0 END)`;
  return client`ts_rank(search_vector, to_tsquery('english', ${tsQuery})) + ${project} + LEAST(coalesce(recalls.xsess, 0), 5) * 0.002`;
}

function buildRelevanceQuery(client: SQL, tsQuery: string, directory: string) {
  return client`
    SELECT content, coalesce(tags, '{}') AS tags,
           ${dateCol(client)} AS date,
           project
    FROM memories
    ${crossSessionJoin(client)}
    ${ftsJoin(client, tsQuery)}
    WHERE ${ftsMatch(client, tsQuery)}
      AND ${visibleRows(client, directory)}
      AND ${notSuperseded(client)}
    ORDER BY ${keywordScore(client, tsQuery, directory)} DESC,
             created_at DESC, id DESC
    LIMIT 20
  `;
}

// Vector half of the hybrid. Always returns rows when any are embedded, so the
// caller merges via rrfMerge. Throws without an embedding column; caller falls back.
function buildVectorQuery(client: SQL, vec: number[], directory: string) {
  return nearest<InjectionRow>(
    client,
    client`content, coalesce(tags, '{}') AS tags, ${dateCol(client)} AS date, project`,
    client`AND ${visibleRows(client, directory)} AND ${notSuperseded(client)}`,
    vec,
    EMBED_CANDIDATES,
  );
}

// The prompt as an OR of stemmed words: websearch_to_tsquery ANDs the terms,
// so one word the memory never uses would zero out the whole query. OR ranks by how many (and how
// rare) the matched terms are, and sanitizing to [a-z0-9]+ tokens keeps
// to_tsquery syntax-safe. Capped at 24 words to bound the query.
// SQLite's FTS5 has no stopword list, so the words Postgres' english config
// drops are removed here instead; a prompt of only stopwords becomes `""`,
// which, like Postgres' empty tsquery, matches nothing.
function orTsQuery(text: string): string {
  const words = (text.toLowerCase().match(/[a-z0-9]+/g) ?? []).slice(0, 24);
  if (!SQLITE) return words.join(" | ");
  const terms = words.filter((w) => !STOPWORDS.has(w));
  if (terms.length === 0) return words.length ? '""' : "";
  return terms.map((w) => `"${w}"`).join(" OR ");
}

// Postgres' english.stop, verbatim (tsearch_data/english.stop, pg 18).
const STOPWORDS = new Set(
  (
    "i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers " +
    "herself it its itself they them their theirs themselves what which who whom this that these those am is " +
    "are was were be been being have has had having do does did doing a an the and but if or because as until " +
    "while of at by for with about against between into through during before after above below to from up " +
    "down in out on off over under again further then once here there when where why how all any both each few " +
    "more most other some such no nor not only own same so than too very s t can will just don should now"
  ).split(" "),
);

async function handleTransform(
  output: { system: string[] },
  directory: string,
  prompt = "",
): Promise<void> {
  if (!directory) return;
  const query = injectionMode === "relevance" ? prompt.trim() : "";
  const cacheKey = `${directory}\u0001${Bun.hash(query).toString(36)}`;
  const cached = injectionCache.get(cacheKey);
  if (cached !== undefined) {
    if (cached) output.system.push(cached);
    return;
  }
  try {
    await withDeadline(ready(), 1000);
    const inject = (q: unknown) => withDeadline(q as PromiseLike<InjectionRow[]>, 1000);
    let rows: InjectionRow[] = [];
    const tsQuery = orTsQuery(query);
    if (tsQuery) {
      // Embed concurrently so it hides behind the keyword round-trip; any
      // embed or vector-query failure degrades to keyword-only.
      const embedding = embed([query], EMBED_QUERY_TIMEOUT_MS);
      const keywordRows = await inject(buildRelevanceQuery(sql, tsQuery, directory));
      rows = keywordRows;
      const vecs = await embedding;
      if (vecs) {
        const vectorRows = await inject(buildVectorQuery(sql, vecs[0], directory)).catch((e: unknown) => {
          logError("embed-query", `mempg vector query failed, keyword-only: ${e instanceof Error ? e.message : String(e)}`);
          return null;
        });
        if (vectorRows) rows = hybridMerge(keywordRows, vectorRows);
      }
    }
    // No signal at all: recency beats an empty block. Vector hits alone count.
    if (rows.length === 0) rows = await inject(buildRecencyQuery(sql, directory));
    const block = formatBlock(collapseDupes(rows), directory);
    if (injectionCache.size >= 32) {
      const firstKey = injectionCache.keys().next().value;
      if (firstKey !== undefined) injectionCache.delete(firstKey);
    }
    injectionCache.set(cacheKey, block);
    if (block) output.system.push(block);
  } catch (e: unknown) {
    logError("inject", `mempg injection failed: ${e instanceof Error ? e.message : String(e)}`);
  }
}

// --- Dispose ---

// omp evaluates this module once per process but runs the extension factory
// once per session (the main session plus every subagent), so all bindings
// share the pool. Without the refcount, one subagent's shutdown would close
// the pool under the main session.
let instances = 0;

function retain(): void {
  instances++;
}

async function dispose(): Promise<void> {
  if (instances > 0) instances--;
  if (instances > 0) return;
  await sql.close().catch(() => {});
}

// --- Agent tools: recall + remember with dedup-on-write ---

// Write caps are abuse guards, far above ordinary memories.
const MAX_CONTENT = 4000;
const MIN_CONTENT = 10;
const MAX_TAGS = 10;
const MAX_TAG_LENGTH = 64;

// Soft nudge, never a rejection: just above formatBlock's 600-char injection cut.
const CONTENT_LENGTH_NUDGE = 700;

function lengthNudge(content: string): string {
  if (content.length <= CONTENT_LENGTH_NUDGE) return "";
  return (
    ` Note: this entry is ${content.length.toLocaleString()} characters - injection truncates at 600, ` +
    "so most of this won't be visible in ambient context. Consider shortening to the essential " +
    "1-3 sentences and putting longer rationale in project docs."
  );
}

// --- Memory types ---

// Mirrors memories_type_check. project_fact is origin-project-only so
// customer-specific facts never surface elsewhere.
const MEMORY_TYPES = ["stack_fact", "project_fact"] as const;
type MemoryType = (typeof MEMORY_TYPES)[number];

// Raw JSON Schema input is not coerced for us: a model sending "3" or null for
// limit would otherwise reach Postgres as LIMIT NaN.
function clampInt(raw: unknown, fallback: number, max: number): number {
  const n = Number(raw);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(Math.max(Math.trunc(n), 1), max);
}

// The model sees a generic failure; the operator sees the real message in the
// server log. Raw driver errors carry host, user and schema details that should
// not end up in a transcript sent to the provider.
function toolError(kind: string, action: string, e: unknown): string {
  logError(kind, `mempg ${action} failed: ${e instanceof Error ? e.message : String(e)}`);
  return `ERROR: memory store unavailable (${action} failed; see the omp log under ~/.omp/logs).`;
}

async function recall(
  args: RecallArgs,
  ctx: { directory: string; sessionID?: string },
): Promise<string> {
  try {
    await ready();
    const limit = clampInt(args.limit, 10, 20);
    const visibleCond = args.global ? sql`` : sql`AND ${visibleRows(sql, ctx.directory)}`;
    const supersededCond = args.includeSuperseded ? sql`` : sql`AND ${notSuperseded(sql)}`;
    const q = args.query ?? "";
    const tsQuery = q ? orTsQuery(q) : "";
    const queryCond = q ? sql`AND ${ftsMatch(sql, tsQuery)}` : sql``;
    // Tags aren't in search_vector, so this filter is the only way to reach them.
    const wantTags = Array.isArray(args.tags) ? args.tags.filter((t) => typeof t === "string" && t) : [];
    const tagCond = !wantTags.length
      ? sql``
      : SQLITE
        ? sql`AND NOT EXISTS (
            SELECT 1 FROM json_each(${JSON.stringify(wantTags)}) want
            WHERE want.value NOT IN (SELECT value FROM json_each(memories.tags))
          )`
        : sql`AND tags @> ${sql.array(wantTags, "text")}`;
    // No raw exposure count feeds ranking: bumping the returned rows would be a
    // rich-get-richer loop.
    const orderBy = q
      ? sql`ORDER BY ${keywordScore(sql, tsQuery, null)} DESC, created_at DESC, id DESC`
      : sql`ORDER BY created_at DESC, id DESC`;

    // Embed concurrently; any embedding failure degrades to keyword-only.
    const embedding = tsQuery ? embed([q], EMBED_QUERY_TIMEOUT_MS) : Promise.resolve(null);
    // A directed query fetches a candidate slice for the RRF merge; the
    // caller's limit is applied after. Browse keeps its plain LIMIT.
    type RecallRow = MemoryRow & { memory_type: string; superseded_by: number | null };
    const cols = sql`id, content, coalesce(tags, '{}') AS tags, ${dateCol(sql)} AS date, project, memory_type, superseded_by`;
    const keywordRows = await sql`
      SELECT ${cols}
      FROM memories
      ${crossSessionJoin(sql)}
      ${q ? ftsJoin(sql, tsQuery) : sql``}
      WHERE 1=1 ${visibleCond} ${supersededCond} ${queryCond} ${tagCond}
      ${orderBy}
      LIMIT ${q ? EMBED_CANDIDATES : limit}
    ` as RecallRow[];

    let rows = keywordRows;
    const vecs = await embedding;
    if (vecs) {
      // Same filters as the keyword half, minus the FTS condition.
      const vectorRows = await nearest<RecallRow>(
        sql,
        cols,
        sql`${visibleCond} ${supersededCond} ${tagCond}`,
        vecs[0],
        EMBED_CANDIDATES,
      ).catch((e: unknown) => {
        logError("embed-query", `mempg vector query failed, keyword-only: ${e instanceof Error ? e.message : String(e)}`);
        return null;
      });
      if (vectorRows) {
        // Reserved slots never exceed half the result: at limit 1-2 the plain
        // merge decides, so a direct query's exact keyword hit cannot be
        // displaced by a merely-adjacent vector row.
        rows = hybridMerge(keywordRows, vectorRows, Math.min(2, Math.floor(limit / 2)));
      }
    }
    if (q) rows = rows.slice(0, limit);

    const ids = rows.map((r) => r.id);
    // Records the cross-session signal (idempotent per session via the PK);
    // must never block or fail the recall.
    if (ids.length > 0 && ctx.sessionID) {
      const record = SQLITE
        ? sql`
            INSERT OR IGNORE INTO memory_recalls (memory_id, session_id)
            SELECT value, ${ctx.sessionID} FROM json_each(${JSON.stringify(ids)})
          `
        : sql`
            INSERT INTO memory_recalls (memory_id, session_id)
            SELECT unnest(${sql.array(ids, "int8")}), ${ctx.sessionID}
            ON CONFLICT (memory_id, session_id) DO NOTHING
          `;
      void record.catch((e: unknown) => logError("xsess", `mempg cross-session record failed: ${e instanceof Error ? e.message : String(e)}`));
    }

    if (rows.length === 0) return "No memories found.";

    return rows
      .map((r) => {
        const tags = tagList(r.tags);
        const tagStr = tags.length ? ` (${tags.join(', ')})` : '';
        // project_fact is the default every pre-column row carries, so
        // printing it is pure noise.
        const typeStr = r.memory_type === "project_fact" ? "" : ` [${r.memory_type}]`;
        const supersededStr = r.superseded_by ? ` [superseded by #${r.superseded_by}]` : '';
        return `[${r.date}] [${r.project}]${typeStr}${tagStr}\n#${r.id}${supersededStr}\n${r.content}`;
      })
      .join('\n---\n');
  } catch (e: unknown) {
    return toolError("recall", "recall", e);
  }
}

// Tag counts so callers reuse established tags; superseded rows' tags excluded.
async function listTags(args: TagsArgs, ctx: { directory: string }, client: SQL = sql): Promise<string> {
  try {
    await ready();
    // High default: rare one-off tags are what this tool exists to surface.
    const limit = clampInt(args.limit, 200, 500);
    const visibleCond = args.global ? client`` : client`AND ${visibleRows(client, ctx.directory)}`;
    // SQLite unnests the JSON array with json_each; the output alias `tag`
    // then serves GROUP BY/ORDER BY on both backends.
    const tagCol = SQLITE ? client`t.value AS tag` : client`tag`;
    const tagSource = SQLITE ? client`json_each(memories.tags) AS t` : client`unnest(tags) AS tag`;
    const rows = await client`
      SELECT ${tagCol}, count(*) AS uses
      FROM memories, ${tagSource}
      WHERE 1=1 ${visibleCond} AND ${notSuperseded(client)}
      GROUP BY tag
      ORDER BY uses DESC, tag ASC
      LIMIT ${limit}
    ` as { tag: string; uses: string | number }[];

    if (rows.length === 0) return "No tags found.";
    return rows.map((r) => `${r.tag} (${r.uses})`).join("\n");
  } catch (e: unknown) {
    return toolError("tags", "tags", e);
  }
}

// A rename must not land a tag validateWrite would reject.
function validateRetag(args: RetagArgs): string | null {
  if (typeof args.old !== "string" || args.old.length === 0) {
    return "ERROR: old must be a non-empty tag string.";
  }
  if (typeof args.new !== "string" || args.new.length === 0) {
    return "ERROR: new must be a non-empty tag string.";
  }
  if (args.new.length > MAX_TAG_LENGTH) {
    return `ERROR: each tag must be a string of at most ${MAX_TAG_LENGTH} characters.`;
  }
  return null;
}

// Same project-boundary guard as forget/update. array_replace() doesn't dedupe:
// a row tagged both `old` and `new` would get `new` twice, hence DISTINCT/unnest.
async function retag(args: RetagArgs, ctx: { directory: string }): Promise<string> {
  try {
    const invalid = validateRetag(args);
    if (invalid) return invalid;
    await ready();

    // SQLite rebuilds the JSON array in its original order, dropping the
    // duplicate the rename can create (GROUP BY keeps each tag's first slot).
    const updated = await (SQLITE
      ? sql`
          UPDATE memories
          SET tags = (
            SELECT json_group_array(tag) FROM (
              SELECT CASE WHEN value = ${args.old} THEN ${args.new} ELSE value END AS tag, min(key) AS pos
              FROM json_each(memories.tags) GROUP BY tag ORDER BY pos
            )
          )
          WHERE EXISTS (SELECT 1 FROM json_each(memories.tags) WHERE value = ${args.old})
            AND ${visibleRows(sql, ctx.directory)}
          RETURNING id
        `
      : sql`
          UPDATE memories
          SET tags = ARRAY(SELECT DISTINCT unnest(array_replace(tags, ${args.old}, ${args.new})))
          WHERE tags @> ARRAY[${args.old}]
            AND ${visibleRows(sql, ctx.directory)}
          RETURNING id
        `) as { id: number }[];

    if (updated.length === 0) {
      return `No memories tagged "${args.old}".`;
    }
    // stack_fact rows show in every project's block: clear all, not one directory.
    injectionCache.clear();
    return `Retagged ${updated.length} ${updated.length === 1 ? "memory" : "memories"}: "${args.old}" → "${args.new}".`;
  } catch (e: unknown) {
    return toolError("retag", "retag", e);
  }
}

// Rejects rather than truncates: a clipped memory loses its tail silently,
// while an error reports the actual size and lets the agent retry shorter.
function validateWrite(args: RememberArgs): string | null {
  const content = typeof args.content === "string" ? args.content : "";
  if (content.length < MIN_CONTENT) {
    return `ERROR: content must be at least ${MIN_CONTENT} characters.`;
  }
  if (content.length > MAX_CONTENT) {
    return `ERROR: content is ${content.length} characters, max ${MAX_CONTENT}; store the essentials in 1-3 sentences and retry.`;
  }
  const tags = args.tags ?? [];
  if (!Array.isArray(tags)) return "ERROR: tags must be an array of strings.";
  if (tags.length > MAX_TAGS) {
    return `ERROR: ${tags.length} tags given, max ${MAX_TAGS}.`;
  }
  const oversized = tags.find((t) => typeof t !== "string" || t.length > MAX_TAG_LENGTH);
  if (oversized !== undefined) {
    return `ERROR: each tag must be a string of at most ${MAX_TAG_LENGTH} characters.`;
  }
  if (args.type !== undefined && !MEMORY_TYPES.includes(args.type)) {
    return `ERROR: type must be one of ${MEMORY_TYPES.join(", ")}.`;
  }
  return null;
}

// Trigram threshold for consolidation and injection collapse: at 0.8 only
// restatements collide.
const DEDUP_SIMILARITY = 0.8;

// Supersede check failure: rolls back the insert and returns a specific message
// instead of toolError's generic one.
class SupersedeError extends Error {}

async function remember(
  args: RememberArgs,
  ctx: { directory: string; sessionID: string },
): Promise<string> {
  try {
    const invalid = validateWrite(args);
    if (invalid) return invalid;
    await ready();

    let supersedesId: number | undefined;
    if (args.supersedes !== undefined) {
      const n = Number(args.supersedes);
      if (!Number.isInteger(n) || n <= 0) {
        return "ERROR: supersedes must be a positive integer (the #id shown by memory_recall).";
      }
      supersedesId = n;
    }

    // Tags are stored verbatim - the project column records origin, not visibility.
    const tags = args.tags ?? [];
    const basename = ctx.directory.split('/').pop() ?? ctx.directory;

    // One transaction: an unreachable supersede target rolls back the insert.
    const newId = await sql.begin(async (tx) => {
      const [{ id }] = await tx`
        INSERT INTO memories (content, tags, session_id, project, memory_type)
        VALUES (${args.content}, ${tagsParam(tx, tags)}, ${ctx.sessionID}, ${ctx.directory}, ${args.type ?? "project_fact"})
        RETURNING id
      ` as { id: number }[];
      if (supersedesId === undefined) return id;

      const linked = await tx`
        UPDATE memories SET superseded_by = ${id}
        WHERE id = ${supersedesId} AND ${visibleRows(tx, ctx.directory)}
        RETURNING id
      ` as { id: number }[];
      if (linked.length === 0) {
        const exists = await tx`SELECT project, memory_type FROM memories WHERE id = ${supersedesId}` as { project: string; memory_type: string }[];
        if (exists.length > 0) {
          throw new SupersedeError(
            `memory #${supersedesId} is a ${exists[0].memory_type} belonging to ${exists[0].project}; cannot supersede it from here. Only that project's agent can mark it superseded.`,
          );
        }
        throw new SupersedeError(`no memory #${supersedesId}; nothing to supersede.`);
      }
      return id;
    });

    // Fire-and-forget: a failure leaves embedding NULL, keyword search keeps
    // working, and the backfill (deploy/backfill.ts) fills the gap later.
    embedInBackground(newId, args.content);

    // Cache is keyed by directory + prompt; clear this directory's keys.
    invalidateInjection(ctx.directory);
    return `Stored memory #${newId} (project ${basename}).${lengthNudge(args.content)}`;
  } catch (e: unknown) {
    if (e instanceof SupersedeError) return `ERROR: ${e.message}`;
    return toolError("remember", "remember", e);
  }
}

// --- Keyword capture ---

// Deterministic capture, no LLM: a trigger phrase in the prompt stores the
// first paragraph after it verbatim (minus the trigger, subject to the code
// and length guards below) through the normal write path.
// Model extraction would make judgment load-bearing on prompt admission.
// Global flag for matchAll (which clones the regex, so no lastIndex leaks).
const MEMORY_TRIGGER_RE =
  /\b(?:remember(?:\s+(?:this|that|to))?|do(?:n'?| no)t forget(?:\s+(?:this|that|to))?|keep (?:this|that )?in mind(?: that)?)\b\s*[:,]?\s*/gi;

// Interrogative follow-ons are questions about the past ("remember when the
// pool broke?"), not storage requests. The list is deliberately narrow:
// "remember that when X happens, do Y" is imperative and must be captured, so
// "when" alone is not enough - only skip the bare question forms.
const INTERROGATIVE_RE = /^(?:when|what|where|why|how|who|whom|whose|which|did)\b/i;

// A trigger only counts as prose, never as code. A candidate is skipped when
// it sits inside a code span or fence (odd backtick
// count before it), is glued to a path/member access (preceded by a backtick,
// dot, or slash), or is a call/code token (followed by "(" even across spaces,
// or by an adjacent backtick - "remember that `bun test` ..." is still prose). The payload stops at the first
// blank line, and a payload over CONTENT_LENGTH_NUDGE is skipped rather than
// truncated - a real "remember that X" is a sentence, not a pasted document.
// Known limitation: a stray unmatched backtick before a genuine trigger
// suppresses capture; the model can still call memory_remember itself.
function extractMemoryRequest(text: string): string | null {
  for (const match of text.matchAll(MEMORY_TRIGGER_RE)) {
    const start = match.index;
    let backticks = 0;
    for (let i = 0; i < start; i++) if (text[i] === "`") backticks++;
    if (backticks % 2 === 1) continue;
    const before = start > 0 ? text[start - 1] : "";
    if (before === "`" || before === "." || before === "/") continue;
    const rest = text.slice(start + match[0].replace(/[\s:,]+$/, "").length);
    if (rest[0] === "`" || rest.trimStart()[0] === "(") continue;
    const payload = text.slice(start + match[0].length).split(/\n\s*\n/, 1)[0].trim();
    if (!payload || INTERROGATIVE_RE.test(payload)) continue;
    if (payload.length > CONTENT_LENGTH_NUDGE) continue;
    return payload;
  }
  return null;
}

// Fire-and-forget: prompt admission must not wait on a database write, and
// the prompt is never mutated. Not idempotent on its own - writes never
// reject as duplicates - so the omp factory calls this once per distinct
// prompt per session (see firstSighting).
async function captureFromPrompt(
  text: string,
  directory: string,
  sessionID: string,
): Promise<void> {
  const content = extractMemoryRequest(text);
  if (!content) return;
  try {
    await remember({ content, tags: ["user-requested"] }, { directory, sessionID });
  } catch (e: unknown) {
    logError("capture", `mempg keyword capture failed: ${e instanceof Error ? e.message : String(e)}`);
  }
}

// project_fact rows are origin-scoped (one customer's agent must not delete
// another's facts); stack_fact is maintainable from any project.
async function forget(
  args: ForgetArgs,
  ctx: { directory: string },
): Promise<string> {
  try {
    const id = Number(args.id);
    if (!Number.isInteger(id) || id <= 0) {
      return "ERROR: id must be a positive integer (the #id shown by memory_recall).";
    }
    await ready();
    const deleted = await sql`
      DELETE FROM memories
      WHERE id = ${id} AND ${visibleRows(sql, ctx.directory)}
      RETURNING id
    ` as { id: number }[];

    if (deleted.length === 0) {
      const exists = await sql`SELECT project, memory_type FROM memories WHERE id = ${id}` as { project: string; memory_type: string }[];
      if (exists.length > 0) {
        return `Memory #${id} is a ${exists[0].memory_type} belonging to ${exists[0].project}; not deleted. Only that project's agent can delete it.`;
      }
      return `No memory #${id}; nothing deleted.`;
    }
    invalidateInjection(ctx.directory);
    return `Deleted memory #${id}.`;
  } catch (e: unknown) {
    return toolError("forget", "forget", e);
  }
}

// created_at is not bumped: the displayed date says when it was learned.
// Omitted tags/type are preserved.
async function updateMemory(
  args: UpdateArgs,
  ctx: { directory: string },
): Promise<string> {
  try {
    const id = Number(args.id);
    if (!Number.isInteger(id) || id <= 0) {
      return "ERROR: id must be a positive integer (the #id shown by memory_recall).";
    }
    const invalid = validateWrite(args);
    if (invalid) return invalid;
    await ready();

    const tags = args.tags ?? [];
    const tagCond = Array.isArray(args.tags)
      ? sql`tags = ${tagsParam(sql, tags)},`
      : sql``;
    const typeCond = args.type !== undefined
      ? sql`memory_type = ${args.type},`
      : sql``;
    const updated = await sql`
      UPDATE memories
      SET content = ${args.content},
          ${tagCond}
          ${typeCond}
          updated_at = ${nowSql(sql)},
          embedding = NULL
      WHERE id = ${id} AND ${visibleRows(sql, ctx.directory)}
      RETURNING id
    ` as { id: number }[];

    if (updated.length === 0) {
      const exists = await sql`SELECT project, memory_type FROM memories WHERE id = ${id}` as { project: string; memory_type: string }[];
      if (exists.length > 0) {
        return `Memory #${id} is a ${exists[0].memory_type} belonging to ${exists[0].project}; not updated. Only that project's agent can edit it.`;
      }
      return `No memory #${id}; nothing updated.`;
    }
    invalidateInjection(ctx.directory);
    // The UPDATE cleared the old vector, so a failed re-embed leaves the row
    // keyword-only rather than matching content it no longer holds.
    embedInBackground(id, args.content);
    return `Updated memory #${id}.${lengthNudge(args.content)}`;
  } catch (e: unknown) {
    return toolError("update", "update", e);
  }
}

// Cosine threshold for consolidate's meaning pass, calibrated for
// embeddinggemma:300m by the bench/ consolidate benches; re-run them on a model change.
const CONSOLIDATE_EMBED_THRESHOLD = 0.83;

// Mirrors visibleRows(): a project_fact clusters only within its own project,
// so consolidation never crosses a boundary forget/update refuse to cross.
function mutuallyVisible(client: SQL) {
  return client`((a.memory_type != 'project_fact' AND b.memory_type != 'project_fact') OR a.project = b.project)`;
}

// TS twin of mutuallyVisible() for the wording pass. SQL's `a.project =
// b.project` is NULL-unsafe; mirrored explicitly since JS `null === null` is true.
function mutuallyVisibleRows(
  a: { memory_type: string; project: string | null },
  b: { memory_type: string; project: string | null },
): boolean {
  if (a.memory_type !== "project_fact" && b.memory_type !== "project_fact") return true;
  return a.project !== null && b.project !== null && a.project === b.project;
}

// Excludes structurally templated auto-logs from the meaning pass: background-
// task status logs, session-compaction summaries and per-app migration
// checklists score high on cosine between genuinely different facts, because
// only a few words vary. Matched by content, not the `auto_capture` tag, which
// sits on correct and incorrect merges alike. Structurally parallel natural
// language (two different systemd units described alike) still false-merges
// sometimes; no pattern separates that from true duplicates.
// isTemplatedContent is the row filter on both backends; the SQL pair form
// prefilters Postgres' O(n^2) self-join. ILIKE's `_` is one wildcard
// character, and Postgres' `.` matches newlines, hence [\s\S].
function isTemplatedAutoLogPair(client: SQL) {
  return client`(
    a.content ILIKE '%background task bg_%' OR b.content ILIKE '%background task bg_%'
    OR a.content ILIKE '%session compacting for project%' OR b.content ILIKE '%session compacting for project%'
    OR a.content ~ 'For APP.{0,3}[0-9]+.{0,3}\\(' OR b.content ~ 'For APP.{0,3}[0-9]+.{0,3}\\('
  )`;
}

const TEMPLATED_AUTO_LOG = [/background task bg[\s\S]/i, /session compacting for project/i, /For APP[\s\S]{0,3}[0-9]+[\s\S]{0,3}\(/];
function isTemplatedContent(content: string): boolean {
  return TEMPLATED_AUTO_LOG.some((re) => re.test(content));
}

// No cosine threshold separates "same fact, reworded" from "same shape,
// different fact" (100 vs 500 req/min), so the meaning pass also compares
// numbers, paths and capitalized names, extracted by regex (no model call).
type DetailSet = {
  numbers: Set<string>;
  paths: Set<string>;
  properNouns: Set<string>;
};

// Deliberately coarse: lowercase names like "bge-m3" are not caught.
function extractDetails(content: string): DetailSet {
  const numbers = new Set<string>();
  for (const m of content.matchAll(/\b\d[\d,.:/-]*\b/g)) {
    const v = m[0].replace(/[.,:/-]+$/, "");
    if (v) numbers.add(v);
  }

  const paths = new Set<string>();
  for (const m of content.matchAll(/(?:~|\.{1,2})?\/[^\s,;:()]+/g)) {
    const v = m[0].replace(/[.,;:]+$/, "");
    if (v.length > 1) paths.add(v);
  }

  const properNouns = new Set<string>();
  for (const sentence of content.split(/(?<=[.!?])\s+/)) {
    const words = sentence.trim().split(/\s+/);
    // Skip index 0: a capitalized sentence-initial word is not distinctive
    // (every sentence starts capitalized regardless of content).
    for (let i = 1; i < words.length; i++) {
      const w = words[i].replace(/^[^A-Za-z]+|[^A-Za-z0-9-]+$/g, "");
      if (w.length > 1 && /^[A-Z][a-zA-Z0-9-]*$/.test(w)) properNouns.add(w);
    }
  }

  return { numbers, paths, properNouns };
}

// Conflicts only when both sides have values and none are `equivalent`;
// absence on one side has nothing to disagree with.
function categoryConflict(a: Set<string>, b: Set<string>, equivalent: (x: string, y: string) => boolean): boolean {
  if (a.size === 0 || b.size === 0) return false;
  for (const x of a) for (const y of b) if (equivalent(x, y)) return false;
  return true;
}

// "1,000" equals "1000"; a version prefix ("2.5" vs "2.5.0") is imprecision,
// but "2.5" vs "3.0" is a real change.
const PLAIN_NUMBER = /^\d+(\.\d+)?$/;
const VERSION_SHAPED = /^\d+(\.\d+)+$/;

function isVersionPrefix(x: string, y: string): boolean {
  const xs = x.split(".");
  const ys = y.split(".");
  if (xs.length >= ys.length) return false;
  return xs.every((seg, i) => seg === ys[i]);
}

function numbersEquivalent(x: string, y: string): boolean {
  const nx = x.replace(/,/g, "");
  const ny = y.replace(/,/g, "");
  if (nx === ny) return true;
  if (PLAIN_NUMBER.test(nx) && PLAIN_NUMBER.test(ny)) return Number(nx) === Number(ny);
  if (VERSION_SHAPED.test(nx) && VERSION_SHAPED.test(ny)) return isVersionPrefix(nx, ny) || isVersionPrefix(ny, nx);
  return false;
}

// Trailing slash ignored; case is significant on Linux paths.
function pathsEquivalent(x: string, y: string): boolean {
  const strip = (v: string) => (v.length > 1 ? v.replace(/\/+$/, "") || v : v);
  return strip(x) === strip(y);
}

// Proper nouns conflict on identity, not on how they happen to be cased
// ("Jira" vs "JIRA" is the same product name).
function properNounsEquivalent(x: string, y: string): boolean {
  return x.toLowerCase() === y.toLowerCase();
}

// One reason per conflicting category, quoting raw (not normalized) values.
function detailConflicts(a: DetailSet, b: DetailSet): string[] {
  const reasons: string[] = [];
  if (categoryConflict(a.numbers, b.numbers, numbersEquivalent)) {
    reasons.push(`numbers differ (${[...a.numbers].join(", ")} vs ${[...b.numbers].join(", ")})`);
  }
  if (categoryConflict(a.paths, b.paths, pathsEquivalent)) {
    reasons.push(`paths differ (${[...a.paths].join(", ")} vs ${[...b.paths].join(", ")})`);
  }
  if (categoryConflict(a.properNouns, b.properNouns, properNounsEquivalent)) {
    reasons.push(`names differ (${[...a.properNouns].join(", ")} vs ${[...b.properNouns].join(", ")})`);
  }
  return reasons;
}

// Deterministic: keep the newest of each near-duplicate cluster, delete the rest,
// and return deleted texts so the calling agent can merge unique facts back.
// Pass 1 is trigram (wording), pass 2 embedding cosine (meaning), each capped at
// 25 clusters; pass 2 routes detailConflicts pairs to [meaning-uncertain].
async function consolidate(args: ConsolidateArgs = {}): Promise<string> {
  const dryRun = args.dryRun === true;
  const verb = dryRun ? "would remove" : "removed";
  try {
    await ready();
    // --- Pass 1: wording (trigram similarity over content) ---
    // Superseded rows are resolved history, not accidental duplicates.
    const rows = await sql`
      SELECT id, content, coalesce(tags, '{}') AS tags, created_at, memory_type, project
      FROM memories
      WHERE superseded_by IS NULL
      ORDER BY created_at DESC, id DESC
    ` as { id: number; content: string; tags: string[] | string; created_at: Date | string; memory_type: string; project: string | null }[];

    // Greedy newest-first clustering against each cluster's newest member.
    // Trigram sets are built once; a size-ratio prefilter skips hopeless pairs.
    type Entry = { row: (typeof rows)[number]; set: Set<string> };
    const entries: Entry[] = rows.map((row) => ({ row, set: trigrams(row.content) }));
    const clusters: Array<Array<Entry>> = [];
    for (const entry of entries) {
      const host = clusters.find((c) => {
        // Checking the anchor alone keeps the whole cluster mutually visible.
        if (!mutuallyVisibleRows(c[0].row, entry.row)) return false;
        const ra = c[0].set.size;
        const rb = entry.set.size;
        // Jaccard >= 0.8 is impossible when one set is much smaller; the
        // comparison itself is the expensive part, so prefilter on sizes.
        if (ra === 0 || rb === 0 || ra > rb * 4 || rb > ra * 4) return false;
        return nearDupeSets(c[0].set, entry.set);
      });
      if (host) host.push(entry);
      else clusters.push([entry]);
    }

    const wordingGroups = clusters.filter((c) => c.length > 1).slice(0, 25);

    let removed = 0;
    let removedGroups = 0;
    let uncertainPairs = 0;
    const report: string[] = [];
    // Tracked even in dryRun so pass 2 sees the same candidates a real run would.
    const wordingRemovedIds: number[] = [];
    for (const cluster of wordingGroups) {
      const survivor = cluster[0].row;
      const removedRows = cluster.slice(1).map((e) => e.row);
      wordingRemovedIds.push(...removedRows.map((r) => r.id));
      removed += removedRows.length;
      removedGroups++;
      report.push(
        `[wording] Kept #${survivor.id}: ${truncateMemory(survivor.content)}\n` +
          removedRows.map((r) => `  ${verb} #${r.id}: ${truncateMemory(r.content)}`).join("\n"),
      );
    }
    if (!dryRun && wordingRemovedIds.length > 0) {
      await sql`DELETE FROM memories WHERE id ${idIn(sql, wordingRemovedIds)}`;
    }

    // --- Pass 2: meaning (embedding cosine similarity) ---
    // Only embedded rows the wording pass left behind, never templated
    // auto-logs. SQLite has no <=>, so it compares in TS (O(n^2), like pass 1).
    type EmbedRow = { id: number; content: string; created_at: Date | string };
    let embedRows: EmbedRow[] | undefined;
    let embedPairs: { a_id: number; b_id: number }[];
    if (SQLITE) {
      const candidates = ((await sql`
        SELECT id, content, created_at, memory_type, project, embedding
        FROM memories
        WHERE embedding IS NOT NULL AND superseded_by IS NULL
          AND id NOT IN (SELECT value FROM json_each(${JSON.stringify(wordingRemovedIds)}))
        ORDER BY created_at DESC, id DESC
      `) as (EmbedRow & { memory_type: string; project: string | null; embedding: Uint8Array })[]).filter(
        (r) => !isTemplatedContent(r.content),
      );
      const vecs = candidates.map((r) => asFloat32(r.embedding));
      embedPairs = [];
      for (let i = 0; i < candidates.length; i++) {
        for (let j = i + 1; j < candidates.length; j++) {
          const [a, b] = [candidates[i], candidates[j]];
          if (vecs[i].length !== vecs[j].length || !mutuallyVisibleRows(a, b)) continue;
          if (cosine(vecs[i], vecs[j]) < CONSOLIDATE_EMBED_THRESHOLD) continue;
          embedPairs.push(a.id < b.id ? { a_id: a.id, b_id: b.id } : { a_id: b.id, b_id: a.id });
        }
      }
      embedRows = candidates;
    } else {
      embedPairs = (await sql`
        SELECT a.id AS a_id, b.id AS b_id
        FROM memories a
        JOIN memories b ON a.id < b.id
        WHERE a.embedding IS NOT NULL
          AND b.embedding IS NOT NULL
          AND a.superseded_by IS NULL
          AND b.superseded_by IS NULL
          AND a.id <> ALL(${sql.array(wordingRemovedIds, "int8")})
          AND b.id <> ALL(${sql.array(wordingRemovedIds, "int8")})
          AND (1 - (a.embedding <=> b.embedding)) >= ${CONSOLIDATE_EMBED_THRESHOLD}
          AND ${mutuallyVisible(sql)}
          AND NOT ${isTemplatedAutoLogPair(sql)}
      `) as { a_id: number; b_id: number }[];
    }

    if (embedPairs.length > 0) {
      const linked = new Set<string>();
      for (const p of embedPairs) linked.add(`${p.a_id}:${p.b_id}`);
      const isLinked = (x: number, y: number) => (x < y ? linked.has(`${x}:${y}`) : linked.has(`${y}:${x}`));

      embedRows ??= ((await sql`
        SELECT id, content, created_at
        FROM memories
        WHERE embedding IS NOT NULL AND superseded_by IS NULL
          AND id <> ALL(${sql.array(wordingRemovedIds, "int8")})
        ORDER BY created_at DESC, id DESC
      `) as EmbedRow[]).filter((r) => !isTemplatedContent(r.content));

      // Same greedy "newest anchors a cluster" shape as the wording pass,
      // matched by the pair list above instead of a text-similarity test.
      const embedClusters: Array<Array<(typeof embedRows)[number]>> = [];
      for (const row of embedRows) {
        const host = embedClusters.find((c) => isLinked(c[0].id, row.id));
        if (host) host.push(row);
        else embedClusters.push([row]);
      }
      const meaningGroups = embedClusters.filter((c) => c.length > 1).slice(0, 25);

      for (const cluster of meaningGroups) {
        const survivor = cluster[0];
        const survivorDetails = extractDetails(survivor.content);
        // A detail-conflicting member leaves the auto-merge alone; the rest
        // of the cluster still merges.
        const clean: (typeof cluster)[number][] = [];
        const uncertain: Array<{ row: (typeof cluster)[number]; reasons: string[] }> = [];
        for (const row of cluster.slice(1)) {
          const reasons = detailConflicts(survivorDetails, extractDetails(row.content));
          if (reasons.length > 0) uncertain.push({ row, reasons });
          else clean.push(row);
        }

        if (!dryRun && clean.length > 0) {
          await sql`DELETE FROM memories WHERE id ${idIn(sql, clean.map((r) => r.id))}`;
        }
        removed += clean.length;
        if (clean.length > 0) {
          removedGroups++;
          report.push(
            `[meaning] Kept #${survivor.id}: ${truncateMemory(survivor.content)}\n` +
              clean.map((r) => `  ${verb} #${r.id}: ${truncateMemory(r.content)}`).join("\n"),
          );
        }
        for (const u of uncertain) {
          uncertainPairs++;
          report.push(
            `[meaning-uncertain] #${survivor.id} vs #${u.row.id} - high similarity but ${u.reasons.join("; ")}; not merged, review manually.\n` +
              `  #${survivor.id}: ${truncateMemory(survivor.content)}\n` +
              `  #${u.row.id}: ${truncateMemory(u.row.content)}`,
          );
        }
      }
    }

    if (report.length === 0) return "No duplicates found; nothing to consolidate.";

    if (removed > 0 && !dryRun) injectionCache.clear();
    const summary: string[] = [];
    if (dryRun) {
      summary.push("DRY RUN - nothing was deleted. Re-run without dryRun (or with dryRun: false) to actually remove these.");
    }
    if (removed > 0) {
      summary.push(
        `${dryRun ? "Would remove" : "Removed"} ${removed} duplicate ${removed === 1 ? "memory" : "memories"} across ${removedGroups} group${removedGroups === 1 ? "" : "s"} ` +
          `(kept the newest of each; [wording] = matched by trigram similarity, [meaning] = matched by embedding similarity).`,
      );
      if (!dryRun) {
        summary.push("Check the removed texts - if any carries a fact the kept memory lacks, merge it in with memory_update:");
      }
    }
    if (uncertainPairs > 0) {
      summary.push(
        `${uncertainPairs} similar pair${uncertainPairs === 1 ? "" : "s"} flagged [meaning-uncertain]: high embedding similarity but a specific ` +
          `number, path, or name differs, so nothing was auto-merged - review each and use memory_update to merge if it's genuinely the same ` +
          `fact, or leave both if they're distinct.`,
      );
    }
    return `${summary.join("\n")}\n\n${report.join("\n")}`;
  } catch (e: unknown) {
    return toolError("consolidate", "consolidate", e);
  }
}

// Clears every cache entry for the directory - the cache keys by directory +
// prompt hash, so a write invalidates them all.
function invalidateInjection(directory: string): void {
  for (const key of [...injectionCache.keys()]) {
    if (key.startsWith(`${directory}\u0001`)) injectionCache.delete(key);
  }
}

// --- Tool table: host-neutral, so the descriptions (which carry the write
// policy the model sees) and input schemas exist exactly once. ---

type ToolJsonSchema = {
  type: "object" | "string" | "number" | "boolean" | "array";
  description?: string;
  properties?: Record<string, ToolJsonSchema>;
  required?: readonly string[];
  additionalProperties?: boolean;
  items?: ToolJsonSchema;
  enum?: readonly string[];
  maxItems?: number;
  maxLength?: number;
};
type ToolCtx = { directory: string; sessionID: string };
type ToolSpec = {
  name: string;
  label: string;
  readOnly: boolean; // omp approval tier: true -> "read", false -> "write"
  description: string;
  input: ToolJsonSchema;
  run: (input: unknown, ctx: ToolCtx) => Promise<string>;
};

const TOOL_SPECS: readonly ToolSpec[] = [
  {
    name: "memory_recall",
    label: "Memory Recall",
    readOnly: true,
    // Essential (always-loaded) tool: memory ops are single-shot calls, not
    // scriptable sequences - hiding them behind discovery only breaks direct
    // invocation without adding value.
    description:
      "Search past memories - your own record of previous sessions, which you otherwise have no access to. " +
      "Search before you debug an error or failing command, change config/infra/deploy/CI, make a design or " +
      "tool choice, or start work in an area you have not touched this session: a past attempt, decision, " +
      "or fix is almost always cheaper to find than to rediscover. Matches on both keywords and meaning, so " +
      "approximate phrasing works. The stack_fact type is always searched; this project's project_fact " +
      "memories are searched by default. An empty query returns the most recent visible memories (recency " +
      "browse mode) rather than an empty result - useful for \"show me the last N memories\" without a " +
      "specific search term.",
    input: {
      type: "object",
      properties: {
        query: { type: "string", description: "Search string (keyword + semantic paraphrase match); omit for the latest memories" },
        tags: {
          type: "array",
          items: { type: "string" },
          description:
            "Only return memories carrying all of these tags. Tags are not full-text " +
            "searchable (search covers content only), so this filter is the only way to reach them.",
        },
        global: {
          type: "boolean",
          description:
            "Also search other projects' project_fact memories (default: only this " +
            "project's project_fact memories, plus all stack_fact memories, " +
            "which are always global).",
        },
        includeSuperseded: {
          type: "boolean",
          description:
            "Include memories that have been superseded by a newer one (default: false, " +
            "hidden). Each included row is annotated with what replaced it - use this to " +
            "review history, not for everyday recall.",
        },
        limit: { type: "number", description: "1-20, default 10" },
      },
      additionalProperties: false,
    },
    run: (input, ctx) => recall(input as RecallArgs, ctx),
  },
  {
    name: "memory_remember",
    label: "Memory Remember",
    readOnly: false,
    // The only place the write policy always reaches the model: injection can fail.
    description:
      "Store a durable memory. This is the only way anything you learn survives past " +
      "this session - unwritten context is lost permanently when the session ends, so " +
      "write immediately rather than deferring to the end of a task. " +
      "stack_fact is shared across all projects; project_fact (the default) is visible " +
      "only in this project unless recalled with global: true. " +
      "Call this whenever the user corrects you or states a preference, you discover a " +
      "non-obvious reason or constraint, you solve something that took more than one " +
      "attempt, or you learn an environment fact not visible in the code. " +
      "Do not store session progress, secrets, or anything the code itself already states. " +
      "Duplicate writes are never rejected - run memory_consolidate afterward to clean up " +
      "near-duplicates if the corpus has accumulated restatements of one fact. " +
      "If this memory corrects, replaces, or reverses an earlier one, pass supersedes: <id> " +
      "(get the id from memory_recall) instead of just narrating the change in prose. The " +
      "old memory is then excluded from normal recall and injection, but stays in history " +
      "rather than being deleted - use memory_recall with includeSuperseded: true to see it.",
    input: {
      type: "object",
      properties: {
        content: {
          type: "string",
          description: `1-3 self-contained sentences capturing the why (${MIN_CONTENT}-${MAX_CONTENT} characters)`,
        },
        type: {
          type: "string",
          enum: [...MEMORY_TYPES],
          description:
            "stack_fact = true about the tooling/stack itself, portable to any project " +
            "using the same stack (e.g. a Terraform module quirk, an ArgoCD gotcha, a " +
            "Helm chart convention) - global. " +
            "project_fact (default) = true about THIS specific project/customer only " +
            "(an environment quirk, a customer's specific request, a one-off workaround) " +
            "- visible only in this project unless the caller asks for global search. " +
            "Test: would this fact help in a different customer's repo using the same " +
            "tools? If yes, stack_fact. If no, project_fact.",
        },
        tags: {
          type: "array",
          items: { type: "string", maxLength: MAX_TAG_LENGTH },
          maxItems: MAX_TAGS,
          description:
            "Fine-grained facets: decision, debug, env, architecture, workaround, " +
            "language:<x>, framework:<x>, tool:<x>. The origin project is recorded " +
            "automatically (a project column, not a tag) - never add project:<name>.",
        },
        supersedes: {
          type: "number",
          description:
            "The #id (from memory_recall) of an earlier memory this one corrects, " +
            "replaces, or reverses. Same project boundary as memory_forget/memory_update: " +
            "a foreign project's project_fact cannot be superseded from here.",
        },
      },
      required: ["content"],
      additionalProperties: false,
    },
    run: (input, ctx) => remember(input as RememberArgs, ctx),
  },
  {
    name: "memory_forget",
    label: "Memory Forget",
    readOnly: false,
    description:
      "Delete a memory by id (get ids from memory_recall). Use for memories that are wrong or obsolete; prefer storing a corrected memory when the old one is still useful history. " +
      "stack_fact can be deleted from any project; project_fact can only be " +
      "deleted by its origin project (the delete will fail with the owning project's name).",
    input: {
      type: "object",
      properties: {
        id: { type: "number", description: "The #id shown by memory_recall" },
      },
      required: ["id"],
      additionalProperties: false,
    },
    run: (input, ctx) => forget(input as ForgetArgs, { directory: ctx.directory }),
  },
  {
    name: "memory_update",
    label: "Memory Update",
    readOnly: false,
    description:
      "Rewrite an existing memory by id (get ids from memory_recall). Use when a memory is outdated but still worth keeping: the corrected content replaces the old, keeping the original learned date. " +
      "Omitted tags/type are kept as-is. stack_fact can be edited from any " +
      "project; project_fact can only be edited by its origin project. " +
      "For obsolete memories use memory_forget; for genuinely new memories use memory_remember.",
    input: {
      type: "object",
      properties: {
        id: { type: "number", description: "The #id shown by memory_recall" },
        content: {
          type: "string",
          description: `1-3 self-contained sentences replacing the old content (${MIN_CONTENT}-${MAX_CONTENT} characters)`,
        },
        tags: {
          type: "array",
          items: { type: "string", maxLength: MAX_TAG_LENGTH },
          maxItems: MAX_TAGS,
          description: "Replaces the tag list; omit to keep the current tags",
        },
        type: {
          type: "string",
          enum: [...MEMORY_TYPES],
          description: "Replaces the memory type; omit to keep the current type",
        },
      },
      required: ["id", "content"],
      additionalProperties: false,
    },
    run: (input, ctx) => updateMemory(input as UpdateArgs, { directory: ctx.directory }),
  },
  {
    name: "memory_consolidate",
    label: "Memory Consolidate",
    readOnly: false,
    description:
      "Remove near-duplicate memories: keeps the newest of each near-duplicate group anywhere in the store and deletes the rest, returning the removed texts. " +
      "Finds duplicates two ways - matching wording (trigram similarity) and matching meaning (embedding similarity, catches the same fact stated in different words). " +
      "A meaning-level pair whose numbers, paths, or names conflict despite high similarity is flagged as [meaning-uncertain] instead of merged - review it and use memory_update yourself. " +
      "After running it, merge any unique fact from the removed texts into the kept memory via memory_update. Deterministic - run it when the user asks to tidy or consolidate memories. " +
      "Pass dryRun: true to preview what would be removed without deleting anything - review the output, then call again without dryRun to commit. " +
      "The preview reflects the corpus at the moment it runs; if memories are added in between, a follow-up real call re-evaluates independently and may not match exactly.",
    input: {
      type: "object",
      properties: {
        dryRun: {
          type: "boolean",
          description: "Preview what would be removed without deleting anything.",
        },
      },
      additionalProperties: false,
    },
    run: (input) => consolidate(input as ConsolidateArgs),
  },
  {
    name: "memory_tags",
    label: "Memory Tags",
    readOnly: true,
    description:
      "List tags currently in use, with counts, most-used first. Check this before writing a new tag to reuse " +
      "an established one instead of minting a near-duplicate (e.g. 'postgres' vs 'tool:postgres'). Read-only, no side effects.",
    input: {
      type: "object",
      properties: {
        limit: { type: "number", description: "1-500, default 200" },
        global: {
          type: "boolean",
          description:
            "Also count tags from other projects' project_fact memories (default: only this " +
            "project's project_fact memories, plus all stack_fact memories, which are always global).",
        },
      },
      additionalProperties: false,
    },
    run: (input, ctx) => listTags(input as TagsArgs, { directory: ctx.directory }),
  },
  {
    name: "memory_retag",
    label: "Memory Retag",
    readOnly: false,
    description:
      "Rename a tag across every memory that has it - e.g. after memory_tags shows both 'postgres' and " +
      "'tool:postgres' exist, use this to collapse them into one. Only affects tags; content and type are " +
      "untouched. project_fact memories can only be retagged from their origin project, same as " +
      "memory_forget/memory_update.",
    input: {
      type: "object",
      properties: {
        old: { type: "string", description: "The existing tag to rename." },
        new: { type: "string", description: "The tag to rename it to." },
      },
      required: ["old", "new"],
      additionalProperties: false,
    },
    run: (input, ctx) => retag(input as RetagArgs, { directory: ctx.directory }),
  },
];

// --- omp schema conversion: omp tools take omptype (zod-like) schemas, the
// table is JSON Schema. Specs are static, so an unsupported shape throws at
// load (and `omp plugin install` rolls back) rather than at call time. ---

type Zod = Pick<ExtensionAPI["zod"], "object" | "string" | "number" | "boolean" | "array" | "enum">;
type ZodLike = ZodLikeSchema<unknown>;

function toZod(z: Zod, s: ToolJsonSchema): ZodLike {
  // Widened before the switch so the exhaustive default can still name it.
  const kind: string = s.type;
  let node: ZodLike;
  switch (s.type) {
    case "string": {
      // z.enum wants a non-empty tuple; MEMORY_TYPES is a non-empty literal list.
      node = s.enum ? z.enum(s.enum as readonly [string, ...string[]]) : z.string();
      if (s.maxLength !== undefined) node = node.max(s.maxLength);
      break;
    }
    case "number":
      node = z.number();
      break;
    case "boolean":
      node = z.boolean();
      break;
    case "array": {
      if (s.items === undefined) throw new Error("mempg: array schema without items");
      node = z.array(toZod(z, s.items));
      if (s.maxItems !== undefined) node = node.max(s.maxItems);
      break;
    }
    case "object": {
      const required = s.required ?? [];
      const shape: Record<string, ZodLike> = {};
      for (const [key, child] of Object.entries(s.properties ?? {})) {
        const field = toZod(z, child);
        shape[key] = required.includes(key) ? field : field.optional();
      }
      node = z.object(shape);
      if (s.additionalProperties === false) node = node.strict();
      break;
    }
    default:
      throw new Error(`mempg: unsupported tool schema type ${kind}`);
  }
  // Described before the parent applies .optional(), so optional fields keep it.
  return s.description === undefined ? node : node.describe(s.description);
}

// --- Capture bookkeeping: module-level so session shutdown (and tests) can
// await in-flight fire-and-forget writes. captureFromPrompt never rejects. ---

const CAPTURE_NOTE =
  "<mempg-capture>The user's \"remember ...\" request in this prompt has been queued for " +
  "automatic verbatim storage by mempg. Do not call memory_remember for that captured request; " +
  "otherwise respond to the user's full prompt normally, including any other tasks it asks for.</mempg-capture>";

// --- session_stop memory checkpoint: a deterministic nudge, not a judgment.
// Prompt text alone does not make models write memory, so busy turns that
// wrote nothing get one hidden continuation asking them to. The gate is
// counted, never model-evaluated; the model only decides WHAT to store. ---

const CHECKPOINT_MIN_TOOL_CALLS = 8;
const CHECKPOINT_NOTE =
  "<mempg-checkpoint>Before this turn ends, check whether it produced anything worth keeping: " +
  WRITE_TRIGGERS.join("; ") +
  ". Store each such fact now with memory_remember (one fact per call, 1-3 sentences). " +
  "If nothing qualifies, reply only \"No new memories.\"</mempg-checkpoint>";

function isObject(x: unknown): x is Record<string, unknown> {
  return typeof x === "object" && x !== null;
}

// Counts tool calls since the latest prompt; skips turns that already wrote
// memory, including via keyword capture (so a captured fact isn't written twice).
function needsMemoryCheckpoint(messages: readonly unknown[]): boolean {
  let lastUser = -1;
  let promptContent: unknown;
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (isObject(m) && m.role === "user") {
      lastUser = i;
      promptContent = m.content;
      break;
    }
  }
  if (lastUser === -1) return false;
  const promptText = typeof promptContent === "string"
    ? promptContent
    : Array.isArray(promptContent)
      ? promptContent.map((b) => (isObject(b) && b.type === "text" && typeof b.text === "string" ? b.text : "")).join("\n")
      : "";
  if (extractMemoryRequest(promptText) !== null) return false;
  let calls = 0;
  for (let i = lastUser + 1; i < messages.length; i++) {
    const m = messages[i];
    if (!isObject(m) || m.role !== "assistant" || !Array.isArray(m.content)) continue;
    for (const block of m.content) {
      if (!isObject(block) || block.type !== "toolCall" || typeof block.name !== "string") continue;
      if (block.name === "memory_remember" || block.name === "memory_update") return false;
      calls++;
    }
  }
  return calls >= CHECKPOINT_MIN_TOOL_CALLS;
}

const inflightCaptures = new Set<Promise<void>>();

async function settleCaptures(): Promise<void> {
  await Promise.all(inflightCaptures);
}

// omp extension factory: runs once per session (main + every subagent) against
// this one module instance. Registers only; nothing touches the DB at load.
function mempg(pi: ExtensionAPI): void {
  setLogSink((line) => pi.logger.error(line));

  retain();
  let released = false;
  pi.on("session_shutdown", async () => {
    if (released) return;
    released = true;
    await settleCaptures();
    await dispose();
  });

  const seenPrompts = new Set<string>();
  const SEEN_PROMPTS_MAX = 32;
  // A before_agent_start chain can re-run for one submission; capture must
  // write once per distinct prompt per session.
  const firstSighting = (prompt: string): boolean => {
    const key = Bun.hash(prompt).toString(36);
    if (seenPrompts.has(key)) return false;
    seenPrompts.add(key);
    if (seenPrompts.size > SEEN_PROMPTS_MAX) {
      const oldest = seenPrompts.values().next().value;
      if (oldest !== undefined) seenPrompts.delete(oldest);
    }
    return true;
  };

  pi.on("session_start", () => {
    // Open the pool before the first prompt needs it: Bun connects lazily, so
    // otherwise the TCP + SCRAM handshake (or SQLite's schema setup) is paid
    // inside the first injection.
    void ready().then(() => sql`SELECT 1`).catch(() => {});
    // Warm the embed model so the first hybrid query isn't a cold load.
    void embed(["mempg session warmup"], EMBED_WRITE_TIMEOUT_MS);
  });

  pi.on("before_agent_start", async (event, ctx) => {
    // Injection: extend the event's systemPrompt (it carries earlier
    // extensions' chained overrides); cwd is read per event since /move changes it.
    const output: { system: string[] } = { system: [] };
    // Keyword capture, main session only: subagent prompts aren't the user's.
    if (ctx.agent.kind === "main" && extractMemoryRequest(event.prompt) !== null) {
      if (firstSighting(event.prompt)) {
        track(inflightCaptures, captureFromPrompt(event.prompt, ctx.cwd, ctx.sessionManager.getSessionId()));
      }
      // memory_remember is always loaded and its description says to write on
      // user statements, so without this the model stores the same request a
      // second time. Emitted on re-runs too: each run rebuilds the prompt.
      output.system.push(CAPTURE_NOTE);
    }
    await handleTransform(output, ctx.cwd, capPromptQuery(event.prompt));
    if (output.system.length === 0) return;
    return { systemPrompt: [...event.systemPrompt, ...output.system] };
  });

  // Main session only (omp never fires session_stop for subagents).
  // stop_hook_active is true inside our own continuation, so this fires at
  // most once per user turn and cannot loop.
  pi.on("session_stop", (event) => {
    if (event.stop_hook_active) return;
    if (!needsMemoryCheckpoint(event.messages)) return;
    return { continue: true, additionalContext: CHECKPOINT_NOTE };
  });

  // Essential on all seven: the descriptions tell the model to call
  // memory_recall/memory_remember unprompted, which discovery-only defeats.
  for (const spec of TOOL_SPECS) {
    pi.registerTool({
      name: spec.name,
      label: spec.label,
      description: spec.description,
      parameters: toZod(pi.zod, spec.input),
      loadMode: "essential",
      approval: spec.readOnly ? "read" : "write",
      async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
        const text = await spec.run(params, { directory: ctx.cwd, sessionID: ctx.sessionManager.getSessionId() });
        return { content: [{ type: "text", text }] };
      },
    });
  }
}

const __internals = {
  sql,
  ready,
  withDeadline,
  formatBlock,
  handleTransform,
  recall,
  remember,
  forget,
  updateMemory,
  consolidate,
  listTags,
  retag,
  extractMemoryRequest,
  needsMemoryCheckpoint,
  captureFromPrompt,
  settleCaptures,
  settleEmbeddings,
  TOOL_SPECS,
  toZod,
  invalidateInjection,
  resolveSslMode,
  clampInt,
  validateWrite,
  logError,
  setLogSink,
  toolError,
  resetRateLimit,
  get injectionMode() {
    return injectionMode;
  },
  setInjectionMode(mode: "relevance" | "recency") {
    injectionMode = mode;
  },
  capPromptQuery,
  buildRecencyQuery,
  embed,
  embedAndStore,
  CONSOLIDATE_EMBED_THRESHOLD,
  mutuallyVisibleRows,
  isTemplatedAutoLogPair,
  isTemplatedContent,
  vectorParam,
  extractDetails,
  detailConflicts,
  rrfMerge,
  hybridMerge,
  get ollamaBase() {
    return ollamaBase;
  },
  // Test hook: point the embed client at a dead endpoint to exercise the
  // keyword-only degradation paths (module state).
  setOllamaBase(base: string) {
    ollamaBase = base;
  },
  retain,
  dispose,
};

// Test internals ride on the default export; module exports stay default-only.
export default Object.assign(mempg, { __internals }) as typeof mempg & {
  __internals: typeof __internals;
};
