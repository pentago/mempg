// mempg - Postgres-backed persistent memory extension for oh-my-pi (omp).
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

const sql = makeSql(defaultConfig);

export interface MemoryRow {
  id: number;
  content: string;
  tags: string[] | null;
  project: string;
  date: string;
}

// Injected lines render each memory's origin project (memories are global),
// so the injection queries must select project; id is never rendered.
type InjectionRow = Pick<MemoryRow, "content" | "tags" | "date" | "project">;

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

class DeadlineError extends Error {
  constructor(ms: number) {
    super(`query exceeded ${ms}ms deadline`);
    this.name = "DeadlineError";
  }
}

// Guards the injection query, which runs in front of every model request: a
// hung database must degrade to "no memories" rather than stall the turn.
//
// This races instead of cancelling. Bun documents query.cancel(), but on bun
// 1.4.2 it is a no-op for an in-flight Postgres query - verified against
// SELECT pg_sleep(5), which ran the full 5s under both .execute()+.cancel()
// and bare .cancel(). So the query is abandoned, not aborted: the caller is
// freed on time while the connection stays busy until the server finishes.
async function withDeadline<T>(query: PromiseLike<T>, ms: number): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const deadline = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new DeadlineError(ms)), ms);
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

// embeddinggemma:300m via Ollama over HTTP (host-installed or the compose service,
// same pattern as the bench). Everything here degrades to keyword-only on any
// failure; embeddings can never break search. Env-only, resolved once at init
// like the DB config. The model name is an env var so it is swappable without
// a code change - a model with different dimensions needs the column re-created
// and a re-embed (deploy/README.md).
const OLLAMA_HOST = process.env.MEMPG_OLLAMA_HOST || "localhost";
const OLLAMA_PORT = Number(process.env.MEMPG_OLLAMA_PORT) || 11434;
const EMBED_MODEL = process.env.MEMPG_EMBED_MODEL || "embeddinggemma:300m";
let ollamaBase = `http://${OLLAMA_HOST}:${OLLAMA_PORT}`;

// Warm bge-m3 embed is ~20ms (measured 2026-09-18, RTX 5070); a cold model
// load is ~2-3s, over the injection deadline - so session_start fires a warm-up and
// every request passes keep_alive to keep the model resident. The query budget
// only trips when racing that warm-up or a wedged daemon, and both fall back
// to keyword-only. The write path is fire-and-forget and tolerates cold loads.
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

// pgvector literal, e.g. "[0.1,-0.2,...]", sent as a text param cast to vector.
function vectorLiteral(v: number[]): string {
  return `[${v.join(",")}]`;
}

// Reciprocal rank fusion (k=60, the standard constant): a row scores
// 1/(60 + 1-based rank) in each list it appears in, so a hit in both lists
// outranks either list's tail. Keyed on content - identical text is the same
// memory. Bench-verified (2026-09-18 synthetic bench, since removed): recall
// >= both the keyword and the embedding parent at 485/5k/50k rows.
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

// The shipped hybrid composition (slot count benched 2026-09-18): the vector list's top
// `reserved` rows are UNCONDITIONAL, the rest is the RRF merge minus the
// picks. Plain RRF alone dilutes vector hits when the keyword half is noisy
// on the real corpus (equal-weight ties favor keyword-listed rows, and
// weighting the vector side does not help - rows present in both lists are
// boosted at any weight). R=2 was the free knee: real paraphrase hit@5
// 8/20 -> 12/20 with zero measurable synthetic cost (R=3: 13/20 but the first
// direct-recall regression, and 3/5 of the block can be nearest-neighbor
// noise on no-signal prompts). reserved=0 or an empty vector list degenerates
// to the plain merge, so the keyword-only fallbacks need no special case.
function hybridMerge<T extends { content: string }>(keywordRows: T[], vectorRows: T[], reserved = 2): T[] {
  const picked = vectorRows.slice(0, reserved);
  const rest = rrfMerge([keywordRows, vectorRows]).filter((r) => !picked.some((p) => p.content === r.content));
  return [...picked, ...rest];
}

// Off the write path: embed one memory and store its vector. Never throws -
// callers fire-and-forget it, so every failure (Ollama down, missing column)
// lands here. A row whose embedding stays NULL is keyword-only until the
// backfill (deploy/backfill.ts) or a later memory_update fills it.
async function embedAndStore(id: number, content: string): Promise<void> {
  const vecs = await embed([content], EMBED_WRITE_TIMEOUT_MS);
  if (!vecs) return;
  await storeEmbedding(id, vecs[0]);
}

// remember() already holds the fresh embedding when the smart-write prefilter
// ran; storing it directly skips the redundant second embed.
async function storeEmbedding(id: number, vec: number[]): Promise<void> {
  try {
    await sql`UPDATE memories SET embedding = ${vectorLiteral(vec)}::vector WHERE id = ${id}`;
  } catch (e: unknown) {
    logError("embed-write", `mempg embedding store failed for #${id}: ${e instanceof Error ? e.message : String(e)}`);
  }
}

// --- Injection pipeline ---

// Injection ranking mode (plan follow-up: relevance over blind recency).
// "relevance" (default) scores all memories - every project - against the
// user's latest prompt via full-text search, falling back to recency when the
// prompt matches nothing; "recency" restores the old last-5 behavior via
// MEMPG_INJECTION=recency. Resolved once at init, env-only.
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

// Approximates pg_trgm similarity in TS: character-trigram Jaccard. Used only
// to collapse near-duplicate rows out of the injection candidates (the
// candidate set is tiny), never for storage decisions.
function trigrams(text: string): Set<string> {
  const s = text.toLowerCase().replace(/\s+/g, " ");
  const out = new Set<string>();
  for (let i = 0; i < s.length - 2; i++) out.add(s.slice(i, i + 3));
  return out;
}

function nearDupe(a: string, b: string): boolean {
  return nearDupeSets(trigrams(a), trigrams(b));
}

function nearDupeSets(A: Set<string>, B: Set<string>): boolean {
  if (A.size === 0 || B.size === 0) return false;
  let inter = 0;
  for (const t of A) if (B.has(t)) inter++;
  return inter / (A.size + B.size - inter) >= DEDUP_SIMILARITY;
}

// Injection fetches a deeper candidate slice (rank-ordered) and greedily drops
// rows that near-dupe an already-kept row, emitting the top 5. Without this,
// duplicate writes (there is no write-time rejection) would fill the 5 slots
// with restatements of one fact.
function collapseDupes(rows: InjectionRow[]): InjectionRow[] {
  const kept: InjectionRow[] = [];
  for (const row of rows) {
    if (kept.length >= 5) break;
    if (!kept.some((k) => nearDupe(k.content, row.content))) kept.push(row);
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
    // Framing goes BEFORE the rows, not after: it must be read as "this is
    // what you already know" before the model sees the list, not skimmed as
    // a footnote once attention has moved on to the rows themselves.
    "This is your memory of this project across sessions. You have no other",
    "access to it - anything not recorded here or retrievable via memory_recall",
    "did not survive. Treat these as established facts you already know, not",
    "suggestions.",
    "",
    `Project: ${projectDir}`,
    rows.length === 0 ? "Memories: none visible here yet." : "Memories:",
  ];
  for (const row of rows) {
    const tags = row.tags ?? [];
    const tagStr = tags.length ? ` [${tags.join(", ")}]` : "";
    // Origin project matters now that memories are shared: basename keeps the
    // injected line short.
    const origin = row.project.split('/').pop() ?? row.project;
    lines.push(`- [${row.date}] (${origin})${tagStr} ${sanitizeMemory(truncateMemory(row.content))}`);
  }
  lines.push("");
  // Concrete, observable triggers replace "non-trivial work": a threshold the
  // model evaluates in its own favor always resolves toward skipping, so each
  // bullet names an event instead of asking for a judgment call. The "lost
  // permanently" framing states the cost of skipping, which the old trailing
  // line never did.
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

// Keyed by directory + prompt hash now that the block depends on the prompt
// (relevance mode): an identical prompt (model retries, re-requests) hits the
// cache; a new prompt queries afresh. A zero-row result caches the
// guidance-only block, so no-match prompts stop re-querying.
const injectionCache = new Map<string, string>();

// djb2 - just a stable key shortener; a same-hash different-prompt collision
// would serve a stale block, which remember/forget invalidation clears.
function hashQuery(text: string): string {
  let h = 5381;
  for (let i = 0; i < text.length; i++) h = ((h << 5) + h + text.charCodeAt(i)) | 0;
  return (h >>> 0).toString(36);
}

// The user's prompt is the retrieval signal; capped because a query is a
// query, not a transcript - FTS is not helped by thousands of characters.
function capPromptQuery(text: string): string {
  const t = text.trim();
  return t.length > 512 ? t.slice(0, 512) : t;
}

// Visibility rule shared by recall and both injection builders: the global
// type (stack_fact) is visible everywhere; project_fact is visible only from
// its origin project unless explicitly opted out with global: true (recall)
// - at injection time there is no caller to opt in, so other projects'
// project_fact rows never surface as ambient context.
function visibleRows(client: SQL, directory: string) {
  return client`(memory_type != 'project_fact' OR project = ${directory})`;
}

// A superseded memory (supersede-tracking: memory_remember's `supersedes`
// argument) is history, not current fact - it must never surface as ambient
// context or in a normal recall. Layered ON TOP of visibleRows rather than
// merged into it: forget and memory_update must still be able to reach a
// superseded row (delete it outright, or fix a bad supersede link), so the
// ownership check they run stays separate from the "is this current"
// check recall/injection run.
function notSuperseded(client: SQL) {
  return client`superseded_by IS NULL`;
}

// Recency query shared by the recency mode and the no-match fallback: latest
// visible rows, newest first. Global by design - the project column records
// origin, not visibility, for global types; project_fact is origin-scoped
// (see visibleRows).
// Query builders are pure functions of the client, so they run unchanged
// against any database, not just the module pool.
function buildRecencyQuery(client: SQL, directory: string) {
  // LIMIT 20: a candidate slice for collapseDupes, not the final block.
  return client`
    SELECT content, coalesce(tags, '{}') AS tags,
           to_char(created_at, 'YYYY-MM-DD') AS date,
           project
    FROM memories
    WHERE ${visibleRows(client, directory)} AND ${notSuperseded(client)}
    ORDER BY created_at DESC
    LIMIT 20
  `;
}

// Cross-session recall tiebreak (spec: cross-session-recall-signal.md):
// COUNT(DISTINCT session_id) from memory_recalls, capped and weighted well
// below the same-project boost (0.01) so it only breaks near-ties, never
// overrides relevance. Deliberately NOT access_count - that column is bumped
// on every recall regardless of session, which is exactly the rich-get-richer
// signal a prior attempt reverted (see recall()'s comment). The PRIMARY KEY on
// (memory_id, session_id) in memory_recalls makes repeated recalls from ONE
// session count once, so this only grows from genuinely separate sessions
// reaching for the same memory - slow to accumulate by design, which is what
// makes it safe to rank on.
function crossSessionJoin(client: SQL) {
  return client`
    LEFT JOIN (
      SELECT memory_id, count(DISTINCT session_id) AS xsess
      FROM memory_recalls
      GROUP BY memory_id
    ) recalls ON recalls.memory_id = memories.id
  `;
}

function crossSessionBoost(client: SQL) {
  return client`LEAST(coalesce(recalls.xsess, 0), 5) * 0.002`;
}

function buildRelevanceQuery(client: SQL, tsQuery: string, directory: string) {
  // Relevance: full-text search over every visible memory - global types
  // from anywhere, project_fact from the origin project - with a small
  // same-project boost to break rank ties toward locally stored memories,
  // plus the smaller cross-session tiebreak above.
  return client`
    SELECT content, coalesce(tags, '{}') AS tags,
           to_char(created_at, 'YYYY-MM-DD') AS date,
           project
    FROM memories
    ${crossSessionJoin(client)}
    WHERE search_vector @@ to_tsquery('english', ${tsQuery})
      AND ${visibleRows(client, directory)}
      AND ${notSuperseded(client)}
    ORDER BY ts_rank(search_vector, to_tsquery('english', ${tsQuery}))
             + (CASE WHEN project = ${directory} THEN 0.01 ELSE 0 END)
             + ${crossSessionBoost(client)} DESC,
             created_at DESC
    LIMIT 20
  `;
}

// Vector half of the hybrid: nearest neighbors by cosine distance over the
// same visibility rules as the keyword half. Unlike FTS this always returns
// rows when embedded rows exist (nearest is always *something*), so the
// caller merges it via rrfMerge - and the recency fallback stays for the
// "no retrieval signal at all" case. Fails on a database without the
// embedding column; the caller catches and runs keyword-only.
function buildVectorQuery(client: SQL, vectorLit: string, directory: string) {
  return client`
    SELECT content, coalesce(tags, '{}') AS tags,
           to_char(created_at, 'YYYY-MM-DD') AS date,
           project
    FROM memories
    WHERE embedding IS NOT NULL
      AND ${visibleRows(client, directory)}
      AND ${notSuperseded(client)}
    ORDER BY embedding <=> ${vectorLit}::vector
    LIMIT ${EMBED_CANDIDATES}
  `;
}

// The prompt as an OR of stemmed words: websearch_to_tsquery ANDs the terms,
// so one word the memory never uses would zero out the whole query (bench:
// recall 0.00-0.02 on multi-word queries). OR ranks by how many (and how
// rare) the matched terms are, and sanitizing to [a-z0-9]+ tokens keeps
// to_tsquery syntax-safe. Capped at 24 words to bound the query.
function orTsQuery(text: string): string {
  const words = text.toLowerCase().match(/[a-z0-9]+/g) ?? [];
  return words.slice(0, 24).join(" | ");
}

// The block depends on the directory plus (in relevance mode) the prompt,
// passed explicitly by the caller.
async function handleTransform(
  output: { system: string[] },
  directory: string,
  prompt = "",
): Promise<void> {
  if (!directory) return;
  const query = injectionMode === "relevance" ? prompt.trim() : "";
  const cacheKey = `${directory}\u0001${hashQuery(query)}`;
  const cached = injectionCache.get(cacheKey);
  if (cached !== undefined) {
    if (cached) output.system.push(cached);
    return;
  }
  try {
    let rows: InjectionRow[];
    const tsQuery = orTsQuery(query);
    if (tsQuery) {
      // Hybrid: the keyword half always runs; the embedding half is started
      // concurrently so a warm bge-m3 (~20ms) hides behind the keyword
      // round-trip. Null embedding (Ollama down/timeout) or a failed vector
      // query (pre-migration database, DeadlineError) degrade to keyword-only.
      const embedding = embed([query], EMBED_QUERY_TIMEOUT_MS);
      const keywordRows = await withDeadline(
        buildRelevanceQuery(sql, tsQuery, directory) as unknown as PromiseLike<InjectionRow[]>,
        1000,
      );
      rows = keywordRows;
      const vecs = await embedding;
      if (vecs) {
        const vectorRows = await withDeadline(
          buildVectorQuery(sql, vectorLiteral(vecs[0]), directory) as unknown as PromiseLike<InjectionRow[]>,
          1000,
        ).catch((e: unknown) => {
          logError("embed-query", `mempg vector query failed, keyword-only: ${e instanceof Error ? e.message : String(e)}`);
          return null;
        });
        if (vectorRows) rows = hybridMerge(keywordRows, vectorRows);
      }
      if (rows.length === 0) {
        // No retrieval signal at all for this prompt - recency beats an empty
        // block. The vector half alone counts as a signal: a zero-overlap
        // paraphrase is exactly what embeddings are for.
        rows = await withDeadline(buildRecencyQuery(sql, directory) as unknown as PromiseLike<InjectionRow[]>, 1000);
      }
    } else {
      rows = await withDeadline(buildRecencyQuery(sql, directory) as unknown as PromiseLike<InjectionRow[]>, 1000);
    }
    const block = formatBlock(collapseDupes(rows), directory);
    // Evict oldest entry when cache exceeds 32
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

// Write caps: these are abuse guards, not style rules. Measured against a real
// 485-memory corpus (p95 content 517 chars, p95 5 tags, longest tag 26 chars),
// so ordinary memories never come close.
const MAX_CONTENT = 4000;
const MIN_CONTENT = 10;
const MAX_TAGS = 10;
const MAX_TAG_LENGTH = 64;

// Soft nudge only - never a rejection. Sits just above the injection block's
// 600-char truncation point (formatBlock): past this length, injection would
// cut the entry off anyway, so write time is the natural place to say so.
const CONTENT_LENGTH_NUDGE = 700;

// A write past this length still succeeds; the response just says so, so the
// calling agent can shorten future entries instead of finding out later that
// most of a long memory never made it into ambient context.
function lengthNudge(content: string): string {
  if (content.length <= CONTENT_LENGTH_NUDGE) return "";
  return (
    ` Note: this entry is ${content.length.toLocaleString()} characters - injection truncates at 600, ` +
    "so most of this won't be visible in ambient context. Consider shortening to the essential " +
    "1-3 sentences and putting longer rationale in project docs."
  );
}

// --- Memory types (plan 2.1: defaulted, never required) ---

// The stored vocabulary mirrors the DB CHECK constraint (memories_type_check);
// rows predate the column, so "required" would break every existing caller -
// type is always defaulted. Visibility follows the type: stack_fact is
// global (tooling knowledge ports across every project using the stack),
// project_fact is origin-project-only (customer-specific facts must not
// surface elsewhere), episodic is reserved for the (cut, opt-in) 1.3 feature;
// remember accepts it so the vocabulary stays in one place.
const MEMORY_TYPES = ["stack_fact", "project_fact", "episodic"] as const;
type MemoryType = (typeof MEMORY_TYPES)[number];

function resolveMemoryType(raw: unknown): MemoryType {
  return MEMORY_TYPES.includes(raw as MemoryType) ? (raw as MemoryType) : "project_fact";
}

// Raw JSON Schema input is not coerced for us: a model sending "3" or null for
// limit would otherwise reach Postgres as LIMIT NaN.
function resolveLimit(raw: unknown): number {
  const n = Number(raw);
  if (!Number.isFinite(n)) return 10;
  return Math.min(Math.max(Math.trunc(n), 1), 20);
}

// memory_tags lists distinct tags, not rows - a much cheaper result per unit,
// so its default and cap are both higher than recall's. Ordering (uses DESC,
// tag ASC) is not itself the issue an earlier review raised; the default
// value was. Verified against this project's own live corpus: it already
// carries ~50 distinct tags, so a default of 50 silently starved rare/new
// tags out of the ordinary (no-limit) call - exactly the tags this tool
// exists to surface (an established, high-count tag needs no lookup; a
// candidate that might already exist as a one-off does). 200 gives a
// realistic personal/team corpus headroom before the hard 500 cap.
function resolveTagsLimit(raw: unknown): number {
  const n = Number(raw);
  if (!Number.isFinite(n)) return 200;
  return Math.min(Math.max(Math.trunc(n), 1), 500);
}

// The model sees a generic failure; the operator sees the real message in the
// server log. Raw driver errors carry host, user and schema details that should
// not end up in a transcript sent to the provider.
function toolError(kind: string, action: string, e: unknown): string {
  logError(kind, `mempg ${action} failed: ${e instanceof Error ? e.message : String(e)}`);
  // 42883 = undefined_function. The only way to hit it here is a database
  // without pg_trgm, which is worth naming: the fix is one statement and the
  // generic message would send the operator hunting.
  if (String((e as { errno?: unknown })?.errno) === "42883") {
    return "ERROR: this database is missing the pg_trgm extension, which memory_remember needs for duplicate detection. Run: CREATE EXTENSION pg_trgm;";
  }
  return `ERROR: memory store unavailable (${action} failed; see the omp log under ~/.omp/logs).`;
}

async function recall(
  args: RecallArgs,
  ctx: { directory: string; sessionID?: string },
): Promise<string> {
  try {
    const limit = resolveLimit(args.limit);
    // Visibility: the global type (stack_fact) everywhere; project_fact only
    // from the origin project unless the caller opts in with global: true.
    const visibleCond = args.global ? sql`` : sql`AND ${visibleRows(sql, ctx.directory)}`;
    // Superseded memories are hidden by default (same posture as injection);
    // includeSuperseded surfaces them for history/audit, annotated below.
    const supersededCond = args.includeSuperseded ? sql`` : sql`AND ${notSuperseded(sql)}`;
    const q = args.query ?? "";
    const tsQuery = q ? orTsQuery(q) : "";
    const queryCond = q
      ? sql`AND search_vector @@ to_tsquery('english', ${tsQuery})`
      : sql``;
    // Tags are not part of search_vector (it covers content only), so they are
    // unreachable by query alone. Matches rows carrying ALL the given tags,
    // served by idx_memories_tags.
    const tagList = Array.isArray(args.tags) ? args.tags.filter((t) => typeof t === "string" && t) : [];
    const tagCond = tagList.length
      ? sql`AND tags @> ${sql.array(tagList, "text")}`
      : sql``;
    // Relevance-ranked when searching; undirected browse is plain recency
    // (matching the injection fallback). A frequency blend was tried and
    // removed: bumping exactly the returned top-5 is a rich-get-richer loop -
    // live corpus rows pinned the top slot after a few runs. access_count
    // stays as data collection; no ranking consumes it. The cross-session
    // tiebreak is different in kind (see crossSessionBoost's comment): it
    // only grows from distinct sessions independently reaching for a memory,
    // not from raw exposure, which is what makes it safe to rank on.
    const orderBy = q
      ? sql`ORDER BY ts_rank(search_vector, to_tsquery('english', ${tsQuery})) + ${crossSessionBoost(sql)} DESC`
      : sql`ORDER BY created_at DESC`;

    // Hybrid: with a searchable query, embed it concurrently so a warm bge-m3
    // (~20ms) hides behind the keyword round-trip; the vector half then merges
    // via RRF. Any embedding failure (Ollama down, no embedding column yet)
    // degrades to exactly the pre-hybrid keyword-only behavior.
    const embedding = tsQuery ? embed([q], EMBED_QUERY_TIMEOUT_MS) : Promise.resolve(null);
    // A directed query fetches a candidate slice for the RRF merge; the
    // caller's limit is applied after. Browse keeps its plain LIMIT.
    const keywordRows = await sql`
      SELECT id, content, coalesce(tags, '{}') AS tags,
             to_char(created_at, 'YYYY-MM-DD') AS date,
             project, memory_type, superseded_by
      FROM memories
      ${crossSessionJoin(sql)}
      WHERE 1=1 ${visibleCond} ${supersededCond} ${queryCond} ${tagCond}
      ${orderBy}
      LIMIT ${q ? EMBED_CANDIDATES : limit}
    ` as (MemoryRow & { memory_type: string; superseded_by: number | null })[];

    let rows = keywordRows;
    const vecs = await embedding;
    if (vecs) {
      // Same filters as the keyword half, minus the FTS condition.
      const vectorRows = await (sql`
        SELECT id, content, coalesce(tags, '{}') AS tags,
               to_char(created_at, 'YYYY-MM-DD') AS date,
               project, memory_type, superseded_by
        FROM memories
        WHERE embedding IS NOT NULL ${visibleCond} ${supersededCond} ${tagCond}
        ORDER BY embedding <=> ${vectorLiteral(vecs[0])}::vector
        LIMIT ${EMBED_CANDIDATES}
      ` as unknown as Promise<(MemoryRow & { memory_type: string; superseded_by: number | null })[]>).catch((e: unknown) => {
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

    // Access ranking is recall-only (plan 2.2): the injection path stays
    // read-only because its per-directory cache would make increments biased.
    // Fire-and-forget so the UPDATE never sits on the read path's latency.
    const ids = rows.map((r) => r.id);
    if (ids.length > 0) {
      void sql`UPDATE memories SET access_count = access_count + 1, last_accessed_at = now() WHERE id = ANY(${sql.array(ids, "int8")})`.catch(
        (e: unknown) => logError("access", `mempg access bump failed: ${e instanceof Error ? e.message : String(e)}`),
      );
    }
    // Cross-session recall signal: records which session reached for each
    // returned memory. The PRIMARY KEY on (memory_id, session_id) makes this
    // naturally idempotent - repeated recalls within one session count once,
    // so only genuinely distinct sessions grow crossSessionBoost's count.
    // Skipped without a sessionID (only __internals callers omit it); never
    // allowed to block or fail the recall itself.
    if (ids.length > 0 && ctx.sessionID) {
      void sql`
        INSERT INTO memory_recalls (memory_id, session_id)
        SELECT unnest(${sql.array(ids, "int8")}), ${ctx.sessionID}
        ON CONFLICT (memory_id, session_id) DO NOTHING
      `.catch((e: unknown) => logError("xsess", `mempg cross-session record failed: ${e instanceof Error ? e.message : String(e)}`));
    }

    if (rows.length === 0) return "No memories found.";

    return rows
      .map((r) => {
        const tags = r.tags ?? [];
        const tagStr = tags.length ? ` (${tags.join(', ')})` : '';
        // episodic is worth surfacing; project_fact is the default every
        // pre-column row carries, so printing it is pure noise.
        const typeStr = r.memory_type === "project_fact" ? "" : ` [${r.memory_type}]`;
        // Only ever set when includeSuperseded surfaced this row - a normal
        // recall never returns a superseded row to annotate in the first place.
        const supersededStr = r.superseded_by ? ` [superseded by #${r.superseded_by}]` : '';
        return `[${r.date}] [${r.project}]${typeStr}${tagStr}\n#${r.id}${supersededStr}\n${r.content}`;
      })
      .join('\n---\n');
  } catch (e: unknown) {
    return toolError("recall", "recall", e);
  }
}

// Lists distinct tags in use, with counts, so a caller can reuse an
// established tag instead of minting a near-duplicate (e.g. "postgres" vs
// "tool:postgres"). Same visibility rule as every other read path; superseded
// rows' tags are excluded (history, not active vocabulary). unnest over the
// existing tags[] needs no new index at this corpus size - the GIN index on
// tags is for the `@>` containment filter recall uses, not this aggregate.
// `client` defaults to the module pool, same as visibleRows/notSuperseded -
// a test can pass a dedicated connection pointed at an isolated schema to
// exercise a genuinely empty table without touching real data.
async function listTags(args: TagsArgs, ctx: { directory: string }, client: SQL = sql): Promise<string> {
  try {
    const limit = resolveTagsLimit(args.limit);
    const visibleCond = args.global ? client`` : client`AND ${visibleRows(client, ctx.directory)}`;
    const rows = await client`
      SELECT tag, count(*) AS uses
      FROM memories, unnest(tags) AS tag
      WHERE 1=1 ${visibleCond} AND ${notSuperseded(client)}
      GROUP BY tag
      ORDER BY uses DESC, tag ASC
      LIMIT ${limit}
    ` as { tag: string; uses: string }[];

    if (rows.length === 0) return "No tags found.";
    return rows.map((r) => `${r.tag} (${r.uses})`).join("\n");
  } catch (e: unknown) {
    return toolError("tags", "tags", e);
  }
}

// Only the length cap is worth enforcing here (mirrors validateWrite's tag
// check, same error shape) - a rename target that would itself be an invalid
// tag to write shouldn't be allowed to land via a different path.
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

// Renames a tag across every row the caller can currently reach - same
// project-boundary guard as forget/update (visibleRows()), not memory_tags'
// global search flag: a write's reach is inherent to which rows the WHERE
// clause can touch, not something to opt into widening. array_replace()
// alone does NOT deduplicate: a row already tagged both `old` and `new`
// would end up with `new` twice (verified directly against Postgres, not
// assumed) - the DISTINCT/unnest wrap is required, not optional polish.
async function retag(args: RetagArgs, ctx: { directory: string }): Promise<string> {
  try {
    const invalid = validateRetag(args);
    if (invalid) return invalid;

    const updated = await sql`
      UPDATE memories
      SET tags = ARRAY(SELECT DISTINCT unnest(array_replace(tags, ${args.old}, ${args.new})))
      WHERE tags @> ARRAY[${args.old}]
        AND ${visibleRows(sql, ctx.directory)}
      RETURNING id
    ` as { id: number }[];

    if (updated.length === 0) {
      return `No memories tagged "${args.old}".`;
    }
    // A rename can touch stack_fact/episodic rows visible from every
    // project's injected block, so a per-directory invalidateInjection()
    // isn't enough - same reasoning as memory_consolidate's real-removal clear.
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

// Trigram similarity threshold for near-duplicate handling (consolidation and
// the injection collapse pass). Measured on a real 485-memory corpus: 0.8 is
// strict enough that only restatements collide. Writes never reject on it -
// duplicates are cleaned up by memory_consolidate and collapsed out of the
// injected block.
const DEDUP_SIMILARITY = 0.8;

// Thrown (and caught) only for the `supersedes` boundary/existence checks
// inside remember()'s transaction - distinct from a generic DB failure so the
// catch block can return the specific message instead of toolError's generic
// one, and so the thrown error rolls back the insert too (supersede failing
// must not silently leave the new memory stored with no link, nor leave it
// stored at all - same "fail loud, not a silent no-op" posture as forget/update).
class SupersedeError extends Error {}

async function remember(
  args: RememberArgs,
  ctx: { directory: string; sessionID: string },
): Promise<string> {
  try {
    const invalid = validateWrite(args);
    if (invalid) return invalid;

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

    // sql.array(tags) alone encodes text[] with quoted elements under bun 1.4.2;
    // the element type hint is required for clean array storage.
    let newId: number;
    if (supersedesId !== undefined) {
      // Atomic: insert the new memory and link the old one in one
      // transaction. If the supersede target is unreachable (wrong project,
      // or gone), the whole thing rolls back - no orphaned insert.
      newId = await sql.begin(async (tx) => {
        const inserted = await tx`
          INSERT INTO memories (content, tags, session_id, project, memory_type)
          VALUES (${args.content}, ${tx.array(tags, "text")}, ${ctx.sessionID}, ${ctx.directory}, ${resolveMemoryType(args.type)})
          RETURNING id
        ` as { id: number }[];
        const id = inserted[0].id;

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
    } else {
      const inserted = await sql`
        INSERT INTO memories (content, tags, session_id, project, memory_type)
        VALUES (${args.content}, ${sql.array(tags, "text")}, ${ctx.sessionID}, ${ctx.directory}, ${resolveMemoryType(args.type)})
        RETURNING id
      ` as { id: number }[];
      newId = inserted[0].id;
    }

    // Fire-and-forget: a failure leaves embedding NULL, keyword search keeps
    // working, and the backfill (deploy/backfill.ts) fills the gap later.
    void embedAndStore(newId, args.content);

    // The injection block is global, but its cache is keyed by the calling
    // directory + prompt; clear the directory's keys. A supersede changes
    // what's current for every project too (the old row stops surfacing),
    // same as any other write.
    invalidateInjection(ctx.directory);
    return `Stored memory #${newId} (project ${basename}).${lengthNudge(args.content)}`;
  } catch (e: unknown) {
    if (e instanceof SupersedeError) return `ERROR: ${e.message}`;
    return toolError("remember", "remember", e);
  }
}

// --- Keyword capture (plan 1.1, revised) ---

// Deterministic capture, no LLM: a trigger phrase in the prompt stores the
// first paragraph after it verbatim (minus the trigger, subject to the code
// and length guards below) through the normal write path.
// Extracting "relevant content" instead would be a model judgment on the
// prompt-admission path - nondeterministic, and it violates the project rule
// that model judgment never becomes load-bearing (memory #1555).
// Global flag so extractMemoryRequest can walk every candidate via matchAll
// (which clones the regex - no lastIndex state leaks between calls).
const MEMORY_TRIGGER_RE =
  /\b(?:remember(?:\s+(?:this|that|to))?|do(?:n'?| no)t forget(?:\s+(?:this|that|to))?|keep (?:this|that )?in mind(?: that)?)\b\s*[:,]?\s*/gi;

// Interrogative follow-ons are questions about the past ("remember when the
// pool broke?"), not storage requests. The list is deliberately narrow:
// "remember that when X happens, do Y" is imperative and must be captured, so
// "when" alone is not enough - only skip the bare question forms.
const INTERROGATIVE_RE = /^(?:when|what|where|why|how|who|whom|whose|which|did)\b/i;

// A trigger only counts as prose, never as code: a pasted spec mentioning the
// identifier `remember()` was once stored verbatim as a 3292-char memory. So a
// candidate is skipped when it sits inside a code span or fence (odd backtick
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

// project_fact rows are origin-scoped (a customer-B agent must not delete
// customer-A's customer facts); the global type (stack_fact) is
// maintainable from any project - that's what global means for it.
async function forget(
  args: ForgetArgs,
  ctx: { directory: string },
): Promise<string> {
  try {
    const id = Number(args.id);
    if (!Number.isInteger(id) || id <= 0) {
      return "ERROR: id must be a positive integer (the #id shown by memory_recall).";
    }
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

// No dedup fall-through, by design: an update that lands close to another
// memory is an intentional correction, not a dupe to reject. updated_at is
// set; created_at is deliberately NOT bumped - the displayed date must keep
// saying when the memory was learned, not when it was last edited. Omitted
// tags/type are preserved, not reset to their defaults.
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

    const tags = args.tags ?? [];
    const tagCond = Array.isArray(args.tags)
      ? sql`tags = ${sql.array(tags, "text")},`
      : sql``;
    const typeCond = args.type !== undefined
      ? sql`memory_type = ${resolveMemoryType(args.type)},`
      : sql``;
    const updated = await sql`
      UPDATE memories
      SET content = ${args.content},
          ${tagCond}
          ${typeCond}
          updated_at = now()
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
    // Content changed, so the embedding must follow - same fire-and-forget
    // path as remember.
    void embedAndStore(id, args.content);
    return `Updated memory #${id}.${lengthNudge(args.content)}`;
  } catch (e: unknown) {
    return toolError("update", "update", e);
  }
}

// Embedding-similarity threshold for consolidate's second pass. Originally
// calibrated for bge-m3 (measured 2026-09-19); re-verified from scratch for
// embeddinggemma:300m (the model mempg.ts now defaults to - see the
// "complete the embeddinggemma:300m migration" spec, repo history) since a
// different model has no guaranteed relationship to another model's cosine
// distribution and that must be measured, not assumed.
//
// Re-ran the exact same calibration methodology: the same 26 hand-labeled
// pairs (bench/embeddinggemma-calibration.ts) re-embedded with
// embeddinggemma:300m. Result: duplicates 0.789-0.925 (mean .866), updates
// 0.540-0.813 (mean .669), distinct pairs topped out at 0.749 (mean .473) -
// a cleaner duplicate/distinct gap than bge-m3 had (0.04 vs 0.024). 0.83
// still sits clear of every distinct pair here too.
//
// Also re-ran the larger, more reliable mempg-shaped-consolidate.ts bench
// (808 pairs, purpose-built to include the exact update-shape failure modes
// - rate limit, path, port, retry-count changes) at 0.83 with
// embeddinggemma:300m: FPR 0.002 with the detail cross-check applied (vs
// bge-m3's 0.005), every update-* category pair that reached the threshold
// was still caught (0 clean merges across all update-* rows, same gate
// bge-m3 passed), and false-negative cost dropped too (1/281 candidates
// wrongly blocked, 0.4%, vs bge-m3's 1.6%). The 26-pair set confirms 0.83
// clears every distinct pair (its own zero-false-merge floor is 0.75); the
// larger bench supports keeping 0.83 - re-measured, not carried over on the
// assumption "the model switch shouldn't matter."
const CONSOLIDATE_EMBED_THRESHOLD = 0.83;

// Mutual visibility for consolidation, mirroring visibleRows(): a project_fact
// may only cluster with another row from its OWN project (regardless of that
// row's type); the global type (stack_fact) clusters with anything.
// Consolidation must never merge/delete across a boundary memory_forget and
// memory_update already refuse to cross.
function mutuallyVisible(client: SQL) {
  return client`((a.memory_type != 'project_fact' AND b.memory_type != 'project_fact') OR a.project = b.project)`;
}

// Same condition as mutuallyVisible(), but for the wording pass: its
// similarity computation happens in TS (trigrams()/nearDupeSets()), not a SQL
// join, so the guard has to be a plain boolean check on two already-fetched
// rows instead of a SQL fragment. `a.project = b.project` is NULL-unsafe in
// SQL (NULL never equals NULL); mirrored here explicitly rather than relying
// on JS's `null === null` (true), which would silently disagree with the SQL
// version for rows with no project set.
function mutuallyVisibleRows(
  a: { memory_type: string; project: string | null },
  b: { memory_type: string; project: string | null },
): boolean {
  if (a.memory_type !== "project_fact" && b.memory_type !== "project_fact") return true;
  return a.project !== null && b.project !== null && a.project === b.project;
}

// Excludes structurally templated, auto-generated content from the meaning
// pass: verified against the real corpus (2026-09-19 audit, see AGENTS.md)
// that three fixed sentence templates - background-task status logs from the
// team/subagent system, session-compaction summaries, and per-app migration
// checklist entries - drive cosine similarity high between GENUINELY
// DIFFERENT facts (different task ids, different sessions, different apps)
// purely because only a few words vary while the rest of the sentence is
// boilerplate. This is a content-pattern check, not the `auto_capture` tag:
// that tag was tested first and rejected - it sits on ~90% of BOTH correct
// and incorrect merges alike in the real corpus, so excluding by tag would
// gut the pass. The three patterns below have zero overlap with any
// confirmed-correct meaning-pass merge in that same audit. A residual, much
// smaller false-merge rate remains on natural-language content whose
// wording happens to be structurally parallel (e.g. two genuinely different
// systemd unit files described in near-identical sentences) - no threshold
// or pattern separates that case from true duplicates without also losing
// real ones (see AGENTS.md), so it is accepted rather than "fixed". Two
// variants: the self-join (a/b aliases) and the plain single-table form.
function isTemplatedAutoLogPair(client: SQL) {
  return client`(
    a.content ILIKE '%background task bg_%' OR b.content ILIKE '%background task bg_%'
    OR a.content ILIKE '%session compacting for project%' OR b.content ILIKE '%session compacting for project%'
    OR a.content ~ 'For APP.{0,3}[0-9]+.{0,3}\\(' OR b.content ~ 'For APP.{0,3}[0-9]+.{0,3}\\('
  )`;
}

function isTemplatedAutoLog(client: SQL) {
  return client`(
    content ILIKE '%background task bg_%'
    OR content ILIKE '%session compacting for project%'
    OR content ~ 'For APP.{0,3}[0-9]+.{0,3}\\('
  )`;
}

// A second, independent signal for the meaning pass, on top of embedding
// similarity: no cosine threshold safely separates "same fact, reworded" from
// "same sentence shape, different fact" - e.g. a rate limit changing from
// 100 to 500 requests/minute, or a system-level vs a user-level systemd
// unit path, both score high on cosine despite differing in exactly the
// specific detail that matters. This extracts those specific details
// (numbers, paths, capitalized names) with regexes only - no model call,
// consistent with the earlier decision to drop the judge model entirely.
type DetailSet = {
  numbers: Set<string>;
  paths: Set<string>;
  properNouns: Set<string>;
};

// Deliberately coarse regexes, expected to need tuning against more real
// content over time - proper-noun extraction especially: a lowercase
// technical name (e.g. "bge-m3") is not capitalized and so is not caught by
// this pass; that is an accepted gap, not a bug, for the first version of
// this check.
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

// A category conflicts only when BOTH sides have extracted values AND those
// values are disjoint - absence on one side is not a conflict (nothing to
// disagree with), per spec: "the API rate limit changed" (no number) must
// not be blocked from merging with "...is 100/min" just because one side
// lacks the detail the other has. "Disjoint" is judged by `equivalent`
// rather than raw string equality, so formatting differences that don't
// change meaning (see the three `*Equivalent` functions below) don't count
// as a real conflict.
function categoryConflict(a: Set<string>, b: Set<string>, equivalent: (x: string, y: string) => boolean): boolean {
  if (a.size === 0 || b.size === 0) return false;
  for (const x of a) for (const y of b) if (equivalent(x, y)) return false;
  return true;
}

// Real notes format the same number differently ("1,000" vs "1000") without
// meaning anything different, so strip thousands separators and compare
// numerically when both sides parse as plain numbers. Version-shaped values
// (2+ dot-separated digit groups, e.g. "2.5.0") get a separate rule: one
// side being a strict prefix of the other ("2.5" vs "2.5.0") is treated as
// imprecision, not a conflict - but "2.5" vs "3.0" still conflicts, since
// that is a real version change, the exact thing this check exists to catch.
const PLAIN_NUMBER = /^\d+(\.\d+)?$/;
const VERSION_SHAPED = /^\d+(\.\d+)+$/;

function isVersionPrefix(x: string, y: string): boolean {
  const xs = x.split(".");
  const ys = y.split(".");
  if (xs.length >= ys.length) return false;
  return xs.every((seg, i) => seg === ys[i]);
}

function numbersEquivalent(x: string, y: string): boolean {
  if (x === y) return true;
  const nx = x.replace(/,/g, "");
  const ny = y.replace(/,/g, "");
  if (nx === ny) return true;
  if (PLAIN_NUMBER.test(nx) && PLAIN_NUMBER.test(ny)) return Number(nx) === Number(ny);
  if (VERSION_SHAPED.test(nx) && VERSION_SHAPED.test(ny)) return isVersionPrefix(nx, ny) || isVersionPrefix(ny, nx);
  return false;
}

// A trailing slash doesn't change what path is meant ("/var/log/app" vs
// "/var/log/app/"); case stays significant, since mempg's actual content
// describes real (case-sensitive Linux) filesystem paths.
function pathsEquivalent(x: string, y: string): boolean {
  const strip = (v: string) => (v.length > 1 ? v.replace(/\/+$/, "") || v : v);
  return strip(x) === strip(y);
}

// Proper nouns conflict on identity, not on how they happen to be cased
// ("Jira" vs "JIRA" is the same product name).
function properNounsEquivalent(x: string, y: string): boolean {
  return x.toLowerCase() === y.toLowerCase();
}

// Returns one human-readable reason per conflicting category (numbers,
// paths, names), or an empty array when the pair has no conflict -
// including the common case where neither side has any extractable detail
// at all, which must never block a merge it has nothing to check. Reasons
// always quote the raw extracted values, never the normalized form used
// internally for comparison, so a report stays readable.
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

// Deterministic consolidation, no model calls inside the plugin: find
// near-duplicate clusters, keep the newest of each, delete the rest. The
// deleted texts are returned verbatim so the CALLING agent - itself a model -
// can merge any unique fact back into the survivor via memory_update. Merging
// is language synthesis, which is the caller's job, not the plugin's.
// Runs on demand (user-invoked), never on a schedule.
//
// Two passes, run in sequence (the second sees whatever the first already
// removed): trigram similarity over content (wording-level restatements,
// unchanged from the original implementation) and embedding cosine
// similarity (meaning-level duplicates worded completely differently, the
// case trigram structurally cannot reach). Each is capped at 25 clusters per
// run so a wildly-duplicated corpus cannot turn into one huge report. The
// meaning pass additionally gates every candidate pair through
// detailConflicts: a specific number/path/name that differs between an
// otherwise-similar pair blocks the auto-merge and routes that pair to
// [meaning-uncertain] in the report instead (see extractDetails' comment).
async function consolidate(args: ConsolidateArgs = {}): Promise<string> {
  const dryRun = args.dryRun === true;
  // Report wording only - the clustering/threshold/exclusion logic below runs
  // identically in both modes; this just labels what happened to each row and
  // gates whether the DELETE statements actually execute.
  const verb = dryRun ? "would remove" : "removed";
  try {
    // --- Pass 1: wording (trigram similarity over content) ---
    // Superseded rows are already-resolved history (an explicit decision was
    // made about them via `supersedes`), not accidental near-duplicates for
    // this pass to guess about - excluded the same way isTemplatedAutoLog
    // excludes a different kind of not-a-candidate row.
    const rows = await sql`
      SELECT id, content, coalesce(tags, '{}') AS tags, created_at, memory_type, project
      FROM memories
      WHERE superseded_by IS NULL
      ORDER BY created_at DESC
    ` as { id: number; content: string; tags: string[]; created_at: Date; memory_type: string; project: string | null }[];

    // Greedy clustering newest-first: each row joins the first cluster whose
    // representative (the newest member) it near-dupes. Trigram sets are built
    // once (rebuilding per comparison made consolidate O(n^2) set-constructions,
    // seconds at 5k rows), and a size-ratio prefilter skips pairs whose Jaccard
    // can never reach the threshold.
    type Entry = { row: (typeof rows)[number]; set: Set<string> };
    const entries: Entry[] = rows.map((row) => ({ row, set: trigrams(row.content) }));
    const clusters: Array<Array<Entry>> = [];
    for (const entry of entries) {
      const host = clusters.find((c) => {
        // Comparisons are always against the cluster's anchor (c[0], the
        // newest member that started it) - never against every existing
        // member - so a single check here is enough to keep the whole
        // cluster mutually visible: if the anchor is global, anything can
        // join it (including rows from different projects, which is
        // correct - the global row is what survives); if the anchor is a
        // project_fact, only its own project's rows can join.
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
    // Tracked regardless of dryRun: in a real run these rows are physically
    // gone by the time pass 2 queries, so pass 2 never sees them; in a dry
    // run nothing was actually deleted, so pass 2 must exclude them itself to
    // see the same candidate set a real run would (and match its report).
    const wordingRemovedIds: number[] = [];
    for (const cluster of wordingGroups) {
      const survivor = cluster[0].row;
      const removedRows = cluster.slice(1).map((e) => e.row);
      if (!dryRun) {
        for (const r of removedRows) {
          await sql`DELETE FROM memories WHERE id = ${r.id}`;
        }
      }
      for (const r of removedRows) wordingRemovedIds.push(r.id);
      removed += removedRows.length;
      removedGroups++;
      // Show what died so the calling agent can merge unique facts back into
      // the survivor.
      report.push(
        `[wording] Kept #${survivor.id}: ${truncateMemory(survivor.content)}\n` +
          removedRows.map((r) => `  ${verb} #${r.id}: ${truncateMemory(r.content)}`).join("\n"),
      );
    }

    // --- Pass 2: meaning (embedding cosine similarity) ---
    // Only rows the wording pass left behind, only embedded rows (a row
    // Ollama never reached - down at write time, pre-migration database - is
    // simply not a candidate, same graceful-degradation posture as everywhere
    // else), and never templated auto-logs (isTemplatedAutoLog - see its
    // comment for why). The pairwise threshold join is pushed into Postgres
    // (pgvector's <=> operator) rather than pulled into TS: cheap for the
    // corpus sizes this tool already accepts an O(n^2) cost for (see the
    // wording pass), and it lets the DB do the floating-point work instead of JS.
    const embedPairs = (await sql`
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

    if (embedPairs.length > 0) {
      const linked = new Set<string>();
      for (const p of embedPairs) linked.add(`${p.a_id}:${p.b_id}`);
      const isLinked = (x: number, y: number) => (x < y ? linked.has(`${x}:${y}`) : linked.has(`${y}:${x}`));

      const embedRows = (await sql`
        SELECT id, content, coalesce(tags, '{}') AS tags, created_at
        FROM memories
        WHERE embedding IS NOT NULL AND superseded_by IS NULL AND NOT ${isTemplatedAutoLog(sql)}
          AND id <> ALL(${sql.array(wordingRemovedIds, "int8")})
        ORDER BY created_at DESC
      `) as { id: number; content: string; tags: string[]; created_at: Date }[];

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
        // Detail cross-check (see extractDetails/detailConflicts): a pair
        // only auto-merges if, on top of the embedding threshold, no
        // specific number/path/name conflicts between it and the survivor.
        // A conflicting member is pulled OUT of the auto-merge on its own -
        // it does not void the rest of an otherwise-clean cluster.
        const clean: (typeof cluster)[number][] = [];
        const uncertain: Array<{ row: (typeof cluster)[number]; reasons: string[] }> = [];
        for (const row of cluster.slice(1)) {
          const reasons = detailConflicts(survivorDetails, extractDetails(row.content));
          if (reasons.length > 0) uncertain.push({ row, reasons });
          else clean.push(row);
        }

        for (const r of clean) {
          if (!dryRun) await sql`DELETE FROM memories WHERE id = ${r.id}`;
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

// Clears every cache entry for the directory - relevance mode keys by
// directory + prompt hash, so a write invalidates them all.
function invalidateInjection(directory: string): void {
  for (const key of [...injectionCache.keys()]) {
    if (key === directory || key.startsWith(`${directory}\u0001`)) {
      injectionCache.delete(key);
    }
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
    // This description is the only place the write policy is guaranteed to
    // reach the model: it is in the tool schema every session, whereas the
    // injected block is missing whenever injection fails (DB down, deadline)
    // and the user may have no project instructions at all.
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
            "episodic = reserved for a future feature; visible everywhere like stack_fact, " +
            "but nothing assigns it automatically today - default to project_fact or " +
            "stack_fact unless you have a specific reason to use it. " +
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
    // User-invoked cleanup, not a write-path gate: writes never reject on
    // duplicates, so call this when the corpus has accumulated near-dupes.
    // The deleted texts come back in the result so the calling agent can
    // merge unique facts into the survivors via memory_update.
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
            "project's project_fact memories, plus all stack_fact/episodic memories, which are always global).",
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

// Deterministic gate for the session_stop checkpoint: counts tool calls made
// since the latest user prompt, and skips turns that already wrote memory -
// via memory_remember/memory_update, or via keyword capture of the prompt
// itself (re-running extractMemoryRequest, the same test the capture path
// uses, so a captured fact is never nudged into a second write).
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

function trackCapture(p: Promise<void>): void {
  inflightCaptures.add(p);
  void p.finally(() => inflightCaptures.delete(p));
}

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
    const key = `${prompt.length}:${hashQuery(prompt)}`;
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
    // otherwise the TCP + SCRAM handshake is paid inside the first injection.
    void sql`SELECT 1`.catch(() => {});
    // Warm the embedding model off the request path: a cold load is ~2-3s
    // (over the query budget), so without this the first hybrid query of a
    // session falls back to keyword-only. keep_alive pins it between requests.
    void embed(["mempg session warmup"], EMBED_WRITE_TIMEOUT_MS);
  });

  pi.on("before_agent_start", async (event, ctx) => {
    // Injection: extend the event's systemPrompt (it carries earlier
    // extensions' chained overrides); cwd is read per event since /move changes it.
    const output: { system: string[] } = { system: [] };
    // Keyword capture: a trigger phrase ("remember that ...") stores the first
    // paragraph after it verbatim (code/length guards apply) through the normal write path. No LLM call, and
    // the prompt is never mutated. Main session only: subagent prompts are
    // written by the parent agent, not the user.
    if (ctx.agent.kind === "main" && extractMemoryRequest(event.prompt) !== null) {
      if (firstSighting(event.prompt)) {
        trackCapture(captureFromPrompt(event.prompt, ctx.cwd, ctx.sessionManager.getSessionId()));
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
  get sql() {
    return sql;
  },
  truncateMemory,
  sanitizeMemory,
  withDeadline,
  formatBlock,
  handleTransform,
  recall,
  remember,
  forget,
  updateMemory,
  consolidate,
  listTags,
  resolveTagsLimit,
  retag,
  validateRetag,
  lengthNudge,
  extractMemoryRequest,
  needsMemoryCheckpoint,
  CHECKPOINT_MIN_TOOL_CALLS,
  CHECKPOINT_NOTE,
  captureFromPrompt,
  settleCaptures,
  TOOL_SPECS,
  toZod,
  invalidateInjection,
  resolveSslMode,
  resolveLimit,
  resolveMemoryType,
  validateWrite,
  logError,
  setLogSink,
  toolError,
  rateLimitOk,
  resetRateLimit,
  get injectionMode() {
    return injectionMode;
  },
  setInjectionMode(mode: "relevance" | "recency") {
    injectionMode = mode;
  },
  capPromptQuery,
  hashQuery,
  orTsQuery,
  buildRecencyQuery,
  buildRelevanceQuery,
  buildVectorQuery,
  visibleRows,
  notSuperseded,
  crossSessionJoin,
  crossSessionBoost,
  embed,
  EMBED_WRITE_TIMEOUT_MS,
  embedAndStore,
  storeEmbedding,
  CONSOLIDATE_EMBED_THRESHOLD,
  mutuallyVisibleRows,
  isTemplatedAutoLog,
  extractDetails,
  detailConflicts,
  rrfMerge,
  hybridMerge,
  vectorLiteral,
  get ollamaBase() {
    return ollamaBase;
  },
  // Test hook: point the embed client at a dead endpoint to exercise the
  // keyword-only degradation paths (module state, like injectionMode).
  setOllamaBase(base: string) {
    ollamaBase = base;
  },
  retain,
  dispose,
};

// Attach test internals to the default export instead of as a named export
// (established contract; module exports stay limited to default).
export default Object.assign(mempg, { __internals }) as typeof mempg & {
  __internals: typeof __internals;
};
