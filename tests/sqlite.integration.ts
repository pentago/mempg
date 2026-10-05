// SQLite backend integration tests, run by `bun run test:sqlite` in their own
// process: the backend is fixed when mempg loads, so this file sets up its
// environment first and stays out of the default `bun test` (Postgres) run by
// its name. It owns its /tmp database, deleted before and after the run.
//
// Embeddings come from a fake Ollama served in-process: a bag-of-words vector
// with a small synonym table, so vector search and the consolidate meaning
// pass are deterministic and need no model.
import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { rm } from "node:fs/promises";

const PATH = process.env.MEMPG_SQLITE_PATH ?? "";
if (process.env.MEMPG_BACKEND !== "sqlite" || !PATH.startsWith("/tmp/")) {
  throw new Error("run via `bun run test:sqlite`: needs MEMPG_BACKEND=sqlite and a /tmp MEMPG_SQLITE_PATH");
}
const removeDb = async () => {
  for (const suffix of ["", "-wal", "-shm"]) await rm(PATH + suffix, { force: true });
};
// Bun opens the SQLite file as soon as mempg loads, so a stale one goes first.
await removeDb();
// Dynamic on purpose: the stale file must be gone before mempg loads and opens it.
const { default: mempg } = await import("../mempg");
const { __internals } = mempg;
const sql = __internals.sql;

const SYNONYMS: Record<string, string> = {
  automobile: "car", vehicle: "car", requires: "needs", fresh: "new", tyres: "tires", shortly: "soon", daemon: "service",
};
function fakeEmbed(text: string): number[] {
  const v = new Array<number>(768).fill(0);
  for (const w of text.toLowerCase().match(/[a-z0-9]+/g) ?? []) {
    v[Number(Bun.hash(SYNONYMS[w] ?? w)) % 768] += 1;
  }
  return v;
}
const ollama = Bun.serve({
  port: 0,
  async fetch(req) {
    const { input } = (await req.json()) as { input: string[] };
    return Response.json({ embeddings: input.map(fakeEmbed) });
  },
});
const EMBED = `http://127.0.0.1:${ollama.port}`;
const OFF = "http://127.0.0.1:9";

// Every non-embedding log line is a swallowed failure on a fire-and-forget
// path (cross-session record, embed write); afterAll asserts there are none.
const logs: string[] = [];

let seq = 0;
const dir = () => `/tmp/mempg-sqlite-test-${++seq}`;

async function store(content: string, directory: string, args: Record<string, unknown> = {}): Promise<number> {
  const out = await __internals.remember({ content, ...args }, { directory, sessionID: "t" });
  const id = out.match(/Stored memory #(\d+)/)?.[1];
  if (!id) throw new Error(out);
  return Number(id);
}

async function count(): Promise<number> {
  const [{ n }] = (await sql`SELECT count(*) AS n FROM memories`) as { n: number }[];
  return n;
}

// Ids in the order recall listed them.
const ranked = (out: string) => [...out.matchAll(/^#(\d+)/gm)].map((m) => Number(m[1]));

// Re-checks a condition set by a fire-and-forget write (cross-session
// record, an embed a gate holds others behind) - paths with no
// promise to await. Each check is itself an awaited query on the single
// SQLite connection, so pending writes get their turn in between; no timer.
async function until(check: () => Promise<boolean>, what: string): Promise<void> {
  for (let i = 0; i < 10_000; i++) if (await check()) return;
  throw new Error(`${what} never happened`);
}

async function untilEmbedded(id: number): Promise<void> {
  await __internals.settleEmbeddings();
  const [row] = (await sql`SELECT embedding IS NOT NULL AS has FROM memories WHERE id = ${id}`) as { has: number }[];
  if (row?.has !== 1) throw new Error(`#${id} was not embedded`);
}

async function inject(directory: string, prompt: string): Promise<string> {
  const o = { system: [] as string[] };
  __internals.invalidateInjection(directory);
  await __internals.handleTransform(o, directory, prompt);
  return o.system.join("");
}

describe("sqlite backend", () => {
  beforeAll(() => {
    __internals.setLogSink((line) => logs.push(line));
    __internals.setOllamaBase(OFF);
  });

  afterAll(async () => {
    await __internals.settleEmbeddings();
    __internals.setLogSink();
    await __internals.dispose();
    ollama.stop(true);
    await removeDb();
    expect(logs.filter((l) => !l.includes("embedding unavailable"))).toEqual([]);
  });

  test("creates the schema on first use with WAL and foreign keys on", async () => {
    await __internals.ready();
    await __internals.ready();
    const [{ foreign_keys }] = (await sql`PRAGMA foreign_keys`) as { foreign_keys: number }[];
    const [{ journal_mode }] = (await sql`PRAGMA journal_mode`) as { journal_mode: string }[];
    expect(foreign_keys).toBe(1);
    expect(journal_mode).toBe("wal");
    const tables = ((await sql`SELECT name FROM sqlite_master WHERE type = 'table'`) as { name: string }[]).map((r) => r.name);
    expect(tables).toEqual(expect.arrayContaining(["memories", "memories_fts", "memory_recalls"]));
  });

  test("recall finds stemmed keywords, filters by all tags, and survives stopword/punctuation queries", async () => {
    const d = dir();
    const id = await store("The zzbuild pipeline is running migrations before deploys.", d, { tags: ["ci", "deploy"] });
    await store("Unrelated zzbuild note about caching layers.", d, { tags: ["ci"] });

    const stemmed = await __internals.recall({ query: "zzbuild migration" }, { directory: d });
    expect(ranked(stemmed)[0]).toBe(id);
    const tagged = await __internals.recall({ tags: ["ci", "deploy"] }, { directory: d });
    expect(tagged).toContain(`#${id}`);
    expect(tagged).not.toContain("caching");
    expect(tagged).toContain("(ci, deploy)");
    expect(await __internals.recall({ query: "the is of" }, { directory: d })).toBe("No memories found.");
    expect(await __internals.recall({ query: "!!!" }, { directory: d })).toBe("No memories found.");
  });

  test("project_fact stays in its project unless global; stack_fact is visible everywhere", async () => {
    const [a, b] = [dir(), dir()];
    const local = await store("zzvis project fact kept in project A.", a);
    const shared = await store("zzvis stack fact shared everywhere.", a, { type: "stack_fact" });
    const fromB = await __internals.recall({ query: "zzvis" }, { directory: b });
    expect(fromB).toContain(`#${shared}`);
    expect(fromB).not.toContain(`#${local}`);
    expect(await __internals.recall({ query: "zzvis", global: true }, { directory: b })).toContain(`#${local}`);
    expect(await __internals.forget({ id: local }, { directory: b })).toContain("not deleted");
    expect(await __internals.forget({ id: local, global: true }, { directory: b })).toBe(`Deleted memory #${local}.`);
  });

  test("supersede hides the old row, rolls back on a foreign target, and unlinks when the new row is forgotten", async () => {
    const [d, other] = [dir(), dir()];
    const oldId = await store("zzsup the limit is 100 requests.", d);
    const newId = await store("zzsup the limit is 500 requests.", d, { supersedes: oldId });
    expect(await __internals.recall({ query: "zzsup" }, { directory: d })).not.toContain(`#${oldId}`);
    expect(await __internals.recall({ query: "zzsup", includeSuperseded: true }, { directory: d })).toContain(
      `#${oldId} [superseded by #${newId}]`,
    );

    const foreign = await store("zzsup foreign project fact.", other);
    const before = await count();
    const refused = await __internals.remember({ content: "zzsup cannot take over.", supersedes: foreign }, { directory: d, sessionID: "t" });
    expect(refused).toContain("cannot supersede");
    expect(await count()).toBe(before);

    await __internals.forget({ id: newId }, { directory: d });
    const [row] = (await sql`SELECT superseded_by FROM memories WHERE id = ${oldId}`) as { superseded_by: number | null }[];
    expect(row.superseded_by).toBeNull();
  });

  // One SQLite connection: overlapping transactions used to fail, and a write
  // issued during a refused supersede was rolled back with it.
  test("parallel writes, including supersedes and a refused one, each succeed or fail on their own", async () => {
    const [d, other] = [dir(), dir()];
    const ctx = { directory: d, sessionID: "t" };
    const olds = [await store("zzpar old fact one.", d), await store("zzpar old fact two.", d)];
    const foreign = await store("zzpar foreign project fact.", other);
    const out = await Promise.all([
      __internals.remember({ content: "zzpar new fact one.", supersedes: olds[0] }, ctx),
      __internals.remember({ content: "zzpar new fact two.", supersedes: olds[1] }, ctx),
      __internals.remember({ content: "zzpar refused takeover.", supersedes: foreign }, ctx),
      __internals.remember({ content: "zzpar plain fact." }, ctx),
    ]);
    expect(out[2]).toContain("cannot supersede");
    for (const i of [0, 1, 3]) expect(out[i]).toMatch(/^Stored memory #\d+/);
    const rows = (await sql`SELECT content FROM memories WHERE content LIKE 'zzpar%' ORDER BY content`) as { content: string }[];
    expect(rows.map((r) => r.content)).toEqual([
      "zzpar foreign project fact.", "zzpar new fact one.", "zzpar new fact two.",
      "zzpar old fact one.", "zzpar old fact two.", "zzpar plain fact.",
    ]);
    const [{ n: linked }] = (await sql`SELECT count(*) AS n FROM memories WHERE id IN (${olds[0]}, ${olds[1]}) AND superseded_by IS NOT NULL`) as {
      n: number;
    }[];
    expect(linked).toBe(2);
  });

  test("update rewrites content, the keyword index, tags and updated_at but keeps created_at", async () => {
    const d = dir();
    const id = await store("zzupd original wording about quokkas.", d, { tags: ["a"] });
    const [before] = (await sql`SELECT created_at FROM memories WHERE id = ${id}`) as { created_at: string }[];
    expect(await __internals.updateMemory({ id, content: "zzupd corrected wording about wombats.", tags: ["b"] }, { directory: d })).toBe(
      `Updated memory #${id}.`,
    );
    expect(await __internals.recall({ query: "quokkas" }, { directory: d })).toBe("No memories found.");
    expect(await __internals.recall({ query: "wombats" }, { directory: d })).toContain("(b)");
    const [after] = (await sql`SELECT created_at, updated_at FROM memories WHERE id = ${id}`) as { created_at: string; updated_at: string }[];
    expect(after.created_at).toBe(before.created_at);
    expect(after.updated_at).toMatch(/^\d{4}-\d\d-\d\dT/);
    expect(await __internals.updateMemory({ id, content: "zzupd from elsewhere." }, { directory: dir() })).toContain("not updated");
  });

  test("recall records each session once; forget cascades those records", async () => {
    const d = dir();
    const id = await store("zzxs cross-session probe.", d);
    for (const sessionID of ["s1", "s1", "s2"]) await __internals.recall({ query: "zzxs" }, { directory: d, sessionID });
    await until(async () => (await sql`SELECT 1 FROM memory_recalls WHERE memory_id = ${id}`).length === 2, "session records");
    const sessions = (await sql`SELECT session_id FROM memory_recalls WHERE memory_id = ${id} ORDER BY session_id`) as { session_id: string }[];
    expect(sessions.map((s) => s.session_id)).toEqual(["s1", "s2"]);
    await __internals.forget({ id }, { directory: d });
    expect(await sql`SELECT 1 FROM memory_recalls WHERE memory_id = ${id}`).toHaveLength(0);
  });

  test("the cross-session tiebreak breaks a near-tie without overriding relevance", async () => {
    const d = dir();
    const stronger = await store("zzrank alpha carries zzrare too.", d);
    // Equal text rank; without the tiebreak the newer row would win on recency.
    const recalled = await store("zzrank beta plain entry.", d);
    const newer = await store("zzrank gamma plain entry.", d);
    for (let s = 0; s < 5; s++) await sql`INSERT INTO memory_recalls (memory_id, session_id) VALUES (${recalled}, ${`rank-${s}`})`;
    const out = await __internals.recall({ query: "zzrank zzrare" }, { directory: d });
    expect(ranked(out).filter((x) => [stronger, recalled, newer].includes(x))).toEqual([stronger, recalled, newer]);
  });

  test("tags lists visible tag counts and retag merges without duplicating or reordering", async () => {
    const d = dir();
    const id = await store("zztag entry one.", d, { tags: ["x-old", "keep", "x-new"] });
    await store("zztag entry two.", d, { tags: ["keep"] });
    expect(await __internals.listTags({}, { directory: d })).toStartWith("keep (2)\n");
    expect(await __internals.retag({ old: "x-old", new: "x-new" }, { directory: d })).toContain("Retagged 1 memory");
    const [row] = (await sql`SELECT tags FROM memories WHERE id = ${id}`) as { tags: string }[];
    expect(JSON.parse(row.tags)).toEqual(["x-new", "keep"]);
    expect(await __internals.retag({ old: "x-old", new: "y" }, { directory: d })).toBe('No memories tagged "x-old".');
  });

  test("injection ranks by the prompt, falls back to recency, and never shows foreign project facts", async () => {
    const [d, other] = [dir(), dir()];
    await store("zzinj the backup job runs at night.", d);
    await store("zzinj newest unrelated note.", d);
    await store("zzinj foreign backup secret.", other);
    const block = await inject(d, "when does the backup run?");
    expect(block).toContain("backup job");
    expect(block).not.toContain("newest unrelated");
    expect(block).not.toContain("foreign");
    expect(await inject(d, "qqqq nothing matches")).toContain("newest unrelated note");
  });

  test("hybrid: an embedded row is found by meaning alone, in recall and injection", async () => {
    __internals.setOllamaBase(EMBED);
    try {
      const d = dir();
      const id = await store("zzfleet car needs new tires soon.", d);
      await untilEmbedded(id);
      // Five newer rows push it out of the recency fallback's top 5.
      for (let i = 0; i < 5; i++) await store(`zzfleet unrelated filler number ${i}.`, d);
      const [row] = (await sql`SELECT length(embedding) AS bytes FROM memories WHERE id = ${id}`) as { bytes: number }[];
      expect(row.bytes).toBe(768 * 4);
      // No shared keyword: "automobile" only reaches the row through the vector.
      expect(await __internals.recall({ query: "automobile" }, { directory: d })).toContain(`#${id}`);
      expect(await inject(d, "automobile")).toContain("zzfleet car");
    } finally {
      __internals.setOllamaBase(OFF);
    }
  });

  test("update drops the old vector, so a failed re-embed leaves the row keyword-only", async () => {
    const d = dir();
    __internals.setOllamaBase(EMBED);
    const id = await store("zzstale car needs new tires soon.", d);
    await untilEmbedded(id);
    __internals.setOllamaBase(OFF);
    await __internals.updateMemory({ id, content: "zzstale the office plants need water." }, { directory: d });
    const [row] = (await sql`SELECT embedding IS NULL AS cleared FROM memories WHERE id = ${id}`) as { cleared: number }[];
    expect(row.cleared).toBe(1);
  });

  test("a late embed of the old content never overwrites the vector of an update", async () => {
    __internals.setOllamaBase(EMBED);
    try {
      const d = dir();
      const original = "zzlate original car tires wording.";
      const id = await store(original, d);
      const updated = "zzlate office plants need water.";
      await __internals.updateMemory({ id, content: updated }, { directory: d });
      await untilEmbedded(id);
      // What a slow embed of the original text would do if it finished now.
      await __internals.embedAndStore(id, original);
      const [row] = (await sql`SELECT embedding FROM memories WHERE id = ${id}`) as { embedding: Uint8Array }[];
      expect([...new Float32Array(row.embedding.slice().buffer)]).toEqual(fakeEmbed(updated));
    } finally {
      __internals.setOllamaBase(OFF);
    }
  });

  test("consolidate: dry run deletes nothing; a real run removes wording and meaning duplicates, flags detail conflicts, skips templated logs", async () => {
    __internals.setOllamaBase(EMBED);
    try {
      await sql`DELETE FROM memories`;
      const d = dir();
      const wordOld = await store("zzcons the staging database resets every night at three.", d);
      const wordNew = await store("zzcons the staging database resets every night at three!", d);
      const meanOld = await store("the zzmean car needs new tires soon", d);
      const meanNew = await store("vehicle zzmean requires fresh tyres shortly", d);
      const portA = await store("zzport service one listens on port 8080 behind proxy", d);
      const portB = await store("zzport daemon one listens on port 9090 behind proxy", d);
      // Templated auto-logs: similar enough for the meaning pass, and their
      // differing task numbers would flag them uncertain if it ever saw them.
      const logA = await store("background task bg_1 finished zzlog car needs new tires", d);
      const logB = await store("background task bg_2 finished zzlog vehicle requires fresh tyres", d);
      for (const id of [wordOld, wordNew, meanOld, meanNew, portA, portB, logA, logB]) await untilEmbedded(id);

      const dry = await __internals.consolidate({ dryRun: true });
      expect(dry).toContain("DRY RUN");
      expect(await count()).toBe(8);

      const real = await __internals.consolidate({});
      expect(real).toContain(`[wording] Kept #${wordNew}`);
      expect(real).toContain(`removed #${wordOld}`);
      expect(real).toContain(`[meaning] Kept #${meanNew}`);
      expect(real).toContain(`removed #${meanOld}`);
      expect(real).toContain(`[meaning-uncertain] #${portB} vs #${portA}`);
      expect(real).not.toContain(`#${logA}`);
      expect(real).not.toContain(`#${logB}`);
      const left = ((await sql`SELECT id FROM memories ORDER BY id`) as { id: number }[]).map((r) => r.id);
      expect(left).toEqual([wordNew, meanNew, portA, portB, logA, logB]);
    } finally {
      __internals.setOllamaBase(OFF);
    }
  });
});
