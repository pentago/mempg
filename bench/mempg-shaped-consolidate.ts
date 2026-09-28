// Tests the detail cross-check (extractDetails/detailConflicts, gating
// consolidate()'s meaning pass) against mempg-SHAPED content. The corpus is
// built from the 40-topic infra vocabulary in bench/config.ts (postgres,
// systemd, wireguard, terraform, ...), programmatically injecting realistic
// numbers/paths/proper-noun values so the failure mode the two real
// incidents were shaped like (a rate limit changing 100->500, a
// system-vs-user-level systemd path) actually occurs at real density - not
// by chance, and not hand-picked pair by pair like the 26-pair set.
//
// Four labeled categories (ground truth is_duplicate, binary):
//   - duplicate:        same fact, reworded, injected detail UNCHANGED (1)
//   - update-<shape>:   same shape, injected detail CHANGED - a genuinely
//                       different fact (rate limit/port/retry-count/path/
//                       tool-name changing) - exactly the two real
//                       incidents' shape (0)
//   - distinct-related: same topic, different aspect, general templates -
//                       the "topically related but different fact" false-
//                       positive class (0)
//   - fn-stress:        hand-picked near-duplicates where a number/path/name
//                       differs only in FORMATTING (comma grouping, trailing
//                       zero, trailing slash, letter case, leading zero) or
//                       is spelled out instead of numeric - true duplicates
//                       (1) that stress-test whether the detail check
//                       produces NEW false negatives, per the spec's
//                       explicit ask (includes some pairs expected to pass
//                       cleanly, to show the design's "absence isn't a
//                       conflict" rule works as intended too)
//
// Setup: none beyond a reachable Ollama + a CREATEDB-capable MEMPG_* user -
// the corpus is generated in-process, no external dataset download.
//   bun bench/mempg-shaped-consolidate.ts [--ollama http://localhost:11434] [--dims 768]
//
// Uses a dedicated scratch database (`mempg-shaped-consolidate-bench`) -
// never the real DB. Not a change to
// extractDetails/detailConflicts/CONSOLIDATE_EMBED_THRESHOLD - measurement
// only.
import { SQL } from "bun";
import mempg from "../mempg.ts";
import { FILLERS, TEMPLATES, TOPICS, makeSql, pick, rng } from "./config.ts";

const { __internals } = mempg;

const args = process.argv.slice(2);
const opt = (name: string, fallback: string): string => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
};
__internals.setOllamaBase(
  opt(
    "--ollama",
    process.env.MEMPG_BENCH_OLLAMA ??
      `http://${process.env.MEMPG_OLLAMA_HOST || "localhost"}:${process.env.MEMPG_OLLAMA_PORT || "11434"}`,
  ),
);
// The vector column's dimension must match whatever model mempg.ts is
// currently configured to embed with (MEMPG_EMBED_MODEL) - there is no way to
// derive dimension from a model name, so it's an explicit flag rather than
// inferred. Defaults to embeddinggemma:300m's 768, the shipped default as of
// the embeddinggemma migration; pass --dims 1024 to reproduce against bge-m3.
const DIMS = Number(opt("--dims", "768"));
// Purely for the printed log line - EMBED_MODEL itself is module-private in
// mempg.ts, so embed() always uses whatever it actually resolved to.
const MODEL_LABEL = process.env.MEMPG_EMBED_MODEL || "embeddinggemma:300m";

const DB_NAME = "mempg-shaped-consolidate-bench";
const PROJECT = "/bench/mempg-shaped-consolidate";

type Pair = { content1: string; content2: string; label: 0 | 1; category: string };
type Shape = { key: string; a: (s: string, v: string) => string; b: (s: string, v: string) => string; values: string[] };

// --- Shape templates: "a" establishes the fact, "b" is a paraphrase of the
// SAME shape (different word order/framing), both referencing the injected
// value {v}. Using the same value on both sides makes a duplicate pair;
// using DIFFERENT pool values makes an update pair (genuinely different
// fact - the shape both real incidents had).
const NUMBER_SHAPES: Shape[] = [
  {
    key: "rate-limit",
    a: (s, v) => `The ${s} rate limit is ${v} requests per minute.`,
    b: (s, v) => `${v} requests per minute is the configured rate limit for the ${s} service.`,
    values: ["100", "500", "1000"],
  },
  {
    key: "port",
    a: (s, v) => `The ${s} service listens on port ${v}.`,
    b: (s, v) => `Port ${v} is where the ${s} service accepts connections.`,
    values: ["5432", "8080", "9090"],
  },
  {
    key: "retry-count",
    a: (s, v) => `Failed ${s} jobs are retried ${v} times before an alert fires.`,
    b: (s, v) => `The ${s} pipeline retries a failed job ${v} times before paging anyone.`,
    values: ["3", "5", "10"],
  },
];

const PATH_SHAPES: Array<{ key: string; a: (s: string, v: string) => string; b: (s: string, v: string) => string; valueTemplates: string[] }> = [
  {
    key: "system-vs-user-unit",
    a: (s, v) => `The ${s} unit is installed system-wide at ${v}.`,
    b: (s, v) => `${v} is where the ${s} unit lives on this machine.`,
    // The exact shape of the second real incident (system vs user systemd path).
    valueTemplates: ["/etc/systemd/system/{slug}.service", "~/.config/systemd/user/{slug}.service"],
  },
  {
    key: "config-location",
    a: (s, v) => `The ${s} config file lives at ${v}.`,
    b: (s, v) => `${v} is where the ${s} configuration is stored.`,
    valueTemplates: ["/etc/{slug}/{slug}.conf", "/opt/{slug}/etc/{slug}.conf"],
  },
  {
    key: "env-values-file",
    a: (s, v) => `The staging values file for ${s} is at ${v}.`,
    b: (s, v) => `${v} holds the staging configuration values for ${s}.`,
    valueTemplates: ["charts_values/environments/staging/{slug}.values.yaml", "charts_values/environments/test/{slug}.values.yaml"],
  },
];

const NAME_SHAPES: Shape[] = [
  {
    key: "alert-routing",
    a: (s, v) => `The ${s} alerts are routed through ${v}.`,
    b: (s, v) => `${v} is the tool used to route ${s} alerts.`,
    values: ["PagerDuty", "Opsgenie"],
  },
  {
    key: "dependency-bot",
    a: (s, v) => `${v} opens the dependency-update pull requests for ${s}.`,
    b: (s, v) => `Dependency PRs for ${s} are opened automatically by ${v}.`,
    values: ["Renovate", "Dependabot"],
  },
  {
    key: "ci-provider",
    a: (s, v) => `${s} pipelines run on ${v}.`,
    b: (s, v) => `${v} is where ${s} pipelines execute.`,
    values: ["GitHub Actions", "GitLab CI"],
  },
];

// --- Hand-picked false-negative stress pairs: genuine near-duplicates where
// a number/path/name differs only cosmetically (or is absent/spelled out on
// one side). Labeled duplicate=1; some are EXPECTED to get blocked (the
// known formatting-sensitivity gap of a regex-only check, documented
// honestly) and some are expected to pass cleanly (the "absence/spelled-out
// isn't a conflict" design already handles these). The spec's own ambiguous
// example (a genuine schedule change) is included and called out below.
const FN_STRESS: Array<[string, string]> = [
  ["The API rate limit is 1,000 requests per minute.", "The API rate limit is 1000 requests per minute."], // comma grouping
  ["The release version is 2.5.0.", "The release is now at version 2.5."], // trailing zero dropped
  ["Application logs are shipped to /var/log/app.", "Application logs land in /var/log/app/."], // trailing slash
  ["Reach out to Jira for ticket status.", "Reach out to JIRA for ticket status."], // letter case
  ["The nightly backup starts at 02:00.", "The nightly backup starts at 2:00 AM."], // leading zero
  ["Meeting moved from 3pm to 4pm.", "The meeting is now scheduled for 4pm."], // spec's own ambiguous example - a genuine change, arguably still "the same meeting"
  ["The staging cluster runs 3 replicas.", "Staging currently runs three replicas of the service."], // digit vs spelled-out - expected to pass (absence, not conflict)
  ["The Terraform state lock times out after 300 seconds.", "The Terraform state lock timeout is 300s."], // unit suffix - expected to pass (same token "300" both sides)
];

const pairs: Pair[] = [];
const rand = rng(1337);

for (const topic of TOPICS) {
  const s = topic.words[0];
  const slug = s.replace(/[^a-z0-9]+/gi, "-");

  for (const shape of NUMBER_SHAPES) {
    pairs.push({ content1: shape.a(s, shape.values[0]), content2: shape.b(s, shape.values[0]), label: 1, category: "duplicate" });
    pairs.push({ content1: shape.a(s, shape.values[0]), content2: shape.b(s, shape.values[1]), label: 0, category: `update-${shape.key}` });
  }
  for (const shape of PATH_SHAPES) {
    const v0 = shape.valueTemplates[0].replace("{slug}", slug);
    const v1 = shape.valueTemplates[1].replace("{slug}", slug);
    pairs.push({ content1: shape.a(s, v0), content2: shape.b(s, v0), label: 1, category: "duplicate" });
    pairs.push({ content1: shape.a(s, v0), content2: shape.b(s, v1), label: 0, category: `update-${shape.key}` });
  }
  for (const shape of NAME_SHAPES) {
    pairs.push({ content1: shape.a(s, shape.values[0]), content2: shape.b(s, shape.values[0]), label: 1, category: "duplicate" });
    pairs.push({ content1: shape.a(s, shape.values[0]), content2: shape.b(s, shape.values[1]), label: 0, category: `update-${shape.key}` });
  }

  // distinct-related: same topic, two independently-sampled template fills. No
  // injected-detail overlap requirement; this is the "topically related but
  // different fact" false-positive class, not the detail-conflict class.
  for (let i = 0; i < 2; i++) {
    const build = () =>
      [0, 1]
        .map(() => pick(rand, TEMPLATES).replaceAll("{s}", () => pick(rand, topic.words)).replaceAll("{f}", pick(rand, FILLERS)))
        .join(" ");
    pairs.push({ content1: build(), content2: build(), label: 0, category: "distinct-related" });
  }
}

for (const [content1, content2] of FN_STRESS) pairs.push({ content1, content2, label: 1, category: "fn-stress" });

console.log(
  `Built ${pairs.length} labeled pairs (${pairs.filter((p) => p.label === 1).length} duplicate / ${pairs.filter((p) => p.label === 0).length} distinct) ` +
    `across ${new Set(pairs.map((p) => p.category)).size} categories.`,
);

// Schema mirrors deploy/init/01-init.sh's memories table (what
// memory_consolidate's meaning pass actually queries), plus a category
// column on the ground-truth table this bench uses for its breakdown.
const DDL = `
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE memories (
  id            serial PRIMARY KEY,
  content       text        NOT NULL,
  tags          text[]      NOT NULL DEFAULT '{}',
  session_id    text,
  project       text,
  created_at    timestamptz DEFAULT now(),
  memory_type   text        NOT NULL DEFAULT 'project_fact',
  embedding     vector(${DIMS}),
  CONSTRAINT memories_type_check CHECK (memory_type IN ('stack_fact', 'project_fact', 'episodic'))
);
CREATE INDEX idx_memories_embedding ON memories USING hnsw (embedding vector_cosine_ops);

CREATE TABLE ground_truth (
  memory_id_1  integer NOT NULL REFERENCES memories(id),
  memory_id_2  integer NOT NULL REFERENCES memories(id),
  is_duplicate boolean NOT NULL,
  category     text    NOT NULL
);
`;

const admin = new SQL(makeSql("postgres"));
await admin`DROP DATABASE IF EXISTS ${admin(DB_NAME)}`;
await admin`CREATE DATABASE ${admin(DB_NAME)}`;
await admin.close({ timeout: 0 });

const db = new SQL(makeSql(DB_NAME));
try {
  await db.unsafe(DDL);

  console.log("Inserting memories (2 rows per pair) + ground-truth mapping...");
  const BATCH = 250;
  for (let b = 0; b < pairs.length; b += BATCH) {
    const slice = pairs.slice(b, b + BATCH);
    await db.begin(async (tx) => {
      for (const p of slice) {
        const [r1] = (await tx`
          INSERT INTO memories (content, project, memory_type) VALUES (${p.content1}, ${PROJECT}, 'project_fact') RETURNING id
        `) as { id: number }[];
        const [r2] = (await tx`
          INSERT INTO memories (content, project, memory_type) VALUES (${p.content2}, ${PROJECT}, 'project_fact') RETURNING id
        `) as { id: number }[];
        await tx`
          INSERT INTO ground_truth (memory_id_1, memory_id_2, is_duplicate, category) VALUES (${r1.id}, ${r2.id}, ${p.label === 1}, ${p.category})
        `;
      }
    });
  }
  console.log(`Inserted ${pairs.length * 2} memories, ${pairs.length} ground-truth rows.`);

  console.log(`Embedding rows via the real pipeline (__internals.embed, ${MODEL_LABEL})...`);
  const allRows = (await db`SELECT id, content FROM memories ORDER BY id`) as { id: number; content: string }[];
  const EMBED_BATCH = 100;
  let embedded = 0;
  let failedBatches = 0;
  for (let b = 0; b < allRows.length; b += EMBED_BATCH) {
    const slice = allRows.slice(b, b + EMBED_BATCH);
    const vecs = await __internals.embed(
      slice.map((r) => r.content),
      30_000,
    );
    if (!vecs) {
      failedBatches++;
      continue;
    }
    await db.begin(async (tx) => {
      for (let i = 0; i < slice.length; i++) {
        await tx`UPDATE memories SET embedding = ${__internals.vectorLiteral(vecs[i])}::vector WHERE id = ${slice[i].id}`;
      }
    });
    embedded += slice.length;
  }
  console.log(`Embedded ${embedded}/${allRows.length} rows (${failedBatches} batches failed - Ollama unavailable for those).`);

  // Negative control - this content is synthetic and infra-flavored, but
  // never matches the three fixed sentence shapes isTemplatedAutoLog
  // targets (mempg's own auto-generated logs).
  const [{ n: templated }] = (await db`
    SELECT count(*)::int AS n FROM memories WHERE ${__internals.isTemplatedAutoLog(db)}
  `) as { n: number }[];
  console.log(`\nExclusion filter (isTemplatedAutoLog) fired on ${templated}/${allRows.length} rows (expected: 0).`);

  const scored = (await db`
    SELECT g.is_duplicate, g.category, (1 - (m1.embedding <=> m2.embedding))::float8 AS cosine, m1.content AS c1, m2.content AS c2
    FROM ground_truth g
    JOIN memories m1 ON m1.id = g.memory_id_1
    JOIN memories m2 ON m2.id = g.memory_id_2
    WHERE m1.embedding IS NOT NULL AND m2.embedding IS NOT NULL
  `) as { is_duplicate: boolean; category: string; cosine: number; c1: string; c2: string }[];
  console.log(`Scored ${scored.length}/${pairs.length} pairs (rest skipped - embedding unavailable).`);

  const dup = scored.filter((r) => r.is_duplicate).map((r) => r.cosine);
  const dist = scored.filter((r) => !r.is_duplicate).map((r) => r.cosine);
  const stats = (xs: number[]) => ({ min: Math.min(...xs), max: Math.max(...xs), mean: xs.reduce((a, b) => a + b, 0) / xs.length });

  console.log("\n=== Calibration table ===");
  const d1 = stats(dup);
  const d0 = stats(dist);
  console.log("category  | min   | max   | mean  | n");
  console.log(`duplicate | ${d1.min.toFixed(3)} | ${d1.max.toFixed(3)} | ${d1.mean.toFixed(3)} | ${dup.length}`);
  console.log(`distinct  | ${d0.min.toFixed(3)} | ${d0.max.toFixed(3)} | ${d0.mean.toFixed(3)} | ${dist.length}`);

  console.log("\n=== Threshold sweep (before the detail check) ===");
  console.log("threshold | precision | recall | FPR   | TP     FP     FN     TN");
  const T = __internals.CONSOLIDATE_EMBED_THRESHOLD;
  const thresholds = [0.75, 0.8, T, 0.86, 0.9];
  for (const t of thresholds) {
    let tp = 0;
    let fp = 0;
    let fn = 0;
    let tn = 0;
    for (const r of scored) {
      const predicted = r.cosine >= t;
      if (predicted && r.is_duplicate) tp++;
      else if (predicted && !r.is_duplicate) fp++;
      else if (!predicted && r.is_duplicate) fn++;
      else tn++;
    }
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const fpr = fp + tn > 0 ? fp / (fp + tn) : 0;
    console.log(
      `${t.toFixed(2)}      | ${precision.toFixed(3)}     | ${recall.toFixed(3)}  | ${fpr.toFixed(3)} | ${tp}    ${fp}    ${fn}    ${tn}${t === T ? " <- shipped" : ""}`,
    );
  }

  // --- Detail cross-check re-run at the shipped threshold.
  console.log(`\n=== Detail cross-check applied on top of the shipped threshold (${T}) ===`);
  const candidates = scored.filter((r) => r.cosine >= T);
  let cleanTp = 0;
  let cleanFp = 0;
  let uncertainTp = 0;
  let uncertainFp = 0;
  const uncertainRows: typeof scored = [];
  for (const r of candidates) {
    const conflicts = __internals.detailConflicts(__internals.extractDetails(r.c1), __internals.extractDetails(r.c2));
    const conflicting = conflicts.length > 0;
    if (conflicting) uncertainRows.push(r);
    if (conflicting && r.is_duplicate) uncertainTp++;
    else if (conflicting && !r.is_duplicate) uncertainFp++;
    else if (!conflicting && r.is_duplicate) cleanTp++;
    else cleanFp++;
  }
  const totalDup = scored.filter((r) => r.is_duplicate).length;
  const totalDist = scored.filter((r) => !r.is_duplicate).length;
  const baselineFp = candidates.filter((r) => !r.is_duplicate).length;
  const baselineTp = candidates.filter((r) => r.is_duplicate).length;

  console.log(`Candidates at threshold (cosine >= ${T}): ${candidates.length} (${baselineTp} true duplicate / ${baselineFp} distinct)`);
  console.log(`  -> clean (no detail conflict, auto-merged same as before): TP=${cleanTp} FP=${cleanFp}`);
  console.log(`  -> [meaning-uncertain] (detail conflict, NOT merged, flagged for review): TP=${uncertainTp} FP=${uncertainFp}`);

  const precisionAuto = cleanTp + cleanFp > 0 ? cleanTp / (cleanTp + cleanFp) : 0;
  const recallAutoOnly = totalDup > 0 ? cleanTp / totalDup : 0;
  const recallIncludingReview = totalDup > 0 ? (cleanTp + uncertainTp) / totalDup : 0;
  const fprAuto = totalDist > 0 ? cleanFp / totalDist : 0;
  console.log(
    `\nprecision (auto-merged only):             ${precisionAuto.toFixed(3)} (baseline: ${(baselineTp / candidates.length).toFixed(3)})`,
  );
  console.log(`recall (auto-merged only, silent):        ${recallAutoOnly.toFixed(3)}`);
  console.log(
    `recall (auto-merged + flagged for review): ${recallIncludingReview.toFixed(3)} (baseline: ${(baselineTp / totalDup).toFixed(3)})`,
  );
  console.log(`FPR (auto-merged only):                    ${fprAuto.toFixed(3)} (baseline: ${(baselineFp / totalDist).toFixed(3)})`);
  console.log(
    `\nPreviously-false-positive pairs now correctly routed to [meaning-uncertain]: ${uncertainFp}/${baselineFp}${baselineFp > 0 ? ` (${((uncertainFp / baselineFp) * 100).toFixed(1)}%)` : ""}`,
  );
  console.log(
    `Previously-true-positive pairs now blocked from auto-merge (review, not silent loss): ${uncertainTp}/${baselineTp}${baselineTp > 0 ? ` (${((uncertainTp / baselineTp) * 100).toFixed(1)}%)` : ""}`,
  );
  console.log(`[meaning-uncertain] bucket size (share of ALL scored pairs): ${(((uncertainTp + uncertainFp) / scored.length) * 100).toFixed(1)}%`);

  // --- Per-category breakdown: whether the check actually catches the exact
  // failure mode it targets (update-* categories) vs categories it was never
  // meant to help with (distinct-related) vs whether it introduces new
  // false negatives (fn-stress).
  console.log("\n=== Per-category breakdown (candidates only, cosine >= threshold) ===");
  console.log("category            | n candidates | clean | uncertain | ground truth");
  const categories = [...new Set(scored.map((r) => r.category))].sort();
  for (const cat of categories) {
    const rows = candidates.filter((r) => r.category === cat);
    if (rows.length === 0) {
      console.log(`${cat.padEnd(19)} | 0            | -     | -         | (no candidates at this threshold)`);
      continue;
    }
    let clean = 0;
    let uncertain = 0;
    for (const r of rows) {
      const conflicts = __internals.detailConflicts(__internals.extractDetails(r.c1), __internals.extractDetails(r.c2));
      if (conflicts.length > 0) uncertain++;
      else clean++;
    }
    const truth = rows[0].is_duplicate ? "duplicate" : "distinct";
    console.log(`${cat.padEnd(19)} | ${String(rows.length).padEnd(12)} | ${String(clean).padEnd(5)} | ${String(uncertain).padEnd(9)} | ${truth}`);
  }

  console.log("\n=== fn-stress pairs individually (false-negative risk detail) ===");
  for (const r of scored.filter((row) => row.category === "fn-stress")) {
    const conflicts = __internals.detailConflicts(__internals.extractDetails(r.c1), __internals.extractDetails(r.c2));
    const candidate = r.cosine >= T;
    const outcome = !candidate ? "below threshold (not a candidate)" : conflicts.length > 0 ? `BLOCKED - ${conflicts.join("; ")}` : "merged cleanly";
    console.log(`cosine=${r.cosine.toFixed(3)} | ${outcome}\n  a: ${r.c1}\n  b: ${r.c2}`);
  }
} finally {
  await db.close({ timeout: 0 }).catch(() => {});
  await __internals.dispose();
}
