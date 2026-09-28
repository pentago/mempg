// Re-calibrates CONSOLIDATE_EMBED_THRESHOLD for embeddinggemma:300m, the
// model mempg.ts switched its default to (see the "complete the
// embeddinggemma:300m migration" spec, repo history). The threshold was
// originally calibrated against bge-m3's cosine distribution and has no
// guaranteed relationship to a different model's numbers - this measures
// the real distribution instead of assuming the value carries over.
//
// Uses the 26 hand-labeled pairs from the original bge-m3 calibration
// (8 duplicate, 8 update, 10 distinct), unchanged, so results are comparable.
//
// No database needed - embed() is a pure HTTP call. Uses whatever
// MEMPG_OLLAMA_HOST/PORT/EMBED_MODEL are already set to (the real production
// config), so this measures the exact model/endpoint the plugin actually
// calls, not a hardcoded stand-in.
//
//   bun bench/embeddinggemma-calibration.ts
import mempg from "../mempg.ts";

const { __internals } = mempg;

// Same fallback the mempg.ts module itself uses - duplicated here (not
// imported) because EMBED_MODEL is module-private; this is purely for the
// printed label, embed() always uses mempg.ts's own resolved value.
const MODEL_LABEL = process.env.MEMPG_EMBED_MODEL || "embeddinggemma:300m";

type Pair = { existing: string; incoming: string };
const PAIRS: Record<"duplicate" | "update" | "distinct", Pair[]> = {
  duplicate: [
    { existing: "The production database runs Postgres 16 on port 5432.", incoming: "Postgres 16 is the production database version, listening on port 5432." },
    { existing: "Releases are tagged only from the main branch after CI passes.", incoming: "We only tag releases off main once CI is green." },
    { existing: "The staging API key rotates every 30 days and lives in Vault.", incoming: "Staging API credentials are stored in Vault and rotate monthly." },
    { existing: "Never run migrations against production without a backup first.", incoming: "Always take a backup before running production migrations." },
    { existing: "The frontend build requires Node 20; Node 18 fails on the crypto polyfill.", incoming: "Building the frontend on Node 18 breaks because of the crypto polyfill - use Node 20." },
    { existing: "User prefers dark-themed diffs and compact tables in all reports.", incoming: "Reports should use compact tables and dark diff themes - that is the user's standing preference." },
    { existing: "The Redis cache for sessions expires after 24 hours of inactivity.", incoming: "Session entries in Redis expire a day after last activity." },
    { existing: "CI runs bun test against a live Postgres service container before merge.", incoming: "Before merging, CI executes the bun test suite against a real Postgres container." },
  ],
  update: [
    { existing: "The staging database runs Postgres 15 on port 5432.", incoming: "The staging database was upgraded to Postgres 16 last week." },
    { existing: "Deploys happen manually via `make deploy` from a maintainer laptop.", incoming: "Deploys moved to GitHub Actions: pushing a tag now triggers the pipeline." },
    { existing: "The API rate limit is 100 requests per minute per key.", incoming: "The API rate limit was raised to 500 requests per minute per key after the caching layer shipped." },
    { existing: "The team uses Jira for issue tracking.", incoming: "The team migrated issue tracking from Jira to Linear in March." },
    { existing: "bge-m3 is the embedding model used for semantic search.", incoming: "Semantic search embeddings were switched from bge-m3 to jina-v3 for the multilingual corpus." },
    { existing: "The backup job runs nightly at 02:00 UTC.", incoming: "After the incident, backups run every 6 hours, not just nightly." },
    { existing: "The monorepo uses npm workspaces.", incoming: "The monorepo migrated from npm workspaces to pnpm workspaces." },
    { existing: "Feature flags are managed in LaunchDarkly.", incoming: "Feature flags moved from LaunchDarkly to a self-hosted Flagsmith instance." },
  ],
  distinct: [
    { existing: "The production database runs Postgres 16 on port 5432.", incoming: "The analytics replica runs Postgres 16 on port 5433 with logical replication." },
    { existing: "CI runs bun test against a live Postgres container before merge.", incoming: "CI lint failures were caused by a biome version mismatch between the runner image and package.json." },
    { existing: "The Redis cache for sessions expires after 24 hours.", incoming: "Redis maxmemory is capped at 512MB with an allkeys-lru eviction policy." },
    { existing: "Releases are tagged only from the main branch after CI passes.", incoming: "Hotfix branches are named hotfix/<ticket> and branch off the latest release tag." },
    { existing: "The staging API key rotates every 30 days and lives in Vault.", incoming: "Vault unsealing requires two of three admin key shares after every restart." },
    { existing: "User prefers compact tables in reports.", incoming: "User asked for the onboarding docs to be rewritten for non-engineers." },
    { existing: "The frontend requires Node 20.", incoming: "The backend was pinned to Python 3.12 because the ORM drops 3.13 support." },
    { existing: "Migrations must never run against production without a backup.", incoming: "The migration framework squashes all migrations older than one year into a baseline." },
    { existing: "bge-m3 is the embedding model for semantic search.", incoming: "Semantic search latency p95 dropped to 40ms after moving embeddings to the GPU host." },
    { existing: "The backup job runs nightly at 02:00 UTC.", incoming: "Restores are tested quarterly by booting the backup into a scratch namespace." },
  ],
};

function cosine(a: number[], b: number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

async function scoreCategory(pairs: Pair[]): Promise<number[]> {
  const scores: number[] = [];
  for (const p of pairs) {
    const vecs = await __internals.embed([p.existing, p.incoming], 30_000);
    if (!vecs) throw new Error(`embed() returned null for pair: "${p.existing}" / "${p.incoming}" - is Ollama reachable?`);
    scores.push(cosine(vecs[0], vecs[1]));
  }
  return scores;
}

const stats = (xs: number[]) => ({
  min: Math.min(...xs),
  max: Math.max(...xs),
  mean: xs.reduce((a, b) => a + b, 0) / xs.length,
});

console.log(`Model: ${MODEL_LABEL} (via ${__internals.ollamaBase})`);
console.log("Scoring the 26 hand-labeled pairs...\n");

const dup = await scoreCategory(PAIRS.duplicate);
const upd = await scoreCategory(PAIRS.update);
const dist = await scoreCategory(PAIRS.distinct);

const d = stats(dup);
const u = stats(upd);
const dst = stats(dist);

console.log("category  | min   | max   | mean  | n  (cf. bench/README.md's embeddinggemma migration table)");
console.log(`duplicate | ${d.min.toFixed(3)} | ${d.max.toFixed(3)} | ${d.mean.toFixed(3)} | ${dup.length}`);
console.log(`update    | ${u.min.toFixed(3)} | ${u.max.toFixed(3)} | ${u.mean.toFixed(3)} | ${upd.length}`);
console.log(`distinct  | ${dst.min.toFixed(3)} | ${dst.max.toFixed(3)} | ${dst.mean.toFixed(3)} | ${dist.length}`);

console.log("\nPer-pair scores:");
for (const [i, s] of dup.entries()) console.log(`  [duplicate] ${s.toFixed(3)} - ${PAIRS.duplicate[i].existing}`);
for (const [i, s] of upd.entries()) console.log(`  [update] ${s.toFixed(3)} - ${PAIRS.update[i].existing}`);
for (const [i, s] of dist.entries()) console.log(`  [distinct] ${s.toFixed(3)} - ${PAIRS.distinct[i].existing}`);

console.log("\n=== Threshold sweep (same candidate thresholds bge-m3 was checked against) ===");
console.log("threshold | false merges (distinct >= t) | duplicates caught | updates caught");
for (const th of [0.7, 0.75, 0.78, 0.8, 0.83, 0.85, 0.88, 0.9, 0.92, 0.95]) {
  const falseMerges = dist.filter((s) => s >= th).length;
  const dupCaught = dup.filter((s) => s >= th).length;
  const updCaught = upd.filter((s) => s >= th).length;
  console.log(`${th.toFixed(2)}      | ${falseMerges}/${dist.length}                         | ${dupCaught}/${dup.length}              | ${updCaught}/${upd.length}`);
}

// The bge-m3 pick (0.83) was chosen to sit clear of every distinct pair
// (zero false merges - the failure mode that matters) while catching most
// duplicates and about half the updates. Apply the identical rule here:
// the lowest threshold with zero distinct-pair false merges.
const candidateThresholds = Array.from({ length: 41 }, (_, i) => 0.5 + i * 0.01);
const safe = candidateThresholds.filter((th) => dist.every((s) => s < th));
const recommended = safe.length > 0 ? Math.min(...safe) : Math.max(...dist) + 0.01;
const dupCaughtAtRec = dup.filter((s) => s >= recommended).length;
const updCaughtAtRec = upd.filter((s) => s >= recommended).length;
console.log(
  `\nRecommended threshold (lowest value with zero distinct-pair false merges): ${recommended.toFixed(2)} ` +
    `-> catches ${dupCaughtAtRec}/${dup.length} duplicates, ${updCaughtAtRec}/${upd.length} updates.`,
);

await __internals.dispose();
