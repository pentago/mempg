// Compares whichever backend MEMPG_BACKEND selects (postgres by default,
// sqlite opt-in) on the operations the agent actually runs: recall,
// injection, remember/update/forget, tags/retag and consolidate.
//
// Quality uses a synthetic known-answer set (below, fictional facts): each
// item is stored, then looked up by a keyword query and by a paraphrase that
// avoids its distinctive words. Rank of the stored id gives hit@1/hit@5/MRR,
// keyword-only (embeddings forced off) and hybrid. Any rows already in the
// database act as distractors, so a copy of a real corpus makes the ranking
// realistic without sending its content anywhere.
//
// MUTATES the target database (inserts, updates, deletes, and a real
// consolidate at the end), so it refuses to run unless MEMPG_DB (postgres) or
// MEMPG_SQLITE_PATH (sqlite) contains "bench":
//   MEMPG_DB=mempg_bench bun bench/backend-compare.ts out.json
//   MEMPG_BACKEND=sqlite MEMPG_SQLITE_PATH=/tmp/mempg-bench.db bun bench/backend-compare.ts out.json
import mempg from "../mempg.ts";

const { __internals } = mempg;
const backend = process.env.MEMPG_BACKEND === "sqlite" ? "sqlite" : "pgvector";
const target = backend === "sqlite" ? (process.env.MEMPG_SQLITE_PATH ?? "") : (process.env.MEMPG_DB ?? "");
if (!target.includes("bench")) {
  console.error(`refusing to run: ${backend} target "${target}" does not contain "bench"`);
  process.exit(1);
}
const outPath = process.argv[2];

const DIR = "/tmp/mempg-bench-project";
const ctx = { directory: DIR, sessionID: `bench-${Date.now()}` };

// A failed write must stop the run: a NaN id would silently count as a miss.
function storedId(out: string): number {
  const id = out.match(/^Stored memory #(\d+)/)?.[1];
  if (!id) throw new Error(`store failed: ${out}`);
  return Number(id);
}

// Fictional facts in the shape of real memories: specific, with names,
// numbers and paths. keyword reuses key terms; paraphrase avoids them.
const ITEMS: Array<{ content: string; keyword: string; paraphrase: string }> = [
  { content: "The quokka-api service must run with GOMAXPROCS=3 because its container limit is 3 CPUs and the Go runtime otherwise oversubscribes threads.", keyword: "quokka-api GOMAXPROCS", paraphrase: "why do we cap the go scheduler thread count for that small container service" },
  { content: "Heliotrope cluster etcd snapshots are written to /srv/etcd-snap every 30 minutes and pruned after 48 hours by a systemd timer.", keyword: "etcd snapshot prune timer", paraphrase: "how often is the key-value store of the control plane backed up and when are old copies deleted" },
  { content: "Marmoset CI runners lose their Docker layer cache on reboot because /var/lib/docker sits on tmpfs; builds after a reboot take about 9 minutes instead of 2.", keyword: "marmoset runner docker cache tmpfs", paraphrase: "why are container image builds slow right after the build machines restart" },
  { content: "The billing-sync cron job must not overlap: it takes a flock on /run/billing-sync.lock and exits 0 when the lock is held.", keyword: "billing-sync flock overlap", paraphrase: "what stops two copies of the invoice synchronisation task from running at the same time" },
  { content: "Pelican staging database resets nightly at 03:15 UTC from the anonymised production dump; never store test fixtures there.", keyword: "pelican staging reset nightly", paraphrase: "is data I put in the pre-production database kept, or does it get wiped" },
  { content: "The wombat frontend needs NODE_OPTIONS=--max-old-space-size=6144 for production builds; the default heap crashes webpack at chunk optimisation.", keyword: "wombat max-old-space-size webpack", paraphrase: "the web app bundler runs out of memory during the release build, what setting fixes it" },
  { content: "Ocelot VPN uses WireGuard on UDP 51871 with MTU 1380; the default MTU fragments packets over the LTE backup link.", keyword: "ocelot wireguard mtu 1380", paraphrase: "why do remote tunnel connections break over the mobile fallback connection" },
  { content: "Terraform state for the lynx account lives in the s3 bucket lynx-tfstate-eu with DynamoDB table lynx-tflock for locking.", keyword: "lynx terraform state bucket", paraphrase: "where is the infrastructure-as-code bookkeeping for that cloud account stored" },
  { content: "Capybara search reindex must run with --batch-size 500; larger batches exceed the 10 MB request limit of the managed OpenSearch domain.", keyword: "capybara reindex batch-size", paraphrase: "rebuilding the search index fails with payload too large errors, how big should chunks be" },
  { content: "The narwhal mobile app pins TLS certificates; rotating the api.narwhal.example certificate requires shipping a new app build first.", keyword: "narwhal certificate pinning rotation", paraphrase: "what must happen before we can replace the https cert used by the phone client" },
  { content: "Kestrel log shipping drops lines longer than 16 KiB; the vector sink truncates them unless max_line_bytes is raised.", keyword: "kestrel vector max_line_bytes", paraphrase: "some very long application output lines never show up in the central logging system" },
  { content: "Axolotl redis is configured with maxmemory-policy allkeys-lfu; sessions are stored elsewhere because eviction would log users out.", keyword: "axolotl redis maxmemory-policy", paraphrase: "why are user logins not kept in the in-memory cache server" },
  { content: "The tapir release script refuses to publish from a dirty worktree and requires the tag to equal package.json version.", keyword: "tapir release dirty worktree tag", paraphrase: "shipping a new version fails when there are uncommitted changes, is that intended" },
  { content: "Gecko prometheus scrapes node exporters every 15s but the blackbox probes every 60s to stay under the upstream rate limit.", keyword: "gecko prometheus scrape interval blackbox", paraphrase: "why do the external uptime checks run less often than the host metrics collection" },
  { content: "Mongoose rabbitmq consumers use prefetch 20; higher values starved the slow PDF workers and grew the dead-letter queue.", keyword: "mongoose rabbitmq prefetch", paraphrase: "how many messages should each queue worker take at once for the document rendering jobs" },
  { content: "Iguana laptops boot with systemd-boot and the fallback entry arch-lts.conf; keep the LTS kernel installed as the recovery path.", keyword: "iguana systemd-boot lts fallback", paraphrase: "what is the backup option at startup if the newest kernel fails on the workstations" },
  { content: "The platypus API paginates with opaque cursors that expire after 10 minutes; clients must restart the listing on HTTP 410.", keyword: "platypus cursor pagination 410", paraphrase: "long running list exports suddenly fail with a gone status, what should the client do" },
  { content: "Falcon backups use restic with a 7 daily, 4 weekly, 12 monthly retention and run restic check --read-data-subset=5% on Sundays.", keyword: "falcon restic retention check", paraphrase: "how long do we keep old copies of the servers and how is their integrity verified" },
  { content: "Dingo SSH access goes through the jump host bastion.dingo.example with ProxyJump; direct port 22 is firewalled from the internet.", keyword: "dingo ssh proxyjump bastion", paraphrase: "how do I get a remote shell on those machines from outside the office network" },
  { content: "The koala python service is locked with uv and must be built on Python 3.12; 3.13 breaks its pinned numpy wheel.", keyword: "koala uv python 3.12 numpy", paraphrase: "which interpreter version is required for that data service and why not the newest one" },
  { content: "Badger postgres runs autovacuum with scale factor 0.02 on the events table; the default let it bloat to 40 GB.", keyword: "badger autovacuum scale factor events", paraphrase: "the big append-heavy table kept growing on disk until cleanup settings were tuned, what value" },
  { content: "Heron nginx upstream keepalive is 64 connections; without it the TLS handshake cost doubled p95 latency to the app servers.", keyword: "heron nginx keepalive upstream", paraphrase: "why did response times at the reverse proxy get much worse without connection reuse" },
  { content: "Vulture feature flags are read from /etc/vulture/flags.toml at start only; changing a flag needs a service restart, not a reload.", keyword: "vulture feature flags restart", paraphrase: "I toggled a switch in the config file and nothing happened, how do I make it take effect" },
  { content: "The meerkat git repo uses a pre-push hook that runs the full test suite; push with --no-verify only for docs-only changes.", keyword: "meerkat pre-push hook tests", paraphrase: "pushing to that repository takes forever because checks run first, when may I skip them" },
  { content: "Jackal DNS zone example-jackal.net has TTL 300 on the api record so failover to the secondary region propagates within five minutes.", keyword: "jackal dns ttl failover", paraphrase: "how quickly do clients follow when we switch traffic to the other region" },
  { content: "Osprey yubikey sudo uses pam_u2f with cue enabled; without the cue prompt users did not know they had to touch the key.", keyword: "osprey pam_u2f cue yubikey", paraphrase: "people kept thinking admin commands had hung when the hardware token was waiting" },
  { content: "The lemur data pipeline writes parquet partitioned by day to /data/lemur/dt=YYYY-MM-DD and compacts files smaller than 64 MB weekly.", keyword: "lemur parquet partition compaction", paraphrase: "how are the analytics output files laid out on disk and when are tiny ones merged" },
  { content: "Tortoise k8s nodes are drained with --delete-emptydir-data and a 300s timeout before kernel upgrades; PDBs block faster drains.", keyword: "tortoise drain emptydir timeout", paraphrase: "what is the procedure for evicting workloads from cluster machines before patching them" },
  { content: "The cobra auth service issues access tokens valid for 15 minutes and refresh tokens for 30 days, rotated on every use.", keyword: "cobra token expiry refresh rotation", paraphrase: "how long does a login stay valid before the client has to renew its credentials" },
  { content: "Walrus image builds pin base images by digest and a weekly renovate job bumps them; tags alone let an upstream push change prod.", keyword: "walrus pin digest renovate", paraphrase: "why do the container definitions reference long hashes instead of simple version labels" },
];

const ms = (t0: number) => performance.now() - t0;
function stats(xs: number[]): { p50: number; p95: number; mean: number } {
  const s = [...xs].sort((a, b) => a - b);
  const q = (p: number) => s[Math.min(s.length - 1, Math.floor(p * s.length))];
  return { p50: q(0.5), p95: q(0.95), mean: xs.reduce((a, b) => a + b, 0) / xs.length };
}
async function timed<T>(fn: () => Promise<T>, into: number[]): Promise<T> {
  const t0 = performance.now();
  const r = await fn();
  into.push(ms(t0));
  return r;
}
const ids = (text: string) => [...text.matchAll(/^#(\d+)/gm)].map((m) => Number(m[1]));
// Tools never throw; an ERROR result or a wrong prefix must still stop the run.
function ok(out: string, prefix: string): string {
  if (out.startsWith("ERROR") || !out.startsWith(prefix)) throw new Error(`unexpected result: ${out.slice(0, 200)}`);
  return out;
}

// Waits for every write-path embed, then insists they all landed.
async function untilEmbedded(): Promise<void> {
  await __internals.settleEmbeddings();
  const [row] = (await __internals.sql`SELECT count(*) AS n FROM memories WHERE project = ${DIR} AND embedding IS NULL`) as { n: number | string }[];
  if (Number(row.n) !== 0) throw new Error(`${row.n} rows left without an embedding; is Ollama up?`);
}

// Only the deliberate keyword-only mode may log; any other line (a failed
// embed write, cross-session record, ...) invalidates the run.
const logs: string[] = [];
__internals.setLogSink((line) => {
  if (!line.includes("embedding unavailable")) logs.push(line);
});

const ollama = __internals.ollamaBase;
const OFF = "http://127.0.0.1:9";
const results: Record<string, unknown> = { backend };
const t: Record<string, number[]> = {};
const lat = (k: string) => (t[k] ??= []);

await __internals.sql`DELETE FROM memories WHERE project = ${DIR}`;
const [{ n: corpus }] = (await __internals.sql`SELECT count(*) AS n FROM memories`) as { n: number | string }[];
results.corpusRows = Number(corpus);
if (!(await __internals.embed(["bench warmup"], 15_000))) throw new Error("Ollama unreachable");

// --- store: synthetic items (stack_fact so every recall sees them) ---
const itemIds: number[] = [];
for (const it of ITEMS) {
  const out = await timed(() => __internals.remember({ content: it.content, type: "stack_fact", tags: ["bench"] }, ctx), lat("remember"));
  itemIds.push(storedId(out));
}
await untilEmbedded();

// --- recall quality + latency ---
function score(ranks: number[]) {
  return {
    hit1: ranks.filter((r) => r === 1).length / ranks.length,
    hit5: ranks.filter((r) => r > 0 && r <= 5).length / ranks.length,
    mrr: ranks.reduce((a, r) => a + (r > 0 ? 1 / r : 0), 0) / ranks.length,
  };
}
for (const mode of ["keyword", "hybrid"] as const) {
  __internals.setOllamaBase(mode === "keyword" ? OFF : ollama);
  for (const kind of ["keyword", "paraphrase"] as const) {
    const ranks: number[] = [];
    for (const [i, it] of ITEMS.entries()) {
      const out = ok(await timed(() => __internals.recall({ query: it[kind], limit: 10 }, ctx), lat(`recall-${mode}`)), "");
      ranks.push(ids(out).indexOf(itemIds[i]) + 1);
    }
    results[`recall ${mode} / ${kind} queries`] = score(ranks);
  }
}
__internals.setOllamaBase(ollama);

// --- injection: cache cleared per call so every call queries ---
let injHits = 0;
for (const it of ITEMS) {
  __internals.invalidateInjection(DIR);
  const o = { system: [] as string[] };
  await timed(() => __internals.handleTransform(o, DIR, it.paraphrase), lat("inject-hybrid"));
  const block = o.system.join("");
  if (!block) throw new Error("injection produced no block");
  if (block.includes(it.content.slice(0, 60))) injHits++;
}
results["injection hybrid / paraphrase in top 5"] = injHits / ITEMS.length;
for (let i = 0; i < 20; i++) {
  __internals.invalidateInjection(DIR);
  const o = { system: [] as string[] };
  await timed(() => __internals.handleTransform(o, DIR, ""), lat("inject-recency"));
  if (!o.system.length) throw new Error("injection produced no block");
}

// --- write path ---
const scratch: number[] = [];
for (let i = 0; i < 50; i++) {
  const out = await timed(() => __internals.remember({ content: `bench scratch memory ${i} about nothing in particular`, tags: ["bench-scratch"] }, ctx), lat("remember"));
  scratch.push(storedId(out));
}
for (const id of scratch) {
  ok(await timed(() => __internals.updateMemory({ id, content: `bench scratch memory ${id} rewritten once` }, ctx), lat("update")), "Updated");
}
for (let i = 0; i < 20; i++) ok(await timed(() => __internals.listTags({}, ctx), lat("tags")), "");
for (let i = 0; i < 10; i++) {
  ok(await timed(() => __internals.retag({ old: "bench-scratch", new: "bench-scratch2" }, ctx), lat("retag")), "Retagged 50");
  ok(await timed(() => __internals.retag({ old: "bench-scratch2", new: "bench-scratch" }, ctx), lat("retag")), "Retagged 50");
}
await untilEmbedded();
for (const id of scratch) ok(await timed(() => __internals.forget({ id }, ctx), lat("forget")), "Deleted");

// --- consolidate: dry runs (no deletes), then one real run ---
let dry = "";
for (let i = 0; i < 5; i++) dry = ok(await timed(() => __internals.consolidate({ dryRun: true }), lat("consolidate-dry")), "");
results["consolidate dry summary"] = dry.split("\n").slice(0, 3).join(" | ");
results["consolidate dry counts"] = {
  wording: (dry.match(/^\[wording\]/gm) ?? []).length,
  meaning: (dry.match(/^\[meaning\]/gm) ?? []).length,
  uncertain: (dry.match(/^\[meaning-uncertain\]/gm) ?? []).length,
};
// Every id the report mentions, so the backends' findings can be diffed.
results["consolidate dry ids"] = [...new Set(dry.match(/#\d+/g) ?? [])].sort();
const real = ok(await timed(() => __internals.consolidate({}), lat("consolidate-real")), "");
results["consolidate real summary"] = real.split("\n")[0];

await __internals.settleEmbeddings();
await __internals.sql`DELETE FROM memories WHERE project = ${DIR}`;
if (logs.length) throw new Error(`run logged failures:\n${logs.join("\n")}`);
results.latencyMs = Object.fromEntries(Object.entries(t).map(([k, v]) => [k, { n: v.length, ...stats(v) }]));
await __internals.dispose();

console.log(JSON.stringify(results, null, 2));
if (outPath) await Bun.write(outPath, JSON.stringify(results, null, 2));
