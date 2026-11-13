// Usage report for mempg: how often agents store and recall memories, how much
// of the storing the session_stop checkpoint prompts, and how often stored
// memories come back. Reads omp session transcripts plus the memory DB
// (never writes rows; configured by the same MEMPG_* env as the plugin). SKILL.md
// covers the workflow and the judging step for --samples.
//
//   bun skills/mempg-eval/eval.ts [--sessions DIR] [--samples FILE]
import { parseArgs } from "node:util";
import { Glob } from "bun";
import mempg from "../../mempg.ts";

const { values: args } = parseArgs({
  options: {
    sessions: { type: "string", default: `${process.env.HOME}/.omp/agent/sessions` },
    samples: { type: "string" },
  },
});

type Part = { type?: string; id?: string; name?: string; arguments?: Record<string, unknown>; text?: string };
type Message = { role?: string; content?: unknown; toolCallId?: string };
type Entry = { type?: string; timestamp?: string; content?: unknown; message?: Message };
type Origin = "organic" | "checkpoint";
type Kind = "main" | "subagent" | "advisor";
type Sample = { key: string; query: unknown; result: string; after: string };

const parts = (m: Message): Part[] => (Array.isArray(m.content) ? m.content : []);
const bump = (map: Map<string, number>, key: string) => map.set(key, (map.get(key) ?? 0) + 1);
const pct = (n: number, d: number) => (d ? `${Math.round((100 * n) / d)}%` : "-");

// Monday of the UTC week, as YYYY-MM-DD.
function week(at: string | Date): string {
  const d = new Date(at);
  d.setUTCDate(d.getUTCDate() - ((d.getUTCDay() + 6) % 7));
  return d.toISOString().slice(0, 10);
}

function table(head: string[], rows: (string | number)[][]): string {
  return [head, head.map(() => "---"), ...rows].map((r) => `| ${r.join(" | ")} |`).join("\n");
}

// What the agent did after a recall: the evidence a judge needs to decide
// whether a returned memory changed anything.
function actionsAfter(msgs: { m: Message }[], from: number): string {
  const out: string[] = [];
  for (const { m } of msgs.slice(from + 1)) {
    if (m.role !== "assistant") continue;
    for (const p of parts(m)) {
      if (p.type === "text" && p.text) out.push(`SAY: ${p.text.slice(0, 600)}`);
      if (p.type === "toolCall") out.push(`CALL ${p.name}: ${JSON.stringify(p.arguments).slice(0, 400)}`);
    }
    if (out.length >= 14) break;
  }
  return out.join("\n").slice(0, 6000);
}

const sessions = new Map<Kind, number>();
const usingAny = new Map<Kind, number>();
const sessionsUsing = new Map<string, number>();
const calls = { organic: new Map<string, number>(), checkpoint: new Map<string, number>() };
const errors = new Map<string, number>();
const weekly = new Map<string, Map<string, number>>();
const stored = { organic: new Set<number>(), checkpoint: new Set<number>() };
const checkpoint = { fired: 0, productive: 0 };
const recallResults = { hit: 0, empty: 0 };
const samples: Sample[] = [];
let toolCalls = 0;

for await (const rel of new Glob("**/*.jsonl").scan(args.sessions)) {
  // Main sessions sit one directory deep; subagent and advisor transcripts nest
  // under their parent session's directory.
  const kind: Kind = rel.split("/").length <= 2 ? "main" : rel.split("/").at(-1)?.startsWith("__advisor") ? "advisor" : "subagent";
  bump(sessions, kind);
  const entries = (await Bun.file(`${args.sessions}/${rel}`).text()).split("\n").flatMap((line) => {
    try {
      return [JSON.parse(line) as Entry];
    } catch {
      return [];
    }
  });

  // Pass 1: tag each message with what prompted it. A checkpoint continuation
  // runs from the injected <mempg-checkpoint> message to the next user message.
  const msgs: { m: Message; origin: Origin }[] = [];
  let wk = "";
  let origin: Origin = "organic";
  let wrote = false;
  const closeCheckpoint = () => {
    if (origin === "checkpoint" && wrote) checkpoint.productive++;
  };
  for (const e of entries) {
    if (e.type === "session" && e.timestamp && !wk) wk = week(e.timestamp);
    if (e.type === "custom_message" && typeof e.content === "string" && e.content.includes("<mempg-checkpoint>")) {
      closeCheckpoint();
      checkpoint.fired++;
      origin = "checkpoint";
      wrote = false;
      continue;
    }
    const m = e.message;
    if (!m) continue;
    if (m.role === "user") {
      closeCheckpoint();
      origin = "organic";
    }
    if (origin === "checkpoint" && parts(m).some((p) => p.name === "memory_remember" || p.name === "memory_update")) wrote = true;
    msgs.push({ m, origin });
  }
  closeCheckpoint();

  // Pass 2: count memory tool calls and classify their results.
  const used = new Set<string>();
  const pending = new Map<string, { tool: string; origin: Origin; at: number; query: unknown }>();
  for (const [at, { m, origin }] of msgs.entries()) {
    for (const p of parts(m)) {
      if (p.type !== "toolCall") continue;
      toolCalls++;
      if (!p.name?.startsWith("memory_") || !p.id) continue;
      bump(calls[origin], p.name);
      used.add(p.name);
      if (!weekly.has(wk)) weekly.set(wk, new Map());
      bump(weekly.get(wk) as Map<string, number>, p.name);
      pending.set(p.id, { tool: p.name, origin, at, query: p.arguments?.query });
    }
    const call = m.role === "toolResult" && m.toolCallId ? pending.get(m.toolCallId) : undefined;
    if (!call) continue;
    const out = parts(m).map((p) => p.text ?? "").join(" ");
    if (out.startsWith("ERROR")) bump(errors, call.tool);
    else if (call.tool === "memory_recall") {
      if (out.includes("No memories found")) recallResults.empty++;
      else {
        recallResults.hit++;
        samples.push({ key: `${rel}:${m.toolCallId}`, query: call.query, result: out.slice(0, 5000), after: actionsAfter(msgs, call.at) });
      }
    } else if (call.tool === "memory_remember") {
      const id = /Stored memory #(\d+)/.exec(out)?.[1];
      if (id) stored[call.origin].add(Number(id));
    }
  }
  for (const tool of used) bump(sessionsUsing, tool);
  if (used.size) bump(usingAny, kind);
}

// --- DB side ---
// No ready(): its SQLite schema init would turn a wrong MEMPG_SQLITE_PATH into
// an empty DB and an all-zero report; without it the first SELECT fails.
const { sql } = mempg.__internals;
type Row = { id: number | string; project: string | null; created_at: string | Date; tags: string | string[]; memory_type: string; superseded_by: unknown };
const rows = (await sql`SELECT id, project, created_at, tags, memory_type, superseded_by FROM memories`) as Row[];
// memory_recalls keeps one row per (memory, session) and loses rows when a
// memory is deleted, so these are "returned in N sessions", not call counts.
const recallRows = (await sql`SELECT memory_id, count(*) AS n FROM memory_recalls GROUP BY memory_id`) as { memory_id: number | string; n: number | string }[];
const [cross] = (await sql`
  SELECT count(*) AS n, sum(CASE WHEN r.session_id <> coalesce(m.session_id, '') THEN 1 ELSE 0 END) AS other
  FROM memory_recalls r JOIN memories m ON m.id = r.memory_id
`) as { n: number | string; other: number | string | null }[];
await sql.close({ timeout: 0 });

const returned = new Map(recallRows.map((r) => [Number(r.memory_id), Number(r.n)]));
const live = rows.filter((r) => r.superseded_by == null);
const byId = new Map(live.map((r) => [Number(r.id), r]));
const histogram = new Map<string, number>();
for (const r of live) {
  const n = returned.get(Number(r.id)) ?? 0;
  bump(histogram, n === 0 ? "0" : n === 1 ? "1" : n <= 4 ? "2-4" : "5+");
}

// --- Report ---
const tools = [...new Set([...calls.organic.keys(), ...calls.checkpoint.keys()])].sort();
const kinds: Kind[] = ["main", "subagent", "advisor"];
const sessionTotal = (sessions.get("main") ?? 0) + (sessions.get("subagent") ?? 0);
const memoryCalls = [...calls.organic.values(), ...calls.checkpoint.values()].reduce((a, b) => a + b, 0);
const out: string[] = [];

out.push(`## Tool calls (${args.sessions})`);
out.push(table(["Session kind", "Sessions", "Using a memory tool"], kinds.map((k) => [k, sessions.get(k) ?? 0, usingAny.get(k) ?? 0])));
out.push(`${memoryCalls} memory calls = ${pct(memoryCalls, toolCalls)} of ${toolCalls} tool calls.`);
out.push(
  table(
    ["Tool", "Organic", "Checkpoint-prompted", "Sessions using", "Errors"],
    tools.map((t) => [t, calls.organic.get(t) ?? 0, calls.checkpoint.get(t) ?? 0, `${sessionsUsing.get(t) ?? 0} (${pct(sessionsUsing.get(t) ?? 0, sessionTotal)})`, errors.get(t) ?? 0]),
  ),
);
out.push(
  `Checkpoint fired ${checkpoint.fired}x; ${checkpoint.productive} (${pct(checkpoint.productive, checkpoint.fired)}) led to a write. ` +
    `Recalls returning rows: ${recallResults.hit}/${recallResults.hit + recallResults.empty}.`,
);

out.push("## Weekly calls (week of session start)");
out.push(table(["Week", ...tools], [...weekly.keys()].sort().map((w) => [w, ...tools.map((t) => weekly.get(w)?.get(t) ?? 0)])));

out.push(`## Memory DB (${process.env.MEMPG_BACKEND === "sqlite" ? "sqlite" : "postgres"})`);
out.push(
  table(
    ["Metric", "Value"],
    [
      ["Rows / live / superseded", `${rows.length} / ${live.length} / ${rows.length - live.length}`],
      ["Live project_fact / stack_fact", `${live.filter((r) => r.memory_type === "project_fact").length} / ${live.filter((r) => r.memory_type === "stack_fact").length}`],
      ["Projects", new Set(rows.map((r) => r.project)).size],
      ['"Remember that..." captures (user-requested tag)', rows.filter((r) => (typeof r.tags === "string" ? JSON.parse(r.tags) : r.tags).includes("user-requested")).length],
      ["Recall rows from another session than the write", `${Number(cross.other ?? 0)}/${Number(cross.n)} (${pct(Number(cross.other ?? 0), Number(cross.n))})`],
      ...["0", "1", "2-4", "5+"].map((b) => [`Live memories returned in ${b} sessions`, histogram.get(b) ?? 0]),
    ],
  ),
);

// Same-week comparison: older memories had more chances to be returned, so
// organic vs checkpoint is only fair within a creation week.
out.push("## Share of live memories returned by a recall, by origin and creation week");
const originWeeks = new Map<string, Record<Origin, { n: number; hit: number }>>();
for (const o of ["organic", "checkpoint"] as const) {
  for (const id of stored[o]) {
    const r = byId.get(id);
    if (!r) continue;
    const w = week(r.created_at);
    if (!originWeeks.has(w)) originWeeks.set(w, { organic: { n: 0, hit: 0 }, checkpoint: { n: 0, hit: 0 } });
    const cell = (originWeeks.get(w) as Record<Origin, { n: number; hit: number }>)[o];
    cell.n++;
    if (returned.has(id)) cell.hit++;
  }
}
out.push(
  table(
    ["Created week", "Organic", "Checkpoint-prompted"],
    [...originWeeks.keys()].sort().map((w) => {
      const c = originWeeks.get(w) as Record<Origin, { n: number; hit: number }>;
      return [w, `${pct(c.organic.hit, c.organic.n)} of ${c.organic.n}`, `${pct(c.checkpoint.hit, c.checkpoint.n)} of ${c.checkpoint.n}`];
    }),
  ),
);

if (args.samples) {
  await Bun.write(args.samples, JSON.stringify(samples));
  out.push(`Wrote ${samples.length} recall samples to ${args.samples}.`);
}
console.log(out.join("\n\n"));
