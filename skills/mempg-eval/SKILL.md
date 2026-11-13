---
name: mempg-eval
description: Use when asked how often mempg memory is stored/recalled, whether the memory plugin brings value, or to evaluate mempg usage and recall usefulness from omp session logs.
---

# mempg usage evaluation

Answers two questions: how much agents use mempg, and whether recalled memories change what they do.

## 1. Counts (deterministic)

Run with the same `MEMPG_*` env the plugin uses (it reads the memory DB and never writes rows):

```bash
bun skill://mempg-eval/eval.ts --samples /tmp/mempg-recall-samples.json
```

`--sessions DIR` overrides `~/.omp/agent/sessions`. The report covers:
- memory tool calls per tool, split into **organic** and **checkpoint-prompted** (issued after the `session_stop` `<mempg-checkpoint>` continuation, until the next user message);
- sessions by kind (main / subagent / advisor) and how many used a memory tool;
- weekly call counts, checkpoint fire count and how often it led to a write;
- DB totals, "remember that…" captures (rows tagged `user-requested`, stored without a tool call), the cross-session share of recalls, and a per-creation-week share of memories ever returned, organic vs checkpoint.

## 2. Usefulness (model-judged)

"Returned by a recall" is not "used": a recall returns up to 10 rows whether or not the agent needs them. Judge the samples in `eval` (Python):

```python
import asyncio, json, random, collections
samples = json.load(open("/tmp/mempg-recall-samples.json"))
random.seed(1)
picked = random.sample(samples, min(100, len(samples)))
b = judge_batch({s["key"]: s for s in picked}, {"use": {"type": "choice",
    "instructions": "An AI coding agent ran memory_recall(query), got result (past memories), then took the actions in after. Did any returned memory materially influence what the agent did next?",
    "criteria": {
        "acted": "agent clearly used a specific returned fact: cites it, follows a stated preference/procedure, avoids a recorded pitfall, or uses a recorded path/command/value it would not obviously know",
        "confirmatory": "returned memories were relevant and consistent with the actions, but the agent would plausibly have done the same anyway",
        "ignored": "returned memories were irrelevant or unused"}}}, intent="Judging recall usefulness")
```

Wait with `while b.status()["running"]: await asyncio.sleep(3)`, then tally `collections.Counter(v["use"]["choice"] for v in b.results().values())`. Spot-check at least 3 `acted` and 2 `ignored` verdicts by reading their samples before trusting the tally. The judge costs real money (~$2.50 per 100 samples on a large model).

## 3. Report

Tables for: calls by tool and origin, weekly trend, DB metrics, same-week origin comparison, usefulness tally. Always state these limits:
- **Injection is not logged.** Memories injected into the system prompt each turn leave no trace, so "returned in 0 sessions" does not mean unused.
- `memory_recalls` keeps one row per (memory, session) and loses rows when a memory is forgotten; it counts sessions, not calls.
- Compare organic vs checkpoint only within a creation week; older memories had more chances to be returned.
- The usefulness tally is a model's judgment over a sample, checked only by your spot-checks.
- Transcripts cover only retained sessions; DB rows can predate them.
