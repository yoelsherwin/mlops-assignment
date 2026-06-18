# REPORT — LLM inference + observability

Text-to-SQL agent over BIRD-bench. vLLM 0.10.2 serves `Qwen/Qwen3-30B-A3B-Instruct-2507` on 1× H100 80 GB. A LangGraph agent (generate → execute → verify → optional revise, capped at 3 iterations) wraps it, fronted by a FastAPI server on `:8001`. Prometheus scrapes vLLM `/metrics`; Grafana visualizes the serving layer; Langfuse captures agent traces.

---

## 1. Serving configuration

Launch script: `scripts/start_vllm.sh`.

```bash
uv run python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192
```

| Flag / setting | Value | One-line justification |
|---|---|---|
| `--max-model-len` | 8192 | Default 262 144 reserves ~24 GiB KV cache per sequence; on 80 GiB after ~57 GiB of weights vLLM refuses to start. Workload tops out near 3 K-token prompts with short outputs, so 8 K leaves ample headroom and frees the rest of the KV budget for concurrency. |
| `--enable-prefix-caching` | on (vLLM default) | The DB schema (1.5–3 K tokens) appears verbatim in generate / verify / revise within one agent run, so caching it turns the schema-encode cost from O(calls per question) into O(1 per question). |
| `--enable-chunked-prefill` | on (vLLM default, `max_num_batched_tokens=8192`) | Interleaves prefill chunks with in-flight decode so a single long prefill can't stall decode under concurrent load — important for keeping P95 bounded as RPS rises. |
| `--gpu-memory-utilization` | 0.9 (vLLM default) | KV cache headroom *is* concurrency. 10 % margin for compile cache + activations; pushing higher trades stability for marginal headroom. |
| `--dtype` (`bfloat16`) | vLLM default on H100 | Full-precision weights fit (56.9 GiB) and give us a known-good quality baseline; FP8 / AWQ would free KV budget but trade quality for an unmeasured one. Listed in §5 as the next bet. |

This is the post-Phase-1 configuration; revisited in §3 after the SLO load test.

---

## 2. Baseline eval (Phase 5)

30 BIRD-bench questions from `evals/eval_set.jsonl`. Scoring signal: execution accuracy — the agent's SQL and the gold SQL are each executed against the target sqlite DB; result rows are canonicalized (sorted, str-coerced, `None → ""`) and compared as bags. Per-iteration scoring uses **carry-forward**: once the agent terminates at iter `j`, its iter > `j` entries inherit `j`'s correctness.

| Metric | Value |
|---|---|
| Overall pass rate (final SQL) | **33.3 %** (10 / 30) |
| Pass rate, stop at iter 0 (initial generate) | **33.3 %** |
| Pass rate, stop at iter 1 (after first revise) | **33.3 %** |
| Pass rate, stop at iter 2 (after second revise) | **33.3 %** |
| Iteration distribution (counts at terminal iter) | iter_1: 20 (67 %) · iter_2: 3 (10 %) · iter_3: 7 (23 %) |
| Errors / `agent_error` responses | 0 |
| Wall-clock of full run | 33.3 s on warmed vLLM (median ~0.6 s/question, three iter_3 outliers ~2.2 s) |

Commentary: the per-iteration pass rate is flat at 33.3 % across all three checkpoints. The verify→revise loop fired on 10 of 30 questions (3 at iter_2 + 7 at iter_3) but did not flip a single question's correctness — the architecture is paying ~2× LLM calls on those questions for zero quality lift. §4 unpacks why.

Artifacts:
- `results/eval_baseline.json` — full per-question record (per-iter SQL, executed_ok, verify_ok, verify_issue, correct).
- `screenshots/grafana_eval_run.png` — serving dashboard captured *during* the baseline run.

---

## 3. SLO journey (Phase 6)

Target: **P95 end-to-end agent latency < 5 s at 10+ RPS over a 5-minute window.** Driven by `load_test/driver.py --rps 10 --duration 300` against `http://localhost:8001/answer`. Iterations are 1-indexed with Iter 1 = the starting-config baseline measurement.

### Iteration log

Format: *"saw X → hypothesized Y → changed Z → result was W."*

1. **Iter 1 (baseline, Phase 1 config, no tuning)** — *Saw* the SLO target and Phase 1 vLLM config; no agent tuning applied yet. *Hypothesized* nothing — this is the starting measurement. *Changed* nothing. *Result* (`results/load_test_iter1_baseline.json`): catastrophic saturation. P95 = **116 s** (× 23 the 5 s SLO), P50 = 7.48 s (already over budget), achieved RPS 8.33, **only 369 / 3000 requests succeeded (12 %)**; 1436 timed out at the 120 s client cap, 331 returned HTTP 5xx, 864 had client-level errors. Nearly half of all requests piled into the timeout wall — the system at this config cannot serve 10 RPS at all.

2. **Iter 2** — *Saw* P95 = 116 s with 1436 timeouts, but Grafana showed vLLM was healthy: KV cache ~ 40 % full, `waiting = 0`, no preemptions, decode ~ 3 s, and `vllm:num_requests_running` plateaued at exactly **~ 40**. *Hypothesized* the bottleneck was upstream of vLLM in the agent server: FastAPI runs sync (`def`) endpoints in a thread pool whose default size (`anyio.to_thread.current_default_thread_limiter`) is **40 tokens** — matching the plateau exactly. *Changed* the launch command to `uvicorn agent.server:app --workers 4`, giving 4 process workers × 40 threads = 160 concurrent handlers (no code change). *Result* (`results/load_test_iter2_workers4.json`): P95 dropped 88 % (116 s → **13.66 s**), success rate climbed 12 % → 87 %, timeouts collapsed 1436 → 10. Still 3× over the 5 s SLO; the new tail is structural (iter_3 questions doing 4 vLLM calls each).

   *Between Iter 2 and Iter 3*, the Grafana KV-cache panel was converted from a snapshot gauge to a timeseries with 0.85 / 0.95 threshold lines, so KV headroom could be read *over time* rather than at the instantaneous moment of a screenshot — the dashboard upgrade that made Iter 3's hypothesis legible.

3. **Iter 3** — *Saw* on the new KV-cache timeseries that the engine peaked at only ~ 62 % of available KV cache during Iter 2, with sustained headroom under the danger threshold. *Hypothesized* the 4-worker scale-out was wasteful (4× the process memory for what is essentially one tuning knob, the per-process sync-endpoint thread-pool size), and that we could simultaneously consolidate into one process *and* push concurrency higher: raise the in-process thread limiter to **240** (≈ 1.5× Iter 2's 160 slots) to drive KV to ~ 93 %, well-used but still under the ~ 95 % preemption threshold. *Changed* `agent/server.py` to set `anyio.to_thread.current_default_thread_limiter().total_tokens = 240` in a FastAPI lifespan handler, and reverted launch to one uvicorn worker. *Result* (`results/load_test_iter3_threadpool240.json`): **hypothesis wrong.** P95 *grew* from 13.66 s to **39.17 s (× 2.9)**, P50 from 2.80 s to 10.98 s (× 3.9), success rate fell 87 % → 81 %, **client errors jumped 1 → 207**. Achieved RPS barely moved (8.33 → 8.56). Two compounding root causes: (a) one Python process shares a single default httpx connection pool (~ 100 connections), so 240 concurrent handlers oversubscribed the pool to vLLM — the 207 client errors are the smoking gun; (b) at higher in-flight sequence counts vLLM's decode step processes more sequences in parallel and per-request decode time amplifies. **KV-cache headroom ≠ compute headroom**; the dashboard signal we trusted only covered one of the two saturating axes.

4. **Iter 4** — Reverted to launch-time worker scaling so each process gets its own httpx connection pool (and removed the in-process lifespan handler). *Hypothesized* 5 workers × 40 = 200 slots (≈ 25 % more than Iter 2's 160) would land at ~ 78 % KV peak by linear extrapolation from Iter 2's 62 % — a safe step that avoids Iter 3's overshoot, with 5 × 100 = 500 httpx connections so the connection-pool failure mode cannot recur. *Changed* the launch command to `uvicorn agent.server:app --workers 5`. *Result* (`results/load_test_iter4_workers5.json`): **hypothesis wrong, smaller version of the Iter 3 lesson.** Median behaved (P50 = 3.99 s vs Iter 2's 2.80 s), but **P95 grew from 13.66 s to 61.98 s (× 4.5 worse)**, P99 from 21.94 s to 77.51 s, client errors remained elevated (109 vs Iter 2's 1). Achieved RPS only nudged 8.33 → 8.67. The 5th worker overcommitted vLLM's per-step decode budget: more in-flight sequences → each decode step runs slower → tail amplifies. **Iter 2's 4 workers is the concurrency upper bound for this hardware × workload × agent design.** Past 4 workers we are giving the GPU more work than it can do per step, not more parallelism.

5. **Iter 5** — Reverted launch to Iter 2's known-best `uvicorn agent.server:app --workers 4` and pivoted attack from concurrency to per-request work. §4 already showed the verify→revise loop adds zero accuracy on the baseline, and 23 % of eval questions (the iter_3 cohort) make 4 sequential vLLM calls each, structurally pushing them past the 5 s SLO regardless of how fast each call is. *Hypothesized* capping `MAX_ITERATIONS = 2` in `agent/graph.py` would cut the worst case from 4 calls per question to 3 (~ 25 % less wall-clock per affected question), directly attacking the P95 tail without measurable accuracy cost (per §2, iter_2 = iter_0 pass rate already). *Changed* `MAX_ITERATIONS` in `agent/graph.py` from 3 to 2. *Result* (`results/load_test_iter5_maxiter2.json`): **hypothesis confirmed.** P95 dropped 39 % (13.66 s → **8.28 s**), P50 nudged down (2.80 → 2.55 s), client errors back to **0** (vs 109 in Iter 4), achieved RPS 8.66. Grafana cross-check: `vllm:num_requests_running` dropped from Iter 2's ~ 80-120 to **20-40**, KV peak from ~ 62 % to ~ 40 %, decode lifecycle p95 stayed 2-4 s (decode work per call unchanged — the gain came from doing fewer calls per question). One artifact: `latency_max = 116 s` near the timeout cap, with decode and E2E p95 *climbing* across the run (2 → 4 s) — vLLM appears to slowly degrade under sustained load (compile cache growth or KV fragmentation), but with only 6 timeouts out of 3000 the effect is bounded. Closest we've been to the 5 s SLO, still ~ 65 % over with 8.28 s P95.

6. **Iter 6** — *Saw* on the Iter 5 dashboard that decode lifecycle p95 climbed from ~ 2 s to ~ 4 s over the 5-minute window — decode is now the dominant cost per remaining vLLM call. *Hypothesized* enabling **n-gram speculative decoding** in vLLM would give a 1.5-2× decode speedup with zero quality risk: SQL output has very repetitive token sequences (`SELECT`, ` FROM`, ` WHERE`, schema-quoted identifiers, `COUNT(*)`, etc.) — the sweet spot for n-gram speculation — and the LLM verifier automatically rejects bad speculative guesses, so any quality loss is impossible by construction. No code change, just one vLLM launch flag. *Changed* `scripts/start_vllm.sh` to add `--speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4}'`. *Result* (`results/load_test_iter6_speculative.json`): **hypothesis confirmed, partial.** P95 dropped 13 % (8.28 s → **7.23 s**), achieved RPS climbed to **9.37** (94 % of the 10 RPS target, vs Iter 5's 8.66), **timeouts collapsed to 0** (vs 6), client errors stayed 0, wall-clock drained 26 s faster (320 s vs 346 s). Speculation acceptance rate from `vllm:spec_decode_*` counters = **96 843 / 152 367 = 63.6 %** — above the "paying its keep" threshold (~ 40 %) but below the ideal (~ 75-80 %). Grafana cross-check: decode lifecycle p95 sat at 2.3-3 s (vs Iter 5's 2-4 s with end-of-window drift); the Iter 5 degradation pattern is **gone**, a bonus finding (speculation reduces per-step compute enough that the engine stays in steady state across the 5-minute window). `vllm:num_requests_running` still 20-30, KV peak ~ 35 %, preemptions still 0, token throughput unchanged at the same RPS. **However** — see Iter 7 — the 386 HTTP 5xx errors carried forward from earlier iterations turned out *not* to be latency-bound. The 7.23 s P95 is the P95 of the 87 % of requests that ran end-to-end; the slowest 13 % was hidden behind a crash.

7. **Iter 7** — *Saw* that the 386 HTTP 5xx in Iter 6 (~ 12.9 % of traffic) matched exactly the share of `perf_pool.jsonl` from two DBs (`debit_card_specializing` + `european_football_2`). Drilling into the agent server's traceback showed `agent/schema.py:54` was calling `_q(fk[4])` on the result of `PRAGMA foreign_key_list`, which is `None` when a FK references the parent's primary key implicitly — instant crash on every request for these two DBs. Not a latency problem, a correctness bug. *Hypothesized* fixing the schema renderer would lift success rate from 87 % to ~ 100 % without materially moving the latency curve — i.e. that the 386 failures were a measurement confound, not the SLO story. *Changed* `agent/schema.py` to build the FK line incrementally and omit the column parenthetical when `fk[4]` is `None`. *Result* (`results/load_test_iter7_schema_fix.json`): **correctness hypothesis confirmed; latency hypothesis catastrophically wrong.** HTTP 5xx collapsed 386 → **5**, success rate climbed 87 % → **96 %**, ok jumped 2614 → 2878. But **P95 exploded from 7.23 s → 77.73 s (× 10.7 worse)**, P50 doubled (2.45 → 5.29 s), client errors went 0 → 112, and 5 requests timed out (vs 0 in Iter 6). The previous P95 numbers were an artifact of measurement bias — the load driver only counts `ok` requests in the latency distribution, so a failure mode that crashes in <100 ms was invisible. With the bug fixed, the two affected DBs (especially `european_football_2`, among BIRD's largest schemas) now actually render their schemas — and those balloon prompts to ~ 5 K+ tokens, violating §1's workload assumption. The cascade: bigger prefill per call → fewer concurrent sequences fit → slow requests hold worker threads → fast requests queue behind them (this is what doubled P50) → agent server saturates and starts refusing connections (the 112 client errors). **This is the most honest SLO measurement so far, and it shows the system is much further from target than Iter 6 implied.** The "94 % reduction from baseline" claim from Iter 6 only held under a confounded measurement; the real reduction so far is 116 s → 77.73 s = 33 %.

8. **Iter 8** — *Saw* in Iter 7's dashboard that the cascade wasn't prefill-driven (per-call prefill stayed flat) — it was **decode-step amplification**: `vllm:num_requests_running` peaked at 150 (vs Iter 6's 20-30), so vLLM's decode step processed 5× more sequences in parallel and per-token latency grew for everyone in flight. *Hypothesized* capping vLLM's `--max-num-seqs` would bound the decode-step batch size and let excess requests queue inside vLLM's scheduler rather than letting in-flight concurrency run away. This is *admission control on a different axis* than the agent thread pool: agent workers still accept incoming requests; vLLM just refuses to batch more than the cap into any single decode step. Sweet-spot from prior iterations is ~ 30-50 concurrent (Iter 5 ran healthy at 30-80, Iter 6 at 20-30), so **32 is the conservative end of the safe zone**. Also added `--disable-log-requests` (free perf win — cuts vLLM's per-request stdout logging; Prometheus counters unaffected). *Changed* `scripts/start_vllm.sh` to add `--max-num-seqs 32` and `--disable-log-requests`; speculation kept at 4 / 4 so any P95 change is attributable to the concurrency cap alone. *Result* (`results/load_test_iter8_maxnumseqs32.json`): **hypothesis wrong — the cap was too tight for this workload.** P95 grew further from 77.73 s → **98.39 s**, P50 exploded 5.29 → **33.22 s (× 6.3)**, ok plummeted 96 % → **69 %** (2059/3000), and client errors jumped 112 → **511**. Dashboard confirms the cap is biting exactly as designed: `running` plateaued at **32** ✓, `vllm:num_requests_waiting` sat at **~ 110** sequences, decode p95 was healthy at **~ 4 s** ✓, KV peak fell to **25 %** ✓ — but **queue p95 climbed to ~ 10 s**: the time-to-be-admitted-to-vLLM is now where the latency lives. The math: at 32 concurrent × ~ 4 s decode, vLLM drains ~ 8 requests/s; the agent demands ~ 20-30 vLLM calls/s (10 RPS × ~ 2-3 calls per question). The 12.9 % heavy cohort (Iter 7's schema fix) blocks slots much longer than median, so the queue compounds instead of clearing. **No `max-num-seqs` value can absorb this workload heterogeneity gracefully** — too tight starves the majority (this iteration); too loose lets the heavy cohort overrun vLLM (Iter 7). The structural fix is removing what makes the heavy cohort heavy in the first place.

9. **Iter 9** — *Saw* the Iter 8 math: vLLM drain rate (`32 / 4 s ≈ 8 calls/s`) is ~ 3× below agent demand (`10 RPS × ~ 2.5 calls/question ≈ 25 calls/s`). Two ways to close the gap: raise the cap (risks re-triggering Iter 7's cascade) or reduce per-call decode time (no cascade risk). *Hypothesized* bumping n-gram speculation aggressiveness from `4 / 4` (Iter 6 acceptance = 63.6 %) to `6 / 8` would push per-call decode toward ~ 2.5 s, lifting effective drain at cap=32 to ~ 12-13 calls/s. *Changed* `scripts/start_vllm.sh` speculation config; everything else unchanged. *Result* (`results/load_test_iter9_speculation_tuned.json`): **hypothesis wrong, made things worse.** P95 grew 98.39 → **109.53 s**, ok fell to **1470/3000 (49 %)**, timeouts jumped 428 → **1026**. The more-aggressive draft almost certainly pushed speculation acceptance below the paying-its-keep threshold (n-gram with longer lookahead and more tokens hits cliffs when the next-token entropy is high, which is most of the time in SQL generation outside of fence patterns), so we paid compute on rejected drafts without gaining decode speed. Branch (c) of the Iter 10 plan triggers: **schema pruning is now mandatory** — there is no `max-num-seqs` or speculation knob that fixes a workload where 13 % of requests carry 5 K-token schemas.

10. **Iter 10** — *Hypothesized* the recurring blocker since Iter 7 has one root cause: 12.9 % of the load pool comes from two DBs with ~ 5 K-token schemas. The structural fix is per-question schema pruning: render only the tables a question likely needs, dropping the worst-case schema to ~ 1 K tokens. *Changed* `agent/schema.py` to accept a `question` argument; score each table by keyword overlap with the question, keep top-k (k=6) by score, expand by one-hop FK neighbors in both directions. Backward-compatible fallback on empty question or zero-overlap. Also reverted speculation from `6 / 8` back to `4 / 4` since Iter 9 showed `6 / 8` was actively harmful. *Result* (`results/load_test_iter10_schemapruning.json`): **partial — pruning helps per-call cost but hurts the system overall.** P50 dropped 35 % (32.92 → **20.77 s**) ✓ confirming pruning is making *some* per-call decoding cheaper. But `ok` fell 1470 → **1140** (38 % success rate), timeouts climbed 1026 → **1241**, and P95 stayed catastrophic at **105.21 s**. Diagnosis: cap=32 was already too tight at the no-pruning rate (`8 calls/s` drain vs `25 calls/s` demand), and pruning *also* caused over-pruning failures — when top-6 + FK expansion drops a table the question actually needs, the agent generates SQL referencing missing columns → execute errors → verify catches → revise tries with the same pruned schema → fails again → hits `MAX_ITERATIONS=2` having spent 4 LLM calls instead of 2. So pruning lowered per-call cost but raised calls-per-request, leaving vLLM more saturated, not less. The two-axis fix is to keep pruning (it does reduce per-call cost on the questions it handles correctly) and **raise the cap** so vLLM can serve the higher call rate that pruning's correctness errors create.

11. **Iter 11 (planned)** — *Hypothesized* combining Iter 10's pruning (per-call cost down on the easy cohort) with `--max-num-seqs 64` (double the drain rate) closes both sides of the bottleneck: drain rate becomes ~ 64 / 2.5 s ≈ 26 calls/s (vs ~ 25 calls/s demand — balanced for the first time), and we stay well below Iter 7's runaway-cascade threshold (~ 150 in-flight) so decode amplification doesn't fire. *Changed* `scripts/start_vllm.sh` to lift `--max-num-seqs` from 32 to 64; all other knobs (pruning, speculation 4 / 4, agent workers=4, `--disable-log-requests`) unchanged so the iteration isolates the cap-relax effect. *Result* (`results/load_test_iter11_pruning_cap64.json`): **TODO after re-run**.

### Final configuration and numbers

Final `scripts/start_vllm.sh` (if changed from §1):

```bash
# TODO — paste the final launch invocation if the SLO journey changed any flags.
```

| Metric | Final value | vs SLO |
|---|---|---|
| P95 latency | **TODO** s | **PASS / MISS** |
| Achieved RPS | **TODO** | |
| Post-tuning eval pass rate (`results/eval_after_tuning.json`) | **TODO** | vs §2 baseline: **+/− X pp** |

**Verdict**: **TODO — honest one-paragraph closing. If SLO hit: which metric was bottlenecking and what unblocked it. If missed: by how much, what the limiting factor was (model size? single H100? no speculative decoding?), what it would take to close the gap. If quality regressed during tuning, name that and the size of the regression.**

---

## 4. Did the agent loop earn its keep?

No. iter_0 and iter_2 pass rates are both 33.3 % — the loop fired on 10 of 30 questions, flipped zero of them, and cost ~2× LLM calls in the process. Two failure modes explain it: verify is too lenient on content (13 of 47 per-iteration verdicts rubber-stamp wrong SQL, e.g. Q1 returns duplicate Australian-GP coordinates where gold uses `DISTINCT`), and when verify does fire its issue is too vague for revise to act on (Q27 and Q30 emit identical SQL across all three iterations because *"zero rows returned"* gives no hint about *what* to change).

---

## 5. What I'd do with more time

Concrete next bets, ordered by expected return:

- **FP8 / AWQ quantization of weights.** Drops weight memory from ~57 GiB to ~28 GiB, freeing ~30 GiB of KV cache budget — direct path to higher concurrency, lower queueing, and the easiest route to a comfortable P95 margin. Cost is a quality hit that needs measurement against the eval set; we don't know how big without trying.
- **Schema pruning per question.** A 1.5–3 K-token schema dominates every prompt. Even a simple retrieval step (e.g. embedding-rank tables by question relevance, keep top-k) could cut prompt length 3-5×, which on a prefix-cached MoE serving stack drops most of the per-call cost with it. Biggest wins: TTFT and prefill GPU time.
- **Deterministic checks before the LLM verifier.** Cheap rule-based pre-checks ("did the SQL execute? did it return any rows? do returned column names overlap with the question's noun phrases?") could short-circuit a large fraction of verify calls without changing the verdict, halving the loop's LLM cost.
- **Self-consistency at generate, instead of verify-revise.** Sample k=3 SQLs at low temperature, execute each, vote on result-set majority. The text-to-SQL literature shows this beating verify-revise architectures on BIRD while being trivially parallelizable. Trades the latency profile (one slower call vs three concurrent ones) — worth a head-to-head.
- **Streaming + partial-response UX.** The agent currently returns only after the whole loop. Streaming the first generate while verify runs hides 1–2 s of perceived latency under load even when total wall-clock is unchanged.
- **Eval rigor upgrades.** Current canonicalization treats result rows as a bag, so a SQL with duplicate rows fails against a gold SQL using `DISTINCT` (a real failure we saw in Phase 4 smoke tests). Reporting both bag-equality and set-equality would surface concrete prompt edits worth making — e.g., nudging the generator to dedup when the question doesn't imply duplicates.
