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

6. **Iter 6** — *Saw* on the Iter 5 dashboard that decode lifecycle p95 climbed from ~ 2 s to ~ 4 s over the 5-minute window — decode is now the dominant cost per remaining vLLM call. *Hypothesized* enabling **n-gram speculative decoding** in vLLM would give a 1.5-2× decode speedup with zero quality risk: SQL output has very repetitive token sequences (`SELECT`, ` FROM`, ` WHERE`, schema-quoted identifiers, `COUNT(*)`, etc.) — the sweet spot for n-gram speculation — and the LLM verifier automatically rejects bad speculative guesses, so any quality loss is impossible by construction. No code change, just one vLLM launch flag. *Changed* `scripts/start_vllm.sh` to add `--speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4}'`. *Result* (`results/load_test_iter6_speculative.json`): **TODO after re-run**.

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
