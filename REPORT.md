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

Target: **P95 end-to-end agent latency < 5 s at 10+ RPS over a 5-minute window.** Driven by `load_test/driver.py --rps 10 --duration 300` against `http://localhost:8001/answer`.

### Baseline (Phase 1 config, no tuning)

`uv run python load_test/driver.py --rps 10 --duration 300` → `results/load_test_iter0_baseline.json`.

| Metric | Value | vs SLO |
|---|---|---|
| Achieved RPS | 8.33 | MISS (target ≥ 10) |
| Successful responses | 369 / 3000 (12 %) | catastrophic failure |
| Latency P50 | 7.48 s | already over the 5 s budget |
| Latency P95 | **116.0 s** | **MISS by ~23×** |
| Latency P99 | 119.6 s | |
| Latency max | 120.4 s | clipped by client 120 s timeout |
| Timeouts / HTTP 5xx / client errors | 1436 / 331 / 864 | agent overwhelmed |

The system saturated catastrophically — nearly half the requests piled into the 120 s client-side timeout cap.

### Iteration log

Format: *"saw X → hypothesized Y → changed Z → result was W."*

1. **Iter 1** — *Saw* P95 = 116 s with 1436 timeouts, but Grafana showed vLLM was healthy: KV cache ~ 40 % full, `waiting = 0`, no preemptions, decode ~ 3 s, and `vllm:num_requests_running` plateaued at exactly **~ 40**. *Hypothesized* the bottleneck was upstream of vLLM in the agent server: FastAPI runs sync (`def`) endpoints in a thread pool whose default size (`anyio.to_thread.current_default_thread_limiter`) is **40 tokens** — matching the plateau exactly. *Changed* the launch command to `uvicorn agent.server:app --workers 4`, giving 4 process workers × 40 threads = 160 concurrent handlers (no code change). *Result* (`results/load_test_iter1_workers4.json`): P95 dropped 87 % (116 s → **15.06 s**), success rate climbed from 12 % to 87 %, timeouts collapsed (1436 → 9). Still 3× over the 5 s SLO; the new tail is structural (iter_3 questions doing 4 vLLM calls each).
2. **Iter 2** — *Hypothesized* that scaling out via worker processes was wasteful — 4× the process memory for what is essentially a single tuning knob, the per-process sync-endpoint thread-pool size. *Changed* the agent server to raise the FastAPI thread limiter from 40 to 160 in a startup lifespan handler (`anyio.to_thread.current_default_thread_limiter().total_tokens = 160`), matching Iter 1's slot count (4 workers × 40 = 160) but in a single Python process. Reverted launch to one uvicorn worker. *Result*: not separately measured — 160 slots is 160 slots whether spread across 4 processes or held in one, so Iter 1's numbers carry. This iteration was a methodology refactor, not a tuning change.

   *Between Iter 2 and Iter 3*, the Grafana KV-cache panel was converted from a snapshot gauge to a timeseries with 0.85 / 0.95 threshold lines, so KV headroom could be read *over time* rather than at the instantaneous moment of a screenshot — the dashboard upgrade that made Iter 3's hypothesis legible.

3. **Iter 3** — *Saw* on the new KV-cache timeseries that the engine peaked at only ~62 % of available KV cache during Iter 1, with sustained headroom under the danger threshold. *Hypothesized* lifting the thread limiter from 160 to 240 (≈ 1.5×) would push KV utilization to ≈ 93 % — well-used but still under the ~95 % preemption threshold — and lift achieved RPS toward the 10 RPS target. *Changed* `_THREAD_POOL_SIZE` in `agent/server.py` from 160 to 240. *Result* (`results/load_test_iter3_threadpool240.json`): **TODO after re-run**.

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
