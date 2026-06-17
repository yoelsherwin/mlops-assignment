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
| Overall pass rate (final SQL) | **TODO** |
| Pass rate, stop at iter 0 (initial generate) | **TODO** |
| Pass rate, stop at iter 1 (after first revise) | **TODO** |
| Pass rate, stop at iter 2 (after second revise) | **TODO** |
| Iteration distribution (counts at terminal iter) | **TODO** |
| Errors / `agent_error` responses | **TODO** |

Commentary: **TODO — does iter-2 meaningfully exceed iter-0? If yes, the verify→revise architecture is doing real work and we quantify the lift here. If not, the loop is theater and we say so.**

Artifacts:
- `results/eval_baseline.json` — full per-question record (per-iter SQL, executed_ok, verify_ok, verify_issue, correct).
- `screenshots/grafana_eval_run.png` — serving dashboard captured *during* the baseline run.

---

## 3. SLO journey (Phase 6)

Target: **P95 end-to-end agent latency < 5 s at 10+ RPS over a 5-minute window.** Driven by `load_test/driver.py --rps 10 --duration 300` against `http://localhost:8001/answer`.

### Baseline (Phase 1 config, no tuning)

| Metric | Value | vs SLO |
|---|---|---|
| Achieved RPS | **TODO** | |
| Latency P50 | **TODO** s | |
| Latency P95 | **TODO** s | **PASS / MISS by Δ s** |
| Latency P99 | **TODO** s | |
| Timeouts / HTTP errors | **TODO** | |

### Iteration log

Format: *"saw X → hypothesized Y → changed Z → result was W."*

1. **Iter 1** — **TODO.** Example shape: *saw KV cache nearing 100 % under load → hypothesized request queueing was bottlenecking the loop → reduced `--max-num-seqs` to 32 to keep batch sizes saner under contention → P95 dropped from X s to Y s, requests-running stabilized.* Grafana before/after: `screenshots/grafana_before.png`, `screenshots/grafana_after.png`.
2. **Iter 2** — **TODO.**
3. **Iter 3 (if needed)** — **TODO.**

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

**TODO — one paragraph.** Cite the per-iteration pass rate from §2:
- If iter-0 ≈ iter-2, the verify→revise architecture isn't doing measurable work — say so and explain why we still kept it (or didn't).
- If iter-2 > iter-0, the lift in percentage points *is* the ROI. Note the cost: roughly 2× LLM calls on questions that triggered a revise, which §3 should show as a latency tax.
- Mention any false-positive / false-negative pattern visible in `per_iter[].verify_ok` vs `per_iter[].correct` from `eval_baseline.json` — those are the prompts most worth tuning next.

---

## 5. What I'd do with more time

Concrete next bets, ordered by expected return:

- **FP8 / AWQ quantization of weights.** Drops weight memory from ~57 GiB to ~28 GiB, freeing ~30 GiB of KV cache budget — direct path to higher concurrency, lower queueing, and the easiest route to a comfortable P95 margin. Cost is a quality hit that needs measurement against the eval set; we don't know how big without trying.
- **Schema pruning per question.** A 1.5–3 K-token schema dominates every prompt. Even a simple retrieval step (e.g. embedding-rank tables by question relevance, keep top-k) could cut prompt length 3-5×, which on a prefix-cached MoE serving stack drops most of the per-call cost with it. Biggest wins: TTFT and prefill GPU time.
- **Deterministic checks before the LLM verifier.** Cheap rule-based pre-checks ("did the SQL execute? did it return any rows? do returned column names overlap with the question's noun phrases?") could short-circuit a large fraction of verify calls without changing the verdict, halving the loop's LLM cost.
- **Self-consistency at generate, instead of verify-revise.** Sample k=3 SQLs at low temperature, execute each, vote on result-set majority. The text-to-SQL literature shows this beating verify-revise architectures on BIRD while being trivially parallelizable. Trades the latency profile (one slower call vs three concurrent ones) — worth a head-to-head.
- **Streaming + partial-response UX.** The agent currently returns only after the whole loop. Streaming the first generate while verify runs hides 1–2 s of perceived latency under load even when total wall-clock is unchanged.
- **Eval rigor upgrades.** Current canonicalization treats result rows as a bag, so a SQL with duplicate rows fails against a gold SQL using `DISTINCT` (a real failure we saw in Phase 4 smoke tests). Reporting both bag-equality and set-equality would surface concrete prompt edits worth making — e.g., nudging the generator to dedup when the question doesn't imply duplicates.
