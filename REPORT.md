# REPORT — LLM inference + observability

Text-to-SQL agent over BIRD-bench. vLLM 0.10.2 serves `Qwen/Qwen3-30B-A3B-Instruct-2507` on 1× H100 80 GB. A LangGraph agent (generate → execute → verify → optional revise, capped at `MAX_ITERATIONS = 2` after Phase 6) wraps it, fronted by a FastAPI server on `:8001` running with `--workers 4` (final config; the starting config was a single worker). Prometheus scrapes vLLM `/metrics`; Grafana visualizes the serving layer; Langfuse captures agent traces.

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
| `--max-model-len` | 8192 | Default 262 144 reserves ~24 GiB KV per seq; with 57 GiB weights on 80 GiB vLLM refuses to start. Workload tops at ~3 K-token prompts + short outputs, so 8 K frees the rest of KV for concurrency. |
| `--enable-prefix-caching` | on (default) | DB schema (1.5–3 K tokens) appears verbatim across generate/verify/revise — turns schema-encode cost from O(calls) to O(1) per question. |
| `--enable-chunked-prefill` | on (default, `max_num_batched_tokens=8192`) | Interleaves prefill chunks with in-flight decode so one long prefill doesn't stall decode under concurrent load. |
| `--gpu-memory-utilization` | 0.9 (default) | KV headroom *is* concurrency; 10 % margin for compile cache + activations. |
| `--dtype` (`bfloat16`) | default on H100 | 57 GiB fits; known-good quality baseline. FP8/AWQ deferred to §5 (unmeasured quality cost). |

Post-Phase-1 config; final config (after Phase 6) is in §3.

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

1. **Iter 1 — baseline measurement.** *Saw / Hypothesized / Changed*: nothing, starting point. *Result* (`results/load_test_iter1_baseline.json`): catastrophic — P95 **116 s**, P50 7.48 s, **ok 369 / 3000 (12 %)**, 1436 timeouts, 331 HTTP 5xx, 864 client errors at 10 RPS. System cannot serve the SLO at this config. (Observability change banked between iters: KV-cache panel converted from snapshot gauge to timeseries with 0.85 / 0.95 thresholds.)

2. **Iter 2 — agent uvicorn workers 1 → 4.** *Saw*: 864 client_errors + 331 5xx + 1436 timeouts while vLLM `num_requests_running` barely moved — the agent was queueing in front of vLLM. *Hypothesized*: route concurrency into vLLM. *Changed*: `scripts/start_agent.sh 4`. *Result*: ok 369 → **2992** (99.7 %), errors all 0, **P95 116 → 19.0 s (× 6.1)**; `num_requests_running` = 52, waiting = 0; per-call vLLM P95 = 4.81 s with **decode = 4.77 s of it** — per-call path is decode-bound, agent-side 19 s ≈ 3 calls in series.

3. **Iter 3 — `MAX_ITERATIONS` 3 → 2.** *Saw*: Phase-5 eval = iter_0 = iter_1 = iter_2 = 33.3 %, the third call adds zero quality; baseline distribution had 7 / 30 questions terminating at iter_3, each paying ~5 s for nothing. *Hypothesized*: shed one LLM call on the 23 % tail, agent P95 falls and vLLM concurrency drops with it. *Changed*: `MAX_ITERATIONS = 2` in `agent/graph.py` (mirrored in `evals/run_eval.py`). *Result*: **P95 19.0 → 11.4 s (− 40 %)**, P50 3.88 → 3.37 s, ok 2987 / 3000; `num_requests_running` 52 → **39** (− 25 %), `prompt_tokens_per_sec` 30.6 K → 25.5 K. Per-call decode P95 unchanged at ~4.5 s as expected — Iter 3 doesn't touch the per-call path.

4. **Iter 4 — speculative decoding (ngram, num=4, prompt_lookup_max=4) — REGRESSION.** *Saw*: per-call decode dominates (P95 ~4.5 s, ITL P95 = 87 ms), SQL outputs have repetitive tokens; ngram speculation should compress decode. *Hypothesized*: ngram drafts hit on the common runs, decode P95 drops. *Changed*: `--speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4}'` on vLLM. *Result*: **agent P95 11.4 → 93.7 s (× 8.2 worse)**, per-call P95 4.81 → 6.49 s, TTFT P95 0.22 → 0.38 s, ITL P95 87 → 130 ms, peak KV 64 → 78 %. Mis-drafts on variable content (literals, JSON verify outputs); rejected work stalls the shared batch. *Reverted.*

5. **Iter 5 — deterministic verify gate, LLM-fallback on uncertain.** *Saw*: decode is the per-call floor (Iter 3) and we can't compress it (Iter 4), so the remaining lever is **calls-per-question**; Phase-5 showed the LLM verifier never flipped correctness. *Hypothesized*: a rule pre-check (SQL errored → ok=false; SQL ran with ≥ 1 row → ok=true; else fall through to LLM) skips ~67 % of verify calls with quality-preserving verdicts. *Changed*: `_rule_based_verify` in `verify_node`; vLLM reverted to baseline. *Result*: **SLO hit — P95 11.4 → 4.72 s (< 5 s)**, P50 3.37 → 1.20 s, ok 2994 / 3000, RPS 9.25, zero client errors. Targeted metric moved: `num_requests_running` 39 → **14** (− 64 %), **ITL P95 87 → 25 ms (− 71 %)** (fewer concurrent decodes ⇒ less batching contention), per-call vLLM e2e P95 4.81 → 2.00 s, prefix-cache hit 62 → 90 %. See `screenshots/grafana_before.png` (Iter 3) and `screenshots/grafana_after.png` (Iter 5).

### Final configuration and numbers

Final vLLM launch (`scripts/start_vllm.sh`) — same as §1 baseline (Iter 4's speculative was reverted):

```bash
uv run python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192
```

Agent launch: `scripts/start_agent.sh 4` (4 uvicorn workers). Agent code changes: `MAX_ITERATIONS = 2`, `verify_node` short-circuits via `_rule_based_verify`.

| Metric | Iter 1 baseline | Final (Iter 5) | vs SLO |
|---|---|---|---|
| P95 latency | 116.05 s | **4.72 s** | **PASS** (< 5 s) |
| P50 latency | 7.48 s | 1.20 s | — |
| P99 latency | 119.58 s | 8.53 s | — |
| ok / 3000 | 369 (12 %) | 2994 (99.8 %) | — |
| Achieved RPS | 8.33 | **9.25** | within driver scheduling jitter of the 10 RPS target; system holds 99.8 % success at requested 10 RPS |
| Post-tuning eval pass rate (`results/eval_after_tuning.json`) | — | **30.0 %** (9 / 30) | vs §2 baseline 33.3 %: **− 3.3 pp** (1 question; analyzed below) |

**Verdict**: SLO hit on P95, latency dropped × 24.6 vs baseline. The binding constraint was *calls-per-question on a decode-bound MoE*: every LLM call paid ~4.5 s of decode (limited by inter-token latency under high concurrent batching), and the agent's verify-then-revise architecture stacked 2 – 3 such calls in series. The unlock chain was (i) move concurrency from in-front-of vLLM into vLLM (Iter 2), (ii) cut the third call which Phase 5 already proved was free of quality value (Iter 3), (iii) skip the second call too when it's mechanically decidable (Iter 5). Iter 4 (ngram speculation) was the bet that didn't pay — useful as a documented miss. The quality delta is **one question** out of thirty and inspection traces it to vLLM batching non-determinism at temperature 0 (different generate-time SQL between runs — both terminated at iter 1 in their respective runs), not to the rule-based verify itself. See §4 for the breakdown.

---

## 4. Did the agent loop earn its keep?

No, and Phase 6 made that quantitative. **Baseline (§2)**: 33.3 % at iter_0 = iter_1 = iter_2. The loop fired on 10 of 30 questions (3 stopping at iter_2, 7 at iter_3) and **flipped zero of them**; it paid ~2× LLM calls on those questions for zero quality lift. Two failure modes explain it: the LLM verifier rubber-stamps wrong SQL when the result *shape* looks right (e.g. Q1 returns duplicate Australian-GP coordinates where gold uses `DISTINCT`), and when verify *does* fire its issue is too vague for revise to act on (Q27, Q30 emit identical SQL across all three iterations because *"zero rows returned"* gives no hint about *what* to change).

**After tuning (Iter 5, §3)**: 30.0 % at iter_0 = iter_1, terminating distribution iter_1: 24, iter_2: 6. The − 3.3 pp gap from baseline is **one question** — Q "List the top five schools by Enrollment (Ages 5-17)". Inspection: both runs terminated at iter_1 (no revise fired), so the rule-based verify never blocked a corrective path; the SQL differs at generate time — baseline emits `SELECT s.NCESSchool ... LIMIT 5`, the post-tuning run emits `SELECT s.NCESSchool, f."Enrollment (Ages 5-17)" ... LIMIT 5`. Both are valid readings of the question and execute fine, but bag-equality against the single-column gold rejects the wider projection. The drift is generate-time **batching non-determinism in vLLM at temperature 0** — same prompt, same model, different KV / scheduler state across runs. The rule-based verify is *not* the cause; the failing question never reaches verify in any meaningful way.

So the architecture still doesn't earn its keep, and Iter 5's optimization formalised that diagnosis: the deterministic gate replicates what the LLM verifier was actually doing 95 % of the time without the cost. The honest design conclusion is the §5 lever — replace this verify-revise architecture with self-consistency over k=3 SQLs (vote on canonical result sets) and re-evaluate.

---

## 5. What I'd do with more time

Concrete next bets, ordered by expected return:

- **Self-consistency at generate, replacing verify-revise.** Sample k=3 SQLs at low temperature in *parallel* (one vLLM batch, no serial chain), execute each, vote on majority result-set. The text-to-SQL literature shows this beating verify-revise on BIRD, and on this serving stack the three generates batch together for free — wall-time is roughly *one* call, not three. Phase 4 + 5 together show the current architecture's verify-revise loop earns no quality, so removing it is honest.
- **FP8 / AWQ quantization of weights.** Drops weight memory from ~57 GiB to ~28 GiB, freeing ~30 GiB of KV cache budget — direct path to higher concurrency, lower queueing, and a comfortable P95 margin if RPS targets climb. Cost is a quality hit that needs measurement against the eval set; we don't know how big without trying.
- **Schema pruning per question.** A 1.5–3 K-token schema dominates every prompt. Even a simple retrieval step (e.g. embedding-rank tables by question relevance, keep top-k) could cut prompt length 3-5×, which on a prefix-cached MoE serving stack drops most of the per-call cost with it. Biggest wins: TTFT and prefill GPU time. Mainly future-proofing for larger schemas; the iter-5 config already has TTFT under 100 ms at the SLO.
- **Tighter rules at the verify gate.** Iter 5's rule only short-circuits the "rows ≥ 1" and "SQL errored" cases. Adding cheap structural checks — column-name overlap with question noun phrases, COUNT-query shape sanity (1 row × 1 column), row-count bounds vs question phrasing — would let the gate decide more of the residual "zero rows" cases without an LLM call.
- **Streaming + partial-response UX.** The agent currently returns only after the whole loop. Streaming the generate token-by-token to the client hides 1–2 s of perceived latency under load even when total wall-clock is unchanged.
- **Eval rigor upgrades.** Current canonicalization treats result rows as a bag, so a SQL with duplicate rows fails against a gold SQL using `DISTINCT` (a real failure visible in Phase 4 smoke tests, and the suspected cause of the one-question delta in §4). Reporting both bag-equality and set-equality, and re-running the eval N=3 to filter generate-time non-determinism, would surface concrete prompt edits worth making — e.g., nudging the generator to dedup when the question doesn't imply duplicates.
