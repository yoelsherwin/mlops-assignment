"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"

# Must match agent.graph.MAX_ITERATIONS. Inlined to avoid pulling the
# langchain/langgraph dependencies into the eval runner.
MAX_ITERATIONS = 3


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------

def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question. Return a dict capturing per-iteration correctness.

    For each iteration the agent ran (generate_sql + any revises), score
    that iteration's SQL independently against the gold rows. This lets
    summarize() compute "what would pass rate have been if we stopped
    after iter k?" - the central agent-value question of Phase 5.
    """
    gold_sql = question["gold_sql"]
    db_id = question["db_id"]
    q_text = question["question"]

    # Gold rows once - matches() canonicalizes both sides downstream.
    gold_ok, gold_rows, gold_err = run_sql(db_id, gold_sql)

    t0 = time.monotonic()
    try:
        resp = httpx.post(
            agent_url,
            json={"question": q_text, "db": db_id},
            timeout=180.0,
        )
        resp.raise_for_status()
        agent_data = resp.json()
    except Exception as e:  # noqa: BLE001
        return {
            "question": q_text,
            "db_id": db_id,
            "gold_sql": gold_sql,
            "gold_executed_ok": gold_ok,
            "gold_error": gold_err,
            "iterations": 0,
            "final_sql": "",
            "final_correct": False,
            "per_iter": [],
            "elapsed_seconds": time.monotonic() - t0,
            "agent_error": f"{type(e).__name__}: {e}",
        }
    elapsed = time.monotonic() - t0

    # Pair each generate_sql/revise entry with the verify entry that
    # follows it. Each pair = one iteration; we score the SQL of that
    # iteration against gold and also record what verify said for it
    # (useful for diagnosing false positives/negatives).
    history = agent_data.get("history", [])
    per_iter: list[dict] = []
    pending: dict | None = None
    for h in history:
        node = h.get("node")
        if node in ("generate_sql", "revise"):
            pending = h
        elif node == "verify" and pending is not None:
            sql = pending.get("sql", "")
            pred_ok, pred_rows, pred_err = run_sql(db_id, sql)
            per_iter.append({
                "sql": sql,
                "executed_ok": pred_ok,
                "execution_error": pred_err,
                "verify_ok": bool(h.get("ok")),
                "verify_issue": h.get("issue", ""),
                "correct": bool(gold_ok and pred_ok and matches(gold_rows, pred_rows)),
            })
            pending = None
    # Defensive: if the last SQL never got a verify entry, still score it.
    if pending is not None:
        sql = pending.get("sql", "")
        pred_ok, pred_rows, pred_err = run_sql(db_id, sql)
        per_iter.append({
            "sql": sql,
            "executed_ok": pred_ok,
            "execution_error": pred_err,
            "verify_ok": None,
            "verify_issue": "",
            "correct": bool(gold_ok and pred_ok and matches(gold_rows, pred_rows)),
        })

    final_correct = per_iter[-1]["correct"] if per_iter else False
    return {
        "question": q_text,
        "db_id": db_id,
        "gold_sql": gold_sql,
        "gold_executed_ok": gold_ok,
        "gold_error": gold_err,
        "iterations": agent_data.get("iterations", len(per_iter)),
        "final_sql": agent_data.get("sql", ""),
        "final_correct": final_correct,
        "per_iter": per_iter,
        "elapsed_seconds": elapsed,
        "agent_error": None,
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results.

    Per-iteration carry-forward: if the agent terminated at iteration j < k
    (verify said ok at j, or it hit MAX_ITERATIONS at j < k), treat the
    question's iteration-k result as identical to its iteration-j result.
    The agent stopped emitting; whatever it had at termination is what
    would have been served had we polled at iteration k.
    """
    n = len(results)
    n_completed = sum(1 for r in results if r.get("agent_error") is None)
    errors = n - n_completed
    overall_pass_rate = (
        sum(1 for r in results if r.get("final_correct")) / n if n else 0.0
    )

    # iter_k counts how many questions would have been correct if we
    # had stopped emitting after iteration k. Carry-forward: once a
    # question's agent terminated, its last per_iter entry is treated
    # as the answer for all higher k.
    iter_correct_counts = [0] * MAX_ITERATIONS
    for r in results:
        per_iter = r.get("per_iter") or []
        last_correct = False
        for k in range(MAX_ITERATIONS):
            if k < len(per_iter):
                last_correct = bool(per_iter[k]["correct"])
            # else: carry last_correct forward (agent already terminated)
            if last_correct:
                iter_correct_counts[k] += 1

    per_iter_pass_rate = {
        f"iter_{k}": (iter_correct_counts[k] / n if n else 0.0)
        for k in range(MAX_ITERATIONS)
    }

    iter_distribution: dict[str, int] = {}
    for r in results:
        key = f"iter_{r.get('iterations', 0)}"
        iter_distribution[key] = iter_distribution.get(key, 0) + 1

    return {
        "n": n,
        "n_completed": n_completed,
        "errors": errors,
        "overall_pass_rate": overall_pass_rate,
        "per_iter_pass_rate": per_iter_pass_rate,
        "iteration_distribution": dict(sorted(iter_distribution.items())),
    }


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
