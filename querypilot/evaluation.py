"""Evaluation harness.

* Execution accuracy: run each question through the agent and compare its
  result rows with the gold query's rows (as multisets, floats rounded to 2
  decimals; order matters only when the gold query has ORDER BY ... LIMIT).
* Schema-retrieval recall@k: the share of questions where every table the
  gold query uses is in the retrieved set. This needs no LLM.
"""
from __future__ import annotations

import collections
import json
import time
from collections.abc import Callable

import sqlglot
from sqlglot import exp

from .agent import AgentConfig, QueryPilot
from .executor import Executor
from .retrieval import SchemaRetriever

CONFIGS = {
    # Whole schema in the prompt, no examples, a single attempt.
    "baseline": AgentConfig(full_schema=True, num_examples=0, max_attempts=1),
    # Retrieved schema with join completion, 3 retrieved examples, self-correction.
    "full": AgentConfig(top_k_tables=3, complete_joins=True, num_examples=3, max_attempts=3),
}


def gold_tables(sql: str) -> set[str]:
    return {t.name for t in sqlglot.parse_one(sql, read="duckdb").find_all(exp.Table)}


def _norm(v):
    if isinstance(v, float):
        return round(v, 2)
    if hasattr(v, "is_integer") and not isinstance(v, int):  # Decimal
        return round(float(v), 2)
    return v


def results_match(gold: list[tuple], pred: list[tuple], ordered: bool) -> bool:
    g = [tuple(_norm(v) for v in r) for r in gold]
    p = [tuple(_norm(v) for v in r) for r in pred]
    if ordered:
        return g == p
    return collections.Counter(g) == collections.Counter(p)


def retrieval_recall(retriever: SchemaRetriever, questions: list[dict], k: int, complete_joins: bool) -> dict:
    hits, sizes = 0, []
    for q in questions:
        got = set(retriever.retrieve(q["question"], k, complete_joins))
        sizes.append(len(got))
        hits += gold_tables(q["sql"]) <= got
    return {"k": k, "complete_joins": complete_joins, "recall": hits / len(questions),
            "avg_tables_in_prompt": sum(sizes) / len(sizes)}


def run_eval(make_agent: Callable[[AgentConfig], QueryPilot], questions: list[dict], config_name: str,
             executor: Executor, role: str = "finance", progress: bool = True) -> dict:
    agent = make_agent(CONFIGS[config_name])
    results = []
    correct = 0
    by_diff = collections.defaultdict(lambda: [0, 0])
    t0 = time.time()
    for i, q in enumerate(questions):
        gold = executor.run(q["sql"]).rows
        ordered = "order by" in q["sql"].lower() and "limit" in q["sql"].lower()
        try:
            state = agent.ask(q["question"], role)
            ok = state.get("status") == "ok" and results_match(gold, state.get("rows") or [], ordered)
            err = state.get("error")
            attempts = state.get("attempts", 0)
            sql = state.get("sql")
        except Exception as e:  # LLM/network failure counts as wrong
            ok, err, attempts, sql = False, f"{type(e).__name__}: {e}", 0, None
        correct += ok
        by_diff[q["difficulty"]][0] += ok
        by_diff[q["difficulty"]][1] += 1
        results.append({"id": q["id"], "question": q["question"], "correct": ok, "attempts": attempts,
                        "sql": sql, "error": err})
        if progress:
            print(f"[{i + 1}/{len(questions)}] {'OK ' if ok else 'BAD'} {q['question']}", flush=True)
    n = len(questions)
    return {
        "config": config_name,
        "questions": n,
        "execution_accuracy": correct / n,
        "by_difficulty": {d: c / t for d, (c, t) in sorted(by_diff.items())},
        "avg_attempts": sum(r["attempts"] for r in results) / n,
        "fixed_by_retry": sum(1 for r in results if r["correct"] and r["attempts"] > 1),
        "seconds": round(time.time() - t0, 1),
        "results": results,
    }


def summary(report: dict) -> str:
    d = ", ".join(f"{k} {v:.0%}" for k, v in report["by_difficulty"].items())
    return (f"{report['config']}: execution accuracy {report['execution_accuracy']:.1%} on {report['questions']} "
            f"questions ({d}); avg attempts {report['avg_attempts']:.2f}; fixed by retry {report['fixed_by_retry']}")


def save(report: dict, path) -> None:
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
