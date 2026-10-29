"""Command line: build the demo warehouse, ask questions, serve the API, run evals."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from . import evalset, evaluation, warehouse
from .agent import AgentConfig, QueryPilot
from .catalog import Catalog
from .executor import Executor
from .llm import ScriptedLLM, from_env
from .retrieval import SchemaRetriever

DATA = pathlib.Path("data")


def _examples():
    path = DATA / "eval" / "examples.jsonl"
    return evalset.load_jsonl(path) if path.exists() else []


def main(argv=None):
    ap = argparse.ArgumentParser(prog="querypilot")
    ap.add_argument("--db", default=str(DATA / "warehouse.duckdb"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="create the sample warehouse and eval set")
    a = sub.add_parser("ask", help="answer one question")
    a.add_argument("question")
    a.add_argument("--role", default="analyst")
    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--port", type=int, default=8000)
    e = sub.add_parser("eval", help="execution accuracy over the eval set (needs an LLM)")
    e.add_argument("--config", choices=list(evaluation.CONFIGS), default="full")
    e.add_argument("--limit", type=int)
    e.add_argument("--oracle", action="store_true", help="answer with the gold SQL (checks the harness itself)")
    e.add_argument("--out")
    r = sub.add_parser("eval-retrieval", help="schema retrieval recall (no LLM needed)")
    r.add_argument("--k", type=int, nargs="+", default=[1, 2, 3, 4])
    args = ap.parse_args(argv)

    if args.cmd == "build":
        warehouse.build(args.db)
        q, ex = evalset.build(DATA / "eval")
        print(f"built {args.db}, {q}, {ex}")
        return

    if args.cmd == "eval-retrieval":
        questions = evalset.load_jsonl(DATA / "eval" / "questions.jsonl")
        retriever = SchemaRetriever(Catalog.load(args.db))
        for k in args.k:
            for complete in (False, True):
                m = evaluation.retrieval_recall(retriever, questions, k, complete)
                print(f"k={k} join_completion={'on ' if complete else 'off'} recall={m['recall']:.1%} "
                      f"avg_tables={m['avg_tables_in_prompt']:.2f}")
        return

    if args.cmd == "eval":
        questions = evalset.load_jsonl(DATA / "eval" / "questions.jsonl")[: args.limit]
        if args.oracle:
            gold = {q["question"]: q["sql"] for q in questions}
            llm = ScriptedLLM(lambda system, msgs: "```sql\n" + gold[msgs[0]["content"].rsplit("Question: ", 1)[1]] + "\n```")
        else:
            llm = from_env()
        make = lambda cfg: QueryPilot.from_paths(args.db, llm, _examples(), cfg)  # noqa: E731
        report = evaluation.run_eval(make, questions, args.config, Executor(args.db))
        print(evaluation.summary(report))
        if args.out:
            evaluation.save(report, args.out)
        return

    agent = QueryPilot.from_paths(args.db, from_env(), _examples(), AgentConfig(), audit_path="audit.jsonl")
    if args.cmd == "ask":
        s = agent.ask(args.question, args.role)
        print(json.dumps({k: s.get(k) for k in ("status", "attempts", "sql", "governed_sql", "lineage", "error")},
                         indent=2, default=str))
        if s["status"] == "ok":
            print(" | ".join(s["columns"]))
            for row in s["rows"][:20]:
                print(" | ".join(str(v) for v in row))
        sys.exit(0 if s["status"] == "ok" else 1)
    if args.cmd == "serve":
        import uvicorn

        from .api import create_app

        uvicorn.run(create_app(agent), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
