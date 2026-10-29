from fastapi.testclient import TestClient

from querypilot import evalset, evaluation
from querypilot.agent import AgentConfig, QueryPilot
from querypilot.api import create_app
from querypilot.audit import AuditLog
from querypilot.llm import ScriptedLLM
from querypilot.retrieval import SchemaRetriever


def test_api(catalog, policies, executor):
    llm = ScriptedLLM(lambda system, msgs: "```sql\nSELECT category, COUNT(*) AS n FROM products GROUP BY category "
                                           "ORDER BY category\n```")
    agent = QueryPilot(catalog, policies, executor, llm, audit=AuditLog(None))
    client = TestClient(create_app(agent))
    assert client.get("/health").json() == {"status": "ok"}
    assert "employees" not in client.get("/tables", params={"role": "support_west"}).json()
    r = client.post("/ask", json={"question": "products per category", "role": "analyst"}).json()
    assert r["status"] == "ok" and r["columns"] == ["category", "n"] and len(r["rows"]) == 7
    assert r["lineage"]["tables"] == ["products"]
    assert client.post("/ask", json={"question": "x", "role": "intruder"}).status_code == 403
    assert client.get("/audit").json()[-1]["question"] == "products per category"


def test_result_comparison():
    assert evaluation.results_match([(1, 2.004)], [(1, 2.0)], ordered=False)
    assert evaluation.results_match([("a",), ("b",)], [("b",), ("a",)], ordered=False)
    assert not evaluation.results_match([("a",), ("b",)], [("b",), ("a",)], ordered=True)
    assert not evaluation.results_match([("a",)], [("a",), ("a",)], ordered=False)


def test_oracle_scores_perfectly(catalog, policies, executor, eval_files):
    questions = evalset.load_jsonl(eval_files[0])[:25]
    gold = {q["question"]: q["sql"] for q in questions}
    llm = ScriptedLLM(lambda s, m: "```sql\n" + gold[m[0]["content"].rsplit("Question: ", 1)[1]] + "\n```")
    make = lambda cfg: QueryPilot(catalog, policies, executor, llm, config=cfg)  # noqa: E731
    report = evaluation.run_eval(make, questions, "full", executor, progress=False)
    assert report["execution_accuracy"] == 1.0


def test_eval_set_gold_queries_all_run(executor, eval_files):
    for path in eval_files:
        for q in evalset.load_jsonl(path):
            assert executor.run(q["sql"]).rows


def test_retrieval_recall(catalog, eval_files):
    questions = evalset.load_jsonl(eval_files[0])
    r = SchemaRetriever(catalog)
    with_joins = evaluation.retrieval_recall(r, questions, 3, True)
    without = evaluation.retrieval_recall(r, questions, 3, False)
    assert with_joins["recall"] > without["recall"]
    assert with_joins["recall"] >= 0.9
    assert with_joins["avg_tables_in_prompt"] < len(catalog.tables)
