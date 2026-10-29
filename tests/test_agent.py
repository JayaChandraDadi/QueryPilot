from querypilot.agent import AgentConfig, QueryPilot
from querypilot.audit import AuditLog
from querypilot.llm import ScriptedLLM


def make(catalog, policies, executor, llm, **cfg):
    return QueryPilot(catalog, policies, executor, llm, examples=[{"question": "How many stores are there?",
                                                                    "sql": "SELECT COUNT(*) FROM stores"}],
                      config=AgentConfig(**cfg), audit=AuditLog(None))


def sql(s):
    return f"Here you go:\n```sql\n{s}\n```"


def test_answers_question(catalog, policies, executor):
    llm = ScriptedLLM([sql("SELECT COUNT(*) AS n FROM customers WHERE region = 'WEST'")])
    agent = make(catalog, policies, executor, llm)
    s = agent.ask("How many customers are in the west region?", "analyst")
    assert s["status"] == "ok" and s["attempts"] == 1
    assert s["columns"] == ["n"] and s["rows"][0][0] > 0
    prompt = llm.calls[0][0]["content"]
    assert "TABLE customers" in prompt and "TABLE employees" not in prompt  # retrieved, not the whole schema
    assert "How many stores are there?" in prompt  # few-shot example included


def test_self_corrects_from_execution_error(catalog, policies, executor):
    llm = ScriptedLLM([
        sql("SELECT COUNT(*) FROM orders WHERE order_date >= '2024-01-01' AND order_date < 2025"),  # type error
        sql("SELECT COUNT(*) AS n FROM orders WHERE order_date >= DATE '2024-01-01' AND order_date < DATE '2025-01-01'"),
    ])
    s = make(catalog, policies, executor, llm).ask("How many orders were placed in 2024?", "analyst")
    assert s["status"] == "ok" and s["attempts"] == 2
    repair_prompt = llm.calls[1][-1]["content"]
    assert "Execution error" in repair_prompt


def test_policy_feedback_lets_model_fix_query(catalog, policies, executor):
    llm = ScriptedLLM([
        sql("SELECT * FROM customers WHERE region = 'EAST' LIMIT 5"),
        sql("SELECT customer_id, region, segment FROM customers WHERE region = 'EAST' LIMIT 5"),
    ])
    s = make(catalog, policies, executor, llm).ask("Show five customers from the east", "analyst")
    assert s["status"] == "ok" and s["attempts"] == 2
    assert "customers.email" in llm.calls[1][-1]["content"]  # the policy message reached the model
    assert s["lineage"]["columns"] == ["customers.customer_id", "customers.region", "customers.segment"]


def test_gives_up_and_audits(catalog, policies, executor):
    llm = ScriptedLLM([sql("SELECT email FROM customers")] * 3)
    agent = make(catalog, policies, executor, llm, max_attempts=3)
    s = agent.ask("List customer emails", "analyst")
    assert s["status"] == "blocked" and s["attempts"] == 3
    rec = agent.audit.records[-1]
    assert rec["status"] == "blocked" and "customers.email" in rec["error"] and rec["role"] == "analyst"


def test_row_filters_apply_to_agent_answers(catalog, policies, executor):
    llm = ScriptedLLM([sql("SELECT region, COUNT(*) AS n FROM customers GROUP BY region")])
    agent = make(catalog, policies, executor, llm)
    s = agent.ask("Customers per region?", "support_west")
    assert [r[0] for r in s["rows"]] == ["WEST"]
    assert s["lineage"]["row_filters"] == {"customers": "region = 'WEST'"}
    assert agent.audit.records[-1]["lineage"]["row_filters"]


def test_baseline_config_uses_full_visible_schema(catalog, policies, executor):
    llm = ScriptedLLM([sql("SELECT 1 AS x")])
    make(catalog, policies, executor, llm, full_schema=True, num_examples=0).ask("anything", "support_west")
    prompt = llm.calls[0][0]["content"]
    assert "TABLE stores" in prompt and "TABLE employees" not in prompt  # employees is not visible to this role
    assert "Solved examples" not in prompt
