# QueryPilot

QueryPilot is a governed text-to-SQL agent. It answers natural-language questions about a data
warehouse by generating SQL with an LLM, enforcing access policies on that SQL, running it
read-only, and logging the lineage of every answer.

It is built with LangGraph, sqlglot, DuckDB and FastAPI. It works with Claude, OpenAI, or a
local model through Ollama.

```
question ──> retrieve ──> generate ──> govern ──> execute ──> answer + audit record
                             ▲            │          │
                             └── repair <─┴──────────┘  (policy, syntax or execution error)
```

## What each step does

**Retrieve.** BM25 over a metadata catalog of table and column descriptions, synonyms and a
business glossary picks the tables a question needs. *Join-path completion* then adds any
bridge tables required to connect them through foreign keys. Similar solved examples are
retrieved as few-shot demonstrations.

**Generate.** The model writes one DuckDB `SELECT` from that schema context. On a retry it
sees its previous SQL plus the exact error (policy message, binder error or execution error),
so it corrects itself.

**Govern.** The SQL is parsed and fully qualified with sqlglot, so policies apply to what the
query actually reads. Aliases, CTEs, subqueries, UNIONs and `SELECT *` all resolve to base
columns. Per role ([`policies.yaml`](querypilot/config/policies.yaml)):
- *allowed tables*
- *denied columns*: blocked anywhere, including WHERE, JOIN, GROUP BY and inside functions, because predicates leak values too
- *masked columns*: can only be selected, and come back as MD5 hashes
- *row filters*: every read of the table is rewritten into a filtered subquery, including reads inside CTEs and subqueries

It also rejects anything that is not a single `SELECT`, table functions such as `read_csv`, and
schema-qualified system tables. The tables and columns read become the answer's lineage.

**Execute.** DuckDB runs the query read-only, with external file and network access disabled,
a timeout, and a row cap.

**Audit.** Every question is appended to a JSON-lines audit log with: role, generated and
governed SQL, lineage, row filters applied, status, and number of attempts.

Example: role `support_west` sees only western data. A question about all regions is rewritten,
not answered with someone else's data:

```sql
-- generated
SELECT region, COUNT(*) AS n FROM customers GROUP BY region
-- executed
SELECT customers.region AS region, COUNT(*) AS n
FROM (SELECT * FROM customers WHERE region = 'WEST') AS customers GROUP BY customers.region
```

## Quick start

```bash
pip install -e '.[dev,anthropic]'
querypilot build                       # sample retail warehouse (DuckDB) + eval set
export QUERYPILOT_LLM=ollama QUERYPILOT_MODEL=qwen2.5-coder:7b     # or:
# export QUERYPILOT_LLM=anthropic QUERYPILOT_MODEL=<claude model id> ANTHROPIC_API_KEY=...
querypilot ask "Which product category had the highest revenue in 2024?" --role analyst
querypilot serve                       # POST /ask {"question": ..., "role": ...}, GET /tables, GET /audit
```

The sample warehouse has six tables (stores, customers, products, orders, order_items,
employees), about 50K orders and 125K line items, all generated from a fixed seed. Three roles
are defined:
- `analyst`: no emails, phones or salaries; names are masked
- `support_west`: WEST rows only; no employees table or credit limits
- `finance`: unrestricted

## Evaluation

The eval set ([`querypilot/evalset.py`](querypilot/evalset.py)) has 120 question/gold-SQL pairs
from 25 templates (41 easy, 49 medium, 30 hard). It comes with a separate bank of 18 few-shot
examples that use different question shapes. Every gold query is checked by the tests to run and
return rows.

```bash
querypilot eval-retrieval                    # schema retrieval recall, no LLM needed
querypilot eval --config baseline --out eval_results/baseline.json
querypilot eval --config full     --out eval_results/full.json
querypilot eval --oracle                     # sanity check: gold SQL scores 100%
```

The two `eval` configurations:
- `baseline`: every visible table in the prompt, no examples, one attempt
- `full`: retrieved schema plus join completion, 3 examples, up to 3 attempts with error feedback

Execution accuracy compares result rows with the gold query's rows: as multisets, with floats
rounded to 2 decimals, and in order when the gold query uses `ORDER BY ... LIMIT`.

### Schema retrieval (measured, no LLM involved)

| tables retrieved (k) | recall, ranking only | recall, with join-path completion | avg. tables in prompt |
|---|---|---|---|
| 2 | 54.2% | 60.0% | 2.2 |
| **3** | 76.7% | **93.3%** | **2.8** (of 6) |
| 4 | 91.7% | 93.3% | 3.1 |

"Recall" here means the share of questions whose gold query's tables are all in the prompt.

### Governance

`tests/test_policy.py` checks 35 adversarial queries that must be blocked. They reach denied
data through:
- aliases, CTEs (including nested `SELECT *`) and derived tables
- UNION / UNION ALL and scalar subqueries
- denied columns used in WHERE, JOIN, ORDER BY or GROUP BY, or inside functions
- misuse of masked columns
- table functions (`read_csv`, `read_parquet`), `information_schema`
- DML/DDL, multi-statement input, COPY and ATTACH

All 35 are blocked. The same file checks 12 legitimate queries that must keep working, and
verifies row filters and masks on the actual query results.

### LLM accuracy

Execution accuracy depends on the model. Run `querypilot eval` with your model of choice and the
report breaks it down by difficulty, attempts per question, and how many answers the repair loop
fixed.

## Tests

```bash
pytest -q   # 64 tests
```

The tests cover:
- the governance suite
- agent behavior with a scripted LLM: self-correction from execution errors and from policy feedback, giving up after N attempts, audit records, row filters on agent answers
- the HTTP API
- the eval harness and retrieval recall

## Layout

```
querypilot/
  agent.py       LangGraph state machine (retrieve, generate, govern, execute, finish)
  retrieval.py   BM25 schema/example retrieval + join-path completion
  policy.py      sqlglot-based access control, masking, row filters, lineage
  catalog.py     metadata catalog (types from DuckDB, descriptions from catalog.yaml)
  executor.py    read-only DuckDB execution with timeout
  llm.py         Anthropic / OpenAI-compatible (Ollama, vLLM) / scripted clients
  evaluation.py  execution accuracy and retrieval recall
  api.py, cli.py
  config/        catalog.yaml, policies.yaml
```
