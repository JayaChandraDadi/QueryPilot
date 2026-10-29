"""The QueryPilot agent as a LangGraph state machine.

    retrieve -> generate -> govern -> execute -> finish
                   ^          |          |
                   +-- repair <----------+   (on a policy, syntax or execution error)

* retrieve: pick the relevant tables (BM25 + join-path completion) and similar
  solved examples.
* generate: ask the model for SQL. On a retry, the previous SQL and the exact
  error go back to the model, so it can correct itself.
* govern: enforce the role's access policies, rewrite for row filters and
  masks, and capture lineage.
* execute: run read-only on DuckDB with a timeout.
* finish: write the audit record.
"""
from __future__ import annotations

import dataclasses
import pathlib
import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from . import prompts
from .audit import AuditLog
from .catalog import Catalog
from .executor import ExecutionError, Executor
from .llm import LLM
from .policy import PolicyEngine, PolicyViolation, QueryError
from .retrieval import ExampleRetriever, SchemaRetriever


@dataclasses.dataclass
class AgentConfig:
    top_k_tables: int = 3
    complete_joins: bool = True
    num_examples: int = 3
    max_attempts: int = 3
    full_schema: bool = False  # baseline: put every visible table in the prompt


class AgentState(TypedDict, total=False):
    question: str
    role: str
    tables: list[str]
    examples: list[dict]
    messages: list[dict]
    sql: str
    governed_sql: str
    error: str | None
    error_kind: str | None
    attempts: int
    columns: list[str]
    rows: list[tuple]
    truncated: bool
    lineage: dict[str, Any]
    status: str
    trace: list[dict]


class QueryPilot:
    def __init__(self, catalog: Catalog, policies: PolicyEngine, executor: Executor, llm: LLM,
                 examples: list[dict] | None = None, config: AgentConfig | None = None,
                 audit: AuditLog | None = None):
        self.catalog = catalog
        self.policies = policies
        self.executor = executor
        self.llm = llm
        self.config = config or AgentConfig()
        self.schema_retriever = SchemaRetriever(catalog)
        self.example_retriever = ExampleRetriever(examples or [])
        self.audit = audit or AuditLog(None)
        self.graph = self._build()

    @classmethod
    def from_paths(cls, db_path: str | pathlib.Path, llm: LLM, examples: list[dict] | None = None,
                   config: AgentConfig | None = None, audit_path: str | None = None) -> "QueryPilot":
        catalog = Catalog.load(db_path)
        return cls(catalog, PolicyEngine.load(catalog), Executor(db_path), llm, examples, config,
                   AuditLog(audit_path))

    # -- nodes ---------------------------------------------------------------

    def _retrieve(self, s: AgentState) -> AgentState:
        visible = self.policies.visible_tables(s["role"])
        if self.config.full_schema:
            tables = visible
        else:
            tables = self.schema_retriever.retrieve(s["question"], self.config.top_k_tables,
                                                    self.config.complete_joins)
            tables = [t for t in tables if t in visible] or visible
        examples = self.example_retriever.retrieve(s["question"], self.config.num_examples)
        messages = prompts.generation_messages(s["question"], self.catalog.describe(tables), examples)
        return {"tables": tables, "examples": examples, "messages": messages, "attempts": 0, "trace": []}

    def _generate(self, s: AgentState) -> AgentState:
        messages = s["messages"]
        if s.get("error"):
            messages = prompts.repair_messages(messages, s["sql"], s["error"])
        t0 = time.perf_counter()
        sql = prompts.extract_sql(self.llm.complete(prompts.SYSTEM, messages))
        trace = s["trace"] + [{"step": "generate", "attempt": s["attempts"] + 1, "sql": sql,
                               "ms": round(1000 * (time.perf_counter() - t0))}]
        return {"sql": sql, "attempts": s["attempts"] + 1, "error": None, "error_kind": None, "trace": trace}

    def _govern(self, s: AgentState) -> AgentState:
        try:
            g = self.policies.govern(s["sql"], s["role"])
        except PolicyViolation as e:
            return {"error": f"Access policy: {e}", "error_kind": "policy",
                    "trace": s["trace"] + [{"step": "govern", "error": str(e)}]}
        except QueryError as e:
            return {"error": f"Invalid query: {e}", "error_kind": "invalid",
                    "trace": s["trace"] + [{"step": "govern", "error": str(e)}]}
        lineage = {"tables": g.tables, "columns": g.columns, "masked_columns": g.masked_columns,
                   "row_filters": g.row_filters}
        return {"governed_sql": g.sql, "lineage": lineage,
                "trace": s["trace"] + [{"step": "govern", "governed_sql": g.sql}]}

    def _execute(self, s: AgentState) -> AgentState:
        try:
            r = self.executor.run(s["governed_sql"])
        except ExecutionError as e:
            return {"error": f"Execution error: {e}", "error_kind": "execution",
                    "trace": s["trace"] + [{"step": "execute", "error": str(e)}]}
        return {"columns": r.columns, "rows": r.rows, "truncated": r.truncated, "status": "ok",
                "trace": s["trace"] + [{"step": "execute", "rows": len(r.rows)}]}

    def _finish(self, s: AgentState) -> AgentState:
        status = s.get("status") or ("blocked" if s.get("error_kind") == "policy" else "failed")
        self.audit.write({
            "role": s["role"], "question": s["question"], "status": status, "attempts": s.get("attempts", 0),
            "sql": s.get("sql"), "governed_sql": s.get("governed_sql"), "lineage": s.get("lineage"),
            "error": s.get("error") if status != "ok" else None, "rows": len(s.get("rows") or []),
        })
        return {"status": status}

    # -- routing -------------------------------------------------------------

    def _after_check(self, s: AgentState) -> str:
        if not s.get("error"):
            return "ok"
        return "retry" if s["attempts"] < self.config.max_attempts else "give_up"

    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("retrieve", self._retrieve)
        g.add_node("generate", self._generate)
        g.add_node("govern", self._govern)
        g.add_node("execute", self._execute)
        g.add_node("finish", self._finish)
        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "generate")
        g.add_edge("generate", "govern")
        g.add_conditional_edges("govern", self._after_check, {"ok": "execute", "retry": "generate", "give_up": "finish"})
        g.add_conditional_edges("execute", self._after_check, {"ok": "finish", "retry": "generate", "give_up": "finish"})
        g.add_edge("finish", END)
        return g.compile()

    def ask(self, question: str, role: str) -> AgentState:
        self.policies.role(role)  # validate early
        return self.graph.invoke({"question": question, "role": role}, {"recursion_limit": 50})
