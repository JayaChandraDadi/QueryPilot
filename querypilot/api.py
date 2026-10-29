"""HTTP API: POST /ask runs the agent for a question under a role."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .agent import QueryPilot
from .policy import PolicyViolation


class AskRequest(BaseModel):
    question: str
    role: str


def create_app(agent: QueryPilot) -> FastAPI:
    app = FastAPI(title="QueryPilot", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/tables")
    def tables(role: str):
        try:
            visible = agent.policies.visible_tables(role)
        except PolicyViolation as e:
            raise HTTPException(404, str(e))
        return {t: {"description": agent.catalog.tables[t].description,
                    "columns": list(agent.catalog.tables[t].columns)} for t in visible}

    @app.post("/ask")
    def ask(req: AskRequest):
        try:
            s = agent.ask(req.question, req.role)
        except PolicyViolation as e:
            raise HTTPException(403, str(e))
        body = {"status": s["status"], "sql": s.get("governed_sql") or s.get("sql"), "attempts": s.get("attempts"),
                "lineage": s.get("lineage"), "error": s.get("error") if s["status"] != "ok" else None}
        if s["status"] == "ok":
            body["columns"] = s["columns"]
            body["rows"] = [list(r) for r in s["rows"]]
            body["truncated"] = s.get("truncated", False)
        return body

    @app.get("/audit")
    def audit(limit: int = 20):
        return agent.audit.tail(limit)

    return app
