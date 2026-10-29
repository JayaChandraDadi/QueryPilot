"""Prompt construction for SQL generation and repair."""
from __future__ import annotations

import re

SYSTEM = """You translate questions about a retail data warehouse into a single DuckDB SQL query.
Rules:
- Return exactly one SELECT statement inside a ```sql code block, and nothing else.
- Use only the tables and columns shown in the schema.
- Follow the business definitions in the schema comments (for example, revenue counts completed orders only).
- Use explicit JOIN ... ON with the listed join keys.
- Give every computed column a short snake_case alias.
- Round money to 2 decimals with ROUND(x, 2) only when the question asks for an amount."""


def generation_messages(question: str, schema: str, examples: list[dict]) -> list[dict]:
    parts = [f"Schema:\n{schema}"]
    if examples:
        shots = "\n\n".join(f"Question: {e['question']}\n```sql\n{e['sql']}\n```" for e in examples)
        parts.append(f"Solved examples:\n{shots}")
    parts.append(f"Question: {question}")
    return [{"role": "user", "content": "\n\n".join(parts)}]


def repair_messages(history: list[dict], sql: str, error: str) -> list[dict]:
    return history + [
        {"role": "assistant", "content": f"```sql\n{sql}\n```"},
        {"role": "user", "content": f"That query failed:\n{error}\n\nReturn a corrected query in a ```sql block."},
    ]


_BLOCK = re.compile(r"```(?:sql)?\s*(.*?)```", re.S | re.I)


def extract_sql(text: str) -> str:
    m = _BLOCK.search(text)
    sql = (m.group(1) if m else text).strip()
    return sql.rstrip(";").strip()
