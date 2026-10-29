"""Metadata catalog: tables, columns, descriptions, relationships and glossary."""
from __future__ import annotations

import collections
import dataclasses
import pathlib

import duckdb
import yaml

CONFIG_DIR = pathlib.Path(__file__).resolve().parent / "config"


@dataclasses.dataclass
class ColumnInfo:
    name: str
    type: str
    description: str = ""
    tags: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class TableInfo:
    name: str
    description: str
    synonyms: list[str]
    columns: dict[str, ColumnInfo]


class Catalog:
    def __init__(self, tables: dict[str, TableInfo], relationships: list[tuple[str, str]], glossary: list[dict]):
        self.tables = tables
        self.relationships = relationships
        self.glossary = glossary
        self._graph = collections.defaultdict(set)
        for a, b in relationships:
            ta, tb = a.split(".")[0], b.split(".")[0]
            self._graph[ta].add(tb)
            self._graph[tb].add(ta)

    @classmethod
    def load(cls, db_path: str | pathlib.Path, catalog_yaml: str | pathlib.Path | None = None) -> "Catalog":
        """Reads column types from the database and descriptions from the YAML catalog."""
        meta = yaml.safe_load(pathlib.Path(catalog_yaml or CONFIG_DIR / "catalog.yaml").read_text())
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            rows = con.execute(
                "SELECT table_name, column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = 'main' ORDER BY table_name, ordinal_position").fetchall()
        finally:
            con.close()
        types: dict[str, dict[str, str]] = collections.defaultdict(dict)
        for t, c, ty in rows:
            types[t][c] = ty
        tables = {}
        for name, tmeta in meta["tables"].items():
            if name not in types:
                raise ValueError(f"catalog table {name} not found in {db_path}")
            cols = {}
            for col, ty in types[name].items():
                cm = (tmeta.get("columns") or {}).get(col) or {}
                cols[col] = ColumnInfo(col, ty, cm.get("description", ""), cm.get("tags", []))
            tables[name] = TableInfo(name, tmeta.get("description", ""), tmeta.get("synonyms", []), cols)
        rels = [tuple(r) for r in meta.get("relationships", [])]
        return cls(tables, rels, meta.get("glossary", []))

    def sqlglot_schema(self) -> dict[str, dict[str, str]]:
        return {t.name: {c.name: c.type for c in t.columns.values()} for t in self.tables.values()}

    def neighbors(self, table: str) -> set[str]:
        return set(self._graph.get(table, set()))

    def join_path(self, start: str, goal: str) -> list[str]:
        """Shortest chain of tables connecting two tables through foreign keys (BFS)."""
        prev = {start: None}
        queue = collections.deque([start])
        while queue:
            t = queue.popleft()
            if t == goal:
                path = []
                while t is not None:
                    path.append(t)
                    t = prev[t]
                return path[::-1]
            for n in sorted(self._graph.get(t, ())):
                if n not in prev:
                    prev[n] = t
                    queue.append(n)
        return []

    def describe(self, tables: list[str]) -> str:
        """Schema context for prompts: DDL-like listing with descriptions and join keys."""
        out = []
        for name in tables:
            t = self.tables[name]
            out.append(f"-- {t.description}")
            out.append(f"TABLE {name} (")
            cols = list(t.columns.values())
            for i, c in enumerate(cols):
                comment = f"  -- {c.description}" if c.description else ""
                out.append(f"  {c.name} {c.type}{',' if i < len(cols) - 1 else ''}{comment}")
            out.append(")")
        joins = [f"{a} = {b}" for a, b in self.relationships
                 if a.split(".")[0] in tables and b.split(".")[0] in tables]
        if joins:
            out.append("-- Join keys: " + "; ".join(joins))
        terms = [g for g in self.glossary if set(g.get("tables", [])) & set(tables)]
        if terms:
            out.append("-- Business definitions:")
            out.extend(f"--   {g['term']}: {g['definition']}" for g in terms)
        return "\n".join(out)
