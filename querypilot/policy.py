"""Data governance for generated SQL.

Every query is parsed and fully qualified with sqlglot before it runs, so
policies apply to what the query actually touches, not to how it is written.
Aliases, CTEs, subqueries, UNIONs and `SELECT *` are all resolved to base
table columns. For each role the engine:

* rejects anything that is not a single read-only SELECT, and table functions
  (read_csv, read_parquet, ...)
* rejects tables outside the role's allow-list
* rejects denied columns wherever they appear (select list, WHERE, JOIN,
  GROUP BY, functions), since predicates can leak values too
* hashes masked columns, which may appear only as plain selected columns
* rewrites every read of a row-filtered table into a filtered subquery

It also returns the lineage (tables and columns read) for the audit log.
"""
from __future__ import annotations

import dataclasses
import pathlib

import sqlglot
import yaml
from sqlglot import exp
from sqlglot.errors import OptimizeError, ParseError
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import traverse_scope

from .catalog import CONFIG_DIR, Catalog


class PolicyViolation(Exception):
    """The query is not allowed for this role."""


class QueryError(Exception):
    """The query is malformed (syntax error, unknown table or column)."""


@dataclasses.dataclass
class Role:
    name: str
    allow_tables: set[str] | None  # None = all
    deny_columns: set[tuple[str, str]]
    mask_columns: set[tuple[str, str]]
    row_filters: dict[str, str]


@dataclasses.dataclass
class GovernedQuery:
    sql: str
    tables: list[str]
    columns: list[str]
    masked_columns: list[str]
    row_filters: dict[str, str]


def _pair(s: str) -> tuple[str, str]:
    t, c = s.split(".")
    return t, c


class PolicyEngine:
    def __init__(self, catalog: Catalog, roles: dict[str, Role]):
        self.catalog = catalog
        self.schema = catalog.sqlglot_schema()
        self.roles = roles

    @classmethod
    def load(cls, catalog: Catalog, path: str | pathlib.Path | None = None) -> "PolicyEngine":
        raw = yaml.safe_load(pathlib.Path(path or CONFIG_DIR / "policies.yaml").read_text())["roles"]
        roles = {}
        for name, r in raw.items():
            allow = r.get("allow_tables", "*")
            roles[name] = Role(
                name=name,
                allow_tables=None if allow == "*" else set(allow),
                deny_columns={_pair(c) for c in r.get("deny_columns", [])},
                mask_columns={_pair(c) for c in r.get("mask_columns", [])},
                row_filters=dict(r.get("row_filters") or {}),
            )
        return cls(catalog, roles)

    def role(self, name: str) -> Role:
        if name not in self.roles:
            raise PolicyViolation(f"unknown role '{name}'")
        return self.roles[name]

    def visible_tables(self, role: str) -> list[str]:
        r = self.role(role)
        return [t for t in self.catalog.tables if r.allow_tables is None or t in r.allow_tables]

    def govern(self, sql: str, role: str) -> GovernedQuery:
        r = self.role(role)
        try:
            statements = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]
        except ParseError as e:
            raise QueryError(f"syntax error: {e}") from e
        if len(statements) != 1:
            raise PolicyViolation("exactly one statement is allowed")
        tree = statements[0]
        if not isinstance(tree, exp.Query):
            raise PolicyViolation(f"only SELECT queries are allowed (got {tree.key.upper()})")

        cte_names = {cte.alias_or_name for cte in tree.find_all(exp.CTE)}
        for t in tree.find_all(exp.Table):
            if not isinstance(t.this, exp.Identifier):
                raise PolicyViolation(f"table functions are not allowed: {t.sql(dialect='duckdb')}")
            if t.name in cte_names:
                continue
            if t.args.get("db") or t.args.get("catalog"):
                raise PolicyViolation(f"schema-qualified tables are not allowed: {t.sql(dialect='duckdb')}")
            if t.name not in self.schema:
                raise QueryError(f"unknown table: {t.name}")
            if r.allow_tables is not None and t.name not in r.allow_tables:
                raise PolicyViolation(f"role {role} may not read table {t.name}")

        try:
            q = qualify(tree.copy(), schema=self.schema, dialect="duckdb", expand_stars=True,
                        validate_qualify_columns=True, quote_identifiers=False)
        except OptimizeError as e:
            raise QueryError(str(e)) from e

        columns: set[tuple[str, str]] = set()
        tables: set[str] = set()
        denied, misused_masks = set(), set()
        mask_targets = []
        for scope in traverse_scope(q):
            projections = scope.expression.expressions if isinstance(scope.expression, exp.Select) else []
            for col in scope.columns:
                source = scope.sources.get(col.table)
                if not isinstance(source, exp.Table) or source.name in cte_names:
                    continue  # a derived table or CTE: its own scope is checked separately
                key = (source.name, col.name)
                tables.add(source.name)
                columns.add(key)
                if key in r.deny_columns:
                    denied.add(key)
                if key in r.mask_columns:
                    parent = col.parent
                    bare = parent in projections or (isinstance(parent, exp.Alias) and parent in projections)
                    if bare:
                        mask_targets.append(col)
                    else:
                        misused_masks.add(key)
            for src in scope.sources.values():
                if isinstance(src, exp.Table) and src.name not in cte_names:
                    tables.add(src.name)
        if denied:
            names = ", ".join(sorted(f"{t}.{c}" for t, c in denied))
            raise PolicyViolation(
                f"role {role} may not access column(s) {names}; select only the columns needed, without them")
        if misused_masks:
            names = ", ".join(sorted(f"{t}.{c}" for t, c in misused_masks))
            raise PolicyViolation(
                f"masked column(s) {names} may only be selected, not filtered, joined, grouped or transformed")

        for col in mask_targets:
            hashed = exp.func("md5", exp.cast(col.copy(), "VARCHAR"))
            if isinstance(col.parent, exp.Alias):
                col.replace(hashed)
            else:
                col.replace(exp.alias_(hashed, col.name))

        applied = {}
        for t in list(q.find_all(exp.Table)):
            if t.name in cte_names or t.name not in r.row_filters:
                continue
            predicate = r.row_filters[t.name]
            inner = sqlglot.parse_one(f"SELECT * FROM {t.name} WHERE {predicate}", read="duckdb")
            t.replace(exp.Subquery(this=inner, alias=exp.TableAlias(this=exp.to_identifier(t.alias_or_name))))
            applied[t.name] = predicate

        return GovernedQuery(
            sql=q.sql(dialect="duckdb"),
            tables=sorted(tables),
            columns=sorted(f"{t}.{c}" for t, c in columns),
            masked_columns=sorted({f"{c.table}.{c.name}" for c in mask_targets}),
            row_filters=applied,
        )
