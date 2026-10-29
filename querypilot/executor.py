"""Read-only query execution on DuckDB with a timeout and a row cap."""
from __future__ import annotations

import dataclasses
import pathlib
import threading

import duckdb


class ExecutionError(Exception):
    pass


@dataclasses.dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple]
    truncated: bool


class Executor:
    def __init__(self, db_path: str | pathlib.Path, timeout_s: float = 10.0, max_rows: int = 1000):
        self.db_path = str(db_path)
        self.timeout_s = timeout_s
        self.max_rows = max_rows

    def run(self, sql: str) -> QueryResult:
        con = duckdb.connect(self.db_path, read_only=True)
        # Defense in depth: no file system or network access from SQL.
        con.execute("SET enable_external_access = false")
        timer = threading.Timer(self.timeout_s, con.interrupt)
        timer.start()
        try:
            cur = con.execute(sql)
            columns = [d[0] for d in cur.description]
            rows = cur.fetchmany(self.max_rows + 1)
        except duckdb.InterruptException as e:
            raise ExecutionError(f"query exceeded {self.timeout_s}s timeout") from e
        except duckdb.Error as e:
            raise ExecutionError(str(e).split("\n")[0]) from e
        finally:
            timer.cancel()
            con.close()
        truncated = len(rows) > self.max_rows
        return QueryResult(columns, rows[: self.max_rows], truncated)
