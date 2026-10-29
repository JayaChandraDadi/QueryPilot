"""Append-only JSON-lines audit log of every question, query and policy decision."""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import threading


class AuditLog:
    def __init__(self, path: str | pathlib.Path | None):
        self.path = pathlib.Path(path) if path else None
        self._lock = threading.Lock()
        self.records: list[dict] = []

    def write(self, record: dict) -> None:
        record = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(), **record}
        with self._lock:
            self.records.append(record)
            if self.path:
                with self.path.open("a") as f:
                    f.write(json.dumps(record, default=str) + "\n")

    def tail(self, n: int = 20) -> list[dict]:
        return self.records[-n:]
