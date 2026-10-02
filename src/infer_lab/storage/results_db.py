"""SQLite benchmark store. One row per (backend, workload point) run.

Kaggle wipes the VM after a session: results live in /kaggle/working/results and should
be committed to the notebook output (or downloaded) at the end of every session.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pandas as pd

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    tag         TEXT,
    backend     TEXT NOT NULL,
    model       TEXT NOT NULL,
    batch_size  INTEGER,
    prompt_len  INTEGER,
    output_len  INTEGER,
    params      TEXT NOT NULL,   -- JSON: full config / backend kwargs
    metrics     TEXT NOT NULL,   -- JSON: summarize() output + extras
    gpu         TEXT NOT NULL    -- JSON: GPUMonitor.summary()
);
"""


class ResultsDB:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def insert(
        self,
        *,
        backend: str,
        model: str,
        batch_size: int | None,
        prompt_len: int | None,
        output_len: int | None,
        params: dict,
        metrics: dict,
        gpu: dict,
        tag: str | None = None,
    ) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO runs (ts, tag, backend, model, batch_size, prompt_len, output_len,"
                " params, metrics, gpu) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (time.time(), tag, backend, model, batch_size, prompt_len, output_len,
                 json.dumps(params), json.dumps(metrics), json.dumps(gpu)),
            )
            return int(cur.lastrowid)

    def to_dataframe(self) -> pd.DataFrame:
        """Flattened view: metric and gpu JSON expanded into columns."""
        with self._conn() as c:
            df = pd.read_sql_query("SELECT * FROM runs ORDER BY id", c)
        if df.empty:
            return df
        metrics = pd.json_normalize(df.pop("metrics").map(json.loads))
        gpu = pd.json_normalize(df.pop("gpu").map(json.loads))
        return pd.concat([df, metrics, gpu.add_prefix("gpu.")], axis=1)
