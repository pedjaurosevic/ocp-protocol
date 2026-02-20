"""
OCP Leaderboard SQLite backend.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path


class LeaderboardDB:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS results (
            id          TEXT PRIMARY KEY,
            provider    TEXT NOT NULL,
            model       TEXT NOT NULL,
            ocp_level   INTEGER,
            ocp_level_name TEXT,
            sasmi_score REAL,
            protocol_version TEXT,
            seed        INTEGER,
            timestamp   REAL,
            submitter   TEXT,
            notes       TEXT,
            full_json   TEXT NOT NULL,
            created_at  REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_results_sasmi ON results(sasmi_score DESC);
        CREATE INDEX IF NOT EXISTS idx_results_level ON results(ocp_level DESC);
        """)
        self.conn.commit()

    def insert_result(self, result: dict, submitter: str | None = None,
                      notes: str | None = None) -> str:
        rid = str(uuid.uuid4())[:8]
        self.conn.execute(
            """INSERT INTO results
               (id, provider, model, ocp_level, ocp_level_name, sasmi_score,
                protocol_version, seed, timestamp, submitter, notes, full_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                rid,
                result.get("provider", "unknown"),
                result.get("model", "unknown"),
                result.get("ocp_level"),
                result.get("ocp_level_name"),
                result.get("sasmi_score"),
                result.get("protocol_version", "0.1.0"),
                result.get("seed"),
                result.get("timestamp", time.time()),
                submitter,
                notes,
                json.dumps(result),
                time.time(),
            )
        )
        self.conn.commit()
        return rid

    def get_leaderboard(self, limit: int = 50, min_level: int = 1) -> list[dict]:
        cur = self.conn.execute(
            """SELECT id, provider, model, ocp_level, ocp_level_name, sasmi_score,
                      protocol_version, seed, timestamp, submitter, notes
               FROM results
               WHERE (ocp_level IS NULL OR ocp_level >= ?)
               ORDER BY COALESCE(sasmi_score, 0) DESC
               LIMIT ?""",
            (min_level, limit),
        )
        return [dict(r) for r in cur.fetchall()]

    def get_result(self, result_id: str) -> dict | None:
        cur = self.conn.execute(
            "SELECT full_json FROM results WHERE id = ?", (result_id,)
        )
        row = cur.fetchone()
        if not row:
            return None
        return json.loads(row["full_json"])

    def get_stats(self) -> dict:
        cur = self.conn.execute(
            """SELECT COUNT(*) as total,
                      AVG(sasmi_score) as avg_sasmi,
                      MAX(sasmi_score) as max_sasmi,
                      COUNT(DISTINCT model) as unique_models
               FROM results"""
        )
        row = dict(cur.fetchone())
        level_cur = self.conn.execute(
            "SELECT ocp_level, COUNT(*) as n FROM results GROUP BY ocp_level ORDER BY ocp_level"
        )
        row["by_level"] = {str(r["ocp_level"]): r["n"] for r in level_cur.fetchall()}
        return row

    def import_from_dir(self, results_dir: Path) -> int:
        """Import all JSON result files from a directory into the DB."""
        count = 0
        for f in sorted(results_dir.glob("ocp_*.json")):
            try:
                data = json.loads(f.read_text())
                self.insert_result(data)
                count += 1
            except Exception:
                continue
        return count
