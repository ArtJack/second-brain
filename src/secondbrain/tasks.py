"""SQLite-backed task storage for Artjeck."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from .config import cfg


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


class TaskStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or cfg.state_db
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    notes TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                )
                """
            )

    def add(self, title: str, notes: str = "") -> dict:
        title = title.strip()
        if not title:
            raise ValueError("task title cannot be empty")
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO tasks (title, notes, created_at) VALUES (?, ?, ?)",
                (title, notes.strip(), _now()),
            )
            task_id = int(cur.lastrowid)
        return self.get(task_id)

    def get(self, task_id: int) -> dict:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(f"task {task_id} not found")
        return dict(row)

    def list(self, status: str = "open", limit: int = 50) -> list[dict]:
        query = "SELECT * FROM tasks"
        params: list[object] = []
        if status != "all":
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def complete(self, task_id: int) -> dict:
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE tasks SET status = 'done', completed_at = ? WHERE id = ?",
                (_now(), task_id),
            )
            if cur.rowcount == 0:
                raise KeyError(f"task {task_id} not found")
        return self.get(task_id)
