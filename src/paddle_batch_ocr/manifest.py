"""SQLite-backed job state for crash-safe large OCR runs."""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class SourceFingerprint:
    size: int
    mtime_ns: int

    @classmethod
    def from_path(cls, path: Path) -> "SourceFingerprint":
        stat = Path(path).stat()
        return cls(size=stat.st_size, mtime_ns=stat.st_mtime_ns)


@dataclass(frozen=True)
class JobRecord:
    source_path: str
    stage: str
    source_size: int
    source_mtime_ns: int
    status: str
    result_path: Optional[str]
    retry_count: int
    error_class: Optional[str]
    error_message: Optional[str]
    worker: Optional[str]
    device: Optional[str]
    started_at: Optional[str]
    finished_at: Optional[str]
    duration_s: Optional[float]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ManifestStore:
    """Small SQLite manifest keyed by ``(source_path, stage)``."""

    def __init__(self, path: Path):
        self.path = Path(path).expanduser().resolve(strict=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                source_path TEXT NOT NULL,
                stage TEXT NOT NULL,
                source_size INTEGER NOT NULL,
                source_mtime_ns INTEGER NOT NULL,
                status TEXT NOT NULL,
                result_path TEXT,
                retry_count INTEGER NOT NULL DEFAULT 0,
                error_class TEXT,
                error_message TEXT,
                worker TEXT,
                device TEXT,
                started_at TEXT,
                finished_at TEXT,
                duration_s REAL,
                PRIMARY KEY (source_path, stage)
            )
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ManifestStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def _normalize_source(path: Path) -> str:
        return os.fspath(Path(path).expanduser().resolve(strict=True))

    def ensure_job(self, source: Path, stage: str) -> JobRecord:
        if not stage or not stage.strip():
            raise ValueError("stage must be a non-empty string")

        source_path = self._normalize_source(source)
        fingerprint = SourceFingerprint.from_path(Path(source_path))
        current = self.get_job(Path(source_path), stage)

        if current is None:
            self._conn.execute(
                """
                INSERT INTO jobs (
                    source_path, stage, source_size, source_mtime_ns, status
                ) VALUES (?, ?, ?, ?, 'pending')
                """,
                (source_path, stage, fingerprint.size, fingerprint.mtime_ns),
            )
        elif (
            current.source_size != fingerprint.size
            or current.source_mtime_ns != fingerprint.mtime_ns
        ):
            self._conn.execute(
                """
                UPDATE jobs
                SET source_size = ?, source_mtime_ns = ?, status = 'pending',
                    result_path = NULL, error_class = NULL, error_message = NULL,
                    worker = NULL, device = NULL, started_at = NULL,
                    finished_at = NULL, duration_s = NULL
                WHERE source_path = ? AND stage = ?
                """,
                (fingerprint.size, fingerprint.mtime_ns, source_path, stage),
            )

        self._conn.commit()
        refreshed = self.get_job(Path(source_path), stage)
        if refreshed is None:
            raise RuntimeError("manifest failed to materialize job record")
        return refreshed

    def get_job(self, source: Path, stage: str) -> Optional[JobRecord]:
        source_path = os.fspath(Path(source).expanduser().resolve(strict=False))
        row = self._conn.execute(
            "SELECT * FROM jobs WHERE source_path = ? AND stage = ?",
            (source_path, stage),
        ).fetchone()
        if row is None:
            return None
        return JobRecord(**dict(row))

    def needs_run(self, source: Path, stage: str) -> bool:
        record = self.ensure_job(source, stage)
        if record.status != "success":
            return True
        if record.result_path and not Path(record.result_path).exists():
            return True
        return False

    def mark_started(
        self,
        source: Path,
        stage: str,
        *,
        worker: Optional[str] = None,
        device: Optional[str] = None,
    ) -> JobRecord:
        record = self.ensure_job(source, stage)
        self._conn.execute(
            """
            UPDATE jobs
            SET status = 'running', started_at = ?, finished_at = NULL,
                duration_s = NULL, error_class = NULL, error_message = NULL,
                worker = ?, device = ?
            WHERE source_path = ? AND stage = ?
            """,
            (_now_iso(), worker, device, record.source_path, stage),
        )
        self._conn.commit()
        refreshed = self.get_job(Path(record.source_path), stage)
        if refreshed is None:
            raise RuntimeError("manifest record disappeared after mark_started")
        return refreshed

    def mark_success(
        self,
        source: Path,
        stage: str,
        *,
        result_path: Optional[Path] = None,
        duration_s: Optional[float] = None,
    ) -> JobRecord:
        record = self.ensure_job(source, stage)
        normalized_result = (
            os.fspath(Path(result_path).expanduser().resolve(strict=False))
            if result_path is not None
            else None
        )
        self._conn.execute(
            """
            UPDATE jobs
            SET status = 'success', result_path = ?, finished_at = ?,
                duration_s = ?, error_class = NULL, error_message = NULL
            WHERE source_path = ? AND stage = ?
            """,
            (normalized_result, _now_iso(), duration_s, record.source_path, stage),
        )
        self._conn.commit()
        refreshed = self.get_job(Path(record.source_path), stage)
        if refreshed is None:
            raise RuntimeError("manifest record disappeared after mark_success")
        return refreshed

    def mark_failure(
        self,
        source: Path,
        stage: str,
        error: BaseException,
        *,
        duration_s: Optional[float] = None,
    ) -> JobRecord:
        record = self.ensure_job(source, stage)
        self._conn.execute(
            """
            UPDATE jobs
            SET status = 'failed', retry_count = retry_count + 1,
                error_class = ?, error_message = ?, finished_at = ?, duration_s = ?
            WHERE source_path = ? AND stage = ?
            """,
            (
                type(error).__name__,
                str(error),
                _now_iso(),
                duration_s,
                record.source_path,
                stage,
            ),
        )
        self._conn.commit()
        refreshed = self.get_job(Path(record.source_path), stage)
        if refreshed is None:
            raise RuntimeError("manifest record disappeared after mark_failure")
        return refreshed

    def summary(self) -> Dict[str, int]:
        rows = self._conn.execute(
            "SELECT status, COUNT(*) AS count FROM jobs GROUP BY status"
        ).fetchall()
        return {row["status"]: int(row["count"]) for row in rows}
