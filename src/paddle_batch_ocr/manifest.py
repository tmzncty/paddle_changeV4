"""SQLite-backed job state for crash-safe large OCR runs."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


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
    intended_result_path: Optional[str]
    execution_profile_json: Optional[str]

    @property
    def execution_profile(self) -> Optional[Dict[str, Any]]:
        if self.execution_profile_json is None:
            return None
        value = json.loads(self.execution_profile_json)
        if not isinstance(value, dict):
            raise ValueError("stored execution profile must decode to a JSON object")
        return value


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_intended_result(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    return os.fspath(Path(path).expanduser().resolve(strict=False))


def _serialize_execution_profile(
    profile: Optional[Mapping[str, object]],
) -> Optional[str]:
    if profile is None:
        return None
    if not isinstance(profile, Mapping):
        raise TypeError("execution_profile must be a mapping")
    try:
        return json.dumps(
            dict(profile),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("execution_profile must contain JSON-serializable values") from exc


class ManifestStore:
    """Small SQLite manifest keyed by ``(source_path, stage)``.

    Each process should open its own ``ManifestStore``. WAL mode plus a 30-second
    busy timeout lets short worker commits wait for one another instead of
    immediately failing with ``database is locked``.

    ``intended_result_path`` and ``execution_profile_json`` describe the output
    target and requested execution semantics even when an attempt fails. Older
    manifests are migrated in place without invalidating successful work merely
    because they do not yet have a trustworthy execution profile.
    """

    def __init__(self, path: Path):
        raw_path = Path(path).expanduser()
        if raw_path.is_symlink():
            raise ValueError(f"refusing symlinked manifest database: {raw_path}")
        self.path = raw_path.resolve(strict=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._init_schema()

    def _init_schema(self) -> None:
        # BEGIN IMMEDIATE serializes first-open migrations across spawned workers.
        # A second process waits, then re-reads table_info after the first process
        # commits instead of racing the same ALTER TABLE statement.
        self._conn.execute("BEGIN IMMEDIATE")
        try:
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
                    intended_result_path TEXT,
                    execution_profile_json TEXT,
                    PRIMARY KEY (source_path, stage)
                )
                """
            )

            columns = {
                str(row["name"])
                for row in self._conn.execute("PRAGMA table_info(jobs)").fetchall()
            }
            if "intended_result_path" not in columns:
                self._conn.execute(
                    "ALTER TABLE jobs ADD COLUMN intended_result_path TEXT"
                )
            if "execution_profile_json" not in columns:
                self._conn.execute(
                    "ALTER TABLE jobs ADD COLUMN execution_profile_json TEXT"
                )

            # A successful historical result is safe evidence for its intended
            # output path. Execution options cannot be reconstructed safely, so
            # legacy profiles deliberately remain NULL rather than being guessed.
            self._conn.execute(
                """
                UPDATE jobs
                SET intended_result_path = result_path
                WHERE intended_result_path IS NULL AND result_path IS NOT NULL
                """
            )
            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ManifestStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def _normalize_source(path: Path) -> str:
        return os.fspath(Path(path).expanduser().resolve(strict=True))

    def ensure_job(
        self,
        source: Path,
        stage: str,
        *,
        intended_result_path: Optional[Path] = None,
        execution_profile: Optional[Mapping[str, object]] = None,
    ) -> JobRecord:
        if not stage or not stage.strip():
            raise ValueError("stage must be a non-empty string")

        source_path = self._normalize_source(source)
        fingerprint = SourceFingerprint.from_path(Path(source_path))
        requested_result = _normalize_intended_result(intended_result_path)
        requested_profile = _serialize_execution_profile(execution_profile)

        # Multiple workers may discover the same source/stage simultaneously.
        # ON CONFLICT makes first registration idempotent instead of turning a
        # benign race into IntegrityError.
        self._conn.execute(
            """
            INSERT INTO jobs (
                source_path, stage, source_size, source_mtime_ns, status,
                intended_result_path, execution_profile_json
            ) VALUES (?, ?, ?, ?, 'pending', ?, ?)
            ON CONFLICT(source_path, stage) DO NOTHING
            """,
            (
                source_path,
                stage,
                fingerprint.size,
                fingerprint.mtime_ns,
                requested_result,
                requested_profile,
            ),
        )
        self._conn.commit()

        current = self.get_job(Path(source_path), stage)
        if current is None:
            raise RuntimeError("manifest failed to materialize job record")

        fingerprint_changed = (
            current.source_size != fingerprint.size
            or current.source_mtime_ns != fingerprint.mtime_ns
        )
        intended_changed = (
            requested_result is not None
            and current.intended_result_path is not None
            and current.intended_result_path != requested_result
        )
        profile_changed = (
            requested_profile is not None
            and current.execution_profile_json is not None
            and current.execution_profile_json != requested_profile
        )

        if fingerprint_changed or intended_changed or profile_changed:
            self._conn.execute(
                """
                UPDATE jobs
                SET source_size = ?, source_mtime_ns = ?, status = 'pending',
                    result_path = NULL, retry_count = 0,
                    error_class = NULL, error_message = NULL,
                    worker = NULL, device = NULL, started_at = NULL,
                    finished_at = NULL, duration_s = NULL,
                    intended_result_path = COALESCE(?, intended_result_path),
                    execution_profile_json = COALESCE(?, execution_profile_json)
                WHERE source_path = ? AND stage = ?
                """,
                (
                    fingerprint.size,
                    fingerprint.mtime_ns,
                    requested_result,
                    requested_profile,
                    source_path,
                    stage,
                ),
            )
            self._conn.commit()
        elif current.status != "success" and (
            (requested_result is not None and current.intended_result_path is None)
            or (requested_profile is not None and current.execution_profile_json is None)
        ):
            # Pending/failed records can safely learn the currently requested
            # target/profile before the next attempt. Successful legacy records
            # keep an unknown profile unknown instead of receiving guessed provenance.
            self._conn.execute(
                """
                UPDATE jobs
                SET intended_result_path = COALESCE(intended_result_path, ?),
                    execution_profile_json = COALESCE(execution_profile_json, ?)
                WHERE source_path = ? AND stage = ?
                """,
                (requested_result, requested_profile, source_path, stage),
            )
            self._conn.commit()

        refreshed = self.get_job(Path(source_path), stage)
        if refreshed is None:
            raise RuntimeError("manifest record disappeared after fingerprint refresh")
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

    def needs_run(
        self,
        source: Path,
        stage: str,
        *,
        intended_result_path: Optional[Path] = None,
        execution_profile: Optional[Mapping[str, object]] = None,
    ) -> bool:
        record = self.ensure_job(
            source,
            stage,
            intended_result_path=intended_result_path,
            execution_profile=execution_profile,
        )
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
        intended_result_path: Optional[Path] = None,
        execution_profile: Optional[Mapping[str, object]] = None,
    ) -> JobRecord:
        record = self.ensure_job(
            source,
            stage,
            intended_result_path=intended_result_path,
            execution_profile=execution_profile,
        )
        normalized_result = _normalize_intended_result(intended_result_path)
        profile_json = _serialize_execution_profile(execution_profile)
        self._conn.execute(
            """
            UPDATE jobs
            SET status = 'running', result_path = NULL,
                started_at = ?, finished_at = NULL, duration_s = NULL,
                error_class = NULL, error_message = NULL,
                worker = ?, device = ?,
                intended_result_path = COALESCE(?, intended_result_path),
                execution_profile_json = COALESCE(?, execution_profile_json)
            WHERE source_path = ? AND stage = ?
            """,
            (
                _now_iso(),
                worker,
                device,
                normalized_result,
                profile_json,
                record.source_path,
                stage,
            ),
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
        normalized_result = _normalize_intended_result(result_path)
        self._conn.execute(
            """
            UPDATE jobs
            SET status = 'success', result_path = ?, finished_at = ?,
                duration_s = ?, error_class = NULL, error_message = NULL,
                intended_result_path = COALESCE(intended_result_path, ?)
            WHERE source_path = ? AND stage = ?
            """,
            (
                normalized_result,
                _now_iso(),
                duration_s,
                normalized_result,
                record.source_path,
                stage,
            ),
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
        intended_result_path: Optional[Path] = None,
        execution_profile: Optional[Mapping[str, object]] = None,
    ) -> JobRecord:
        record = self.ensure_job(
            source,
            stage,
            intended_result_path=intended_result_path,
            execution_profile=execution_profile,
        )
        self._conn.execute(
            """
            UPDATE jobs
            SET status = 'failed', result_path = NULL,
                retry_count = retry_count + 1,
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
