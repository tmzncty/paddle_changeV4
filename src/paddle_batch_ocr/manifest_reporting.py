"""Read-only aggregate and row queries for an existing job manifest."""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple
from urllib.parse import quote


VALID_STATUSES = {"pending", "running", "success", "failed"}


class ManifestReportingError(RuntimeError):
    """Raised when a manifest cannot be inspected safely."""


@dataclass(frozen=True)
class ManifestReport:
    total: int
    status: Mapping[str, int]
    stages: Mapping[str, Mapping[str, int]]
    error_classes: Mapping[str, int]
    retry_total: int
    duration_total_s: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "total": self.total,
            "status": dict(self.status),
            "stages": {
                stage: dict(counts)
                for stage, counts in self.stages.items()
            },
            "error_classes": dict(self.error_classes),
            "retry_total": self.retry_total,
            "duration_total_s": self.duration_total_s,
        }


def _readonly_connection(path: Path) -> sqlite3.Connection:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise ManifestReportingError(
            f"refusing symlinked manifest database: {raw}"
        )

    try:
        resolved = raw.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ManifestReportingError(
            f"manifest database does not exist: {raw}"
        ) from exc

    if not resolved.is_file():
        raise ManifestReportingError(
            f"manifest path is not a file: {resolved}"
        )

    # URI mode=ro ensures inspection cannot create schema, WAL files, or mutate
    # task state even if a future query accidentally contains a write.
    uri = "file:{}?mode=ro".format(
        quote(os.fspath(resolved), safe="/")
    )
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=5.0)
        connection.execute("PRAGMA query_only=ON")
    except sqlite3.Error as exc:
        raise ManifestReportingError(
            f"cannot open manifest read-only: {resolved}: {exc}"
        ) from exc

    connection.row_factory = sqlite3.Row
    return connection


def _ensure_jobs_table(connection: sqlite3.Connection) -> None:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='jobs'"
    ).fetchone()
    if row is None:
        raise ManifestReportingError("manifest does not contain a jobs table")


def _job_filters(
    *,
    status: Optional[str],
    stage: Optional[str],
    error_class: Optional[str],
) -> Tuple[str, List[object]]:
    if status is not None and status not in VALID_STATUSES:
        raise ValueError(
            "status must be one of pending, running, success, failed"
        )
    if stage is not None and not stage.strip():
        raise ValueError("stage must be non-empty when provided")
    if error_class is not None and not error_class.strip():
        raise ValueError("error_class must be non-empty when provided")

    clauses = []
    params: List[object] = []
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    if stage is not None:
        clauses.append("stage = ?")
        params.append(stage)
    if error_class is not None:
        clauses.append("error_class = ?")
        params.append(error_class)

    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    return where, params


def read_manifest_report(path: Path) -> ManifestReport:
    """Aggregate status, stage and error information entirely in SQLite."""

    connection = _readonly_connection(path)
    try:
        _ensure_jobs_table(connection)

        status_rows = connection.execute(
            "SELECT status, COUNT(*) AS count FROM jobs GROUP BY status ORDER BY status"
        ).fetchall()
        status = {
            str(row["status"]): int(row["count"])
            for row in status_rows
        }

        stage_rows = connection.execute(
            """
            SELECT stage, status, COUNT(*) AS count
            FROM jobs
            GROUP BY stage, status
            ORDER BY stage, status
            """
        ).fetchall()
        stages: Dict[str, Dict[str, int]] = {}
        for row in stage_rows:
            stage = str(row["stage"])
            stages.setdefault(stage, {})[str(row["status"])] = int(row["count"])

        error_rows = connection.execute(
            """
            SELECT error_class, COUNT(*) AS count
            FROM jobs
            WHERE error_class IS NOT NULL AND error_class != ''
            GROUP BY error_class
            ORDER BY count DESC, error_class
            """
        ).fetchall()
        error_classes = {
            str(row["error_class"]): int(row["count"])
            for row in error_rows
        }

        totals = connection.execute(
            """
            SELECT COUNT(*) AS total,
                   COALESCE(SUM(retry_count), 0) AS retry_total,
                   COALESCE(SUM(duration_s), 0.0) AS duration_total_s
            FROM jobs
            """
        ).fetchone()
        if totals is None:
            raise ManifestReportingError("cannot aggregate manifest totals")

        return ManifestReport(
            total=int(totals["total"]),
            status=status,
            stages=stages,
            error_classes=error_classes,
            retry_total=int(totals["retry_total"]),
            duration_total_s=float(totals["duration_total_s"]),
        )
    except sqlite3.Error as exc:
        raise ManifestReportingError(
            f"cannot query manifest: {exc}"
        ) from exc
    finally:
        connection.close()


def count_manifest_jobs(
    path: Path,
    *,
    status: Optional[str] = None,
    stage: Optional[str] = None,
    error_class: Optional[str] = None,
) -> int:
    """Count all rows matching the same filters used by paged job queries."""

    where, params = _job_filters(
        status=status,
        stage=stage,
        error_class=error_class,
    )
    connection = _readonly_connection(path)
    try:
        _ensure_jobs_table(connection)
        row = connection.execute(
            "SELECT COUNT(*) AS count FROM jobs" + where,
            tuple(params),
        ).fetchone()
        if row is None:
            return 0
        return int(row["count"])
    except sqlite3.Error as exc:
        raise ManifestReportingError(
            f"cannot count manifest jobs: {exc}"
        ) from exc
    finally:
        connection.close()


def query_manifest_jobs(
    path: Path,
    *,
    status: Optional[str] = None,
    stage: Optional[str] = None,
    error_class: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> Tuple[Dict[str, object], ...]:
    """Return deterministic filtered manifest rows without mutating the DB."""

    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10000:
        raise ValueError("limit must be an integer between 1 and 10000")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be an integer >= 0")

    where, params = _job_filters(
        status=status,
        stage=stage,
        error_class=error_class,
    )
    query = (
        "SELECT source_path, stage, source_size, source_mtime_ns, status, "
        "result_path, retry_count, error_class, error_message, worker, device, "
        "started_at, finished_at, duration_s FROM jobs"
        + where
        + " ORDER BY source_path, stage LIMIT ? OFFSET ?"
    )
    params.extend((limit, offset))

    connection = _readonly_connection(path)
    try:
        _ensure_jobs_table(connection)
        rows = connection.execute(query, tuple(params)).fetchall()
        return tuple(dict(row) for row in rows)
    except sqlite3.Error as exc:
        raise ManifestReportingError(
            f"cannot query manifest jobs: {exc}"
        ) from exc
    finally:
        connection.close()
