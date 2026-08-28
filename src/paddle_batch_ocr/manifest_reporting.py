"""Read-only aggregate and row queries for an existing job manifest."""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Set, Tuple
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
    intended_result_count: int
    execution_profile_count: int

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
            "provenance": {
                "intended_result_count": self.intended_result_count,
                "execution_profile_count": self.execution_profile_count,
            },
        }


@dataclass(frozen=True)
class ManifestJobPage:
    total_matching: int
    jobs: Tuple[Dict[str, object], ...]


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


def _jobs_columns(connection: sqlite3.Connection) -> Set[str]:
    rows = connection.execute("PRAGMA table_info(jobs)").fetchall()
    if not rows:
        raise ManifestReportingError("manifest does not contain a jobs table")
    return {str(row["name"]) for row in rows}


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
    """Aggregate a single transactionally consistent manifest snapshot."""

    connection = _readonly_connection(path)
    try:
        columns = _jobs_columns(connection)
        connection.execute("BEGIN")

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

        intended_expr = (
            "SUM(CASE WHEN intended_result_path IS NOT NULL THEN 1 ELSE 0 END)"
            if "intended_result_path" in columns
            else "0"
        )
        profile_expr = (
            "SUM(CASE WHEN execution_profile_json IS NOT NULL THEN 1 ELSE 0 END)"
            if "execution_profile_json" in columns
            else "0"
        )
        totals = connection.execute(
            """
            SELECT COUNT(*) AS total,
                   COALESCE(SUM(retry_count), 0) AS retry_total,
                   COALESCE(SUM(duration_s), 0.0) AS duration_total_s,
                   {intended_expr} AS intended_result_count,
                   {profile_expr} AS execution_profile_count
            FROM jobs
            """.format(
                intended_expr=intended_expr,
                profile_expr=profile_expr,
            )
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
            intended_result_count=int(totals["intended_result_count"]),
            execution_profile_count=int(totals["execution_profile_count"]),
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
        _jobs_columns(connection)
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


def query_manifest_job_page(
    path: Path,
    *,
    status: Optional[str] = None,
    stage: Optional[str] = None,
    error_class: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> ManifestJobPage:
    """Read count and page rows from one consistent SQLite snapshot."""

    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10000:
        raise ValueError("limit must be an integer between 1 and 10000")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be an integer >= 0")

    where, params = _job_filters(
        status=status,
        stage=stage,
        error_class=error_class,
    )

    connection = _readonly_connection(path)
    try:
        columns = _jobs_columns(connection)
        intended_select = (
            "intended_result_path"
            if "intended_result_path" in columns
            else "NULL AS intended_result_path"
        )
        profile_select = (
            "execution_profile_json"
            if "execution_profile_json" in columns
            else "NULL AS execution_profile_json"
        )
        query = (
            "SELECT source_path, stage, source_size, source_mtime_ns, status, "
            "result_path, retry_count, error_class, error_message, worker, device, "
            "started_at, finished_at, duration_s, "
            + intended_select
            + ", "
            + profile_select
            + " FROM jobs"
            + where
            + " ORDER BY source_path, stage LIMIT ? OFFSET ?"
        )

        connection.execute("BEGIN")
        total_row = connection.execute(
            "SELECT COUNT(*) AS count FROM jobs" + where,
            tuple(params),
        ).fetchone()
        total_matching = int(total_row["count"]) if total_row is not None else 0

        page_params = list(params)
        page_params.extend((limit, offset))
        rows = connection.execute(query, tuple(page_params)).fetchall()
        return ManifestJobPage(
            total_matching=total_matching,
            jobs=tuple(dict(row) for row in rows),
        )
    except sqlite3.Error as exc:
        raise ManifestReportingError(
            f"cannot query manifest jobs: {exc}"
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
    """Backward-compatible rows-only wrapper around the snapshot page query."""

    return query_manifest_job_page(
        path,
        status=status,
        stage=stage,
        error_class=error_class,
        limit=limit,
        offset=offset,
    ).jobs
