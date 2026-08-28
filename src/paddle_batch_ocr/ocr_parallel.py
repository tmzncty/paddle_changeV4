"""Spawn-based multi-process OCR execution.

The parallel layer deliberately reuses the serial task/result contract. Each
worker owns one lazy PaddleX pipeline and, when enabled, one SQLite connection.
"""

from __future__ import annotations

import atexit
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .manifest import ManifestStore
from .ocr_runner import (
    OcrBatchResult,
    OcrRunnerError,
    OcrTask,
    OcrTaskResult,
    _validate_existing_result,
)
from .paddlex_adapter import PipelineRef, create_ocr_pipeline, predict_one_to_json


@dataclass(frozen=True)
class ParallelWorkerConfig:
    pipeline_ref: PipelineRef
    device: str
    engine: Optional[str]
    use_hpip: Optional[bool]
    predict_kwargs: Dict[str, object]
    execution_profile: Dict[str, object]
    overwrite: bool
    manifest_path: Optional[str]
    create_pipeline_fn: Optional[Callable[..., object]] = None


_WORKER_CONFIG: Optional[ParallelWorkerConfig] = None
_WORKER_PIPELINE: Optional[object] = None
_WORKER_PIPELINE_ERROR: Optional[Exception] = None
_WORKER_STORE: Optional[ManifestStore] = None


def _close_worker_store() -> None:
    global _WORKER_STORE
    if _WORKER_STORE is not None:
        try:
            _WORKER_STORE.close()
        finally:
            _WORKER_STORE = None


def _init_parallel_worker(config: ParallelWorkerConfig) -> None:
    """Initialize per-process state without loading a PaddleX model yet."""

    global _WORKER_CONFIG, _WORKER_PIPELINE, _WORKER_PIPELINE_ERROR, _WORKER_STORE

    _WORKER_CONFIG = config
    _WORKER_PIPELINE = None
    _WORKER_PIPELINE_ERROR = None
    _WORKER_STORE = (
        ManifestStore(Path(config.manifest_path))
        if config.manifest_path is not None
        else None
    )
    atexit.register(_close_worker_store)


def _get_worker_pipeline() -> object:
    """Create at most one PaddleX pipeline in the current process."""

    global _WORKER_PIPELINE, _WORKER_PIPELINE_ERROR

    config = _WORKER_CONFIG
    if config is None:
        raise OcrRunnerError("parallel OCR worker was not initialized")

    if _WORKER_PIPELINE is not None:
        return _WORKER_PIPELINE

    if _WORKER_PIPELINE_ERROR is not None:
        raise OcrRunnerError(
            "PaddleX pipeline initialization previously failed in this worker: "
            f"{type(_WORKER_PIPELINE_ERROR).__name__}: {_WORKER_PIPELINE_ERROR}"
        ) from _WORKER_PIPELINE_ERROR

    try:
        _WORKER_PIPELINE = create_ocr_pipeline(
            config.pipeline_ref,
            device=config.device,
            engine=config.engine,
            use_hpip=config.use_hpip,
            create_pipeline_fn=config.create_pipeline_fn,
        )
    except Exception as exc:
        _WORKER_PIPELINE_ERROR = exc
        raise

    return _WORKER_PIPELINE


def _execute_parallel_task(task: OcrTask) -> OcrTaskResult:
    """Execute one already-preflighted OCR task in a worker process."""

    config = _WORKER_CONFIG
    if config is None:
        return OcrTaskResult(
            source=task.source,
            output_json=task.output_json,
            status="failed",
            error="OcrRunnerError: parallel OCR worker was not initialized",
        )

    store = _WORKER_STORE
    try:
        if store is not None:
            store.mark_started(
                task.source,
                "ocr",
                worker=f"pid-{os.getpid()}",
                device=config.device,
                intended_result_path=task.output_json,
                execution_profile=config.execution_profile,
            )

        predict_one_to_json(
            _get_worker_pipeline(),
            task.source,
            task.output_json,
            overwrite=config.overwrite,
            predict_kwargs=config.predict_kwargs,
        )

        if store is not None:
            store.mark_success(
                task.source,
                "ocr",
                result_path=task.output_json,
            )

        return OcrTaskResult(
            source=task.source,
            output_json=task.output_json,
            status="success",
        )
    except Exception as exc:
        if store is not None:
            try:
                store.mark_failure(
                    task.source,
                    "ocr",
                    exc,
                    intended_result_path=task.output_json,
                    execution_profile=config.execution_profile,
                )
            except Exception:
                pass
        return OcrTaskResult(
            source=task.source,
            output_json=task.output_json,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )


def _prepare_tasks(
    tasks: Tuple[OcrTask, ...],
    *,
    manifest_path: Optional[Path],
    resume: bool,
    overwrite: bool,
    execution_profile: Dict[str, object],
) -> Tuple[List[Tuple[int, OcrTask]], Dict[int, OcrTaskResult]]:
    """Resolve resume/stale/existing-output semantics before spawning workers."""

    pending: List[Tuple[int, OcrTask]] = []
    prepared: Dict[int, OcrTaskResult] = {}
    store = ManifestStore(manifest_path) if manifest_path is not None else None

    try:
        for index, task in enumerate(tasks):
            try:
                previous = (
                    store.get_job(task.source, "ocr") if store is not None else None
                )

                # Existing files with no manifest history can be adopted, but
                # their original execution profile is unknowable and must stay NULL.
                if (
                    task.output_json.exists()
                    and resume
                    and store is not None
                    and previous is None
                ):
                    _validate_existing_result(task.output_json)
                    store.mark_success(
                        task.source,
                        "ocr",
                        result_path=task.output_json,
                    )
                    prepared[index] = OcrTaskResult(
                        source=task.source,
                        output_json=task.output_json,
                        status="skipped",
                    )
                    continue

                needs_run = (
                    store.needs_run(
                        task.source,
                        "ocr",
                        intended_result_path=task.output_json,
                        execution_profile=execution_profile,
                    )
                    if store is not None
                    else True
                )

                if task.output_json.exists():
                    can_adopt = (
                        resume
                        and (
                            store is None
                            or not needs_run
                        )
                    )
                    if can_adopt:
                        _validate_existing_result(task.output_json)
                        if store is not None:
                            store.mark_success(
                                task.source,
                                "ocr",
                                result_path=task.output_json,
                            )
                        prepared[index] = OcrTaskResult(
                            source=task.source,
                            output_json=task.output_json,
                            status="skipped",
                        )
                        continue

                    if not overwrite:
                        reason = (
                            "existing result is stale according to manifest; use --overwrite"
                            if (
                                store is not None
                                and previous is not None
                                and needs_run
                            )
                            else "OCR output already exists; enable resume or overwrite"
                        )
                        raise OcrRunnerError(f"{reason}: {task.output_json}")

                pending.append((index, task))
            except Exception as exc:
                if store is not None:
                    try:
                        store.mark_failure(
                            task.source,
                            "ocr",
                            exc,
                            intended_result_path=task.output_json,
                            execution_profile=execution_profile,
                        )
                    except Exception:
                        pass
                prepared[index] = OcrTaskResult(
                    source=task.source,
                    output_json=task.output_json,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
    finally:
        if store is not None:
            store.close()

    return pending, prepared


def run_ocr_parallel(
    tasks: Tuple[OcrTask, ...],
    *,
    workers: int,
    pipeline_ref: PipelineRef,
    device: Optional[str],
    engine: Optional[str],
    use_hpip: Optional[bool],
    manifest_path: Optional[Path],
    resume: bool,
    overwrite: bool,
    predict_kwargs: Dict[str, object],
    execution_profile: Dict[str, object],
    create_pipeline_fn: Optional[Callable[..., object]] = None,
    start_method: str = "spawn",
) -> OcrBatchResult:
    """Execute tasks with one lazy pipeline per spawned CPU worker process."""

    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 2:
        raise OcrRunnerError("parallel OCR requires workers >= 2")

    if device != "cpu":
        raise OcrRunnerError(
            "workers > 1 currently requires explicit device='cpu'; "
            "GPU worker pools need an explicit device map and are not enabled yet"
        )

    if start_method != "spawn":
        raise OcrRunnerError(
            "parallel OCR currently supports only the 'spawn' start method"
        )

    pending, results = _prepare_tasks(
        tasks,
        manifest_path=manifest_path,
        resume=resume,
        overwrite=overwrite,
        execution_profile=execution_profile,
    )

    if not pending:
        return OcrBatchResult(
            tasks=tuple(results[index] for index in range(len(tasks)))
        )

    config = ParallelWorkerConfig(
        pipeline_ref=pipeline_ref,
        device=device,
        engine=engine,
        use_hpip=use_hpip,
        predict_kwargs=dict(predict_kwargs),
        execution_profile=dict(execution_profile),
        overwrite=overwrite,
        manifest_path=(
            os.fspath(Path(manifest_path).expanduser())
            if manifest_path is not None
            else None
        ),
        create_pipeline_fn=create_pipeline_fn,
    )

    context = multiprocessing.get_context(start_method)
    max_workers = min(workers, len(pending))

    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=context,
        initializer=_init_parallel_worker,
        initargs=(config,),
    ) as executor:
        future_map = {
            executor.submit(_execute_parallel_task, task): (index, task)
            for index, task in pending
        }

        for future in as_completed(future_map):
            index, task = future_map[future]
            try:
                results[index] = future.result()
            except Exception as exc:
                if manifest_path is not None:
                    try:
                        with ManifestStore(manifest_path) as store:
                            store.mark_failure(
                                task.source,
                                "ocr",
                                exc,
                                intended_result_path=task.output_json,
                                execution_profile=execution_profile,
                            )
                    except Exception:
                        pass
                results[index] = OcrTaskResult(
                    source=task.source,
                    output_json=task.output_json,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )

    return OcrBatchResult(
        tasks=tuple(results[index] for index in range(len(tasks)))
    )
