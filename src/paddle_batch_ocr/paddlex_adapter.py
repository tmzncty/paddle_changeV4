"""Thin, lazy adapters around the current PaddleX pipeline/result API.

This module deliberately contains no top-level PaddleX import. The core package
must remain usable for configuration, PDF reconstruction and manifest tooling on
machines where Paddle/PaddleX is not installed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Union

from .io_utils import atomic_write_json
from .ocr_schema import OcrPage, OcrSchemaError, parse_ocr_page


class PaddleXDependencyError(RuntimeError):
    """Raised when OCR execution is requested without PaddleX installed."""


class PaddleXResultError(RuntimeError):
    """Raised when a PaddleX result cannot be converted to the stable OCR contract."""


PipelineRef = Union[str, os.PathLike[str]]


def _unwrap_result_mapping(data: Mapping[str, object]) -> Mapping[str, object]:
    """Unwrap the common ``{"res": {...}}`` result envelope when present."""

    nested = data.get("res")
    has_direct_ocr_fields = any(
        field in data for field in ("rec_texts", "rec_text", "rec_polys", "dt_polys")
    )
    if not has_direct_ocr_fields and isinstance(nested, Mapping):
        return nested
    return data


def result_mapping(result: object) -> Mapping[str, object]:
    """Return a PaddleX/PaddleOCR result as a mapping without writing a file.

    Current PaddleX Result objects expose a ``json`` attribute whose content is
    equivalent to ``save_to_json`` output. Mapping inputs are also accepted so
    the adapter can be unit-tested without importing PaddleX.
    """

    if isinstance(result, Mapping):
        data = result
    else:
        try:
            json_value = getattr(result, "json")
        except (AttributeError, RuntimeError) as exc:
            raise PaddleXResultError("PaddleX result does not expose a json attribute") from exc

        # Accept a callable accessor defensively for older/custom Result wrappers,
        # while treating the documented property form as the primary API.
        if callable(json_value):
            json_value = json_value()
        if not isinstance(json_value, Mapping):
            raise PaddleXResultError(
                f"PaddleX result json must be a mapping, got {type(json_value).__name__}"
            )
        data = json_value

    return _unwrap_result_mapping(data)


def normalize_ocr_result(result: object) -> Dict[str, object]:
    """Validate a Result object and return its stable JSON mapping.

    Validation goes through ``parse_ocr_page`` so modern ``rec_polys`` /
    ``rec_texts`` pairing and historical field fallbacks share one contract.
    The original mapping fields are preserved for provenance and future use.
    """

    mapping = result_mapping(result)
    try:
        page = parse_ocr_page(mapping)
    except OcrSchemaError as exc:
        raise PaddleXResultError(f"PaddleX OCR result violates the OCR schema: {exc}") from exc

    normalized: Dict[str, object] = dict(mapping)
    normalized["_paddle_batch_ocr"] = {
        "schema": 1,
        "polygon_field": page.polygon_field,
        "text_field": page.text_field,
    }
    return normalized


def parse_paddlex_ocr_result(result: object) -> OcrPage:
    """Convert a current/legacy result directly into the internal OCR page model."""

    try:
        return parse_ocr_page(result_mapping(result))
    except OcrSchemaError as exc:
        raise PaddleXResultError(f"PaddleX OCR result violates the OCR schema: {exc}") from exc


def create_pipeline_kwargs(
    *,
    device: Optional[str] = None,
    engine: Optional[str] = None,
    use_hpip: Optional[bool] = None,
    hpi_config: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Build current PaddleX ``create_pipeline`` kwargs without legacy aliases."""

    kwargs: Dict[str, object] = {}
    if device not in (None, "auto"):
        kwargs["device"] = device
    if engine is not None:
        kwargs["engine"] = engine
    if use_hpip is not None:
        kwargs["use_hpip"] = use_hpip
    if hpi_config is not None:
        kwargs["hpi_config"] = dict(hpi_config)
    return kwargs


def create_ocr_pipeline(
    pipeline: PipelineRef = "OCR",
    *,
    device: Optional[str] = None,
    engine: Optional[str] = None,
    use_hpip: Optional[bool] = None,
    hpi_config: Optional[Mapping[str, object]] = None,
    create_pipeline_fn: Optional[Callable[..., object]] = None,
) -> object:
    """Instantiate the current PaddleX OCR pipeline lazily.

    ``pipeline`` may be ``"OCR"`` or a local PaddleX pipeline YAML path. The
    optional callable injection exists for dependency-free contract tests.
    """

    if isinstance(pipeline, os.PathLike):
        pipeline_ref = os.fspath(Path(pipeline).expanduser().resolve(strict=True))
    elif isinstance(pipeline, str):
        pipeline_ref = pipeline
    else:
        raise TypeError("pipeline must be a pipeline name or filesystem path")

    if create_pipeline_fn is None:
        try:
            from paddlex import create_pipeline as create_pipeline_fn  # type: ignore
        except ImportError as exc:
            raise PaddleXDependencyError(
                "OCR execution requires PaddleX; install PaddlePaddle first and then paddlex[ocr]"
            ) from exc

    kwargs = create_pipeline_kwargs(
        device=device,
        engine=engine,
        use_hpip=use_hpip,
        hpi_config=hpi_config,
    )
    return create_pipeline_fn(pipeline=pipeline_ref, **kwargs)


def predict_one(pipeline: object, image_path: Path) -> object:
    """Run one image through a pipeline and require exactly one Result object."""

    source = Path(image_path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise FileNotFoundError(source)

    predict_iter = getattr(pipeline, "predict_iter", None)
    if callable(predict_iter):
        output: Iterable[object] = predict_iter(input=os.fspath(source))
    else:
        predict = getattr(pipeline, "predict", None)
        if not callable(predict):
            raise PaddleXResultError("pipeline exposes neither predict_iter nor predict")
        output = predict(input=os.fspath(source))

    iterator = iter(output)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise PaddleXResultError(f"PaddleX returned no OCR result for {source}") from exc

    try:
        next(iterator)
    except StopIteration:
        return first
    raise PaddleXResultError(f"PaddleX returned multiple OCR results for one image: {source}")


def predict_one_to_json(
    pipeline: object,
    image_path: Path,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Predict one image, validate the Result contract, and atomically save JSON."""

    result = predict_one(pipeline, image_path)
    payload = normalize_ocr_result(result)
    return atomic_write_json(output_path, payload, overwrite=overwrite)
