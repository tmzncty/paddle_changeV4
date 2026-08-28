"""Thin, lazy adapters around the current PaddleX pipeline/result API.

This module deliberately contains no top-level PaddleX import. The core package
must remain usable for configuration, PDF reconstruction and manifest tooling on
machines where Paddle/PaddleX is not installed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Optional, Union

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
    """Return a PaddleX/PaddleOCR result as its documented export mapping.

    Current PaddleX ``OCRResult`` behaves like a Mapping *and* exposes a
    documented ``json`` export. The raw Mapping contains runtime-only objects
    such as ``vis_fonts`` and, depending on pipeline settings, image arrays.
    PaddleX's ``OCRResult._to_json()`` intentionally excludes those fields.

    Therefore prefer the Result ``json`` export whenever present, even if the
    object also satisfies ``Mapping``. Plain historical dictionaries continue
    to be accepted directly for compatibility and dependency-free tests.
    """

    try:
        json_value = getattr(result, "json")
    except (AttributeError, RuntimeError):
        json_value = None

    if json_value is not None:
        if callable(json_value):
            json_value = json_value()
        if not isinstance(json_value, Mapping):
            raise PaddleXResultError(
                f"PaddleX result json must be a mapping, got {type(json_value).__name__}"
            )
        data: Mapping[str, object] = json_value
    elif isinstance(result, Mapping):
        data = result
    else:
        raise PaddleXResultError(
            "PaddleX result exposes neither a json mapping nor a Mapping interface"
        )

    return _unwrap_result_mapping(data)


def _json_safe(value: object, *, path: str = "result") -> object:
    """Convert PaddleX/NumPy values to ordinary JSON-compatible Python values.

    PaddleX's exported JSON mapping can contain NumPy arrays/scalars before its
    final serializer converts them. Normalize those values without importing
    NumPy into the core package. Unknown values fail explicitly instead of being
    silently stringified.
    """

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        converted: Dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise PaddleXResultError(
                    f"{path} contains non-string mapping key {key!r}"
                )
            converted[key] = _json_safe(item, path=f"{path}.{key}")
        return converted
    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe(tolist(), path=path)
        except Exception as exc:
            if isinstance(exc, PaddleXResultError):
                raise
            raise PaddleXResultError(
                f"cannot convert array-like value at {path} to JSON-safe data"
            ) from exc

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item(), path=path)
        except Exception as exc:
            if isinstance(exc, PaddleXResultError):
                raise
            raise PaddleXResultError(
                f"cannot convert scalar-like value at {path} to JSON-safe data"
            ) from exc

    raise PaddleXResultError(
        f"unsupported PaddleX result value at {path}: {type(value).__name__}"
    )


def normalize_ocr_result(result: object) -> Dict[str, object]:
    """Validate a Result object and return a JSON-safe stable mapping."""

    mapping = result_mapping(result)
    try:
        page = parse_ocr_page(mapping)
    except OcrSchemaError as exc:
        raise PaddleXResultError(f"PaddleX OCR result violates the OCR schema: {exc}") from exc

    converted = _json_safe(mapping)
    if not isinstance(converted, dict):
        raise PaddleXResultError("normalized PaddleX OCR result is not a mapping")
    normalized: Dict[str, object] = converted
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
    """Instantiate the current PaddleX OCR pipeline lazily."""

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
                "OCR execution requires PaddleX; install PaddlePaddle first and then paddle-batch-ocr[ocr]"
            ) from exc

    kwargs = create_pipeline_kwargs(
        device=device,
        engine=engine,
        use_hpip=use_hpip,
        hpi_config=hpi_config,
    )
    return create_pipeline_fn(pipeline=pipeline_ref, **kwargs)


def default_ocr_predict_kwargs(
    *,
    use_doc_orientation_classify: bool = False,
    use_doc_unwarping: bool = False,
    use_textline_orientation: bool = False,
) -> Dict[str, object]:
    """Return the lightweight General OCR predict options used by this project."""

    for name, value in (
        ("use_doc_orientation_classify", use_doc_orientation_classify),
        ("use_doc_unwarping", use_doc_unwarping),
        ("use_textline_orientation", use_textline_orientation),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be bool")

    return {
        "use_doc_orientation_classify": use_doc_orientation_classify,
        "use_doc_unwarping": use_doc_unwarping,
        "use_textline_orientation": use_textline_orientation,
    }


def predict_one(
    pipeline: object,
    image_path: Path,
    *,
    predict_kwargs: Optional[Mapping[str, object]] = None,
) -> object:
    """Run one image through a pipeline and require exactly one Result object."""

    source = Path(image_path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise FileNotFoundError(source)

    kwargs = dict(predict_kwargs or {})
    predict_iter = getattr(pipeline, "predict_iter", None)
    if callable(predict_iter):
        output: Iterable[object] = predict_iter(input=os.fspath(source), **kwargs)
    else:
        predict = getattr(pipeline, "predict", None)
        if not callable(predict):
            raise PaddleXResultError("pipeline exposes neither predict_iter nor predict")
        output = predict(input=os.fspath(source), **kwargs)

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
    predict_kwargs: Optional[Mapping[str, object]] = None,
) -> Path:
    """Predict one image, validate the Result contract, and atomically save JSON."""

    result = predict_one(pipeline, image_path, predict_kwargs=predict_kwargs)
    payload = normalize_ocr_result(result)
    return atomic_write_json(output_path, payload, overwrite=overwrite)
