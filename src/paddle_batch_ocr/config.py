"""Project configuration with conservative defaults and path-conflict checks."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

from .safety import UnsafePathError, is_within, validate_destructive_target


class ConfigError(ValueError):
    """Raised when a project configuration is invalid or unsafe."""


def _as_path(value: object, *, base_dir: Path) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ConfigError(f"expected a filesystem path, got {type(value).__name__}")
    path = Path(os.fspath(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve(strict=False)


def _as_manifest_path(value: object, *, base_dir: Path) -> Path:
    """Make a manifest path absolute without erasing final symlink identity.

    Manifest readers/writers reject symlinked database paths before opening
    SQLite. Calling ``resolve()`` here would follow the link too early and make
    that safety boundary impossible to enforce. Other containment checks still
    resolve paths through ``is_within`` when they need realpath semantics.
    """

    if not isinstance(value, (str, os.PathLike)):
        raise ConfigError(f"expected a filesystem path, got {type(value).__name__}")
    path = Path(os.fspath(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.absolute()


def _positive_int(value: object, *, name: str, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{name} must be an integer >= 1")
    return value


def _bool_value(value: object, *, name: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be true or false")
    return value


def _device_value(value: object) -> str:
    if not isinstance(value, str):
        raise ConfigError("device must be a string")
    device = value.lower()
    if device in {"auto", "cpu", "gpu"} or re.fullmatch(r"gpu:\d+", device):
        return device
    raise ConfigError("device must be 'auto', 'cpu', 'gpu', or a GPU selector such as 'gpu:0'")


@dataclass(frozen=True)
class InputSource:
    path: Path
    kind: str

    def __post_init__(self) -> None:
        if self.kind not in {"pdf", "image"}:
            raise ConfigError("input source kind must be 'pdf' or 'image'")


@dataclass(frozen=True)
class RuntimeConfig:
    ocr_workers: int = 1
    pdf_prep_workers: int = 1
    render_workers: int = 1
    batch_size: int = 1
    device: str = "auto"

    def __post_init__(self) -> None:
        for name in ("ocr_workers", "pdf_prep_workers", "render_workers", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigError(f"{name} must be an integer >= 1")
        _device_value(self.device)


@dataclass(frozen=True)
class ProjectConfig:
    input_sources: Tuple[InputSource, ...]
    output_root: Path
    log_dir: Path
    cache_root: Path
    manifest_path: Path
    paddle_config: Optional[Path] = None
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    delete_temp_images: bool = False
    overwrite: bool = False
    resume: bool = True

    def validate_paths(self) -> None:
        roots = [source.path for source in self.input_sources]
        named_paths = {
            "output_root": self.output_root,
            "log_dir": self.log_dir,
            "cache_root": self.cache_root,
            "manifest_path": self.manifest_path,
        }

        for source_root in roots:
            for name, target in named_paths.items():
                if target == source_root or is_within(target, source_root):
                    raise ConfigError(
                        f"{name} ({target}) must not be equal to or nested inside input source ({source_root})"
                    )

        if self.output_root == self.cache_root or is_within(self.output_root, self.cache_root):
            raise ConfigError("output_root must not be equal to or nested inside cache_root")
        if self.cache_root == self.output_root or is_within(self.cache_root, self.output_root):
            raise ConfigError("cache_root must not be equal to or nested inside output_root")

        if (
            self.log_dir == self.cache_root
            or is_within(self.log_dir, self.cache_root)
            or is_within(self.cache_root, self.log_dir)
        ):
            raise ConfigError("log_dir and cache_root must not overlap")

        if self.manifest_path == self.cache_root or is_within(self.manifest_path, self.cache_root):
            raise ConfigError("manifest_path must not be stored inside cache_root")

        try:
            validate_destructive_target(self.cache_root / "temp", self.cache_root)
        except UnsafePathError as exc:
            raise ConfigError(f"unsafe cache_root: {exc}") from exc


def _load_mapping(path: Path) -> Mapping[str, Any]:
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as handle:
        if suffix == ".json":
            data = json.load(handle)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
            except ImportError as exc:
                raise ConfigError(
                    "YAML configuration requires PyYAML; install paddle-batch-ocr[yaml] or use JSON"
                ) from exc
            data = yaml.safe_load(handle)
        else:
            raise ConfigError("project config must use .json, .yaml, or .yml")

    if not isinstance(data, Mapping):
        raise ConfigError("project config root must be a mapping/object")
    return data


def config_from_mapping(data: Mapping[str, Any], *, base_dir: Path) -> ProjectConfig:
    raw_sources = data.get("input_sources")
    if not isinstance(raw_sources, Sequence) or isinstance(raw_sources, (str, bytes)) or not raw_sources:
        raise ConfigError("input_sources must be a non-empty list")

    sources = []
    for index, raw in enumerate(raw_sources):
        if not isinstance(raw, Mapping):
            raise ConfigError(f"input_sources[{index}] must be an object")
        if "path" not in raw:
            raise ConfigError(f"input_sources[{index}].path is required")
        kind = raw.get("type", raw.get("kind"))
        if not isinstance(kind, str):
            raise ConfigError(f"input_sources[{index}].type is required")
        sources.append(InputSource(path=_as_path(raw["path"], base_dir=base_dir), kind=kind.lower()))

    for required in ("output_root", "log_dir", "cache_root"):
        if required not in data:
            raise ConfigError(f"{required} is required")

    output_root = _as_path(data["output_root"], base_dir=base_dir)
    log_dir = _as_path(data["log_dir"], base_dir=base_dir)
    cache_root = _as_path(data["cache_root"], base_dir=base_dir)

    manifest_raw = data.get("manifest_path")
    manifest_path = (
        _as_manifest_path(manifest_raw, base_dir=base_dir)
        if manifest_raw not in (None, "")
        else (log_dir / "manifest.sqlite3").absolute()
    )

    raw_runtime = data.get("runtime", {})
    if not isinstance(raw_runtime, Mapping):
        raise ConfigError("runtime must be an object")

    runtime = RuntimeConfig(
        ocr_workers=_positive_int(raw_runtime.get("ocr_workers"), name="ocr_workers", default=1),
        pdf_prep_workers=_positive_int(raw_runtime.get("pdf_prep_workers"), name="pdf_prep_workers", default=1),
        render_workers=_positive_int(raw_runtime.get("render_workers"), name="render_workers", default=1),
        batch_size=_positive_int(raw_runtime.get("batch_size"), name="batch_size", default=1),
        device=_device_value(raw_runtime.get("device", data.get("device", "auto"))),
    )

    paddle_config_raw = data.get("paddle_config")
    paddle_config = _as_path(paddle_config_raw, base_dir=base_dir) if paddle_config_raw not in (None, "") else None

    config = ProjectConfig(
        input_sources=tuple(sources),
        output_root=output_root,
        log_dir=log_dir,
        cache_root=cache_root,
        manifest_path=manifest_path,
        paddle_config=paddle_config,
        runtime=runtime,
        delete_temp_images=_bool_value(
            data.get("delete_temp_images"), name="delete_temp_images", default=False
        ),
        overwrite=_bool_value(data.get("overwrite"), name="overwrite", default=False),
        resume=_bool_value(data.get("resume"), name="resume", default=True),
    )
    config.validate_paths()
    return config


def load_config(path: Union[os.PathLike[str], str]) -> ProjectConfig:
    config_path = Path(path).expanduser().resolve(strict=True)
    return config_from_mapping(_load_mapping(config_path), base_dir=config_path.parent)
