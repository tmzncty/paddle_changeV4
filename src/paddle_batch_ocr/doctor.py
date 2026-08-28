"""Dependency-light environment diagnostics for OCR hosts."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

from .config import ProjectConfig


def _package_version(distribution: str) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _physical_memory_bytes() -> Optional[int]:
    if hasattr(os, "sysconf"):
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            pages = int(os.sysconf("SC_PHYS_PAGES"))
            if page_size > 0 and pages > 0:
                return page_size * pages
        except (OSError, ValueError, TypeError):
            pass
    return None


def _nearest_existing(path: Path) -> Path:
    candidate = path.expanduser().resolve(strict=False)
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _free_bytes(path: Path) -> Optional[int]:
    try:
        return shutil.disk_usage(_nearest_existing(path)).free
    except OSError:
        return None


def _nvidia_summary() -> Optional[str]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "nvidia-smi present but probe failed"
    output = result.stdout.strip()
    return output or f"nvidia-smi exited with code {result.returncode}"


@dataclass(frozen=True)
class DoctorReport:
    python: str
    platform: str
    cpu_count: Optional[int]
    memory_bytes: Optional[int]
    packages: Dict[str, Optional[str]]
    nvidia: Optional[str]
    path_free_bytes: Dict[str, Optional[int]]
    warnings: tuple[str, ...]

    def as_dict(self) -> dict:
        return asdict(self)


def collect_doctor_report(config: Optional[ProjectConfig] = None) -> DoctorReport:
    paths: Dict[str, Optional[int]] = {}
    warnings = []

    if config is not None:
        for label, path in (
            ("output_root", config.output_root),
            ("log_dir", config.log_dir),
            ("cache_root", config.cache_root),
        ):
            paths[label] = _free_bytes(path)

        for source in config.input_sources:
            if not source.path.exists():
                warnings.append(f"input does not exist: {source.path}")

        if config.paddle_config is not None and not config.paddle_config.exists():
            warnings.append(f"Paddle config does not exist: {config.paddle_config}")

        if config.runtime.ocr_workers > max(1, os.cpu_count() or 1):
            warnings.append("ocr_workers exceeds visible CPU count")

        if config.runtime.batch_size > 64:
            warnings.append("batch_size is high; verify accelerator memory before production use")

    packages = {
        "paddlepaddle": _package_version("paddlepaddle"),
        "paddlepaddle-gpu": _package_version("paddlepaddle-gpu"),
        "paddlex": _package_version("paddlex"),
        "paddleocr": _package_version("paddleocr"),
        "PyMuPDF": _package_version("PyMuPDF"),
        "Pillow": _package_version("Pillow"),
    }

    if packages["paddlex"] is None and packages["paddleocr"] is None:
        warnings.append("PaddleX/PaddleOCR is not installed in this Python environment")

    return DoctorReport(
        python=sys.version.split()[0],
        platform=platform.platform(),
        cpu_count=os.cpu_count(),
        memory_bytes=_physical_memory_bytes(),
        packages=packages,
        nvidia=_nvidia_summary(),
        path_free_bytes=paths,
        warnings=tuple(warnings),
    )
