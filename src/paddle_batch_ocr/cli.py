"""Unified command-line entry point for the refactored project."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Optional

from . import __version__
from .cache import clean_temp_cache
from .config import ConfigError, ProjectConfig, load_config
from .doctor import collect_doctor_report


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def _count_files(root: Path, kind: str) -> int:
    if not root.exists():
        return 0

    count = 0
    for _, _, files in os.walk(root):
        for filename in files:
            suffix = Path(filename).suffix.lower()
            if kind == "pdf" and suffix == ".pdf":
                count += 1
            elif kind == "image" and suffix in IMAGE_EXTENSIONS:
                count += 1
    return count


def _format_bytes(value: Optional[int]) -> str:
    if value is None:
        return "unknown"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


def _load_optional_config(path: Optional[str]) -> Optional[ProjectConfig]:
    return load_config(path) if path else None


def command_doctor(args: argparse.Namespace) -> int:
    config = _load_optional_config(args.config)
    report = collect_doctor_report(config)

    if args.json:
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
        return 0

    print(f"Python: {report.python}")
    print(f"Platform: {report.platform}")
    print(f"CPU count: {report.cpu_count}")
    print(f"Memory: {_format_bytes(report.memory_bytes)}")
    print("Packages:")
    for name, version in report.packages.items():
        print(f"  {name}: {version or 'not installed'}")
    print(f"NVIDIA: {report.nvidia or 'not detected'}")

    if report.path_free_bytes:
        print("Disk free:")
        for name, value in report.path_free_bytes.items():
            print(f"  {name}: {_format_bytes(value)}")

    if report.warnings:
        print("Warnings:")
        for warning in report.warnings:
            print(f"  - {warning}")

    return 0


def command_scan(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    total = 0

    for source in config.input_sources:
        count = _count_files(source.path, source.kind)
        total += count
        print(f"{source.kind:5} {count:8d}  {source.path}")

    print(f"total {total:8d}")
    return 0


def command_cache_clean(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    result = clean_temp_cache(
        config.cache_root,
        execute=args.execute,
        recreate=not args.no_recreate,
    )

    action = "cleaned" if result.executed else "would clean"
    print(f"{action}: {result.target}")
    print(f"existed: {result.existed}")
    if not result.executed:
        print("dry-run only; pass --execute to perform deletion")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paddle-batch-ocr",
        description="Safety-first orchestration for large PaddleX/PaddleOCR OCR jobs.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Inspect runtime, dependencies, GPU and disk")
    doctor.add_argument("--config", help="Optional project JSON/YAML config")
    doctor.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    doctor.set_defaults(func=command_doctor)

    scan = subparsers.add_parser("scan", help="Count configured input files without OCR")
    scan.add_argument("--config", required=True, help="Project JSON/YAML config")
    scan.set_defaults(func=command_scan)

    cache = subparsers.add_parser("cache", help="Cache maintenance")
    cache_subparsers = cache.add_subparsers(dest="cache_command", required=True)
    clean = cache_subparsers.add_parser("clean", help="Safely clean only <cache_root>/temp")
    clean.add_argument("--config", required=True, help="Project JSON/YAML config")
    clean.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete; without this flag the command is a dry-run",
    )
    clean.add_argument(
        "--no-recreate",
        action="store_true",
        help="Do not recreate the temp directory after deletion",
    )
    clean.set_defaults(func=command_cache_clean)

    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
        return int(args.func(args))
    except (ConfigError, FileNotFoundError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
