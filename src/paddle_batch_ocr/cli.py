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
from .manifest import ManifestStore
from .ocr_runner import OcrRunnerError, run_ocr_batch
from .orchestrator import ProjectRunError, run_project
from .pdf_render import PdfRenderError, render_pdf
from .safety import UnsafePathError
from .searchable_pdf import SearchablePdfError, build_searchable_pdf
from .stdio import redirect_process_stdout_to_stderr

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

def _count_files(root: Path, kind: str) -> int:
    if not root.exists(): return 0
    if root.is_file():
        suffix = root.suffix.lower(); return int((kind == "pdf" and suffix == ".pdf") or (kind == "image" and suffix in IMAGE_EXTENSIONS))
    count = 0
    for _, _, files in os.walk(root):
        for filename in files:
            suffix = Path(filename).suffix.lower()
            if kind == "pdf" and suffix == ".pdf": count += 1
            elif kind == "image" and suffix in IMAGE_EXTENSIONS: count += 1
    return count

def _format_bytes(value: Optional[int]) -> str:
    if value is None: return "unknown"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB": return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"

def _load_optional_config(path: Optional[str]) -> Optional[ProjectConfig]: return load_config(path) if path else None

def command_doctor(args):
    report = collect_doctor_report(_load_optional_config(args.config))
    if args.json: print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2)); return 0
    print(f"Python: {report.python}\nPlatform: {report.platform}\nCPU count: {report.cpu_count}\nMemory: {_format_bytes(report.memory_bytes)}")
    print("Packages:")
    for name, version in report.packages.items(): print(f"  {name}: {version or 'not installed'}")
    print(f"NVIDIA: {report.nvidia or 'not detected'}")
    if report.path_free_bytes:
        print("Disk free:")
        for name, value in report.path_free_bytes.items(): print(f"  {name}: {_format_bytes(value)}")
    if report.warnings:
        print("Warnings:")
        for warning in report.warnings: print(f"  - {warning}")
    return 0

def command_scan(args):
    config = load_config(args.config); total = 0
    for source in config.input_sources:
        count = _count_files(source.path, source.kind); total += count; print(f"{source.kind:5} {count:8d}  {source.path}")
    print(f"total {total:8d}"); return 0

def command_cache_clean(args):
    config = load_config(args.config); result = clean_temp_cache(config.cache_root, execute=args.execute, recreate=not args.no_recreate); action = "cleaned" if result.executed else "would clean"; print(f"{action}: {result.target}\nexisted: {result.existed}")
    if not result.executed: print("dry-run only; pass --execute to perform deletion")
    return 0

def command_manifest_status(args):
    config = load_config(args.config)
    if config.manifest_path.exists():
        with ManifestStore(config.manifest_path) as store: summary = store.summary()
    else: summary = {}
    payload = {"manifest_path": str(config.manifest_path), "exists": config.manifest_path.exists(), "status": summary, "total": sum(summary.values())}
    if args.json: print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"manifest: {payload['manifest_path']}\nexists: {payload['exists']}")
        for status, count in sorted(summary.items()): print(f"{status:8} {count:8d}")
        print(f"total    {payload['total']:8d}")
    return 0

def command_render(args):
    result = render_pdf(Path(args.input_pdf), Path(args.output), dpi=args.dpi, overwrite=args.overwrite); payload = {"source": str(result.source), "output_dir": str(result.output_dir), "page_count": result.page_count, "dpi": result.dpi, "pages": [str(path) for path in result.page_paths]}
    if args.json: print(json.dumps(payload, ensure_ascii=False, indent=2))
    else: print(f"rendered: {result.source}\npages: {result.page_count}\ndpi: {result.dpi}\noutput: {result.output_dir}")
    return 0

def command_searchable_pdf(args):
    result = build_searchable_pdf(Path(args.images), Path(args.ocr_json), Path(args.output), overwrite=args.overwrite, y_offset=args.y_offset, fontname=args.fontname); payload = {"output_pdf": str(result.output_pdf), "page_count": result.page_count, "text_line_count": result.text_line_count, "images": [str(path) for path in result.image_paths], "ocr_json": [str(path) for path in result.json_paths]}
    if args.json: print(json.dumps(payload, ensure_ascii=False, indent=2))
    else: print(f"searchable PDF: {result.output_pdf}\npages: {result.page_count}\ntext lines: {result.text_line_count}")
    return 0

def _ocr_payload(result):
    return {"success": result.success_count, "skipped": result.skipped_count, "failed": result.failed_count, "tasks": [{"source": str(task.source), "output_json": str(task.output_json), "status": task.status, "error": task.error} for task in result.tasks]}

def command_ocr(args):
    run_kwargs = {"pipeline_ref": args.pipeline, "device": args.device, "engine": args.engine, "use_hpip": True if args.use_hpip else None, "manifest_path": Path(args.manifest) if args.manifest else None, "resume": not args.no_resume, "overwrite": args.overwrite, "use_doc_orientation_classify": args.use_doc_orientation_classify, "use_doc_unwarping": args.use_doc_unwarping, "use_textline_orientation": args.use_textline_orientation, "workers": args.workers}
    if args.json:
        with redirect_process_stdout_to_stderr(): result = run_ocr_batch(Path(args.input), Path(args.output), **run_kwargs)
    else: result = run_ocr_batch(Path(args.input), Path(args.output), **run_kwargs)
    payload = _ocr_payload(result)
    if args.json: print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"success: {result.success_count}\nskipped: {result.skipped_count}\nfailed: {result.failed_count}")
        for task in result.tasks:
            if task.status == "failed": print(f"FAILED {task.source}: {task.error}")
    return 1 if result.failed_count else 0

def command_run(args):
    config = load_config(args.config)
    if args.json:
        with redirect_process_stdout_to_stderr(): result = run_project(config, dpi=args.dpi)
    else: result = run_project(config, dpi=args.dpi)
    payload = {"success": result.success_count, "failed": result.failed_count, "items": [{"source": str(item.source), "kind": item.kind, "status": item.status, "pages_dir": str(item.pages_dir) if item.pages_dir else None, "ocr_dir": str(item.ocr_dir) if item.ocr_dir else None, "searchable_pdf": str(item.searchable_pdf) if item.searchable_pdf else None, "error": item.error} for item in result.items]}
    if args.json: print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"success: {result.success_count}\nfailed: {result.failed_count}")
        for item in result.items:
            print(f"{item.status.upper():7} {item.kind:5} {item.source}")
            if item.error: print(f"  {item.error}")
    return 1 if result.failed_count else 0

def build_parser():
    parser = argparse.ArgumentParser(prog="paddle-batch-ocr", description="Safety-first orchestration for large PaddleX/PaddleOCR OCR jobs."); parser.add_argument("--version", action="version", version=__version__); subparsers = parser.add_subparsers(dest="command", required=True)
    doctor = subparsers.add_parser("doctor", help="Inspect runtime, dependencies, GPU and disk"); doctor.add_argument("--config"); doctor.add_argument("--json", action="store_true"); doctor.set_defaults(func=command_doctor)
    scan = subparsers.add_parser("scan", help="Count configured input files without OCR"); scan.add_argument("--config", required=True); scan.set_defaults(func=command_scan)
    render = subparsers.add_parser("render", help="Render every PDF page to a transactional PNG directory"); render.add_argument("input_pdf"); render.add_argument("--output", required=True); render.add_argument("--dpi", type=int, default=144); render.add_argument("--overwrite", action="store_true"); render.add_argument("--json", action="store_true"); render.set_defaults(func=command_render)
    searchable = subparsers.add_parser("searchable-pdf", help="Build a searchable PDF from page images and Paddle OCR JSON"); searchable.add_argument("--images", required=True); searchable.add_argument("--ocr-json", required=True); searchable.add_argument("--output", required=True); searchable.add_argument("--fontname", default="china-s"); searchable.add_argument("--y-offset", type=float, default=0.0); searchable.add_argument("--overwrite", action="store_true"); searchable.add_argument("--json", action="store_true"); searchable.set_defaults(func=command_searchable_pdf)
    ocr = subparsers.add_parser("ocr", help="Run PaddleX OCR on an image or recursively scanned image directory"); ocr.add_argument("input"); ocr.add_argument("--output", required=True); ocr.add_argument("--pipeline", default="OCR"); ocr.add_argument("--device", default="auto"); ocr.add_argument("--workers", type=int, default=1, help="OCR worker processes; workers>1 currently requires --device cpu"); ocr.add_argument("--engine", choices=("paddle", "paddle_static", "paddle_dynamic", "hpi", "flexible", "transformers", "onnxruntime", "genai_client")); ocr.add_argument("--use-hpip", action="store_true"); ocr.add_argument("--use-doc-orientation-classify", action="store_true"); ocr.add_argument("--use-doc-unwarping", action="store_true"); ocr.add_argument("--use-textline-orientation", action="store_true"); ocr.add_argument("--manifest"); ocr.add_argument("--no-resume", action="store_true"); ocr.add_argument("--overwrite", action="store_true"); ocr.add_argument("--json", action="store_true"); ocr.set_defaults(func=command_ocr)
    run = subparsers.add_parser("run", help="Execute configured PDF render -> OCR -> searchable-PDF and image OCR stages"); run.add_argument("--config", required=True); run.add_argument("--dpi", type=int, default=144); run.add_argument("--json", action="store_true"); run.set_defaults(func=command_run)
    cache = subparsers.add_parser("cache"); cache_sub = cache.add_subparsers(dest="cache_command", required=True); clean = cache_sub.add_parser("clean"); clean.add_argument("--config", required=True); clean.add_argument("--execute", action="store_true"); clean.add_argument("--no-recreate", action="store_true"); clean.set_defaults(func=command_cache_clean)
    manifest = subparsers.add_parser("manifest"); manifest_sub = manifest.add_subparsers(dest="manifest_command", required=True); status = manifest_sub.add_parser("status"); status.add_argument("--config", required=True); status.add_argument("--json", action="store_true"); status.set_defaults(func=command_manifest_status)
    return parser

def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None); return int(args.func(args))
    except (ConfigError, UnsafePathError, OcrRunnerError, ProjectRunError, PdfRenderError, SearchablePdfError, FileNotFoundError, OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")

if __name__ == "__main__": raise SystemExit(main())
