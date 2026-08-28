"""Command-line surface for manifest-driven targeted retries."""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

from .config import load_config
from .retry import (
    RetryCandidate,
    RetryError,
    RetryExecutionResult,
    RetryPlan,
    execute_retry_plan,
    plan_failed_retries,
)
from .stdio import redirect_process_stdout_to_stderr


def _candidate_state(candidate: RetryCandidate) -> str:
    if candidate.eligible:
        return "eligible"
    reason = candidate.reason or ""
    if reason.startswith("retry target already exists"):
        return "blocked"
    return "ineligible"


def _candidate_payload(candidate: RetryCandidate) -> Dict[str, object]:
    return {
        "source": str(candidate.source),
        "stage": candidate.stage,
        "intended_result": (
            str(candidate.intended_result)
            if candidate.intended_result is not None
            else None
        ),
        "state": _candidate_state(candidate),
        "reason": candidate.reason,
    }


def _plan_payload(plan: RetryPlan, *, execute: bool) -> Dict[str, object]:
    candidates = [_candidate_payload(candidate) for candidate in plan.candidates]
    return {
        "mode": "execute" if execute else "dry-run",
        "manifest_path": str(plan.manifest_path),
        "total_matching": plan.total_matching,
        "page_count": len(plan.candidates),
        "eligible": sum(item["state"] == "eligible" for item in candidates),
        "blocked": sum(item["state"] == "blocked" for item in candidates),
        "ineligible": sum(item["state"] == "ineligible" for item in candidates),
        "overwrite": plan.overwrite,
        "candidates": candidates,
    }


def _execution_payload(result: RetryExecutionResult) -> Dict[str, object]:
    return {
        "success": result.success_count,
        "failed": result.failed_count,
        "ineligible": result.ineligible_count,
        "items": [
            {
                "source": str(item.source),
                "stage": item.stage,
                "intended_result": (
                    str(item.intended_result)
                    if item.intended_result is not None
                    else None
                ),
                "status": item.status,
                "error": item.error,
            }
            for item in result.items
        ],
    }


def command_manifest_retry_failed(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
        plan = plan_failed_retries(
            config,
            stage=args.stage,
            error_class=args.error_class,
            limit=args.limit,
            offset=args.offset,
            overwrite=args.overwrite,
        )

        if not args.execute:
            payload = _plan_payload(plan, execute=False)
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                print("mode: dry-run")
                print(f"manifest: {plan.manifest_path}")
                print(f"total_matching: {plan.total_matching}")
                print(f"page_count: {len(plan.candidates)}")
                print(f"eligible: {payload['eligible']}")
                print(f"blocked: {payload['blocked']}")
                print(f"ineligible: {payload['ineligible']}")
                for candidate in plan.candidates:
                    state = _candidate_state(candidate).upper()
                    target = candidate.intended_result or "-"
                    print(
                        f"{state:10} {candidate.stage:14} "
                        f"{candidate.source} -> {target}"
                    )
                    if candidate.reason:
                        print(f"  {candidate.reason}")
                print("dry-run only; pass --execute to run eligible retries")
            return 0

        # Paddle / oneDNN may write through native fd 1. Preserve strict JSON
        # stdout in execute --json mode exactly like the normal OCR/run commands.
        if args.json:
            with redirect_process_stdout_to_stderr():
                result = execute_retry_plan(config, plan)
        else:
            result = execute_retry_plan(config, plan)

        plan_payload = _plan_payload(plan, execute=True)
        execution = _execution_payload(result)
        payload = {**plan_payload, "execution": execution}

        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print("mode: execute")
            print(f"eligible: {plan_payload['eligible']}")
            print(f"blocked: {plan_payload['blocked']}")
            print(f"ineligible: {plan_payload['ineligible']}")
            print(f"success: {result.success_count}")
            print(f"failed: {result.failed_count}")
            for item in result.items:
                print(f"{item.status.upper():10} {item.stage:14} {item.source}")
                if item.error:
                    print(f"  {item.error}")

        return 1 if result.failed_count else 0
    except RetryError as exc:
        # The main CLI already normalizes ValueError into a concise parser-style
        # error rather than a traceback.
        raise ValueError(str(exc)) from exc


def install_manifest_retry_subparser(
    manifest_subparsers: argparse._SubParsersAction,
) -> None:
    retry = manifest_subparsers.add_parser(
        "retry-failed",
        help="Plan or explicitly execute provenance-validated failed-job retries",
    )
    retry.add_argument("--config", required=True)
    retry.add_argument(
        "--stage",
        choices=("ocr", "render", "searchable_pdf"),
        help="Restrict retries to one supported stage",
    )
    retry.add_argument("--error-class", help="Restrict retries to one error class")
    retry.add_argument("--limit", type=int, default=100)
    retry.add_argument("--offset", type=int, default=0)
    retry.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow eligible retries to replace an existing intended target",
    )
    retry.add_argument(
        "--execute",
        action="store_true",
        help="Execute eligible retries; without this flag the command is read-only",
    )
    retry.add_argument("--json", action="store_true")
    retry.set_defaults(func=command_manifest_retry_failed)
