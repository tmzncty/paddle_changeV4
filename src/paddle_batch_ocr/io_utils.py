"""Crash-conscious filesystem publication helpers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional


def _publish_temp(temp_path: Path, target: Path, *, overwrite: bool) -> None:
    if overwrite:
        os.replace(str(temp_path), str(target))
    else:
        # Hard-link publication is atomic and refuses to replace an existing
        # target. The temp file and final target are always on the same
        # filesystem because the temp is created in target.parent.
        os.link(str(temp_path), str(target))
        temp_path.unlink()


def atomic_write_bytes(
    path: Path,
    data: bytes,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically publish bytes without exposing a partially-written target."""

    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")

    target = Path(path).expanduser().resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)

    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(target.parent),
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())

        _publish_temp(temp_path, target, overwrite=overwrite)
        temp_path = None
        return target
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def atomic_write_json(
    path: Path,
    data: Any,
    *,
    overwrite: bool = False,
    indent: int = 2,
) -> Path:
    """Write JSON through a same-directory temporary file.

    ``overwrite=False`` uses an atomic hard-link publish step so concurrent
    workers cannot silently replace an existing result. ``overwrite=True``
    publishes with ``os.replace``.
    """

    target = Path(path).expanduser().resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)

    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(target.parent),
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(data, handle, ensure_ascii=False, indent=indent)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        _publish_temp(temp_path, target, overwrite=overwrite)
        temp_path = None
        return target
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
