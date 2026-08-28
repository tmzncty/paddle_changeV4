"""Crash-conscious filesystem writes used by future OCR/PDF stages."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


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

    temp_path = None
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

        if overwrite:
            os.replace(str(temp_path), str(target))
            temp_path = None
        else:
            # Publishing a hard link is atomic and fails with FileExistsError if
            # another worker has already produced the target.
            os.link(str(temp_path), str(target))
            temp_path.unlink()
            temp_path = None

        return target
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
