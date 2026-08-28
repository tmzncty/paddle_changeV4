"""Safety primitives for destructive filesystem operations.

Legacy scripts contain recursive cache deletion.  New code should validate
paths here before any delete/overwrite implementation is allowed to act.
"""

from __future__ import annotations

import os
from os import PathLike
from pathlib import Path
from typing import Union

Pathish = Union[str, PathLike[str]]


class UnsafePathError(ValueError):
    """Raised when a destructive target is outside the configured safety boundary."""


def _resolve(path: Pathish) -> Path:
    return Path(os.fspath(path)).expanduser().resolve(strict=False)


def is_within(path: Pathish, root: Pathish) -> bool:
    """Return whether *path* is equal to or contained by *root*.

    ``Path.relative_to`` avoids string-prefix mistakes such as accepting
    ``/data/cache-evil`` when the allowed root is ``/data/cache``.
    """

    resolved_path = _resolve(path)
    resolved_root = _resolve(root)

    try:
        resolved_path.relative_to(resolved_root)
    except ValueError:
        return False
    return True


def validate_destructive_target(
    target: Pathish,
    allowed_root: Pathish,
    *,
    allow_root: bool = False,
) -> Path:
    """Validate a path before recursive deletion or equivalent destructive work.

    Rules:

    - the configured allowed root itself must not be the filesystem root;
    - the target must be inside the allowed root;
    - the target must not be the filesystem root, user home, or current working
      directory;
    - deleting the allowed root itself is rejected unless ``allow_root=True``.

    The function performs validation only.  It does not delete anything.
    """

    resolved_target = _resolve(target)
    resolved_root = _resolve(allowed_root)

    filesystem_root = Path(resolved_root.anchor or os.sep).resolve(strict=False)
    home = Path.home().resolve(strict=False)
    cwd = Path.cwd().resolve(strict=False)

    if resolved_root == filesystem_root:
        raise UnsafePathError("filesystem root cannot be used as an allowed destructive root")

    if not is_within(resolved_target, resolved_root):
        raise UnsafePathError(
            f"destructive target {resolved_target} is outside allowed root {resolved_root}"
        )

    if resolved_target in {filesystem_root, home, cwd}:
        raise UnsafePathError(f"refusing destructive operation on protected path {resolved_target}")

    if resolved_target == resolved_root and not allow_root:
        raise UnsafePathError(
            "refusing to operate on the allowed root itself without allow_root=True"
        )

    return resolved_target
