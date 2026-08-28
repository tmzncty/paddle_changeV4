"""Safe cache cleanup planning and execution."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .safety import validate_destructive_target


@dataclass(frozen=True)
class CacheCleanupResult:
    target: Path
    existed: bool
    executed: bool
    recreated: bool


def cache_temp_dir(cache_root: Path) -> Path:
    return Path(cache_root).expanduser().resolve(strict=False) / "temp"


def clean_temp_cache(
    cache_root: Path,
    *,
    execute: bool = False,
    recreate: bool = True,
) -> CacheCleanupResult:
    """Clean only ``<cache_root>/temp``.

    Dry-run is the default. The cache root itself is never recursively deleted.
    When ``execute=True``, path validation happens immediately before deletion.
    """

    root = Path(cache_root).expanduser().resolve(strict=False)
    target = cache_temp_dir(root)
    safe_target = validate_destructive_target(target, root)
    existed = safe_target.exists()

    if not execute:
        return CacheCleanupResult(
            target=safe_target,
            existed=existed,
            executed=False,
            recreated=False,
        )

    if existed:
        if safe_target.is_symlink() or safe_target.is_file():
            safe_target.unlink()
        else:
            shutil.rmtree(safe_target)

    recreated = False
    if recreate:
        safe_target.mkdir(parents=True, exist_ok=True)
        recreated = True

    return CacheCleanupResult(
        target=safe_target,
        existed=existed,
        executed=True,
        recreated=recreated,
    )
