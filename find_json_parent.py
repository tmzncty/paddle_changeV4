"""Compatibility helper for discovering directories that contain OCR JSON files.

This module exists because historical searchable-PDF scripts import
``find_unique_json_parent_paths`` from ``find_json_parent``.  The original
helper was not present in the public repository even though the same logic was
later inlined into ``pdf_creator_with_text_layer7.py``.

Keep this module deliberately small while the project is being refactored.
Future code should move discovery into the package-level implementation and
leave this file as a compatibility shim for legacy scripts.
"""

from __future__ import annotations

import os
from os import PathLike
from typing import Union

Pathish = Union[str, PathLike[str]]


def find_unique_json_parent_paths(base_dir: Pathish) -> list[str]:
    """Return sorted directories below *base_dir* that contain ``.json`` files.

    The behavior intentionally mirrors the discovery loop currently embedded
    in ``pdf_creator_with_text_layer7.py``:

    - walk recursively from ``base_dir``;
    - treat a directory as one logical JSON parent when it contains at least
      one filename ending in ``.json``;
    - return each parent only once;
    - return an empty list when the root does not exist or is not traversable.

    Paths are returned in deterministic sorted order so callers and tests do
    not depend on filesystem traversal order.
    """

    parents: set[str] = set()

    for root, _, files in os.walk(os.fspath(base_dir)):
        if any(filename.endswith(".json") for filename in files):
            parents.add(root)

    return sorted(parents)
