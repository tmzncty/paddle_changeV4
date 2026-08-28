"""Filesystem discovery helpers.

These functions are dependency-free so they can be tested on every pull
request without installing Paddle or CUDA libraries.
"""

from __future__ import annotations

import os
from os import PathLike
from typing import Union

Pathish = Union[str, PathLike[str]]


def find_json_parent_paths(base_dir: Pathish) -> list[str]:
    """Return sorted directories below *base_dir* that contain ``.json`` files.

    This preserves the historical behavior used by the searchable-PDF scripts
    while making traversal order deterministic.
    """

    parents: set[str] = set()

    for root, _, files in os.walk(os.fspath(base_dir)):
        if any(filename.endswith(".json") for filename in files):
            parents.add(root)

    return sorted(parents)
