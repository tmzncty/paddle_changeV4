"""Compatibility shim for historical searchable-PDF scripts.

Legacy code imports ``find_unique_json_parent_paths`` from this root-level
module.  The maintained implementation now lives in ``src/paddle_batch_ocr``;
this shim keeps the old import working when a script is executed directly from
a repository checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from paddle_batch_ocr.discovery import find_json_parent_paths

find_unique_json_parent_paths = find_json_parent_paths

__all__ = ["find_unique_json_parent_paths"]
