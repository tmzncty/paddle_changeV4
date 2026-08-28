"""Small stdio helpers for machine-readable CLI modes.

Paddle/PaddleX includes native C/C++ code that may write directly to process
file descriptor 1. Replacing ``sys.stdout`` alone does not catch those writes,
so JSON-producing commands need a short-lived fd-level redirect while third
party inference code is running.
"""

from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Iterator


def _flush_native_stdio() -> None:
    """Best-effort flush of C stdio buffers without adding a runtime dependency."""

    try:
        import ctypes

        libc = ctypes.CDLL(None)
        fflush = libc.fflush
        fflush.argtypes = [ctypes.c_void_p]
        fflush.restype = ctypes.c_int
        fflush(None)
    except Exception:
        # fd redirection still works when the platform does not expose fflush.
        pass


@contextlib.contextmanager
def redirect_process_stdout_to_stderr() -> Iterator[None]:
    """Route Python *and native* stdout writes to stderr temporarily.

    When stdout/stderr do not expose real file descriptors (for example
    ``io.StringIO`` in unit tests), fall back to Python-level redirection.
    The original fd 1 is always restored before the caller emits its own JSON.
    """

    try:
        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        with contextlib.redirect_stdout(sys.stderr):
            yield
        return

    if stdout_fd == stderr_fd:
        with contextlib.redirect_stdout(sys.stderr):
            yield
        return

    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    _flush_native_stdio()

    saved_stdout_fd = os.dup(stdout_fd)
    try:
        os.dup2(stderr_fd, stdout_fd)
        with contextlib.redirect_stdout(sys.stderr):
            try:
                yield
            finally:
                try:
                    sys.stderr.flush()
                except Exception:
                    pass
                _flush_native_stdio()
    finally:
        os.dup2(saved_stdout_fd, stdout_fd)
        os.close(saved_stdout_fd)
