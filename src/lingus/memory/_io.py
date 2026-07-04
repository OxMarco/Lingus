"""Crash-safe persistence helper for the JSON-backed memory layers.

`Path.write_text` is not atomic: a crash (or a force-quit `sys.exit`) partway
through leaves a truncated file, and the memory loaders swallow the resulting
`JSONDecodeError` and start blank — silent memory loss. Writing to a temp file
and renaming it into place makes the swap atomic: a reader (the next boot) sees
either the old file or the fully-written new one, never a torn one. This is the
durability guarantee that keeps the archive JSON-backed instead of reaching for a
database at one-streamer scale (CLAUDE.md §6/§10).
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # PID-tagged temp name so concurrent writers don't clobber each other's temp.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)  # atomic on POSIX and Windows when src/dst share a dir
    finally:
        # No-op if the rename succeeded; on a write failure it clears the partial
        # temp so it can't accumulate or be mistaken for real data.
        with contextlib.suppress(OSError):
            tmp.unlink()
