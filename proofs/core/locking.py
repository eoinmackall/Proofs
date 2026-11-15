"""Process locks held on files: the run lock (core/checkpoint.py).

hold(path) takes an exclusive flock on path without waiting and keeps it for
as long as the process lives; is_held(path) asks whether another process
holds one. A run holds its run.lock that way, so two processes can never run
the same run, and proofs prune can tell a live run's streams from the rest.
The lock is on a sidecar file, never on a file that is replaced whole (a
lock on a replaced inode would exclude nobody), and lock files are never
deleted while their run may be live.

Where fcntl does not exist (Windows) every hold succeeds and nothing is
held: two processes can then run one run, and prune cannot see live runs.

This module imports nothing from this package.
"""

from __future__ import annotations

import os
from typing import Optional, Union

try:
    import fcntl
except ImportError:  # Windows: no flock; see the module docstring
    fcntl = None  # type: ignore[assignment]

__all__ = ["hold", "is_held"]


def hold(path: Union[str, os.PathLike]) -> Optional[int]:
    """Take an exclusive lock on path itself without waiting, and keep it:
    the open descriptor is returned, and the lock lasts until it is closed
    or the process exits. None when another process holds it. Where there
    is no flock, every hold succeeds."""
    os.makedirs(os.path.dirname(os.fspath(path)) or ".", exist_ok=True)
    fd = os.open(os.fspath(path), os.O_RDWR | os.O_CREAT, 0o644)
    if fcntl is None:
        return fd
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd


def is_held(path: Union[str, os.PathLike]) -> bool:
    """Whether another process holds path's lock (hold) right now. A path
    that does not exist is not held."""
    if fcntl is None or not os.path.exists(path):
        return False
    fd = os.open(os.fspath(path), os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return True
    finally:
        os.close(fd)
    return False
