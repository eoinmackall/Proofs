"""Cross-process file locks for the per-user files' read-modify-writes.

The thread locks in main.py (DAG_LOCK) and certificates.py (_LOCK) keep
the threads of one run from interleaving their read-modify-writes, but two
commands of the same user — proofs run in one terminal, proofs verify or
proofs repair in another — are two processes, and a thread lock does not
reach across them. Without a lock both can read the same file, each add
its own change, and the second os.replace lands over the first: a lemma
or a certificate line silently lost.

file_lock(path) is that lock: an exclusive flock on a sidecar file,
<path>.lock, held for the whole read-modify-write. The sidecar, not the
file itself, because the writers replace the file (temp file, os.replace)
and a lock on the old inode would not exclude a writer that opened the
new one. Lock files are gitignored (*.lock) and never deleted — deleting
one while another process waits on it would let a third lock a fresh
inode beside it.

Lock order: DAG files before certificate files. A commit holds its DAG
file's lock while it writes the certificate (main._certify_before_write),
and proofs prune takes every DAG file's lock, in sorted order, before the
certificate file's. Nothing may take a DAG file's lock while holding a
certificate file's, or the two can deadlock.

Where fcntl does not exist (Windows) the lock is a no-op: the thread locks
still serialise one run, and concurrent commands of one user are unsafe
there, as they were before this module.

This module is deliberately standalone, the way merkle.py is: it imports
nothing from this package.
"""

from __future__ import annotations

import contextlib
import os
from typing import Iterator, Union

try:
    import fcntl
except ImportError:  # Windows: no flock; see the module docstring
    fcntl = None  # type: ignore[assignment]

__all__ = ["file_lock", "LOCK_SUFFIX"]

LOCK_SUFFIX = ".lock"


@contextlib.contextmanager
def file_lock(path: Union[str, os.PathLike]) -> Iterator[None]:
    """Hold an exclusive lock on path's sidecar, <path>.lock, for the
    block: blocks until every other process holding it lets go. The
    directory is created if it is not there yet, the way the writers
    create it before their first write."""
    if fcntl is None:
        yield
        return
    lock_path = os.fspath(path) + LOCK_SUFFIX
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
