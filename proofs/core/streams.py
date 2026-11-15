"""Append-only record streams: the files dags/, certificates/ and
computations/ are made of.

Every run writes its records to a stream of its own, so no two processes —
on one machine or on two — ever write the same file:

    conjectures/NAME/<kind>/<user_id>/<run_id>/000001.jsonl
                                               000002.jsonl ...

run_id is the run's id (core/checkpoint.py), held by one process at a time.
The files are numbered in hex, six digits, and one is at most MAX_FILE_BYTES
(40 MB): a record that would carry a file past that starts the next one, and
a record is never split between files. Each file starts with a header line —
whose stream it is, as a JSON object of "type": "header" with the writer's
identity and the run id — and every other line is one record, a JSON object
with a "type". Nothing in a stream is rewritten, except by
proofs prune (rewrite()).

A writer opens the file for each append rather than holding it open, so a
git pull that replaces the file while a run is going loses nothing.

Reading is strict: a line that is not a JSON object stops the read with the
file and line, and so do git's conflict markers. The only conflict a stream
can have is a prune on one machine against a live run's appends on another
(rare); it is resolved by hand, never guessed at.
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple

try:                    # about twice as fast on LaTeX-heavy records; optional
    from orjson import loads as _fast_loads
except ImportError:
    _fast_loads = None

MAX_FILE_BYTES = 40 * 1024 * 1024
HEADER = "header"

_FILE_RE = re.compile(r"[0-9a-f]{6}\.jsonl\Z")
_CONFLICT_RE = re.compile(r"(<{7}|={7}|>{7})( |$)")


class StreamError(Exception):
    """A stream file that cannot be read as records: names the file and
    the line."""

    def __init__(self, path: Path, line: int, message: str) -> None:
        self.path, self.line = path, line
        super().__init__(f"{path}:{line}: {message}")


def file_name(n: int) -> str:
    return f"{n:06x}.jsonl"


def run_dir(base: Path, user_id: str, run_id: str) -> Path:
    return Path(base) / user_id / run_id


def stream_files(directory: Path) -> List[Path]:
    """One run's stream files, in order."""
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if _FILE_RE.match(p.name))


def run_dirs(base: Path) -> List[Tuple[str, str, Path]]:
    """Every (user_id, run_id, directory) under base, sorted."""
    out: List[Tuple[str, str, Path]] = []
    if not Path(base).is_dir():
        return out
    for user in sorted(Path(base).iterdir()):
        if not user.is_dir():
            continue
        for run in sorted(user.iterdir()):
            if run.is_dir():
                out.append((user.name, run.name, run))
    return out


def _lines(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise StreamError(path, 0, f"cannot be read: {e}") from e
    # Records end in \n and nothing else: splitlines() would also break a
    # record at U+0085, U+2028 or U+2029, which JSON keeps raw in a string.
    for n, line in enumerate(text.split("\n"), 1):
        if not line.strip():
            continue
        if _CONFLICT_RE.match(line):
            raise StreamError(
                path, n,
                "a git conflict marker: resolve the merge by hand (keep "
                "every record line both sides have), then run again",
            )
        try:
            obj = _loads(line)
        except ValueError as e:
            raise StreamError(path, n, f"not a JSON record ({e})") from None
        if not isinstance(obj, dict):
            raise StreamError(path, n, "not a JSON object")
        yield n, obj


def _loads(line: str) -> Any:
    """One record line: orjson when it is installed, the standard library
    otherwise — and for any line orjson refuses, so what counts as a
    record, and the error for one that is not, are the standard
    library's. orjson reads an integer wider than 64 bits as a float, which
    would change a computation's recorded output and so the lemma's hash:
    a line that parses to any float is read again by the standard library,
    which gives a real float the same value and a wide integer exactly."""
    if _fast_loads is not None:
        try:
            obj = _fast_loads(line)
        except ValueError:
            pass
        else:
            if not _has_float(obj):
                return obj
    return json.loads(line)


def _has_float(obj: Any) -> bool:
    stack = [obj]
    while stack:
        o = stack.pop()
        if isinstance(o, float):
            return True
        if isinstance(o, dict):
            stack.extend(o.values())
        elif isinstance(o, list):
            stack.extend(o)
    return False


def read_file(path: Path) -> Tuple[Optional[Dict[str, Any]], List[Tuple[int, Dict[str, Any]]]]:
    """(the header, [(line number, record), ...]) of one stream file."""
    header: Optional[Dict[str, Any]] = None
    records: List[Tuple[int, Dict[str, Any]]] = []
    for n, obj in _lines(path):
        if obj.get("type") == HEADER:
            if header is None:
                header = obj
            continue
        records.append((n, obj))
    return header, records


def records(base: Path) -> Iterator[Tuple[str, str, Path, int, Dict[str, Any]]]:
    """Every record of every stream under base: (user_id, run_id, file,
    line, record), run by run, file by file, line by line."""
    for user_id, run_id, directory in run_dirs(base):
        for path in stream_files(directory):
            _header, recs = read_file(path)
            for n, rec in recs:
                yield user_id, run_id, path, n, rec


def rewrite(path: Path,
            edit: Callable[[int, Dict[str, Any]], Optional[Dict[str, Any]]]) -> int:
    """Rewrite one stream file through edit (proofs prune): edit(line,
    record) returns the record to write in its place — itself, or a
    changed one — or None to drop it. The header and the records written
    keep their order; the file is replaced atomically (temp file,
    os.replace), and only when something changed. A file left with no
    record is deleted. Returns how many records went."""
    header, recs = read_file(path)
    kept: List[Dict[str, Any]] = []
    changed = False
    for n, rec in recs:
        out = edit(n, rec)
        if out is None or out != rec:
            changed = True
        if out is not None:
            kept.append(out)
    removed = len(recs) - len(kept)
    if not changed:
        return 0
    if not kept:
        path.unlink()
        return removed
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        if header is not None:
            f.write(_dumps(header))
        for rec in kept:
            f.write(_dumps(rec))
    os.replace(tmp, path)
    return removed


def _dumps(obj: Mapping[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"


class Writer:
    """One run's stream of one kind: appends records, a file at a time.

    header is the identity every file of the stream starts with (the
    "type": "header" key is added). Thread-safe: the loops of a parallel run
    share one writer per kind."""

    def __init__(self, base: Path, user_id: str, run_id: str,
                 header: Mapping[str, Any]) -> None:
        self.directory = run_dir(base, user_id, run_id)
        self.header = {"type": HEADER, **header}
        self._lock = threading.Lock()

    def records(self) -> List[Dict[str, Any]]:
        """What this run's stream already holds (a resumed run)."""
        out: List[Dict[str, Any]] = []
        for path in stream_files(self.directory):
            out.extend(rec for _n, rec in read_file(path)[1])
        return out

    def append(self, record: Mapping[str, Any]) -> Path:
        """Append one record; returns the file it went to."""
        data = _dumps(record)
        size = len(data.encode("utf-8"))
        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            files = stream_files(self.directory)
            path = files[-1] if files else self.directory / file_name(1)
            current = path.stat().st_size if path.exists() else 0
            if current and current + size > MAX_FILE_BYTES:
                path = self.directory / file_name(int(path.stem, 16) + 1)
                current = 0
            with open(path, "a", encoding="utf-8") as f:
                if current == 0:
                    f.write(_dumps(self.header))
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            return path
