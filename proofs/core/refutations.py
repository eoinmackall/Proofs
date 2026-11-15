"""Refutations: the record that a lemma was rejected.

A refutation is a file in the conjecture's refutations/ directory, one file
per rejection, named for the lemma, the verifier and the moment:

    conjectures/NAME/refutations/<user_id>--<run_id>--<hex>--<verifier>--<timestamp>.json

(the lemma's id, <user_id>/<run_id>/<hex>, with its slashes as "--"). One
JSON object per file, fields in this order:

    {
      "lemma_id":      "<the lemma's id>",
      "hash":          "<the lemma's Merkle hash at the moment of the rejection>",
      "verifier":      "<the user_id of the user whose run rejected it>",
      "model":         "<the model under which the rejection happened>",
      "date":          "<ISO 8601 UTC time of the rejection>",
      "verdict":       "reject",
      "justification": "<the model's justification for the rejection>",
      "file":          "<the stream file holding the version it rejects>"
    }

file is that version's stream file, relative to the conjecture
(dags/<user_id>/<run_id>/000001.jsonl), so a person can find the proof the
objection is to. The justification is the text passed back to the
verifiers when the lemma is verified again, so a known objection is not
forgotten.

A refutation counts only while its hash matches the lemma's current Merkle
hash (is_counting); when the lemma changes — its proof, its statement, or
anything it cites — the refutation is stale: the record of an objection to a
proof no longer in the files, shown to the verifiers all the same. Refutation
files are written only by proofs verify and proofs repair (a rejection
inside proofs run is feedback to the loop's own reviser, not a record), and
deleted when the lemma is verified with the objection on the table and
accepted, or when proofs prune removes the lemma.

Refutations are rare, so they are plain files, not a stream: a file is
addressed by its name, which is why the name carries the identity; the
timestamp is full, to the second, and two rejections of one lemma by one
verifier in one second are told apart by a -2, -3 suffix.
"""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

# The directory beside dags/ and certificates/ that holds the refutation
# files.
DIR_NAME = "refutations"

# The field order of a refutation file: identity first (lemma_id, hash),
# then who rejected with what (verifier, model), then the record of
# it (date, verdict, justification).
FIELDS = (
    "lemma_id", "hash", "verifier", "model", "date",
    "verdict", "justification", "file",
)

# One lock for the whole process, the certificates.py way: a read-modify-
# write on the directory (the -2 collision guard in record) must be atomic
# within this process.
_LOCK = threading.Lock()

# A filename piece: the lemma id and the verifier go into the name,
# and a lemma_id is model-generated, so whatever the model makes of an id
# must survive in a filename. Path-hostile characters become underscores;
# the un-abbreviated values always stay in the file's fields.
_NAME_HOSTILE = re.compile(r"[^A-Za-z0-9._-]+")


def dir_for(conjecture_dir: Union[str, os.PathLike]) -> Path:
    """The conjecture's refutations/ directory."""
    return Path(conjecture_dir) / DIR_NAME


def now_iso() -> str:
    """The current UTC time as an ISO 8601 stamp to the second, the same
    style certificates.py dates its lines in."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _name_stamp(when: str) -> str:
    """The date field as a filename piece: 2025-01-02T03:04:05Z becomes
    20250102T030405Z — the same moment, without the characters a filename
    would have to keep quoting."""
    return when.replace("-", "").replace(":", "")


def _safe(part: Any) -> str:
    """A value as a filename piece: path-hostile characters (and runs of
    them) become underscores, and an empty result becomes "_" rather than
    leaving two dashes next to each other in the name."""
    return _NAME_HOSTILE.sub("_", str(part)).strip(".") or "_"


def name_for(lemma_id: str, verifier: str, when: str) -> str:
    """The refutation file's name: the lemma (its id's slashes as "--"),
    the verifier and the full timestamp (see the module docstring). when
    is an ISO 8601 stamp (now_iso())."""
    lemma = "--".join(_safe(part) for part in str(lemma_id).split("/"))
    return f"{lemma}--{_safe(verifier)}--{_name_stamp(when)}.json"


def _dumps(obj: Mapping[str, Any]) -> str:
    """The record as one JSON line: FIELDS first, in order, then whatever
    extra fields the record has (forward compatibility, the
    certificates.py way), keys sorted within each group, and a newline so
    a hand-editing editor that strips trailing newlines cannot join this
    file's last line with nothing."""
    out: Dict[str, Any] = {}
    for k in FIELDS:
        if k in obj:
            out[k] = obj[k]
    for k in sorted(obj):
        if k not in out:
            out[k] = obj[k]
    # No sort_keys: it would re-sort the whole object alphabetically and
    # undo the FIELDS-first order built above.
    return json.dumps(out, ensure_ascii=False) + "\n"


def record(
    conjecture_dir: Union[str, os.PathLike],
    lemma_id: str, hash: str, verifier: str,
    model: str, verdict: str, justification: str, file: str,
    date: Optional[str] = None,
) -> Path:
    """Write one refutation file for a rejection of the lemma and return
    its path.

    The file is the rejection as it happened: the lemma, its Merkle hash at
    the moment of the rejection, the user whose run rejected it and the
    model under which the rejection happened, the moment, the verdict, the
    model's justification, and the stream file holding the version
    rejected.

    Two rejections of the same lemma by the same verifier in the same
    second are two different events and both are kept: the first takes
    the plain name, the second appends -2, the third -3, and so on.
    """
    when = date if date is not None else now_iso()
    obj = {
        "lemma_id": str(lemma_id),
        "hash": str(hash),
        "verifier": str(verifier),
        "model": str(model),
        "date": when,
        "verdict": str(verdict),
        "justification": str(justification),
        "file": str(file),
    }
    base = name_for(lemma_id, verifier, when)
    with _LOCK:
        directory = dir_for(conjecture_dir)
        os.makedirs(directory, exist_ok=True)
        # The temp file is per writer and the name is claimed with
        # os.link, which fails when the name exists: an exists() test
        # followed by os.replace would let two processes (proofs verify
        # and proofs repair of the same user, in the same second) pick the
        # same free name and the second replace would erase the first
        # rejection. The link is atomic, so the file appears whole.
        tmp = directory / f".{base}.{os.getpid()}.{threading.get_ident()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(_dumps(obj))
        try:
            path = directory / base
            n = 2
            while True:
                try:
                    os.link(tmp, path)
                    break
                except FileExistsError:
                    path = directory / (base[: -len(".json")] + f"-{n}.json")
                    n += 1
        finally:
            os.remove(tmp)
    return path


def load(path: Union[str, os.PathLike]) -> Dict[str, Any]:
    """The refutation of one file as a dict.

    Raises ValueError if the file is not a JSON object: a refutation file
    is small and a broken one is a file to look at, not a record to skip
    (callers that read many files, like load_all, do the skipping).
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a refutation file (not a JSON object)")
    return data


def load_all(
    conjecture_dir: Union[str, os.PathLike],
) -> List[Tuple[Path, Dict[str, Any]]]:
    """Every refutation on disk, as (path, record) pairs, oldest first (by
    date, then name). A file that cannot be read or parsed is skipped: a
    refutation that cannot be read cannot count."""
    directory = dir_for(conjecture_dir)
    if not directory.is_dir():
        return []
    out: List[Tuple[Path, Dict[str, Any]]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            out.append((path, load(path)))
        except (OSError, ValueError):
            continue
    out.sort(key=lambda pr: (str(pr[1].get("date") or ""), pr[0].name))
    return out


def is_counting(ref: Mapping[str, Any], current_hash: str) -> bool:
    """Whether a refutation counts: only while its hash matches the
    lemma's current Merkle hash.

    The same rule certificates.py checks for an acceptance, checked for
    the record of a rejection. current_hash is the hash the caller
    recomputed from the DAG and references.md as they stand now
    (merkle.Merkle over the in-memory DAG's lemmas and references); the
    refutation's own hash is frozen at the moment of the rejection, so
    the comparison is always against a fresh computation, never against
    itself.
    """
    return ref.get("hash") == current_hash


def delete(path: Union[str, os.PathLike]) -> None:
    """Delete one refutation file: the way a refutation is dismissed, the
    lemma having been verified with the objection on the table and
    accepted. A missing file is not an error: a dismissal that already
    happened is still a dismissal.
    """
    p = Path(path)
    if p.is_file():
        p.unlink()
