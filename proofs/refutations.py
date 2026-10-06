"""Refutations: the record that a lemma was rejected.

A refutation is a file in the conjecture's refutations/ directory, one file
per rejection, named for the lemma's owner, the lemma, the verifier and the
moment of the rejection:

    conjectures/NAME/refutations/<user_id>--<lemma_id>--<verifier>--<timestamp>.json

One JSON object per file, fields in this order:

    {
      "user_id":       "<the lemma's owner>",
      "lemma_id":      "<the lemma's id within that user's file>",
      "hash":          "<the lemma's Merkle hash at the moment of the rejection>",
      "verifier":      "<the user_id of the user whose run rejected it>",
      "model":         "<the model under which the rejection happened>",
      "date":          "<ISO 8601 UTC time of the rejection>",
      "verdict":       "reject",
      "justification": "<the model's justification for the rejection>"
    }

user_id and lemma_id identify the lemma, the pair the way a certificate
does (a bare lemma_id is unique only within one user's DAG file), the
verifier is the user_id of the user whose run rejected it — the same
reading as a certificate's verifier, not the name of the verifier agent —
and the justification is the model's own reason for the rejection: the
text that must be passed back to the verifiers when the lemma is verified
again, so a known objection is not forgotten.

A refutation counts only while its hash matches the lemma's current Merkle
hash, recomputed from the DAG files and references.md as they stand now;
the comparison is a module job the way is_valid is in certificates.py, and
the "current" hash is the caller's job (merkle.Merkle over load_dag()'s
lemmas and load_references()). When the lemma changes — its proof, its
statement, or anything it cites, in any user's file — the match breaks,
the refutation is stale, and it becomes the record of an objection to a
proof that is no longer in the files: the stale refutations are the ones a
re-verification is told about, with their justifications, and they are
dropped once the changed lemma is accepted.

Refutation files are written only by proofs verify and proofs repair, and
only for lemmas that are already in the DAG (the hash they carry is the
lemma's Merkle hash at the moment of rejection, which exists only for
lemmas in the DAG): a rejection inside proofs run is feedback to the
loop's own reviser, not a committed record, so the run never writes one.
They are read by proofs verify and proofs repair (the verifiers see the
stale justifications; repair is addressed to one file by its path) and
deleted when the refutation is dismissed — the lemma was verified with
the objection on the table and accepted. Nothing else in the system
writes or reads these files.

A file is addressed by its name, which is why the name carries the
identity: the timestamp is full, to the second, so two rejections of the
same lemma by the same verifier in the same second are distinguished by a
-2, -3 suffix rather than overwritten, the way a renamed lemma id is
distinguished from the one it replaced.
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

# The field order of a refutation file: identity first (user_id, lemma_id,
# hash), then who rejected with what (verifier, model), then the record of
# it (date, verdict, justification).
FIELDS = (
    "user_id", "lemma_id", "hash", "verifier", "model", "date",
    "verdict", "justification",
)

# One lock for the whole process, the certificates.py way: a read-modify-
# write on the directory (the -2 collision guard in record) must be atomic
# within this process.
_LOCK = threading.Lock()

# A filename piece: the user_id, lemma_id and verifier go into the name,
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


def name_for(
    user_id: str, lemma_id: str, verifier: str, when: str
) -> str:
    """The refutation file's name: the owner, the lemma, the verifier and
    the full timestamp, in that order (see the module docstring). when is
    an ISO 8601 stamp (now_iso())."""
    return (
        f"{_safe(user_id)}--{_safe(lemma_id)}--{_safe(verifier)}"
        f"--{_name_stamp(when)}.json"
    )


def file_for(
    conjecture_dir: Union[str, os.PathLike],
    user_id: str, lemma_id: str, verifier: str, when: str,
) -> Path:
    """The refutation file's path: dir_for(conjecture_dir) / name_for(...)."""
    return dir_for(conjecture_dir) / name_for(user_id, lemma_id, verifier, when)


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
    user_id: str, lemma_id: str, hash: str, verifier: str,
    model: str, verdict: str, justification: str,
    date: Optional[str] = None,
) -> Path:
    """Write one refutation file for a rejection of the lemma and return
    its path.

    The file is the rejection as it happened: the lemma by its owner and id,
    its Merkle hash at the moment of the rejection (the caller recomputed
    it, the way main.record_certificate recomputes a certificate's hash),
    the user whose run rejected it and the model under which the rejection
    happened, the moment, the verdict, and the model's justification.

    Two rejections of the same lemma by the same verifier in the same
    second are two different events and both are kept: the first takes
    the plain name, the second appends -2, the third -3, and so on.
    """
    when = date if date is not None else now_iso()
    obj = {
        "user_id": str(user_id),
        "lemma_id": str(lemma_id),
        "hash": str(hash),
        "verifier": str(verifier),
        "model": str(model),
        "date": when,
        "verdict": str(verdict),
        "justification": str(justification),
    }
    base = name_for(user_id, lemma_id, verifier, when)
    with _LOCK:
        directory = dir_for(conjecture_dir)
        os.makedirs(directory, exist_ok=True)
        path = directory / base
        n = 2
        while path.exists():
            path = directory / (base[: -len(".json")] + f"-{n}.json")
            n += 1
        tmp = str(path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(_dumps(obj))
        os.replace(tmp, path)
    return path


def load(path: Union[str, os.PathLike]) -> Dict[str, Any]:
    """The refutation of one file as a dict.

    Raises ValueError if the file is not a JSON object: a refutation file
    is small and a broken one is a file to look at, not a record to skip
    (callers that read many files, like for_lemma, do the skipping).
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a refutation file (not a JSON object)")
    return data


def for_lemma(
    conjecture_dir: Union[str, os.PathLike],
    user_id: str, lemma_id: str,
) -> List[Tuple[Path, Dict[str, Any]]]:
    """Every refutation of the lemma, as (path, record) pairs, oldest
    first.

    The directory is scanned, not the names parsed: a file counts as the
    lemma's when its user_id and lemma_id fields say so, so a
    hand-renamed file is still found and a file for another lemma is
    never touched. A file that cannot be read or parsed is skipped here,
    the way a broken certificate line is skipped in certificates.load:
    the callers ask for refutations and get refutations. The order is by
    date, then name, so the verifiers see the first objection first and
    two undated files still have a deterministic order.
    """
    directory = dir_for(conjecture_dir)
    if not directory.is_dir():
        return []
    out: List[Tuple[Path, Dict[str, Any]]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = load(path)
        except (OSError, ValueError):
            continue
        if (
            str(data.get("user_id") or "") == str(user_id)
            and str(data.get("lemma_id") or "") == str(lemma_id)
        ):
            out.append((path, data))
    out.sort(key=lambda pr: (str(pr[1].get("date") or ""), pr[0].name))
    return out


def is_counting(ref: Mapping[str, Any], current_hash: str) -> bool:
    """Whether a refutation counts: only while its hash matches the
    lemma's current Merkle hash.

    The same rule certificates.py checks for an acceptance, checked for
    the record of a rejection. current_hash is the hash the caller
    recomputed from the DAG and references.md as they stand now
    (merkle.Merkle over load_dag()'s lemmas and load_references()); the
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
