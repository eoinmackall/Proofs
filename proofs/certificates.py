"""Certificates: the record that a lemma was accepted.

Certificates are stored in the conjecture's certificates/ directory, one
JSONL file per user, named for the user who issued them — the verifier:

    conjectures/NAME/certificates/<verifier>.jsonl

The file's first line is the owner's identity — the config's user.id,
user.name and user.email, one JSON object, the same keys config.py holds —
the per-user files' header, the way the user's DAG file carries it at the
top of its object. It is written and refreshed by the owner's own runs — the
only runs that write the file — and every rewrite carries it through at the
top. It is metadata about the file, not a certificate: load() does not
return it, and a line is the header by carrying user.id — a key a
certificate line never has, its identity field being user_id — with none of
a certificate's key fields.

Below the header, one line per certificate, one JSON object per line,
fields in this order:

    {
      "user_id":  "<the lemma's owner>",
      "lemma_id": "<the lemma's id within that user's DAG file>",
      "hash":     "<the lemma's Merkle hash when it was accepted>",
      "verifier": "<the user_id of the person who verified it>",
      "model":    "<the verifier/model's name, or empty for a human>",
      "date":     "<ISO 8601 UTC time of the latest acceptance>",
      "count":    <how many times this exact certificate has been issued>
    }

user_id and lemma_id identify the lemma (a bare lemma_id is unique only
within one user's file), the verifier is the user_id of the person who
performed the verification, and model is the name of the verifier model
that did it (an empty model is a human acceptance, which older versions of
the menu could record; none is written now). A user
writes only their own certificate file: the file is named for the verifier,
the user whose run issued the certificate, never for the lemma's owner, so
a certificate for another user's lemma still lands in the verifier's own
file. No two users ever write the same file, and git has no per-line
merging to do. The certificates for one lemma are therefore spread over
every verifier's file; load_all() reads them together (suspension, proofs
status).

A line is identified by (user_id, lemma_id, hash, verifier, model): issuing
a certificate whose five fields match an existing line increments that
line's count and updates its date to the latest run instead of appending a
duplicate, while a different hash — a fresh proof of the same lemma, or the
same proof accepted by another verifier or model — is a different
certificate and gets its own line.

Validity is not stored, it is checked: a certificate is valid only if its
hash matches the lemma's current Merkle hash (see merkle.py), recomputed
from the DAG and references.md as they stand now. The hash covers what
the verifiers were shown — the lemma's statement and proof and the
statements of everything it cites — so a change to the lemma, or to the
statement of a result it cites, breaks the match, and the certificate is
known not to cover the proof any more. A new proof of a cited lemma under
the same statement does not: whether the ground is sound is suspension's
question, not the certificate's (see suspension.py).
This module records and compares hashes; computing the "current" one is the
caller's job (merkle.Merkle over the DAG and the references collection).

record() and prune() are the only writers, and both are read-modify-
writes of the user's file rewritten whole: record() updates or appends
one line, prune() drops the lines that no longer match. The rewrite is
atomic in the same sense as the DAG writes (temp file beside it,
os.replace) so a reader or git never sees a half-written file, and each
read-modify-write runs under a process-wide lock because a parallel run
has several loops accepting lemmas at once. A line that is not a JSON
object is kept verbatim on the rewrite: the file is a committed shared
record, and a write must never drop what it did not write.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

try:  # inside the proofs package (installed, or `python -m proofs`)
    from . import locking, merkle
except ImportError:  # top-level modules (`python proofs/certificates.py`)
    import locking
    import merkle

# The directory beside dags/ that holds the certificate files.
DIR_NAME = "certificates"

# The field order of a line in the file: identity first (user_id,
# lemma_id, hash), then who and what accepted (verifier, model), then the
# record of it (date, count).
FIELDS = ("user_id", "lemma_id", "hash", "verifier", "model", "date", "count")

# The five fields that identify a line. Same lemma, same hash, same
# verifier, same model is the same certificate issued again: it updates
# the existing line (count += 1, date refreshed) rather than appending.
KEY_FIELDS = ("user_id", "lemma_id", "hash", "verifier", "model")

# The keys of the file's header line: the owner's identity, named the way
# config.py names them (its USER_ID_KEY and company, config.py's file the
# record of the same identity), with the empty string where the config
# holds none. A certificate line never carries them — its identity field
# is user_id, an underscore — which is what lets is_header tell the two
# apart.
USER_ID_KEY = "user.id"
USER_NAME_KEY = "user.name"
USER_EMAIL_KEY = "user.email"
USER_KEYS = (USER_ID_KEY, USER_NAME_KEY, USER_EMAIL_KEY)

# One lock for the whole process: in a parallel run several loops accept
# lemmas at the same moment, and the read-modify-write of one file must be
# atomic within this process. Across processes on one machine (proofs run
# and proofs verify of the same user) the file lock in locking.py does the
# same job; across machines the files are per-user and coordinated by git,
# the same way the DAG files are.
_LOCK = threading.Lock()


def dir_for(conjecture_dir: Union[str, os.PathLike]) -> Path:
    """The conjecture's certificates/ directory (created on first write)."""
    return Path(conjecture_dir) / DIR_NAME


def file_for(
    conjecture_dir: Union[str, os.PathLike], verifier: str
) -> Path:
    """verifier's certificate file: certificates/<verifier>.jsonl.

    The file holds the certificates that user issued, whoever owns the
    lemmas they cover — which is why it is addressed by the certificate's
    verifier field rather than its user_id."""
    return dir_for(conjecture_dir) / (str(verifier) + ".jsonl")


def header(
    user_id: str, user_name: str = "", user_email: str = ""
) -> Dict[str, str]:
    """The file's header line as a mapping: the owner's identity, the
    config's user.id, user.name and user.email (config.py's keys), with
    the empty string where the config holds none."""
    return {
        USER_ID_KEY: str(user_id),
        USER_NAME_KEY: str(user_name),
        USER_EMAIL_KEY: str(user_email),
    }


def is_header(obj: Any) -> bool:
    """True of the file's header line: a JSON object that carries user.id
    and none of a certificate's key fields. The two shapes cannot be
    confused: a certificate line never carries user.id — its identity
    field is user_id — and a line that did carry it while still holding a
    certificate's key fields is a hand-edited certificate, not a header.
    """
    return (
        isinstance(obj, Mapping)
        and USER_ID_KEY in obj
        and not any(k in obj for k in KEY_FIELDS)
    )


def key_of(cert: Any) -> Optional[Tuple[Any, ...]]:
    """The (user_id, lemma_id, hash, verifier, model) of a certificate line,
    or None if it is not an object carrying the five key fields. Two lines
    with the same key are the same certificate: the second is an update of
    the first, not a new line."""
    if not isinstance(cert, Mapping):
        return None
    try:
        return tuple(cert[f] for f in KEY_FIELDS)
    except (KeyError, TypeError):
        return None


def load(path: Union[str, os.PathLike]) -> List[Dict[str, Any]]:
    """All the certificate lines of path, in file order.

    A line that is not a JSON object is skipped: the readers (prune,
    status, a git pull) ask for certificates and get certificates, and
    neither does the header line — the owner's identity, the file's first
    line, which is metadata about the file, not a certificate. record()
    does not skip — it keeps such a line verbatim: readers must not see
    junk, but a rewrite must not drop it. A missing file is simply empty:
    no file means no certificates yet."""
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f.read().splitlines():
            parsed = _parse_line(line)
            if parsed is not None and not is_header(parsed):
                out.append(parsed)
    return out


def load_all(conjecture_dir: Union[str, os.PathLike]) -> List[Dict[str, Any]]:
    """Every certificate line of every user's file, file by file in name
    order. A lemma's certificates are spread over the files of everyone
    who verified it, so a reader that asks about a lemma — rather than
    about one user's work — reads them all."""
    d = dir_for(conjecture_dir)
    if not d.is_dir():
        return []
    out: List[Dict[str, Any]] = []
    for path in sorted(d.glob("*.jsonl")):
        out.extend(load(path))
    return out


def is_valid(cert: Mapping[str, Any], current_hash: str) -> bool:
    """A certificate is valid only if its hash matches the lemma's current
    Merkle hash. current_hash is the hash the caller recomputed from the
    DAG and references.md as they stand now
    (merkle.Merkle over load_dag()'s lemmas and load_references()): the
    certificate's own hash is frozen at the moment of acceptance, so the
    comparison is against a fresh computation, never against itself."""
    return cert.get("hash") == current_hash


def now_iso() -> str:
    """The time a certificate is dated: the run's moment, ISO 8601 UTC to
    the second. Second precision is the granularity that matters — a
    certificate is re-issued by runs, not by heartbeats — and it keeps
    the file's diff small."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def record(
    conjecture_dir: Union[str, os.PathLike],
    user_id: str,
    lemma_id: str,
    hash: str,
    verifier: str,
    model: str,
    date: Optional[str] = None,
    user: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Record one certificate in the verifier's file, or update it.

    The upsert rule (see the module docstring): a line with the same
    (user_id, lemma_id, hash, verifier, model) already there gets its count
    incremented and its date updated to this run's, and no duplicate line is
    appended; otherwise a new line with count 1 is appended. The file is
    rewritten whole and atomically (temp file beside it, os.replace).
    Returns the line as written: the existing line with its new count and
    date, or the new one.

    user is the verifier's identity (header()'s mapping) as the caller's
    config holds it: the file's header line, written and refreshed on every
    write, since the verifier's own run is the only one that writes the
    file. When omitted, the header it finds is carried through as is, and
    no header is invented for one the file does not have."""
    path = file_for(conjecture_dir, verifier)
    when = date if date is not None else now_iso()
    key = (str(user_id), str(lemma_id), str(hash), str(verifier), str(model))
    new_line: Dict[str, Any] = {
        "user_id": str(user_id),
        "lemma_id": str(lemma_id),
        "hash": str(hash),
        "verifier": str(verifier),
        "model": str(model),
        "date": when,
        "count": 1,
    }
    with _LOCK, locking.file_lock(path):
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        replaced = False
        header_line: Optional[str] = None
        out: List[str] = []
        for line in text.splitlines():
            parsed = _parse_line(line)
            if parsed is not None and is_header(parsed):
                # The file's header line: carried through the rewrite, at
                # the top, stamped from the caller's config when the
                # owner's own run passed it. A second header line (a hand
                # edit) is not a certificate and goes.
                if header_line is None:
                    header_line = (
                        _dumps_header(user) if user is not None else line
                    )
                continue
            if not replaced and parsed is not None and key_of(parsed) == key:
                # The same certificate issued again: count it, refresh the
                # date, keep the line where it is. A count a hand edit left
                # un-integer is reset, not crashed on: the file is a shared
                # committed record, and a rewrite must survive its contents.
                try:
                    count = int(parsed.get("count") or 1)
                except (TypeError, ValueError):
                    count = 1
                parsed["count"] = count + 1
                parsed["date"] = when
                out.append(_dumps(parsed))
                new_line = parsed
                replaced = True
            else:
                out.append(line)
        if user is not None and header_line is None:
            header_line = _dumps_header(user)
        if not replaced:
            out.append(_dumps(new_line))
        if header_line is not None:
            out.insert(0, header_line)
        os.makedirs(path.parent, exist_ok=True)
        tmp = str(path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\n".join(out) + "\n")
        os.replace(tmp, path)
    return new_line


def prune(
    conjecture_dir: Union[str, os.PathLike],
    user_id: str,
    hashes: Mapping[Tuple[str, str], Optional[str]],
    user: Optional[Mapping[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Drop the user's certificates that no longer match: the file side
    of proofs prune (main.prune_entry()).

    hashes is the conjecture's lemmas with the hash each has now, keyed
    by the (user_id, lemma_id) pair: merkle.Merkle over the complete
    DAG and references.md, the same computation every user who loads the
    same files makes, with a pair the DAG holds but cannot hash (a
    citation cycle) present as None. A line is kept only if its pair has
    a hash and the line's hash field equals it — a certificate is valid
    only while it matches the lemma's current hash, so a line for a lemma
    that is no longer in the DAG, or that no longer has a hash, matches
    nothing now and goes.

    Only the certificates user_id issued are pruned: the user's own file,
    certificates/<user_id>.jsonl, is the only one read and rewritten, so
    the other users' files are never touched, and within it a line whose
    verifier is someone else (a hand edit) is kept verbatim rather than
    judged. The file is left alone entirely when every line still
    matches and the header it carries is still the one the caller's
    config holds. The rewrite is atomic in record()'s sense (temp file
    beside it, os.replace), runs under the same process lock, and keeps a
    line that is not a JSON object verbatim, record()'s rule for a shared
    committed record. Returns (the lines kept, the lines dropped), each
    as parsed, so the caller can say why each dropped line went.

    user is the owner's identity (header()'s mapping) as the caller's
    config holds it, passed by the owner's own prune: the file's header
    line is never pruned — it names no lemma — and a rewrite refreshes it
    from the identity, which is what the rewrite condition above is about
    (a changed name or email is a header that no longer matches the
    config). When omitted — this prune never writes another user's file,
    so in practice only by a caller who holds no identity to stamp — the
    header it finds is carried through as is.
    """
    path = file_for(conjecture_dir, user_id)
    if not path.is_file():
        return [], []
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    raw_kept: List[str] = []
    with _LOCK, locking.file_lock(path):
        if not path.is_file():
            return [], []
        text = path.read_text(encoding="utf-8")
        header_on_disk: Optional[str] = None
        for line in text.splitlines():
            parsed = _parse_line(line)
            if parsed is None:
                raw_kept.append(line)
                continue
            if is_header(parsed):
                # The file's header line: the owner's identity, metadata
                # about the file, not a certificate — never dropped, kept
                # at the top. A second one (a hand edit) goes.
                if header_on_disk is None:
                    header_on_disk = line
                continue
            if str(parsed.get("verifier") or "") != str(user_id):
                raw_kept.append(line)
                continue
            pair = (
                str(parsed.get("user_id") or ""),
                str(parsed.get("lemma_id") or ""),
            )
            current = hashes.get(pair)
            if current is not None and parsed.get("hash") == current:
                kept.append(parsed)
                raw_kept.append(line)
            else:
                dropped.append(parsed)
        header_out = _dumps_header(user) if user is not None else header_on_disk
        if dropped or header_out != header_on_disk:
            os.makedirs(path.parent, exist_ok=True)
            tmp = str(path) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                if header_out is not None:
                    f.write(header_out + "\n")
                for l in raw_kept:
                    f.write(l + "\n")
            os.replace(tmp, path)
    return kept, dropped


def issue(
    conjecture_dir: Union[str, os.PathLike],
    lemmas: Mapping[Any, Mapping[str, Any]],
    references: Any,
    user_id: str,
    lemma_id: str,
    verifier: str,
    model: str,
    date: Optional[str] = None,
    user: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Compute the lemma's current Merkle hash and record a certificate
    for it. lemmas is the complete DAG's lemmas mapping as load_dag()
    returns it under "lemmas" — keyed by (user_id, lemma_id) pairs, so the
    hash is the one every user who loads the same DAG and references.md
    computes — and references is the references collection from
    references.md. Returns the line as written, or None if the DAG has no
    such lemma. A MerkleCycleError propagates: a hash that cannot be
    computed is a state the caller must report, not one to paper over."""
    m = merkle.Merkle(lemmas, references)
    if not m.has(user_id, lemma_id):
        return None
    return record(
        conjecture_dir,
        user_id,
        lemma_id,
        m.hash(user_id, lemma_id),
        verifier,
        model,
        date=date,
        user=user,
    )


def _parse_line(line: str) -> Optional[Dict[str, Any]]:
    """The line as a dict, or None: blank, or not a JSON object. Lines are
    judged one at a time because the file is JSONL — one certificate per
    line, one bad line must not void the rest."""
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _dumps_header(h: Mapping[str, Any]) -> str:
    """The header line: the known keys in USER_KEYS order, then any key a
    hand edit added — the rewrite's rule for a shared committed record,
    applied to the header's own fields."""
    ordered: Dict[str, Any] = {k: h[k] for k in USER_KEYS if k in h}
    for k in h:
        if k not in ordered:
            ordered[k] = h[k]
    return json.dumps(ordered, separators=(",", ":"), ensure_ascii=False)


def _dumps(cert: Mapping[str, Any]) -> str:
    """One line, no terminating newline (the caller joins and terminates):
    the known fields in FIELDS order first, then any field a hand edit
    added — a rewrite must not drop what it did not write."""
    ordered: Dict[str, Any] = {f: cert[f] for f in FIELDS if f in cert}
    for k in cert:
        if k not in ordered:
            ordered[k] = cert[k]
    return json.dumps(ordered, separators=(",", ":"), ensure_ascii=False)
