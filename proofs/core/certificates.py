"""Certificates: the record that a lemma was accepted.

A certificate is one record in the verifier's run stream (streams.py):

    conjectures/NAME/certificates/<verifier>/<run_id>/000001.jsonl

    {"type": "certificate",
     "lemma_id": "<the lemma's id>",
     "hash":     "<the lemma's Merkle hash when it was accepted>",
     "verifier": "<the user.id of the user whose run accepted it>",
     "model":    "<the model that did the verification>",
     "date":     "<ISO 8601 UTC time of the acceptance>"}

Each acceptance is a record of its own, appended and never changed by the run
that wrote it: the same lemma accepted again under the same hash by the same
verifier and model is a second record. proofs prune merges such duplicates to
shrink the files: the first (the earliest) is kept with a "count" field, the
number of acceptances it stands for, and the rest are deleted. A record
without "count" stands for one. How many times a certificate was issued is
summed in the loaded DAG for its current hash, and by tally() for proofs
status. A citation recheck also records a check_id for crash recovery.
A certificate always goes into the
stream of the run that issued it, whoever owns the lemma, so no two users —
no two runs — ever write the same file.

Validity is not stored, it is checked: a certificate is valid only while its
hash matches the lemma's current Merkle hash (merkle.py), recomputed from
the DAG and references.md as they stand. The hash covers what the verifiers
were shown — the lemma's statement and proof and the statements of
everything it cites — so a change to any of those breaks the match, and the
certificate is known not to cover the proof any more. A certificate that
matches nothing is harmless until proofs prune deletes it: prune keeps a
certificate only while its hash is its lemma's current hash.
"""

from __future__ import annotations

import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

from . import streams

# The directory beside dags/ that holds the certificate streams.
DIR_NAME = "certificates"
TYPE = "certificate"
FIELDS = ("lemma_id", "hash", "verifier", "model", "date")


def dir_for(conjecture_dir: Union[str, os.PathLike]) -> Path:
    return Path(conjecture_dir) / DIR_NAME


def now_iso() -> str:
    """The current UTC time in the format certificates record it."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def record(writer: "streams.Writer", lemma_id: str, lemma_hash: str,
           verifier: str, model: str, *, check_id: Optional[str] = None) -> Dict[str, Any]:
    """Append a certificate to the run's certificate stream; returns it."""
    cert = {
        "type": TYPE,
        "lemma_id": lemma_id,
        "hash": lemma_hash,
        "verifier": verifier,
        "model": model or "",
        "date": now_iso(),
    }
    if check_id is not None:
        cert["check_id"] = check_id
    writer.append(cert)
    return cert


def load_all(conjecture_dir: Union[str, os.PathLike]) -> List[Dict[str, Any]]:
    """Every certificate of every verifier's runs."""
    return [
        rec for _u, _r, _p, _n, rec in streams.records(dir_for(conjecture_dir))
        if rec.get("type") == TYPE
    ]


def count(cert: Mapping[str, Any]) -> int:
    """How many acceptances a record stands for: its "count" (proofs
    prune's merge of duplicates), 1 when it has none or a bad one."""
    n = cert.get("count")
    return n if isinstance(n, int) and not isinstance(n, bool) and n >= 1 else 1


def key(cert: Mapping[str, Any]) -> tuple:
    """What makes two certificate records duplicates: the same lemma,
    hash, verifier and model."""
    return (str(cert.get("lemma_id") or ""), str(cert.get("hash") or ""),
            str(cert.get("verifier") or ""), str(cert.get("model") or ""))


def is_valid(cert: Mapping[str, Any], current_hash: str) -> bool:
    """Whether a certificate covers the proof whose hash is current_hash."""
    return bool(current_hash) and cert.get("hash") == current_hash


def tally(lines: Iterable[Mapping[str, Any]], lemma_id: str) -> List[Dict[str, Any]]:
    """A lemma's certificates, one entry per distinct (hash, verifier,
    model): how many times it was issued and when last — what proofs status
    shows."""
    counts: Counter = Counter()
    last: Dict[tuple, str] = {}
    for c in lines:
        if c.get("lemma_id") != lemma_id:
            continue
        k = key(c)[1:]
        counts[k] += count(c)
        last[k] = max(last.get(k, ""), str(c.get("date") or ""))
    return [
        {"hash": h, "verifier": v, "model": m, "count": n, "date": last[(h, v, m)]}
        for (h, v, m), n in sorted(counts.items(), key=lambda kv: last[kv[0]])
    ]
