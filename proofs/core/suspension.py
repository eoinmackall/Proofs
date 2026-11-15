"""Suspension: the state in which a lemma is not to be built on.

A lemma is suspended if

  * it has a refutation whose hash matches the lemma — the Merkle hash the
    DAG and references.md have now, the way refutations.is_counting checks
    it: the proof in the file is one a verification has rejected;
  * it has no valid certificate — no line in any user's certificate file
    whose hash matches the lemma's current Merkle hash, the way
    certificates.is_valid checks it; or
  * its versions state different things — a fork no reader can decide
    (core/dag.py), until a person deletes all but one; or
  * it cites a suspended lemma — a proof standing on a lemma it must not
    stand on, transitively.

Suspension is never stored in the DAG: compute() derives it from the
in-memory DAG (core/dag.py) — the lemmas, their hashes, the certificates
and the refutations as the command loaded them, plus whatever it has
committed since — and a lemma is suspended exactly while those say so:
until a refutation is dismissed, or a fresh certificate is issued for the
proof that is in the file now.

Suspension is what keeps proofs run from building on a refuted or
unverified lemma: a suspended lemma is hidden from the planner and
the prover (the views and id sets that take the suspension do
the hiding), but proofs verify, proofs repair and proofs status still load
it — suspension hides a lemma from the construction, not from the
verification that is the only way back to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Set, Tuple,
)

from . import certificates, refutations

__all__ = ["Suspension", "compute"]

@dataclass(frozen=True)
class Suspension:
    """The suspension of a loaded DAG, with the reason for each lemma.

    Compute it with compute(), the only constructor this module offers,
    and hold it only as long as the DAG, references.md and the two file
    directories it was computed against stand as they stood: suspension is
    recomputed each time the DAG is loaded, and this describes the state
    it was computed for, not the state in between.
    """

    suspended: FrozenSet[str]
    """The ids of the lemmas that are not to be built on."""

    refuted: FrozenSet[str]
    """...of which the suspension carries a counting refutation — a
    refutation whose hash matches the lemma's current Merkle hash, the
    case the run warns about by name."""

    _reasons: Mapping[str, Tuple[str, ...]]
    """Why each lemma is suspended, one clause per rule that holds of it —
    a counting refutation, no valid certificate, no Merkle hash, and each
    suspended lemma it cites — so a warning can name the case, not just
    the state."""

    def __contains__(self, lemma_id: object) -> bool:
        return lemma_id in self.suspended

    def has(self, lemma_id: str) -> bool:
        """Whether the lemma is suspended."""
        return str(lemma_id) in self.suspended

    def reasons(self, lemma_id: str) -> List[str]:
        """Why the lemma is suspended, one clause per rule that holds of
        it — [] when it is not suspended at all."""
        return list(self._reasons.get(str(lemma_id), ()))


def _index(records: Iterable[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    """Certificate or refutation records, grouped by lemma id. A record for
    a version of the lemma that is no longer in the files drops out of
    compute() of its own accord — compared by hash, it simply stops
    counting."""
    index: Dict[str, List[Mapping[str, Any]]] = {}
    for rec in records:
        lemma_id = str(rec.get("lemma_id") or "").strip()
        if lemma_id:
            index.setdefault(lemma_id, []).append(rec)
    return index


def _own_reasons(
    lemma_id: str,
    h: Optional[str],
    certs: Iterable[Mapping[str, Any]],
    refus: Iterable[Mapping[str, Any]],
    conflicts: Optional[Mapping[str, List[str]]],
) -> Tuple[List[str], bool]:
    """Why the lemma is suspended on its own account — every rule but the
    transitive one — given its hash and its own certificates and
    refutations; and whether a refutation counts against it."""
    why: List[str] = []
    is_refuted = False
    if h is None:
        why.append(
            "no Merkle hash (a citation cycle the hash refuses to cut "
            "through), so no certificate is valid for it"
        )
    else:
        counting = [r for r in refus if refutations.is_counting(r, h)]
        if counting:
            is_refuted = True
            why.append(
                f"{len(counting)} refutation(s) match its hash — the "
                f"proof in the file is one a verification has rejected"
            )
        if not any(certificates.is_valid(c, h) for c in certs):
            why.append("no valid certificate")
    if conflicts and lemma_id in conflicts:
        why.append(
            "its versions state different things ("
            + ", ".join(conflicts[lemma_id]) + "); delete all but one"
        )
    return why, is_refuted


def with_lemma(
    susp: Suspension,
    lemma_id: str,
    node: Mapping[str, Any],
    h: Optional[str],
    certs: Iterable[Mapping[str, Any]],
    refus: Iterable[Mapping[str, Any]],
    conflicts: Optional[Mapping[str, List[str]]] = None,
) -> Suspension:
    """susp with a lemma just committed added: its own status, and nothing
    else, since nothing cites it yet. The same rules as compute(), for one
    lemma; the same object back when the lemma is not suspended."""
    why, is_refuted = _own_reasons(lemma_id, h, certs, refus, conflicts)
    if not why:
        why = [
            f"cites suspended {c}"
            for c in (node.get("cited_lemmas") or []) if c in susp.suspended
        ]
    if not why:
        return susp
    reasons = dict(susp._reasons)
    reasons[lemma_id] = why
    return Suspension(
        susp.suspended | {lemma_id},
        (susp.refuted | {lemma_id}) if is_refuted else susp.refuted,
        reasons,
    )


def compute(
    lemmas: Mapping[str, Mapping[str, Any]],
    hashes: Mapping[str, Optional[str]],
    certificate_lines: Iterable[Mapping[str, Any]],
    refutation_records: Iterable[Mapping[str, Any]],
    conflicts: Optional[Mapping[str, List[str]]] = None,
) -> Suspension:
    """The suspension of lemmas, from plain data: the DAG's lemmas, keyed by
    id with each node's cited_lemmas a list of ids; each
    lemma's current Merkle hash (None for a lemma of a citation cycle, which
    has none); every certificate line; every refutation record; and the
    lemmas whose versions state different things, with where those versions
    are. The in-memory DAG (core/dag.py, Dag.suspension) holds them all.
    """
    certificates_by_lemma = _index(certificate_lines)
    refutations_by_lemma = _index(refutation_records)

    refuted: Set[str] = set()
    reasons: Dict[str, List[str]] = {}
    for lemma_id in lemmas:
        why, is_refuted = _own_reasons(
            lemma_id, hashes.get(lemma_id),
            certificates_by_lemma.get(lemma_id, []),
            refutations_by_lemma.get(lemma_id, []), conflicts,
        )
        if is_refuted:
            refuted.add(lemma_id)
        if why:
            reasons[lemma_id] = why
    # The last rule is transitive: a lemma that cites a suspended lemma is
    # suspended itself, and so is one citing that one. Spread it once along
    # who-cites-whom, so each citation is followed once.
    citers: Dict[str, List[str]] = {}
    for lemma_id, node in lemmas.items():
        for c in node.get("cited_lemmas") or []:
            citers.setdefault(c, []).append(lemma_id)
    suspended: Set[str] = set(reasons)
    queue = list(suspended)
    while queue:
        for lemma_id in citers.get(queue.pop(), ()):
            if lemma_id not in suspended:
                suspended.add(lemma_id)
                queue.append(lemma_id)
    for lemma_id, node in lemmas.items():
        if lemma_id in suspended and lemma_id not in reasons:
            reasons[lemma_id] = [
                f"cites suspended {c}"
                for c in (node.get("cited_lemmas") or []) if c in suspended
            ]
    return Suspension(frozenset(suspended), frozenset(refuted), reasons)
