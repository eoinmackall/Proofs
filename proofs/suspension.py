"""Suspension: the state in which a lemma is not to be built on.

A lemma is suspended if

  * it has a refutation whose hash matches the lemma — the Merkle hash the
    DAG and references.md have now, the way refutations.is_counting checks
    it: the proof in the file is one a verification has rejected;
  * it has no valid certificate — no line in any user's certificate file
    whose hash matches the lemma's current Merkle hash, the way
    certificates.is_valid checks it; or
  * it cites a suspended lemma — a proof standing on a lemma it must not
    stand on, transitively.

Suspension is never stored in the DAG: compute() derives it from the DAG,
references.md, the certificates/ and refutations/ files each time the DAG
is loaded, and a lemma is suspended exactly while those files say so —
until a refutation is dismissed, or a fresh certificate is issued for the
proof that is in the file now.

Suspension is what keeps proofs run from building on a refuted or
unverified lemma: a suspended lemma is hidden from the planner, the
selector and the prover (the views and id sets that take the suspension do
the hiding), but proofs verify, proofs repair and proofs status still load
it — suspension hides a lemma from the construction, not from the
verification that is the only way back to it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

try:  # package form: `python -m proofs`, `pip install -e .`
    from . import certificates, merkle, refutations
except ImportError:  # running the package directory as a script
    import certificates
    import merkle
    import refutations

__all__ = ["Suspension", "compute"]

# (user_id, lemma_id): a lemma's full identity in the loaded DAG. A bare
# lemma_id is unique only within one user's file.
Pair = Tuple[str, str]


@dataclass(frozen=True)
class Suspension:
    """The suspension of a loaded DAG, with the reason for each lemma.

    Compute it with compute(), the only constructor this module offers,
    and hold it only as long as the DAG, references.md and the two file
    directories it was computed against stand as they stood: suspension is
    recomputed each time the DAG is loaded, and this describes the state
    it was computed for, not the state in between.
    """

    suspended: FrozenSet[Pair]
    """The (user_id, lemma_id) pairs that are not to be built on."""

    refuted: FrozenSet[Pair]
    """...of which the suspension carries a counting refutation — a
    refutation whose hash matches the lemma's current Merkle hash, the
    case the run warns about by name."""

    _reasons: Mapping[Pair, Tuple[str, ...]]
    """Why each pair is suspended, one clause per rule that holds of it —
    a counting refutation, no valid certificate, no Merkle hash, and each
    suspended lemma it cites — so a warning can name the case, not just
    the state."""

    def __contains__(self, pair: object) -> bool:
        p = _pair_of(pair)
        return p is not None and p in self.suspended

    def has(self, user_id: str, lemma_id: str) -> bool:
        """Whether (user_id, lemma_id) is suspended."""
        return (str(user_id), str(lemma_id)) in self.suspended

    def reasons(self, user_id: str, lemma_id: str) -> List[str]:
        """Why (user_id, lemma_id) is suspended, one clause per rule that
        holds of it — [] when it is not suspended at all."""
        return list(self._reasons.get((str(user_id), str(lemma_id)), ()))


def _pair_of(rec: object) -> Optional[Pair]:
    """The (user_id, lemma_id) a record names, or None when it names no
    lemma: a pair as given, or a certificate line or refutation record
    missing its user_id or lemma_id, which names nothing to suspend."""
    if isinstance(rec, (tuple, list)):
        if len(rec) != 2:
            return None
        user_id, lemma_id = rec
    else:
        user_id = rec.get("user_id")  # type: ignore[union-attr]
        lemma_id = rec.get("lemma_id")  # type: ignore[union-attr]
    user_id = str(user_id or "").strip()
    lemma_id = str(lemma_id or "").strip()
    if not user_id or not lemma_id:
        return None
    return (user_id, lemma_id)


def _label(pair: Pair) -> str:
    """The owner-qualified form of a pair, for a warning: user_id:lemma_id."""
    return f"{pair[0]}:{pair[1]}"


def _certificate_index(
    conjecture_dir: Union[str, os.PathLike]
) -> Dict[Pair, List[Mapping[str, Any]]]:
    """Every certificate line on disk, grouped by (user_id, lemma_id).

    Each line in each verifier's certificates/ file is the claim that,
    when it was written, the owner's proof of that lemma had been
    verified;
    compute() compares each line's hash with the lemma's current Merkle
    hash through certificates.is_valid, so a line for a version of the
    lemma that is no longer in the file drops out of the computation of
    its own accord — neither deleted nor consulted, it simply stops
    counting."""
    index: Dict[Pair, List[Mapping[str, Any]]] = {}
    for line in certificates.load_all(conjecture_dir):
        pair = _pair_of(line)
        if pair is None:
            continue
        index.setdefault(pair, []).append(line)
    return index


def _refutation_index(
    conjecture_dir: Union[str, os.PathLike]
) -> Dict[Pair, List[Mapping[str, Any]]]:
    """Every refutation file on disk, grouped by (user_id, lemma_id).

    A file the refutations module will not read — a torn checkout, a half
    written record — is skipped rather than raised on: a refutation that
    cannot be read cannot count, and suspension runs in the conservative
    direction (the lemma without a valid certificate), not a crash."""
    index: Dict[Pair, List[Mapping[str, Any]]] = {}
    d = refutations.dir_for(conjecture_dir)
    if d.is_dir():
        for path in sorted(d.glob("*.json")):
            try:
                rec = refutations.load(path)
            except (OSError, ValueError):
                continue
            pair = _pair_of(rec)
            if pair is None:
                continue
            index.setdefault(pair, []).append(rec)
    return index


def compute(
    lemmas: Mapping[Pair, Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
    conjecture_dir: Union[str, os.PathLike],
) -> Suspension:
    """The suspension of lemmas, as the files stand now.

    lemmas is the whole DAG's lemmas mapping as load_dag() hands it over —
    keyed by (user_id, lemma_id) pairs, each node's cited_lemmas the list
    of pairs the load migrated it to; references is the contents of
    references.md — the Merkle hashes are computed over lemmas and
    references, the same way the certificates' and refutations' hashes are
    stamped; conjecture_dir is the conjecture's root, the directory whose
    certificates/ and refutations/ subdirectories are read.
    """
    tree = merkle.Merkle(lemmas, references)
    hashes: Dict[Pair, Optional[str]] = {}
    for pair in lemmas:
        try:
            hashes[pair] = tree.hash(*pair)
        except merkle.MerkleCycleError:
            # No hash exists for a lemma of a citation cycle: no
            # certificate can be valid for it (there is no hash for a
            # line to match) and no refutation can count against it (none
            # to match). Both rules therefore point the same way, and the
            # warning says which one it is.
            hashes[pair] = None

    certificates_by_lemma = _certificate_index(conjecture_dir)
    refutations_by_lemma = _refutation_index(conjecture_dir)

    refuted: Set[Pair] = set()
    reasons: Dict[Pair, List[str]] = {}
    for pair in lemmas:
        why: List[str] = []
        h = hashes.get(pair)
        if h is None:
            why.append(
                "no Merkle hash (a citation cycle the hash refuses to cut "
                "through), so no certificate is valid for it"
            )
        else:
            counting = [
                r
                for r in refutations_by_lemma.get(pair, [])
                if refutations.is_counting(r, h)
            ]
            if counting:
                refuted.add(pair)
                why.append(
                    f"{len(counting)} refutation(s) match its hash — the "
                    f"proof in the file is one a verification has rejected"
                )
            if not any(
                certificates.is_valid(c, h)
                for c in certificates_by_lemma.get(pair, [])
            ):
                why.append("no valid certificate")
        if why:
            reasons[pair] = why

    # The third rule is transitive, so take the fixpoint: a lemma that
    # cites a suspended lemma is suspended itself, and one that cites that
    # one with it, until a pass adds nothing new.
    suspended: Set[Pair] = set(reasons)
    while True:
        newly = False
        for pair, node in lemmas.items():
            if pair in suspended:
                continue
            cites = [
                c
                for c in (node.get("cited_lemmas") or [])
                if isinstance(c, tuple) and len(c) == 2 and c in suspended
            ]
            if cites:
                suspended.add(pair)
                reasons.setdefault(pair, []).extend(
                    f"cites suspended {_label(c)}" for c in cites
                )
                newly = True
        if not newly:
            break
    return Suspension(frozenset(suspended), frozenset(refuted), reasons)
