"""Merkle hashes of the DAG's lemmas.

A lemma's hash is the SHA-256 of a canonical JSON object holding the
lemma's normalized statement, its normalized proof, and the sorted
statement hashes of everything the lemma cites:

    {
        "cited":     [<hash>, <hash>, ...],
        "proof":     "<normalized proof>",
        "statement": "<normalized statement>"
    }

written with sorted keys and no whitespace (canonical_json), UTF-8
encoded, and hashed. A cited lemma contributes the SHA-256 of its
normalized statement, and a cited reference the SHA-256 of its
normalized formal statement — the same function for both
(statement_hash). The ids, the slogans and a cited lemma's proof are
deliberately out, because a citation pins the result, not its label or
how it was established: a maintainer who rewords a slogan, or a repair
that re-proves a lemma under the same statement, must not invalidate the
proofs that cite it.

The hash therefore covers exactly what a verifier is shown: the lemma's
own statement and proof and the statements of what it cites. A new
statement for a lemma changes the hash of every lemma that cites it
directly, and those must be verified again; a new proof under the same
statement changes only the lemma's own hash. Whether the ground a lemma
stands on is sound is not the hash's question but suspension's
(suspension.py): a lemma that cites a lemma without a valid certificate,
or with a counting refutation, is suspended, transitively, all the way to
the top of the DAG.

No hash is stored in any DAG file: it is a function of the files'
contents, recomputed on demand. That is also what makes it shared —
every user who loads the same DAG files and the same references.md
computes the same hashes, with no exchange of state, which is how a
certificate or refutation written by one user stays checkable by every
other.

"Normalized" is a fixed whitespace canonical form (normalize()): no
surrounding whitespace, line endings to \n, no trailing line whitespace,
no blank-line runs. Text that differs only in those cosmetics hashes
alike, so reformatting a proof is not a new proof; a change to any
word, symbol or line break is.

Hashes are computed on demand, in dependency order, with each result
cached: Merkle.hash() recurses into a lemma's citations before hashing
the lemma itself, and every hash it computes is kept in Merkle.cache,
so the whole DAG costs one pass per lemma no matter how often the
results are asked for afterwards, and in what order. The recursion is no
longer needed for the value — a cited lemma contributes only its
statement — but it is kept for what it rules out: a DAG with a citation
cycle says so with a MerkleCycleError, and a lemma in a cycle has no
hash, so no certificate can be valid for it. Without that, two lemmas
each certified on the other's statement would be circular reasoning that
nothing flags.

This module is deliberately standalone: it imports nothing from this
package, so everything that keys off these hashes — certificates.py,
refutations.py, suspension.py and resolution.py, and the prune, verify,
status and repair commands — can build a Merkle without the run loop.
"""

from __future__ import annotations

import hashlib
import json
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
)

__all__ = [
    "Merkle",
    "MerkleCycleError",
    "canonical_json",
    "normalize",
    "reference_hash",
    "sha256_hex",
    "statement_hash",
]

# A lemma is identified by the pair (user_id, lemma_id); a bare lemma_id
# is unique only within one user's file, so on its own it names no
# particular lemma (see workspace.py).
LemmaKey = Tuple[str, str]


class MerkleCycleError(Exception):
    """A citation cycle: these lemmas' hashes depend on each other, so
    none of them can be computed in dependency order. The DAG is corrupt
    (or hand-edited into a loop); nothing is hashed rather than a hash
    being computed from a cycle."""


# ----------------------------------------------------------------------------
# Canonical forms
# ----------------------------------------------------------------------------
def normalize(text: Any) -> str:
    """The canonical whitespace form of a statement or proof.

    The surrounding whitespace goes, line endings become \n, trailing
    whitespace is dropped from every line, and runs of blank lines
    collapse to a single blank line. Two texts that differ only in those
    cosmetics normalize to the same string, and so hash alike; any other
    difference survives into the hash.

    A non-string reads as itself: None becomes "", anything else
    str() — the hash has to exist for whatever the file holds, and a
    field the model smuggled in as a number is still content.
    """
    if text is None:
        text = ""
    if not isinstance(text, str):
        text = str(text)
    lines = [
        line.rstrip()
        for line in text.replace("\r\n", "\n").replace("\r", "\n").strip().split("\n")
    ]
    out: List[str] = []
    for line in lines:
        if line == "" and out and out[-1] == "":
            continue
        out.append(line)
    return "\n".join(out)


def canonical_json(obj: Any) -> str:
    """The canonical JSON form of a value: sorted keys, compact
    separators, non-ASCII kept as UTF-8. A hash is only as comparable
    across users as this form is deterministic, and json.dumps is
    deterministic exactly in this shape."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def sha256_hex(data: bytes) -> str:
    """The SHA-256 of data, as the lowercase hex string every file and
    log line in this project spells a hash in."""
    return hashlib.sha256(data).hexdigest()


def statement_hash(statement: Any) -> str:
    """A cited result's contribution to a lemma's hash: the SHA-256 of
    its normalized statement. The same for a cited lemma (its
    "statement") and a cited reference (its "formal statement"), since a
    verifier is shown the statement and nothing else of either."""
    return sha256_hex(normalize(statement).encode("utf-8"))


def reference_hash(reference: Mapping[str, Any]) -> str:
    """A cited reference's contribution to a lemma's hash: the
    statement_hash of its "formal statement". The id
    and the slogan are deliberately not in it: a citation pins the
    result, not its label."""
    return statement_hash(reference.get("formal statement", ""))


def _unresolved(kind: str, *parts: str) -> str:
    """The contribution of a citation that resolves to nothing: the
    SHA-256 of a canonical marker carrying the citation's identity.

    A pair no file owns, or an id no reference has, still has to
    contribute something: the hash is a function of the files' contents,
    and every user must compute the same one. The marker keeps that —
    it is deterministic, differs from every other citation's marker, and
    changes when the citation is fixed, renamed or dropped, so the
    lemma's hash moves with it."""
    marker = {"unresolved_citation": {"kind": kind, "parts": list(parts)}}
    return sha256_hex(canonical_json(marker).encode("utf-8"))


def _citation_pair(entry: Any) -> Optional[LemmaKey]:
    """The (user_id, lemma_id) pair an entry names, or None.

    The shapes a lemma citation takes across the files: the in-memory
    tuple, a two-item list, and the stored {"user_id", "lemma_id"}
    object (the way cited_lemmas is written in a DAG file, and the way
    the prover's JSON spells it). A bare id is not a pair: a lemma_id
    is unique only within one user's file, so on its own it names no
    particular lemma."""
    if isinstance(entry, Mapping):
        uid = entry.get("user_id")
        lid = entry.get("lemma_id")
        if isinstance(uid, str) and isinstance(lid, str) and uid and lid:
            return (uid.strip(), lid.strip())
    elif isinstance(entry, (tuple, list)) and len(entry) == 2:
        uid, lid = entry
        if isinstance(uid, str) and isinstance(lid, str) and uid and lid:
            return (uid.strip(), lid.strip())
    return None


# ----------------------------------------------------------------------------
# The Merkle hashes of one complete DAG
# ----------------------------------------------------------------------------
class Merkle:
    """The on-demand, cached Merkle hashes of one complete DAG.

    Built from the complete DAG's lemmas — a mapping keyed by the
    (user_id, lemma_id) pair, the shape load_dag() in main.py returns
    under "lemmas" — and the reference collection (the objects of
    references.md, each with its "id" and "formal statement"). A Merkle
    is a snapshot of that data, not a live view: build one per load, the
    way the loop builds a DAG per iteration.

    hash() asks for one lemma's hash and computes it in dependency
    order: every hash it cites is computed first, each kept in cache as
    it is made, so asking for the hash of the top of the DAG computes
    each cited lemma exactly once, and every later ask is a cache hit.
    all() forces the whole DAG; the order it asks in is irrelevant, for
    the same reason.
    """

    def __init__(
        self,
        lemmas: Mapping[Any, Mapping[str, Any]],
        references: Optional[Iterable[Mapping[str, Any]]] = None,
    ) -> None:
        self._lemmas: Dict[LemmaKey, Mapping[str, Any]] = {}
        for key, node in lemmas.items():
            if not isinstance(node, Mapping):
                raise ValueError(
                    f"lemma {key!r} is not a JSON object; a lemma node "
                    f"holds statement, proof and its citations"
                )
            self._lemmas[_lemma_key(key)] = node

        # lemma_id -> the users' files that own it: the resolution of a
        # bare id in the legacy citation fields, the way load_dag() reads
        # it (a bare id exactly one lemma in the complete DAG owns is
        # that lemma's pair, the legacy spelling of a lemma citation).
        self._owners: Dict[str, List[LemmaKey]] = {}
        for key in self._lemmas:
            self._owners.setdefault(key[1], []).append(key)

        self._ref_hashes: Dict[str, str] = {}
        for ref in references or []:
            if not isinstance(ref, Mapping):
                continue
            rid = str(ref.get("id") or "").strip()
            if rid:
                self._ref_hashes[rid] = reference_hash(ref)

        # lemma hash, keyed by pair, cached as each is computed. Public:
        # "each result cached" is part of the contract (see the module
        # docstring), and a caller that asks the same lemma twice should
        # be able to see that it was computed once.
        self.cache: Dict[LemmaKey, str] = {}
        # The hash() chain currently being computed: a second entry for a
        # pair on it is a citation cycle.
        self._stack: List[LemmaKey] = []

    # -- asking ------------------------------------------------------------
    def hash(self, user_id: str, lemma_id: str) -> str:
        """The Merkle hash of the lemma the pair names.

        Computed on demand, in dependency order: the hashes of everything
        the lemma cites are computed (and cached) before the lemma's
        own is — not for their value, since a cited lemma contributes
        only its statement, but so that a citation cycle anywhere below
        the lemma is found. A second ask for the same pair returns the
        cached result without recomputing anything.

        Raises KeyError for a pair the DAG does not hold, and
        MerkleCycleError when the citations run in a circle.
        """
        pair = (str(user_id), str(lemma_id))
        cached = self.cache.get(pair)
        if cached is not None:
            return cached
        if pair not in self._lemmas:
            raise KeyError(
                f"({user_id!r}, {lemma_id!r}) is not a lemma of this DAG"
            )
        if pair in self._stack:
            cycle = self._stack[self._stack.index(pair):] + [pair]
            raise MerkleCycleError(
                "citation cycle: "
                + " -> ".join(f"{u}:{l}" for u, l in cycle)
            )
        self._stack.append(pair)
        try:
            digest = sha256_hex(
                canonical_json(
                    {
                        "statement": normalize(
                            self._lemmas[pair].get("statement")
                        ),
                        "proof": normalize(self._lemmas[pair].get("proof")),
                        "cited": self._cited_hashes(self._lemmas[pair]),
                    }
                ).encode("utf-8")
            )
        finally:
            self._stack.pop()
        self.cache[pair] = digest
        return digest

    def all(self) -> Dict[LemmaKey, str]:
        """Every lemma's hash, in one call.

        The cache makes the asking order irrelevant — every lemma is
        computed once, its citations first — so this is the hash of the
        complete DAG in the shape prune, status and the certificates
        consume it: a map from the (user_id, lemma_id) pair to its
        hash."""
        return {pair: self.hash(*pair) for pair in self._lemmas}

    def reference_hash(self, ref_id: str) -> Optional[str]:
        """A cited reference's contribution to a lemma's hash: the SHA-256
        of its normalized formal statement, or None for an id the
        reference collection does not hold."""
        return self._ref_hashes.get(str(ref_id).strip())

    def has(self, user_id: str, lemma_id: str) -> bool:
        """Whether the DAG holds the lemma the pair names."""
        return (str(user_id), str(lemma_id)) in self._lemmas

    # -- internals ---------------------------------------------------------
    def _citations(
        self, node: Mapping[str, Any]
    ) -> Tuple[List[LemmaKey], List[str]]:
        """A node's citations, in the two stored fields' shapes: the cited
        lemma pairs and the cited reference ids.

        Reads cited_lemmas and cited_references, plus the legacy mixed
        "dependencies" list (pair objects and bare reference ids in one
        field) the older files still carry — resolved exactly the way
        load_dag() migrates it, so a Merkle built over raw file data
        hashes a file the same way the run loop hashes its in-memory
        view of it: a pair object is a lemma citation, a bare id in
        cited_lemmas or dependencies that exactly one lemma in the
        complete DAG owns is read as that lemma's pair (the legacy
        spelling of a lemma citation), and whatever is left is a
        reference citation. Duplicates collapse; the order is the
        file's, and the hashing sorts anyway."""
        pairs: List[LemmaKey] = []
        refs: List[str] = []
        seen_pairs: Set[LemmaKey] = set()
        seen_refs: Set[str] = set()
        for field in ("cited_lemmas", "dependencies", "cited_references"):
            entries = node.get(field)
            if not isinstance(entries, list):
                continue
            for entry in entries:
                pair = _citation_pair(entry)
                if pair is not None:
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        pairs.append(pair)
                    continue
                if isinstance(entry, str):
                    rid = entry.strip()
                    if not rid:
                        continue
                    if (
                        field in ("cited_lemmas", "dependencies")
                        and len(self._owners.get(rid, [])) == 1
                    ):
                        pair = self._owners[rid][0]
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            pairs.append(pair)
                    elif rid not in seen_refs:
                        seen_refs.add(rid)
                        refs.append(rid)
                # anything else (a number, null) names nothing: dropped
        return pairs, refs

    def _cited_hashes(self, node: Mapping[str, Any]) -> List[str]:
        """The sorted, duplicate-free hashes of everything the node cites:
        each cited lemma's statement hash, each cited reference's formal-
        statement hash, and the marker of _unresolved() for a citation
        that resolves to nothing. Sorting is what "the sorted hashes of
        everything it cites" says: the set of cited hashes, not their
        order in the file, is part of the lemma's content."""
        pairs, refs = self._citations(node)
        hashes: List[str] = []
        for pair in pairs:
            if pair in self._lemmas:
                # Hashed for the cycle check (see hash()); the value is
                # the cited statement's alone.
                self.hash(*pair)
                hashes.append(
                    statement_hash(self._lemmas[pair].get("statement"))
                )
            else:
                hashes.append(_unresolved("lemma", *pair))
        for rid in refs:
            h = self._ref_hashes.get(rid)
            if h is None:
                h = _unresolved("reference", rid)
            hashes.append(h)
        return sorted(set(hashes))


def _lemma_key(key: Any) -> LemmaKey:
    """A mapping key as the (user_id, lemma_id) pair it must be.

    A bare string is rejected rather than guessed at: a lemma_id is
    unique only within one user's file, so a mapping keyed by bare ids
    is not a complete DAG and hashing it would hash the wrong thing."""
    if isinstance(key, (tuple, list)) and len(key) == 2:
        uid, lid = key
        if isinstance(uid, str) and isinstance(lid, str) and uid and lid:
            return (uid, lid)
    raise ValueError(
        f"{key!r} is not a (user_id, lemma_id) pair; a lemma is "
        f"identified by the pair, a bare lemma_id names no particular "
        f"lemma"
    )
