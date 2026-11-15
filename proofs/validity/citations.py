"""Checking the citations a proof declares against the DAG and the
reference collection.
"""

from __future__ import annotations

import re
from typing import Any, Collection, List, Optional, Set, Tuple

from ..core.dag import _citation_display, _lemma_id_from


def check_citations(
    raw_lemmas: Any,
    raw_references: Any,
    dag: Any,
    ref_ids: Set[str],
    suspended: Optional[Collection[str]] = None,
) -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    """The prover's declared citations, normalized and checked against the
    DAG.

    cited_lemmas is a list of lemma ids, cited_references a list of
    reference ids. Returns (cited_lemmas, cited_references, unmatched): the
    surviving lemma ids, the surviving reference ids, and (label, kind) for
    every entry that matched nothing — kind is "lemma" for an id no lemma
    has, "reference" for an id that is neither a lemma nor a reference,
    "suspended" for a suspended lemma, and "other" for an entry that is not
    an id at all. The caller sends the unmatched back to the prover as
    feedback; the "reference" ones are the maintainer's to supply, since
    references.md is the conjecture's committed collection (see parsing.py).

    Lenient on shape, strict on existence: an id in the wrong list is read
    as the citation it is — a lemma id among the references is a lemma, a
    reference id among the lemmas a reference — but whatever an entry names
    must exist, or it is unmatched.

    A suspended lemma is unmatched, as "suspended": the prover was not shown
    it and may not build on it (see suspension.py). Let through, it would
    suspend the new lemma the moment it was committed — and in a repair, a
    citation of the lemma being re-proved, or of one of its dependents,
    would close a citation cycle once the new version superseded the old."""
    hidden = set(suspended or ())
    cited_lemmas: List[str] = []
    cited_references: List[str] = []
    unmatched: List[Tuple[str, str]] = []
    seen: Set[str] = set()

    def take(ident: str) -> None:
        if ident in seen:
            return
        seen.add(ident)
        if ident in hidden:
            unmatched.append((ident, "suspended"))
        elif ident in dag.lemmas:
            cited_lemmas.append(ident)
        elif ident in ref_ids:
            cited_references.append(ident)
        elif "/" in ident:
            unmatched.append((ident, "lemma"))
        else:
            unmatched.append((ident, "reference"))

    for raw in (raw_lemmas, raw_references):
        for entry in raw if isinstance(raw, list) else []:
            ident = _lemma_id_from(entry)
            if ident is not None:
                take(ident)
            else:
                label = _citation_display(entry)
                if label not in seen:
                    seen.add(label)
                    unmatched.append((label, "other"))
    return cited_lemmas, cited_references, unmatched


def scan_citations(
    proof: str, dag: Any, references: List[Any],
    suspended: Optional[Collection[str]] = None,
) -> Tuple[List[str], List[str]]:
    """Fallback edge recovery: which known results appear in the proof text.

    Only used when the prover ignored its output format entirely, in which
    case the alternative is a lemma that silently claims to stand on its
    own. Scans lemma ids and reference ids, matched whole (so .../1 does not
    match inside .../10). Everything it names exists by construction; a
    suspended lemma is never recovered — the verifiers, shown no statement
    for it, see the gap."""
    hidden = set(suspended or ())
    cited_lemmas = [
        lid for lid in sorted(dag.lemmas)
        if lid not in hidden and re.search(rf"(?<![\w/]){re.escape(lid)}(?![\w/])", proof)
    ]
    cited_references: List[str] = []
    for ref in references:
        rid = str(ref.get("id") or "")
        if rid and re.search(rf"\b{re.escape(rid)}\b", proof):
            cited_references.append(rid)
    return cited_lemmas, cited_references
