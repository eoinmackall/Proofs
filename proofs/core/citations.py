"""Who cites whom, read backwards, and what the citations add up to.

A lemma records only what it cites (cited_lemmas). The other direction —
which lemmas cite this one — is what suspension spreads along, what proofs
repair climbs and what proofs prune follows, and what search_lemmas filters
by. build_cited_by makes the whole map in one pass over every citation;
the Dag keeps the map it built (Dag.cited_by_map) and answers for one
lemma from it (Dag.cited_by), carrying it across its own commits rather
than building it again.

The lemma window's strategies (loops/strategies.py) read two more things
off the citations, each built in one pass in dependency order and kept by
the Dag the same way:

  * a lemma's depth: the longest citation chain beneath it, 0 for a lemma
    citing nothing (depths);
  * a sketch of its ancestry — every lemma it rests on, directly or
    through a chain — small enough to keep for every lemma, from which the
    overlap of two ancestries is estimated (Ancestry).
"""

from __future__ import annotations

import hashlib
from array import array
from typing import Any, Dict, List, Mapping, Optional, Sequence

try:                                    # optional: pip install proofs[fast]
    import numpy as _np
except ImportError:                     # pragma: no cover - depends on the machine
    _np = None


def build_cited_by(lemmas: Mapping[str, Mapping[str, Any]]) -> Dict[str, List[str]]:
    """Every cited id, with the lemmas that cite it, each citer once, in
    lemmas' order. A cited id lemmas does not hold (a lemma an earlier
    prune removed) has its citers all the same; a lemma nothing cites has
    no entry."""
    out: Dict[str, List[str]] = {}
    for lid, node in lemmas.items():
        for cited in node.get("cited_lemmas") or ():
            citers = out.get(cited)
            if citers is None:
                out[cited] = [lid]
            elif citers[-1] != lid:     # a lemma that lists an id twice
                citers.append(lid)
    return out


def _cited(lemmas: Mapping[str, Mapping[str, Any]], lid: str) -> List[str]:
    """What lid cites among lemmas."""
    return [c for c in lemmas[lid].get("cited_lemmas") or () if c in lemmas]


def dependency_order(lemmas: Mapping[str, Mapping[str, Any]],
                     cited_by: Mapping[str, List[str]]) -> List[str]:
    """lemmas with each after every lemma it cites (Kahn's algorithm, from
    the lemmas citing nothing upward). A lemma on a citation cycle, or
    citing its way up to one, is left out: it has no place in the order."""
    waiting = {lid: len(set(_cited(lemmas, lid))) for lid in lemmas}
    order = [lid for lid, n in waiting.items() if n == 0]
    for lid in order:                   # order grows as lemmas come free
        for up in set(cited_by.get(lid, ())):
            if up in waiting:
                waiting[up] -= 1
                if waiting[up] == 0:
                    order.append(up)
    return order


def depths(lemmas: Mapping[str, Mapping[str, Any]],
           cited_by: Mapping[str, List[str]]) -> Dict[str, int]:
    """Every lemma's depth: the longest citation chain beneath it, 0 for a
    lemma citing nothing. A lemma dependency_order leaves out (a cycle)
    has none."""
    out: Dict[str, int] = {}
    for lid in dependency_order(lemmas, cited_by):
        out[lid] = 1 + max((out[c] for c in _cited(lemmas, lid)), default=-1)
    return out


def depth_of(lemmas: Mapping[str, Mapping[str, Any]], depth: Mapping[str, int],
             node: Mapping[str, Any]) -> int:
    """The depth of a lemma just committed, from the depths of what it
    cites: nothing cites it yet, so no other depth moves."""
    return 1 + max((depth.get(c, 0) for c in node.get("cited_lemmas") or ()
                    if c in lemmas), default=-1)


class Ancestry:
    """MinHash sketches of every lemma's ancestry.

    Fix K hash functions of a lemma id. The sketch of a set of lemmas is,
    for each, the smallest hash of any lemma in the set; two sets' sketches
    agree in a position with probability exactly the sets' Jaccard
    similarity (shared lemmas over all lemmas), so the fraction of agreeing
    positions estimates it, within about 1/(2√K).

    What is kept per lemma is the sketch of the lemma together with its
    ancestry (_closure). That of a new lemma is the position-wise minimum
    of its own hashes and the closures of what it cites, so every one is
    built in one pass in dependency order, and a commit adds one. A lemma's
    ancestry alone (sketch) is the minimum over the closures of what it
    cites: a lemma citing nothing has the empty sketch, every position at
    its maximum, so two such lemmas look alike and unlike everything else.

    With numpy (the fast extra) K is 64 and the sketches are numpy arrays;
    without it K is 16 and they are arrays of the standard library.
    """

    K = 64 if _np is not None else 16
    EMPTY = 0xFFFFFFFF

    def __init__(self, closure: Dict[str, Any]) -> None:
        self._closure = closure

    @staticmethod
    def _hashes(lemma_id: str) -> Any:
        raw = hashlib.shake_128(lemma_id.encode("utf-8")).digest(4 * Ancestry.K)
        if _np is not None:
            return _np.frombuffer(raw, dtype="<u4").copy()
        out = array("I")
        out.frombytes(raw)
        return out

    @classmethod
    def _min(cls, sketches: Sequence[Any]) -> Any:
        if not sketches:
            if _np is not None:
                return _np.full(cls.K, cls.EMPTY, dtype="<u4")
            return array("I", [cls.EMPTY] * cls.K)
        if _np is not None:
            return _np.minimum.reduce(sketches) if len(sketches) > 1 else sketches[0]
        return array("I", map(min, zip(*sketches)))

    @classmethod
    def build(cls, lemmas: Mapping[str, Mapping[str, Any]],
              cited_by: Mapping[str, List[str]]) -> "Ancestry":
        closure: Dict[str, Any] = {}
        for lid in dependency_order(lemmas, cited_by):
            closure[lid] = cls._min(
                [cls._hashes(lid)] + [closure[c] for c in _cited(lemmas, lid)])
        return cls(closure)

    def with_lemma(self, lemmas: Mapping[str, Mapping[str, Any]],
                   node: Mapping[str, Any]) -> "Ancestry":
        """These sketches with a lemma just committed (node, in lemmas):
        nothing cites it yet, so no other sketch moves."""
        closure = dict(self._closure)
        closure[node["id"]] = self._min(
            [self._hashes(node["id"])]
            + [closure[c] for c in node.get("cited_lemmas") or ()
               if c in lemmas and c in closure])
        return Ancestry(closure)

    def sketch(self, lemmas: Mapping[str, Mapping[str, Any]], lemma_id: str) -> Any:
        """The sketch of lemma_id's ancestry, without lemma_id itself."""
        return self._min([self._closure[c] for c in _cited(lemmas, lemma_id)
                          if c in self._closure])

    def sketches(self, lemmas: Mapping[str, Mapping[str, Any]],
                 lemma_ids: Sequence[str]) -> Any:
        """The sketches of many lemmas' ancestries: with numpy one matrix,
        a row each, made in one reduction over every cited lemma's closure
        (a lemma citing nothing gets the empty row); a list otherwise."""
        if _np is None:
            return [self.sketch(lemmas, lid) for lid in lemma_ids]
        rows: List[Any] = []
        starts: List[int] = []
        empty = _np.full(self.K, self.EMPTY, dtype="<u4")
        for lid in lemma_ids:
            starts.append(len(rows))
            cited = [self._closure[c] for c in _cited(lemmas, lid) if c in self._closure]
            rows.extend(cited or (empty,))
        if not rows:
            return _np.empty((0, self.K), dtype="<u4")
        return _np.minimum.reduceat(_np.stack(rows), _np.array(starts), axis=0)

    @staticmethod
    def numpy() -> Optional[Any]:
        """numpy, when installed; None otherwise."""
        return _np
