"""Resolution: when the conjecture counts as proved or disproved.

The conjecture is settled by a lemma, never by an agent's say-so. Two
lemma ids are reserved for it:

    conjecture            the conjecture itself, as conjecture.md states it
    conjecture_negation   that the conjecture is false, by an explicit
                          counterexample

and the statement under either id is written by the code, never by a
model: whoever proposes one — the planner, the lemma generator, the
operator at the menu — has the statement it gave replaced by the pinned
text (pin()), the verbatim contents of conjecture.md, or that text
wrapped in NEGATION_PREFIX. A reserved candidate is otherwise a candidate
like any other: the selector weighs it against the rest, the prover
writes its proof from the lemmas in the DAG, and the three verifiers
check it.

Whether the conjecture is settled is computed, never stored, the way
suspension is (suspension.py): find() looks for a lemma, in any user's
DAG file, that

  * states the pinned text exactly, up to merkle.normalize()'s whitespace
    cosmetics — so an edit to conjecture.md leaves an older resolution
    behind on its own; and
  * is not suspended — so it holds a valid certificate, and everything it
    cites, all the way down, is certified and unrefuted.

The test is on the statement, not on the id: the id is only the trigger
for pinning, and a proof committed under a renamed id (conjecture_2,
after a collision) settles the conjecture as well as the original would.
A refutation, or a lemma left uncertified, anywhere below a resolving
lemma suspends it, and the conjecture is open again.

This module is deliberately standalone, the way merkle.py is: it imports
nothing from this package but merkle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Collection, Dict, List, Mapping, Optional, Tuple

try:  # package form: `python -m proofs`, `pip install -e .`
    from . import merkle
except ImportError:  # running the package directory as a script
    import merkle

__all__ = [
    "PROVE_ID",
    "DISPROVE_ID",
    "RESERVED_IDS",
    "NEGATION_PREFIX",
    "Resolution",
    "is_reserved",
    "pinned_statement",
    "pin",
    "find",
    "describe",
]

PROVE_ID = "conjecture"
DISPROVE_ID = "conjecture_negation"
RESERVED_IDS = (PROVE_ID, DISPROVE_ID)

NEGATION_PREFIX = (
    "The following conjecture is false. Exhibit an explicit counterexample "
    "to it and prove that it is one.\n\n"
)

Pair = Tuple[str, str]


@dataclass(frozen=True)
class Resolution:
    """A lemma that settles the conjecture."""

    kind: str          # "proved" or "disproved"
    pair: Pair         # (user_id, lemma_id) of the settling lemma


def is_reserved(lemma_id: Any) -> bool:
    """Whether lemma_id is one of the two ids whose statement is pinned."""
    return str(lemma_id or "").strip() in RESERVED_IDS


def pinned_statement(lemma_id: str, conjecture: str) -> Optional[str]:
    """The statement the code writes for a reserved id, None for any other."""
    lemma_id = str(lemma_id or "").strip()
    text = str(conjecture or "").strip()
    if lemma_id == PROVE_ID:
        return text
    if lemma_id == DISPROVE_ID:
        return NEGATION_PREFIX + text
    return None


def pin(candidate: Dict[str, Any], conjecture: str) -> Dict[str, Any]:
    """The candidate with its statement pinned, if its id is reserved.

    Any other candidate is returned unchanged. The id is stripped, so
    " conjecture" is the reserved id too."""
    stmt = pinned_statement(candidate.get("id", ""), conjecture)
    if stmt is None:
        return candidate
    return {**candidate, "id": str(candidate["id"]).strip(), "statement": stmt}


def find(
    lemmas: Mapping[Pair, Mapping[str, Any]],
    suspended: Collection[Pair],
    conjecture: str,
) -> List[Resolution]:
    """Every lemma that settles the conjecture now, sorted by pair.

    lemmas is the complete DAG's lemmas, keyed by (user_id, lemma_id), and
    suspended the pairs suspension.compute() hides. Empty when the
    conjecture is open. Holding both kinds at once means the DAG proves
    the conjecture and its negation; the caller reports that rather than
    picking one."""
    if not str(conjecture or "").strip():
        return []
    targets: Dict[str, str] = {
        merkle.normalize(pinned_statement(PROVE_ID, conjecture)): "proved",
        merkle.normalize(pinned_statement(DISPROVE_ID, conjecture)): "disproved",
    }
    out: List[Resolution] = []
    for pair in sorted(lemmas):
        if pair in suspended:
            continue
        node = lemmas[pair]
        kind = targets.get(merkle.normalize(node.get("statement", "")))
        if kind is None:
            continue
        out.append(
            Resolution(kind=kind, pair=(str(pair[0]), str(pair[1])))
        )
    return out


def describe(res: Resolution) -> str:
    """One line for the log: what is settled, and by which lemma."""
    return f"the conjecture is {res.kind} by {res.pair[0]}:{res.pair[1]}"
