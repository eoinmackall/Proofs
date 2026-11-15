"""Resolution: when the conjecture counts as proved or disproved.

The conjecture is settled by a lemma, never by an agent's say-so. A
candidate attempts it by saying which way, in its "settles" field:

    conjecture            the conjecture itself, as conjecture.md states it
    conjecture_negation   that the conjecture is false, by an explicit
                          counterexample

and the statement of such a candidate is written by the code, never by a
model: whoever proposes one — the planner, the lemma generator, the
operator at the menu — has the statement it gave replaced by the pinned
text (pin()), the verbatim contents of conjecture.md, or that text
wrapped in NEGATION_PREFIX. It is otherwise a candidate like any other,
with an id like any other: it can be picked like the rest, the prover
writes its proof from the lemmas in the DAG, and the three verifiers
check it. A pinned statement is never revised (is_pinned).

Whether the conjecture is settled is computed, never stored, the way
suspension is (suspension.py): find() looks for a lemma, anyone's, that

  * states the pinned text exactly, up to merkle.normalize()'s whitespace
    cosmetics — so an edit to conjecture.md leaves an older resolution
    behind on its own; and
  * is not suspended — so it holds a valid certificate, and everything it
    cites, all the way down, is certified and unrefuted.

The test is on the statement: "settles" only asks for the pinned text, and
the lemma a proof of it is committed as settles the conjecture by what it
states.
A refutation, or a lemma left uncertified, anywhere below a resolving
lemma suspends it, and the conjecture is open again.

This module is deliberately standalone, the way merkle.py is: it imports
nothing from this package but merkle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Collection, Dict, List, Mapping, Optional

from . import merkle

__all__ = [
    "PROVE",
    "DISPROVE",
    "SETTLES",
    "NEGATION_PREFIX",
    "Resolution",
    "is_pinned",
    "pinned_statement",
    "pin",
    "find",
    "describe",
]

PROVE = "conjecture"
DISPROVE = "conjecture_negation"
SETTLES = (PROVE, DISPROVE)

NEGATION_PREFIX = (
    "The following conjecture is false. Exhibit an explicit counterexample "
    "to it and prove that it is one.\n\n"
)

@dataclass(frozen=True)
class Resolution:
    """A lemma that settles the conjecture."""

    kind: str          # "proved" or "disproved"
    lemma_id: str      # the settling lemma


def pinned_statement(settles: Any, conjecture: str) -> Optional[str]:
    """The statement the code writes for a "settles" value, None for
    anything else."""
    settles = str(settles or "").strip()
    text = str(conjecture or "").strip()
    if settles == PROVE:
        return text
    if settles == DISPROVE:
        return NEGATION_PREFIX + text
    return None


def is_pinned(statement: Any, conjecture: str) -> bool:
    """Whether statement is the conjecture or its negation (up to
    whitespace): a statement the code wrote, never to be revised."""
    s = merkle.normalize(statement)
    return any(s == merkle.normalize(pinned_statement(k, conjecture)) for k in SETTLES)


def pin(candidate: Dict[str, Any], conjecture: str) -> Dict[str, Any]:
    """The candidate with its statement pinned, if it settles the
    conjecture one way or the other; "settles" is not kept. Any other
    candidate is returned without the field."""
    out = {k: v for k, v in candidate.items() if k != "settles"}
    stmt = pinned_statement(candidate.get("settles"), conjecture)
    if stmt is not None:
        out["statement"] = stmt
    return out


def find(
    lemmas: Mapping[str, Mapping[str, Any]],
    suspended: Collection[str],
    conjecture: str,
) -> List[Resolution]:
    """Every lemma that settles the conjecture now, sorted by id.

    lemmas is the complete DAG's lemmas, keyed by id, and suspended the ids
    the suspension hides. Empty when the
    conjecture is open. Holding both kinds at once means the DAG proves
    the conjecture and its negation; the caller reports that rather than
    picking one."""
    if not str(conjecture or "").strip():
        return []
    targets: Dict[str, str] = {
        merkle.normalize(pinned_statement(PROVE, conjecture)): "proved",
        merkle.normalize(pinned_statement(DISPROVE, conjecture)): "disproved",
    }
    out: List[Resolution] = []
    for lemma_id in sorted(lemmas):
        if lemma_id in suspended:
            continue
        node = lemmas[lemma_id]
        kind = targets.get(merkle.normalize(node.get("statement", "")))
        if kind is None:
            continue
        out.append(Resolution(kind=kind, lemma_id=lemma_id))
    return out


def describe(res: Resolution) -> str:
    """One line for the log: what is settled, and by which lemma."""
    return f"the conjecture is {res.kind} by {res.lemma_id}"
