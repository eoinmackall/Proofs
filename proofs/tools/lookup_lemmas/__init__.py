"""lookup_lemmas: how an agent reads a proved lemma the lemma window lists by
id only.

The lemma window (loops/prompts.py, lemma_window) shows the agents the most
recently proved lemmas in full and lists the rest by id only; an agent shown
the window is offered this tool whenever it leaves some out.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Set

from ..base import NO_ROOM, Tool



def lemma_lookup(
    lemmas: Mapping[str, Mapping[str, Any]],
    suspended: Optional[Set[str]] = None,
    max_lookup: int = 10,
) -> Tool:
    """lookup_lemmas over the DAG's lemmas (Dag.lemmas, keyed by id). It
    returns what the lemma window shows for a lemma in full — the statement
    and the citations, not the proof — and refuses a suspended lemma, as the
    window does: a lemma no agent may build on must not come back in through
    a lookup. One call returns at most max_lookup lemmas (tools.max_lookup),
    for the same reason a call's rounds are bounded."""
    suspended = suspended or set()

    def handler(args: Dict[str, Any], budget: Optional[int] = None) -> str:
        wanted = args.get("lemmas")
        if not isinstance(wanted, list) or not wanted:
            return 'error: pass "lemmas", a non-empty list of lemma ids'
        found: List[Dict[str, Any]] = []
        for entry in wanted[:max_lookup]:
            if isinstance(entry, dict):
                entry = entry.get("id")
            lid = str(entry or "").strip()
            if lid not in lemmas:
                found.append({"id": lid, "error": "no such proved lemma"})
            elif lid in suspended:
                found.append({"id": lid, "error": "suspended: not available to build on"})
            else:
                node = lemmas[lid]
                found.append({
                    "id": lid,
                    "statement": str(node.get("statement", "")),
                    "cited_lemmas": list(node.get("cited_lemmas", [])),
                    "cited_references": list(node.get("cited_references", [])),
                })
        note = (
            f" (only the first {max_lookup} were looked up)"
            if len(wanted) > max_lookup else ""
        )
        text = json.dumps({"lemmas": found}, indent=2) + note
        if budget is None or len(text) <= budget:
            return text
        # Not all of it fits the room left: whole statements, as many as
        # fit, in the order asked for, and a line saying what was left out.
        # Half a statement is worse than none — the model would build on a
        # hypothesis list cut short.
        def left_out(n: int) -> str:
            return (
                f"\n({n} of the {len(found)} lemmas found were left out: "
                "room in the context window is running out. Look up fewer "
                "at a time, or answer with what you have.)"
            )
        kept: List[Dict[str, Any]] = []
        for entry in found:
            trial = json.dumps({"lemmas": kept + [entry]}, indent=2)
            if len(trial) + len(note) + len(left_out(len(found))) > budget:
                break
            kept.append(entry)
        if not kept:
            return NO_ROOM
        return (json.dumps({"lemmas": kept}, indent=2) + note
                + left_out(len(found) - len(kept)))

    return Tool(
        name="lookup_lemmas",
        description=(
            "Read the statements of proved lemmas listed in "
            "other_proved_lemmas by id only. Give each by its id. At most "
            f"{max_lookup} lemmas per call."
        ),
        parameters={
            "type": "object",
            "properties": {
                "lemmas": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["lemmas"],
        },
        handler=handler,
        takes_budget=True,
        hint=(
            "other_proved_lemmas lists proved lemmas by id only. Call "
            "lookup_lemmas to read the statement of any of them before "
            "relying on it."
        ),
    )


class LookupLemmas:
    """The plug-in: offered to an agent shown the lemma window, when the
    window lists some lemmas by id only."""

    name = "lookup_lemmas"

    def offer(self, ctx: Any, call: Any) -> Optional[Tool]:
        if call.lemmas is None or not call.index:
            return None
        return lemma_lookup(
            call.lemmas, call.suspended, max_lookup=ctx.settings.max_lookup,
        )
