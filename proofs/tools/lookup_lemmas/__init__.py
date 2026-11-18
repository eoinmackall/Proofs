"""lookup_lemmas: how an agent reads a proved lemma it has only the id of.

The lemma window (loops/prompts.py, lemma_window) shows the agents the most
recently proved lemmas in full, each with the ids of the lemmas it cites;
search_lemmas returns lemmas the same way. A cited lemma not shown in full
is one lookup away. An agent shown the window is offered this tool
whenever the window leaves some lemmas out.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Set

from ..base import Tool, fit_entries


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
        for entry in (wanted if max_lookup < 0 else wanted[:max_lookup]):
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
            if 0 <= max_lookup < len(wanted) else ""
        )

        # Not all of it may fit the room left: whole statements, as many as
        # fit, in the order asked for, and a line saying what was left out.
        def render(kept: List[Dict[str, Any]], left_out: int) -> str:
            text = json.dumps({"lemmas": kept}, indent=2) + note
            if left_out:
                text += (
                    f"\n({left_out} of the {len(found)} lemmas found were left "
                    "out: room in the context window is running out. Look up "
                    "fewer at a time, or answer with what you have.)"
                )
            return text

        return fit_entries(found, render, budget)

    return Tool(
        name="lookup_lemmas",
        description=(
            "Read the statements of proved lemmas by id: ids from a "
            "cited_lemmas list, or from search results."
            + ("" if max_lookup < 0 else f" At most {max_lookup} lemmas per call.")
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
            "Lemmas not shown in full appear by id only, in cited_lemmas "
            "lists. Call lookup_lemmas to read the statement of any of them "
            "before relying on it."
        ),
    )


class LookupLemmas:
    """The plug-in: offered to an agent shown the lemma window, when the
    window leaves some lemmas out."""

    name = "lookup_lemmas"

    def offer(self, ctx: Any, call: Any) -> Optional[Tool]:
        if call.lemmas is None or not call.hidden:
            return None
        return lemma_lookup(
            call.lemmas, call.suspended, max_lookup=ctx.settings.max_lookup,
        )
