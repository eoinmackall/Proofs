"""search_lemmas: how an agent finds a proved lemma the lemma window does not
show.

The lemma window (loops/prompts.py, lemma_window) shows a bounded number of
lemmas however large the DAG grows, and says only how many it leaves out.
This tool searches every proved lemma: by words or phrases from the
statement, ranked (index.py), and by citation — the lemmas that cite a
given lemma (cites), or that it cites (cited_by) — either alone, newest
first, or with a query ranking them. A lemma comes back as the window
shows one: its statement and its citations, never its proof.

It never returns a suspended lemma (a lemma no agent may build on must not
come back in through a search), nor one the agent's prompt already shows
in full, nor one the agent's DAG snapshot does not hold: the index is
shared by the run's threads and kept up to date by its commits, so it can
be ahead of the snapshot a prompt was built from.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set

from ..base import Tool, fit_entries
from .index import LemmaIndex

NO_MATCH = ("No proved lemma matched. Try other words, or the other "
            "notation (LaTeX or Unicode).")


def _lemma_id(raw: Any) -> str:
    return str(raw).strip() if isinstance(raw, str) else ""


def lemma_search(
    index: LemmaIndex,
    dag: Any,
    window: Set[str],
    suspended: Optional[Set[str]] = None,
    max_search: int = 5,
) -> Tool:
    """search_lemmas over index, for an agent shown dag (the snapshot its
    prompt was built from) with the lemmas in window shown in full. One call
    returns at most max_search lemmas (tools.max_search; -1, no limit); the
    rest of the results are a further call away (offset)."""
    lemmas = dag.lemmas
    suspended = suspended or set()

    def usable(lid: str) -> bool:
        return lid in lemmas and lid not in suspended and lid not in window

    def newest_first(ids: List[str]) -> List[str]:
        return sorted(ids, key=lambda lid: (str(lemmas[lid].get("proved_at") or ""), lid),
                      reverse=True)

    def entry(lid: str) -> Dict[str, Any]:
        node = lemmas[lid]
        return {
            "id": lid,
            "statement": str(node.get("statement", "")),
            "cited_lemmas": list(node.get("cited_lemmas", [])),
            "cited_references": list(node.get("cited_references", [])),
        }

    def handler(args: Dict[str, Any], budget: Optional[int] = None) -> str:
        query = args.get("query")
        query = query.strip() if isinstance(query, str) else ""
        cites, cited_by = _lemma_id(args.get("cites")), _lemma_id(args.get("cited_by"))
        offset = args.get("offset", 0)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            return 'error: "offset" must be a whole number, 0 or more'
        if not (query or cites or cited_by):
            return 'error: pass "query", "cites" or "cited_by"'
        for name, lid in (("cites", cites), ("cited_by", cited_by)):
            if lid and lid not in lemmas:
                return f"error: {name}: {lid} is not a proved lemma"
            if lid in suspended:
                return f"error: {name}: {lid} is suspended: not available to build on"
        error = index.wait()
        if error:
            return f"error: search is unavailable: {error}"

        want = None if max_search < 0 else offset + max_search + 1
        if cites or cited_by:
            pools = []
            if cites:
                pools.append(set(dag.cited_by(cites)))
            if cited_by:
                pools.append(set(lemmas[cited_by].get("cited_lemmas", [])))
            pool = [lid for lid in set.intersection(*pools) if usable(lid)]
            ranked = (index.rank(query, {lid: str(lemmas[lid].get("statement", ""))
                                         for lid in pool})
                      if query else newest_first(pool))
        else:
            ranked = index.search(query, want, usable)

        page = ranked[offset:] if max_search < 0 else ranked[offset:offset + max_search]
        more = len(ranked) > offset + len(page)
        entries = [entry(lid) for lid in page]

        # Whole statements, as many as fit the room left; next_offset then
        # starts where the cut did.
        def render(kept: List[Dict[str, Any]], left_out: int) -> str:
            out: Dict[str, Any] = {"lemmas": kept}
            if not kept and not left_out:
                out["note"] = NO_MATCH if offset == 0 else "No more results."
            if more or left_out:
                out["more"] = True
                out["next_offset"] = offset + len(kept)
            if left_out:
                out["note"] = (
                    f"{left_out} more results on this page were left out: room "
                    "in the context window is running out. Answer with what "
                    "you have, or search again from next_offset."
                )
            return json.dumps(out, indent=2)

        return fit_entries(entries, render, budget)

    per_call = "" if max_search < 0 else f" At most {max_search} lemmas per call."
    return Tool(
        name="search_lemmas",
        description=(
            "Search every proved lemma in the DAG, including those not "
            "listed in your prompt: by words or phrases from the statement, "
            "best match first; or the lemmas that cite a given lemma "
            "(cites) or that it cites (cited_by), newest first unless a "
            "query ranks them. Lemmas your prompt shows in full are not "
            "returned." + per_call + " For more, search again with "
            "next_offset as offset."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": 'Words or "quoted phrases" from a lemma '
                                   "statement; a word ending in * matches "
                                   "every word it begins.",
                },
                "cites": {"type": "string",
                          "description": "Only lemmas that cite this lemma id."},
                "cited_by": {"type": "string",
                             "description": "Only lemmas this lemma id cites."},
                "offset": {"type": "integer", "minimum": 0,
                           "description": "Skip this many results: the "
                                          "next_offset of an earlier search."},
            },
        },
        handler=handler,
        takes_budget=True,
        hint=(
            "search_lemmas searches all the proved lemmas in the DAG, not "
            "only those listed here, by words or phrases from their "
            "statements, and finds the lemmas citing, or cited by, a given "
            "lemma. Statements mix LaTeX and Unicode notation (\\ell and ℓ): "
            "if a search finds nothing useful, try other words or the other "
            "notation."
        ),
    )


def run_index(ctx: Any) -> Optional[LemmaIndex]:
    """The run's index, started on first use from the DAG as it stands, or
    None when there is none to be had (said once). The relevance strategy
    (loops/strategies.py) searches it too."""
    index = ctx.run.search_index
    if index is None:
        # Under the DAG lock: no commit falls between the DAG the build
        # reads and the index the commits update.
        with ctx.run.dag_lock:
            index = ctx.run.search_index
            if index is None:
                index = ctx.run.search_index = LemmaIndex.start(ctx.dag.lemmas)
    if index.failed():
        if not index.reported:
            index.reported = True
            ctx.ui.log(f"note: search_lemmas is not offered: {index.error}", err=True)
        return None
    return index


class SearchLemmas:
    """The plug-in: offered to an agent shown the lemma window, when the
    window leaves some lemmas out."""

    name = "search_lemmas"

    def offer(self, ctx: Any, call: Any) -> Optional[Tool]:
        if call.lemmas is None or call.dag is None or not call.hidden:
            return None
        index = run_index(ctx)
        if index is None:
            return None
        return lemma_search(index, call.dag, call.window, call.suspended,
                            max_search=ctx.settings.max_search)
