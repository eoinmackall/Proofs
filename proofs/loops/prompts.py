"""Prompt assembly: the lemma window, the agents' views of the DAG and the
references, and fitting a prompt into the context window.
"""

from __future__ import annotations

import hashlib
import json
import random
import textwrap
from pathlib import Path
from typing import Any, Callable, Collection, Dict, List, Optional, Sequence, Tuple

from .. import tools as tools_mod
from ..core import resolution
from ..core.dag import load_file, new_lemma_id, owner_of
from ..tools.run_code import computations
from .reasoning import (
    COMPACT_MIN_ROOM, _estimate_tokens, _lookup_room, _native_tools, _reserve,
    _room,
)

# request a server accepted and then dropped must not stall the run longer.
# This is the longest gap between bytes received (requests' read timeout),
# and on a non-streaming call that is the whole generation.

# How many lemmas the planner shortlists per iteration. Raising this costs
# planner tokens and, in human mode, attention; five is about as many
# candidates as can be compared without re-reading the conjecture.
PLANNER_CANDIDATES = 5


# How many parsed references (references.md) the planner and the reviser are
# shown at all. The rule is deliberately blunt: below the limit each of them
# gets an id + slogan per result, so they can build on a named theorem rather
# than re-prove it from scratch; at or above it, none. A shortlist of hundreds
# of slogans costs more in tokens and attention than it returns, and the
# prover sees the full collection regardless, so a named result the planner
# misses is one proving round away, not lost. A strategy for large
# collections is deliberately not built yet.
PLANNER_REFERENCE_LIMIT = 100


_TIEBREAK_SEED = random.getrandbits(64)


def _window_sizes(ctx) -> Tuple[int, int, int]:
    """(lemmas_all, my_lemmas, lemmas_common) as this call should see them:
    _fit_prompt's override when it is rebuilding a prompt, else the
    module's (the flags')."""
    override = ctx.loop.window_override
    return override if override is not None else (
        ctx.settings.lemmas_all, ctx.settings.my_lemmas, ctx.settings.lemmas_common)


def agent_tools(
    ctx,
    role: str,
    dag: Any,
    suspended: Optional[Collection[str]] = None,
) -> Optional["tools_mod.ToolSet"]:
    """The tools an agent shown the lemma window is offered (tools/): today
    lookup_lemmas, over the same lemmas the window chose from, when the
    window leaves some out; and run_code under the role's hint, when the
    run was given --tools and the role has one. None when no tool offers
    itself, so a small DAG's prompts carry no tools."""
    _shown, _common, index = lemma_window(ctx, dag, suspended)
    return tools_mod.offered(ctx, tools_mod.Call(
        role=role, lemmas=dag.lemmas, suspended=set(suspended or ()),
        index=index,
    ))


def verifier_tools(ctx, agent: str = "") -> Optional["tools_mod.ToolSet"]:
    """What a verifier is offered: it is shown no lemma window, so only a
    tool that needs none — run_code, under the verifier's hint (a
    computation may justify a rejection, never an acceptance), when the run
    was given --tools; else nothing."""
    return tools_mod.offered(ctx, tools_mod.Call(role="verifier", agent=agent))


def rejected_block(ctx, rejected: List[Dict[str, str]]) -> str:
    """The reject list as the planning agents read it: the statements only."""
    statements = [e["statement"] for e in rejected]
    return (
        f"Rejected lemmas (the most recent; each went through all "
        f"{ctx.settings.max_proof_attempts} prover rounds without a proof the verifiers "
        f"accepted — do not propose these again unchanged):\n"
        + (json.dumps(statements, indent=2) if statements else "(none)")
    )


def replan_note(shown: List[Dict[str, Any]], notes: List[str]) -> str:
    """The operator's 'replan' at the human menu, as the next planning call
    reads it: the statements that were turned down, and why."""
    statements = [str(c.get("statement") or "") for c in shown]
    return (
        "The operator turned down the last shortlist without trying it:\n"
        f"{json.dumps(statements, indent=2)}\n"
        "Operator's reason: " + (" ".join(notes) or "(none given)")
    )


def lemma_window(
    ctx,
    dag: Any,
    suspended: Optional[Collection[str]] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """Which proved lemmas the agents see in full, and the index of the rest.

    The window is chosen from the lemmas that are not suspended — choosing
    first and filtering after would fill it with lemmas no agent may use.
    Newest first by proved_at, id breaking ties: the lemmas_all newest
    lemmas, anyone's, and the my_lemmas newest of the current user's own,
    deduplicated, so the two can show fewer than their sum. -1 for either
    is no limit.

    Then the common lemmas: of the eligible lemmas not already shown,
    the lemmas_common cited most often by the shown ones, counting direct
    citations only, ties broken by _tiebreak. A lemma no shown lemma
    cites does not rank, so the group can be smaller than lemmas_common.

    Returns (the shown ids, oldest first — the order the proof grew in;
    the common ids, most cited first; every other eligible lemma's id,
    sorted). The index is empty when everything is shown.
    """
    hidden = set(suspended or ())
    eligible = [lid for lid in dag.lemmas if lid not in hidden]
    newest = sorted(
        eligible,
        key=lambda lid: (str(dag.lemmas[lid].get("proved_at") or ""), lid),
        reverse=True,
    )

    def take(ids: List[str], n: int) -> List[str]:
        return ids if n < 0 else ids[:n]

    # The flags' sizes, or _fit_prompt's smaller ones while it rebuilds a
    # prompt to fit the window (ctx.loop.window_override).
    lemmas_all, my_lemmas, lemmas_common = _window_sizes(ctx)
    chosen = set(take(newest, lemmas_all))
    chosen.update(take([lid for lid in newest if owner_of(lid) == ctx.user], my_lemmas))
    shown = [lid for lid in reversed(newest) if lid in chosen]

    eligible_set = set(eligible)
    counts: Dict[str, int] = {}
    for lid in shown:
        for dep in dag.lemmas[lid].get("cited_lemmas", []):
            if dep in eligible_set and dep not in chosen:
                counts[dep] = counts.get(dep, 0) + 1
    common = take(sorted(counts, key=lambda d: (-counts[d], _tiebreak(d))),
                  lemmas_common)
    chosen.update(common)
    index = sorted(lid for lid in eligible if lid not in chosen)
    return shown, common, index


def _tiebreak(lemma_id: str) -> bytes:
    """A lemma's place among equally cited lemmas: random, from the run's
    seed, but the same for the whole run — so a tie does not reshuffle the
    prompt from one call to the next."""
    return hashlib.sha256(f"{_TIEBREAK_SEED}:{lemma_id}".encode("utf-8")).digest()


def _index_note(common: List[str], index: List[str]) -> str:
    """What an agent is told about the lemma window, beside it in the view:
    one sentence per list the view holds. It names no tool — a tool's own
    hint does that, and only while the tool is offered (tools/)."""
    parts = ["proved_lemmas gives the statements of the most recently "
             "proved lemmas."]
    if common:
        parts.append("common_lemmas gives the statements of older lemmas "
                     "that those cite most often.")
    if index:
        parts.append("other_proved_lemmas lists the ids of every other "
                     "proved lemma, without statements; they are proved "
                     "lemmas all the same and may be cited.")
    return " ".join(parts)


def _entry(dag: Any, lemma_id: str, with_citations: bool = False) -> Dict[str, Any]:
    node = dag.lemmas[lemma_id]
    entry: Dict[str, Any] = {"id": lemma_id, "statement": str(node.get("statement", ""))}
    if with_citations:
        entry["cited_lemmas"] = list(node.get("cited_lemmas", []))
        entry["cited_references"] = list(node.get("cited_references", []))
    return entry


def _view(ctx, dag: Any, suspended: Optional[Collection[str]],
          with_citations: bool) -> Dict[str, Any]:
    shown, common, index = lemma_window(ctx, dag, suspended)
    view: Dict[str, Any] = {
        "proved_lemmas": [_entry(dag, lid, with_citations) for lid in shown]
    }
    if common:
        view["common_lemmas"] = [_entry(dag, lid) for lid in common]
    if index:
        view["other_proved_lemmas"] = index
    if common or index:
        view["note"] = _index_note(common, index)
    return view


def planner_dag_view(
    ctx, dag: Any, suspended: Optional[Collection[str]] = None,
) -> Dict[str, Any]:
    """What the planning agents (and the reviser) see of the DAG: the
    statements and the citation structure, never the proofs. Each lemma
    is listed by its id with its statement, the ids it cites and the
    references it cites.

    The suspended lemmas are not in the view: a refuted or uncertified
    lemma is not a proved lemma, and the planner must not be told it is
    one (see suspension.py). Only the lemma window is listed in full; the
    rest of the proved lemmas are in other_proved_lemmas by id."""
    return _view(ctx, dag, suspended, with_citations=True)


def prover_context(
    ctx, dag: Any, suspended: Optional[Collection[str]] = None,
) -> Dict[str, Any]:
    """Everything the prover is allowed to cite: every proved statement, by
    the id it is cited by — never a proof (the prover sees that a lemma
    holds, not how), since shipping every proof would grow without bound
    against the context window.

    The suspended lemmas are not in the view: the prover may not build on
    a lemma that is not to be built on, and a proof that cites one would
    inherit its suspension (see suspension.py). Only the lemma window is
    listed in full; the rest are in other_proved_lemmas by id."""
    return _view(ctx, dag, suspended, with_citations=False)


def verifier_context(
    dag: Any,
    cited_lemmas: List[str],
    cited_references: List[str],
    references: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Statements of exactly the results the prover claims to have used,
    keyed by id: lemmas and references in one set, statements only.

    Deliberately not the whole DAG or the whole reference collection: the
    verifier agents are asked to reject the use of results not in the
    provided set, which only bites if the set is the prover's declared
    citations — a proof leaning on a result it never declared then reads as
    an unjustified leap, which is what it is."""
    statements: Dict[str, Dict[str, str]] = {}
    for lid in cited_lemmas:
        if lid in dag.lemmas:
            statements[lid] = {"statement": str(dag.lemmas[lid].get("statement", ""))}
    ref_by_id = {str(r.get("id") or ""): r for r in references}
    for rid in cited_references:
        ref = ref_by_id.get(rid)
        if ref is not None:
            statements[rid] = {"statement": str(ref.get("formal statement", ""))}
    return {"cited_results": statements}


def computations_block(
    ctx,
    node: Dict[str, Any], sources: Optional[Dict[str, Path]] = None,
) -> str:
    """The computations a proof cites, as the verifiers are shown them
    (computations.context): "" for a proof with none."""
    shown = computations.context(Path(ctx.paths.conjecture_root), node, sources)
    if not shown:
        return ""
    return (
        "\n\nComputations the proof cites (each script exactly as it ran, "
        "and the RESULT the harness recorded from it; the proof refers to "
        "each as [code:<id>]. They were run once, by the prover's harness, "
        "and are not run again for you):\n"
        + json.dumps(shown, indent=2)
    )


def _fit_prompt(
    ctx,
    role: str,
    system_prompt: str,
    build: Callable[[], Tuple[str, Optional["tools_mod.ToolSet"]]],
    verbose: bool,
    dag: Any,
    suspended: Optional[Collection[str]] = None,
    parts: Optional[Dict[str, str]] = None,
) -> Tuple[str, Optional["tools_mod.ToolSet"]]:
    """An agent's (user prompt, tools), fitted to the context window.

    build() makes the prompt and the tools for the window sizes in effect:
    it calls planner_dag_view or prover_context, and agent_tools, so the
    prompt and the tools always see the same window (agent_tools offers
    lookup_lemmas once anything is left out of it). The lemma window is the
    part of a prompt that can give way without losing anything: a lemma
    moved from the full listing into the id index is one lookup away.

    The configured sizes are tried first, and a prompt that fits is
    returned exactly as built. One whose estimate leaves less than the
    reply's reserve plus LOOKUP_ROOM is rebuilt with all three sizes
    halved (15 -> 7 -> 3 -> 1 -> 0; -1, no limit, halves from the DAG's
    size) under ctx.loop.window_override, which is per thread, until one fits; the
    last build is kept when none does — reason() then closes the tools if
    the prompt leaves only the reserve, and returns "too_large", unsent,
    if it leaves less than COMPACT_MIN_ROOM. In that case the parts of the
    prompt are logged by size (`parts` names the call site's fixed pieces
    — the conjecture, the references block, the lemma or proof being
    worked on), since then something other than the window is too large.
    """
    num_ctx = ctx.reasoning_options["num_ctx"]
    limit = num_ctx - _reserve(num_ctx) - _lookup_room(num_ctx)

    def estimate(user_prompt: str, tools: Optional["tools_mod.ToolSet"]) -> int:
        extra = ""
        if tools:
            # What reason() adds while it offers the tools: the specs or the
            # protocol, and the hints on the task.
            extra = (json.dumps(tools.specs()) if _native_tools(ctx)
                     else tools.json_protocol()) + tools.prompt_hints()
        return _estimate_tokens(ctx, [
            {"content": system_prompt}, {"content": user_prompt},
            {"content": extra},
        ])

    def build_with(sizes: Tuple[int, int, int]):
        saved = ctx.loop.window_override
        ctx.loop.window_override = sizes
        try:
            return build()
        finally:
            ctx.loop.window_override = saved

    configured = _window_sizes(ctx)
    sizes = configured
    user_prompt, tools = build()
    used = estimate(user_prompt, tools)
    total = len(dag.lemmas)
    while used > limit and any(sizes):
        sizes = tuple((total if n < 0 else n) // 2 for n in sizes)
        user_prompt, tools = build_with(sizes)
        used = estimate(user_prompt, tools)

    def shown(n: int) -> str:
        return "all" if n < 0 else str(n)

    if sizes != configured:
        ctx.ui.log(
            f"  {role}: showing {shown(sizes[0])}+{shown(sizes[1])} lemmas "
            f"in full and {shown(sizes[2])} common instead of "
            f"{shown(configured[0])}+{shown(configured[1])} and "
            f"{shown(configured[2])} to fit the {num_ctx:,}-token window "
            f"(the rest stay available by lookup).",
            verbose,
        )
    if _room(used, num_ctx) < COMPACT_MIN_ROOM:
        # Only reached with every size at 0: the room limit is below
        # half the window, so the loop above ran to the end.
        saved = ctx.loop.window_override
        ctx.loop.window_override = sizes
        try:
            _shown, _common, index = lemma_window(ctx, dag, suspended)
        finally:
            ctx.loop.window_override = saved
        sized = {"the system prompt": system_prompt, **(parts or {})}
        if index:
            sized["the id index"] = json.dumps(index, indent=2)
        ratio = max(ctx.model.backend.chars_per_token(), 1.0)
        largest = sorted(
            ((label, int(len(text) / ratio)) for label, text in sized.items()
             if len(text) >= ratio),
            key=lambda item: -item[1],
        )
        ctx.ui.log(
            f"  {role}: the prompt (~{used:,} tokens) does not fit the "
            f"{num_ctx:,}-token window even with no lemmas in full; its "
            f"largest parts: "
            + ", ".join(f"{label} ~{n:,}" for label, n in largest)
            + ".",
            verbose,
        )
    return user_prompt, tools


def candidate(ctx, raw: Any, conjecture: str) -> Optional[Dict[str, Any]]:
    """A proposed lemma as the run takes it: a new id (new_lemma_id) and
    the statement, pinned when the entry settles the conjecture
    (resolution.pin). None for an entry without a statement: dropped, not
    repaired. Whatever id the model wrote, and any field the run does not
    use (a dependencies list a planner volunteers must not pre-empt the
    prover), is left behind."""
    if not isinstance(raw, dict):
        return None
    statement = str(raw.get("statement") or "").strip()
    if not statement:
        return None
    pinned = resolution.pin(
        {"statement": statement, "settles": raw.get("settles")}, conjecture)
    return {"id": new_lemma_id(ctx), **pinned}


def planner_candidates(ctx, planner_res: Dict[str, Any], conjecture: str) -> List[Dict[str, Any]]:
    """Normalise a planner reply into a list of candidate lemmas, each
    under a new id (candidate()), at most PLANNER_CANDIDATES of them.

    Accepts the single-lemma shape as well, so a conjecture directory holding
    an old per-conjecture planner.md override keeps working: it simply gets a
    shortlist of length one, and every path below treats that as a shortlist
    that happens to be short.
    """
    raw = planner_res.get("candidate_lemmas")
    if not isinstance(raw, list):
        single = planner_res.get("next_lemma")
        raw = [single] if isinstance(single, dict) else []
    out: List[Dict[str, Any]] = []
    for item in raw:
        cand = candidate(ctx, item, conjecture)
        if cand is not None:
            out.append(cand)
        if len(out) >= PLANNER_CANDIDATES:
            break
    return out


def _wrap(text: str, indent: str = "      ", width: int = 78) -> str:
    return textwrap.fill(
        " ".join(str(text).split()),
        width=width,
        initial_indent=indent,
        subsequent_indent=indent,
    )


def render(
    shortlist: Sequence[Tuple[Dict[str, Any], List[str]]],
    plan_summary: str = "",
) -> str:
    """The candidate list, printed the same way in both modes: each
    candidate with its notes (a parallel run's "claimed by loop-1").

    Automation logs it so that a run you walk away from still leaves a record
    of the four roads not taken.
    """
    out: List[str] = []
    if plan_summary:
        out.append(_wrap(f"Strategy: {plan_summary}", indent="  "))
    for i, (cand, notes) in enumerate(shortlist, start=1):
        flag = "  " + "; ".join(notes) if notes else ""
        out.append(f"  [{i}] {cand.get('id', '?')}{flag}")
        out.append(_wrap(cand.get("statement", "(no statement)")))
    return "\n".join(out)


def log_candidates(
    ctx,
    candidates: List[Dict[str, Any]],
    plan_summary: str,
    verbose: bool,
) -> None:
    """Print the shortlist in automation too, not just when a human is asked.

    A run you walked away from should still leave a record of the four roads
    not taken; without it the log shows a decision and none of the choice.
    """
    ctx.ui.log("Planner shortlist:", verbose)
    ctx.ui.log(render([(c, []) for c in candidates], plan_summary), verbose)


def select_lemma(ctx, usable: List[Dict[str, Any]], verbose: bool) -> Dict[str, Any]:
    """Pick, uniformly at random, one of the candidates."""
    chosen = random.choice(usable)
    if len(usable) > 1:
        ctx.ui.log(f"Picked {chosen['id']} at random from {len(usable)} candidates.", verbose)
    return chosen


def _build_reference_blocks(
    ctx,
    verbose: bool,
) -> Tuple[List[Dict[str, Any]], str, str, str]:
    """The reference context the prompts carry, built once per run.

    The prover gets the full collection unconditionally — it is the tool
    that cites — so it is rendered once here and reused on every attempt.
    The serial run and the parallel threads share these strings; they are
    read-only once built.
    """
    references = ctx.dag.references
    prover_refs = [
        {
            "id": r["id"],
            "slogan": r.get("slogan", ""),
            "formal statement": r.get("formal statement", ""),
        }
        for r in references
    ]
    reference_block = (
        "\n\nKnown references (theorem-level results from the parsed "
        "collection; you may cite any of them in cited_references by id "
        "without proving them yourself):\n"
        + json.dumps(prover_refs, indent=2)
        if prover_refs
        else ""
    )
    # The planner and the reviser share one short view of the collection:
    # below PLANNER_REFERENCE_LIMIT they get an id + slogan per result, at or
    # above it none. See the limit's comment for why a blunt rule beats a
    # graded one here.
    short_refs = [
        {
            "id": r["id"],
            "slogan": r.get("slogan", ""),
        }
        for r in references
    ]
    if references:
        if len(references) < PLANNER_REFERENCE_LIMIT:
            planner_ref_block = (
                "\n\nKnown references (named theorem-level results the prover "
                "may cite by id; the prover sees their full statements):\n"
                + json.dumps(short_refs, indent=2)
            )
            reviser_ref_block = (
                "\n\nKnown references (named theorem-level results from the "
                "parsed collection; treat them as established):\n"
                + json.dumps(short_refs, indent=2)
            )
        else:
            planner_ref_block = ""
            reviser_ref_block = ""
            ctx.ui.log(
                f"{len(references)} references parsed; that is at or "
                f"above the planner limit ({PLANNER_REFERENCE_LIMIT}), so the "
                f"planner and the reviser get none. The prover sees them all."
            )
    else:
        planner_ref_block = ""
        reviser_ref_block = ""
    return references, reference_block, planner_ref_block, reviser_ref_block


def _build_comments_block(ctx) -> str:
    """comments.md, if the operator left one: free-form notes on possible
    approaches to a proof or counterexample, handed to the planner verbatim
    on every iteration. The planner is the only agent that reads it, and an
    absent file degrades to the pre-comments prompt, the way the reference
    blocks do."""
    comments = load_file(ctx.paths.comments_file)
    return (
        "\n\nHuman comments (the operator's notes on possible approaches to a "
        "proof or counterexample; suggestions to weigh, not instructions):\n"
        + comments
        if comments
        else ""
    )
