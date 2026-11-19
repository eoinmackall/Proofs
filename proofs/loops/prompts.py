"""Prompt assembly: the lemma window, the agents' views of the DAG and the
references, and fitting a prompt into the context window.
"""

from __future__ import annotations

import json
import random
import textwrap
from pathlib import Path
from typing import Any, Callable, Collection, Dict, List, Optional, Sequence, Tuple

from .. import tools as tools_mod
from ..core import resolution
from ..core.dag import load_file, new_lemma_id
from ..tools.run_code import computations
from . import strategies
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


def agent_tools(
    ctx,
    role: str,
    dag: Any,
    suspended: Optional[Collection[str]] = None,
) -> Optional["tools_mod.ToolSet"]:
    """The tools an agent shown the lemma window is offered (tools/): today
    lookup_lemmas and search_lemmas, over the same lemmas the window chose
    from, when the window leaves some out; and run_code under the role's
    hint, when the run was given --tools and the role has one. None when no
    tool offers itself, so a small DAG's prompts carry no tools."""
    shown, hidden = lemma_window(ctx, dag, suspended)
    return tools_mod.offered(ctx, tools_mod.Call(
        role=role, lemmas=dag.lemmas, suspended=set(suspended or ()),
        dag=dag, window=set(shown), hidden=hidden,
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
        + ("its" if ctx.settings.max_proof_attempts < 0
           else str(ctx.settings.max_proof_attempts))
        + f" prover rounds without a proof the verifiers "
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
) -> Tuple[List[str], int]:
    """Which proved lemmas the agent is shown in full, and how many it is
    not: the window its planner or prover chose (strategies.py,
    ctx.loop.window) — chosen here, for the agent alone, when none was —
    with each part cut to _fit_prompt's sizes while it rebuilds a prompt to
    fit (ctx.loop.window_override), and without a lemma suspended since.

    The agents reach the lemmas not shown through the tools (search_lemmas,
    lookup_lemmas), never through a list in the prompt: a list that grows
    with the DAG would outgrow any context window.
    """
    suspended = set(suspended or ())
    if ctx.loop.window is None:
        ctx.loop.window = strategies.choose(ctx, dag, suspended, "")
    shown = ctx.loop.window.shown(dag, suspended, ctx.loop.window_override)
    eligible = len(dag.lemmas) - sum(1 for lid in suspended if lid in dag.lemmas)
    return shown, max(eligible - len(shown), 0)


def _index_note(hidden: int) -> str:
    """What an agent is told about the lemma window, beside it in the view.
    It names no tool — a tool's own hint does that, and only while the
    tool is offered (tools/) — and says nothing of how the lemmas shown
    were chosen."""
    return (f"{hidden:,} other proved lemma(s) are not listed here; they are "
            "proved lemmas all the same and may be cited, as may every lemma "
            "a cited_lemmas list names.")


def _entry(dag: Any, lemma_id: str, with_references: bool = False) -> Dict[str, Any]:
    """A lemma as the window shows it: its id, its statement and the ids of
    the lemmas it cites; with_references, the references it cites too."""
    node = dag.lemmas[lemma_id]
    entry: Dict[str, Any] = {
        "id": lemma_id,
        "statement": str(node.get("statement", "")),
        "cited_lemmas": list(node.get("cited_lemmas", [])),
    }
    if with_references:
        entry["cited_references"] = list(node.get("cited_references", []))
    return entry


def _view(ctx, dag: Any, suspended: Optional[Collection[str]],
          with_references: bool) -> Dict[str, Any]:
    shown, hidden = lemma_window(ctx, dag, suspended)
    view: Dict[str, Any] = {
        "proved_lemmas": [_entry(dag, lid, with_references) for lid in shown]
    }
    if hidden:
        view["note"] = _index_note(hidden)
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
    one (see suspension.py). Only the lemma window is listed; the rest of
    the proved lemmas are found with the tools."""
    return _view(ctx, dag, suspended, with_references=True)


def prover_context(
    ctx, dag: Any, suspended: Optional[Collection[str]] = None,
) -> Dict[str, Any]:
    """What the prover sees of what it may cite: the lemma window's
    statements, each by the id it is cited by, with the ids it cites —
    never a proof (the prover sees that a lemma holds, not how), since
    shipping every proof would grow without bound against the context
    window.

    The suspended lemmas are not in the view: the prover may not build on
    a lemma that is not to be built on, and a proof that cites one would
    inherit its suspension (see suspension.py). Only the lemma window is
    listed; the rest of the proved lemmas are found with the tools."""
    return _view(ctx, dag, suspended, with_references=False)


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
    parts: Optional[Dict[str, str]] = None,
) -> Tuple[str, Optional["tools_mod.ToolSet"]]:
    """An agent's (user prompt, tools), fitted to the context window.

    build() makes the prompt and the tools for the window sizes in effect:
    it calls planner_dag_view or prover_context, and agent_tools, so the
    prompt and the tools always see the same window (agent_tools offers
    lookup_lemmas and search_lemmas once anything is left out of it). The
    lemma window is the part of a prompt that can give way without losing
    anything: a lemma left out of it is one search away, and its id stays
    in the cited_lemmas of the lemmas shown.

    The window's own sizes are tried first, and a prompt that fits is
    returned exactly as built. One whose estimate leaves less than the
    reply's reserve plus LOOKUP_ROOM is rebuilt with each of the window's
    parts halved (50 -> 25 -> 12 -> ... -> 0), keeping the lemmas its
    strategy preferred, under ctx.loop.window_override, which is per
    thread, until one fits; the
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

    def build_with(sizes: Tuple[int, ...]):
        saved = ctx.loop.window_override
        ctx.loop.window_override = sizes
        try:
            return build()
        finally:
            ctx.loop.window_override = saved

    user_prompt, tools = build()
    used = estimate(user_prompt, tools)
    # The window build() was shown (lemma_window chooses one if none was).
    configured = ctx.loop.window.sizes() if ctx.loop.window is not None else ()
    sizes = configured
    while used > limit and any(sizes):
        sizes = tuple(n // 2 for n in sizes)
        user_prompt, tools = build_with(sizes)
        used = estimate(user_prompt, tools)
    if sizes != configured:
        ctx.ui.log(
            f"  {role}: showing at most {sum(sizes)} lemmas instead of "
            f"{sum(configured)} to fit the {num_ctx:,}-token window (the "
            f"rest stay available by search and lookup).",
            verbose,
        )
    if _room(used, num_ctx) < COMPACT_MIN_ROOM:
        # Only reached with every size at 0: the room limit is below
        # half the window, so the loop above ran to the end.
        sized = {"the system prompt": system_prompt, **(parts or {})}
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
    (resolution.pin) — so a settling entry may leave its statement empty.
    None for any other entry without a statement: dropped, not repaired.
    Whatever id the model wrote, and any field the run does not use (a
    dependencies list a planner volunteers must not pre-empt the prover),
    is left behind."""
    if not isinstance(raw, dict):
        return None
    statement = str(raw.get("statement") or "").strip()
    pinned = resolution.pin(
        {"statement": statement, "settles": raw.get("settles")}, conjecture)
    if not pinned["statement"]:
        return None
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
