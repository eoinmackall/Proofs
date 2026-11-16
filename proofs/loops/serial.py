"""The serial run: plan, pick, prove, repeat."""

from __future__ import annotations

import itertools
import json
from typing import Any, Dict, List, Optional, Tuple

from .. import tools as tools_mod
from ..core import resolution
from ..core.checkpoint import (
    discard_checkpoint, drop_rejection, load_checkpoint, note_rejection,
    save_checkpoint,
)
from ..core.dag import (
    _committed_in_run, _committed_uncertified, _settled, _uncertified_message,
    _warn_suspended, commit_lemma_to_dag, load_file, new_lemma_id,
)
from ..core.interrupts import _commit_section
from ..validity.steps import VERIFIER_AGENTS
from .human import _operator_citation_check, _verify_operator_proof, choose
from .json_repair import parse_json_or_none
from .mode import check_mode_toggle
from .prompts import (
    _build_comments_block, _build_reference_blocks, _fit_prompt, agent_tools,
    log_candidates, planner_candidates, planner_dag_view, rejected_block,
    replan_note, select_lemma,
)
from .proof import run_proof_loop
from .reasoning import extract, reason
from .schemas import PLANNER_SCHEMA


def _of_budget(ctx, iteration: int) -> str:
    """An iteration as the log shows it: against the budget, when there is
    one."""
    if ctx.settings.unlimited_iterations():
        return str(iteration)
    return f"{iteration}/{ctx.settings.max_iterations}"


def _serial_on_round(
    ctx,
    iteration: int,
    rejected_lemmas: List[Dict[str, str]],
    state: Dict[str, Any],
    verbose: bool,
) -> None:
    """The serial round boundary: publish the in-flight state into
    ctx.run.live_state (the SIGINT handler's source) and into the checkpoint, so a
    Ctrl-C mid-round re-runs exactly that round."""
    ctx.run.live_state["in_flight"] = state
    save_checkpoint(
        ctx, iteration, rejected_lemmas, state,
        plan=ctx.run.live_state.get("plan"), verbose=verbose,
    )


def _serial_on_proof(
    ctx,
    lemma_id: str, node: Dict[str, Any], verified_hash: Optional[str],
    verbose: bool,
) -> str:
    """The serial success boundary: commit the lemma through the same
    check-and-write as the parallel run, so the engine's on_proof
    contract — return the id the lemma was committed under — holds in both
    modes."""
    with _commit_section(ctx):
        # All three verifier checks accepted this proof: the commit
        # certifies it, then writes it, the way the parallel on_proof's
        # does.
        return commit_lemma_to_dag(ctx, lemma_id, node, verified_hash, verbose)


def run_loop(ctx, verbose: bool = True) -> Dict[str, Any]:
    """Execute the multi-agent theorem proving loop.

    Args:
        verbose: If True, print progress to the console.

    Returns:
        The final DAG, so callers running with --no-verbose still get a result.
    """

    conjecture = load_file(ctx.paths.conjecture_file)
    if not conjecture:
        ctx.ui.log(f"{ctx.paths.conjecture_file} is empty — write your conjecture there first.", verbose)
        return ctx.dag

    planner_sys = load_file(ctx.paths.prompt_paths["planner.md"])
    prover_sys = load_file(ctx.paths.prompt_paths["prover.md"])
    # One prompt per verification step; VERIFIER_AGENTS names the three.
    verifier_sys = {name: load_file(ctx.paths.prompt_paths[name]) for name in VERIFIER_AGENTS}
    # One reviser for both failures: a rejected proof, and a prover that
    # overflowed the context window. The user message tells it which case it
    # is — a full rejected proof with the verifier's reasoning, or a
    # summarised trace plus the partial answer.
    reviser_sys = load_file(ctx.paths.prompt_paths["reviser.md"])

    empty = [name for name, text in (
        ("planner.md", planner_sys), ("prover.md", prover_sys), *verifier_sys.items(),
        ("reviser.md", reviser_sys),
    ) if not text]
    if empty:
        # A missing system prompt does not crash — it produces an agent with
        # no instructions, which fails in ways that look like model problems.
        ctx.ui.log(f"Missing or empty prompt files: {', '.join(empty)}. Aborting.", verbose)
        return ctx.dag

    references, reference_block, planner_ref_block, reviser_ref_block = (
        _build_reference_blocks(ctx, verbose)
    )
    ref_ids = {str(r["id"]) for r in references}
    comments_block = _build_comments_block(ctx)

    # Checkpoint restore: the DAG is the files as they stood at startup plus
    # the run's own commits (ctx.dag), so a cancelled run leaves only
    # run-level state to restore — where the
    # iteration budget stood, the plan, the planner's reject list, and the
    # lemma that was in flight, at the step it had reached. A missing or
    # corrupt checkpoint degrades to a fresh run, never to a crash.
    cp = load_checkpoint(ctx, verbose)
    if cp is None:
        start_iteration = 1
        rejected_lemmas: List[Dict[str, str]] = []
        in_flight: Optional[Dict[str, Any]] = None
        plan: Optional[Dict[str, Any]] = None
    else:
        start_iteration = cp["iteration"]
        rejected_lemmas = cp["rejected_lemmas"]
        in_flight = cp["in_flight"]
        plan = cp["plan"]
        if in_flight is None:
            ctx.ui.log(
                f"Resuming at iteration {_of_budget(ctx, start_iteration)} "
                f"({len(rejected_lemmas)} rejected lemma(s) on record).",
                verbose,
            )
        else:
            ctx.ui.log(
                f"Resuming at iteration {_of_budget(ctx, start_iteration)}, "
                f"in-flight {in_flight['lemma_id']} at prover round "
                f"{in_flight['attempt']}/{ctx.settings.max_proof_attempts}, "
                f"{in_flight['stage']} step.",
                verbose,
            )
    # The checkpoint's plan stands in for the first planner call of a
    # resumed run (no staleness check: the planner runs again as soon as
    # this plan's lemma is settled).
    saved_plan = plan
    if (not ctx.settings.unlimited_iterations()
            and start_iteration > ctx.settings.max_iterations):
        ctx.ui.log(
            f"\nThe checkpoint stands at iteration {start_iteration}, past "
            f"--max-iterations ({ctx.settings.max_iterations}). Re-run with a larger "
            f"--max-iterations to continue, or --fresh to restart the budget "
            f"(the DAG is kept either way).",
            verbose,
        )
        return ctx.dag

    warned_dangling = set()
    # The suspension warning last told this run: told again only when its
    # counts change.
    warned_suspended: Dict[str, Any] = {}
    ctx.run.live_state["rejected_lemmas"] = rejected_lemmas
    ctx.run.live_state["in_flight"] = in_flight
    ctx.run.live_state["plan"] = plan
    # The operator's reason for rejecting the last shortlist ('replan' at
    # the human menu), for the next planner call only: it is not a
    # rejected lemma, and it is not checkpointed.
    operator_note = ""

    # With run.max_iterations -1 the loop ends only when the conjecture is
    # settled, or the run is stopped.
    for iteration in itertools.count(start_iteration):
        if (not ctx.settings.unlimited_iterations()
                and iteration > ctx.settings.max_iterations):
            break
        ctx.ui.header(iteration=_of_budget(ctx, iteration))
        ctx.ui.log(f"\n--- Iteration {_of_budget(ctx, iteration)} ---", verbose)
        check_mode_toggle(ctx, verbose)
        dag = ctx.dag

        # The suspension, computed from the in-memory DAG's hashes,
        # certificates and refutations (see suspension.py): the lemmas this
        # iteration's planner and prover are hidden from. It is never
        # stored in the DAG — the computation is the only record of it.
        susp = dag.suspension()
        _warn_suspended(ctx, susp, verbose, warned_suspended)
        suspended = susp.suspended

        # Settled? Only a certified lemma stating the conjecture (or its
        # negation) verbatim says so — this run's, or one another user's
        # pulled streams hold (resolution.py). Nothing is stored: the
        # resolution is the lemma, and it lasts exactly as long as the
        # lemma stays unsuspended.
        if _settled(ctx, dag, suspended, conjecture, verbose):
            discard_checkpoint(ctx)
            return dag

        # Citations that no longer resolve dangle every proof that used
        # them, and the verifier would reject those for good reason: a lemma
        # id no stream holds (pruned, or not pulled), or a reference id
        # references.md no longer has (a re-parse dropped it). Counted, not
        # listed (there may be thousands), and told again only when the
        # count grows.
        dangling = {
            key for key, node in dag.lemmas.items()
            if any(d not in dag.lemmas for d in node.get("cited_lemmas", []))
            or any(d not in ref_ids for d in node.get("cited_references", []))
        }
        if len(dangling) > len(warned_dangling):
            warned_dangling = dangling
            ctx.ui.log(
                f"{len(dangling):,} lemma(s) cite a lemma or reference that "
                f"is not there; verification of one will see a citation "
                f"with no statement behind it."
            )

        # Checkpoint: this iteration is now the resumable position. If the
        # previous run was cancelled mid-proof, `in_flight` names the lemma
        # that goes straight back to the prover; otherwise the iteration
        # starts with planning. The proof loop and the end-of-iteration
        # bookkeeping keep it up to date, and the SIGINT handler writes it
        # on Ctrl-C from ctx.run.live_state.
        ctx.run.live_state["iteration"] = iteration
        ctx.run.live_state["in_flight"] = in_flight
        save_checkpoint(ctx, iteration, rejected_lemmas, in_flight,
                        plan=plan, verbose=verbose)

        # An in-flight lemma this run already committed (the run stopped
        # between the commit and the checkpoint) is not resumed: its id is
        # in the DAG, and whatever became of it since is repair's business.
        committed = _committed_in_run(dag, in_flight)
        resuming = in_flight is not None and committed is None
        uncertified = (
            _committed_uncertified(ctx, dag, susp, in_flight)
            if committed is not None else None
        )
        if uncertified is not None:
            ctx.ui.log(_uncertified_message(ctx, uncertified))
            in_flight = None
            ctx.run.live_state["in_flight"] = None
        if resuming:
            ctx.ui.log(
                f"In-flight {in_flight['lemma_id']}: resuming prover round "
                f"{in_flight['attempt']}/{ctx.settings.max_proof_attempts} at its "
                f"{in_flight['stage']} step (what the round had finished is "
                f"kept; only the interrupted call is re-run).",
                verbose,
            )
        elif in_flight is not None:
            # The lemma was accepted between two checkpoints: its proof is
            # already in the DAG, so there is nothing in flight to resume.
            ctx.ui.log(
                f"Checkpoint's in-flight {in_flight['lemma_id']} is already "
                f"in the DAG; planning as usual.",
                verbose,
            )
            in_flight = None
            ctx.run.live_state["in_flight"] = None
        if not resuming:

            # ---------------- Step 1: Planner ----------------
            ctx.status(lemma="", attempt="")
            saved = saved_plan
            saved_plan = None
            if saved is not None:
                # Its candidates keep the ids they were given; one committed
                # before the checkpoint caught up is spent.
                saved["candidate_lemmas"] = [
                    c for c in saved["candidate_lemmas"] if c["id"] not in dag.lemmas
                ]
            if saved is not None and saved["candidate_lemmas"]:
                planner_res = saved
                candidates = list(saved["candidate_lemmas"])
                ctx.ui.log("Using the plan saved in the run's checkpoint.", verbose)
            else:
                def build_planner() -> Tuple[str, Optional["tools_mod.ToolSet"]]:
                    return (
                        f"Conjecture:\n{conjecture}\n\n"
                        f"Proved lemmas so far:\n{json.dumps(planner_dag_view(ctx, dag, suspended), indent=2)}"
                        f"{planner_ref_block}{comments_block}\n\n"
                        f"{rejected_block(ctx, rejected_lemmas)}"
                        + (f"\n\n{operator_note}" if operator_note else ""),
                        agent_tools(ctx, "planner", dag, suspended),
                    )
                planner_user, planner_tools = _fit_prompt(
                    ctx,
                    "planner", planner_sys, build_planner, verbose, dag,
                    suspended,
                    parts={
                        "the conjecture": conjecture,
                        "the references block": planner_ref_block,
                        "the comments": comments_block,
                        "the rejected lemmas": rejected_block(ctx, rejected_lemmas),
                    },
                )
                operator_note = ""
                plan_text, plan_status, _partial = reason(
                    ctx,
                    planner_sys, planner_user, "planner", verbose,
                    tools=planner_tools,
                )
                check_mode_toggle(ctx, verbose)
                if plan_status == "too_large":
                    # Nothing left to shrink, and in a serial run nothing but
                    # this planner moves the DAG: every later iteration would
                    # build the same prompt. Stop, with the checkpoint at this
                    # iteration, rather than spend the budget on unsent calls.
                    ctx.ui.log(
                        "The planner's prompt does not fit the context window "
                        "even with no lemmas shown in full (its largest parts "
                        "are logged above). Stopping the run: re-run with a "
                        "larger window (--num-ctx, and the server's own) or a "
                        "smaller part; the checkpoint keeps this iteration.",
                        verbose,
                    )
                    return ctx.dag
                if plan_status == "ceiling" and not plan_text:
                    ctx.ui.log("Planner exhausted its token budget. Re-planning.", verbose)
                    continue
                if not plan_text:
                    ctx.ui.log("Planner produced nothing. Re-planning next iteration.", verbose)
                    continue

                # planner.md demands raw JSON output; extraction is only the fallback.
                planner_res = parse_json_or_none(plan_text)
                if planner_res is None:
                    planner_res = extract(
                        ctx,
                        "Extract the plan. Copy every candidate lemma the text proposes "
                        "into candidate_lemmas.",
                        plan_text,
                        PLANNER_SCHEMA,
                        "planner",
                        verbose,
                    )
                # Each candidate gets its id here; one that settles the
                # conjecture gets the code's statement, never the
                # planner's, and then competes at selection like any other.
                candidates = planner_candidates(ctx, planner_res, conjecture)
            if not candidates:
                ctx.ui.log("Planner proposed no usable lemma. Re-planning.", verbose)
                continue
            # Checkpoint: the plan, so a cancelled run picks from it
            # instead of planning again.
            plan = {
                "plan_summary": str(planner_res.get("plan_summary") or ""),
                "candidate_lemmas": candidates,
            }
            ctx.run.live_state["plan"] = plan
            save_checkpoint(ctx, iteration, rejected_lemmas, None,
                            plan=plan, verbose=verbose)

            shortlist = [(c, []) for c in candidates]
            summary = str(planner_res.get("plan_summary") or "")

            # ---------------- Step 1b: Selection ----------------
            # The one place the two modes differ, and so the one place MODE is
            # read. Everything after this block runs identically whether the lemma
            # was chosen by a person or at random, which is what keeps
            # --mode auto honest as a control.
            next_lemma: Optional[Dict[str, Any]] = None
            if ctx.run.mode == "human":
                check = _operator_citation_check(dag, ref_ids, suspended)
                choice = choose(
                    ctx,
                    shortlist, lambda: new_lemma_id(ctx), ctx.run.hotkey, summary,
                    check_citations=check,
                )
                # The operator's own proof goes through the verifiers here;
                # a rejected one (or one with no verdict) puts the same
                # menu back, with no re-plan in between.
                own_committed: Optional[str] = None
                while choice.action == "own_proof":
                    verified = _verify_operator_proof(
                        ctx,
                        choice, dag, references, ref_ids, suspended,
                        conjecture, verifier_sys, verbose,
                    )
                    if verified is not None:
                        lid, node, vh = verified
                        own_committed = _serial_on_proof(ctx, lid, node, vh, verbose)
                        drop_rejection(rejected_lemmas, node["statement"])
                        # Proved, so out of the plan a resumed run picks from.
                        plan["candidate_lemmas"] = [
                            c for c in plan["candidate_lemmas"] if c.get("id") != lid
                        ]
                        break
                    choice = choose(
                        ctx,
                        shortlist, lambda: new_lemma_id(ctx), ctx.run.hotkey,
                        summary, check_citations=check,
                    )
                if own_committed is not None:
                    save_checkpoint(ctx, iteration + 1, rejected_lemmas, None,
                                    plan=plan, verbose=verbose)
                    continue
                if choice.action == "quit":
                    ctx.ui.log(
                        "\nStopped by the operator; DAG and checkpoint kept — "
                        "re-run the same command to resume.",
                        verbose,
                    )
                    # Nothing to write: every commit is already on disk,
                    # and rewriting the file from this iteration's snapshot
                    # would undo whatever reached it since (a repair, a
                    # pull).
                    return dag
                if choice.action == "replan":
                    # Told to the next planner call only: the shortlist was
                    # turned down, not tried, so none of it goes on the
                    # reject list.
                    operator_note = replan_note(candidates, choice.notes)
                    ctx.ui.log("Shortlist rejected; asking the planner again.", verbose)
                    continue
                if choice.action == "auto":
                    ctx.run.mode = "auto"
                    ctx.ui.log(f"Automation resumed. {ctx.run.hotkey.hint()}", verbose)
                elif choice.lemma is not None:
                    # A written lemma that settles the conjecture is pinned too.
                    next_lemma = resolution.pin(choice.lemma, conjecture)

            if next_lemma is None:
                # Automatic selection: one candidate, uniformly at random.
                log_candidates(ctx, candidates, summary, verbose)
                next_lemma = select_lemma(ctx, candidates, verbose)

        if not resuming and plan is not None:
            # The lemma leaves the plan as it is picked: a resumed run
            # picks from what is left.
            plan["candidate_lemmas"] = [
                c for c in plan["candidate_lemmas"] if c.get("id") != next_lemma["id"]
            ]
        if resuming:
            lemma_id = in_flight["lemma_id"]
            lemma_stmt = in_flight["target"]["statement"]
            ctx.ui.log(f"Next Lemma [{lemma_id}]: {lemma_stmt}", verbose)
        else:
            lemma_id = next_lemma["id"]
            lemma_stmt = next_lemma["statement"]
            ctx.ui.log(f"Next Lemma [{lemma_id}]: {lemma_stmt}", verbose)

        # ---------------- Steps 2-5: the proof loop ----------------
        # prover -> verifiers -> reviser, repeated within the iteration. The
        # prover writes a proof of `target`; the verifiers then check it —
        # three atomic passes, one per verifier agent — and one reject ends the
        # counting and sends the proof to the reviser with the verdict's
        # reasoning. The reviser then either keeps the statement (the prover
        # re-tries with the verdict as feedback), revises it (the revised
        # statement becomes the new target), or decomposes it into a smaller
        # lemma (judged too hard as stated from the last proof; the prover
        # restarts on the new one with a fresh budget). Only a proof that
        # survives all
        # three verifiers enters the DAG; after run.max_proof_attempts prover
        # rounds the lemma is given up for this iteration and the planner is
        # asked again.
        #
        # No human intervention here, in either mode. Choosing 'p' at the menu
        # *is* the decision to let the models settle it; asking again
        # afterwards would put the operator back in the loop they just
        # delegated out of. If you want a say over this lemma, assert it.
        # The shared engine (run_proof_loop, above) owns steps 2-5; the
        # serial run hands it the serial versions of the round and success
        # boundaries: publish the in-flight state into ctx.run.live_state and the
        # checkpoint, and commit proved lemmas to the DAG. The lock is a
        # formality here — one thread — and the engine's contract does not
        # change between the two modes.
        result = run_proof_loop(
            ctx,
            verbose=verbose,
            conjecture=conjecture,
            prover_sys=prover_sys,
            reviser_sys=reviser_sys,
            reference_block=reference_block,
            reviser_ref_block=reviser_ref_block,
            references=references,
            verifier_sys=verifier_sys,
            ref_ids=ref_ids,
            initial=in_flight if resuming else None,
            fresh_target=None if resuming else next_lemma,
            on_round=lambda state: _serial_on_round(
                ctx,
                iteration, rejected_lemmas, state, verbose
            ),
            on_proof=lambda lid, node, vh: _serial_on_proof(
                ctx,
                lid, node, vh, verbose
            ),
            get_dag=lambda: ctx.dag,
            toggle_hotkey=lambda: check_mode_toggle(ctx, verbose),
        )
        lemma_id = result["lemma_id"]
        proved = result["proved"]

        # ---------------- Step 6: Record the outcome for the planner --------
        if proved:
            drop_rejection(rejected_lemmas, result["target"]["statement"])
        elif result["exhausted"]:
            note_rejection(
                ctx,
                rejected_lemmas, lemma_id, result["target"]["statement"]
            )
            ctx.ui.log(
                f"Lemma {lemma_id} left unproved after "
                f"{ctx.settings.max_proof_attempts} prover rounds; the planner will see "
                f"it as rejected.",
                verbose,
            )
            ctx.status(event=f"gave up {lemma_id}")
        else:
            ctx.ui.log(f"Lemma {lemma_id} given up unproved.", verbose)

        # The lemma is settled one way or another: clear the in-flight state
        # and move the resumable position to the next iteration, with the
        # reject list as it now stands.
        in_flight = None
        ctx.run.live_state["in_flight"] = None
        save_checkpoint(ctx, iteration + 1, rejected_lemmas, None,
                        plan=plan, verbose=verbose)

    # The last iteration's lemma may have been the one that settles it.
    dag = ctx.dag
    susp = dag.suspension()
    if _settled(ctx, dag, susp.suspended, conjecture, verbose):
        discard_checkpoint(ctx)
        return dag

    ctx.ui.log(
        f"\nReached run.max_iterations ({ctx.settings.max_iterations}) without settling the "
        f"conjecture (proved or disproved). The checkpoint keeps the budget "
        f"position: re-run with a larger --max-iterations to continue, or "
        f"--fresh to restart it. The DAG is kept either way.",
        verbose,
    )
    return ctx.dag
