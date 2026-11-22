"""The proof loop shared by the serial run and every parallel loop:
prover -> verifiers -> reviser.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .. import tools as tools_mod
from ..core import resolution
from ..core.dag import _verified_hash, new_lemma_id
from ..tools.run_code import computations, files
from ..validity import verdicts
from ..validity.citations import check_citations, scan_citations
from ..validity.steps import role_of, verifier_steps
from ..workspace import settings as settings_mod
from . import citation_checks, strategies
from .json_repair import parse_json_or_none
from .prompts import (
    _fit_prompt, agent_tools, computations_block, planner_dag_view,
    prover_context, verifier_context,
)
from .reasoning import (
    _incomplete_report, _reason_until_answered, _recorded_model, _same_model,
    _wait_for_server, compaction_passes, extract,
)
from .schemas import REVISER_SCHEMA
from .verifiers import _run_verifier, fan_out, stopping, verifier_prompt


def _parse_reviser_decision(ctx, revision_text: str, verbose: bool) -> Dict[str, Any]:
    """Parse a reviser's JSON decision, with an extraction fallback.

    reviser.md demands raw JSON; when the model wraps it in prose, extract()
    recovers the fields. Returns {} when there is nothing to recover — the
    caller treats that as "no usable decision".
    """
    if not revision_text:
        return {}
    res = parse_json_or_none(revision_text)
    if res is None:
        res = extract(
            ctx,
            "Extract the revision decision. action is one of 'keep', "
            "'revise_statement' or 'new_lemma': 'keep' if the statement is "
            "kept unchanged, 'revise_statement' only if the text proposes a "
            "changed statement for the same lemma, 'new_lemma' only if the "
            "text proposes a different, smaller lemma to prove instead. Copy "
            "new_statement verbatim; it is the empty string when the "
            "statement is kept. If the text uses "
            "'statement_revision' instead of 'action', map true to "
            "'revise_statement' and false to 'keep'.",
            revision_text,
            REVISER_SCHEMA,
            "reviser",
            verbose,
        )
    return res or {}


def _restore_failed(ctx, flight, why: str, verbose: bool) -> None:
    """A resumed loop's saved proof does not stand any more: say so. The
    caller re-runs the prover round, at no cost to the budget."""
    citation_checks.recover_checks(ctx, [flight])
    ctx.ui.log(
        f"The saved proof of {flight['lemma_id']} no longer stands ({why}); re-running "
        f"the prover round.",
        verbose,
    )


def run_proof_loop(ctx, **kwargs) -> Dict[str, Any]:
    """_proof_loop (which documents the loop), in the loop's own code
    directory (tools/run_code/files.py): the one the resumed round saved,
    or a new one. The directory goes when the lemma settles — proved, out
    of attempts, or given up — and stays for a loop that stopped or was
    interrupted, whose checkpoint resumes in it."""
    if ctx.code.runner is None:
        return _proof_loop(ctx, **kwargs)
    name = str((kwargs.get("initial") or {}).get("code_dir") or "")
    if not re.fullmatch(r"[0-9a-f]{8}", name):
        name = files.new_loop_dir_name()
    code_dir = ctx.code.runner.scratch_dir / name
    prev, ctx.loop.code_dir = ctx.loop.code_dir, code_dir
    try:
        result = _proof_loop(ctx, **kwargs)
    finally:
        ctx.loop.code_dir = prev
    if not result.get("stopped"):
        shutil.rmtree(code_dir, ignore_errors=True)
    return result


def _as_list(raw: Any) -> List[Any]:
    if not raw:
        return []
    return list(raw) if isinstance(raw, list) else [raw]


def _proof_loop(
    ctx,
    *,
    verbose: bool,
    conjecture: str,
    prover_sys: str,
    reviser_sys: str,
    reference_block: str,
    reviser_ref_block: str,
    references: List[Dict[str, Any]],
    verifier_sys: Dict[str, str],
    ref_ids: Set[str],
    initial: Optional[Dict[str, Any]],
    fresh_target: Optional[Dict[str, Any]],
    on_round,
    on_proof,
    get_dag,
    toggle_hotkey,
    should_stop=None,
    allow_decomposition: bool = True,
    audit_citations: bool = False,
) -> Dict[str, Any]:
    """Steps 2-5 of the proof loop: prover -> verifiers -> reviser, shared
    by the serial run and by every parallel loop.

    `allow_decomposition` keeps the reviser's new_lemma option open: the
    run leaves it on, where a lemma that is too hard as stated is the
    reviser's to decompose. proofs repair turns it off — the repair
    re-proves the named lemma and the lemma keeps its id, so a
    decomposition would commit a different lemma and leave the refutation
    sitting on the one it named. With it off, a new_lemma decision falls
    back to keeping the statement, the way a new_lemma with no statement
    already does; on the overflow path it falls back to giving up
    on the lemma for this pass, since retrying the statement that just
    overflowed would only overflow it again.

    `should_stop`, when given, is polled before every prover round: the
    parallel loops pass the run's stop test so a resolution or a Ctrl-C
    stops a loop between rounds instead of letting it spend the rest of its
    prover budget on a lemma no one will keep. A stop returns the current
    position marked "stopped" — the last on_round already published the
    resumable round, so the caller keeps it for the next run. The serial
    run passes nothing (its stop is the KeyboardInterrupt).

    `initial` is the in-flight state a cancelled prover round left behind
    (None for a fresh start); when given, the engine re-runs that round with
    that feedback instead of starting from attempt 1. `fresh_target` is the
    lemma the caller picked for a fresh start; it is ignored when resuming.

    When audit_citations is enabled (agent-produced proofs in proofs run),
    the post-acceptance stage checkpoints the sampled direct citations and
    passes their completed checks to on_proof for publication with the node.

    The engine never touches the DAG or the checkpoint itself: `on_round` is
    called at each prover round with the in-flight state that round
    establishes (the caller publishes it and checkpoints it), and `on_proof`
    is called with the lemma's DAG node when a proof survives all three
    verifier checks, and with the node's hash as the verifiers checked it
    (_verified_hash), which the caller hands to commit_lemma_to_dag (the
    caller certifies and commits it under the lemma's id and returns it;
    a smaller lemma the reviser decomposes onto gets an id of its own,
    new_lemma_id). The
    reject list is the caller's too: the engine only reports, in
    "exhausted", whether the lemma it ended on spent every prover round.

    The suspension (see suspension.py) is recomputed at every round from
    the DAG get_dag() returns: the prover's context and the reviser's
    view of the DAG omit the suspended lemmas, so the proof cannot build on
    them. A target that restates a suspended lemma's statement is a fresh
    proof, not a build on the suspended lemma: it is committed under a new
    id of its own, and the suspended lemma is lifted only by proofs verify
    or proofs
    repair re-accepting it.

    Returns {"proved", "exhausted", "lemma_id", "committed_id",
    "claimed_id", "target", "proof", "attempt_notes"}. exhausted is True
    when the final target (after any revisions and decompositions) went
    through all run.max_proof_attempts prover rounds unproved — the one outcome
    the caller puts on the reject list; a lemma given up early (an overflow
    the reviser could not rescue) or stopped is not. committed_id is the id the proof was
    committed under (None when not proved); the caller's on_proof decides
    it. claimed_id is the id this claim started from: lemma_id is wherever
    the prover ended up, and a decomposition may have moved it, so the
    caller frees its buffer slot keyed on claimed_id, not lemma_id. The
    caller records the final target when exhausted and clears its
    in-flight state; the engine leaves no other traces.
    """
    resuming = initial is not None
    in_flight = initial if initial is not None else {}
    proof = ""  # bound before the return; the overflow-break paths skip the assignment
    committed_id: Optional[str] = None  # set when on_proof commits

    if resuming:
        lemma_id = in_flight["lemma_id"]
        # The claim id is the id this claim started from, kept apart from
        # the target because a decomposition may have moved the prover onto
        # a different lemma since the claim: the buffer slot to free on
        # settle is keyed on the claim, not on wherever the prover ended
        # up. Checkpoints written before this field existed carry only the
        # target's id; the fallback is the old behavior.
        claimed_id = str(in_flight.get("claimed_id") or lemma_id)
        target: Dict[str, Any] = in_flight["target"]
    else:
        lemma_id = fresh_target["id"]
        claimed_id = lemma_id
        # The target is the statement only: the candidate's other fields
        # (priority, notes) steer the loops, not the prover.
        target: Dict[str, Any] = {
            "id": lemma_id,
            "statement": fresh_target["statement"],
        }
    # The prover sees only the most recent failure: feedback is rebuilt
    # after each rejected round, so a retry reads the last verdict and
    # diagnosis, not the history of every earlier attempt. guidance is the
    # exception: the reviser's diagnosis that came with the current
    # statement (a new lemma or a revised one) — the route, the useful
    # lemmas, the mistakes to avoid — kept for every prover round on that
    # statement, replaced when the reviser sets another, and dropped with
    # the loop: it is never stored with the lemma.
    if resuming:
        feedback: List[str] = list(in_flight.get("feedback") or [])
        guidance = str(in_flight.get("guidance") or "")
        attempt_notes: List[str] = list(in_flight.get("attempt_notes") or [])
        first_attempt = in_flight["attempt"]
        last_proof = str(in_flight.get("last_proof") or "")
        # The scripts the last proof cited (their ids), which the next
        # prover round is shown again (the carry-over), until the
        # statement changes.
        last_code_runs: List[Any] = _as_list(in_flight.get("last_code_runs"))
    else:
        feedback: List[str] = []
        guidance = ""
        attempt_notes: List[str] = []
        first_attempt = 1
        last_proof = ""
        last_code_runs: List[Any] = []
    # A loop resuming past its prover call (stage verify or revise; see
    # core/checkpoint.py): the prover's saved output stands in for the
    # call, once, and the verdicts saved with it for the verifiers that
    # had finished. If the saved proof no longer stands — a citation that
    # stopped resolving, a script gone from the scratch directory — the
    # round is re-run from the prover, at no cost to the lemma's budget.
    restore: Optional[Dict[str, Any]] = None
    if resuming and in_flight.get("stage") in ("verify", "revise", "citations") \
            and in_flight.get("proof"):
        restore = dict(in_flight, prover_model=str(in_flight.get('prover_model') or ''),
                       verdicts=list(in_flight.get('verdicts') or []))
    proved = False
    attempt = first_attempt
    # `attempt` is a while-loop counter rather than a for-range because the
    # reviser can reset it: a lemma it decomposes into a smaller one starts
    # its own prover rounds from 1 (see the new_lemma branch below).
    while settings_mod.within(attempt, ctx.settings.max_proof_attempts):
        if should_stop is not None and should_stop():
            return {
                "proved": False,
                "exhausted": False,
                "lemma_id": lemma_id,
                "committed_id": None,
                "claimed_id": claimed_id,
                "target": target,
                "proof": last_proof,
                "attempt_notes": attempt_notes,
                "stopped": True,
            }
        dag = get_dag()
        # The suspension as this round stands (see suspension.py),
        # recomputed with the DAG rather than taken once per lemma: in a
        # parallel run a sibling's commit, or a certificate that failed to
        # land, changes it between rounds, and a lemma suspended since the
        # last round must not be offered to the prover in this one.
        suspended = dag.suspension().suspended
        # The lemmas this prover round, and the reviser after it, are shown
        # (strategies.py): the planner's in a serial run, chosen now in a
        # parallel loop or a repair.
        strategies.for_prover(ctx, dag, suspended, str(target.get("statement") or ""))
        # Checkpoint: this prover round is now the resumable position, and
        # every step of it below publishes where it got to (stage, the
        # prover's output, each verdict). If the run is cancelled inside
        # it, the next run resumes at that step — the interrupted call is
        # lost, nothing completed before it is.
        saved, restore = restore, None
        restoring = saved is not None
        in_flight_state = {
            "lemma_id": lemma_id,
            "claimed_id": claimed_id,
            "target": target,
            "attempt": attempt,
            "feedback": feedback,
            "guidance": guidance,
            "attempt_notes": attempt_notes,
            "last_proof": last_proof,
            "last_code_runs": last_code_runs,
            "code_dir": ctx.loop.code_dir.name if ctx.loop.code_dir else "",
            "stage": saved["stage"] if restoring else "prove",
            "proof": saved["proof"] if restoring else None,
            "prover_model": saved["prover_model"] if restoring else "",
            "verdicts": saved["verdicts"] if restoring else [],
        }
        if restoring and saved['stage'] == 'citations':
            in_flight_state.update(audit_hash=saved['audit_hash'], checks=saved['checks'])
        on_round(in_flight_state)
        ctx.status(
            lemma=lemma_id,
            attempt=settings_mod.of_limit(attempt, ctx.settings.max_proof_attempts),
        )
        ctx.ui.log(
            f"Proof attempt "
            f"{settings_mod.of_limit(attempt, ctx.settings.max_proof_attempts)} "
            f"for {lemma_id}.",
            verbose,
        )

        # ---------------- Step 2: Prover ----------------
        # prover.md instructs the model to reply with {"lemma_id",
        # "cited_lemmas", "cited_references", "proof"} — the two citation
        # lists are separate: lemma ids, and reference ids. We parse that
        # JSON locally
        # rather than via a second model call, so the proof text can never
        # be abridged or paraphrased; if the model ignored the format, its
        # content is taken verbatim as the proof.
        if restoring:
            ctx.ui.log(
                f"Resuming {lemma_id} at its {saved['stage']} step, with "
                f"the prover's saved output.",
                verbose,
            )
            answer = (json.dumps(saved["proof"]), "", None)
        else:
            # The carry-over: the scripts the last proof cited, shown again
            # so the prover can cite them unchanged or edit them. They are
            # the only files the prover's directory keeps from earlier
            # rounds — no other file's text is in the new call's context.
            carry = ""
            if ctx.loop.code_dir is not None and ctx.loop.code_loop:
                prover_files = files.Scripts(ctx.loop.code_dir / "prover")
                c_entries, c_sources, _problems, c_names = \
                    prover_files.resolve(last_code_runs)
                prover_files.keep_only(c_names)
                carry = files.carried_block(c_entries, c_names, c_sources)

            def build_prover() -> Tuple[str, Optional["tools_mod.ToolSet"]]:
                return (
                    f"Conjecture:\n{conjecture}\n\n"
                    f"Available proved lemmas:\n"
                    f"{json.dumps(prover_context(ctx, dag, suspended), indent=2)}"
                    f"{reference_block}\n\n"
                    f"Lemma to prove:\n{json.dumps(target, indent=2)}\n"
                    + (
                        f"\nReviser's guidance for this statement (it holds for "
                        f"every attempt at it):\n{guidance}\n"
                        if guidance
                        else ""
                    )
                    + (
                        f"\nFeedback from the previous attempt "
                        f"(the most recent rejection only):\n"
                        f"{json.dumps(feedback, indent=2)}"
                        if feedback
                        else ""
                    )
                    + (f"\n\n{carry}" if carry else "")
                    + (f"\n\n{ctx.code.answer_note}" if ctx.code.answer_note else ""),
                    agent_tools(ctx, "prover", dag, suspended),
                )
            prover_user, prover_tools = _fit_prompt(
                ctx,
                "prover", prover_sys, build_prover, verbose,
                parts={
                    "the conjecture": conjecture,
                    "the references block": reference_block,
                    "the lemma": json.dumps(target, indent=2),
                    "the reviser's guidance": guidance,
                    "the feedback": json.dumps(feedback, indent=2) if feedback else "",
                    "the carried computations": carry,
                },
            )
            # An unreachable server is waited out, not counted: the round
            # re-runs from the top (where a stop is noticed) once it answers.
            answer = _reason_until_answered(
                ctx,
                should_stop, verbose,
                prover_sys, prover_user, "prover",
                tools=prover_tools,
            )
        if answer is None:
            continue
        proof_text, prover_status, prover_partial = answer
        if not restoring:
            toggle_hotkey()
        # The model that wrote this proof, for the lemma record: a restored
        # round's is the one saved with its proof, which a resumed run on
        # another model did not write ("" from a checkpoint that predates
        # the field: not known).
        prover_model = saved["prover_model"] if restoring else _recorded_model(ctx)

        if prover_status == "too_large":
            # The prompt, not the lemma, is what does not fit: the lemma
            # was never attempted, so it is neither decomposed nor put on
            # the reject list, and re-sending the same prompt would fail
            # the same way. Given up for this pass, with a note that says
            # why.
            ctx.ui.log(
                f"The prover's prompt for {lemma_id} does not fit the "
                f"context window even with no lemmas shown in full (its "
                f"largest parts are logged above); giving the lemma up for "
                f"now, unattempted.",
                verbose,
            )
            attempt_notes.append(
                f"Prover attempt {attempt}: not attempted — the prover's "
                f"prompt did not fit the context window. This says nothing "
                f"about the lemma itself."
            )
            break

        if prover_status == "ceiling":
            # The prover exhausted the window and the compaction rescue
            # (bounded by llm.max_compaction_passes) could not finish it. The
            # partial work is still worth reading: compact it one last time
            # and let the reviser pick a smaller lemma or revise the
            # statement from it, instead of discarding it and asking the
            # planner to decompose blind.
            report = _incomplete_report(
                ctx,
                "prover", prover_partial or {}, verbose,
            )
            if not report:
                ctx.ui.log(
                    f"Lemma {lemma_id} is too large to prove in one call "
                    f"and left nothing to build on. Asking the planner to "
                    f"decompose it.",
                    verbose,
                )
                attempt_notes.append(
                    "Lemma too large: the prover exhausted its entire token "
                    "budget without completing a proof and left nothing to "
                    "resume from. Decompose this into smaller, "
                    "independently provable lemmas rather than re-proposing "
                    "it."
                )
                break
            ctx.ui.log(
                f"Lemma {lemma_id} overflowed the context window after "
                f"{compaction_passes(ctx)}; asking the "
                f"reviser to pick a smaller lemma or revise the statement "
                f"from the partial proof.",
                verbose,
            )
            def build_overflow() -> Tuple[str, Optional["tools_mod.ToolSet"]]:
                return (
                    f"Conjecture:\n{conjecture}\n\n"
                    f"Proved lemmas so far:\n{json.dumps(planner_dag_view(ctx, dag, suspended), indent=2)}"
                    f"{reviser_ref_block}\n\n"
                    f"Target lemma:\n{json.dumps(target, indent=2)}\n\n"
                    f"{report}",
                    agent_tools(ctx, "reviser", dag, suspended),
                )
            overflow_user, overflow_tools = _fit_prompt(
                ctx,
                "reviser", reviser_sys, build_overflow, verbose,
                parts={
                    "the conjecture": conjecture,
                    "the references block": reviser_ref_block,
                    "the lemma": json.dumps(target, indent=2),
                    "the partial proof": report,
                },
            )
            answer = _reason_until_answered(
                ctx,
                should_stop, verbose,
                reviser_sys, overflow_user, "reviser",
                tools=overflow_tools,
            )
            if answer is None:
                continue
            revision_text, _revision_status, _revision_partial = answer
            toggle_hotkey()
            if _revision_status == "too_large":
                # Not re-sent: it would not fit any better. With no
                # decision the lemma is given up below, the way an
                # overflow the reviser cannot rescue always is.
                ctx.ui.log(
                    "The reviser's prompt for the overflow does not fit the "
                    "context window even with no lemmas shown in full (its "
                    "largest parts are logged above).",
                    verbose,
                )
            revision_res = _parse_reviser_decision(ctx, revision_text, verbose)
            action = str(revision_res.get("action") or "").strip().lower()
            diagnosis = str(revision_res.get("diagnosis") or "").strip()
            new_stmt = str(revision_res.get("new_statement") or "").strip()
            overflow_feedback = [
                f"Prover: the proof of {lemma_id} overflowed the context "
                f"window after {compaction_passes(ctx)}; "
                f"the reviser is choosing a smaller lemma or a corrected "
                f"statement from the partial work."
            ]
            # The diagnosis rides along only where the reviser's choice
            # takes it: the guidance of a new or revised statement, or the
            # notes of a lemma given up on.
            diagnosis_note = (
                [f"Reviser's diagnosis: {diagnosis}"] if diagnosis else []
            )
            if allow_decomposition and action == "new_lemma" and new_stmt:
                # The reviser found a smaller lemma worth trying. Set the
                # overflowing lemma aside and start the prover on the
                # smaller lemma with a fresh budget and no inherited
                # verdict.
                old_id = lemma_id
                lemma_id = new_lemma_id(ctx)
                target = {"id": lemma_id, "statement": new_stmt}
                attempt_notes = [
                    f"Overflowed from {old_id}: the prover ran out of the "
                    f"context window; the reviser is trying the smaller "
                    f"{lemma_id} instead."
                ]
                feedback = []
                guidance = diagnosis
                last_proof = ""
                last_code_runs = []
                attempt = 1
                ctx.ui.log(
                    f"Prover overflowed {old_id}; now trying the smaller "
                    f"lemma {lemma_id}:\n{new_stmt}",
                    verbose,
                )
                continue
            if (
                action == "revise_statement"
                and new_stmt
                and new_stmt != target["statement"]
                # the conjecture's statement is pinned, not revisable
                and not resolution.is_pinned(target["statement"], conjecture)
            ):
                # The partial work shows the statement is false or badly
                # posed, and the reviser has corrected it. The lemma keeps
                # its id and its attempt budget — the overflowing round
                # still counts — and the prover re-runs on the correction
                # with this round's feedback, the way a rejected-proof
                # revision does.
                target = {"id": lemma_id, "statement": new_stmt}
                last_proof = ""
                last_code_runs = []
                overflow_feedback.append(
                    f"Revised statement proposed: {new_stmt}"
                )
                feedback = overflow_feedback
                guidance = diagnosis
                attempt_notes.extend(overflow_feedback + diagnosis_note)
                attempt += 1
                ctx.ui.log(
                    f"Reviser revised the lemma after the overflow; the "
                    f"prover starts again from:\n{new_stmt}",
                    verbose,
                )
                continue
            if allow_decomposition:
                ctx.ui.log(
                    "Reviser could not pick a usable smaller lemma or "
                    "revised statement from the partial proof; the planner "
                    "will decompose.",
                    verbose,
                )
                attempt_notes.extend(overflow_feedback + diagnosis_note)
                attempt_notes.append(
                    "Lemma too large: the prover exhausted its context "
                    "window and the reviser could neither split it further "
                    "nor revise its statement. Decompose into smaller, "
                    "independently provable lemmas."
                )
            else:
                ctx.ui.log(
                    "Reviser could not pick a usable revised statement "
                    "from the partial proof, and decomposition is not "
                    "allowed here.",
                    verbose,
                )
                attempt_notes.extend(overflow_feedback + diagnosis_note)
                attempt_notes.append(
                    "Lemma too large: the prover exhausted its context "
                    "window and the reviser could not revise its statement "
                    "(decomposition is not allowed here)."
                )
            break

        proof = proof_text
        cited_lemmas: List[str] = []
        cited_refs: List[str] = []
        prover_res = parse_json_or_none(proof_text)
        raw_cited_lemmas: Any = None
        raw_cited_refs: Any = None
        raw_code_runs: Any = None
        if prover_res is not None:
            candidate = prover_res.get("proof")
            if isinstance(candidate, str) and candidate.strip():
                proof = candidate.strip()
            raw_cited_lemmas = prover_res.get("cited_lemmas")
            raw_cited_refs = prover_res.get("cited_references")
            raw_code_runs = prover_res.get("code_runs")

        if not proof and restoring:
            _restore_failed(ctx, in_flight_state, "it holds no proof", verbose)
            continue
        if not proof:
            # The model's failure, not the server's (an outage was waited
            # out above): it spends the round, and the next one is told.
            ctx.ui.log(f"Prover produced no proof for {lemma_id}.", verbose)
            feedback = [
                "Your previous attempt returned no proof at all. Write the "
                "complete proof, in the format asked for."
            ]
            attempt_notes.append(
                f"Prover attempt {attempt}: the prover returned no proof."
            )
            attempt += 1
            continue

        # The last proof of this lemma: what the verifiers are about to
        # check, what the reviser will judge its difficulty by, and what
        # the checkpoint records beside the in-flight state.
        last_proof = proof
        last_code_runs = _as_list(raw_code_runs)

        # The DAG's edges now come from here, in the two stored fields:
        # cited_lemmas, a list of lemma ids, and cited_references, a list
        # of reference ids. A response with neither list means the
        # prover ignored its format, and scanning the text beats recording
        # a lemma as standing on nothing; the scan only ever names
        # something that exists, so what it recovers cannot be a phantom.
        if not (
            isinstance(raw_cited_lemmas, list)
            or isinstance(raw_cited_refs, list)
        ):
            cited_lemmas, cited_refs = scan_citations(
                proof, dag, references, suspended
            )
            if cited_lemmas or cited_refs:
                ctx.ui.log(
                    f"  Prover declared no citations; recovered "
                    f"{', '.join(str(d) for d in [*cited_lemmas, *cited_refs])} "
                    f"from the proof text.",
                    verbose,
                )
        else:
            # Every cited pair is checked against the complete DAG and
            # every id against the reference collection; whatever matches
            # nothing is sent back to the prover as feedback and the round
            # is spent, so a proof that claims a result nobody has proved
            # never reaches the verifiers to be rejected there instead.
            cited_lemmas, cited_refs, unmatched = check_citations(
                raw_cited_lemmas, raw_cited_refs, dag, ref_ids,
                # A proof resumed at its citation checks already passed its
                # verifiers with these edges, and a suspension since (perhaps
                # from its own recheck) is inherited at commit. Any earlier
                # step re-checks them, so a newly suspended citation sends
                # the round back to the prover.
                () if audit_citations and restoring and saved['stage'] == 'citations' else suspended,
            )
            if unmatched and restoring:
                _restore_failed(
                    ctx, in_flight_state,
                    "it cites " + ", ".join(label for label, _ in unmatched)
                    + ", which can no longer be cited",
                    verbose,
                )
                continue
            if unmatched:
                unmatched_labels = [label for label, _ in unmatched]
                suspended_labels = [
                    label for label, kind in unmatched if kind == "suspended"
                ]
                unknown_labels = [
                    label for label, kind in unmatched if kind != "suspended"
                ]
                if unknown_labels:
                    ctx.ui.log(
                        f"  Prover cited {len(unknown_labels)} result(s) "
                        f"that match no lemma and no reference: "
                        f"{', '.join(unknown_labels)} — sent back as "
                        f"feedback.",
                        verbose,
                    )
                if suspended_labels:
                    ctx.ui.log(
                        f"  Prover cited {len(suspended_labels)} "
                        f"suspended lemma(s): {', '.join(suspended_labels)} "
                        f"— not to be built on; sent back as feedback.",
                        verbose,
                    )
                missing_refs = [
                    label for label, kind in unmatched
                    if kind == "reference"
                ]
                if missing_refs:
                    ctx.ui.log(
                        f"  none of these is in "
                        f"{os.path.basename(ctx.paths.references_file)}: "
                        f"{', '.join(missing_refs)} — this run cannot add "
                        f"references. Request the missing one(s) from the "
                        f"conjecture's maintainer, who keeps the file with "
                        f"proofs parse and commits it.",
                        verbose,
                    )
                problems = []
                if unknown_labels:
                    problems.append(
                        "cited result(s) that do not exist in the DAG or "
                        "the reference collection: "
                        + ", ".join(unknown_labels)
                    )
                if suspended_labels:
                    problems.append(
                        "cited lemma(s) that are suspended — refuted, "
                        "uncertified, or standing on one that is — and may "
                        "not be built on: "
                        + ", ".join(suspended_labels)
                        + " (prove what you need from them yourself, or "
                        "use other results)"
                    )
                feedback = [
                    "Your proof "
                    + "; and ".join(problems)
                    + ". Cite only lemmas from the 'Available proved "
                    "lemmas' list, each by the exact id that list shows, "
                    "and references from the 'Known references' list by "
                    "id. The two go in separate lists: cited_lemmas and "
                    "cited_references."
                ]
                attempt_notes.append(
                    f"Prover attempt {attempt}: the proof cited "
                    f"{len(unmatched)} unknown or suspended result(s) "
                    f"({', '.join(unmatched_labels)}); the citations were "
                    "sent back unverified and the proof never reached the "
                    "verifiers."
                )
                attempt += 1
                continue

        # The computations the proof rests on: each cited script id read
        # back from the prover's directory (files.Scripts.resolve), so what
        # the node records as a script's output is the harness's record,
        # never the model's. A bad citation is sent back like a
        # bad lemma citation, before any verifier sees the proof.
        code: List[Dict[str, Any]] = []
        code_sources: Dict[str, Path] = {}
        if restoring and not raw_code_runs \
                and computations.unknown_references(proof, []):
            _restore_failed(ctx, in_flight_state, "its computations are not cited", verbose)
            continue
        if not raw_code_runs and computations.unknown_references(proof, []):
            ctx.ui.log(
                "  The proof refers to computations ([code:...]) but "
                "cites no run in code_runs — sent back as feedback.",
                verbose,
            )
            feedback = [
                "Your proof refers to computations ([code:...]) but lists "
                "no script in code_runs. List the script id of every script "
                "the proof rests on, or remove the references."
            ]
            attempt_notes.append(
                f"Prover attempt {attempt}: the proof referred to "
                "computations it did not cite; it never reached the "
                "verifiers."
            )
            attempt += 1
            continue
        if raw_code_runs:
            if ctx.loop.code_dir is None:
                code_problems = ["this run has no code tools, so it can cite no scripts"]
            else:
                code, code_sources, code_problems, _names = files.Scripts(
                    ctx.loop.code_dir / "prover"
                ).resolve(_as_list(raw_code_runs))
            if not code_problems:
                code_problems = computations.unknown_references(proof, code)
            if code_problems and restoring:
                _restore_failed(ctx, in_flight_state, "; ".join(code_problems), verbose)
                continue
            if code_problems:
                ctx.ui.log(
                    f"  Prover's code_runs: {'; '.join(code_problems)} — "
                    f"sent back as feedback.",
                    verbose,
                )
                feedback = [
                    "Your proof's code_runs: " + "; ".join(code_problems)
                    + ". List only the script ids of scripts whose last run "
                    "is ok, as run_code reported them."
                ]
                attempt_notes.append(
                    f"Prover attempt {attempt}: code_runs cited scripts that "
                    f"do not stand ({'; '.join(code_problems)}); the proof "
                    "never reached the verifiers."
                )
                attempt += 1
                continue
            ctx.ui.log(
                f"  The proof rests on {len(code)} computation(s): "
                + ", ".join(f"{computations.short(e)} ({e['lang']})" for e in code),
                verbose,
            )

        # The proof as the verifiers will check it, and its hash over the
        # DAG they are shown (_verified_hash): what every verdict records,
        # and what the commit certifies.
        proved_node = {
            "statement": target["statement"],
            "proof": proof,
            "cited_lemmas": list(cited_lemmas or []),
            "cited_references": list(cited_refs or []),
        }
        if code:
            proved_node["code"] = code
            # Not stored: where the scripts are until the commit copies
            # them into code/ (computations.store).
            proved_node["_code_sources"] = code_sources
        # Not hashed: the model the lemma record names (dag._record).
        proved_node["_model"] = prover_model
        verified_hash = _verified_hash(dag, proved_node)

        # Checkpoint: the prover's output for this round, so a cancelled
        # run resumes at the verifiers instead of re-running the prover.
        if not restoring:
            in_flight_state["stage"] = "verify"
            in_flight_state["proof"] = dict(proof=proof, cited_lemmas=cited_lemmas,
                                            cited_references=cited_refs, code_runs=last_code_runs)
            in_flight_state["prover_model"] = prover_model
            in_flight_state["verdicts"] = []
            on_round(in_flight_state)

        all_cited = [*cited_lemmas, *cited_refs]
        if all_cited:
            ctx.ui.log(
                f"Proof generated for {lemma_id} ({len(proof)} chars, "
                f"cites: "
                f"{', '.join(str(d) for d in all_cited)}).",
                verbose,
            )
        else:
            ctx.ui.log(
                f"Proof generated for {lemma_id} ({len(proof)} chars, "
                f"no citations).",
                verbose,
            )

        # ---------------- Step 3: the three verifier checks -----------
        # Three atomic checks, one per verifier agent, in the order
        # verifier_1, verifier_2, verifier_3. Each is a single call to its
        # own agent; one reject ends the counting and carries its
        # reasoning to the reviser; only a proof that survives all three
        # is accepted.
        verifier_user = (
            f"Conjecture:\n{conjecture}\n\n"
            f"Cited results (statements of exactly the lemmas and "
            f"references the proof declares it used; nothing else is "
            f"available to it):\n"
            f"{json.dumps(verifier_context(dag, cited_lemmas, cited_refs, references), indent=2)}\n\n"
            f"Target lemma:\n{json.dumps(target, indent=2)}\n\n"
            f"Proposed proof:\n{proof}"
            + computations_block(ctx, {"code": code}, code_sources)
        )
        reject_just: Optional[str] = None
        stopped = False
        steps = verifier_steps({"code": code})
        # The verdicts a resumed round saved: each counts only while the
        # proof's hash is the one it checked and it came from the model
        # this run uses; any other is dropped and its verifier runs again.
        kept = [
            v for v in in_flight_state["verdicts"]
            if v.get("hash") == verified_hash
            and _same_model(ctx, v.get("model", ""), v.get("requested", ""))
        ]
        if len(kept) < len(in_flight_state["verdicts"]):
            # The parent may now be rejected or revised; do not lose completed
            # checks of its original citations if its batch is abandoned.
            citation_checks.recover_checks(ctx, [in_flight_state])
            ctx.ui.log(
                f"  {len(in_flight_state['verdicts']) - len(kept)} saved "
                f"verdict(s) on {lemma_id} no longer apply (the proof's hash "
                f"or the model changed); those verifiers run again.",
                verbose,
            )
        in_flight_state["verdicts"] = kept
        saved_verdicts = {v["agent"]: v for v in kept}
        # A saved rejection (or missing verdict) settles the round as it
        # did when it came in; any other saved verdict stands in for its
        # verifier, and the rest run.
        missing: Optional[Tuple[int, str]] = None
        for step, agent in enumerate(steps, 1):
            prior = saved_verdicts.get(agent)
            if prior is None:
                continue
            decision, justification = prior["decision"], prior["justification"]
            ctx.ui.log(
                f"Verifier {step}/{len(steps)} ({role_of(agent)}): "
                f"{decision.upper() or '???'} (saved) — {justification}",
                verbose,
            )
            if decision == verdicts.REJECT and reject_just is None:
                reject_just = justification
                ctx.status(event=f"verifier {step}/{len(steps)} rejected")
            elif decision != verdicts.ACCEPT and missing is None:
                missing = (step, justification)
        to_run = [] if reject_just is not None or missing is not None else [
            agent for agent in steps if agent not in saved_verdicts
        ]
        prompts = {agent: verifier_prompt(ctx, verifier_sys, agent)
                   for agent in to_run}

        def run_step(step_ctx, agent: str) -> Optional[Tuple[str, str]]:
            """One verifier, the server's outages waited out: its verdict,
            or None when should_stop fired while waiting. An unreachable
            server is waited out and the same step retried: the proof is
            not at fault, and keeping it saves re-running the prover."""
            step, role = steps.index(agent) + 1, role_of(agent)
            while True:
                decision, justification = _run_verifier(
                    step_ctx, role, prompts[agent], verifier_user, verbose,
                )
                if decision != verdicts.UNAVAILABLE:
                    return decision, justification
                step_ctx.ui.log(
                    f"Verifier {step}/{len(steps)} ({role}): "
                    f"{decision.upper()} — {justification}",
                    verbose,
                )
                if not _wait_for_server(
                    step_ctx, role,
                    lambda: stopping(step_ctx) or (
                        should_stop is not None and should_stop()),
                    verbose,
                ):
                    return None

        # Parallel verifiers (ctx.settings.parallel_verifiers): the first
        # rejection settles the round, as the first one does in step
        # order, and the rest are not waited for; a missing verdict starts
        # no further verifier but waits for the running ones, since a
        # rejection among them is the better reason to give the reviser.
        with fan_out(ctx, to_run, run_step, ctx.settings.parallel_verifiers) as results:
            if results.parallel:
                ctx.status(role=f"{results.workers} verifiers")
            for agent, outcome in results:
                if outcome is None:
                    stopped = True
                    break
                decision, justification = outcome
                step = steps.index(agent) + 1
                ctx.ui.log(
                    f"Verifier {step}/{len(steps)} ({role_of(agent)}): "
                    f"{decision.upper() or '???'} — {justification}",
                    verbose,
                )
                # Checkpoint: the verdict, as soon as it is in.
                in_flight_state["verdicts"].append({
                    "agent": agent,
                    "decision": decision,
                    "justification": justification,
                    "model": _recorded_model(ctx),
                    "requested": ctx.model.name,
                    "hash": verified_hash,
                })
                if decision != verdicts.ACCEPT:
                    in_flight_state["stage"] = "revise"
                on_round(in_flight_state)
                if decision == verdicts.REJECT:
                    rest = results.abandoned()
                    if rest:
                        ctx.ui.log(
                            f"Not waiting for "
                            f"{', '.join(role_of(a) for a in rest)}.",
                            verbose,
                        )
                    ctx.status(event=f"verifier {step}/{len(steps)} rejected")
                    reject_just = justification
                    break
                if decision != verdicts.ACCEPT and missing is None:
                    missing = (step, justification)
                    results.drain()
        if reject_just is None and missing is not None and not stopped:
            step, reject_just = missing
            ctx.status(event=f"verifier {step}/{len(steps)} gave no verdict")
        if stopped:
            # The top of the loop returns the published round as stopped.
            continue

        if reject_just is None:
            # ---------------- Step 4: DAG update ----------------
            # All three verifier checks accepted the same proof.
            ctx.ui.log(
                f"Lemma {lemma_id} passed all {len(steps)} "
                f"verifier checks.",
                verbose,
            )
            if audit_citations and (in_flight_state['stage'] == 'citations' or
                                    (cited_lemmas and ctx.settings.citation_recheck_base != -1)):
                # Canonical edges also for pre-feature prover checkpoints.
                in_flight_state['proof'].update(cited_lemmas=cited_lemmas, cited_references=cited_refs)
                ready = citation_checks.run_checks(
                    ctx, in_flight_state, on_round, proved_node, verified_hash,
                    conjecture, verifier_sys, verbose, should_stop,
                )
                if not ready:
                    return dict(proved=False, exhausted=False, stopped=True, lemma_id=lemma_id,
                                committed_id=None, claimed_id=claimed_id, target=target,
                                proof=proof, attempt_notes=attempt_notes)
                committed_id = on_proof(lemma_id, proved_node, verified_hash, in_flight_state['checks'])
            else:
                committed_id = on_proof(lemma_id, proved_node, verified_hash)
            ctx.status(event=f"committed {committed_id}")
            proved = True
            break

        # ---------------- Step 5: Reviser ----------------
        ctx.ui.log(
            f"Proof rejected for {lemma_id}. Sending the proof and the "
            f"verdict to the reviser.",
            verbose,
        )
        def build_reviser() -> Tuple[str, Optional["tools_mod.ToolSet"]]:
            return (
                f"Conjecture:\n{conjecture}\n\n"
                f"Proved lemmas so far:\n{json.dumps(planner_dag_view(ctx, dag, suspended), indent=2)}"
                f"{reviser_ref_block}\n\n"
                f"Target lemma:\n{json.dumps(target, indent=2)}\n\n"
                f"The last proof of this lemma (the one just rejected; judge "
                f"the lemma's difficulty from it):\n{proof}\n\n"
                f"Verifier's reasoning (why the proof failed):\n{reject_just}",
                agent_tools(ctx, "reviser", dag, suspended),
            )
        reviser_user, reviser_tools = _fit_prompt(
            ctx,
            "reviser", reviser_sys, build_reviser, verbose,
            parts={
                "the conjecture": conjecture,
                "the references block": reviser_ref_block,
                "the lemma": json.dumps(target, indent=2),
                "the proof": proof,
                "the verifier's reasoning": reject_just or "",
            },
        )
        answer = _reason_until_answered(
            ctx,
            should_stop, verbose,
            reviser_sys, reviser_user, "reviser",
            tools=reviser_tools,
        )
        if answer is None:
            continue
        revision_text, _revision_status, _partial = answer
        toggle_hotkey()

        revision_res: Dict[str, Any] = {}
        if revision_text:
            revision_res = _parse_reviser_decision(ctx, revision_text, verbose)
        elif _revision_status == "too_large":
            # Not re-sent: the same prompt would not fit any better. The
            # prover's next round, a different prompt, goes ahead.
            ctx.ui.log(
                "The reviser's prompt does not fit the context window even "
                "with no lemmas shown in full (its largest parts are logged "
                "above); the verdict alone goes back to the prover.",
                verbose,
            )
        else:
            ctx.ui.log(
                "Reviser returned nothing; the verdict alone goes back "
                "to the prover.",
                verbose,
            )

        # `action` is the current contract; older per-conjecture
        # overrides that still emit statement_revision are read the old
        # way so they keep working.
        action = str(revision_res.get("action") or "").strip().lower()
        if action not in ("keep", "revise_statement", "new_lemma"):
            action = (
                "revise_statement"
                if bool(revision_res.get("statement_revision", False))
                else "keep"
            )
        diagnosis = str(revision_res.get("diagnosis") or "").strip()
        new_stmt = str(revision_res.get("new_statement") or "").strip()

        feedback = [f"Verifier: {reject_just}"]
        # A kept statement's diagnosis is the fix for this rejection and
        # goes with it, for the next round only; a new or revised
        # statement's diagnosis becomes its guidance instead (below).
        diagnosis_note = (
            [f"Reviser's diagnosis: {diagnosis}"] if diagnosis else []
        )

        if allow_decomposition and action == "new_lemma" and new_stmt:
            # Decomposition: the target is too hard to prove as stated.
            # Restart the prover on the smaller lemma with a fresh attempt
            # budget and no inherited verdict; only the lemma the loop ends
            # on can reach the reject list.
            old_id = lemma_id
            lemma_id = new_lemma_id(ctx)
            target = {"id": lemma_id, "statement": new_stmt}
            attempt_notes = [
                f"Decomposed from {old_id} by the reviser (the original "
                f"target was judged too hard to prove as stated)."
            ]
            # A fresh lemma has no rejection of its own, so the prover
            # starts clean rather than inheriting the old proof's verdict;
            # what the reviser learned from it comes as guidance.
            feedback = []
            guidance = diagnosis
            last_proof = ""
            last_code_runs = []
            attempt = 1
            ctx.ui.log(
                f"Reviser decomposed {old_id}; the prover now starts on "
                f"the smaller lemma {lemma_id}:\n{new_stmt}",
                verbose,
            )
            continue

        if action == "revise_statement" and resolution.is_pinned(target["statement"], conjecture):
            # A pinned statement is the conjecture (or its negation)
            # itself: revising it would only prove something else in the
            # conjecture's place, which settles nothing.
            ctx.ui.log(
                f"Reviser proposed a revised statement for {lemma_id}, "
                f"whose statement is pinned to conjecture.md; keeping it.",
                verbose,
            )
        elif action == "revise_statement" and new_stmt and new_stmt != target["statement"]:
            target = {"id": lemma_id, "statement": new_stmt}
            last_proof = ""
            last_code_runs = []
            ctx.ui.log(
                f"Reviser revised the lemma; the prover starts again "
                f"from:\n{new_stmt}",
                verbose,
            )
            feedback.append(f"Revised statement proposed: {new_stmt}")
            guidance = diagnosis
            attempt_notes.extend(diagnosis_note)
            diagnosis_note = []
        elif action == "revise_statement":
            ctx.ui.log(
                "Reviser flagged a statement revision but gave none "
                "usable; keeping the statement.",
                verbose,
            )
        elif action == "new_lemma":
            # The reviser wanted to decompose but could not: it gave no
            # new statement, or decomposition is not allowed here at all
            # (a repair re-proves the lemma under its own id); fall back to
            # keeping the statement.
            if allow_decomposition:
                ctx.ui.log(
                    "Reviser proposed a new lemma but gave no statement for "
                    "it; keeping the statement instead.",
                    verbose,
                )
            else:
                ctx.ui.log(
                    "Reviser wanted to decompose, but the repair keeps "
                    "the lemma's id; keeping the statement instead.",
                    verbose,
                )
        else:
            ctx.ui.log(
                "Reviser kept the statement; the prover re-tries with "
                "the verdict as feedback.",
                verbose,
            )
        # The statement stands (a revision empties diagnosis_note above),
        # so the diagnosis is the fix.
        feedback.extend(diagnosis_note)
        attempt_notes.extend(feedback)
        attempt += 1
        # Loop continues: the next prover round works on `target`, with
        # only this round's feedback.

    return {
        "proved": proved,
        # The while condition is the only exit that leaves attempt past
        # the budget: a proof, an overflow give-up, or a stop leaves it.
        "exhausted": (not proved and not settings_mod.within(
            attempt, ctx.settings.max_proof_attempts)),
        "lemma_id": lemma_id,
        "committed_id": committed_id,
        "claimed_id": claimed_id,
        "target": target,
        "proof": proof,
        "attempt_notes": attempt_notes,
        "stopped": False,
    }
