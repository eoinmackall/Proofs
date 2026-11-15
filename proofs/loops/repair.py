"""proofs repair: re-verify, and if need be re-prove, the lemma a refutation
names, then its dependents.
"""

from __future__ import annotations

import argparse
import signal
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core import certificates, merkle, refutations
from ..core.context import Code, Context
from ..core.dag import (
    _dependent_chain, commit_reproof, load_file,
)
from ..core.interrupts import (
    _commit_section, _on_sigint_repair, _raise_interrupt,
)
from ..validity import verdicts
from ..workspace.settings import (
    DEFAULT_LEMMAS_ALL, DEFAULT_LEMMAS_COMMON, DEFAULT_MY_LEMMAS,
    DEFAULT_TOOL_MODE,
)
from .prompts import _build_reference_blocks
from .proof import run_proof_loop
from .verify import (
    EXIT_NO_VERDICT, _verifier_prompts, _verify_session, verify_stored_lemma,
)


def _fork_refusal(dag: Any, lemma_id: str) -> Optional[str]:
    """Why a repair may not re-prove lemma_id, or None: a forked lemma
    (core/dag.py) is refused, since a reproof supersedes only the current
    version and the others would stay beside it. What to do first depends
    on the fork: one whose versions agree and one of which is certified,
    proofs prune resolves; any other, a person settles by deleting all but
    one version."""
    if lemma_id not in dag.forked:
        return None
    h = dag.hashes.get(lemma_id)
    certified = lemma_id not in dag.conflicts and bool(h) and any(
        c.get("lemma_id") == lemma_id and certificates.is_valid(c, h)
        for c in dag.certificates
    )
    if certified:
        todo = ("one of its versions is certified, so proofs prune resolves "
                "the fork (it keeps that version and deletes the rest); "
                "prune, then repair again if the refutation still stands")
    else:
        todo = (f"delete all but one of its versions by hand "
                f"({', '.join(dag.forked[lemma_id])}), then repair")
    return f"{lemma_id} has more than one current version (a fork): {todo}."


def repair_entry(
    *,
    path: str,
    user_id: str,
    model: str,
    host: Optional[str],
    api_key: Optional[str],
    backend: str,
    num_ctx: Optional[int],
    verbose: bool,
    parser: argparse.ArgumentParser,
    lemmas_all: int = DEFAULT_LEMMAS_ALL,
    my_lemmas: int = DEFAULT_MY_LEMMAS,
    tool_mode: str = DEFAULT_TOOL_MODE,
    lemmas_common: int = DEFAULT_LEMMAS_COMMON,
    code: Optional[Code] = None,
) -> int:
    """proofs repair path_to_refutation (see refutations.py,
    repair_entry()): the refutation file supplies the lemma's id, so
    nothing else names the lemma.

    The first step is to re-verify the existing proof with the
    refutation's justifications as input — the verification a stored
    lemma gets from proofs verify, in which the lemma's refutations are
    on the table and the file addressed carries its justification. If the
    proof is accepted, the refutation is dismissed and its file deleted
    (the acceptance dismisses the lemma's refutations and re-issues its
    certificate), and the repair stops.

    If it is rejected, a new refutation file is written for the new
    rejection, and the lemma is re-proven through the prover, verifier
    and reviser, the rejection's justification as the prover's first
    feedback. The repair re-proves the lemma, and the lemma keeps its
    id, so the engine runs with decomposition off: the reviser's options
    are keep and revise only. An accepted proof is committed as a reproof
    of the lemma — a new version under the same id, superseding the one
    rejected, in this run's own stream (core/dag.py) — certified, and the
    lemma's refutations dismissed.

    A new proof under the same statement changes only the lemma's own
    hash — a dependent's hash covers the statements it cites, not their
    proofs (see merkle.py) — so its dependents keep their certificates.
    When the reviser revised the statement, the lemmas citing it lose
    theirs the moment the reproof is committed. The repair therefore
    climbs the dependency chain in dependency order, re-verifying each
    dependent whose certificate no longer matches and re-proving a
    rejected one the same way, its own rejection as the justification. A
    lemma that cannot be re-proved keeps the proof it had and the
    refutation that counts against it, and the climb goes on, so it and
    its dependents stay suspended.

    Returns the exit code: 0 when the lemma was re-verified or re-proved
    and every dependent re-verified or re-proved; 1 when a re-proof
    failed (its lemma and its dependents stay suspended) or a
    re-verification could not hash the lemma; EXIT_NO_VERDICT when a
    verifier gave no verdict (the server failed, or its reply could not be
    read). A missing verdict is not a rejection: no refutation is written,
    nothing is re-proved, and the repair stops where it is — the lemmas
    already repaired stand and the rest are left as they were. On the
    named lemma, the refutation file is still there, so the same command
    can be re-run; on a dependent, the named lemma's refutation is already
    dismissed, so the log lists the dependents still to verify with proofs
    verify.
    """
    ctx = Context(user=user_id, code=code if code is not None else Code())
    # Resolved first, as cli/main.py resolves PATH: the conjecture root is the
    # refutation file's grandparent, which a bare relative name (run inside
    # refutations/) does not have until it is made absolute.
    p = Path(path).resolve()
    try:
        ref = refutations.load(p)
    except (OSError, ValueError) as e:
        parser.error(f"{path} is not a readable refutation file: {e}")
    lemma_id = str(ref.get("lemma_id") or "")
    if not lemma_id:
        parser.error(f"{path} names no lemma (its lemma_id is required)")
    if p.parent.name != refutations.DIR_NAME:
        parser.error(f"{path} is not in a {refutations.DIR_NAME}/ directory")
    root = p.parent.parent
    # A refutation written by a re-execution (proofs verify --run-code:
    # its model is exec:<tool>) is a script that did not reproduce, a fact
    # no reading can answer. The lemma's re-verification re-runs its
    # scripts too, so only a re-execution that reproduces dismisses it.
    from_execution = verdicts.is_execution_refutation(ref)
    if from_execution and ctx.code.runner is None:
        parser.error(
            f"{p.name} was written by a re-execution of the lemma's scripts "
            f"({ref.get('model')}); repairing it re-runs them, so pass "
            f"--tools"
        )
    _verify_session(
        ctx,
        conjecture_dir=str(root), user_id=user_id, model=model, host=host,
        api_key=api_key, backend_choice=backend, num_ctx=num_ctx,
        verbose=verbose, parser=parser,
    )
    ctx.settings.lemmas_all, ctx.settings.my_lemmas = lemmas_all, my_lemmas
    ctx.settings.lemmas_common = lemmas_common
    ctx.settings.tool_mode = tool_mode
    dag = ctx.dag
    if lemma_id not in dag.lemmas:
        parser.error(
            f"{lemma_id} (named by {p.name}) is not a lemma of the "
            f"DAG {root}; there is nothing to repair"
        )
    refusal = _fork_refusal(dag, lemma_id)
    if refusal is not None:
        ctx.ui.log(f"Not repairing: {refusal}", verbose)
        return 1
    ctx.ui.log(f"Repairing {lemma_id} from {p.name}", verbose)
    # The lemma a refutation names is usually suspended (the refutation may
    # be stale, but the files say so only once they are read), and the
    # repair is precisely the case the suspension warns about (see
    # suspension.py).
    susp = dag.suspension()
    if susp.has(lemma_id):
        ctx.ui.log(
            f"{lemma_id} is suspended ({'; '.join(susp.reasons(lemma_id))}); "
            f"the repair re-verifies it anyway — an acceptance re-issues the "
            f"certificate and lifts the suspension.",
            verbose,
        )
    conjecture_text = load_file(ctx.paths.conjecture_file)
    if not conjecture_text:
        ctx.ui.log(
            f"{ctx.paths.conjecture_file} is empty; the re-verification and the "
            f"re-proof go on without the conjecture, the lemma's "
            f"statement and the DAG being the substance.",
            verbose,
        )
    verifier_sys = _verifier_prompts(ctx, parser)
    # Ctrl-C stays a KeyboardInterrupt (caught below), but one landing
    # while a re-proof is committed and certified waits for both.
    previous_handler = signal.signal(
        signal.SIGINT, lambda signum, frame: _on_sigint_repair(ctx, signum, frame)
    )
    ctx.run.deferred_interrupt = _raise_interrupt
    try:
        try:
            ctx.code.run_code = from_execution
            try:
                outcome, just = verify_stored_lemma(
                    ctx,
                    ctx.dag, ctx.dag.references, lemma_id, conjecture_text,
                    verifier_sys, verbose,
                )
            finally:
                ctx.code.run_code = False
        except merkle.MerkleCycleError as e:
            ctx.ui.log(f"{e}", verbose)
            return 1
        if outcome == "no_verdict":
            # Not a rejection: re-proving would replace a proof nobody
            # rejected.
            ctx.ui.log(
                f"{lemma_id} got no verdict; the repair stops "
                f"with nothing written. Re-run it once the server answers.",
                verbose,
            )
            return EXIT_NO_VERDICT
        if outcome == "accepted":
            # The acceptance already dismissed the lemma's refutations
            # (the addressed file among them) and re-issued its
            # certificate.
            if p.is_file():
                refutations.delete(p)
                ctx.ui.log(f"Refutation dismissed: {p}", verbose)
            ctx.ui.log(f"{lemma_id} accepted; the repair is done.",
                verbose)
            return 0

        # The re-verification rejected the stored proof and wrote a new
        # refutation for the rejection. The repair's next step:
        # re-prove the lemma through the prover, verifier and reviser,
        # the rejection's justification as the prover's first feedback,
        # and on acceptance commit the new proof as a reproof under the
        # same id.
        ctx.ui.log(
            f"{lemma_id} rejected on re-verification; "
            f"re-proving it through the prover, verifier and reviser.",
            verbose,
        )
        prover_sys = load_file(ctx.paths.prompt_paths["prover.md"])
        reviser_sys = load_file(ctx.paths.prompt_paths["reviser.md"])
        empty = [
            name for name, text in (
                ("prover.md", prover_sys), ("reviser.md", reviser_sys),
            ) if not text
        ]
        if empty:
            parser.error(f"no prompt text for: {', '.join(empty)}")
        references, reference_block, _planner_ref_block, reviser_ref_block = (
            _build_reference_blocks(ctx, verbose)
        )
        ref_ids = {str(r["id"]) for r in references}

        def reprove(lid: str, justification: Optional[str]) -> bool:
            """The repair's re-proof of the stored lemma lid:
            the engine's prover -> verifier -> reviser rounds on the
            lemma's statement as it stands in the file, the rejection's
            justification as the prover's first feedback, decomposition
            off — the repair re-proves the lemma and the lemma keeps its
            id, so the reviser's choices are keep and revise only. An
            accepted proof is committed as a reproof under the same id,
            certified the way an acceptance certifies, and the lemma's
            refutations dismissed. Returns
            whether the proof was accepted; a failure leaves the stored
            proof and its counting refutation in the file, the lemma
            suspended.
            """
            node = ctx.dag.lemmas.get(lid)
            if node is None:
                ctx.ui.log(
                    f"{lid} is not in the DAG any more; nothing "
                    f"to re-prove.",
                    verbose,
                )
                return False
            # A dependent the climb reaches may be forked too.
            refusal = _fork_refusal(ctx.dag, lid)
            if refusal is not None:
                ctx.ui.log(f"Not re-proving: {refusal}", verbose)
                return False
            # The engine recomputes the suspension every round: the lemma
            # is suspended (that is why it is being re-proved), and so are
            # its dependents, so the prover and the reviser see only the
            # lemmas that are not — the lemma cannot cite itself.
            initial = {
                "lemma_id": lid,
                "claimed_id": lid,
                "target": {
                    "id": lid,
                    "statement": str(node.get("statement") or ""),
                },
                "attempt": 1,
                # The rejection's justification is the prover's first
                # feedback, the engine's most-recent-failure-only as it
                # was when the lemma's proof was last rejected.
                "feedback": (
                    [f"Verifier: {justification}"] if justification else []
                ),
                "guidance": "",
                "attempt_notes": [],
                "last_proof": "",
            }

            def on_proof(
                proof_lid: str, proof_node: Dict[str, Any],
                verified_hash: Optional[str],
            ) -> str:
                with _commit_section(ctx):
                    commit_reproof(ctx, lid, proof_node, verified_hash, verbose)
                return lid

            result = run_proof_loop(
                ctx,
                verbose=verbose,
                conjecture=conjecture_text,
                prover_sys=prover_sys,
                reviser_sys=reviser_sys,
                reference_block=reference_block,
                reviser_ref_block=reviser_ref_block,
                references=references,
                verifier_sys=verifier_sys,
                ref_ids=ref_ids,
                initial=initial,
                fresh_target=None,
                on_round=lambda state: None,
                on_proof=on_proof,
                get_dag=lambda: ctx.dag,
                toggle_hotkey=lambda: None,
                allow_decomposition=False,
            )
            if not result["proved"]:
                ctx.ui.log(
                    f"{lid} could not be re-proved after "
                    f"{ctx.settings.max_proof_attempts} prover round(s); the stored "
                    f"proof stays, with its refutation counting against "
                    f"it.",
                    verbose,
                )
                for note in result["attempt_notes"]:
                    ctx.ui.log(f"   · {note}", verbose)
                return False
            # The acceptance put every objection on the table: the
            # lemma's refutations are dismissed, the addressed file among
            # them, the way the re-verification's acceptance would have
            # dismissed them.
            dismissed = []
            for ref_path, _ref in ctx.dag.refutations_for(lid):
                try:
                    refutations.delete(ref_path)
                    dismissed.append(ref_path)
                    ctx.ui.log(f"Refutation dismissed: {ref_path}", verbose)
                except OSError as e:
                    ctx.ui.log(
                        f"Could not delete the refutation "
                        f"{ref_path}: {e}",
                        verbose,
                    )
            ctx.run.dag = ctx.dag.without_refutations(dismissed)
            ctx.ui.log(
                f"{lid} re-proved; the new version stands under its id.",
                verbose,
            )
            return True

        if not reprove(lemma_id, just):
            ctx.ui.log(
                f"The repair of {lemma_id} failed; it and the "
                f"lemmas that depend on it stay suspended.",
                verbose,
            )
            return 1

        # A new proof under the same statement changes only the lemma's
        # own hash: a dependent's hash covers the statements it cites,
        # not their proofs, so its certificate still matches and nothing
        # above needs verifying. A statement the reviser revised does
        # change the hash of every lemma citing it, and those lose their
        # certificates. Climb the dependency chain in dependency order,
        # re-verifying each dependent whose certificate no longer matches
        # — checked as the climb reaches it, so a dependent whose own
        # statement is revised on the way takes its citers along — and
        # re-proving a rejected one the same way, its own rejection as the
        # justification. A dependent that cannot be re-proved stays
        # suspended and the climb goes on.
        chain = _dependent_chain(ctx.dag, lemma_id)
        if not chain:
            ctx.ui.log(
                f"The repair is done: {lemma_id} re-proved, "
                f"and nothing depends on it.",
                verbose,
            )
            return 0
        ctx.ui.log(
            f"Checking the {len(chain)} dependent(s) of {lemma_id} "
            f"in dependency order, re-verifying those whose certificates "
            f"no longer match: "
            + ", ".join(chain)
            + ".",
            verbose,
        )
        failed: List[str] = []
        reverified = 0
        for lid in chain:
            dag = ctx.dag
            try:
                h = dag.hash(lid)
            except merkle.MerkleCycleError as e:
                ctx.ui.log(f"{e}", verbose)
                failed.append(lid)
                continue
            if any(
                certificates.is_valid(c, h)
                for c in dag.certificates if c.get("lemma_id") == lid
            ):
                continue
            reverified += 1
            try:
                dep_outcome, dep_just = verify_stored_lemma(
                    ctx,
                    dag, dag.references, lid, conjecture_text, verifier_sys,
                    verbose,
                )
            except merkle.MerkleCycleError as e:
                ctx.ui.log(f"{e}", verbose)
                failed.append(lid)
                continue
            if dep_outcome == "no_verdict":
                # Most likely the server is down, and every later check
                # would fail the same way: stop the climb rather than
                # spend it. The repaired lemma stands (its refutation is
                # already dismissed, so this repair cannot be re-run); the
                # dependents not yet verified stay suspended until proofs
                # verify accepts each.
                rest = chain[chain.index(lid):]
                ctx.ui.log(
                    f"{lid} got no verdict; the climb stops with "
                    f"nothing written for it. Still to verify, with proofs "
                    f"verify once the server answers: "
                    + ", ".join(rest)
                    + (". Already unproved: " + ", ".join(failed)
                       if failed else "")
                    + ".",
                    verbose,
                )
                return EXIT_NO_VERDICT
            if dep_outcome == "accepted":
                continue
            ctx.ui.log(
                f"{lid} rejected on re-verification; re-proving it "
                f"the same way.",
                verbose,
            )
            if not reprove(lid, dep_just):
                failed.append(lid)
        if failed:
            ctx.ui.log(
                f"The repair finished with {len(failed)} lemma(s) "
                f"unproved: "
                + ", ".join(failed)
                + "; they and their dependents stay suspended.",
                verbose,
            )
            return 1
        ctx.ui.log(
            f"The repair is done: {lemma_id} re-proved; "
            f"{reverified} of its {len(chain)} dependent(s) needed "
            f"re-verifying, the rejected ones re-proved.",
            verbose,
        )
        return 0
    except KeyboardInterrupt:
        ctx.ui.log(
            "\nInterrupted; the lemmas already repaired stand, and the "
            "rest keep the proofs they had, suspended.",
            verbose,
        )
        return 130
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        ctx.run.deferred_interrupt = None
