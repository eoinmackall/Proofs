"""proofs repair: re-verify, and if need be re-prove, the lemma a refutation
names, then its dependents.
"""

from __future__ import annotations

import argparse
import signal
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core import certificates, merkle, refutations
from ..core.context import Code, Context
from ..core.dag import (
    _dependency_order, _dependent_chain, commit_reproof, describe_suspension,
    load_file, suspension_kinds,
)
from ..core.interrupts import (
    _commit_section, _on_sigint_repair, _raise_interrupt,
)
from ..validity import verdicts
from ..workspace.settings import (
    DEFAULT_LEMMAS_ALL, DEFAULT_MY_LEMMAS, DEFAULT_STRATEGY, DEFAULT_WINDOW_SIZE,
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


class _Repairer:
    """The steps of a repair, over one verify session: the re-verification
    of a refuted lemma with its objection on the table, the re-proof of a
    rejected one, and the climb up its dependents. proofs repair runs them
    on the one lemma its refutation file names; proofs repair --all on
    every refuted lemma in turn."""

    def __init__(self, ctx, verbose: bool, parser: argparse.ArgumentParser) -> None:
        self.ctx = ctx
        self.verbose = verbose
        self.parser = parser
        self.conjecture_text = load_file(ctx.paths.conjecture_file)
        if not self.conjecture_text:
            ctx.ui.log(
                f"{ctx.paths.conjecture_file} is empty; the re-verification and the "
                f"re-proof go on without the conjecture, the lemma's "
                f"statement and the DAG being the substance.",
                verbose,
            )
        self.verifier_sys = _verifier_prompts(ctx, parser)
        self._proof_inputs: Optional[Dict[str, Any]] = None

    def _inputs(self) -> Dict[str, Any]:
        """The prover's and reviser's prompts and the reference blocks,
        read the first time a lemma has to be re-proved."""
        if self._proof_inputs is None:
            ctx = self.ctx
            prover_sys = load_file(ctx.paths.prompt_paths["prover.md"])
            reviser_sys = load_file(ctx.paths.prompt_paths["reviser.md"])
            empty = [
                name for name, text in (
                    ("prover.md", prover_sys), ("reviser.md", reviser_sys),
                ) if not text
            ]
            if empty:
                self.parser.error(f"no prompt text for: {', '.join(empty)}")
            references, reference_block, _planner_ref_block, reviser_ref_block = (
                _build_reference_blocks(ctx, self.verbose)
            )
            self._proof_inputs = {
                "prover_sys": prover_sys,
                "reviser_sys": reviser_sys,
                "references": references,
                "reference_block": reference_block,
                "reviser_ref_block": reviser_ref_block,
                "ref_ids": {str(r["id"]) for r in references},
            }
        return self._proof_inputs

    def verify(self, lid: str) -> Tuple[str, Optional[str]]:
        return verify_stored_lemma(
            self.ctx,
            self.ctx.dag, self.ctx.dag.references, lid, self.conjecture_text,
            self.verifier_sys, self.verbose,
        )

    def reprove(self, lid: str, justification: Optional[str]) -> bool:
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
        ctx, verbose = self.ctx, self.verbose
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
        inputs = self._inputs()
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
            conjecture=self.conjecture_text,
            prover_sys=inputs["prover_sys"],
            reviser_sys=inputs["reviser_sys"],
            reference_block=inputs["reference_block"],
            reviser_ref_block=inputs["reviser_ref_block"],
            references=inputs["references"],
            verifier_sys=self.verifier_sys,
            ref_ids=inputs["ref_ids"],
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
                f"{lid} could not be re-proved"
                + ("" if ctx.settings.max_proof_attempts < 0
                   else f" after {ctx.settings.max_proof_attempts} prover round(s)")
                + "; the stored "
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

    def repair(
        self, lemma_id: str, from_execution: bool, touched: Set[str],
        path: Optional[Path] = None,
    ) -> Tuple[str, List[str]]:
        """Repair lemma_id: re-verify it with its refutations on the table
        (re-running its scripts when from_execution), re-prove it when that
        rejects it, and climb its dependents when the re-proof stands.
        path, when given, is the refutation file the repair was asked from.

        Every lemma this re-verifies or re-proves is added to touched, and
        a dependent already in touched is not checked again (repair --all
        tries each lemma once). Returns (outcome, failed): outcome is
        "accepted" (the re-verification accepted it; nothing above it
        changes), "reproved" (re-proved, and the climb done), "failed" (it
        could not be re-proved, or its hash could not be computed),
        "refused" (a fork) or "no_verdict" (a verifier gave none: the
        repair stopped where it was); failed are the lemmas left unproved
        — lemma_id itself, or dependents the climb could not re-prove."""
        ctx, verbose = self.ctx, self.verbose
        refusal = _fork_refusal(ctx.dag, lemma_id)
        if refusal is not None:
            ctx.ui.log(f"Not repairing: {refusal}", verbose)
            return "refused", [lemma_id]
        touched.add(lemma_id)
        try:
            ctx.code.run_code = from_execution
            try:
                outcome, just = self.verify(lemma_id)
            finally:
                ctx.code.run_code = False
        except merkle.MerkleCycleError as e:
            ctx.ui.log(f"{e}", verbose)
            return "failed", [lemma_id]
        if outcome == "no_verdict":
            # Not a rejection: re-proving would replace a proof nobody
            # rejected.
            ctx.ui.log(
                f"{lemma_id} got no verdict; the repair stops "
                f"with nothing written. Re-run it once the server answers.",
                verbose,
            )
            return "no_verdict", []
        if outcome == "accepted":
            # The acceptance already dismissed the lemma's refutations
            # (the addressed file among them) and re-issued its
            # certificate.
            if path is not None and path.is_file():
                refutations.delete(path)
                ctx.ui.log(f"Refutation dismissed: {path}", verbose)
            ctx.ui.log(f"{lemma_id} accepted; the repair is done.",
                verbose)
            return "accepted", []

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
        if not self.reprove(lemma_id, just):
            ctx.ui.log(
                f"The repair of {lemma_id} failed; it and the "
                f"lemmas that depend on it stay suspended.",
                verbose,
            )
            return "failed", [lemma_id]
        return self.climb(lemma_id, touched)

    def climb(self, lemma_id: str, touched: Set[str]) -> Tuple[str, List[str]]:
        """After lemma_id's re-proof: its dependents (see repair())."""
        ctx, verbose = self.ctx, self.verbose
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
            return "reproved", []
        ctx.ui.log(
            f"Checking the {len(chain):,} dependent(s) of {lemma_id} "
            f"in dependency order, re-verifying those whose certificates "
            f"no longer match.",
            verbose,
        )
        failed: List[str] = []
        reverified = 0
        for lid in chain:
            if lid in touched:
                continue
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
            touched.add(lid)
            try:
                dep_outcome, dep_just = self.verify(lid)
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
                # verify accepts each. Counted, not listed: verify
                # --uncertified finds them.
                rest = len(chain) - chain.index(lid)
                ctx.ui.log(
                    f"{lid} got no verdict; the climb stops with "
                    f"nothing written for it. {rest:,} dependent(s) are "
                    f"still to check: once the server answers, proofs "
                    f"verify {Path(ctx.paths.conjecture_root).name} "
                    f"--uncertified verifies those that need it"
                    + (f". {len(failed):,} could not be re-proved (the log "
                       f"above says why for each)" if failed else "")
                    + ".",
                    verbose,
                )
                return "no_verdict", failed
            if dep_outcome == "accepted":
                continue
            ctx.ui.log(
                f"{lid} rejected on re-verification; re-proving it "
                f"the same way.",
                verbose,
            )
            if not self.reprove(lid, dep_just):
                failed.append(lid)
        if failed:
            ctx.ui.log(
                f"The repair finished with {len(failed):,} lemma(s) "
                f"unproved (the log above says why for each); they and "
                f"their dependents stay suspended.",
                verbose,
            )
            return "reproved", failed
        ctx.ui.log(
            f"The repair is done: {lemma_id} re-proved; "
            f"{reverified} of its {len(chain)} dependent(s) needed "
            f"re-verifying, the rejected ones re-proved.",
            verbose,
        )
        return "reproved", []


def _guarded(ctx, verbose: bool, body) -> int:
    """Run body() with the repair's Ctrl-C handling: a KeyboardInterrupt,
    except while a re-proof is committed and certified, which waits for
    both."""
    previous_handler = signal.signal(
        signal.SIGINT, lambda signum, frame: _on_sigint_repair(ctx, signum, frame)
    )
    ctx.run.deferred_interrupt = _raise_interrupt
    try:
        return body()
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


def _session(
    ctx, *, root: Path, user_id: str, model: str, host: Optional[str],
    api_key: Optional[str], backend: str, num_ctx: Optional[int],
    verbose: bool, parser: argparse.ArgumentParser, lemmas_all: int,
    my_lemmas: int, tool_mode: str, strategy: Tuple[str, float],
    window_size: int, parallel: int = 1,
    settings_flags: Optional[Dict[str, Any]] = None,
) -> None:
    """The verify session, with the re-proof's lemma window and tool mode."""
    _verify_session(
        ctx,
        conjecture_dir=str(root), user_id=user_id, model=model, host=host,
        api_key=api_key, backend_choice=backend, num_ctx=num_ctx,
        verbose=verbose, parser=parser, parallel=parallel,
        settings_flags=settings_flags,
    )
    ctx.settings.lemmas_all, ctx.settings.my_lemmas = lemmas_all, my_lemmas
    ctx.settings.strategy, ctx.settings.strategy_dial = strategy
    ctx.settings.window_size = window_size
    ctx.settings.tool_mode = tool_mode


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
    strategy: Tuple[str, float] = DEFAULT_STRATEGY,
    window_size: int = DEFAULT_WINDOW_SIZE,
    code: Optional[Code] = None,
    parallel: int = 1,
    settings_flags: Optional[Dict[str, Any]] = None,
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
    dismissed, so the log counts the dependents still to check and points
    to proofs verify --uncertified.

    parallel (--parallel) runs up to that many of a lemma's verifiers at
    once — in the re-verifications and in the re-proof's rounds — and the
    first rejection ends the check.
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
    _session(
        ctx, root=root, user_id=user_id, model=model, host=host,
        api_key=api_key, backend=backend, num_ctx=num_ctx, verbose=verbose,
        parser=parser, lemmas_all=lemmas_all, my_lemmas=my_lemmas,
        tool_mode=tool_mode, strategy=strategy, window_size=window_size,
        parallel=parallel,
        settings_flags=settings_flags,
    )
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
    repairer = _Repairer(ctx, verbose, parser)

    def body() -> int:
        outcome, failed = repairer.repair(lemma_id, from_execution, set(), p)
        if outcome == "no_verdict":
            return EXIT_NO_VERDICT
        return 1 if failed else 0

    return _guarded(ctx, verbose, body)


def repair_all_entry(
    *,
    root: str,
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
    strategy: Tuple[str, float] = DEFAULT_STRATEGY,
    window_size: int = DEFAULT_WINDOW_SIZE,
    code: Optional[Code] = None,
    parallel: int = 1,
    settings_flags: Optional[Dict[str, Any]] = None,
) -> int:
    """proofs repair --all DIR: repair every lemma a refutation counts
    against, each the way proofs repair repairs the one its file names,
    cited lemmas first.

    The refuted lemmas are read once, at the start, one per lemma however
    many refutation files it has; a refutation written during the run —
    a re-verification's rejection, or a dependent's in a climb — waits for
    the next run. Every lemma is tried at most once: one a climb has
    already re-verified or re-proved is not repaired again, whatever came
    of it. The refuted lemmas are walked together with the lemmas
    suspended only for what they cite, in dependency order, so that a
    refuted lemma citing one that stays suspended — forked, on a cycle,
    citing what is not there, uncertified (proofs verify --uncertified is
    its way back), or one this run could not repair — is skipped, and so
    is everything above it: the prover could not cite what it rests on. A
    lemma refuted by a re-execution needs --tools to be repaired, and is
    skipped without it.

    A missing verdict stops the run, as it stops a repair. The log has a
    repair's lines for each lemma repaired and counts otherwise, never a
    list of ids. Returns 0 when nothing failed, 1 when a lemma could not
    be re-proved or was refused, EXIT_NO_VERDICT on a missing verdict, 130
    on Ctrl-C.

    parallel (--parallel) runs up to that many of a lemma's verifiers at
    once, as in proofs repair; the lemmas themselves are repaired one at a
    time.
    """
    ctx = Context(user=user_id, code=code if code is not None else Code())
    _session(
        ctx, root=Path(root), user_id=user_id, model=model, host=host,
        api_key=api_key, backend=backend, num_ctx=num_ctx, verbose=verbose,
        parser=parser, lemmas_all=lemmas_all, my_lemmas=my_lemmas,
        tool_mode=tool_mode, strategy=strategy, window_size=window_size,
        parallel=parallel,
        settings_flags=settings_flags,
    )
    name = Path(ctx.paths.conjecture_root).name
    dag = ctx.dag
    kinds = suspension_kinds(dag)
    refuted = kinds["refuted"]
    line = describe_suspension(dag, name)
    ctx.ui.log(line or "No lemma is suspended.", verbose)
    if not refuted:
        if line:
            ctx.ui.log("No refutation counts: nothing to repair.", verbose)
        return 0
    # What no repair in this run brings back: whatever cites one of these
    # is skipped. A lemma this run fails to repair joins them.
    blocked = set(kinds["conflict"] | kinds["cycle"] | kinds["dangling"]
                  | kinds["uncertified"])
    by_execution = {
        str(r.get("lemma_id")) for _path, r in dag.refutations
        if str(r.get("lemma_id")) in refuted
        and verdicts.is_execution_refutation(r)
        and refutations.is_counting(r, dag.hashes.get(str(r.get("lemma_id"))) or "")
    }
    order = _dependency_order(dag, refuted | kinds["cites"])
    total = len(refuted)
    repairer = _Repairer(ctx, verbose, parser)
    touched: Set[str] = set()
    counts: Counter = Counter()
    ctx.ui.log(f"Repairing up to {total:,} refuted lemma(s), cited lemmas first.",
               verbose)

    def summary() -> None:
        settled = sum(counts[k] for k in (
            "accepted", "reproved", "failed", "refused", "skipped",
            "needs_tools", "earlier"))
        parts = [
            f"{counts[key]:,} {what}"
            for key, what in (
                ("failed", "could not be re-proved"),
                ("refused", "forked (settle by hand)"),
                ("skipped", "skipped, citing a lemma that stays suspended"),
                ("needs_tools", "refuted by a re-execution (pass --tools)"),
                ("earlier", "already reached by an earlier climb"),
            )
            if counts[key]
        ]
        if total - settled:
            parts.append(f"{total - settled:,} not reached")
        ctx.ui.log(
            f"\nRepaired {counts['accepted'] + counts['reproved']:,} of "
            f"{total:,} refuted lemma(s) ({counts['accepted']:,} accepted on "
            f"re-verification, {counts['reproved']:,} re-proved)"
            + ("; " + "; ".join(parts) if parts else "") + "."
            + (f" {counts['climb_failed']:,} dependent(s) could not be "
               f"re-proved in the climbs." if counts["climb_failed"] else ""),
            verbose,
        )
        ctx.ui.log(describe_suspension(ctx.dag, name) or
                   "No lemma is suspended now.", verbose)

    def body() -> int:
        n = 0
        try:
            for lid in order:
                cited = ctx.dag.lemmas[lid].get("cited_lemmas") or []
                if any(c in blocked for c in cited):
                    blocked.add(lid)
                    if lid in refuted:
                        counts["skipped"] += 1
                    continue
                if lid not in refuted:
                    continue
                if lid in touched:
                    counts["earlier"] += 1
                    continue
                if lid in by_execution and ctx.code.runner is None:
                    counts["needs_tools"] += 1
                    blocked.add(lid)
                    continue
                n += 1
                ctx.ui.log(f"\n[{n:,}/{total:,}] Repairing {lid}", verbose)
                outcome, failed = repairer.repair(lid, lid in by_execution, touched)
                counts[outcome] += 1
                blocked.update(failed)
                counts["climb_failed"] += sum(1 for f in failed if f != lid)
                if outcome == "no_verdict":
                    return EXIT_NO_VERDICT
        finally:
            summary()
        failed_any = counts["failed"] or counts["refused"] or counts["climb_failed"]
        return 1 if failed_any else 0

    return _guarded(ctx, verbose, body)
