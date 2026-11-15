"""proofs verify: the verifier checks on a stored proof."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .. import llm
from ..core import certificates, merkle, refutations
from ..core.checkpoint import attach_run, open_fresh_run
from ..core.context import Code, Context, Paths
from ..core.dag import (
    _citation_closure, load_file, writer,
)
from ..tools.run_code import computations, execution
from ..validity import verdicts
from ..validity.steps import VERIFIER_AGENTS, role_of, verifier_steps
from ..workspace import layout as workspace
from ..workspace import settings as settings_mod
from .prompts import computations_block, verifier_context
from .reasoning import REQUEST_TIMEOUT, _recorded_model
from .verifiers import _run_verifier, verifier_prompt

# proofs verify / proofs repair: the exit code when a verifier gave no
# verdict (the server failed, or the reply could not be read). Distinct from
# 1, a real rejection, and from argparse's 2; nothing was written.
EXIT_NO_VERDICT = 3


def _verifier_prompts(ctx, parser: argparse.ArgumentParser) -> Dict[str, str]:
    """The three verifier prompts as the prompt paths give them, each
    checked non-empty: a verifier with an empty system prompt is not a
    verifier, and a missing prompt file is a setup error, not a verdict."""
    verifier_sys = {name: load_file(ctx.paths.prompt_paths[name]) for name in VERIFIER_AGENTS}
    missing = [name for name in verifier_sys if not verifier_sys[name]]
    if missing:
        parser.error(f"no prompt text for: {', '.join(missing)}")
    return verifier_sys


# Limit kills (a timeout, the memory cap) say nothing about a script's
# claim, only about this machine: such a re-execution is inconclusive.
def _reexecute(
    ctx,
    lemma_id: str, node: Dict[str, Any], verbose: bool,
) -> Tuple[str, str, str]:
    """proofs verify --run-code: run the lemma's scripts again and compare.

    Every script is shown in full and the operator asked to confirm before
    anything runs — the scripts are other people's code, and the sandbox
    is a second line, not the first. Returns (outcome, message, tool):
    outcome is "skip" (nothing to run, or --run-code not given), "match"
    (every script exited 0 with the RESULT the lemma records; each
    execution is recorded in the run's computations stream), "mismatch" (one did not: message says
    how, and is the refutation's justification) or "inconclusive" (the
    operator declined, a script is missing or needs a tool this run does
    not have, or a run hit this machine's limits: nothing is recorded
    either way). tool names the tool of a mismatching run."""
    code = computations.entries(node)
    if not ctx.code.run_code or not code:
        return "skip", "", ""
    root = Path(ctx.paths.conjecture_root)
    scripts = []
    for e in code:
        text = computations.load_script(root, e)
        if text is None:
            return ("inconclusive",
                    f"script {computations.short(e)} has no intact copy on "
                    f"disk, so it cannot be re-run", "")
        if e["lang"] not in ctx.code.runner.tools:
            return ("inconclusive",
                    f"script {computations.short(e)} is {e['lang']}, and this "
                    f"verification was not given that tool (--tools)", "")
        scripts.append((e, text))
    ctx.ui.log(f"\n{lemma_id} rests on {len(scripts)} computation(s). Read "
               f"them before running them:")
    for e, text in scripts:
        ctx.ui.log(f"\n----- [code:{computations.short(e)}] {e['lang']}, "
                   f"recorded RESULT: {json.dumps(e.get('produced_output'), sort_keys=True)}")
        ctx.ui.log(text.rstrip("\n"))
        ctx.ui.log("-----")
    if not sys.stdin.isatty():
        return ("inconclusive",
                "--run-code asks before running anything, and there is no "
                "terminal to ask on", "")
    try:
        with ctx.ui.paused():
            answer = input(f"Run these {len(scripts)} script(s) in the sandbox? [y/N] ")
    except EOFError:
        answer = ""
    if answer.strip().lower() not in ("y", "yes"):
        return "inconclusive", "the operator declined to run the scripts", ""
    for e, text in scripts:
        sid = computations.short(e)
        try:
            res = ctx.code.runner.run(e["lang"], text)
        except execution.HarnessError as ex:
            return "inconclusive", f"script {sid}: {ex}", ""
        found = ctx.code.runner.tools[e["lang"]]
        tool = f"{found.spec.name} {found.version}"
        ctx.ui.log(f"  [code:{sid}] re-run with {tool}: {res.status}"
            + (f" — {res.failure}" if res.failure else
               f", RESULT {json.dumps(res.result, sort_keys=True)}"),
            verbose)
        if res.limit_killed:
            return ("inconclusive",
                    f"script {sid} hit this machine's limits ({res.failure}); "
                    f"raise them (proofs limits) and re-run", "")
        recorded = e.get("produced_output")
        if res.status == "ok" and computations.result_hash(res.result) == \
                computations.result_hash(recorded):
            computations.record_run(writer(ctx, computations.DIR_NAME), res.log)
            continue
        if res.status == "ok":
            how = (f"it printed RESULT {json.dumps(res.result, sort_keys=True)}, "
                   f"but the lemma records "
                   f"{json.dumps(recorded, sort_keys=True)}")
        else:
            how = res.failure
            if res.stderr_tail:
                how += "\nstderr (tail):\n" + res.stderr_tail
        return ("mismatch",
                f"Re-execution of script [code:{sid}] with {tool} "
                f"({found.path}) does not reproduce the computation the proof "
                f"rests on: {how}. Scratch run: {res.run_id}.", tool)
    return "match", "", ""


def verify_stored_lemma(
    ctx,
    dag: Any,
    references: List[Dict[str, Any]],
    lemma_id: str,
    conjecture_text: str,
    verifier_sys: Dict[str, str],
    verbose: bool,
) -> Tuple[str, Optional[str]]:
    """The three verifier checks on a lemma that is already in the DAG,
    with the refutation bookkeeping around them (see refutations.py).

    The lemma's current Merkle hash is recomputed from the DAG and
    references as they stand now, and every refutation file the lemma has
    is loaded. A refutation counts while its hash matches the current
    one (refutations.is_counting); the ones that do not are stale —
    objections to versions of the lemma that are no longer in the files —
    and their justifications are passed to the verifiers all the same,
    so a known objection is not forgotten. The
    counting ones go on the table as well, marked as such: an objection
    to the proof as it stands now is the one a re-verification is
    answering, and proofs repair runs on exactly that case. A lemma with
    a counting refutation is warned about before the checks start.

    A rejection ends the count, the run loop's rule, and the rejecting
    verifier's justification is what the new refutation file records,
    under the user who ran the verification and the model under which it
    happened. An acceptance issues the certificate — in this run's own
    certificate stream, the verifier recorded the way a certificate
    records it — and
    dismisses the refutations whose justifications were on the table: the
    stale ones, and the counting ones, answered.

    A step that gives no verdict at all (_run_verifier's "" or
    "unavailable": the server failed or replied with nothing, the reply
    could not be read, or the pass hit the context wall) ends the count too, but writes nothing: a
    refutation is the record of a verifier's rejection, and a missing
    verdict is not one — written, it would suspend the lemma and everything
    above it for a server outage. Nothing is issued or dismissed either;
    the lemma stands exactly as it did, and the verification can be re-run.
    An acceptance whose certificate cannot be written ends the same way:
    the refutations are kept rather than dismissed with no record of the
    acceptance in their place.

    Returns (outcome, justification): ("accepted", None), ("rejected", the
    rejecting justification), or ("no_verdict", why there was none).
    """
    node = dag.lemmas.get(str(lemma_id))
    if node is None:
        raise KeyError(f"{lemma_id!r} is not a lemma of the DAG")
    h = dag.hash(lemma_id)

    on_file = dag.refutations_for(lemma_id)
    stale = 0
    # A counting refutation from a re-execution (model exec:<tool>) says a
    # script did not reproduce; only a re-execution that does can answer
    # it, so the verifiers' reading alone may not dismiss it.
    exec_counting = [
        path for path, ref in on_file
        if refutations.is_counting(ref, h)
        and verdicts.is_execution_refutation(ref)
    ]
    for path, ref in on_file:
        if refutations.is_counting(ref, h):
            ctx.ui.log(
                f"{lemma_id} has a refutation that counts ({path.name}): "
                f"the proof it rejects is the proof in the file now.",
                verbose,
            )
        else:
            stale += 1
    if stale:
        ctx.ui.log(
            f"{lemma_id} has {stale} stale refutation(s); their "
            f"justifications go to the verifiers.",
            verbose,
        )

    # --run-code: the computations first. A script that no longer gives
    # what the lemma records is a rejection on its own, before any
    # verifier reads the proof; an inconclusive re-run leaves the reading
    # verification to decide.
    exec_name = exec_counting[0].name if exec_counting else None
    why = verdicts.unanswerable_without_reexecution(
        lemma_id, exec_name, ctx.code.run_code,
    )
    if why is not None:
        ctx.ui.log(why, verbose)
        return "no_verdict", why
    outcome, message, tool = _reexecute(ctx, lemma_id, node, verbose)
    why = verdicts.unsettled_by_reexecution(lemma_id, exec_name, outcome, message)
    if why is not None:
        ctx.ui.log(why, verbose)
        return "no_verdict", why
    if outcome == "mismatch":
        path = refutations.record(
            ctx.paths.conjecture_root, lemma_id, h, ctx.user,
            verdicts.execution_model(tool), "reject", message, node["file"],
        )
        ctx.run.dag = ctx.dag.with_refutation(path, refutations.load(path))
        ctx.ui.log(f"{message}\nRefutation for {lemma_id} written: {path}", verbose)
        return "rejected", message
    if outcome == "inconclusive":
        ctx.ui.log(f"  Re-execution inconclusive: {message}. Nothing is recorded "
            f"for it; the verifiers read the proof as usual.", verbose)
    elif outcome == "match":
        ctx.ui.log(f"  Every computation of {lemma_id} reproduced; recorded in "
            f"the run's computations stream.", verbose)

    cited_lemmas = [str(c) for c in node.get("cited_lemmas", [])]
    cited_references = [str(r) for r in node.get("cited_references", [])]
    statement = str(node.get("statement") or "")
    proof = dag.proof(lemma_id)

    notes = [
        {
            "verifier": str(ref.get("verifier") or ""),
            "model": str(ref.get("model") or ""),
            "date": str(ref.get("date") or ""),
            "verdict": str(ref.get("verdict") or ""),
            "hash": str(ref.get("hash") or ""),
            "hash_matches_current": refutations.is_counting(ref, h),
            "justification": str(ref.get("justification") or ""),
        }
        for _path, ref in on_file
    ]
    refutation_block = ""
    if notes:
        refutation_block = (
            "Known refutations (rejections of this lemma on record, oldest "
            "first; each carries the verifier's justification. A hash that "
            "no longer matches the lemma's is an objection to an earlier "
            "version of it, kept so the objection is not lost when the "
            "lemma changes; one that still matches is an objection to the "
            "proof as it stands):\n"
            f"{json.dumps(notes, indent=2)}\n\n"
        )

    verifier_user = (
        f"Conjecture:\n{conjecture_text}\n\n"
        "Cited results (statements of exactly the lemmas and references "
        "the proof declares it used; nothing else is available to it):\n"
        f"{json.dumps(verifier_context(dag, cited_lemmas, cited_references, references), indent=2)}\n\n"
        f"{refutation_block}"
        "Target lemma:\n"
        f"{json.dumps({'id': lemma_id, 'statement': statement}, indent=2)}\n\n"
        f"Proposed proof:\n{proof}"
        + computations_block(ctx, node)
    )
    steps = verifier_steps(node)
    ctx.status(lemma=lemma_id, attempt="")
    for step, agent in enumerate(steps, 1):
        role = role_of(agent)
        decision, justification = _run_verifier(
            ctx,
            role, verifier_prompt(ctx, verifier_sys, agent), verifier_user, verbose,
        )
        ctx.ui.log(
            f"Verifier {step}/{len(steps)} ({role}): "
            f"{decision.upper() or '???'} — {justification}",
            verbose,
        )
        kind = verdicts.record_kind(decision)
        if kind == verdicts.REFUTATION:
            path = refutations.record(
                ctx.paths.conjecture_root, lemma_id, h, ctx.user,
                _recorded_model(ctx), "reject", justification, node["file"],
            )
            ctx.run.dag = ctx.dag.with_refutation(path, refutations.load(path))
            ctx.ui.log(f"Refutation for {lemma_id} written: {path}", verbose)
            ctx.status(event=f"verifier {step}/{len(steps)} rejected")
            return "rejected", justification
        if kind == verdicts.NOTHING:
            ctx.ui.log(
                f"No verdict on {lemma_id} from {role}; no refutation is "
                f"written and nothing is certified. Re-run once the server "
                f"is answering.",
                verbose,
            )
            return "no_verdict", justification
    try:
        line = certificates.record(
            writer(ctx, certificates.DIR_NAME), lemma_id, h, ctx.user,
            _recorded_model(ctx),
        )
    except OSError as e:
        # The acceptance is only as good as its record: without the
        # certificate the lemma stays suspended, and dismissing the
        # refutations now would delete the objections while leaving
        # nothing in their place. Keep them, and say the verification has
        # to be re-run — the no-verdict outcome: nothing was written.
        ctx.ui.log(
            f"{lemma_id} passed the verifier checks, but the certificate "
            f"could not be written ({e}); its refutations are kept and "
            f"nothing is recorded. Re-run the verification once the "
            f"certificate file is writable.",
            verbose,
        )
        return "no_verdict", f"the certificate could not be written: {e}"
    ctx.run.dag = ctx.dag.with_certificate(line)
    ctx.ui.log(
        f"Certificate for {lemma_id}: accepted by {ctx.user} (model {_recorded_model(ctx)}).",
        verbose,
    )
    dismissed = []
    for path, _ref in on_file:
        try:
            refutations.delete(path)
            dismissed.append(path)
            ctx.ui.log(f"Refutation dismissed: {path}", verbose)
        except OSError as e:
            ctx.ui.log(f"Could not delete the refutation {path}: {e}", verbose)
    ctx.run.dag = ctx.dag.without_refutations(dismissed)
    ctx.ui.log(
        f"Lemma {lemma_id} passed all {len(steps)} verifier checks.",
        verbose,
    )
    return "accepted", None


def _verify_session(
    ctx,
    *,
    conjecture_dir: str,
    user_id: str,
    model: str,
    host: Optional[str],
    api_key: Optional[str],
    backend_choice: str,
    num_ctx: Optional[int],
    verbose: bool,
    parser: argparse.ArgumentParser,
) -> None:
    """The shared setup of proofs verify and proofs repair: the backend
    and the conjecture's paths, the run's, minus the run's — no
    checkpoint, no hotkey, no SIGINT handler. A verify or repair writes
    only refutation files and its own run's streams (certificates,
    reproofs, executions). Fills the context main() fills, so
    the in-memory DAG (ctx.dag) and the verifier plumbing below run
    unchanged. user_id is the user running the verification: the verifier
    of record in any certificate issued and any refutation written.

    The certificates and refutations this session writes name the model
    _recorded_model() reports, not the --model request.
    """
    try:
        ctx.settings = settings_mod.resolve(
            workspace.find_root(Path(conjecture_dir).resolve()),
        )
    except settings_mod.SettingsError as e:
        parser.error(str(e))
    ctx.model.name = model
    try:
        connection = llm.connect(
            model, host, REQUEST_TIMEOUT, api_key, backend_choice, num_ctx,
        )
    except llm.AuthError as e:
        parser.error(str(e))
    ctx.attach_model(connection)
    if connection.notes:
        ctx.ui.log(connection.notes, verbose)
    ctx.user = user_id
    try:
        paths = workspace.resolve(conjecture_dir, user_id)
    except workspace.ConjectureNotFound as e:
        parser.error(str(e))
    ctx.paths = Paths.from_run_paths(paths)
    # A run of its own, with a lock and no checkpoint (core/checkpoint.py).
    attach_run(ctx, open_fresh_run(paths.root, user_id), checkpoint=False)
    ctx.ui.log(workspace.describe(paths), verbose)


def verify_entry(
    *,
    lemma_id: str,
    root: str,
    full: bool,
    user_id: str,
    model: str,
    host: Optional[str],
    api_key: Optional[str],
    backend: str,
    num_ctx: Optional[int],
    verbose: bool,
    parser: argparse.ArgumentParser,
    run_code: bool = False,
    code: Optional[Code] = None,
) -> int:
    """proofs verify NAME LEMMA_ID [--full]: the three verifier checks on
    the stored proof of a lemma that is already in the DAG, with the
    refutation and certificate bookkeeping that verify carries — a
    rejection writes a refutation file, an acceptance issues a certificate
    and dismisses the refutations whose justifications were on the table.

    root is the conjecture directory. With --full the lemma's cited lemmas are verified first, in
    dependency order, the target last, and a rejection anywhere in that
    order stops the verification. Returns the exit code: 0 when every
    lemma verified was accepted, 1 when one was rejected (its refutation
    written), EXIT_NO_VERDICT when a verifier gave no verdict (nothing
    written; re-run once the server answers).

    run_code (--run-code, which needs --tools) re-executes each lemma's
    scripts before its verifiers read it (_reexecute), asking first.
    """
    ctx = Context(user=user_id, code=code if code is not None else Code())
    ctx.code.run_code = run_code
    _verify_session(
        ctx,
        conjecture_dir=root, user_id=user_id, model=model, host=host,
        api_key=api_key, backend_choice=backend, num_ctx=num_ctx,
        verbose=verbose, parser=parser,
    )
    dag = ctx.dag
    if lemma_id not in dag.lemmas:
        parser.error(f"{lemma_id} is not a lemma of the DAG {root}")
    order = _citation_closure(dag, lemma_id) if full else [lemma_id]
    # Suspension is computed, not stored, and it does not reach in here:
    # verify loads the suspended lemmas too (see suspension.py) —
    # it is the path back to them. The warning names a lemma the run must
    # not build on as one.
    susp = dag.suspension()
    for lid in order:
        if susp.has(lid):
            ctx.ui.log(
                f"{lid} is suspended ({'; '.join(susp.reasons(lid))}); "
                f"it is loaded and verified anyway — the verification is how "
                f"the suspension is lifted.",
                verbose,
            )
    conjecture_text = load_file(ctx.paths.conjecture_file)
    verifier_sys = _verifier_prompts(ctx, parser)
    ctx.ui.log(
        f"Verifying {len(order)} lemma(s): " + ", ".join(order),
        verbose,
    )
    try:
        for lid in order:
            outcome, _just = verify_stored_lemma(
                ctx,
                ctx.dag, ctx.dag.references, lid, conjecture_text,
                verifier_sys, verbose,
            )
            if outcome == "rejected":
                ctx.ui.log(f"{lid} rejected; verification stops here.", verbose)
                return 1
            if outcome == "no_verdict":
                ctx.ui.log(
                    f"{lid} got no verdict; verification stops here, "
                    f"with nothing written.",
                    verbose,
                )
                return EXIT_NO_VERDICT
    except merkle.MerkleCycleError as e:
        ctx.ui.log(f"{e}", verbose)
        return 1
    return 0
