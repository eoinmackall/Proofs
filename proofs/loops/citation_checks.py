"""The post-acceptance stage of proofs run (never verify or repair)."""

from __future__ import annotations

import secrets
from typing import Any, Callable, Iterable, Mapping, Optional

from ..core import audits, certificates
from ..core.audits import Check
from ..core.dag import Dag, writer
from ..core.interrupts import _commit_section
from ..validity.steps import verifier_steps
from .reasoning import (
    VERIFIER_ROLE, VERIFIER_ROLES, VerifierFailover, _recorded_model,
    _requested_model, _same_model, pinned,
)
from .verify import review_stored


def selected(base: int, count: int) -> bool:
    """Exactly base**-count, without floats or constructing a huge power.
    Stops at the first losing draw (at most two draws on average for p>=2)."""
    return base != -1 and (base == 1 or all(secrets.randbelow(base) == 0 for _ in range(count)))


def _pending(dag: Dag, lid: str) -> Check:
    return {"lemma_id": lid, "hash": dag.hash(lid), "file": dag.lemmas[lid]["file"],
            "check_id": secrets.token_hex(16), "verdicts": []}


def _review_one(ctx, dag: Dag, check: Check, checkpoint: Callable[[], None],
                conjecture: str, verifier_sys: Mapping[str, str], verbose: bool,
                interrupted: Callable[[], bool]) -> tuple[str, Optional[str]]:
    lid = check["lemma_id"]
    on_file = dag.refutations_for(lid)
    objections = {path.name: ref for path, ref in on_file}
    if objections != check.get("objections", {}):
        check["verdicts"] = []  # agents must have read the same objections
    check["objections"] = objections
    steps = set(verifier_steps(dag.lemmas[lid]))
    # A verifier's call that fails over (--with-fallback) restarts the
    # check on the fallback: the verdicts the verifiers' own model gave no
    # longer match the model in use, and are dropped by the filter below,
    # so their verifiers run again there. The pins go when the check does.
    with pinned(ctx, {}):
        while True:
            kept = [v for v in check["verdicts"] if v["decision"] == "accept"
                    and v["hash"] == check["hash"]
                    and _same_model(ctx, VERIFIER_ROLE, v["model"], v["requested"])]
            check["verdicts"] = kept

            def progress(agent: str, decision: str, why: str,
                         kept: list = kept) -> None:
                with _commit_section(ctx):  # the received verdict and its checkpoint belong together
                    model = _recorded_model(ctx, VERIFIER_ROLE)
                    kept.append({"agent": agent, "model": model,
                                 "requested": _requested_model(ctx, VERIFIER_ROLE),
                                 "decision": decision, "justification": why, "hash": check["hash"]})
                    if decision == "reject":
                        check.update(outcome="rejected", justification=why, model=model)
                    elif steps <= {v["agent"] for v in kept}:
                        check.update(outcome="accepted", model=model)
                    checkpoint()

            try:
                return review_stored(
                    ctx, dag, dag.references, lid, conjecture, verifier_sys, verbose, on_file,
                    accepted=[v["agent"] for v in kept], on_verdict=progress,
                    should_stop=interrupted,
                )
            except VerifierFailover as e:
                ctx.ui.log(
                    f"  A verifier's model cannot serve the check of {lid} "
                    f"({e.why}); running it again on the fallback, "
                    f"{e.model.label()}.",
                    verbose,
                )
                ctx.loop.pinned.update({role: e.model for role in VERIFIER_ROLES})


def run_checks(ctx, state: dict[str, Any], save: Callable[[dict[str, Any]], None],
               node: Mapping[str, Any], parent_hash: Optional[str], conjecture: str,
               verifier_sys: Mapping[str, str], verbose: bool,
               should_stop: Optional[Callable[[], bool]]) -> bool:
    """Return True to commit, False to pause without losing this stage.
    Sample against one snapshot, check the latest suspension before each
    review. No DAG lock is held across a model call."""
    lids = list(dict.fromkeys(node["cited_lemmas"]))
    last_suspension, blocked = None, False

    def suspended(dag: Dag) -> bool:
        nonlocal last_suspension, blocked
        current = dag.suspension()
        if current is not last_suspension:
            blocked = any(lid in current for lid in lids)
            last_suspension = current
        return blocked

    def interrupted() -> bool:
        return bool(should_stop and should_stop()) or suspended(ctx.dag)

    dag = ctx.dag
    if state.get("stage") != "citations" or state.get("audit_hash") != parent_hash:
        base = ctx.settings.citation_recheck_base
        checks = [] if suspended(dag) else [
            _pending(dag, lid) for lid in lids if selected(base, dag.lemmas[lid].get("verified_count", 0))
        ]
        state.update(stage="citations", audit_hash=parent_hash, checks=checks)
        save(state)  # even an empty selection is a decision, not a reroll
    checks = state["checks"]
    for i, check in enumerate(checks):
        dag = ctx.dag
        if check.get("outcome") == "rejected" or suspended(dag):
            break
        lid = check["lemma_id"]
        if dag.hash(lid) != check["hash"]:
            # The selection stands, but a review of a previous version
            # cannot be used to certify this one.
            checks[i] = check = _pending(dag, lid)
            save(state)
        if check.get("outcome") == "accepted":
            continue
        if should_stop and should_stop():
            return False
        ctx.ui.log(f"Rechecking citation {lid} of {state['lemma_id']}.", verbose)
        outcome, why = _review_one(ctx, dag, check, lambda: save(state), conjecture,
                                   verifier_sys, verbose, interrupted)
        if outcome == "rejected":
            break  # including a rejection received while the run was stopping
        if outcome == "no_verdict":
            if suspended(ctx.dag):
                break
            ctx.ui.log(f"Citation {lid} is still pending ({why}); resume this run to retry.")
            return False
    return True


def _unwritten(check: Check) -> bool:
    return bool(check.get("outcome")) and not check.get("published")


def recover_checks(ctx, flights: Iterable[Optional[Mapping[str, Any]]], *,
                   keep_staged: bool = False) -> None:
    """Reconcile receipts in one pass. On an ordinary resume, keep unwritten
    acceptances staged for the parent. Rejections must be made visible before
    checking for resolution. When abandoning a parent, flush all its finished
    checks (even if the parent no longer restores).

    A flight's abandoned_checks (finished results the checkpoint salvaged
    from a check list it could not resume, checkpoint.py) are always
    flushed: no parent will carry them to a commit. Only those of lemmas
    still in the DAG are written."""
    pending = [f for f in flights if f and f["lemma_id"] not in ctx.dag.lemmas]
    staged = [c for f in pending for c in f.get("checks", ()) if _unwritten(c)]
    abandoned = [c for f in pending for c in f.get("abandoned_checks", ()) if _unwritten(c)]
    if not staged and not abandoned:
        return
    with _commit_section(ctx), ctx.run.dag_lock:
        dag = ctx.dag
        abandoned = [c for c in abandoned if c["lemma_id"] in dag.lemmas]
        keys = {c["check_id"] for c in (*staged, *abandoned)}
        existing = {c["check_id"] for c in dag.certificates if c.get("check_id") in keys}
        existing.update(r["check_id"] for _, r in dag.refutations if r.get("check_id") in keys)
        if keep_staged:
            staged = [c for c in staged if c["check_id"] in existing or c["outcome"] == "rejected"
                      or c["hash"] != dag.hashes.get(c["lemma_id"])]
        checks = [*staged, *abandoned]
        if not checks:
            return
        try:
            evidence = audits.write_checks(ctx, dag, checks, writer(ctx, certificates.DIR_NAME),
                                           existing=existing)
            ctx.run.dag = dag.with_evidence(evidence)
        except OSError as exc:
            raise audits.CitationCommitError(str(exc)) from exc
        for c in checks:
            c["published"] = True
