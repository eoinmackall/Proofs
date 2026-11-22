"""Runs and their checkpoints: where each loop stopped, so an interrupted
run resumes there instead of re-running what it had finished.

Every command that calls a model is a run, with a run id of its own: 16
random hex digits, drawn when the run starts. A run's directory is local
to the machine and gitignored:

    conjectures/NAME/checkpoints/<user_id>/<run_id>/
        checkpoint.json     the checkpoint (proofs run only)
        run.lock            held by the process running it, for as long
                            as it runs (locking.hold)

The lock is a sidecar rather than the checkpoint file itself because the
checkpoint is replaced whole on every write (temp file, os.replace), and a
lock on the old inode would exclude nobody. Two processes can therefore
never run the same run, while separate runs of one user on one conjecture
go side by side, each with its own checkpoint.

Which run proofs run continues (open_run): with --resume RUN_ID, that run;
with --fresh, a new run with an empty checkpoint; with neither, the user's
most recent run on the conjecture — the newest checkpoint — unless another
process holds it, in which case a new run starts. proofs verify and proofs
repair always start a new run (open_fresh_run), with a lock and no
checkpoint. A run that settles the conjecture deletes its checkpoint, so
the next proofs run starts a new run; any other exit keeps it, with no
loop in flight after a clean one, so the next run starts from its plan.

The checkpoint is written after every step of a loop, not only at the start
of a prover round. Two shapes, by the kind of run:

    serial (version 1)
        {"version": 1, "iteration": N, "rejected_lemmas": [...],
         "plan": {"plan_summary", "candidate_lemmas"},     when there is one
         "in_flight": {...}}                                when a lemma is

    parallel (version 2)
        {"version": 2,
         "loops": {"loop-0": {"in_flight": {...} or null,
                              "iterations_used": n}, ...},
         "rejected_lemmas": [...],
         "plan": {"plan_summary", "priorities"} or null,
         "buffer": [{"lemma": {...}, "claimed_by": "loop-0" or null}, ...]}

iteration is the next iteration a resuming serial run starts at;
rejected_lemmas the planner's reject list; plan the latest planner output,
which a resuming run uses instead of calling the planner (with no
staleness check: the planner runs again as soon as the run moves on);
buffer the parallel run's waiting candidates and which loop has claimed
each. Every candidate carries the permanent id it was given when proposed
(core.dag.new_lemma_id); a resumed run reserves all of them, so it never
draws one again. An in-flight record is one loop's lemma:

    lemma_id, claimed_id, target      the lemma, the claim it started from,
                                      and its statement as the reviser last
                                      left it
    attempt                           the prover round
    feedback, guidance, attempt_notes what the next prover round is told
    last_proof                        the last proof of it (what the reviser
                                      judges its difficulty by)
    last_code_runs                    the script ids that proof cited (the
                                      carry-over the next prover round is
                                      shown)
    code_dir                          the loop's code directory under the
                                      run's scratch (tools/run_code/files.py)
    stage                             "prove", "verify", "citations" or "revise": the
                                      step the loop resumes at
    proof                             the prover's parsed output for this
                                      round (proof, cited_lemmas,
                                      cited_references, code_runs: script
                                      ids), written
                                      as soon as the prover returns
    prover_model                      the model that wrote proof, which the
                                      committed lemma names (a resumed run
                                      may use another)
    verdicts                          each finished verifier's verdict:
                                      {agent, decision, justification,
                                       model, requested, hash}, written as
                                      each finishes

The citations stage keeps the accepted proof's hash and its selected
checks (including an empty selection), their receipt IDs, per-agent
verdicts and completed outcomes until the batch is committed. A check list
that does not validate is not resumed (the loop goes back to "verify"),
but its finished, unwritten results are handed on as abandoned_checks and
written by recovery, so a received rejection is never dropped.

A loop resuming at "verify" runs only the verifiers missing from verdicts,
and at "revise" goes straight to the reviser. A saved verdict counts only
while the hash it checked is the proof's hash now and it came from the
model this run uses (loops.proof); otherwise it is dropped and its
verifier runs again. A certificate is issued at the commit, from the
accepting verdicts.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set

from . import locking

CHECKPOINTS_DIRNAME = "checkpoints"
CHECKPOINT_FILENAME = "checkpoint.json"
RUN_LOCK_FILENAME = "run.lock"

STAGES = ("prove", "verify", "revise", "citations")


class CitationCheckpointError(OSError):
    """No durable intent for a batch that might write verification evidence."""


def report_checkpoint_stop(ctx, err: OSError, verbose: bool = True) -> None:
    """The clean stop for a CitationCheckpointError. It is raised before any
    agent is called or record written, so the files on disk are consistent
    and the last checkpoint that did land is still a valid resume point."""
    ctx.ui.log(
        f"Could not write the checkpoint ({err}); stopping the run. Nothing "
        f"already recorded is lost: re-run the same command to resume once "
        f"the checkpoint can be written.",
        verbose,
    )


class RunError(Exception):
    """The run asked for cannot be run: missing, or running elsewhere."""


@dataclass
class Run:
    id: str
    dir: Path
    # The descriptor holding run.lock: kept open for the life of the process.
    lock_fd: int
    # Whether this run continues one that has a checkpoint.
    resumed: bool

    @property
    def checkpoint_file(self) -> Path:
        return self.dir / CHECKPOINT_FILENAME


def attach_run(ctx, run: Run, checkpoint: bool = True) -> None:
    """Make run the command's run: its id, its checkpoint (none for verify
    and repair), and the scratch directory its code executions write under
    (scratch/<user_id>/<run_id>/)."""
    ctx.run_id = run.id
    ctx.run.handle = run
    ctx.paths.checkpoint_file = str(run.checkpoint_file) if checkpoint else ""
    if ctx.code.runner is not None:
        ctx.code.runner.scratch_dir = ctx.code.runner.scratch_dir / run.id


def runs_dir(conjecture_root: Path, user_id: str) -> Path:
    return Path(conjecture_root) / CHECKPOINTS_DIRNAME / user_id


def new_run_id() -> str:
    return secrets.token_hex(8)


def _take(directory: Path) -> Optional[int]:
    return locking.hold(directory / RUN_LOCK_FILENAME)


def open_fresh_run(conjecture_root: Path, user_id: str) -> Run:
    """A new run: a new run id, its directory, and its lock."""
    while True:
        run_id = new_run_id()
        directory = runs_dir(conjecture_root, user_id) / run_id
        if directory.exists():
            continue
        directory.mkdir(parents=True)
        fd = _take(directory)
        if fd is not None:
            return Run(run_id, directory, fd, resumed=False)


def open_run(
    conjecture_root: Path,
    user_id: str,
    fresh: bool = False,
    resume: Optional[str] = None,
) -> Run:
    """The run proofs run continues (see the module docstring)."""
    base = runs_dir(conjecture_root, user_id)
    if resume is not None:
        directory = base / resume
        if not (directory / CHECKPOINT_FILENAME).is_file():
            known = ", ".join(
                d.name for d in sorted(base.iterdir())
                if (d / CHECKPOINT_FILENAME).is_file()
            ) if base.is_dir() else ""
            raise RunError(
                f"there is no run {resume} with a checkpoint in {base}"
                + (f"; the runs with one are {known}" if known else "")
            )
        fd = _take(directory)
        if fd is None:
            raise RunError(
                f"run {resume} is running in another process; it cannot be "
                f"resumed twice"
            )
        return Run(resume, directory, fd, resumed=True)
    if not fresh and base.is_dir():
        with_checkpoint = [
            d for d in base.iterdir()
            if (d / CHECKPOINT_FILENAME).is_file()
        ]
        if with_checkpoint:
            latest = max(
                with_checkpoint,
                key=lambda d: (d / CHECKPOINT_FILENAME).stat().st_mtime,
            )
            fd = _take(latest)
            if fd is not None:
                return Run(latest.name, latest, fd, resumed=True)
    return open_fresh_run(conjecture_root, user_id)


def note_rejection(
    ctx,
    rejected: List[Dict[str, str]], lemma_id: str, statement: str
) -> None:
    """Put a lemma on the reject list, in place: most recent last, an
    earlier entry with the same id or statement replaced, and the list cut
    to the newest run.max_rejected_lemmas."""
    rejected[:] = [
        e for e in rejected
        if e["id"] != lemma_id and e["statement"] != statement
    ]
    rejected.append({"id": lemma_id, "statement": statement})
    excess = len(rejected) - ctx.settings.max_rejected_lemmas
    if excess > 0:
        del rejected[:excess]


def drop_rejection(rejected: List[Dict[str, str]], statement: str) -> None:
    """Take a statement off the reject list, in place: it is proved now.
    By statement, since a rejected lemma proposed again comes back under a
    new id."""
    rejected[:] = [e for e in rejected if e["statement"] != statement]


def clean_rejected(ctx, raw: Any) -> List[Dict[str, str]]:
    """A checkpoint's reject list, validated and cut to the cap. Anything
    that is not an {"id", "statement"} entry is dropped."""
    out: List[Dict[str, str]] = []
    for e in raw if isinstance(raw, list) else []:
        if not isinstance(e, dict):
            continue
        lid = str(e.get("id") or "").strip()
        stmt = str(e.get("statement") or "").strip()
        if lid and stmt:
            note_rejection(ctx, out, lid, stmt)
    return out


def save_checkpoint(
    ctx,
    iteration: int,
    rejected_lemmas: List[Dict[str, str]],
    in_flight: Optional[Dict[str, Any]],
    plan: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> None:
    """Write the serial checkpoint atomically (see the module docstring)."""
    if not ctx.paths.checkpoint_file:
        return
    data: Dict[str, Any] = {
        "version": 1,
        "iteration": int(iteration),
        "rejected_lemmas": [dict(e) for e in rejected_lemmas],
    }
    if plan is not None:
        data["plan"] = plan
    if in_flight is not None:
        data["in_flight"] = in_flight
    _atomic_write_checkpoint_file(ctx, data, verbose)


def _read(ctx, verbose: bool, discard_corrupt: bool) -> Optional[Dict[str, Any]]:
    path = ctx.paths.checkpoint_file
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        ctx.ui.log(f"Checkpoint {path} is unreadable ({e}); ignoring it.", verbose)
        if discard_corrupt:
            discard_checkpoint(ctx)
        return None
    if not isinstance(data, dict):
        ctx.ui.log(f"Checkpoint {path} is malformed; ignoring it.", verbose)
        if discard_corrupt:
            discard_checkpoint(ctx)
        return None
    return data


def load_checkpoint(ctx, verbose: bool = True) -> Optional[Dict[str, Any]]:
    """Read and validate the serial checkpoint.

    Returns {"iteration", "rejected_lemmas", "in_flight", "plan"}, or None
    when there is no usable checkpoint. A checkpoint that cannot be trusted
    is discarded rather than fatal: the DAG is the permanent record, and the
    checkpoint only saves re-paying for a cancelled run's completed work.
    """
    data = _read(ctx, verbose, discard_corrupt=True)
    if data is None:
        return None
    name = os.path.basename(ctx.paths.checkpoint_file)
    version = data.get("version")
    if version == 2:
        # A parallel run's checkpoint is valid state for a parallel run;
        # the serial loader must not read it as (and discard it as) a
        # corrupt serial one.
        ctx.ui.log(f"Run {ctx.run_id}'s {name} is a parallel-mode "
                   f"checkpoint; re-run with --parallel to resume it.", verbose)
        return None
    if version != 1:
        ctx.ui.log(f"Checkpoint {ctx.paths.checkpoint_file} has an unknown "
                   f"version ({version!r}); ignoring it.", verbose)
        discard_checkpoint(ctx)
        return None
    iteration = data.get("iteration")
    if not isinstance(iteration, int) or iteration < 1:
        ctx.ui.log(f"Checkpoint {ctx.paths.checkpoint_file} has no usable "
                   f"iteration; ignoring it.", verbose)
        discard_checkpoint(ctx)
        return None
    in_flight = None
    if data.get("in_flight") is not None:
        in_flight = _validate_in_flight(ctx, data["in_flight"])
        if in_flight is None:
            ctx.ui.log(
                "Checkpoint's in-flight state is unusable (bad lemma, "
                "statement, or prover round); planning as usual instead.",
                verbose,
            )
    out = {
        "iteration": iteration,
        "rejected_lemmas": clean_rejected(ctx, data.get("rejected_lemmas")),
        "in_flight": in_flight,
        "plan": _validate_plan(data.get("plan"), "candidate_lemmas"),
    }
    _reserve_ids(ctx, [in_flight], out["rejected_lemmas"],
                 (out["plan"] or {}).get("candidate_lemmas") or [])
    return out


def _reserve_ids(ctx, in_flight: List[Any], rejected: List[Dict[str, str]],
                 candidates: List[Dict[str, Any]]) -> None:
    """Mark every lemma id a resumed checkpoint still holds as handed out
    (ctx.run.issued_ids), so core.dag.new_lemma_id never draws one again."""
    ids = [str(e.get("id") or "") for e in [*rejected, *candidates]]
    for f in in_flight:
        if f is not None:
            ids += [f["lemma_id"], f["claimed_id"]]
    with ctx.run.id_lock:
        ctx.run.issued_ids.update(i for i in ids if i)


def _validate_plan(raw: Any, list_key: str) -> Optional[Dict[str, Any]]:
    """A saved plan, or None: a plan_summary, and the list the planner
    writes — candidate_lemmas (serial: {"id", "statement"} objects) or
    priorities (parallel: strings)."""
    if not isinstance(raw, dict):
        return None
    items = raw.get(list_key)
    if not isinstance(items, list):
        return None
    if list_key == "candidate_lemmas":
        items = [c for c in items if isinstance(c, dict)
                 and str(c.get("id") or "").strip()
                 and str(c.get("statement") or "").strip()]
    else:
        items = [p for p in items if isinstance(p, str) and p.strip()]
    return {"plan_summary": str(raw.get("plan_summary") or ""), list_key: items}


def _validate_verdicts(raw: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for v in raw if isinstance(raw, list) else []:
        if not isinstance(v, dict) or not isinstance(v.get("agent"), str):
            continue
        out.append({
            "agent": v["agent"],
            "decision": str(v.get("decision") or ""),
            "justification": str(v.get("justification") or ""),
            "model": str(v.get("model") or ""),
            "requested": str(v.get("requested") or ""),
            "hash": v.get("hash") if isinstance(v.get("hash"), str) else None,
        })
    return out


def flights_in(data: Any) -> Iterator[Dict[str, Any]]:
    """The in-flight records in either checkpoint shape (also used by prune)."""
    if not isinstance(data, dict):
        return
    loops = data.get('loops')
    slots = loops.values() if isinstance(loops, dict) else (data,)
    for slot in slots:
        if isinstance(slot, dict) and isinstance(slot.get('in_flight'), dict):
            yield slot['in_flight']


_CHECK_ID = re.compile(r'[0-9a-f]{32}')
_CHECK_KEYS = ('lemma_id', 'hash', 'file', 'check_id')


def _well_formed_check(check: Any) -> bool:
    """The fields every check carries, and that its record would be written from."""
    return (isinstance(check, dict)
            and all(isinstance(check.get(k), str) and check[k] for k in _CHECK_KEYS)
            and bool(_CHECK_ID.fullmatch(check['check_id'])))


def _salvage_checks(raw: Any) -> List[Dict[str, Any]]:
    """The finished, unwritten results in a check list that cannot be
    resumed, each cut down to what its record needs: lemma_id, hash, file,
    check_id, outcome, model, and a rejection's justification. Objections
    are left out, so a salvaged acceptance dismisses no refutation."""
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for check in raw if isinstance(raw, list) else []:
        if (not _well_formed_check(check) or check['check_id'] in seen
                or check.get('published')
                or check.get('outcome') not in ('accepted', 'rejected')
                or not isinstance(check.get('model'), str)
                or (check['outcome'] == 'rejected'
                    and not isinstance(check.get('justification'), str))):
            continue
        seen.add(check['check_id'])
        keys = (*_CHECK_KEYS, 'outcome', 'model') + (
            ('justification',) if check['outcome'] == 'rejected' else ())
        out.append({k: check[k] for k in keys})
    return out


def _validate_checks(raw: Any) -> Optional[List[Dict[str, Any]]]:
    """A selected (possibly empty) queue; None if it must be selected anew."""
    if not isinstance(raw, list):
        return None
    checks: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for check in raw:
        if (not _well_formed_check(check)
                or check['lemma_id'] in seen
                or check.get('outcome') not in (None, 'accepted', 'rejected')
                or not isinstance(check.get('objections', {}), dict)):
            return None
        seen.add(check['lemma_id'])
        if check.get('outcome') and not isinstance(check.get('model'), str):
            return None
        if check.get('outcome') == 'rejected' and not isinstance(check.get('justification'), str):
            return None
        checks.append(dict(check, verdicts=_validate_verdicts(check.get('verdicts'))))
    return checks


def _validate_in_flight(ctx, raw: Any) -> Optional[Dict[str, Any]]:
    """The in-flight state a resuming loop is handed, or None (see the
    module docstring for the fields). Every field but the lemma, its
    statement and the prover round falls back to "nothing done yet": a
    record without a usable stage resumes at the prover."""
    if not isinstance(raw, dict):
        return None
    lemma_id = str(raw.get("lemma_id") or "")
    target = raw.get("target")
    attempt = raw.get("attempt")
    statement = (
        str(target.get("statement") or "").strip() if isinstance(target, dict) else ""
    )
    if not (
        lemma_id
        and statement
        and isinstance(attempt, int)
        and 1 <= attempt
        and (ctx.settings.max_proof_attempts < 0
             or attempt <= ctx.settings.max_proof_attempts)
    ):
        return None
    stage = raw.get("stage") if raw.get("stage") in STAGES else "prove"
    proof = raw.get("proof") if isinstance(raw.get("proof"), dict) else None
    if proof is None:
        stage = "prove"
    checks = _validate_checks(raw.get('checks')) if stage == 'citations' else None
    if stage == 'citations':
        cited = proof.get('cited_lemmas')
        if (checks is None or not isinstance(cited, list) or
                not {c['lemma_id'] for c in checks}.issubset(str(lid) for lid in cited)):
            stage, checks = 'verify', None
    # A list that cannot be resumed may still hold finished results nobody
    # wrote; recovery writes them (citation_checks.recover_checks).
    abandoned = _salvage_checks(raw.get('checks')) if checks is None else []
    return {
        **({'checks': checks, 'audit_hash': raw.get('audit_hash')} if checks is not None else {}),
        **({'abandoned_checks': abandoned} if abandoned else {}),
        "lemma_id": lemma_id,
        "claimed_id": str(raw.get("claimed_id") or lemma_id),
        "target": {"id": lemma_id, "statement": statement},
        "attempt": attempt,
        "feedback": [n for n in (raw.get("feedback") or []) if isinstance(n, str)],
        "guidance": str(raw.get("guidance") or ""),
        "attempt_notes": [
            n for n in (raw.get("attempt_notes") or []) if isinstance(n, str)
        ],
        "last_proof": str(raw.get("last_proof") or ""),
        "last_code_runs": [
            r for r in (raw.get("last_code_runs") or []) if isinstance(r, str)
        ],
        "code_dir": str(raw.get("code_dir") or ""),
        "stage": stage,
        "proof": proof,
        "prover_model": str(raw.get("prover_model") or "") if proof else "",
        "verdicts": _validate_verdicts(raw.get("verdicts")) if proof else [],
    }


def load_parallel_checkpoint(
    ctx, loop_ids: List[str], verbose: bool = True
) -> Optional[Dict[str, Any]]:
    """Read and validate the parallel (version 2) checkpoint.

    Returns {"loops": {loop_id: {"in_flight", "iterations_used"}},
    "rejected_lemmas", "plan", "buffer"}, or None when there is nothing to
    resume. A checkpoint written by a run with a different set of loops is
    ignored rather than half-applied: per-loop slots are not transferable
    between runs of different shape.
    """
    data = _read(ctx, verbose, discard_corrupt=False)
    if data is None:
        return None
    name = os.path.basename(ctx.paths.checkpoint_file)
    if data.get("version") != 2:
        ctx.ui.log(f"Run {ctx.run_id}'s {name} is not a parallel-mode "
                   f"checkpoint; ignoring it.", verbose)
        return None
    raw_loops = data.get("loops")
    if not isinstance(raw_loops, dict) or set(raw_loops) != set(loop_ids):
        ctx.ui.log(f"Run {ctx.run_id}'s {name} was written by a run with a "
                   f"different set of loops; ignoring it.", verbose)
        return None
    loops: Dict[str, Dict[str, Any]] = {}
    for lid in loop_ids:
        raw = raw_loops.get(lid)
        if not isinstance(raw, dict):
            raw = {}
        in_flight = _validate_in_flight(ctx, raw.get("in_flight"))
        if in_flight is None and raw.get("in_flight") is not None:
            ctx.ui.log(f"{name} has a malformed in-flight record for {lid}; "
                       f"that loop starts fresh.", verbose)
        iterations_used = raw.get("iterations_used", 0)
        if not isinstance(iterations_used, int) or iterations_used < 0:
            iterations_used = 0
        loops[lid] = {"in_flight": in_flight, "iterations_used": iterations_used}
    buffer = []
    for e in data.get("buffer") or []:
        if not isinstance(e, dict) or not isinstance(e.get("lemma"), dict):
            continue
        lemma = e["lemma"]
        if not str(lemma.get("id") or "").strip() or not str(lemma.get("statement") or "").strip():
            continue
        if lemma["id"] in ctx.dag.lemmas:
            continue  # committed before the checkpoint caught up: spent
        claimed = e.get("claimed_by")
        buffer.append({
            "lemma": lemma,
            "claimed_by": claimed if claimed in loops else None,
        })
    out = {
        "loops": loops,
        "rejected_lemmas": clean_rejected(ctx, data.get("rejected_lemmas")),
        "plan": _validate_plan(data.get("plan"), "priorities"),
        "buffer": buffer,
    }
    _reserve_ids(ctx, [lp["in_flight"] for lp in loops.values()],
                 out["rejected_lemmas"], [e["lemma"] for e in buffer])
    return out


def _atomic_write_checkpoint_file(ctx, data: Dict[str, Any], verbose: bool) -> None:
    """The shared atomic write: temp file beside the checkpoint, then
    os.replace over it. Readers see either the whole old file or the whole
    new one, never a half-written mix.

    The temp file is named per writer (pid + thread id), not shared: two
    writers opening one .tmp truncate each other's in-progress dump, and
    the first replace to land would promote the intermixed bytes to the
    checkpoint file itself. With a private tmp, os.replace alone keeps the
    file whole for readers; the run's checkpoint lock (RunState), held by
    the callers that snapshot shared state, is what stops an older
    snapshot from landing over a newer one."""
    if not ctx.paths.checkpoint_file:
        return
    tmp = f"{ctx.paths.checkpoint_file}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp, ctx.paths.checkpoint_file)
    except OSError as e:
        ctx.ui.log(f"Could not write checkpoint {ctx.paths.checkpoint_file}: {e}", verbose)
        try:
            os.remove(tmp)
        except OSError:
            pass
        if any(f.get('stage') == 'citations' for f in flights_in(data)):
            # This is also the transaction intent: don't call agents or
            # append receipts if recovery wouldn't know their check IDs.
            raise CitationCheckpointError(str(e)) from e


def discard_checkpoint(ctx) -> None:
    """Remove the checkpoint file if present.

    Silent on failure: a checkpoint is a convenience file, and its absence
    is always a valid state (a fresh run).
    """
    if ctx.paths.checkpoint_file and os.path.exists(ctx.paths.checkpoint_file):
        try:
            os.remove(ctx.paths.checkpoint_file)
        except OSError:
            pass
