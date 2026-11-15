"""proofs prune: the garbage collector of the append-only files.

A lemma whose refutation still counts — the refutation's hash is the lemma's
current hash, so the proof in the files is the one a verification rejected,
and no repair has replaced it — is wrong as it stands. proofs prune removes
it: every record of it (its original and any reproofs) from the dags/
streams, every certificate of it from the certificates/ streams, and its
refutation files. The lemmas citing it stay, suspended — proofs repair can
still revive them — unless --with-dependents is given, which removes them
too, transitively. The computations/ streams and code/ are left alone.

Prune also compacts what it keeps. A lemma repaired one or more times has
its current version — a reproof, possibly in another run's stream —
rewritten in place, in the same file and line, as a plain lemma record
with no "supersedes", and every other version of it (the original, the
reproofs between, the losers of a fork) is deleted. The stream header of
that file still records who wrote the proof. And every certificate whose
hash is not the current hash of its lemma is deleted: it covers a proof
that is no longer in the files, or a lemma that is gone.

Prune is the one writer that rewrites a stream (streams.rewrite), and it
never touches a stream a live run is appending to: a run whose lock is held
on this machine (core/checkpoint.py) is skipped, and so is any lemma with a
record in one of its streams — it is neither removed nor compacted. The next prune, once the run has ended,
finishes the job. A run on another machine holds no lock here; if it
appends to a stream this prune rewrote, git reports a conflict in that
file, resolved by hand (keep every record line both sides have).

Prune also clears the machine's scratch: everything under scratch/ except a
live run's, and, for a run that is not live, the executions its checkpoint
still needs (the computations its in-flight proofs cite). Nothing is kept by
age. The run directories under checkpoints/ with nothing to resume go too.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

from ..workspace import layout as workspace
from . import certificates, locking, refutations, streams
from .checkpoint import CHECKPOINT_FILENAME, CHECKPOINTS_DIRNAME, RUN_LOCK_FILENAME
from .context import Context
from .dag import DIR_NAME as DAGS_DIRNAME
from .dag import LEMMA, REPROOF, dependents


def live_runs(root: Path) -> Set[Tuple[str, str]]:
    """The (user_id, run_id) of every run a process on this machine is
    running now: its run lock is held."""
    out: Set[Tuple[str, str]] = set()
    for user_id, run_id, directory in streams.run_dirs(root / CHECKPOINTS_DIRNAME):
        if locking.is_held(directory / RUN_LOCK_FILENAME):
            out.add((user_id, run_id))
    return out


def _needed_scratch(checkpoint: Path) -> Set[str]:
    """The executions a run's checkpoint still needs: the code_runs its
    in-flight proofs cite (serial and parallel shapes alike)."""
    try:
        data = json.loads(checkpoint.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    flights: List[Any] = []
    if isinstance(data, dict):
        flights.append(data.get("in_flight"))
        for loop in (data.get("loops") or {}).values():
            if isinstance(loop, dict):
                flights.append(loop.get("in_flight"))
    out: Set[str] = set()
    for f in flights:
        proof = f.get("proof") if isinstance(f, dict) else None
        runs = proof.get("code_runs") if isinstance(proof, dict) else None
        for r in runs if isinstance(runs, list) else [runs]:
            if isinstance(r, str) and r.strip():
                out.add(r.strip())
    return out


def _clear_scratch(root: Path, live: Set[Tuple[str, str]]) -> Tuple[int, int]:
    """Remove the scratch a run no longer needs (see the module
    docstring). Returns (executions removed, run directories removed)."""
    removed_execs = removed_runs = 0
    for user_id, run_id, directory in streams.run_dirs(root / workspace.SCRATCH_DIRNAME):
        if (user_id, run_id) in live:
            continue
        keep = _needed_scratch(
            root / CHECKPOINTS_DIRNAME / user_id / run_id / CHECKPOINT_FILENAME)
        if not keep:
            shutil.rmtree(directory, ignore_errors=True)
            removed_runs += 1
            continue
        for d in directory.iterdir():
            if d.name in keep:
                continue
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
            else:
                d.unlink()
            removed_execs += 1
    return removed_execs, removed_runs


def _clear_runs(root: Path, live: Set[Tuple[str, str]]) -> int:
    """Remove the run directories under checkpoints/ that hold nothing to
    resume — no checkpoint (a verify or repair, or a settled run) — and no
    live run. Returns how many went."""
    removed = 0
    for user_id, run_id, directory in streams.run_dirs(root / CHECKPOINTS_DIRNAME):
        if (user_id, run_id) in live or (directory / CHECKPOINT_FILENAME).exists():
            continue
        shutil.rmtree(directory, ignore_errors=True)
        removed += 1
    return removed


def _rewrite(base: Path, live: Set[Tuple[str, str]], edit) -> int:
    """Rewrite every stream under base that is not a live run's through
    edit(path, line, record) (see streams.rewrite). Returns how many records
    went."""
    removed = 0
    for user_id, run_id, directory in streams.run_dirs(base):
        if (user_id, run_id) in live:
            continue
        for path in streams.stream_files(directory):
            removed += streams.rewrite(
                path, lambda line, rec, path=path: edit(path, line, rec))
    return removed


def prune_entry(
    *,
    dir: str,
    user_id: str,
    parser: argparse.ArgumentParser,
    with_dependents: bool = False,
) -> int:
    """proofs prune NAME [--with-dependents] (see the module docstring).
    No model is involved. Returns the exit code: 0 when the prune ran
    (including a no-op), 1 when a file could not be read or rewritten."""
    ctx = Context(user=user_id)
    try:
        paths = workspace.resolve(dir, user_id)
    except workspace.ConjectureNotFound as e:
        parser.error(str(e))
    ctx.paths.conjecture_root = str(paths.root)
    ctx.paths.dags_dir = str(paths.dags_dir)
    ctx.paths.references_file = str(paths.references)
    ctx.ui.log(workspace.describe(paths))
    root = Path(paths.root)
    try:
        dag = ctx.dag
        live = live_runs(root)

        refuted = {
            lid for lid in dag.lemmas
            if dag.hashes.get(lid) is not None and any(
                refutations.is_counting(ref, dag.hashes[lid])
                for _path, ref in dag.refutations_for(lid)
            )
        }
        wanted = set(refuted)
        if with_dependents:
            # A lemma citing one an earlier prune removed (an id no stream
            # holds) depends on a removed lemma all the same.
            gone = {
                c for node in dag.lemmas.values()
                for c in node.get("cited_lemmas", []) if c not in dag.lemmas
            }
            wanted |= dependents(dag, refuted | gone)
        # A lemma whose versions state different things is a person's to
        # settle: prune leaves it, and its certificates and refutations,
        # alone.
        conflicted = set(dag.conflicts)
        wanted -= conflicted

        def certified(lid: str) -> bool:
            h = dag.hashes.get(lid)
            return bool(h) and any(
                c.get("lemma_id") == lid and certificates.is_valid(c, h)
                for c in dag.certificates
            )

        # A lemma with a record in a live run's stream is left for the next
        # prune: that stream may not be rewritten now.
        live_ids: Set[str] = set()
        record_count: Dict[str, int] = {}
        for u, r, _p, _n, rec in streams.records(root / DAGS_DIRNAME):
            if rec.get("type") not in (LEMMA, REPROOF):
                continue
            lid = str(rec.get("id") or "")
            record_count[lid] = record_count.get(lid, 0) + 1
            if (u, r) in live:
                live_ids.add(lid)
        in_live = wanted & live_ids
        remove = wanted - in_live
        # What compaction rewrites: every lemma kept with more than its one
        # record, or whose one record is a reproof (its original gone) —
        # but a fork only once the version it keeps is certified.
        compact = {
            lid for lid, node in dag.lemmas.items()
            if lid not in remove and lid not in live_ids and lid not in conflicted
            and (record_count.get(lid, 0) > 1 or node.get("type") == REPROOF)
            and (lid not in dag.forked or certified(lid))
        }
        # Every other record of a compacted lemma goes, so its current one
        # must be exactly where the DAG says; a lemma whose is not (a file
        # changed since the load) is left for the next prune.
        current: Set[Tuple[Path, int]] = set()
        moved: Set[str] = set()
        files: Dict[Path, Dict[int, Dict[str, Any]]] = {}
        for lid in sorted(compact):
            where = ((root / dag.lemmas[lid]["file"]).resolve(), dag.lemmas[lid]["line"])
            if where[0] not in files:
                files[where[0]] = dict(streams.read_file(where[0])[1]) if where[0].exists() else {}
            if str(files[where[0]].get(where[1], {}).get("id") or "") != lid:
                moved.add(lid)
                continue
            current.add(where)
        compact -= moved

        def edit_lemma(path: Path, line: int, rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            lid = rec.get("id")
            if rec.get("type") not in (LEMMA, REPROOF) or (lid not in remove and lid not in compact):
                return rec
            if lid in remove:
                return None
            if (path.resolve(), line) not in current:
                return None
            out = {k: v for k, v in rec.items() if k != "supersedes"}
            out["type"] = LEMMA
            return out

        lemma_records = _rewrite(root / DAGS_DIRNAME, live, edit_lemma)
        # A certificate stays only while its hash is its lemma's current
        # hash: one of a removed lemma, of a lemma no stream holds, or of a
        # proof no longer current goes. Of the duplicates that stay (same
        # lemma, hash, verifier, model), the earliest is kept with the count
        # of them all, and the rest go. Live runs' streams are not touched.
        kept_ids = set(dag.lemmas) - remove

        def current_cert(rec: Mapping[str, Any]) -> bool:
            lid = rec.get("lemma_id")
            return (lid in kept_ids and bool(dag.hashes.get(lid))
                    and rec.get("hash") == dag.hashes[lid])

        groups: Dict[tuple, List[Tuple[str, int, Path, int, Dict[str, Any]]]] = {}
        stale_certs = 0
        for order, (u, r, path, line, rec) in enumerate(
                streams.records(certificates.dir_for(root))):
            if (u, r) in live or rec.get("type") != certificates.TYPE \
                    or rec.get("lemma_id") in conflicted:
                continue
            if not current_cert(rec):
                stale_certs += 1
                continue
            groups.setdefault(certificates.key(rec), []).append(
                (str(rec.get("date") or ""), order, path.resolve(), line, rec))
        totals: Dict[Tuple[Path, int], int] = {}
        duplicates: Set[Tuple[Path, int]] = set()
        for members in groups.values():
            first, *rest = sorted(members, key=lambda m: m[:2])
            totals[(first[2], first[3])] = sum(certificates.count(m[4]) for m in members)
            duplicates |= {(m[2], m[3]) for m in rest}

        def edit_cert(path: Path, line: int, rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            if rec.get("type") != certificates.TYPE or rec.get("lemma_id") in conflicted:
                return rec
            where = (path.resolve(), line)
            if not current_cert(rec) or where in duplicates:
                return None
            n = totals.get(where, 1)
            return {**rec, "count": n} if n != certificates.count(rec) else rec

        _rewrite(certificates.dir_for(root), live, edit_cert)
        # A refutation goes with its lemma, and once it no longer counts
        # while the lemma's current version is certified: the proof it
        # rejected is not the one in the files any more.
        refutation_files = stale_refutations = 0
        for path, ref in dag.refutations:
            lid = ref.get("lemma_id")
            if lid in conflicted or lid in in_live:
                continue
            if lid in remove or lid not in dag.lemmas:
                refutations.delete(path)
                refutation_files += 1
            elif (not refutations.is_counting(ref, dag.hashes.get(lid) or "")
                    and certified(lid)):
                refutations.delete(path)
                stale_refutations += 1
        scratch_execs, scratch_runs = _clear_scratch(root, live)
        _clear_runs(root, live)
    except (OSError, streams.StreamError) as e:
        ctx.ui.log(f"Could not read or rewrite the conjecture's files: {e}")
        return 1

    for lid in sorted(remove):
        why = ("its refutation counts" if lid in refuted
               else "it depends on a removed lemma")
        ctx.ui.log(f"Removed {lid}: {why}.")
    for lid in sorted(moved):
        ctx.ui.log(f"Kept {lid} uncompacted for now: its current version is not "
                   f"where the DAG read it. Prune again.")
    for lid in sorted(compact):
        ctx.ui.log(f"Compacted {lid}: its current version, in "
                   f"{dag.lemmas[lid]['file']}, is now its only record.")
    for lid in sorted(in_live):
        ctx.ui.log(f"Kept {lid} for now: a run on this machine is writing to a "
                   f"stream that holds it. Prune again once it has ended.")
    if not with_dependents and remove:
        left = dependents(dag, remove) - remove
        if left:
            ctx.ui.log(
                f"{len(left)} lemma(s) citing what was removed stay, "
                f"suspended (proofs repair can revive them; --with-dependents "
                f"removes them): " + ", ".join(sorted(left)))
    for lid in sorted(conflicted):
        ctx.ui.log(f"Left {lid} alone: its versions state different things; "
                   f"delete all but one by hand.")
    merged = len(duplicates)
    did = (remove or compact or stale_certs or merged or refutation_files
           or stale_refutations or scratch_runs or scratch_execs)
    ctx.ui.log(
        f"Pruned {len(remove)} lemma(s) and compacted {len(compact)} "
        f"({lemma_records} record(s) removed); removed {stale_certs} stale "
        f"certificate(s) and merged {merged} duplicate(s); removed "
        f"{refutation_files} refutation file(s) with their lemmas and "
        f"{stale_refutations} stale; cleared {scratch_runs} scratch run(s) "
        f"and {scratch_execs} execution(s)."
        if did else "Nothing to prune."
    )
    return 0
