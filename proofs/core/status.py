"""proofs status: one lemma's hash, certificates, refutations and suspension."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..tools.run_code import computations
from ..workspace import layout as workspace
from . import certificates
from .context import Context
from .dag import _citation_closure, _settled, load_file


def status_entry(
    *,
    lemma_id: str,
    root: str,
    user_id: str,
    parser: argparse.ArgumentParser,
) -> int:
    """proofs status NAME LEMMA_ID: for each lemma in the named lemma's
    dependency closure — dependencies first, the target last, the order
    proofs verify --full verifies in — its hash, the stream file its current
    version is in, every valid certificate with how many times it was issued
    (certificates.tally: the one place a count is made), its suspension and
    its computations; then whether the conjecture is settled.

    No model is involved: a certificate is valid only while its hash
    matches the lemma's current Merkle hash, a function of the files alone.
    A certificate for a version of the lemma no longer current is stale,
    and counted, not listed. A lemma on a citation cycle has no hash, so no
    certificate is valid for it. Reads the files; writes nothing.

    Returns the exit code: 0 when the status was reported.
    """
    ctx = Context(user=user_id)
    try:
        paths = workspace.resolve(root, user_id)
    except workspace.ConjectureNotFound as e:
        parser.error(str(e))
    ctx.paths.conjecture_root = str(paths.root)
    ctx.paths.dags_dir = str(paths.dags_dir)
    ctx.paths.references_file = str(paths.references)
    ctx.ui.log(workspace.describe(paths))
    dag = ctx.dag
    for warning in dag.warnings:
        ctx.ui.log(f"warning: {warning}")
    if lemma_id not in dag.lemmas:
        parser.error(f"{lemma_id} is not a lemma of the DAG {root}")
    order = _citation_closure(dag, lemma_id)
    run_lines = computations.load_runs(Path(ctx.paths.conjecture_root))
    susp = dag.suspension()
    ctx.ui.log(
        f"Status for {lemma_id}: {len(order)} lemma(s) in its dependency "
        f"closure, dependencies first:"
    )
    for i, lid in enumerate(order, 1):
        h = dag.hashes.get(lid)
        node = dag.lemmas[lid]
        if h is None:
            ctx.ui.log(f"  {i}/{len(order)} {lid}  ({node['file']})")
            ctx.ui.log("     no Merkle hash (a citation cycle) — no "
                       "certificate is valid for it")
            continue
        ctx.ui.log(f"  {i}/{len(order)} {lid}  hash {h}  ({node['file']})")
        issued = certificates.tally(dag.certificates, lid)
        valid = [c for c in issued if c["hash"] == h]
        stale = sum(c["count"] for c in issued if c["hash"] != h)
        for c in valid:
            ctx.ui.log(
                f"     verifier={c['verifier']} model={c['model']} "
                f"last={c['date']} count={c['count']}"
            )
        if not valid:
            ctx.ui.log("     (no valid certificates)")
        if stale:
            ctx.ui.log(f"     {stale} stale certificate(s) for an earlier "
                       f"version are not listed")
        if susp.has(lid):
            ctx.ui.log(f"     suspended: {'; '.join(susp.reasons(lid))}")
        for line in computations.describe(Path(ctx.paths.conjecture_root), node, run_lines):
            ctx.ui.log(f"     {line}")
    # Whether the conjecture is settled, recomputed the way the run
    # computes it (resolution.py).
    if _settled(ctx, dag, susp.suspended, load_file(paths.conjecture), True) is None:
        ctx.ui.log("Conjecture: open (no certified lemma states it or its negation).")
    return 0
