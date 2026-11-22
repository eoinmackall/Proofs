"""Evidence from citation rechecks. The proof loop checkpoints the decisions;
only the commit (or crash recovery) turns them into durable records."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Collection, Iterable

from . import certificates, refutations

if TYPE_CHECKING:
    from . import streams
    from .dag import Dag

# One selected recheck as the checkpoint holds it (checkpoint.py lists the
# fields): lemma_id, hash, file, check_id, verdicts, objections, and once
# finished outcome, model and (for a rejection) justification.
Check = dict[str, Any]


class CitationCommitError(Exception):
    """A worker must reload the disk records before retrying this batch."""


@dataclass
class Evidence:
    certificates: list[dict[str, Any]] = field(default_factory=list)
    refutations: list[tuple[Path, dict[str, Any]]] = field(default_factory=list)
    dismissed: list[Path] = field(default_factory=list)


def write_checks(ctx, dag: Dag, checks: Iterable[Check],
                 cert_writer: streams.Writer, *,
                 existing: Collection[str] = ()) -> Evidence:
    """Under the DAG lock. `existing` identifies records already on disk
    after a crash. No mutation of the DAG snapshot. A late accept never
    dismisses a refutation that was not actually put to these verifiers."""
    evidence = Evidence()
    for check in checks:
        if check.get('published') or not check.get('outcome'):
            continue
        lid, h, key = check['lemma_id'], check['hash'], check['check_id']
        if check['outcome'] == 'rejected':
            if key not in existing:
                path = refutations.record(
                    dag.root, lid, h, ctx.user, check['model'], 'reject',
                    check['justification'], check['file'], check_id=key,
                )
                evidence.refutations.append((path, refutations.load(path)))
                ctx.ui.log(f'Citation recheck rejected {lid}: {path.name}')
        elif check['outcome'] == 'accepted':
            if key not in existing:
                evidence.certificates.append(certificates.record(
                    cert_writer, lid, h, ctx.user, check['model'], check_id=key,
                ))
            shown = check.get('objections')
            if shown and dag.hashes.get(lid) == h:
                for path, ref in dag.refutations_for(lid):
                    # Match the contents too: a file name could be reused.
                    if shown.get(path.name) == ref:
                        try:
                            refutations.delete(path)
                        except OSError as exc:
                            ctx.ui.log(f'Could not dismiss {path}: {exc}')
                        else:
                            evidence.dismissed.append(path)
    return evidence
