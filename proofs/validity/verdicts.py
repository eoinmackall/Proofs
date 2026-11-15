"""Verdicts: how a verifier's review is read, what each verdict records,
and when a computation's re-execution settles a lemma.

A verdict is a (decision, justification) pair. decision is ACCEPT or REJECT
only when the verifier itself gave that verdict; UNAVAILABLE when the
server could not be reached at all; NO_VERDICT ("") for any other failure
— the model returned nothing, the reply could not be read, the pass hit
the context wall — with the justification saying which.

What a verdict produces (record_kind): an acceptance a certificate, a
rejection a refutation, anything else nothing at all. A missing verdict is
not a rejection: written as a refutation it would suspend the lemma, and
everything above it, for a server outage. In the proof loop the same rule
reads the other way — only an acceptance counts, so a check that cannot be
completed sends the proof to the reviser like a rejection.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

ACCEPT = "accept"
REJECT = "reject"
UNAVAILABLE = "unavailable"
NO_VERDICT = ""

CERTIFICATE = "certificate"
REFUTATION = "refutation"
NOTHING = "nothing"

Verdict = Tuple[str, str]


def unread_verdict(role: str, status: str, review: str) -> Optional[Verdict]:
    """The verdict of a review that cannot be read at all, from the call's
    status (loops.reasoning.reason): None when there is a review to read."""
    if status == "unavailable":
        return (
            UNAVAILABLE,
            f"{role} could not reach the server (transport errors on every "
            f"attempt).",
        )
    if status == "ceiling" and not review:
        return (
            NO_VERDICT,
            f"{role} exhausted its token budget without reaching a verdict; "
            "the proof is likely too long to review in one pass.",
        )
    if status == "too_large":
        return (
            NO_VERDICT,
            f"{role}'s prompt does not fit the context window, so it was "
            "never sent: the proof and the results it cites are too long "
            "to review in one pass.",
        )
    if not review:
        return (
            NO_VERDICT,
            f"{role} returned no review (the server failed or replied "
            f"with nothing, after retries).",
        )
    return None


def read_verdict(role: str, parsed: Optional[Mapping[str, Any]]) -> Verdict:
    """The verdict a review's parsed JSON ({"decision", "justification"})
    states, or NO_VERDICT when it states none."""
    res = parsed or {}
    decision = str(res.get("decision", "")).strip().lower()
    if decision not in (ACCEPT, REJECT):
        return (
            NO_VERDICT,
            f"{role}'s review could not be read as a verdict "
            f"(decision {res.get('decision')!r}).",
        )
    justification = str(res.get("justification") or "(no justification)").strip()
    return decision, justification


def record_kind(decision: str) -> str:
    """What a verdict on a stored lemma writes: CERTIFICATE, REFUTATION, or
    NOTHING (see the module docstring)."""
    if decision == ACCEPT:
        return CERTIFICATE
    if decision == REJECT:
        return REFUTATION
    return NOTHING


# ----------------------------------------------------------------------------
# Re-execution
# ----------------------------------------------------------------------------
# A refutation written by a re-execution (proofs verify --run-code) records
# a script that did not reproduce what the lemma says it gave: a fact no
# reading can answer. Its model is "exec:<tool version>".
EXECUTION_MODEL_PREFIX = "exec:"


def execution_model(tool: str) -> str:
    """The model a re-execution's refutation records."""
    return f"{EXECUTION_MODEL_PREFIX}{tool}"


def is_execution_refutation(ref: Mapping[str, Any]) -> bool:
    return str(ref.get("model") or "").startswith(EXECUTION_MODEL_PREFIX)


def unanswerable_without_reexecution(
    lemma_id: str, refutation_name: Optional[str], run_code: bool,
) -> Optional[str]:
    """Why a verification cannot settle a lemma that has a counting
    re-execution refutation (refutation_name) when it will not re-run the
    scripts, or None. Only a re-execution that reproduces them answers it."""
    if refutation_name is None or run_code:
        return None
    return (
        f"{lemma_id} has a refutation from a re-execution of its scripts "
        f"({refutation_name}); only a re-execution that reproduces "
        f"them can answer it. Verify with --tools and --run-code."
    )


def unsettled_by_reexecution(
    lemma_id: str, refutation_name: Optional[str], outcome: str, message: str,
) -> Optional[str]:
    """Why a re-execution leaves a counting re-execution refutation
    unanswered — it was inconclusive, neither reproducing the scripts nor
    contradicting them — or None."""
    if refutation_name is None or outcome in ("match", "mismatch"):
        return None
    return (
        f"{lemma_id} has a refutation from a re-execution of its "
        f"scripts, and this re-execution was inconclusive ({message}); "
        f"nothing is written."
    )

