"""The verification steps: one call per verifier agent."""

from __future__ import annotations

from typing import Dict, Tuple

from ..core.dag import load_file
from ..validity import verdicts
from .json_repair import parse_json_or_none
from .mode import check_mode_toggle
from .prompts import verifier_tools
from .reasoning import extract, reason
from .schemas import VERIFIER_SCHEMA


def verifier_prompt(ctx, loaded: Dict[str, str], agent: str) -> str:
    """The system prompt of a verification step: from the prompts the
    caller loaded, or read now (coding_verifier.md, only a code-backed
    proof needs it)."""
    if agent not in loaded:
        loaded[agent] = load_file(ctx.paths.prompt_paths[agent])
    return loaded[agent]


def _run_verifier(
    ctx,
    role: str,
    system_prompt: str,
    user_prompt: str,
    verbose: bool,
) -> Tuple[str, str]:
    """One atomic verification step: a single call to one verifier agent.

    Returns (decision, justification). decision is "accept" or "reject"
    only when the verifier itself gave that verdict; it is "unavailable"
    when the server could not be reached at all, and "" when there is
    no verdict for any other reason — the model returned nothing, the reply
    could not be parsed, or the pass hit the context wall — and the
    justification then says which. The proof loop waits out "unavailable"
    and retries the step (an outage is not the proof's failure), and
    treats "" like a reject
    (a check that cannot be completed can never count as an acceptance, so
    the proof goes to the reviser rather than into the DAG on the strength
    of the other two verifiers); verify and repair do not, because a
    refutation file is the record of a verifier's rejection, and a missing
    verdict is not one.
    """
    review, review_status, _partial = reason(
        ctx,
        system_prompt, user_prompt, role, verbose,
        tools=verifier_tools(ctx, role),
    )
    check_mode_toggle(ctx, verbose)
    unread = verdicts.unread_verdict(role, review_status, review)
    if unread is not None:
        return unread
    # The verifier prompts demand raw JSON output; extraction is the fallback.
    res = parse_json_or_none(review)
    if res is None:
        res = extract(
            ctx,
            "Extract the verdict. decision is 'accept' only if the review "
            "endorses the proof without unresolved objections.",
            review,
            VERIFIER_SCHEMA,
            role,
            verbose,
        )
    return verdicts.read_verdict(role, res)
