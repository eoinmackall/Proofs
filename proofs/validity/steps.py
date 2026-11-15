"""What verification consists of: the verifier agents a proof must pass."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from ..tools.run_code import computations

# The verification steps: a proof enters the DAG only after every one of
# these verifier agents has accepted it. Each is a separate agent — its own
# prompt file and role, run as a single atomic check — rather than one
# verifier run repeatedly, so each step can grow its own focus, and the
# prompts are already specialized: verifier_1 checks logic and computations,
# verifier_2 specializes in logic (definitions, cited results, the
# conclusion), and verifier_3 in computations and completeness. One reject
# ends the counting and sends the proof to the reviser.
VERIFIER_AGENTS = ("verifier_1.md", "verifier_2.md", "verifier_3.md")

# The fourth check, for a proof that rests on a computation.
CODING_VERIFIER = "coding_verifier.md"


def verifier_steps(node: Dict[str, Any]) -> Tuple[str, ...]:
    """The verification steps a proof goes through: the three verifier
    checks, and coding_verifier.md as a fourth when the proof rests on a
    computation."""
    if computations.entries(node):
        return (*VERIFIER_AGENTS, CODING_VERIFIER)
    return VERIFIER_AGENTS


def role_of(agent: str) -> str:
    """A step's role, its prompt file's name: "verifier_1.md" -> "verifier_1"."""
    return agent[:-3]
