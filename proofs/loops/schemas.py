"""Response schemas (llama.cpp builds a grammar from these, so required keys
and enum values are guaranteed rather than hoped for).
"""

from __future__ import annotations

from typing import Any, Dict

# No dependencies field: the planner proposes statements, and which proved
# lemmas a proof leans on is settled by the prover while it writes the proof.
# No id either: the code gives every candidate its id (core.dag.new_lemma_id).
_LEMMA_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "statement": {"type": "string"},
        # The candidate attempts the conjecture (or its negation); its
        # statement is then the code's (resolution.pin).
        "settles": {"type": "string", "enum": ["conjecture", "conjecture_negation"]},
        # Parallel runs with --tools: the lemma needs a computation, so only
        # a loop that may run code takes it (LemmaBuffer).
        "computation": {"type": "boolean"},
    },
    "required": ["statement"],
}


# minItems/maxItems are deliberately absent: llama.cpp's schema support does
# not cover array cardinality, so writing them here would look like an
# enforced guarantee while enforcing nothing. The count is requested in
# planner.md and trimmed in planner_candidates().
#
# No flag says the conjecture is settled: that is computed from the DAG
# (resolution.find), and the planner attempts the conjecture by proposing
# a candidate with a "settles" field (resolution.SETTLES).
PLANNER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "plan_summary": {"type": "string"},
        "candidate_lemmas": {"type": "array", "items": _LEMMA_SCHEMA},
    },
    "required": [
        "plan_summary",
        "candidate_lemmas",
    ],
}


# Parallel mode: the planner no longer proposes lemmas (the lemma generator
# does), so its schema drops candidate_lemmas and adds the ordered priorities
# the generator works from. The generator reuses _LEMMA_SCHEMA for its batch.
PARALLEL_PLANNER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "plan_summary": {"type": "string"},
        "priorities": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "plan_summary",
    ],
}


PARALLEL_LEMMA_GENERATOR_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "candidate_lemmas": {"type": "array", "items": _LEMMA_SCHEMA},
    },
    "required": ["candidate_lemmas"],
}


VERIFIER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["accept", "reject"]},
        "justification": {"type": "string"},
    },
    "required": ["decision", "justification"],
}


# new_statement is a string, not nullable: the reviser writes the
# empty string when a field does not apply to the chosen action, which a
# grammar can enforce but a ["string", "null"] union is fiddlier to ask one
# for. action is an enum so a grammar can force one of the three choices.
REVISER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        # Which of the three revisions the reviser has chosen.
        #   keep              -> the statement is kept; the prover re-tries it
        #                        with the verdict as feedback.
        #   revise_statement  -> the statement is corrected (same id).
        #   new_lemma         -> the target is too hard as stated; a smaller
        #                        lemma (new_statement, under a new id the
        #                        code draws) is tried instead.
        "action": {
            "type": "string",
            "enum": ["keep", "revise_statement", "new_lemma"],
        },
        "diagnosis": {"type": "string"},
        "new_statement": {"type": "string"},
    },
    "required": ["action", "diagnosis", "new_statement"],
}
