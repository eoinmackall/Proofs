"""Multi-agent proof loop: planner -> selector -> prover -> verifier -> reviser -> DAG.

Design note on thinking vs. structured output
---------------------------------------------
llama.cpp's `json_schema` response_format is *ignored* while reasoning is on,
so the model is free to wrap its JSON in a markdown fence, and the
server's own JSON parser then 500s (ggml-org/llama.cpp#20345). The fix has
two halves:

  * `llm_backend.py` neutralizes the conflict per call: thinking is turned
    off for exactly the calls that carry a schema (the extraction stage),
    and left on everywhere else.
  * `extract()` additionally degrades to prompt-only with a JSON repair
    pipeline in `clean_json_text()`, in case the model still wraps the
    output in a fence or emits stray prose.

The reasoning stage stays schema-free, because that is exactly where the
model needs to think.

All seven agent prompts (planner.md, selector.md, prover.md, verifier_1.md,
verifier_2.md, verifier_3.md and reviser.md) instruct the model to emit
strict JSON directly, so each
response is first
parsed as-is; the extraction stage only runs as a fallback when the model
fails to comply. The prover never round-trips through a second model call at
all: its JSON is parsed locally and, failing that, the reasoning content is
taken verbatim, so the proof text can't be abridged or paraphrased.

Cancelling and resuming
-----------------------
The run keeps a checkpoint, `dag.checkpoint.json`, beside the DAG it is
building (so a `--dag` run checkpoints beside its own file). It holds what
`dag.json` does not: the position of the iteration budget, the planner's
reject list (`failed_attempts`), and — when the run is cancelled in the
middle of a proof — the lemma that was in flight, with the target statement
(the reviser may have revised it), the prover round it was on, the
feedback that round was about to be given, and the last proof of it (the one
the verifiers just rejected — the material the reviser judges its difficulty
by when it next sees this lemma). Checkpoints are written at safe
boundaries — the top of each iteration, the top of every prover round, and
the end of each iteration — and by the SIGINT handler itself on Ctrl-C,
which writes the last safe boundary's state and exits 130; a second Ctrl-C
force-exits. Re-running the same command resumes from the checkpoint:
the budget continues where it stopped, the planner keeps seeing the same
reject notes, and a lemma cancelled mid-proof goes straight back to the
prover for the interrupted round rather than through planning again (which
is what makes a mid-proof Ctrl-C cost the interrupted call, not the whole
iteration). An interrupted call is never restored — that one is re-run —
everything completed before it is not. When the planner settles the
conjecture the checkpoint is deleted; `--fresh` deletes it too, without
touching `dag.json`.

Planning is a shortlist, not a decision
---------------------------------------
The planner proposes PLANNER_CANDIDATES (5) lemmas per iteration rather than
one, ordered best-first, and something downstream picks exactly one. In
--mode auto that picker is the selector agent: screen_candidates() filters
the shortlist against the DAG, and when more than one candidate survives,
selector.md is asked to weigh them all and name the one it judges most
likely to come through the prover and the verifiers. The planner's best-first
order is the fallback, not the default — it is what gets taken when the
selector fails, names a candidate it was not offered, or when only one
candidate survives, where there is nothing left to choose. A candidate whose
id is already in the DAG is dropped from the shortlist — the rest still go to
selection — rather than stalling the whole iteration.

Proving is a loop, not a single pass
------------------------------------
A lemma enters the DAG only after all three verifier agents have accepted the
same proof. Verification is three atomic steps rather than one: verifier_1,
verifier_2 and verifier_3 are separate agents — separate prompt files and
separate roles — each a single call, so each step can grow its own focus:
the prompts are already specialized — verifier_1 checks logic and
computations, verifier_2 specializes in logic (definitions, cited results,
the conclusion), and verifier_3 in computations and completeness. Any single
reject ends the counting and sends the proof to the reviser with the
verifier's reasoning about the failure, along with the rejected proof itself
(the last proof), the lemmas already proved, and the known references. The
reviser decides where the fault lies: a fault in the
argument keeps the statement and sends the prover back with the verdict as
feedback; a fault in the statement comes back as a revised statement, which
becomes the prover's new target; and a lemma judged too hard to prove as
stated — from the shape of its last proof — is decomposed into a smaller
lemma (a fresh id, a sub-statement the last proof left unjustified), which
the prover then starts on with a fresh budget. The loop allows
MAX_PROOF_ATTEMPTS prover rounds per lemma per iteration
(--max-proof-attempts), then gives the lemma up and asks the planner again.

When a call runs into the context wall
--------------------------------------
The server is stateless, so a truncated exchange — prompt, thinking trace,
answer-so-far, summing to about num_ctx — cannot be re-sent whole. reason()
therefore gets pi-style compaction rescues before reporting "ceiling":
the thinking trace (scratch, regenerable) is summarised into a structured
summary — Goal, Progress, Key Decisions, Next Steps, Critical Context, plus
a <cited-lemmas> tracking block — in chunked passes if it is big; the task
prompt and the answer written so far are kept verbatim, the answer ending
exactly where it was cut; and the model is asked to resume from the cut with
the whole headroom of the now-smaller prompt. A continuation that hits the
wall itself is not a failure but the next pass: its trace folds into the
summary, its text extends the verbatim answer, and the model resumes from
the new cut. The rescue is bounded to MAX_COMPACTION_PASSES passes (two:
the first usually finishes an answer that merely ran long, the second
covers the genuinely long proofs). Still unfinished after that many — or a
pass that leaves nothing to build on, or no headroom left — degrades to the
"ceiling" with the partial work ({"summary", "thinking", "answer"}) for the
caller to hand on: the proof loop compacts it a final time via
_incomplete_report() and sends the summary plus the partial answer to the
reviser (the overflow case of reviser.md) to pick a smaller lemma or revise
the statement; only when there is nothing to build on, or the reviser has
nothing usable, does the planner decompose.

In --mode human the picker is a person, and picking is two decisions rather
than one. A number and Enter accepts that candidate as true on your
authority: it goes straight into the DAG, with no prover call and no verifier
calls, marked "provenance": "operator". A trailing 'p' — "3p" — sends it down
the ordinary pipeline instead, for when you are confident it is the right next
step but not that it is true. You can also write your own lemma ('w', or 'wp'
to have it proved), or send the planner back to think again.

You can cross between them mid-run, in either direction, without restarting.
'h' toggles: press it during a planner, prover, verifier or reviser call and
the mode flips at the next planning step, which is the earliest a change of
mode can mean anything. 'a' at the menu does the same thing for the case where the
menu is already up and the listener is therefore paused. See interaction.py
for why one of these is a keystroke and the other is a line of input.

Usage
-----
    python main.py --conjecture algebra_example
    python main.py --conjecture algebra_example --model Qwen3.5-122B-Q4_K_M
    python main.py --conjecture algebra_example --model qwen3.8-27b
    python main.py --conjecture algebra_example --no-verbose --max-iterations 25
    python main.py --conjecture algebra_example --mode human
    python parsing.py --conjecture algebra_example

parsing.py is standalone: it reads conjectures/<name>/references/ (any .tex
or .md file in it), asks the model for the theorem-level results, and writes
them as a strict-JSON array to conjectures/<name>/references.md. main.py then
offers those references to the prover (citable by id), to the verifiers (the
formal statements of the cited ones only) and to the planner and the reviser
(an id + slogan shortlist, only while the collection is small enough to be
one).

A conjecture may also carry a comments.md beside its conjecture.md: free-form
operator notes on possible approaches to a proof or a counterexample. The
loop reads it once and hands it to the planner verbatim on every iteration as
a "Human comments" section — the planner is told to treat it as suggestions
to weigh (a viable suggested route should get one of the five candidates),
not as instructions. No other agent sees the file, and a directory without
one runs exactly as before.

--conjecture names a directory under conjectures/ holding a conjecture.md.
The DAG is written beside it as dag.json and shared by every model: point a
second model at a conjecture already under way and it continues from the
lemmas the first one proved. Pass --dag to give a run its own file instead.
"""

import argparse
import json
import os
import re
import signal
import sys
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

import interaction
import llm_backend
import workspace

# ----------------------------------------------------------------------------
# Configuration (all overridable from the command line; see main())
# ----------------------------------------------------------------------------
# qwen3.8-27b — llama-server reached over the SSH tunnel on 8081
# (the only live inference server on this machine).
MODEL_NAME = "qwen3.8-27b"
DEFAULT_CONJECTURE = "algebra_example"

# Set in main(). BACKEND owns the HTTP conversation; PROFILE is what the
# capability probe found (context ceiling, thinking support, tokenizer ratio).
BACKEND: Any = None
PROFILE: Any = None

# Resolved by workspace.resolve(): a prompt may be overridden per conjecture,
# so these are paths rather than the bare filenames they used to be.
CONJECTURE_FILE = ""
DAG_FILE = ""
# Set in main(), beside DAG_FILE: dag.json -> dag.checkpoint.json. See the
# "Cancelling and resuming" section of the module docstring.
CHECKPOINT_FILE = ""
REFERENCES_FILE = ""
# Set in main(), beside REFERENCES_FILE: the optional operator notes the
# planner sees as "Human comments" (see the module docstring).
COMMENTS_FILE = ""
PROMPT_PATHS: Dict[str, str] = {name: name for name in workspace.PROMPT_FILES}

MAX_ITERATIONS = 10
LLM_MAX_RETRIES = 3
REQUEST_TIMEOUT = 1800       # Thinking models are slow; give them room

# How many lemmas the planner shortlists per iteration. Raising this costs
# planner tokens and, in human mode, attention; five is about as many
# candidates as can be compared without re-reading the conjecture.
PLANNER_CANDIDATES = 5

# How many parsed references (references.md) the planner and the reviser are
# shown at all. The rule is deliberately blunt: below the limit each of them
# gets an id + slogan per result, so they can build on a named theorem rather
# than re-prove it from scratch; at or above it, none. A shortlist of hundreds
# of slogans costs more in tokens and attention than it returns, and the
# prover sees the full collection regardless, so a named result the planner
# misses is one proving round away, not lost. A strategy for large
# collections is deliberately not built yet.
PLANNER_REFERENCE_LIMIT = 100

# The verification steps: a proof enters the DAG only after every one of
# these verifier agents has accepted it. Each is a separate agent — its own
# prompt file and role, run as a single atomic check — rather than one
# verifier run repeatedly, so each step can grow its own focus, and the
# prompts are already specialized: verifier_1 checks logic and computations,
# verifier_2 specializes in logic (definitions, cited results, the
# conclusion), and verifier_3 in computations and completeness. One reject
# ends the counting and sends the proof to the reviser.
VERIFIER_AGENTS = ("verifier_1.md", "verifier_2.md", "verifier_3.md")
# How many prover rounds one iteration may spend on a lemma before giving it
# up and re-planning.
MAX_PROOF_ATTEMPTS = 3

# "auto" reproduces the original behaviour end to end. "human" stops at every
# planning step. Set from --mode, then mutated by the hotkey and the menu, so
# it is genuinely a run-time toggle rather than a launch-time one.
DEFAULT_MODE = "auto"
MODE = DEFAULT_MODE
HOTKEY_KEYS = "h"            # single keystroke; see interaction.HotKey
HOTKEY: interaction.HotKey = interaction.NullHotKey()

# Default context window. main() reassigns it after probing: with no
# --num-ctx the server's own -c is used as-is (the KV cache is already
# allocated at launch, so there is nothing to request); an explicit
# --num-ctx always wins — clamped only when the ceiling is a *probed*
# llama-server -c (a fact), since on a non-llama.cpp endpoint the value on
# record is a default, not a measurement, and clamping an explicit request
# to a guess would defeat it. This constant stands in when the probe fails.
# 65536 matches the Qwen3.8-27B reference launch line in llm_backend.py
# (-c 65536).
NUM_CTX = 65536
# No static per-role generation budgets. reason() and extract() give every
# call the full headroom — num_ctx minus the prompt, minus a safety margin —
# because num_predict is a ceiling, not a target: the model stops at EOS
# whatever the number, so a tight budget can only ever end in a truncated
# generation paid for in full, followed by a retry with a bigger number. The
# response is allowed to use the whole window, which means a truncation now
# reports "ceiling" — the lemma is too big for the context — instead of
# quietly undersizing the answer.

# Thinking level: True, or "low"/"medium"/"high"/"max" on models that support
# levels. Set to None to omit the field entirely (the model's default).
# Never set this to False — see the module docstring.
#
# These are requests, not guarantees. The backend checks them against the
# probed capabilities and drops or downgrades: asking a non-thinking model to
# think is a 400 from the server, and a level string sent to a model that
# only understands booleans is ignored silently, which is worse.
THINK = {
    "planner": True,
    "selector": True,
    "prover": True,
    "verifier_1": True,
    "verifier_2": True,
    "verifier_3": True,
    "reviser": True,
    "parser": True,   # parsing.py: reading mathematics, then extracting it
    "parallel_planner": True,
    "parallel_lemma_generator": True,
}

# Sampling. The presence penalty is the delicate one: it pushes the model off
# tokens it has already used, which suppresses repetition loops in agentic
# coding but is actively harmful here — mathematics *requires* hammering the
# same symbols (\epsilon, n, x_i) over and over, and the extraction stage's
# whole job is verbatim copying.
#
# The right value is model-specific, but there is no per-model baseline in
# this codebase: a model runs on its own card's default, and whatever is set
# in these option dicts wins. Keep the dicts minimal for that reason.
REASONING_OPTIONS: Dict[str, Any] = {
    "temperature": 0.7,          # Greedy decoding degrades thinking models.
    "num_ctx": NUM_CTX,
    # presence_penalty: deliberately absent — the server's default applies.
    # num_predict is set per call by reason(); see the NUM_CTX comment.
}

EXTRACT_OPTIONS: Dict[str, Any] = {
    "temperature": 0.0,          # Mechanical, and grammar-constrained anyway.
    "presence_penalty": 0.0,     # Must be 0 on every model: this stage copies,
                                 # it doesn't write. Stated explicitly so it
                                 # overrides whatever the model's default is.
    "min_p": 0.0,
    "num_ctx": NUM_CTX,
    # num_predict is set per call by extract(); see the NUM_CTX comment.
}

# Per-role reasoning temperature overrides.
TEMPERATURES = {
    "planner": 0.7,
    "selector": 0.5,   # Comparing a shortlist, not generating one.
    "prover": 0.6,     # Slightly tighter: rigour over exploration.
    "verifier_1": 0.8,
    "verifier_2": 0.8,
    "verifier_3": 0.8,   # Looser, so the three independent checks don't
                        # collapse into one review of the first pass.
    "reviser": 0.7,    # Diagnosing a failure is planning-scale judgement.
    "parser": 0.3,     # Extracting from a fixed source, not exploring.
    "parallel_planner": 0.3,          # Steering a run, not exploring one.
    "parallel_lemma_generator": 0.6,  # Proposing lemmas is generation.
}

# Pi-style compaction on the context wall: a truncated call is resumed by
# summarising its thinking trace and re-sending task + answer-so-far (see
# _resume_compacted, between _headroom and reason below). A truncated
# continuation is not a failure but the next pass; the passes run until the
# answer completes, nothing is left to build on, or MAX_COMPACTION_PASSES is
# spent. COMPACT_CHUNK_TOKENS bounds one compaction pass's input,
# COMPACT_TAIL_CHARS is what the compactor sees of the answer-so-far,
# COMPACT_MIN_ROOM is the smallest headroom a continuation is worth attempting.
#
# COMPACT_THRESHOLD is the fraction of the window a call is allowed to fill
# (prompt + thinking + answer) before it is cut off and compacted. It is the
# single knob for "how long a role may think before compaction": the generation
# budget is set so that prompt + budget lands on COMPACT_THRESHOLD * num_ctx,
# leaving (1 - COMPACT_THRESHOLD) of the window as a safety margin rather than
# a fixed 512 tokens. 0.99 leaves a ~1% margin; note the prompt itself eats
# into the window, so the generation budget is really
# (COMPACT_THRESHOLD * num_ctx) - prompt — a 30k prompt on a 90k window caps
# generation near 60k regardless of this value. Raise toward 1.0 for maximum
# thinking room; lower it only if a run reports the server tripping the wall.
COMPACT_ENABLED = True
COMPACT_THRESHOLD = 0.99
COMPACT_CHUNK_TOKENS = 24000
COMPACT_TAIL_CHARS = 2000
COMPACT_MIN_ROOM = 2048

# How many compaction-and-resume passes a single prover call may get before
# the partial work is handed to the reviser. Two: the first pass usually
# finishes an answer that merely ran long, the second covers the genuinely
# long proofs. Still unfinished after the second, the lemma is too large for
# one call — the trace and partial answer are compacted a final time and the
# reviser picks a smaller lemma from them (or revises the statement),
# instead of the prover grinding on
# an unbounded number of passes (see _resume_compacted and the proof loop).
MAX_COMPACTION_PASSES = 2

# Set at runtime if schema-constrained extraction proves incompatible with the
# model's default thinking (empty content, output stranded in .thinking).
_SCHEMA_MODE_BROKEN = False


def log(message: str, verbose: bool = True) -> None:
    """Single logging chokepoint, so --no-verbose silences everything."""
    if verbose:
        print(message)


# ----------------------------------------------------------------------------
# Response schemas (llama.cpp builds a grammar from these, so required keys
# and enum values are guaranteed rather than hoped for).
# ----------------------------------------------------------------------------
# No dependencies field: the planner proposes statements, and which proved
# lemmas a proof leans on is settled by the prover while it writes the proof.
_LEMMA_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "statement": {"type": "string"},
    },
    "required": ["id", "statement"],
}

# minItems/maxItems are deliberately absent: llama.cpp's schema support does
# not cover array cardinality, so writing them here would look like an
# enforced guarantee while enforcing nothing. The count is requested in
# planner.md and trimmed in planner_candidates().
PLANNER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "is_conjecture_proved": {"type": "boolean"},
        "is_conjecture_disproved": {"type": "boolean"},
        "plan_summary": {"type": "string"},
        "candidate_lemmas": {"type": "array", "items": _LEMMA_SCHEMA},
    },
    "required": [
        "is_conjecture_proved",
        "is_conjecture_disproved",
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
        "is_conjecture_proved": {"type": "boolean"},
        "is_conjecture_disproved": {"type": "boolean"},
        "plan_summary": {"type": "string"},
        "priorities": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "is_conjecture_proved",
        "is_conjecture_disproved",
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

# The selector picks by id, but "one of these particular ids" is not
# something a grammar can enforce, so the offered ids are listed in the
# prompt and the answer is checked against them in select_lemma(), which
# falls back to the planner's first candidate when it does not check out.
SELECTOR_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "selected_id": {"type": "string"},
    },
    "required": ["selected_id"],
}

VERIFIER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["accept", "reject"]},
        "justification": {"type": "string"},
    },
    "required": ["decision", "justification"],
}

# new_statement and new_id are strings, not nullable: the reviser writes the
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
        #                        lemma (new_id + new_statement) is tried instead.
        "action": {
            "type": "string",
            "enum": ["keep", "revise_statement", "new_lemma"],
        },
        "diagnosis": {"type": "string"},
        "new_id": {"type": "string"},
        "new_statement": {"type": "string"},
    },
    "required": ["action", "diagnosis", "new_id", "new_statement"],
}


# ----------------------------------------------------------------------------
# File helpers
# ----------------------------------------------------------------------------
def load_file(filepath: Any) -> str:
    """Read prompt or markdown file contents. Accepts str or Path."""
    if not filepath or not os.path.exists(filepath):
        return ""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_dag() -> Dict[str, Any]:
    """Load current DAG state from file. Absent file => empty DAG."""
    if os.path.exists(DAG_FILE):
        with open(DAG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"lemmas": {}}


def save_dag(dag: Dict[str, Any]) -> None:
    """Save updated DAG state to file, creating it if needed.

    Whole-file write: temp file beside the DAG, then os.replace, the
    checkpoint's way. The parallel loops read the DAG from several threads
    at once, and a plain open("w") lets a reader catch the file mid-write
    (an empty or half-formed file) and crash the loop on the parse."""
    tmp = DAG_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(dag, f, indent=2)
    os.replace(tmp, DAG_FILE)


# ----------------------------------------------------------------------------
# Checkpointing: the state a cancelled run leaves behind, so the next run
# resumes instead of starting over. The DAG itself is the permanent record;
# this is the run-level state that would otherwise be paid for again: where
# the iteration budget stood, the planner's reject list, and the lemma that
# was mid-proof. See "Cancelling and resuming" in the module docstring.
# ----------------------------------------------------------------------------
# The latest safe-boundary state, updated by run_loop() as it goes. The
# SIGINT handler reads it so a Ctrl-C between boundaries still writes what
# the last boundary established, without reaching into run_loop's locals.
_LIVE_STATE: Dict[str, Any] = {
    "iteration": None,
    "failed_attempts": None,
    "in_flight": None,
}
_sigint_count = 0
# The live ParallelState, or None outside a parallel run. The SIGINT handler
# checks it first so a Ctrl-C in parallel mode snapshots the shared state
# (version-2 checkpoint) instead of the serial _LIVE_STATE, which a parallel
# run never updates.
_PARALLEL: "ParallelState" = None
# The DAG file lock, held across load-then-save by every DAG writer that
# can run concurrently with another: the serial success boundary and each
# parallel loop's. Two writers can't add the same id twice or clobber
# each other's addition. Readers reload under it too (fresh per round), so
# a loop never plans or proves against a stale DAG.
DAG_LOCK = threading.Lock()
# The checkpoint-file lock, held by ParallelState.write_checkpoint() across
# the whole snapshot-then-replace. In parallel mode every loop's claim
# (begin_lemma), round boundary (set_in_flight), DAG commit (add_lemma) and
# reject note (record_failure) rewrites the checkpoint, as does the SIGINT
# handler's snapshot, so the writers are serialised as a unit: the slots
# and the reject list snapshotted are the ones that land in the file, and
# an older snapshot that snapshots first but replaces last can never land
# over a newer one. (The file stays whole even without the lock — the temp
# file is per writer and the swap is os.replace — so the lock is about
# which snapshot wins, not about readers seeing a half-write.)
CHECKPOINT_LOCK = threading.Lock()


def checkpoint_path_for(dag_path: str) -> str:
    """The checkpoint file for a DAG file: dag.json -> dag.checkpoint.json.

    Beside the DAG, not beside the conjecture directory's name, so a run
    pointed at its own --dag file checkpoints beside exactly that file.
    """
    root, ext = os.path.splitext(dag_path)
    return f"{root}.checkpoint{ext or '.json'}"


def save_checkpoint(
    iteration: int,
    failed_attempts: Dict[str, List[str]],
    in_flight: Optional[Dict[str, Any]],
    verbose: bool = True,
) -> None:
    """Write the checkpoint atomically, beside the DAG.

    `iteration` is the next iteration a resuming run should start at, and
    `in_flight` (None unless a prover round is the resumable position) names
    the lemma to go straight back to the prover with: its target as the
    reviser last left it, the prover round to re-run, the feedback and reject
    notes that round carries, and the last proof of it (the one just rejected
    — the material the reviser judges its difficulty by when it next sees this
    lemma).
    """
    if not CHECKPOINT_FILE:
        return
    data: Dict[str, Any] = {
        "version": 1,
        "iteration": int(iteration),
        "failed_attempts": {k: list(v) for k, v in failed_attempts.items()},
    }
    if in_flight is not None:
        data["in_flight"] = in_flight
    _atomic_write_checkpoint_file(data, verbose)


def load_checkpoint(verbose: bool = True) -> Optional[Dict[str, Any]]:
    """Read and validate the checkpoint.

    Returns {"iteration", "failed_attempts", "in_flight"}, or None when there
    is no usable checkpoint. A checkpoint that cannot be trusted is discarded
    rather than fatal: the DAG is the permanent record, and the checkpoint
    only saves re-paying for a cancelled run's completed work. Corrupt files
    are a warning, not a crash, for the same reason.
    """
    if not CHECKPOINT_FILE or not os.path.exists(CHECKPOINT_FILE):
        return None
    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log(f"⚠️  Checkpoint {CHECKPOINT_FILE} is unreadable ({e}); ignoring it.",
            verbose)
        discard_checkpoint()
        return None
    if not isinstance(data, dict):
        log(f"⚠️  Checkpoint {CHECKPOINT_FILE} is malformed; ignoring it.",
            verbose)
        discard_checkpoint()
        return None
    version = data.get("version")
    if version == 2:
        # A parallel-mode checkpoint is valid state for a parallel run; the
        # serial loader must not read it as (and discard it as) a corrupt
        # serial one. Re-running without --parallel is a different shape of
        # run, so the honest answer is "re-run with --parallel".
        log(f"⚠️  {os.path.basename(CHECKPOINT_FILE)} is a parallel-mode "
            f"checkpoint; re-run with --parallel to resume it.", verbose)
        return None
    if version not in (None, 1):
        log(f"⚠️  Checkpoint {CHECKPOINT_FILE} has an unknown version "
            f"({version!r}); ignoring it.", verbose)
        discard_checkpoint()
        return None
    iteration = data.get("iteration")
    if not isinstance(iteration, int) or iteration < 1:
        log(f"⚠️  Checkpoint {CHECKPOINT_FILE} has no usable iteration; ignoring it.",
            verbose)
        discard_checkpoint()
        return None
    failed_raw = data.get("failed_attempts")
    failed_attempts: Dict[str, List[str]] = {}
    if isinstance(failed_raw, dict):
        failed_attempts = {
            k: [n for n in v if isinstance(n, str)]
            for k, v in failed_raw.items()
            if isinstance(v, list)
        }
    in_flight: Optional[Dict[str, Any]] = None
    raw = data.get("in_flight")
    if isinstance(raw, dict):
        lemma_id = str(raw.get("lemma_id") or "")
        target = raw.get("target")
        attempt = raw.get("attempt")
        statement = str(target.get("statement") or "").strip() if isinstance(target, dict) else ""
        if lemma_id and statement and isinstance(attempt, int) and 1 <= attempt <= MAX_PROOF_ATTEMPTS:
            clean_target: Dict[str, Any] = {"id": lemma_id, "statement": statement}
            in_flight = {
                "lemma_id": lemma_id,
                "target": clean_target,
                "attempt": attempt,
                "feedback": [n for n in (raw.get("feedback") or []) if isinstance(n, str)],
                "attempt_notes": [n for n in (raw.get("attempt_notes") or []) if isinstance(n, str)],
                "last_proof": str(raw.get("last_proof") or ""),
            }
        else:
            log(
                "⚠️  Checkpoint's in-flight state is unusable (bad lemma, "
                "statement, or prover round); planning as usual instead.",
                verbose,
            )
    return {
        "iteration": iteration,
        "failed_attempts": failed_attempts,
        "in_flight": in_flight,
    }


def _validate_in_flight(raw: Any) -> Optional[Dict[str, Any]]:
    """The in-flight state a resuming prover round is handed, or None.

    The shape the serial and parallel checkpoints share: the lemma id, the
    target as the reviser last left it, the prover round to re-run, the
    feedback and reject notes it carries, and the last proof of it (the one
    just rejected — what the reviser judges its difficulty by).
    """
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
        and 1 <= attempt <= MAX_PROOF_ATTEMPTS
    ):
        return None
    clean_target: Dict[str, Any] = {"id": lemma_id, "statement": statement}
    return {
        "lemma_id": lemma_id,
        "target": clean_target,
        "attempt": attempt,
        "feedback": [n for n in (raw.get("feedback") or []) if isinstance(n, str)],
        "attempt_notes": [
            n for n in (raw.get("attempt_notes") or []) if isinstance(n, str)
        ],
        "last_proof": str(raw.get("last_proof") or ""),
    }


def load_parallel_checkpoint(
    loop_ids: List[str], verbose: bool = True
) -> Optional[Dict[str, Any]]:
    """Read and validate the version-2 (parallel-mode) checkpoint.

    Returns {"loops": {loop_id: {"in_flight", "iterations_used"}},
    "failed_attempts"}, or None when there is nothing to resume. A
    checkpoint written by a run with a different set of loops is ignored
    rather than half-applied: per-loop slots are not transferable between
    runs of different shape. As with the serial loader, a file that is not
    ours to interpret is left alone unless it is corrupt.
    """
    if not CHECKPOINT_FILE or not os.path.exists(CHECKPOINT_FILE):
        return None
    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log(f"⚠️  Checkpoint {CHECKPOINT_FILE} is unreadable ({e}); ignoring it.",
            verbose)
        return None
    if not isinstance(data, dict) or data.get("version") != 2:
        log(f"⚠️  {os.path.basename(CHECKPOINT_FILE)} is not a parallel-mode "
            f"(v2) checkpoint; ignoring it.", verbose)
        return None
    raw_loops = data.get("loops")
    if not isinstance(raw_loops, dict) or set(raw_loops) != set(loop_ids):
        log(f"⚠️  {os.path.basename(CHECKPOINT_FILE)} was written by a run "
            f"with a different set of loops; ignoring it.", verbose)
        return None
    failed_raw = data.get("failed_attempts")
    if not isinstance(failed_raw, dict):
        log(f"⚠️  {os.path.basename(CHECKPOINT_FILE)} has no usable "
            f"failed-attempts section; ignoring it.", verbose)
        return None
    failed_attempts: Dict[str, List[str]] = {
        k: [n for n in v if isinstance(n, str)]
        for k, v in failed_raw.items()
        if isinstance(v, list)
    }
    loops: Dict[str, Dict[str, Any]] = {}
    for lid in loop_ids:
        raw = raw_loops.get(lid)
        if not isinstance(raw, dict):
            raw = {}
        in_flight = _validate_in_flight(raw.get("in_flight"))
        if in_flight is None and isinstance(raw, dict) and raw.get("in_flight") is not None:
            log(f"⚠️  {os.path.basename(CHECKPOINT_FILE)} has a malformed "
                f"in-flight record for {lid}; that loop starts fresh.", verbose)
        iterations_used = raw.get("iterations_used", 0)
        if not isinstance(iterations_used, int) or iterations_used < 0:
            iterations_used = 0
        loops[lid] = {"in_flight": in_flight, "iterations_used": iterations_used}
    return {"loops": loops, "failed_attempts": failed_attempts}


def _atomic_write_checkpoint_file(data: Dict[str, Any], verbose: bool) -> None:
    """The shared atomic write: temp file beside the checkpoint, then
    os.replace over it. Readers see either the whole old file or the whole
    new one, never a half-written mix.

    The temp file is named per writer (pid + thread id), not shared: two
    writers opening one .tmp truncate each other's in-progress dump, and
    the first replace to land would promote the intermixed bytes to the
    checkpoint file itself — while the other's replace then raises on the
    already-renamed name and is swallowed as a lost update. With a private
    tmp, os.replace alone keeps the file whole for readers even when two
    processes write the same checkpoint; CHECKPOINT_LOCK, held by the
    callers that snapshot shared state, is what stops an older snapshot
    from landing over a newer one."""
    if not CHECKPOINT_FILE:
        return
    tmp = f"{CHECKPOINT_FILE}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp, CHECKPOINT_FILE)
    except OSError as e:
        log(f"⚠️  Could not write checkpoint {CHECKPOINT_FILE}: {e}", verbose)
        try:
            os.remove(tmp)
        except OSError:
            pass


def discard_checkpoint() -> None:
    """Remove the checkpoint file if present.

    Silent on failure: a checkpoint is a convenience file, and its absence
    is always a valid state (a fresh run).
    """
    if CHECKPOINT_FILE and os.path.exists(CHECKPOINT_FILE):
        try:
            os.remove(CHECKPOINT_FILE)
        except OSError:
            pass


def _on_sigint(signum, frame) -> None:
    """Ctrl-C: write the checkpoint, then exit 130 like the shell expects.

    A second Ctrl-C force-exits without waiting on the file. The checkpoint
    is written from _LIVE_STATE — the last safe boundary run_loop() reached —
    so an interrupt mid-LLM-call loses only that call: the next run re-runs
    the interrupted prover round, not the whole iteration.
    """
    global _sigint_count
    _sigint_count += 1
    if _sigint_count > 1:
        print("\nForced exit.", file=sys.stderr)
        os._exit(130)
    if _PARALLEL is not None:
        # Parallel mode: the shared state is the checkpoint. Ask every thread
        # to stop at its next boundary, snapshot the whole run under the
        # state lock, and exit; the daemon threads die with the process and
        # the next run resumes from the snapshot.
        _PARALLEL.request_stop()
        print("\n⏹ Interrupted — writing checkpoint...", file=sys.stderr)
        _PARALLEL.write_checkpoint()
        print(
            f"   {os.path.basename(CHECKPOINT_FILE)} written. Re-run the same "
            f"command to resume where this run stopped.",
            file=sys.stderr,
        )
        sys.exit(130)
    print("\n⏹ Interrupted — writing checkpoint...", file=sys.stderr)
    state = _LIVE_STATE
    if (
        state["iteration"] is not None
        and state["failed_attempts"] is not None
        and CHECKPOINT_FILE
    ):
        save_checkpoint(
            state["iteration"],
            state["failed_attempts"],
            state["in_flight"],
            verbose=False,
        )
        print(
            f"   {os.path.basename(CHECKPOINT_FILE)} written. Re-run the same "
            f"command to resume where this run stopped.",
            file=sys.stderr,
        )
    else:
        print("   (no checkpoint state yet — nothing written)", file=sys.stderr)
    sys.exit(130)


# ----------------------------------------------------------------------------
# Server context 400s: a mis-sized window, not a transport hiccup
# ----------------------------------------------------------------------------
_CONTEXT_FAILURE_SEEN = False


def _context_length_message(
    e: "llm_backend.ContextLengthError", budget: int
) -> str:
    """What to tell the operator when the server 400s a call as too long.

    The limit in it is the server's own number, read from the body of its
    400 — a measurement, not this run's budget; the budget is what the run
    asked for. Naming both, and only the flag that fixes it, keeps the
    failure from looking like a model or a network problem."""
    if e.prompt_tokens is not None and e.prompt_tokens >= e.server_limit:
        return (
            f"\n⛔ The server rejected a call as too long: the prompt alone "
            f"({e.prompt_tokens} tokens) already exceeds its context window "
            f"({e.server_limit} tokens), and this run budgets {budget}. No "
            f"--num-ctx value fixes a prompt that no longer fits — the DAG "
            f"has grown past the model's window. Re-run against a server "
            f"launched with a larger --max-model-len, or start a fresh DAG "
            f"for this conjecture."
        )
    return (
        f"\n⛔ The server rejected a call as too long: its context window is "
        f"{e.server_limit} tokens (the server's own number, from its 400), "
        f"but this run budgets {budget}, so every call whose prompt plus "
        f"generation budget outruns the window is rejected outright — a "
        f"rejection the compaction rescue never sees, because it only "
        f"triggers on a truncated reply. Re-run with --num-ctx "
        f"{e.server_limit}, matched to the server's --max-model-len."
    )


def _fail_context_length(e: "llm_backend.ContextLengthError",
                         stop: Any = None) -> None:
    """Run-level handling of a server context 400: one loud message per
    run, then the stop the caller names (parallel mode stops the shared
    state; serial mode exits from main instead). The checkpoint is
    untouched — it holds the last safe boundary, which is exactly where a
    re-run with a corrected --num-ctx should resume from. Logged
    unconditionally: a fatal the operator must see is not verbose noise."""
    global _CONTEXT_FAILURE_SEEN
    if not _CONTEXT_FAILURE_SEEN:
        _CONTEXT_FAILURE_SEEN = True
        log(_context_length_message(e, NUM_CTX))
        log(
            "The checkpoint is safe; re-running resumes from the last "
            "safe boundary.",
        )
    if stop is not None:
        stop()


# ----------------------------------------------------------------------------
# JSON repair (fallback path; the schema grammar should make it unnecessary)
# ----------------------------------------------------------------------------
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_OPEN_THINK_RE = re.compile(r"<think>.*", re.DOTALL)
_HEX = set("0123456789abcdefABCDEF")


def _repair_escapes(text: str) -> str:
    """Fix backslash escapes inside JSON strings, without corrupting text
    that is already correctly escaped.

    A regex can't do this: in `\\\\subset` (a valid escaped backslash followed
    by 's'), a lookahead-based pattern matches the *second* backslash of the
    pair and doubles it, turning valid JSON into an invalid `\\s` escape.
    The scanner consumes `\\\\` pairs atomically so that can't happen.

    Heuristics inside strings:
      \\ + \\            -> valid pair, keep
      \\ + " or /        -> valid escape, keep
      \\u + 4 hex digits -> valid unicode escape, keep
      \\ + bfnrtu + [a-z]-> LaTeX command (\\frac, \\neq, \\to), double it
      \\ + bfnrtu        -> genuine control escape, keep
      \\ + anything else -> invalid (\\alpha, \\subset, \\{), double it
    """
    out: List[str] = []
    i, n = 0, len(text)
    in_str = False
    while i < n:
        ch = text[i]
        if not in_str:
            if ch == '"':
                in_str = True
            out.append(ch)
            i += 1
            continue
        if ch == '"':
            in_str = False
            out.append(ch)
            i += 1
            continue
        if ch != "\\":
            out.append(ch)
            i += 1
            continue

        nxt = text[i + 1] if i + 1 < n else ""
        if nxt == "\\":
            out.append("\\\\")          # valid pair — consume both, untouchable
            i += 2
        elif nxt in '"/':
            out.append("\\" + nxt)      # valid escape
            i += 2
        elif nxt == "u" and set(text[i + 2 : i + 6]) <= _HEX and len(text[i + 2 : i + 6]) == 4:
            out.append(text[i : i + 6])  # valid \uXXXX
            i += 6
        elif nxt in "bfnrtu":
            follow = text[i + 2 : i + 3]
            if follow.islower() and follow.isalpha():
                out.append("\\\\" + nxt)  # \frac, \neq, \to — LaTeX, not control
            else:
                out.append("\\" + nxt)    # genuine \n, \t, ...
            i += 2
        elif nxt == "":
            out.append("\\\\")          # trailing backslash at end of text
            i += 1
        else:
            out.append("\\\\" + nxt)    # \alpha, \subset, \{ — invalid, double
            i += 2
    return "".join(out)


def _extract_json_object(text: str) -> str:
    """First balanced {...} span, ignoring braces inside strings."""
    start = text.find("{")
    if start == -1:
        return text
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]  # unbalanced => truncated


def _salvage_truncated(text: str) -> str:
    """Close an unterminated string and any open braces, so a cut-off reply is
    still usable instead of lost."""
    in_str, esc, depth = False, False, 0
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
    repaired = text.rstrip()
    if in_str:
        repaired = repaired.rstrip("\\") + '"'
    return repaired + "}" * max(depth, 0)


def clean_json_text(raw: str) -> str:
    """Normalise model output into parseable JSON.

    Repairs escalate and each is tried only if the previous stage fails to
    parse — text that is already valid JSON is returned byte-for-byte
    untouched, so the repair heuristics can never corrupt a correct reply.
    """
    text = _THINK_RE.sub("", raw)
    if "<think>" in text:
        text = _OPEN_THINK_RE.sub("", text)
    text = _FENCE_RE.sub("", text).strip()
    text = _extract_json_object(text)

    candidates = [text]
    repaired = _repair_escapes(text)
    if repaired != text:
        candidates.append(repaired)
    candidates.append(_salvage_truncated(repaired))

    for candidate in candidates:
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue
    return candidates[-1]  # let the caller surface the real parse error


def parse_json_or_none(text: str) -> Optional[Dict[str, Any]]:
    """Parse an agent reply that already complies with its prompt file.

    planner.md, selector.md, prover.md, the three verifier prompts
    (verifier_1.md, verifier_2.md, verifier_3.md) and reviser.md all end
    with "Output strictly valid JSON ... no markdown fences and no extra
    text", so the reasoning stage's content is usually the JSON object itself. When it
    parses, we use it directly and skip the extraction call entirely.
    """
    if not text:
        return None
    try:
        obj = json.loads(clean_json_text(text))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


# ----------------------------------------------------------------------------
# LLM plumbing
# ----------------------------------------------------------------------------
def _headroom(messages: List[Dict[str, str]], num_ctx: int) -> int:
    """Tokens available for generation after the prompt, with a safety margin.

    num_ctx caps prompt + generation together, so this is the most a response
    can ever get: anything more is cut off by the context window, not the
    budget, after a long generation.

    The prompt size used to be estimated as chars // 4. That is optimistic for
    LaTeX-dense text and, worse, it is tokenizer-specific — the whole point of
    this project is now to run the same conjecture through four different
    tokenizers. llm_backend.chars_per_token() reads the running estimate that
    note_usage() revises from each call's real prompt token count, so the
    estimate converges on the truth for whichever model is loaded.

    The budget is set so that prompt + generation lands on
    COMPACT_THRESHOLD * num_ctx: the model may think and write until the window
    is COMPACT_THRESHOLD full, and only then is it cut off for compaction.
    """
    chars = sum(len(m["content"]) for m in messages)
    ratio = llm_backend.chars_per_token()
    used = int(chars / max(ratio, 1.0)) + 1
    return max(int(num_ctx * COMPACT_THRESHOLD) - used, 0)


COMPACT_SYSTEM = (
    "You are a compaction pass, not the agent. You are given part of a "
    "mathematician's reasoning trace that was cut off by the context limit. "
    "Write ONLY the structured summary in the format below — do not continue "
    "the mathematics, do not add claims, no preamble, no fences. If a previous "
    "pass's summary is included, fold it in: the result must cover everything "
    "it covered, plus this part.\n\n"
    "Format (fill every section, 'none' where nothing applies):\n"
    "## Goal\n[the task the trace was working on, in your own words]\n"
    "## Progress\n### Done\n- [x] ...\n### In Progress\n- [ ] ...\n"
    "### Blocked\n- ...\n## Key Decisions\n- **[decision]**: rationale\n"
    "## Next Steps\n1. [what must happen next, starting exactly where the "
    "trace stopped]\n## Critical Context\n- [definitions, facts, partial "
    "results needed to continue]\n<cited-lemmas>\n[comma-separated lemma ids "
    "the trace cited or used, or 'none']\n</cited-lemmas>"
)


def _split_inline_thinking(content: str, thinking: str) -> Tuple[str, str]:
    """Recover (content, thinking) when the model inlined its reasoning
    instead of separating it."""
    if not thinking and "\u003cthink\u003e" in content:
        m = re.search(r"\u003cthink\u003e(.*?)\u003c/think\u003e", content, re.DOTALL)
        if m:
            thinking = m.group(1).strip()
        content = _THINK_RE.sub("", content).strip()
    return content, thinking


def _split_trace_chunks(text: str, max_tokens: int) -> List[str]:
    """Split a trace into line-aligned chunks of at most ~max_tokens, so a
    compaction pass never needs more than a fraction of the window. Breaking
    on newlines keeps no LaTeX display cut mid-line; a chunk may still exceed
    the limit by one very long line, which the pass's headroom absorbs."""
    ratio = llm_backend.chars_per_token()
    max_chars = max(int(max_tokens * ratio), 1000)
    chunks: List[str] = []
    cur = ""
    for line in text.split("\n"):
        if cur and len(cur) + len(line) + 1 > max_chars:
            chunks.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        chunks.append(cur)
    return chunks


def _compact_pass(part: str, prev: str, tail: str,
                  part_no: int, n_parts: int, verbose: bool) -> str:
    """One summarisation pass over part of the cut-off trace.

    Returns the folded summary, or `prev` unchanged on any failure: the
    compaction must never lose what it already had, and it is a rescue, not a
    new stage that can fail the call.
    """
    user = (
        f"Part {part_no} of {n_parts} of the cut-off reasoning trace:\n"
        f"[Thinking trace]:\n{part}"
    )
    if prev:
        user = (
            "The summary of all previous parts — fold it in, drop nothing "
            f"from it:\n{prev}\n\n" + user
        )
    if tail:
        user += (
            "\n\n[The answer written before the cut, its tail — the work "
            f"ends here, mid-sentence]:\n{tail}"
        )
    messages = [
        {"role": "system", "content": COMPACT_SYSTEM},
        {"role": "user", "content": user},
    ]
    options = {
        **EXTRACT_OPTIONS,
        "num_predict": _headroom(messages, EXTRACT_OPTIONS["num_ctx"]),
    }
    try:
        reply = BACKEND.chat(messages, think=None, schema=None,
                             options=options)
    except (requests.RequestException, ValueError, KeyError) as e:
        log(f"  ⚠️  compaction pass {part_no}/{n_parts} transport error: {e}",
            verbose)
        return prev
    summary = (reply.content or "").strip()
    if reply.truncated or not summary:
        log(f"  ⚠️  compaction pass {part_no}/{n_parts} was truncated or "
            f"empty; keeping the previous summary.", verbose)
        return prev
    return summary


def _compact_trace(role: str, thinking: str, tail: str, verbose: bool,
                   prev_summary: str = "") -> str:
    """Summarise a cut-off reasoning trace in pi-style passes: chunked, each
    pass folding the previous summary in. prev_summary carries an earlier
    compaction round's result across, so a trace compacted in several rounds
    (a continuation that itself hit the wall) never loses what the first
    rounds established. Returns the summary, or prev_summary unchanged when
    the trace was empty or nothing survived the passes."""
    if not thinking.strip():
        return prev_summary
    parts = _split_trace_chunks(thinking, COMPACT_CHUNK_TOKENS)
    log(f"  📦 {role} trace too big to re-send; compacting in "
        f"{len(parts)} pass(es).", verbose)
    summary = prev_summary
    for i, part in enumerate(parts, 1):
        summary = _compact_pass(part, summary, tail, i, len(parts), verbose)
    return summary


def _resume_compacted(
    role: str,
    system_prompt: str,
    user_prompt: str,
    thinking: str,
    content: str,
    think: Any,
    verbose: bool,
    max_passes: int = MAX_COMPACTION_PASSES,
) -> Tuple[str, Optional[Dict[str, str]]]:
    """Pi-style overflow recovery after a truncated call.

    One pass: compact the reasoning trace (the scratch, summarised), keep
    the task prompt and the answer-so-far verbatim (the recent work), and
    ask the model to resume from the cut with the whole headroom of the
    now-smaller prompt: P + summary + C is strictly less than P + trace + C
    was.

    A continuation that hits the wall is not a failure but the next pass: its
    trace joins the summary, its content extends the verbatim answer, and the
    model resumes from the new cut. The summary carries across passes, so
    nothing established earlier is lost.

    Returns (answer, None) once a pass completes the answer, or ("", state)
    when the work is still unfinished after `max_passes` passes (or a pass can
    no longer go on). `state` is {"summary", "thinking", "answer"}, the
    partial work for the caller to hand on — the proof loop compacts it a
    final time and sends it to the reviser to pick a smaller lemma or
    revise the statement.
    """
    answer = content
    summary = ""
    want_think = think if (think is not None and think is not False) else None

    def _partial() -> Dict[str, str]:
        # The work-so-far for a caller to hand on: the running summary, the
        # latest (not-yet-compacted) trace, and the verbatim answer.
        return {"summary": summary, "thinking": thinking, "answer": answer}

    for _pass in range(max_passes):
        tail = answer[-COMPACT_TAIL_CHARS:]
        summary = _compact_trace(role, thinking, tail, verbose, summary)

        parts = [
            user_prompt,
            "---",
            "Your previous attempt at this task was cut off by the context "
            "limit.",
        ]
        if summary:
            parts += [
                "Summary of your reasoning trace (what is established, where "
                "it stopped, what it cited):",
                summary,
            ]
        if answer.strip():
            parts += [
                "Your answer so far, verbatim, ending exactly where it was "
                "cut off:",
                answer,
                "Resume from exactly that point and finish the answer. Do "
                "not repeat anything already written.",
            ]
        else:
            parts += [
                "Your previous answer had not started: the limit was hit "
                "during reasoning. Using the summary as your notes, write "
                "the complete answer now.",
            ]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n\n".join(parts)},
        ]
        # Show where the continuation prompt's tokens go: the original task is
        # re-sent in full, then the compaction (trace summary + verbatim
        # answer-so-far) rides on top. The two are different fixes, so keep
        # them separate in the log.
        _ratio = llm_backend.chars_per_token()
        log(
            f"  📦 {role} continuation prompt: task~{int(len(user_prompt)/_ratio)}"
            f" + summary~{int(len(summary)/_ratio)}"
            f" + answer~{int(len(answer)/_ratio)} tokens",
            verbose,
        )
        room = _headroom(messages, REASONING_OPTIONS["num_ctx"])
        if room < COMPACT_MIN_ROOM:
            log(f"  ⛔ {role} compaction left only ~{room} tokens of room; "
                f"not enough to continue.", verbose)
            return "", _partial()
        options = {
            **REASONING_OPTIONS,
            "temperature": TEMPERATURES.get(role, REASONING_OPTIONS["temperature"]),
            "num_predict": room,
        }
        try:
            reply = BACKEND.chat(messages, think=want_think, schema=None,
                                 options=options)
        except (requests.RequestException, ValueError, KeyError) as e:
            log(f"  ⚠️  {role} continuation transport error: {e}", verbose)
            return "", _partial()
        cont_content, cont_thinking = _split_inline_thinking(
            reply.content or "", reply.thinking or "")
        if reply.truncated:
            if not (cont_content.strip() or cont_thinking.strip()):
                log(f"  ⛔ {role} continuation was truncated with nothing "
                    f"to build on; the answer does not fit the window.",
                    verbose)
                return "", _partial()
            # The continuation itself hit the wall: not a failure, the next
            # pass. Its trace joins the summary, its content extends the
            # verbatim answer, and the model resumes from the new cut.
            answer += cont_content
            thinking = cont_thinking
            log(f"  🔁 {role} continuation hit the wall; the next pass "
                f"compacts its trace and resumes from the new cut.",
                verbose)
            continue
        if not cont_content.strip():
            log(f"  ⚠️  {role} continuation returned empty content.", verbose)
            return "", _partial()
        return (answer + cont_content).strip(), None
    # All `max_passes` passes spent and the answer is still unfinished.
    log(
        f"  ⛔ {role} still unfinished after {max_passes} compaction passes; "
        f"handing the partial work on.",
        verbose,
    )
    return "", _partial()


def _incomplete_report(
    role: str, partial: Dict[str, str], verbose: bool,
) -> str:
    """The third compaction, for a prover that overflowed the window.

    The two bounded compaction passes left the last (cut-off) trace not yet
    summarised; fold it into the running summary the way a continuation pass
    would have, and pair the summary with the verbatim partial answer. The
    result is what the reviser reads to pick a smaller lemma or revise the
    statement. Returns the report text, or "" when there is nothing to
    report.
    """
    thinking = str(partial.get("thinking") or "")
    answer = str(partial.get("answer") or "")
    if not (thinking.strip() or answer.strip()):
        return ""
    summary = str(partial.get("summary") or "")
    summary = _compact_trace(
        role, thinking, answer[-COMPACT_TAIL_CHARS:], verbose, summary,
    )
    parts = [
        "The prover ran out of its context window and did not finish: after "
        f"{MAX_COMPACTION_PASSES} compaction-and-resume passes the proof was "
        "still incomplete. Read its work below and choose a smaller lemma it "
        "should try instead — or a corrected statement, if its work shows "
        "the statement is false or badly posed.",
    ]
    if summary.strip():
        parts += [
            "Summary of its reasoning trace (what it established, where it "
            "stopped, what it cited):",
            summary,
        ]
    if answer.strip():
        parts += [
            "The partial answer it wrote, verbatim, ending exactly where it "
            "was cut off:",
            answer,
        ]
    return "\n\n".join(parts)


def _parse_reviser_decision(revision_text: str, verbose: bool) -> Dict[str, Any]:
    """Parse a reviser's JSON decision, with an extraction fallback.

    reviser.md demands raw JSON; when the model wraps it in prose, extract()
    recovers the fields. Returns {} when there is nothing to recover — the
    caller treats that as "no usable decision".
    """
    if not revision_text:
        return {}
    res = parse_json_or_none(revision_text)
    if res is None:
        res = extract(
            "Extract the revision decision. action is one of 'keep', "
            "'revise_statement' or 'new_lemma': 'keep' if the statement is "
            "kept unchanged, 'revise_statement' only if the text proposes a "
            "changed statement for the same lemma, 'new_lemma' only if the "
            "text proposes a different, smaller lemma to prove instead. Copy "
            "new_statement verbatim; it is the empty string when the "
            "statement is kept. Copy new_id verbatim; it is the empty string "
            "unless action is 'new_lemma'. If the text uses "
            "'statement_revision' instead of 'action', map true to "
            "'revise_statement' and false to 'keep'.",
            revision_text,
            REVISER_SCHEMA,
            "reviser",
            verbose,
        )
    return res or {}


def reason(
    system_prompt: str,
    user_prompt: str,
    role: str,
    think: Any = True,
    verbose: bool = True,
) -> Tuple[str, str, Optional[Dict[str, str]]]:
    """Stage 1: free-form reasoning. No `format`, so thinking is preserved.

    Returns (content, status, partial). status is "" on success, or "ceiling"
    when the role exhausted the context window without finishing — a signal
    that the task is too large, not that the call failed. `partial` is None on
    success; on "ceiling" after a bounded compaction rescue it carries the
    partial work ({"summary", "thinking", "answer"}) for the caller to hand
    on — the proof loop compacts it a final time and sends it to the reviser.

    Truncated output is never returned as if it were complete: `done_reason ==
    "length"` means the model was cut off mid-sentence, and the budget already
    covers the whole window. Before a truncation is reported as "ceiling" the
    exchange gets a pi-style compaction rescue — the thinking trace
    summarised, the answer written so far kept verbatim, the model asked to
    resume from the cut (see _resume_compacted). The rescue is bounded by
    MAX_COMPACTION_PASSES passes: a continuation that is itself truncated
    becomes the next pass's input, and once the passes are spent (or a pass
    leaves nothing to build on, or no headroom is left) the partial work is
    returned for the caller to decide what to do — for the prover, handing it
    to the reviser to pick a smaller lemma or revise the statement.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    options: Dict[str, Any] = {
        **REASONING_OPTIONS,
        "temperature": TEMPERATURES.get(role, REASONING_OPTIONS["temperature"]),
        # As much room as the window has left, recomputed for this prompt:
        # the prompt differs every iteration and every role, and there is no
        # point asking for more than the window holds or less than it does.
        "num_predict": _headroom(messages, REASONING_OPTIONS["num_ctx"]),
    }
    # `think` is passed through as a request. The backend reconciles it with
    # the probed capabilities and never sends False, whatever we ask for.
    want_think = think if (think is not None and think is not False) else None

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            reply = BACKEND.chat(messages, think=want_think, schema=None,
                                 options=options)
            content = reply.content or ""
            thinking = reply.thinking or ""
            content, thinking = _split_inline_thinking(content, thinking)

            hit_ceiling = reply.truncated

            if content.strip() and not hit_ceiling:
                return content.strip(), "", None

            if hit_ceiling:
                spent = (reply.usage.get("completion_tokens")
                         or (len(thinking) + len(content)) // 4)
                log(
                    f"  ⛔ {role} hit the context wall "
                    f"(num_ctx={options['num_ctx']}, generated {spent} tokens, "
                    f"~{len(thinking) // 4} of them thinking).",
                    verbose,
                )
                # A pi-style compaction rescue, when there is something to
                # resume from: the trace gets summarised, the answer-so-far
                # is kept verbatim, and the model is asked to finish from the
                # cut. The rescue is bounded by MAX_COMPACTION_PASSES passes;
                # if the work is still unfinished then, its partial state is
                # returned so the proof loop can hand it to the reviser.
                partial: Optional[Dict[str, str]] = None
                if COMPACT_ENABLED and (thinking.strip() or content.strip()):
                    resumed, partial = _resume_compacted(
                        role, system_prompt, user_prompt, thinking, content,
                        want_think, verbose,
                    )
                    if resumed:
                        log(f"  ✅ {role} completed after compaction + "
                            f"continuation.", verbose)
                        return resumed, "", None
                return "", "ceiling", partial

            log(f"  ⚠️  {role} returned empty content (attempt {attempt}).", verbose)
        except (requests.RequestException, ValueError, KeyError) as e:
            log(f"  ⚠️  {role} transport error (attempt {attempt}): {e}", verbose)

    # Retries exhausted on empty replies or transport errors; nothing to salvage.
    return "", "", None


def extract(
    instruction: str,
    source_text: str,
    schema: Dict[str, Any],
    role: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Stage 2: convert stage-1 prose into schema-conformant JSON.

    Thinking is never requested here, and the backend goes further and
    disables reasoning outright for this call, because on llama.cpp it is
    thinking that voids the grammar. That is the backend's problem, not
    this function's.
    """
    system = (
        "You convert a mathematician's written work into JSON. Copy the "
        "content faithfully: do not add claims, do not evaluate the "
        "mathematics, do not summarise beyond what is asked. "
        f"{instruction}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": source_text},
    ]

    # Schema-constrained first; drop to prompt-only if the grammar and the
    # model's default thinking mode conflict (symptom: empty content, output
    # stranded in message.thinking). The repair pipeline in clean_json_text
    # reclaims JSON from free-form output on the fallback path. The conflict
    # is a property of the model, not of one call, so once
    # discovered it's remembered for the rest of the run.
    global _SCHEMA_MODE_BROKEN
    use_schema = not _SCHEMA_MODE_BROKEN

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        options = dict(EXTRACT_OPTIONS)
        call_messages = messages
        if not use_schema:
            # Free-form mode: the schema moves from grammar to prompt.
            call_messages = [
                {
                    "role": "system",
                    "content": system
                    + " Respond with ONLY a JSON object matching this schema, "
                    "no fences, no commentary: "
                    + json.dumps(schema),
                },
            ] + messages[1:]
        # The extraction is a copy of the source, so it can be as long as the
        # source was; give it whatever this (larger) prompt leaves of the
        # window. That may be less than the source itself — the source plus a
        # copy of it do not always both fit — which is window physics, not a
        # budget to be nudged.
        options["num_predict"] = _headroom(call_messages, options["num_ctx"])

        content: Optional[str] = None
        try:
            # think is left as None throughout: whichever server we are on,
            # this stage wants the grammar honoured, and the backend knows
            # what that costs. No new mathematics happens here.
            reply = BACKEND.chat(
                call_messages,
                think=None,
                schema=schema if use_schema else None,
                options=options,
            )
            content = reply.content or ""
            if reply.truncated:
                log(f"  ⚠️  {role} extraction hit the token ceiling.", verbose)

            if use_schema and not content.strip():
                # Grammar/thinking conflict: qwen3.x thinks by default, the
                # format grammar suppresses the answer, and the output lands
                # in the thinking channel. Deterministic, so don't retry
                # same-mode.
                stranded = len(reply.thinking or "")
                log(
                    f"  ⚠️  {role} extraction returned empty content with format "
                    f"set ({stranded} chars stranded in thinking). Falling back "
                    f"to prompt-only JSON for the rest of the run.",
                    verbose,
                )
                use_schema = False
                _SCHEMA_MODE_BROKEN = True
                continue

            return json.loads(clean_json_text(content))
        except (requests.RequestException, KeyError, TypeError) as e:
            log(f"  ⚠️  {role} extraction transport error (attempt {attempt}): {e}", verbose)
        except json.JSONDecodeError as e:
            log(f"  ⚠️  {role} extraction parse error (attempt {attempt}): {e}", verbose)
            if content is not None:
                log(f"     raw ({len(content)} chars) head: {content[:200]!r}", verbose)
            if content is not None and content.strip():
                messages = messages[:2] + [
                    {"role": "assistant", "content": content[:1500]},
                    {
                        "role": "user",
                        "content": (
                            f"That was not valid JSON ({e}). Reply with ONLY the "
                            "JSON object, no fences and no commentary."
                        ),
                    },
                ]

    log(f"  ❌ {role} extraction failed after all retries.", verbose)
    return {}


# ----------------------------------------------------------------------------
# Context filtering
# ----------------------------------------------------------------------------
def planner_dag_view(dag: Dict[str, Any]) -> Dict[str, Any]:
    """Planner sees lemma statements + dependency structure, never proofs."""
    return {
        "proved_lemmas": {
            lid: {
                "statement": node["statement"],
                "dependencies": node.get("dependencies", []),
            }
            for lid, node in dag["lemmas"].items()
        }
    }


def prover_context(dag: Dict[str, Any]) -> Dict[str, Any]:
    """Everything the prover is allowed to cite: every proved statement.

    This used to be the direct dependencies' full proofs plus a statement
    index, on the strength of the planner having declared which lemmas the
    argument would need. With that declaration gone there is no way to know in
    advance which proofs would be worth shipping, and shipping all of them
    grows without bound — prompt tokens and generation tokens share num_ctx,
    so by lemma thirty the prover would be paying to re-read every proof it
    has ever written in order to have room to write one more.

    The cost is real: the prover sees *that* a lemma holds, not how it was
    established. If a conjecture turns out to need the latter, the fix is a
    second prover call — one to ask which lemmas it wants, one to prove with
    those proofs attached — not to widen this.
    """
    return {
        "proved_lemmas": {
            lid: {"statement": node["statement"]}
            for lid, node in dag["lemmas"].items()
        }
    }


def verifier_context(
    dag: Dict[str, Any],
    cited: List[str],
    references: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Statements of exactly the results the prover claims to have used.

    DAG lemmas and reference results are merged into one set, keyed by id and
    carrying the statement only. Deliberately not the whole DAG or the whole
    reference collection: the verifier agents are asked to reject "use of
    results not
    present in the provided set", which only bites if the set is the prover's
    declared citations: a proof leaning on a result it never declared then
    reads as an unjustified leap, which is what it is.
    """
    statements = {
        lid: {"statement": dag["lemmas"][lid]["statement"]}
        for lid in cited
        if lid in dag["lemmas"]
    }
    for ref in references:
        lid = str(ref.get("id") or "")
        if lid in cited and lid not in statements:
            statements[lid] = {
                "statement": str(ref.get("formal statement", ""))
            }
    return {"cited_results": statements}


def scan_citations(
    proof: str, dag: Dict[str, Any], references: List[Dict[str, Any]]
) -> List[str]:
    """Fallback edge recovery: which known ids appear in the proof text.

    Only used when the prover ignored its output format entirely, in which
    case the alternative is a node with no edges at all — a lemma that
    silently claims to stand on its own. Scans lemma ids and reference ids;
    word-boundary matching, so lemma_1 does not match inside lemma_10.
    """
    ids = set(dag["lemmas"])
    ids.update(str(r.get("id")) for r in references if r.get("id"))
    return [
        lid for lid in ids
        if re.search(rf"\b{re.escape(lid)}\b", proof)
    ]


def load_references() -> List[Dict[str, Any]]:
    """Load references.md — parsing.py's output.

    A strict JSON array of { id, slogan, "formal statement", reference,
    tags } objects, written once by parsing.py; this module only reads it.
    A missing file returns [], which is exactly the pre-features run: every
    call site degrades to the old prompts. A corrupt file warns and returns
    [] too — the right fix is to re-run parsing.py, not to hand the loop a
    subset. Ids (ref_N) are assigned by parsing.py, never here, so a prover
    that cites one is citing a name the file vouches for.
    """
    if not REFERENCES_FILE or not os.path.exists(REFERENCES_FILE):
        return []
    try:
        with open(REFERENCES_FILE, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if text.startswith("```"):
            text = _FENCE_RE.sub("", text).strip()
        data = json.loads(text)
    except (OSError, json.JSONDecodeError) as e:
        log(f"⚠️  {REFERENCES_FILE} is not valid JSON ({e}); ignoring it. "
            f"Re-run parsing.py if you expected references here.")
        return []
    if not isinstance(data, list):
        log(f"⚠️  {REFERENCES_FILE} is not a JSON array; ignoring it.")
        return []
    return [
        r for r in data
        if isinstance(r, dict) and str(r.get("id") or "").strip()
    ]


# ----------------------------------------------------------------------------
# Candidate handling
# ----------------------------------------------------------------------------
def planner_candidates(planner_res: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalise a planner reply into a list of well-formed candidate lemmas.

    Accepts the single-lemma shape as well, so a conjecture directory holding
    an old per-conjecture planner.md override keeps working: it simply gets a
    shortlist of length one, and every path below treats that as a shortlist
    that happens to be short.

    Candidates missing an id or a statement are dropped rather than passed on
    as half-lemmas, and duplicate ids are collapsed — models asked for five
    distinct routes will occasionally give the same lemma two numbers.
    """
    raw = planner_res.get("candidate_lemmas")
    if not isinstance(raw, list):
        single = planner_res.get("next_lemma")
        raw = [single] if isinstance(single, dict) else []

    out: List[Dict[str, Any]] = []
    seen: set = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        lemma_id = str(item.get("id") or "").strip()
        statement = str(item.get("statement") or "").strip()
        if not lemma_id or not statement or lemma_id in seen:
            continue
        seen.add(lemma_id)
        # Any dependencies field is dropped rather than honoured: a planner
        # that volunteers one (an old per-conjecture override, or a model
        # embellishing the schema) must not get to pre-empt the prover.
        out.append({"id": lemma_id, "statement": statement})
        if len(out) >= PLANNER_CANDIDATES:
            break
    return out


def screen_candidates(
    dag: Dict[str, Any], candidates: List[Dict[str, Any]]
) -> List[Tuple[Dict[str, Any], List[str]]]:
    """Pair each candidate with the reasons it cannot be used as it stands.

    Only one check survives now that candidates carry no dependencies: an id
    already in the DAG. Proving it again gains nothing, and writing it again
    would overwrite a node that other lemmas may already cite. An empty
    problem list means the candidate is ready for the prover.
    """
    screened: List[Tuple[Dict[str, Any], List[str]]] = []
    for cand in candidates:
        problems: List[str] = []
        if cand["id"] in dag["lemmas"]:
            problems.append(f"{cand['id']} is already in the DAG")
        screened.append((cand, problems))
    return screened


def log_candidates(
    screened: List[Tuple[Dict[str, Any], List[str]]],
    plan_summary: str,
    verbose: bool,
) -> None:
    """Print the shortlist in automation too, not just when a human is asked.

    A run you walked away from should still leave a record of the four roads
    not taken; without it the log shows a decision and none of the choice.
    """
    log("🗺  Planner shortlist:", verbose)
    log(interaction.render(screened, plan_summary), verbose)


def select_lemma(
    usable: List[Dict[str, Any]],
    dag: Dict[str, Any],
    conjecture: str,
    plan_summary: str,
    failed_attempts: Dict[str, List[str]],
    verbose: bool,
    proposer: str = "the planner",
) -> Dict[str, Any]:
    """Ask the selector agent to pick, from the candidates that survived
    screening, the one it judges most likely to come through the prover and
    the verifiers.

    The planner's best-first order is the fallback, not the default: if the
    selector fails, runs out of room, or names a lemma it was not offered, the
    first offered candidate is taken — which is what automation did before the
    selector existed, and which keeps --mode auto able to make progress even
    when this extra call goes wrong.
    """
    selector_sys = load_file(PROMPT_PATHS["selector.md"])
    selector_user = (
        f"Conjecture:\n{conjecture}\n\n"
        f"Planner's strategy summary:\n{plan_summary or '(none)'}\n\n"
        f"Proved lemmas so far:\n{json.dumps(planner_dag_view(dag), indent=2)}\n\n"
        f"Previously rejected attempts (routes the prover and the verifiers "
        f"have already found wanting):\n"
        f"{json.dumps(failed_attempts, indent=2)}\n\n"
        f"Candidate lemmas, in the order {proposer} offered them. Exactly "
        f"one will be sent to the prover; choose from these only:\n"
        f"{json.dumps(usable, indent=2)}"
    )
    text, _status, _partial = reason(selector_sys, selector_user, "selector",
                           THINK["selector"], verbose)
    check_mode_toggle(verbose)

    res: Optional[Dict[str, Any]] = None
    if text:
        # selector.md demands raw JSON output; extraction is the fallback.
        res = parse_json_or_none(text)
        if res is None:
            res = extract(
                "Extract the selector's decision. selected_id must be one of "
                "the offered candidate ids, copied verbatim.",
                text,
                SELECTOR_SCHEMA,
                "selector",
                verbose,
            )
    res = res or {}
    selected_id = str(res.get("selected_id") or "").strip()
    for cand in usable:
        if cand["id"] == selected_id:
            log(f"🧭 Selector picked {cand['id']}.", verbose)
            return cand

    if not text or not selected_id:
        log(
            f"⚠️  Selector gave no usable choice; taking the first candidate "
            f"instead ({usable[0]['id']}).",
            verbose,
        )
    else:
        log(
            f"⚠️  Selector chose {selected_id!r}, which was not among the "
            f"candidates offered; taking the first candidate instead "
            f"({usable[0]['id']}).",
            verbose,
        )
    return usable[0]


def check_mode_toggle(verbose: bool = True) -> None:
    """Apply a hotkey press, if one has landed since the last check.

    Called at several points in an iteration rather than only before the menu,
    because a press lands whenever the operator changes their mind and that is
    usually somewhere in the middle of a long prover, verifier or reviser call.
    MODE is
    still only *read* at the planning step, so checking early changes nothing
    functionally — it means the log acknowledges the key at the moment you
    press it instead of twenty minutes later, which is the difference between
    a toggle that feels responsive and one you press twice because the first
    press seemed not to register.

    The press is a toggle, so it works from either mode: it is the way back to
    automation from anywhere in the loop except the menu itself, where the
    listener is paused and 'a' does the same job.
    """
    global MODE
    if not HOTKEY.take():
        return
    MODE = "auto" if MODE == "human" else "human"
    log(
        "\n▶ Automation resumed; planning runs unattended from here."
        if MODE == "auto"
        else "\n⏸ Manual control engaged; you pick at the next planning step.",
        verbose,
    )


def _run_verifier(
    role: str,
    system_prompt: str,
    user_prompt: str,
    verbose: bool,
) -> Tuple[str, str]:
    """One atomic verification step: a single call to one verifier agent.

    Returns (decision, justification). A pass that hits the context wall
    without a verdict, or returns nothing parseable, is a reject — a check
    that cannot be completed can never count as an acceptance, so the proof
    goes to the reviser rather than into the DAG on the strength of the
    other two verifiers.
    """
    review, review_status, _partial = reason(
        system_prompt, user_prompt, role, THINK[role], verbose,
    )
    check_mode_toggle(verbose)
    if review_status == "ceiling" and not review:
        return (
            "reject",
            f"{role} exhausted its token budget without reaching a verdict; "
            "the proof is likely too long to review in one pass.",
        )
    # The verifier prompts demand raw JSON output; extraction is the fallback.
    res = parse_json_or_none(review) if review else None
    if res is None and review:
        res = extract(
            "Extract the verdict. decision is 'accept' only if the review "
            "endorses the proof without unresolved objections.",
            review,
            VERIFIER_SCHEMA,
            role,
            verbose,
        )
    res = res or {}
    decision = str(res.get("decision", "")).strip().lower()
    justification = str(res.get("justification") or "(no justification)").strip()
    return decision, justification


def run_proof_loop(
    *,
    verbose: bool,
    conjecture: str,
    prover_sys: str,
    reviser_sys: str,
    reference_block: str,
    reviser_ref_block: str,
    references: List[Dict[str, Any]],
    verifier_sys: Dict[str, str],
    ref_ids: Set[str],
    initial: Optional[Dict[str, Any]],
    fresh_target: Optional[Dict[str, Any]],
    on_round,
    on_proof,
    failed_attempts: Dict[str, List[str]],
    failed_lock: "threading.Lock",
    get_dag,
    toggle_hotkey,
    should_stop=None,
) -> Dict[str, Any]:
    """Steps 2-5 of the proof loop: prover -> verifiers -> reviser, shared
    by the serial run and by every parallel loop.

    `should_stop`, when given, is polled before every prover round: the
    parallel loops pass the run's stop test so a resolution or a Ctrl-C
    stops a loop between rounds instead of letting it spend the rest of its
    prover budget on a lemma no one will keep. A stop returns the current
    position marked "stopped" — the last on_round already published the
    resumable round, so the caller keeps it for the next run. The serial
    run passes nothing (its stop is the KeyboardInterrupt).

    `initial` is the in-flight state a cancelled prover round left behind
    (None for a fresh start); when given, the engine re-runs that round with
    that feedback instead of starting from attempt 1. `fresh_target` is the
    lemma the caller picked for a fresh start; it is ignored when resuming.

    The engine never touches the DAG or the checkpoint itself: `on_round` is
    called at each prover round with the in-flight state that round
    establishes (the caller publishes it and checkpoints it), and `on_proof`
    is called with the lemma's DAG node when a proof survives all three
    verifier checks (the caller commits it). Every failed_attempts mutation
    happens under `failed_lock` — a formality in the serial run, where the
    lock is never contended, and what keeps the shared reject list coherent
    in a parallel one.

    Returns {"proved", "lemma_id", "target", "proof", "attempt_notes"}. The
    caller records the final failure when not proved (from attempt_notes)
    and clears its in-flight state; the engine leaves no other traces.
    """
    resuming = initial is not None
    in_flight = initial if initial is not None else {}
    proof = ""  # bound before the return; the overflow-break paths skip the assignment

    if resuming:
        lemma_id = in_flight["lemma_id"]
        target: Dict[str, Any] = in_flight["target"]
    else:
        lemma_id = fresh_target["id"]
        # The target is the statement only: the candidate's other fields
        # (priority, notes) steer the loops, not the prover.
        target: Dict[str, Any] = {
            "id": lemma_id,
            "statement": fresh_target["statement"],
        }
    # The prover sees only the most recent failure: feedback is rebuilt
    # after each rejected round, so a retry reads the last verdict and
    # diagnosis, not the history of every earlier attempt. That history
    # is kept for the planner in failed_attempts — decomposing or
    # rerouting is its job, not the prover's.
    if resuming:
        feedback: List[str] = list(in_flight.get("feedback") or [])
        attempt_notes: List[str] = list(in_flight.get("attempt_notes") or [])
        first_attempt = in_flight["attempt"]
        last_proof = str(in_flight.get("last_proof") or "")
    else:
        feedback: List[str] = []
        attempt_notes: List[str] = list(failed_attempts.get(lemma_id, []))
        first_attempt = 1
        last_proof = ""
    proved = False
    attempt = first_attempt
    # `attempt` is a while-loop counter rather than a for-range because the
    # reviser can reset it: a lemma it decomposes into a smaller one starts
    # its own prover rounds from 1 (see the new_lemma branch below).
    while attempt <= MAX_PROOF_ATTEMPTS:
        if should_stop is not None and should_stop():
            return {
                "proved": False,
                "lemma_id": lemma_id,
                "target": target,
                "proof": last_proof,
                "attempt_notes": attempt_notes,
                "stopped": True,
            }
        dag = get_dag()
        # Checkpoint: this prover round is now the resumable position. If
        # the run is cancelled anywhere inside it, the next run re-runs
        # the round with this target and this feedback — the round's own
        # in-flight call is lost, nothing completed before it is.
        in_flight_state = {
            "lemma_id": lemma_id,
            "target": target,
            "attempt": attempt,
            "feedback": feedback,
            "attempt_notes": attempt_notes,
            "last_proof": last_proof,
        }
        on_round(in_flight_state)
        log(
            f"📝 Proof attempt {attempt}/{MAX_PROOF_ATTEMPTS} for {lemma_id}.",
            verbose,
        )

        # ---------------- Step 2: Prover ----------------
        # prover.md instructs the model to reply with {"lemma_id",
        # "cited_lemmas", "proof"}. We parse that JSON locally rather than
        # via a second model call, so the proof text can never be abridged
        # or paraphrased; if the model ignored the format, its content is
        # taken verbatim as the proof.
        prover_user = (
            f"Conjecture:\n{conjecture}\n\n"
            f"Available proved lemmas:\n"
            f"{json.dumps(prover_context(dag), indent=2)}"
            f"{reference_block}\n\n"
            f"Lemma to prove:\n{json.dumps(target, indent=2)}\n"
            + (
                f"\nFeedback from the previous attempt "
                f"(the most recent rejection only):\n"
                f"{json.dumps(feedback, indent=2)}"
                if feedback
                else ""
            )
        )
        proof_text, prover_status, prover_partial = reason(
            prover_sys, prover_user, "prover", THINK["prover"], verbose
        )
        toggle_hotkey()

        if prover_status == "ceiling":
            # The prover exhausted the window and the compaction rescue
            # (bounded by MAX_COMPACTION_PASSES) could not finish it. The
            # partial work is still worth reading: compact it one last time
            # and let the reviser pick a smaller lemma or revise the
            # statement from it, instead of discarding it and asking the
            # planner to decompose blind.
            report = _incomplete_report(
                "prover", prover_partial or {}, verbose,
            )
            if not report:
                log(
                    f"⛔ Lemma {lemma_id} is too large to prove in one call "
                    f"and left nothing to build on. Asking the planner to "
                    f"decompose it.",
                    verbose,
                )
                attempt_notes.append(
                    "Lemma too large: the prover exhausted its entire token "
                    "budget without completing a proof and left nothing to "
                    "resume from. Decompose this into smaller, "
                    "independently provable lemmas rather than re-proposing "
                    "it."
                )
                break
            log(
                f"⛔ Lemma {lemma_id} overflowed the context window after "
                f"{MAX_COMPACTION_PASSES} compaction passes; asking the "
                f"reviser to pick a smaller lemma or revise the statement "
                f"from the partial proof.",
                verbose,
            )
            overflow_user = (
                f"Conjecture:\n{conjecture}\n\n"
                f"Proved lemmas so far:\n{json.dumps(planner_dag_view(dag), indent=2)}"
                f"{reviser_ref_block}\n\n"
                f"Target lemma:\n{json.dumps(target, indent=2)}\n\n"
                f"{report}"
            )
            revision_text, _revision_status, _revision_partial = reason(
                reviser_sys, overflow_user, "reviser",
                THINK["reviser"], verbose,
            )
            toggle_hotkey()
            revision_res = _parse_reviser_decision(revision_text, verbose)
            action = str(revision_res.get("action") or "").strip().lower()
            diagnosis = str(revision_res.get("diagnosis") or "").strip()
            new_id = str(revision_res.get("new_id") or "").strip()
            new_stmt = str(revision_res.get("new_statement") or "").strip()
            overflow_feedback = [
                f"Prover: the proof of {lemma_id} overflowed the context "
                f"window after {MAX_COMPACTION_PASSES} compaction passes; "
                f"the reviser is choosing a smaller lemma or a corrected "
                f"statement from the partial work."
            ]
            if diagnosis:
                overflow_feedback.append(f"Reviser's diagnosis: {diagnosis}")
            if (
                action == "new_lemma"
                and new_id
                and new_stmt
                and new_id not in dag["lemmas"]
            ):
                # The reviser found a smaller lemma worth trying. Set the
                # overflowing lemma aside (its history is recorded under its
                # own id) and start the prover on the smaller lemma with a
                # fresh budget and no inherited verdict.
                old_id = lemma_id
                failed_attempts[old_id] = (
                    list(attempt_notes)
                    + list(overflow_feedback)
                    + [
                        f"Overflowed: {old_id} ran out of the prover's "
                        f"context window; the reviser is instead trying "
                        f"{new_id}."
                    ]
                )
                lemma_id = new_id
                target = {"id": lemma_id, "statement": new_stmt}
                attempt_notes = [
                    f"Overflowed from {old_id}: the prover ran out of the "
                    f"context window; the reviser is trying the smaller "
                    f"{lemma_id} instead."
                ]
                feedback = []
                last_proof = ""
                attempt = 1
                log(
                    f"↻ Prover overflowed {old_id}; now trying the smaller "
                    f"lemma {lemma_id}:\n{new_stmt}",
                    verbose,
                )
                continue
            if (
                action == "revise_statement"
                and new_stmt
                and new_stmt != target["statement"]
            ):
                # The partial work shows the statement is false or badly
                # posed, and the reviser has corrected it. The lemma keeps
                # its id and its attempt budget — the overflowing round
                # still counts — and the prover re-runs on the correction
                # with this round's feedback, the way a rejected-proof
                # revision does.
                target = {"id": lemma_id, "statement": new_stmt}
                last_proof = ""
                overflow_feedback.append(
                    f"Revised statement proposed: {new_stmt}"
                )
                feedback = overflow_feedback
                attempt_notes.extend(overflow_feedback)
                attempt += 1
                log(
                    f"↻ Reviser revised the lemma after the overflow; the "
                    f"prover starts again from:\n{new_stmt}",
                    verbose,
                )
                continue
            log(
                "⛔ Reviser could not pick a usable smaller lemma or "
                "revised statement from the partial proof; the planner "
                "will decompose.",
                verbose,
            )
            attempt_notes.extend(overflow_feedback)
            attempt_notes.append(
                "Lemma too large: the prover exhausted its context window "
                "and the reviser could neither split it further nor "
                "revise its statement. Decompose into smaller, "
                "independently provable lemmas."
            )
            break

        proof = proof_text
        declared: Optional[List[str]] = None
        prover_res = parse_json_or_none(proof_text)
        if prover_res is not None:
            candidate = prover_res.get("proof")
            if isinstance(candidate, str) and candidate.strip():
                proof = candidate.strip()
            raw_cited = prover_res.get("cited_lemmas")
            if isinstance(raw_cited, list):
                declared = [str(c).strip() for c in raw_cited if str(c).strip()]

        if not proof:
            log(f"❌ Prover produced no proof for {lemma_id}.", verbose)
            attempt_notes.append("Prover returned nothing.")
            break

        # The last proof of this lemma: what the verifiers are about to
        # check, what the reviser will judge its difficulty by, and what
        # the checkpoint records beside the in-flight state.
        last_proof = proof

        # The DAG's edges now come from here. An empty declared list is
        # taken at face value — a proof from first principles has no
        # dependencies — but a missing one means the prover ignored its
        # format, and scanning the text beats recording a lemma as
        # standing on nothing.
        if declared is None:
            dep_ids = scan_citations(proof, dag, references)
            if dep_ids:
                log(
                    f"  ℹ️  Prover declared no citations; recovered "
                    f"{', '.join(dep_ids)} from the proof text.",
                    verbose,
                )
        else:
            # Citations may name DAG lemmas or parsed references; the
            # verifier's context is the union of the two, so both are
            # kept. A name in neither set is dropped, and the verifier is
            # about to see a proof that leans on a result absent from its
            # context, which is exactly the unjustified step it is meant
            # to catch.
            dep_ids = [c for c in declared if c in dag["lemmas"] or c in ref_ids]
            phantom = [
                c for c in declared if c not in dag["lemmas"] and c not in ref_ids
            ]
            if phantom:
                log(
                    f"  ⚠️  Prover cited ids that are neither lemmas nor "
                    f"references: {', '.join(phantom)}.",
                    verbose,
                )

        log(
            f"✍️ Proof generated for {lemma_id} ({len(proof)} chars"
            f"{', cites ' + ', '.join(dep_ids) if dep_ids else ', no citations'}).",
            verbose,
        )

        # ---------------- Step 3: the three verifier checks -----------
        # Three atomic checks, one per verifier agent, in the order
        # verifier_1, verifier_2, verifier_3. Each is a single call to its
        # own agent; one reject ends the counting and carries its
        # reasoning to the reviser; only a proof that survives all three
        # is accepted.
        verifier_user = (
            f"Conjecture:\n{conjecture}\n\n"
            f"Cited results (statements of exactly the lemmas and "
            f"references the proof declares it used; nothing else is "
            f"available to it):\n"
            f"{json.dumps(verifier_context(dag, dep_ids, references), indent=2)}\n\n"
            f"Target lemma:\n{json.dumps(target, indent=2)}\n\n"
            f"Proposed proof:\n{proof}"
        )
        reject_just: Optional[str] = None
        for step, agent in enumerate(VERIFIER_AGENTS, 1):
            role = agent[:-3]   # "verifier_1.md" -> "verifier_1"
            decision, justification = _run_verifier(
                role, verifier_sys[agent], verifier_user, verbose,
            )
            log(
                f"🔍 Verifier {step}/{len(VERIFIER_AGENTS)} ({role}): "
                f"{decision.upper() or '???'} — {justification}",
                verbose,
            )
            if decision != "accept":
                reject_just = justification
                break

        if reject_just is None:
            # ---------------- Step 4: DAG update ----------------
            # All three verifier checks accepted the same proof.
            log(
                f"✅ Lemma {lemma_id} passed all {len(VERIFIER_AGENTS)} "
                f"verifier checks. Adding to DAG.",
                verbose,
            )
            on_proof(lemma_id, {
                "statement": target["statement"],
                "proof": proof,
                "dependencies": dep_ids,
            })
            with failed_lock:
                failed_attempts.pop(lemma_id, None)
            proved = True
            break

        # ---------------- Step 5: Reviser ----------------
        log(
            f"❌ Proof rejected for {lemma_id}. Sending the proof and the "
            f"verdict to the reviser.",
            verbose,
        )
        reviser_user = (
            f"Conjecture:\n{conjecture}\n\n"
            f"Proved lemmas so far:\n{json.dumps(planner_dag_view(dag), indent=2)}"
            f"{reviser_ref_block}\n\n"
            f"Target lemma:\n{json.dumps(target, indent=2)}\n\n"
            f"The last proof of this lemma (the one just rejected; judge "
            f"the lemma's difficulty from it):\n{proof}\n\n"
            f"Verifier's reasoning (why the proof failed):\n{reject_just}"
        )
        revision_text, _revision_status, _partial = reason(
            reviser_sys, reviser_user, "reviser", THINK["reviser"], verbose
        )
        toggle_hotkey()

        revision_res: Optional[Dict[str, Any]] = None
        if revision_text:
            # reviser.md demands raw JSON output; extraction is the
            # fallback.
            revision_res = parse_json_or_none(revision_text)
            if revision_res is None:
                revision_res = extract(
                    "Extract the revision decision. action is one of "
                    "'keep', 'revise_statement' or 'new_lemma': 'keep' if "
                    "the statement is kept unchanged, 'revise_statement' "
                    "only if the text proposes a changed statement for the "
                    "same lemma, 'new_lemma' only if the text proposes a "
                    "different, smaller lemma to prove instead. Copy "
                    "new_statement verbatim; it is the empty string when "
                    "the statement is kept. Copy new_id verbatim; it is the "
                    "empty string unless action is 'new_lemma'. If the text "
                    "uses 'statement_revision' instead of 'action', map "
                    "true to 'revise_statement' and false to 'keep'.",
                    revision_text,
                    REVISER_SCHEMA,
                    "reviser",
                    verbose,
                )
        else:
            log(
                "⚠️  Reviser returned nothing; the verdict alone goes back "
                "to the prover.",
                verbose,
            )
        revision_res = revision_res or {}

        # `action` is the current contract; older per-conjecture
        # overrides that still emit statement_revision are read the old
        # way so they keep working.
        action = str(revision_res.get("action") or "").strip().lower()
        if action not in ("keep", "revise_statement", "new_lemma"):
            action = (
                "revise_statement"
                if bool(revision_res.get("statement_revision", False))
                else "keep"
            )
        diagnosis = str(revision_res.get("diagnosis") or "").strip()
        new_id = str(revision_res.get("new_id") or "").strip()
        new_stmt = str(revision_res.get("new_statement") or "").strip()

        feedback = [f"Verifier: {reject_just}"]
        if diagnosis:
            feedback.append(f"Reviser's diagnosis: {diagnosis}")

        if (
            action == "new_lemma"
            and new_id
            and new_stmt
            and new_id not in dag["lemmas"]
        ):
            # Decomposition: the target is too hard to prove as stated.
            # Record the old lemma's history — and why it was set aside —
            # under its own id, then restart the prover on the smaller
            # lemma with a fresh attempt budget and no inherited verdict.
            old_id = lemma_id
            # The current round's rejection is the old lemma's last word;
            # fold it into its history before we set the lemma aside.
            with failed_lock:
                failed_attempts[old_id] = (
                    list(attempt_notes)
                    + list(feedback)
                    + [
                        f"Decomposed: {old_id} was judged too hard to prove "
                        f"as stated; the reviser is instead trying {new_id}."
                    ]
                )
            lemma_id = new_id
            target = {"id": lemma_id, "statement": new_stmt}
            attempt_notes = [
                f"Decomposed from {old_id} by the reviser (the original "
                f"target was judged too hard to prove as stated)."
            ]
            # A fresh lemma has no rejection of its own, so the prover
            # starts clean rather than inheriting the old proof's verdict.
            feedback = []
            last_proof = ""
            attempt = 1
            log(
                f"↻ Reviser decomposed {old_id}; the prover now starts on "
                f"the smaller lemma {lemma_id}:\n{new_stmt}",
                verbose,
            )
            continue

        if action == "revise_statement" and new_stmt and new_stmt != target["statement"]:
            target = {"id": lemma_id, "statement": new_stmt}
            last_proof = ""
            log(
                f"↻ Reviser revised the lemma; the prover starts again "
                f"from:\n{new_stmt}",
                verbose,
            )
            feedback.append(f"Revised statement proposed: {new_stmt}")
        elif action == "revise_statement":
            log(
                "⚠️  Reviser flagged a statement revision but gave none "
                "usable; keeping the statement.",
                verbose,
            )
        elif action == "new_lemma":
            # The reviser wanted to decompose but the new id was missing or
            # already taken; fall back to keeping the statement.
            log(
                "⚠️  Reviser proposed a new lemma with a missing or "
                "already-taken id; keeping the statement instead.",
                verbose,
            )
        else:
            log(
                "↻ Reviser kept the statement; the prover re-tries with "
                "the verdict as feedback.",
                verbose,
            )
        attempt_notes.extend(feedback)
        attempt += 1
        # Loop continues: the next prover round works on `target`, with
        # only this round's feedback.

    return {
        "proved": proved,
        "lemma_id": lemma_id,
        "target": target,
        "proof": proof,
        "attempt_notes": attempt_notes,
        "stopped": False,
    }


def _serial_on_round(
    iteration: int,
    failed_attempts: Dict[str, List[str]],
    state: Dict[str, Any],
    verbose: bool,
) -> None:
    """The serial round boundary: publish the in-flight state into
    _LIVE_STATE (the SIGINT handler's source) and into the checkpoint, so a
    Ctrl-C mid-round re-runs exactly that round."""
    _LIVE_STATE["in_flight"] = state
    save_checkpoint(iteration, failed_attempts, state, verbose)


def _serial_on_proof(
    lemma_id: str, node: Dict[str, Any], verbose: bool
) -> None:
    """The serial success boundary: add the lemma to the DAG, reading it
    fresh under the DAG lock so the write composes with any later reader."""
    with DAG_LOCK:
        dag = load_dag()
        dag["lemmas"][lemma_id] = node
        save_dag(dag)


def _build_reference_blocks(
    verbose: bool,
) -> Tuple[List[Dict[str, Any]], str, str, str]:
    """The reference context the prompts carry, built once per run.

    The prover gets the full collection unconditionally — it is the tool
    that cites — so it is rendered once here and reused on every attempt.
    The serial run and the parallel threads share these strings; they are
    read-only once built.
    """
    references = load_references()
    prover_refs = [
        {
            "id": r["id"],
            "slogan": r.get("slogan", ""),
            "formal statement": r.get("formal statement", ""),
            "tags": r.get("tags", []),
        }
        for r in references
    ]
    reference_block = (
        "\n\nKnown references (theorem-level results from the parsed "
        "collection; you may cite any of them by id in cited_lemmas without "
        "proving them yourself):\n"
        + json.dumps(prover_refs, indent=2)
        if prover_refs
        else ""
    )
    # The planner and the reviser share one short view of the collection:
    # below PLANNER_REFERENCE_LIMIT they get an id + slogan per result, at or
    # above it none. See the limit's comment for why a blunt rule beats a
    # graded one here.
    short_refs = [
        {
            "id": r["id"],
            "slogan": r.get("slogan", ""),
            "tags": r.get("tags", []),
        }
        for r in references
    ]
    if references:
        if len(references) < PLANNER_REFERENCE_LIMIT:
            planner_ref_block = (
                "\n\nKnown references (named theorem-level results the prover "
                "may cite by id; the prover sees their full statements):\n"
                + json.dumps(short_refs, indent=2)
            )
            reviser_ref_block = (
                "\n\nKnown references (named theorem-level results from the "
                "parsed collection; treat them as established):\n"
                + json.dumps(short_refs, indent=2)
            )
        else:
            planner_ref_block = ""
            reviser_ref_block = ""
            log(
                f"ℹ️  {len(references)} references parsed; that is at or "
                f"above the planner limit ({PLANNER_REFERENCE_LIMIT}), so the "
                f"planner and the reviser get none. The prover sees them all."
            )
    else:
        planner_ref_block = ""
        reviser_ref_block = ""
    return references, reference_block, planner_ref_block, reviser_ref_block


def _build_comments_block() -> str:
    """comments.md, if the operator left one: free-form notes on possible
    approaches to a proof or counterexample, handed to the planner verbatim
    on every iteration. The planner is the only agent that reads it, and an
    absent file degrades to the pre-comments prompt, the way the reference
    blocks do."""
    comments = load_file(COMMENTS_FILE)
    return (
        "\n\nHuman comments (the operator's notes on possible approaches to a "
        "proof or counterexample; suggestions to weigh, not instructions):\n"
        + comments
        if comments
        else ""
    )


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
def run_loop(verbose: bool = True) -> Dict[str, Any]:
    """Execute the multi-agent theorem proving loop.

    Args:
        verbose: If True, print progress to the console.

    Returns:
        The final DAG, so callers running with --no-verbose still get a result.
    """
    global MODE            # the hotkey and the menu both retarget it mid-run

    conjecture = load_file(CONJECTURE_FILE)
    if not conjecture:
        log(f"{CONJECTURE_FILE} is empty — write your conjecture there first.", verbose)
        return load_dag()

    planner_sys = load_file(PROMPT_PATHS["planner.md"])
    selector_sys = load_file(PROMPT_PATHS["selector.md"])
    prover_sys = load_file(PROMPT_PATHS["prover.md"])
    # One prompt per verification step; VERIFIER_AGENTS names the three.
    verifier_sys = {name: load_file(PROMPT_PATHS[name]) for name in VERIFIER_AGENTS}
    # One reviser for both failures: a rejected proof, and a prover that
    # overflowed the context window. The user message tells it which case it
    # is — a full rejected proof with the verifier's reasoning, or a
    # summarised trace plus the partial answer.
    reviser_sys = load_file(PROMPT_PATHS["reviser.md"])

    empty = [name for name, text in (
        ("planner.md", planner_sys), ("selector.md", selector_sys),
        ("prover.md", prover_sys), *verifier_sys.items(),
        ("reviser.md", reviser_sys),
    ) if not text]
    if empty:
        # A missing system prompt does not crash — it produces an agent with
        # no instructions, which fails in ways that look like model problems.
        log(f"Missing or empty prompt files: {', '.join(empty)}. Aborting.", verbose)
        return load_dag()

    references, reference_block, planner_ref_block, reviser_ref_block = (
        _build_reference_blocks(verbose)
    )
    ref_ids = {str(r["id"]) for r in references}
    comments_block = _build_comments_block()

    # Checkpoint restore: the DAG is reloaded from disk every iteration, so a
    # cancelled run leaves only run-level state to restore — where the
    # iteration budget stood, the planner's reject list, and the lemma that
    # was in flight, if the cancellation landed mid-proof. A missing or
    # corrupt checkpoint degrades to a fresh run, never to a crash.
    cp = load_checkpoint(verbose)
    if cp is not None and not os.path.exists(DAG_FILE):
        log("⚠️  Checkpoint without a DAG file; discarding it.", verbose)
        discard_checkpoint()
        cp = None
    if cp is None:
        start_iteration = 1
        failed_attempts: Dict[str, List[str]] = {}
        in_flight: Optional[Dict[str, Any]] = None
    else:
        start_iteration = cp["iteration"]
        failed_attempts = cp["failed_attempts"]
        in_flight = cp["in_flight"]
        if in_flight is None:
            log(
                f"♻ Resuming at iteration {start_iteration}/{MAX_ITERATIONS} "
                f"({len(failed_attempts)} rejected lemma(s) on record).",
                verbose,
            )
        else:
            log(
                f"♻ Resuming at iteration {start_iteration}/{MAX_ITERATIONS}, "
                f"in-flight {in_flight['lemma_id']} at prover round "
                f"{in_flight['attempt']}/{MAX_PROOF_ATTEMPTS}.",
                verbose,
            )
    if start_iteration > MAX_ITERATIONS:
        log(
            f"\n⏹ The checkpoint stands at iteration {start_iteration}, past "
            f"--max-iterations ({MAX_ITERATIONS}). Re-run with a larger "
            f"--max-iterations to continue, or --fresh to restart the budget "
            f"(the DAG is kept either way).",
            verbose,
        )
        return load_dag()

    warned_dangling = set()
    _LIVE_STATE["failed_attempts"] = failed_attempts
    _LIVE_STATE["in_flight"] = in_flight

    for iteration in range(start_iteration, MAX_ITERATIONS + 1):
        log(f"\n--- Iteration {iteration}/{MAX_ITERATIONS} ---", verbose)
        check_mode_toggle(verbose)
        dag = load_dag()

        # An existing DAG may cite reference ids; one that no longer exists in
        # references.md (a re-parse dropped it) dangles every proof that used
        # it, and the verifier would reject those for good reason. Warn once
        # per (lemma, dependency) pair rather than every iteration.
        known_ids = set(dag["lemmas"]) | ref_ids
        for lid, node in dag["lemmas"].items():
            for dep in node.get("dependencies", []):
                if dep not in known_ids and (lid, dep) not in warned_dangling:
                    warned_dangling.add((lid, dep))
                    log(
                        f"⚠️  DAG node {lid} cites {dep}, which is neither a "
                        f"lemma nor a parsed reference; verification of it "
                        f"will see a citation with no statement behind it."
                    )

        # Checkpoint: this iteration is now the resumable position. If the
        # previous run was cancelled mid-proof, `in_flight` names the lemma
        # that goes straight back to the prover; otherwise the iteration
        # starts with planning. The proof loop and the end-of-iteration
        # bookkeeping keep it up to date, and the SIGINT handler writes it
        # on Ctrl-C from _LIVE_STATE.
        _LIVE_STATE["iteration"] = iteration
        _LIVE_STATE["in_flight"] = in_flight
        save_checkpoint(iteration, failed_attempts, in_flight, verbose)

        resuming = in_flight is not None and in_flight["lemma_id"] not in dag["lemmas"]
        if resuming:
            log(
                f"♻ In-flight {in_flight['lemma_id']}: re-running prover round "
                f"{in_flight['attempt']}/{MAX_PROOF_ATTEMPTS} (earlier rounds' "
                f"feedback is kept; the interrupted round itself is re-run).",
                verbose,
            )
        elif in_flight is not None:
            # The lemma was accepted between two checkpoints: its proof is
            # already in the DAG, so there is nothing in flight to resume.
            log(
                f"♻ Checkpoint's in-flight {in_flight['lemma_id']} is already "
                f"in the DAG; planning as usual.",
                verbose,
            )
            in_flight = None
            _LIVE_STATE["in_flight"] = None
        if not resuming:

            # ---------------- Step 1: Planner ----------------
            planner_user = (
                f"Conjecture:\n{conjecture}\n\n"
                f"Proved lemmas so far:\n{json.dumps(planner_dag_view(dag), indent=2)}"
                f"{planner_ref_block}{comments_block}\n\n"
                f"Previously rejected attempts (avoid or decompose these):\n"
                f"{json.dumps(failed_attempts, indent=2)}"
            )
            plan_text, plan_status, _partial = reason(
                planner_sys, planner_user, "planner", THINK["planner"], verbose
            )
            check_mode_toggle(verbose)
            if plan_status == "ceiling" and not plan_text:
                log("Planner exhausted its token budget. Re-planning.", verbose)
                continue
            if not plan_text:
                log("Planner produced nothing. Re-planning next iteration.", verbose)
                continue

            # planner.md demands raw JSON output; extraction is only the fallback.
            planner_res = parse_json_or_none(plan_text)
            if planner_res is None:
                planner_res = extract(
                    "Extract the plan. Copy every candidate lemma the text proposes "
                    "into candidate_lemmas, preserving the order it presents them "
                    "in. "
                    "is_conjecture_proved must be true only if the text "
                    "explicitly concludes the conjecture is fully proved; "
                    "is_conjecture_disproved must be true only if the text "
                    "explicitly concludes that a full counterexample to the "
                    "conjecture has been established.",
                    plan_text,
                    PLANNER_SCHEMA,
                    "planner",
                    verbose,
                )

            if planner_res.get("is_conjecture_proved"):
                log("\n🎉 Conjecture has been fully proved!", verbose)
                dag["conjecture"] = conjecture
                dag["status"] = "proved"
                save_dag(dag)
                discard_checkpoint()
                return dag

            if planner_res.get("is_conjecture_disproved"):
                log(
                    "\n💥 Conjecture has been disproved: a counterexample to it "
                    "stands in the DAG.",
                    verbose,
                )
                dag["conjecture"] = conjecture
                dag["status"] = "disproved"
                save_dag(dag)
                discard_checkpoint()
                return dag

            candidates = planner_candidates(planner_res)
            if not candidates:
                log("Planner proposed no usable lemma. Re-planning.", verbose)
                continue

            screened = screen_candidates(dag, candidates)
            summary = str(planner_res.get("plan_summary") or "")

            # ---------------- Step 1b: Selection ----------------
            # The one place the two modes differ, and so the one place MODE is
            # read. Everything after this block runs identically whether the lemma
            # was chosen by a person or by the selector, which is what keeps
            # --mode auto honest as a control.
            next_lemma: Optional[Dict[str, Any]] = None
            if MODE == "human":
                choice = interaction.choose(
                    screened, dag["lemmas"].keys(), HOTKEY, summary
                )
                if choice.action == "quit":
                    log(
                        "\n⏹ Stopped by the operator; DAG and checkpoint kept — "
                        "re-run the same command to resume.",
                        verbose,
                    )
                    save_dag(dag)
                    return dag
                if choice.action == "replan":
                    # Recorded against every candidate shown, because the channel
                    # the planner reads is keyed by lemma id and the point is that
                    # it should not come back with this same shortlist.
                    for cand, _ in screened:
                        for note in choice.notes:
                            failed_attempts.setdefault(cand["id"], []).append(note)
                    log("↩︎ Shortlist rejected; asking the planner again.", verbose)
                    continue
                if choice.action == "auto":
                    MODE = "auto"
                    log(f"▶ Automation resumed. {HOTKEY.hint()}", verbose)
                elif choice.action == "assert":
                    # The operator has vouched for it, so there is nothing for the
                    # prover or the verifiers to do: no proof is generated, no
                    # review is run, and the lemma is in the DAG before the next
                    # iteration reloads it. `provenance` marks it as resting on a
                    # person rather than on a machine-checked argument — absent on
                    # every node written by the pipeline, and on every DAG file
                    # that predates this, so read it with .get().
                    asserted = choice.lemma or {}
                    dag["lemmas"][asserted["id"]] = {
                        "statement": asserted["statement"],
                        "proof": "Asserted by the operator; not machine-proved.",
                        "dependencies": [],   # nothing was cited; nothing was proved
                        "provenance": "operator",
                    }
                    save_dag(dag)
                    failed_attempts.pop(asserted["id"], None)
                    log(
                        f"🖊  Lemma {asserted['id']} accepted on your authority "
                        f"and added to the DAG unproved.",
                        verbose,
                    )
                    continue
                else:
                    next_lemma = choice.lemma

            if next_lemma is None:
                # Automatic selection. Screening removes the candidates whose ids
                # are already in the DAG; of what survives, the selector agent
                # weighs each one and names the one it judges most likely to
                # succeed.
                # The planner's best-first order is only the fallback: with
                # exactly one candidate surviving there is nothing to choose, and
                # select_lemma() itself falls back when the selector fails.
                log_candidates(screened, summary, verbose)
                usable = [cand for cand, problems in screened if not problems]
                if not usable:
                    log("No candidate is currently provable; re-planning.", verbose)
                    for cand, problems in screened:
                        failed_attempts.setdefault(cand["id"], []).append(
                            f"Planning error: {'; '.join(problems)}."
                        )
                    continue
                if len(usable) == 1:
                    next_lemma = usable[0]
                    if next_lemma is not screened[0][0]:
                        log(
                            f"↷ Skipped {screened[0][0]['id']} "
                            f"({'; '.join(screened[0][1])}); took {next_lemma['id']}.",
                            verbose,
                        )
                else:
                    next_lemma = select_lemma(
                        usable, dag, conjecture, summary, failed_attempts, verbose
                    )

        if resuming:
            lemma_id = in_flight["lemma_id"]
            lemma_stmt = in_flight["target"]["statement"]
            log(f"📌 Next Lemma [{lemma_id}]: {lemma_stmt}", verbose)
        else:
            lemma_id = next_lemma["id"]
            lemma_stmt = next_lemma["statement"]
            log(f"📌 Next Lemma [{lemma_id}]: {lemma_stmt}", verbose)

        # ---------------- Steps 2-5: the proof loop ----------------
        # prover -> verifiers -> reviser, repeated within the iteration. The
        # prover writes a proof of `target`; the verifiers then check it —
        # three atomic passes, one per verifier agent — and one reject ends the
        # counting and sends the proof to the reviser with the verdict's
        # reasoning. The reviser then either keeps the statement (the prover
        # re-tries with the verdict as feedback), revises it (the revised
        # statement becomes the new target), or decomposes it into a smaller
        # lemma (judged too hard as stated from the last proof; the prover
        # restarts on the new one with a fresh budget). Only a proof that
        # survives all
        # three verifiers enters the DAG; after MAX_PROOF_ATTEMPTS prover
        # rounds the lemma is given up for this iteration and the planner is
        # asked again.
        #
        # No human intervention here, in either mode. Choosing 'p' at the menu
        # *is* the decision to let the models settle it; asking again
        # afterwards would put the operator back in the loop they just
        # delegated out of. If you want a say over this lemma, assert it.
        # The shared engine (run_proof_loop, above) owns steps 2-5; the
        # serial run hands it the serial versions of the round and success
        # boundaries: publish the in-flight state into _LIVE_STATE and the
        # checkpoint, and commit proved lemmas to the DAG. The lock is a
        # formality here — one thread — and the engine's contract does not
        # change between the two modes.
        serial_failed_lock = threading.Lock()
        result = run_proof_loop(
            verbose=verbose,
            conjecture=conjecture,
            prover_sys=prover_sys,
            reviser_sys=reviser_sys,
            reference_block=reference_block,
            reviser_ref_block=reviser_ref_block,
            references=references,
            verifier_sys=verifier_sys,
            ref_ids=ref_ids,
            initial=in_flight if resuming else None,
            fresh_target=None if resuming else next_lemma,
            on_round=lambda state: _serial_on_round(
                iteration, failed_attempts, state, verbose
            ),
            on_proof=lambda lid, node: _serial_on_proof(lid, node, verbose),
            failed_attempts=failed_attempts,
            failed_lock=serial_failed_lock,
            get_dag=load_dag,
            toggle_hotkey=lambda: check_mode_toggle(verbose),
        )
        lemma_id = result["lemma_id"]
        proved = result["proved"]

        # ---------------- Step 6: Record the outcome for the planner --------
        if not proved:
            if result["attempt_notes"]:
                failed_attempts[lemma_id] = result["attempt_notes"]
            log(
                f"↷ Lemma {lemma_id} left unproved after the proof loop; the "
                f"planner will see the feedback.",
                verbose,
            )

        # The lemma is settled one way or another: clear the in-flight state
        # and move the resumable position to the next iteration, with the
        # reject list as it now stands.
        in_flight = None
        _LIVE_STATE["in_flight"] = None
        save_checkpoint(iteration + 1, failed_attempts, None, verbose)

    log(
        f"\n⏹ Reached MAX_ITERATIONS ({MAX_ITERATIONS}) without settling the "
        f"conjecture (proved or disproved). The checkpoint keeps the budget "
        f"position: re-run with a larger --max-iterations to continue, or "
        f"--fresh to restart it. The DAG is kept either way.",
        verbose,
    )
    return load_dag()


# ----------------------------------------------------------------------------
# Parallel mode
#
# N proof loops share one DAG, one version-2 checkpoint, and one
# possible-lemma buffer, steered by two extra threads:
#
#   planner    re-plans the shared strategy whenever the DAG changes
#   generator  keeps the buffer stocked with fresh candidate lemmas
#   loop-N     pulls a lemma from the buffer, runs the shared proof engine
#              (run_proof_loop) on it, records the outcome, repeats
#
# When --mode human, loop 0 is the operator's loop: each fresh start goes
# to the menu with the buffer as its candidate list — the parallel version
# of the serial shortlist menu — and the hotkey toggles it between the menu
# and automation exactly as in the serial run. Every write to the shared
# state goes through ParallelState, and every write to the DAG goes through
# the DAG lock with a fresh read, so a lemma can never be added twice and a
# reader can never see a half-state.
# ----------------------------------------------------------------------------

_QUIT = object()  # sentinel: the human loop asked the run to stop


class LemmaBuffer:
    """The shared possible-lemma list: exactly N+5 fixed slots.

    The generator fills the empty slots; the proof loops claim them. The
    slots are the flow control — a full buffer means the loops are behind,
    and the generator waits for a slot instead of piling up work. Claiming
    is atomic: a selector that picked a lemma another loop has taken in the
    meantime is re-run on what remains, so two loops never work on one id.
    """

    def __init__(self, size: int) -> None:
        # Each slot is {"lemma": candidate, "claimed_by": loop id or None},
        # or None while the slot is empty.
        self.slots: List[Optional[Dict[str, Any]]] = [None] * size
        self.cond = threading.Condition()

    def __len__(self) -> int:
        with self.cond:
            return len(self.slots)

    def snapshot(self) -> List[Dict[str, Any]]:
        """The current contents in slot order — what the menu renders and
        the generator's prompt carries."""
        with self.cond:
            return [
                {"lemma": e["lemma"], "claimed_by": e["claimed_by"]}
                for e in self.slots
                if e is not None
            ]

    def _unclaimed_locked(self) -> List[Dict[str, Any]]:
        # Call with self.cond held — the lock is not reentrant, so the
        # public unclaimed() below is the only way in from the outside.
        return [
            e["lemma"]
            for e in self.slots
            if e is not None and e["claimed_by"] is None
        ]

    def unclaimed(self) -> List[Dict[str, Any]]:
        with self.cond:
            return self._unclaimed_locked()

    def has_room(self) -> bool:
        with self.cond:
            return any(e is None for e in self.slots)

    def fill(self, candidates: List[Dict[str, Any]]) -> int:
        """Put fresh candidates into the empty slots, first come first
        served, skipping any whose id already sits in the buffer. Returns
        how many landed."""
        added = 0
        with self.cond:
            for cand in candidates:
                if any(
                    e is not None and e["lemma"]["id"] == cand["id"]
                    for e in self.slots
                ):
                    continue
                for i, e in enumerate(self.slots):
                    if e is None:
                        self.slots[i] = {"lemma": cand, "claimed_by": None}
                        added += 1
                        break
            if added:
                self.cond.notify_all()
        return added

    def claim(self, loop_id: str, selector) -> Optional[Dict[str, Any]]:
        """The atomic pick: snapshot the unclaimed lemmas, let the selector
        choose outside the lock (its LLM call is never made under it), then
        commit under the lock — and if the choice was claimed in the
        meantime, re-run the selector on what remains. Returns the claimed
        candidate, or None when the buffer runs out under the feet."""
        while True:
            with self.cond:
                usable = self._unclaimed_locked()
                if not usable:
                    return None
            chosen = selector(usable)
            if chosen is None:
                return None
            with self.cond:
                for e in self.slots:
                    if (
                        e is not None
                        and e["lemma"]["id"] == chosen["id"]
                        and e["claimed_by"] is None
                    ):
                        e["claimed_by"] = loop_id
                        self.cond.notify_all()
                        return e["lemma"]
                # Stolen between snapshot and commit: go around again.

    def claim_id(self, loop_id: str, lemma_id: str) -> Optional[Dict[str, Any]]:
        """Claim a named lemma — the human loop's pick. Same atomicity as
        claim(): None when the id is not in the buffer unclaimed."""
        with self.cond:
            for e in self.slots:
                if (
                    e is not None
                    and e["lemma"]["id"] == lemma_id
                    and e["claimed_by"] is None
                ):
                    e["claimed_by"] = loop_id
                    self.cond.notify_all()
                    return e["lemma"]
        return None

    def consume(self, lemma_id: str) -> None:
        """Free the slot a settled lemma occupied. Proved, disproved by
        decomposition, or abandoned, the lemma has left the list either
        way; the id's territory is the generator's to offer again.
        No-op when the id is not in the buffer (a written lemma)."""
        with self.cond:
            for i, e in enumerate(self.slots):
                if e is not None and e["lemma"]["id"] == lemma_id:
                    self.slots[i] = None
                    self.cond.notify_all()
                    return

    def wait_for(self, predicate, wake: threading.Event) -> bool:
        """Sleep while predicate() is false. True when it holds, False
        when `wake` (a stop) fires first."""
        with self.cond:
            while not predicate():
                if wake.is_set():
                    return False
                self.cond.wait()
            return True


class ParallelState:
    """Everything the parallel threads share, plus the checkpoint writer.

    One lock per concern, never held across an LLM call:
      state_lock    the per-loop slots, the plan, the DAG version
      failed_lock   the shared reject list
      DAG_LOCK      the DAG file itself (module global)
      CHECKPOINT_LOCK  the checkpoint's snapshot-then-replace (module global)
    The checkpoint is a whole-file rewrite taken under CHECKPOINT_LOCK —
    the loop slots under state_lock, the reject list under failed_lock,
    then one atomic replace over a per-writer temp file — so the file is
    always a consistent, current state even though many threads poke at
    it.
    """

    def __init__(self, loop_ids: List[str], max_iterations: int) -> None:
        self.buffer = LemmaBuffer(len(loop_ids) + 5)
        self.loops: Dict[str, Dict[str, Any]] = {
            lid: {"in_flight": None, "iterations_used": 0} for lid in loop_ids
        }
        self.max_iterations = max_iterations
        self.plan: Optional[Dict[str, Any]] = None
        self.plan_stale = False
        self.plan_cond = threading.Condition()
        self.dag_version = 0
        self.failed_attempts: Dict[str, List[str]] = {}
        self.failed_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.resolution: Optional[str] = None
        self.state_lock = threading.Lock()
        # How many proof loops are still inside their loops. The stop the
        # planner and generator wait on may only fire when this reaches zero:
        # a loop that has spent its budget may still be running the engine on
        # its final claim, and stopping it mid-engine would orphan a lemma
        # that was claimed, budgeted, and never proved.
        self.live_loops = len(loop_ids)

    # -- run control ------------------------------------------------------

    def running(self) -> bool:
        return not self.stop_event.is_set()

    def note_loop_exit(self) -> bool:
        """A proof loop left its loop. True if it was the last one out."""
        with self.state_lock:
            self.live_loops = max(0, self.live_loops - 1)
            return self.live_loops == 0

    def request_stop(self) -> None:
        self._stop_all()

    def set_resolution(self, kind: str) -> None:
        with self.state_lock:
            if self.resolution is not None:
                return
            self.resolution = kind
        self._stop_all()

    def _stop_all(self) -> None:
        self.stop_event.set()
        with self.buffer.cond:
            self.buffer.cond.notify_all()
        with self.plan_cond:
            self.plan_cond.notify_all()

    # -- the plan ----------------------------------------------------------

    def wait_for_work(self) -> bool:
        """Planner: sleep until there is no plan, or a stale one."""
        with self.plan_cond:
            while self.plan is not None and not self.plan_stale:
                if self.stop_event.is_set():
                    return False
                self.plan_cond.wait()
            return True

    def wait_for_plan(self) -> bool:
        """Generator: sleep until some plan exists. A stale one still
        steers the generator — it keeps the buffer stocked while the
        planner catches up — so only the first plan is waited for."""
        with self.plan_cond:
            while self.plan is None:
                if self.stop_event.is_set():
                    return False
                self.plan_cond.wait()
            return True

    def install_plan(
        self, plan: Dict[str, Any], dag_version_before: int
    ) -> bool:
        """Install a plan. Returns True when the DAG moved while it was
        being written: the plan is installed already marked stale, so the
        planner goes straight back around — a plan is never left steering
        a DAG it was not written for."""
        with self.state_lock:
            self.plan = {
                "plan_summary": str(plan.get("plan_summary") or "").strip(),
                "priorities": [
                    p
                    for p in (plan.get("priorities") or [])
                    if isinstance(p, dict) and str(p.get("id") or "").strip()
                ],
            }
            moved = self.dag_version > dag_version_before
            self.plan_stale = moved
        with self.plan_cond:
            self.plan_cond.notify_all()
        return moved

    def mark_plan_stale(self) -> None:
        """The operator's 'replan' and the DAG changes both land here."""
        with self.plan_cond:
            self.plan_stale = True
            self.plan_cond.notify_all()

    def plan_snapshot(self) -> Optional[Dict[str, Any]]:
        with self.state_lock:
            if self.plan is None:
                return None
            return {
                "plan_summary": self.plan["plan_summary"],
                "priorities": list(self.plan["priorities"]),
            }

    # -- per-loop slots -----------------------------------------------------

    def iterations_used(self, loop_id: str) -> int:
        with self.state_lock:
            return self.loops[loop_id]["iterations_used"]

    def begin_lemma(self, loop_id: str) -> None:
        """A fresh lemma starts: spend one of the loop's iterations and
        checkpoint the spend, so a cancelled run does not re-spend it."""
        with self.state_lock:
            self.loops[loop_id]["iterations_used"] += 1
        self.write_checkpoint()

    def set_in_flight(
        self, loop_id: str, state: Optional[Dict[str, Any]]
    ) -> None:
        """The engine's round boundary: publish this loop's resumable
        position (None when its lemma is settled)."""
        with self.state_lock:
            self.loops[loop_id]["in_flight"] = state
        self.write_checkpoint()

    def take_in_flight(self, loop_id: str) -> Optional[Dict[str, Any]]:
        """This loop's pending round at start-of-run.

        Deliberately not cleared: the engine's first on_round replaces the
        published state, and until then the position stays visible — the
        generator's duplicate check sees the lemma is taken, and a crash
        before the first on_round leaves the old round to re-take, losing
        nothing. The budget was already spent when the lemma was claimed,
        so re-taking it never spends an iteration twice."""
        with self.state_lock:
            pending = self.loops[loop_id]["in_flight"]
            return dict(pending) if pending is not None else None

    def in_flight_ids(self) -> Set[str]:
        """The lemmas the loops are holding in flight right now: a fresh
        run's buffer starts empty, so without this the generator could
        re-offer a lemma whose loop is about to resume it, and two loops
        could then prove the same lemma in parallel."""
        with self.state_lock:
            return {
                d["in_flight"]["lemma_id"]
                for d in self.loops.values()
                if d.get("in_flight") is not None
            }

    # -- the DAG -------------------------------------------------------------

    def add_lemma(self, lemma_id: str, node: Dict[str, Any]) -> None:
        """Commit a lemma to the shared DAG — the atomic part of the
        design. Read the DAG fresh under the DAG lock, add the lemma,
        write it back whole; then bump the version the planner watches,
        clear the lemma's reject notes, mark the plan stale (the DAG
        changed, so the plan owes a re-read), and checkpoint. The version
        is visible only after the file is down, so a replan triggered by
        it always sees the new lemma."""
        with DAG_LOCK:
            dag = load_dag()
            dag["lemmas"][lemma_id] = node
            save_dag(dag)
        with self.state_lock:
            self.dag_version += 1
        with self.failed_lock:
            self.failed_attempts.pop(lemma_id, None)
        self.mark_plan_stale()
        self.write_checkpoint()

    def dag_version_snapshot(self) -> int:
        with self.state_lock:
            return self.dag_version

    # -- the shared reject list -----------------------------------------------

    def record_failure(self, lemma_id: str, notes: List[str]) -> None:
        with self.failed_lock:
            self.failed_attempts[lemma_id] = list(notes)
        self.write_checkpoint()

    def failed_snapshot(self) -> Dict[str, List[str]]:
        with self.failed_lock:
            return {k: list(v) for k, v in self.failed_attempts.items()}

    # -- checkpoint -------------------------------------------------------------

    def apply_checkpoint(self, cp: Dict[str, Any]) -> None:
        with self.state_lock:
            for lid, slot in self.loops.items():
                entry = cp["loops"].get(lid) or {}
                slot["in_flight"] = entry.get("in_flight")
                slot["iterations_used"] = int(entry.get("iterations_used") or 0)
        with self.failed_lock:
            self.failed_attempts = {
                str(k): [n for n in v if isinstance(n, str)]
                for k, v in (cp.get("failed_attempts") or {}).items()
                if isinstance(v, list)
            }

    def write_checkpoint(self) -> None:
        # CHECKPOINT_LOCK spans the snapshot and the replace: with the two
        # separated, a writer that snapshots first could replace last and
        # land an older state over a newer one's.
        with CHECKPOINT_LOCK:
            with self.state_lock:
                loops = {
                    lid: {
                        "in_flight": slot["in_flight"],
                        "iterations_used": slot["iterations_used"],
                    }
                    for lid, slot in self.loops.items()
                }
            failed = self.failed_snapshot()
            _atomic_write_checkpoint_file(
                {"version": 2, "loops": loops, "failed_attempts": failed},
                verbose=False,
            )


def _clean_candidate(raw: Any) -> Optional[Dict[str, Any]]:
    """A generated or written lemma entry, validated the way the planner's
    candidates are: an id and a non-empty statement. Anything else is
    dropped, not repaired — the prompt tells the generator the shape, and a
    dropped entry costs one generator round, not a proof."""
    if not isinstance(raw, dict):
        return None
    lemma_id = str(raw.get("id") or "").strip()
    statement = str(raw.get("statement") or "").strip()
    if not lemma_id or not statement:
        return None
    return {"id": lemma_id, "statement": statement}


def _parallel_planner(
    state: ParallelState,
    verbose: bool,
    planner_sys: str,
    conjecture: str,
    comments_block: str,
    ref_block: str,
) -> None:
    """The steering planner: keeps the shared strategy current.

    Runs once at start (no plan yet), then whenever the DAG changes — a
    lemma added by any loop lands in add_lemma, which marks the plan
    stale — or the operator asks for a re-plan. If the DAG moves again
    during the planning call itself, the plan is installed already marked
    stale and the planner goes straight back around: no DAG change is
    lost, and the plan that steers the loops was always written for the
    DAG as it stood."""
    while state.running():
        if not state.wait_for_work():
            return
        if not state.running():
            return
        dag = load_dag()
        dag_version_before = state.dag_version_snapshot()
        plan = state.plan_snapshot()
        plan_user = (
            f"Conjecture:\n{conjecture}\n\n"
            f"Proved lemmas so far:\n{json.dumps(planner_dag_view(dag), indent=2)}"
            f"{ref_block}{comments_block}\n\n"
            f"Rejected attempts so far, keyed by lemma id (feedback from the "
            f"loops' prover, verifier and reviser rounds; routes already "
            f"found wanting):\n{json.dumps(state.failed_snapshot(), indent=2)}\n\n"
            f"Current plan, if any (revise it against the DAG: keep what "
            f"still holds, drop what is done or dead, re-order what moved):\n"
            + (json.dumps(plan, indent=2) if plan else "(none yet)")
            + "\n"
        )
        text, _status, _partial = reason(
            planner_sys, plan_user, "parallel_planner",
            THINK["parallel_planner"], verbose,
        )
        check_mode_toggle(verbose)
        plan_res = parse_json_or_none(text) if text else None
        if plan_res is None and text:
            plan_res = extract(
                "Extract the plan. The two flags are booleans; plan_summary "
                "is a short string; priorities is a list of objects with id "
                "and reason.",
                text,
                PARALLEL_PLANNER_SCHEMA,
                "parallel_planner",
                verbose,
            )
        if not plan_res or not isinstance(plan_res, dict):
            log(
                "⚠️  Parallel planner gave no usable plan; retrying in 10s.",
                verbose,
            )
            if state.stop_event.wait(10):
                return
            continue
        moved = state.install_plan(plan_res, dag_version_before)
        if bool(plan_res.get("is_conjecture_proved")) or bool(
            plan_res.get("is_conjecture_disproved")
        ):
            kind = (
                "proved"
                if plan_res.get("is_conjecture_proved")
                else "disproved"
            )
            log(
                f"\n🏁 Planner judges the conjecture {kind}; the loops are "
                f"being stopped.",
                verbose,
            )
            state.set_resolution(kind)
            return
        log(
            "🧠 Planner set the strategy"
            + (" — the DAG moved while it was writing, re-planning now."
               if moved else "."),
            verbose,
        )


def _parallel_generator(
    state: ParallelState,
    verbose: bool,
    generator_sys: str,
    conjecture: str,
    ref_block: str,
) -> None:
    """Keeps the possible-lemma list stocked.

    Each round it asks the generator for five candidate lemmas and fills
    the buffer's empty slots with the ones that are new: an id already in
    the DAG, the buffer, or the reject list is dropped, so no id is ever
    claimed twice. The buffer's fixed N+5 slots are the flow control — a
    full buffer means the loops are behind, and the generator waits for a
    slot instead of piling up work."""
    while state.running():
        if not state.wait_for_plan():
            return
        if not state.running():
            return
        # Wait for a free slot first: the buffer is the queue and the
        # generator its only writer.
        if not state.buffer.wait_for(state.buffer.has_room, state.stop_event):
            return
        plan = state.plan_snapshot() or {}
        dag = load_dag()
        buffer_snapshot = state.buffer.snapshot()
        # Ids the new batch must not repeat: proved, buffered (claimed or
        # not), or already rejected.
        known = set(dag["lemmas"].keys())
        known |= {e["lemma"]["id"] for e in buffer_snapshot}
        known |= set(state.failed_snapshot().keys())
        # A lemma a loop holds in flight is off-limits too: the run
        # resumed it, and it is not in this run's buffer.
        known |= state.in_flight_ids()
        gen_user = (
            f"Conjecture:\n{conjecture}\n\n"
            f"Current plan (follow its priorities; these are the lemmas the "
            f"loops are working toward):\n{json.dumps(plan, indent=2)}\n\n"
            f"Proved lemmas so far (already in the DAG; never restate "
            f"them):\n{json.dumps(planner_dag_view(dag), indent=2)}\n\n"
            f"Possible lemmas currently in the buffer (claimed_by marks the "
            f"loop that has picked one up; never duplicate any of "
            f"these):\n{json.dumps(buffer_snapshot, indent=2)}\n\n"
            f"Rejected attempts so far, keyed by lemma id (statements and "
            f"routes already found wanting):\n"
            f"{json.dumps(state.failed_snapshot(), indent=2)}\n"
        )
        text, _status, _partial = reason(
            generator_sys, gen_user, "parallel_lemma_generator",
            THINK["parallel_lemma_generator"], verbose,
        )
        check_mode_toggle(verbose)
        batch = parse_json_or_none(text) if text else None
        if batch is None and text:
            batch = extract(
                "Extract the candidate lemmas: an object with a "
                "candidate_lemmas list; each lemma has an id and a "
                "statement.",
                text,
                PARALLEL_LEMMA_GENERATOR_SCHEMA,
                "parallel_lemma_generator",
                verbose,
            )
        lemmas = (batch or {}).get("candidate_lemmas") or []
        fresh: List[Dict[str, Any]] = []
        for raw in lemmas:
            cand = _clean_candidate(raw)
            if cand is None or cand["id"] in known:
                continue
            known.add(cand["id"])
            fresh.append(cand)
        added = state.buffer.fill(fresh)
        log(
            f"🧪 Lemma generator offered {len(fresh)} new candidate(s); "
            f"{added} entered the buffer "
            f"({len(state.buffer.snapshot())}/{len(state.buffer)} slots full).",
            verbose,
        )
        if added == 0:
            # Nothing landed (no offer, or the buffer filled under the feet):
            # back off so a stalled planner and a stalled generator cannot
            # spin together.
            if state.stop_event.wait(10):
                return


def _auto_claim(
    state: ParallelState,
    loop_id: str,
    verbose: bool,
    conjecture: str,
) -> Optional[Dict[str, Any]]:
    """The unattended claim: snapshot the unclaimed buffer, let the
    selector agent weigh them, commit atomically. None when the buffer
    holds nothing the loops have not already taken."""

    def pick(usable: List[Dict[str, Any]]) -> Dict[str, Any]:
        dag = load_dag()
        plan = state.plan_snapshot() or {}
        failed = state.failed_snapshot()
        return select_lemma(
            usable,
            dag,
            conjecture,
            str(plan.get("plan_summary") or ""),
            failed,
            verbose,
            proposer="the lemma generator",
        )

    return state.buffer.claim(loop_id, pick)


def _parallel_human_step(
    state: ParallelState,
    loop_id: str,
    verbose: bool,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """One human planning step: the buffer on the menu, as the serial run
    puts the planner's shortlist. Loops until the operator commits to a
    lemma, hands the loop to automation, or quits — a quit stops the whole
    run, not just this loop, and says so. Re-plans and asserts are
    recorded and the menu comes right back, the serial way."""
    global MODE
    while True:
        dag = load_dag()
        snapshot = state.buffer.snapshot()
        screened = [
            (
                entry["lemma"],
                [f"claimed by {entry['claimed_by']}"]
                if entry["claimed_by"]
                else [],
            )
            for entry in snapshot
        ]
        plan = state.plan_snapshot() or {}
        summary = str(plan.get("plan_summary") or "")
        choice = interaction.choose(
            screened, dag["lemmas"].keys(), HOTKEY, summary
        )
        if choice.action == "quit":
            log(
                "\n⏹ Stopped by the operator; the DAG and the checkpoint "
                "are kept — re-run the same command to resume.",
                verbose,
            )
            state.request_stop()
            return "quit", None
        if choice.action == "replan":
            with state.failed_lock:
                for cand, _problems in screened:
                    for note in choice.notes:
                        state.failed_attempts.setdefault(cand["id"], []).append(
                            note
                        )
            state.mark_plan_stale()
            state.write_checkpoint()
            log("↩︎ Buffer rejected; the planner is asked again.", verbose)
            continue
        if choice.action == "auto":
            MODE = "auto"
            log(f"▶ {loop_id} handed to automation. {HOTKEY.hint()}", verbose)
            return "auto", None
        if choice.action == "assert":
            asserted = choice.lemma or {}
            state.add_lemma(
                asserted["id"],
                {
                    "statement": asserted["statement"],
                    "proof": "Asserted by the operator; not machine-proved.",
                    "dependencies": [],
                    "provenance": "operator",
                },
            )
            state.buffer.consume(asserted["id"])
            log(
                f"🖊  Lemma {asserted['id']} accepted on your authority and "
                f"added to the DAG unproved.",
                verbose,
            )
            continue
        # 'prove': a buffer lemma, claimed atomically, or a written one.
        lemma = choice.lemma or {}
        lemma_id = str(lemma.get("id") or "").strip()
        if lemma_id in {e["lemma"]["id"] for e in snapshot}:
            claimed = state.buffer.claim_id(loop_id, lemma_id)
            if claimed is None:
                log(
                    f"  {lemma_id} was claimed in the meantime; pick another.",
                    verbose,
                )
                continue
            return "prove", claimed
        cand = _clean_candidate(lemma)
        if cand is None:
            log(
                "  That written lemma has no usable id or statement; "
                "pick another.",
                verbose,
            )
            continue
        return "prove", cand


def _parallel_proof_loop(
    state: ParallelState,
    loop_id: str,
    loop_index: int,
    verbose: bool,
    conjecture: str,
    prover_sys: str,
    reviser_sys: str,
    reference_block: str,
    reviser_ref_block: str,
    references: List[Dict[str, Any]],
    verifier_sys: Dict[str, str],
    ref_ids: Set[str],
) -> None:
    """One of the N proof loops: pull a lemma from the buffer, run the
    shared proof engine on it, record the outcome, repeat — until the loop
    spends its iteration budget, the run resolves, or it is stopped.

    Loop 0 is the operator's while --mode human: each fresh start goes to
    the menu first (the buffer is the shortlist), and the hotkey still
    toggles it between the menu and automation. The engine's callbacks are
    the loop's share of the shared state: on_round publishes the resumable
    position, on_proof commits a lemma to the DAG, and the loop's failure
    notes join the shared reject list under its lock."""
    is_human = lambda: loop_index == 0 and MODE == "human"
    log(
        f"🚀 {loop_id} starts"
        + (" (the human loop — it asks before each lemma)."
           if is_human() else " (automatic)."),
        verbose,
    )
    while state.running():
        pending = state.take_in_flight(loop_id)
        if pending is not None:
            # A prover round a previous run cancelled mid-flight: re-run
            # exactly that round. It does not spend a new iteration — the
            # one it started under was already spent — so the resume is
            # checked before the budget: a loop that exhausted its budget
            # still finishes the lemma it died holding.
            dag_now = load_dag()
            if pending["lemma_id"] in dag_now.get("lemmas", {}):
                # The stop landed after the lemma was proved: on_proof had
                # already committed it. The serial run's same case — nothing
                # to resume; drop the position and the buffer claim, and go
                # about the next lemma (the spent iteration is spent).
                state.set_in_flight(loop_id, None)
                state.buffer.consume(pending["lemma_id"])
                log(
                    f"\u267b {loop_id}'s in-flight {pending['lemma_id']} is "
                    f"already in the DAG; resuming is moot.",
                    verbose,
                )
                continue
            initial = pending
            fresh_target = None
            log(
                f"♻ {loop_id} resumes {pending['lemma_id']} at prover round "
                f"{pending['attempt']}/{MAX_PROOF_ATTEMPTS}.",
                verbose,
            )
        else:
            if state.iterations_used(loop_id) >= state.max_iterations:
                log(
                    f"⏹ {loop_id} has spent its budget "
                    f"({state.max_iterations}/{state.max_iterations}).",
                    verbose,
                )
                # The stop that frees the planner and generator is sent from
                # the loop exit below, not here: this loop has spent its
                # budget, but a sibling may still be proving the lemma it
                # claimed on its last iteration, and a stop now would kill
                # that engine before its lemma was ever proved.
                break
            if is_human():
                action, lemma = _parallel_human_step(state, loop_id, verbose)
                if action == "quit":
                    # The operator ended the run, the serial way: stop every
                    # loop (their in-flight lemmas stay in the checkpoint),
                    # not just this one.
                    state.request_stop()
                    break
                if action == "prove":
                    fresh_target = lemma
                else:  # 'auto': the operator delegated the pick.
                    fresh_target = _auto_claim(
                        state, loop_id, verbose, conjecture
                    )
            else:
                fresh_target = _auto_claim(
                    state, loop_id, verbose, conjecture
                )
            if fresh_target is None:
                # Nothing unclaimed in the buffer: the generator is behind.
                # Wait for it to fill a slot (or the run to stop).
                if not state.buffer.wait_for(
                    state.buffer.unclaimed, state.stop_event
                ):
                    break
                continue
            initial = None
            # The iteration is spent the moment the lemma is claimed, the
            # serial way: a run cancelled mid-lemma resumes it without
            # charging the budget again.
            state.begin_lemma(loop_id)
        result = run_proof_loop(
            verbose=verbose,
            conjecture=conjecture,
            prover_sys=prover_sys,
            reviser_sys=reviser_sys,
            reference_block=reference_block,
            reviser_ref_block=reviser_ref_block,
            references=references,
            verifier_sys=verifier_sys,
            ref_ids=ref_ids,
            initial=initial,
            fresh_target=fresh_target,
            on_round=lambda state_dict: state.set_in_flight(
                loop_id, state_dict
            ),
            on_proof=lambda lid, node: state.add_lemma(lid, node),
            failed_attempts=state.failed_attempts,
            failed_lock=state.failed_lock,
            get_dag=load_dag,
            toggle_hotkey=lambda: check_mode_toggle(verbose),
            should_stop=lambda: not state.running(),
        )
        if result["stopped"]:
            # A stop landed mid-round: the engine's last on_round already
            # published the resumable position, and the buffer claim is
            # kept — the lemma is still this loop's to finish.
            break
        # The lemma is settled one way or another: clear the resumable
        # position (the engine's last on_round may still hold it), free the
        # buffer slot it occupied, and record the failure where the planner
        # and the other loops can see it.
        state.set_in_flight(loop_id, None)
        state.buffer.consume(result["lemma_id"])
        if not result["proved"]:
            if result["attempt_notes"]:
                state.record_failure(result["lemma_id"], result["attempt_notes"])
        # A proved lemma is already in the DAG (on_proof committed it, which
        # bumped the version and marked the plan stale): the loop simply
        # goes for its next lemma.
        if not state.running():
            break
    # Every exit path (budget spent, human quit, a stop, the buffer waiting
    # giving up) funnels here: announce this loop is out, and if it was the
    # last, send the stop that frees the planner and generator — nothing
    # more will ever be claimed or added, so there is no point left in
    # their waiting.
    if state.note_loop_exit():
        state.request_stop()


def run_parallel(parallel_loops: int, verbose: bool = True) -> Dict[str, Any]:
    """The parallel entry point: N proof loops around one shared DAG.

    The loops share run_proof_loop() with the serial run; what is new here
    is the shared state (ParallelState), the buffer that feeds the loops,
    and the two threads that steer them — the planner, which re-plans
    whenever the DAG changes, and the generator, which keeps the
    possible-lemma list stocked. Loop 0 is the operator's while --mode
    human. Returns the DAG, as run_loop does."""
    global _PARALLEL

    log(
        f"\n🧵 Parallel mode: {parallel_loops} proof loop(s), one planner, "
        f"one lemma generator, one DAG.",
        verbose,
    )

    conjecture = load_file(CONJECTURE_FILE)
    if not conjecture:
        log(
            f"{CONJECTURE_FILE} is empty — write your conjecture there "
            f"first.",
            verbose,
        )
        return load_dag()

    planner_sys = load_file(PROMPT_PATHS["parallel_planner.md"])
    generator_sys = load_file(PROMPT_PATHS["parallel_lemma_generator.md"])
    prover_sys = load_file(PROMPT_PATHS["prover.md"])
    # One prompt per verification step; VERIFIER_AGENTS names the three.
    verifier_sys = {name: load_file(PROMPT_PATHS[name]) for name in VERIFIER_AGENTS}
    reviser_sys = load_file(PROMPT_PATHS["reviser.md"])
    empty = [
        name
        for name, text in (
            ("parallel_planner.md", planner_sys),
            ("parallel_lemma_generator.md", generator_sys),
            ("prover.md", prover_sys),
            *verifier_sys.items(),
            ("reviser.md", reviser_sys),
        )
        if not text
    ]
    if empty:
        log(f"Missing or empty prompt files: {', '.join(empty)}. Aborting.", verbose)
        return load_dag()

    references, reference_block, planner_ref_block, reviser_ref_block = (
        _build_reference_blocks(verbose)
    )
    # The planner and the generator steer by the short view (the same one
    # the serial planner gets — empty at or above the limit, by the same
    # rule); the prover in each loop gets the full collection through
    # reference_block.
    ref_block = planner_ref_block
    ref_ids = {str(r["id"]) for r in references}
    comments_block = _build_comments_block()

    loop_ids = [f"loop-{i}" for i in range(parallel_loops)]
    state = ParallelState(loop_ids, MAX_ITERATIONS)
    cp = load_parallel_checkpoint(loop_ids, verbose)
    if cp is not None and not os.path.exists(DAG_FILE):
        # The serial run's rule, kept: deleting dag.json is how the operator
        # resets a conjecture, and a checkpoint outliving its DAG would
        # resurrect the old budget and in-flight lemma.
        log("\u26a0\ufe0f  Checkpoint without a DAG file; discarding it.", verbose)
        discard_checkpoint()
        cp = None
    if cp is not None:
        state.apply_checkpoint(cp)
        pending = sum(
            1
            for lid in loop_ids
            if (cp["loops"].get(lid) or {}).get("in_flight") is not None
        )
        used = sum(
            (cp["loops"].get(lid) or {}).get("iterations_used", 0)
            for lid in loop_ids
        )
        log(
            f"♻ Resumed the parallel checkpoint: {used} lemma(s) spent of "
            f"{MAX_ITERATIONS} each, {pending} prover round(s) in flight.",
            verbose,
        )
    state.write_checkpoint()
    # A server context 400 (a mis-sized --num-ctx) would otherwise kill
    # one thread with a traceback while its siblings keep 400-storming;
    # these wrappers turn it into the one loud message plus a stop of the
    # shared state, so the run ends cleanly with its checkpoint intact.
    def _planner_thread() -> None:
        try:
            _parallel_planner(state, verbose, planner_sys, conjecture,
                              comments_block, ref_block)
        except llm_backend.ContextLengthError as e:
            _fail_context_length(e, stop=state.request_stop)

    def _generator_thread() -> None:
        try:
            _parallel_generator(state, verbose, generator_sys, conjecture,
                                ref_block)
        except llm_backend.ContextLengthError as e:
            _fail_context_length(e, stop=state.request_stop)

    def _proof_thread(i: int, lid: str) -> None:
        try:
            _parallel_proof_loop(
                state, lid, i, verbose, conjecture, prover_sys, reviser_sys,
                reference_block, reviser_ref_block, references, verifier_sys,
                ref_ids,
            )
        except llm_backend.ContextLengthError as e:
            _fail_context_length(e, stop=state.request_stop)

    threads = [
        threading.Thread(
            target=_planner_thread,
            name="parallel-planner",
            daemon=True,
        ),
        threading.Thread(
            target=_generator_thread,
            name="parallel-lemma-generator",
            daemon=True,
        ),
    ]
    for i, lid in enumerate(loop_ids):
        threads.append(
            threading.Thread(
                target=_proof_thread,
                args=(i, lid),
                name=f"parallel-{lid}",
                daemon=True,
            )
        )
    _PARALLEL = state
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        _PARALLEL = None

    if state.resolution is not None:
        with DAG_LOCK:
            dag = load_dag()
            dag["conjecture"] = conjecture
            dag["status"] = state.resolution
            save_dag(dag)
        discard_checkpoint()
        log(
            f"\n🎉 Conjecture has been {state.resolution} — the DAG "
            f"records it and the checkpoint is cleared.",
            verbose,
        )
        return dag
    state.write_checkpoint()
    buffer_snapshot = state.buffer.snapshot()
    log(
        f"\n⏹ Parallel run finished without settling the conjecture: "
        f"the buffer still holds {len(buffer_snapshot)} candidate(s). "
        f"The checkpoint keeps every loop's budget position and in-flight "
        f"round: re-run the same command to continue, or --fresh to "
        f"restart. The DAG is kept either way.",
        verbose,
    )
    return load_dag()


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main() -> None:
    global MODEL_NAME, CONJECTURE_FILE, DAG_FILE, REFERENCES_FILE
    global COMMENTS_FILE
    global MAX_ITERATIONS, NUM_CTX
    global BACKEND, PROFILE, PROMPT_PATHS, MODE, HOTKEY
    global MAX_PROOF_ATTEMPTS, CHECKPOINT_FILE

    parser = argparse.ArgumentParser(
        description="Run the multi-agent theorem prover."
    )
    # Adds --verbose / --no-verbose flags (defaults to True)
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable console logging (default: --verbose)",
    )
    parser.add_argument(
        "--host", default=None,
        help="OpenAI-compatible base URL: a local llama-server, or a hosted "
             "API's base (e.g. https://api.openai.com, without the /v1 "
             "suffix — it is added automatically). Defaults to "
             "$LLAMA_HOST or localhost:8081 (8080 is taken by open-webui).",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="API key sent as 'Authorization: Bearer <key>'. Needed for a "
             "hosted API; defaults to $LLM_API_KEY. Leave unset for a plain "
             "local llama-server (or one launched without --api-key).",
    )
    parser.add_argument(
        "--model", default=MODEL_NAME,
        help="Model name: the --alias llama-server was launched with, or the "
             "provider's model id (e.g. gpt-4o).",
    )
    parser.add_argument(
        "--backend", choices=("auto", "llamacpp", "openai"), default="auto",
        help=(
            "Which dialect of the OpenAI-compatible endpoint to speak. "
            "auto (default): probe /props — an answer means llama.cpp (full "
            "option set, thinking suppressed for schema calls), no answer "
            "means a generic endpoint (standard OpenAI fields). llamacpp / "
            "openai force one side of that decision, for when the probe "
            "can't see the real server (a proxy in front of llama-server, a "
            "provider that happens to serve /props)."
        ),
    )
    parser.add_argument(
        "--conjecture", default=DEFAULT_CONJECTURE,
        help=f"Directory under {workspace.CONJECTURES_ROOT}/ containing "
             f"{workspace.CONJECTURE_FILENAME} (default: {DEFAULT_CONJECTURE})",
    )
    parser.add_argument(
        "--dag", default=None,
        help="Override the DAG path. By default every run of a conjecture "
             "shares conjectures/<name>/dag.json, whatever model wrote it.",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=MAX_ITERATIONS,
        help=f"Loop iterations before giving up (default: {MAX_ITERATIONS})",
    )
    parser.add_argument(
        "--max-proof-attempts", type=int, default=MAX_PROOF_ATTEMPTS,
        help=(
            "How many prover rounds the proof loop may spend on one lemma "
            f"per iteration (default: {MAX_PROOF_ATTEMPTS})"
        ),
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Ignore and delete any existing checkpoint, restarting the "
            "iteration budget from 1. The DAG itself is kept: delete "
            "dag.json for a genuinely clean run."
        ),
    )
    parser.add_argument(
        "--parallel",
        type=int,
        nargs="?",
        const=1,
        default=None,
        metavar="N",
        help=(
            "Run N proof loops in parallel, steered by one planner and one "
            "lemma generator sharing one DAG. The human mode, when set, "
            "takes the place of one of the loops (loop-0). "
            "Bare --parallel is N=1."
        ),
    )
    parser.add_argument(
        "--num-ctx", type=int, default=None,
        help=(
            "Context window in tokens. Defaults to the server's context — "
            "the -c llama-server was launched with, as probed (lower it if "
            "VRAM is tight) — or 65536 on a hosted --host API, where "
            "nothing is probed. An explicit value replaces that default; "
            "against a probed llama-server it is clamped to the ceiling."
        ),
    )
    parser.add_argument(
        "--mode", choices=("auto", "human"), default=DEFAULT_MODE,
        help="auto: the loop picks a lemma from the planner's shortlist and "
             "runs unattended (default). human: the shortlist is put to you at "
             "every planning step. Switchable mid-run either way.",
    )
    parser.add_argument(
        "--hotkey", default=HOTKEY_KEYS,
        help=f"Key that toggles between manual and automatic mid-run "
             f"(default: {HOTKEY_KEYS!r}). Pass '' to disable the listener.",
    )
    args = parser.parse_args()

    MODEL_NAME = args.model
    MAX_ITERATIONS = args.max_iterations
    MAX_PROOF_ATTEMPTS = max(1, args.max_proof_attempts)
    MODE = args.mode

    # Backend first: the probe tells us the real context ceiling, which an
    # explicit --num-ctx is clamped to. Doing this before resolving paths
    # also means a dead server is reported before a missing directory.
    BACKEND = llm_backend.make_backend(
        args.model, args.host, REQUEST_TIMEOUT, api_key=args.api_key,
        backend=args.backend
    )
    PROFILE = BACKEND.probe()
    # A key the API rejected is not a "degrade to defaults" situation: every
    # call in the run would 401. Fail now, before spending a token on it.
    if PROFILE.auth_error:
        parser.error(PROFILE.auth_error)

    # Resolve the context window: an explicit --num-ctx wins. It is clamped
    # only against a *probed* ceiling — the -c a real llama-server reports,
    # which is a fact (the KV cache is fixed at launch). On a non-llama.cpp
    # endpoint the recorded ceiling is a default, not a measurement, so the
    # explicit value replaces it; see the NUM_CTX comment.
    if args.num_ctx is None:
        NUM_CTX = PROFILE.context_limit
    elif PROFILE.probed_llama_cpp:
        NUM_CTX = min(args.num_ctx, PROFILE.context_limit)
    else:
        NUM_CTX = args.num_ctx
    REASONING_OPTIONS["num_ctx"] = NUM_CTX
    EXTRACT_OPTIONS["num_ctx"] = NUM_CTX
    # The banner warns when an *explicit* --num-ctx is clamped; in the auto
    # case the resolution above has already made the two agree.
    log(llm_backend.describe(
        PROFILE, args.num_ctx if args.num_ctx is not None else NUM_CTX
    ), args.verbose)

    try:
        paths = workspace.resolve(args.conjecture, args.dag)
    except workspace.ConjectureNotFound as e:
        parser.error(str(e))
        return  # unreachable; parser.error exits, but keeps type checkers calm

    CONJECTURE_FILE = str(paths.conjecture)
    DAG_FILE = str(paths.dag)
    CHECKPOINT_FILE = checkpoint_path_for(DAG_FILE)
    REFERENCES_FILE = str(paths.references)
    COMMENTS_FILE = str(paths.comments)
    PROMPT_PATHS = {name: str(p) for name, p in paths.prompts.items()}
    log(workspace.describe(paths), args.verbose)
    if args.fresh:
        if os.path.exists(CHECKPOINT_FILE):
            discard_checkpoint()
            log(f"♻ --fresh: deleted {os.path.basename(CHECKPOINT_FILE)}; "
                f"the iteration budget restarts from 1. The DAG is kept.",
                args.verbose)
        else:
            log("♻ --fresh: no checkpoint to delete.", args.verbose)

    # The listener puts the terminal in cbreak mode, so it has to be stopped on
    # every exit path — including a traceback — or the shell you return to has
    # no echo. Hence try/finally rather than a stop() at the end of the run.
    HOTKEY = interaction.HotKey(
        args.hotkey,
        enabled=bool(args.hotkey),
        on_press=lambda: print(
            f"\n⇄ Mode change queued: "
            f"{'automatic' if MODE == 'human' else 'manual'} "
            f"from the next planning step."
        ),
    ).start()
    if MODE == "auto" and HOTKEY.hint():
        log(f"mode=auto {HOTKEY.hint()}", args.verbose)
    elif MODE == "human":
        log("mode=human — you pick the lemma at every planning step.", args.verbose)

    # Ctrl-C writes the checkpoint from the safe-boundary state in
    # _LIVE_STATE and exits 130; a second Ctrl-C force-quits. Installed
    # now, once the paths exist, so the handler can always write.
    signal.signal(signal.SIGINT, _on_sigint)

    try:
        if args.parallel is not None:
            run_parallel(max(1, args.parallel), verbose=args.verbose)
        else:
            run_loop(verbose=args.verbose)
    except llm_backend.ContextLengthError as e:
        _fail_context_length(e)
        sys.exit(1)
    finally:
        HOTKEY.stop()


if __name__ == "__main__":
    main()
