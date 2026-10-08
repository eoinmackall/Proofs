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
The run keeps a checkpoint, `dags/<user_id>_dag.checkpoint.json`, beside
the user's DAG file it is building (it is per user, the way the DAG file
is, and gitignored). It holds what the DAG files do not: the position of
the iteration budget, the planner's reject list (`failed_attempts`), and —
when the run is cancelled in the middle of a proof — the lemma that was in
flight, with the target statement (the reviser may have revised it), the
prover round it was on, the feedback that round was about to be given, and
the last proof of it (the one the verifiers just rejected — the material
the reviser judges its difficulty by when it next sees this lemma).
Checkpoints are written at safe
boundaries — the top of each iteration, the top of every prover round, and
the end of each iteration — and by the SIGINT handler itself on Ctrl-C,
which writes the last safe boundary's state and exits 130; a second Ctrl-C
force-exits. Re-running the same command resumes from the checkpoint:
the budget continues where it stopped, the planner keeps seeing the same
reject notes, and a lemma cancelled mid-proof goes straight back to the
prover for the interrupted round rather than through planning again (which
is what makes a mid-proof Ctrl-C cost the interrupted call, not the whole
iteration). An interrupted call is never restored — that one is re-run —
everything completed before it is not. When the conjecture is settled
(resolution.py) the checkpoint is deleted; `--fresh` deletes it too,
without touching the DAG files.

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
than one: which lemma, and who proves it. A number and Enter means you prove
that candidate yourself: you paste or type your proof and its citations, and
the three verifiers check it exactly as they check the prover's — accepted, it
enters the DAG with you as its last prover; rejected, the objection is shown
and the menu comes back. Nothing enters the DAG on anyone's say-so. A trailing
'p' — "3p" — sends it down the ordinary pipeline instead, the prover writing
the proof. You can also write your own lemma ('w' to prove it yourself, 'wp'
to have it proved), or send the planner back to think again.

You can cross between them mid-run, in either direction, without restarting.
'h' toggles: press it during a planner, prover, verifier or reviser call and
the mode flips at the next planning step, which is the earliest a change of
mode can mean anything. 'a' at the menu does the same thing for the case where the
menu is already up and the listener is therefore paused. See interaction.py
for why one of these is a keystroke and the other is a line of input.

Usage
-----
    proofs run conjectures/algebra_example
    proofs run conjectures/algebra_example --model Qwen3.5-122B-Q4_K_M
    proofs run conjectures/algebra_example --model qwen3.8-27b
    proofs run conjectures/algebra_example --no-verbose --max-iterations 25
    proofs run conjectures/algebra_example --mode human
    proofs parse conjectures/algebra_example
    proofs verify conjectures/algebra_example/dags/alice_dag.json lemma_42 --full
    proofs repair conjectures/algebra_example/refutations/alice--lemma_42--bob--20250102T030405Z.json
    proofs status conjectures/algebra_example/dags/alice_dag.json lemma_42
    proofs export conjectures/algebra_example -o ~/tex
    proofs export conjectures/algebra_example/dags/alice_dag.json lemma_42 -o ~/tex

proofs run calls main() with DIR as --conjecture; verify, repair, prune,
status and export call the *_entry functions below.

proofs parse (parsing.py) is standalone: the conjecture's maintainer points
it at conjectures/<name>/references/ (any .tex or .md file in it), it asks
the model for the theorem-level results, and writes them as a strict-JSON
array to conjectures/<name>/references.md, which the maintainer commits with
the conjecture. The reference collection is a shared, committed artefact:
every other user pulls references.md and never regenerates it, and a result
they need that is missing is requested from the maintainer. main.py only
ever reads that file, and offers those references to the prover (citable by
id), to the verifiers (the formal statements of the cited ones only) and to
the planner and the reviser (an id + slogan shortlist, only while the
collection is small enough to be one).

A conjecture directory may also carry a comments.md beside its
conjecture.md: free-form operator notes on possible approaches to a proof or
a counterexample. The loop reads it once and hands it to the planner verbatim
on every iteration as a "Human comments" section — the planner is told to
treat it as suggestions to weigh (a viable suggested route should get one of
the five candidates), not as instructions. No other agent sees the file, and
a directory without one runs without the section. The file is gitignored and
never committed or shared: each user keeps their own.

Sharing a conjecture
--------------------
Several users work on one conjecture by committing, pushing and pulling its
directory through git; nothing here runs git itself. DAG files are per
user: conjectures/<name>/dags/<user_id>_dag.json holds one user's lemmas
(user_id from the config file, never emitted by a model), and every
command reads every file in dags/ to see the complete DAG while writing new
lemmas to the current user's file only. A lemma is identified by the pair
(user_id, lemma_id); a lemma_id is unique only within one user's file, and
no hashes are stored in the DAG files.

Trust is recorded beside the DAG, never in it. Each acceptance of a lemma
is a certificate in conjectures/<name>/certificates/<verifier>.jsonl, the
file of the user whose run accepted it (certificates.py); each rejection
by proofs verify or proofs repair is a file in conjectures/<name>/
refutations/ (refutations.py). Both carry the lemma's Merkle hash
(merkle.py), so either one lapses on its own when the lemma, or anything it
cites, changes. A lemma without a valid certificate, with a refutation
that still matches it, or citing such a lemma is suspended (suspension.py):
hidden from the planner, the selector and the prover until proofs verify or
proofs repair re-accepts it. The conjecture is settled by an unsuspended
lemma stating it, or its negation, verbatim (resolution.py).
"""

import argparse
import contextlib
import glob
import hashlib
import json
import os
import random
import re
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import requests

try:  # inside the proofs package (installed, or `python -m proofs`)
    from . import (
        certificates, config, export, interaction, llm_backend, locking,
        merkle, refutations, resolution, suspension, workspace,
    )
    from . import tools as tools_mod
except ImportError:  # top-level modules (`python proofs/main.py`)
    import certificates, config, export, interaction, llm_backend, locking
    import merkle, refutations, resolution, suspension, workspace
    import tools as tools_mod

# ----------------------------------------------------------------------------
# Configuration (all overridable from the command line; see main())
# ----------------------------------------------------------------------------
# The default --model, sent as the "model" field of every request. A hosted
# API serves the model it names; a llama-server ignores it and answers with
# whatever it loaded. Certificates and refutations do not record it but
# _recorded_model(), the model that actually did the work.
MODEL_NAME = "qwen3.8-27b"
DEFAULT_CONJECTURE = "algebra_example"

# Set in main(). BACKEND owns the HTTP conversation; PROFILE is what the
# capability probe found (context ceiling, thinking support, tokenizer ratio).
BACKEND: Any = None
PROFILE: Any = None

# Resolved by workspace.resolve(): a prompt may be overridden per conjecture,
# so these are paths rather than bare filenames.
CONJECTURE_FILE = ""
# Set in main(): the conjecture's root, the parent of DAGS_DIR. The
# certificates/ directory with the certificate files lives here too
# (see certificates.py).
CONJECTURE_ROOT = ""
# Set in main(): the user this run writes as, resolved from the config file
# before the paths are (see config.py). The model never
# emits it; it is stored in every lemma the run commits.
USER = ""
# Set in main(): the per-user DAG files live here, one per user
# (DAGS_DIR/<user_id>_dag.json). load_dag() merges every file in it into the
# complete DAG; only DAG_FILE is ever written. See workspace.py.
DAGS_DIR = ""
# The current user's DAG file, DAGS_DIR/<user_id>_dag.json.
DAG_FILE = ""
# Set in main(), beside DAG_FILE: DAGS_DIR/<user_id>_dag.json ->
# DAGS_DIR/<user_id>_dag.checkpoint.json. The checkpoint is per user, the
# way the DAG file is, and gitignored. See the "Cancelling and resuming"
# section of the module docstring.
CHECKPOINT_FILE = ""
REFERENCES_FILE = ""
# Set in main(), beside REFERENCES_FILE: the optional, gitignored operator
# notes the planner sees as "Human comments" — this user's own, never
# committed or shared (see the module docstring).
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

# The lemma window: how many proved lemmas reach the agents in full. A
# complete DAG of many users' files would otherwise put every statement
# into every prompt. Shown in full are the LEMMAS_ALL most recently proved
# lemmas, anyone's, together with the MY_LEMMAS most recently proved of the
# current user's own (the two overlap, and are deduplicated); every other
# proved lemma is listed by id only, as an index (lemma_window). -1 is no
# limit. Recency is the lemma's proved_at stamp, written at commit.
LEMMAS_ALL = 25
MY_LEMMAS = 25
# A third, fixed-size group: the LEMMAS_COMMON lemmas outside the two
# recency windows that the shown lemmas cite most often (direct citations
# only), shown statement-only. Ties are broken in an order random per run
# but fixed within it (_TIEBREAK_SEED), so the prompt does not churn from
# call to call. A lemma nothing shown cites does not rank.
LEMMAS_COMMON = 10
_TIEBREAK_SEED = random.getrandbits(64)


def _window_size(text: str) -> int:
    """A lemma-window size: a count, or -1 for no limit."""
    n = int(text)
    if n < -1:
        raise argparse.ArgumentTypeError(f"{n} is neither a count nor -1")
    return n


# How agents call tools (tools.py): "auto" tries the server's native tool
# calling and falls back to the JSON protocol the first time the server
# refuses it (a llama-server without --jinja); "json" uses the protocol
# from the start. _NATIVE_TOOLS_OK is what auto has learned: None until a
# refusal, then False for the rest of the process.
TOOL_MODE = "auto"
_NATIVE_TOOLS_OK: Optional[bool] = None


def _native_tools() -> bool:
    return TOOL_MODE == "auto" and _NATIVE_TOOLS_OK is not False


def _disable_native_tools(why: str, verbose: bool) -> None:
    global _NATIVE_TOOLS_OK
    if _NATIVE_TOOLS_OK is not False:
        _NATIVE_TOOLS_OK = False
        log(
            f"The server refused native tool calls ({why[:200]}); using "
            f"the JSON tool protocol for the rest of the run. Launch "
            f"llama-server with --jinja for native tool calls.",
            verbose,
        )


def agent_tools(
    dag: Dict[str, Any],
    suspended: Optional[Set[Tuple[str, str]]] = None,
) -> Optional["tools_mod.ToolSet"]:
    """The tools an agent shown the lemma window is offered: lookup_lemmas,
    over the same lemmas the window chose from — or None when the window
    shows every proved lemma, so a small DAG's prompts carry no tools."""
    _shown, _common, index = lemma_window(dag, suspended)
    if not index:
        return None
    return tools_mod.ToolSet([tools_mod.lemma_lookup(dag["lemmas"], suspended)])


def add_tool_flags(parser: argparse.ArgumentParser) -> None:
    """--tool-mode: taken by proofs run and proofs repair, beside the
    lemma window's flags."""
    parser.add_argument(
        "--tool-mode", choices=("auto", "json"), default=TOOL_MODE,
        help=(
            "How agents call tools such as lookup_lemmas. auto (default): "
            "the server's native tool calling, falling back to a JSON "
            "protocol if the server refuses it (llama-server needs --jinja "
            "for native calls). json: the JSON protocol from the start."
        ),
    )


def add_lemma_window_flags(parser: argparse.ArgumentParser) -> None:
    """--lemmas-all and --my-lemmas, the lemma window's two sizes: taken by
    proofs run and by proofs repair, the two commands whose agents are
    shown the proved lemmas."""
    parser.add_argument(
        "--lemmas-all", type=_window_size, default=LEMMAS_ALL, metavar="N",
        help=(
            "Show the agents the N most recently proved lemmas, anyone's, "
            f"in full (default: {LEMMAS_ALL}; -1 for all). The rest are "
            "listed by id."
        ),
    )
    parser.add_argument(
        "--lemmas-common", type=_window_size, default=LEMMAS_COMMON,
        metavar="K",
        help=(
            "Also show the K lemmas the shown lemmas cite most often, "
            f"statement only (default: {LEMMAS_COMMON}; -1 for every cited "
            "lemma). Ties are broken randomly, fixed for the run."
        ),
    )
    parser.add_argument(
        "--my-lemmas", type=_window_size, default=MY_LEMMAS, metavar="N",
        help=(
            "Also show the N most recently proved of your own lemmas in "
            f"full (default: {MY_LEMMAS}; -1 for all). Overlap with "
            "--lemmas-all is shown once."
        ),
    )

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
# Parallel mode: how many times a proof loop that crashed (an unexpected
# exception, not a stop) is restarted in a row — the count resets whenever
# the loop finishes a lemma — before it is given up for the run, and how
# long it waits before each restart. The bound matters because a
# restart resumes the round it crashed in: a crash that repeats on that
# round would otherwise restart forever.
MAX_LOOP_RESTARTS = 3
# proofs verify / proofs repair: the exit code when a verifier gave no
# verdict (the server failed, or the reply could not be read). Distinct from
# 1, a real rejection, and from argparse's 2; nothing was written.
EXIT_NO_VERDICT = 3
LOOP_RESTART_DELAY = 10
# How long the proof loop waits before retrying a call the server never
# answered (reason()'s "unavailable"): an outage is not the model's
# failure, so the round is not counted, and the call is retried until the
# server is back — or the run is stopped (Ctrl-C, or a parallel stop),
# with the round kept in the checkpoint.
SERVER_WAIT_SECONDS = 60

# "auto" runs unattended: the selector agent picks from the shortlist.
# "human" stops at every planning step. Set from --mode, then mutated by the
# hotkey and the menu, so it is genuinely a run-time toggle rather than a
# launch-time one.
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
# to a guess would defeat it. This constant stands in when the probe fails,
# and matches llm_backend.Profile's default context_limit.
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
# Never set this to False: reason() turns a False into None, because
# turning thinking off is the backend's job, for schema calls only — see
# the module docstring.
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


def _recorded_model() -> str:
    """The model name certificates and refutations record: the model that
    did the work, as llm_backend's recorded_model() reads it — the file a
    llama-server loaded, or the model a hosted API's reply names — rather
    than the --model request, which a llama-server ignores. MODEL_NAME
    before a backend exists."""
    return BACKEND.recorded_model() if BACKEND is not None else MODEL_NAME


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
#
# No flag says the conjecture is settled: that is computed from the DAG
# (resolution.find), and the planner attempts the conjecture by proposing
# a candidate under a reserved id (resolution.RESERVED_IDS).
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


def _dep_to_json(dep: Any) -> Any:
    """A dependency as it is stored in a DAG file: a lemma as the
    {"user_id", "lemma_id"} object that identifies it, a parsed reference as
    its bare id (reference ids are global, so bare is right)."""
    if isinstance(dep, tuple):
        return {"user_id": dep[0], "lemma_id": dep[1]}
    return str(dep)


def _citation_label(dep: Any) -> str:
    """How a citation is spelled for a log line: user:lemma for a lemma,
    the bare id for a reference."""
    if isinstance(dep, tuple):
        return f"{dep[0]}:{dep[1]}"
    return str(dep)


def lemma_id_set(
    dag: Dict[str, Any],
    suspended: Optional[Set[Tuple[str, str]]] = None,
) -> Set[str]:
    """Every lemma_id in the complete DAG, whoever proved it: the id space
    a new lemma must not restate — the "already in the DAG" test, taken
    across every user's file.

    Given the suspension (suspension.Suspension.suspended, recomputed from
    the hashes, the certificates and the refutations — see suspension.py), a
    suspended lemma's id is not in this set: the id is up for grabs again,
    for the users whose files do not hold it — a proof
    under an id the DAG holds only under suspension is a fresh proof, not a
    build on the suspended lemma, and screening it out would make the id
    dead for everyone until the suspended node is repaired. The suspended
    node itself is lifted by proofs verify or proofs repair re-accepting
    it, which re-issues the certificate for the proof in the file."""
    return {
        lid
        for (u, lid) in dag["lemmas"]
        if suspended is None or (u, lid) not in suspended
    }


def _warn_suspended(
    susp: "suspension.Suspension",
    verbose: bool,
    warned: Optional[Set[Tuple[str, str]]] = None,
    dag: Optional[Dict[str, Any]] = None,
) -> None:
    """The run's warning for the suspended lemmas (see suspension.py): a
    lemma with a refutation whose hash matches it, or without a valid
    certificate, or citing a lemma that has either, is hidden from the
    planner, the selector and the prover, and the operator is told so. With
    a warned set, once per lemma per run, the way the dangling-citation
    warning is told; without one, about every suspended lemma now.

    One case gets its own message: a lemma committed but never certified —
    suspended only for having no certificate, with no refutation and no
    certificate line in any file, ever. A commit certifies before it
    writes (_certify_before_write), so that is a certificate that was not
    issued: its write failed, or the hash had moved since the
    verification, as the committing run logged; or a run from before the
    certify-first order stopped between its commit and its certificate.
    Nothing is wrong with the proof that anyone has recorded, and proofs
    verify is how it gets its certificate."""
    pending = [
        p for p in sorted(susp.suspended)
        if warned is None or p not in warned
    ]
    if not pending:
        return
    ever_certified = {
        (str(c.get("user_id") or ""), str(c.get("lemma_id") or ""))
        for c in certificates.load_all(CONJECTURE_ROOT)
    }
    for u, lid in pending:
        if warned is not None:
            warned.add((u, lid))
        if (
            susp.reasons(u, lid) == ["no valid certificate"]
            and (u, lid) not in ever_certified
        ):
            dag_path = os.path.join(DAGS_DIR, f"{u}_dag.json")
            log(
                f"{u}:{lid} was committed but never certified — its "
                f"certificate was not issued (see the log of the run that "
                f"committed it). It is "
                f"suspended (hidden from the planner, the selector and the "
                f"prover) until certified: run proofs verify {dag_path} "
                f"{lid}.",
                verbose,
            )
            continue
        log(
            f"{u}:{lid} is suspended ({'; '.join(susp.reasons(u, lid))}); "
            f"it is hidden from the planner, the selector and the prover, "
            f"and proofs run will not build on it. proofs verify and "
            f"proofs repair still load it: re-accepting it (and any "
            f"suspended lemma it cites) re-issues the certificate and "
            f"lifts the suspension.",
            verbose,
        )


def _settled(
    dag: Dict[str, Any],
    suspended: Set[Tuple[str, str]],
    conjecture: str,
    verbose: bool,
) -> Optional[str]:
    """Whether the DAG settles the conjecture now (resolution.find): the
    kind, "proved" or "disproved", or None while it is open. Logs the
    lemma that settles it. A DAG that holds both a proof and a disproof
    settles nothing — it is inconsistent, and the run is told so and goes
    on, since a repair is the way out of it, not a stop."""
    found = resolution.find(dag["lemmas"], suspended, conjecture)
    kinds = {r.kind for r in found}
    if len(kinds) > 1:
        log(
            "The DAG both proves and disproves the conjecture ("
            + "; ".join(_citation_label(r.pair) for r in found)
            + "): at least one of these proofs is wrong. Treating the "
            "conjecture as open; proofs verify --full on each will show "
            "where.",
            verbose,
        )
        return None
    if not found:
        return None
    for r in found:
        log(f"{resolution.describe(r)}.", verbose)
    return found[0].kind


def _committed_uncertified(
    dag: Dict[str, Any],
    susp: "suspension.Suspension",
    in_flight: Dict[str, Any],
) -> bool:
    """Whether a checkpoint's in-flight lemma is already in the current
    user's file, committed but never certified: the state a run leaves when
    it stops between the commit of an accepted proof and its certificate
    (a kill, or a certificate write that failed — Ctrl-C is held off for
    that step, see _commit_section).

    Suspended (no certificate), so the resume's "already in the DAG" test
    does not see it, and without this the resume would re-prove it and
    commit the new proof beside it as id_2. Only that state answers yes:
    the user's own lemma under the in-flight id, stating the in-flight
    target, suspended, not refuted, and with no certificate line at all in
    any file — never certified, rather than certified once and gone stale.
    A lemma the user's file holds under suspension for any other reason (a
    refutation, a stale certificate) is a fresh proof's to replace, and the
    resume goes ahead."""
    pair = (USER, str(in_flight.get("lemma_id") or ""))
    node = dag["lemmas"].get(pair)
    if node is None or pair not in susp.suspended or pair in susp.refuted:
        return False
    target = in_flight.get("target") or {}
    if merkle.normalize(node.get("statement")) != merkle.normalize(
        target.get("statement")
    ):
        return False
    return not any(
        str(c.get("user_id") or "") == pair[0]
        and str(c.get("lemma_id") or "") == pair[1]
        for c in certificates.load_all(CONJECTURE_ROOT)
    )


def _uncertified_message(lemma_id: str) -> str:
    return (
        f"In-flight {lemma_id} is already in {os.path.basename(DAG_FILE)}"
        f", committed but never certified (the last run stopped between "
        f"the two). It is suspended until certified: run proofs verify "
        f"{DAG_FILE} {lemma_id}. Not re-proving it."
    )


def _lemma_pair_from(entry: Any) -> Optional[Tuple[str, str]]:
    """The (user_id, lemma_id) pair an entry declares, or None.

    A pair is a dict with usable "user_id"/"lemma_id" strings — the shape
    it has in a DAG file and in the prover's JSON — or a two-item tuple /
    list of the same (the shape a pair has in memory, or across a
    checkpoint). A bare id is not a pair: a lemma_id is unique only within
    one user's file, so on its own it names no particular lemma."""
    if isinstance(entry, dict):
        uid = entry.get("user_id")
        lid = entry.get("lemma_id")
        if isinstance(uid, str) and isinstance(lid, str) and uid and lid:
            return (uid.strip(), lid.strip())
    elif isinstance(entry, (tuple, list)) and len(entry) == 2:
        uid, lid = entry
        if isinstance(uid, str) and isinstance(lid, str) and uid and lid:
            return (uid.strip(), lid.strip())
    return None


def _citation_display(entry: Any) -> str:
    """How a citation entry reads in a log line: a pair as
    "user_id:lemma_id", a bare id as itself, anything else as JSON."""
    pair = _lemma_pair_from(entry)
    if pair is not None:
        return f"{pair[0]}:{pair[1]}"
    if isinstance(entry, str):
        return entry
    try:
        return json.dumps(entry, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return repr(entry)


def _check_prover_citations(
    raw_lemmas: Any,
    raw_references: Any,
    dag: Dict[str, Any],
    ref_ids: Set[str],
    suspended: Optional[Set[Tuple[str, str]]] = None,
) -> Tuple[List[Tuple[str, str]], List[str], List[Tuple[str, str]]]:
    """The prover's declared citations, normalized and checked against the
    complete DAG (see load_dag()).

    cited_lemmas is an array of {"user_id", "lemma_id"} objects, every pair
    of which is checked against the complete DAG; cited_references is a
    separate list of reference ids. Returns (cited_lemmas,
    cited_references, unmatched): the surviving pairs, the surviving
    reference ids, and (label, kind) for every entry that matched nothing —
    kind is "lemma" for a pair that names no lemma in any user's file,
    "reference" for an id that is in neither the DAG nor the reference
    collection, "suspended" for a pair that names a suspended lemma, and
    "other" for an entry that is neither a pair nor an id.
    The caller sends the unmatched back to the prover as feedback; the
    "reference" ones are the maintainer's to supply, since references.md is
    the conjecture's committed collection and only the maintainer adds to
    it (see parsing.py).

    Lenient on shape, strict on existence: a bare id in cited_lemmas that
    is a reference id (or exactly one lemma's id) is read as the citation
    it can mean, and a pair misplaced in cited_references is read as a
    lemma citation — but whatever an entry names, it must exist, or it is
    unmatched.

    Given the suspension, a suspended lemma is unmatched too, as
    "suspended": the prover was not shown it and may not build on it
    (see suspension.py). Let through, it would suspend the new
    lemma the moment it was committed — and in a repair, a citation of the
    lemma being re-proved, or of one of its dependents (all suspended by
    the rejection that started the repair), would close a citation cycle
    once the new proof replaced the old one in place. A bare id resolves
    only to an unsuspended lemma for the same reason."""
    suspended = suspended or set()
    owners: Dict[str, List[str]] = {}
    suspended_ids: Set[str] = set()
    for u, lid in dag["lemmas"]:
        if (u, lid) in suspended:
            suspended_ids.add(lid)
        else:
            owners.setdefault(lid, []).append(u)

    cited_lemmas: List[Tuple[str, str]] = []
    cited_references: List[str] = []
    unmatched: List[Tuple[str, str]] = []
    seen_pairs: Set[Tuple[str, str]] = set()
    seen_refs: Set[str] = set()
    seen_unmatched: Set[str] = set()

    def report_unmatched(label: str, kind: str) -> None:
        if label not in seen_unmatched:
            seen_unmatched.add(label)
            unmatched.append((label, kind))

    def take_pair(pair: Tuple[str, str]) -> None:
        if pair in suspended:
            report_unmatched(f"{pair[0]}:{pair[1]}", "suspended")
        elif pair in dag["lemmas"]:
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                cited_lemmas.append(pair)
        else:
            report_unmatched(f"{pair[0]}:{pair[1]}", "lemma")

    def take_ref(rid: str) -> None:
        if rid in ref_ids:
            if rid not in seen_refs:
                seen_refs.add(rid)
                cited_references.append(rid)
        else:
            report_unmatched(rid, "reference")

    def take_bare(rid: str) -> None:
        if rid in ref_ids:
            take_ref(rid)
        elif len(owners.get(rid, [])) == 1:
            take_pair((owners[rid][0], rid))
        elif rid in suspended_ids and rid not in owners:
            report_unmatched(rid, "suspended")
        else:
            report_unmatched(rid, "reference")

    for entry in raw_lemmas if isinstance(raw_lemmas, list) else []:
        pair = _lemma_pair_from(entry)
        if pair is not None:
            take_pair(pair)
        elif isinstance(entry, str) and entry.strip():
            take_bare(entry.strip())
        else:
            report_unmatched(_citation_display(entry), "other")
    for entry in raw_references if isinstance(raw_references, list) else []:
        if isinstance(entry, str) and entry.strip():
            take_bare(entry.strip())
        else:
            pair = _lemma_pair_from(entry)
            if pair is not None:
                take_pair(pair)
            else:
                report_unmatched(_citation_display(entry), "other")
    return cited_lemmas, cited_references, unmatched


def load_dag() -> Dict[str, Any]:
    """The complete DAG: the merge of every user's DAG file in dags/.

    Each file in dags/ is one user's lemmas, named <user_id>_dag.json, and
    the merge keys them by the pair (user_id, lemma_id) — a bare lemma_id
    is unique only within one user's file, so the pair is the only key that
    cannot collide. The file stores neither half inside a node (see
    _stored_node): in memory every node is given both, user_id from the
    file's name — the lemma's owner — and lemma_id from its key. A stored
    node's last_prover_id names who wrote its proof, not who owns it, and
    is not an identity. Citations are stored the same way
    (see load_dag()): cited_lemmas is an array of {"user_id",
    "lemma_id"} objects, read in memory as a list of (user_id, lemma_id)
    tuples, and cited_references a separate array of bare reference ids.
    Older files may still carry the mixed "dependencies" list (pair objects
    and bare reference ids in one field); the second pass below migrates
    it.

    This is a read-only view: the run writes new lemmas to the current
    user's file alone (commit_lemma_to_dag) and never to another user's
    file.
    """
    dag: Dict[str, Any] = {"lemmas": {}}
    if not DAGS_DIR or not os.path.isdir(DAGS_DIR):
        return dag
    for path in sorted(glob.glob(os.path.join(DAGS_DIR, "*_dag.json"))):
        file_user = os.path.basename(path)[: -len("_dag.json")]
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            # Unreadable means corrupt or cut off; it is another user's
            # file, so the run goes on without its lemmas rather than
            # dying — but the operator must see it, so the warning is
            # logged unconditionally (a load_dag() per iteration repeats
            # it, which is the point: a corrupt file is a state the run
            # should not be able to shrug off silently).
            print(
                f"{os.path.basename(path)} is unreadable ({e}); its "
                f"lemmas are skipped.",
            )
            continue
        if not isinstance(data, dict):
            continue
        lemmas = data.get("lemmas")
        if not isinstance(lemmas, dict):
            continue
        for lid, node in lemmas.items():
            if not isinstance(node, dict):
                continue
            node = dict(node)
            node["user_id"] = file_user
            node["lemma_id"] = str(lid)
            key = (file_user, str(lid))
            if key not in dag["lemmas"]:
                dag["lemmas"][key] = node
    # Second pass: split each node's citations into the two stored fields
    # (see load_dag()). A lemma citation is the (user_id, lemma_id)
    # pair its file names it, a reference citation the reference's bare id.
    # Older files still carry the mixed "dependencies" list (pair objects
    # and bare reference ids in one field): a pair object becomes a lemma
    # pair, and a bare id a reference citation — except that a bare id
    # exactly one lemma in the merge owns is read as that lemma's pair, the
    # legacy spelling of a lemma citation. Whatever cannot be resolved is
    # kept in the field its shape names, where the dangling-citation check
    # in the run loop will see it.
    owners: Dict[str, List[str]] = {}
    for u, lid in dag["lemmas"]:
        owners.setdefault(lid, []).append(u)
    for node in dag["lemmas"].values():
        cited_lemmas: List[Tuple[str, str]] = []
        cited_references: List[str] = []
        seen_pairs: Set[Tuple[str, str]] = set()
        seen_refs: Set[str] = set()
        for field in ("cited_lemmas", "dependencies"):
            entries = node.get(field)
            if not isinstance(entries, list):
                continue
            for entry in entries:
                pair = _lemma_pair_from(entry)
                if pair is not None:
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        cited_lemmas.append(pair)
                    continue
                if isinstance(entry, str):
                    rid = entry.strip()
                    if not rid:
                        continue
                    if len(owners.get(rid, [])) == 1:
                        pair = (owners[rid][0], rid)
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            cited_lemmas.append(pair)
                    elif rid not in seen_refs:
                        # A reference id, or a bare id nothing owns: kept
                        # as a reference citation, where the dangling check
                        # will flag it if the id is gone.
                        seen_refs.add(rid)
                        cited_references.append(rid)
                # anything else (a number, null) names nothing: dropped
        entries = node.get("cited_references")
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, str):
                    rid = entry.strip()
                    if rid and rid not in seen_refs:
                        seen_refs.add(rid)
                        cited_references.append(rid)
        node["cited_lemmas"] = cited_lemmas
        node["cited_references"] = cited_references
        # The legacy field is migrated, not mirrored: drop it so a save of
        # a loaded DAG writes only the two stored fields.
        node.pop("dependencies", None)
    return dag


def load_user_dag() -> Dict[str, Any]:
    """The current user's own DAG file, or an empty one.

    The file the run writes to, read whole: commit_lemma_to_dag updates one
    lemma in it and keeps whatever else the file holds. A file that exists but is not a JSON
    object is an error, not an empty DAG: it is the operator's data, and
    overwriting it would be worse than failing loudly.
    """
    if os.path.exists(DAG_FILE):
        with open(DAG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"{DAG_FILE} must hold a JSON object")
        return data
    return {"lemmas": {}}


def _user_header() -> Dict[str, str]:
    """The user identity stamped at the top of the per-user files this run
    writes as its own user: the config's user.id, user.name and user.email
    (see config.py) — user.id is the run's USER, the file's
    name, and name and email are whatever the config holds, the empty
    string where it holds none. The DAG file carries it as the first
    entries of its object, the certificate file as its first line: the
    header that says whose file this is, beside the name that already said
    it, so the file stands alone when it is read out of context."""
    return {
        config.USER_ID_KEY: USER,
        config.USER_NAME_KEY: config.get(config.USER_NAME_KEY) or "",
        config.USER_EMAIL_KEY: config.get(config.USER_EMAIL_KEY) or "",
    }


def _write_user_file(data: Dict[str, Any]) -> None:
    """Whole-file write of the current user's DAG file: temp file beside it,
    then os.replace, the checkpoint's way. The parallel loops read the DAG
    from several threads at once, and a plain open("w") lets a reader catch
    the file mid-write (an empty or half-formed file) and crash the loop on
    the parse. The file's first entries are the run's user identity, the
    header the per-user files carry (see config.py), kept
    at the top and refreshed from the config on every write.

    Call it holding DAG_LOCK and the file lock on DAG_FILE (locking.py):
    the temp file's name is fixed, and the write is the second half of a
    read-modify-write another thread or process must not interleave
    with."""
    header = _user_header()
    data = {**header, **{k: v for k, v in data.items() if k not in header}}
    os.makedirs(DAGS_DIR, exist_ok=True)
    tmp = DAG_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, DAG_FILE)


def _ensure_user_dag_file() -> None:
    """The current user's DAG file, created empty if absent.

    Both loops discard a checkpoint that has no DAG file beside it: the
    operator deleting the user's DAG file is how a run resets, and a
    checkpoint outliving its DAG would resurrect the old budget and
    in-flight lemma. An interrupted run must resume, not reset, so the
    file is in place before the first checkpoint can be written.

    The existence test is repeated under the locks: another process of the
    same user may create the file, and commit lemmas to it, between a test
    outside them and the write.
    """
    if os.path.exists(DAG_FILE):
        return
    with DAG_LOCK, locking.file_lock(DAG_FILE):
        if not os.path.exists(DAG_FILE):
            _write_user_file({"lemmas": {}})


def _stored_node(node: Dict[str, Any], last_prover_id: str) -> Dict[str, Any]:
    """A lemma node in the shape a DAG file stores it, and nothing else:

        statement, proof, cited_lemmas, cited_references,
        last_prover_id, proved_at

    The owner is not stored in the node — it is the file's name,
    <owner>_dag.json — and neither is the lemma_id, which is the node's
    key; load_dag() supplies both in memory. last_prover_id is the user
    whose run wrote the proof now in the file: the owner for a lemma their
    own run proved, the repairer for one re-proved by proofs repair.
    Citations are stored the load_dag() way: cited_lemmas as
    {"user_id", "lemma_id"} objects naming each cited lemma's owner,
    cited_references as bare ids. proved_at is the moment of this write,
    the lemma window's recency. Neither last_prover_id nor proved_at is in
    the Merkle hash, which covers only the statement, the proof and the
    cited statements."""
    return {
        "statement": node["statement"],
        "proof": node["proof"],
        "cited_lemmas": [
            _dep_to_json(dep) for dep in node.get("cited_lemmas", [])
        ],
        "cited_references": list(node.get("cited_references", [])),
        "last_prover_id": last_prover_id,
        "proved_at": certificates.now_iso(),
    }


def commit_lemma_to_dag(
    lemma_id: str, node: Dict[str, Any], verified_hash: Optional[str],
    verbose: bool = True,
) -> Tuple[str, bool]:
    """Certify node and commit it to the current user's DAG file under
    lemma_id, or under a fresh non-colliding id if lemma_id is already
    taken in that file. The id check and the write are one step under the
    DAG lock, so the check can never go stale: a writer that finds its id
    taken commits under lemma_id_2 (or _3, ...) rather than over the first
    proof. A collision is with the current user's own lemmas: lemma_id is
    unique only within one user's DAG, so a lemma another user proved under
    the same id is no collision at all. The node is stored in the shape
    _stored_node gives it, with the current user as its last_prover_id and
    the moment of the commit as proved_at (the lemma window's recency).
    Returns (committed_id, renamed).

    All three verifier checks accepted node, so it is certified under the
    id it is committed under — before the write, not after
    (_certify_before_write): readers load the DAG and the certificates
    without a lock, and in this order the lemma is never in the file
    without its certificate. A reader between the two writes sees a
    certificate for a lemma the DAG does not hold yet, which counts for
    nothing; the reverse order would show it a committed lemma with no
    certificate, suspended. verified_hash is the hash the verifiers
    checked (_verified_hash), compared with the hash at the commit there.

    The check and the write are one step across processes too: the file
    lock (locking.py) keeps a proofs repair or a second run of the same
    user from replacing the file between this read and this write."""
    with DAG_LOCK, locking.file_lock(DAG_FILE):
        data = load_user_dag()
        lemmas = data.get("lemmas")
        if not isinstance(lemmas, dict):
            lemmas = {}
        if lemma_id not in lemmas:
            committed_id = lemma_id
        else:
            suffix = 2
            while f"{lemma_id}_{suffix}" in lemmas:
                suffix += 1
            committed_id = f"{lemma_id}_{suffix}"
        _certify_before_write(
            USER, committed_id, node, verified_hash, verbose,
        )
        lemmas[committed_id] = _stored_node(node, USER)
        data["lemmas"] = lemmas
        _write_user_file(data)
        return committed_id, committed_id != lemma_id


# ----------------------------------------------------------------------------
# Certificates
# ----------------------------------------------------------------------------
# The record that a lemma was accepted (see certificates.py). The run
# writes a certificate the moment an acceptance happens — the three verifier
# checks accepting a proof, the prover's or the operator's —
# into this run's user's file, the verifier's, whoever owns the lemma. The storage, the upsert rule and the validity check
# live in certificates.py; this is the run's one entry point to them.

def _certify_before_write(
    owner: str, lemma_id: str, node: Dict[str, Any],
    verified_hash: Optional[str], verbose: bool = True,
) -> None:
    """Record a certificate for a lemma this run just accepted, before the
    lemma is written: the commit's first half (commit_lemma_to_dag,
    _replace_lemma_in_owner_file).

    The caller holds DAG_LOCK and the file lock on the owner's DAG file,
    and writes node to (owner, lemma_id) right after, so the hash is the
    one the lemma will have once the write lands: the DAG loaded from the
    files with node in place under its pair, over references.md. The
    write adds only last_prover_id and proved_at, neither of which the
    Merkle hash covers. The line covers that proof and stays valid until
    the lemma's statement, its proof, or the statement of anything it
    cites changes. The model recorded is the verifier's (_recorded_model).

    The order is the point. Readers — load_dag(), suspension.compute —
    take no lock, so whichever file is written first is what a reader can
    see alone. A certificate whose lemma is not in the DAG yet counts for
    nothing (suspension only looks up the lemmas the DAG holds), while a
    lemma in the DAG without its certificate is suspended. Certified
    first, the lemma is never in the file without its certificate; a
    crash between the two writes leaves an orphan line that matches
    nothing, and proofs prune clears it.

    A certificate is the record of an acceptance; a failure to write it
    must not undo the acceptance, so every failure here degrades to a
    warning and the caller writes the lemma anyway — suspended, until
    proofs verify certifies it.

    owner is the lemma's owner, the certificate's user_id: the current
    user for a run's commit, while proofs repair commits a re-proof in
    place in the owner's file (see repair_entry()). The verifier is either
    way the user whose run accepted the proof, and the line goes into that
    user's certificates/ file — never the owner's.

    verified_hash is the hash of the version the verifiers checked
    (_verified_hash: the proof over the DAG and references they were shown).
    When the hash at the commit differs — a lemma it cites, or
    references.md, changed between the verification and the commit (a
    proofs repair, a pull) — the certificate would cover a version nobody
    verified, so none is issued: the lemma stays suspended until proofs
    verify checks the version in the file. None skips the comparison
    (_verified_hash gives None for a proof whose hash cannot be computed,
    and the hash here then reports why).
    """
    model = _recorded_model()
    try:
        lemmas = dict(load_dag()["lemmas"])
        lemmas[(owner, lemma_id)] = {
            **node, "user_id": owner, "lemma_id": lemma_id,
        }
        h = merkle.Merkle(lemmas, load_references()).hash(owner, lemma_id)
    except merkle.MerkleCycleError as e:
        log(f"No certificate for {lemma_id}: {e}", verbose)
        return
    if verified_hash is not None and h != verified_hash:
        dag_path = os.path.join(DAGS_DIR, f"{owner}_dag.json")
        log(
            f"No certificate for {lemma_id}: something it stands on (a "
            f"cited lemma's statement, or references.md) changed between the "
            f"verification and the commit, so the version in the file is "
            f"not the one the verifiers checked. It stays suspended until "
            f"verified again: run proofs verify {dag_path} {lemma_id}.",
            verbose,
        )
        return
    try:
        cert = certificates.record(
            CONJECTURE_ROOT, owner, lemma_id, h, USER, model,
            user=_user_header(),
        )
    except OSError as e:
        log(
            f"Could not write the certificate for {lemma_id}: {e}",
            verbose,
        )
        return
    how = f"model {model}" if model else "human acceptance"
    log(
        f"Certificate for {lemma_id}: accepted by {USER} ({how}), "
        f"count {cert['count']}.",
        verbose,
    )


# ----------------------------------------------------------------------------
# Verify and repair: the verification path proofs verify and proofs repair
# share (see verify_entry(), refutations.py, repair_entry())
# ----------------------------------------------------------------------------
# A lemma that is already in the DAG is verified the way the proof loop
# verifies a proposed proof: the three verifier agents in order, each a
# single atomic check, and the first rejection ends the count (a proof the
# verifiers are not all for is not accepted, whatever the rest would say).
# What differs from the loop is what the verdict does to the record, and
# only proofs verify and proofs repair do this: a rejection writes a
# refutation file (see refutations.py), an acceptance issues a certificate
# (see certificates.py), and the refutations whose justifications were on
# the table are dismissed. proofs run never takes this path: a rejection
# inside the loop is feedback to its own reviser, not a committed record,
# and refutations are not made during the run.


def _citation_closure(
    dag: Dict[str, Any], user_id: str, lemma_id: str,
) -> List[Tuple[str, str]]:
    """The lemma's cited-lemma closure in dependency order: every lemma it
    cites, dependencies first, the target last — the order proofs verify
    --full verifies in, cited lemmas before the lemma that cites them.

    A cited pair the DAG does not hold is skipped (a dangling citation is
    the verifier's to see as an absent result, not the closure's to
    chase), and a citation cycle is cut at the first repeat rather than
    followed: the Merkle hash of a cycle would raise MerkleCycleError
    anyway, so the closure cannot loop where the hash would refuse to.
    """
    lemmas = dag["lemmas"]
    out: List[Tuple[str, str]] = []
    done: Set[Tuple[str, str]] = set()
    on_stack: Set[Tuple[str, str]] = set()

    def visit(u: str, lid: str) -> None:
        pair = (u, lid)
        if pair in done or pair in on_stack:
            return
        node = lemmas.get(pair)
        if node is None:
            return
        on_stack.add(pair)
        for cited in node.get("cited_lemmas", []):
            if isinstance(cited, tuple) and len(cited) == 2:
                visit(cited[0], cited[1])
        on_stack.discard(pair)
        done.add(pair)
        out.append(pair)

    visit(user_id, lemma_id)
    return out


def _dependent_chain(
    dag: Dict[str, Any], start: Tuple[str, str],
) -> List[Tuple[str, str]]:
    """start's transitive dependents in dependency order: every lemma of
    the DAG that cites start directly or transitively, each before the
    lemmas that cite it — the reverse of _citation_closure's order, the
    order proofs repair climbs the dependency chain (see repair_entry()).

    The set is the reverse-citation closure of start; the order is a
    post-order depth-first pass over the set's own citations, so a lemma
    comes after every lemma it cites that is in the set. start itself is
    not a dependent of itself and is never in the result, and a citation
    cycle is cut at the first repeat, the way _citation_closure cuts one:
    a lemma in a cycle has no Merkle hash to re-verify, and the climb
    will not loop where the hash would refuse to.
    """
    lemmas = dag["lemmas"]
    # The reverse edges: for each lemma, the lemmas that cite it.
    cited_by: Dict[Tuple[str, str], List[Tuple[str, str]]] = {
        key: [] for key in lemmas
    }
    for key, node in lemmas.items():
        for cited in node.get("cited_lemmas", []):
            if isinstance(cited, tuple) and len(cited) == 2 and cited in lemmas:
                cited_by[cited].append(key)
    # The dependent set: everything that reaches start by following the
    # reverse edges — nothing else, so a dependent's unrelated citations
    # are not pulled into the climb.
    in_set: Set[Tuple[str, str]] = set()
    queue: List[Tuple[str, str]] = [start]
    while queue:
        for dep in cited_by.get(queue.pop(), []):
            if dep not in in_set:
                in_set.add(dep)
                queue.append(dep)
    # Dependency order: emit a lemma after the lemmas it cites that are in
    # the set; sorted start points keep the result deterministic.
    out: List[Tuple[str, str]] = []
    done: Set[Tuple[str, str]] = set()
    on_stack: Set[Tuple[str, str]] = set()

    def visit(u: str, lid: str) -> None:
        pair = (u, lid)
        if pair in done or pair in on_stack:
            return
        on_stack.add(pair)
        for cited in lemmas[pair].get("cited_lemmas", []):
            if isinstance(cited, tuple) and len(cited) == 2 and cited in in_set:
                visit(cited[0], cited[1])
        on_stack.discard(pair)
        done.add(pair)
        if pair != start:
            out.append(pair)

    for pair in sorted(in_set):
        visit(pair[0], pair[1])
    return out


def _dependency_order(
    dag: Dict[str, Any], pairs: Set[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    """The given lemmas in dependency order: every lemma after the lemmas
    it cites, dependencies first — the order proofs export writes them in
    (see export.py), each result resting on the ones above it.

    A post-order depth-first pass over the set's own citations, the way
    _citation_closure and _dependent_chain order their sets: a citation
    the set does not hold is not pulled in (an export of one user's
    lemmas writes that user's lemmas; their dependencies live in the
    other users' files, written by their own exports), and a citation
    cycle is cut at the first repeat, the way those cut one — a cycle's
    lemmas still get an order, one after the other, rather than an
    export that never ends. Sorted start points keep the result
    deterministic, the way _dependent_chain's do.
    """
    lemmas = dag["lemmas"]
    out: List[Tuple[str, str]] = []
    done: Set[Tuple[str, str]] = set()
    on_stack: Set[Tuple[str, str]] = set()

    def visit(u: str, lid: str) -> None:
        pair = (u, lid)
        if pair in done or pair in on_stack:
            return
        on_stack.add(pair)
        for cited in lemmas[pair].get("cited_lemmas", []):
            if isinstance(cited, tuple) and len(cited) == 2 and cited in pairs:
                visit(cited[0], cited[1])
        on_stack.discard(pair)
        done.add(pair)
        out.append(pair)

    for pair in sorted(pairs):
        visit(pair[0], pair[1])
    return out


def _replace_lemma_in_owner_file(
    owner: str, lemma_id: str, node: Dict[str, Any],
    verified_hash: Optional[str], verbose: bool = True,
) -> None:
    """Replace the lemma in place in the owner's DAG file (see
    repair_entry()): the (user_id, lemma_id) pair the refutation named is
    kept, in the owner's file, and the node's statement, proof and citations
    are what the re-proof established. Unlike the run's commit, which writes
    only the current user's file, this may write another user's file — the
    only write to another user's DAG file anywhere in the system.

    The node is stored in the shape a run's commit writes (_stored_node),
    so a repaired lemma reads the same as a run-committed one, with the
    repairer — the current user — as its last_prover_id: the owner stays
    the file's name, and the node says who wrote the proof now in it. A
    field of the node it replaces that _stored_node does not write is
    dropped with it. The write
    is under the DAG lock and the file lock (locking.py) and atomic, the
    way the run's file writes are, so a proofs run of the owner on this
    machine cannot commit over it or under it.

    The re-proof is certified under the same locks before the file is
    replaced (_certify_before_write), the order commit_lemma_to_dag
    certifies in and for the same reason: a reader between the two writes
    sees the old proof with its old certificate, or the new proof with its
    new one, never the new proof uncertified.
    """
    path = os.path.join(DAGS_DIR, f"{owner}_dag.json")
    with DAG_LOCK, locking.file_lock(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"{path} must hold a JSON object")
        lemmas = data.get("lemmas")
        if not isinstance(lemmas, dict) or lemma_id not in lemmas:
            raise KeyError(f"{owner}:{lemma_id} is not in {path}")
        _certify_before_write(owner, lemma_id, node, verified_hash, verbose)
        # A re-proof is a fresh proof: its proved_at moves it to the front
        # of the lemma window, as a new commit does.
        lemmas[lemma_id] = _stored_node(node, USER)
        if owner == USER:
            # The run's own file: refresh the identity header at the top,
            # the way _write_user_file does it. Another user's file keeps
            # the header the owner stamped — the run does not hold the
            # owner's name or email, and a header is the owner's to write.
            header = _user_header()
            data = {**header, **{k: v for k, v in data.items()
                                 if k not in header}}
        data["lemmas"] = lemmas
        os.makedirs(DAGS_DIR, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)


def _verifier_prompts(parser: argparse.ArgumentParser) -> Dict[str, str]:
    """The three verifier prompts as the prompt paths give them, each
    checked non-empty: a verifier with an empty system prompt is not a
    verifier, and a missing prompt file is a setup error, not a verdict."""
    verifier_sys = {name: load_file(PROMPT_PATHS[name]) for name in VERIFIER_AGENTS}
    missing = [name for name in verifier_sys if not verifier_sys[name]]
    if missing:
        parser.error(f"no prompt text for: {', '.join(missing)}")
    return verifier_sys


def verify_stored_lemma(
    dag: Dict[str, Any],
    references: List[Dict[str, Any]],
    user_id: str,
    lemma_id: str,
    conjecture_text: str,
    verifier_sys: Dict[str, str],
    verbose: bool,
) -> Tuple[str, Optional[str]]:
    """The three verifier checks on a lemma that is already in the DAG,
    with the refutation bookkeeping around them (see refutations.py).

    The lemma's current Merkle hash is recomputed from the DAG and
    references as they stand now, and every refutation file the lemma has
    is loaded. A refutation counts while its hash matches the current
    one (refutations.is_counting); the ones that do not are stale —
    objections to versions of the lemma that are no longer in the files —
    and their justifications are passed to the verifiers all the same,
    so a known objection is not forgotten. The
    counting ones go on the table as well, marked as such: an objection
    to the proof as it stands now is the one a re-verification is
    answering, and proofs repair runs on exactly that case. A lemma with
    a counting refutation is warned about before the checks start.

    A rejection ends the count, the run loop's rule, and the rejecting
    verifier's justification is what the new refutation file records,
    under the user who ran the verification and the model under which it
    happened. An acceptance issues the certificate — the owner's file,
    the verifier recorded the way a certificate records it — and
    dismisses the refutations whose justifications were on the table: the
    stale ones, and the counting ones, answered.

    A step that gives no verdict at all (_run_verifier's "" or
    "unavailable": the server failed or replied with nothing, the reply
    could not be read, or the pass hit the context wall) ends the count too, but writes nothing: a
    refutation is the record of a verifier's rejection, and a missing
    verdict is not one — written, it would suspend the lemma and everything
    above it for a server outage. Nothing is issued or dismissed either;
    the lemma stands exactly as it did, and the verification can be re-run.
    An acceptance whose certificate cannot be written ends the same way:
    the refutations are kept rather than dismissed with no record of the
    acceptance in their place.

    Returns (outcome, justification): ("accepted", None), ("rejected", the
    rejecting justification), or ("no_verdict", why there was none).
    """
    node = dag["lemmas"].get((str(user_id), str(lemma_id)))
    if node is None:
        raise KeyError(f"({user_id!r}, {lemma_id!r}) is not a lemma of the DAG")
    m = merkle.Merkle(dag["lemmas"], references)
    h = m.hash(user_id, lemma_id)

    on_file = refutations.for_lemma(CONJECTURE_ROOT, user_id, lemma_id)
    stale = 0
    for path, ref in on_file:
        if refutations.is_counting(ref, h):
            log(
                f"{lemma_id} has a refutation that counts ({path.name}): "
                f"the proof it rejects is the proof in the file now.",
                verbose,
            )
        else:
            stale += 1
    if stale:
        log(
            f"{lemma_id} has {stale} stale refutation(s); their "
            f"justifications go to the verifiers.",
            verbose,
        )

    cited_lemmas = [c for c in node.get("cited_lemmas", []) if isinstance(c, tuple)]
    cited_references = [str(r) for r in node.get("cited_references", [])]
    statement = str(node.get("statement") or "")
    proof = str(node.get("proof") or "")

    notes = [
        {
            "verifier": str(ref.get("verifier") or ""),
            "model": str(ref.get("model") or ""),
            "date": str(ref.get("date") or ""),
            "verdict": str(ref.get("verdict") or ""),
            "hash": str(ref.get("hash") or ""),
            "hash_matches_current": refutations.is_counting(ref, h),
            "justification": str(ref.get("justification") or ""),
        }
        for _path, ref in on_file
    ]
    refutation_block = ""
    if notes:
        refutation_block = (
            "Known refutations (rejections of this lemma on record, oldest "
            "first; each carries the verifier's justification. A hash that "
            "no longer matches the lemma's is an objection to an earlier "
            "version of it, kept so the objection is not lost when the "
            "lemma changes; one that still matches is an objection to the "
            "proof as it stands):\n"
            f"{json.dumps(notes, indent=2)}\n\n"
        )

    verifier_user = (
        f"Conjecture:\n{conjecture_text}\n\n"
        "Cited results (statements of exactly the lemmas and references "
        "the proof declares it used; nothing else is available to it):\n"
        f"{json.dumps(verifier_context(dag, cited_lemmas, cited_references, references), indent=2)}\n\n"
        f"{refutation_block}"
        "Target lemma:\n"
        f"{json.dumps({'user_id': user_id, 'lemma_id': lemma_id, 'statement': statement}, indent=2)}\n\n"
        f"Proposed proof:\n{proof}"
    )
    for step, agent in enumerate(VERIFIER_AGENTS, 1):
        role = agent[:-3]
        decision, justification = _run_verifier(role, verifier_sys[agent], verifier_user, verbose)
        log(
            f"Verifier {step}/{len(VERIFIER_AGENTS)} ({role}): "
            f"{decision.upper() or '???'} — {justification}",
            verbose,
        )
        if decision == "reject":
            path = refutations.record(
                CONJECTURE_ROOT, user_id, lemma_id, h, USER, _recorded_model(),
                "reject", justification,
            )
            log(f"Refutation for {lemma_id} written: {path}", verbose)
            return "rejected", justification
        if decision != "accept":
            log(
                f"No verdict on {lemma_id} from {role}; no refutation is "
                f"written and nothing is certified. Re-run once the server "
                f"is answering.",
                verbose,
            )
            return "no_verdict", justification
    try:
        certificates.record(
            CONJECTURE_ROOT, user_id, lemma_id, h, USER, _recorded_model(),
            user=_user_header(),
        )
    except OSError as e:
        # The acceptance is only as good as its record: without the
        # certificate the lemma stays suspended, and dismissing the
        # refutations now would delete the objections while leaving
        # nothing in their place. Keep them, and say the verification has
        # to be re-run — the no-verdict outcome: nothing was written.
        log(
            f"{lemma_id} passed the verifier checks, but the certificate "
            f"could not be written ({e}); its refutations are kept and "
            f"nothing is recorded. Re-run the verification once the "
            f"certificate file is writable.",
            verbose,
        )
        return "no_verdict", f"the certificate could not be written: {e}"
    log(
        f"Certificate for {lemma_id}: accepted by {USER} (model {_recorded_model()}).",
        verbose,
    )
    for path, _ref in on_file:
        try:
            refutations.delete(path)
            log(f"Refutation dismissed: {path}", verbose)
        except OSError as e:
            log(f"Could not delete the refutation {path}: {e}", verbose)
    log(
        f"Lemma {lemma_id} passed all {len(VERIFIER_AGENTS)} verifier checks.",
        verbose,
    )
    return "accepted", None


def _verify_session(
    *,
    conjecture_dir: str,
    user_id: str,
    model: str,
    host: Optional[str],
    api_key: Optional[str],
    backend_choice: str,
    num_ctx: Optional[int],
    verbose: bool,
    parser: argparse.ArgumentParser,
) -> None:
    """The shared setup of proofs verify and proofs repair: the backend
    and the conjecture's paths, the run's, minus the run's — no
    checkpoint, no hotkey, no SIGINT handler, and the DAG files are
    read, never written (a verify or repair writes only refutations and
    the running user's own certificate file). Sets the same globals main() sets, so
    load_dag, load_references and the verifier plumbing below run
    unchanged. user_id is the user running the verification: the verifier
    of record in any certificate issued and any refutation written.

    The certificates and refutations this session writes name the model
    _recorded_model() reports, not the --model request.
    """
    global MODEL_NAME, CONJECTURE_FILE, CONJECTURE_ROOT, DAGS_DIR, DAG_FILE
    global REFERENCES_FILE, COMMENTS_FILE, USER, BACKEND, PROFILE, NUM_CTX
    global PROMPT_PATHS
    MODEL_NAME = model
    BACKEND = llm_backend.make_backend(
        model, host, REQUEST_TIMEOUT, api_key=api_key, backend=backend_choice,
    )
    PROFILE = BACKEND.probe()
    if PROFILE.auth_error:
        parser.error(PROFILE.auth_error)
    if num_ctx is None:
        NUM_CTX = PROFILE.context_limit
    elif PROFILE.probed_llama_cpp:
        NUM_CTX = min(num_ctx, PROFILE.context_limit)
    else:
        NUM_CTX = num_ctx
    REASONING_OPTIONS["num_ctx"] = NUM_CTX
    EXTRACT_OPTIONS["num_ctx"] = NUM_CTX
    log(
        llm_backend.describe(
            PROFILE, num_ctx if num_ctx is not None else NUM_CTX
        ),
        verbose,
    )
    USER = user_id
    try:
        paths = workspace.resolve(conjecture_dir, user_id)
    except workspace.ConjectureNotFound as e:
        parser.error(str(e))
    CONJECTURE_FILE = str(paths.conjecture)
    CONJECTURE_ROOT = str(paths.root)
    DAGS_DIR = str(paths.dags_dir)
    DAG_FILE = str(paths.dag)
    REFERENCES_FILE = str(paths.references)
    COMMENTS_FILE = str(paths.comments)
    PROMPT_PATHS = {name: str(p) for name, p in paths.prompts.items()}
    log(workspace.describe(paths), verbose)


def verify_entry(
    *,
    path: str,
    lemma_id: str,
    owner: str,
    root: str,
    full: bool,
    user_id: str,
    model: str,
    host: Optional[str],
    api_key: Optional[str],
    backend: str,
    num_ctx: Optional[int],
    verbose: bool,
    parser: argparse.ArgumentParser,
) -> int:
    """proofs verify PATH lemma_id [--full]: the three verifier checks on
    the stored proof of a lemma that is already in the DAG, with the
    refutation and certificate bookkeeping that verify carries — a
    rejection writes a refutation file, an acceptance issues a certificate
    and dismisses the refutations whose justifications were on the table.

    PATH is the lemma's user's DAG file, which supplies the lemma's
    user_id (owner, the file's name the way load_dag reads it), and the
    conjecture's root (root) is the file's grandparent, so the
    verification runs against the conjecture the lemma's DAG belongs to.
    With --full the lemma's cited lemmas are verified first, in
    dependency order, the target last, and a rejection anywhere in that
    order stops the verification. Returns the exit code: 0 when every
    lemma verified was accepted, 1 when one was rejected (its refutation
    written), EXIT_NO_VERDICT when a verifier gave no verdict (nothing
    written; re-run once the server answers).
    """
    _verify_session(
        conjecture_dir=root, user_id=user_id, model=model, host=host,
        api_key=api_key, backend_choice=backend, num_ctx=num_ctx,
        verbose=verbose, parser=parser,
    )
    dag = load_dag()
    references = load_references()
    m = merkle.Merkle(dag["lemmas"], references)
    if not m.has(owner, lemma_id):
        parser.error(
            f"{owner}:{lemma_id} (from {path}) is not a lemma of the DAG {root}"
        )
    order = _citation_closure(dag, owner, lemma_id) if full else [(owner, lemma_id)]
    # Suspension is computed, not stored, and it does not reach in here:
    # verify loads the suspended lemmas too (see suspension.py) —
    # it is the path back to them. The warning names a lemma the run must
    # not build on as one.
    susp = suspension.compute(dag["lemmas"], references, CONJECTURE_ROOT)
    for u, lid in order:
        if susp.has(u, lid):
            log(
                f"{u}:{lid} is suspended ({'; '.join(susp.reasons(u, lid))}); "
                f"it is loaded and verified anyway — the verification is how "
                f"the suspension is lifted.",
                verbose,
            )
    conjecture_text = load_file(CONJECTURE_FILE)
    verifier_sys = _verifier_prompts(parser)
    log(
        f"Verifying {len(order)} lemma(s) for {owner}: "
        + ", ".join(lid for _u, lid in order),
        verbose,
    )
    try:
        for u, lid in order:
            outcome, _just = verify_stored_lemma(
                dag, references, u, lid, conjecture_text, verifier_sys, verbose
            )
            if outcome == "rejected":
                log(f"{lid} rejected; verification stops here.", verbose)
                return 1
            if outcome == "no_verdict":
                log(
                    f"{lid} got no verdict; verification stops here, "
                    f"with nothing written.",
                    verbose,
                )
                return EXIT_NO_VERDICT
    except merkle.MerkleCycleError as e:
        log(f"{e}", verbose)
        return 1
    return 0


def repair_entry(
    *,
    path: str,
    user_id: str,
    model: str,
    host: Optional[str],
    api_key: Optional[str],
    backend: str,
    num_ctx: Optional[int],
    verbose: bool,
    parser: argparse.ArgumentParser,
    lemmas_all: int = LEMMAS_ALL,
    my_lemmas: int = MY_LEMMAS,
    tool_mode: str = TOOL_MODE,
    lemmas_common: int = LEMMAS_COMMON,
) -> int:
    """proofs repair path_to_refutation (see refutations.py,
    repair_entry()): the refutation file supplies the lemma's user_id and
    lemma_id, so nothing else names the lemma.

    The first step is to re-verify the existing proof with the
    refutation's justifications as input — the verification a stored
    lemma gets from proofs verify, in which the lemma's refutations are
    on the table and the file addressed carries its justification. If the
    proof is accepted, the refutation is dismissed and its file deleted
    (the acceptance dismisses the lemma's refutations and re-issues its
    certificate), and the repair stops.

    If it is rejected, a new refutation file is written for the new
    rejection, and the lemma is re-proven through the prover, verifier
    and reviser, the rejection's justification as the prover's first
    feedback. The repair re-proves the lemma, and the lemma keeps its
    id, so the engine runs with decomposition off: the reviser's options
    are keep and revise only. An accepted proof is committed in place —
    the same user_id and lemma_id, in the owner's file, the one write to
    another user's DAG file — certified, and the lemma's refutations
    dismissed.

    A new proof under the same statement changes only the lemma's own
    hash — a dependent's hash covers the statements it cites, not their
    proofs (see merkle.py) — so its dependents keep their certificates.
    When the reviser revised the statement, the lemmas citing it lose
    theirs the moment the in-place commit happens. The repair therefore
    climbs the dependency chain in dependency order, re-verifying each
    dependent whose certificate no longer matches and re-proving a
    rejected one the same way, its own rejection as the justification. A
    lemma that cannot be re-proved keeps the proof it had and the
    refutation that counts against it, and the climb goes on, so it and
    its dependents stay suspended.

    Returns the exit code: 0 when the lemma was re-verified or re-proved
    and every dependent re-verified or re-proved; 1 when a re-proof
    failed (its lemma and its dependents stay suspended) or a
    re-verification could not hash the lemma; EXIT_NO_VERDICT when a
    verifier gave no verdict (the server failed, or its reply could not be
    read). A missing verdict is not a rejection: no refutation is written,
    nothing is re-proved, and the repair stops where it is — the lemmas
    already repaired stand and the rest are left as they were. On the
    named lemma, the refutation file is still there, so the same command
    can be re-run; on a dependent, the named lemma's refutation is already
    dismissed, so the log lists the dependents still to verify with proofs
    verify.
    """
    # Resolved first, as cli.py resolves PATH: the conjecture root is the
    # refutation file's grandparent, which a bare relative name (run inside
    # refutations/) does not have until it is made absolute.
    p = Path(path).resolve()
    try:
        ref = refutations.load(p)
    except (OSError, ValueError) as e:
        parser.error(f"{path} is not a readable refutation file: {e}")
    owner = str(ref.get("user_id") or "")
    lemma_id = str(ref.get("lemma_id") or "")
    if not owner or not lemma_id:
        parser.error(
            f"{path} names no lemma (its user_id and lemma_id are required)"
        )
    if p.parent.name != refutations.DIR_NAME:
        parser.error(f"{path} is not in a {refutations.DIR_NAME}/ directory")
    root = p.parent.parent
    global LEMMAS_ALL, MY_LEMMAS, TOOL_MODE, LEMMAS_COMMON
    LEMMAS_ALL, MY_LEMMAS, TOOL_MODE = lemmas_all, my_lemmas, tool_mode
    LEMMAS_COMMON = lemmas_common
    _verify_session(
        conjecture_dir=str(root), user_id=user_id, model=model, host=host,
        api_key=api_key, backend_choice=backend, num_ctx=num_ctx,
        verbose=verbose, parser=parser,
    )
    dag = load_dag()
    references = load_references()
    m = merkle.Merkle(dag["lemmas"], references)
    if not m.has(owner, lemma_id):
        parser.error(
            f"{owner}:{lemma_id} (named by {p.name}) is not a lemma of the "
            f"DAG {root}; there is nothing to repair"
        )
    log(f"Repairing {owner}:{lemma_id} from {p.name}", verbose)
    # The lemma a refutation names is usually suspended (the refutation may
    # be stale, but the files say so only once they are read), and the
    # repair is precisely the case the suspension warns about (see
    # suspension.py).
    susp = suspension.compute(dag["lemmas"], references, CONJECTURE_ROOT)
    if susp.has(owner, lemma_id):
        log(
            f"{owner}:{lemma_id} is suspended ({'; '.join(susp.reasons(owner, lemma_id))}); "
            f"the repair re-verifies it anyway — an acceptance re-issues the "
            f"certificate and lifts the suspension.",
            verbose,
        )
    conjecture_text = load_file(CONJECTURE_FILE)
    if not conjecture_text:
        log(
            f"{CONJECTURE_FILE} is empty; the re-verification and the "
            f"re-proof go on without the conjecture, the lemma's "
            f"statement and the DAG being the substance.",
            verbose,
        )
    verifier_sys = _verifier_prompts(parser)
    # Ctrl-C stays a KeyboardInterrupt (caught below), but one landing
    # while a re-proof is rewritten in place and certified waits for both.
    global _DEFERRED_INTERRUPT
    previous_handler = signal.signal(signal.SIGINT, _on_sigint_repair)
    _DEFERRED_INTERRUPT = _raise_interrupt
    # Whether any lemma was re-proved in place: its replaced proof's
    # certificate is now stale, and the repair ends with the prune hint.
    reproved = False
    try:
        try:
            outcome, just = verify_stored_lemma(
                dag, references, owner, lemma_id, conjecture_text,
                verifier_sys, verbose,
            )
        except merkle.MerkleCycleError as e:
            log(f"{e}", verbose)
            return 1
        if outcome == "no_verdict":
            # Not a rejection: re-proving would replace a proof nobody
            # rejected.
            log(
                f"{owner}:{lemma_id} got no verdict; the repair stops "
                f"with nothing written. Re-run it once the server answers.",
                verbose,
            )
            return EXIT_NO_VERDICT
        if outcome == "accepted":
            # The acceptance already dismissed the lemma's refutations
            # (the addressed file among them) and re-issued its
            # certificate.
            if p.is_file():
                refutations.delete(p)
                log(f"Refutation dismissed: {p}", verbose)
            log(f"{owner}:{lemma_id} accepted; the repair is done.",
                verbose)
            return 0

        # The re-verification rejected the stored proof and wrote a new
        # refutation for the rejection. The repair's next step:
        # re-prove the lemma through the prover, verifier and reviser,
        # the rejection's justification as the prover's first feedback,
        # and on acceptance commit the new proof in place under the same
        # pair.
        log(
            f"{owner}:{lemma_id} rejected on re-verification; "
            f"re-proving it through the prover, verifier and reviser.",
            verbose,
        )
        prover_sys = load_file(PROMPT_PATHS["prover.md"])
        reviser_sys = load_file(PROMPT_PATHS["reviser.md"])
        empty = [
            name for name, text in (
                ("prover.md", prover_sys), ("reviser.md", reviser_sys),
            ) if not text
        ]
        if empty:
            parser.error(f"no prompt text for: {', '.join(empty)}")
        references, reference_block, _planner_ref_block, reviser_ref_block = (
            _build_reference_blocks(verbose)
        )
        ref_ids = {str(r["id"]) for r in references}

        def reprove(u: str, lid: str, justification: Optional[str]) -> bool:
            """The repair's re-proof of the stored lemma (u, lid):
            the engine's prover -> verifier -> reviser rounds on the
            lemma's statement as it stands in the file, the rejection's
            justification as the prover's first feedback, decomposition
            off — the repair re-proves the lemma and the lemma keeps its
            id, so the reviser's choices are keep and revise only. An
            accepted proof is committed in place in the owner's file
            under the same pair, certified the way an acceptance
            certifies, and the lemma's refutations dismissed. Returns
            whether the proof was accepted; a failure leaves the stored
            proof and its counting refutation in the file, the lemma
            suspended.
            """
            nonlocal reproved
            dag0 = load_dag()
            node = dag0["lemmas"].get((u, lid))
            if node is None:
                log(
                    f"{u}:{lid} is not in the DAG any more; nothing "
                    f"to re-prove.",
                    verbose,
                )
                return False
            # The engine recomputes the suspension every round: the lemma
            # is suspended (that is why it is being re-proved), and so are
            # its dependents, so the prover and the reviser see only the
            # lemmas that are not — the lemma cannot cite itself.
            initial = {
                "lemma_id": lid,
                "claimed_id": lid,
                "target": {
                    "id": lid,
                    "statement": str(node.get("statement") or ""),
                },
                "attempt": 1,
                # The rejection's justification is the prover's first
                # feedback, the engine's most-recent-failure-only as it
                # was when the lemma's proof was last rejected.
                "feedback": (
                    [f"Verifier: {justification}"] if justification else []
                ),
                "attempt_notes": [],
                "last_proof": "",
            }

            def on_proof(
                proof_lid: str, proof_node: Dict[str, Any],
                verified_hash: Optional[str],
            ) -> str:
                with _commit_section():
                    _replace_lemma_in_owner_file(
                        u, proof_lid, proof_node, verified_hash, verbose,
                    )
                return proof_lid

            result = run_proof_loop(
                verbose=verbose,
                conjecture=conjecture_text,
                prover_sys=prover_sys,
                reviser_sys=reviser_sys,
                reference_block=reference_block,
                reviser_ref_block=reviser_ref_block,
                references=references,
                verifier_sys=verifier_sys,
                ref_ids=ref_ids,
                initial=initial,
                fresh_target=None,
                on_round=lambda state: None,
                on_proof=on_proof,
                failed_attempts={},
                failed_lock=threading.Lock(),
                get_dag=load_dag,
                toggle_hotkey=lambda: None,
                allow_decomposition=False,
            )
            if not result["proved"]:
                log(
                    f"{u}:{lid} could not be re-proved after "
                    f"{MAX_PROOF_ATTEMPTS} prover round(s); the stored "
                    f"proof stays, with its refutation counting against "
                    f"it.",
                    verbose,
                )
                for note in result["attempt_notes"]:
                    log(f"   · {note}", verbose)
                return False
            # The acceptance put every objection on the table: the
            # lemma's refutations are dismissed, the addressed file among
            # them, the way the re-verification's acceptance would have
            # dismissed them.
            for ref_path, _ref in refutations.for_lemma(CONJECTURE_ROOT, u, lid):
                try:
                    refutations.delete(ref_path)
                    log(f"Refutation dismissed: {ref_path}", verbose)
                except OSError as e:
                    log(
                        f"Could not delete the refutation "
                        f"{ref_path}: {e}",
                        verbose,
                    )
            reproved = True
            log(
                f"{u}:{lid} re-proved in place; the new proof stands "
                f"under its own id.",
                verbose,
            )
            return True

        if not reprove(owner, lemma_id, just):
            log(
                f"The repair of {owner}:{lemma_id} failed; it and the "
                f"lemmas that depend on it stay suspended.",
                verbose,
            )
            return 1

        # A new proof under the same statement changes only the lemma's
        # own hash: a dependent's hash covers the statements it cites,
        # not their proofs, so its certificate still matches and nothing
        # above needs verifying. A statement the reviser revised does
        # change the hash of every lemma citing it, and those lose their
        # certificates. Climb the dependency chain in dependency order,
        # re-verifying each dependent whose certificate no longer matches
        # — checked as the climb reaches it, so a dependent whose own
        # statement is revised on the way takes its citers along — and
        # re-proving a rejected one the same way, its own rejection as the
        # justification. A dependent that cannot be re-proved stays
        # suspended and the climb goes on.
        chain = _dependent_chain(load_dag(), (owner, lemma_id))
        if not chain:
            log(
                f"The repair is done: {owner}:{lemma_id} re-proved, "
                f"and nothing depends on it.",
                verbose,
            )
            return 0
        log(
            f"Checking the {len(chain)} dependent(s) of {owner}:{lemma_id} "
            f"in dependency order, re-verifying those whose certificates "
            f"no longer match: "
            + ", ".join(f"{u}:{lid}" for u, lid in chain)
            + ".",
            verbose,
        )
        failed: List[Tuple[str, str]] = []
        reverified = 0
        for u, lid in chain:
            dag = load_dag()
            try:
                h = merkle.Merkle(dag["lemmas"], references).hash(u, lid)
            except merkle.MerkleCycleError as e:
                log(f"{e}", verbose)
                failed.append((u, lid))
                continue
            if any(
                certificates.is_valid(c, h)
                for c in certificates.load_all(CONJECTURE_ROOT)
                if (c.get("user_id"), c.get("lemma_id")) == (u, lid)
            ):
                continue
            reverified += 1
            try:
                dep_outcome, dep_just = verify_stored_lemma(
                    dag, references, u, lid, conjecture_text, verifier_sys,
                    verbose,
                )
            except merkle.MerkleCycleError as e:
                log(f"{e}", verbose)
                failed.append((u, lid))
                continue
            if dep_outcome == "no_verdict":
                # Most likely the server is down, and every later check
                # would fail the same way: stop the climb rather than
                # spend it. The repaired lemma stands (its refutation is
                # already dismissed, so this repair cannot be re-run); the
                # dependents not yet verified stay suspended until proofs
                # verify accepts each.
                rest = chain[chain.index((u, lid)):]
                log(
                    f"{u}:{lid} got no verdict; the climb stops with "
                    f"nothing written for it. Still to verify, with proofs "
                    f"verify once the server answers: "
                    + ", ".join(f"{a}:{b}" for a, b in rest)
                    + (f". Already unproved: "
                       + ", ".join(f"{a}:{b}" for a, b in failed)
                       if failed else "")
                    + ".",
                    verbose,
                )
                return EXIT_NO_VERDICT
            if dep_outcome == "accepted":
                continue
            log(
                f"{u}:{lid} rejected on re-verification; re-proving it "
                f"the same way.",
                verbose,
            )
            if not reprove(u, lid, dep_just):
                failed.append((u, lid))
        if failed:
            log(
                f"The repair finished with {len(failed)} lemma(s) "
                f"unproved: "
                + ", ".join(f"{u}:{lid}" for u, lid in failed)
                + "; they and their dependents stay suspended.",
                verbose,
            )
            return 1
        log(
            f"The repair is done: {owner}:{lemma_id} re-proved; "
            f"{reverified} of its {len(chain)} dependent(s) needed "
            f"re-verifying, the rejected ones re-proved.",
            verbose,
        )
        return 0
    except KeyboardInterrupt:
        log(
            "\nInterrupted; the lemmas already repaired stand, and the "
            "rest keep the proofs they had, suspended.",
            verbose,
        )
        return 130
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        _DEFERRED_INTERRUPT = None
        if reproved:
            _prune_hint(root, verbose)


def _prune_hint(root: Path, verbose: bool) -> None:
    """After a repair that re-proved a lemma: say how many of the current
    user's certificates no longer match — the lines proofs prune would
    drop — and the command that drops them. A re-proof leaves the
    certificate of the proof it replaced in the file (certificates are a
    record, never rewritten by an acceptance), and the repair itself does
    not prune: that is the user's call, the way it is everywhere else.
    Silent when nothing is stale, or when the files cannot be read."""
    try:
        dag = load_dag()
        m = merkle.Merkle(dag["lemmas"], load_references())
        lines = certificates.load(certificates.file_for(CONJECTURE_ROOT, USER))
    except (OSError, ValueError):
        return
    stale = 0
    for line in lines:
        # prune judges only the lines the user issued (see
        # certificates.prune); a hand-edited line for another verifier
        # is kept, so it is not counted here either.
        if str(line.get("verifier") or "") != USER:
            continue
        pair = (str(line.get("user_id") or ""), str(line.get("lemma_id") or ""))
        try:
            h: Optional[str] = m.hash(*pair)
        except (KeyError, merkle.MerkleCycleError):
            h = None
        if h is None or not certificates.is_valid(line, h):
            stale += 1
    if stale:
        try:
            shown = os.path.relpath(root)
        except ValueError:  # another drive (Windows): keep it absolute
            shown = str(root)
        log(
            f"{stale} of your certificate(s) no longer match a lemma (the "
            f"replaced proofs' among them). They are harmless, but to drop "
            f"them run: proofs prune {shown}",
            verbose,
        )


@contextlib.contextmanager
def _all_dag_file_locks():
    """Hold DAG_LOCK and the file lock (locking.py) on every DAG file in
    dags/, taken in sorted order, for the block. The lock order is DAG
    files before certificate files, everywhere (see locking.py): a commit
    holds one DAG file's lock while it writes the certificate, so this
    takes no certificate lock until every DAG lock is held."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(DAG_LOCK)
        for path in sorted(glob.glob(os.path.join(DAGS_DIR, "*_dag.json"))):
            stack.enter_context(locking.file_lock(path))
        yield


def prune_entry(
    *,
    dir: str,
    user_id: str,
    parser: argparse.ArgumentParser,
) -> int:
    """proofs prune DIR: recompute the Merkle hash of every lemma in the
    conjecture and drop the current user's certificates that no longer
    match.

    No model is involved — the hash is a function of the DAG files and
    references.md alone, the way certificates are checked everywhere else
    (see certificates.py) — so this entry sets only the file globals
    load_dag() and load_references() read: no backend, no probe, no
    prompt. It writes exactly one file, the user's own certificate file,
    and never another user's: only the certificates the user issued as
    verifier are pruned, whoever owns the lemmas they cover.

    A line goes when its lemma's hash has moved (the proof, the statement,
    or the statement of something it cites changed), when the lemma is no
    longer in the
    DAG at all, or when the lemma has no hash any more (a citation cycle,
    the case suspension.compute reads as "no certificate is valid for
    it") — a certificate that matches nothing is the stale record prune
    exists to clear, and the log names which case each line was.

    Returns the exit code: 0 when the prune ran (including a no-op), 1
    when the user's certificate file could not be read or rewritten.
    """
    global CONJECTURE_ROOT, DAGS_DIR, DAG_FILE, REFERENCES_FILE, USER
    try:
        paths = workspace.resolve(dir, user_id)
    except workspace.ConjectureNotFound as e:
        parser.error(str(e))
    CONJECTURE_ROOT = str(paths.root)
    DAGS_DIR = str(paths.dags_dir)
    DAG_FILE = str(paths.dag)
    REFERENCES_FILE = str(paths.references)
    USER = user_id
    log(workspace.describe(paths))
    # Every DAG file stays locked from the load to the rewrite. A commit
    # certifies before it writes the lemma (_certify_before_write), so for
    # a moment a fresh certificate names a lemma no DAG file holds yet;
    # read in that moment, it would look like a line for a lemma that is
    # gone, and go. The commit holds the lock on the file it writes for
    # both of its writes, so with every file locked here, prune sees each
    # commit either not started or finished. proofs repair writes another
    # user's file, which is why it is every file and not the user's own.
    try:
        with _all_dag_file_locks():
            dag = load_dag()
            references = load_references()
            m = merkle.Merkle(dag["lemmas"], references)
            # Every lemma's current hash, a pair the DAG holds but cannot
            # hash (a citation cycle) kept as None: its certificates match
            # nothing either, and the map still says the lemma was there,
            # so the log can name the reason. Per pair, the way
            # suspension.compute does it: Merkle.all() would raise on the
            # first cycle and hash nothing after.
            hashes: Dict[Tuple[str, str], Optional[str]] = {}
            for pair in dag["lemmas"]:
                try:
                    hashes[pair] = m.hash(*pair)
                except merkle.MerkleCycleError:
                    hashes[pair] = None
            kept, dropped = certificates.prune(
                CONJECTURE_ROOT, USER, hashes, user=_user_header()
            )
    except OSError as e:
        log(f"Could not read or rewrite {USER}'s certificate file: {e}")
        return 1
    if dropped:
        for line in dropped:
            pair = (
                str(line.get("user_id") or ""),
                str(line.get("lemma_id") or ""),
            )
            if pair not in dag["lemmas"]:
                reason = "the lemma is no longer in the DAG"
            elif hashes.get(pair) is None:
                reason = "the lemma has no current hash (a citation cycle)"
            else:
                reason = "its hash no longer matches the lemma's"
            log(f"Dropped certificate for {pair[0]}:{pair[1]} — {reason}.")
        log(f"Pruned {len(dropped)} certificate(s) for {USER}; {len(kept)} kept.")
    else:
        log(f"Nothing to prune: all {len(kept)} certificate(s) for {USER} still match.")
    return 0


def status_entry(
    *,
    path: str,
    lemma_id: str,
    owner: str,
    root: str,
    user_id: str,
    parser: argparse.ArgumentParser,
) -> int:
    """proofs status PATH lemma_id: for each lemma in the named lemma's
    dependency closure, list every valid certificate, each with its
    verifier, model, date and count — then whether the conjecture is
    settled.

    No model is involved — a certificate is valid only while its hash
    matches the lemma's current Merkle hash (see certificates.py), and that
    hash is a function of the DAG files and references.md alone, the way
    prune's recomputation is — so this entry sets only the file globals
    load_dag() and load_references() read: no backend, no probe, no prompt.
    It reads the DAG files, references.md and the certificate files, and
    writes nothing.

    The closure is the lemma's citation closure in dependency order
    (_citation_closure): the dependencies first, the target last, the order
    proofs verify --full verifies in. A certificate is valid for a lemma
    only while its hash matches the lemma's current Merkle hash
    (certificates.is_valid), and the pair's lines are spread over every
    verifier's file, certificates/<verifier>.jsonl, the way record() writes
    them, so all the files are read (certificates.load_all); each valid line
    is listed with its verifier, model, date and count. A line for the pair
    whose hash no longer matches is stale — the record of an acceptance of a
    version of the lemma that is no longer in the files — and is counted,
    not listed. A lemma whose hash cannot be computed (a citation cycle) has
    no valid certificate: no hash exists for a line to match, the case prune
    logs the same way.

    Returns the exit code: 0 when the status was reported.
    """
    global CONJECTURE_ROOT, DAGS_DIR, DAG_FILE, REFERENCES_FILE, USER
    try:
        paths = workspace.resolve(root, user_id)
    except workspace.ConjectureNotFound as e:
        parser.error(str(e))
    CONJECTURE_ROOT = str(paths.root)
    DAGS_DIR = str(paths.dags_dir)
    DAG_FILE = str(paths.dag)
    REFERENCES_FILE = str(paths.references)
    USER = user_id
    log(workspace.describe(paths))
    dag = load_dag()
    references = load_references()
    m = merkle.Merkle(dag["lemmas"], references)
    if not m.has(owner, lemma_id):
        parser.error(
            f"{owner}:{lemma_id} (from {path}) is not a lemma of the DAG {root}"
        )
    order = _citation_closure(dag, owner, lemma_id)
    try:
        all_lines = certificates.load_all(CONJECTURE_ROOT)
    except OSError as e:
        log(f"Could not read the certificate files: {e}")
        all_lines = []
    # Per lemma, the way prune_entry computes it: a pair the DAG holds
    # but cannot hash (a citation cycle) is kept as None, and its status
    # is "no certificate is valid for it", not a crash.
    hashes: Dict[Tuple[str, str], Optional[str]] = {}
    for u, lid in order:
        try:
            hashes[(u, lid)] = m.hash(u, lid)
        except merkle.MerkleCycleError:
            hashes[(u, lid)] = None
    log(
        f"Status for {owner}:{lemma_id}: {len(order)} lemma(s) in its "
        f"dependency closure, dependencies first:"
    )
    for i, (u, lid) in enumerate(order, 1):
        h = hashes[(u, lid)]
        if h is None:
            log(f"  {i}/{len(order)} {u}:{lid}")
            log(
                "     no Merkle hash (a citation cycle) — no "
                "certificate is valid for it"
            )
            continue
        of_lemma = [
            c for c in all_lines
            if str(c.get("user_id") or "") == u
            and str(c.get("lemma_id") or "") == lid
        ]
        valid = [c for c in of_lemma if certificates.is_valid(c, h)]
        stale = [c for c in of_lemma if not certificates.is_valid(c, h)]
        log(f"  {i}/{len(order)} {u}:{lid}  hash {h}")
        if valid:
            for c in valid:
                log(
                    f"     verifier={c.get('verifier') or ''} "
                    f"model={c.get('model') or ''} "
                    f"date={c.get('date') or ''} "
                    f"count={c.get('count')}"
                )
        else:
            log("     (no valid certificates)")
        if stale:
            log(
                f"     {len(stale)} stale certificate(s) for an earlier "
                f"version of this lemma are not listed"
            )
    # Whether the conjecture is settled, recomputed the way the run
    # computes it (resolution.py): an unsuspended lemma, anyone's, stating
    # conjecture.md (or its negation) verbatim.
    susp = suspension.compute(dag["lemmas"], references, CONJECTURE_ROOT)
    if _settled(dag, susp.suspended, load_file(paths.conjecture), True) is None:
        log("Conjecture: open (no certified lemma states it or its negation).")
    return 0


# ----------------------------------------------------------------------------
# Export: the LaTeX document of a DAG (see export.py)
# ----------------------------------------------------------------------------
def export_entry(
    *,
    path: str,
    lemma_id: Optional[str],
    owner: Optional[str],
    out_dir: str,
    root: str,
    user_id: str,
    parser: argparse.ArgumentParser,
) -> int:
    """proofs export PATH [lemma_id] (see export.py): write the
    LaTeX document of the DAG, in dependency order.

    No model is involved — the document is a function of the DAG files,
    references.md and conjecture.md alone, the way a Merkle hash is a
    function of the files' contents — so this entry sets only the file
    globals load_dag() and load_references() read: no backend, no probe,
    no prompt. It reads those files and writes exactly one: the LaTeX
    document in the out_dir directory, named by export.name_for for what
    PATH names — proof.tex for the complete DAG,
    <user_id>_proof.tex for a user's lemmas,
    <user_id>--<lemma_id>_proof.tex for a lemma and its dependencies.
    out_dir is the directory the user asked the document to be written to
    (-o/--out), created if it is not there yet: the export is the user's
    artifact, not the project's data, so it never lands in the project
    unless told to. The document's lemmas come in
    dependency order, dependencies first, the order the document reads in:
    a section always stands on the sections above it when they are its
    dependencies.

    What PATH names is what the document holds (see export.py): with owner
    None (PATH was the conjecture directory) it is the complete DAG, every
    user's lemmas; with owner set (PATH was that user's DAG file) it is that
    user's lemmas, or — with lemma_id — that lemma and everything it cites,
    whichever user's file each dependency is in, the _citation_closure
    order, the same order proofs verify --full verifies in. A user's
    lemmas and the complete DAG are ordered by _dependency_order over the
    set they are: an export of one user's
    lemmas writes that user's lemmas, and the dependencies in the other
    users' files are not pulled in — they are cross-referenced, and each
    user's export writes its own.

    Returns the exit code: 0 when the document was written.
    """
    global CONJECTURE_ROOT, DAGS_DIR, DAG_FILE, REFERENCES_FILE, USER
    try:
        paths = workspace.resolve(root, user_id)
    except workspace.ConjectureNotFound as e:
        parser.error(str(e))
    CONJECTURE_ROOT = str(paths.root)
    DAGS_DIR = str(paths.dags_dir)
    DAG_FILE = str(paths.dag)
    REFERENCES_FILE = str(paths.references)
    USER = user_id
    log(workspace.describe(paths))
    dag = load_dag()
    references = load_references()
    lemmas = dag["lemmas"]
    if owner is None:
        order = _dependency_order(dag, set(lemmas))
        users = {u for u, _ in order}
        name = export.name_for()
        scope = (
            f"the complete DAG — {len(order)} lemma(s) from "
            f"{len(users)} user(s)"
        )
    elif lemma_id is None:
        order = _dependency_order(
            dag, {p for p in lemmas if p[0] == owner}
        )
        name = export.name_for(owner=owner)
        scope = f"{owner}'s lemmas — {len(order)} lemma(s)"
    else:
        if (owner, lemma_id) not in lemmas:
            parser.error(
                f"{owner}:{lemma_id} (from {path}) is not a lemma of the "
                f"DAG {root}"
            )
        order = _citation_closure(dag, owner, lemma_id)
        name = export.name_for(owner=owner, lemma_id=lemma_id)
        scope = (
            f"{owner}'s lemma {lemma_id} and everything it cites — "
            f"{len(order)} lemma(s)"
        )
    document = export.render(
        conjecture_name=paths.root.name,
        conjecture=load_file(paths.conjecture),
        scope=scope,
        order=order,
        lemmas=lemmas,
        references=references,
    )
    out = export.write(Path(out_dir) / name, document)
    log(
        f"Exported {len(order)} lemma(s) to {out} — dependency order, "
        f"dependencies first."
    )
    return 0


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
# Commit sections: the commit of an accepted lemma and its certificate are
# one step that Ctrl-C must not split — a lemma committed without its
# certificate is suspended, and a resume would re-prove it into id_2.
# _commit_section() counts the sections in progress, in any thread; the
# SIGINT handler defers the interrupt until the count is back to zero (the
# serial run replays it at the section's end; the parallel run waits for
# the worker threads' sections to finish, at most COMMIT_WAIT_SECONDS).
_COMMIT_COND = threading.Condition()
_COMMITS_IN_PROGRESS = 0
_SIGINT_PENDING = False
COMMIT_WAIT_SECONDS = 30
# What a deferred interrupt does when its commit section ends: None is the
# run's (_interrupt_serial: checkpoint, exit 130); proofs repair, which has
# no checkpoint, sets its own.
_DEFERRED_INTERRUPT = None
# The live ParallelState, or None outside a parallel run. The SIGINT handler
# checks it first so a Ctrl-C in parallel mode snapshots the shared state
# (version-2 checkpoint) instead of the serial _LIVE_STATE, which a parallel
# run never updates.
_PARALLEL: "ParallelState" = None
# The DAG file lock, held across load-then-save by every DAG writer that
# can run concurrently with another: the serial success boundary and each
# parallel loop's. Two writers can't clobber each other's addition —
# commit_lemma_to_dag checks the id under the lock, and a writer that
# finds its id taken commits under a fresh one. Readers do not take it:
# load_dag() reads the files as they stand (fresh per round), which is why
# a commit certifies before it writes (_certify_before_write) — a reader
# can catch the moment between the two, and in that order it sees nothing
# wrong there.
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
    """The checkpoint file for a DAG file:
    dags/<user_id>_dag.json -> dags/<user_id>_dag.checkpoint.json.

    Beside the user's DAG file, so the checkpoint is per user the way the
    DAG file is, and gitignored by the *.checkpoint.json rule like the
    temporary files beside it.
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
        log(f"Checkpoint {CHECKPOINT_FILE} is unreadable ({e}); ignoring it.",
            verbose)
        discard_checkpoint()
        return None
    if not isinstance(data, dict):
        log(f"Checkpoint {CHECKPOINT_FILE} is malformed; ignoring it.",
            verbose)
        discard_checkpoint()
        return None
    version = data.get("version")
    if version == 2:
        # A parallel-mode checkpoint is valid state for a parallel run; the
        # serial loader must not read it as (and discard it as) a corrupt
        # serial one. Re-running without --parallel is a different shape of
        # run, so the honest answer is "re-run with --parallel".
        log(f"{os.path.basename(CHECKPOINT_FILE)} is a parallel-mode "
            f"checkpoint; re-run with --parallel to resume it.", verbose)
        return None
    if version not in (None, 1):
        log(f"Checkpoint {CHECKPOINT_FILE} has an unknown version "
            f"({version!r}); ignoring it.", verbose)
        discard_checkpoint()
        return None
    iteration = data.get("iteration")
    if not isinstance(iteration, int) or iteration < 1:
        log(f"Checkpoint {CHECKPOINT_FILE} has no usable iteration; ignoring it.",
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
                "claimed_id": str(raw.get("claimed_id") or lemma_id),
                "target": clean_target,
                "attempt": attempt,
                "feedback": [n for n in (raw.get("feedback") or []) if isinstance(n, str)],
                "attempt_notes": [n for n in (raw.get("attempt_notes") or []) if isinstance(n, str)],
                "last_proof": str(raw.get("last_proof") or ""),
            }
        else:
            log(
                "Checkpoint's in-flight state is unusable (bad lemma, "
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
    just rejected — what the reviser judges its difficulty by). The claim
    id is kept beside the lemma id for the same reason the engine does:
    a decomposition may have moved the prover off the claimed id, and the
    buffer slot to free on settle is keyed on the claim. Checkpoints from
    before the field existed fall back to the lemma id.
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
        "claimed_id": str(raw.get("claimed_id") or lemma_id),
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
        log(f"Checkpoint {CHECKPOINT_FILE} is unreadable ({e}); ignoring it.",
            verbose)
        return None
    if not isinstance(data, dict) or data.get("version") != 2:
        log(f"{os.path.basename(CHECKPOINT_FILE)} is not a parallel-mode "
            f"(v2) checkpoint; ignoring it.", verbose)
        return None
    raw_loops = data.get("loops")
    if not isinstance(raw_loops, dict) or set(raw_loops) != set(loop_ids):
        log(f"{os.path.basename(CHECKPOINT_FILE)} was written by a run "
            f"with a different set of loops; ignoring it.", verbose)
        return None
    failed_raw = data.get("failed_attempts")
    if not isinstance(failed_raw, dict):
        log(f"{os.path.basename(CHECKPOINT_FILE)} has no usable "
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
            log(f"{os.path.basename(CHECKPOINT_FILE)} has a malformed "
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
        log(f"Could not write checkpoint {CHECKPOINT_FILE}: {e}", verbose)
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


@contextlib.contextmanager
def _commit_section():
    """Hold off Ctrl-C while an accepted lemma is committed and certified
    (see _COMMIT_COND). In the serial run an interrupt that arrived inside
    is replayed when the section ends."""
    global _COMMITS_IN_PROGRESS, _SIGINT_PENDING
    with _COMMIT_COND:
        _COMMITS_IN_PROGRESS += 1
    try:
        yield
    finally:
        with _COMMIT_COND:
            _COMMITS_IN_PROGRESS -= 1
            _COMMIT_COND.notify_all()
            replay = (
                _SIGINT_PENDING
                and _COMMITS_IN_PROGRESS == 0
                and threading.current_thread() is threading.main_thread()
            )
            if replay:
                _SIGINT_PENDING = False
        if replay:
            (_DEFERRED_INTERRUPT or _interrupt_serial)()


def _raise_interrupt() -> None:
    raise KeyboardInterrupt


def _on_sigint_repair(signum, frame) -> None:
    """Ctrl-C during proofs repair: a KeyboardInterrupt, as with no
    handler at all, except that one landing while a re-proved lemma is
    being rewritten in place and certified is held off until both are
    done (_commit_section) — split, the lemma would be left rewritten and
    uncertified. A second Ctrl-C force-exits."""
    global _sigint_count, _SIGINT_PENDING
    _sigint_count += 1
    if _sigint_count > 1:
        print("\nForced exit.", file=sys.stderr)
        os._exit(130)
    if _COMMITS_IN_PROGRESS:
        _SIGINT_PENDING = True
        print(
            "\nInterrupted — finishing the in-place commit of the "
            "re-proved lemma first (Ctrl-C again to force)...",
            file=sys.stderr,
        )
        return
    raise KeyboardInterrupt


def _on_sigint(signum, frame) -> None:
    """Ctrl-C: write the checkpoint, then exit 130 like the shell expects.

    A second Ctrl-C force-exits without waiting on the file. The checkpoint
    is written from _LIVE_STATE — the last safe boundary run_loop() reached —
    so an interrupt mid-LLM-call loses only that call: the next run re-runs
    the interrupted prover round, not the whole iteration.
    """
    global _sigint_count, _SIGINT_PENDING
    _sigint_count += 1
    if _sigint_count > 1:
        print("\nForced exit.", file=sys.stderr)
        os._exit(130)
    if _PARALLEL is not None:
        # Parallel mode: the shared state is the checkpoint. Ask every thread
        # to stop at its next boundary, let any commit in progress finish
        # (its lemma and its certificate land together), snapshot the whole
        # run under the state lock, and exit; the daemon threads die with
        # the process and the next run resumes from the snapshot.
        _PARALLEL.request_stop()
        print("\nInterrupted — writing checkpoint...", file=sys.stderr)
        with _COMMIT_COND:
            if _COMMITS_IN_PROGRESS:
                print(
                    f"   waiting for {_COMMITS_IN_PROGRESS} lemma commit(s) "
                    f"to finish (Ctrl-C again to force)...",
                    file=sys.stderr,
                )
            _COMMIT_COND.wait_for(
                lambda: _COMMITS_IN_PROGRESS == 0, timeout=COMMIT_WAIT_SECONDS
            )
        _PARALLEL.write_checkpoint()
        print(
            f"   {os.path.basename(CHECKPOINT_FILE)} written. Re-run the same "
            f"command to resume where this run stopped.",
            file=sys.stderr,
        )
        sys.exit(130)
    if _COMMITS_IN_PROGRESS:
        # Serial mode: the handler runs on the main thread, which is inside
        # the commit section itself, so it cannot wait for it — it defers,
        # and the section replays the interrupt as it ends.
        _SIGINT_PENDING = True
        print(
            "\nInterrupted — finishing the commit of the accepted lemma "
            "first (Ctrl-C again to force)...",
            file=sys.stderr,
        )
        return
    _interrupt_serial()


def _interrupt_serial() -> None:
    """The serial interrupt: write the checkpoint from _LIVE_STATE and exit
    130."""
    print("\nInterrupted — writing checkpoint...", file=sys.stderr)
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
            f"\nThe server rejected a call as too long: the prompt alone "
            f"({e.prompt_tokens} tokens) already exceeds its context window "
            f"({e.server_limit} tokens), and this run budgets {budget}. No "
            f"--num-ctx value fixes a prompt that no longer fits — the DAG "
            f"has grown past the model's window. Re-run against a server "
            f"launched with a larger --max-model-len, or start a fresh DAG "
            f"for this conjecture."
        )
    return (
        f"\nThe server rejected a call as too long: its context window is "
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
    LaTeX-dense text and, worse, it is tokenizer-specific — and the users
    sharing a conjecture may each run a different model, with a different
    tokenizer. llm_backend.chars_per_token() reads the running estimate that
    note_usage() revises from each call's real prompt token count, so the
    estimate converges on the truth for whichever model is loaded.

    The budget is set so that prompt + generation lands on
    COMPACT_THRESHOLD * num_ctx: the model may think and write until the window
    is COMPACT_THRESHOLD full, and only then is it cut off for compaction.
    """
    # A tool-call turn carries its calls beside an often-empty content.
    chars = sum(
        len(str(m.get("content") or "")) + len(json.dumps(m.get("tool_calls") or ""))
        for m in messages
    )
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
        log(f"  compaction pass {part_no}/{n_parts} transport error: {e}",
            verbose)
        return prev
    summary = (reply.content or "").strip()
    if reply.truncated or not summary:
        log(f"  compaction pass {part_no}/{n_parts} was truncated or "
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
    log(f"  {role} trace too big to re-send; compacting in "
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
            f"  {role} continuation prompt: task~{int(len(user_prompt)/_ratio)}"
            f" + summary~{int(len(summary)/_ratio)}"
            f" + answer~{int(len(answer)/_ratio)} tokens",
            verbose,
        )
        room = _headroom(messages, REASONING_OPTIONS["num_ctx"])
        if room < COMPACT_MIN_ROOM:
            log(f"  {role} compaction left only ~{room} tokens of room; "
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
            log(f"  {role} continuation transport error: {e}", verbose)
            return "", _partial()
        cont_content, cont_thinking = _split_inline_thinking(
            reply.content or "", reply.thinking or "")
        if reply.truncated:
            if not (cont_content.strip() or cont_thinking.strip()):
                log(f"  {role} continuation was truncated with nothing "
                    f"to build on; the answer does not fit the window.",
                    verbose)
                return "", _partial()
            # The continuation itself hit the wall: not a failure, the next
            # pass. Its trace joins the summary, its content extends the
            # verbatim answer, and the model resumes from the new cut.
            answer += cont_content
            thinking = cont_thinking
            log(f"  {role} continuation hit the wall; the next pass "
                f"compacts its trace and resumes from the new cut.",
                verbose)
            continue
        if not cont_content.strip():
            log(f"  {role} continuation returned empty content.", verbose)
            return "", _partial()
        return (answer + cont_content).strip(), None
    # All `max_passes` passes spent and the answer is still unfinished.
    log(
        f"  {role} still unfinished after {max_passes} compaction passes; "
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
    tools: Optional["tools_mod.ToolSet"] = None,
) -> Tuple[str, str, Optional[Dict[str, str]]]:
    """Stage 1: free-form reasoning. No `format`, so thinking is preserved.

    Returns (content, status, partial). status is "" on success, or "ceiling"
    when the role exhausted the context window without finishing — a signal
    that the task is too large, not that the call failed — or "unavailable"
    when every attempt failed to reach the server (a transport error, never
    a reply): an outage, not the model's failure, which the proof loop
    waits out instead of counting against the lemma. A model that replied
    with nothing, every attempt, is "" with empty content. `partial` is None on
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

    tools, when given, is what the agent may call before it answers
    (tools.py): natively, through the request's `tools` field, or — on a
    server that refuses that, or under --tool-mode json — through the JSON
    protocol the system prompt then describes. Each round's results stay in
    the conversation, up to tools.MAX_TOOL_ROUNDS rounds; then the agent is
    told to answer. A compaction rescue re-sends the task with the results
    folded into it as text, so what the agent looked up survives the cut.
    """
    base_system = system_prompt
    native = bool(tools) and _native_tools()
    if tools and not native:
        system_prompt = base_system + "\n\n" + tools.json_protocol()
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    options: Dict[str, Any] = {
        **REASONING_OPTIONS,
        "temperature": TEMPERATURES.get(role, REASONING_OPTIONS["temperature"]),
    }
    # `think` is passed through as a request. The backend reconciles it with
    # the probed capabilities and never sends False, whatever we ask for.
    want_think = think if (think is not None and think is not False) else None
    # The tool calls made and their results, as text: what a compaction
    # rescue folds into the task it re-sends.
    tool_notes: List[str] = []
    rounds = 0

    def _record(name: str, arguments: Any, result: str) -> None:
        shown = arguments if isinstance(arguments, str) else json.dumps(arguments)
        log(f"  {role} called {name}({shown[:200]})", verbose)
        tool_notes.append(f"{name}({shown}) returned:\n{result}")

    attempt = 0
    # Whether the server ever answered this call: if not, the empty result
    # is an outage ("unavailable"), not the model's.
    answered = False
    while attempt < LLM_MAX_RETRIES:
        offer = bool(tools) and rounds < tools_mod.MAX_TOOL_ROUNDS
        # As much room as the window has left, recomputed for this prompt:
        # the prompt differs every iteration and every role — and grows
        # with each tool round — and there is no point asking for more than
        # the window holds or less than it does.
        options["num_predict"] = _headroom(messages, REASONING_OPTIONS["num_ctx"])
        try:
            if native:
                reply = BACKEND.chat(
                    messages, think=want_think, schema=None, options=options,
                    tools=tools.specs(),
                    tool_choice=None if offer else "none",
                )
            else:
                reply = BACKEND.chat(messages, think=want_think, schema=None,
                                     options=options)
        except llm_backend.ToolsUnsupported as e:
            # The server takes no native tools: the JSON protocol from here
            # on, for this call and every later one. Not an attempt spent.
            _disable_native_tools(str(e), verbose)
            native = False
            system_prompt = base_system + "\n\n" + tools.json_protocol()
            messages[0] = {"role": "system", "content": system_prompt}
            continue
        except (requests.RequestException, ValueError, KeyError) as e:
            attempt += 1
            log(f"  {role} transport error (attempt {attempt}): {e}", verbose)
            continue

        answered = True
        content = reply.content or ""
        thinking = reply.thinking or ""
        content, thinking = _split_inline_thinking(content, thinking)
        hit_ceiling = reply.truncated

        if native and reply.tool_calls and not hit_ceiling:
            if offer:
                rounds += 1
                messages.append({
                    "role": "assistant",
                    "content": reply.content or "",
                    "tool_calls": [
                        {"id": c.id, "type": "function",
                         "function": {"name": c.name, "arguments": c.arguments}}
                        for c in reply.tool_calls
                    ],
                })
                for c in reply.tool_calls:
                    result = tools.run(c.name, c.arguments)
                    _record(c.name, c.arguments, result)
                    messages.append({"role": "tool", "tool_call_id": c.id,
                                     "content": result})
                continue
            if not content.strip():
                # Still calling with the tools closed and nothing written.
                attempt += 1
                log(f"  {role} kept calling tools after its last round "
                    f"(attempt {attempt}).", verbose)
                continue

        if not native and tools and not hit_ceiling:
            call = tools.parse_json_call(content)
            if call is not None:
                messages.append({"role": "assistant", "content": content})
                if offer:
                    rounds += 1
                    name, arguments = call
                    result = tools.run(name, arguments)
                    _record(name, arguments, result)
                    last = rounds >= tools_mod.MAX_TOOL_ROUNDS
                    messages.append({"role": "user", "content": (
                        f"Result of {name}:\n{result}\n\n"
                        + ("That was your last tool call: give your answer "
                           "now, in the format asked for."
                           if last else
                           "Call another tool, or give your answer in the "
                           "format asked for.")
                    )})
                else:
                    attempt += 1
                    messages.append({"role": "user", "content": (
                        "No more tool calls: give your answer now, in the "
                        "format asked for."
                    )})
                continue

        if content.strip() and not hit_ceiling:
            return content.strip(), "", None

        if hit_ceiling:
            spent = (reply.usage.get("completion_tokens")
                     or (len(thinking) + len(content)) // 4)
            log(
                f"  {role} hit the context wall "
                f"(num_ctx={options['num_ctx']}, generated {spent} tokens, "
                f"~{len(thinking) // 4} of them thinking).",
                verbose,
            )
            # A pi-style compaction rescue, when there is something to
            # resume from: the trace gets summarised, the answer-so-far
            # is kept verbatim, and the model is asked to finish from the
            # cut. The rescue is bounded by MAX_COMPACTION_PASSES passes;
            # if the work is still unfinished then, its partial state is
            # returned so the proof loop can hand it to the reviser. The
            # continuation offers no tools; what was looked up rides along
            # in the task.
            partial: Optional[Dict[str, str]] = None
            if COMPACT_ENABLED and (thinking.strip() or content.strip()):
                task = user_prompt
                if tool_notes:
                    task += (
                        "\n\n---\nResults of the tool calls you made:\n\n"
                        + "\n\n".join(tool_notes)
                    )
                resumed, partial = _resume_compacted(
                    role, base_system, task, thinking, content,
                    want_think, verbose,
                )
                if resumed:
                    log(f"  {role} completed after compaction + "
                        f"continuation.", verbose)
                    return resumed, "", None
            return "", "ceiling", partial

        attempt += 1
        log(f"  {role} returned empty content (attempt {attempt}).", verbose)

    # Retries exhausted on empty replies or transport errors; nothing to salvage.
    return "", ("" if answered else "unavailable"), None


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
                log(f"  {role} extraction hit the token ceiling.", verbose)

            if use_schema and not content.strip():
                # Grammar/thinking conflict: qwen3.x thinks by default, the
                # format grammar suppresses the answer, and the output lands
                # in the thinking channel. Deterministic, so don't retry
                # same-mode.
                stranded = len(reply.thinking or "")
                log(
                    f"  {role} extraction returned empty content with format "
                    f"set ({stranded} chars stranded in thinking). Falling back "
                    f"to prompt-only JSON for the rest of the run.",
                    verbose,
                )
                use_schema = False
                _SCHEMA_MODE_BROKEN = True
                continue

            parsed = json.loads(clean_json_text(content))
            if not isinstance(parsed, dict):
                # A bare value (a string, a list of strings) is no answer
                # to an object schema; every caller reads the result with
                # .get. Same as a failed extraction: nothing usable.
                log(
                    f"  {role} extraction returned a JSON "
                    f"{type(parsed).__name__}, not an object; treating it "
                    f"as no answer.",
                    verbose,
                )
                return {}
            return parsed
        except (requests.RequestException, KeyError, TypeError) as e:
            log(f"  {role} extraction transport error (attempt {attempt}): {e}", verbose)
        except json.JSONDecodeError as e:
            log(f"  {role} extraction parse error (attempt {attempt}): {e}", verbose)
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

    log(f"  {role} extraction failed after all retries.", verbose)
    return {}


# ----------------------------------------------------------------------------
# Context filtering
# ----------------------------------------------------------------------------
def lemma_window(
    dag: Dict[str, Any],
    suspended: Optional[Set[Tuple[str, str]]] = None,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], Dict[str, List[str]]]:
    """Which proved lemmas the agents see in full, and the index of the rest.

    The window is chosen from the lemmas that are not suspended — choosing
    first and filtering after would fill it with lemmas no agent may use.
    Newest first by proved_at, a lemma without one (committed before the
    stamp existed) counting as older than any stamped lemma, file order
    breaking ties: the LEMMAS_ALL newest lemmas, anyone's, and the
    MY_LEMMAS newest of the current user's own, deduplicated, so the two
    can show fewer than their sum. -1 for either is no limit.

    Then the common lemmas: of the eligible lemmas not already shown,
    the LEMMAS_COMMON cited most often by the shown ones, counting direct
    citations only, ties broken by _tiebreak. A lemma no shown lemma
    cites does not rank, so the group can be smaller than LEMMAS_COMMON.

    Returns (the shown pairs, oldest first — the order the proof grew in;
    the common pairs, most cited first; the index of every other eligible
    lemma, as {user_id: [lemma_id, ...]}, each list sorted). The index is
    empty when everything is shown.
    """
    position = {pair: i for i, pair in enumerate(dag["lemmas"])}
    eligible = [
        pair for pair in dag["lemmas"]
        if suspended is None or pair not in suspended
    ]
    newest = sorted(
        eligible,
        key=lambda pair: (
            str(dag["lemmas"][pair].get("proved_at") or ""),
            position[pair],
        ),
        reverse=True,
    )

    def take(pairs: List[Tuple[str, str]], n: int) -> List[Tuple[str, str]]:
        return pairs if n < 0 else pairs[:n]

    chosen = set(take(newest, LEMMAS_ALL))
    chosen.update(take([p for p in newest if p[0] == USER], MY_LEMMAS))
    shown = [pair for pair in reversed(newest) if pair in chosen]

    eligible_set = set(eligible)
    counts: Dict[Tuple[str, str], int] = {}
    for pair in shown:
        for dep in dag["lemmas"][pair].get("cited_lemmas", []):
            if isinstance(dep, tuple) and dep in eligible_set and dep not in chosen:
                counts[dep] = counts.get(dep, 0) + 1
    common = take(
        sorted(counts, key=lambda p: (-counts[p], _tiebreak(p))),
        LEMMAS_COMMON,
    )
    chosen.update(common)

    index: Dict[str, List[str]] = {}
    for u, lid in sorted(p for p in eligible if p not in chosen):
        index.setdefault(u, []).append(lid)
    return shown, common, index


def _tiebreak(pair: Tuple[str, str]) -> bytes:
    """A lemma's place among equally cited lemmas: random, from the run's
    seed, but the same for the whole run — so a tie does not reshuffle the
    prompt from one call to the next."""
    return hashlib.sha256(
        f"{_TIEBREAK_SEED}:{pair[0]}:{pair[1]}".encode("utf-8")
    ).digest()


# What an agent is told about the index, beside it in the view.
_INDEX_NOTE = (
    "proved_lemmas lists the most recently proved lemmas in full. "
    "common_lemmas lists, statement only, older lemmas those cite often. "
    "other_proved_lemmas lists every other proved lemma by user_id "
    "(the key) and lemma_id, statement omitted; they are proved lemmas "
    "all the same. Call lookup_lemmas to read the statement of any of them "
    "before relying on it."
)


def planner_dag_view(
    dag: Dict[str, Any],
    suspended: Optional[Set[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    """Planner sees lemma statements + dependency structure, never proofs.

    A flat list, not a dict keyed by id: a bare lemma_id is unique only
    within one user's file, so in the complete DAG a lemma is the pair
    (user_id, lemma_id), and each entry carries both fields plus the node's
    citations: the cited lemma pairs and the cited reference ids (bare).

    Given the suspension, the suspended lemmas are not in the view: a
    refuted or uncertified lemma is not a proved lemma, and the planner
    must not be told it is one (see suspension.py). (This view is
    part of how a suspended lemma is hidden from the planner, the selector
    and the prover: the selector sees it through planner_dag_view, the
    prover through prover_context, and the screening's id test through
    lemma_id_set.)

    Only the lemma window is listed in full; the rest of the proved
    lemmas are in other_proved_lemmas by id (lemma_window).
"""
    shown, common, index = lemma_window(dag, suspended)
    view: Dict[str, Any] = {
        "proved_lemmas": [
            {
                "user_id": u,
                "lemma_id": lid,
                "statement": str(dag["lemmas"][(u, lid)].get("statement", "")),
                "cited_lemmas": [
                    _dep_to_json(dep)
                    for dep in dag["lemmas"][(u, lid)].get("cited_lemmas", [])
                ],
                "cited_references": list(
                    dag["lemmas"][(u, lid)].get("cited_references", [])
                ),
            }
            for u, lid in shown
        ]
    }
    if common:
        view["common_lemmas"] = [
            {
                "user_id": u,
                "lemma_id": lid,
                "statement": str(dag["lemmas"][(u, lid)].get("statement", "")),
            }
            for u, lid in common
        ]
    if index:
        view["other_proved_lemmas"] = index
    if common or index:
        view["note"] = _INDEX_NOTE
    return view


def prover_context(
    dag: Dict[str, Any],
    suspended: Optional[Set[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
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

    Each lemma is listed with the user_id and lemma_id it is cited by: a
    bare lemma_id is unique only within one user's file, so a citation of a
    lemma is the pair, and the prover's cited_lemmas carries the pair
    objects — one {"user_id", "lemma_id"} object per cited lemma, with the
    cited reference ids in its separate cited_references list.

    Given the suspension, the suspended lemmas are not in the view: the
    prover may not build on a lemma that is not to be built on, and a proof
    that cites one would inherit its suspension (see suspension.py).

    Only the lemma window is listed in full; the rest of the proved
    lemmas are in other_proved_lemmas by id (lemma_window).
    """
    shown, common, index = lemma_window(dag, suspended)
    view: Dict[str, Any] = {
        "proved_lemmas": [
            {
                "user_id": u,
                "lemma_id": lid,
                "statement": str(dag["lemmas"][(u, lid)].get("statement", "")),
            }
            for u, lid in shown
        ]
    }
    if common:
        view["common_lemmas"] = [
            {
                "user_id": u,
                "lemma_id": lid,
                "statement": str(dag["lemmas"][(u, lid)].get("statement", "")),
            }
            for u, lid in common
        ]
    if index:
        view["other_proved_lemmas"] = index
    if common or index:
        view["note"] = _INDEX_NOTE
    return view


def verifier_context(
    dag: Dict[str, Any],
    cited_lemmas: List[Tuple[str, str]],
    cited_references: List[str],
    references: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Statements of exactly the results the prover claims to have used.

    DAG lemmas and reference results are merged into one set and carry the
    statement only: a lemma citation (the (user_id, lemma_id) pair) is keyed
    by the "user:lemma" spelling of the pair, a reference by its bare id.
    The two lists are the citation check's output (see load_dag()): every
    pair is a lemma of the complete DAG and every id a reference of the
    collection, so a statement missing here names a citation the check let
    through — the verifier sees the gap as an absent result. Deliberately
    not the whole DAG or the whole reference collection: the verifier agents
    are asked to reject "use of results not present in the provided set",
    which only bites if the set is the prover's declared citations: a proof
    leaning on a result it never declared then reads as an unjustified leap,
    which is what it is.
    """
    statements: Dict[str, Dict[str, str]] = {}
    for pair in cited_lemmas:
        if pair in dag["lemmas"]:
            statements[_citation_label(pair)] = {
                "statement": str(dag["lemmas"][pair].get("statement", ""))
            }
    ref_by_id = {str(r.get("id") or ""): r for r in references}
    for rid in cited_references:
        ref = ref_by_id.get(rid)
        if ref is not None:
            statements[rid] = {
                "statement": str(ref.get("formal statement", ""))
            }
    return {"cited_results": statements}


def scan_citations(
    proof: str, dag: Dict[str, Any], references: List[Dict[str, Any]],
    suspended: Optional[Set[Tuple[str, str]]] = None,
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Fallback edge recovery: which known results appear in the proof text.

    Only used when the prover ignored its output format entirely, in which
    case the alternative is a node with no citations at all — a lemma that
    silently claims to stand on its own. Scans lemma ids and reference ids;
    word-boundary matching, so lemma_1 does not match inside lemma_10. A
    lemma id several users' files share cannot be resolved from the text
    alone, so it is left to the prover's declaration rather than guessed.

    Returns (cited_lemmas, cited_references) in the two stored fields'
    shapes. Everything it names exists by construction, so the citation
    check has nothing to reject.

    Given the suspension, a suspended lemma is never recovered: the text
    naming one is a proof leaning on a lemma it may not build on, and the
    verifiers, shown no statement for it, see the gap. Ownership is still
    counted over every lemma, so an id a suspended lemma shares with an
    unsuspended one stays ambiguous rather than resolving to the other.
    """
    suspended = suspended or set()
    owners: Dict[str, List[str]] = {}
    for u, lid in dag["lemmas"]:
        owners.setdefault(lid, []).append(u)
    cited_lemmas: List[Tuple[str, str]] = []
    for lid, users in sorted(owners.items()):
        if (
            len(users) == 1
            and (users[0], lid) not in suspended
            and re.search(rf"\b{re.escape(lid)}\b", proof)
        ):
            cited_lemmas.append((users[0], lid))
    cited_references: List[str] = []
    for ref in references:
        rid = str(ref.get("id") or "")
        if rid and re.search(rf"\b{re.escape(rid)}\b", proof):
            cited_references.append(rid)
    return cited_lemmas, cited_references


def load_references() -> List[Dict[str, Any]]:
    """Load references.md — proofs parse's output (parsing.py).

    A strict JSON array of { id, slogan, "formal statement", reference } objects.
    The file is the conjecture's committed reference
    collection (see parsing.py): the maintainer parses it and
    commits it, every other user pulls it, and this module only ever reads
    it — a missing or corrupt file is never repaired here.
    A missing file returns [], and every call site degrades to prompts
    with no references section. A corrupt file warns and returns [] too —
    the right fix is for the maintainer to re-run proofs parse, not to
    hand the loop a subset. Ids (ref_N) are assigned by parsing.py,
    never here, so a prover that cites one is citing a name the file vouches
    for.
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
        log(f"{REFERENCES_FILE} is not valid JSON ({e}); ignoring it. "
            f"Re-run proofs parse if you expected references here.")
        return []
    if not isinstance(data, list):
        log(f"{REFERENCES_FILE} is not a JSON array; ignoring it.")
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
    dag: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    suspended: Optional[Set[Tuple[str, str]]] = None,
) -> List[Tuple[Dict[str, Any], List[str]]]:
    """Pair each candidate with the reasons it cannot be used as it stands.

    Only one check survives now that candidates carry no dependencies: an id
    already in the DAG — under any user: proving it again gains nothing, and
    writing it again would only fork the id across users' files. An empty
    problem list means the candidate is ready for the prover.

    Given the suspension, the test runs against the lemmas that are not
    suspended (see suspension.py): a candidate that restates a
    suspended lemma's id is not screened out, because a proof under that id
    is a fresh proof, not a build on the suspended lemma — the id is up for
    grabs for the users whose files do not hold it, and screening it out
    would make the id dead for everyone until the suspended node is
    repaired. The suspended node itself is lifted by proofs verify or
    proofs repair re-accepting it, not by this screen.
    """
    screened: List[Tuple[Dict[str, Any], List[str]]] = []
    for cand in candidates:
        problems: List[str] = []
        if cand["id"] in lemma_id_set(dag, suspended):
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
    log("Planner shortlist:", verbose)
    log(interaction.render(screened, plan_summary), verbose)


def select_lemma(
    usable: List[Dict[str, Any]],
    dag: Dict[str, Any],
    conjecture: str,
    plan_summary: str,
    failed_attempts: Dict[str, List[str]],
    verbose: bool,
    proposer: str = "the planner",
    suspended: Optional[Set[Tuple[str, str]]] = None,
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
        f"Proved lemmas so far:\n{json.dumps(planner_dag_view(dag, suspended), indent=2)}\n\n"
        f"Previously rejected attempts (routes the prover and the verifiers "
        f"have already found wanting):\n"
        f"{json.dumps(failed_attempts, indent=2)}\n\n"
        f"Candidate lemmas, in the order {proposer} offered them. Exactly "
        f"one will be sent to the prover; choose from these only:\n"
        f"{json.dumps(usable, indent=2)}"
    )
    text, _status, _partial = reason(selector_sys, selector_user, "selector",
                           THINK["selector"], verbose,
                           tools=agent_tools(dag, suspended))
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
            log(f"Selector picked {cand['id']}.", verbose)
            return cand

    if not text or not selected_id:
        log(
            f"Selector gave no usable choice; taking the first candidate "
            f"instead ({usable[0]['id']}).",
            verbose,
        )
    else:
        log(
            f"Selector chose {selected_id!r}, which was not among the "
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
        "\nAutomation resumed; planning runs unattended from here."
        if MODE == "auto"
        else "\nManual control engaged; you pick at the next planning step.",
        verbose,
    )


def _run_verifier(
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
        system_prompt, user_prompt, role, THINK[role], verbose,
    )
    check_mode_toggle(verbose)
    if review_status == "unavailable":
        return (
            "unavailable",
            f"{role} could not reach the server (transport errors on every "
            f"attempt).",
        )
    if review_status == "ceiling" and not review:
        return (
            "",
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
    if not review:
        return (
            "",
            f"{role} returned no review (the server failed or replied "
            f"with nothing, after retries).",
        )
    res = res or {}
    decision = str(res.get("decision", "")).strip().lower()
    if decision not in ("accept", "reject"):
        return (
            "",
            f"{role}'s review could not be read as a verdict "
            f"(decision {res.get('decision')!r}).",
        )
    justification = str(res.get("justification") or "(no justification)").strip()
    return decision, justification


def _wait_for_server(role: str, should_stop, verbose: bool) -> bool:
    """Wait SERVER_WAIT_SECONDS after a call the server never answered.
    True when the caller should retry, False when should_stop fired
    during the wait (the engine then returns its published round as
    stopped). Logged unconditionally: a stalled run must say why."""
    log(
        f"The server did not answer the {role}; this round is not "
        f"counted. Retrying in {SERVER_WAIT_SECONDS}s (Ctrl-C stops the "
        f"run, and a re-run resumes this round)."
    )
    deadline = time.monotonic() + SERVER_WAIT_SECONDS
    while time.monotonic() < deadline:
        if should_stop is not None and should_stop():
            return False
        time.sleep(1)
    return True


def _reason_until_answered(should_stop, verbose: bool, *args, **kwargs):
    """reason(), retried while the server is unreachable: its result once
    the server answers, or None when should_stop fired while waiting."""
    role = args[2]
    while True:
        result = reason(*args, verbose=verbose, **kwargs)
        if result[1] != "unavailable":
            return result
        if not _wait_for_server(role, should_stop, verbose):
            return None


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
    allow_decomposition: bool = True,
) -> Dict[str, Any]:
    """Steps 2-5 of the proof loop: prover -> verifiers -> reviser, shared
    by the serial run and by every parallel loop.

    `allow_decomposition` keeps the reviser's new_lemma option open: the
    run leaves it on, where a lemma that is too hard as stated is the
    reviser's to decompose. proofs repair turns it off — the repair
    re-proves the named lemma and the lemma keeps its id, so a
    decomposition would commit a different lemma and leave the refutation
    sitting on the one it named. With it off, a new_lemma decision falls
    back to keeping the statement, the way a new_lemma with a missing or
    taken id already does; on the overflow path it falls back to giving up
    on the lemma for this pass, since retrying the statement that just
    overflowed would only overflow it again.

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
    verifier checks, and with the node's hash as the verifiers checked it
    (_verified_hash), which the caller hands to commit_lemma_to_dag (the
    caller certifies and commits it and returns the id it committed under — the lemma's
    id, unless that id was already taken and the commit renamed it). Every failed_attempts mutation happens under
    `failed_lock` — a formality in the serial run, where the lock is never
    contended, and what keeps the shared reject list coherent in a parallel
    one.

    The suspension (see suspension.py) is recomputed at every round from
    the DAG get_dag() returns: the prover's context and the reviser's
    view of the DAG omit the suspended lemmas, so the proof cannot build on
    them. A target that restates a suspended lemma's id is a fresh proof,
    not a build on the suspended lemma: the commit lands in this run's
    user's own file, under the id if it is free there and under a fresh id
    beside it otherwise (commit_lemma_to_dag renames on collision), and a
    node suspended in some file is lifted by proofs verify or proofs
    repair re-accepting it.

    Returns {"proved", "lemma_id", "committed_id", "claimed_id", "target",
    "proof", "attempt_notes"}. committed_id is the id the proof was
    committed under (None when not proved); the caller's on_proof decides
    it. claimed_id is the id this claim started from: lemma_id is wherever
    the prover ended up, and a decomposition may have moved it, so the
    caller frees its buffer slot keyed on claimed_id, not lemma_id. The
    caller records the final failure when not proved (from attempt_notes)
    and clears its in-flight state; the engine leaves no other traces.
    """
    resuming = initial is not None
    in_flight = initial if initial is not None else {}
    proof = ""  # bound before the return; the overflow-break paths skip the assignment
    committed_id: Optional[str] = None  # set when on_proof commits

    if resuming:
        lemma_id = in_flight["lemma_id"]
        # The claim id is the id this claim started from, kept apart from
        # the target because a decomposition may have moved the prover onto
        # a different lemma since the claim: the buffer slot to free on
        # settle is keyed on the claim, not on wherever the prover ended
        # up. Checkpoints written before this field existed carry only the
        # target's id; the fallback is the old behavior.
        claimed_id = str(in_flight.get("claimed_id") or lemma_id)
        target: Dict[str, Any] = in_flight["target"]
    else:
        lemma_id = fresh_target["id"]
        claimed_id = lemma_id
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
        with failed_lock:
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
                "committed_id": None,
                "claimed_id": claimed_id,
                "target": target,
                "proof": last_proof,
                "attempt_notes": attempt_notes,
                "stopped": True,
            }
        dag = get_dag()
        # The suspension as this round stands (see suspension.py),
        # recomputed with the DAG rather than taken once per lemma: in a
        # parallel run a sibling's commit, or a certificate that failed to
        # land, changes it between rounds, and a lemma suspended since the
        # last round must not be offered to the prover in this one.
        suspended = suspension.compute(
            dag["lemmas"], references, CONJECTURE_ROOT
        ).suspended
        # Checkpoint: this prover round is now the resumable position. If
        # the run is cancelled anywhere inside it, the next run re-runs
        # the round with this target and this feedback — the round's own
        # in-flight call is lost, nothing completed before it is.
        in_flight_state = {
            "lemma_id": lemma_id,
            "claimed_id": claimed_id,
            "target": target,
            "attempt": attempt,
            "feedback": feedback,
            "attempt_notes": attempt_notes,
            "last_proof": last_proof,
        }
        on_round(in_flight_state)
        log(
            f"Proof attempt {attempt}/{MAX_PROOF_ATTEMPTS} for {lemma_id}.",
            verbose,
        )

        # ---------------- Step 2: Prover ----------------
        # prover.md instructs the model to reply with {"lemma_id",
        # "cited_lemmas", "cited_references", "proof"} — the two citation
        # lists are separate: pair objects for lemmas, bare ids for
        # references (see load_dag()). We parse that JSON locally
        # rather than via a second model call, so the proof text can never
        # be abridged or paraphrased; if the model ignored the format, its
        # content is taken verbatim as the proof.
        prover_user = (
            f"Conjecture:\n{conjecture}\n\n"
            f"Available proved lemmas:\n"
            f"{json.dumps(prover_context(dag, suspended), indent=2)}"
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
        # An unreachable server is waited out, not counted: the round
        # re-runs from the top (where a stop is noticed) once it answers.
        answer = _reason_until_answered(
            should_stop, verbose,
            prover_sys, prover_user, "prover", THINK["prover"],
            tools=agent_tools(dag, suspended),
        )
        if answer is None:
            continue
        proof_text, prover_status, prover_partial = answer
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
                    f"Lemma {lemma_id} is too large to prove in one call "
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
                f"Lemma {lemma_id} overflowed the context window after "
                f"{MAX_COMPACTION_PASSES} compaction passes; asking the "
                f"reviser to pick a smaller lemma or revise the statement "
                f"from the partial proof.",
                verbose,
            )
            overflow_user = (
                f"Conjecture:\n{conjecture}\n\n"
                f"Proved lemmas so far:\n{json.dumps(planner_dag_view(dag, suspended), indent=2)}"
                f"{reviser_ref_block}\n\n"
                f"Target lemma:\n{json.dumps(target, indent=2)}\n\n"
                f"{report}"
            )
            answer = _reason_until_answered(
                should_stop, verbose,
                reviser_sys, overflow_user, "reviser", THINK["reviser"],
                tools=agent_tools(dag, suspended),
            )
            if answer is None:
                continue
            revision_text, _revision_status, _revision_partial = answer
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
            # The id test reads the DAG fresh: the round's snapshot is
            # stale in a parallel run, and a decomposition onto an
            # already-proved id would burn rounds that add_lemma's
            # commit-time guard would drop anyway. The id is matched
            # against every user's lemmas: a lemma_id is unique only
            # within one user's file.
            if (
                allow_decomposition
                and action == "new_lemma"
                and new_id
                and new_stmt
                # a smaller lemma never takes a reserved id: that id's
                # statement is pinned to the whole conjecture
                and not resolution.is_reserved(new_id)
                and new_id not in lemma_id_set(get_dag())
            ):
                # The reviser found a smaller lemma worth trying. Set the
                # overflowing lemma aside (its history is recorded under its
                # own id) and start the prover on the smaller lemma with a
                # fresh budget and no inherited verdict.
                old_id = lemma_id
                with failed_lock:
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
                    f"Prover overflowed {old_id}; now trying the smaller "
                    f"lemma {lemma_id}:\n{new_stmt}",
                    verbose,
                )
                continue
            if (
                action == "revise_statement"
                and new_stmt
                and new_stmt != target["statement"]
                # the conjecture's statement is pinned, not revisable
                and not resolution.is_reserved(lemma_id)
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
                    f"Reviser revised the lemma after the overflow; the "
                    f"prover starts again from:\n{new_stmt}",
                    verbose,
                )
                continue
            if allow_decomposition:
                log(
                    "Reviser could not pick a usable smaller lemma or "
                    "revised statement from the partial proof; the planner "
                    "will decompose.",
                    verbose,
                )
                attempt_notes.extend(overflow_feedback)
                attempt_notes.append(
                    "Lemma too large: the prover exhausted its context "
                    "window and the reviser could neither split it further "
                    "nor revise its statement. Decompose into smaller, "
                    "independently provable lemmas."
                )
            else:
                log(
                    "Reviser could not pick a usable revised statement "
                    "from the partial proof, and decomposition is not "
                    "allowed here.",
                    verbose,
                )
                attempt_notes.extend(overflow_feedback)
                attempt_notes.append(
                    "Lemma too large: the prover exhausted its context "
                    "window and the reviser could not revise its statement "
                    "(decomposition is not allowed here)."
                )
            break

        proof = proof_text
        cited_lemmas: List[Tuple[str, str]] = []
        cited_refs: List[str] = []
        prover_res = parse_json_or_none(proof_text)
        raw_cited_lemmas: Any = None
        raw_cited_refs: Any = None
        if prover_res is not None:
            candidate = prover_res.get("proof")
            if isinstance(candidate, str) and candidate.strip():
                proof = candidate.strip()
            raw_cited_lemmas = prover_res.get("cited_lemmas")
            raw_cited_refs = prover_res.get("cited_references")

        if not proof:
            # The model's failure, not the server's (an outage was waited
            # out above): it spends the round, and the next one is told.
            log(f"Prover produced no proof for {lemma_id}.", verbose)
            feedback = [
                "Your previous attempt returned no proof at all. Write the "
                "complete proof, in the format asked for."
            ]
            attempt_notes.append(
                f"Prover attempt {attempt}: the prover returned no proof."
            )
            attempt += 1
            continue

        # The last proof of this lemma: what the verifiers are about to
        # check, what the reviser will judge its difficulty by, and what
        # the checkpoint records beside the in-flight state.
        last_proof = proof

        # The DAG's edges now come from here, in the two stored fields
        # (see load_dag()): cited_lemmas, an array of
        # {"user_id", "lemma_id"} objects, and cited_references, an array
        # of bare reference ids. A response with neither list means the
        # prover ignored its format, and scanning the text beats recording
        # a lemma as standing on nothing; the scan only ever names
        # something that exists, so what it recovers cannot be a phantom.
        if not (
            isinstance(raw_cited_lemmas, list)
            or isinstance(raw_cited_refs, list)
        ):
            cited_lemmas, cited_refs = scan_citations(
                proof, dag, references, suspended
            )
            if cited_lemmas or cited_refs:
                log(
                    f"  Prover declared no citations; recovered "
                    f"{', '.join(_citation_label(d) for d in [*cited_lemmas, *cited_refs])} "
                    f"from the proof text.",
                    verbose,
                )
        else:
            # Every cited pair is checked against the complete DAG and
            # every id against the reference collection; whatever matches
            # nothing is sent back to the prover as feedback and the round
            # is spent, so a proof that claims a result nobody has proved
            # never reaches the verifiers to be rejected there instead.
            cited_lemmas, cited_refs, unmatched = _check_prover_citations(
                raw_cited_lemmas, raw_cited_refs, dag, ref_ids, suspended
            )
            if unmatched:
                unmatched_labels = [label for label, _ in unmatched]
                suspended_labels = [
                    label for label, kind in unmatched if kind == "suspended"
                ]
                unknown_labels = [
                    label for label, kind in unmatched if kind != "suspended"
                ]
                if unknown_labels:
                    log(
                        f"  Prover cited {len(unknown_labels)} result(s) "
                        f"that match no lemma and no reference: "
                        f"{', '.join(unknown_labels)} — sent back as "
                        f"feedback.",
                        verbose,
                    )
                if suspended_labels:
                    log(
                        f"  Prover cited {len(suspended_labels)} "
                        f"suspended lemma(s): {', '.join(suspended_labels)} "
                        f"— not to be built on; sent back as feedback.",
                        verbose,
                    )
                missing_refs = [
                    label for label, kind in unmatched
                    if kind == "reference"
                ]
                if missing_refs:
                    log(
                        f"  none of these is in "
                        f"{os.path.basename(REFERENCES_FILE)}: "
                        f"{', '.join(missing_refs)} — this run cannot add "
                        f"references. Request the missing one(s) from the "
                        f"conjecture's maintainer, who keeps the file with "
                        f"proofs parse and commits it.",
                        verbose,
                    )
                problems = []
                if unknown_labels:
                    problems.append(
                        "cited result(s) that do not exist in the DAG or "
                        "the reference collection: "
                        + ", ".join(unknown_labels)
                    )
                if suspended_labels:
                    problems.append(
                        "cited lemma(s) that are suspended — refuted, "
                        "uncertified, or standing on one that is — and may "
                        "not be built on: "
                        + ", ".join(suspended_labels)
                        + " (prove what you need from them yourself, or "
                        "use other results)"
                    )
                feedback = [
                    "Your proof "
                    + "; and ".join(problems)
                    + ". Cite only lemmas from the 'Available proved "
                    "lemmas' list, each as the exact {\"user_id\", "
                    "\"lemma_id\"} object that list shows, and cite "
                    "references from the 'Known references' list by id. "
                    "The two go in separate lists: cited_lemmas and "
                    "cited_references."
                ]
                attempt_notes.append(
                    f"Prover attempt {attempt}: the proof cited "
                    f"{len(unmatched)} unknown or suspended result(s) "
                    f"({', '.join(unmatched_labels)}); the citations were "
                    "sent back unverified and the proof never reached the "
                    "verifiers."
                )
                attempt += 1
                continue

        all_cited = [*cited_lemmas, *cited_refs]
        if all_cited:
            log(
                f"Proof generated for {lemma_id} ({len(proof)} chars, "
                f"cites: "
                f"{', '.join(_citation_label(d) for d in all_cited)}).",
                verbose,
            )
        else:
            log(
                f"Proof generated for {lemma_id} ({len(proof)} chars, "
                f"no citations).",
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
            f"{json.dumps(verifier_context(dag, cited_lemmas, cited_refs, references), indent=2)}\n\n"
            f"Target lemma:\n{json.dumps(target, indent=2)}\n\n"
            f"Proposed proof:\n{proof}"
        )
        reject_just: Optional[str] = None
        stopped = False
        for step, agent in enumerate(VERIFIER_AGENTS, 1):
            role = agent[:-3]   # "verifier_1.md" -> "verifier_1"
            # An unreachable server is waited out and the same step
            # retried: the proof is not at fault, and keeping it saves
            # re-running the prover.
            while True:
                decision, justification = _run_verifier(
                    role, verifier_sys[agent], verifier_user, verbose,
                )
                log(
                    f"Verifier {step}/{len(VERIFIER_AGENTS)} ({role}): "
                    f"{decision.upper() or '???'} — {justification}",
                    verbose,
                )
                if decision != "unavailable":
                    break
                if not _wait_for_server(role, should_stop, verbose):
                    stopped = True
                    break
            if stopped:
                break
            if decision != "accept":
                reject_just = justification
                break
        if stopped:
            # The top of the loop returns the published round as stopped.
            continue

        if reject_just is None:
            # ---------------- Step 4: DAG update ----------------
            # All three verifier checks accepted the same proof. The
            # reject notes are cleared before the commit so that a commit
            # renamed under a taken id (on_proof returns a different id)
            # can still record its note under the original id afterwards
            # without this pop wiping it.
            log(
                f"Lemma {lemma_id} passed all {len(VERIFIER_AGENTS)} "
                f"verifier checks. Adding to DAG.",
                verbose,
            )
            with failed_lock:
                failed_attempts.pop(lemma_id, None)
            proved_node = {
                "statement": target["statement"],
                "proof": proof,
                # The round may have ended before the citations were
                # parsed (ceiling or overflow break); a lemma proved and
                # committed never took that path, but the loop must not
                # crash on it.
                "cited_lemmas": list(cited_lemmas or []),
                "cited_references": list(cited_refs or []),
            }
            committed_id = on_proof(
                lemma_id, proved_node,
                _verified_hash(dag, references, proved_node),
            )
            proved = True
            break

        # ---------------- Step 5: Reviser ----------------
        log(
            f"Proof rejected for {lemma_id}. Sending the proof and the "
            f"verdict to the reviser.",
            verbose,
        )
        reviser_user = (
            f"Conjecture:\n{conjecture}\n\n"
            f"Proved lemmas so far:\n{json.dumps(planner_dag_view(dag, suspended), indent=2)}"
            f"{reviser_ref_block}\n\n"
            f"Target lemma:\n{json.dumps(target, indent=2)}\n\n"
            f"The last proof of this lemma (the one just rejected; judge "
            f"the lemma's difficulty from it):\n{proof}\n\n"
            f"Verifier's reasoning (why the proof failed):\n{reject_just}"
        )
        answer = _reason_until_answered(
            should_stop, verbose,
            reviser_sys, reviser_user, "reviser", THINK["reviser"],
            tools=agent_tools(dag, suspended),
        )
        if answer is None:
            continue
        revision_text, _revision_status, _partial = answer
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
                "Reviser returned nothing; the verdict alone goes back "
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

        # The id test reads the DAG fresh: the round's snapshot is stale
        # in a parallel run, and a decomposition onto an already-proved id
        # would burn rounds that add_lemma's commit-time guard would drop
        # anyway. The id is matched against every user's lemmas: a
        # lemma_id is unique only within one user's file.
        if (
            allow_decomposition
            and action == "new_lemma"
            and new_id
            and new_stmt
            # a smaller lemma never takes a reserved id: that id's
            # statement is pinned to the whole conjecture
            and not resolution.is_reserved(new_id)
            and new_id not in lemma_id_set(get_dag())
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
                f"Reviser decomposed {old_id}; the prover now starts on "
                f"the smaller lemma {lemma_id}:\n{new_stmt}",
                verbose,
            )
            continue

        if action == "revise_statement" and resolution.is_reserved(lemma_id):
            # The statement of a reserved id is the conjecture (or its
            # negation) itself: revising it would only prove something
            # else under the conjecture's name, which settles nothing.
            log(
                f"Reviser proposed a revised statement for {lemma_id}, "
                f"whose statement is pinned to conjecture.md; keeping it.",
                verbose,
            )
        elif action == "revise_statement" and new_stmt and new_stmt != target["statement"]:
            target = {"id": lemma_id, "statement": new_stmt}
            last_proof = ""
            log(
                f"Reviser revised the lemma; the prover starts again "
                f"from:\n{new_stmt}",
                verbose,
            )
            feedback.append(f"Revised statement proposed: {new_stmt}")
        elif action == "revise_statement":
            log(
                "Reviser flagged a statement revision but gave none "
                "usable; keeping the statement.",
                verbose,
            )
        elif action == "new_lemma":
            # The reviser wanted to decompose but could not: the new id was
            # missing or already taken, or decomposition is not allowed
            # here at all (a repair re-proves the lemma under its own id);
            # fall back to keeping the statement.
            if allow_decomposition:
                log(
                    "Reviser proposed a new lemma with a missing, "
                    "reserved or already-taken id; keeping the statement "
                    "instead.",
                    verbose,
                )
            else:
                log(
                    "Reviser wanted to decompose, but the repair keeps "
                    "the lemma's id; keeping the statement instead.",
                    verbose,
                )
        else:
            log(
                "Reviser kept the statement; the prover re-tries with "
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
        "committed_id": committed_id,
        "claimed_id": claimed_id,
        "target": target,
        "proof": proof,
        "attempt_notes": attempt_notes,
        "stopped": False,
    }


def _verified_hash(
    dag: Dict[str, Any],
    references: List[Dict[str, Any]],
    node: Dict[str, Any],
) -> Optional[str]:
    """The Merkle hash of a proof as the verifiers checked it: the node's
    statement, proof and citations, over the DAG and references they were
    shown — the round's snapshot, not the files as they stand at the
    commit. _certify_before_write compares it with the hash at the commit
    and certifies only when the two agree.

    The node is hashed under a key no lemma can have (an empty-looking
    pair a user.id never matches), so it neither replaces a lemma of the
    snapshot nor depends on the id it will be committed under: the hash
    covers no id. None when the hash cannot be computed (a citation cycle
    below it), and _certify_before_write then reports the cycle itself."""
    key = ("\x00verified", "\x00verified")
    try:
        return merkle.Merkle({**dag["lemmas"], key: node}, references).hash(*key)
    except merkle.MerkleCycleError:
        return None


def _operator_citation_check(
    dag: Dict[str, Any],
    ref_ids: Set[str],
    suspended: Set[Tuple[str, str]],
) -> Callable[[List[Any], List[Any]], Optional[str]]:
    """The menu's check_citations (see interaction.choose): the operator's
    citations, run through the prover's check, and the reason they cannot
    be used — an unknown or suspended lemma, a reference the collection
    does not hold — or None when every one resolves."""

    def check(raw_lemmas: List[Any], raw_refs: List[Any]) -> Optional[str]:
        _l, _r, unmatched = _check_prover_citations(
            raw_lemmas, raw_refs, dag, ref_ids, suspended
        )
        if not unmatched:
            return None
        why = {
            "lemma": "no such lemma in the DAG",
            "suspended": "suspended — refuted, uncertified, or standing on one that is",
            "reference": "no such lemma or reference",
            "other": "not a citation",
        }
        return "cannot cite " + "; ".join(
            f"{label} ({why.get(kind, kind)})" for label, kind in unmatched
        )

    return check


def _verify_operator_proof(
    choice: "interaction.Choice",
    dag: Dict[str, Any],
    references: List[Dict[str, Any]],
    ref_ids: Set[str],
    suspended: Set[Tuple[str, str]],
    conjecture: str,
    verifier_sys: Dict[str, str],
    verbose: bool,
) -> Optional[Tuple[str, Dict[str, Any], Optional[str]]]:
    """The operator's own proof of a lemma (an "own_proof" Choice) through
    the three verifier checks, exactly as the proof loop puts the prover's:
    the conjecture, the statements of what the proof cites and nothing
    else, the target, the proof. No lemma enters the DAG on the operator's
    say-so; the operator stands in for the prover, not for the verifiers.

    Returns (lemma_id, node, verified_hash) for the caller to commit and
    certify when all three accept — the operator is the commit's
    last_prover_id, as the run's user — or None, the reason logged, when
    a citation does not resolve, a verifier rejects the proof (its
    objection is shown), or a verifier gives no verdict. The caller then
    puts the menu back."""
    lemma = resolution.pin(choice.lemma or {}, conjecture)
    lemma_id = str(lemma.get("id") or "").strip()
    statement = str(lemma.get("statement") or "").strip()
    proof = str(choice.proof or "").strip()
    if not lemma_id or not statement or not proof:
        log("  That lemma has no usable id, statement or proof.", verbose)
        return None
    cited_lemmas, cited_refs, unmatched = _check_prover_citations(
        choice.cited_lemmas, choice.cited_references, dag, ref_ids, suspended
    )
    if unmatched:
        log(
            "  Your proof cites results that cannot be used: "
            + ", ".join(label for label, _ in unmatched)
            + ". Nothing was verified.",
            verbose,
        )
        return None
    target = {"id": lemma_id, "statement": statement}
    log(
        f"Verifying your proof of {lemma_id} ({len(proof)} chars"
        + (
            ", cites: " + ", ".join(
                _citation_label(d) for d in [*cited_lemmas, *cited_refs]
            )
            if cited_lemmas or cited_refs else ", no citations"
        )
        + ").",
        verbose,
    )
    verifier_user = (
        f"Conjecture:\n{conjecture}\n\n"
        f"Cited results (statements of exactly the lemmas and "
        f"references the proof declares it used; nothing else is "
        f"available to it):\n"
        f"{json.dumps(verifier_context(dag, cited_lemmas, cited_refs, references), indent=2)}\n\n"
        f"Target lemma:\n{json.dumps(target, indent=2)}\n\n"
        f"Proposed proof:\n{proof}"
    )
    for step, agent in enumerate(VERIFIER_AGENTS, 1):
        role = agent[:-3]
        decision, justification = _run_verifier(
            role, verifier_sys[agent], verifier_user, verbose,
        )
        log(
            f"Verifier {step}/{len(VERIFIER_AGENTS)} ({role}): "
            f"{decision.upper() or '???'} — {justification}",
            verbose,
        )
        if decision == "reject":
            log(
                f"Your proof of {lemma_id} was rejected by {role}; nothing "
                f"was added. Back to the menu.",
                verbose,
            )
            return None
        if decision != "accept":
            log(
                f"No verdict on your proof of {lemma_id} from {role} (the "
                f"server failed or the reply could not be read); nothing "
                f"was added. Back to the menu.",
                verbose,
            )
            return None
    log(
        f"Your proof of {lemma_id} passed all {len(VERIFIER_AGENTS)} "
        f"verifier checks. Adding to DAG.",
        verbose,
    )
    node = {
        "statement": statement,
        "proof": proof,
        "cited_lemmas": list(cited_lemmas),
        "cited_references": list(cited_refs),
    }
    return lemma_id, node, _verified_hash(dag, references, node)


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
    lemma_id: str, node: Dict[str, Any], verified_hash: Optional[str],
    verbose: bool,
) -> str:
    """The serial success boundary: commit the lemma through the same
    check-and-write as the parallel run, so the engine's on_proof
    contract — return the id the lemma was committed under — holds in both
    modes. A serial run has a single writer, so the id is rarely taken —
    typically because the user's own file already holds a suspended lemma
    under it, an id the screening lets through — and the commit is then
    renamed beside it."""
    with _commit_section():
        # All three verifier checks accepted this proof: the commit
        # certifies it, then writes it, the way the parallel on_proof's
        # does.
        committed, _renamed = commit_lemma_to_dag(
            lemma_id, node, verified_hash, verbose,
        )
    return committed


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
        }
        for r in references
    ]
    reference_block = (
        "\n\nKnown references (theorem-level results from the parsed "
        "collection; you may cite any of them in cited_references by id "
        "without proving them yourself):\n"
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
                f"{len(references)} references parsed; that is at or "
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
    _ensure_user_dag_file()
    cp = load_checkpoint(verbose)
    if cp is not None and not os.path.exists(DAG_FILE):
        log("Checkpoint without a DAG file; discarding it.", verbose)
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
                f"Resuming at iteration {start_iteration}/{MAX_ITERATIONS} "
                f"({len(failed_attempts)} rejected lemma(s) on record).",
                verbose,
            )
        else:
            log(
                f"Resuming at iteration {start_iteration}/{MAX_ITERATIONS}, "
                f"in-flight {in_flight['lemma_id']} at prover round "
                f"{in_flight['attempt']}/{MAX_PROOF_ATTEMPTS}.",
                verbose,
            )
    if start_iteration > MAX_ITERATIONS:
        log(
            f"\nThe checkpoint stands at iteration {start_iteration}, past "
            f"--max-iterations ({MAX_ITERATIONS}). Re-run with a larger "
            f"--max-iterations to continue, or --fresh to restart the budget "
            f"(the DAG is kept either way).",
            verbose,
        )
        return load_dag()

    warned_dangling = set()
    # Suspended lemmas already warned about this run: the warning is told
    # once per lemma, the way the dangling-citation warning is told.
    warned_suspended: Set[Tuple[str, str]] = set()
    _LIVE_STATE["failed_attempts"] = failed_attempts
    _LIVE_STATE["in_flight"] = in_flight

    for iteration in range(start_iteration, MAX_ITERATIONS + 1):
        log(f"\n--- Iteration {iteration}/{MAX_ITERATIONS} ---", verbose)
        check_mode_toggle(verbose)
        dag = load_dag()

        # The suspension, recomputed from the hashes, the certificates and
        # the refutations as they stand now (see suspension.py): the
        # lemmas this iteration's planner, selector and prover are hidden
        # from. It is never stored in the DAG — this recomputation is the
        # only record of it — so a lemma re-proved and certified since the
        # last iteration is simply no longer in it.
        susp = suspension.compute(dag["lemmas"], references, CONJECTURE_ROOT)
        _warn_suspended(susp, verbose, warned_suspended, dag)
        suspended = susp.suspended

        # Settled? Only a certified lemma stating the conjecture (or its
        # negation) verbatim says so — this run's, or one another user's
        # pulled DAG file holds (resolution.py). Nothing is stored: the
        # resolution is the lemma, and it lasts exactly as long as the
        # lemma stays unsuspended.
        if _settled(dag, suspended, conjecture, verbose):
            discard_checkpoint()
            return dag

        # Citations that no longer resolve dangle every proof that used
        # them, and the verifier would reject those for good reason: a lemma
        # pair no user's file holds (a hand-edited DAG), or a reference id
        # references.md no longer has (a re-parse dropped it). Warn once
        # per (lemma, citation) pair rather than every iteration. The check
        # runs on the complete DAG: a lemma citation is the (user_id,
        # lemma_id) pair its file names it, a reference citation the
        # reference's bare id.
        for key, node in dag["lemmas"].items():
            for dep in [*node.get("cited_lemmas", []),
                        *node.get("cited_references", [])]:
                exists = (
                    dep in dag["lemmas"] if isinstance(dep, tuple)
                    else dep in ref_ids
                )
                if exists or (key, dep) in warned_dangling:
                    continue
                warned_dangling.add((key, dep))
                if isinstance(dep, tuple):
                    shown, kind = _citation_label(dep), "proved lemma"
                else:
                    shown, kind = dep, "parsed reference"
                log(
                    f"DAG node {_citation_label(key)} cites "
                    f"{shown}, which is not a {kind}; verification of it "
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

        resuming = (
            in_flight is not None
            and in_flight["lemma_id"] not in lemma_id_set(dag, suspended)
        )
        if resuming and _committed_uncertified(dag, susp, in_flight):
            log(_uncertified_message(in_flight["lemma_id"]))
            resuming = False
            in_flight = None
            _LIVE_STATE["in_flight"] = None
        if resuming:
            log(
                f"In-flight {in_flight['lemma_id']}: re-running prover round "
                f"{in_flight['attempt']}/{MAX_PROOF_ATTEMPTS} (earlier rounds' "
                f"feedback is kept; the interrupted round itself is re-run).",
                verbose,
            )
        elif in_flight is not None:
            # The lemma was accepted between two checkpoints: its proof is
            # already in the DAG, so there is nothing in flight to resume.
            log(
                f"Checkpoint's in-flight {in_flight['lemma_id']} is already "
                f"in the DAG; planning as usual.",
                verbose,
            )
            in_flight = None
            _LIVE_STATE["in_flight"] = None
        if not resuming:

            # ---------------- Step 1: Planner ----------------
            planner_user = (
                f"Conjecture:\n{conjecture}\n\n"
                f"Proved lemmas so far:\n{json.dumps(planner_dag_view(dag, suspended), indent=2)}"
                f"{planner_ref_block}{comments_block}\n\n"
                f"Previously rejected attempts (avoid or decompose these):\n"
                f"{json.dumps(failed_attempts, indent=2)}"
            )
            plan_text, plan_status, _partial = reason(
                planner_sys, planner_user, "planner", THINK["planner"], verbose,
                tools=agent_tools(dag, suspended),
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
                    "in.",
                    plan_text,
                    PLANNER_SCHEMA,
                    "planner",
                    verbose,
                )

            # A candidate under a reserved id is the conjecture (or its
            # negation) itself: its statement is the code's, never the
            # planner's. It then competes at selection like any other.
            candidates = [
                resolution.pin(c, conjecture)
                for c in planner_candidates(planner_res)
            ]
            if not candidates:
                log("Planner proposed no usable lemma. Re-planning.", verbose)
                continue

            screened = screen_candidates(dag, candidates, suspended)
            summary = str(planner_res.get("plan_summary") or "")

            # ---------------- Step 1b: Selection ----------------
            # The one place the two modes differ, and so the one place MODE is
            # read. Everything after this block runs identically whether the lemma
            # was chosen by a person or by the selector, which is what keeps
            # --mode auto honest as a control.
            next_lemma: Optional[Dict[str, Any]] = None
            if MODE == "human":
                check = _operator_citation_check(dag, ref_ids, suspended)
                choice = interaction.choose(
                    screened, lemma_id_set(dag, suspended), HOTKEY, summary,
                    check_citations=check,
                )
                # The operator's own proof goes through the verifiers here;
                # a rejected one (or one with no verdict) puts the same
                # menu back, with no re-plan in between.
                own_committed: Optional[str] = None
                while choice.action == "own_proof":
                    verified = _verify_operator_proof(
                        choice, dag, references, ref_ids, suspended,
                        conjecture, verifier_sys, verbose,
                    )
                    if verified is not None:
                        lid, node, vh = verified
                        own_committed = _serial_on_proof(lid, node, vh, verbose)
                        failed_attempts.pop(lid, None)
                        break
                    choice = interaction.choose(
                        screened, lemma_id_set(dag, suspended), HOTKEY,
                        summary, check_citations=check,
                    )
                if own_committed is not None:
                    save_checkpoint(iteration + 1, failed_attempts, None, verbose)
                    continue
                if choice.action == "quit":
                    log(
                        "\nStopped by the operator; DAG and checkpoint kept — "
                        "re-run the same command to resume.",
                        verbose,
                    )
                    # Nothing to write: every commit is already on disk,
                    # and rewriting the file from this iteration's snapshot
                    # would undo whatever reached it since (a repair, a
                    # pull).
                    return dag
                if choice.action == "replan":
                    # Recorded against every candidate shown, because the channel
                    # the planner reads is keyed by lemma id and the point is that
                    # it should not come back with this same shortlist.
                    for cand, _ in screened:
                        for note in choice.notes:
                            failed_attempts.setdefault(cand["id"], []).append(note)
                    log("Shortlist rejected; asking the planner again.", verbose)
                    continue
                if choice.action == "auto":
                    MODE = "auto"
                    log(f"Automation resumed. {HOTKEY.hint()}", verbose)
                elif choice.lemma is not None:
                    # A written lemma under a reserved id is pinned too.
                    next_lemma = resolution.pin(choice.lemma, conjecture)

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
                    # No note goes on the reject list: the screening alone
                    # keeps these ids out, and a note per screened candidate
                    # per iteration only grew the planner's prompt.
                    log("No candidate is currently provable; re-planning.", verbose)
                    continue
                if len(usable) == 1:
                    next_lemma = usable[0]
                    if next_lemma is not screened[0][0]:
                        log(
                            f"Skipped {screened[0][0]['id']} "
                            f"({'; '.join(screened[0][1])}); took {next_lemma['id']}.",
                            verbose,
                        )
                else:
                    next_lemma = select_lemma(
                        usable, dag, conjecture, summary, failed_attempts, verbose,
                        suspended=suspended,
                    )

        if resuming:
            lemma_id = in_flight["lemma_id"]
            lemma_stmt = in_flight["target"]["statement"]
            log(f"Next Lemma [{lemma_id}]: {lemma_stmt}", verbose)
        else:
            lemma_id = next_lemma["id"]
            lemma_stmt = next_lemma["statement"]
            log(f"Next Lemma [{lemma_id}]: {lemma_stmt}", verbose)

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
            on_proof=lambda lid, node, vh: _serial_on_proof(
                lid, node, vh, verbose
            ),
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
                f"Lemma {lemma_id} left unproved after the proof loop; the "
                f"planner will see the feedback.",
                verbose,
            )

        # The lemma is settled one way or another: clear the in-flight state
        # and move the resumable position to the next iteration, with the
        # reject list as it now stands.
        in_flight = None
        _LIVE_STATE["in_flight"] = None
        save_checkpoint(iteration + 1, failed_attempts, None, verbose)

    # The last iteration's lemma may have been the one that settles it.
    dag = load_dag()
    susp = suspension.compute(dag["lemmas"], references, CONJECTURE_ROOT)
    if _settled(dag, susp.suspended, conjecture, verbose):
        discard_checkpoint()
        return dag

    log(
        f"\nReached MAX_ITERATIONS ({MAX_ITERATIONS}) without settling the "
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
# the check-and-write in commit_lemma_to_dag under the DAG lock, so a node
# can never be overwritten (a colliding id is renamed, not clobbered) and
# a reader can never see a half-state.
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
        # Call with self.cond held. The Condition's lock is an RLock, so
        # the public unclaimed() below may also be called with it held
        # (wait_for's predicate is); this is the form for code that holds
        # it already and wants no second acquire.
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

    def release_claims(
        self, loop_id: str, keep: Optional[str] = None
    ) -> List[str]:
        """Unclaim every slot loop_id holds except keep: the claims a
        crashed loop left without a published round. Returns the ids
        released."""
        released: List[str] = []
        with self.cond:
            for e in self.slots:
                if (
                    e is not None
                    and e["claimed_by"] == loop_id
                    and e["lemma"]["id"] != keep
                ):
                    e["claimed_by"] = None
                    released.append(e["lemma"]["id"])
            if released:
                self.cond.notify_all()
        return released

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
        # How many lemmas each loop has settled (proved or given up) this
        # run: the crash-restart count resets when this moves, so the
        # restart bound is on crashes with no finished lemma between them.
        # Run-local, not checkpointed.
        self.finished: Dict[str, int] = {lid: 0 for lid in loop_ids}
        # How many rounds the planner and the generator have completed —
        # plans installed, batches offered — for their crash-restart count,
        # the way finished is for the loops'. Run-local, not checkpointed.
        self.steering_rounds: Dict[str, int] = {"planner": 0, "generator": 0}
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
                # Plain strings, one gap per entry, the shape
                # parallel_planner.md and PARALLEL_PLANNER_SCHEMA ask for.
                "priorities": [
                    str(p).strip()
                    for p in (plan.get("priorities") or [])
                    if isinstance(p, str) and p.strip()
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

    def finish_lemma(self, loop_id: str) -> None:
        """The loop's lemma settled, proved or not."""
        with self.state_lock:
            self.finished[loop_id] += 1

    def lemmas_finished(self, loop_id: str) -> int:
        with self.state_lock:
            return self.finished[loop_id]

    def finish_steering_round(self, name: str) -> None:
        """The planner ("planner") or the generator ("generator")
        completed a round."""
        with self.state_lock:
            self.steering_rounds[name] += 1

    def steering_rounds_done(self, name: str) -> int:
        with self.state_lock:
            return self.steering_rounds[name]

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

    def add_lemma(
        self, lemma_id: str, node: Dict[str, Any],
        verified_hash: Optional[str], verbose: bool = True,
    ) -> str:
        """Commit a lemma to the shared DAG — the atomic part of the
        design. The commit itself is check-and-write under the DAG lock
        (commit_lemma_to_dag): two loops can land on one id — the reviser
        decomposes against the DAG as it stood at the top of its own
        round, and the id it picks can also sit in the buffer where a
        sibling is proving it — and a caller that finds its id taken
        takes a fresh non-colliding id (lemma_id_2, ...) instead, so no
        node is ever overwritten and no proof is dropped. The commit
        certifies the lemma before it writes it (verified_hash, see
        commit_lemma_to_dag), so the planner this wakes, and any loop
        reloading the DAG, never finds it uncertified. Then bump the
        version the planner watches, clear the lemma's reject notes, mark
        the plan stale (the DAG changed, so the plan owes a re-read), and
        checkpoint. The version is visible only after the file is down, so
        a replan triggered by it always sees the new lemma. Returns the id
        the lemma was committed under — lemma_id unless it was already
        taken."""
        committed, _renamed = commit_lemma_to_dag(
            lemma_id, node, verified_hash, verbose,
        )
        with self.state_lock:
            self.dag_version += 1
        with self.failed_lock:
            self.failed_attempts.pop(lemma_id, None)
        self.mark_plan_stale()
        self.write_checkpoint()
        return committed

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


def _clean_candidate(raw: Any, conjecture: str) -> Optional[Dict[str, Any]]:
    """A generated or written lemma entry, validated the way the planner's
    candidates are: an id and a non-empty statement. Anything else is
    dropped, not repaired — the prompt tells the generator the shape, and a
    dropped entry costs one generator round, not a proof. An entry under a
    reserved id has its statement pinned to the conjecture (or its
    negation), whatever it said (resolution.pin)."""
    if not isinstance(raw, dict):
        return None
    lemma_id = str(raw.get("id") or "").strip()
    statement = str(raw.get("statement") or "").strip()
    if not lemma_id or not statement:
        return None
    return resolution.pin({"id": lemma_id, "statement": statement}, conjecture)


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
    warned: Set[Tuple[str, str]] = set()
    while state.running():
        if not state.wait_for_work():
            return
        if not state.running():
            return
        # The version is read before the DAG, never after: a commit landing
        # between the two would otherwise be counted in the version but
        # missing from the DAG this plan is written from, and install_plan
        # would then clear the stale flag mark_plan_stale had just set.
        dag_version_before = state.dag_version_snapshot()
        dag = load_dag()
        # The suspension as this round stands to be planned (see
        # suspension.py): the planner's view omits the suspended lemmas, and
        # the warning is told here, where every DAG change is seen.
        susp = suspension.compute(dag["lemmas"], load_references(), CONJECTURE_ROOT)
        _warn_suspended(susp, verbose, warned, dag)
        # Settled? The planner wakes on every DAG change, so this is where
        # a lemma that settles the conjecture is noticed — a loop's proof,
        # the operator's, or (at start) another user's pulled
        # file. The planner's own opinion never settles it (resolution.py).
        kind = _settled(dag, susp.suspended, conjecture, verbose)
        if kind is not None:
            log("The loops are being stopped.", verbose)
            state.set_resolution(kind)
            return
        plan = state.plan_snapshot()
        plan_user = (
            f"Conjecture:\n{conjecture}\n\n"
            f"Proved lemmas so far:\n{json.dumps(planner_dag_view(dag, susp.suspended), indent=2)}"
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
            tools=agent_tools(dag, susp.suspended),
        )
        check_mode_toggle(verbose)
        plan_res = parse_json_or_none(text) if text else None
        if plan_res is None and text:
            plan_res = extract(
                "Extract the plan. plan_summary is a short string; "
                "priorities is a list of strings, each one gap to close, "
                "in the order the text gives them.",
                text,
                PARALLEL_PLANNER_SCHEMA,
                "parallel_planner",
                verbose,
            )
        if not plan_res or not isinstance(plan_res, dict):
            log(
                "Parallel planner gave no usable plan; retrying in 10s.",
                verbose,
            )
            if state.stop_event.wait(10):
                return
            continue
        moved = state.install_plan(plan_res, dag_version_before)
        state.finish_steering_round("planner")
        log(
            "Planner set the strategy"
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
        susp = suspension.compute(dag["lemmas"], load_references(), CONJECTURE_ROOT)
        buffer_snapshot = state.buffer.snapshot()
        # Ids the new batch must not repeat: proved (under any user's
        # name: a lemma_id is unique only within one user's file, and not
        # suspended — a suspended lemma's id is up for grabs again for the
        # users whose files do not hold it), buffered (claimed or not), or
        # already rejected.
        known = lemma_id_set(dag, susp.suspended)
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
            f"them):\n{json.dumps(planner_dag_view(dag, susp.suspended), indent=2)}\n\n"
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
            tools=agent_tools(dag, susp.suspended),
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
            cand = _clean_candidate(raw, conjecture)
            if cand is None or cand["id"] in known:
                continue
            known.add(cand["id"])
            fresh.append(cand)
        added = state.buffer.fill(fresh)
        state.finish_steering_round("generator")
        log(
            f"Lemma generator offered {len(fresh)} new candidate(s); "
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
        susp = suspension.compute(
            dag["lemmas"], load_references(), CONJECTURE_ROOT
        )
        return select_lemma(
            usable,
            dag,
            conjecture,
            str(plan.get("plan_summary") or ""),
            failed,
            verbose,
            proposer="the lemma generator",
            suspended=susp.suspended,
        )

    return state.buffer.claim(loop_id, pick)


def _parallel_human_step(
    state: ParallelState,
    loop_id: str,
    verbose: bool,
    conjecture: str,
    references: List[Dict[str, Any]],
    ref_ids: Set[str],
    verifier_sys: Dict[str, str],
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """One human planning step: the buffer on the menu, as the serial run
    puts the planner's shortlist. Loops until the operator commits to a
    lemma, hands the loop to automation, or quits — a quit stops the whole
    run, not just this loop, and says so. Re-plans are recorded and the
    menu comes right back, the serial way; so does the operator's own
    proof, committed when the verifiers accept it and not otherwise."""
    global MODE
    while True:
        dag = load_dag()
        susp = suspension.compute(
            dag["lemmas"], load_references(), CONJECTURE_ROOT
        )
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
            screened, lemma_id_set(dag, susp.suspended), HOTKEY, summary,
            check_citations=_operator_citation_check(
                dag, ref_ids, susp.suspended
            ),
        )
        if choice.action == "quit":
            log(
                "\nStopped by the operator; the DAG and the checkpoint "
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
            log("Buffer rejected; the planner is asked again.", verbose)
            continue
        if choice.action == "auto":
            MODE = "auto"
            log(f"{loop_id} handed to automation. {HOTKEY.hint()}", verbose)
            return "auto", None
        if choice.action == "own_proof":
            own = resolution.pin(choice.lemma or {}, conjecture)
            own_id = str(own.get("id") or "").strip()
            # A buffered id is claimed before the operator's proof is
            # verified, the way a 'prove' pick is: a lemma another loop is
            # proving is not the operator's to prove over (its slot would be
            # freed under that loop, which would go on to commit a
            # duplicate as id_2), and one still unclaimed must not be
            # picked up by a loop while the verifiers run.
            buffered = own_id in {e["lemma"]["id"] for e in state.buffer.snapshot()}
            if buffered and state.buffer.claim_id(loop_id, own_id) is None:
                log(
                    f"  {own_id} is claimed by another loop, which is "
                    f"proving it; pick another, or let that loop finish.",
                    verbose,
                )
                continue
            verified = _verify_operator_proof(
                choice, dag, references, ref_ids, susp.suspended,
                conjecture, verifier_sys, verbose,
            )
            if verified is None:
                if buffered:
                    # Back in the buffer for any loop, this one included.
                    state.buffer.release_claims(loop_id)
                continue
            lid, node, vh = verified
            with _commit_section():
                committed = state.add_lemma(lid, node, vh, verbose)
                state.buffer.consume(lid)
                if committed != lid:
                    # A sibling loop committed this id while the verifiers
                    # ran: its entry stands under the id, the operator's
                    # proof goes in under the renamed one.
                    log(
                        f"Lemma {lid} entered the DAG in the meantime; "
                        f"yours went in as {committed}.",
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
        cand = _clean_candidate(lemma, conjecture)
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
        f"{loop_id} starts"
        + (" (the human loop — it asks before each lemma)."
           if is_human() else " (automatic)."),
        verbose,
    )
    while state.running():
        # The suspension as this claim stands to be run (see
        # suspension.py; the engine recomputes its own every round): the
        # "already in the DAG" test below runs against
        # the lemmas that are not suspended — a suspended lemma a previous
        # run died re-proving is not "already in the DAG" for this test, it
        # is exactly what the resume is for.
        dag_now = load_dag()
        susp_now = suspension.compute(
            dag_now["lemmas"], load_references(), CONJECTURE_ROOT
        )
        suspended_now = susp_now.suspended
        pending = state.take_in_flight(loop_id)
        if pending is not None:
            # A prover round a previous run cancelled mid-flight: re-run
            # exactly that round. It does not spend a new iteration — the
            # one it started under was already spent — so the resume is
            # checked before the budget: a loop that exhausted its budget
            # still finishes the lemma it died holding.
            if _committed_uncertified(dag_now, susp_now, pending):
                state.set_in_flight(loop_id, None)
                state.buffer.consume(
                    pending.get("claimed_id") or pending["lemma_id"]
                )
                log(_uncertified_message(pending["lemma_id"]))
                continue
            if pending["lemma_id"] in lemma_id_set(dag_now, suspended_now):
                # The stop landed after the lemma was proved: on_proof had
                # already committed it. The serial run's same case — nothing
                # to resume; drop the position and the buffer claim, and go
                # about the next lemma (the spent iteration is spent).
                state.set_in_flight(loop_id, None)
                state.buffer.consume(
                    pending.get("claimed_id") or pending["lemma_id"]
                )
                log(
                    f"{loop_id}'s in-flight {pending['lemma_id']} is "
                    f"already in the DAG; resuming is moot.",
                    verbose,
                )
                continue
            initial = pending
            fresh_target = None
            log(
                f"{loop_id} resumes {pending['lemma_id']} at prover round "
                f"{pending['attempt']}/{MAX_PROOF_ATTEMPTS}.",
                verbose,
            )
        else:
            if state.iterations_used(loop_id) >= state.max_iterations:
                log(
                    f"{loop_id} has spent its budget "
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
                action, lemma = _parallel_human_step(
                    state, loop_id, verbose, conjecture,
                    references, ref_ids, verifier_sys,
                )
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
        def on_proof(
            lid: str, node: Dict[str, Any], verified_hash: Optional[str],
        ) -> str:
            # add_lemma checks the id against the DAG as it stands at
            # write time, not as it stood at the top of this loop's round:
            # a sibling that committed the same id in the meantime keeps
            # its proof under the id, and this one is committed under a
            # fresh non-colliding id rather than dropped or stacked on
            # top of it. The rename goes on the shared reject list, under
            # the id it happened under, where the planner and the other
            # loops can see that the two proofs may state different
            # things.
            with _commit_section():
                # All three verifier checks accepted this proof: add_lemma
                # certifies it under the id it is committed under (a
                # renamed id is the lemma that is in the DAG now), then
                # writes it.
                committed = state.add_lemma(lid, node, verified_hash, verbose)
                if committed != lid:
                    log(
                        f"{loop_id}: lemma {lid} was taken by another route "
                        f"while this proof was in flight; committed under "
                        f"{committed} instead.",
                        verbose,
                    )
                    with state.failed_lock:
                        state.failed_attempts.setdefault(lid, []).append(
                            f"Id collision: this loop's proof of {lid} was "
                            f"committed under {committed} because the id was "
                            f"already in the DAG; the two proofs may state "
                            f"different things."
                        )
                    state.write_checkpoint()
            return committed
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
            on_proof=on_proof,
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
        # buffer slot the claim occupied, and record the failure where the
        # planner and the other loops can see it. The slot is keyed on the
        # claim id, not on the final target: a decomposition settles on a
        # lemma the buffer never held, so freeing by the final id would
        # leak the claim's slot and the generator would never be told the
        # claimed id is free. The failure note, by contrast, belongs on
        # the final target — that is the statement that was actually tried.
        state.set_in_flight(loop_id, None)
        state.buffer.consume(result["claimed_id"])
        state.finish_lemma(loop_id)
        if not result["proved"]:
            if result["attempt_notes"]:
                state.record_failure(result["lemma_id"], result["attempt_notes"])
        # A proved lemma is already in the DAG (on_proof committed it, which
        # bumped the version and marked the plan stale): the loop simply
        # goes for its next lemma.
        if not state.running():
            break
    # The loop's exit is announced by its thread wrapper (_proof_thread in
    # run_parallel), in a finally, so a crash announces it too.


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
        f"\nParallel mode: {parallel_loops} proof loop(s), one planner, "
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
    _ensure_user_dag_file()
    cp = load_parallel_checkpoint(loop_ids, verbose)
    if cp is not None and not os.path.exists(DAG_FILE):
        # The serial run's rule, kept: deleting the user's DAG file is how
        # the operator resets the run, and a checkpoint outliving its DAG
        # would resurrect the old budget and in-flight lemma.
        log("Checkpoint without a DAG file; discarding it.", verbose)
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
            f"Resumed the parallel checkpoint: {used} lemma(s) spent of "
            f"{MAX_ITERATIONS} each, {pending} prover round(s) in flight.",
            verbose,
        )
    state.write_checkpoint()
    # The planner and the generator are restarted after a crash, the way a
    # proof loop is: without them the loops would wait on an empty buffer
    # (no generator) or on a plan that never comes (no planner) for the
    # rest of the run. A server context 400 (a mis-sized --num-ctx) is not
    # a crash — every restart would 400 the same way — and turns into the
    # one loud message plus a stop of the shared state, so the run ends
    # cleanly with its checkpoint intact. So does a thread that crashes
    # MAX_LOOP_RESTARTS times in a row without completing a round: the
    # run cannot go on without it, and a stop beats a hang.
    def _steering_thread(name: str, run) -> None:
        restarts = 0
        rounds_at_start = state.steering_rounds_done(name)
        while True:
            try:
                run()
                return
            except llm_backend.ContextLengthError as e:
                _fail_context_length(e, stop=state.request_stop)
                return
            except Exception as e:
                # Logged unconditionally: a crash the operator must see is
                # not verbose noise.
                log(
                    f"The parallel {name} crashed: {type(e).__name__}: "
                    f"{e}\n" + traceback.format_exc().rstrip()
                )
                if state.steering_rounds_done(name) > rounds_at_start:
                    restarts = 0
                if restarts >= MAX_LOOP_RESTARTS:
                    log(
                        f"The parallel {name} crashed {restarts + 1} "
                        f"times in a row without completing a round; "
                        f"stopping the run. The checkpoint keeps every "
                        f"loop's position: re-run the same command to "
                        f"resume."
                    )
                    state.request_stop()
                    return
                restarts += 1
                rounds_at_start = state.steering_rounds_done(name)
                log(
                    f"The parallel {name}: restart {restarts}/"
                    f"{MAX_LOOP_RESTARTS} in {LOOP_RESTART_DELAY}s."
                )
                if state.stop_event.wait(LOOP_RESTART_DELAY):
                    return

    def _planner_thread() -> None:
        _steering_thread("planner", lambda: _parallel_planner(
            state, verbose, planner_sys, conjecture, comments_block,
            ref_block,
        ))

    def _generator_thread() -> None:
        _steering_thread("generator", lambda: _parallel_generator(
            state, verbose, generator_sys, conjecture, ref_block,
        ))

    def _proof_thread(i: int, lid: str) -> None:
        restarts = 0
        finished_at_start = state.lemmas_finished(lid)
        try:
            while True:
                try:
                    _parallel_proof_loop(
                        state, lid, i, verbose, conjecture, prover_sys,
                        reviser_sys, reference_block, reviser_ref_block,
                        references, verifier_sys, ref_ids,
                    )
                    return
                except llm_backend.ContextLengthError as e:
                    # A mis-sized window, not a crash: every restart would
                    # 400 the same way.
                    _fail_context_length(e, stop=state.request_stop)
                    return
                except Exception as e:
                    # Logged unconditionally: a crash the operator must see
                    # is not verbose noise.
                    log(
                        f"{lid} crashed: {type(e).__name__}: {e}\n"
                        + traceback.format_exc().rstrip()
                    )
                    # The bound is on crashes in a row: a loop that finished
                    # a lemma since its last (re)start was making progress,
                    # so this crash starts the count again.
                    if state.lemmas_finished(lid) > finished_at_start:
                        restarts = 0
                    if restarts >= MAX_LOOP_RESTARTS:
                        log(
                            f"{lid} crashed {restarts + 1} times in a row "
                            f"without finishing a lemma; giving it "
                            f"up for this run. Its in-flight round stays in "
                            f"the checkpoint for the next run."
                        )
                        return
                    restarts += 1
                    finished_at_start = state.lemmas_finished(lid)
                    # The restart resumes the round the loop last published
                    # (take_in_flight), and its claim with it; any other
                    # claim the crash left behind (claimed, but no round
                    # published yet) goes back to the buffer, or its slot
                    # would stay claimed for the rest of the run.
                    pending = state.take_in_flight(lid)
                    keep = (
                        (pending.get("claimed_id") or pending["lemma_id"])
                        if pending is not None else None
                    )
                    released = state.buffer.release_claims(lid, keep=keep)
                    log(
                        f"{lid}: restart {restarts}/{MAX_LOOP_RESTARTS} in "
                        f"{LOOP_RESTART_DELAY}s"
                        + (f", resuming {pending['lemma_id']}"
                           if pending is not None else "")
                        + (f"; released {', '.join(released)} to the buffer"
                           if released else "")
                        + "."
                    )
                    if state.stop_event.wait(LOOP_RESTART_DELAY):
                        return
        finally:
            # Every exit path (budget spent, human quit, a stop, the buffer
            # waiting giving up, an unexpected exception) funnels here:
            # announce this loop is out, and if it was the last, send the
            # stop that frees the planner and generator — nothing more will
            # ever be claimed or added, so there is no point left in their
            # waiting. Without the finally, a loop that crashed would never
            # be counted out and the run would hang at join().
            if state.note_loop_exit():
                state.request_stop()

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

    if state.resolution is None:
        # The last loop out may have committed the settling lemma after
        # the planner stopped watching.
        dag = load_dag()
        susp = suspension.compute(dag["lemmas"], references, CONJECTURE_ROOT)
        kind = _settled(dag, susp.suspended, conjecture, verbose)
        if kind is not None:
            state.set_resolution(kind)
    if state.resolution is not None:
        # Nothing is written to the DAG: the settling lemma is the record,
        # and the resolution is recomputed from it (resolution.py).
        discard_checkpoint()
        log(
            f"\nConjecture {state.resolution}; the checkpoint is cleared.",
            verbose,
        )
        return load_dag()
    state.write_checkpoint()
    buffer_snapshot = state.buffer.snapshot()
    log(
        f"\nParallel run finished without settling the conjecture: "
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
def main(argv: Optional[List[str]] = None, prog: Optional[str] = None) -> None:
    """Run the main proof loop.

    argv is the argument list (defaults to sys.argv[1:]); prog is the name
    usage lines show. The proofs command calls this as
    main(argv=["--conjecture", DIR, ...], prog="proofs run"); running the
    file directly (python proofs/main.py --conjecture NAME) parses
    sys.argv the same way.
    """
    global MODEL_NAME, CONJECTURE_FILE, CONJECTURE_ROOT, DAGS_DIR, DAG_FILE
    global REFERENCES_FILE, COMMENTS_FILE
    global MAX_ITERATIONS, NUM_CTX, USER, LEMMAS_ALL, MY_LEMMAS, TOOL_MODE
    global LEMMAS_COMMON
    global BACKEND, PROFILE, PROMPT_PATHS, MODE, HOTKEY
    global MAX_PROOF_ATTEMPTS, CHECKPOINT_FILE

    parser = argparse.ArgumentParser(
        prog=prog, description="Run the multi-agent theorem prover."
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
             "$LLAMA_HOST or http://localhost:8081.",
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
            "iteration budget from 1. The DAG files are kept: delete "
            "dags/<user_id>_dag.json (the current user's file) for a "
            "genuinely clean run."
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
    add_lemma_window_flags(parser)
    add_tool_flags(parser)
    args = parser.parse_args(argv)
    LEMMAS_ALL = args.lemmas_all
    MY_LEMMAS = args.my_lemmas
    LEMMAS_COMMON = args.lemmas_common
    TOOL_MODE = args.tool_mode

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

    # The run's user, before the paths: the DAG file is named after it,
    # and every lemma the run commits is stored under it. The model never
    # emits a user_id; this is the one that goes in (see config.py).
    try:
        USER = config.ensure_user_id()
        paths = workspace.resolve(args.conjecture, USER)
    except config.UserConfigError as e:
        parser.error(str(e))
        return  # unreachable; parser.error exits, but keeps type checkers calm
    except workspace.ConjectureNotFound as e:
        parser.error(str(e))
        return  # unreachable; parser.error exits, but keeps type checkers calm

    CONJECTURE_FILE = str(paths.conjecture)
    CONJECTURE_ROOT = str(paths.root)
    DAGS_DIR = str(paths.dags_dir)
    DAG_FILE = str(paths.dag)
    CHECKPOINT_FILE = checkpoint_path_for(DAG_FILE)
    REFERENCES_FILE = str(paths.references)
    COMMENTS_FILE = str(paths.comments)
    PROMPT_PATHS = {name: str(p) for name, p in paths.prompts.items()}
    log(workspace.describe(paths), args.verbose)
    if args.fresh:
        if os.path.exists(CHECKPOINT_FILE):
            discard_checkpoint()
            log(f"--fresh: deleted {os.path.basename(CHECKPOINT_FILE)}; "
                f"the iteration budget restarts from 1. The DAG is kept.",
                args.verbose)
        else:
            log("--fresh: no checkpoint to delete.", args.verbose)

    # The listener puts the terminal in cbreak mode, so it has to be stopped on
    # every exit path — including a traceback — or the shell you return to has
    # no echo. Hence try/finally rather than a stop() at the end of the run.
    HOTKEY = interaction.HotKey(
        args.hotkey,
        enabled=bool(args.hotkey),
        on_press=lambda: print(
            f"\nMode change queued: "
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
