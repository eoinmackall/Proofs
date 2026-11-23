"""Multi-agent proof loop: planner -> prover -> verifier -> reviser -> DAG.

Design note on thinking vs. structured output
---------------------------------------------
llama.cpp's `json_schema` response_format is *ignored* while reasoning is on,
so the model is free to wrap its JSON in a markdown fence, and the
server's own JSON parser then 500s (ggml-org/llama.cpp#20345). The fix has
two halves:

  * `llm/openai_compat.py` neutralizes the conflict per call: thinking is turned
    off for exactly the calls that carry a schema (the extraction stage),
    and left on everywhere else.
  * `extract()` additionally degrades to prompt-only with a JSON repair
    pipeline in `clean_json_text()`, in case the model still wraps the
    output in a fence or emits stray prose.

The reasoning stage stays schema-free, because that is exactly where the
model needs to think.

All six agent prompts (planner.md, prover.md, verifier_1.md,
verifier_2.md, verifier_3.md and reviser.md) instruct the model to emit
strict JSON directly, so each
response is first
parsed as-is; the extraction stage only runs as a fallback when the model
fails to comply. The prover never round-trips through a second model call at
all: its JSON is parsed locally and, failing that, the reasoning content is
taken verbatim, so the proof text can't be abridged or paraphrased.

Cancelling and resuming
-----------------------
Every proofs run is a run with an id of its own and a checkpoint, kept in
conjectures/NAME/checkpoints/<user_id>/<run_id>/ (local and gitignored;
see core/checkpoint.py). The checkpoint holds what the DAG files do not:
the position of the iteration budget, the latest plan, the planner's reject
list (`rejected_lemmas`: the statements of the last run.max_rejected_lemmas
lemmas that went through every prover round unproved), and each loop's
lemma in flight — its statement as the reviser last left it, the prover
round, the step it is at (prove, verify, revise), the prover's output for
that round and the verdicts of the verifiers that have finished. It is
written after every step, and by the SIGINT handler itself on Ctrl-C,
which writes the last step's state and exits 130; a second Ctrl-C
force-exits. Re-running the same command continues the most recent run
from its checkpoint: the budget goes on where it stopped, the saved plan
stands in for the planner, and a loop cancelled mid-lemma picks up at the
step it was on — the verifiers that had not finished, or the reviser — so a
Ctrl-C costs the interrupted call and nothing before it. --fresh starts a
new run instead, and --resume RUN_ID continues a given one. When the
conjecture is settled (resolution.py) the run's checkpoint is deleted.

Planning is a shortlist, not a decision
---------------------------------------
The planner proposes PLANNER_CANDIDATES (5) lemmas per iteration rather than
one, in no particular order, and something downstream picks exactly one. In
--mode auto the pick is random: one candidate is chosen uniformly at random
(select_lemma). The parallel run's equivalent is the buffer, which the proof
loops drain oldest-first. Every candidate is given its permanent id by the
code when it is proposed (core.dag.new_lemma_id), so no two can collide and
none needs screening.

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
run.max_proof_attempts prover rounds per lemma per iteration
(--max-proof-attempts), then gives the lemma up and asks the planner again;
the statement it gave up on goes on the reject list the planner reads.

When a call runs into the context wall
--------------------------------------
The server is stateless, so a truncated exchange — prompt, thinking trace,
answer-so-far, summing to about ctx_budget — cannot be re-sent whole. reason()
therefore gets pi-style compaction rescues before reporting "ceiling":
the thinking trace (scratch, regenerable) is summarised into a structured
summary — Goal, Progress, Key Decisions, Next Steps, Critical Context, plus
a <cited-lemmas> tracking block — in chunked passes if it is big; the task
prompt and the answer written so far are kept verbatim, the answer ending
exactly where it was cut; and the model is asked to resume from the cut with
the whole headroom of the now-smaller prompt. A continuation that hits the
wall itself is not a failure but the next pass: its trace folds into the
summary, its text extends the verbatim answer, and the model resumes from
the new cut. The rescue is bounded to llm.max_compaction_passes passes (two:
the first usually finishes an answer that merely ran long, the second
covers the genuinely long proofs). Still unfinished after that many — or a
pass that leaves nothing to build on, or no headroom left — degrades to the
"ceiling" with the partial work ({"summary", "thinking", "answer"}) for the
caller to hand on: the proof loop compacts it a final time via
_incomplete_report() and sends the summary plus the partial answer to the
reviser (the overflow case of reviser.md) to pick a smaller lemma or revise
the statement; only when there is nothing to build on, or the reviser has
nothing usable, does the planner decompose.

Room for the reply
------------------
The window is shared by the prompt and the reply, and the server is told
only the max_tokens reason() asks for, so the room for thinking and
answering is kept here. Every agent prompt that shows the lemma window is
fitted before it is sent (_fit_prompt): one that would leave less than the
reply's reserve (RESERVE_TOKENS) plus room for lookups (LOOKUP_ROOM) is
rebuilt with a smaller window — the lemmas moved out of it are one search
away. Inside a call, the prompt is sized from the server's
own count of the last one plus what was appended since; lookups close once
the prompt reaches the reserve, a result that would cut into it is trimmed
to fit, and each result tells the model how much room is left. A prompt
that leaves less than COMPACT_MIN_ROOM even so is never sent: reason()
returns "too_large", and each caller does something other than send it
again.

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
menu is already up and the listener is therefore paused. See ui/input.py
for why one of these is a keystroke and the other is a line of input.

Usage
-----
    proofs run conjectures/algebra_example --host http://localhost:8081 --model qwen3.8-27b
    proofs run conjectures/algebra_example --host http://localhost:8081 --model qwen3.8-27b --mode human
    proofs parse conjectures/algebra_example
    proofs verify algebra_example alice/3f9a2c1b4d5e6f70/42 --full
    proofs repair conjectures/algebra_example/refutations/alice--3f9a2c1b4d5e6f70--42--bob--20250102T030405Z.json
    proofs status algebra_example alice/3f9a2c1b4d5e6f70/42
    proofs export algebra_example --format latex -o ~/tex
    proofs export algebra_example alice/3f9a2c1b4d5e6f70/42 --format html -o ~/graphs

proofs run calls main() with DIR as --conjecture; verify, repair, prune,
status and export call the *_entry functions below.

proofs parse (parsing.py) is standalone: the conjecture's maintainer points
it at conjectures/<name>/references/ (any .tex or .md file in it), it asks
the model for the theorem-level results, and writes them as a strict-JSON
array to conjectures/<name>/references.md, which the maintainer commits with
the conjecture. The reference collection is a shared, committed artefact:
every other user pulls references.md and never regenerates it, and a result
they need that is missing is requested from the maintainer. The run only
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
directory through git; nothing here runs git itself. Every run appends to
streams of its own, conjectures/<name>/dags/<user_id>/<run_id>/ (user_id
from the config file, never emitted by a model; see core/streams.py), and
every command reads every run's streams to see the complete DAG. A lemma's
id, <user_id>/<run_id>/<hex>, is drawn when the candidate is proposed and
is unique everywhere; no hashes are stored in the streams.

Trust is recorded beside the DAG, never in it. Each acceptance of a lemma
is a certificate in the accepting run's stream under
conjectures/<name>/certificates/ (certificates.py); each rejection
by proofs verify or proofs repair is a file in conjectures/<name>/
refutations/ (refutations.py). Both carry the lemma's Merkle hash
(merkle.py), so either one lapses on its own when the lemma, or anything it
cites, changes. A lemma without a valid certificate, with a refutation
that still matches it, or citing such a lemma is suspended (suspension.py):
hidden from the planner and the prover until proofs verify or
proofs repair re-accepts it. The conjecture is settled by an unsuspended
lemma stating it, or its negation, verbatim (resolution.py).

"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path
from typing import List, Optional

from .. import llm
from ..cli.flags import (
    add_backend_flags, add_lemma_window_flags, add_settings_flags,
    add_tool_flags, model_flags,
)
from ..loops.mode import DEFAULT_MODE, HOTKEY_KEYS
from ..loops.parallel import run_parallel
from ..loops.models import attach as attach_models
from ..loops.models import build as build_models
from ..loops.reasoning import _fail_context_length, _fail_quota
from ..loops.serial import run_loop
from ..tools.run_code.setup import add_code_flags, setup_code
from ..ui import live
from ..ui.input import HotKey
from ..workspace import config
from ..workspace import layout as workspace
from ..workspace import settings as settings_mod
from . import streams
from .checkpoint import RunError, attach_run, open_run
from .context import Context, Paths
from .dag import _warn_suspended
from .interrupts import _on_sigint


def main(
    argv: Optional[List[str]] = None,
    prog: Optional[str] = None,
    root: Optional[Path] = None,
) -> None:
    """Run the main proof loop.

    argv is the argument list (defaults to sys.argv[1:]); prog is the name
    usage lines show; root is the workspace root a bare --conjecture NAME
    is looked up under (workspace.resolve), None outside a workspace. The
    proofs command calls this as main(argv=["--conjecture", DIR, ...],
    prog="proofs run", root=...).
    """
    ctx = Context()

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
    add_backend_flags(parser)
    parser.add_argument(
        "--conjecture", required=True,
        help=f"A conjecture directory containing "
             f"{workspace.CONJECTURE_FILENAME}: a name under the workspace "
             f"root's {workspace.CONJECTURES_ROOT}/, or a path to it "
             "(required)",
    )
    add_settings_flags(parser)
    which = parser.add_mutually_exclusive_group()
    which.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Start a new run, with an empty checkpoint, instead of "
            "continuing the most recent one. The DAG files are kept."
        ),
    )
    which.add_argument(
        "--resume", default=None, metavar="RUN_ID",
        help=(
            "Continue the given run (by default the most recent one is "
            "continued, or a new one started if it is running elsewhere)."
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
    parser.add_argument(
        "--ui", choices=("auto", "live", "plain"), default="auto",
        help=(
            "How the run reports: live keeps a status board of every agent "
            "at the bottom of the terminal, under the scrolling log; plain "
            "prints the log lines only. auto (default): live on a terminal, "
            "plain otherwise."
        ),
    )
    add_lemma_window_flags(parser)
    add_tool_flags(parser)
    add_code_flags(parser)
    args = parser.parse_args(argv)
    ctx.ui, ui_note = live.choose(args.ui, args.verbose)
    if ui_note:
        ctx.ui.log(f"note: {ui_note}", err=True)
    # The settings first: flags over the conjecture's .proofs-local.json
    # over the workspace's over the defaults — so the conjecture's
    # directory before them, and the model routing, which is checked
    # (offline) before anything slow: a typo in a route is reported before
    # the sandbox relaunch and the probes.
    try:
        conjecture_dir = workspace.conjecture_dir(args.conjecture, root)
        ctx.settings = settings_mod.resolve(
            root, settings_mod.flag_values(args), conjecture_dir)
        models = build_models(ctx.settings, model_flags(args))
    except (workspace.ConjectureNotFound, settings_mod.SettingsError,
            llm.RoutingError) as e:
        parser.error(str(e))
    # Before the backend probe: with --tools this re-execs the command
    # inside the sandbox, and the probe is the relaunched process's to make.
    ctx.code = setup_code(args, parser, root, ui=ctx.ui)
    ctx.settings.lemmas_all = args.lemmas_all
    ctx.settings.my_lemmas = args.my_lemmas
    ctx.settings.strategy, ctx.settings.strategy_dial = args.strategy
    ctx.settings.window_size = args.window_size
    ctx.settings.tool_mode = args.tool_mode

    ctx.run.mode = args.mode

    # The models next: each probe tells the real context ceiling, which an
    # explicit window is clamped to (llm.connect), and a dead server is
    # reported before the run's files are touched.
    try:
        attach_models(ctx, models, args.verbose)
    except (llm.ProbeError, llm.RoutingError) as e:
        parser.error(str(e))

    # The run's user, before the paths: the run's streams are named after it,
    # and every lemma the run commits is stored under it. The model never
    # emits a user_id; this is the one that goes in (see config.py).
    try:
        ctx.user = config.ensure_user_id()
        paths = workspace.resolve(args.conjecture, ctx.user, root)
    except config.UserConfigError as e:
        parser.error(str(e))
        return  # unreachable; parser.error exits, but keeps type checkers calm
    except workspace.ConjectureNotFound as e:
        parser.error(str(e))
        return  # unreachable; parser.error exits, but keeps type checkers calm

    ctx.paths = Paths.from_run_paths(paths)
    ctx.ui.log(workspace.describe(paths), args.verbose)
    try:
        run = open_run(paths.root, ctx.user, fresh=args.fresh, resume=args.resume)
    except RunError as e:
        parser.error(str(e))
    attach_run(ctx, run)
    ctx.ui.log(
        f"run {run.id}" + (" (continuing it)" if run.resumed else " (new)"),
        args.verbose,
    )
    ctx.ui.header(conjecture=paths.root.name, model=ctx.models.summary(),
                  run=run.id)
    # The DAG is built once, here (a stream that cannot be read stops the
    # run with its file and line), and what the build found worth saying
    # is said once.
    try:
        dag = ctx.dag
    except streams.StreamError as e:
        parser.error(str(e))
    for warning in dag.warnings:
        ctx.ui.log(f"warning: {warning}", err=True)
    # Build the initial suspension cache. A citation recheck can later
    # refute a stored lemma and invalidate this cache for the new snapshot.
    _warn_suspended(ctx, dag.suspension(), args.verbose)

    # The listener puts the terminal in cbreak mode, so it has to be stopped on
    # every exit path — including a traceback — or the shell you return to has
    # no echo. Hence try/finally rather than a stop() at the end of the run.
    ctx.run.hotkey = HotKey(
        args.hotkey,
        enabled=bool(args.hotkey),
        on_press=lambda: ctx.ui.log(
            f"\nMode change queued: "
            f"{'automatic' if ctx.run.mode == 'human' else 'manual'} "
            f"from the next planning step."
        ),
    ).start()
    if ctx.run.mode == "auto" and ctx.run.hotkey.hint():
        ctx.ui.log(f"mode=auto {ctx.run.hotkey.hint()}", args.verbose)
    elif ctx.run.mode == "human":
        ctx.ui.log("mode=human — you pick the lemma at every planning step.", args.verbose)

    # Ctrl-C writes the checkpoint from the safe-boundary state in
    # ctx.run.live_state and exits 130; a second Ctrl-C force-quits. Installed
    # now, once the paths exist, so the handler can always write.
    signal.signal(
        signal.SIGINT, lambda signum, frame: _on_sigint(ctx, signum, frame)
    )

    ctx.ui.start()
    try:
        if args.parallel is not None:
            run_parallel(ctx, max(1, args.parallel), verbose=args.verbose)
        else:
            run_loop(ctx, verbose=args.verbose)
    except llm.ContextLengthError as e:
        _fail_context_length(ctx, e)
        sys.exit(1)
    except llm.QuotaExhausted as e:
        _fail_quota(ctx, e)
        sys.exit(1)
    finally:
        ctx.ui.close()
        ctx.run.hotkey.stop()
