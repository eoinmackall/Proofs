"""Ctrl-C: commit sections it must not split, and the checkpoint it writes."""

from __future__ import annotations

import contextlib
import os
import sys
import threading

from ..tools.run_code import execution
from .checkpoint import save_checkpoint

COMMIT_WAIT_SECONDS = 30


@contextlib.contextmanager
def _commit_section(ctx):
    """Hold off Ctrl-C while an accepted lemma is committed and certified
    (see RunState.commit_cond). In the serial run an interrupt that arrived inside
    is replayed when the section ends."""
    with ctx.run.commit_cond:
        ctx.run.commits_in_progress += 1
    try:
        yield
    finally:
        with ctx.run.commit_cond:
            ctx.run.commits_in_progress -= 1
            ctx.run.commit_cond.notify_all()
            replay = (
                ctx.run.sigint_pending
                and ctx.run.commits_in_progress == 0
                and threading.current_thread() is threading.main_thread()
            )
            if replay:
                ctx.run.sigint_pending = False
        if replay:
            if ctx.run.deferred_interrupt is not None:
                ctx.run.deferred_interrupt()
            else:
                _interrupt_serial(ctx)


def _raise_interrupt() -> None:
    raise KeyboardInterrupt


def _on_sigint_repair(ctx, signum, frame) -> None:
    """Ctrl-C during proofs repair: a KeyboardInterrupt, as with no
    handler at all, except that one landing while a re-proved lemma is
    being certified and committed is held off until both are done
    (_commit_section) — split, the reproof would be left uncertified. A
    second Ctrl-C force-exits."""
    ctx.run.sigint_count += 1
    if ctx.run.sigint_count > 1:
        ctx.ui.close()
        ctx.ui.log("\nForced exit.", err=True)
        os._exit(130)
    if ctx.run.commits_in_progress:
        ctx.run.sigint_pending = True
        ctx.ui.log(
            "\nInterrupted — finishing the commit of the re-proved "
            "lemma first (Ctrl-C again to force)...",
            err=True,
        )
        return
    raise KeyboardInterrupt


def _on_sigint(ctx, signum, frame) -> None:
    """Ctrl-C: write the checkpoint, then exit 130 like the shell expects.

    A second Ctrl-C force-exits without waiting on the file. The checkpoint
    is written from ctx.run.live_state — the last step run_loop() reached —
    so an interrupt mid-LLM-call loses only that call: the next run re-runs
    the interrupted prover round, not the whole iteration.
    """
    ctx.run.sigint_count += 1
    # Code executions run in their own session, where the terminal's
    # Ctrl-C never reaches them: they are killed here, first, whatever
    # happens to the run (execution.kill_all).
    execution.kill_all()
    if ctx.run.sigint_count > 1:
        ctx.ui.close()
        ctx.ui.log("\nForced exit.", err=True)
        os._exit(130)
    if ctx.run.parallel is not None:
        # Parallel mode: the shared state is the checkpoint. Ask every thread
        # to stop at its next boundary, let any commit in progress finish
        # (its lemma and its certificate land together), snapshot the whole
        # run under the state lock, and exit; the daemon threads die with
        # the process and the next run resumes from the snapshot.
        ctx.run.parallel.request_stop()
        ctx.ui.log("\nInterrupted — writing checkpoint...", err=True)
        with ctx.run.commit_cond:
            if ctx.run.commits_in_progress:
                ctx.ui.log(
                    f"   waiting for {ctx.run.commits_in_progress} lemma commit(s) "
                    f"to finish (Ctrl-C again to force)...",
                    err=True,
                )
            ctx.run.commit_cond.wait_for(
                lambda: ctx.run.commits_in_progress == 0, timeout=COMMIT_WAIT_SECONDS
            )
        ctx.run.parallel.write_checkpoint()
        ctx.ui.log(
            f"   {os.path.basename(ctx.paths.checkpoint_file)} written. Re-run the same "
            f"command to resume where this run stopped.",
            err=True,
        )
        sys.exit(130)
    if ctx.run.commits_in_progress:
        # Serial mode: the handler runs on the main thread, which is inside
        # the commit section itself, so it cannot wait for it — it defers,
        # and the section replays the interrupt as it ends.
        ctx.run.sigint_pending = True
        ctx.ui.log(
            "\nInterrupted — finishing the commit of the accepted lemma "
            "first (Ctrl-C again to force)...",
            err=True,
        )
        return
    _interrupt_serial(ctx)


def _interrupt_serial(ctx) -> None:
    """The serial interrupt: write the checkpoint from ctx.run.live_state and exit
    130."""
    ctx.ui.log("\nInterrupted — writing checkpoint...", err=True)
    state = ctx.run.live_state
    if (
        state["iteration"] is not None
        and state["rejected_lemmas"] is not None
        and ctx.paths.checkpoint_file
    ):
        save_checkpoint(
            ctx,
            state["iteration"],
            state["rejected_lemmas"],
            state["in_flight"],
            plan=state.get("plan"),
            verbose=False,
        )
        ctx.ui.log(
            f"   {os.path.basename(ctx.paths.checkpoint_file)} written. Re-run the same "
            f"command to resume where this run stopped.",
            err=True,
        )
    else:
        ctx.ui.log("   (no checkpoint state yet — nothing written)", err=True)
    sys.exit(130)
