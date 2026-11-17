"""The verification steps: one call per verifier agent."""

from __future__ import annotations

import queue
import threading
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from ..core.dag import load_file
from ..validity import verdicts
from .json_repair import parse_json_or_none
from .mode import check_mode_toggle
from .prompts import verifier_tools
from .reasoning import Cancelled, extract, reason
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


class _QuietWhenCancelled:
    """A verifier thread's UI: the shared one, until the thread is
    cancelled. A verifier nobody is waiting for any more says nothing:
    its log lines would read as part of the verification that moved on,
    and its status would put back the row fan_out took away."""

    def __init__(self, ui: Any, cancel: threading.Event) -> None:
        self._ui = ui
        self._cancel = cancel

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ui, name)

    def log(self, *args: Any, **kwargs: Any) -> None:
        if not self._cancel.is_set():
            self._ui.log(*args, **kwargs)

    def status(self, *args: Any, **kwargs: Any) -> None:
        if not self._cancel.is_set():
            self._ui.status(*args, **kwargs)


def stopping(ctx) -> bool:
    """Whether ctx is a verifier thread fan_out has cancelled."""
    return ctx.loop.cancel is not None and ctx.loop.cancel.is_set()


class fan_out:
    """Run a proof's verification steps, up to `workers` at once, and hand
    back each step's result as it comes in.

        with fan_out(ctx, steps, run_step, workers) as results:
            for agent, result in results:
                ...

    run_step(step_ctx, agent) is one step; its result is the caller's to
    read. With workers <= 1 (or a single step) the steps run one after
    another in the calling thread on ctx itself, exactly as a plain loop
    would. Otherwise each step gets a thread of its own context
    (Context.for_loop: its own status row, "<agent>/<role>", and the code
    directory the calling loop would have given it) and results arrive in
    the order the verifiers finish, not in step order.

    The caller decides when the verification is over. drain() starts no
    further step but keeps handing back the ones already running (a
    missing verdict: a later rejection is still worth having). Leaving
    the with block — a break on a rejection, a return, an exception —
    cancels the rest: no step starts, and a running one sends no further
    request (reasoning.Cancelled) and is never waited for. The request it
    is in the middle of cannot be taken back; it finishes on the server
    and its reply is dropped. The threads are daemons, so a process that
    exits does not wait for them.

    An exception in run_step is raised in the calling thread, in place of
    that step's result.
    """

    def __init__(
        self,
        ctx,
        steps: Sequence[str],
        run_step: Callable[[Any, str], Any],
        workers: int,
    ) -> None:
        self.ctx = ctx
        self.steps = list(steps)
        self.run_step = run_step
        self.workers = max(1, min(workers, len(self.steps)))
        self._cancel = threading.Event()
        self._draining = False
        # Under _lock: the steps running, and the results not yet handed
        # back. A worker files its result and stops counting as running in
        # one step, so the calling thread never sees a step that is
        # neither. Steps are started by the calling thread only, between
        # results, so a drain() is in force before the next one starts.
        self._lock = threading.Lock()
        self._running: List[str] = []
        self._results: "queue.Queue[Tuple[str, bool, Any]]" = queue.Queue()

    @property
    def parallel(self) -> bool:
        return self.workers > 1

    def __enter__(self) -> "fan_out":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def drain(self) -> None:
        """Start no further step; the running ones are still handed back."""
        self._draining = True

    def close(self) -> None:
        """Stop: start nothing more, and cancel the steps still running."""
        self._draining = True
        self._cancel.set()
        with self._lock:
            running, self._running = self._running, []
        for agent in running:
            self.ctx.ui.done(self._row(agent))

    def abandoned(self) -> List[str]:
        """The steps still running (call before leaving the with block: these
        are the ones it cancels)."""
        with self._lock:
            return list(self._running)

    def _row(self, agent: str) -> str:
        return f"{self.ctx.loop.agent}/{agent[:-3]}"

    def _start(self, agent: str) -> None:
        with self._lock:
            self._running.append(agent)
        threading.Thread(target=self._worker, args=(agent,), daemon=True).start()

    def _worker(self, agent: str) -> None:
        row = self._row(agent)
        code_dir = self.ctx.loop.code_dir
        if code_dir is None and self.ctx.code.runner is not None:
            # Where files.loop_dir() puts a verifier outside a proof loop,
            # so the scripts land where a sequential run's do.
            code_dir = self.ctx.code.runner.scratch_dir / self.ctx.loop.agent
        sctx = self.ctx.for_loop(
            agent=row, code_loop=self.ctx.loop.code_loop,
            code_dir=code_dir, cancel=self._cancel,
        )
        sctx.ui = _QuietWhenCancelled(self.ctx.ui, self._cancel)
        try:
            result, ok = self.run_step(sctx, agent), True
        except Cancelled:
            return
        except BaseException as e:  # raised again in the calling thread
            result, ok = e, False
        finally:
            self.ctx.ui.done(row)
        with self._lock:
            if self._cancel.is_set():
                return
            self._results.put((agent, ok, result))
            self._running.remove(agent)

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        if not self.parallel:
            for agent in self.steps:
                if self._draining:
                    return
                yield agent, self.run_step(self.ctx, agent)
            return
        todo = list(self.steps)
        while True:
            while todo and not self._draining \
                    and len(self._running) < self.workers:
                self._start(todo.pop(0))
            with self._lock:
                if not self._running and self._results.empty():
                    return
            try:
                # A timeout, so Ctrl-C reaches the calling thread.
                agent, ok, result = self._results.get(timeout=0.5)
            except queue.Empty:
                continue
            if not ok:
                raise result
            yield agent, result
