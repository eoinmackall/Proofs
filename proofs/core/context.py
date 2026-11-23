"""The explicit context every part of a run is handed, in place of module
globals.

One Context per command invocation, made by the entry point (proofs run,
verify, repair, status, prune, export) and passed to whatever needs it.
Its pieces:

    user      the user.id the command acts as
    run_id    the run this command is (core.checkpoint)
    paths     the conjecture's files (Paths), set once the conjecture
              is resolved
    settings  the tunables: flags over .proofs-local.json over the
              defaults (workspace.settings.Settings)
    models    the models the command talks to and the role each serves,
              with what the run has learned about each
              (llm.routing.Models: models.for_role(role)); None for the
              commands that call no model
    code      the run_code machinery, when the command was given --tools
              (Code)
    run       state the whole process shares and mutates: the mode, the
              hotkey, the interrupt bookkeeping, the locks (RunState)
    loop      the calling thread's own state (LoopState): each thread of
              a parallel run works on its own copy of the Context, made
              by for_loop(), which shares every other piece
    ui        where output goes (ui.plain.PlainUI unless a renderer is
              installed)

The pieces are created as the command needs them: a command that calls no
model never makes Models, one without --tools never fills Code.
"""

from __future__ import annotations

import copy
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set, Tuple

from ..ui.plain import PlainUI
from ..workspace.settings import Settings


@dataclass
class Paths:
    """The conjecture's files, as the run reads and writes them."""

    conjecture_root: str = ""
    conjecture_file: str = ""
    dags_dir: str = ""
    checkpoint_file: str = ""
    references_file: str = ""
    comments_file: str = ""
    prompt_paths: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_run_paths(cls, paths: Any, checkpoint_file: str = "") -> "Paths":
        """From workspace.resolve()'s RunPaths."""
        return cls(
            conjecture_root=str(paths.root),
            conjecture_file=str(paths.conjecture),
            dags_dir=str(paths.dags_dir),
            checkpoint_file=checkpoint_file,
            references_file=str(paths.references),
            comments_file=str(paths.comments),
            prompt_paths={name: str(p) for name, p in paths.prompts.items()},
        )


@dataclass
class Code:
    """The run_code machinery (setup_code): empty without --tools."""

    # The runner every run_code call goes through.
    runner: Any = None
    # What makes the run_code tool for each role that is offered it
    # ("prover", "reviser", "verifier"), given the agent's directory
    # (tools/run_code/tool.py: run_code with all but scripts bound).
    tools: Dict[str, Any] = field(default_factory=dict)
    # How many proof loops may run code: the machine-wide job slots this
    # process holds.
    loops: int = 0
    # The note the prover's task carries on how to cite a computation.
    answer_note: str = ""
    # proofs verify --run-code: re-execute a lemma's scripts first.
    run_code: bool = False


def _null_hotkey() -> Any:
    from ..ui.input import NullHotKey
    return NullHotKey()


@dataclass
class RunState:
    """What the whole process shares and mutates while it runs."""

    # "auto" or "human"; the hotkey and the menu retarget it mid-run.
    mode: str = "auto"
    # The mode toggle (ui.input.HotKey); a listener that never fires
    # until proofs run starts one.
    hotkey: Any = field(default_factory=_null_hotkey)
    # The latest safe-boundary state of a serial run, for the SIGINT
    # handler.
    live_state: Dict[str, Any] = field(default_factory=lambda: {
        "iteration": None,
        "rejected_lemmas": None,
        "in_flight": None,
        "plan": None,
    })
    # The live ParallelState, or None outside a parallel run.
    parallel: Any = None
    # The interrupt bookkeeping (core.interrupts).
    sigint_count: int = 0
    commit_cond: threading.Condition = field(default_factory=threading.Condition)
    commits_in_progress: int = 0
    sigint_pending: bool = False
    deferred_interrupt: Any = None
    # The open run (core.checkpoint.Run): it holds the run's lock for as
    # long as the process lives.
    handle: Any = None
    # The in-memory DAG (core.dag.Dag), loaded on first use: the state the
    # command started from, plus its own commits. Here rather than on the
    # Context so every thread's copy shares it.
    dag: Any = None
    # The search index over the DAG's statements (tools/search_lemmas),
    # started the first time search_lemmas is offered and kept up to date
    # by the commits; None until then.
    search_index: Any = None
    # The lemma window the planning agents were last shown
    # (loops/strategies.py), chosen when the planner starts: the parallel
    # lemma generator is shown the same one, and so, in a serial run, are
    # the prover and the reviser (window_from_planner).
    planning_window: Any = None
    window_from_planner: bool = False
    # The once-a-run notes already given (by what each is about).
    said: Set[str] = field(default_factory=set)
    # The run's stream writers, by kind (core.dag.writer), and every lemma id
    # the run has handed out (core.dag.new_lemma_id), committed or not.
    writers: Dict[str, Any] = field(default_factory=dict)
    issued_ids: Set[str] = field(default_factory=set)
    id_lock: Any = field(default_factory=threading.Lock)
    # The DAG lock (commits, one at a time) and the checkpoint lock (see core.dag and
    # core.checkpoint).
    dag_lock: Any = field(default_factory=threading.Lock)
    checkpoint_lock: Any = field(default_factory=threading.Lock)


@dataclass
class LoopState:
    """The calling thread's own state."""

    # The agent this thread is, as the status board names its row: "main"
    # for a serial run, verify or repair; "loop-N" for the workers of a
    # parallel run.
    agent: str = "main"
    # Whether this thread's loop may run code (the first code.loops loops
    # of a parallel run).
    code_loop: bool = True
    # The code directory of the proof loop this thread is running
    # (tools/run_code/files.py); None outside one.
    code_dir: Optional[Path] = None
    # The lemma window this thread's agents are shown (loops/strategies.py,
    # Window), chosen when its planner or prover starts; and _fit_prompt's
    # smaller sizes for its parts while it rebuilds a prompt too large for
    # the context window.
    window: Any = None
    window_override: Optional[Tuple[int, ...]] = None
    # Set (a threading.Event) on a verifier thread of a parallel
    # verification (loops/verifiers.py, fan_out): once it fires, the
    # thread's next model request raises reasoning.Cancelled instead.
    cancel: Any = None
    # Model fallback (--with-fallback, loops/reasoning.py): the roles this
    # thread's calls are pinned to another model for, by role — a call
    # failed over, or a verification check restarted on the fallback
    # (fan_out hands a check's pins to its verifier threads); the model
    # that served each role's latest answer, which is the one its record
    # names; and how to fit each role's prompt again for another model's
    # window (prompts._fit_prompt): (the prompt it fitted, a refit).
    pinned: Dict[str, Any] = field(default_factory=dict)
    served: Dict[str, Any] = field(default_factory=dict)
    fitted: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Context:
    user: str = ""
    # The run this command is (core.checkpoint): its id names the run's
    # directory, checkpoint and scratch. Empty for the commands that call
    # no model.
    run_id: str = ""
    paths: Paths = field(default_factory=Paths)
    settings: Settings = field(default_factory=Settings)
    models: Any = None
    code: Code = field(default_factory=Code)
    run: RunState = field(default_factory=RunState)
    loop: LoopState = field(default_factory=LoopState)
    ui: Any = field(default_factory=PlainUI)

    @property
    def dag(self) -> Any:
        """The in-memory DAG (core.dag.Dag): loaded from the files the first
        time it is asked for, then changed only by this command's commits."""
        if self.run.dag is None:
            from .dag import Dag
            self.run.dag = Dag.load(self)
        return self.run.dag

    def status(self, **fields: str) -> None:
        """Update this thread's row on the status board (ui.status): role,
        lemma, attempt, event."""
        self.ui.status(self.loop.agent, **fields)

    def for_loop(self, **loop_fields: Any) -> "Context":
        """A copy for one thread: its own LoopState, every other piece
        shared."""
        clone = copy.copy(self)
        clone.loop = LoopState(**loop_fields)
        return clone
