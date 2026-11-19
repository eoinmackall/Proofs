"""Parallel mode

N workers share one DAG, one version-2 checkpoint, and one possible-lemma
buffer, and N is every model call the run makes at once: there is no
planner or generator thread. Each worker, between lemmas, takes the most
urgent task there is (ParallelState.next_task):

  plan      re-plan the shared strategy — at the start, after
            run.replan_every lemmas have been committed since the plan in
            force was written, or when the operator asks
  generate  restock the buffer with up to k fresh candidate lemmas
            (batch_size) — when the stock runs low, or when workers have
            nothing to claim
  prove     claim the oldest lemma in the buffer, run the shared proof
            engine (run_proof_loop) on it, record the outcome

When --mode human, loop 0 is the operator's loop: each fresh start goes
to the menu with the buffer as its candidate list — the parallel version
of the serial shortlist menu — and the hotkey toggles it between the menu
and automation exactly as in the serial run. It takes plan and generate
tasks like any other worker. Every write to the shared state goes through
ParallelState, and every write to the DAG goes through commit_lemma_to_dag
under the DAG lock, each lemma under the id its candidate was given, so a
reader can never see a half-state.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
import traceback
from typing import Any, Callable, Collection, Dict, List, Optional, Set, Tuple

from .. import llm
from .. import tools as tools_mod
from ..core import merkle, resolution
from ..workspace import settings as settings_mod
from ..core.checkpoint import (
    _atomic_write_checkpoint_file, clean_rejected, discard_checkpoint,
    drop_rejection, load_parallel_checkpoint, note_rejection,
)
from ..core.dag import (
    _committed_in_run, _committed_uncertified, _settled, _uncertified_message,
    commit_lemma_to_dag, load_file, new_lemma_id,
)
from ..core.interrupts import _commit_section
from ..validity.steps import VERIFIER_AGENTS
from .human import _operator_citation_check, _verify_operator_proof, choose
from .json_repair import parse_json_or_none
from .mode import check_mode_toggle
from . import strategies
from .prompts import (
    _build_comments_block, _build_reference_blocks, _fit_prompt, agent_tools,
    candidate, planner_dag_view, rejected_block, replan_note,
)
from .proof import run_proof_loop
from .reasoning import Cancelled, _fail_context_length, extract, reason
from .schemas import PARALLEL_LEMMA_GENERATOR_SCHEMA, PARALLEL_PLANNER_SCHEMA

LOOP_RESTART_DELAY = 10

# How long a plan or generate task waits before it is tried again after a
# round that came back with nothing usable.
STEERING_RETRY_DELAY = 10


def batch_size(num_ctx: int, loops: int) -> int:
    """k, the most candidate lemmas one generator call is asked for: 4 at a
    64K window, two more for each doubling and two fewer for each halving
    (2 at 32K, 6 at 128K), at least 1 and at most the loops. Each candidate
    in a call shares its thinking and its searches with the others, so a
    larger window affords a few more, and a few good candidates beat many
    thin ones."""
    k = round(4 + 2 * math.log2(max(num_ctx, 1) / 65536))
    return max(1, min(k, loops))


_QUIT = object()  # sentinel: the human loop asked the run to stop


class LemmaBuffer:
    """The shared possible-lemma list: exactly N+2k fixed slots (k is
    batch_size) — room for the N claimed lemmas and two batches, one being
    used up while the next is written.

    The generate tasks fill the empty slots; the proof loops claim them. A
    generate task reserves the slots its batch will take before it calls
    the model (reserve), so generators running at once never ask for more
    than the slots can take. The unattended claim treats the buffer as a
    queue: the oldest unclaimed lemma goes first, by arrival order rather
    than slot position (a freed slot is refilled by whatever arrives next).
    Claiming is atomic, so two loops never work on one id.
    """

    def __init__(self, size: int, marked_cap: int = 0) -> None:
        # Each slot is {"lemma": candidate, "claimed_by": loop id or None,
        # "seq": arrival number}, or None while the slot is empty. seq is
        # what makes the buffer a queue; a released claim keeps its place.
        self.slots: List[Optional[Dict[str, Any]]] = [None] * size
        self._next_seq = 0
        # Reentrant: ParallelState schedules under this same condition and
        # calls the _locked helpers from inside it.
        self.cond = threading.Condition()
        # The empty slots each running generate task has reserved, by the
        # worker running it.
        self.reservations: Dict[str, int] = {}
        # A lemma marked "computation" (a parallel run with --tools) is
        # claimed only by a loop that may run code, and at most marked_cap
        # of them — one per coding loop — sit in the slots at once. One
        # beyond that is discarded, as any candidate without a slot is.
        self.marked_cap = marked_cap
        # Called after every change to the slots, with
        # self.cond released: ParallelState sets it to its checkpoint
        # writer, which takes self.cond itself (checkpoint_entries).
        self.on_change: Callable[[], None] = lambda: None

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

    def _room_locked(self) -> int:
        """How many new candidates the slots can take: the empty slots no
        generate task has reserved. Call with self.cond held."""
        empty = sum(1 for e in self.slots if e is None)
        return max(0, empty - sum(self.reservations.values()))

    def _stock_locked(self, code: bool) -> int:
        """The lemmas waiting for a loop that may (code) or may not run
        code, and those on their way: the unclaimed slots it could claim,
        and the reserved slots, whatever their batches turn out to hold. A
        loop without code does not count the marked lemmas it can never
        take. Call with self.cond held."""
        return (len(self._claimable_locked(code))
                + sum(self.reservations.values()))

    def _reserve_locked(self, owner: str, n: int) -> int:
        """Reserve up to n empty slots for owner's generate task; returns
        how many. Call with self.cond held."""
        n = min(n, self._room_locked())
        if n > 0:
            self.reservations[owner] = self.reservations.get(owner, 0) + n
        return n

    def release(self, owner: str) -> None:
        """Drop owner's reservation, if it still has one: its generate task
        ended without filling (a failed call, a crash)."""
        with self.cond:
            if self.reservations.pop(owner, 0):
                self.cond.notify_all()

    def _marked_locked(self) -> int:
        return sum(1 for e in self.slots
                   if e is not None and e["lemma"].get("computation"))

    def _place_locked(self, cand: Dict[str, Any]) -> bool:
        for i, e in enumerate(self.slots):
            if e is None:
                self.slots[i] = {
                    "lemma": cand, "claimed_by": None, "seq": self._next_seq,
                }
                self._next_seq += 1
                return True
        return False

    def _statements_locked(self) -> Set[str]:
        """The normalized statements of every lemma in the slots. Call
        with self.cond held."""
        return {
            merkle.normalize(e["lemma"]["statement"])
            for e in self.slots if e is not None
        }

    def fill(
        self, candidates: List[Dict[str, Any]],
        taken: Collection[str] = (),
        owner: Optional[str] = None,
    ) -> Tuple[int, int]:
        """Put fresh candidates into the empty slots, first come first
        served, in place of owner's reservation (released here, so the
        slots it held are the ones the candidates take). A candidate whose
        statement (up to merkle.normalize's whitespace) is already in the
        buffer, or in taken, the normalized statements the loops hold in
        flight, is dropped: every candidate has a new id, so only the
        statement can tell that two loops would be proving the same
        thing. A marked candidate past the cap is discarded (see
        __init__). Returns (how many landed in the slots, how many were
        dropped as duplicates)."""
        added = duplicates = 0
        with self.cond:
            if owner is not None:
                self.reservations.pop(owner, None)
            for cand in candidates:
                statement = merkle.normalize(cand["statement"])
                if statement in taken or statement in self._statements_locked():
                    duplicates += 1
                    continue
                if cand.get("computation") and \
                        self._marked_locked() >= self.marked_cap:
                    continue
                if self._place_locked(cand):
                    added += 1
            if added or owner is not None:
                self.cond.notify_all()
        if added:
            self.on_change()
        return added, duplicates

    def _claimable_locked(self, code: bool) -> List[Dict[str, Any]]:
        return [
            e for e in self.slots
            if e is not None and e["claimed_by"] is None
            and (code or not e["lemma"].get("computation"))
        ]

    def claim_oldest(self, loop_id: str, code: bool = True) -> Optional[Dict[str, Any]]:
        """The unattended pick: claim the unclaimed lemma that entered the
        buffer first. A loop that may run code takes a lemma marked
        "computation" first, the oldest of them — no other loop can; a loop
        that may not never takes one. None when nothing is claimable."""
        with self.cond:
            claimed = self._claim_oldest_locked(loop_id, code)
        if claimed is not None:
            self.on_change()
        return claimed

    def _claim_oldest_locked(self, loop_id: str, code: bool) -> Optional[Dict[str, Any]]:
        """claim_oldest's claim, without its checkpoint: the caller holds
        self.cond, and calls on_change once it has let go of it."""
        waiting = self._claimable_locked(code)
        if not waiting:
            return None
        marked = [e for e in waiting if e["lemma"].get("computation")]
        oldest = min(marked or waiting, key=lambda e: e["seq"])
        oldest["claimed_by"] = loop_id
        self.cond.notify_all()
        return oldest["lemma"]

    def claim_id(self, loop_id: str, lemma_id: str) -> Optional[Dict[str, Any]]:
        """Claim a named lemma — the human loop's pick. Same atomicity as
        claim_oldest(): None when the id is not in the buffer unclaimed."""
        claimed = None
        with self.cond:
            for e in self.slots:
                if (
                    e is not None
                    and e["lemma"]["id"] == lemma_id
                    and e["claimed_by"] is None
                ):
                    e["claimed_by"] = loop_id
                    self.cond.notify_all()
                    claimed = e["lemma"]
                    break
        if claimed is not None:
            self.on_change()
        return claimed

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
        if released:
            self.on_change()
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
                    break
            else:
                return
        self.on_change()

    def restore(self, entries: List[Dict[str, Any]]) -> int:
        """Put a checkpoint's buffer back (core/checkpoint.py): each
        {"lemma", "claimed_by"} into a slot in its saved order, the claimed
        ones first, an unclaimed marked lemma past the cap discarded, the
        way fill() would. The slots are sized by this run's batch
        (batch_size), so a buffer saved with a larger window may not fit:
        returns how many unclaimed lemmas were dropped for want of a
        slot."""
        dropped = 0
        with self.cond:
            ordered = ([e for e in entries if e.get("claimed_by")]
                       + [e for e in entries if not e.get("claimed_by")])
            for e in ordered:
                cand = e["lemma"]
                if cand.get("computation") and not e.get("claimed_by") and \
                        self._marked_locked() >= self.marked_cap:
                    continue
                if self._place_locked(cand):
                    for slot in self.slots:
                        if slot is not None and slot["lemma"] is cand:
                            slot["claimed_by"] = e.get("claimed_by")
                else:
                    dropped += 1
            self.cond.notify_all()
        return dropped

    def checkpoint_entries(self) -> List[Dict[str, Any]]:
        """The buffer as the checkpoint keeps it: the slots in arrival
        order."""
        with self.cond:
            slots = sorted((e for e in self.slots if e is not None),
                           key=lambda e: e["seq"])
            return [{"lemma": e["lemma"], "claimed_by": e["claimed_by"]}
                    for e in slots]


class ParallelState:
    """Everything the parallel workers share, the scheduler that hands them
    their tasks, and the checkpoint writer.

    One lock per concern, never held across an LLM call:
      cond          the buffer's condition (LemmaBuffer.cond): the slots,
                    the reservations, and the scheduler's own state — the
                    plan, the running plan and generate tasks, the waiting
                    workers; every worker with nothing to do waits on it
      state_lock    the per-loop slots, the DAG version, the operator note
      rejected_lock the shared reject list
      run.dag_lock  the commits to the run's streams (Context.run)
      run.checkpoint_lock  the checkpoint's snapshot-then-replace
    cond is taken before state_lock or rejected_lock, never after, and is
    never held while the checkpoint is written. The checkpoint is a
    whole-file rewrite taken under the checkpoint lock — the loop slots
    under state_lock, the reject list under rejected_lock, the plan and the
    buffer under cond, then one atomic replace over a per-writer temp file
    — so the file is always a consistent, current state even though many
    threads poke at it.
    """

    def __init__(
        self, ctx, loop_ids: List[str], max_iterations: int,
        conjecture: str, verbose: bool,
    ) -> None:
        self.ctx = ctx
        self.conjecture = conjecture
        self.verbose = verbose
        self.batch = batch_size(ctx.settings.num_ctx, len(loop_ids))
        self.buffer = LemmaBuffer(len(loop_ids) + 2 * self.batch,
                                  marked_cap=self.ctx.code.loops)
        self.buffer.on_change = self.write_checkpoint
        self.cond = self.buffer.cond
        self.loops: Dict[str, Dict[str, Any]] = {
            lid: {"in_flight": None, "iterations_used": 0} for lid in loop_ids
        }
        # How many tasks each worker has completed this run — lemmas
        # settled, plans installed, batches offered: the crash-restart count
        # resets when this moves, so the restart bound is on crashes with
        # no finished task between them. Run-local, not checkpointed.
        self.progress: Dict[str, int] = {lid: 0 for lid in loop_ids}
        self.max_iterations = max_iterations
        # The plan in force (under cond); the DAG version when it was
        # installed (plan_due); and whether the operator has asked for a
        # new one.
        self.plan: Optional[Dict[str, Any]] = None
        self.plan_version = 0
        self.plan_stale = False
        # The running plan task's epoch and cancel event: the operator's
        # replan supersedes a plan being written (request_replan) by moving
        # the epoch on, and a plan from an older epoch is never installed.
        self.plan_epoch = 0
        self.plan_cancel: Optional[threading.Event] = None
        # The scheduler's state, under cond: whether a worker is planning,
        # the focus (a priority's index) of each running generate task by
        # its worker, how many workers are waiting for a task, and the plan
        # and generate tasks held back — until a time (a round that gave
        # nothing usable), or until their inputs change (a prompt too
        # large to send).
        self.planning = False
        self.generators: Dict[str, int] = {}
        self.idle = 0
        self.not_before: Dict[str, float] = {}
        self.hold: Dict[str, Tuple[Any, ...]] = {}
        self.dag_version = 0
        self.rejected_lemmas: List[Dict[str, str]] = []
        self.rejected_lock = threading.Lock()
        # The operator's last 'replan' reason, for the next plan only
        # (under state_lock; not checkpointed).
        self.operator_note = ""
        self.stop_event = threading.Event()
        self.resolution: Optional[str] = None
        self.state_lock = threading.Lock()
        # Keeps two commits from checking the DAG at once.
        self.dag_check_lock = threading.Lock()
        # The workers still inside their loops, each with whether it may
        # run code. The run stops when none is left: a worker that has
        # spent its budget may still be running the engine on its final
        # claim, and stopping it mid-engine would orphan a lemma that was
        # claimed, budgeted, and never proved.
        self.live: Dict[str, bool] = {
            lid: _may_run_code(ctx, i) for i, lid in enumerate(loop_ids)
        }

    # -- run control ------------------------------------------------------

    def running(self) -> bool:
        return not self.stop_event.is_set()

    def note_loop_exit(self, loop_id: str) -> bool:
        """A worker left its loop. True if it was the last one out."""
        with self.state_lock:
            self.live.pop(loop_id, None)
            return not self.live

    def has_work(self) -> bool:
        """Whether a live worker has work, or can get some without a new
        batch: a lemma it holds (claimed, or in flight — a decomposition's
        smaller lemma is in no slot), or one in the buffer it may claim. A
        given-up worker's round stays in flight for the next run, and a
        marked lemma waits for a coding loop: neither counts once no live
        worker can take it."""
        with self.cond:
            with self.state_lock:
                live = dict(self.live)
                if any(self.loops[lid]["in_flight"] is not None for lid in live):
                    return True
            return any(
                e is not None and e["claimed_by"] in live
                for e in self.buffer.slots
            ) or bool(self.buffer._claimable_locked(any(live.values())))

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
        with self.cond:
            self.cond.notify_all()

    # -- the scheduler -----------------------------------------------------

    def next_task(
        self, loop_id: str, code: bool, human: bool,
        on_wait: Callable[[], None] = lambda: None,
    ) -> Optional[Tuple[str, Any]]:
        """The worker's next task, the most urgent first; None on a stop.

          ("plan", (epoch, cancel))  a plan is due (_plan_due_locked) and
                               nobody is writing one: the task's epoch, and
                               the event the operator's replan sets to
                               supersede it (request_replan)
          ("generate", (n, focus))  the buffer needs restocking (_restock):
                               n slots are reserved for this worker's
                               batch; focus is the priority it is pointed
                               at when other generate tasks are running,
                               else None
          ("menu", None)       the human loop's pick, once there is a plan
                               and none is being written
          ("prove", lemma)     the oldest lemma this worker may claim,
                               claimed

        A worker with none of these waits — on_wait is told, once — until
        something changes. No new batch is written while a new plan is
        being written: it would be steered by the plan on its way out."""
        claimed = None
        waited = False
        with self.cond:
            self.idle += 1
            try:
                while True:
                    if self.stop_event.is_set():
                        return None
                    now = time.monotonic()
                    wake: Optional[float] = None
                    if not self.planning and self._plan_due_locked():
                        ready, wake = self._ready_locked("planner", now, wake)
                        if ready:
                            self.planning = True
                            self.plan_cancel = threading.Event()
                            return ("plan", (self.plan_epoch, self.plan_cancel))
                    if self.plan is not None and not self.planning:
                        n = self._restock_locked(code)
                        if n:
                            ready, wake = self._ready_locked("generator", now, wake)
                            if ready:
                                n = self.buffer._reserve_locked(loop_id, n)
                                focus = self._focus_locked()
                                self.generators[loop_id] = focus if focus is not None else 0
                                return ("generate", (n, focus))
                    if human:
                        # Once the plan being written, if any, is in, and
                        # the buffer has a lemma or none is on its way.
                        if self.plan is not None and not self.planning and (
                                self.buffer._claimable_locked(code)
                                or not self.generators):
                            return ("menu", None)
                    else:
                        claimed = self.buffer._claim_oldest_locked(loop_id, code)
                        if claimed is not None:
                            break
                    if not waited:
                        waited = True
                        on_wait()
                    self.cond.wait(None if wake is None else max(0.0, wake - now))
            finally:
                self.idle -= 1
        self.buffer.on_change()
        return ("prove", claimed)

    def _plan_due_locked(self) -> bool:
        """Whether a new plan is due: there is none, the operator asked for
        one, or run.replan_every lemmas have been committed since the plan
        in force was installed (never, at -1). Call with cond held."""
        if self.plan is None or self.plan_stale:
            return True
        every = self.ctx.settings.replan_every
        return (every >= 0
                and self.dag_version_snapshot() - self.plan_version >= every)

    def _restock_locked(self, code: bool) -> int:
        """How many candidates a generate task should be asked for now, 0
        for none. The stock is the asking worker's own
        (LemmaBuffer._stock_locked: what it could claim, and what is on its
        way), so the workers without code restock when the lemmas they can
        take run low, however many marked ones wait for the coding loops.
        It is restocked when it falls to one batch, so that a batch is still
        there to be used while the next one is written — and also while it
        is short of the workers waiting with nothing to claim, which is what
        starts more than one generate task at once (at the start, or after
        a failed call). Call with cond held."""
        stock = self.buffer._stock_locked(code)
        starved = not self.buffer._claimable_locked(code) and stock < self.idle
        if stock > self.batch and not starved:
            return 0
        return min(self.batch, self.buffer._room_locked())

    def _focus_locked(self) -> Optional[int]:
        """The priority a new generate task is pointed at: None when it is
        the only one (it reads the whole plan), else the first priority no
        running task has — the first running task counts as having the
        first, the one a whole-plan reading starts from. Call with cond
        held."""
        if not self.generators or self.plan is None:
            return None
        count = len(self.plan["priorities"])
        if count < 2:
            return None
        taken = set(self.generators.values())
        free = [i for i in range(count) if i not in taken]
        return free[0] if free else len(self.generators) % count

    def _ready_locked(
        self, name: str, now: float, wake: Optional[float],
    ) -> Tuple[bool, Optional[float]]:
        """Whether the plan or generate task (name) may run now, and the
        earliest time a held-back one may (to wait until). Call with cond
        held."""
        until = self.not_before.get(name, 0.0)
        if until > now:
            return False, until if wake is None else min(wake, until)
        held = self.hold.get(name)
        if held is not None:
            if _steering_inputs(self) == held:
                return False, wake
            del self.hold[name]
        return True, wake

    def end_planning(self, epoch: int) -> None:
        """The plan task of epoch ended, a plan installed or not. A task the
        operator superseded no longer holds the planning flag: the plan
        written in its place may be under way."""
        with self.cond:
            if epoch == self.plan_epoch:
                self.planning = False
                self.plan_cancel = None
            self.cond.notify_all()

    def request_replan(self, note: str) -> bool:
        """The operator's replan, with the note the next plan is told: due
        at once, held back by nothing. A plan being written when it comes
        in is superseded — written without the note, it would answer a
        question the operator has moved past — so it is cancelled at its
        next model request and never installed, and the new plan may start
        on the next free worker. True when a plan was superseded."""
        with self.cond:
            superseded = self.planning
            if superseded:
                self.plan_epoch += 1
                if self.plan_cancel is not None:
                    self.plan_cancel.set()
                self.plan_cancel = None
                self.planning = False
            self.not_before.pop("planner", None)
            self.hold.pop("planner", None)
            with self.state_lock:
                self.operator_note = note
            self.plan_stale = True
            self.cond.notify_all()
        return superseded

    def end_generating(self, loop_id: str) -> None:
        """loop_id's generate task ended: its reservation, if it did not
        fill, goes back."""
        with self.cond:
            self.generators.pop(loop_id, None)
            self.cond.notify_all()
        self.buffer.release(loop_id)

    def retry_later(self, name: str, epoch: Optional[int] = None) -> None:
        """The plan or generate task (name) came back with nothing usable:
        hold it back for STEERING_RETRY_DELAY, so a model that keeps
        failing is not called in a tight loop. A plan task passes its
        epoch: one the operator superseded holds nothing back, since the
        replan cleared the hold for the plan written in its place."""
        with self.cond:
            if epoch is not None and epoch != self.plan_epoch:
                return
            self.not_before[name] = time.monotonic() + STEERING_RETRY_DELAY
            self.cond.notify_all()

    def hold_until_changed(
        self, name: str, inputs: Tuple[Any, ...], epoch: Optional[int] = None,
    ) -> None:
        """The plan or generate task's prompt (name) was too large to
        send: hold it back until what it is built from changes, since the
        same inputs build the same prompt. inputs are the ones the prompt
        was built from (_steering_inputs, taken before it was built), not
        the ones now: a change made while the call was on the server frees
        the task at once. epoch as for retry_later."""
        with self.cond:
            if epoch is not None and epoch != self.plan_epoch:
                return
            self.hold[name] = inputs
            self.cond.notify_all()

    def stuck(self, inputs: Tuple[Any, ...]) -> bool:
        """Whether a steering prompt built from inputs that was too large
        to send can never be built again any smaller: nothing has changed
        since (_steering_inputs), and no live worker has work to change
        anything (has_work). A run is stopped on this alone."""
        return not self.has_work() and _steering_inputs(self) == inputs

    def note_progress(self, loop_id: str) -> None:
        """loop_id completed a task: a lemma settled, a plan installed, a
        batch offered."""
        with self.state_lock:
            self.progress[loop_id] += 1

    def progress_of(self, loop_id: str) -> int:
        with self.state_lock:
            return self.progress[loop_id]

    # -- the plan ----------------------------------------------------------

    def install_plan(self, plan: Dict[str, Any], epoch: int) -> bool:
        """Install a plan written by the plan task of epoch. False when
        the operator superseded the task, and the plan is dropped."""
        with self.cond:
            if epoch != self.plan_epoch:
                return False
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
            self.plan_version = self.dag_version_snapshot()
            self.plan_stale = False
            self.cond.notify_all()
        self.write_checkpoint()
        return True

    def plan_snapshot(self) -> Optional[Dict[str, Any]]:
        with self.cond:
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
        """The lemmas the loops are holding in flight right now: whether
        any loop still has work, for the steering tasks' too-large stop."""
        with self.state_lock:
            return {
                d["in_flight"]["lemma_id"]
                for d in self.loops.values()
                if d.get("in_flight") is not None
            }

    def in_flight_statements(self) -> Set[str]:
        """The normalized statements the loops are proving right now — each
        in-flight target as the reviser last left it, a decomposition's
        smaller lemma included, which the buffer never held. The generator
        drops a candidate that restates one (LemmaBuffer.fill): a resumed
        run's buffer starts without the lemmas its loops resume, and a
        decomposed target is in no slot at all."""
        with self.state_lock:
            return {
                merkle.normalize(d["in_flight"]["target"]["statement"])
                for d in self.loops.values()
                if d.get("in_flight") is not None
            }

    # -- the DAG -------------------------------------------------------------

    def add_lemma(
        self, lemma_id: str, node: Dict[str, Any],
        verified_hash: Optional[str], verbose: bool = True,
    ) -> str:
        """Commit a lemma to the shared DAG (commit_lemma_to_dag, under the
        DAG lock, so the loops commit one at a time). Then bump the DAG
        version (the count the next re-plan waits on), take the lemma off
        the reject list, check whether the DAG now settles the conjecture
        (check_dag), wake the scheduler, and checkpoint. Returns the id the
        lemma was committed under."""
        committed = commit_lemma_to_dag(self.ctx, lemma_id, node, verified_hash, verbose)
        with self.state_lock:
            self.dag_version += 1
        with self.rejected_lock:
            drop_rejection(self.rejected_lemmas, node["statement"])
        self.check_dag()
        with self.cond:
            self.cond.notify_all()
        self.write_checkpoint()
        return committed

    def check_dag(self) -> None:
        """After every change to the DAG, and once at the start: stop the
        run when the DAG settles the conjecture — a loop's proof, the
        operator's, or (at the start) another user's pulled file. No
        model's opinion ever settles it (resolution.py)."""
        with self.dag_check_lock:
            dag = self.ctx.dag
            kind = _settled(self.ctx, dag, dag.suspension().suspended,
                            self.conjecture, self.verbose)
        if kind is not None:
            self.ctx.ui.log("The loops are being stopped.", self.verbose)
            self.set_resolution(kind)

    def dag_version_snapshot(self) -> int:
        with self.state_lock:
            return self.dag_version

    # -- the shared reject list -----------------------------------------------

    def record_rejection(self, lemma_id: str, statement: str) -> None:
        with self.rejected_lock:
            note_rejection(self.ctx, self.rejected_lemmas, lemma_id, statement)
        with self.cond:
            self.cond.notify_all()
        self.write_checkpoint()

    def rejected_snapshot(self) -> List[Dict[str, str]]:
        with self.rejected_lock:
            return [dict(e) for e in self.rejected_lemmas]

    def take_operator_note(self, epoch: int) -> str:
        """The note for the plan task of epoch, taken: nothing for a task
        already superseded, which must leave the note to the plan written
        in its place."""
        with self.cond:
            if epoch != self.plan_epoch:
                return ""
            with self.state_lock:
                note, self.operator_note = self.operator_note, ""
                return note

    def return_operator_note(self, note: str, epoch: int) -> None:
        """The plan task of epoch took the note and wrote no plan: the note
        waits for the next one, unless the operator has written a newer
        one (or superseded the task, with a newer one)."""
        if not note:
            return
        with self.cond:
            if epoch != self.plan_epoch:
                return
            with self.state_lock:
                if not self.operator_note:
                    self.operator_note = note

    def superseded(self, epoch: int) -> bool:
        """Whether the operator's replan has superseded the plan task of
        epoch (request_replan)."""
        with self.cond:
            return epoch != self.plan_epoch

    # -- checkpoint -------------------------------------------------------------

    def apply_checkpoint(self, cp: Dict[str, Any]) -> int:
        """Take a checkpoint's state. Returns how many buffered lemmas did
        not fit this run's buffer (LemmaBuffer.restore)."""
        with self.state_lock:
            for lid, slot in self.loops.items():
                entry = cp["loops"].get(lid) or {}
                slot["in_flight"] = entry.get("in_flight")
                slot["iterations_used"] = int(entry.get("iterations_used") or 0)
        with self.rejected_lock:
            self.rejected_lemmas = clean_rejected(self.ctx, cp.get("rejected_lemmas"))
        # The saved plan steers the run from the start; the next plan is
        # due after run.replan_every commits in this run.
        if cp.get("plan") is not None:
            with self.cond:
                self.plan = cp["plan"]
                self.plan_version = 0
                self.plan_stale = False
        # A claim survives only where its loop holds the lemma in flight;
        # any other claim (taken, but no round published) is released.
        held = {
            lid: (slot["in_flight"] or {}).get("claimed_id")
            for lid, slot in self.loops.items()
        }
        return self.buffer.restore([
            {"lemma": e["lemma"],
             "claimed_by": e["claimed_by"]
             if e["claimed_by"] and held.get(e["claimed_by"]) == e["lemma"]["id"]
             else None}
            for e in cp.get("buffer") or []
        ])

    def write_checkpoint(self) -> None:
        # The checkpoint lock spans the snapshot and the replace: with the two
        # separated, a writer that snapshots first could replace last and
        # land an older state over a newer one's.
        with self.ctx.run.checkpoint_lock:
            with self.state_lock:
                loops = {
                    lid: {
                        "in_flight": slot["in_flight"],
                        "iterations_used": slot["iterations_used"],
                    }
                    for lid, slot in self.loops.items()
                }
            rejected = self.rejected_snapshot()
            plan = self.plan_snapshot()
            buffer = self.buffer.checkpoint_entries()
            _atomic_write_checkpoint_file(
                self.ctx,
                {"version": 2, "loops": loops, "rejected_lemmas": rejected,
                 "plan": plan, "buffer": buffer},
                verbose=False,
            )


def _may_run_code(ctx, i: int) -> bool:
    """Whether worker i may run code: the first code.loops workers may
    (see Code.loops), and every worker in a run without code tools."""
    return ctx.code.runner is None or i < ctx.code.loops


def _clean_candidate(ctx, raw: Any, conjecture: str) -> Optional[Dict[str, Any]]:
    """A generated lemma entry, taken the way the planner's candidates
    are (prompts.candidate: a new id, the statement, pinned when it settles
    the conjecture), with its computation mark in a run with code loops.
    An entry without a statement is dropped, not repaired, unless it
    settles the conjecture, whose statement the code writes — the prompt
    tells the generator the shape, and a dropped entry costs one generator
    round, not a proof."""
    cand = candidate(ctx, raw, conjecture)
    if cand is not None and raw.get("computation") is True and ctx.code.loops > 0:
        cand["computation"] = True
    return cand


def _steering_inputs(state: ParallelState) -> Tuple[Any, ...]:
    """What the steering prompts are built from, in a comparable form: the
    DAG's version, the plan, the reject list and the buffer."""
    return (
        state.dag_version_snapshot(),
        json.dumps(state.plan_snapshot(), sort_keys=True, default=str),
        json.dumps(state.rejected_snapshot(), sort_keys=True, default=str),
        json.dumps(state.buffer.snapshot(), sort_keys=True, default=str),
    )


# What the steering planner is told about the references and the operator's
# comments, each only while its prompt carries them (_planner_system): an
# instruction about a section that is not there invites the model to
# imagine one.
_PLANNER_REFERENCES_NOTE = """\
If "Known references" are provided, they name theorem-level results the
provers are allowed to cite by id without proving them. Priorities should
build on them rather than re-prove them."""

_PLANNER_COMMENTS_NOTE = """\
If "Human comments" are provided, they are the operator's notes on how to
attack the conjecture — suggested approaches to a proof or a counterexample,
objects or theorems worth trying, or routes worth avoiding. Weigh them: when
a comment suggests a viable route and the state of the proof allows it, make
one of your priorities a concrete version of that route. They are
suggestions, not instructions — a comment may be mistaken or stale."""


# The line in parallel_planner.md the notes take the place of, so that the
# output format stays the last thing the planner reads.
_PLANNER_NOTES_MARKER = "{{references_and_comments}}"


def _planner_system(planner_sys: str, ref_block: str, comments_block: str) -> str:
    """parallel_planner.md, with the note on the references when the
    planner is shown references (ref_block is empty without them, and at
    or above PLANNER_REFERENCE_LIMIT) and the note on the comments when
    there is a comments.md, in place of the marker line; the marker goes
    when neither applies. A prompt override without the marker gets the
    notes at its end."""
    notes = "\n\n".join(note for note, block in (
        (_PLANNER_REFERENCES_NOTE, ref_block),
        (_PLANNER_COMMENTS_NOTE, comments_block),
    ) if block)
    if _PLANNER_NOTES_MARKER in planner_sys:
        return re.sub(r"\n*" + re.escape(_PLANNER_NOTES_MARKER) + r"\n*",
                      f"\n\n{notes}\n\n" if notes else "\n\n", planner_sys)
    return f"{planner_sys.rstrip()}\n\n{notes}\n" if notes else planner_sys


def _plan_round(
    ctx,
    state: ParallelState,
    verbose: bool,
    planner_sys: str,
    conjecture: str,
    comments_block: str,
    ref_block: str,
    epoch: int,
    operator_note: str,
) -> bool:
    """One plan task: write the shared strategy and install it. True when
    a plan was installed.

    A plan is due at the start, when the operator asks, and after
    run.replan_every commits under the plan in force
    (ParallelState._plan_due_locked). The
    buffer is left as it is: its lemmas were written from the old plan,
    and the loops still take them. epoch is the task's: a plan the
    operator superseded while it was being written is dropped
    (install_plan). operator_note is the operator's note, taken for this
    task (_run_plan)."""
    # Before anything the prompt is built from is read: a change from here
    # on shows up as a change, whether or not the prompt saw it.
    inputs = _steering_inputs(state)
    dag = ctx.dag
    susp = dag.suspension()
    # Each plan is written from scratch: the previous plan is not shown, so
    # the new one is written from the DAG alone.
    rejected = rejected_block(ctx, state.rejected_snapshot())
    # The window for this plan (strategies.py); the generator chooses its
    # own on every call.
    strategies.for_planner(ctx, dag, susp.suspended,
                           strategies.plan_query(conjecture, state.plan_snapshot()))

    def build_plan() -> Tuple[str, Optional["tools_mod.ToolSet"]]:
        return (
            f"Conjecture:\n{conjecture}\n\n"
            f"Proved lemmas so far:\n{json.dumps(planner_dag_view(ctx, dag, susp.suspended), indent=2)}"
            f"{ref_block}{comments_block}\n\n"
            f"{rejected}\n"
            + (f"\n{operator_note}\n" if operator_note else ""),
            agent_tools(ctx, "parallel_planner", dag, susp.suspended),
        )
    plan_user, plan_tools = _fit_prompt(
        ctx,
        "parallel_planner", planner_sys, build_plan, verbose,
        parts={
            "the conjecture": conjecture,
            "the references block": ref_block,
            "the comments": comments_block,
            "the rejected lemmas": rejected,
        },
    )
    text, _status, _partial = reason(
        ctx,
        planner_sys, plan_user, "parallel_planner", verbose,
        tools=plan_tools,
    )
    check_mode_toggle(ctx, verbose)
    if state.superseded(epoch):
        # The operator asked for a new plan while this call was on the
        # server: whatever it returned, the plan written in its place is
        # the one that counts — this round neither installs, nor holds the
        # plan task back, nor stops the run.
        ctx.ui.log("The operator asked for a new plan while this one was "
                   "being written; it is dropped.", verbose)
        return False
    if _status == "too_large":
        # The same prompt would not fit any better: hold the task until
        # what it is built from changes. With no plan yet, nothing changed
        # since the prompt was built, and no live worker holding or able to
        # claim a lemma, nothing can — nothing is generated without a plan
        # — so the run stops instead of hanging.
        if state.plan_snapshot() is None and state.stuck(inputs):
            ctx.ui.log(
                "The parallel planner's prompt does not fit the context "
                "window even with no lemmas shown in full (its largest "
                "parts are logged above), and there is no plan to steer "
                "by. Stopping the run: re-run with a larger window "
                "(--num-ctx, and the server's own) or a smaller part.",
                verbose,
            )
            state.request_stop()
            return False
        ctx.ui.log(
            "The parallel planner's prompt does not fit the context "
            "window even with no lemmas shown in full; planning again "
            "once the DAG, the reject list or the buffer changes.",
            verbose,
        )
        state.hold_until_changed("planner", inputs, epoch)
        return False
    plan_res = parse_json_or_none(text) if text else None
    if plan_res is None and text:
        plan_res = extract(
            ctx,
            "Extract the plan. plan_summary is a short string; "
            "priorities is a list of strings, each one gap to close, "
            "in the order the text gives them.",
            text,
            PARALLEL_PLANNER_SCHEMA,
            "parallel_planner",
            verbose,
        )
    if not plan_res or not isinstance(plan_res, dict):
        ctx.ui.log(
            f"Parallel planner gave no usable plan; retrying in "
            f"{STEERING_RETRY_DELAY}s.",
            verbose,
        )
        state.retry_later("planner", epoch)
        return False
    if not state.install_plan(plan_res, epoch):
        ctx.ui.log("The operator asked for a new plan while this one was "
                   "being written; it is dropped.", verbose)
        return False
    state.note_progress(ctx.loop.agent)
    every = ctx.settings.replan_every
    ctx.ui.log(
        "Planner set the strategy"
        + ("." if every < 0 else
           f"; the next plan is due after {every} commit(s)."),
        verbose,
    )


def _generate_round(
    ctx,
    state: ParallelState,
    verbose: bool,
    generator_sys: str,
    conjecture: str,
    needed: int,
    focus: Optional[int],
) -> None:
    """One generate task: ask the generator for `needed` candidate lemmas
    — the slots reserved for them (ParallelState.next_task) — and fill the
    slots with the ones that are new: a statement already in the buffer or
    in flight on a loop is dropped (LemmaBuffer.fill), so no two loops
    prove the same statement at once. The DAG and the reject list are not
    checked here: the prompt shows the reject list and the lemma window,
    the generator reaches the rest of the DAG through the search and
    lookup tools, and a statement proved again, or tried again, is the
    generator's call to make. With other generate tasks running, focus
    points this one at a priority of its own, so their batches overlap
    less."""
    # Before anything the prompt is built from is read (as for the plan).
    inputs = _steering_inputs(state)
    plan = state.plan_snapshot() or {}
    dag = ctx.dag
    susp = dag.suspension()
    buffer_snapshot = state.buffer.snapshot()
    rejected = rejected_block(ctx, state.rejected_snapshot())
    strategies.for_generator(ctx, dag, susp.suspended,
                             strategies.plan_query(conjecture, plan))
    priorities = plan.get("priorities") or []
    focus_note = (
        f"\nAnother lemma generator is working from the same plan at the "
        f"same time. To overlap it less, concentrate this round on priority "
        f"{focus + 1}: {priorities[focus]}\n"
        if focus is not None and focus < len(priorities) else ""
    )

    def build_gen() -> Tuple[str, Optional["tools_mod.ToolSet"]]:
        return (
            f"Conjecture:\n{conjecture}\n\n"
            f"Current plan (follow its priorities; these are the lemmas the "
            f"loops are working toward):\n{json.dumps(plan, indent=2)}\n\n"
            f"Proved lemmas so far (already in the DAG; never restate "
            f"them):\n{json.dumps(planner_dag_view(ctx, dag, susp.suspended), indent=2)}\n\n"
            f"Possible lemmas currently in the buffer (claimed_by marks the "
            f"loop that has picked one up; never duplicate any of "
            f"these):\n{json.dumps(buffer_snapshot, indent=2)}\n\n"
            f"{rejected}\n"
            + _computation_note(ctx, state)
            + focus_note
            + f"\nPropose exactly {needed} candidate lemma(s) this "
            f"round.\n",
            agent_tools(ctx, "parallel_lemma_generator", dag, susp.suspended),
        )
    gen_user, gen_tools = _fit_prompt(
        ctx,
        "parallel_lemma_generator", generator_sys, build_gen, verbose,
        parts={
            "the conjecture": conjecture,
            "the plan": json.dumps(plan, indent=2),
            "the buffer": json.dumps(buffer_snapshot, indent=2),
            "the rejected lemmas": rejected,
        },
    )
    text, _status, _partial = reason(
        ctx,
        generator_sys, gen_user, "parallel_lemma_generator", verbose,
        tools=gen_tools,
    )
    check_mode_toggle(ctx, verbose)
    if _status == "too_large":
        # As for the planner: hold the task until the prompt's inputs
        # change rather than re-send it — and stop only when they have not
        # changed since it was built and nothing can change them: no live
        # worker holds a lemma or can claim one.
        if state.stuck(inputs):
            ctx.ui.log(
                "The lemma generator's prompt does not fit the context "
                "window even with no lemmas shown in full (its largest "
                "parts are logged above), and no loop has work. "
                "Stopping the run: re-run with a larger window "
                "(--num-ctx, and the server's own) or a smaller part.",
                verbose,
            )
            state.request_stop()
            return
        ctx.ui.log(
            "The lemma generator's prompt does not fit the context "
            "window even with no lemmas shown in full; generating again "
            "once the DAG, the plan, the reject list or the buffer "
            "changes.",
            verbose,
        )
        state.hold_until_changed("generator", inputs)
        return
    batch = parse_json_or_none(text) if text else None
    if batch is None and text:
        batch = extract(
            ctx,
            "Extract the candidate lemmas: an object with a "
            "candidate_lemmas list; each lemma has a statement.",
            text,
            PARALLEL_LEMMA_GENERATOR_SCHEMA,
            "parallel_lemma_generator",
            verbose,
        )
    lemmas = (batch or {}).get("candidate_lemmas") or []
    fresh: List[Dict[str, Any]] = []
    for raw in lemmas:
        cand = _clean_candidate(ctx, raw, conjecture)
        if cand is not None:
            fresh.append(cand)
    fresh = fresh[:needed]
    added, duplicates = state.buffer.fill(
        fresh, taken=state.in_flight_statements(), owner=ctx.loop.agent)
    state.note_progress(ctx.loop.agent)
    ctx.ui.log(
        f"Lemma generator offered {len(fresh)} new candidate(s); "
        + (f"{duplicates} restated a buffered or in-flight lemma and "
           f"were dropped; " if duplicates else "")
        + f"{added} entered the buffer "
        f"({len(state.buffer.snapshot())}/{len(state.buffer)} slots full).",
        verbose,
    )
    if added == 0:
        # Nothing landed (no offer, or every offer a duplicate or a marked
        # lemma past the cap): hold the task back, so a generator that keeps failing does not take
        # worker after worker.
        state.retry_later("generator")


def _computation_note(ctx, state: ParallelState) -> str:
    """What the lemma generator is told about marking a lemma for
    computation: nothing in a run without code tools."""
    if ctx.code.loops == 0:
        return ""
    marked = sum(1 for e in state.buffer.snapshot() if e["lemma"].get("computation"))
    return (
        f"\n{ctx.code.loops} of the proof loops can run code "
        f"({', '.join(ctx.code.runner.languages)}). Add \"computation\": true to "
        f"a candidate only when proving it needs a computer search or an "
        f"exact finite check; only those loops take a marked candidate. At "
        f"most {ctx.code.loops} marked candidate(s) sit in the buffer at "
        f"once, and {marked} do now; a marked candidate beyond that is "
        f"discarded. Leave the field out otherwise.\n"
        # This round restocks the loops without code (ctx is the worker
        # running it): a marked lemma is no use to them.
        + (f"This round restocks the loops that cannot run code, and the "
           f"coding loops already have {marked} marked lemma(s) waiting: "
           f"prefer lemmas any loop can prove.\n"
           if not ctx.loop.code_loop and marked else "")
    )


def _auto_claim(ctx, state: ParallelState, loop_id: str, verbose: bool) -> Optional[Dict[str, Any]]:
    """The unattended claim: the oldest unclaimed lemma in the buffer,
    first in, first out. None when the buffer holds nothing the loops have
    not already taken."""
    claimed = state.buffer.claim_oldest(loop_id, code=ctx.loop.code_loop)
    if claimed is not None:
        _log_claim(ctx, loop_id, claimed, verbose)
    return claimed


def _log_claim(ctx, loop_id: str, claimed: Dict[str, Any], verbose: bool) -> None:
    """An unattended claim, as the log tells it."""
    ctx.ui.log(f"[{loop_id}] Took {claimed['id']}, "
        + ("marked for computation." if claimed.get("computation")
           else "the oldest in the buffer."), verbose)


def _parallel_human_step(
    ctx,
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
    lemma, hands the loop to automation, asks for a re-plan, or quits — the
    action is returned with the lemma, if any — and a quit stops the whole
    run, not just this loop, and says so. A re-plan is recorded and handed
    back to the scheduler, which writes the new plan (this worker, when no
    other is free) before the menu comes back; the operator's own proof is
    committed when the verifiers accept it and not otherwise, and the menu
    comes right back."""
    while True:
        dag = ctx.dag
        susp = dag.suspension()
        snapshot = state.buffer.snapshot()
        shortlist = [
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
        choice = choose(
            ctx,
            shortlist, lambda: new_lemma_id(ctx), ctx.run.hotkey, summary,
            check_citations=_operator_citation_check(
                dag, ref_ids, susp.suspended
            ),
        )
        if choice.action == "quit":
            ctx.ui.log(
                "\nStopped by the operator; the DAG and the checkpoint "
                "are kept — re-run the same command to resume.",
                verbose,
            )
            state.request_stop()
            return "quit", None
        if choice.action == "replan":
            # For the next plan only: the buffer was turned down, not
            # tried, so none of it goes on the reject list. A plan being
            # written right now is superseded by this one.
            superseded = state.request_replan(
                replan_note([cand for cand, _ in shortlist], choice.notes)
            )
            ctx.ui.log(
                "Buffer rejected; the planner is asked again"
                + (" (the plan being written is dropped for it)."
                   if superseded else "."),
                verbose,
            )
            return "replan", None
        if choice.action == "auto":
            ctx.run.mode = "auto"
            ctx.ui.log(f"{loop_id} handed to automation. {ctx.run.hotkey.hint()}", verbose)
            return "auto", None
        if choice.action == "own_proof":
            own = resolution.pin(choice.lemma or {}, conjecture)
            own_id = str(own.get("id") or "").strip()
            # A buffered id is claimed before the operator's proof is
            # verified, the way a 'prove' pick is: a lemma another loop is
            # proving is not the operator's to prove over (its slot would be
            # freed under that loop, and the two would commit one id), and
            # one still unclaimed must not be picked up by a loop while the
            # verifiers run.
            buffered = own_id in {e["lemma"]["id"] for e in state.buffer.snapshot()}
            if buffered and state.buffer.claim_id(loop_id, own_id) is None:
                ctx.ui.log(
                    f"  {own_id} is claimed by another loop, which is "
                    f"proving it; pick another, or let that loop finish.",
                    verbose,
                )
                continue
            verified = _verify_operator_proof(
                ctx,
                choice, dag, references, ref_ids, susp.suspended,
                conjecture, verifier_sys, verbose,
            )
            if verified is None:
                if buffered:
                    # Back in the buffer for any loop, this one included.
                    state.buffer.release_claims(loop_id)
                continue
            lid, node, vh = verified
            with _commit_section(ctx):
                state.add_lemma(lid, node, vh, verbose)
                state.buffer.consume(lid)
            continue
        # 'prove': a buffer lemma, claimed atomically, or a written one.
        lemma = choice.lemma or {}
        lemma_id = str(lemma.get("id") or "").strip()
        if lemma_id in {e["lemma"]["id"] for e in snapshot}:
            claimed = state.buffer.claim_id(loop_id, lemma_id)
            if claimed is None:
                ctx.ui.log(
                    f"  {lemma_id} was claimed in the meantime; pick another.",
                    verbose,
                )
                continue
            return "prove", claimed
        # A written lemma: its id is drawn already (the menu's new_id).
        return "prove", resolution.pin(lemma, conjecture)


def _run_plan(ctx, state: ParallelState, verbose: bool, steering: Dict[str, str],
              conjecture: str, epoch: int, cancel: threading.Event) -> None:
    """A plan task, as the worker runs it: under the task's cancel event,
    which the operator's replan sets (its next model request then raises
    Cancelled), with the scheduler's planning flag cleared however the
    round ends. The operator's note goes back for the next plan however
    this one ends without a plan installed, a crash included."""
    ctx.loop.cancel = cancel
    operator_note = state.take_operator_note(epoch)
    installed = False
    try:
        installed = _plan_round(
            ctx, state, verbose, steering["planner_sys"], conjecture,
            steering["comments_block"], steering["ref_block"], epoch,
            operator_note)
    except Cancelled:
        ctx.ui.log("The operator asked for a new plan; the one being written "
                   "is dropped.", verbose)
    finally:
        if not installed:
            state.return_operator_note(operator_note, epoch)
        ctx.loop.cancel = None
        state.end_planning(epoch)


def _run_generate(ctx, state: ParallelState, verbose: bool, steering: Dict[str, str],
                  conjecture: str, needed: int, focus: Optional[int]) -> None:
    """A generate task, as the worker runs it: its reservation goes back
    however the round ends, if the round did not fill it."""
    try:
        _generate_round(ctx, state, verbose, steering["generator_sys"],
                        conjecture, needed, focus)
    finally:
        state.end_generating(ctx.loop.agent)


def _parallel_worker(
    ctx,
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
    steering: Dict[str, str],
) -> None:
    """One of the N workers: finish the lemma it holds in flight, if any,
    then take task after task from the scheduler (ParallelState.next_task)
    — write a plan, write a batch of candidates, or claim a lemma and run
    the shared proof engine on it — until the worker spends its iteration
    budget, the run resolves, or it is stopped. steering holds the plan and
    generate tasks' prompts: planner_sys, generator_sys, comments_block and
    ref_block.

    Loop 0 is the operator's while --mode human: each lemma it proves is
    picked at the menu (the buffer is the shortlist), and the hotkey still
    toggles it between the menu and automation. The engine's callbacks are
    the loop's share of the shared state: on_round publishes the resumable
    position, on_proof commits a lemma to the DAG, and a lemma that spends
    every prover round joins the shared reject list under its lock."""
    is_human = lambda: loop_index == 0 and ctx.run.mode == "human"
    ctx.ui.log(
        f"{loop_id} starts"
        + (" (the human loop — it asks before each lemma)."
           if is_human() else " (automatic)."),
        verbose,
    )
    while state.running():
        dag_now = ctx.dag
        susp_now = dag_now.suspension()
        pending = state.take_in_flight(loop_id)
        if pending is not None:
            # A prover round a previous run cancelled mid-flight: re-run
            # exactly that round. It does not spend a new iteration — the
            # one it started under was already spent — so the resume is
            # checked before the budget: a loop that exhausted its budget
            # still finishes the lemma it died holding.
            uncertified = _committed_uncertified(ctx, dag_now, susp_now, pending)
            if uncertified is not None:
                state.set_in_flight(loop_id, None)
                state.buffer.consume(
                    pending.get("claimed_id") or pending["lemma_id"]
                )
                ctx.ui.log(_uncertified_message(ctx, uncertified))
                continue
            if _committed_in_run(dag_now, pending) is not None:
                # The stop landed after the lemma was proved: on_proof had
                # already committed it. The serial run's same case — nothing
                # to resume; drop the position and the buffer claim, and go
                # about the next lemma (the spent iteration is spent).
                state.set_in_flight(loop_id, None)
                state.buffer.consume(
                    pending.get("claimed_id") or pending["lemma_id"]
                )
                ctx.ui.log(
                    f"{loop_id}'s in-flight {pending['lemma_id']} is "
                    f"already in the DAG; resuming is moot.",
                    verbose,
                )
                continue
            initial = pending
            fresh_target = None
            ctx.ui.log(
                f"{loop_id} resumes {pending['lemma_id']} at prover round "
                f"{settings_mod.of_limit(pending['attempt'], ctx.settings.max_proof_attempts)}, "
                f"{pending['stage']} step.",
                verbose,
            )
        else:
            if (state.max_iterations >= 0
                    and state.iterations_used(loop_id) >= state.max_iterations):
                ctx.ui.log(
                    f"{loop_id} has spent its budget "
                    f"({state.max_iterations}/{state.max_iterations}).",
                    verbose,
                )
                # The run's stop is sent from the last worker's exit (the
                # thread wrapper in run_parallel), not here: this worker has
                # spent its budget, but a sibling may still be proving the
                # lemma it claimed on its last iteration, and a stop now
                # would kill that engine before its lemma was ever proved.
                # The siblings take the plan and generate tasks from here.
                break
            task = state.next_task(
                loop_id, ctx.loop.code_loop, is_human(),
                on_wait=lambda: ctx.status(role="waiting for work", lemma="",
                                           attempt=""),
            )
            if task is None:
                break
            kind, payload = task
            if kind == "plan":
                epoch, cancel = payload
                _run_plan(ctx, state, verbose, steering, conjecture, epoch,
                          cancel)
                continue
            if kind == "generate":
                needed, focus = payload
                _run_generate(ctx, state, verbose, steering, conjecture,
                              needed, focus)
                continue
            if kind == "menu":
                action, lemma = _parallel_human_step(
                    ctx,
                    state, loop_id, verbose, conjecture,
                    references, ref_ids, verifier_sys,
                )
                if action == "quit":
                    # The operator ended the run, the serial way: stop every
                    # loop (their in-flight lemmas stay in the checkpoint),
                    # not just this one.
                    state.request_stop()
                    break
                if action == "replan":
                    # Back to the scheduler, which has a plan due.
                    continue
                if action == "prove":
                    fresh_target = lemma
                else:  # 'auto': the operator delegated the pick.
                    fresh_target = _auto_claim(ctx, state, loop_id, verbose)
                if fresh_target is None:
                    # Nothing to claim just now: the scheduler waits for it.
                    continue
            else:
                fresh_target = payload
                _log_claim(ctx, loop_id, fresh_target, verbose)
            initial = None
            # The iteration is spent the moment the lemma is claimed, the
            # serial way: a run cancelled mid-lemma resumes it without
            # charging the budget again.
            state.begin_lemma(loop_id)
        def on_proof(
            lid: str, node: Dict[str, Any], verified_hash: Optional[str],
        ) -> str:
            # All three verifier checks accepted this proof: add_lemma
            # certifies it under the new id it is committed under, then
            # writes it.
            with _commit_section(ctx):
                return state.add_lemma(lid, node, verified_hash, verbose)
        result = run_proof_loop(
            ctx,
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
            get_dag=lambda: ctx.dag,
            toggle_hotkey=lambda: check_mode_toggle(ctx, verbose),
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
        # claimed id is free. The rejection, by contrast, records the final
        # target — that is the statement that was actually tried.
        state.set_in_flight(loop_id, None)
        state.buffer.consume(result["claimed_id"])
        state.note_progress(loop_id)
        if result["exhausted"]:
            state.record_rejection(
                result["lemma_id"], result["target"]["statement"]
            )
            ctx.status(event=f"gave up {result['lemma_id']}")
        # A proved lemma is already in the DAG (on_proof committed it, which
        # bumped the version the next re-plan waits on): the worker simply
        # goes for its next task.
        if not state.running():
            break


    # The worker's exit is announced by its thread wrapper (_worker_thread
    # in run_parallel), in a finally, so a crash announces it too.


def run_parallel(ctx, parallel_loops: int, verbose: bool = True) -> Dict[str, Any]:
    """The parallel entry point: N workers around one shared DAG.

    The workers share run_proof_loop() with the serial run; what is new
    here is the shared state (ParallelState), the buffer that feeds the
    loops, and the scheduler that hands each worker its next task — a
    plan, a batch of candidates, or a lemma to prove — so that the run
    never has more than N model calls in flight. Loop 0 is the operator's
    while --mode human. Returns the DAG, as run_loop does."""

    conjecture = load_file(ctx.paths.conjecture_file)
    if not conjecture:
        ctx.ui.log(
            f"{ctx.paths.conjecture_file} is empty — write your conjecture there "
            f"first.",
            verbose,
        )
        return ctx.dag

    planner_sys = load_file(ctx.paths.prompt_paths["parallel_planner.md"])
    generator_sys = load_file(ctx.paths.prompt_paths["parallel_lemma_generator.md"])
    prover_sys = load_file(ctx.paths.prompt_paths["prover.md"])
    # One prompt per verification step; VERIFIER_AGENTS names the three.
    verifier_sys = {name: load_file(ctx.paths.prompt_paths[name]) for name in VERIFIER_AGENTS}
    reviser_sys = load_file(ctx.paths.prompt_paths["reviser.md"])
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
        ctx.ui.log(f"Missing or empty prompt files: {', '.join(empty)}. Aborting.", verbose)
        return ctx.dag

    references, reference_block, planner_ref_block, reviser_ref_block = (
        _build_reference_blocks(ctx, verbose)
    )
    # The planner and the generator steer by the short view (the same one
    # the serial planner gets — empty at or above the limit, by the same
    # rule); the prover in each loop gets the full collection through
    # reference_block.
    ref_block = planner_ref_block
    ref_ids = {str(r["id"]) for r in references}
    comments_block = _build_comments_block(ctx)
    planner_sys = _planner_system(planner_sys, ref_block, comments_block)

    loop_ids = [f"loop-{i}" for i in range(parallel_loops)]
    state = ParallelState(ctx, loop_ids, ctx.settings.max_iterations,
                          conjecture, verbose)
    every = ctx.settings.replan_every
    ctx.ui.log(
        f"\nParallel mode: {parallel_loops} worker(s), each planning, "
        f"generating or proving as the run needs; up to {state.batch} "
        f"candidate(s) per generator call, {len(state.buffer)} buffer slots, "
        + ("re-planning only when asked." if every < 0 else
           f"re-planning every {every} commit(s)."),
        verbose,
    )
    cp = load_parallel_checkpoint(ctx, loop_ids, verbose)
    if cp is not None:
        dropped = state.apply_checkpoint(cp)
        pending = sum(
            1
            for lid in loop_ids
            if (cp["loops"].get(lid) or {}).get("in_flight") is not None
        )
        used = sum(
            (cp["loops"].get(lid) or {}).get("iterations_used", 0)
            for lid in loop_ids
        )
        ctx.ui.log(
            f"Resumed the parallel checkpoint: {used} lemma(s) spent"
            + ("" if ctx.settings.unlimited_iterations()
               else f" of {ctx.settings.max_iterations} each")
            + f", {pending} lemma(s) in flight, {len(cp.get('buffer') or [])} "
            f"in the buffer"
            + (", and the saved plan." if cp.get("plan") else ".")
            + (f" {dropped} buffered candidate(s) did not fit this run's "
               f"{len(state.buffer)} slots (a smaller window means smaller "
               f"batches) and were dropped." if dropped else ""),
            verbose,
        )
    state.write_checkpoint()
    steering = {"planner_sys": planner_sys, "generator_sys": generator_sys,
                "comments_block": comments_block, "ref_block": ref_block}

    # A worker is restarted after a crash. A server context 400 (a
    # mis-sized --num-ctx) is not a crash — every restart would 400 the
    # same way — and turns into the one loud message plus a stop of the
    # shared state, so the run ends cleanly with its checkpoint intact. A
    # worker that crashes run.max_loop_restarts times in a row without
    # completing a task is given up for the run.
    def _worker_thread(i: int, lid: str) -> None:
        # Each thread works on its own copy of the context.
        lctx = ctx.for_loop(
            agent=lid,
            code_loop=_may_run_code(ctx, i),
        )
        restarts = 0
        progress_at_start = state.progress_of(lid)
        try:
            while True:
                try:
                    _parallel_worker(
                        lctx,
                        state, lid, i, verbose, conjecture, prover_sys,
                        reviser_sys, reference_block, reviser_ref_block,
                        references, verifier_sys, ref_ids, steering,
                    )
                    return
                except llm.ContextLengthError as e:
                    # A mis-sized window, not a crash: every restart would
                    # 400 the same way.
                    _fail_context_length(lctx, e, stop=state.request_stop)
                    return
                except Exception as e:
                    # Logged unconditionally: a crash the operator must see
                    # is not verbose noise.
                    lctx.ui.log(
                        f"{lid} crashed: {type(e).__name__}: {e}\n"
                        + traceback.format_exc().rstrip()
                    )
                    # The bound is on crashes in a row: a worker that
                    # finished a task since its last (re)start was making
                    # progress, so this crash starts the count again.
                    if state.progress_of(lid) > progress_at_start:
                        restarts = 0
                    if not settings_mod.below(restarts, ctx.settings.max_loop_restarts):
                        lctx.ui.log(
                            f"{lid} crashed {restarts + 1} times in a row "
                            f"without finishing a task; giving it "
                            f"up for this run. Its in-flight round stays in "
                            f"the checkpoint for the next run."
                        )
                        return
                    restarts += 1
                    progress_at_start = state.progress_of(lid)
                    # The restart resumes the round the worker last
                    # published (take_in_flight), and its claim with it; any
                    # other claim the crash left behind (claimed, but no
                    # round published yet) goes back to the buffer, or its
                    # slot would stay claimed for the rest of the run.
                    pending = state.take_in_flight(lid)
                    keep = (
                        (pending.get("claimed_id") or pending["lemma_id"])
                        if pending is not None else None
                    )
                    released = state.buffer.release_claims(lid, keep=keep)
                    lctx.ui.log(
                        f"{lid}: restart "
                        f"{settings_mod.of_limit(restarts, ctx.settings.max_loop_restarts)} in "
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
            # Every exit path (budget spent, human quit, a stop, an
            # unexpected exception) funnels here: announce this worker is
            # out, and if it was the last, stop the run — nothing more will
            # ever be claimed or added. Without the finally, a worker that
            # crashed would never be counted out and the run would hang at
            # join().
            lctx.ui.done(lid)
            if state.note_loop_exit(lid):
                state.request_stop()

    threads = [
        threading.Thread(
            target=_worker_thread,
            args=(i, lid),
            name=f"parallel-{lid}",
            daemon=True,
        )
        for i, lid in enumerate(loop_ids)
    ]
    ctx.run.parallel = state
    try:
        # The DAG may settle the conjecture already (another user's pulled
        # file); after this, every commit is checked as it lands.
        state.check_dag()
        if state.running():
            for t in threads:
                t.start()
            for t in threads:
                t.join()
    finally:
        ctx.run.parallel = None

    if state.resolution is None:
        # A settling lemma is caught as it is committed (check_dag); this
        # is the last word for one whose check did not run to the end.
        dag = ctx.dag
        susp = dag.suspension()
        kind = _settled(ctx, dag, susp.suspended, conjecture, verbose)
        if kind is not None:
            state.set_resolution(kind)
    if state.resolution is not None:
        # Nothing is written to the DAG: the settling lemma is the record,
        # and the resolution is recomputed from it (resolution.py).
        discard_checkpoint(ctx)
        ctx.ui.log(
            f"\nConjecture {state.resolution}; the checkpoint is cleared.",
            verbose,
        )
        return ctx.dag
    state.write_checkpoint()
    buffer_snapshot = state.buffer.snapshot()
    ctx.ui.log(
        f"\nParallel run finished without settling the conjecture: "
        f"the buffer still holds {len(buffer_snapshot)} candidate(s). "
        f"The checkpoint keeps every loop's budget position and in-flight "
        f"round: re-run the same command to continue, or --fresh to "
        f"restart. The DAG is kept either way.",
        verbose,
    )
    return ctx.dag
