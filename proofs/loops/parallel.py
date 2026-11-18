"""Parallel mode

N proof loops share one DAG, one version-2 checkpoint, and one
possible-lemma buffer, steered by two extra threads:

  planner    re-plans the shared strategy whenever the DAG changes
  generator  keeps the buffer stocked with fresh candidate lemmas
  loop-N     pulls a lemma from the buffer, runs the shared proof engine
             (run_proof_loop) on it, records the outcome, repeats

When --mode human, loop 0 is the operator's loop: each fresh start goes
to the menu with the buffer as its candidate list — the parallel version
of the serial shortlist menu — and the hotkey toggles it between the menu
and automation exactly as in the serial run. Every write to the shared
state goes through ParallelState, and every write to the DAG goes through
commit_lemma_to_dag under the DAG lock, each lemma under the id its
candidate was given, so a reader can never see a half-state.
"""

from __future__ import annotations

import json
import threading
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
    _warn_suspended, commit_lemma_to_dag, load_file, new_lemma_id,
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
from .reasoning import _fail_context_length, extract, reason
from .schemas import PARALLEL_LEMMA_GENERATOR_SCHEMA, PARALLEL_PLANNER_SCHEMA

LOOP_RESTART_DELAY = 10


_QUIT = object()  # sentinel: the human loop asked the run to stop


class LemmaBuffer:
    """The shared possible-lemma list: exactly N+5 fixed slots.

    The generator fills the empty slots; the proof loops claim them. The
    slots are the flow control — a full buffer means the loops are behind,
    and the generator waits for a slot instead of piling up work. The
    unattended claim treats the buffer as a queue: the oldest unclaimed
    lemma goes first, by arrival order rather than slot position (a freed
    slot is refilled by whatever arrives next). Claiming is atomic, so two
    loops never work on one id.
    """

    def __init__(self, size: int, marked_cap: int = 0) -> None:
        # Each slot is {"lemma": candidate, "claimed_by": loop id or None,
        # "seq": arrival number}, or None while the slot is empty. seq is
        # what makes the buffer a queue; a released claim keeps its place.
        self.slots: List[Optional[Dict[str, Any]]] = [None] * size
        self._next_seq = 0
        self.cond = threading.Condition()
        # A lemma marked "computation" (a parallel run with --tools) is
        # claimed only by a loop that may run code, and at most marked_cap
        # of them — the coding loops — sit in the slots at once. One beyond
        # that is held here, out of the slots, until a marked slot frees:
        # held, not dropped, and not crowding out the loops without code.
        self.marked_cap = marked_cap
        self.held: List[Dict[str, Any]] = []
        # Called after every change to the slots or the held list, with
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

    def has_room(self) -> bool:
        with self.cond:
            return any(e is None for e in self.slots)

    def held_ids(self) -> Set[str]:
        with self.cond:
            return {c["id"] for c in self.held}

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

    def _promote_held_locked(self) -> int:
        """Move held marked lemmas into the slots while the cap and the
        free slots allow. Call with self.cond held."""
        moved = 0
        while self.held and self._marked_locked() < self.marked_cap \
                and self._place_locked(self.held[0]):
            self.held.pop(0)
            moved += 1
        return moved

    def _statements_locked(self) -> Set[str]:
        """The normalized statements of every lemma in the slots or held.
        Call with self.cond held."""
        return {
            merkle.normalize(c["statement"])
            for c in [*(e["lemma"] for e in self.slots if e is not None),
                      *self.held]
        }

    def fill(
        self, candidates: List[Dict[str, Any]],
        taken: Collection[str] = (),
    ) -> Tuple[int, int]:
        """Put fresh candidates into the empty slots, first come first
        served. A candidate whose statement (up to merkle.normalize's
        whitespace) is already in the buffer — in a slot or held — or in
        taken, the normalized statements the loops hold in flight, is
        dropped: every candidate has a new id, so only the statement can
        tell that two loops would be proving the same thing. A marked
        candidate past the cap is held (see __init__). Returns (how many
        landed in the slots, how many were dropped as duplicates)."""
        added = duplicates = held = 0
        with self.cond:
            added += self._promote_held_locked()
            for cand in candidates:
                statement = merkle.normalize(cand["statement"])
                if statement in taken or statement in self._statements_locked():
                    duplicates += 1
                    continue
                if cand.get("computation") and \
                        self._marked_locked() >= self.marked_cap:
                    self.held.append(cand)
                    held += 1
                    continue
                if self._place_locked(cand):
                    added += 1
            if added:
                self.cond.notify_all()
        if added or held:
            self.on_change()
        return added, duplicates

    def _claimable_locked(self, code: bool) -> List[Dict[str, Any]]:
        return [
            e for e in self.slots
            if e is not None and e["claimed_by"] is None
            and (code or not e["lemma"].get("computation"))
        ]

    def claimable(self, code: bool) -> bool:
        """Whether a loop that may (code) or may not run code has anything
        to claim."""
        with self.cond:
            return bool(self._claimable_locked(code))

    def claim_oldest(self, loop_id: str, code: bool = True) -> Optional[Dict[str, Any]]:
        """The unattended pick: claim the unclaimed lemma that entered the
        buffer first. A loop that may run code takes a lemma marked
        "computation" first, the oldest of them — no other loop can; a loop
        that may not never takes one. None when nothing is claimable."""
        with self.cond:
            waiting = self._claimable_locked(code)
            if not waiting:
                return None
            marked = [e for e in waiting if e["lemma"].get("computation")]
            oldest = min(marked or waiting, key=lambda e: e["seq"])
            oldest["claimed_by"] = loop_id
            self.cond.notify_all()
        self.on_change()
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
                    self._promote_held_locked()
                    self.cond.notify_all()
                    break
            else:
                return
        self.on_change()

    def restore(self, entries: List[Dict[str, Any]]) -> None:
        """Put a checkpoint's buffer back (core/checkpoint.py): each
        {"lemma", "claimed_by"} into a slot in its saved order, a marked
        lemma past the cap held, the way fill() would."""
        with self.cond:
            for e in entries:
                cand = e["lemma"]
                if cand.get("computation") and \
                        self._marked_locked() >= self.marked_cap:
                    self.held.append(cand)
                    continue
                if self._place_locked(cand):
                    for slot in self.slots:
                        if slot is not None and slot["lemma"] is cand:
                            slot["claimed_by"] = e.get("claimed_by")
            self.cond.notify_all()

    def checkpoint_entries(self) -> List[Dict[str, Any]]:
        """The buffer as the checkpoint keeps it: the slots in arrival
        order, then the held lemmas, unclaimed."""
        with self.cond:
            slots = sorted((e for e in self.slots if e is not None),
                           key=lambda e: e["seq"])
            return (
                [{"lemma": e["lemma"], "claimed_by": e["claimed_by"]} for e in slots]
                + [{"lemma": c, "claimed_by": None} for c in self.held]
            )

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
      rejected_lock the shared reject list
      run.dag_lock  the commits to the run's streams (Context.run)
      run.checkpoint_lock  the checkpoint's snapshot-then-replace
    The checkpoint is a whole-file rewrite taken under the checkpoint lock —
    the loop slots under state_lock, the reject list under rejected_lock,
    then one atomic replace over a per-writer temp file — so the file is
    always a consistent, current state even though many threads poke at
    it.
    """

    def __init__(self, ctx, loop_ids: List[str], max_iterations: int) -> None:
        self.ctx = ctx
        self.buffer = LemmaBuffer(len(loop_ids) + 5, marked_cap=self.ctx.code.loops)
        self.buffer.on_change = self.write_checkpoint
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
        self.rejected_lemmas: List[Dict[str, str]] = []
        self.rejected_lock = threading.Lock()
        # The operator's last 'replan' reason, for the next plan only
        # (under state_lock; not checkpointed).
        self.operator_note = ""
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
        self.write_checkpoint()
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
        """The lemmas the loops are holding in flight right now: whether
        any loop still has work, for the steering threads' too-large
        stop."""
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
        DAG lock, so the loops commit one at a time). The commit certifies
        the lemma before it writes it, so the planner this wakes never finds
        it uncertified. Then bump the version the planner watches, take the
        lemma off the reject list, mark the plan stale, and checkpoint.
        Returns the id the lemma was committed under."""
        committed = commit_lemma_to_dag(self.ctx, lemma_id, node, verified_hash, verbose)
        with self.state_lock:
            self.dag_version += 1
        with self.rejected_lock:
            drop_rejection(self.rejected_lemmas, node["statement"])
        self.mark_plan_stale()
        self.write_checkpoint()
        return committed

    def dag_version_snapshot(self) -> int:
        with self.state_lock:
            return self.dag_version

    # -- the shared reject list -----------------------------------------------

    def record_rejection(self, lemma_id: str, statement: str) -> None:
        with self.rejected_lock:
            note_rejection(self.ctx, self.rejected_lemmas, lemma_id, statement)
        self.write_checkpoint()

    def rejected_snapshot(self) -> List[Dict[str, str]]:
        with self.rejected_lock:
            return [dict(e) for e in self.rejected_lemmas]

    def set_operator_note(self, note: str) -> None:
        with self.state_lock:
            self.operator_note = note
        self.mark_plan_stale()

    def take_operator_note(self) -> str:
        with self.state_lock:
            note, self.operator_note = self.operator_note, ""
            return note

    # -- checkpoint -------------------------------------------------------------

    def apply_checkpoint(self, cp: Dict[str, Any]) -> None:
        with self.state_lock:
            for lid, slot in self.loops.items():
                entry = cp["loops"].get(lid) or {}
                slot["in_flight"] = entry.get("in_flight")
                slot["iterations_used"] = int(entry.get("iterations_used") or 0)
        with self.rejected_lock:
            self.rejected_lemmas = clean_rejected(self.ctx, cp.get("rejected_lemmas"))
        # The saved plan steers the run from the start; the planner waits
        # for the DAG to move before it plans again.
        if cp.get("plan") is not None:
            with self.state_lock:
                self.plan = cp["plan"]
                self.plan_stale = False
        # A claim survives only where its loop holds the lemma in flight;
        # any other claim (taken, but no round published) is released.
        held = {
            lid: (slot["in_flight"] or {}).get("claimed_id")
            for lid, slot in self.loops.items()
        }
        self.buffer.restore([
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


def _clean_candidate(ctx, raw: Any, conjecture: str) -> Optional[Dict[str, Any]]:
    """A generated lemma entry, taken the way the planner's candidates
    are (prompts.candidate: a new id, the statement, pinned when it settles
    the conjecture), with its computation mark in a run with code loops.
    An entry without a statement is dropped, not repaired — the prompt
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


def _wait_for_steering_change(ctx, state: ParallelState) -> bool:
    """Wait for a steering prompt's inputs to change after one was too
    large to send (reason()'s "too_large"): the same inputs would build the
    same prompt. True once they have changed, False on a stop."""
    before = _steering_inputs(state)
    while not state.stop_event.wait(10):
        if _steering_inputs(state) != before:
            return True
    return False


def _parallel_planner(
    ctx,
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
    warned: Dict[str, Any] = {}
    while state.running():
        ctx.status(role="waiting for the DAG to change")
        if not state.wait_for_work():
            return
        if not state.running():
            return
        # The version is read before the DAG, never after: a commit landing
        # between the two would otherwise be counted in the version but
        # missing from the DAG this plan is written from, and install_plan
        # would then clear the stale flag mark_plan_stale had just set.
        dag_version_before = state.dag_version_snapshot()
        dag = ctx.dag
        # The suspension as this round stands to be planned (see
        # suspension.py): the planner's view omits the suspended lemmas, and
        # the warning is told here, where every DAG change is seen.
        susp = dag.suspension()
        _warn_suspended(ctx, susp, verbose, warned)
        # Settled? The planner wakes on every DAG change, so this is where
        # a lemma that settles the conjecture is noticed — a loop's proof,
        # the operator's, or (at start) another user's pulled
        # file. The planner's own opinion never settles it (resolution.py).
        kind = _settled(ctx, dag, susp.suspended, conjecture, verbose)
        if kind is not None:
            ctx.ui.log("The loops are being stopped.", verbose)
            state.set_resolution(kind)
            return
        # Each pass plans from scratch: the previous plan is not shown, so
        # the new one is written from the DAG alone.
        operator_note = state.take_operator_note()
        rejected = rejected_block(ctx, state.rejected_snapshot())
        # The window for this plan, and for the generator working from it
        # (strategies.py): chosen now, held until the next pass.
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
        if _status == "too_large":
            # The same prompt would not fit any better: wait for what it is
            # built from to change. With no plan yet and no loop in flight,
            # nothing can — the generator waits for a plan, the loops for
            # the generator — so the run stops instead of hanging.
            if state.plan_snapshot() is None and not state.in_flight_ids():
                ctx.ui.log(
                    "The parallel planner's prompt does not fit the context "
                    "window even with no lemmas shown in full (its largest "
                    "parts are logged above), and there is no plan to steer "
                    "by. Stopping the run: re-run with a larger window "
                    "(--num-ctx, and the server's own) or a smaller part.",
                    verbose,
                )
                state.request_stop()
                return
            ctx.ui.log(
                "The parallel planner's prompt does not fit the context "
                "window even with no lemmas shown in full; planning again "
                "once the DAG, the reject list or the buffer changes.",
                verbose,
            )
            if not _wait_for_steering_change(ctx, state):
                return
            continue
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
                "Parallel planner gave no usable plan; retrying in 10s.",
                verbose,
            )
            if state.stop_event.wait(10):
                return
            continue
        moved = state.install_plan(plan_res, dag_version_before)
        state.finish_steering_round("planner")
        ctx.ui.log(
            "Planner set the strategy"
            + (" — the DAG moved while it was writing, re-planning now."
               if moved else "."),
            verbose,
        )


def _parallel_generator(
    ctx,
    state: ParallelState,
    verbose: bool,
    generator_sys: str,
    conjecture: str,
    ref_block: str,
) -> None:
    """Keeps the possible-lemma list stocked.

    Each round it asks the generator for five candidate lemmas and fills
    the buffer's empty slots with the ones that are new: a statement
    already in the buffer or in flight on a loop is dropped
    (LemmaBuffer.fill), so no two loops prove the same statement at once.
    The DAG and the reject list are not checked here: the prompt shows
    them, and a statement proved again, or tried again, is the
    generator's call to make. The buffer's fixed N+5 slots are the flow
    control — a full buffer means the loops are behind, and the generator
    waits for a slot instead of piling up work."""
    while state.running():
        ctx.status(role="waiting for a plan")
        if not state.wait_for_plan():
            return
        if not state.running():
            return
        # Wait for a free slot first: the buffer is the queue and the
        # generator its only writer.
        ctx.status(role="waiting for a free slot")
        if not state.buffer.wait_for(state.buffer.has_room, state.stop_event):
            return
        plan = state.plan_snapshot() or {}
        dag = ctx.dag
        susp = dag.suspension()
        buffer_snapshot = state.buffer.snapshot()
        rejected = rejected_block(ctx, state.rejected_snapshot())
        strategies.for_generator(ctx, dag, susp.suspended,
                                 strategies.plan_query(conjecture, plan))

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
                + _computation_note(ctx, state),
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
            # As for the planner: wait for the prompt's inputs to change
            # rather than re-send it — and stop when nothing can change
            # them, with the buffer empty and no loop in flight.
            if not buffer_snapshot and not state.in_flight_ids():
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
            if not _wait_for_steering_change(ctx, state):
                return
            continue
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
        added, duplicates = state.buffer.fill(
            fresh, taken=state.in_flight_statements())
        state.finish_steering_round("generator")
        ctx.ui.log(
            f"Lemma generator offered {len(fresh)} new candidate(s); "
            + (f"{duplicates} restated a buffered or in-flight lemma and "
               f"were dropped; " if duplicates else "")
            + f"{added} entered the buffer "
            f"({len(state.buffer.snapshot())}/{len(state.buffer)} slots full).",
            verbose,
        )
        if added == 0:
            # Nothing landed (no offer, or the buffer filled under the feet):
            # back off so a stalled planner and a stalled generator cannot
            # spin together.
            if state.stop_event.wait(10):
                return


def _computation_note(ctx, state: ParallelState) -> str:
    """What the lemma generator is told about marking a lemma for
    computation: nothing in a run without code tools."""
    if ctx.code.loops == 0:
        return ""
    marked = sum(1 for e in state.buffer.snapshot() if e["lemma"].get("computation"))
    held = len(state.buffer.held_ids())
    return (
        f"\n{ctx.code.loops} of the proof loops can run code "
        f"({', '.join(ctx.code.runner.languages)}). Add \"computation\": true to "
        f"a candidate only when proving it needs a computer search or an "
        f"exact finite check; only those loops take a marked candidate. At "
        f"most {ctx.code.loops} marked candidate(s) wait in the buffer at once; "
        f"{marked} do now"
        + (f", and {held} more are held until one frees" if held else "")
        + ". Leave the field out otherwise.\n"
    )


def _auto_claim(ctx, state: ParallelState, loop_id: str, verbose: bool) -> Optional[Dict[str, Any]]:
    """The unattended claim: the oldest unclaimed lemma in the buffer,
    first in, first out. None when the buffer holds nothing the loops have
    not already taken."""
    claimed = state.buffer.claim_oldest(loop_id, code=ctx.loop.code_loop)
    if claimed is not None:
        ctx.ui.log(f"[{loop_id}] Took {claimed['id']}, "
            + ("marked for computation." if claimed.get("computation")
               else "the oldest in the buffer."), verbose)
    return claimed


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
    lemma, hands the loop to automation, or quits — a quit stops the whole
    run, not just this loop, and says so. Re-plans are recorded and the
    menu comes right back, the serial way; so does the operator's own
    proof, committed when the verifiers accept it and not otherwise."""
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
            # tried, so none of it goes on the reject list.
            state.set_operator_note(
                replan_note([cand for cand, _ in shortlist], choice.notes)
            )
            ctx.ui.log("Buffer rejected; the planner is asked again.", verbose)
            continue
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


def _parallel_proof_loop(
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
) -> None:
    """One of the N proof loops: pull a lemma from the buffer, run the
    shared proof engine on it, record the outcome, repeat — until the loop
    spends its iteration budget, the run resolves, or it is stopped.

    Loop 0 is the operator's while --mode human: each fresh start goes to
    the menu first (the buffer is the shortlist), and the hotkey still
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
                # The stop that frees the planner and generator is sent from
                # the loop exit below, not here: this loop has spent its
                # budget, but a sibling may still be proving the lemma it
                # claimed on its last iteration, and a stop now would kill
                # that engine before its lemma was ever proved.
                break
            if is_human():
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
                if action == "prove":
                    fresh_target = lemma
                else:  # 'auto': the operator delegated the pick.
                    fresh_target = _auto_claim(ctx, state, loop_id, verbose)
            else:
                fresh_target = _auto_claim(ctx, state, loop_id, verbose)
            if fresh_target is None:
                # Nothing unclaimed in the buffer: the generator is behind.
                # Wait for it to fill a slot (or the run to stop).
                code_loop = ctx.loop.code_loop
                ctx.status(role="waiting for a lemma", lemma="", attempt="")
                if not state.buffer.wait_for(
                    lambda: state.buffer.claimable(code_loop), state.stop_event
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
        state.finish_lemma(loop_id)
        if result["exhausted"]:
            state.record_rejection(
                result["lemma_id"], result["target"]["statement"]
            )
            ctx.status(event=f"gave up {result['lemma_id']}")
        # A proved lemma is already in the DAG (on_proof committed it, which
        # bumped the version and marked the plan stale): the loop simply
        # goes for its next lemma.
        if not state.running():
            break


    # The loop's exit is announced by its thread wrapper (_proof_thread in
    # run_parallel), in a finally, so a crash announces it too.


def run_parallel(ctx, parallel_loops: int, verbose: bool = True) -> Dict[str, Any]:
    """The parallel entry point: N proof loops around one shared DAG.

    The loops share run_proof_loop() with the serial run; what is new here
    is the shared state (ParallelState), the buffer that feeds the loops,
    and the two threads that steer them — the planner, which re-plans
    whenever the DAG changes, and the generator, which keeps the
    possible-lemma list stocked. Loop 0 is the operator's while --mode
    human. Returns the DAG, as run_loop does."""

    ctx.ui.log(
        f"\nParallel mode: {parallel_loops} proof loop(s), one planner, "
        f"one lemma generator.",
        verbose,
    )

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

    loop_ids = [f"loop-{i}" for i in range(parallel_loops)]
    state = ParallelState(ctx, loop_ids, ctx.settings.max_iterations)
    cp = load_parallel_checkpoint(ctx, loop_ids, verbose)
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
        ctx.ui.log(
            f"Resumed the parallel checkpoint: {used} lemma(s) spent"
            + ("" if ctx.settings.unlimited_iterations()
               else f" of {ctx.settings.max_iterations} each")
            + f", {pending} lemma(s) in flight, {len(cp.get('buffer') or [])} "
            f"in the buffer"
            + (", and the saved plan." if cp.get("plan") else "."),
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
    # run.max_loop_restarts times in a row without completing a round: the
    # run cannot go on without it, and a stop beats a hang.
    def _steering_thread(name: str, run) -> None:
        try:
            _steer(name, run)
        finally:
            ctx.ui.done(name)

    def _steer(name: str, run) -> None:
        restarts = 0
        rounds_at_start = state.steering_rounds_done(name)
        while True:
            try:
                run()
                return
            except llm.ContextLengthError as e:
                _fail_context_length(ctx, e, stop=state.request_stop)
                return
            except Exception as e:
                # Logged unconditionally: a crash the operator must see is
                # not verbose noise.
                ctx.ui.log(
                    f"The parallel {name} crashed: {type(e).__name__}: "
                    f"{e}\n" + traceback.format_exc().rstrip()
                )
                if state.steering_rounds_done(name) > rounds_at_start:
                    restarts = 0
                if not settings_mod.below(restarts, ctx.settings.max_loop_restarts):
                    ctx.ui.log(
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
                ctx.ui.log(
                    f"The parallel {name}: restart "
                    f"{settings_mod.of_limit(restarts, ctx.settings.max_loop_restarts)} "
                    f"in {LOOP_RESTART_DELAY}s."
                )
                if state.stop_event.wait(LOOP_RESTART_DELAY):
                    return

    def _planner_thread() -> None:
        _steering_thread("planner", lambda: _parallel_planner(
            ctx.for_loop(agent="planner"),
            state, verbose, planner_sys, conjecture, comments_block,
            ref_block,
        ))

    def _generator_thread() -> None:
        _steering_thread("generator", lambda: _parallel_generator(
            ctx.for_loop(agent="generator"),
            state, verbose, generator_sys, conjecture, ref_block,
        ))

    def _proof_thread(i: int, lid: str) -> None:
        # The first code.loops loops may run code (see Code.loops); each
        # thread works on its own copy of the context.
        lctx = ctx.for_loop(
            agent=lid,
            code_loop=ctx.code.runner is None or i < ctx.code.loops,
        )
        restarts = 0
        finished_at_start = state.lemmas_finished(lid)
        try:
            while True:
                try:
                    _parallel_proof_loop(
                        lctx,
                        state, lid, i, verbose, conjecture, prover_sys,
                        reviser_sys, reference_block, reviser_ref_block,
                        references, verifier_sys, ref_ids,
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
                    # The bound is on crashes in a row: a loop that finished
                    # a lemma since its last (re)start was making progress,
                    # so this crash starts the count again.
                    if state.lemmas_finished(lid) > finished_at_start:
                        restarts = 0
                    if not settings_mod.below(restarts, ctx.settings.max_loop_restarts):
                        lctx.ui.log(
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
            # Every exit path (budget spent, human quit, a stop, the buffer
            # waiting giving up, an unexpected exception) funnels here:
            # announce this loop is out, and if it was the last, send the
            # stop that frees the planner and generator — nothing more will
            # ever be claimed or added, so there is no point left in their
            # waiting. Without the finally, a loop that crashed would never
            # be counted out and the run would hang at join().
            lctx.ui.done(lid)
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
    ctx.run.parallel = state
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        ctx.run.parallel = None

    if state.resolution is None:
        # The last loop out may have committed the settling lemma after
        # the planner stopped watching.
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
