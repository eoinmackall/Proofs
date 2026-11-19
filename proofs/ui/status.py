"""The status board: what each agent is doing.

One row per agent — the serial run's loop, or each worker of a parallel
run (loop-0, loop-1, ...) — updated by the loops as they move
(Context.status). A row holds:

    role      what the agent is doing now: "planner", "prover",
              "verifier 2/3", "reviser", "waiting for work", ...
    lemma     the lemma it is on: the candidate's id until it is committed
    attempt   the prover round, "k/N"
    event     the last thing that happened to it: "verifier 2 rejected",
              "committed lemma_a", ...
    since     when the current role started (time.monotonic())

The board only records; a renderer (ui.live) draws it, and the plain UI
does not.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from typing import Dict, List, Tuple

FIELDS = ("role", "lemma", "attempt", "event")


@dataclass(frozen=True)
class Row:
    role: str = ""
    lemma: str = ""
    attempt: str = ""
    event: str = ""
    since: float = 0.0


class Board:
    def __init__(self) -> None:
        self._rows: Dict[str, Row] = {}
        self._lock = threading.Lock()

    def update(self, agent: str, **fields: str) -> None:
        unknown = set(fields) - set(FIELDS)
        if unknown:
            raise TypeError(f"unknown status field(s): {', '.join(sorted(unknown))}")
        with self._lock:
            row = self._rows.get(agent) or Row(since=time.monotonic())
            if "role" in fields and fields["role"] != row.role:
                row = replace(row, since=time.monotonic())
            self._rows[agent] = replace(row, **fields)

    def remove(self, agent: str) -> None:
        with self._lock:
            self._rows.pop(agent, None)

    def snapshot(self) -> List[Tuple[str, Row]]:
        """The rows in the order the agents first reported."""
        with self._lock:
            return list(self._rows.items())
