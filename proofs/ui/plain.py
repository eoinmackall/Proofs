"""Line-by-line output: what the run prints when no live renderer is
installed (not on a terminal, or --no-verbose)."""

from __future__ import annotations

import contextlib
import sys
from typing import Iterator

from .status import Board


class PlainUI:
    """Every message the run writes goes through log(), so --no-verbose
    silences everything that passes verbose. The status board is kept but
    not drawn: in plain output the log lines are the record."""

    def __init__(self) -> None:
        self.board = Board()

    def log(self, message: str, verbose: bool = True, err: bool = False) -> None:
        """One message: to stdout, or to stderr with err (warnings and the
        interrupt notices). verbose False drops it."""
        if verbose:
            print(message, file=sys.stderr if err else sys.stdout)

    def header(self, **fields: object) -> None:
        """The run-wide line a live renderer draws above the board
        (conjecture, model, run, iteration); not drawn here."""

    def start(self) -> None:
        """Begin drawing (a live renderer); nothing to do here."""

    def close(self) -> None:
        """Stop drawing, restoring the terminal; nothing to do here."""

    def status(self, agent: str, **fields: str) -> None:
        """Update agent's row on the board (ui.status)."""
        self.board.update(agent, **fields)

    def done(self, agent: str) -> None:
        """agent has stopped: its row goes."""
        self.board.remove(agent)

    @contextlib.contextmanager
    def paused(self) -> Iterator[None]:
        """Around anything that reads the terminal (the human menu, a
        pasted proof, a y/N): a live renderer stops drawing for it."""
        yield
