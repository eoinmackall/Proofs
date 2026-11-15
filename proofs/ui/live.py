"""The live terminal block: the status board drawn under the scrolling log.

While a run is live, the bottom of the terminal holds one line about the
run (the conjecture, the model, the run id, the iteration, how long it has
been going) and a table with one row per agent — what it is doing, on which
lemma, which prover round, for how long, and the last thing that happened
to it. Log lines scroll above the block as they would in plain output.

The block redraws itself twice a second, so the time column moves while a
call is waiting on the server; nothing is streamed from the model. Anything
that reads the terminal (the human menu, a pasted proof, a y/N) runs inside
paused(), which takes the block off the screen until it is done.

rich is imported here only: a run without it (or not on a terminal) uses
ui.plain instead (choose()).
"""

from __future__ import annotations

import contextlib
import sys
import threading
import time
from typing import Any, Dict, Iterator

from .plain import PlainUI


def choose(mode: str, verbose: bool) -> Any:
    """The UI a run uses. mode is --ui: "plain", "live", or "auto" — live on
    a terminal, with --verbose, when rich is installed. Returns
    (ui, note): note says why a requested live UI is not used, or "" ."""
    if mode == "plain" or not verbose:
        return PlainUI(), ""
    if mode == "auto" and not sys.stdout.isatty():
        return PlainUI(), ""
    try:
        return LiveUI(), ""
    except ImportError:
        return PlainUI(), ("the live status board needs the rich package "
                           "(pip install rich, or re-enter nix develop); "
                           "printing plain lines instead")


def _elapsed(seconds: float) -> str:
    seconds = int(seconds)
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class _Block:
    """What the block draws, rebuilt on every refresh."""

    def __init__(self, ui: "LiveUI") -> None:
        self.ui = ui

    def __rich__(self) -> Any:
        from rich.console import Group
        from rich.table import Table
        from rich.text import Text

        ui = self.ui
        now = time.monotonic()
        head = ui.header_fields()
        parts = []
        for key in ("conjecture", "model", "run", "iteration"):
            if head.get(key):
                parts.append(f"{key} {head[key]}")
        parts.append(f"up {_elapsed(now - ui.started)}")
        header = Text("  ".join(parts), style="bold")

        table = Table(box=None, pad_edge=False, expand=True, show_edge=False,
                      header_style="dim")
        table.add_column("agent", no_wrap=True)
        table.add_column("doing", no_wrap=True, overflow="ellipsis")
        table.add_column("lemma", no_wrap=True, overflow="ellipsis", max_width=32)
        table.add_column("round", no_wrap=True, justify="right")
        table.add_column("for", no_wrap=True, justify="right")
        table.add_column("last", overflow="ellipsis", no_wrap=True, ratio=1)
        for agent, row in ui.board.snapshot():
            table.add_row(
                agent, row.role or "-", row.lemma, row.attempt,
                _elapsed(now - row.since) if row.since else "",
                Text(row.event, style="dim"),
            )
        return Group(Text("─" * 8, style="dim"), header, table)


class LiveUI(PlainUI):
    """PlainUI's log, printed above a live block that draws the board."""

    def __init__(self) -> None:
        from rich.console import Console
        from rich.live import Live

        super().__init__()
        self.started = time.monotonic()
        self._header: Dict[str, str] = {}
        self._lock = threading.Lock()
        self.console = Console()
        # redirect_stdout/stderr: a stray print from anywhere still lands
        # above the block instead of through it.
        self._live = Live(
            _Block(self), console=self.console, refresh_per_second=2,
            transient=True, redirect_stdout=True, redirect_stderr=True,
        )
        self._running = False
        self._pauses = 0

    # -- the run's lifetime ------------------------------------------------

    def start(self) -> None:
        if not self._running:
            self._live.start()
            self._running = True

    def close(self) -> None:
        if self._running:
            self._live.stop()
            self._running = False

    # -- PlainUI's surface -------------------------------------------------

    def log(self, message: str, verbose: bool = True, err: bool = False) -> None:
        if not verbose:
            return
        if not self._running:
            super().log(message, verbose, err)
            return
        # markup off: a lemma statement full of [brackets] is text, not
        # rich markup.
        self.console.print(message, markup=False, highlight=False,
                           style="red" if err else None)

    def header(self, **fields: Any) -> None:
        with self._lock:
            for key, value in fields.items():
                self._header[key] = "" if value is None else str(value)

    def header_fields(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._header)

    @contextlib.contextmanager
    def paused(self) -> Iterator[None]:
        with self._lock:
            self._pauses += 1
            first = self._pauses == 1
        was_running = self._running
        if first and was_running:
            self._live.stop()
            self._running = False
        try:
            yield
        finally:
            with self._lock:
                self._pauses -= 1
                last = self._pauses == 0
            if last and was_running:
                self._live.start()
                self._running = True
