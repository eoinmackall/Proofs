"""The live terminal block: a compact view of the run, redrawn in place.

While a run is live, the bottom of the terminal holds one block:

    the last few log lines (LOG_LINES), each with the agent that logged
        it and what kind of line it is (tool call, accept, reject, ...)
    one line about the run (the conjecture, the model, the run id, the
        iteration, how long it has been going)
    a table with one row per agent — what it is doing, on which lemma,
        which prover round, for how long, and the last thing that happened
        to it — at most MAX_AGENT_ROWS of them, and fewer on a short
        terminal; the rest are counted by role on a last line

The log does not scroll above the block: each message replaces the oldest
of the lines shown, and the block keeps its place. --ui plain prints every
line instead. When the run ends, the block goes and the lines it last
showed are printed, so the terminal is left with how the run ended.

The block redraws itself twice a second, so the time column moves while a
call is waiting on the server; nothing is streamed from the model. Only
that redraw writes to the terminal while the block is up — a log() just
records the line, and a stray print from anywhere is caught (_Sink) and
recorded the same way — so no two threads ever race to move the cursor.
Anything that reads the terminal (the human menu, a pasted proof, a y/N)
runs inside paused(), which takes the block off the screen until it is done.

rich is imported here only: a run without it (or not on a terminal) uses
ui.plain instead (choose()).
"""

from __future__ import annotations

import collections
import contextlib
import io
import re
import sys
import threading
import time
from typing import Any, Deque, Dict, Iterator, List, Tuple

from .plain import PlainUI

# How many log lines the block shows above the run line.
LOG_LINES = 3
# The most agent rows the board draws, the "+N more" line included; a
# terminal too short for them gets fewer.
MAX_AGENT_ROWS = 12

# A log line's kind, by what it says: the first pattern that matches wins,
# and a line matching none has no kind. An err line is an error unless it
# says it is a warning. The kind is drawn in its colour in a column of its
# own; a leading "warning: " or "error: " is dropped from the text, since
# the column says it.
_KINDS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("warning", re.compile(r"^\s*warning:", re.I)),
    ("tool call", re.compile(r"\bcalled \w+\(")),
    ("accept", re.compile(r": ACCEPT —")),
    ("reject", re.compile(r": REJECT —|\brejected\b")),
    ("commit", re.compile(r"Adding to DAG|\bcommitted\b")),
    ("error", re.compile(r"\b(crashed|error|failed|Traceback)\b", re.I)),
    ("compaction", re.compile(r"\bcompact", re.I)),
)
_KIND_STYLES = {
    "tool call": "#458588",
    "accept": "#b8bb26",
    "reject": "#fe8019",
    "commit": "#8ec07c",
    "error": "#cc241d",
    "warning": "#d3869b",
    "compaction": "#fabd2f",
}
_LEADING_LABEL = re.compile(r"^\s*(warning|error):\s*", re.I)


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


def _kind(message: str, err: bool) -> str:
    for kind, pattern in _KINDS:
        if pattern.search(message):
            return kind
    return "error" if err else ""


def _one_line(message: str) -> str:
    """A message as the block shows it: its first non-blank line, with
    "..." when lines were left out (a shortlist, a traceback)."""
    lines = [line for line in message.splitlines() if line.strip()]
    if not lines:
        return ""
    text = _LEADING_LABEL.sub("", lines[0].strip(), count=1)
    return text + " ..." if len(lines) > 1 else text


class _Entry:
    __slots__ = ("agent", "kind", "text")

    def __init__(self, agent: str, kind: str, text: str) -> None:
        self.agent, self.kind, self.text = agent, kind, text


def _log_lines(entries: List[_Entry]) -> Any:
    """The log lines as a grid: agent, kind (in its colour), text."""
    from rich.table import Table
    from rich.text import Text

    grid = Table.grid(padding=(0, 2), expand=True)
    grid.add_column(no_wrap=True, style="dim")
    grid.add_column(no_wrap=True)
    grid.add_column(no_wrap=True, overflow="ellipsis", ratio=1)
    for entry in entries:
        # Text, not str: a str cell is rich markup, and a lemma statement
        # full of [brackets] is text.
        grid.add_row(Text(entry.agent),
                     Text(entry.kind, style=_KIND_STYLES.get(entry.kind, "")),
                     Text(entry.text))
    return grid


class _Block:
    """What the block draws, rebuilt on every refresh."""

    # The lines the block always has besides the log and the agent rows:
    # the rule, the run line and the table's column names.
    _FIXED = 3

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
        header = Text("  ".join(parts), style="bold", no_wrap=True,
                      overflow="ellipsis")

        log = ui.recent()

        rows = ui.board.snapshot()
        room = ui.console.size.height - self._FIXED - len(log)
        shown, hidden = _cap(rows, max(1, min(MAX_AGENT_ROWS, room)))

        table = Table(box=None, pad_edge=False, expand=True, show_edge=False,
                      header_style="dim")
        table.add_column("agent", no_wrap=True)
        table.add_column("status", no_wrap=True, overflow="ellipsis")
        table.add_column("lemma", no_wrap=True, overflow="ellipsis", max_width=32)
        table.add_column("round", no_wrap=True, justify="right")
        table.add_column("for", no_wrap=True, justify="right")
        table.add_column("last", overflow="ellipsis", no_wrap=True, ratio=1)
        for agent, row in shown:
            table.add_row(
                agent, row.role or "-", row.lemma, row.attempt,
                _elapsed(now - row.since) if row.since else "",
                Text(row.event, style="dim"),
            )
        parts = [Text("─" * 8, style="dim")]
        if log:
            parts.append(_log_lines(log))
        parts += [header, table]
        if hidden:
            parts.append(Text(_more(hidden), style="dim", no_wrap=True,
                              overflow="ellipsis"))
        return Group(*parts)


def _cap(rows: List[Tuple[str, Any]], limit: int) -> Tuple[list, list]:
    """The rows to draw and the rows to count, in at most limit lines (the
    count's line among them). All of them, in the order the agents first
    reported, when they fit; otherwise the ones longest at what they are
    doing first — a stuck agent stays in view."""
    if len(rows) <= limit:
        return rows, []
    rows = sorted(rows, key=lambda item: item[1].since or float("inf"))
    keep = max(0, limit - 1)
    return rows[:keep], rows[keep:]


def _more(hidden: List[Tuple[str, Any]]) -> str:
    """"+58 more: 41 prover, 12 verifier, 5 waiting" — the hidden rows
    counted by the first word of what they are doing, most first."""
    counts = collections.Counter(
        (row.role.split() or ["-"])[0] for _, row in hidden
    )
    return f"+{len(hidden)} more: " + ", ".join(
        f"{n} {role}" for role, n in counts.most_common()
    )


class _Sink(io.TextIOBase):
    """sys.stdout / sys.stderr while the block is up: a stray print from
    anywhere becomes a log line, by whole lines, instead of writing
    through the block."""

    def __init__(self, ui: "LiveUI", real: Any, err: bool) -> None:
        self._ui, self._real, self._err = ui, real, err
        self._partial = ""
        self._lock = threading.Lock()

    def write(self, text: str) -> int:
        with self._lock:
            *lines, self._partial = (self._partial + text).split("\n")
        for line in lines:
            self._ui.log(line, err=self._err)
        return len(text)

    def flush(self) -> None:
        with self._lock:
            line, self._partial = self._partial, ""
        if line:
            self._ui.log(line, err=self._err)

    def fileno(self) -> int:
        return self._real.fileno()

    def isatty(self) -> bool:
        return False

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return getattr(self._real, "encoding", "utf-8")


class LiveUI(PlainUI):
    """PlainUI's surface, drawn as one compact live block."""

    def __init__(self) -> None:
        from rich.console import Console
        from rich.live import Live

        super().__init__()
        self.started = time.monotonic()
        self._header: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._recent: Deque[_Entry] = collections.deque(maxlen=LOG_LINES)
        # Which agent each thread is, from its status() calls: its log
        # lines are labelled with it.
        self._agents: Dict[int, str] = {}
        # The terminal itself, held before _Sink takes sys.stdout's place.
        self._streams = (sys.stdout, sys.stderr)
        self.console = Console(file=sys.stdout)
        self._live = Live(
            _Block(self), console=self.console, refresh_per_second=2,
            transient=True, redirect_stdout=False, redirect_stderr=False,
        )
        self._running = False
        self._pauses = 0

    # -- drawing on and off ------------------------------------------------

    def _show(self) -> None:
        sys.stdout = _Sink(self, self._streams[0], err=False)
        sys.stderr = _Sink(self, self._streams[1], err=True)
        self._live.start()
        self._running = True

    def _hide(self) -> None:
        self._live.stop()
        self._running = False
        # The terminal back first: a sink's flush logs, and with the block
        # off a log prints to sys.stdout.
        sinks = (sys.stdout, sys.stderr)
        sys.stdout, sys.stderr = self._streams
        for sink in sinks:
            if isinstance(sink, _Sink):
                sink.flush()

    # -- the run's lifetime ------------------------------------------------

    def start(self) -> None:
        if not self._running:
            self._show()

    def close(self) -> None:
        if self._running:
            self._hide()
            # The block is gone (transient); its last lines stay behind.
            log = self.recent()
            if log:
                self.console.print(_log_lines(log))

    # -- PlainUI's surface -------------------------------------------------

    def log(self, message: str, verbose: bool = True, err: bool = False) -> None:
        if not verbose:
            return
        if not self._running:
            super().log(message, verbose, err)
            return
        text = _one_line(message)
        if not text:
            return
        agent = self._agents.get(threading.get_ident(), "")
        with self._lock:
            self._recent.append(_Entry(agent, _kind(message, err), text))

    def recent(self) -> List[_Entry]:
        with self._lock:
            return list(self._recent)

    def status(self, agent: str, **fields: str) -> None:
        self._agents[threading.get_ident()] = agent
        super().status(agent, **fields)

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
            self._hide()
        try:
            yield
        finally:
            with self._lock:
                self._pauses -= 1
                last = self._pauses == 0
            if last and was_running:
                self._show()
