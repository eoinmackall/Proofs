"""Human-in-the-loop controls: a hotkey that flips the run between automation
and manual selection, and the menu that manual selection puts up.

Two mechanisms, kept separate because they solve opposite problems:

  * The MENU (`choose`) is line-oriented and reads with input(). It runs at
    exactly one point in the loop — the planning step — and only when the loop
    has already stopped and is waiting for a person, so ordinary cooked-mode
    input is exactly right: you get editing, backspace, and somewhere to type
    a lemma of your own. "A couple of keystrokes" here means `2` then Enter,
    and that is the entire interaction: nothing downstream asks again.

  * The HOTKEY is a single keystroke, for the opposite situation: a run is 900
    seconds into a prover call and you have changed your mind about who should
    be driving. Nothing can be read line-wise there, so a daemon thread
    watches stdin in cbreak mode and sets a flag that run_loop() checks at
    several points per iteration. It toggles, so it works in both directions;
    because the loop is blocked on HTTP, the press is acknowledged as soon as
    the current call returns and takes effect at the next planning step.

The two cannot both own the terminal, which is what `HotKey.paused()` is for:
it takes stdin out of cbreak mode, waits for the listener thread to actually
be idle, and only then lets the menu call input(). Skipping that wait is the
classic bug — the listener sits inside its 0.2 s read and swallows the first
character typed at the prompt.

Everything degrades quietly. No tty (nohup, a pipe, CI), no termios (Windows
falls back to msvcrt; anything else gets no listener at all): the hotkey
reports itself unavailable, `--mode human` still works, and `--mode auto`
behaves exactly as it did before this module existed.
"""

from __future__ import annotations

import contextlib
import re
import sys
import textwrap
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # POSIX
    import os
    import select
    import termios
    import tty
    _HAVE_TERMIOS = True
except ImportError:  # pragma: no cover - Windows
    _HAVE_TERMIOS = False

try:  # Windows
    import msvcrt
    _HAVE_MSVCRT = True
except ImportError:
    _HAVE_MSVCRT = False


# ----------------------------------------------------------------------------
# Hotkey listener
# ----------------------------------------------------------------------------
class HotKey:
    """Watches stdin for a single character while the main thread works.

    Usage:

        hk = HotKey("h", on_press=lambda: print("manual mode queued"))
        hk.start()
        try:
            ...
            if hk.take():        # true exactly once per press
                go_manual()
        finally:
            hk.stop()
    """

    def __init__(
        self,
        keys: str = "h",
        enabled: bool = True,
        on_press: Optional[Any] = None,
    ) -> None:
        self.keys = {k.lower() for k in keys if k.strip()}
        self.on_press = on_press
        self.available = bool(
            enabled
            and self.keys
            and sys.stdin.isatty()
            and (_HAVE_TERMIOS or _HAVE_MSVCRT)
        )
        self._pressed = threading.Event()
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._idle = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._saved: Any = None

    # -- terminal mode -------------------------------------------------------
    def _to_cbreak(self) -> None:
        if not _HAVE_TERMIOS or not self.available:
            return
        fd = sys.stdin.fileno()
        self._saved = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        # Anything typed while we were in cooked mode is not a hotkey press.
        termios.tcflush(fd, termios.TCIFLUSH)

    def _to_cooked(self) -> None:
        if not _HAVE_TERMIOS or self._saved is None:
            return
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._saved)
        self._saved = None

    # -- key reading ---------------------------------------------------------
    def _read_key(self, timeout: float) -> str:
        """One character, or "" if none arrived within `timeout`.

        The timeout is what makes stop() and paused() possible at all: a bare
        blocking read cannot be interrupted, so the thread would outlive the
        run and keep eating keystrokes belonging to the shell.
        """
        if _HAVE_TERMIOS:
            ready, _, _ = select.select([sys.stdin], [], [], timeout)
            if not ready:
                return ""
            return os.read(sys.stdin.fileno(), 1).decode(errors="ignore")
        if _HAVE_MSVCRT:  # pragma: no cover - Windows
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if msvcrt.kbhit():
                    return msvcrt.getwch()
                time.sleep(0.02)
        return ""

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._pause.is_set():
                self._idle.set()
                time.sleep(0.05)
                continue
            self._idle.clear()
            try:
                ch = self._read_key(0.2)
            except (OSError, ValueError):
                # stdin closed under us; nothing useful left to watch.
                break
            if ch and ch.lower() in self.keys and not self._pressed.is_set():
                self._pressed.set()
                if self.on_press:
                    try:
                        self.on_press()
                    except Exception:
                        pass
        self._idle.set()

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> "HotKey":
        if not self.available or self._thread is not None:
            return self
        self._to_cbreak()
        self._thread = threading.Thread(
            target=self._run, name="hotkey", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._to_cooked()

    def take(self) -> bool:
        """True exactly once per press, then rearms."""
        if self._pressed.is_set():
            self._pressed.clear()
            return True
        return False

    @contextlib.contextmanager
    def paused(self):
        """Hand the terminal back to line-oriented input() for a while."""
        if self._thread is None:
            yield
            return
        self._pause.set()
        self._idle.wait(timeout=1.0)   # let the thread finish any pending read
        self._to_cooked()
        try:
            yield
        finally:
            self._to_cbreak()
            self._pause.clear()

    def hint(self) -> str:
        if not self.available:
            return ""
        keys = "/".join(sorted(self.keys))
        return f"(press '{keys}' at any time to switch between manual and auto)"


class NullHotKey(HotKey):
    """Stand-in for runs with no terminal: never fires, never grabs stdin."""

    def __init__(self) -> None:
        super().__init__(keys="", enabled=False)


# ----------------------------------------------------------------------------
# Menu
# ----------------------------------------------------------------------------
@dataclass
class Choice:
    """What the operator decided at a planning step.

    action is one of:
        "assert"  - `lemma` is true on the operator's authority; it goes into
                    the DAG as it stands, with no prover and no verifiers
        "prove"   - send `lemma` down the usual prover/verifier pipeline
        "replan"  - discard all candidates, ask the planner again
        "auto"    - hand control back to the automation, starting now
        "quit"    - stop the run and keep the DAG as it stands

    "assert" and "prove" are separate because selecting a lemma and vouching
    for one are separate judgements. You can be sure candidate 3 is the right
    next step and still have no idea whether it is true.
    """
    action: str
    lemma: Optional[Dict[str, Any]] = None
    notes: List[str] = field(default_factory=list)


_MENU = (
    "  [1-9]  accept as true — straight into the DAG, no proof run\n"
    "  [1p]   send that one to the prover and verifiers instead\n"
    "  [w]    write your own lemma     [wp] write one and have it proved\n"
    "  [r]    none of these, re-plan   [a]  resume automation\n"
    "  [q]    quit and keep the DAG"
)

# A trailing 'p' is the whole grammar: it means "hand this to the models"
# wherever it appears. "3p" is the gesture as described — pick the candidate,
# then hit a key. "p3" reads more naturally to some people and costs nothing
# to accept, so both work, and likewise "wp" / "pw".
_PICK_RE = re.compile(r"(p?)(\d+)(p?)$")


def _wrap(text: str, indent: str = "      ", width: int = 78) -> str:
    return textwrap.fill(
        " ".join(str(text).split()),
        width=width,
        initial_indent=indent,
        subsequent_indent=indent,
    )


def render(
    screened: Sequence[Tuple[Dict[str, Any], List[str]]],
    plan_summary: str = "",
) -> str:
    """The candidate list, printed the same way in both modes.

    Automation logs it so that a run you walk away from still leaves a record
    of the four roads not taken.
    """
    out: List[str] = []
    if plan_summary:
        out.append(_wrap(f"Strategy: {plan_summary}", indent="  "))
    for i, (cand, problems) in enumerate(screened, start=1):
        flag = "  ⚠ " + "; ".join(problems) if problems else ""
        out.append(f"  [{i}] {cand.get('id', '?')}{flag}")
        out.append(_wrap(cand.get("statement", "(no statement)")))
    return "\n".join(out)


def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return "q"


def choose(
    screened: Sequence[Tuple[Dict[str, Any], List[str]]],
    proved_ids: Iterable[str],
    hotkey: HotKey,
    plan_summary: str = "",
) -> Choice:
    """Put the candidate list to the operator and return their decision."""
    proved = set(proved_ids)
    print("\n" + "─" * 72)
    print(f"Planner proposes {len(screened)} next steps:\n")
    print(render(screened, plan_summary))
    print("\n" + _MENU)

    with hotkey.paused():
        while True:
            reply = _ask("> ").lower()

            if reply in ("q", "quit"):
                return Choice("quit")
            if reply in ("a", "auto"):
                return Choice("auto")
            if reply in ("r", "replan", ""):
                why = _ask("reason for the planner (optional): ")
                note = why or "Rejected by the operator without a stated reason."
                return Choice("replan", notes=[note])
            if reply in ("w", "write", "wp", "pw", "writep"):
                lemma = _write_own(proved)
                if lemma is None:
                    continue
                return Choice(
                    "prove" if reply in ("wp", "pw", "writep") else "assert",
                    lemma=lemma,
                )
            if reply in ("?", "h", "help"):
                print(_MENU)
                continue

            match = _PICK_RE.match(reply)
            if match and 1 <= int(match.group(2)) <= len(screened):
                cand, _ = screened[int(match.group(2)) - 1]
                if cand["id"] in proved:
                    # Nothing sensible to do: proving it again gains nothing
                    # and asserting it would overwrite a node other lemmas
                    # already cite. Say so and let them pick again.
                    print(f"  {cand['id']} is already in the DAG. Pick another.")
                    continue
                return Choice(
                    "prove" if (match.group(1) or match.group(3)) else "assert",
                    lemma=cand,
                )

            print("  ? unrecognised. " + _MENU)


def _write_own(proved: set) -> Optional[Dict[str, Any]]:
    """Let the operator state the next lemma themselves.

    Worth having: the most valuable thing a mathematician can contribute to a
    run like this is usually not picking between the model's five ideas but
    supplying the sixth one it didn't think of. Whether it is then asserted or
    proved is the caller's business — `w` vouches for it, `wp` sends it to the
    models, exactly as a digit and a digit with `p` do for a candidate.

    Only an id and a statement are asked for, the same two fields the planner
    supplies. You are standing in for the planner here, not for the prover.
    """
    lemma_id = _ask("  id (e.g. lemma_7): ")
    if not lemma_id or lemma_id == "q":
        return None
    if lemma_id in proved:
        print(f"  {lemma_id} is already in the DAG; pick another id.")
        return None

    statement = _ask("  statement: ")
    if not statement:
        print("  empty statement; nothing added.")
        return None

    return {"id": lemma_id, "statement": statement}
