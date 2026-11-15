"""Terminal input: the hotkey that flips a run between automation and
manual selection, and the readers the human menu (loops/human.py) uses.

Two mechanisms, kept separate because they solve opposite problems:

  * The MENU (loops/human.py) is line-oriented and reads with input()
    (ask). It runs at exactly one point in the loop — the planning step —
    and only when the loop has already stopped and is waiting for a
    person, so ordinary cooked-mode input is exactly right: you get
    editing, backspace, and somewhere to type a lemma of your own. A
    pasted proof is read whole by read_block.

  * The HOTKEY is a single keystroke, for the opposite situation: a run is 900
    seconds into a prover call and you have changed your mind about who should
    be driving. Nothing can be read line-wise there, so a daemon thread
    watches stdin in cbreak mode and sets a flag that the loops check at
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

import atexit
import codecs
import contextlib
import sys
import threading
import time
from typing import Any, List, Optional, Tuple

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


def ask(prompt: str) -> Optional[str]:
    """One line from the operator, stripped — or None at end of input
    (Ctrl-D, a closed stdin) or a KeyboardInterrupt. None is kept apart from
    any typed answer so each prompt decides what giving up means there: a
    "q" stood in for it once, and the statement prompt took it for a
    statement."""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None


# Bracketed paste: a terminal that has it switched on wraps whatever is
# pasted in these markers, so a paste arrives as one block, newlines and
# all, and only a typed Enter submits.
_PASTE_ON, _PASTE_OFF = "\x1b[?2004h", "\x1b[?2004l"
_PASTE_START, _PASTE_END = "\x1b[200~", "\x1b[201~"
# Shift+Enter, in the terminals that report it apart from Enter at all:
# the kitty keyboard protocol and xterm's modifyOtherKeys.
_SHIFT_ENTER = ("\x1b[13;2u", "\x1b[27;2;13~")


# The terminal settings read_block replaced, while it holds the terminal
# raw: restored by read_block itself, or — when the process exits from
# another thread while the read is blocked (Ctrl-C in a parallel run exits
# from the main thread) — by the atexit hook below, so the shell is never
# left without echo.
_RAW_SAVED: Optional[Tuple[int, Any]] = None


def _restore_terminal() -> None:
    global _RAW_SAVED
    if _RAW_SAVED is not None:
        fd, saved = _RAW_SAVED
        _RAW_SAVED = None
        try:
            sys.stdout.write(_PASTE_OFF)
            sys.stdout.flush()
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        except (OSError, ValueError, termios.error):
            pass


atexit.register(_restore_terminal)


def read_block(prompt: str) -> Optional[str]:
    """A block of text, line breaks and all, from the operator — or None
    when they cancel (Ctrl-D on an empty block, or a KeyboardInterrupt
    where Ctrl-C raises one; in a proofs run, Ctrl-C stops the run as it
    does anywhere else).

    On a POSIX terminal the read is raw, with bracketed paste switched on:
    a paste arrives whole, newlines included, and a typed Enter submits.
    A line break is typed as Alt+Enter or Ctrl+J, both of which every
    terminal sends apart from Enter, or as Shift+Enter where the terminal
    sends that apart too (most send it as a plain Enter, which submits).
    Editing is minimal: Backspace, and Ctrl-U to clear the current line;
    other keys that move the cursor are ignored.

    Anywhere else — no terminal, or no termios (Windows) — the text is
    read a line at a time and ended by a line holding a single ".".
    """
    sys.stdout.write(prompt)
    sys.stdout.flush()
    if not (_HAVE_TERMIOS and sys.stdin.isatty()):
        return _read_block_lines()
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    raw = termios.tcgetattr(fd)
    # Characters one at a time and no echo (we echo), Enter (CR) kept apart
    # from Ctrl-J (LF), flow control off so Ctrl-S/Q are just keys. ISIG and
    # output processing stay on: Ctrl-C still interrupts, and "\n" still
    # prints as a new line.
    raw[0] &= ~(termios.ICRNL | termios.INLCR | termios.IXON)
    raw[3] &= ~(termios.ICANON | termios.ECHO | termios.IEXTEN)
    raw[6][termios.VMIN] = 1
    raw[6][termios.VTIME] = 0
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buf: List[str] = []
    pending = ""      # undecided bytes of an escape sequence
    in_paste = False
    global _RAW_SAVED
    try:
        _RAW_SAVED = (fd, saved)
        termios.tcsetattr(fd, termios.TCSADRAIN, raw)
        sys.stdout.write(_PASTE_ON)
        sys.stdout.flush()
        while True:
            chunk = decoder.decode(os.read(fd, 4096))
            text = pending + chunk
            pending = ""
            i = 0
            while i < len(text):
                if text.startswith("\x1b", i):
                    seq = _escape_at(text, i)
                    if seq is None:  # incomplete: wait for the rest
                        pending = text[i:]
                        break
                    i += len(seq)
                    if seq == _PASTE_START:
                        in_paste = True
                    elif seq == _PASTE_END:
                        in_paste = False
                    elif seq in ("\x1b\r", "\x1b\n") or seq in _SHIFT_ENTER:
                        _put(buf, "\n")
                    # any other sequence (arrows, function keys) is ignored
                    continue
                ch = text[i]
                i += 1
                if in_paste:
                    if ch == "\r":
                        if text.startswith("\n", i):
                            i += 1   # CRLF is one line break
                        _put(buf, "\n")
                    else:
                        _put(buf, ch)
                    continue
                if ch == "\r":
                    sys.stdout.write("\n")
                    return "".join(buf)
                if ch == "\n":
                    _put(buf, "\n")
                elif ch == "\x04":   # Ctrl-D
                    if not buf:
                        sys.stdout.write("\n")
                        return None
                elif ch in ("\x7f", "\x08"):
                    _backspace(buf)
                elif ch == "\x15":   # Ctrl-U
                    while buf and buf[-1] != "\n":
                        _backspace(buf)
                elif ch == "\t" or ch >= " ":
                    _put(buf, ch)
            sys.stdout.flush()
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        return None
    finally:
        _restore_terminal()


def _escape_at(text: str, i: int) -> Optional[str]:
    """The escape sequence starting at text[i] ("\\x1b"), or None when the
    text ends before the sequence does. Handles CSI sequences (ESC [ ...
    final byte), SS3 (ESC O x) and ESC followed by one character (Alt+key,
    Alt+Enter among them)."""
    if i + 1 >= len(text):
        return None
    nxt = text[i + 1]
    if nxt == "[":
        j = i + 2
        while j < len(text):
            if "\x40" <= text[j] <= "\x7e":
                return text[i:j + 1]
            j += 1
        return None
    if nxt == "O":
        return text[i:i + 3] if i + 2 < len(text) else None
    return text[i:i + 2]


def _put(buf: List[str], ch: str) -> None:
    """Append ch to the block and echo it."""
    buf.append(ch)
    sys.stdout.write(ch)


def _backspace(buf: List[str]) -> None:
    """Remove the last character from the block and from the screen. A
    removed line break moves the cursor back to the end of the line above
    (a line longer than the terminal is wide has wrapped, and the cursor
    lands on its last screen row instead — the text is right either way)."""
    if not buf:
        return
    ch = buf.pop()
    if ch != "\n":
        sys.stdout.write("\b \b")
        return
    line = "".join(buf).rsplit("\n", 1)[-1]
    sys.stdout.write("\x1b[A\r")
    if line:
        sys.stdout.write(f"\x1b[{len(line)}C")


def _read_block_lines() -> Optional[str]:
    """read_block without a terminal: lines up to one holding a single
    ".", or None at end of input before any line."""
    print('  (end the proof with a line holding a single ".")')
    lines: List[str] = []
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            print()
            return "\n".join(lines) if lines else None
        if line.strip() == ".":
            return "\n".join(lines)
        lines.append(line)
