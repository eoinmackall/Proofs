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

import atexit
import codecs
import contextlib
import json
import re
import sys
import textwrap
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

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
        "own_proof" - the operator proves `lemma` themselves: `proof` is
                      their proof and `cited_lemmas` / `cited_references`
                      its citations, in the prover's raw shapes, already
                      checked by the menu's check_citations when it was
                      given one. The caller puts the proof through the
                      three verifiers, exactly as it would the prover's;
                      nothing enters the DAG on the operator's say-so
        "prove"     - send `lemma` down the usual prover/verifier pipeline
        "replan"    - discard all candidates, ask the planner again
        "auto"      - hand control back to the automation, starting now
        "quit"      - stop the run and keep the DAG as it stands

    "own_proof" and "prove" differ only in who writes the proof. Either
    way it is the verifiers who decide whether it goes in.
    """
    action: str
    lemma: Optional[Dict[str, Any]] = None
    notes: List[str] = field(default_factory=list)
    proof: Optional[str] = None
    cited_lemmas: List[Any] = field(default_factory=list)
    cited_references: List[Any] = field(default_factory=list)


_MENU = (
    "  [1-9]  prove that one yourself — your proof goes to the verifiers\n"
    "  [1p]   send that one to the proving pipeline instead\n"
    "  [w]    write your own lemma and proof   [wp] write one and have it proved\n"
    "  [r]    none of these, re-plan           [a]  resume automation\n"
    "  [q]    quit and keep the DAG\n"
)

# A trailing 'p' is the whole grammar: it means "hand this to the models"
# wherever it appears. "3p" is the gesture as described — pick the candidate,
# then hit a key. "p3" reads more naturally to some people and costs nothing
# to accept, so both work, and likewise "wp" / "pw".
_PICK_RE = re.compile(r"(p?)(\d+)(p?)$")

# check_citations: given the raw cited_lemmas and cited_references, the
# reason they cannot be used (an unknown or suspended lemma, a reference the
# collection does not hold), or None when every citation resolves.
CitationCheck = Callable[[List[Any], List[Any]], Optional[str]]


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
        flag = "  " + "; ".join(problems) if problems else ""
        out.append(f"  [{i}] {cand.get('id', '?')}{flag}")
        out.append(_wrap(cand.get("statement", "(no statement)")))
    return "\n".join(out)


def _ask(prompt: str) -> Optional[str]:
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


def choose(
    screened: Sequence[Tuple[Dict[str, Any], List[str]]],
    proved_ids: Iterable[str],
    hotkey: HotKey,
    plan_summary: str = "",
    check_citations: Optional[CitationCheck] = None,
) -> Choice:
    """Put the candidate list to the operator and return their decision.

    check_citations, when given, is run on the citations of a proof the
    operator writes, before the menu returns it: a citation that does not
    resolve is reported and the citations asked for again, so a pasted
    proof is not lost to a typo in an id."""
    proved = set(proved_ids)
    print("\n" + "─" * 72)
    print(f"Planner proposes {len(screened)} next steps:\n")
    print(render(screened, plan_summary))
    print("\n" + _MENU)

    with hotkey.paused():
        while True:
            answer = _ask("> ")
            if answer is None:  # end of input at the menu: quit
                return Choice("quit")
            reply = answer.lower()

            if reply in ("q", "quit"):
                return Choice("quit")
            if reply in ("a", "auto"):
                return Choice("auto")
            if reply in ("r", "replan", ""):
                # End of input here is no reason given, not a reason "q".
                why = _ask("reason for the planner (optional): ")
                note = why or "Rejected by the operator without a stated reason."
                return Choice("replan", notes=[note])
            if reply in ("w", "write", "wp", "pw", "writep"):
                lemma = _write_own(proved)
                if lemma is None:
                    continue
                if reply in ("wp", "pw", "writep"):
                    return Choice("prove", lemma=lemma)
                own = _own_proof(lemma, check_citations)
                if own is None:
                    print(_MENU)
                    continue
                return own
            if reply in ("?", "h", "help"):
                print(_MENU)
                continue

            match = _PICK_RE.match(reply)
            if match and 1 <= int(match.group(2)) <= len(screened):
                cand, _ = screened[int(match.group(2)) - 1]
                if cand["id"] in proved:
                    # Nothing sensible to do: proving it again gains nothing
                    # and committing it would sit beside a node other lemmas
                    # already cite. Say so and let them pick again.
                    print(f"  {cand['id']} is already in the DAG. Pick another.")
                    continue
                if match.group(1) or match.group(3):
                    return Choice("prove", lemma=cand)
                own = _own_proof(cand, check_citations)
                if own is None:
                    print(_MENU)
                    continue
                return own

            print("  ? unrecognised. " + _MENU)


def _write_own(proved: set) -> Optional[Dict[str, Any]]:
    """Let the operator state the next lemma themselves.

    Worth having: the most valuable thing a mathematician can contribute to a
    run like this is usually not picking between the model's five ideas but
    supplying the sixth one it didn't think of. Whether the operator then
    proves it or the models do is the caller's business — `w` asks for the
    operator's proof next, `wp` sends it to the models, exactly as a digit
    and a digit with `p` do for a candidate.

    Only an id and a statement are asked for here, the same two fields the
    planner supplies.
    """
    # End of input, an empty line or a lone "q" at either prompt cancels the
    # written lemma and returns to the menu.
    lemma_id = _ask("  id (e.g. lemma_7): ")
    if not lemma_id or lemma_id == "q":
        return None
    if lemma_id in proved:
        print(f"  {lemma_id} is already in the DAG; pick another id.")
        return None

    statement = _ask("  statement: ")
    if not statement or statement == "q":
        print("  no statement; nothing added.")
        return None

    return {"id": lemma_id, "statement": statement}


# ----------------------------------------------------------------------------
# The operator's proof
# ----------------------------------------------------------------------------
def _own_proof(
    lemma: Dict[str, Any], check_citations: Optional[CitationCheck]
) -> Optional[Choice]:
    """Ask the operator for a proof of `lemma` and its citations, and
    return them as an "own_proof" Choice — or None when they cancel, which
    puts the menu back.

    The proof is read with read_block (paste it, or type it; Enter
    submits). A pasted JSON object with a "proof" field — the prover's own
    output shape — is taken whole, its cited_lemmas and cited_references
    with it. Otherwise the citations are asked for on one line (see
    parse_citations). Citations that do not parse, or that check_citations
    finds do not resolve, are reported and asked for again; the proof is
    kept.
    """
    print(f"\n  Proof of {lemma.get('id', '?')}:")
    print(_wrap(str(lemma.get("statement", "")), indent="    "))
    proof = read_block(
        "  Paste or type your proof. Enter submits; Alt+Enter or Ctrl+J "
        "(or Shift+Enter, where your terminal sends it) starts a new line; "
        "Ctrl+D on an empty proof cancels.\n"
    )
    if proof is None or not proof.strip():
        print("  no proof; back to the menu.")
        return None

    cited: Optional[Tuple[List[Any], List[Any]]] = None
    as_json = _proof_object(proof)
    if as_json is not None:
        proof_text, raw = as_json
        try:
            cited = _citations_from_json(raw)
        except ValueError as e:
            print(f"  the pasted object's citations are not usable: {e}")
            return None
        proof = proof_text
        problem = check_citations(*cited) if check_citations else None
        if problem:
            print(f"  {problem}")
            cited = None  # ask for them on the line instead
    while cited is None:
        line = _ask(
            "  cites (lemmas as user:lemma_id, references by id, comma-"
            "separated; or the prover's JSON; empty for none): "
        )
        if line is None or line == "q":
            print("  cancelled; back to the menu.")
            return None
        try:
            attempt = parse_citations(line)
        except ValueError as e:
            print(f"  {e}")
            continue
        problem = check_citations(*attempt) if check_citations else None
        if problem:
            print(f"  {problem}")
            continue
        cited = attempt
    return Choice(
        "own_proof", lemma=lemma, proof=proof.strip(),
        cited_lemmas=cited[0], cited_references=cited[1],
    )


# An id as the files spell one: a user_id, a lemma_id or a reference id.
_ID_RE = re.compile(r"[A-Za-z0-9_.\-]+$")


def parse_citations(text: str) -> Tuple[List[Any], List[Any]]:
    """The operator's citations line as (cited_lemmas, cited_references),
    in the prover's raw shapes.

    Either the prover's JSON — an object with "cited_lemmas" (a list of
    {"user_id", "lemma_id"} objects) and "cited_references" (a list of
    ids), or a bare list of such entries — or the short form: entries
    separated by commas or spaces, a lemma as user_id:lemma_id and a
    reference by its id. An empty line is no citations. Anything else
    raises ValueError saying what is wrong; nothing is guessed. Whether
    the entries name results that exist is not checked here (see
    check_citations in choose()).
    """
    text = text.strip()
    if not text:
        return [], []
    if text[0] in "{[":
        try:
            raw = json.loads(text)
        except ValueError as e:
            raise ValueError(f"not valid JSON: {e}") from None
        return _citations_from_json(raw)
    lemmas: List[Any] = []
    refs: List[Any] = []
    for token in re.split(r"[,\s]+", text):
        if not token:
            continue
        if ":" in token:
            user_id, _, lemma_id = token.partition(":")
            if not (_ID_RE.match(user_id) and _ID_RE.match(lemma_id)):
                raise ValueError(
                    f"{token!r} is not a lemma citation; write user_id:lemma_id"
                )
            lemmas.append({"user_id": user_id, "lemma_id": lemma_id})
        elif _ID_RE.match(token):
            refs.append(token)
        else:
            raise ValueError(
                f"{token!r} is not an id; cite a lemma as user_id:lemma_id "
                f"and a reference by its id"
            )
    return lemmas, refs


def _citations_from_json(raw: Any) -> Tuple[List[Any], List[Any]]:
    """Citations in the prover's JSON shapes, checked for shape only: an
    object with list-valued "cited_lemmas" / "cited_references" (either may
    be missing), or a list whose entries are {"user_id", "lemma_id"}
    objects (lemmas) and id strings (references)."""
    if isinstance(raw, dict):
        unknown = set(raw) - {"cited_lemmas", "cited_references", "proof"}
        if unknown:
            raise ValueError(
                f"unexpected key(s) {', '.join(sorted(unknown))}; the "
                f"citations are cited_lemmas and cited_references"
            )
        lemmas = raw.get("cited_lemmas", [])
        refs = raw.get("cited_references", [])
        if not isinstance(lemmas, list) or not isinstance(refs, list):
            raise ValueError("cited_lemmas and cited_references must be lists")
        entries = [*lemmas, *refs]
    elif isinstance(raw, list):
        entries = raw
    else:
        raise ValueError("expected a JSON object or list of citations")
    out_lemmas: List[Any] = []
    out_refs: List[Any] = []
    for entry in entries:
        if (
            isinstance(entry, dict)
            and set(entry) == {"user_id", "lemma_id"}
            and all(isinstance(v, str) and _ID_RE.match(v) for v in entry.values())
        ):
            out_lemmas.append(dict(entry))
        elif isinstance(entry, str) and _ID_RE.match(entry):
            out_refs.append(entry)
        else:
            raise ValueError(
                f"{json.dumps(entry)} is neither a {{\"user_id\", "
                f"\"lemma_id\"}} object nor a reference id"
            )
    return out_lemmas, out_refs


def _proof_object(text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """(proof, the object) when the pasted text is the prover's JSON — an
    object with a string "proof" — else None, and the text is the proof
    itself."""
    stripped = text.strip()
    if not stripped.startswith("{"):
        return None
    try:
        obj = json.loads(stripped)
    except ValueError:
        return None
    if isinstance(obj, dict) and isinstance(obj.get("proof"), str):
        return obj["proof"], obj
    return None


# ----------------------------------------------------------------------------
# Multi-line input
# ----------------------------------------------------------------------------
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
