"""JSON repair: the fallback path for a reply that is not clean JSON (the
schema grammar should make it unnecessary).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, Iterator, List, Optional

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


_OPEN_THINK_RE = re.compile(r"<think>.*", re.DOTALL)


# Where a JSON object can start: a brace followed by a key's opening quote
# (or the closing brace of an empty object). LaTeX braces ({Q}, \frac{a}{b},
# {x : x>0}) never look like this, so a reply's prose is skipped over.
_OBJECT_START_RE = re.compile(r'\{\s*["}]')


def strip_thinking(text: str) -> str:
    """text without the reasoning a model left in it: every <think>...</think>
    span; an unclosed <think> and all after it; and, when the chat template
    opened <think> in the prompt, so only the closing tag was generated,
    everything up to the last </think>."""
    text = _THINK_RE.sub("", text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    if "<think>" in text:
        text = _OPEN_THINK_RE.sub("", text)
    return text


_HEX = set("0123456789abcdefABCDEF")


def _repair_escapes(text: str) -> str:
    """Fix backslash escapes inside JSON strings, without corrupting text
    that is already correctly escaped.

    A regex can't do this: in `\\\\subset` (a valid escaped backslash followed
    by 's'), a lookahead-based pattern matches the *second* backslash of the
    pair and doubles it, turning valid JSON into an invalid `\\s` escape.
    The scanner consumes `\\\\` pairs atomically so that can't happen.

    Heuristics inside strings:
      \\ + \\            -> valid pair, keep
      \\ + " or /        -> valid escape, keep
      \\u + 4 hex digits -> valid unicode escape, keep
      \\ + bfnrtu + [a-z]-> LaTeX command (\\frac, \\neq, \\to), double it
      \\ + bfnrtu        -> genuine control escape, keep
      \\ + anything else -> invalid (\\alpha, \\subset, \\{), double it
    """
    out: List[str] = []
    i, n = 0, len(text)
    in_str = False
    while i < n:
        ch = text[i]
        if not in_str:
            if ch == '"':
                in_str = True
            out.append(ch)
            i += 1
            continue
        if ch == '"':
            in_str = False
            out.append(ch)
            i += 1
            continue
        if ch != "\\":
            out.append(ch)
            i += 1
            continue

        nxt = text[i + 1] if i + 1 < n else ""
        if nxt == "\\":
            out.append("\\\\")          # valid pair — consume both, untouchable
            i += 2
        elif nxt in '"/':
            out.append("\\" + nxt)      # valid escape
            i += 2
        elif nxt == "u" and set(text[i + 2 : i + 6]) <= _HEX and len(text[i + 2 : i + 6]) == 4:
            out.append(text[i : i + 6])  # valid \uXXXX
            i += 6
        elif nxt in "bfnrtu":
            follow = text[i + 2 : i + 3]
            if follow.islower() and follow.isalpha():
                out.append("\\\\" + nxt)  # \frac, \neq, \to — LaTeX, not control
            else:
                out.append("\\" + nxt)    # genuine \n, \t, ...
            i += 2
        elif nxt == "":
            out.append("\\\\")          # trailing backslash at end of text
            i += 1
        else:
            out.append("\\\\" + nxt)    # \alpha, \subset, \{ — invalid, double
            i += 2
    return "".join(out)


def _extract_json_object(text: str) -> str:
    """First balanced {...} span, ignoring braces inside strings."""
    start = text.find("{")
    if start == -1:
        return text
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]  # unbalanced => truncated


def _salvage_truncated(text: str) -> str:
    """Close an unterminated string and every open bracket and brace, in
    nesting order, so a cut-off reply is still usable instead of lost: a
    plan cut off inside its list of lemmas keeps the lemmas before the
    cut. A trailing comma or colon left by the cut is dropped first."""
    in_str, esc = False, False
    stack: List[str] = []
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    repaired = text.rstrip()
    if in_str:
        repaired = repaired.rstrip("\\") + '"'
    else:
        repaired = repaired.rstrip(",: \n\t")
    return repaired + "".join(reversed(stack))


def _loads(text: str) -> Any:
    """json.loads, taking raw control characters inside strings: models
    often write a real newline in a long string (a proof) where JSON wants
    \\n, and that is no reason to lose the reply."""
    return json.loads(text, strict=False)


def _object_candidates(raw: str) -> Iterator[str]:
    """The texts to try as a JSON object, in order. The reply without its
    reasoning and fences; then, from each place an object can start
    (_OBJECT_START_RE) — the reply's own start first — its first balanced
    {...} span, as written, with its escapes repaired, and with a truncated
    end closed. So the prose a model writes before its object, LaTeX
    braces in that prose, and an answer that restarted its object after a
    cut-off (the context-wall continuation) all still leave the object
    findable."""
    text = _FENCE_RE.sub("", strip_thinking(raw)).strip()
    starts = [0] + [m.start() for m in _OBJECT_START_RE.finditer(text)]
    seen = set()
    for start in dict.fromkeys(starts):
        span = _extract_json_object(text[start:])
        repaired = _repair_escapes(span)
        for candidate in (span, repaired, _salvage_truncated(repaired)):
            if candidate not in seen:
                seen.add(candidate)
                yield candidate


def clean_json_text(raw: str, expect: Optional[Iterable[str]] = None) -> str:
    """Normalise model output into parseable JSON.

    Repairs escalate and each is tried only if the previous stage fails to
    parse — text that is already valid JSON is returned byte-for-byte
    untouched, so the repair heuristics can never corrupt a correct reply.
    Parse the result with strict=False (raw newlines in strings are
    accepted here). expect, as for parse_json_or_none: the keys the
    answer has, so an object nested inside a reply whose own object could
    not be read is never taken for the answer.
    """
    wanted = tuple(expect or ())
    last = raw
    first_parsed: Optional[str] = None
    for candidate in _object_candidates(raw):
        last = candidate
        try:
            obj = _loads(candidate)
        except json.JSONDecodeError:
            continue
        if not wanted or (isinstance(obj, dict) and any(k in obj for k in wanted)):
            return candidate
        if first_parsed is None:
            first_parsed = candidate
    # No object with an expected key: the reply's own object (the first
    # that parses — never one nested inside it), for the caller to find
    # without the fields it asked for; else the last attempt, so the
    # caller surfaces the real parse error.
    return first_parsed if first_parsed is not None else last


def parse_json_or_none(
    text: str, expect: Optional[Iterable[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Parse an agent reply that already complies with its prompt file.

    The agent prompts all end with "Output strictly valid JSON ... no
    markdown fences and no extra text", so the reasoning stage's content is
    usually the JSON object itself. When it parses, we use it directly and
    skip the extraction call entirely.

    expect names the keys the agent's answer has (any one of them will
    do): the first object found that has one is the answer. A reply's
    prose can hold other small objects — a citation written as
    {"user_id": ..., "lemma_id": ...} — and without expect the first object
    of any shape is taken. None when there is no such object.
    """
    if not text:
        return None
    wanted = tuple(expect or ())
    for candidate in _object_candidates(text):
        try:
            obj = _loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and (not wanted or any(k in obj for k in wanted)):
            return obj
    return None
